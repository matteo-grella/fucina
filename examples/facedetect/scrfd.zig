//! SCRFD det_10g face detector forward — mirrors the C++/ggml
//! `scrfd_forward` (refs/face-detect.cpp `src/scrfd_graph.cpp:189`). ResNet-style
//! backbone (BN folded into conv bias, like ArcFace) + PAFPN neck + 3 shared GFL
//! stride heads → 9 raw heads (score/bbox/kps × strides 8/16/32). Channel-last
//! `Tensor(.{.h,.w,.c})`, composed from nn.zig ops + the recognizer's loaders.
//! The head maps' natural HWC layout already equals insightface's flatten_head
//! (ct = a·C+c anchor-major), so no reorder is needed.
//!
//! Weights load ONCE into `Model` — a lazy by-name cache (the
//! `src/llm/parakeet/weights.zig` pattern): each conv weight dequants/repacks
//! on first use and is reused by every later forward, so repeated detection is
//! pure compute (mirrors the reference's load-once graph lifecycle).

const std = @import("std");
const fucina = @import("fucina");
const nn = @import("nn.zig");
const rec = @import("recognizer.zig");
const image = @import("image.zig");

const gguf = fucina.gguf;
const ExecContext = fucina.ExecContext;
const Map = nn.Map;
const ConvW = fucina.Tensor(.{ .oc, .kh, .kw, .c });
const Channels = nn.Channels;

/// Session-lifetime SCRFD weight cache. Borrows `file` (must outlive the
/// model); conv weights and head scales load on first use.
pub const Model = struct {
    allocator: std.mem.Allocator,
    file: *const gguf.File,
    convs: std.StringHashMapUnmanaged(ConvEntry) = .empty,

    const ConvEntry = struct {
        w: ConvW,
        b: Channels,
        prep: fucina.PreparedConvWeights,
    };

    pub fn init(allocator: std.mem.Allocator, file: *const gguf.File) Model {
        return .{ .allocator = allocator, .file = file };
    }

    pub fn deinit(self: *Model) void {
        var it = self.convs.iterator();
        while (it.next()) |e| {
            e.value_ptr.w.deinit();
            e.value_ptr.b.deinit();
            e.value_ptr.prep.deinit();
        }
        self.convs.deinit(self.allocator);
        self.* = undefined;
    }

    /// The conv weight+bias pair for `wn`/`bn`, loaded+repacked on first use.
    /// Keyed by the weight name (a static string in this file, so keys are
    /// borrowed). Stride-1 call sites get load-time Winograd weight planes
    /// (`.empty` — inert — for the shapes the Winograd route never takes);
    /// each weight name is used at a single stride in this net.
    fn convPair(self: *Model, ctx: *ExecContext, wn: []const u8, bn: []const u8, stride: usize) !*const ConvEntry {
        if (self.convs.getPtr(wn)) |v| return v;
        var w = try rec.loadConvW(ctx, self.allocator, self.file, wn);
        errdefer w.deinit();
        var b = try rec.loadVec(ctx, self.allocator, self.file, bn);
        errdefer b.deinit();
        var prep: fucina.PreparedConvWeights = if (stride == 1) try w.prepareConv2dWeights(ctx) else .empty;
        errdefer prep.deinit();
        try self.convs.put(self.allocator, wn, .{ .w = w, .b = b, .prep = prep });
        return self.convs.getPtr(wn).?;
    }
};

fn conv(ctx: *ExecContext, model: *Model, x: *const Map, wn: []const u8, bn: []const u8, stride: usize, pad: usize, do_relu: bool) !Map {
    const e = try model.convPair(ctx, wn, bn, stride);
    if (do_relu) {
        // Fused epilogue: identical values to conv2d + relu, one fewer pass.
        return x.conv2dPreparedRelu(ctx, &e.w, &e.prep, &e.b, .{ stride, stride }, .{ pad, pad }, 1, .{ .h, .w, .c });
    }
    return x.conv2dPrepared(ctx, &e.w, &e.prep, &e.b, .{ stride, stride }, .{ pad, pad }, 1, .{ .h, .w, .c });
}

fn conv3(ctx: *ExecContext, model: *Model, x: *const Map, wn: []const u8, bn: []const u8, do_relu: bool) !Map {
    return conv(ctx, model, x, wn, bn, 1, 1, do_relu);
}

/// Residual block: conv3(relu) → conv3 → +identity → relu.
fn resBlock(ctx: *ExecContext, model: *Model, x: *const Map, w1: []const u8, b1: []const u8, w2: []const u8, b2: []const u8) !Map {
    var y = try conv3(ctx, model, x, w1, b1, true);
    defer y.deinit();
    var y2 = try conv3(ctx, model, &y, w2, b2, false);
    defer y2.deinit();
    var s = try y2.add(ctx, x);
    defer s.deinit();
    return s.relu(ctx);
}

/// Downsample block: conv(s2,relu) → conv3 → +[avgpool→conv1×1] shortcut → relu.
fn downBlock(ctx: *ExecContext, model: *Model, x: *const Map, w1: []const u8, b1: []const u8, w2: []const u8, b2: []const u8, scw: []const u8, scb: []const u8) !Map {
    var y = try conv(ctx, model, x, w1, b1, 2, 1, true);
    defer y.deinit();
    var y2 = try conv3(ctx, model, &y, w2, b2, false);
    defer y2.deinit();
    var sc = try nn.avgPool2x2(ctx, x);
    defer sc.deinit();
    var sc2 = try conv(ctx, model, &sc, scw, scb, 1, 0, false);
    defer sc2.deinit();
    var s = try y2.add(ctx, &sc2);
    defer s.deinit();
    return s.relu(ctx);
}

pub const Heads = struct {
    score: [3][]f32,
    bbox: [3][]f32,
    kps: [3][]f32,
    feat_w: [3]usize,
    feat_h: [3]usize,

    pub fn deinit(self: *Heads, al: std.mem.Allocator) void {
        for (0..3) |i| {
            al.free(self.score[i]);
            al.free(self.bbox[i]);
            al.free(self.kps[i]);
        }
    }
};

const StrideHead = struct {
    sw: [3][]const u8,
    sb: [3][]const u8,
    cls_w: []const u8,
    cls_b: []const u8,
    reg_w: []const u8,
    reg_b: []const u8,
    kps_w: []const u8,
    kps_b: []const u8,
    scale: []const u8,
};

const stride_heads = [3]StrideHead{
    .{ .sw = .{ "det.667", "det.671", "det.675" }, .sb = .{ "det.669", "det.673", "det.677" }, .cls_w = "det.bbox_head.stride_cls.(8, 8).weight", .cls_b = "det.bbox_head.stride_cls.(8, 8).bias", .reg_w = "det.bbox_head.stride_reg.(8, 8).weight", .reg_b = "det.bbox_head.stride_reg.(8, 8).bias", .kps_w = "det.bbox_head.stride_kps.(8, 8).weight", .kps_b = "det.bbox_head.stride_kps.(8, 8).bias", .scale = "det.bbox_head.scales.0.scale" },
    .{ .sw = .{ "det.679", "det.683", "det.687" }, .sb = .{ "det.681", "det.685", "det.689" }, .cls_w = "det.bbox_head.stride_cls.(16, 16).weight", .cls_b = "det.bbox_head.stride_cls.(16, 16).bias", .reg_w = "det.bbox_head.stride_reg.(16, 16).weight", .reg_b = "det.bbox_head.stride_reg.(16, 16).bias", .kps_w = "det.bbox_head.stride_kps.(16, 16).weight", .kps_b = "det.bbox_head.stride_kps.(16, 16).bias", .scale = "det.bbox_head.scales.1.scale" },
    .{ .sw = .{ "det.691", "det.695", "det.699" }, .sb = .{ "det.693", "det.697", "det.701" }, .cls_w = "det.bbox_head.stride_cls.(32, 32).weight", .cls_b = "det.bbox_head.stride_cls.(32, 32).bias", .reg_w = "det.bbox_head.stride_reg.(32, 32).weight", .reg_b = "det.bbox_head.stride_reg.(32, 32).bias", .kps_w = "det.bbox_head.stride_kps.(32, 32).weight", .kps_b = "det.bbox_head.stride_kps.(32, 32).bias", .scale = "det.bbox_head.scales.2.scale" },
};

fn ownedData(al: std.mem.Allocator, m: *const Map) ![]f32 {
    const d = try m.dataConst();
    const out = try al.alloc(f32, d.len);
    @memcpy(out, d);
    return out;
}

/// Run one stride head (feat → 3 stem convs → cls/reg/kps), writing the flattened
/// score (sigmoid), bbox (·scale) and kps into `out` at index `i`.
fn runHead(ctx: *ExecContext, al: std.mem.Allocator, model: *Model, feat: *const Map, hd: StrideHead, i: usize, out: *Heads) !void {
    var h0 = try conv3(ctx, model, feat, hd.sw[0], hd.sb[0], true);
    defer h0.deinit();
    var h1 = try conv3(ctx, model, &h0, hd.sw[1], hd.sb[1], true);
    defer h1.deinit();
    var h2 = try conv3(ctx, model, &h1, hd.sw[2], hd.sb[2], true);
    defer h2.deinit();

    out.feat_h[i] = h2.dim(.h);
    out.feat_w[i] = h2.dim(.w);

    var cls = try conv3(ctx, model, &h2, hd.cls_w, hd.cls_b, false);
    defer cls.deinit();
    var score = try cls.sigmoid(ctx);
    defer score.deinit();
    out.score[i] = try ownedData(al, &score);

    var reg = try conv3(ctx, model, &h2, hd.reg_w, hd.reg_b, false);
    defer reg.deinit();
    const sv = (try rec.toF32(al, try rec.info(model.file, hd.scale)));
    defer al.free(sv);
    var bbox = try reg.scale(ctx, sv[0]);
    defer bbox.deinit();
    out.bbox[i] = try ownedData(al, &bbox);

    var kps = try conv3(ctx, model, &h2, hd.kps_w, hd.kps_b, false);
    defer kps.deinit();
    out.kps[i] = try ownedData(al, &kps);
}

/// SCRFD forward on the 640² blob image (FDR1) → the 9 raw heads. One-shot
/// (loads + tears down a Model; repeated callers hold a `Model`).
pub fn forward(ctx: *ExecContext, al: std.mem.Allocator, io: std.Io, file: *const gguf.File, blob_path: []const u8) !Heads {
    const bytes = try rec.readFile(io, al, blob_path);
    defer al.free(bytes);
    var img = try image.fromRaw(al, bytes);
    defer img.deinit();
    return forwardImage(ctx, al, file, &img);
}

/// One-shot SCRFD forward on an in-memory 640² RGB blob image.
pub fn forwardImage(ctx: *ExecContext, al: std.mem.Allocator, file: *const gguf.File, img: *const image.Image) !Heads {
    var model = Model.init(al, file);
    defer model.deinit();
    return forwardImageWith(ctx, al, &model, img);
}

/// SCRFD forward using a caller-held `Model` (weights cached across calls).
pub fn forwardImageWith(ctx: *ExecContext, al: std.mem.Allocator, model: *Model, img: *const image.Image) !Heads {
    const buf = try al.alloc(f32, img.width * img.height * 3);
    defer al.free(buf);
    for (img.pixels, 0..) |px, i| buf[i] = (@as(f32, @floatFromInt(px)) - 127.5) / 128.0;
    var x0 = try fucina.Tensor(.{ .h, .w, .c }).fromSlice(ctx, .{ img.height, img.width, 3 }, buf);
    defer x0.deinit();

    // Stem: conv s2 → conv3 → conv3 → maxpool 2×2.
    var st0 = try conv(ctx, model, &x0, "det.547", "det.549", 2, 1, true);
    defer st0.deinit();
    var st1 = try conv3(ctx, model, &st0, "det.551", "det.553", true);
    defer st1.deinit();
    var st2 = try conv3(ctx, model, &st1, "det.555", "det.557", true);
    defer st2.deinit();
    var p = try nn.maxPool2x2(ctx, &st2);
    defer p.deinit();

    // layer1 (56ch, 3 residual blocks) → c2.
    var l1a = try resBlock(ctx, model, &p, "det.559", "det.561", "det.563", "det.565");
    defer l1a.deinit();
    var l1b = try resBlock(ctx, model, &l1a, "det.567", "det.569", "det.571", "det.573");
    defer l1b.deinit();
    var c2 = try resBlock(ctx, model, &l1b, "det.575", "det.577", "det.579", "det.581");
    defer c2.deinit();

    // layer2 (88ch): down + 3 res → c3 (stride 8).
    var s8a = try downBlock(ctx, model, &c2, "det.583", "det.585", "det.587", "det.589", "det.591", "det.593");
    defer s8a.deinit();
    var s8b = try resBlock(ctx, model, &s8a, "det.595", "det.597", "det.599", "det.601");
    defer s8b.deinit();
    var s8c = try resBlock(ctx, model, &s8b, "det.603", "det.605", "det.607", "det.609");
    defer s8c.deinit();
    var c3 = try resBlock(ctx, model, &s8c, "det.611", "det.613", "det.615", "det.617");
    defer c3.deinit();

    // layer3 (88ch): down + 1 res → c4 (stride 16).
    var s16a = try downBlock(ctx, model, &c3, "det.619", "det.621", "det.623", "det.625", "det.627", "det.629");
    defer s16a.deinit();
    var c4 = try resBlock(ctx, model, &s16a, "det.631", "det.633", "det.635", "det.637");
    defer c4.deinit();

    // layer4 (224ch): down + 2 res → c5 (stride 32).
    var s32a = try downBlock(ctx, model, &c4, "det.639", "det.641", "det.643", "det.645", "det.647", "det.649");
    defer s32a.deinit();
    var s32b = try resBlock(ctx, model, &s32a, "det.651", "det.653", "det.655", "det.657");
    defer s32b.deinit();
    var c5 = try resBlock(ctx, model, &s32b, "det.659", "det.661", "det.663", "det.665");
    defer c5.deinit();

    // PAFPN neck: laterals (1×1) → top-down (upsample+add) → fpn → bottom-up → pafpn.
    var l0 = try conv(ctx, model, &c3, "det.neck.lateral_convs.0.conv.weight", "det.neck.lateral_convs.0.conv.bias", 1, 0, false);
    defer l0.deinit();
    var la1 = try conv(ctx, model, &c4, "det.neck.lateral_convs.1.conv.weight", "det.neck.lateral_convs.1.conv.bias", 1, 0, false);
    defer la1.deinit();
    var la2 = try conv(ctx, model, &c5, "det.neck.lateral_convs.2.conv.weight", "det.neck.lateral_convs.2.conv.bias", 1, 0, false);
    defer la2.deinit();

    var up = try nn.upsample2xNearest(ctx, &la2);
    defer up.deinit();
    var p1m = try la1.add(ctx, &up);
    defer p1m.deinit();
    var up2 = try nn.upsample2xNearest(ctx, &p1m);
    defer up2.deinit();
    var p0m = try l0.add(ctx, &up2);
    defer p0m.deinit();

    // fpn_convs.1/.2 carry NO own bias — the ONNX export feeds them
    // downsample_convs.0/.1's bias. Match exactly.
    var f0 = try conv3(ctx, model, &p0m, "det.neck.fpn_convs.0.conv.weight", "det.neck.fpn_convs.0.conv.bias", false);
    defer f0.deinit();
    var f1 = try conv3(ctx, model, &p1m, "det.neck.fpn_convs.1.conv.weight", "det.neck.downsample_convs.0.conv.bias", false);
    defer f1.deinit();
    var f2 = try conv3(ctx, model, &la2, "det.neck.fpn_convs.2.conv.weight", "det.neck.downsample_convs.1.conv.bias", false);
    defer f2.deinit();

    var d0 = try conv(ctx, model, &f0, "det.neck.downsample_convs.0.conv.weight", "det.neck.downsample_convs.0.conv.bias", 2, 1, false);
    defer d0.deinit();
    var n1 = try f1.add(ctx, &d0);
    defer n1.deinit();
    var d1 = try conv(ctx, model, &n1, "det.neck.downsample_convs.1.conv.weight", "det.neck.downsample_convs.1.conv.bias", 2, 1, false);
    defer d1.deinit();
    var n2 = try f2.add(ctx, &d1);
    defer n2.deinit();

    var pa0 = try conv3(ctx, model, &n1, "det.neck.pafpn_convs.0.conv.weight", "det.neck.pafpn_convs.0.conv.bias", false);
    defer pa0.deinit();
    var pa1 = try conv3(ctx, model, &n2, "det.neck.pafpn_convs.1.conv.weight", "det.neck.pafpn_convs.1.conv.bias", false);
    defer pa1.deinit();

    // 3 stride heads on (f0, pa0, pa1).
    var out: Heads = undefined;
    const feats = [3]*const Map{ &f0, &pa0, &pa1 };
    for (0..3) |i| try runHead(ctx, al, model, feats[i], stride_heads[i], i, &out);
    return out;
}
