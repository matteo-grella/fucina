//! ArcFace IResNet-50 (w600k_r50) recognizer forward — mirrors the C++/ggml
//! `arcface_embed` (refs/face-detect.cpp `src/arcface_graph.cpp`). Channel-last
//! `Tensor(.{ .h, .w, .c })` feature maps, composed from the `nn.zig`
//! primitives (conv2d + channelAffine + prelu).
//!
//! Key fold convention (from the reference): the standalone BatchNorms are only
//! `bn1` (per IR block), `bn2` (head), and `features` (head BN1d). Every other
//! BN — the stem BN, each block's bn2/bn3, and the downsample BN — is PRE-FOLDED
//! by the converter into the *bias* of the preceding conv (with its weight
//! already scaled by γ/√(var+ε)). So those convs carry their BN for free.
//!
//! Weights load ONCE into `Model` (GGUF dequant + layout repack + BN fold are
//! load-time work, mirroring the reference's load-once graph lifecycle); every
//! `Model.embedImage` call is pure compute.

const std = @import("std");
const fucina = @import("fucina");
const nn = @import("nn.zig");
const image = @import("image.zig");

const gguf = fucina.gguf;
const ExecContext = fucina.ExecContext;
const Map = nn.Map;
const Channels = nn.Channels;
const ConvW = fucina.Tensor(.{ .oc, .kh, .kw, .c });
const FcW = fucina.Tensor(.{ .out, .in });
const FcVec = fucina.Tensor(.{.out});

const bn_eps: f32 = 1e-5; // kRecBnEps

// --- weight loading (GGUF `rec.*` → Fucina tensors) ------------------------

pub fn info(file: *const gguf.File, name: []const u8) !gguf.TensorInfo {
    for (file.tensors) |t| {
        if (std.mem.eql(u8, t.name, name)) return t;
    }
    return error.MissingTensor;
}

/// Read a whole file into owned bytes via the std.Io model (the repo's file API).
pub fn readFile(io: std.Io, allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    var handle = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer handle.close(io);
    const stat = try handle.stat(io);
    const len: usize = @intCast(stat.size);
    const bytes = try allocator.alloc(u8, len);
    errdefer allocator.free(bytes);
    var read_len: usize = 0;
    while (read_len < bytes.len) {
        const n = try handle.readStreaming(io, &.{bytes[read_len..]});
        if (n == 0) return error.EndOfStream;
        read_len += n;
    }
    return bytes;
}

/// Dequantize a tensor's raw little-endian bytes to owned f32 (F32 verbatim,
/// F16 widened). Alignment-safe (reads via readInt, GGUF mmap isn't aligned).
pub fn toF32(allocator: std.mem.Allocator, t: gguf.TensorInfo) ![]f32 {
    switch (t.ggml_type) {
        .f32 => {
            const n = t.data.len / 4;
            const out = try allocator.alloc(f32, n);
            for (0..n) |i| out[i] = @bitCast(std.mem.readInt(u32, t.data[i * 4 ..][0..4], .little));
            return out;
        },
        .f16 => {
            const n = t.data.len / 2;
            const out = try allocator.alloc(f32, n);
            for (0..n) |i| out[i] = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, t.data[i * 2 ..][0..2], .little))));
            return out;
        },
        else => return error.UnsupportedDtype,
    }
}

/// 1-D vector tensor (BN param / conv bias / PReLU slope, all `[c]` logically).
pub fn loadVec(ctx: *ExecContext, allocator: std.mem.Allocator, file: *const gguf.File, name: []const u8) !Channels {
    const t = try info(file, name);
    const data = try toF32(allocator, t);
    defer allocator.free(data);
    return fucina.Tensor(.{.c}).fromSlice(ctx, .{data.len}, data);
}

/// Conv weight: GGUF ggml order `[kw,kh,cin,cout]` → Fucina `[cout,kh,kw,cin]`.
pub fn loadConvW(ctx: *ExecContext, allocator: std.mem.Allocator, file: *const gguf.File, name: []const u8) !ConvW {
    const t = try info(file, name);
    const raw = try toF32(allocator, t); // ggml flat: ((co*CIN+ci)*KH+kh)*KW+kw
    defer allocator.free(raw);
    const kw = t.dims[0];
    const kh = t.dims[1];
    const cin = t.dims[2];
    const cout = t.dims[3];
    const buf = try allocator.alloc(f32, cout * kh * kw * cin);
    defer allocator.free(buf);
    for (0..cout) |co| {
        for (0..kh) |y| {
            for (0..kw) |x| {
                for (0..cin) |ci| {
                    buf[((co * kh + y) * kw + x) * cin + ci] = raw[((co * cin + ci) * kh + y) * kw + x];
                }
            }
        }
    }
    return ConvW.fromSlice(ctx, .{ cout, kh, kw, cin }, buf);
}

/// FC weight `rec.fc.weight`: ggml `[in=25088(CHW), out=512]` → Fucina
/// `[.out, .in]` with the input dim re-indexed from the reference's NCHW flatten
/// (c·49 + h·7 + w) to Fucina's HWC `merge(.h,.w,.c)` order ((h·7+w)·512 + c).
fn loadFcW(ctx: *ExecContext, allocator: std.mem.Allocator, file: *const gguf.File) !FcW {
    const t = try info(file, "rec.fc.weight");
    const raw = try toF32(allocator, t); // ggml flat: in_chw + IN*out
    defer allocator.free(raw);
    const OUT: usize = 512;
    const IN: usize = 25088;
    const C: usize = 512;
    const H: usize = 7;
    const W: usize = 7;
    const buf = try allocator.alloc(f32, OUT * IN);
    defer allocator.free(buf);
    for (0..OUT) |o| {
        for (0..C) |c| {
            for (0..H) |h| {
                for (0..W) |w| {
                    const in_chw = c * (H * W) + h * W + w;
                    const in_hwc = (h * W + w) * C + c;
                    buf[o * IN + in_hwc] = raw[in_chw + IN * o];
                }
            }
        }
    }
    return FcW.fromSlice(ctx, .{ OUT, IN }, buf);
}

/// Load + fold a standalone BatchNorm2d by tensor-name prefix (`<p>.weight/
/// .bias/.running_mean/.running_var`) into a per-channel (scale, shift) pair.
fn loadBnFold(ctx: *ExecContext, allocator: std.mem.Allocator, file: *const gguf.File, prefix: []const u8, eps: f32) !nn.BnScaleShift {
    var buf: [128]u8 = undefined;
    var g = try loadVec(ctx, allocator, file, try std.fmt.bufPrint(&buf, "{s}.weight", .{prefix}));
    defer g.deinit();
    var b = try loadVec(ctx, allocator, file, try std.fmt.bufPrint(&buf, "{s}.bias", .{prefix}));
    defer b.deinit();
    var m = try loadVec(ctx, allocator, file, try std.fmt.bufPrint(&buf, "{s}.running_mean", .{prefix}));
    defer m.deinit();
    var v = try loadVec(ctx, allocator, file, try std.fmt.bufPrint(&buf, "{s}.running_var", .{prefix}));
    defer v.deinit();
    return nn.bnFold(ctx, &g, &b, &m, &v, eps);
}

// --- IR block table (arcface_graph.cpp:39-68) ------------------------------

const Block = struct {
    stage: u8,
    idx: u8,
    c1w: []const u8,
    c1b: []const u8,
    pa: []const u8,
    c2w: []const u8,
    c2b: []const u8,
    dsw: ?[]const u8 = null,
    dsb: ?[]const u8 = null,
    stride: usize,
};

const blocks = [_]Block{
    .{ .stage = 1, .idx = 0, .c1w = "rec.688", .c1b = "rec.689", .pa = "rec.844", .c2w = "rec.691", .c2b = "rec.692", .dsw = "rec.694", .dsb = "rec.695", .stride = 2 },
    .{ .stage = 1, .idx = 1, .c1w = "rec.697", .c1b = "rec.698", .pa = "rec.845", .c2w = "rec.700", .c2b = "rec.701", .stride = 1 },
    .{ .stage = 1, .idx = 2, .c1w = "rec.703", .c1b = "rec.704", .pa = "rec.846", .c2w = "rec.706", .c2b = "rec.707", .stride = 1 },
    .{ .stage = 2, .idx = 0, .c1w = "rec.709", .c1b = "rec.710", .pa = "rec.847", .c2w = "rec.712", .c2b = "rec.713", .dsw = "rec.715", .dsb = "rec.716", .stride = 2 },
    .{ .stage = 2, .idx = 1, .c1w = "rec.718", .c1b = "rec.719", .pa = "rec.848", .c2w = "rec.721", .c2b = "rec.722", .stride = 1 },
    .{ .stage = 2, .idx = 2, .c1w = "rec.724", .c1b = "rec.725", .pa = "rec.849", .c2w = "rec.727", .c2b = "rec.728", .stride = 1 },
    .{ .stage = 2, .idx = 3, .c1w = "rec.730", .c1b = "rec.731", .pa = "rec.850", .c2w = "rec.733", .c2b = "rec.734", .stride = 1 },
    .{ .stage = 3, .idx = 0, .c1w = "rec.736", .c1b = "rec.737", .pa = "rec.851", .c2w = "rec.739", .c2b = "rec.740", .dsw = "rec.742", .dsb = "rec.743", .stride = 2 },
    .{ .stage = 3, .idx = 1, .c1w = "rec.745", .c1b = "rec.746", .pa = "rec.852", .c2w = "rec.748", .c2b = "rec.749", .stride = 1 },
    .{ .stage = 3, .idx = 2, .c1w = "rec.751", .c1b = "rec.752", .pa = "rec.853", .c2w = "rec.754", .c2b = "rec.755", .stride = 1 },
    .{ .stage = 3, .idx = 3, .c1w = "rec.757", .c1b = "rec.758", .pa = "rec.854", .c2w = "rec.760", .c2b = "rec.761", .stride = 1 },
    .{ .stage = 3, .idx = 4, .c1w = "rec.763", .c1b = "rec.764", .pa = "rec.855", .c2w = "rec.766", .c2b = "rec.767", .stride = 1 },
    .{ .stage = 3, .idx = 5, .c1w = "rec.769", .c1b = "rec.770", .pa = "rec.856", .c2w = "rec.772", .c2b = "rec.773", .stride = 1 },
    .{ .stage = 3, .idx = 6, .c1w = "rec.775", .c1b = "rec.776", .pa = "rec.857", .c2w = "rec.778", .c2b = "rec.779", .stride = 1 },
    .{ .stage = 3, .idx = 7, .c1w = "rec.781", .c1b = "rec.782", .pa = "rec.858", .c2w = "rec.784", .c2b = "rec.785", .stride = 1 },
    .{ .stage = 3, .idx = 8, .c1w = "rec.787", .c1b = "rec.788", .pa = "rec.859", .c2w = "rec.790", .c2b = "rec.791", .stride = 1 },
    .{ .stage = 3, .idx = 9, .c1w = "rec.793", .c1b = "rec.794", .pa = "rec.860", .c2w = "rec.796", .c2b = "rec.797", .stride = 1 },
    .{ .stage = 3, .idx = 10, .c1w = "rec.799", .c1b = "rec.800", .pa = "rec.861", .c2w = "rec.802", .c2b = "rec.803", .stride = 1 },
    .{ .stage = 3, .idx = 11, .c1w = "rec.805", .c1b = "rec.806", .pa = "rec.862", .c2w = "rec.808", .c2b = "rec.809", .stride = 1 },
    .{ .stage = 3, .idx = 12, .c1w = "rec.811", .c1b = "rec.812", .pa = "rec.863", .c2w = "rec.814", .c2b = "rec.815", .stride = 1 },
    .{ .stage = 3, .idx = 13, .c1w = "rec.817", .c1b = "rec.818", .pa = "rec.864", .c2w = "rec.820", .c2b = "rec.821", .stride = 1 },
    .{ .stage = 4, .idx = 0, .c1w = "rec.823", .c1b = "rec.824", .pa = "rec.865", .c2w = "rec.826", .c2b = "rec.827", .dsw = "rec.829", .dsb = "rec.830", .stride = 2 },
    .{ .stage = 4, .idx = 1, .c1w = "rec.832", .c1b = "rec.833", .pa = "rec.866", .c2w = "rec.835", .c2b = "rec.836", .stride = 1 },
    .{ .stage = 4, .idx = 2, .c1w = "rec.838", .c1b = "rec.839", .pa = "rec.867", .c2w = "rec.841", .c2b = "rec.842", .stride = 1 },
};

/// All rec.* weights, loaded/repacked/BN-folded once; forwards are pure compute.
pub const Model = struct {
    stem_w: ConvW,
    stem_b: Channels,
    stem_prep: fucina.PreparedConvWeights,
    stem_alpha: Channels,
    blocks: [blocks.len]LoadedBlock,
    bn2: nn.BnScaleShift,
    fcw: FcW,
    fcb: FcVec,
    features_scale: FcVec,
    features_shift: FcVec,

    const LoadedBlock = struct {
        bn1: nn.BnScaleShift,
        w1: ConvW,
        b1: Channels,
        prep1: fucina.PreparedConvWeights,
        alpha: Channels,
        w2: ConvW,
        b2: Channels,
        prep2: fucina.PreparedConvWeights,
        ds_w: ?ConvW,
        ds_b: ?Channels,
        stride: usize,
    };

    pub fn load(ctx: *ExecContext, allocator: std.mem.Allocator, file: *const gguf.File) !Model {
        var self: Model = undefined;

        self.stem_w = try loadConvW(ctx, allocator, file, "rec.685");
        errdefer self.stem_w.deinit();
        self.stem_b = try loadVec(ctx, allocator, file, "rec.686");
        errdefer self.stem_b.deinit();
        // Naturally `.empty`: the stem's cin = 3 is below the Winograd gate.
        self.stem_prep = try self.stem_w.prepareConv2dWeights(ctx);
        errdefer self.stem_prep.deinit();
        self.stem_alpha = try loadVec(ctx, allocator, file, "rec.843");
        errdefer self.stem_alpha.deinit();

        var loaded: usize = 0;
        errdefer for (self.blocks[0..loaded]) |*b| deinitBlock(b);
        for (blocks, 0..) |blk, i| {
            var pbuf: [64]u8 = undefined;
            const prefix = try std.fmt.bufPrint(&pbuf, "rec.layer{d}.{d}.bn1", .{ blk.stage, blk.idx });
            var bn1 = try loadBnFold(ctx, allocator, file, prefix, bn_eps);
            errdefer bn1.deinit();
            var w1 = try loadConvW(ctx, allocator, file, blk.c1w);
            errdefer w1.deinit();
            var b1 = try loadVec(ctx, allocator, file, blk.c1b);
            errdefer b1.deinit();
            var prep1 = try w1.prepareConv2dWeights(ctx);
            errdefer prep1.deinit();
            var alpha = try loadVec(ctx, allocator, file, blk.pa);
            errdefer alpha.deinit();
            var w2 = try loadConvW(ctx, allocator, file, blk.c2w);
            errdefer w2.deinit();
            var b2 = try loadVec(ctx, allocator, file, blk.c2b);
            errdefer b2.deinit();
            // conv2 runs at the block's stride; Winograd (and thus the
            // prepared planes) only applies at stride 1.
            var prep2: fucina.PreparedConvWeights = if (blk.stride == 1) try w2.prepareConv2dWeights(ctx) else .empty;
            errdefer prep2.deinit();
            var ds_w: ?ConvW = if (blk.dsw) |n| try loadConvW(ctx, allocator, file, n) else null;
            errdefer if (ds_w) |*w| w.deinit();
            const ds_b: ?Channels = if (blk.dsb) |n| try loadVec(ctx, allocator, file, n) else null;
            self.blocks[i] = .{ .bn1 = bn1, .w1 = w1, .b1 = b1, .prep1 = prep1, .alpha = alpha, .w2 = w2, .b2 = b2, .prep2 = prep2, .ds_w = ds_w, .ds_b = ds_b, .stride = blk.stride };
            loaded = i + 1;
        }

        self.bn2 = try loadBnFold(ctx, allocator, file, "rec.bn2", bn_eps);
        errdefer self.bn2.deinit();
        self.fcw = try loadFcW(ctx, allocator, file);
        errdefer self.fcw.deinit();
        self.fcb = try loadFcVec(ctx, allocator, file, "rec.fc.bias");
        errdefer self.fcb.deinit();
        const fss = try loadFeaturesBn(ctx, allocator, file);
        self.features_scale = fss.scale;
        self.features_shift = fss.shift;
        return self;
    }

    fn deinitBlock(b: *LoadedBlock) void {
        b.bn1.deinit();
        b.w1.deinit();
        b.b1.deinit();
        b.prep1.deinit();
        b.alpha.deinit();
        b.w2.deinit();
        b.b2.deinit();
        b.prep2.deinit();
        if (b.ds_w) |*w| w.deinit();
        if (b.ds_b) |*bb| bb.deinit();
        b.* = undefined;
    }

    pub fn deinit(self: *Model) void {
        self.stem_w.deinit();
        self.stem_b.deinit();
        self.stem_prep.deinit();
        self.stem_alpha.deinit();
        for (&self.blocks) |*b| deinitBlock(b);
        self.bn2.deinit();
        self.fcw.deinit();
        self.fcb.deinit();
        self.features_scale.deinit();
        self.features_shift.deinit();
        self.* = undefined;
    }

    /// One IBasicBlock: bn1 → conv1(3×3,s1,p1) → prelu → conv2(3×3,s=stride,p1)
    /// → + downsample(1×1,s=stride,p0 on the raw input) → residual add.
    fn irBlock(ctx: *ExecContext, x: *const Map, blk: *const LoadedBlock) !Map {
        var y0 = try x.channelAffine(ctx, &blk.bn1.scale, &blk.bn1.shift);
        defer y0.deinit();
        var y1 = try y0.conv2dPrepared(ctx, &blk.w1, &blk.prep1, &blk.b1, .{ 1, 1 }, .{ 1, 1 }, 1, .{ .h, .w, .c });
        defer y1.deinit();
        var y2 = try y1.prelu(ctx, &blk.alpha);
        defer y2.deinit();
        var y3 = try y2.conv2dPrepared(ctx, &blk.w2, &blk.prep2, &blk.b2, .{ blk.stride, blk.stride }, .{ 1, 1 }, 1, .{ .h, .w, .c });
        defer y3.deinit();

        if (blk.ds_w) |*dw| {
            var idn = try x.conv2d(ctx, dw, &blk.ds_b.?, .{ blk.stride, blk.stride }, .{ 0, 0 }, 1, .{ .h, .w, .c });
            defer idn.deinit();
            return y3.add(ctx, &idn);
        }
        return y3.add(ctx, x);
    }

    /// IResNet-50 embed on an in-memory 112² RGB crop → the raw 512-d feature
    /// vector (caller L2-normalizes / cosines).
    pub fn embedImage(self: *const Model, ctx: *ExecContext, allocator: std.mem.Allocator, img: *const image.Image) ![]f32 {
        // Crop → normalized [112,112,3] map: (pixel − 127.5) / 127.5, RGB HWC.
        const np = img.width * img.height * 3;
        const pbuf = try allocator.alloc(f32, np);
        defer allocator.free(pbuf);
        for (img.pixels, 0..) |p, i| pbuf[i] = (@as(f32, @floatFromInt(p)) - 127.5) / 127.5;
        var x0 = try fucina.Tensor(.{ .h, .w, .c }).fromSlice(ctx, .{ img.height, img.width, 3 }, pbuf);
        defer x0.deinit();

        // Stem: conv(3→64,k3,s1,p1)+folded-BN bias → PReLU.
        var sc = try x0.conv2dPrepared(ctx, &self.stem_w, &self.stem_prep, &self.stem_b, .{ 1, 1 }, .{ 1, 1 }, 1, .{ .h, .w, .c });
        defer sc.deinit();
        var cur = try sc.prelu(ctx, &self.stem_alpha);

        // 24 IR blocks.
        for (&self.blocks) |*blk| {
            const nx = try irBlock(ctx, &cur, blk);
            cur.deinit();
            cur = nx;
        }
        defer cur.deinit();

        // Head: bn2 → flatten(HWC) → FC → BN1d(features).
        var bn2 = try cur.channelAffine(ctx, &self.bn2.scale, &self.bn2.shift);
        defer bn2.deinit();
        var flat = try bn2.merge(ctx, .in, .{ .h, .w, .c }); // [25088]
        defer flat.deinit();
        var logits = try flat.dot(ctx, &self.fcw, .in); // [512]
        defer logits.deinit();
        var biased = try logits.add(ctx, &self.fcb);
        defer biased.deinit();
        var feat = try biased.channelAffine(ctx, &self.features_scale, &self.features_shift);
        defer feat.deinit();

        const data = try feat.dataConst();
        const out = try allocator.alloc(f32, data.len);
        @memcpy(out, data);
        return out;
    }
};

fn loadFcVec(ctx: *ExecContext, allocator: std.mem.Allocator, file: *const gguf.File, name: []const u8) !FcVec {
    const data = try toF32(allocator, try info(file, name));
    defer allocator.free(data);
    return FcVec.fromSlice(ctx, .{data.len}, data);
}

/// BatchNorm1d over the 512-d feature vector (`rec.features.*`), eps = 1e-5,
/// folded to a `[.out]` (scale, shift) pair at load.
fn loadFeaturesBn(ctx: *ExecContext, allocator: std.mem.Allocator, file: *const gguf.File) !struct { scale: FcVec, shift: FcVec } {
    var g = try loadFcVec(ctx, allocator, file, "rec.features.weight");
    defer g.deinit();
    var b = try loadFcVec(ctx, allocator, file, "rec.features.bias");
    defer b.deinit();
    var m = try loadFcVec(ctx, allocator, file, "rec.features.running_mean");
    defer m.deinit();
    var v = try loadFcVec(ctx, allocator, file, "rec.features.running_var");
    defer v.deinit();
    var ve = try v.addScalar(ctx, bn_eps);
    defer ve.deinit();
    var sd = try ve.sqrt(ctx);
    defer sd.deinit();
    var scale = try g.div(ctx, &sd);
    errdefer scale.deinit();
    var ms = try m.mul(ctx, &scale);
    defer ms.deinit();
    const shift = try b.sub(ctx, &ms);
    return .{ .scale = scale, .shift = shift };
}

/// One-shot embed from a crop file: load the model, forward once
/// (repeated-forward callers hold a `Model`).
pub fn embed(ctx: *ExecContext, allocator: std.mem.Allocator, io: std.Io, file: *const gguf.File, crop_path: []const u8) ![]f32 {
    const bytes = try readFile(io, allocator, crop_path);
    defer allocator.free(bytes);
    var img = try image.fromRaw(allocator, bytes);
    defer img.deinit();
    return embedImage(ctx, allocator, file, &img);
}

/// One-shot embed on an in-memory crop (model loaded and torn down inside).
pub fn embedImage(ctx: *ExecContext, allocator: std.mem.Allocator, file: *const gguf.File, img: *const image.Image) ![]f32 {
    var model = try Model.load(ctx, allocator, file);
    defer model.deinit();
    return model.embedImage(ctx, allocator, img);
}
