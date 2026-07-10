//! GenderAge MobileNet-0.25 head — mirrors the C++/ggml `genderage_forward`
//! (refs/face-detect.cpp `src/genderage_graph.cpp`). Reuses
//! the recognizer's GGUF loaders; genderage-specific bits: BN eps = 1e-3 (mxnet
//! export, NOT ArcFace's 1e-5), bias-less convs with SEPARATE (unfolded) BN,
//! depthwise-separable blocks (groups = C), the `_gamma/_beta/_moving_{mean,var}`
//! BN naming, and the built-in input normalize `(x − op1)·op2`.
//!
//! Weights load ONCE into `Model` — a lazy by-name cache (conv weights repack,
//! BN stats fold to per-channel scale/shift on first use; names are built at
//! runtime, so the cache owns its keys).

const std = @import("std");
const fucina = @import("fucina");
const nn = @import("nn.zig");
const image = @import("image.zig");
const rec = @import("recognizer.zig");

const gguf = fucina.gguf;
const ExecContext = fucina.ExecContext;
const Map = nn.Map;
const FcOut = fucina.Tensor(.{.out});
const FcW = fucina.Tensor(.{ .out, .c });

const ga_eps: f32 = 1e-3; // kGaBnEps (mxnet export)

/// Session-lifetime genderage weight cache. Borrows `file`; owns its name keys
/// (they are runtime-formatted).
pub const Model = struct {
    allocator: std.mem.Allocator,
    file: *const gguf.File,
    convs: std.StringHashMapUnmanaged(ConvEntry) = .empty,
    bns: std.StringHashMapUnmanaged(nn.BnScaleShift) = .empty,

    const ConvW = fucina.Tensor(.{ .oc, .kh, .kw, .c });

    const ConvEntry = struct {
        w: ConvW,
        prep: fucina.PreparedConvWeights,
    };

    pub fn init(allocator: std.mem.Allocator, file: *const gguf.File) Model {
        return .{ .allocator = allocator, .file = file };
    }

    pub fn deinit(self: *Model) void {
        var cit = self.convs.iterator();
        while (cit.next()) |e| {
            self.allocator.free(e.key_ptr.*);
            e.value_ptr.w.deinit();
            e.value_ptr.prep.deinit();
        }
        self.convs.deinit(self.allocator);
        var bit = self.bns.iterator();
        while (bit.next()) |e| {
            self.allocator.free(e.key_ptr.*);
            e.value_ptr.deinit();
        }
        self.bns.deinit(self.allocator);
        self.* = undefined;
    }

    /// Conv weight (+ Winograd planes for stride-1 sites — `.empty` for the
    /// shapes this net actually has: 1×1 pointwise, depthwise cin/g = 1, and
    /// the cin = 3 stem never take the Winograd route), loaded on first use.
    fn convW(self: *Model, ctx: *ExecContext, name: []const u8, stride: usize) !*const ConvEntry {
        if (self.convs.getPtr(name)) |v| return v;
        const key = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(key);
        var w = try rec.loadConvW(ctx, self.allocator, self.file, name);
        errdefer w.deinit();
        var prep: fucina.PreparedConvWeights = if (stride == 1) try w.prepareConv2dWeights(ctx) else .empty;
        errdefer prep.deinit();
        try self.convs.put(self.allocator, key, .{ .w = w, .prep = prep });
        return self.convs.getPtr(key).?;
    }

    /// Folded BN (scale, shift) for a `<prefix>_gamma/_beta/_moving_*` node.
    fn bn(self: *Model, ctx: *ExecContext, prefix: []const u8) !*const nn.BnScaleShift {
        if (self.bns.getPtr(prefix)) |v| return v;
        const key = try self.allocator.dupe(u8, prefix);
        errdefer self.allocator.free(key);
        var ss = try loadGaBnFold(ctx, self.allocator, self.file, prefix);
        errdefer ss.deinit();
        try self.bns.put(self.allocator, key, ss);
        return self.bns.getPtr(key).?;
    }
};

fn loadGaBnFold(ctx: *ExecContext, allocator: std.mem.Allocator, file: *const gguf.File, prefix: []const u8) !nn.BnScaleShift {
    var buf: [128]u8 = undefined;
    var g = try rec.loadVec(ctx, allocator, file, try std.fmt.bufPrint(&buf, "{s}_gamma", .{prefix}));
    defer g.deinit();
    var b = try rec.loadVec(ctx, allocator, file, try std.fmt.bufPrint(&buf, "{s}_beta", .{prefix}));
    defer b.deinit();
    var m = try rec.loadVec(ctx, allocator, file, try std.fmt.bufPrint(&buf, "{s}_moving_mean", .{prefix}));
    defer m.deinit();
    var v = try rec.loadVec(ctx, allocator, file, try std.fmt.bufPrint(&buf, "{s}_moving_var", .{prefix}));
    defer v.deinit();
    return nn.bnFold(ctx, &g, &b, &m, &v, ga_eps);
}

fn loadScalar(allocator: std.mem.Allocator, file: *const gguf.File, name: []const u8) !f32 {
    const data = try rec.toF32(allocator, try rec.info(file, name));
    defer allocator.free(data);
    return data[0];
}

/// Standard conv (groups=1, bias-less) → BN (folded affine) → ReLU.
fn convBnRelu(ctx: *ExecContext, model: *Model, x: *const Map, conv_w: []const u8, bn_prefix: []const u8, stride: usize, pad: usize) !Map {
    const e = try model.convW(ctx, conv_w, stride);
    var c = try x.conv2dPrepared(ctx, &e.w, &e.prep, null, .{ stride, stride }, .{ pad, pad }, 1, .{ .h, .w, .c });
    defer c.deinit();
    const ss = try model.bn(ctx, bn_prefix);
    var a = try c.channelAffine(ctx, &ss.scale, &ss.shift);
    defer a.deinit();
    return a.relu(ctx);
}

/// Depthwise conv (groups = C, bias-less) → BN → ReLU.
fn dwConvBnRelu(ctx: *ExecContext, model: *Model, x: *const Map, conv_w: []const u8, bn_prefix: []const u8, stride: usize, pad: usize) !Map {
    const groups = x.dim(.c);
    const e = try model.convW(ctx, conv_w, stride);
    var c = try x.conv2dPrepared(ctx, &e.w, &e.prep, null, .{ stride, stride }, .{ pad, pad }, groups, .{ .h, .w, .c });
    defer c.deinit();
    const ss = try model.bn(ctx, bn_prefix);
    var a = try c.channelAffine(ctx, &ss.scale, &ss.shift);
    defer a.deinit();
    return a.relu(ctx);
}

/// One depthwise-separable block conv_N: dw(k3,s=dw_stride,p1) then pw(1×1,s1,p0).
fn sepBlock(ctx: *ExecContext, model: *Model, x: *const Map, n: usize, dw_stride: usize) !Map {
    var nbuf: [64]u8 = undefined;
    var pbuf: [64]u8 = undefined;
    const dw_w = try std.fmt.bufPrint(&nbuf, "ga.conv_{d}_dw_conv2d_weight", .{n});
    const dw_bn = try std.fmt.bufPrint(&pbuf, "ga.conv_{d}_dw_batchnorm", .{n});
    var dw = try dwConvBnRelu(ctx, model, x, dw_w, dw_bn, dw_stride, 1);
    defer dw.deinit();
    const pw_w = try std.fmt.bufPrint(&nbuf, "ga.conv_{d}_conv2d_weight", .{n});
    const pw_bn = try std.fmt.bufPrint(&pbuf, "ga.conv_{d}_batchnorm", .{n});
    return convBnRelu(ctx, model, &dw, pw_w, pw_bn, 1, 0);
}

fn loadGaFc(ctx: *ExecContext, allocator: std.mem.Allocator, file: *const gguf.File, name: []const u8) !FcW {
    const t = try rec.info(file, name);
    const data = try rec.toF32(allocator, t); // ggml [in, out], in-fastest == Fucina [out, in]
    defer allocator.free(data);
    const in = t.dims[0];
    const out = t.dims[1];
    return FcW.fromSlice(ctx, .{ out, in }, data);
}

/// One task branch (t0 gender / t1 age): conv_13_t*(dw s2 + pw) → conv_14_t*(dw
/// s1 + pw) → global-avg-pool → FC. Returns the FC output `[.out]`.
fn taskBranch(ctx: *ExecContext, allocator: std.mem.Allocator, model: *Model, trunk: *const Map, comptime tag: []const u8, comptime fc: []const u8) !FcOut {
    var x1 = try dwConvBnRelu(ctx, model, trunk, "ga.conv_13_dw_" ++ tag ++ "_conv2d_weight", "ga.conv_13_dw_" ++ tag ++ "_batchnorm", 2, 1);
    defer x1.deinit();
    var x2 = try convBnRelu(ctx, model, &x1, "ga.conv_13_" ++ tag ++ "_conv2d_weight", "ga.conv_13_" ++ tag ++ "_batchnorm", 1, 0);
    defer x2.deinit();
    var x3 = try dwConvBnRelu(ctx, model, &x2, "ga.conv_14_dw_" ++ tag ++ "_conv2d_weight", "ga.conv_14_dw_" ++ tag ++ "_batchnorm", 1, 1);
    defer x3.deinit();
    var x4 = try convBnRelu(ctx, model, &x3, "ga.conv_14_" ++ tag ++ "_conv2d_weight", "ga.conv_14_" ++ tag ++ "_batchnorm", 1, 0);
    defer x4.deinit();

    // Global average pool over the whole spatial extent → [.c].
    var ph = try x4.mean(ctx, .h);
    defer ph.deinit();
    var pooled = try ph.mean(ctx, .w); // [.c]
    defer pooled.deinit();

    var fcw = try loadGaFc(ctx, allocator, model.file, fc ++ "_weight"); // [.out, .c]
    defer fcw.deinit();
    var logits = try pooled.dot(ctx, &fcw, .c); // [.out]
    defer logits.deinit();
    const bdata = try rec.toF32(allocator, try rec.info(model.file, fc ++ "_bias"));
    defer allocator.free(bdata);
    var fcb = try FcOut.fromSlice(ctx, .{bdata.len}, bdata);
    defer fcb.deinit();
    return logits.add(ctx, &fcb);
}

/// Full genderage forward → `[g0, g1, age_raw]`. One-shot (model built inside).
pub fn forward(ctx: *ExecContext, allocator: std.mem.Allocator, io: std.Io, file: *const gguf.File, crop_path: []const u8) ![3]f32 {
    const bytes = try rec.readFile(io, allocator, crop_path);
    defer allocator.free(bytes);
    var img = try image.fromRaw(allocator, bytes);
    defer img.deinit();
    return forwardImage(ctx, allocator, file, &img);
}

/// One-shot genderage forward on an in-memory 96² RGB crop.
pub fn forwardImage(ctx: *ExecContext, allocator: std.mem.Allocator, file: *const gguf.File, img: *const image.Image) ![3]f32 {
    var model = Model.init(allocator, file);
    defer model.deinit();
    return forwardImageWith(ctx, allocator, &model, img);
}

/// GenderAge forward using a caller-held `Model` → `[g0, g1, age_raw]`.
pub fn forwardImageWith(ctx: *ExecContext, allocator: std.mem.Allocator, model: *Model, img: *const image.Image) ![3]f32 {
    // Raw [0,255] RGB crop — the net does its own normalization.
    const np = img.width * img.height * 3;
    const pbuf = try allocator.alloc(f32, np);
    defer allocator.free(pbuf);
    for (img.pixels, 0..) |p, i| pbuf[i] = @floatFromInt(p);
    var x0 = try fucina.Tensor(.{ .h, .w, .c }).fromSlice(ctx, .{ img.height, img.width, 3 }, pbuf);
    defer x0.deinit();

    // Built-in normalize: (x − scalar_op1) · scalar_op2.
    const s1 = try loadScalar(allocator, model.file, "ga.scalar_op1");
    const s2 = try loadScalar(allocator, model.file, "ga.scalar_op2");
    var xn = try x0.subScalar(ctx, s1);
    defer xn.deinit();
    var xs = try xn.scale(ctx, s2);
    defer xs.deinit();

    // Stem conv_1 (3→16, k3 s2 p1) + BN + ReLU.
    var cur = try convBnRelu(ctx, model, &xs, "ga.conv_1_conv2d_weight", "ga.conv_1_batchnorm", 2, 1);

    // conv_2..conv_12 sep blocks; dw stride 2 at conv_3/5/7.
    const dw_stride = [_]usize{ 0, 0, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1 };
    var n: usize = 2;
    while (n <= 12) : (n += 1) {
        const nx = try sepBlock(ctx, model, &cur, n, dw_stride[n]);
        cur.deinit();
        cur = nx;
    }
    defer cur.deinit();

    var gender = try taskBranch(ctx, allocator, model, &cur, "t0", "ga.fullyconnected0");
    defer gender.deinit();
    var age = try taskBranch(ctx, allocator, model, &cur, "t1", "ga.fullyconnected1");
    defer age.deinit();

    const gd = try gender.dataConst();
    const ad = try age.dataConst();
    return .{ gd[0], gd[1], ad[0] };
}

pub const Result = struct { gender: u8, age: i32 }; // gender byte: 'M' / 'F'

/// Decode: gender = argmax(g0,g1) → 'M' if g1>g0 else 'F'; age = round(raw·100).
pub fn analyze(ctx: *ExecContext, allocator: std.mem.Allocator, io: std.Io, file: *const gguf.File, crop_path: []const u8) !Result {
    const o = try forward(ctx, allocator, io, file, crop_path);
    return decode(o);
}

pub fn analyzeImage(ctx: *ExecContext, allocator: std.mem.Allocator, file: *const gguf.File, img: *const image.Image) !Result {
    return decode(try forwardImage(ctx, allocator, file, img));
}

pub fn analyzeImageWith(ctx: *ExecContext, allocator: std.mem.Allocator, model: *Model, img: *const image.Image) !Result {
    return decode(try forwardImageWith(ctx, allocator, model, img));
}

fn decode(o: [3]f32) Result {
    return .{
        .gender = if (o[1] > o[0]) 'M' else 'F',
        .age = @intFromFloat(@round(o[2] * 100.0)),
    };
}
