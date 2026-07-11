//! Behavioral tests for the public autograd tensor facade (`ag/tensor.zig`).
//! Split out of tensor.zig so the implementation file stays navigable; these
//! tests exercise only the public surface (plus `core.GradState`).
const std = @import("std");
const backend_mod = @import("../backend.zig");
const dtype_mod = @import("../dtype.zig");
const exec_mod = @import("../exec.zig");
const control = @import("control.zig");
const core = @import("core.zig");
const ag_tensor = @import("tensor.zig");
const gradcheck_mod = @import("gradcheck.zig");

const DType = dtype_mod.DType;
const ExecContext = exec_mod.ExecContext;
const GradState = core.GradState;
const Tensor = ag_tensor.Tensor;
const RawTensor = @import("../tensor.zig").Tensor;

test "tagged autograd exposes Tensor facade operations" {
    const T = Tensor(.{ .batch, .d });
    try std.testing.expect(T.dtype == .f32);
    const expected = .{
        "withTags",
        "fromSlice",
        "variableFromSlice",
        "asRawTensor",
        "detach",
        "requiresGrad",
        "backward",
        "backwardWithGrad",
        "grad",
        "shape",
        "to",
        "materialize",
        "alignTo",
        "permuteTo",
        "transpose",
        "insertAxis",
        "squeeze",
        "split",
        "merge",
        "broadcastTo",
        "add",
        "sub",
        "mul",
        "div",
        "causalDepthwiseConv1d",
        "groupedCausalConv1d",
        "gated",
        "glu",
        "swiglu",
        "splitGated",
        "unary",
        "relu",
        "exp",
        "sqrt",
        "rsqrt",
        "sigmoid",
        "silu",
        "log",
        "neg",
        "abs",
        "sin",
        "cos",
        "tanh",
        "fastTanh",
        "gelu",
        "quickGelu",
        "clamp",
        "sum",
        "mean",
        "variance",
        "standardizeAxis",
        "sumAll",
        "sumMany",
        "flatten",
        "gather",
        "narrow",
        "concat",
        "setSlice",
        "setRows",
        "argmax",
        "max",
        "min",
        "topK",
        "softmax",
        "rmsNorm",
        "layerNorm",
        "crossEntropy",
        "crossEntropyExt",
        "rope",
        "dot",
        "floor",
        "ceil",
        "round",
        "sign",
        "reciprocal",
        "maximum",
        "minimum",
        "pow",
        "isnan",
        "isinf",
        "isfinite",
        "any",
        "all",
        "prod",
        "cumprod",
        "norm",
        "logsumexp",
        "logSoftmax",
        "reshape",
        "sliceStep",
        "diagonal",
        "diag",
        "trace",
        "nonzero",
        "indexAdd",
        "takeAlongAxis",
        "scatterAdd",
        "scatter",
        "arange",
        "linspace",
        "oneHot",
        "rand",
        "randn",
        "bernoulli",
    };
    inline for (expected) |name| {
        if (!@hasDecl(T, name)) @compileError("missing tagged autograd op: " ++ name);
    }
}

test "public Tensor accepts non-f32 dtype specs for token tensors" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const TokenIds = Tensor(.{ .dtype = .u16, .tags = .{ .batch, .seq } });
    try std.testing.expect(TokenIds.dtype == .u16);
    try std.testing.expect(!@hasDecl(TokenIds, "add"));

    var ids = try TokenIds.fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer ids.deinit();

    try std.testing.expect(!ids.requiresGrad());
    try std.testing.expectEqual(@as(usize, 2), ids.dim(.batch));
    try std.testing.expectEqual(@as(usize, 3), ids.dim(.seq));
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, ids.asRawTensor().shape.slice());
    try std.testing.expectEqualSlices(u16, &.{ 1, 2, 3, 4, 5, 6 }, ids.asRawTensor().dataConst());
}

test "public non-float Tensor excludes autograd at comptime" {
    inline for (.{ DType.bool, DType.u16, DType.i64 }) |non_f32_dtype| {
        const T = Tensor(.{ .dtype = non_f32_dtype, .tags = .{ .batch, .d } });
        if (@hasDecl(T, "variable")) @compileError("non-float Tensor exposes variable");
        if (@hasDecl(T, "variableFromSlice")) @compileError("non-float Tensor exposes variableFromSlice");
        if (@hasDecl(T, "backward")) @compileError("non-float Tensor exposes backward");
        if (@hasDecl(T, "grad")) @compileError("non-float Tensor exposes grad");
    }
}

test "public 16-bit float Tensor exposes leaf autograd but never backward" {
    // f16/bf16 tensors can be trainable LEAVES (f32 gradients); they are
    // never losses, so the backward entry points stay f32-only. The f64
    // decls exist but are comptime errors on instantiation (f64 training
    // is unsupported — gradients are always f32).
    inline for (.{ DType.bf16, DType.f16, DType.f64 }) |float_dtype| {
        const T = Tensor(.{ .dtype = float_dtype, .tags = .{ .batch, .d } });
        inline for (.{ "variable", "variableFromSlice", "grad", "gradView", "zeroGrad", "detach", "requiresGrad" }) |decl_name| {
            if (!@hasDecl(T, decl_name)) @compileError("typed float Tensor missing " ++ decl_name);
        }
        if (@hasDecl(T, "backward")) @compileError("typed float Tensor exposes backward");
        if (@hasDecl(T, "backwardWithGrad")) @compileError("typed float Tensor exposes backwardWithGrad");
    }
}

test "public integer and bool Tensor excludes float math at comptime" {
    inline for (.{ DType.bool, DType.u16, DType.i64 }) |non_float_dtype| {
        comptime expectNoFloatMath(non_float_dtype);
    }
}

test "public block-quantized Tensor exposes only quantized operations" {
    inline for (.{ DType.q1_0, DType.q4_0, DType.q4_1, DType.q5_0, DType.q5_1, DType.q8_0, DType.q8_1, DType.q2_k, DType.q3_k, DType.q4_k, DType.q5_k, DType.q6_k, DType.q8_k, DType.iq1_s, DType.iq1_m, DType.iq2_xxs, DType.iq2_xs, DType.iq2_s, DType.iq3_xxs, DType.iq3_s, DType.iq4_nl, DType.iq4_xs, DType.tq1_0, DType.tq2_0, DType.mxfp4, DType.nvfp4 }) |quant_dtype| {
        const Q = Tensor(.{ .dtype = quant_dtype, .tags = .{ .out, .in } });
        try std.testing.expect(Q.dtype == quant_dtype);
        if (!@hasDecl(Q, "fromBlocks")) @compileError("quantized Tensor missing fromBlocks");
        if (!@hasDecl(Q, "to")) @compileError("quantized Tensor missing to");
        if (!@hasDecl(Q, "materialize")) @compileError("quantized Tensor missing materialize");
        if (!@hasDecl(Q, "concat")) @compileError("quantized Tensor missing concat");
        if (!@hasDecl(Q, "getRows")) @compileError("quantized Tensor missing getRows");
        if (@hasDecl(Q, "add")) @compileError("quantized Tensor exposes add");
        if (@hasDecl(Q, "softmax")) @compileError("quantized Tensor exposes softmax");
        if (@hasDecl(Q, "variable")) @compileError("quantized Tensor exposes autograd variable");
    }
}

test "public q8_0 Tensor dequantizes and gathers rows" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const Q = Tensor(.{ .dtype = .q8_0, .tags = .{ .out, .in } });
    var blocks = [_]dtype_mod.BlockQ8_0{
        .{ .d = f16TestBits(1), .qs = [_]i8{1} ** dtype_mod.q8_0_block_size },
        .{ .d = f16TestBits(2), .qs = [_]i8{3} ** dtype_mod.q8_0_block_size },
    };

    var q = try Q.fromBlocks(&ctx, .{ 2, dtype_mod.q8_0_block_size }, &blocks);
    defer q.deinit();

    var dense = try q.to(&ctx, .f32);
    defer dense.deinit();
    try std.testing.expectEqual(@as(f32, 1), dense.asRawTensor().dataConst()[0]);
    try std.testing.expectEqual(@as(f32, 6), dense.asRawTensor().dataConst()[dtype_mod.q8_0_block_size]);

    var row = try q.getRows(&ctx, .out, &.{1}, .batch);
    defer row.deinit();
    try std.testing.expectEqual(@as(f32, 6), row.asRawTensor().dataConst()[0]);
    try std.testing.expectEqual(@as(usize, 1), row.dim(.batch));
    try std.testing.expectEqual(@as(usize, dtype_mod.q8_0_block_size), row.dim(.in));

    var joined = try q.concat(&ctx, .out, &.{&q});
    defer joined.deinit();
    try std.testing.expectEqual(@as(usize, 4), joined.dim(.out));
    try std.testing.expectEqual(@as(usize, dtype_mod.q8_0_block_size), joined.dim(.in));

    var joined_dense = try joined.to(&ctx, .f32);
    defer joined_dense.deinit();
    try std.testing.expectEqual(@as(f32, 1), joined_dense.asRawTensor().dataConst()[0]);
    try std.testing.expectEqual(@as(f32, 6), joined_dense.asRawTensor().dataConst()[dtype_mod.q8_0_block_size]);
    try std.testing.expectEqual(@as(f32, 1), joined_dense.asRawTensor().dataConst()[2 * dtype_mod.q8_0_block_size]);
}

test "public q8_1 Tensor dequantizes and gathers rows but is not a matmul RHS dtype" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const Q = Tensor(.{ .dtype = .q8_1, .tags = .{ .out, .in } });
    var blocks = [_]dtype_mod.BlockQ8_1{
        .{ .ds = .{ f16TestBits(1), f16TestBits(@floatFromInt(dtype_mod.q8_1_block_size)) }, .qs = [_]i8{1} ** dtype_mod.q8_1_block_size },
        .{ .ds = .{ f16TestBits(2), f16TestBits(@floatFromInt(6 * dtype_mod.q8_1_block_size)) }, .qs = [_]i8{3} ** dtype_mod.q8_1_block_size },
    };

    var q = try Q.fromBlocks(&ctx, .{ 2, dtype_mod.q8_1_block_size }, &blocks);
    defer q.deinit();

    try std.testing.expect(!dtype_mod.supportsQuantizedMatmulRhs(.q8_1));

    var dense = try q.to(&ctx, .f32);
    defer dense.deinit();
    try std.testing.expectEqual(@as(f32, 1), dense.asRawTensor().dataConst()[0]);
    try std.testing.expectEqual(@as(f32, 6), dense.asRawTensor().dataConst()[dtype_mod.q8_1_block_size]);

    var row = try q.getRows(&ctx, .out, &.{1}, .batch);
    defer row.deinit();
    try std.testing.expectEqual(@as(f32, 6), row.asRawTensor().dataConst()[0]);
    try std.testing.expectEqual(@as(usize, 1), row.dim(.batch));
    try std.testing.expectEqual(@as(usize, dtype_mod.q8_1_block_size), row.dim(.in));
}

test "public q8_k Tensor dequantizes and gathers rows but is not a matmul RHS dtype" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const Q = Tensor(.{ .dtype = .q8_k, .tags = .{ .out, .in } });
    var blocks = [_]dtype_mod.BlockQ8_K{
        .{ .d = 1, .qs = [_]i8{1} ** dtype_mod.qk_k_block_size, .bsums = [_]i16{16} ** (dtype_mod.qk_k_block_size / 16) },
        .{ .d = 2, .qs = [_]i8{3} ** dtype_mod.qk_k_block_size, .bsums = [_]i16{48} ** (dtype_mod.qk_k_block_size / 16) },
    };

    var q = try Q.fromBlocks(&ctx, .{ 2, dtype_mod.qk_k_block_size }, &blocks);
    defer q.deinit();

    try std.testing.expect(!dtype_mod.supportsQuantizedMatmulRhs(.q8_k));

    var dense = try q.to(&ctx, .f32);
    defer dense.deinit();
    try std.testing.expectEqual(@as(f32, 1), dense.asRawTensor().dataConst()[0]);
    try std.testing.expectEqual(@as(f32, 6), dense.asRawTensor().dataConst()[dtype_mod.qk_k_block_size]);

    var row = try q.getRows(&ctx, .out, &.{1}, .batch);
    defer row.deinit();
    try std.testing.expectEqual(@as(f32, 6), row.asRawTensor().dataConst()[0]);
    try std.testing.expectEqual(@as(usize, 1), row.dim(.batch));
    try std.testing.expectEqual(@as(usize, dtype_mod.qk_k_block_size), row.dim(.in));
}

test "public f32 Tensor dot dispatches to quantized RHS Tensor" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const X = Tensor(.{ .dtype = .f32, .tags = .{ .batch, .in } });
    const W = Tensor(.{ .dtype = .q8_0, .tags = .{ .out, .in } });

    var x_values = [_]f32{1} ** dtype_mod.q8_0_block_size;
    var x = try X.fromSlice(&ctx, .{ 1, dtype_mod.q8_0_block_size }, &x_values);
    defer x.deinit();

    var blocks = [_]dtype_mod.BlockQ8_0{
        .{ .d = f16TestBits(1), .qs = [_]i8{1} ** dtype_mod.q8_0_block_size },
        .{ .d = f16TestBits(1), .qs = [_]i8{2} ** dtype_mod.q8_0_block_size },
    };
    var w = try W.fromBlocks(&ctx, .{ 2, dtype_mod.q8_0_block_size }, &blocks);
    defer w.deinit();

    var y = try x.dot(&ctx, &w, .in);
    defer y.deinit();

    try std.testing.expectEqual(@as(usize, 1), y.dim(.batch));
    try std.testing.expectEqual(@as(usize, 2), y.dim(.out));
    try std.testing.expectApproxEqAbs(@as(f32, 32), y.asRawTensor().dataConst()[0], 1e-2);
    try std.testing.expectApproxEqAbs(@as(f32, 64), y.asRawTensor().dataConst()[1], 1e-2);
}

test "public f32 Tensor dot dispatches to quantized RHS Tensor with multiple left free axes" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const X = Tensor(.{ .dtype = .f32, .tags = .{ .batch, .seq, .in } });
    const W = Tensor(.{ .dtype = .q8_0, .tags = .{ .out, .in } });

    var x_values = [_]f32{1} ** (2 * dtype_mod.q8_0_block_size);
    var x = try X.fromSlice(&ctx, .{ 2, 1, dtype_mod.q8_0_block_size }, &x_values);
    defer x.deinit();

    var blocks = [_]dtype_mod.BlockQ8_0{
        .{ .d = f16TestBits(1), .qs = [_]i8{1} ** dtype_mod.q8_0_block_size },
        .{ .d = f16TestBits(1), .qs = [_]i8{2} ** dtype_mod.q8_0_block_size },
    };
    var w = try W.fromBlocks(&ctx, .{ 2, dtype_mod.q8_0_block_size }, &blocks);
    defer w.deinit();

    var y = try x.dot(&ctx, &w, .in);
    defer y.deinit();

    try std.testing.expectEqual(@as(usize, 2), y.dim(.batch));
    try std.testing.expectEqual(@as(usize, 1), y.dim(.seq));
    try std.testing.expectEqual(@as(usize, 2), y.dim(.out));
    for (y.asRawTensor().dataConst(), &[_]f32{ 32, 64, 32, 64 }) |got, expected| {
        try std.testing.expectApproxEqAbs(expected, got, 1e-2);
    }
}

test "public f32 Tensor dot dispatches to legacy q4_1 RHS Tensor" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const X = Tensor(.{ .dtype = .f32, .tags = .{ .batch, .in } });
    const W = Tensor(.{ .dtype = .q4_1, .tags = .{ .out, .in } });

    var x_values = [_]f32{1} ** dtype_mod.q4_1_block_size;
    var x = try X.fromSlice(&ctx, .{ 1, dtype_mod.q4_1_block_size }, &x_values);
    defer x.deinit();

    var blocks = [_]dtype_mod.BlockQ4_1{
        .{ .dm = .{ f16TestBits(1), f16TestBits(0) }, .qs = [_]u8{0x11} ** (dtype_mod.q4_1_block_size / 2) },
        .{ .dm = .{ f16TestBits(1), f16TestBits(0) }, .qs = [_]u8{0x22} ** (dtype_mod.q4_1_block_size / 2) },
    };
    var w = try W.fromBlocks(&ctx, .{ 2, dtype_mod.q4_1_block_size }, &blocks);
    defer w.deinit();

    var y = try x.dot(&ctx, &w, .in);
    defer y.deinit();

    try std.testing.expectEqual(@as(usize, 1), y.dim(.batch));
    try std.testing.expectEqual(@as(usize, 2), y.dim(.out));
    try std.testing.expectApproxEqAbs(@as(f32, 32), y.asRawTensor().dataConst()[0], 1e-1);
    try std.testing.expectApproxEqAbs(@as(f32, 64), y.asRawTensor().dataConst()[1], 1e-1);
}

test "public f32 Tensor dot dispatches to K-quant RHS Tensor" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const X = Tensor(.{ .dtype = .f32, .tags = .{ .batch, .in } });
    const W = Tensor(.{ .dtype = .q4_k, .tags = .{ .out, .in } });

    var x_values = [_]f32{1} ** dtype_mod.qk_k_block_size;
    var x = try X.fromSlice(&ctx, .{ 1, dtype_mod.qk_k_block_size }, &x_values);
    defer x.deinit();

    var blocks = [_]dtype_mod.BlockQ4_K{.{
        .dm = [_]u16{ 0, 0 },
        .scales = [_]u8{0} ** dtype_mod.k_scale_size,
        .qs = [_]u8{0} ** (dtype_mod.qk_k_block_size / 2),
    }};
    var w = try W.fromBlocks(&ctx, .{ 1, dtype_mod.qk_k_block_size }, &blocks);
    defer w.deinit();

    var y = try x.dot(&ctx, &w, .in);
    defer y.deinit();

    try std.testing.expectEqual(@as(usize, 1), y.dim(.batch));
    try std.testing.expectEqual(@as(usize, 1), y.dim(.out));
    try std.testing.expectEqual(@as(f32, 0), y.asRawTensor().dataConst()[0]);
}

test "public packed Q6_Kx4 RHS batched rows match stacked single rows" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const m = 13;
    const n = 12;
    const k = 512;
    const blocks_per_row = k / dtype_mod.qk_k_block_size;

    const X = Tensor(.{ .dtype = .f32, .tags = .{ .batch, .in } });
    const W = Tensor(.{ .dtype = .q6_k, .tags = .{ .out, .in } });

    const x_values = try allocator.alloc(f32, m * k);
    defer allocator.free(x_values);
    for (x_values, 0..) |*v, i| {
        const signed: i32 = @as(i32, @intCast((i * 17 + 11) % 251)) - 125;
        v.* = @as(f32, @floatFromInt(signed)) / 11.0;
    }
    var x = try X.fromSlice(&ctx, .{ m, k }, x_values);
    defer x.deinit();

    const blocks = try allocator.alloc(dtype_mod.BlockQ6_K, n * blocks_per_row);
    defer allocator.free(blocks);
    for (blocks, 0..) |*b, bi| {
        b.d = f16TestBits(0.03 + 0.001 * @as(f32, @floatFromInt(bi % 5)));
        for (&b.scales, 0..) |*s, i| s.* = @intCast(@as(i32, @intCast((i * 5 + bi * 3) % 64)) - 32);
        for (&b.ql, 0..) |*q, i| q.* = @intCast((i * 31 + bi * 5) % 256);
        for (&b.qh, 0..) |*q, i| q.* = @intCast((i * 13 + bi * 11) % 256);
    }
    var w = try W.fromBlocks(&ctx, .{ n, k }, blocks);
    defer w.deinit();
    var packed_rhs = try w.packRhs(&ctx);
    defer packed_rhs.deinit();

    var batched = try x.dotPacked(&ctx, &packed_rhs, .in, .out);
    defer batched.deinit();
    const batched_data = batched.asRawTensor().dataConst();

    for (0..m) |row| {
        var one = try x.narrow(&ctx, .batch, row, 1);
        defer one.deinit();
        var single = try one.dotPacked(&ctx, &packed_rhs, .in, .out);
        defer single.deinit();
        for (single.asRawTensor().dataConst(), 0..) |expected, col| {
            try expectPackedClose(expected, batched_data[row * n + col]);
        }
    }
}

test "public packed Q8_0x4 RHS batched rows match stacked single rows" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const m = 13;
    const n = 12;
    const k = 64;
    const blocks_per_row = k / dtype_mod.q8_0_block_size;

    const X = Tensor(.{ .dtype = .f32, .tags = .{ .batch, .in } });
    const W = Tensor(.{ .dtype = .q8_0, .tags = .{ .out, .in } });

    const x_values = try allocator.alloc(f32, m * k);
    defer allocator.free(x_values);
    for (x_values, 0..) |*v, i| {
        const signed: i32 = @as(i32, @intCast((i * 19 + 7) % 251)) - 125;
        v.* = @as(f32, @floatFromInt(signed)) / 9.0;
    }
    var x = try X.fromSlice(&ctx, .{ m, k }, x_values);
    defer x.deinit();

    const blocks = try allocator.alloc(dtype_mod.BlockQ8_0, n * blocks_per_row);
    defer allocator.free(blocks);
    for (blocks, 0..) |*b, bi| {
        b.d = f16TestBits(0.025 + 0.001 * @as(f32, @floatFromInt(bi % 7)));
        for (&b.qs, 0..) |*q, i| q.* = @intCast(@as(i32, @intCast((i * 23 + bi * 5) % 255)) - 127);
    }
    var w = try W.fromBlocks(&ctx, .{ n, k }, blocks);
    defer w.deinit();
    var packed_rhs = try w.packRhs(&ctx);
    defer packed_rhs.deinit();

    var batched = try x.dotPacked(&ctx, &packed_rhs, .in, .out);
    defer batched.deinit();
    const batched_data = batched.asRawTensor().dataConst();

    for (0..m) |row| {
        var one = try x.narrow(&ctx, .batch, row, 1);
        defer one.deinit();
        var single = try one.dotPacked(&ctx, &packed_rhs, .in, .out);
        defer single.deinit();
        for (single.asRawTensor().dataConst(), 0..) |expected, col| {
            try expectPackedClose(expected, batched_data[row * n + col]);
        }
    }
}

test "public splitSwiGlu packed Q8_0x4 RHS dot matches unfused path" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // m=13: LHS quantization scratch fits the 512-block stack fast path;
    // m=1027: 257 row groups x 2 blocks/row = 514 blocks, crossing into the
    // pooled ScratchLease fallback.
    for ([_]usize{ 13, 1027 }) |m| {
        const n = 12;
        const k = 64;
        const blocks_per_row = k / dtype_mod.q8_0_block_size;

        const GU = Tensor(.{ .dtype = .f32, .tags = .{ .batch, .gate_up } });
        const W = Tensor(.{ .dtype = .q8_0, .tags = .{ .out, .in } });

        const gate_up_values = try allocator.alloc(f32, m * k * 2);
        defer allocator.free(gate_up_values);
        for (gate_up_values, 0..) |*v, i| {
            const signed: i32 = @as(i32, @intCast((i * 31 + 11) % 257)) - 128;
            v.* = @as(f32, @floatFromInt(signed)) / 13.0;
        }
        var gate_up = try GU.fromSlice(&ctx, .{ m, k * 2 }, gate_up_values);
        defer gate_up.deinit();

        const blocks = try allocator.alloc(dtype_mod.BlockQ8_0, n * blocks_per_row);
        defer allocator.free(blocks);
        for (blocks, 0..) |*b, bi| {
            b.d = f16TestBits(0.02 + 0.001 * @as(f32, @floatFromInt(bi % 9)));
            for (&b.qs, 0..) |*q, i| q.* = @intCast(@as(i32, @intCast((i * 17 + bi * 13) % 255)) - 127);
        }
        var w = try W.fromBlocks(&ctx, .{ n, k }, blocks);
        defer w.deinit();
        var packed_rhs = try w.packRhs(&ctx);
        defer packed_rhs.deinit();

        var activated = try gate_up.splitGated(&ctx, .swiglu, .gate_up, .in);
        defer activated.deinit();
        var expected = try activated.dotPacked(&ctx, &packed_rhs, .in, .out);
        defer expected.deinit();

        var fused = try gate_up.splitSwiGluDotPacked(&ctx, &packed_rhs, .gate_up, .out);
        defer fused.deinit();

        const expected_data = expected.asRawTensor().dataConst();
        const fused_data = fused.asRawTensor().dataConst();
        for (expected_data, fused_data) |e, actual| {
            try expectPackedClose(e, actual);
        }
    }
}

test "public rmsNormMul packed RHS dot matches the unfused pair" {
    // rmsNormMulDotPacked normalizes into task-private scratch with the
    // exact kernels the unfused rmsNormMul dispatch uses (rows kernel /
    // scalar loop by the same threshold); the LHS quantizer ARRANGEMENT can
    // differ from the packed matmul's internal one at the last ulp, so the
    // pin is the packed-route tolerance (the splitSwiGluDotPacked
    // precedent), across q8_0x4 and K-quant (x4-group + per-row tail)
    // routes and both dispatch regimes.
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const eps: f32 = 1e-6;
    inline for (.{ .q8_0, .q4_k }) |wdtype| {
        const k = if (wdtype == .q8_0) 64 else 2 * dtype_mod.qk_k_block_size;
        const n = if (wdtype == .q8_0) 12 else 16; // Q4_Kx8 packs 8 columns per group
        for ([_]usize{ 1, 5, 13, 130 }) |m| {
            const X = Tensor(.{ .dtype = .f32, .tags = .{ .batch, .in } });
            const NW = Tensor(.{ .dtype = .f32, .tags = .{.in} });
            const W = Tensor(.{ .dtype = wdtype, .tags = .{ .out, .in } });

            const x_values = try allocator.alloc(f32, m * k);
            defer allocator.free(x_values);
            for (x_values, 0..) |*v, i| {
                const signed: i32 = @as(i32, @intCast((i * 37 + 5) % 251)) - 125;
                v.* = @as(f32, @floatFromInt(signed)) / 17.0;
            }
            var x = try X.fromSlice(&ctx, .{ m, k }, x_values);
            defer x.deinit();

            const nw_values = try allocator.alloc(f32, k);
            defer allocator.free(nw_values);
            for (nw_values, 0..) |*v, i| v.* = 0.5 + @as(f32, @floatFromInt(i % 7)) * 0.25;
            var norm_w = try NW.fromSlice(&ctx, .{k}, nw_values);
            defer norm_w.deinit();

            var w = switch (wdtype) {
                .q8_0 => blk: {
                    const blocks_per_row = k / dtype_mod.q8_0_block_size;
                    const blocks = try allocator.alloc(dtype_mod.BlockQ8_0, n * blocks_per_row);
                    defer allocator.free(blocks);
                    for (blocks, 0..) |*b, bi| {
                        b.d = f16TestBits(0.02 + 0.001 * @as(f32, @floatFromInt(bi % 9)));
                        for (&b.qs, 0..) |*q, i| q.* = @intCast(@as(i32, @intCast((i * 17 + bi * 13) % 255)) - 127);
                    }
                    break :blk try W.fromBlocks(&ctx, .{ n, k }, blocks);
                },
                .q4_k => blk: {
                    const blocks_per_row = k / dtype_mod.qk_k_block_size;
                    const blocks = try allocator.alloc(dtype_mod.BlockQ4_K, n * blocks_per_row);
                    defer allocator.free(blocks);
                    for (blocks, 0..) |*b, bi| {
                        b.dm = .{ f16TestBits(0.011 + 0.0007 * @as(f32, @floatFromInt(bi % 5))), f16TestBits(0.003) };
                        for (&b.scales, 0..) |*sc, i| sc.* = @intCast((i * 11 + bi * 7) % 64);
                        for (&b.qs, 0..) |*q, i| q.* = @intCast((i * 29 + bi * 3) % 256);
                    }
                    break :blk try W.fromBlocks(&ctx, .{ n, k }, blocks);
                },
                else => unreachable,
            };
            defer w.deinit();
            var packed_rhs = try w.packRhs(&ctx);
            defer packed_rhs.deinit();

            var normed = try x.rmsNormMul(&ctx, .in, &norm_w, eps);
            defer normed.deinit();
            var expected = try normed.dotPacked(&ctx, &packed_rhs, .in, .out);
            defer expected.deinit();

            var fused = try x.rmsNormMulDotPacked(&ctx, &norm_w, eps, &packed_rhs, .in, .out);
            defer fused.deinit();

            for (expected.asRawTensor().dataConst(), fused.asRawTensor().dataConst()) |e, actual| {
                try expectPackedClose(e, actual);
            }
        }
    }
}

fn expectPackedClose(expected: f32, actual: f32) !void {
    const tol = 1e-3 * @max(@as(f32, 1), @abs(expected));
    try std.testing.expect(@abs(expected - actual) <= tol);
}

fn expectNoFloatMath(comptime non_float_dtype: DType) void {
    const T = Tensor(.{ .dtype = non_float_dtype, .tags = .{ .batch, .d } });
    const forbidden = .{
        "to",
        "add",
        "sub",
        "mul",
        "div",
        "gated",
        "glu",
        "swiglu",
        "unary",
        "relu",
        "exp",
        "sqrt",
        "rsqrt",
        "sigmoid",
        "silu",
        "log",
        "neg",
        "abs",
        "sin",
        "cos",
        "tanh",
        "gelu",
        "quickGelu",
        "clamp",
        "sum",
        "mean",
        "variance",
        "sumAll",
        "sumMany",
        "flatten",
        "argmax",
        "max",
        "min",
        "topK",
        "softmax",
        "rmsNorm",
        "layerNorm",
        "crossEntropy",
        "crossEntropyExt",
        "rope",
        "dot",
    };

    inline for (forbidden) |decl_name| {
        if (@hasDecl(T, decl_name)) @compileError("non-float Tensor exposes float operation: " ++ decl_name);
    }
}

fn f16TestBits(value: f32) u16 {
    const h: f16 = @floatCast(value);
    return @bitCast(h);
}

test "public non-f32 float Tensor exposes forward math at comptime" {
    inline for (.{ DType.bf16, DType.f16, DType.f64 }) |float_dtype| {
        const T = Tensor(.{ .dtype = float_dtype, .tags = .{ .batch, .d } });
        const expected = .{
            "to",        "add",        "sub",        "mul",       "div",        "sum",        "mean",      "sumAll",
            "dot",       "split",      "merge",      "flatten",   "reshape",    "sliceStep",  "flip",      "roll",
            "stack",     "repeatAxis", "scale",      "divScalar", "unary",      "relu",       "exp",       "sqrt",
            "rsqrt",     "sigmoid",    "silu",       "log",       "log1p",      "neg",        "abs",       "sin",
            "cos",       "tanh",       "fastTanh",   "softcap30", "softcap15",  "gelu",       "quickGelu", "elu",
            "geluErf",   "floor",      "ceil",       "round",     "sign",       "reciprocal", "leakyRelu", "clamp",
            "addScalar", "subScalar",  "powScalar",  "maximum",   "minimum",    "gated",      "glu",       "swiglu",
            "geglu",     "softmax",    "logSoftmax", "rmsNorm",   "rmsNormMul", "layerNorm",  "cumsum",    "cumprod",
            "where",     "maskedFill", "compare",    "pad",       "max",        "min",        "argmax",    "prod",
            "variance",  "logsumexp",  "einsum",
        };
        inline for (expected) |decl_name| {
            if (!@hasDecl(T, decl_name)) @compileError("non-f32 float Tensor missing forward operation: " ++ decl_name);
        }
    }
}

test "public non-f32 Tensor supports tag-only views" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var ids = try Tensor(.{ .dtype = .i64, .rank = 2 }).fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer ids.deinit();

    try std.testing.expect(@TypeOf(ids).axis_tags[0] == ._0);
    try std.testing.expect(@TypeOf(ids).axis_tags[1] == ._1);

    var named = try ids.withTags(&ctx, .{ .batch, .seq });
    defer named.deinit();
    try std.testing.expect(@TypeOf(named).dtype == .i64);
    try std.testing.expect(@TypeOf(named).axis_tags[0] == .batch);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, named.asRawTensor().shape.slice());

    var transposed = try named.transpose(&ctx, .{ .seq, .batch });
    defer transposed.deinit();
    var transposed_data: [6]i64 = undefined;
    try transposed.asRawTensor().copyTo(&transposed_data);
    try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, transposed.asRawTensor().shape.slice());
    try std.testing.expectEqualSlices(i64, &.{ 1, 4, 2, 5, 3, 6 }, &transposed_data);

    var row = try Tensor(.{ .dtype = .i64, .tags = .{.seq} }).fromSlice(&ctx, .{3}, &.{ 7, 8, 9 });
    defer row.deinit();
    var broadcasted = try row.broadcastTo(&ctx, .{ .batch, .seq }, .{ 2, 3 });
    defer broadcasted.deinit();
    var broadcasted_data: [6]i64 = undefined;
    try broadcasted.asRawTensor().copyTo(&broadcasted_data);
    try std.testing.expectEqualSlices(i64, &.{ 7, 8, 9, 7, 8, 9 }, &broadcasted_data);

    var gathered = try named.gather(&ctx, .batch, &.{ 1, 0 }, .token);
    defer gathered.deinit();
    try std.testing.expect(@TypeOf(gathered).axis_tags[0] == .token);
    try std.testing.expectEqualSlices(i64, &.{ 4, 5, 6, 1, 2, 3 }, gathered.asRawTensor().dataConst());

    var narrowed = try named.narrow(&ctx, .seq, 1, 2);
    defer narrowed.deinit();
    var narrowed_data: [4]i64 = undefined;
    try narrowed.asRawTensor().copyTo(&narrowed_data);
    try std.testing.expectEqualSlices(i64, &.{ 2, 3, 5, 6 }, &narrowed_data);

    var extra = try Tensor(.{ .dtype = .i64, .tags = .{ .batch, .seq } }).fromSlice(&ctx, .{ 2, 1 }, &.{ 10, 11 });
    defer extra.deinit();
    var joined = try named.concat(&ctx, .seq, &.{&extra});
    defer joined.deinit();
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 10, 4, 5, 6, 11 }, joined.asRawTensor().dataConst());

    var update = try Tensor(.{ .dtype = .i64, .tags = .{ .batch, .seq } }).fromSlice(&ctx, .{ 2, 2 }, &.{ 20, 21, 22, 23 });
    defer update.deinit();
    var replaced = try joined.setSlice(&ctx, .seq, 1, &update);
    defer replaced.deinit();
    try std.testing.expectEqualSlices(i64, &.{ 1, 20, 21, 10, 4, 22, 23, 11 }, replaced.asRawTensor().dataConst());
}

test "public non-f32 float Tensor supports forward math" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try Tensor(.{ .dtype = .bf16, .tags = .{ .batch, .d } }).fromSlice(&ctx, .{ 2, 2 }, &.{
        dtype_mod.f32ToBf16(1),
        dtype_mod.f32ToBf16(2),
        dtype_mod.f32ToBf16(3),
        dtype_mod.f32ToBf16(4),
    });
    defer a.deinit();
    var b = try Tensor(.{ .dtype = .bf16, .tags = .{ .batch, .d } }).fromSlice(&ctx, .{ 2, 2 }, &.{
        dtype_mod.f32ToBf16(10),
        dtype_mod.f32ToBf16(20),
        dtype_mod.f32ToBf16(30),
        dtype_mod.f32ToBf16(40),
    });
    defer b.deinit();

    var sum = try a.add(&ctx, &b);
    defer sum.deinit();
    try std.testing.expectEqual(@as(f32, 11), dtype_mod.bf16ToF32(sum.asRawTensor().dataConst()[0]));
    try std.testing.expectEqual(@as(f32, 44), dtype_mod.bf16ToF32(sum.asRawTensor().dataConst()[3]));

    var reduced = try sum.sum(&ctx, .d);
    defer reduced.deinit();
    try std.testing.expect(@TypeOf(reduced).dtype == .f32);
    try std.testing.expectEqualSlices(f32, &.{ 33, 77 }, reduced.asRawTensor().dataConst());

    var as_f32 = try reduced.to(&ctx, .f32);
    defer as_f32.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 33, 77 }, as_f32.asRawTensor().dataConst());

    var f32_source = try Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ 1, 2 }, &.{ 1.5, -2.25 });
    defer f32_source.deinit();
    var bf16_cast = try f32_source.to(&ctx, .bf16);
    defer bf16_cast.deinit();
    var f32_roundtrip = try bf16_cast.to(&ctx, .f32);
    defer f32_roundtrip.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1.5, -2.25 }, f32_roundtrip.asRawTensor().dataConst());

    var left = try Tensor(.{ .dtype = .f16, .tags = .{ .m, .k } }).fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer left.deinit();
    var right = try Tensor(.{ .dtype = .f16, .tags = .{ .k, .n } }).fromSlice(&ctx, .{ 3, 2 }, &.{ 7, 8, 9, 10, 11, 12 });
    defer right.deinit();
    var product = try left.dot(&ctx, &right, .k);
    defer product.deinit();
    try std.testing.expectEqualSlices(f16, &.{ 58, 64, 139, 154 }, product.asRawTensor().dataConst());
}

test "public non-f32 float Tensor dot supports multi-free and batch tags" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var left = try Tensor(.{ .dtype = .f16, .tags = .{ .m, .h, .k } }).fromSlice(&ctx, .{ 2, 2, 2 }, &.{
        1, 2,
        3, 4,
        5, 6,
        7, 8,
    });
    defer left.deinit();
    var right = try Tensor(.{ .dtype = .f16, .tags = .{ .k, .n, .v } }).fromSlice(&ctx, .{ 2, 2, 2 }, &.{
        1, 2,
        3, 4,
        5, 6,
        7, 8,
    });
    defer right.deinit();
    var product = try left.dot(&ctx, &right, .k);
    defer product.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 2, 2 }, product.asRawTensor().shape.slice());
    try std.testing.expectEqualSlices(f16, &.{
        11, 14, 17, 20,
        23, 30, 37, 44,
        35, 46, 57, 68,
        47, 62, 77, 92,
    }, product.asRawTensor().dataConst());

    var batched_left = try Tensor(.{ .dtype = .f16, .tags = .{ .batch, .m, .k } }).fromSlice(&ctx, .{ 2, 2, 2 }, &.{
        1, 2,
        3, 4,
        5, 6,
        7, 8,
    });
    defer batched_left.deinit();
    var batched_right = try Tensor(.{ .dtype = .f16, .tags = .{ .batch, .k, .n } }).fromSlice(&ctx, .{ 2, 2, 2 }, &.{
        1, 2,
        3, 4,
        5, 6,
        7, 8,
    });
    defer batched_right.deinit();
    var batched_product = try batched_left.dot(&ctx, &batched_right, .k);
    defer batched_product.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 2 }, batched_product.asRawTensor().shape.slice());
    try std.testing.expectEqualSlices(f16, &.{
        7,  10,
        15, 22,
        67, 78,
        91, 106,
    }, batched_product.asRawTensor().dataConst());
}

test "typed float widened unary family matches the narrowed f32 reference" {
    @setEvalBranchQuota(1_000_000);
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Positive inputs so log/sqrt/rsqrt stay in-domain for every op.
    var x32 = try Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ 2, 2 }, &.{ 0.5, 1.25, 2.0, 3.5 });
    defer x32.deinit();

    inline for (.{ DType.f16, DType.bf16 }) |float_dtype| {
        const Scalar = dtype_mod.Scalar(float_dtype);
        var x_t = try x32.to(&ctx, float_dtype);
        defer x_t.deinit();

        const unary_names = .{
            "relu", "exp",       "sqrt",      "rsqrt",     "sigmoid", "silu",      "log",   "log1p",
            "neg",  "abs",       "sin",       "cos",       "tanh",    "fastTanh",  "gelu",  "quickGelu",
            "elu",  "geluErf",   "floor",     "ceil",      "round",   "sign",      "reciprocal",
            "softcap30", "softcap15",
        };
        inline for (unary_names) |name| {
            var got = try @field(@TypeOf(x_t), name)(&x_t, &ctx);
            defer got.deinit();
            var ref32 = try @field(@TypeOf(x32), name)(&x32, &ctx);
            defer ref32.deinit();
            var ref = try ref32.to(&ctx, float_dtype);
            defer ref.deinit();
            try std.testing.expectEqualSlices(Scalar, ref.asRawTensor().dataConst(), got.asRawTensor().dataConst());
        }

        // The generic entry, the parameterized pointwise ops, and the
        // scalar variants take the same widen -> f32 -> narrow route.
        var via_unary = try x_t.unary(&ctx, .silu);
        defer via_unary.deinit();
        var silu_ref32 = try x32.silu(&ctx);
        defer silu_ref32.deinit();
        var silu_ref = try silu_ref32.to(&ctx, float_dtype);
        defer silu_ref.deinit();
        try std.testing.expectEqualSlices(Scalar, silu_ref.asRawTensor().dataConst(), via_unary.asRawTensor().dataConst());

        var leaky = try x_t.leakyRelu(&ctx, 0.1);
        defer leaky.deinit();
        var leaky_ref32 = try x32.leakyRelu(&ctx, 0.1);
        defer leaky_ref32.deinit();
        var leaky_ref = try leaky_ref32.to(&ctx, float_dtype);
        defer leaky_ref.deinit();
        try std.testing.expectEqualSlices(Scalar, leaky_ref.asRawTensor().dataConst(), leaky.asRawTensor().dataConst());

        var clamped = try x_t.clamp(&ctx, 1.0, 2.0);
        defer clamped.deinit();
        var clamp_ref32 = try x32.clamp(&ctx, 1.0, 2.0);
        defer clamp_ref32.deinit();
        var clamp_ref = try clamp_ref32.to(&ctx, float_dtype);
        defer clamp_ref.deinit();
        try std.testing.expectEqualSlices(Scalar, clamp_ref.asRawTensor().dataConst(), clamped.asRawTensor().dataConst());

        var scaled = try x_t.scale(&ctx, 3.0);
        defer scaled.deinit();
        var shifted = try x_t.addScalar(&ctx, 1.5);
        defer shifted.deinit();
        var shifted_back = try shifted.subScalar(&ctx, 1.5);
        defer shifted_back.deinit();
        var halved = try x_t.divScalar(&ctx, 2.0);
        defer halved.deinit();
        var squared = try x_t.powScalar(&ctx, 2.0);
        defer squared.deinit();
        // 0.5/1.25/2.0/3.5 are exact in f16 AND bf16, so exact-arithmetic
        // scalar results survive the narrow bit-for-bit.
        var squared_ref32 = try x32.powScalar(&ctx, 2.0);
        defer squared_ref32.deinit();
        var squared_ref = try squared_ref32.to(&ctx, float_dtype);
        defer squared_ref.deinit();
        try std.testing.expectEqualSlices(Scalar, squared_ref.asRawTensor().dataConst(), squared.asRawTensor().dataConst());
        try std.testing.expectEqualSlices(Scalar, x_t.asRawTensor().dataConst(), shifted_back.asRawTensor().dataConst());
    }
}

test "typed float widened binary, gated, and mask ops match the narrowed f32 reference" {
    @setEvalBranchQuota(1_000_000);
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a32 = try Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ 2, 2 }, &.{ 1.0, -2.0, 3.0, -4.0 });
    defer a32.deinit();
    var b32 = try Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ 2, 2 }, &.{ -1.5, 2.5, 0.5, 4.0 });
    defer b32.deinit();
    var bias32 = try Tensor(.{.d}).fromSlice(&ctx, .{2}, &.{ 0.5, -0.5 });
    defer bias32.deinit();

    inline for (.{ DType.f16, DType.bf16 }) |float_dtype| {
        const Scalar = dtype_mod.Scalar(float_dtype);
        var a_t = try a32.to(&ctx, float_dtype);
        defer a_t.deinit();
        var b_t = try b32.to(&ctx, float_dtype);
        defer b_t.deinit();
        var bias_t = try bias32.to(&ctx, float_dtype);
        defer bias_t.deinit();

        var hi = try a_t.maximum(&ctx, &b_t);
        defer hi.deinit();
        var hi_ref32 = try a32.maximum(&ctx, &b32);
        defer hi_ref32.deinit();
        var hi_ref = try hi_ref32.to(&ctx, float_dtype);
        defer hi_ref.deinit();
        try std.testing.expectEqualSlices(Scalar, hi_ref.asRawTensor().dataConst(), hi.asRawTensor().dataConst());

        // maximum against a lower-rank operand exercises the tag broadcast.
        var hi_bias = try a_t.minimum(&ctx, &bias_t);
        defer hi_bias.deinit();
        var hi_bias_ref32 = try a32.minimum(&ctx, &bias32);
        defer hi_bias_ref32.deinit();
        var hi_bias_ref = try hi_bias_ref32.to(&ctx, float_dtype);
        defer hi_bias_ref.deinit();
        try std.testing.expectEqualSlices(Scalar, hi_bias_ref.asRawTensor().dataConst(), hi_bias.asRawTensor().dataConst());

        inline for (.{ "glu", "swiglu", "geglu" }) |name| {
            var got = try @field(@TypeOf(a_t), name)(&a_t, &ctx, &b_t);
            defer got.deinit();
            var ref32 = try @field(@TypeOf(a32), name)(&a32, &ctx, &b32);
            defer ref32.deinit();
            var ref = try ref32.to(&ctx, float_dtype);
            defer ref.deinit();
            try std.testing.expectEqualSlices(Scalar, ref.asRawTensor().dataConst(), got.asRawTensor().dataConst());
        }

        var mask = try a_t.compare(&ctx, .gt, 0.0);
        defer mask.deinit();
        var mask_ref32 = try a32.compare(&ctx, .gt, 0.0);
        defer mask_ref32.deinit();
        var mask_ref = try mask_ref32.to(&ctx, float_dtype);
        defer mask_ref.deinit();
        try std.testing.expectEqualSlices(Scalar, mask_ref.asRawTensor().dataConst(), mask.asRawTensor().dataConst());

        var tensor_mask = try a_t.compare(&ctx, .lt, &b_t);
        defer tensor_mask.deinit();

        var filled = try a_t.maskedFill(&ctx, &tensor_mask, 9.0);
        defer filled.deinit();
        var mask32 = try a32.compare(&ctx, .lt, &b32);
        defer mask32.deinit();
        var filled_ref32 = try a32.maskedFill(&ctx, &mask32, 9.0);
        defer filled_ref32.deinit();
        var filled_ref = try filled_ref32.to(&ctx, float_dtype);
        defer filled_ref.deinit();
        try std.testing.expectEqualSlices(Scalar, filled_ref.asRawTensor().dataConst(), filled.asRawTensor().dataConst());

        var chosen = try a_t.where(&ctx, &tensor_mask, &b_t);
        defer chosen.deinit();
        var chosen_ref32 = try a32.where(&ctx, &mask32, &b32);
        defer chosen_ref32.deinit();
        var chosen_ref = try chosen_ref32.to(&ctx, float_dtype);
        defer chosen_ref.deinit();
        try std.testing.expectEqualSlices(Scalar, chosen_ref.asRawTensor().dataConst(), chosen.asRawTensor().dataConst());
    }
}

test "typed float softmax and norm family matches the narrowed f32 reference" {
    @setEvalBranchQuota(1_000_000);
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x32 = try Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ 2, 3 }, &.{ 0.5, -1.0, 2.0, 1.5, 0.25, -0.75 });
    defer x32.deinit();
    var w32 = try Tensor(.{.d}).fromSlice(&ctx, .{3}, &.{ 1.0, 0.5, 2.0 });
    defer w32.deinit();

    inline for (.{ DType.f16, DType.bf16 }) |float_dtype| {
        const Scalar = dtype_mod.Scalar(float_dtype);
        var x_t = try x32.to(&ctx, float_dtype);
        defer x_t.deinit();
        var w_t = try w32.to(&ctx, float_dtype);
        defer w_t.deinit();

        var soft = try x_t.softmax(&ctx, .d, .{});
        defer soft.deinit();
        var soft_ref32 = try x32.softmax(&ctx, .d, .{});
        defer soft_ref32.deinit();
        var soft_ref = try soft_ref32.to(&ctx, float_dtype);
        defer soft_ref.deinit();
        try std.testing.expectEqualSlices(Scalar, soft_ref.asRawTensor().dataConst(), soft.asRawTensor().dataConst());

        var logsoft = try x_t.logSoftmax(&ctx, .d);
        defer logsoft.deinit();
        var logsoft_ref32 = try x32.logSoftmax(&ctx, .d);
        defer logsoft_ref32.deinit();
        var logsoft_ref = try logsoft_ref32.to(&ctx, float_dtype);
        defer logsoft_ref.deinit();
        try std.testing.expectEqualSlices(Scalar, logsoft_ref.asRawTensor().dataConst(), logsoft.asRawTensor().dataConst());

        var rms = try x_t.rmsNorm(&ctx, .d, 1e-6);
        defer rms.deinit();
        var rms_ref32 = try x32.rmsNorm(&ctx, .d, 1e-6);
        defer rms_ref32.deinit();
        var rms_ref = try rms_ref32.to(&ctx, float_dtype);
        defer rms_ref.deinit();
        try std.testing.expectEqualSlices(Scalar, rms_ref.asRawTensor().dataConst(), rms.asRawTensor().dataConst());

        var rms_mul = try x_t.rmsNormMul(&ctx, .d, &w_t, 1e-6);
        defer rms_mul.deinit();
        var rms_mul_ref32 = try x32.rmsNormMul(&ctx, .d, &w32, 1e-6);
        defer rms_mul_ref32.deinit();
        var rms_mul_ref = try rms_mul_ref32.to(&ctx, float_dtype);
        defer rms_mul_ref.deinit();
        try std.testing.expectEqualSlices(Scalar, rms_mul_ref.asRawTensor().dataConst(), rms_mul.asRawTensor().dataConst());

        var ln = try x_t.layerNorm(&ctx, .d, 1e-5, .{});
        defer ln.deinit();
        var ln_ref32 = try x32.layerNorm(&ctx, .d, 1e-5, .{});
        defer ln_ref32.deinit();
        var ln_ref = try ln_ref32.to(&ctx, float_dtype);
        defer ln_ref.deinit();
        try std.testing.expectEqualSlices(Scalar, ln_ref.asRawTensor().dataConst(), ln.asRawTensor().dataConst());
    }
}

test "typed float widened reductions return f32 and scans keep the dtype" {
    @setEvalBranchQuota(1_000_000);
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Exact in f16 AND bf16: widen(narrow(x)) == x, so the widened
    // reductions must equal the f32 reference bit-for-bit.
    var x32 = try Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ 2, 3 }, &.{ 1.0, -2.5, 3.0, 0.5, 4.0, -1.5 });
    defer x32.deinit();

    inline for (.{ DType.f16, DType.bf16 }) |float_dtype| {
        const Scalar = dtype_mod.Scalar(float_dtype);
        var x_t = try x32.to(&ctx, float_dtype);
        defer x_t.deinit();

        var top = try x_t.max(&ctx, .d);
        defer top.deinit();
        comptime std.debug.assert(@TypeOf(top).dtype == .f32);
        try std.testing.expectEqualSlices(f32, &.{ 3.0, 4.0 }, top.asRawTensor().dataConst());

        var bottom = try x_t.min(&ctx, .d);
        defer bottom.deinit();
        comptime std.debug.assert(@TypeOf(bottom).dtype == .f32);
        try std.testing.expectEqualSlices(f32, &.{ -2.5, -1.5 }, bottom.asRawTensor().dataConst());

        var best = try x_t.argmax(&ctx, .d);
        defer best.deinit();
        comptime std.debug.assert(@TypeOf(best).dtype == .i64);
        try std.testing.expectEqualSlices(i64, &.{ 2, 1 }, best.asRawTensor().dataConst());

        var product = try x_t.prod(&ctx, .d);
        defer product.deinit();
        comptime std.debug.assert(@TypeOf(product).dtype == .f32);
        try std.testing.expectEqualSlices(f32, &.{ -7.5, -3.0 }, product.asRawTensor().dataConst());

        var spread = try x_t.variance(&ctx, .d, 0);
        defer spread.deinit();
        comptime std.debug.assert(@TypeOf(spread).dtype == .f32);
        var spread_ref = try x32.variance(&ctx, .d, 0);
        defer spread_ref.deinit();
        try std.testing.expectEqualSlices(f32, spread_ref.asRawTensor().dataConst(), spread.asRawTensor().dataConst());

        var lse = try x_t.logsumexp(&ctx, .d);
        defer lse.deinit();
        comptime std.debug.assert(@TypeOf(lse).dtype == .f32);
        var lse_ref = try x32.logsumexp(&ctx, .d);
        defer lse_ref.deinit();
        try std.testing.expectEqualSlices(f32, lse_ref.asRawTensor().dataConst(), lse.asRawTensor().dataConst());

        var running = try x_t.cumsum(&ctx, .d);
        defer running.deinit();
        comptime std.debug.assert(@TypeOf(running).dtype == float_dtype);
        var running_ref32 = try x32.cumsum(&ctx, .d);
        defer running_ref32.deinit();
        var running_ref = try running_ref32.to(&ctx, float_dtype);
        defer running_ref.deinit();
        try std.testing.expectEqualSlices(Scalar, running_ref.asRawTensor().dataConst(), running.asRawTensor().dataConst());

        var running_prod = try x_t.cumprod(&ctx, .d);
        defer running_prod.deinit();
        comptime std.debug.assert(@TypeOf(running_prod).dtype == float_dtype);
        var running_prod_ref32 = try x32.cumprod(&ctx, .d);
        defer running_prod_ref32.deinit();
        var running_prod_ref = try running_prod_ref32.to(&ctx, float_dtype);
        defer running_prod_ref.deinit();
        try std.testing.expectEqualSlices(Scalar, running_prod_ref.asRawTensor().dataConst(), running_prod.asRawTensor().dataConst());
    }
}

test "typed float einsum matches the narrowed f32 lowering" {
    @setEvalBranchQuota(1_000_000);
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var left32 = try Tensor(.{ .m, .k }).fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer left32.deinit();
    var right32 = try Tensor(.{ .k, .n }).fromSlice(&ctx, .{ 3, 2 }, &.{ 7, 8, 9, 10, 11, 12 });
    defer right32.deinit();

    inline for (.{ DType.f16, DType.bf16 }) |float_dtype| {
        const Scalar = dtype_mod.Scalar(float_dtype);
        var left_t = try left32.to(&ctx, float_dtype);
        defer left_t.deinit();
        var right_t = try right32.to(&ctx, float_dtype);
        defer right_t.deinit();

        var got = try left_t.einsum(&ctx, &right_t, .{ .m, .n });
        defer got.deinit();
        comptime std.debug.assert(@TypeOf(got).dtype == float_dtype);
        var ref32 = try left32.einsum(&ctx, &right32, .{ .m, .n });
        defer ref32.deinit();
        var ref = try ref32.to(&ctx, float_dtype);
        defer ref.deinit();
        try std.testing.expectEqualSlices(Scalar, ref.asRawTensor().dataConst(), got.asRawTensor().dataConst());

        // einsum agrees with the native typed dot on the same contraction
        // (both accumulate in f32 and narrow once).
        var via_dot = try left_t.dot(&ctx, &right_t, .k);
        defer via_dot.deinit();
        try std.testing.expectEqualSlices(Scalar, via_dot.asRawTensor().dataConst(), got.asRawTensor().dataConst());
    }
}

test "typed float structural ops preserve values across dtypes" {
    @setEvalBranchQuota(1_000_000);
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x32 = try Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ 2, 4 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer x32.deinit();

    inline for (.{ DType.f16, DType.bf16, DType.f64 }) |float_dtype| {
        const Scalar = dtype_mod.Scalar(float_dtype);
        var x_t = try x32.to(&ctx, float_dtype);
        defer x_t.deinit();
        var flat_ref = try x_t.flatten(&ctx, .flat);
        defer flat_ref.deinit();

        var halves = try x_t.split(&ctx, .d, .{ .half, .pair }, .{ 2, 2 });
        defer halves.deinit();
        try std.testing.expectEqualSlices(usize, &.{ 2, 2, 2 }, halves.asRawTensor().shape.slice());
        var remerged = try halves.merge(&ctx, .d, .{ .half, .pair });
        defer remerged.deinit();
        var remerged_mat = try remerged.materialize(&ctx);
        defer remerged_mat.deinit();
        try std.testing.expectEqualSlices(Scalar, x_t.asRawTensor().dataConst(), remerged_mat.asRawTensor().dataConst());

        var reshaped = try x_t.reshape(&ctx, .{ .a, .b }, .{ 4, 2 });
        defer reshaped.deinit();
        try std.testing.expectEqualSlices(usize, &.{ 4, 2 }, reshaped.asRawTensor().shape.slice());
        try std.testing.expectEqualSlices(Scalar, flat_ref.asRawTensor().dataConst(), blk: {
            var reflat = try reshaped.flatten(&ctx, .flat);
            defer reflat.deinit();
            break :blk reflat.asRawTensor().dataConst();
        });

        var stepped = try x_t.sliceStep(&ctx, .d, 0, 2, 2);
        defer stepped.deinit();
        var stepped_mat = try stepped.materialize(&ctx);
        defer stepped_mat.deinit();
        var stepped_ref32 = try x32.sliceStep(&ctx, .d, 0, 2, 2);
        defer stepped_ref32.deinit();
        var stepped_ref = try stepped_ref32.to(&ctx, float_dtype);
        defer stepped_ref.deinit();
        try std.testing.expectEqualSlices(Scalar, stepped_ref.asRawTensor().dataConst(), stepped_mat.asRawTensor().dataConst());

        var flipped = try x_t.flip(&ctx, .d);
        defer flipped.deinit();
        var flipped_ref32 = try x32.flip(&ctx, .d);
        defer flipped_ref32.deinit();
        var flipped_ref = try flipped_ref32.to(&ctx, float_dtype);
        defer flipped_ref.deinit();
        try std.testing.expectEqualSlices(Scalar, flipped_ref.asRawTensor().dataConst(), flipped.asRawTensor().dataConst());

        var rolled = try x_t.roll(&ctx, .d, 1);
        defer rolled.deinit();
        var rolled_ref32 = try x32.roll(&ctx, .d, 1);
        defer rolled_ref32.deinit();
        var rolled_ref = try rolled_ref32.to(&ctx, float_dtype);
        defer rolled_ref.deinit();
        try std.testing.expectEqualSlices(Scalar, rolled_ref.asRawTensor().dataConst(), rolled.asRawTensor().dataConst());

        var stacked = try x_t.stack(&ctx, .copy, 0, &.{&x_t});
        defer stacked.deinit();
        try std.testing.expectEqualSlices(usize, &.{ 2, 2, 4 }, stacked.asRawTensor().shape.slice());

        var repeated = try x_t.repeatAxis(&ctx, .batch, 2);
        defer repeated.deinit();
        try std.testing.expectEqualSlices(usize, &.{ 4, 4 }, repeated.asRawTensor().shape.slice());

        // scale/divScalar run the native typed kernel on every float dtype
        // (f64 included). Integer inputs scaled by 3 are exact everywhere.
        var tripled = try x_t.scale(&ctx, 3.0);
        defer tripled.deinit();
        var thirds = try tripled.divScalar(&ctx, 3.0);
        defer thirds.deinit();
        var tripled_ref32 = try x32.scale(&ctx, 3.0);
        defer tripled_ref32.deinit();
        var tripled_ref = try tripled_ref32.to(&ctx, float_dtype);
        defer tripled_ref.deinit();
        try std.testing.expectEqualSlices(Scalar, tripled_ref.asRawTensor().dataConst(), tripled.asRawTensor().dataConst());
        try std.testing.expectEqualSlices(Scalar, x_t.asRawTensor().dataConst(), thirds.asRawTensor().dataConst());
    }

    // pad computes through f32 and is f16/bf16 only.
    inline for (.{ DType.f16, DType.bf16 }) |float_dtype| {
        const Scalar = dtype_mod.Scalar(float_dtype);
        var x_t = try x32.to(&ctx, float_dtype);
        defer x_t.deinit();
        var padded = try x_t.pad(&ctx, .d, 1, 1, 0.5);
        defer padded.deinit();
        var padded_ref32 = try x32.pad(&ctx, .d, 1, 1, 0.5);
        defer padded_ref32.deinit();
        var padded_ref = try padded_ref32.to(&ctx, float_dtype);
        defer padded_ref.deinit();
        try std.testing.expectEqualSlices(Scalar, padded_ref.asRawTensor().dataConst(), padded.asRawTensor().dataConst());
    }
}

test "tagged public tensor insertAxis and squeeze are zero-copy views and validate dims" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();

    var with_head = try x.insertAxis(&ctx, .head, 1);
    defer with_head.deinit();
    try std.testing.expect(with_head.asRawTensor().buffer == x.asRawTensor().buffer);
    try std.testing.expectEqualSlices(usize, &.{ 2, 1, 3 }, with_head.asRawTensor().shape.slice());

    var squeezed = try with_head.squeeze(&ctx, .head);
    defer squeezed.deinit();
    try std.testing.expect(squeezed.asRawTensor().buffer == x.asRawTensor().buffer);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, squeezed.asRawTensor().shape.slice());

    try std.testing.expectError(error.InvalidShape, x.squeeze(&ctx, .batch));
}

test "tagged public grad state does not duplicate the raw value" {
    if (@hasField(GradState, "value")) @compileError("GradState must not own Tensor.value");
}

test "tagged autograd numeric rank uses generated axis tags" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(3).variable(
        &ctx,
        try ctx.fromSlice(
            &.{ 2, 3, 2 },
            &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 },
        ),
    );
    defer x.deinit();

    try std.testing.expectEqual(@as(usize, 3), @TypeOf(x).tag_count);
    try std.testing.expect(@TypeOf(x).axis_tags[0] == ._0);
    try std.testing.expect(@TypeOf(x).axis_tags[1] == ._1);
    try std.testing.expect(@TypeOf(x).axis_tags[2] == ._2);
    try std.testing.expectEqual(@as(usize, 2), @TypeOf(x).axis(._2));

    var reduced = try x.sumMany(&ctx, 2);
    defer reduced.deinit();
    try std.testing.expect(@TypeOf(reduced).axis_tags[0] == ._2);
    try std.testing.expectEqualSlices(usize, &.{2}, reduced.asRawTensor().shape.slice());
    try std.testing.expectEqualSlices(f32, &.{ 36, 42 }, reduced.asRawTensor().dataConst());

    var loss = try reduced.sumAll(&ctx);
    defer loss.deinit();

    try loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }, gx.asRawTensor().dataConst());
}

test "tagged public tensor leakyRelu forward and backward" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{.d}).variable(
        &ctx,
        try ctx.fromSlice(&.{3}, &.{ -2, 0, 3 }),
    );
    defer x.deinit();

    var y = try x.leakyRelu(&ctx, 0.25);
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ -0.5, 0, 3 }, y.asRawTensor().dataConst());

    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0.25, 0.25, 1 }, gx.asRawTensor().dataConst());
}

fn fastTanhDerivativeForTest(value: f32) f32 {
    const a: f32 = 2.45550750702956;
    const b: f32 = 0.893229853513558;
    const c: f32 = 0.821226666969744;
    const d: f32 = 2.44506634652299;
    const e: f32 = 0.814642734961073;

    const ax = @abs(value);
    const dax: f32 = if (value > 0) 1 else if (value < 0) -1 else 0;
    const x2 = value * value;
    const p = a + a * ax + (b + c * ax) * x2;
    const dp = a * dax + c * dax * x2 + (b + c * ax) * 2 * value;
    const numerator = value * p;
    const dnumerator = p + value * dp;

    const q = value + e * value * ax;
    const dq = 1 + e * (ax + value * dax);
    const r = @abs(q);
    const dr: f32 = if (q > 0) dq else if (q < 0) -dq else 0;
    const denominator = d + (d + x2) * r;
    const ddenominator = 2 * value * r + (d + x2) * dr;
    return (dnumerator * denominator - numerator * ddenominator) / (denominator * denominator);
}

test "tagged public tensor fastTanh forward and backward" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{.d}).variable(
        &ctx,
        try ctx.fromSlice(&.{4}, &.{ -2, -0.25, 0, 3 }),
    );
    defer x.deinit();

    var y = try x.fastTanh(&ctx);
    defer y.deinit();
    for (x.asRawTensor().dataConst(), y.asRawTensor().dataConst()) |value, actual| {
        try std.testing.expectApproxEqAbs(backend_mod.ops.fastTanhScalar(value), actual, 1e-6);
    }

    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    for (x.asRawTensor().dataConst(), gx.asRawTensor().dataConst()) |value, actual| {
        try std.testing.expectApproxEqAbs(fastTanhDerivativeForTest(value), actual, 1e-5);
    }
}

test "tagged public tensor detach severs the gradient graph" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ 1, 2, 3 });
    defer x.deinit();

    var y = try x.mul(&ctx, &x);
    defer y.deinit();
    try std.testing.expect(y.requiresGrad());

    var detached = try y.detach(&ctx);
    defer detached.deinit();
    try std.testing.expect(!detached.requiresGrad());
    try std.testing.expectEqualSlices(f32, &.{ 1, 4, 9 }, try detached.dataConst());

    var loss = try detached.sumAll(&ctx);
    defer loss.deinit();
    try std.testing.expect(!loss.requiresGrad());
    try std.testing.expectError(error.NoGradientGraph, loss.backward(&ctx));
    try std.testing.expect((try x.grad(&ctx)) == null);
}

test "tagged public backwardWithGrad seeds non-scalar outputs" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{.d}).variableFromSlice(&ctx, .{2}, &.{ 3, 4 });
    defer x.deinit();
    var c = try Tensor(.{.d}).fromSlice(&ctx, .{2}, &.{ 10, 20 });
    defer c.deinit();
    var y = try x.mul(&ctx, &c);
    defer y.deinit();

    // A non-scalar output has no implicit seed; a mis-shaped output
    // gradient is rejected before any state changes.
    try std.testing.expectError(error.MissingOutputGradient, y.backward(&ctx));
    var bad_grad = try Tensor(.{.d}).fromSlice(&ctx, .{3}, &.{ 1, 1, 1 });
    defer bad_grad.deinit();
    try std.testing.expectError(error.ShapeMismatch, y.backwardWithGrad(&ctx, &bad_grad));

    // The output gradient is read as a value: dloss/dx = grad_output * c.
    var grad_output = try Tensor(.{.d}).fromSlice(&ctx, .{2}, &.{ 1, 10 });
    defer grad_output.deinit();
    try y.backwardWithGrad(&ctx, &grad_output);
    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 10, 200 }, try gx.dataConst());
    try std.testing.expectEqualSlices(f32, &.{ 1, 10 }, try grad_output.dataConst());

    // The completed pass consumed the graph; constants have no graph at all.
    try std.testing.expectError(error.BackwardAlreadyRun, y.backwardWithGrad(&ctx, &grad_output));
    try std.testing.expectError(error.NoGradientGraph, c.backwardWithGrad(&ctx, &grad_output));
}

test "tagged public tensor rejects a second backward over a consumed graph" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{.d}).variableFromSlice(&ctx, .{2}, &.{ 3, 5 });
    defer x.deinit();

    var sq = try x.mul(&ctx, &x);
    defer sq.deinit();
    var loss = try sq.sumAll(&ctx);
    defer loss.deinit();

    // Scalar outputs may take an explicit output gradient too:
    // dloss/dx = 2 * 2x.
    var grad_output = try Tensor(.{}).scalar(&ctx, 2);
    defer grad_output.deinit();
    try loss.backwardWithGrad(&ctx, &grad_output);
    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 12, 20 }, try gx.dataConst());

    // Interior states keep their gradients, so a second pass over the SAME
    // graph would compound them; it fails loudly instead — zeroGrad resets
    // gradients, not the consumed graph.
    try std.testing.expectError(error.BackwardAlreadyRun, loss.backward(&ctx));
    x.zeroGrad();
    loss.zeroGrad();
    try std.testing.expectError(error.BackwardAlreadyRun, loss.backward(&ctx));

    // The micro-batch idiom is untouched: a FRESH graph over the same leaf
    // runs and accumulates into the leaf as before.
    var sq2 = try x.mul(&ctx, &x);
    defer sq2.deinit();
    var loss2 = try sq2.sumAll(&ctx);
    defer loss2.deinit();
    try loss2.backward(&ctx);
    var gx2 = (try x.grad(&ctx)).?;
    defer gx2.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 6, 10 }, try gx2.dataConst());
}

test "tagged public noGrad scope suppresses graph recording" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ 1, 2, 3 });
    defer x.deinit();

    {
        var guard = control.noGrad();
        defer guard.close();
        var y = try x.mul(&ctx, &x);
        defer y.deinit();
        try std.testing.expect(!y.requiresGrad());
        try std.testing.expectEqualSlices(f32, &.{ 1, 4, 9 }, try y.dataConst());
    }

    var y = try x.mul(&ctx, &x);
    defer y.deinit();
    try std.testing.expect(y.requiresGrad());
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 2, 4, 6 }, try gx.dataConst());
}

test "tagged autograd no-grad expressions do not retain graph intermediates" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();
    var bias = try Tensor(.{.d}).fromSlice(&ctx, .{3}, &.{ 10, 20, 30 });
    defer bias.deinit();

    var y = try x.add(&ctx, &bias);
    defer y.deinit();

    try std.testing.expect(!y.requiresGrad());
    try std.testing.expect(y.grad_state == null);
    try std.testing.expectEqualSlices(f32, &.{ 11, 22, 33, 14, 25, 36 }, y.asRawTensor().dataConst());
}

test "tagged public tensor stores no-grad values inline" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const raw = try ctx.fromSliceRank(1, .{3}, &.{ 1, 2, 3 });
    const buffer = raw.buffer;
    var x = try Tensor(1).fromTensor(&ctx, raw);
    defer x.deinit();

    try std.testing.expect(x.grad_state == null);
    try std.testing.expect(x.asRawTensor().buffer == buffer);

    var y = try x.add(&ctx, x);
    defer y.deinit();
    try std.testing.expect(y.grad_state == null);
    try std.testing.expectEqualSlices(f32, &.{ 2, 4, 6 }, y.asRawTensor().dataConst());
}

test "tagged public tensor applies no-grad views without graph state" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();

    var p = try x.permuteTo(&ctx, .{ .d, .batch });
    defer p.deinit();
    var p_data = [_]f32{0} ** 6;
    try p.asRawTensor().copyTo(&p_data);
    try std.testing.expect(p.grad_state == null);
    try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, p.asRawTensor().shape.slice());
    try std.testing.expectEqualSlices(f32, &.{ 1, 4, 2, 5, 3, 6 }, &p_data);

    var expanded = try x.insertAxis(&ctx, .head, 1);
    defer expanded.deinit();
    try std.testing.expect(expanded.grad_state == null);
    try std.testing.expectEqualSlices(usize, &.{ 2, 1, 3 }, expanded.asRawTensor().shape.slice());

    var squeezed = try expanded.squeeze(&ctx, .head);
    defer squeezed.deinit();
    var squeezed_data = [_]f32{0} ** 6;
    try squeezed.asRawTensor().copyTo(&squeezed_data);
    try std.testing.expect(squeezed.grad_state == null);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6 }, &squeezed_data);

    var split = try x.split(&ctx, .d, .{ .row, .col }, .{ 1, 3 });
    defer split.deinit();
    try std.testing.expect(split.grad_state == null);
    try std.testing.expectEqualSlices(usize, &.{ 2, 1, 3 }, split.asRawTensor().shape.slice());

    var merged = try split.merge(&ctx, .d2, .{ .row, .col });
    defer merged.deinit();
    var merged_data = [_]f32{0} ** 6;
    try merged.asRawTensor().copyTo(&merged_data);
    try std.testing.expect(merged.grad_state == null);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6 }, &merged_data);
}

test "tagged public tensor concatenates and narrows with gradients" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try Tensor(.{ .row, .d }).variableFromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer a.deinit();
    var b = try Tensor(.{ .row, .d }).variableFromSlice(&ctx, .{ 1, 2 }, &.{ 5, 6 });
    defer b.deinit();

    var joined = try a.concat(&ctx, .row, &.{&b});
    defer joined.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, joined.asRawTensor().shape.slice());
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6 }, joined.asRawTensor().dataConst());

    var loss = try joined.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var ga = (try a.grad(&ctx)).?;
    defer ga.deinit();
    var gb = (try b.grad(&ctx)).?;
    defer gb.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 1, 1 }, ga.asRawTensor().dataConst());
    try std.testing.expectEqualSlices(f32, &.{ 1, 1 }, gb.asRawTensor().dataConst());

    var x = try Tensor(.{ .row, .d }).variableFromSlice(&ctx, .{ 4, 2 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer x.deinit();
    var sliced = try x.narrow(&ctx, .row, 1, 2);
    defer sliced.deinit();
    try std.testing.expect(sliced.asRawTensor().buffer == x.asRawTensor().buffer);
    try std.testing.expectEqualSlices(f32, &.{ 3, 4, 5, 6 }, sliced.asRawTensor().dataConst());

    var slice_loss = try sliced.sumAll(&ctx);
    defer slice_loss.deinit();
    try slice_loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 1, 1, 1, 1, 0, 0 }, gx.asRawTensor().dataConst());
}

fn concatGradcheckLoss(ctx: *ExecContext, a: *const Tensor(.{.d}), b: *const Tensor(.{.d})) !Tensor(.{}) {
    var y = try a.concat(ctx, .d, &.{b});
    defer y.deinit();
    var w = try Tensor(.{.d}).fromSlice(ctx, .{7}, &.{ 1, -2, 3, 0.5, -1, 2, -3 });
    defer w.deinit();
    var z = try y.mul(ctx, &w);
    defer z.deinit();
    return z.sumAll(ctx);
}

test "public tensor concat gradient passes gradcheck for both parents" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var a = try Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ 2, -3, 4 });
    defer a.deinit();
    var b = try Tensor(.{.d}).variableFromSlice(&ctx, .{4}, &.{ 5, 7, -11, 2 });
    defer b.deinit();
    const result = try gradcheck_mod.gradcheck(&ctx, concatGradcheckLoss, .{ &a, &b }, .{});
    try std.testing.expectEqual(@as(usize, 7), result.checked);
}

test "public tensor concat stays no-grad for two and more than sixteen inputs" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try Tensor(.{.d}).fromSlice(&ctx, .{2}, &.{ 1, 2 });
    defer a.deinit();
    var b = try Tensor(.{.d}).fromSlice(&ctx, .{2}, &.{ 3, 4 });
    defer b.deinit();

    var pair = try a.concat(&ctx, .d, &.{&b});
    defer pair.deinit();
    try std.testing.expect(pair.grad_state == null);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4 }, pair.asRawTensor().dataConst());

    // 1 + 17 = 18 inputs exceeds the small-inline metadata bound (16).
    var inputs: [17]Tensor(.{.d}) = undefined;
    var created: usize = 0;
    defer for (inputs[0..created]) |*input| input.deinit();
    var ptrs: [17]*const Tensor(.{.d}) = undefined;
    for (&inputs, &ptrs, 0..) |*input, *ptr, i| {
        const value: f32 = @floatFromInt(i + 1);
        input.* = try Tensor(.{.d}).fromSlice(&ctx, .{1}, &.{value});
        created += 1;
        ptr.* = input;
    }
    var joined = try a.concat(&ctx, .d, &ptrs);
    defer joined.deinit();
    try std.testing.expect(joined.grad_state == null);
    try std.testing.expectEqualSlices(
        f32,
        &.{ 1, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 },
        joined.asRawTensor().dataConst(),
    );

    // Variables under a noGrad scope take the same no-grad path (concat's
    // metadata gate mirrors finishOp's wants_grad + isGradEnabled check).
    var v = try Tensor(.{.d}).variableFromSlice(&ctx, .{2}, &.{ 5, 6 });
    defer v.deinit();
    {
        var guard = control.noGrad();
        defer guard.close();
        var y = try v.concat(&ctx, .d, &.{&v});
        defer y.deinit();
        try std.testing.expect(!y.requiresGrad());
        try std.testing.expect(y.grad_state == null);
        try std.testing.expectEqualSlices(f32, &.{ 5, 6, 5, 6 }, y.asRawTensor().dataConst());
    }
}

test "public tensor concat backpropagates through more than sixteen inputs" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // 1 + 17 = 18 inputs exceeds the small-inline metadata bound (16), so the
    // backward parents/sizes take the heap fallback.
    var first = try Tensor(.{.d}).variableFromSlice(&ctx, .{1}, &.{1});
    defer first.deinit();
    var inputs: [17]Tensor(.{.d}) = undefined;
    var created: usize = 0;
    defer for (inputs[0..created]) |*input| input.deinit();
    var ptrs: [17]*const Tensor(.{.d}) = undefined;
    for (&inputs, &ptrs, 0..) |*input, *ptr, i| {
        const value: f32 = @floatFromInt(i + 2);
        input.* = try Tensor(.{.d}).variableFromSlice(&ctx, .{1}, &.{value});
        created += 1;
        ptr.* = input;
    }

    var joined = try first.concat(&ctx, .d, &ptrs);
    defer joined.deinit();
    var expected: [18]f32 = undefined;
    for (&expected, 0..) |*e, i| e.* = @floatFromInt(i + 1);
    try std.testing.expectEqualSlices(f32, &expected, joined.asRawTensor().dataConst());

    // Distinct per-position weights make gradient mis-routing detectable.
    var weights: [18]f32 = undefined;
    for (&weights, 0..) |*wv, i| wv.* = @as(f32, @floatFromInt(i + 1)) * 0.5 - 4;
    var w = try Tensor(.{.d}).fromSlice(&ctx, .{18}, &weights);
    defer w.deinit();
    var z = try joined.mul(&ctx, &w);
    defer z.deinit();
    var loss = try z.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var gfirst = (try first.grad(&ctx)).?;
    defer gfirst.deinit();
    try std.testing.expectEqualSlices(f32, weights[0..1], gfirst.asRawTensor().dataConst());
    for (inputs[0..created], 0..) |*input, i| {
        var gi = (try input.grad(&ctx)).?;
        defer gi.deinit();
        try std.testing.expectEqualSlices(f32, weights[i + 1 ..][0..1], gi.asRawTensor().dataConst());
    }
}

test "public f32 Tensor einsum contracts and backpropagates through both operands" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try Tensor(.{ .m, .k }).variableFromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer a.deinit();
    var b = try Tensor(.{ .k, .n }).variableFromSlice(&ctx, .{ 2, 2 }, &.{ 5, 6, 7, 8 });
    defer b.deinit();

    var y = try a.einsum(&ctx, &b, .{ .m, .n });
    defer y.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, y.asRawTensor().shape.slice());
    try std.testing.expectEqualSlices(f32, &.{ 19, 22, 43, 50 }, y.asRawTensor().dataConst());

    // The output tag order is the equation: swapping it transposes the result.
    var yt = try a.einsum(&ctx, &b, .{ .n, .m });
    defer yt.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 19, 43, 22, 50 }, yt.asRawTensor().dataConst());

    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var ga = (try a.grad(&ctx)).?;
    defer ga.deinit();
    var gb = (try b.grad(&ctx)).?;
    defer gb.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 11, 15, 11, 15 }, ga.asRawTensor().dataConst());
    try std.testing.expectEqualSlices(f32, &.{ 4, 4, 6, 6 }, gb.asRawTensor().dataConst());
}

test "public f32 Tensor einsum broadcasts gradients over summed-away axes" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // .s is private to `a` and dropped from the output: the forward sums over
    // it, so its gradient replicates along that axis.
    var a = try Tensor(.{ .s, .k }).variableFromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer a.deinit();
    var b = try Tensor(.{ .k, .n }).variableFromSlice(&ctx, .{ 2, 2 }, &.{ 5, 6, 7, 8 });
    defer b.deinit();

    var y = try a.einsum(&ctx, &b, .{.n});
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 62, 72 }, y.asRawTensor().dataConst());

    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var ga = (try a.grad(&ctx)).?;
    defer ga.deinit();
    var gb = (try b.grad(&ctx)).?;
    defer gb.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 11, 15, 11, 15 }, ga.asRawTensor().dataConst());
    try std.testing.expectEqualSlices(f32, &.{ 4, 4, 6, 6 }, gb.asRawTensor().dataConst());
}

test "public f32 Tensor einsum backward compiles when the operand-tag union exceeds max_rank" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // 5 + 5 tags sharing only .k: the union has 9 tags (> max_rank 8) while
    // every tensor involved, including the rank-8 result, fits. The backward
    // record's membership sets must not be rank-capped.
    var a = try Tensor(.{ .a, .b, .c, .d, .k }).variableFromSlice(&ctx, .{ 2, 1, 2, 1, 3 }, &.{
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    });
    defer a.deinit();
    var b = try Tensor(.{ .k, .e, .f, .g, .h }).variableFromSlice(&ctx, .{ 3, 2, 1, 2, 1 }, &.{
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    });
    defer b.deinit();

    var y = try a.einsum(&ctx, &b, .{ .a, .b, .c, .d, .e, .f, .g, .h });
    defer y.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 1, 2, 1, 2, 1, 2, 1 }, y.asRawTensor().shape.slice());

    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var ga = (try a.grad(&ctx)).?;
    defer ga.deinit();
    var gb = (try b.grad(&ctx)).?;
    defer gb.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 1, 2, 1, 3 }, ga.asRawTensor().shape.slice());
    try std.testing.expectEqualSlices(usize, &.{ 3, 2, 1, 2, 1 }, gb.asRawTensor().shape.slice());
}

test "public einsumMany folds a four-operand chain and matches chained dot" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .a, .b }).fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();
    var y = try Tensor(.{ .b, .c }).fromSlice(&ctx, .{ 3, 2 }, &.{ 1, -1, 2, 0, -2, 1 });
    defer y.deinit();
    var z = try Tensor(.{ .c, .d }).fromSlice(&ctx, .{ 2, 2 }, &.{ 3, 1, 0, -1 });
    defer z.deinit();
    var w = try Tensor(.{ .d, .e }).fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 0, -1, 1, 2 });
    defer w.deinit();

    var folded = try ag_tensor.einsumMany(&ctx, .{ .a, .e }, .{ &x, &y, &z, &w });
    defer folded.deinit();

    var xy = try x.dot(&ctx, &y, .b);
    defer xy.deinit();
    var xyz = try xy.dot(&ctx, &z, .c);
    defer xyz.deinit();
    var ref = try xyz.dot(&ctx, &w, .d);
    defer ref.deinit();

    try std.testing.expectEqualSlices(usize, ref.asRawTensor().shape.slice(), folded.asRawTensor().shape.slice());
    try std.testing.expectEqualSlices(f32, ref.asRawTensor().dataConst(), folded.asRawTensor().dataConst());
}

test "public einsumMany folds a three-operand chain and matches chained dot" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // LoRA-delta shape: x[s,i] · A[r,i] · B[o,r] -> [s,o] as one equation.
    var x = try Tensor(.{ .s, .i }).fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();
    var a = try Tensor(.{ .r, .i }).fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 0, 2, -1, 3, 1 });
    defer a.deinit();
    var b = try Tensor(.{ .o, .r }).fromSlice(&ctx, .{ 2, 2 }, &.{ 2, 1, 0, -2 });
    defer b.deinit();

    var y = try ag_tensor.einsumMany(&ctx, .{ .s, .o }, .{ &x, &a, &b });
    defer y.deinit();

    var xa = try x.dot(&ctx, &a, .i);
    defer xa.deinit();
    var ref = try xa.dot(&ctx, &b, .r);
    defer ref.deinit();

    try std.testing.expectEqualSlices(usize, ref.asRawTensor().shape.slice(), y.asRawTensor().shape.slice());
    try std.testing.expectEqualSlices(f32, ref.asRawTensor().dataConst(), y.asRawTensor().dataConst());
}

test "tagged public tensor setSlice and setRows propagate assignment gradients" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var base = try Tensor(.{ .row, .d }).variableFromSlice(&ctx, .{ 4, 2 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer base.deinit();
    var update = try Tensor(.{ .row, .d }).variableFromSlice(&ctx, .{ 2, 2 }, &.{ 10, 20, 30, 40 });
    defer update.deinit();

    var y = try base.setSlice(&ctx, .row, 1, &update);
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 10, 20, 30, 40, 7, 8 }, y.asRawTensor().dataConst());

    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var gb = (try base.grad(&ctx)).?;
    defer gb.deinit();
    var gu = (try update.grad(&ctx)).?;
    defer gu.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 0, 0, 0, 0, 1, 1 }, gb.asRawTensor().dataConst());
    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 1, 1 }, gu.asRawTensor().dataConst());

    var rows_base = try Tensor(.{ .row, .d }).variableFromSlice(&ctx, .{ 4, 2 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer rows_base.deinit();
    var rows_update = try Tensor(.{ .row, .d }).variableFromSlice(&ctx, .{ 2, 2 }, &.{ 50, 60, 70, 80 });
    defer rows_update.deinit();

    var rows = try rows_base.setRows(&ctx, .row, &.{ 2, 0 }, &rows_update);
    defer rows.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 70, 80, 3, 4, 50, 60, 7, 8 }, rows.asRawTensor().dataConst());

    var rows_loss = try rows.sumAll(&ctx);
    defer rows_loss.deinit();
    try rows_loss.backward(&ctx);

    var grb = (try rows_base.grad(&ctx)).?;
    defer grb.deinit();
    var gru = (try rows_update.grad(&ctx)).?;
    defer gru.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 1, 1, 0, 0, 1, 1 }, grb.asRawTensor().dataConst());
    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 1, 1 }, gru.asRawTensor().dataConst());
}

test "tagged public tensor exposes argmax and topK sampling helpers" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .batch, .d }).variableFromSlice(&ctx, .{ 2, 4 }, &.{ 1, 5, 2, 3, -1, 2, 4, 0 });
    defer x.deinit();

    var arg = try x.argmax(&ctx, .d);
    defer arg.deinit();
    try std.testing.expect(!arg.requiresGrad());
    try std.testing.expectEqualSlices(usize, &.{2}, arg.asRawTensor().shape.slice());
    comptime std.debug.assert(@TypeOf(arg).dtype == .i64);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2 }, arg.asRawTensor().dataConst());

    var top = try x.topK(&ctx, .d, 2, .k);
    defer top.deinit();
    try std.testing.expect(top.values.requiresGrad());
    try std.testing.expect(!top.indices.requiresGrad());
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, top.values.asRawTensor().shape.slice());
    try std.testing.expectEqualSlices(f32, &.{ 5, 3, 4, 2 }, top.values.asRawTensor().dataConst());
    comptime std.debug.assert(@TypeOf(top.indices).dtype == .i64);
    try std.testing.expectEqualSlices(i64, &.{ 1, 3, 2, 1 }, top.indices.asRawTensor().dataConst());
}

test "tagged public tensor causal depthwise conv uses optional history state" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var input = try Tensor(.{ .time, .channel }).fromSlice(&ctx, .{ 3, 2 }, &.{
        1, 10,
        2, 20,
        3, 30,
    });
    defer input.deinit();
    var kernel = try Tensor(.{ .channel, .tap }).fromSlice(&ctx, .{ 2, 3 }, &.{
        1,  2,  3,
        10, 20, 30,
    });
    defer kernel.deinit();

    var no_state = try input.causalDepthwiseConv1d(&ctx, .time, .channel, .tap, &kernel, null);
    defer no_state.deinit();
    try std.testing.expectEqualSlices(f32, &.{
        3,  300,
        8,  800,
        14, 1400,
    }, no_state.asRawTensor().dataConst());

    var state = [_]f32{
        -2, -20,
        4,  40,
    };
    var with_state = try input.causalDepthwiseConv1d(&ctx, .time, .channel, .tap, &kernel, &state);
    defer with_state.deinit();
    try std.testing.expectEqualSlices(f32, &.{
        9,  900,
        12, 1200,
        14, 1400,
    }, with_state.asRawTensor().dataConst());
}

test "tagged public tensor causal depthwise conv propagates input and kernel gradients" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var input = try Tensor(.{ .time, .channel }).variableFromSlice(&ctx, .{ 3, 2 }, &.{
        1, 10,
        2, 20,
        3, 30,
    });
    defer input.deinit();
    var kernel = try Tensor(.{ .channel, .tap }).variableFromSlice(&ctx, .{ 2, 3 }, &.{
        1,  2,  3,
        10, 20, 30,
    });
    defer kernel.deinit();
    var state = [_]f32{
        -1,  -10,
        0.5, 5,
    };

    var out = try input.causalDepthwiseConv1d(&ctx, .time, .channel, .tap, &kernel, &state);
    defer out.deinit();
    var loss = try out.sumAll(&ctx);
    defer loss.deinit();

    try loss.backward(&ctx);

    var gx = (try input.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{
        6, 60,
        5, 50,
        3, 30,
    }, gx.asRawTensor().dataConst());

    var gk = (try kernel.grad(&ctx)).?;
    defer gk.deinit();
    try std.testing.expectEqualSlices(f32, &.{
        0.5, 3.5, 6,
        5,   35,  60,
    }, gk.asRawTensor().dataConst());
}

test "tagged public tensor general causal conv mixes channels with optional state" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var input = try Tensor(.{ .time, .in }).fromSlice(&ctx, .{ 3, 2 }, &.{
        1, 10,
        2, 20,
        3, 30,
    });
    defer input.deinit();
    // w[k][i][o], k=1 is the newest tap.
    var weight = try Tensor(.{ .tap, .in, .out }).fromSlice(&ctx, .{ 2, 2, 2 }, &.{
        10, 20,
        30, 40,
        1,  2,
        3,  4,
    });
    defer weight.deinit();

    var no_state = try input.causalConv1d(&ctx, .time, .in, .tap, .out, &weight, 1, null);
    defer no_state.deinit();
    try std.testing.expectEqualSlices(f32, &.{
        31,  42,
        372, 504,
        713, 966,
    }, no_state.asRawTensor().dataConst());

    var state = [_]f32{ 5, 7 };
    var with_state = try input.causalConv1d(&ctx, .time, .in, .tap, .out, &weight, 1, &state);
    defer with_state.deinit();
    try std.testing.expectEqualSlices(f32, &.{
        291, 422,
        372, 504,
        713, 966,
    }, with_state.asRawTensor().dataConst());

    // Dilation 2, in=out=1: y[t] = 10*x[t-2] + x[t], state covers t<2 history.
    var mono = try Tensor(.{ .time, .in }).fromSlice(&ctx, .{ 4, 1 }, &.{ 1, 2, 3, 4 });
    defer mono.deinit();
    var taps2 = try Tensor(.{ .tap, .in, .out }).fromSlice(&ctx, .{ 2, 1, 1 }, &.{ 10, 1 });
    defer taps2.deinit();
    var dilated_state = [_]f32{ 100, 200 };
    var dilated = try mono.causalConv1d(&ctx, .time, .in, .tap, .out, &taps2, 2, &dilated_state);
    defer dilated.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1001, 2002, 13, 24 }, dilated.asRawTensor().dataConst());
}

test "tagged public tensor general causal conv propagates input and weight gradients" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var input = try Tensor(.{ .time, .in }).variableFromSlice(&ctx, .{ 3, 2 }, &.{
        1, 10,
        2, 20,
        3, 30,
    });
    defer input.deinit();
    var weight = try Tensor(.{ .tap, .in, .out }).variableFromSlice(&ctx, .{ 2, 2, 2 }, &.{
        10, 20,
        30, 40,
        1,  2,
        3,  4,
    });
    defer weight.deinit();
    var state = [_]f32{ 5, 7 };

    var out = try input.causalConv1d(&ctx, .time, .in, .tap, .out, &weight, 1, &state);
    defer out.deinit();
    var loss = try out.sumAll(&ctx);
    defer loss.deinit();

    try loss.backward(&ctx);

    var gx = (try input.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{
        33, 77,
        33, 77,
        3,  7,
    }, gx.asRawTensor().dataConst());

    var gw = (try weight.grad(&ctx)).?;
    defer gw.deinit();
    try std.testing.expectEqualSlices(f32, &.{
        8,  8,
        37, 37,
        6,  6,
        60, 60,
    }, gw.asRawTensor().dataConst());
}

test "tagged public tensor grouped causal conv partitions channels with state" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var input = try Tensor(.{ .time, .in }).fromSlice(&ctx, .{ 3, 4 }, &.{
        1, 10, 100, 1000,
        2, 20, 200, 2000,
        3, 30, 300, 3000,
    });
    defer input.deinit();
    // w[k][local_input][out]; groups=2, so out 0/1 read input 0/1 and
    // out 2/3 read input 2/3.
    var weight = try Tensor(.{ .tap, .in_group, .out }).fromSlice(&ctx, .{ 2, 2, 4 }, &.{
        10,  20,   30,    40,
        1,   2,    3,     4,
        5,   6,    7,     8,
        0.5, 0.25, 0.125, 0.0625,
    });
    defer weight.deinit();
    var state = [_]f32{ 9, 90, 900, 9000 };

    var out = try input.groupedCausalConv1d(&ctx, .time, .in, .tap, .in_group, .out, &weight, 1, 2, &state);
    defer out.deinit();
    try std.testing.expectEqualSlices(f32, &.{
        190, 368.5, 54825, 72862.5,
        40,  57,    7650,  9725,
        70,  105.5, 14475, 18587.5,
    }, out.asRawTensor().dataConst());
}

test "tagged public tensor grouped causal conv propagates input and weight gradients" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var input = try Tensor(.{ .time, .in }).variableFromSlice(&ctx, .{ 3, 4 }, &.{
        1, 10, 100, 1000,
        2, 20, 200, 2000,
        3, 30, 300, 3000,
    });
    defer input.deinit();
    var weight = try Tensor(.{ .tap, .in_group, .out }).variableFromSlice(&ctx, .{ 2, 2, 4 }, &.{
        10,  20,   30,    40,
        1,   2,    3,     4,
        5,   6,    7,     8,
        0.5, 0.25, 0.125, 0.0625,
    });
    defer weight.deinit();
    var state = [_]f32{ 9, 90, 900, 9000 };

    var out = try input.groupedCausalConv1d(&ctx, .time, .in, .tap, .in_group, .out, &weight, 1, 2, &state);
    defer out.deinit();
    var loss = try out.sumAll(&ctx);
    defer loss.deinit();

    try loss.backward(&ctx);

    var gx = (try input.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{
        41, 3.75, 85, 7.1875,
        41, 3.75, 85, 7.1875,
        11, 0.75, 15, 0.1875,
    }, gx.asRawTensor().dataConst());

    var gw = (try weight.grad(&ctx)).?;
    defer gw.deinit();
    try std.testing.expectEqualSlices(f32, &.{
        12,  12,  1200,  1200,
        120, 120, 12000, 12000,
        6,   6,   600,   600,
        60,  60,  6000,  6000,
    }, gw.asRawTensor().dataConst());
}

test "tagged public tensor general causal conv dilated gradients" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var input = try Tensor(.{ .time, .in }).variableFromSlice(&ctx, .{ 4, 1 }, &.{ 1, 2, 3, 4 });
    defer input.deinit();
    var weight = try Tensor(.{ .tap, .in, .out }).variableFromSlice(&ctx, .{ 2, 1, 1 }, &.{ 10, 1 });
    defer weight.deinit();
    var state = [_]f32{ 100, 200 };

    var out = try input.causalConv1d(&ctx, .time, .in, .tap, .out, &weight, 2, &state);
    defer out.deinit();
    var loss = try out.sumAll(&ctx);
    defer loss.deinit();

    try loss.backward(&ctx);

    var gx = (try input.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 11, 11, 1, 1 }, gx.asRawTensor().dataConst());

    var gw = (try weight.grad(&ctx)).?;
    defer gw.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 303, 10 }, gw.asRawTensor().dataConst());
}

test "tagged public tensor keeps constant operands alive for gradients" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(1).variableFromSlice(&ctx, .{3}, &.{ 1, 2, 3 });
    defer x.deinit();
    var scale = try Tensor(1).fromSlice(&ctx, .{3}, &.{ 10, 20, 30 });
    defer scale.deinit();

    var y = try x.mul(&ctx, &scale);
    defer y.deinit();
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();

    try loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 10, 20, 30 }, gx.asRawTensor().dataConst());
    try std.testing.expect((try scale.grad(&ctx)) == null);
}

test "tagged autograd reduces broadcast pointwise gradients by tag" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .batch, .d }).variable(&ctx, try ctx.fromSlice(&.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 }));
    defer x.deinit();
    var bias = try Tensor(.{.d}).variable(&ctx, try ctx.fromSlice(&.{3}, &.{ 10, 20, 30 }));
    defer bias.deinit();

    var y = try x.add(&ctx, &bias);
    defer y.deinit();
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();

    try loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    var gb = (try bias.grad(&ctx)).?;
    defer gb.deinit();

    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 1, 1, 1, 1 }, gx.asRawTensor().dataConst());
    try std.testing.expectEqualSlices(f32, &.{ 2, 2, 2 }, gb.asRawTensor().dataConst());
}

test "tagged autograd permutes and reduces named axes" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .batch, .seq, .d }).variable(
        &ctx,
        try ctx.fromSlice(
            &.{ 2, 3, 2 },
            &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 },
        ),
    );
    defer x.deinit();

    var p = try x.permuteTo(&ctx, .{ .d, .batch, .seq });
    defer p.deinit();
    var reduced = try p.sumMany(&ctx, .{ .batch, .seq });
    defer reduced.deinit();
    var loss = try reduced.sumAll(&ctx);
    defer loss.deinit();

    try loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }, gx.asRawTensor().dataConst());
}

test "strided view autograd scatters a transposed alias gradient" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .batch, .d }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();

    var xt = try x.viewWithStrides(&ctx, .{ .d, .batch }, .{ 3, 2 }, .{ 1, 3 });
    defer xt.deinit();
    var weights = try Tensor(.{ .d, .batch }).fromSlice(&ctx, .{ 3, 2 }, &.{ 10, 20, 30, 40, 50, 60 });
    defer weights.deinit();

    var weighted = try xt.mul(&ctx, &weights);
    defer weighted.deinit();
    var loss = try weighted.sumAll(&ctx);
    defer loss.deinit();

    try loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 10, 30, 50, 20, 40, 60 }, gx.asRawTensor().dataConst());
}

test "tagged autograd split and merge axes keep gradients as views" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .batch, .flat }).variable(&ctx, try ctx.fromSlice(&.{ 2, 6 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }));
    defer x.deinit();

    var split = try x.split(&ctx, .flat, .{ .row, .col }, .{ 2, 3 });
    defer split.deinit();
    var merged = try split.merge(&ctx, .flat2, .{ .row, .col });
    defer merged.deinit();
    var loss = try merged.sumAll(&ctx);
    defer loss.deinit();

    try loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }, gx.asRawTensor().dataConst());
}

test "tagged autograd squeeze can produce a scalar-tag value" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{.singleton}).variable(&ctx, try ctx.fromSlice(&.{1}, &.{2}));
    defer x.deinit();

    var y = try x.squeeze(&ctx, .singleton);
    defer y.deinit();
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();

    try loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{1}, gx.asRawTensor().dataConst());
}

test "tagged autograd broadcasts scalar-tag values as raw scalar tensors" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{}).variable(&ctx, try ctx.scalar(2));
    defer x.deinit();

    var y = try x.broadcastTo(&ctx, .{}, .{});
    defer y.deinit();
    try std.testing.expectEqualSlices(usize, &.{1}, y.asRawTensor().shape.slice());
    try std.testing.expectEqual(@as(f32, 2), y.asRawTensor().item());

    var loss = try y.sumAll(&ctx);
    defer loss.deinit();

    try loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{1}, gx.asRawTensor().dataConst());
}

test "tagged autograd dot contracts by tag and propagates gradients" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try Tensor(.{ .batch, .m, .k }).variable(
        &ctx,
        try ctx.fromSlice(&.{ 2, 2, 3 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }),
    );
    defer a.deinit();
    var b = try Tensor(.{ .batch, .k, .n }).variable(
        &ctx,
        try ctx.fromSlice(&.{ 2, 3, 2 }, &.{ 1, 10, 2, 20, 3, 30, 4, 40, 5, 50, 6, 60 }),
    );
    defer b.deinit();

    var y = try a.dot(&ctx, &b, .k);
    defer y.deinit();
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();

    try loss.backward(&ctx);

    var ga = (try a.grad(&ctx)).?;
    defer ga.deinit();
    var gb = (try b.grad(&ctx)).?;
    defer gb.deinit();

    try std.testing.expectEqualSlices(f32, &.{ 11, 22, 33, 11, 22, 33, 44, 55, 66, 44, 55, 66 }, ga.asRawTensor().dataConst());
    try std.testing.expectEqualSlices(f32, &.{ 5, 5, 7, 7, 9, 9, 17, 17, 19, 19, 21, 21 }, gb.asRawTensor().dataConst());
}

test "tagged autograd dot handles non-physical axis order through raw graph ops" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try Tensor(.{ .batch, .m, .k }).variable(
        &ctx,
        try ctx.fromSlice(&.{ 2, 2, 3 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }),
    );
    defer a.deinit();
    var b = try Tensor(.{ .n, .batch, .k }).variable(
        &ctx,
        try ctx.fromSlice(&.{ 2, 2, 3 }, &.{ 1, 2, 3, 4, 5, 6, 10, 20, 30, 40, 50, 60 }),
    );
    defer b.deinit();

    var y = try a.dot(&ctx, &b, .k);
    defer y.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 2 }, y.asRawTensor().shape.slice());
    try std.testing.expectEqualSlices(f32, &.{ 14, 140, 32, 320, 122, 1220, 167, 1670 }, y.asRawTensor().dataConst());

    var loss = try y.sumAll(&ctx);
    defer loss.deinit();

    try loss.backward(&ctx);

    var ga = (try a.grad(&ctx)).?;
    defer ga.deinit();
    var gb = (try b.grad(&ctx)).?;
    defer gb.deinit();

    try std.testing.expectEqualSlices(f32, &.{ 11, 22, 33, 11, 22, 33, 44, 55, 66, 44, 55, 66 }, ga.asRawTensor().dataConst());
    try std.testing.expectEqualSlices(f32, &.{ 5, 7, 9, 17, 19, 21, 5, 7, 9, 17, 19, 21 }, gb.asRawTensor().dataConst());
}

test "tagged autograd gathers embedding rows and scatter-adds gradients" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var table = try Tensor(.{ .vocab, .d }).variable(
        &ctx,
        try ctx.fromSlice(&.{ 4, 2 }, &.{ 1, 10, 2, 20, 3, 30, 4, 40 }),
    );
    defer table.deinit();

    var y = try table.gather(&ctx, .vocab, &.{ 2, 0, 2 }, .token);
    defer y.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, y.asRawTensor().shape.slice());
    try std.testing.expectEqualSlices(f32, &.{ 3, 30, 1, 10, 3, 30 }, y.asRawTensor().dataConst());

    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var grad = (try table.grad(&ctx)).?;
    defer grad.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 0, 0, 2, 2, 0, 0 }, grad.asRawTensor().dataConst());
}

test "tagged autograd differentiates div mean and scalar unary math" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{.d}).variableFromSlice(&ctx, .{2}, &.{ 1, 4 });
    defer x.deinit();
    var denom = try Tensor(.{.d}).fromSlice(&ctx, .{2}, &.{ 2, 8 });
    defer denom.deinit();

    var q = try x.div(&ctx, &denom);
    defer q.deinit();
    var exp_x = try x.exp(&ctx);
    defer exp_x.deinit();
    var sqrt_x = try x.sqrt(&ctx);
    defer sqrt_x.deinit();
    var rsqrt_x = try x.rsqrt(&ctx);
    defer rsqrt_x.deinit();
    var sigmoid_x = try x.sigmoid(&ctx);
    defer sigmoid_x.deinit();
    var silu_x = try x.silu(&ctx);
    defer silu_x.deinit();
    var log_x = try x.log(&ctx);
    defer log_x.deinit();

    var total = try q.add(&ctx, &exp_x);
    defer total.deinit();
    var total2 = try total.add(&ctx, &sqrt_x);
    defer total2.deinit();
    var total3 = try total2.add(&ctx, &rsqrt_x);
    defer total3.deinit();
    var total4 = try total3.add(&ctx, &sigmoid_x);
    defer total4.deinit();
    var total5 = try total4.add(&ctx, &silu_x);
    defer total5.deinit();
    var total6 = try total5.add(&ctx, &log_x);
    defer total6.deinit();
    var loss = try total6.sumAll(&ctx);
    defer loss.deinit();

    try loss.backward(&ctx);

    var grad = (try x.grad(&ctx)).?;
    defer grad.deinit();
    const data = grad.asRawTensor().dataConst();
    const s0 = testSigmoid(1);
    const s1 = testSigmoid(4);
    const expected = [_]f32{
        1.0 / 2.0 + @exp(@as(f32, 1)) + 0.5 / @sqrt(@as(f32, 1)) - 0.5 / (@as(f32, 1) * @sqrt(@as(f32, 1))) + s0 * (1 - s0) + s0 * (1 + @as(f32, 1) * (1 - s0)) + 1.0 / 1.0,
        1.0 / 8.0 + @exp(@as(f32, 4)) + 0.5 / @sqrt(@as(f32, 4)) - 0.5 / (@as(f32, 4) * @sqrt(@as(f32, 4))) + s1 * (1 - s1) + s1 * (1 + @as(f32, 4) * (1 - s1)) + 1.0 / 4.0,
    };
    try expectCloseSlices(&expected, data, 1e-5);

    var x2 = try Tensor(.{ .batch, .d }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x2.deinit();
    var mean = try x2.mean(&ctx, .d);
    defer mean.deinit();
    var mean_loss = try mean.sumAll(&ctx);
    defer mean_loss.deinit();
    try mean_loss.backward(&ctx);
    var mean_grad = (try x2.grad(&ctx)).?;
    defer mean_grad.deinit();
    try expectCloseSlices(&.{ 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0 }, mean_grad.asRawTensor().dataConst(), 1e-6);
}

test "tagged autograd differentiates extended unary ops and clamp" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{.d}).variableFromSlice(&ctx, .{2}, &.{ -0.75, 1.25 });
    defer x.deinit();

    var neg_x = try x.neg(&ctx);
    defer neg_x.deinit();
    var abs_x = try x.abs(&ctx);
    defer abs_x.deinit();
    var sin_x = try x.sin(&ctx);
    defer sin_x.deinit();
    var cos_x = try x.cos(&ctx);
    defer cos_x.deinit();
    var tanh_x = try x.tanh(&ctx);
    defer tanh_x.deinit();
    var gelu_x = try x.gelu(&ctx);
    defer gelu_x.deinit();
    var quick_x = try x.quickGelu(&ctx);
    defer quick_x.deinit();
    var clamp_x = try x.clamp(&ctx, -0.5, 1.0);
    defer clamp_x.deinit();

    var total = try neg_x.add(&ctx, &abs_x);
    defer total.deinit();
    var total2 = try total.add(&ctx, &sin_x);
    defer total2.deinit();
    var total3 = try total2.add(&ctx, &cos_x);
    defer total3.deinit();
    var total4 = try total3.add(&ctx, &tanh_x);
    defer total4.deinit();
    var total5 = try total4.add(&ctx, &gelu_x);
    defer total5.deinit();
    var total6 = try total5.add(&ctx, &quick_x);
    defer total6.deinit();
    var total7 = try total6.add(&ctx, &clamp_x);
    defer total7.deinit();
    var loss = try total7.sumAll(&ctx);
    defer loss.deinit();

    try loss.backward(&ctx);

    var grad = (try x.grad(&ctx)).?;
    defer grad.deinit();

    const x0: f32 = -0.75;
    const x1: f32 = 1.25;
    const expected = [_]f32{
        -1 - 1 + @cos(x0) - @sin(x0) + testTanhDerivative(x0) + testGeluDerivative(x0) + testQuickGeluDerivative(x0) + 0,
        -1 + 1 + @cos(x1) - @sin(x1) + testTanhDerivative(x1) + testGeluDerivative(x1) + testQuickGeluDerivative(x1) + 0,
    };
    try expectCloseSlices(&expected, grad.asRawTensor().dataConst(), 1e-5);
}

test "tagged autograd differentiates fused glu and swiglu" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try Tensor(.{.d}).variableFromSlice(&ctx, .{2}, &.{ 2, -3 });
    defer a.deinit();
    var gate = try Tensor(.{.d}).variableFromSlice(&ctx, .{2}, &.{ 0, 1 });
    defer gate.deinit();

    var y = try a.swiglu(&ctx, &gate);
    defer y.deinit();
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var ga = (try a.grad(&ctx)).?;
    defer ga.deinit();
    var gg = (try gate.grad(&ctx)).?;
    defer gg.deinit();

    const s0 = testSigmoid(0);
    const s1 = testSigmoid(1);
    try expectCloseSlices(&.{ 0 * s0, 1 * s1 }, ga.asRawTensor().dataConst(), 1e-6);
    try expectCloseSlices(&.{ 2 * s0 * (1 + 0 * (1 - s0)), -3 * s1 * (1 + 1 * (1 - s1)) }, gg.asRawTensor().dataConst(), 1e-6);

    var left = try Tensor(.{.d}).fromSlice(&ctx, .{2}, &.{ 2, -3 });
    defer left.deinit();
    var y_glu = try left.glu(&ctx, &gate);
    defer y_glu.deinit();
    try expectCloseSlices(&.{ 2 * s0, -3 * s1 }, y_glu.asRawTensor().dataConst(), 1e-6);
}

test "tagged autograd f32 cast preserves the graph" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ 1, -2, 4 });
    defer x.deinit();

    var y = try x.to(&ctx, .f32);
    defer y.deinit();
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var grad = (try x.grad(&ctx)).?;
    defer grad.deinit();
    try expectCloseSlices(&.{ 1, 1, 1 }, grad.asRawTensor().dataConst(), 1e-6);
    // f16/bf16 narrows are differentiable (the mixed-precision seam);
    // f64 stays a no-grad-only cast.
    var narrowed = try x.to(&ctx, .f16);
    defer narrowed.deinit();
    try std.testing.expect(narrowed.requiresGrad());
    try std.testing.expectError(error.GradientCastUnsupported, x.to(&ctx, .f64));
}

test "tagged autograd differentiates splitSwiGlu along the fused axis" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .batch, .ff }).variableFromSlice(&ctx, .{ 1, 4 }, &.{ 0, 1, 2, -3 });
    defer x.deinit();

    var y = try x.splitGated(&ctx, .swiglu, .ff, .d);
    defer y.deinit();
    const s0 = testSigmoid(0);
    const s1 = testSigmoid(1);
    try expectCloseSlices(&.{ 2 * 0 * s0, -3 * 1 * s1 }, y.asRawTensor().dataConst(), 1e-6);

    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var grad = (try x.grad(&ctx)).?;
    defer grad.deinit();
    try expectCloseSlices(
        &.{ 2 * s0 * (1 + 0 * (1 - s0)), -3 * s1 * (1 + 1 * (1 - s1)), 0 * s0, 1 * s1 },
        grad.asRawTensor().dataConst(),
        1e-6,
    );
}

test "tagged autograd differentiates splitGlu along the fused axis" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .batch, .ff }).variableFromSlice(&ctx, .{ 2, 4 }, &.{
        2, -3, 0, 0,
        4, 6,  0, 0,
    });
    defer x.deinit();

    var y = try x.splitGated(&ctx, .glu, .ff, .d);
    defer y.deinit();
    try expectCloseSlices(&.{ 1, -1.5, 2, 3 }, y.asRawTensor().dataConst(), 1e-6);

    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var grad = (try x.grad(&ctx)).?;
    defer grad.deinit();
    try expectCloseSlices(
        &.{ 0.5, 0.5, 0.5, -0.75, 0.5, 0.5, 1, 1.5 },
        grad.asRawTensor().dataConst(),
        1e-6,
    );
}

test "tagged autograd rmsNormMul matches rmsNorm followed by weight multiply" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const x_values = [_]f32{ 1, 2, 3, 2, 0, 4 };
    const w_values = [_]f32{ 0.5, -1.5, 2 };

    var x = try Tensor(.{ .batch, .d }).variableFromSlice(&ctx, .{ 2, 3 }, &x_values);
    defer x.deinit();
    var w = try Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &w_values);
    defer w.deinit();
    var fused = try x.rmsNormMul(&ctx, .d, &w, 1e-5);
    defer fused.deinit();
    var fused_loss = try fused.sumAll(&ctx);
    defer fused_loss.deinit();
    try fused_loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    var gw = (try w.grad(&ctx)).?;
    defer gw.deinit();

    var x_ref = try Tensor(.{ .batch, .d }).variableFromSlice(&ctx, .{ 2, 3 }, &x_values);
    defer x_ref.deinit();
    var w_ref = try Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &w_values);
    defer w_ref.deinit();
    var norm = try x_ref.rmsNorm(&ctx, .d, 1e-5);
    defer norm.deinit();
    var ref_y = try norm.mul(&ctx, &w_ref);
    defer ref_y.deinit();
    var ref_loss = try ref_y.sumAll(&ctx);
    defer ref_loss.deinit();
    try ref_loss.backward(&ctx);

    var gx_ref = (try x_ref.grad(&ctx)).?;
    defer gx_ref.deinit();
    var gw_ref = (try w_ref.grad(&ctx)).?;
    defer gw_ref.deinit();
    try expectCloseSlices(gx_ref.asRawTensor().dataConst(), gx.asRawTensor().dataConst(), 1e-5);
    try expectCloseSlices(gw_ref.asRawTensor().dataConst(), gw.asRawTensor().dataConst(), 1e-5);
}

test "tagged autograd rmsNormMulAdd matches rmsNormMul plus residual add" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const x_values = [_]f32{ 1, -2, 3, 4, 0.5, -1 };
    const w_values = [_]f32{ 0.5, -1.5, 2 };
    const r_values = [_]f32{ 0.25, 0.5, -0.75, 1.25, -1, 0.125 };
    const gy_values = [_]f32{ 1, -0.5, 0.25, -1.5, 2, 0.75 };

    var x = try Tensor(.{ .batch, .d }).variableFromSlice(&ctx, .{ 2, 3 }, &x_values);
    defer x.deinit();
    var w = try Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &w_values);
    defer w.deinit();
    var r = try Tensor(.{ .batch, .d }).variableFromSlice(&ctx, .{ 2, 3 }, &r_values);
    defer r.deinit();
    var gy = try Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ 2, 3 }, &gy_values);
    defer gy.deinit();

    var fused = try x.rmsNormMulAdd(&ctx, .d, &w, &r, 1e-5);
    defer fused.deinit();
    var fused_weighted = try fused.mul(&ctx, &gy);
    defer fused_weighted.deinit();
    var fused_loss = try fused_weighted.sumAll(&ctx);
    defer fused_loss.deinit();
    try fused_loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    var gw = (try w.grad(&ctx)).?;
    defer gw.deinit();
    var gr = (try r.grad(&ctx)).?;
    defer gr.deinit();

    var x_ref = try Tensor(.{ .batch, .d }).variableFromSlice(&ctx, .{ 2, 3 }, &x_values);
    defer x_ref.deinit();
    var w_ref = try Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &w_values);
    defer w_ref.deinit();
    var r_ref = try Tensor(.{ .batch, .d }).variableFromSlice(&ctx, .{ 2, 3 }, &r_values);
    defer r_ref.deinit();
    var norm = try x_ref.rmsNormMul(&ctx, .d, &w_ref, 1e-5);
    defer norm.deinit();
    var ref_y = try r_ref.add(&ctx, &norm);
    defer ref_y.deinit();
    var ref_weighted = try ref_y.mul(&ctx, &gy);
    defer ref_weighted.deinit();
    var ref_loss = try ref_weighted.sumAll(&ctx);
    defer ref_loss.deinit();
    try ref_loss.backward(&ctx);

    var gx_ref = (try x_ref.grad(&ctx)).?;
    defer gx_ref.deinit();
    var gw_ref = (try w_ref.grad(&ctx)).?;
    defer gw_ref.deinit();
    var gr_ref = (try r_ref.grad(&ctx)).?;
    defer gr_ref.deinit();

    try expectCloseSlices(ref_y.asRawTensor().dataConst(), fused.asRawTensor().dataConst(), 1e-6);
    try expectCloseSlices(gx_ref.asRawTensor().dataConst(), gx.asRawTensor().dataConst(), 1e-5);
    try expectCloseSlices(gw_ref.asRawTensor().dataConst(), gw.asRawTensor().dataConst(), 1e-5);
    try expectCloseSlices(gr_ref.asRawTensor().dataConst(), gr.asRawTensor().dataConst(), 1e-6);
}

test "tagged autograd grouped causal attention matches finite differences" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const q_values = [_]f32{ 0.2, -0.4, 0.5, 0.1 };
    const k_values = [_]f32{ 0.3, -0.2, -0.1, 0.4 };
    const v_values = [_]f32{ 0.7, -0.6, 0.2, 0.5 };
    const kv_head_for_head = [_]usize{0};
    const scale_value: f32 = 0.7;

    var q = try Tensor(.{ .seq, .head, .d }).variableFromSlice(&ctx, .{ 2, 1, 2 }, &q_values);
    defer q.deinit();
    var k = try Tensor(.{ .seq, .kv_head, .d }).variableFromSlice(&ctx, .{ 2, 1, 2 }, &k_values);
    defer k.deinit();
    var v = try Tensor(.{ .seq, .kv_head, .d }).variableFromSlice(&ctx, .{ 2, 1, 2 }, &v_values);
    defer v.deinit();

    var y = try q.groupedAttention(&ctx, &k, &v, kv_head_for_head[0..], .out, scale_value, .{});
    defer y.deinit();
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var gq = (try q.grad(&ctx)).?;
    defer gq.deinit();
    var gk = (try k.grad(&ctx)).?;
    defer gk.deinit();
    var gv = (try v.grad(&ctx)).?;
    defer gv.deinit();

    const eps: f32 = 1e-3;
    var q_work = q_values;
    for (q_values, 0..) |_, i| {
        q_work = q_values;
        q_work[i] += eps;
        const plus = try groupedAttentionTinyLoss(&ctx, q_work[0..], k_values[0..], v_values[0..], kv_head_for_head[0..], scale_value);
        q_work[i] -= 2 * eps;
        const minus = try groupedAttentionTinyLoss(&ctx, q_work[0..], k_values[0..], v_values[0..], kv_head_for_head[0..], scale_value);
        const expected = (plus - minus) / (2 * eps);
        try std.testing.expectApproxEqAbs(expected, gq.asRawTensor().dataConst()[i], 2e-2);
    }

    var k_work = k_values;
    for (k_values, 0..) |_, i| {
        k_work = k_values;
        k_work[i] += eps;
        const plus = try groupedAttentionTinyLoss(&ctx, q_values[0..], k_work[0..], v_values[0..], kv_head_for_head[0..], scale_value);
        k_work[i] -= 2 * eps;
        const minus = try groupedAttentionTinyLoss(&ctx, q_values[0..], k_work[0..], v_values[0..], kv_head_for_head[0..], scale_value);
        const expected = (plus - minus) / (2 * eps);
        try std.testing.expectApproxEqAbs(expected, gk.asRawTensor().dataConst()[i], 2e-2);
    }

    var v_work = v_values;
    for (v_values, 0..) |_, i| {
        v_work = v_values;
        v_work[i] += eps;
        const plus = try groupedAttentionTinyLoss(&ctx, q_values[0..], k_values[0..], v_work[0..], kv_head_for_head[0..], scale_value);
        v_work[i] -= 2 * eps;
        const minus = try groupedAttentionTinyLoss(&ctx, q_values[0..], k_values[0..], v_work[0..], kv_head_for_head[0..], scale_value);
        const expected = (plus - minus) / (2 * eps);
        try std.testing.expectApproxEqAbs(expected, gv.asRawTensor().dataConst()[i], 2e-2);
    }
}

test "tagged autograd grouped causal attention GEMM backward matches finite differences" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const S = 16;
    const H = 4;
    const KV = 2;
    const D = 16;
    const kv_head_for_head = [_]usize{ 0, 0, 1, 1 };
    const scale_value: f32 = 0.35;

    var q_values: [S * H * D]f32 = undefined;
    var k_values: [S * KV * D]f32 = undefined;
    var v_values: [S * KV * D]f32 = undefined;
    for (&q_values, 0..) |*value, i| {
        const x = @as(f32, @floatFromInt(i));
        value.* = @sin(x * 0.071) * 0.4 + @cos(x * 0.037) * 0.11;
    }
    for (&k_values, 0..) |*value, i| {
        const x = @as(f32, @floatFromInt(i));
        value.* = @cos(x * 0.053) * 0.3 - @sin(x * 0.019) * 0.07;
    }
    for (&v_values, 0..) |*value, i| {
        const x = @as(f32, @floatFromInt(i));
        value.* = @sin(x * 0.041 + 0.2) * 0.5;
    }

    var q = try Tensor(.{ .seq, .head, .d }).variableFromSlice(&ctx, .{ S, H, D }, &q_values);
    defer q.deinit();
    var k = try Tensor(.{ .seq, .kv_head, .d }).variableFromSlice(&ctx, .{ S, KV, D }, &k_values);
    defer k.deinit();
    var v = try Tensor(.{ .seq, .kv_head, .d }).variableFromSlice(&ctx, .{ S, KV, D }, &v_values);
    defer v.deinit();

    var y = try q.groupedAttention(&ctx, &k, &v, kv_head_for_head[0..], .out, scale_value, .{});
    defer y.deinit();
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var gq = (try q.grad(&ctx)).?;
    defer gq.deinit();
    var gk = (try k.grad(&ctx)).?;
    defer gk.deinit();
    var gv = (try v.grad(&ctx)).?;
    defer gv.deinit();

    const eps: f32 = 1e-2;
    const q_probe = [_]usize{ 0, 137, q_values.len - 5 };
    var q_work = q_values;
    for (q_probe) |i| {
        q_work = q_values;
        q_work[i] += eps;
        const plus = try groupedAttentionLoss(&ctx, S, H, KV, D, q_work[0..], k_values[0..], v_values[0..], kv_head_for_head[0..], scale_value);
        q_work[i] -= 2 * eps;
        const minus = try groupedAttentionLoss(&ctx, S, H, KV, D, q_work[0..], k_values[0..], v_values[0..], kv_head_for_head[0..], scale_value);
        const expected = (plus - minus) / (2 * eps);
        try std.testing.expectApproxEqAbs(expected, gq.asRawTensor().dataConst()[i], 2e-2);
    }

    const k_probe = [_]usize{ 3, 91, k_values.len - 7 };
    var k_work = k_values;
    for (k_probe) |i| {
        k_work = k_values;
        k_work[i] += eps;
        const plus = try groupedAttentionLoss(&ctx, S, H, KV, D, q_values[0..], k_work[0..], v_values[0..], kv_head_for_head[0..], scale_value);
        k_work[i] -= 2 * eps;
        const minus = try groupedAttentionLoss(&ctx, S, H, KV, D, q_values[0..], k_work[0..], v_values[0..], kv_head_for_head[0..], scale_value);
        const expected = (plus - minus) / (2 * eps);
        try std.testing.expectApproxEqAbs(expected, gk.asRawTensor().dataConst()[i], 2e-2);
    }

    const v_probe = [_]usize{ 11, 173, v_values.len - 1 };
    var v_work = v_values;
    for (v_probe) |i| {
        v_work = v_values;
        v_work[i] += eps;
        const plus = try groupedAttentionLoss(&ctx, S, H, KV, D, q_values[0..], k_values[0..], v_work[0..], kv_head_for_head[0..], scale_value);
        v_work[i] -= 2 * eps;
        const minus = try groupedAttentionLoss(&ctx, S, H, KV, D, q_values[0..], k_values[0..], v_work[0..], kv_head_for_head[0..], scale_value);
        const expected = (plus - minus) / (2 * eps);
        try std.testing.expectApproxEqAbs(expected, gv.asRawTensor().dataConst()[i], 2e-2);
    }
}

fn bidirectionalAttentionTinyLoss(
    ctx: *ExecContext,
    q_values: []const f32,
    k_values: []const f32,
    v_values: []const f32,
    kv_head_for_head: []const usize,
    scale_value: f32,
) !f32 {
    var q = try Tensor(.{ .seq, .head, .d }).fromSlice(ctx, .{ 2, 1, 2 }, q_values);
    defer q.deinit();
    var k = try Tensor(.{ .seq, .kv_head, .d }).fromSlice(ctx, .{ 3, 1, 2 }, k_values);
    defer k.deinit();
    var v = try Tensor(.{ .seq, .kv_head, .d }).fromSlice(ctx, .{ 3, 1, 2 }, v_values);
    defer v.deinit();
    var y = try q.groupedAttention(ctx, &k, &v, kv_head_for_head, .out, scale_value, .{ .mask = .bidirectional });
    defer y.deinit();
    var loss = try y.sumAll(ctx);
    defer loss.deinit();
    return loss.item();
}

test "tagged autograd grouped bidirectional attention matches finite differences" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // q_seq (2) < kv_seq (3): query 0 attends key 2 — a position the causal
    // kernels would mask — so the FD check fails if any causal bound leaks
    // into the bidirectional forward or backward.
    const q_values = [_]f32{ 0.2, -0.4, 0.5, 0.1 };
    const k_values = [_]f32{ 0.3, -0.2, -0.1, 0.4, 0.6, -0.5 };
    const v_values = [_]f32{ 0.7, -0.6, 0.2, 0.5, -0.3, 0.8 };
    const kv_head_for_head = [_]usize{0};
    const scale_value: f32 = 0.7;

    var q = try Tensor(.{ .seq, .head, .d }).variableFromSlice(&ctx, .{ 2, 1, 2 }, &q_values);
    defer q.deinit();
    var k = try Tensor(.{ .seq, .kv_head, .d }).variableFromSlice(&ctx, .{ 3, 1, 2 }, &k_values);
    defer k.deinit();
    var v = try Tensor(.{ .seq, .kv_head, .d }).variableFromSlice(&ctx, .{ 3, 1, 2 }, &v_values);
    defer v.deinit();

    var y = try q.groupedAttention(&ctx, &k, &v, kv_head_for_head[0..], .out, scale_value, .{ .mask = .bidirectional });
    defer y.deinit();
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var gq = (try q.grad(&ctx)).?;
    defer gq.deinit();
    var gk = (try k.grad(&ctx)).?;
    defer gk.deinit();
    var gv = (try v.grad(&ctx)).?;
    defer gv.deinit();

    const eps: f32 = 1e-3;
    var q_work = q_values;
    for (q_values, 0..) |_, i| {
        q_work = q_values;
        q_work[i] += eps;
        const plus = try bidirectionalAttentionTinyLoss(&ctx, q_work[0..], k_values[0..], v_values[0..], kv_head_for_head[0..], scale_value);
        q_work[i] -= 2 * eps;
        const minus = try bidirectionalAttentionTinyLoss(&ctx, q_work[0..], k_values[0..], v_values[0..], kv_head_for_head[0..], scale_value);
        const expected = (plus - minus) / (2 * eps);
        try std.testing.expectApproxEqAbs(expected, gq.asRawTensor().dataConst()[i], 2e-2);
    }

    var k_work = k_values;
    for (k_values, 0..) |_, i| {
        k_work = k_values;
        k_work[i] += eps;
        const plus = try bidirectionalAttentionTinyLoss(&ctx, q_values[0..], k_work[0..], v_values[0..], kv_head_for_head[0..], scale_value);
        k_work[i] -= 2 * eps;
        const minus = try bidirectionalAttentionTinyLoss(&ctx, q_values[0..], k_work[0..], v_values[0..], kv_head_for_head[0..], scale_value);
        const expected = (plus - minus) / (2 * eps);
        try std.testing.expectApproxEqAbs(expected, gk.asRawTensor().dataConst()[i], 2e-2);
    }

    var v_work = v_values;
    for (v_values, 0..) |_, i| {
        v_work = v_values;
        v_work[i] += eps;
        const plus = try bidirectionalAttentionTinyLoss(&ctx, q_values[0..], k_values[0..], v_work[0..], kv_head_for_head[0..], scale_value);
        v_work[i] -= 2 * eps;
        const minus = try bidirectionalAttentionTinyLoss(&ctx, q_values[0..], k_values[0..], v_work[0..], kv_head_for_head[0..], scale_value);
        const expected = (plus - minus) / (2 * eps);
        try std.testing.expectApproxEqAbs(expected, gv.asRawTensor().dataConst()[i], 2e-2);
    }
}

test "tagged grouped bidirectional biased attention: constant bias matches plain, grads rejected" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const q_values = [_]f32{ 0.2, -0.4, 0.5, 0.1 };
    const k_values = [_]f32{ 0.3, -0.2, -0.1, 0.4, 0.6, -0.5 };
    const v_values = [_]f32{ 0.7, -0.6, 0.2, 0.5, -0.3, 0.8 };
    const kv_head_for_head = [_]usize{0};
    const scale_value: f32 = 0.7;
    // Constant bias: softmax shift-invariance makes the biased result equal
    // the plain bidirectional path up to summation-order rounding.
    const bias_values = [_]f32{1.0} ** (2 * 3);

    var q = try Tensor(.{ .seq, .head, .d }).fromSlice(&ctx, .{ 2, 1, 2 }, &q_values);
    defer q.deinit();
    var k = try Tensor(.{ .seq, .kv_head, .d }).fromSlice(&ctx, .{ 3, 1, 2 }, &k_values);
    defer k.deinit();
    var v = try Tensor(.{ .seq, .kv_head, .d }).fromSlice(&ctx, .{ 3, 1, 2 }, &v_values);
    defer v.deinit();
    var bias = try Tensor(.{ .sq, .skv }).fromSlice(&ctx, .{ 2, 3 }, &bias_values);
    defer bias.deinit();

    var got = try q.groupedAttention(&ctx, &k, &v, kv_head_for_head[0..], .out, scale_value, .{ .mask = .bidirectional, .bias = &bias });
    defer got.deinit();
    var plain = try q.groupedAttention(&ctx, &k, &v, kv_head_for_head[0..], .out, scale_value, .{ .mask = .bidirectional });
    defer plain.deinit();
    for (plain.asRawTensor().dataConst(), got.asRawTensor().dataConst()) |e, g| {
        try std.testing.expectApproxEqAbs(e, g, 1e-5);
    }

    // Inference-only: no VJP exists for the biased forward, so ANY
    // grad-requiring operand is rejected — q here, and the bias itself.
    var qg = try Tensor(.{ .seq, .head, .d }).variableFromSlice(&ctx, .{ 2, 1, 2 }, &q_values);
    defer qg.deinit();
    try std.testing.expectError(
        error.UnsupportedGradient,
        qg.groupedAttention(&ctx, &k, &v, kv_head_for_head[0..], .out, scale_value, .{ .mask = .bidirectional, .bias = &bias }),
    );
    var bias_grad = try Tensor(.{ .sq, .skv }).variableFromSlice(&ctx, .{ 2, 3 }, &bias_values);
    defer bias_grad.deinit();
    try std.testing.expectError(
        error.UnsupportedGradient,
        q.groupedAttention(&ctx, &k, &v, kv_head_for_head[0..], .out, scale_value, .{ .mask = .bidirectional, .bias = &bias_grad }),
    );
}

fn groupedAttentionTinyLoss(
    ctx: *ExecContext,
    q_values: []const f32,
    k_values: []const f32,
    v_values: []const f32,
    kv_head_for_head: []const usize,
    scale_value: f32,
) !f32 {
    var q = try Tensor(.{ .seq, .head, .d }).fromSlice(ctx, .{ 2, 1, 2 }, q_values);
    defer q.deinit();
    var k = try Tensor(.{ .seq, .kv_head, .d }).fromSlice(ctx, .{ 2, 1, 2 }, k_values);
    defer k.deinit();
    var v = try Tensor(.{ .seq, .kv_head, .d }).fromSlice(ctx, .{ 2, 1, 2 }, v_values);
    defer v.deinit();
    var y = try q.groupedAttention(ctx, &k, &v, kv_head_for_head, .out, scale_value, .{});
    defer y.deinit();
    var loss = try y.sumAll(ctx);
    defer loss.deinit();
    return loss.item();
}

fn groupedAttentionLoss(
    ctx: *ExecContext,
    comptime S: usize,
    comptime H: usize,
    comptime KV: usize,
    comptime D: usize,
    q_values: []const f32,
    k_values: []const f32,
    v_values: []const f32,
    kv_head_for_head: []const usize,
    scale_value: f32,
) !f32 {
    var q = try Tensor(.{ .seq, .head, .d }).fromSlice(ctx, .{ S, H, D }, q_values);
    defer q.deinit();
    var k = try Tensor(.{ .seq, .kv_head, .d }).fromSlice(ctx, .{ S, KV, D }, k_values);
    defer k.deinit();
    var v = try Tensor(.{ .seq, .kv_head, .d }).fromSlice(ctx, .{ S, KV, D }, v_values);
    defer v.deinit();
    var y = try q.groupedAttention(ctx, &k, &v, kv_head_for_head, .out, scale_value, .{});
    defer y.deinit();
    var loss = try y.sumAll(ctx);
    defer loss.deinit();
    return loss.item();
}

fn testGelu(x: f32) f32 {
    return 0.5 * x * (1 + std.math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
}

fn testGeluDeriv(x: f32) f32 {
    const a: f32 = 0.7978845608028654;
    const x2 = x * x;
    const u = a * (x + 0.044715 * x * x2);
    const t = std.math.tanh(u);
    const du = a * (1 + 3 * 0.044715 * x2);
    return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * du;
}

test "tagged tanh/gelu/geglu match scalar reference over the SIMD path" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const N = 40; // > 4*vector_len so the vectorized tanh/gelu path runs, plus a tail
    var vals: [N]f32 = undefined;
    for (&vals, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(i)) - 20.0) * 0.7; // ~ -14 .. 13

    var x = try Tensor(.{.d}).fromSlice(&ctx, .{N}, &vals);
    defer x.deinit();
    var th = try x.tanh(&ctx);
    defer th.deinit();
    var ge = try x.gelu(&ctx);
    defer ge.deinit();
    var gg = try x.geglu(&ctx, &x); // x * gelu(x)
    defer gg.deinit();

    for (0..N) |i| {
        try std.testing.expectApproxEqAbs(std.math.tanh(vals[i]), th.asRawTensor().dataConst()[i], 1e-4);
        try std.testing.expectApproxEqAbs(testGelu(vals[i]), ge.asRawTensor().dataConst()[i], 1e-3);
        try std.testing.expectApproxEqAbs(vals[i] * testGelu(vals[i]), gg.asRawTensor().dataConst()[i], 2e-3);
    }
}

test "tagged autograd differentiates fused geglu (gelu gate)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try Tensor(.{.d}).variableFromSlice(&ctx, .{2}, &.{ 2, -3 });
    defer a.deinit();
    var gate = try Tensor(.{.d}).variableFromSlice(&ctx, .{2}, &.{ 0, 1 });
    defer gate.deinit();

    // geglu(a, gate) = a * gelu(gate) (GELU tanh approximation), matching Gemma.
    var y = try a.geglu(&ctx, &gate);
    defer y.deinit();
    const g0 = testGelu(0);
    const g1 = testGelu(1);
    try expectCloseSlices(&.{ 2 * g0, -3 * g1 }, y.asRawTensor().dataConst(), 1e-6);

    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var ga = (try a.grad(&ctx)).?;
    defer ga.deinit();
    var gg = (try gate.grad(&ctx)).?;
    defer gg.deinit();
    try expectCloseSlices(&.{ g0, g1 }, ga.asRawTensor().dataConst(), 1e-6);
    try expectCloseSlices(&.{ 2 * testGeluDeriv(0), -3 * testGeluDeriv(1) }, gg.asRawTensor().dataConst(), 1e-6);
}

test "tagged autograd softmax backward follows stable row-wise VJP" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .batch, .d }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 2, 4, 8 });
    defer x.deinit();
    var w = try Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ 2, 3 }, &.{ 0.5, -1, 2, 1, 0, -0.5 });
    defer w.deinit();

    var y = try x.softmax(&ctx, .d, .{});
    defer y.deinit();
    var weighted = try y.mul(&ctx, &w);
    defer weighted.deinit();
    var loss = try weighted.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var grad = (try x.grad(&ctx)).?;
    defer grad.deinit();

    var expected: [6]f32 = undefined;
    expectedSoftmaxWeightedGrad(.{ 1, 2, 3 }, .{ 0.5, -1, 2 }, expected[0..3]);
    expectedSoftmaxWeightedGrad(.{ 2, 4, 8 }, .{ 1, 0, -0.5 }, expected[3..6]);
    try expectCloseSlices(&expected, grad.asRawTensor().dataConst(), 1e-5);
}

test "tagged autograd softmaxExt applies scaled additive masks and gradients" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .batch, .d }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 2, 4, 8 });
    defer x.deinit();
    var mask = try Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ 2, 3 }, &.{ 0, -1, 0.5, -0.5, 0, -2 });
    defer mask.deinit();
    var w = try Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ 2, 3 }, &.{ 0.5, -1, 2, 1, 0, -0.5 });
    defer w.deinit();

    var y = try x.softmax(&ctx, .d, .{ .mask = &mask, .scale = 0.5 });
    defer y.deinit();
    var weighted = try y.mul(&ctx, &w);
    defer weighted.deinit();
    var loss = try weighted.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var expected_y: [6]f32 = undefined;
    var expected_grad: [6]f32 = undefined;
    expectedSoftmaxExtWeighted(.{ 1, 2, 3 }, .{ 0, -1, 0.5 }, .{ 0.5, -1, 2 }, 0.5, 1, null, expected_y[0..3], expected_grad[0..3]);
    expectedSoftmaxExtWeighted(.{ 2, 4, 8 }, .{ -0.5, 0, -2 }, .{ 1, 0, -0.5 }, 0.5, 1, null, expected_y[3..6], expected_grad[3..6]);
    try expectCloseSlices(&expected_y, y.asRawTensor().dataConst(), 1e-6);

    var grad = (try x.grad(&ctx)).?;
    defer grad.deinit();
    try expectCloseSlices(&expected_grad, grad.asRawTensor().dataConst(), 1e-6);
}

test "tagged autograd softmaxExt applies causal masks and gradients" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .seq, .src }).variableFromSlice(&ctx, .{ 3, 3 }, &.{
        1, 2, 3,
        2, 4, 8,
        3, 1, 0,
    });
    defer x.deinit();
    var w = try Tensor(.{ .seq, .src }).fromSlice(&ctx, .{ 3, 3 }, &.{
        0.5, -1,   2,
        1,   0,    -0.5,
        -1,  0.25, 2,
    });
    defer w.deinit();

    var y = try x.softmax(&ctx, .src, .{ .causal = .{ .query_tag = .seq }, .scale = 0.5 });
    defer y.deinit();
    var weighted = try y.mul(&ctx, &w);
    defer weighted.deinit();
    var loss = try weighted.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    const neg_inf = -std.math.inf(f32);
    var expected_y: [9]f32 = undefined;
    var expected_grad: [9]f32 = undefined;
    expectedSoftmaxExtWeighted(.{ 1, 2, 3 }, .{ 0, neg_inf, neg_inf }, .{ 0.5, -1, 2 }, 0.5, 1, null, expected_y[0..3], expected_grad[0..3]);
    expectedSoftmaxExtWeighted(.{ 2, 4, 8 }, .{ 0, 0, neg_inf }, .{ 1, 0, -0.5 }, 0.5, 1, null, expected_y[3..6], expected_grad[3..6]);
    expectedSoftmaxExtWeighted(.{ 3, 1, 0 }, .{ 0, 0, 0 }, .{ -1, 0.25, 2 }, 0.5, 1, null, expected_y[6..9], expected_grad[6..9]);
    try expectCloseSlices(&expected_y, y.asRawTensor().dataConst(), 1e-6);

    var grad = (try x.grad(&ctx)).?;
    defer grad.deinit();
    try expectCloseSlices(&expected_grad, grad.asRawTensor().dataConst(), 1e-6);
}

test "tagged autograd softmaxExt applies causal source offsets and gradients" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .seq, .src }).variableFromSlice(&ctx, .{ 2, 4 }, &.{
        1, 2, 3, 4,
        2, 4, 8, 1,
    });
    defer x.deinit();
    var w = try Tensor(.{ .seq, .src }).fromSlice(&ctx, .{ 2, 4 }, &.{
        0.5, -1, 2,    3,
        1,   0,  -0.5, 2,
    });
    defer w.deinit();

    var y = try x.softmax(&ctx, .src, .{ .causal = .{ .query_tag = .seq, .source_offset = 2 }, .scale = 0.5 });
    defer y.deinit();
    var weighted = try y.mul(&ctx, &w);
    defer weighted.deinit();
    var loss = try weighted.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    const neg_inf = -std.math.inf(f32);
    var expected_y: [8]f32 = undefined;
    var expected_grad: [8]f32 = undefined;
    expectedSoftmaxExtWeighted(.{ 1, 2, 3, 4 }, .{ 0, 0, 0, neg_inf }, .{ 0.5, -1, 2, 3 }, 0.5, 1, null, expected_y[0..4], expected_grad[0..4]);
    expectedSoftmaxExtWeighted(.{ 2, 4, 8, 1 }, .{ 0, 0, 0, 0 }, .{ 1, 0, -0.5, 2 }, 0.5, 1, null, expected_y[4..8], expected_grad[4..8]);
    try expectCloseSlices(&expected_y, y.asRawTensor().dataConst(), 1e-6);

    var grad = (try x.grad(&ctx)).?;
    defer grad.deinit();
    try expectCloseSlices(&expected_grad, grad.asRawTensor().dataConst(), 1e-6);
}

test "tagged autograd softmaxExt supports sink denominator mass" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .head, .key }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 0, 0, 0, 0, 0, 0 });
    defer x.deinit();
    var sinks = [_]f32{ 0, @log(@as(f32, 3)) };

    var y = try x.softmax(&ctx, .key, .{ .sinks = sinks[0..], .head_tag = .head });
    defer y.deinit();
    try expectCloseSlices(&.{ 0.25, 0.25, 0.25, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0 }, y.asRawTensor().dataConst(), 1e-6);

    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var grad = (try x.grad(&ctx)).?;
    defer grad.deinit();
    try expectCloseSlices(&.{ 0.0625, 0.0625, 0.0625, 1.0 / 12.0, 1.0 / 12.0, 1.0 / 12.0 }, grad.asRawTensor().dataConst(), 1e-6);
}

test "tagged autograd softmaxExt applies ggml-style ALiBi max_bias to broadcast masks" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .head, .key }).variableFromSlice(&ctx, .{ 3, 4 }, &.{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
    defer x.deinit();
    var mask = try Tensor(.{.key}).fromSlice(&ctx, .{4}, &.{ 0, -1, -2, -3 });
    defer mask.deinit();

    var y = try x.softmax(&ctx, .key, .{ .mask = &mask, .max_bias = 8.0, .head_tag = .head });
    defer y.deinit();

    var expected: [12]f32 = undefined;
    expectedSoftmaxExtProbs(.{ 0, 0, 0, 0 }, .{ 0, -1, -2, -3 }, 1, 1.0 / 16.0, null, expected[0..4]);
    expectedSoftmaxExtProbs(.{ 0, 0, 0, 0 }, .{ 0, -1, -2, -3 }, 1, 1.0 / 256.0, null, expected[4..8]);
    expectedSoftmaxExtProbs(.{ 0, 0, 0, 0 }, .{ 0, -1, -2, -3 }, 1, 1.0 / 4.0, null, expected[8..12]);
    try expectCloseSlices(&expected, y.asRawTensor().dataConst(), 1e-6);
}

test "tagged autograd softmaxExt rejects differentiable masks" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .batch, .d }).variableFromSlice(&ctx, .{ 1, 3 }, &.{ 1, 2, 3 });
    defer x.deinit();
    var mask = try Tensor(.{ .batch, .d }).variableFromSlice(&ctx, .{ 1, 3 }, &.{ 0, -1, 0 });
    defer mask.deinit();

    try std.testing.expectError(error.UnsupportedGradient, x.softmax(&ctx, .d, .{ .mask = &mask }));
}

test "tagged autograd cross entropy fuses stable loss and logits gradient" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var logits = try Tensor(.{ .token, .vocab }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 2, 0, -1 });
    defer logits.deinit();

    var loss = try logits.crossEntropy(&ctx, .vocab, &.{ 2, 0 });
    defer loss.deinit();

    const expected_loss = (expectedCrossEntropy(.{ 1, 2, 3 }, 2) + expectedCrossEntropy(.{ 2, 0, -1 }, 0)) / 2;
    try std.testing.expectApproxEqAbs(expected_loss, loss.asRawTensor().item(), 1e-6);

    try loss.backward(&ctx);
    var grad = (try logits.grad(&ctx)).?;
    defer grad.deinit();

    var expected: [6]f32 = undefined;
    expectedCrossEntropyGrad(.{ 1, 2, 3 }, 2, 0.5, expected[0..3]);
    expectedCrossEntropyGrad(.{ 2, 0, -1 }, 0, 0.5, expected[3..6]);
    try expectCloseSlices(&expected, grad.asRawTensor().dataConst(), 1e-6);
}

fn crossEntropyExtScalarLossForTest(
    ctx: *ExecContext,
    data: []const f32,
    labels: []const usize,
    comptime options: exec_mod.CrossEntropyOptions,
    weights: []const f32,
) !f32 {
    var logits = try ctx.fromSliceRank(2, .{ 4, 7 }, data);
    defer logits.deinit();
    var loss = try ctx.crossEntropyLossExAxisRank(2, &logits, 1, labels, options);
    defer loss.deinit();
    if (comptime options.reduction == .none) {
        var acc: f32 = 0;
        for (loss.dataConst(), weights) |value, weight| acc += value * weight;
        return acc;
    }
    return loss.item();
}

test "tagged autograd crossEntropyExt matches finite differences across options" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const logit_values = [_]f32{
        0.4,  -1.2, 0.8,  1.5,  -0.3, 0.1,  -0.9,
        1.1,  0.2,  -0.7, 0.5,  -1.4, 0.9,  0.3,
        -0.6, 1.3,  0.0,  -0.2, 0.7,  -1.1, 0.6,
        0.9,  -0.5, 1.2,  -0.8, 0.3,  0.4,  -0.1,
    };
    const weights = [_]f32{ 0.5, -1.25, 2, 0.75 };

    inline for (.{
        exec_mod.CrossEntropyOptions{},
        exec_mod.CrossEntropyOptions{ .reduction = .sum },
        exec_mod.CrossEntropyOptions{ .reduction = .none },
        exec_mod.CrossEntropyOptions{ .ignore_index = 9, .label_smoothing = 0.1 },
        exec_mod.CrossEntropyOptions{ .reduction = .sum, .ignore_index = 9, .label_smoothing = 0.1 },
        exec_mod.CrossEntropyOptions{ .reduction = .none, .ignore_index = 9 },
    }) |options| {
        const labels: []const usize = if (comptime options.ignore_index != null) &.{ 2, 9, 0, 6 } else &.{ 2, 5, 0, 6 };

        var logits = try Tensor(.{ .token, .vocab }).variableFromSlice(&ctx, .{ 4, 7 }, &logit_values);
        defer logits.deinit();

        var loss_value: f32 = undefined;
        if (comptime options.reduction == .none) {
            var losses = try logits.crossEntropyExt(&ctx, .vocab, labels, options);
            defer losses.deinit();
            // The class tag is removed like sum/mean over an axis.
            try std.testing.expect(@TypeOf(losses).axis_tags.len == 1);
            try std.testing.expect(@TypeOf(losses).axis_tags[0] == .token);
            var w = try Tensor(.{.token}).fromSlice(&ctx, .{4}, &weights);
            defer w.deinit();
            var weighted = try losses.mul(&ctx, &w);
            defer weighted.deinit();
            var loss = try weighted.sumAll(&ctx);
            defer loss.deinit();
            loss_value = loss.asRawTensor().item();
            try loss.backward(&ctx);
        } else {
            var loss = try logits.crossEntropyExt(&ctx, .vocab, labels, options);
            defer loss.deinit();
            loss_value = loss.asRawTensor().item();
            try loss.backward(&ctx);
        }

        // The facade forward agrees with the exec-level kernel.
        try std.testing.expectApproxEqAbs(
            try crossEntropyExtScalarLossForTest(&ctx, &logit_values, labels, options, &weights),
            loss_value,
            1e-6,
        );

        var grad = (try logits.grad(&ctx)).?;
        defer grad.deinit();
        const gd = grad.asRawTensor().dataConst();

        const h: f32 = 1e-2;
        var work = logit_values;
        for (logit_values, 0..) |_, i| {
            work = logit_values;
            work[i] += h;
            const plus = try crossEntropyExtScalarLossForTest(&ctx, &work, labels, options, &weights);
            work[i] -= 2 * h;
            const minus = try crossEntropyExtScalarLossForTest(&ctx, &work, labels, options, &weights);
            const expected = (plus - minus) / (2 * h);
            try std.testing.expectApproxEqAbs(expected, gd[i], 2e-3);
        }

        // Ignored positions (row 1 when ignore_index is set) get exactly zero.
        if (comptime options.ignore_index != null) {
            for (gd[7..14]) |value| try std.testing.expectEqual(@as(f32, 0), value);
        }
    }
}

test "tagged autograd crossEntropyExt default options match crossEntropy" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var logits = try Tensor(.{ .token, .vocab }).fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 2, 0, -1 });
    defer logits.deinit();

    var legacy = try logits.crossEntropy(&ctx, .vocab, &.{ 2, 0 });
    defer legacy.deinit();
    var ext = try logits.crossEntropyExt(&ctx, .vocab, &.{ 2, 0 }, .{});
    defer ext.deinit();
    try std.testing.expectEqual(legacy.asRawTensor().item(), ext.asRawTensor().item());
}

test "tagged autograd rope supports half split mode and inverse backward" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .seq, .d }).variableFromSlice(&ctx, .{ 2, 4 }, &.{ 1, 2, 3, 4, 1, 2, 3, 4 });
    defer x.deinit();

    var y = try x.rope(&ctx, .seq, .d, .{ .positions = &.{ 0, 1 }, .theta_base = 10000 }, .half);
    defer y.deinit();

    const c0 = @cos(@as(f32, 1));
    const s0 = @sin(@as(f32, 1));
    const c1 = @cos(@as(f32, 0.01));
    const s1 = @sin(@as(f32, 0.01));
    const expected_y = [_]f32{
        1,               2,               3,               4,
        1 * c0 - 3 * s0, 2 * c1 - 4 * s1, 1 * s0 + 3 * c0, 2 * s1 + 4 * c1,
    };
    try expectCloseSlices(&expected_y, y.asRawTensor().dataConst(), 1e-6);

    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var grad = (try x.grad(&ctx)).?;
    defer grad.deinit();
    const expected_grad = [_]f32{
        1,       1,       1,       1,
        c0 + s0, c1 + s1, c0 - s0, c1 - s1,
    };
    try expectCloseSlices(&expected_grad, grad.asRawTensor().dataConst(), 1e-6);
}

test "tagged autograd partial rope passes tail dims and inverts rotated gradient" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .seq, .d }).variableFromSlice(&ctx, .{ 2, 6 }, &.{ 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6 });
    defer x.deinit();
    var table = try ctx.prepareRopeTable(&.{ 0, 1 }, 4, 10000, false);
    defer table.deinit();

    var y = try x.rope(&ctx, .seq, .d, &table, .half);
    defer y.deinit();

    const c0 = @cos(@as(f32, 1));
    const s0 = @sin(@as(f32, 1));
    const c1 = @cos(@as(f32, 0.01));
    const s1 = @sin(@as(f32, 0.01));
    const expected_y = [_]f32{
        1,               2,               3,               4,               5, 6,
        1 * c0 - 3 * s0, 2 * c1 - 4 * s1, 1 * s0 + 3 * c0, 2 * s1 + 4 * c1, 5, 6,
    };
    try expectCloseSlices(&expected_y, y.asRawTensor().dataConst(), 1e-6);

    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var grad = (try x.grad(&ctx)).?;
    defer grad.deinit();
    const expected_grad = [_]f32{
        1,       1,       1,       1,       1, 1,
        c0 + s0, c1 + s1, c0 - s0, c1 - s1, 1, 1,
    };
    try expectCloseSlices(&expected_grad, grad.asRawTensor().dataConst(), 1e-6);
}

test "tagged raw rope interleaved mode rotates adjacent feature pairs" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .seq, .d }).fromSlice(&ctx, .{ 1, 4 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();
    var y = try x.rope(&ctx, .seq, .d, .{ .positions = &.{1}, .theta_base = 10000 }, .interleaved);
    defer y.deinit();

    const c0 = @cos(@as(f32, 1));
    const s0 = @sin(@as(f32, 1));
    const c1 = @cos(@as(f32, 0.01));
    const s1 = @sin(@as(f32, 0.01));
    try expectCloseSlices(&.{
        1 * c0 - 2 * s0,
        1 * s0 + 2 * c0,
        3 * c1 - 4 * s1,
        3 * s1 + 4 * c1,
    }, y.asRawTensor().dataConst(), 1e-6);
}

test "tagged rope matches ggml context shift composition for signed positions" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .seq, .head, .d }).fromSlice(
        &ctx,
        .{ 3, 2, 4 },
        &.{
            0.1,  0.2,  0.3,  0.4,
            0.5,  0.6,  0.7,  0.8,
            -0.1, -0.2, -0.3, -0.4,
            -0.5, -0.6, -0.7, -0.8,
            1.1,  1.2,  1.3,  1.4,
            1.5,  1.6,  1.7,  1.8,
        },
    );
    defer x.deinit();

    var first_half = try x.rope(&ctx, .seq, .d, .{ .positions = &.{ 100, 101, 102 }, .theta_base = 10000 }, .half);
    defer first_half.deinit();
    var shifted_half = try first_half.rope(&ctx, .seq, .d, .{ .positions = &.{ -67, -67, -67 }, .theta_base = 10000 }, .half);
    defer shifted_half.deinit();
    var direct_half = try x.rope(&ctx, .seq, .d, .{ .positions = &.{ 33, 34, 35 }, .theta_base = 10000 }, .half);
    defer direct_half.deinit();
    try expectCloseSlices(direct_half.asRawTensor().dataConst(), shifted_half.asRawTensor().dataConst(), 1e-5);

    var first_interleaved = try x.rope(&ctx, .seq, .d, .{ .positions = &.{ 100, 101, 102 }, .theta_base = 10000 }, .interleaved);
    defer first_interleaved.deinit();
    var shifted_interleaved = try first_interleaved.rope(&ctx, .seq, .d, .{ .positions = &.{ -67, -67, -67 }, .theta_base = 10000 }, .interleaved);
    defer shifted_interleaved.deinit();
    var direct_interleaved = try x.rope(&ctx, .seq, .d, .{ .positions = &.{ 33, 34, 35 }, .theta_base = 10000 }, .interleaved);
    defer direct_interleaved.deinit();
    try expectCloseSlices(direct_interleaved.asRawTensor().dataConst(), shifted_interleaved.asRawTensor().dataConst(), 1e-5);
}

test "tagged ggml-inspired axis coverage for softmax rmsnorm and cross entropy" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var logits = try Tensor(.{ .vocab, .token, .batch }).variableFromSlice(
        &ctx,
        .{ 4, 2, 2 },
        &.{
            1, 2,  3,  4,
            2, 0,  -1, 3,
            0, -2, 1,  2,
            4, 3,  2,  1,
        },
    );
    defer logits.deinit();

    var probs = try logits.softmax(&ctx, .vocab, .{});
    defer probs.deinit();
    try expectSoftmaxAxisSumsClose(probs.asRawTensor().dataConst(), 4, 4, 1e-6);

    var loss = try logits.crossEntropy(&ctx, .vocab, &.{ 3, 0, 2, 1 });
    defer loss.deinit();
    try loss.backward(&ctx);
    var grad = (try logits.grad(&ctx)).?;
    defer grad.deinit();
    try expectCrossEntropyGradAxis0SumsClose(grad.asRawTensor().dataConst(), 4, 4, 1e-6);

    var x = try Tensor(.{ .d, .token, .batch }).variableFromSlice(
        &ctx,
        .{ 4, 2, 1 },
        &.{ 1, 2, 3, 4, -1, -2, -3, -4 },
    );
    defer x.deinit();
    var y = try x.rmsNorm(&ctx, .d, 1e-6);
    defer y.deinit();
    try expectRmsNormAxisMeanSquareClose(y.asRawTensor().dataConst(), 4, 2, 1, 1e-5);
}

test "tagged rmsNormMulRopeHalfPrepared matches materialized non-contiguous input" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var base_values: [2 * 8]f32 = undefined;
    for (&base_values, 0..) |*value, i| value.* = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 5)) / 7.0;
    var base = try Tensor(.{ .seq, .wide }).fromSlice(&ctx, .{ 2, 8 }, &base_values);
    defer base.deinit();

    var view4 = try base.narrow(&ctx, .wide, 2, 4);
    defer view4.deinit();
    var strided = try view4.split(&ctx, .wide, .{ .head, .d }, .{ 1, 4 });
    defer strided.deinit();
    var materialized = try strided.materialize(&ctx);
    defer materialized.deinit();

    var weight = try Tensor(.{.d}).fromSlice(&ctx, .{4}, &.{ 0.5, 1.5, -2.0, 0.75 });
    defer weight.deinit();
    const positions = [_]i32{ 0, 1 };
    var table = try ctx.prepareRopeTable(&positions, 4, 10_000, false);
    defer table.deinit();

    var got = try strided.rmsNormMulRopeHalfPrepared(&ctx, .seq, .d, &weight, 1e-6, &table);
    defer got.deinit();
    var expected = try materialized.rmsNormMulRopeHalfPrepared(&ctx, .seq, .d, &weight, 1e-6, &table);
    defer expected.deinit();

    for (got.asRawTensor().dataConst(), expected.asRawTensor().dataConst()) |actual, wanted| {
        try std.testing.expectApproxEqAbs(wanted, actual, 1e-6);
    }
}

test "tagged autograd rms norm backward matches row-wise formula" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .batch, .d }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 2, 0, 4 });
    defer x.deinit();

    var y = try x.rmsNorm(&ctx, .d, 1e-5);
    defer y.deinit();
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var grad = (try x.grad(&ctx)).?;
    defer grad.deinit();

    var expected: [6]f32 = undefined;
    expectedRmsNormSumGrad(.{ 1, 2, 3 }, 1e-5, expected[0..3]);
    expectedRmsNormSumGrad(.{ 2, 0, 4 }, 1e-5, expected[3..6]);
    try expectCloseSlices(&expected, grad.asRawTensor().dataConst(), 1e-5);
}

/// Scalar probe for the layerNorm finite-difference checks:
/// loss = Σ (layerNorm[Affine](x) ⊙ r). `w_values == null` exercises the
/// plain (non-affine) op.
fn layerNormLossForTest(
    ctx: *ExecContext,
    x_values: []const f32,
    w_values: ?[]const f32,
    b_values: ?[]const f32,
    r_values: []const f32,
    rows: usize,
    cols: usize,
    eps: f32,
) !f32 {
    var x = try Tensor(.{ .token, .d }).fromSlice(ctx, .{ rows, cols }, x_values);
    defer x.deinit();
    var r = try Tensor(.{ .token, .d }).fromSlice(ctx, .{ rows, cols }, r_values);
    defer r.deinit();

    var y = blk: {
        if (w_values) |wv| {
            var w = try Tensor(.{.d}).fromSlice(ctx, .{cols}, wv);
            defer w.deinit();
            var b = try Tensor(.{.d}).fromSlice(ctx, .{cols}, b_values.?);
            defer b.deinit();
            break :blk try x.layerNorm(ctx, .d, eps, .{ .weight = &w, .bias = &b });
        }
        break :blk try x.layerNorm(ctx, .d, eps, .{});
    };
    defer y.deinit();
    var weighted = try y.mul(ctx, &r);
    defer weighted.deinit();
    var loss = try weighted.sumAll(ctx);
    defer loss.deinit();
    return loss.asRawTensor().item();
}

/// Scalar probe for the variance finite-difference checks:
/// loss = Σ variance(x, ddof) ⊙ r.
fn varianceLossForTest(
    ctx: *ExecContext,
    x_values: []const f32,
    r_values: []const f32,
    rows: usize,
    cols: usize,
    ddof: u1,
) !f32 {
    var x = try Tensor(.{ .token, .d }).fromSlice(ctx, .{ rows, cols }, x_values);
    defer x.deinit();
    var v = try x.variance(ctx, .d, ddof);
    defer v.deinit();
    var r = try Tensor(.{.token}).fromSlice(ctx, .{rows}, r_values);
    defer r.deinit();
    var weighted = try v.mul(ctx, &r);
    defer weighted.deinit();
    var loss = try weighted.sumAll(ctx);
    defer loss.deinit();
    return loss.asRawTensor().item();
}

fn standardizeReferenceValue(
    x_values: []const f32,
    cols: usize,
    row: usize,
    col: usize,
    valid_len: ?usize,
    options: exec_mod.StandardizeOptions,
) f64 {
    const valid = valid_len orelse cols;
    if (valid == 0 or col >= valid) return 0;

    const row_values = x_values[row * cols ..][0..cols];
    var mean: f64 = 0;
    for (0..valid) |i| mean += row_values[i];
    mean /= @floatFromInt(valid);

    var variance: f64 = 0;
    if (valid > options.ddof) {
        for (0..valid) |i| {
            const centered = @as(f64, row_values[i]) - mean;
            variance += centered * centered;
        }
        variance /= @floatFromInt(valid - @as(usize, options.ddof));
    }

    const denom = switch (options.eps_mode) {
        .outside_sqrt => @sqrt(variance) + @as(f64, options.eps),
        .inside_sqrt => @sqrt(variance + @as(f64, options.eps)),
    };
    return (@as(f64, row_values[col]) - mean) / denom;
}

/// Scalar probe for the standardize finite-difference checks:
/// loss = Σ standardize(x, options) ⊙ r.
fn standardizeLossForTest(
    ctx: *ExecContext,
    x_values: []const f32,
    r_values: []const f32,
    rows: usize,
    cols: usize,
    valid_len: ?usize,
    options: exec_mod.StandardizeOptions,
) !f32 {
    var x = try Tensor(.{ .token, .d }).fromSlice(ctx, .{ rows, cols }, x_values);
    defer x.deinit();
    var y = if (valid_len) |valid|
        try x.standardizeAxis(ctx, .d, .{ .valid_len = valid, .ddof = options.ddof, .eps = options.eps, .eps_mode = options.eps_mode, .accumulation = options.accumulation })
    else
        try x.standardizeAxis(ctx, .d, options);
    defer y.deinit();
    var r = try Tensor(.{ .token, .d }).fromSlice(ctx, .{ rows, cols }, r_values);
    defer r.deinit();
    var weighted = try y.mul(ctx, &r);
    defer weighted.deinit();
    var loss = try weighted.sumAll(ctx);
    defer loss.deinit();
    return loss.asRawTensor().item();
}

test "tagged autograd layerNorm matches the f64 closed form (PyTorch golden)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Golden spot-checks: the expected values come from evaluating the exact
    // closed form torch.nn.LayerNorm implements — y = (x−μ)/√(σ²+eps)·w + b
    // with the BIASED σ² — in f64 inside the test (the f64 closed form is
    // the PyTorch golden; torch evaluates the same formula).
    const Case = struct {
        rows: usize,
        cols: usize,
        x: []const f32,
        w: ?[]const f32 = null,
        b: ?[]const f32 = null,
        eps: f32,
    };
    const cases = [_]Case{
        .{ .rows = 1, .cols = 4, .x = &.{ 1, 2, 3, 4 }, .eps = 1e-5 },
        .{ .rows = 1, .cols = 3, .x = &.{ 0.5, -1.5, 2.0 }, .w = &.{ 2, 0.5, -1 }, .b = &.{ 0.1, -0.2, 0.3 }, .eps = 1e-6 },
        .{ .rows = 2, .cols = 2, .x = &.{ 1, 1, -3, 5 }, .w = &.{ -0.5, 1.5 }, .b = &.{ 0.25, -1 }, .eps = 1e-5 },
    };

    for (cases) |case| {
        var x = try Tensor(.{ .token, .d }).fromSlice(&ctx, .{ case.rows, case.cols }, case.x);
        defer x.deinit();

        var y = blk: {
            if (case.w) |wv| {
                var w = try Tensor(.{.d}).fromSlice(&ctx, .{case.cols}, wv);
                defer w.deinit();
                var b = try Tensor(.{.d}).fromSlice(&ctx, .{case.cols}, case.b.?);
                defer b.deinit();
                break :blk try x.layerNorm(&ctx, .d, case.eps, .{ .weight = &w, .bias = &b });
            }
            break :blk try x.layerNorm(&ctx, .d, case.eps, .{});
        };
        defer y.deinit();

        const yd = y.asRawTensor().dataConst();
        const n = @as(f64, @floatFromInt(case.cols));
        for (0..case.rows) |row| {
            var sum: f64 = 0;
            for (case.x[row * case.cols ..][0..case.cols]) |value| sum += value;
            const mean = sum / n;
            var sumsq: f64 = 0;
            for (case.x[row * case.cols ..][0..case.cols]) |value| {
                const centered = @as(f64, value) - mean;
                sumsq += centered * centered;
            }
            const inv_sigma = 1 / @sqrt(sumsq / n + @as(f64, case.eps));
            for (0..case.cols) |col| {
                var want = (@as(f64, case.x[row * case.cols + col]) - mean) * inv_sigma;
                if (case.w) |wv| want = want * wv[col] + case.b.?[col];
                try std.testing.expectApproxEqAbs(@as(f32, @floatCast(want)), yd[row * case.cols + col], 1e-5);
            }
        }
    }
}

test "tagged autograd layerNorm and layerNormAffine match finite differences" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const rows = 3;
    const cols = 4;
    const x_values = [_]f32{
        0.4,  -1.2, 0.8,  1.5,
        1.1,  0.2,  -0.7, 0.5,
        -0.6, 1.3,  0.0,  -0.2,
    };
    const w_values = [_]f32{ 0.5, -1.25, 2, 0.75 };
    const b_values = [_]f32{ -0.3, 0.6, 0.1, -1 };
    const r_values = [_]f32{
        0.7,  -0.4, 1.1, 0.3,
        -0.9, 0.5,  0.2, -1.3,
        0.6,  -0.8, 1.4, 0.1,
    };
    const eps: f32 = 1e-5;
    const h: f32 = 1e-2;

    // Plain layerNorm: dx via finite differences.
    {
        var x = try Tensor(.{ .token, .d }).variableFromSlice(&ctx, .{ rows, cols }, &x_values);
        defer x.deinit();
        var r = try Tensor(.{ .token, .d }).fromSlice(&ctx, .{ rows, cols }, &r_values);
        defer r.deinit();
        var y = try x.layerNorm(&ctx, .d, eps, .{});
        defer y.deinit();
        var weighted = try y.mul(&ctx, &r);
        defer weighted.deinit();
        var loss = try weighted.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);

        var grad = (try x.grad(&ctx)).?;
        defer grad.deinit();
        const gd = grad.asRawTensor().dataConst();
        var work = x_values;
        for (x_values, 0..) |_, i| {
            work = x_values;
            work[i] += h;
            const plus = try layerNormLossForTest(&ctx, &work, null, null, &r_values, rows, cols, eps);
            work[i] -= 2 * h;
            const minus = try layerNormLossForTest(&ctx, &work, null, null, &r_values, rows, cols, eps);
            try std.testing.expectApproxEqAbs((plus - minus) / (2 * h), gd[i], 2e-3);
        }
    }

    // Affine layerNorm: dx, dweight, dbias via finite differences.
    {
        var x = try Tensor(.{ .token, .d }).variableFromSlice(&ctx, .{ rows, cols }, &x_values);
        defer x.deinit();
        var w = try Tensor(.{.d}).variableFromSlice(&ctx, .{cols}, &w_values);
        defer w.deinit();
        var b = try Tensor(.{.d}).variableFromSlice(&ctx, .{cols}, &b_values);
        defer b.deinit();
        var r = try Tensor(.{ .token, .d }).fromSlice(&ctx, .{ rows, cols }, &r_values);
        defer r.deinit();
        var y = try x.layerNorm(&ctx, .d, eps, .{ .weight = &w, .bias = &b });
        defer y.deinit();
        var weighted = try y.mul(&ctx, &r);
        defer weighted.deinit();
        var loss = try weighted.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);

        var gx = (try x.grad(&ctx)).?;
        defer gx.deinit();
        const gxd = gx.asRawTensor().dataConst();
        var x_work = x_values;
        for (x_values, 0..) |_, i| {
            x_work = x_values;
            x_work[i] += h;
            const plus = try layerNormLossForTest(&ctx, &x_work, &w_values, &b_values, &r_values, rows, cols, eps);
            x_work[i] -= 2 * h;
            const minus = try layerNormLossForTest(&ctx, &x_work, &w_values, &b_values, &r_values, rows, cols, eps);
            try std.testing.expectApproxEqAbs((plus - minus) / (2 * h), gxd[i], 2e-3);
        }

        var gw = (try w.grad(&ctx)).?;
        defer gw.deinit();
        const gwd = gw.asRawTensor().dataConst();
        var w_work = w_values;
        for (w_values, 0..) |_, i| {
            w_work = w_values;
            w_work[i] += h;
            const plus = try layerNormLossForTest(&ctx, &x_values, &w_work, &b_values, &r_values, rows, cols, eps);
            w_work[i] -= 2 * h;
            const minus = try layerNormLossForTest(&ctx, &x_values, &w_work, &b_values, &r_values, rows, cols, eps);
            try std.testing.expectApproxEqAbs((plus - minus) / (2 * h), gwd[i], 2e-3);
        }

        var gb = (try b.grad(&ctx)).?;
        defer gb.deinit();
        const gbd = gb.asRawTensor().dataConst();
        var b_work = b_values;
        for (b_values, 0..) |_, i| {
            b_work = b_values;
            b_work[i] += h;
            const plus = try layerNormLossForTest(&ctx, &x_values, &w_values, &b_work, &r_values, rows, cols, eps);
            b_work[i] -= 2 * h;
            const minus = try layerNormLossForTest(&ctx, &x_values, &w_values, &b_work, &r_values, rows, cols, eps);
            try std.testing.expectApproxEqAbs((plus - minus) / (2 * h), gbd[i], 2e-3);
        }
    }
}

test "tagged autograd layerNormAffine prunes gradients per operand" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const rows = 2;
    const cols = 3;
    const x_values = [_]f32{ 0.4, -1.2, 0.8, 1.1, 0.2, -0.7 };
    const w_values = [_]f32{ 0.5, -1.25, 2 };
    const b_values = [_]f32{ -0.3, 0.6, 0.1 };

    // Full run: every operand is a variable.
    var gw_full: [cols]f32 = undefined;
    var gb_full: [cols]f32 = undefined;
    {
        var x = try Tensor(.{ .token, .d }).variableFromSlice(&ctx, .{ rows, cols }, &x_values);
        defer x.deinit();
        var w = try Tensor(.{.d}).variableFromSlice(&ctx, .{cols}, &w_values);
        defer w.deinit();
        var b = try Tensor(.{.d}).variableFromSlice(&ctx, .{cols}, &b_values);
        defer b.deinit();
        var y = try x.layerNorm(&ctx, .d, 1e-5, .{ .weight = &w, .bias = &b });
        defer y.deinit();
        var loss = try y.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);

        var gw = (try w.grad(&ctx)).?;
        defer gw.deinit();
        @memcpy(&gw_full, gw.asRawTensor().dataConst());
        var gb = (try b.grad(&ctx)).?;
        defer gb.deinit();
        @memcpy(&gb_full, gb.asRawTensor().dataConst());
    }

    // Weight-only: x and bias are constants; the weight grad is bitwise the
    // same as the full run (same serial param pass on the same inputs).
    {
        var x = try Tensor(.{ .token, .d }).fromSlice(&ctx, .{ rows, cols }, &x_values);
        defer x.deinit();
        var w = try Tensor(.{.d}).variableFromSlice(&ctx, .{cols}, &w_values);
        defer w.deinit();
        var b = try Tensor(.{.d}).fromSlice(&ctx, .{cols}, &b_values);
        defer b.deinit();
        var y = try x.layerNorm(&ctx, .d, 1e-5, .{ .weight = &w, .bias = &b });
        defer y.deinit();
        try std.testing.expect(y.requiresGrad());
        var loss = try y.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);
        var gw = (try w.grad(&ctx)).?;
        defer gw.deinit();
        try std.testing.expectEqualSlices(f32, &gw_full, gw.asRawTensor().dataConst());
    }

    // Bias-only.
    {
        var x = try Tensor(.{ .token, .d }).fromSlice(&ctx, .{ rows, cols }, &x_values);
        defer x.deinit();
        var w = try Tensor(.{.d}).fromSlice(&ctx, .{cols}, &w_values);
        defer w.deinit();
        var b = try Tensor(.{.d}).variableFromSlice(&ctx, .{cols}, &b_values);
        defer b.deinit();
        var y = try x.layerNorm(&ctx, .d, 1e-5, .{ .weight = &w, .bias = &b });
        defer y.deinit();
        var loss = try y.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);
        var gb = (try b.grad(&ctx)).?;
        defer gb.deinit();
        try std.testing.expectEqualSlices(f32, &gb_full, gb.asRawTensor().dataConst());
    }

    // All-constant inputs stay grad-free.
    {
        var x = try Tensor(.{ .token, .d }).fromSlice(&ctx, .{ rows, cols }, &x_values);
        defer x.deinit();
        var w = try Tensor(.{.d}).fromSlice(&ctx, .{cols}, &w_values);
        defer w.deinit();
        var b = try Tensor(.{.d}).fromSlice(&ctx, .{cols}, &b_values);
        defer b.deinit();
        var y = try x.layerNorm(&ctx, .d, 1e-5, .{ .weight = &w, .bias = &b });
        defer y.deinit();
        try std.testing.expect(!y.requiresGrad());
    }
}

test "tagged autograd max and min route gradients to the first extremum" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Row 0 has a duplicated max (3 at indices 1 and 2); row 1 a duplicated
    // max (5 at 0 and 2): the gradient lands only on the FIRST occurrence.
    var x = try Tensor(.{ .token, .d }).variableFromSlice(&ctx, .{ 2, 4 }, &.{
        1, 3,  3, 2,
        5, -1, 5, 4,
    });
    defer x.deinit();

    var m = try x.max(&ctx, .d);
    defer m.deinit();
    try std.testing.expect(@TypeOf(m).axis_tags.len == 1);
    try std.testing.expect(@TypeOf(m).axis_tags[0] == .token);
    try std.testing.expectEqualSlices(f32, &.{ 3, 5 }, m.asRawTensor().dataConst());

    var loss = try m.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    var grad = (try x.grad(&ctx)).?;
    defer grad.deinit();
    try std.testing.expectEqualSlices(f32, &.{
        0, 1, 0, 0,
        1, 0, 0, 0,
    }, grad.asRawTensor().dataConst());

    // min with a duplicated extremum (-2 at indices 1 and 3).
    var x2 = try Tensor(.{ .token, .d }).variableFromSlice(&ctx, .{ 1, 5 }, &.{ 4, -2, 7, -2, 0 });
    defer x2.deinit();
    var mn = try x2.min(&ctx, .d);
    defer mn.deinit();
    try std.testing.expectEqualSlices(f32, &.{-2}, mn.asRawTensor().dataConst());
    var loss2 = try mn.sumAll(&ctx);
    defer loss2.deinit();
    try loss2.backward(&ctx);
    var grad2 = (try x2.grad(&ctx)).?;
    defer grad2.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0, 1, 0, 0, 0 }, grad2.asRawTensor().dataConst());
}

test "tagged autograd variance matches torch semantics and finite differences" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const rows = 2;
    const cols = 4;
    const x_values = [_]f32{
        0.4, -1.2, 0.8, 1.5,
        1.1, 0.2,  3,   0.5,
    };
    const r_values = [_]f32{ 0.5, -1.25 };
    const h: f32 = 1e-2;

    for ([_]u1{ 0, 1 }) |ddof| {
        var x = try Tensor(.{ .token, .d }).variableFromSlice(&ctx, .{ rows, cols }, &x_values);
        defer x.deinit();
        var v = try x.variance(&ctx, .d, ddof);
        defer v.deinit();
        try std.testing.expect(@TypeOf(v).axis_tags.len == 1);

        // Forward vs the f64 closed form (ddof 0 = biased, 1 = torch.var).
        const vd = v.asRawTensor().dataConst();
        const n = @as(f64, @floatFromInt(cols));
        for (0..rows) |row| {
            var sum: f64 = 0;
            for (x_values[row * cols ..][0..cols]) |value| sum += value;
            const mean = sum / n;
            var sumsq: f64 = 0;
            for (x_values[row * cols ..][0..cols]) |value| {
                const centered = @as(f64, value) - mean;
                sumsq += centered * centered;
            }
            const want = sumsq / (n - @as(f64, @floatFromInt(ddof)));
            try std.testing.expectApproxEqAbs(@as(f32, @floatCast(want)), vd[row], 1e-5);
        }

        var r = try Tensor(.{.token}).fromSlice(&ctx, .{rows}, &r_values);
        defer r.deinit();
        var weighted = try v.mul(&ctx, &r);
        defer weighted.deinit();
        var loss = try weighted.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);

        var grad = (try x.grad(&ctx)).?;
        defer grad.deinit();
        const gd = grad.asRawTensor().dataConst();
        var work = x_values;
        for (x_values, 0..) |_, i| {
            work = x_values;
            work[i] += h;
            const plus = try varianceLossForTest(&ctx, &work, &r_values, rows, cols, ddof);
            work[i] -= 2 * h;
            const minus = try varianceLossForTest(&ctx, &work, &r_values, rows, cols, ddof);
            try std.testing.expectApproxEqAbs((plus - minus) / (2 * h), gd[i], 2e-3);
        }
    }
}

test "tagged autograd standardizeAxis supports ddof eps valid-prefix and finite differences" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const rows = 2;
    const cols = 5;
    const x_values = [_]f32{
        0.4, -1.2, 0.8, 1.5, -0.7,
        1.1, 0.2,  3,   0.5, 2.4,
    };
    const r_values = [_]f32{
        0.5, -1.25, 0.75, 0.2, -0.4,
        1.4, -0.6,  0.3,  2.1, -0.9,
    };
    const cases = [_]struct {
        valid_len: ?usize,
        options: exec_mod.StandardizeOptions,
    }{
        .{ .valid_len = null, .options = .{ .ddof = 0, .eps = 1e-4, .eps_mode = .inside_sqrt, .accumulation = .f32 } },
        .{ .valid_len = null, .options = .{ .ddof = 1, .eps = 1e-5, .eps_mode = .outside_sqrt, .accumulation = .f64 } },
        .{ .valid_len = 3, .options = .{ .ddof = 1, .eps = 1e-5, .eps_mode = .outside_sqrt, .accumulation = .f64 } },
    };
    const h: f32 = 1e-2;

    for (cases) |case| {
        var x = try Tensor(.{ .token, .d }).variableFromSlice(&ctx, .{ rows, cols }, &x_values);
        defer x.deinit();
        var y = if (case.valid_len) |valid|
            try x.standardizeAxis(&ctx, .d, .{ .valid_len = valid, .ddof = case.options.ddof, .eps = case.options.eps, .eps_mode = case.options.eps_mode, .accumulation = case.options.accumulation })
        else
            try x.standardizeAxis(&ctx, .d, case.options);
        defer y.deinit();

        const yd = y.asRawTensor().dataConst();
        for (0..rows) |row| {
            for (0..cols) |col| {
                const want = standardizeReferenceValue(&x_values, cols, row, col, case.valid_len, case.options);
                try std.testing.expectApproxEqAbs(@as(f32, @floatCast(want)), yd[row * cols + col], 1e-5);
            }
        }

        if (case.valid_len) |valid| {
            for (0..rows) |row| {
                for (valid..cols) |col| {
                    try std.testing.expectEqual(@as(f32, 0), yd[row * cols + col]);
                }
            }
        }

        var r = try Tensor(.{ .token, .d }).fromSlice(&ctx, .{ rows, cols }, &r_values);
        defer r.deinit();
        var weighted = try y.mul(&ctx, &r);
        defer weighted.deinit();
        var loss = try weighted.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);

        var grad = (try x.grad(&ctx)).?;
        defer grad.deinit();
        const gd = grad.asRawTensor().dataConst();
        var work = x_values;
        for (x_values, 0..) |_, i| {
            work = x_values;
            work[i] += h;
            const plus = try standardizeLossForTest(&ctx, &work, &r_values, rows, cols, case.valid_len, case.options);
            work[i] -= 2 * h;
            const minus = try standardizeLossForTest(&ctx, &work, &r_values, rows, cols, case.valid_len, case.options);
            try std.testing.expectApproxEqAbs((plus - minus) / (2 * h), gd[i], 4e-3);
        }
    }
}

fn testSigmoid(value: f32) f32 {
    if (value >= 0) {
        const z = @exp(-value);
        return 1 / (1 + z);
    }
    const z = @exp(value);
    return z / (1 + z);
}

fn testTanhDerivative(value: f32) f32 {
    const t = std.math.tanh(value);
    return 1 - t * t;
}

fn testGeluDerivative(value: f32) f32 {
    const sqrt_2_over_pi: f32 = 0.7978845608028654;
    const x2 = value * value;
    const u = sqrt_2_over_pi * (value + 0.044715 * value * x2);
    const t = std.math.tanh(u);
    return 0.5 * (1 + t) + 0.5 * value * (1 - t * t) * sqrt_2_over_pi * (1 + 3 * 0.044715 * x2);
}

fn testQuickGeluDerivative(value: f32) f32 {
    const s = testSigmoid(1.702 * value);
    return s + value * 1.702 * s * (1 - s);
}

fn expectCloseSlices(expected: []const f32, actual: []const f32, tolerance: f32) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |e, a| {
        try std.testing.expectApproxEqAbs(e, a, tolerance);
    }
}

fn expectSoftmaxAxisSumsClose(values: []const f32, class_count: usize, inner: usize, tolerance: f32) !void {
    try std.testing.expectEqual(@as(usize, 0), values.len % (class_count * inner));
    const outer = values.len / (class_count * inner);
    for (0..outer) |outer_i| {
        const base = outer_i * class_count * inner;
        for (0..inner) |inner_i| {
            var sum: f32 = 0;
            for (0..class_count) |class_i| {
                sum += values[base + class_i * inner + inner_i];
            }
            try std.testing.expectApproxEqAbs(@as(f32, 1), sum, tolerance);
        }
    }
}

fn expectCrossEntropyGradAxis0SumsClose(values: []const f32, class_count: usize, inner: usize, tolerance: f32) !void {
    try std.testing.expectEqual(@as(usize, 0), values.len % (class_count * inner));
    const outer = values.len / (class_count * inner);
    for (0..outer) |outer_i| {
        const base = outer_i * class_count * inner;
        for (0..inner) |inner_i| {
            var sum: f32 = 0;
            for (0..class_count) |class_i| {
                sum += values[base + class_i * inner + inner_i];
            }
            try std.testing.expectApproxEqAbs(@as(f32, 0), sum, tolerance);
        }
    }
}

fn expectRmsNormAxisMeanSquareClose(values: []const f32, axis_dim: usize, inner: usize, expected: f32, tolerance: f32) !void {
    try std.testing.expectEqual(@as(usize, 0), values.len % (axis_dim * inner));
    const outer = values.len / (axis_dim * inner);
    for (0..outer) |outer_i| {
        const base = outer_i * axis_dim * inner;
        for (0..inner) |inner_i| {
            var sumsq: f32 = 0;
            for (0..axis_dim) |axis_i| {
                const value = values[base + axis_i * inner + inner_i];
                sumsq += value * value;
            }
            try std.testing.expectApproxEqAbs(expected, sumsq / @as(f32, @floatFromInt(axis_dim)), tolerance);
        }
    }
}

fn expectedSoftmaxWeightedGrad(comptime logits: [3]f32, comptime weights: [3]f32, out: []f32) void {
    var max_value = logits[0];
    inline for (1..3) |i| max_value = @max(max_value, logits[i]);

    var probs: [3]f32 = undefined;
    var sum_exp: f32 = 0;
    inline for (0..3) |i| {
        probs[i] = @exp(logits[i] - max_value);
        sum_exp += probs[i];
    }
    inline for (0..3) |i| probs[i] /= sum_exp;

    var dot: f32 = 0;
    inline for (0..3) |i| dot += probs[i] * weights[i];
    inline for (0..3) |i| out[i] = probs[i] * (weights[i] - dot);
}

fn expectedSoftmaxExtProbs(comptime logits: anytype, comptime mask: anytype, scale_value: f32, slope: f32, sink: ?f32, out: []f32) void {
    const len = logits.len;
    std.debug.assert(mask.len == len);
    std.debug.assert(out.len == len);

    var max_value = logits[0] * scale_value + mask[0] * slope;
    inline for (1..len) |i| {
        max_value = @max(max_value, logits[i] * scale_value + mask[i] * slope);
    }
    if (sink) |sink_value| max_value = @max(max_value, sink_value);

    var sum_exp: f32 = 0;
    inline for (0..len) |i| {
        out[i] = @exp(logits[i] * scale_value + mask[i] * slope - max_value);
        sum_exp += out[i];
    }
    if (sink) |sink_value| sum_exp += @exp(sink_value - max_value);

    inline for (0..len) |i| out[i] /= sum_exp;
}

fn expectedSoftmaxExtWeighted(
    comptime logits: anytype,
    comptime mask: anytype,
    comptime weights: anytype,
    scale_value: f32,
    slope: f32,
    sink: ?f32,
    probs_out: []f32,
    grad_out: []f32,
) void {
    const len = logits.len;
    std.debug.assert(weights.len == len);
    std.debug.assert(grad_out.len == len);
    expectedSoftmaxExtProbs(logits, mask, scale_value, slope, sink, probs_out);

    var dot: f32 = 0;
    inline for (0..len) |i| dot += probs_out[i] * weights[i];
    inline for (0..len) |i| grad_out[i] = scale_value * probs_out[i] * (weights[i] - dot);
}

fn expectedCrossEntropy(comptime logits: [3]f32, comptime label: usize) f32 {
    var max_value = logits[0];
    inline for (1..3) |i| max_value = @max(max_value, logits[i]);
    var sum_exp: f32 = 0;
    inline for (0..3) |i| sum_exp += @exp(logits[i] - max_value);
    return @log(sum_exp) + max_value - logits[label];
}

fn expectedCrossEntropyGrad(comptime logits: [3]f32, comptime label: usize, scale_value: f32, out: []f32) void {
    var max_value = logits[0];
    inline for (1..3) |i| max_value = @max(max_value, logits[i]);

    var probs: [3]f32 = undefined;
    var sum_exp: f32 = 0;
    inline for (0..3) |i| {
        probs[i] = @exp(logits[i] - max_value);
        sum_exp += probs[i];
    }

    inline for (0..3) |i| {
        var grad = probs[i] / sum_exp;
        if (i == label) grad -= 1;
        out[i] = grad * scale_value;
    }
}

fn expectedRmsNormSumGrad(comptime row: [3]f32, eps: f32, out: []f32) void {
    var sumsq: f32 = 0;
    var dot: f32 = 0;
    inline for (0..3) |i| {
        sumsq += row[i] * row[i];
        dot += row[i];
    }
    const inv_n = 1.0 / 3.0;
    const inv_rms = 1 / @sqrt(sumsq * inv_n + eps);
    const correction = dot * inv_n * inv_rms * inv_rms * inv_rms;
    inline for (0..3) |i| {
        out[i] = inv_rms - row[i] * correction;
    }
}

test "tagged public tensor argmax and topK reduce leading (non-trailing) axes" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .row, .d }).fromSlice(&ctx, .{ 3, 2 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();

    var arg = try x.argmax(&ctx, .row);
    defer arg.deinit();
    try std.testing.expectEqualSlices(i64, &.{ 2, 2 }, try arg.dataConst());

    var top = try x.topK(&ctx, .row, 2, .k);
    defer top.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 5, 6, 3, 4 }, try top.values.dataConst());
    try std.testing.expectEqualSlices(i64, &.{ 2, 2, 1, 1 }, try top.indices.dataConst());
}

test "tagged public tensor withTags and transpose share the source buffer" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var bias = try Tensor(.{.d}).fromSlice(&ctx, .{3}, &.{ 10, 20, 30 });
    defer bias.deinit();
    var retagged = try bias.withTags(&ctx, .{.feature});
    defer retagged.deinit();
    try std.testing.expect(retagged.asRawTensor().buffer == bias.asRawTensor().buffer);

    var x = try Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();
    var transposed = try x.transpose(&ctx, .{ .d, .batch });
    defer transposed.deinit();
    try std.testing.expect(transposed.asRawTensor().buffer == x.asRawTensor().buffer);
    try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, transposed.asRawTensor().shape.slice());
    try std.testing.expectEqualSlices(usize, &.{ 1, 3 }, transposed.asRawTensor().strides.slice());
}

fn allocationFailureProbe(allocator: std.mem.Allocator) !void {
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ 1, 2, 3 });
    defer x.deinit();
    var w = try Tensor(.{ .out, .d }).fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer w.deinit();

    var y = try w.dot(&ctx, &x, .d);
    defer y.deinit();
    var z = try y.mul(&ctx, &y);
    defer z.deinit();
    var loss = try z.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
}

test "tagged autograd ops release exactly once under induced allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationFailureProbe, .{});
}

fn allocationFailureProbeMulUnscoped(allocator: std.mem.Allocator) !void {
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ 1, 2, 3 });
    defer x.deinit();

    var y = try x.mul(&ctx, &x);
    defer y.deinit();
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
}

test "tagged tracked mul (unscoped) releases exactly once under induced allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationFailureProbeMulUnscoped, .{});
}

fn allocationFailureProbeMulScoped(allocator: std.mem.Allocator) !void {
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ 1, 2, 3 });
    defer x.deinit();

    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const y = try x.mul(&ctx, &x);
        const loss = try y.sumAll(&ctx);
        try loss.backward(&ctx);
    }

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
}

test "tagged tracked mul (exec scope) releases exactly once under induced allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationFailureProbeMulScoped, .{});
}

/// Minimal counting wrapper (mirrors bench/alloc.zig CountingAllocator, which
/// tests cannot import across the bench/src module boundary).
const AllocCounter = struct {
    child: std.mem.Allocator,
    alloc_count: usize = 0,

    fn allocator(self: *AllocCounter) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .remap = remap,
                .free = free,
            },
        };
    }

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *AllocCounter = @ptrCast(@alignCast(ctx));
        const ptr = self.child.rawAlloc(len, alignment, ret_addr) orelse return null;
        self.alloc_count += 1;
        return ptr;
    }

    fn resize(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *AllocCounter = @ptrCast(@alignCast(ctx));
        return self.child.rawResize(buf, alignment, new_len, ret_addr);
    }

    fn remap(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        const self: *AllocCounter = @ptrCast(@alignCast(ctx));
        return self.child.rawRemap(buf, alignment, new_len, ret_addr);
    }

    fn free(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        const self: *AllocCounter = @ptrCast(@alignCast(ctx));
        self.child.rawFree(buf, alignment, ret_addr);
    }
};

test "tagged tracked pointwise op on a warmed context performs exactly one allocation" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var counter = AllocCounter{ .child = gpa.allocator() };

    var ctx: ExecContext = undefined;
    ctx.init(counter.allocator());
    defer ctx.deinit();

    var x = try Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ 1, 2, 3 });
    defer x.deinit();

    // Warm the runtime: the first op's value buffer (and any lazy runtime
    // state) is allocated here and returns to the buffer pool on deinit.
    var warm = try x.mul(&ctx, &x);
    warm.deinit();

    // The counted op reuses the pooled value buffer, so the only heap
    // allocation left is the single co-allocated GradState+record node.
    counter.alloc_count = 0;
    var y = try x.mul(&ctx, &x);
    defer y.deinit();
    try std.testing.expectEqual(@as(usize, 1), counter.alloc_count);
}

fn fillKQuantTestWeight(comptime Block: type, blocks: []Block, seed: usize) void {
    // Arbitrary bytes are structurally valid K-quant blocks; keep scales small
    // via the f16 d/dm fields so dequantized magnitudes stay sane.
    for (blocks, 0..) |*b, bi| {
        const bytes = std.mem.asBytes(b);
        for (bytes, 0..) |*byte, i| byte.* = @truncate(i * 31 + bi * 17 + seed * 13 + 7);
        if (comptime @hasField(Block, "dm")) {
            b.dm[0] = f16TestBits(0.02 + 0.001 * @as(f32, @floatFromInt(bi % 9)));
            b.dm[1] = f16TestBits(0.01);
        } else {
            b.d = f16TestBits(0.02 + 0.001 * @as(f32, @floatFromInt(bi % 9)));
        }
    }
}

fn checkFusedSplitSwiGluKQuant(comptime weight_dtype: DType, comptime rhs_layout: backend_mod.PackedRhsLayout, m: usize) !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const n = 16;
    const k = 256;
    const blocks_per_row = k / 256;

    const GU = Tensor(.{ .dtype = .f32, .tags = .{ .batch, .gate_up } });
    const W = Tensor(.{ .dtype = weight_dtype, .tags = .{ .out, .in } });
    const Block = dtype_mod.Storage(weight_dtype);

    const gate_up_values = try allocator.alloc(f32, m * k * 2);
    defer allocator.free(gate_up_values);
    for (gate_up_values, 0..) |*v, i| {
        const signed: i32 = @as(i32, @intCast((i * 29 + 5) % 251)) - 125;
        v.* = @as(f32, @floatFromInt(signed)) / 17.0;
    }
    var gate_up = try GU.fromSlice(&ctx, .{ m, k * 2 }, gate_up_values);
    defer gate_up.deinit();

    const blocks = try allocator.alloc(Block, n * blocks_per_row);
    defer allocator.free(blocks);
    fillKQuantTestWeight(Block, blocks, m);
    var w = try W.fromBlocks(&ctx, .{ n, k }, blocks);
    defer w.deinit();
    // Explicit layout, not packRhs: the q4_k case must force x8 on MMLA
    // hardware to exercise the fused x8 kernel (no fused MMLA kernel exists).
    var packed_rhs = try w.packRhsLayout(&ctx, rhs_layout);
    defer packed_rhs.deinit();

    var activated = try gate_up.splitGated(&ctx, .swiglu, .gate_up, .in);
    defer activated.deinit();
    var expected = try activated.dotPacked(&ctx, &packed_rhs, .in, .out);
    defer expected.deinit();

    var fused = try gate_up.splitSwiGluDotPacked(&ctx, &packed_rhs, .gate_up, .out);
    defer fused.deinit();

    // The fused op mirrors the NATIVE backend's x4-prefix LHS dispatch, so on
    // the native backend results are bit-identical to the unfused path; the
    // scalar reference backend dispatches the unfused path differently (plain
    // rows), so there comparison is tolerance-based.
    if (comptime backend_mod.active_kind == .native) {
        try std.testing.expectEqualSlices(f32, expected.asRawTensor().dataConst(), fused.asRawTensor().dataConst());
    } else {
        for (expected.asRawTensor().dataConst(), fused.asRawTensor().dataConst()) |e, actual| {
            try expectPackedClose(e, actual);
        }
    }
}

test "public splitSwiGlu packed Q4_Kx8 RHS dot matches unfused path bit-exactly" {
    // m=13: padded-x4 small path; m=3: rows path; m=68: padded-x4 large path.
    for ([_]usize{ 3, 13, 68 }) |m| {
        try checkFusedSplitSwiGluKQuant(.q4_k, .q4_kx8, m);
    }
}

test "public splitSwiGlu packed Q5_Kx8 RHS dot matches unfused path bit-exactly" {
    // m=8: exact-x4 path; m=13: rows path; m=130: x4 prefix + 2-row tail.
    for ([_]usize{ 8, 13, 130 }) |m| {
        try checkFusedSplitSwiGluKQuant(.q5_k, .q5_kx8, m);
    }
}

test "public splitSwiGlu packed Q6_Kx4 RHS dot matches unfused path bit-exactly" {
    for ([_]usize{ 5, 12 }) |m| {
        try checkFusedSplitSwiGluKQuant(.q6_k, .q6_kx4, m);
    }
}

test "public gegluQuant packed Q8_0x4 RHS dot matches unfused path" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    for ([_]usize{ 1, 13, 16 }) |m| {
        const n = 12;
        const k = 64;
        const blocks_per_row = k / dtype_mod.q8_0_block_size;

        const A = Tensor(.{ .dtype = .f32, .tags = .{ .batch, .ffn } });
        const W = Tensor(.{ .dtype = .q8_0, .tags = .{ .out, .in } });

        const values = try allocator.alloc(f32, m * k);
        defer allocator.free(values);
        const up_values = try allocator.alloc(f32, m * k);
        defer allocator.free(up_values);
        for (values, up_values, 0..) |*g, *u, i| {
            const signed: i32 = @as(i32, @intCast((i * 37 + 3) % 241)) - 120;
            g.* = @as(f32, @floatFromInt(signed)) / 11.0;
            u.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 23 + 9) % 233)) - 116)) / 9.0;
        }
        var gate = try A.fromSlice(&ctx, .{ m, k }, values);
        defer gate.deinit();
        var up = try A.fromSlice(&ctx, .{ m, k }, up_values);
        defer up.deinit();

        const blocks = try allocator.alloc(dtype_mod.BlockQ8_0, n * blocks_per_row);
        defer allocator.free(blocks);
        for (blocks, 0..) |*b, bi| {
            b.d = f16TestBits(0.02 + 0.001 * @as(f32, @floatFromInt(bi % 9)));
            for (&b.qs, 0..) |*q, i| q.* = @intCast(@as(i32, @intCast((i * 19 + bi * 7) % 255)) - 127);
        }
        var w = try W.fromBlocks(&ctx, .{ n, k }, blocks);
        defer w.deinit();
        var packed_rhs = try w.packRhs(&ctx);
        defer packed_rhs.deinit();

        var gate_act = try gate.unary(&ctx, .gelu_quant);
        defer gate_act.deinit();
        var gated = try up.mul(&ctx, &gate_act);
        defer gated.deinit();
        var gated_in = try gated.withTags(&ctx, .{ .batch, .in });
        defer gated_in.deinit();
        var expected = try gated_in.dotPacked(&ctx, &packed_rhs, .in, .out);
        defer expected.deinit();

        var fused = try gate.gegluQuantDotPacked(&ctx, &up, &packed_rhs, .ffn, .out);
        defer fused.deinit();

        // Like the swiglu Q8_0 test above: the unfused Q8_0 dot quantizes its
        // LHS with a different m%4 grouping than the fused padded-x4 layout,
        // so comparison is tolerance-based (the K-quant fused ops mirror the
        // unfused dispatch exactly and are tested bit-exact).
        for (expected.asRawTensor().dataConst(), fused.asRawTensor().dataConst()) |e, actual| {
            try expectPackedClose(e, actual);
        }
    }
}

test "tagged autograd dot with quantized RHS propagates gradient to lhs" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const X = Tensor(.{ .batch, .in });
    const W = Tensor(.{ .dtype = .q8_0, .tags = .{ .out, .in } });

    var x_values = [_]f32{1} ** dtype_mod.q8_0_block_size;
    var x = try X.variableFromSlice(&ctx, .{ 1, dtype_mod.q8_0_block_size }, &x_values);
    defer x.deinit();

    var blocks = [_]dtype_mod.BlockQ8_0{
        .{ .d = f16TestBits(1), .qs = [_]i8{1} ** dtype_mod.q8_0_block_size },
        .{ .d = f16TestBits(1), .qs = [_]i8{2} ** dtype_mod.q8_0_block_size },
    };
    var w = try W.fromBlocks(&ctx, .{ 2, dtype_mod.q8_0_block_size }, &blocks);
    defer w.deinit();

    var y = try x.dot(&ctx, &w, .in);
    defer y.deinit();
    try std.testing.expect(y.requiresGrad());

    // Weight the two output columns differently so gx distinguishes the rows.
    var c = try Tensor(.{ .batch, .out }).fromSlice(&ctx, .{ 1, 2 }, &.{ 3, 5 });
    defer c.deinit();
    var weighted = try y.mul(&ctx, &c);
    defer weighted.deinit();
    var loss = try weighted.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    // dL/dx[k] = 3 * W_deq[0][k] + 5 * W_deq[1][k] = 3*1 + 5*2 = 13 everywhere.
    for (gx.asRawTensor().dataConst()) |g| {
        try std.testing.expectApproxEqAbs(@as(f32, 13), g, 1e-3);
    }
}

test "tagged autograd dot with f16 RHS propagates gradient to lhs" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const X = Tensor(.{ .batch, .in });
    const W = Tensor(.{ .dtype = .f16, .tags = .{ .out, .in } });

    var x = try X.variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();
    const w_values = [_]f16{ 0.5, -1, 2, 1.5, 0.25, -0.5 };
    var w = try W.fromSlice(&ctx, .{ 2, 3 }, &w_values);
    defer w.deinit();

    var y = try x.dot(&ctx, &w, .in);
    defer y.deinit();
    try std.testing.expect(y.requiresGrad());

    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    // dL/dx[b][k] = sum over rows of W: {0.5+1.5, -1+0.25, 2-0.5}.
    try expectCloseSlices(&.{ 2, -0.75, 1.5, 2, -0.75, 1.5 }, gx.asRawTensor().dataConst(), 1e-5);
}

test "tagged autograd dot with bf16 RHS propagates gradient to lhs" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const X = Tensor(.{ .batch, .in });
    const W = Tensor(.{ .dtype = .bf16, .tags = .{ .out, .in } });

    var x = try X.variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();
    // All exactly representable in bf16, so widening introduces no error.
    const w_values = [_]u16{
        dtype_mod.f32ToBf16(0.5),
        dtype_mod.f32ToBf16(-1),
        dtype_mod.f32ToBf16(2),
        dtype_mod.f32ToBf16(1.5),
        dtype_mod.f32ToBf16(0.25),
        dtype_mod.f32ToBf16(-0.5),
    };
    var w = try W.fromSlice(&ctx, .{ 2, 3 }, &w_values);
    defer w.deinit();

    var y = try x.dot(&ctx, &w, .in);
    defer y.deinit();
    try std.testing.expect(y.requiresGrad());
    try expectCloseSlices(&.{ 4.5, 0.5, 9, 4.25 }, y.asRawTensor().dataConst(), 1e-5);

    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    // dL/dx[b][k] = sum over rows of W: {0.5+1.5, -1+0.25, 2-0.5}.
    try expectCloseSlices(&.{ 2, -0.75, 1.5, 2, -0.75, 1.5 }, gx.asRawTensor().dataConst(), 1e-5);
}

test "tagged autograd bf16 RHS dot matches f32 reference across GEMV/GEMM shapes" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Odd k exercises the SIMD tail; m covers GEMV (1) and the row-tile GEMM
    // splits (8, 64). Values live on 1/8 and 1/16 binary grids: bf16 widening
    // is exact and every partial product/sum is exactly representable in f32,
    // so forward and backward match the f64 reference bit-for-bit regardless
    // of accumulation order (tolerances below are purely defensive).
    const k: usize = 1027;
    const n: usize = 3;

    const w_data = try allocator.alloc(u16, n * k);
    defer allocator.free(w_data);
    const w_f32 = try allocator.alloc(f32, n * k);
    defer allocator.free(w_f32);
    for (w_data, 0..) |*value, idx| {
        const centered: i32 = @intCast((idx * 3) % 13);
        value.* = dtype_mod.f32ToBf16(@as(f32, @floatFromInt(centered - 6)) * 0.0625);
        w_f32[idx] = dtype_mod.bf16ToF32(value.*);
    }
    const W = Tensor(.{ .dtype = .bf16, .tags = .{ .out, .in } });
    var w = try W.fromSlice(&ctx, .{ n, k }, w_data);
    defer w.deinit();

    for ([_]usize{ 1, 8, 64 }) |m| {
        const x_data = try allocator.alloc(f32, m * k);
        defer allocator.free(x_data);
        for (x_data, 0..) |*value, idx| {
            const centered: i32 = @intCast(idx % 11);
            value.* = @as(f32, @floatFromInt(centered - 5)) * 0.125;
        }

        var x = try Tensor(.{ .batch, .in }).variableFromSlice(&ctx, .{ m, k }, x_data);
        defer x.deinit();

        var y = try x.dot(&ctx, &w, .in);
        defer y.deinit();
        const yd = y.asRawTensor().dataConst();
        for (0..m) |b| {
            for (0..n) |j| {
                var expected: f64 = 0;
                for (0..k) |p| {
                    expected += @as(f64, x_data[b * k + p]) * @as(f64, w_f32[j * k + p]);
                }
                try std.testing.expectApproxEqAbs(@as(f32, @floatCast(expected)), yd[b * n + j], 1e-4);
            }
        }

        var loss = try y.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);

        var gx = (try x.grad(&ctx)).?;
        defer gx.deinit();
        const gd = gx.asRawTensor().dataConst();
        for (0..m) |b| {
            for (0..k) |p| {
                var expected: f64 = 0;
                for (0..n) |j| expected += @as(f64, w_f32[j * k + p]);
                try std.testing.expectApproxEqAbs(@as(f32, @floatCast(expected)), gd[b * k + p], 1e-5);
            }
        }
    }
}

test "tagged autograd bf16 RHS dot accepts a non-contiguous lhs view" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Same rows as the contiguous test ([1,2,3] and [4,5,6] per batch), but
    // stored [in, batch] so the transposed view handed to dot is strided.
    var x = try Tensor(.{ .in, .batch }).variableFromSlice(&ctx, .{ 3, 2 }, &.{ 1, 4, 2, 5, 3, 6 });
    defer x.deinit();
    var xt = try x.transpose(&ctx, .{ .batch, .in });
    defer xt.deinit();

    const W = Tensor(.{ .dtype = .bf16, .tags = .{ .out, .in } });
    var w = try W.fromSlice(&ctx, .{ 2, 3 }, &.{
        dtype_mod.f32ToBf16(0.5),
        dtype_mod.f32ToBf16(-1),
        dtype_mod.f32ToBf16(2),
        dtype_mod.f32ToBf16(1.5),
        dtype_mod.f32ToBf16(0.25),
        dtype_mod.f32ToBf16(-0.5),
    });
    defer w.deinit();

    var y = try xt.dot(&ctx, &w, .in);
    defer y.deinit();
    try expectCloseSlices(&.{ 4.5, 0.5, 9, 4.25 }, y.asRawTensor().dataConst(), 1e-5);

    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    // Column sums of W land transposed in the [in, batch] layout.
    try expectCloseSlices(&.{ 2, 2, -0.75, -0.75, 1.5, 1.5 }, gx.asRawTensor().dataConst(), 1e-5);
}

test "tagged autograd bf16 RHS dot works under exec scope" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .batch, .in }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();
    const W = Tensor(.{ .dtype = .bf16, .tags = .{ .out, .in } });
    var w = try W.fromSlice(&ctx, .{ 2, 3 }, &.{
        dtype_mod.f32ToBf16(0.5),
        dtype_mod.f32ToBf16(-1),
        dtype_mod.f32ToBf16(2),
        dtype_mod.f32ToBf16(1.5),
        dtype_mod.f32ToBf16(0.25),
        dtype_mod.f32ToBf16(-0.5),
    });
    defer w.deinit();

    // Two steps with per-iteration scopes: no keep, no defer, no leaks.
    for (0..2) |_| {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const y = try x.dot(&ctx, &w, .in);
        const loss = try y.sumAll(&ctx);
        try loss.backward(&ctx);

        var gx = (try x.grad(&ctx)).?;
        defer gx.deinit();
        try expectCloseSlices(&.{ 2, -0.75, 1.5, 2, -0.75, 1.5 }, gx.asRawTensor().dataConst(), 1e-5);
        x.zeroGrad();
    }
}

test "tagged autograd bf16 RHS dot fallback path stays correct and differentiable" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .batch, .in }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();
    // RHS stored [contract, free] misses the TransB fast path and exercises
    // the cast-to-f32 + typedDotRaw fallback. Same weights as the fast-path
    // test, transposed.
    const W = Tensor(.{ .dtype = .bf16, .tags = .{ .in, .out } });
    var w = try W.fromSlice(&ctx, .{ 3, 2 }, &.{
        dtype_mod.f32ToBf16(0.5),
        dtype_mod.f32ToBf16(1.5),
        dtype_mod.f32ToBf16(-1),
        dtype_mod.f32ToBf16(0.25),
        dtype_mod.f32ToBf16(2),
        dtype_mod.f32ToBf16(-0.5),
    });
    defer w.deinit();

    var y = try x.dot(&ctx, &w, .in);
    defer y.deinit();
    try std.testing.expect(y.requiresGrad());
    try expectCloseSlices(&.{ 4.5, 0.5, 9, 4.25 }, y.asRawTensor().dataConst(), 1e-5);

    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try expectCloseSlices(&.{ 2, -0.75, 1.5, 2, -0.75, 1.5 }, gx.asRawTensor().dataConst(), 1e-5);
}

test "tagged autograd topK values backward scatters into source" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .batch, .d }).variableFromSlice(&ctx, .{ 2, 4 }, &.{ 1, 5, 2, 3, -1, 2, 4, 0 });
    defer x.deinit();

    var top = try x.topK(&ctx, .d, 2, .k);
    defer top.deinit();
    // values rows: {5, 3} at indices {1, 3}; {4, 2} at indices {2, 1}.
    var c = try Tensor(.{ .batch, .k }).fromSlice(&ctx, .{ 2, 2 }, &.{ 2, 3, 5, 7 });
    defer c.deinit();
    var weighted = try top.values.mul(&ctx, &c);
    defer weighted.deinit();
    var loss = try weighted.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try expectCloseSlices(&.{ 0, 2, 0, 3, 0, 7, 5, 0 }, gx.asRawTensor().dataConst(), 1e-6);

    // Non-last reduction axis exercises the inner-stride scatter path.
    var x2 = try Tensor(.{ .d, .batch }).variableFromSlice(&ctx, .{ 3, 2 }, &.{ 1, 6, 5, 2, 3, 4 });
    defer x2.deinit();
    var top2 = try x2.topK(&ctx, .d, 1, .k);
    defer top2.deinit();
    var loss2 = try top2.values.sumAll(&ctx);
    defer loss2.deinit();
    try loss2.backward(&ctx);

    var gx2 = (try x2.grad(&ctx)).?;
    defer gx2.deinit();
    try expectCloseSlices(&.{ 0, 1, 1, 0, 0, 0 }, gx2.asRawTensor().dataConst(), 1e-6);
}

test "tagged autograd windowed grouped causal attention matches finite differences" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const q_values = [_]f32{ 0.2, -0.4, 0.5, 0.1, -0.3, 0.6, 0.05, -0.15 };
    const k_values = [_]f32{ 0.3, -0.2, -0.1, 0.4, 0.25, -0.35, 0.15, 0.45 };
    const v_values = [_]f32{ 0.7, -0.6, 0.2, 0.5, -0.4, 0.3, 0.1, -0.2 };
    const kv_head_for_head = [_]usize{0};
    const scale_value: f32 = 0.7;
    const window: usize = 2;

    var q = try Tensor(.{ .seq, .head, .d }).variableFromSlice(&ctx, .{ 4, 1, 2 }, &q_values);
    defer q.deinit();
    var k = try Tensor(.{ .seq, .kv_head, .d }).variableFromSlice(&ctx, .{ 4, 1, 2 }, &k_values);
    defer k.deinit();
    var v = try Tensor(.{ .seq, .kv_head, .d }).variableFromSlice(&ctx, .{ 4, 1, 2 }, &v_values);
    defer v.deinit();

    var y = try q.groupedAttention(&ctx, &k, &v, kv_head_for_head[0..], .out, scale_value, .{ .window = window });
    defer y.deinit();
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var gq = (try q.grad(&ctx)).?;
    defer gq.deinit();
    var gk = (try k.grad(&ctx)).?;
    defer gk.deinit();
    var gv = (try v.grad(&ctx)).?;
    defer gv.deinit();

    const eps: f32 = 1e-3;
    var q_work = q_values;
    for (q_values, 0..) |_, i| {
        q_work = q_values;
        q_work[i] += eps;
        const plus = try windowedAttentionTinyLoss(&ctx, q_work[0..], k_values[0..], v_values[0..], kv_head_for_head[0..], scale_value, window);
        q_work[i] -= 2 * eps;
        const minus = try windowedAttentionTinyLoss(&ctx, q_work[0..], k_values[0..], v_values[0..], kv_head_for_head[0..], scale_value, window);
        const expected = (plus - minus) / (2 * eps);
        try std.testing.expectApproxEqAbs(expected, gq.asRawTensor().dataConst()[i], 2e-2);
    }
    var k_work = k_values;
    for (k_values, 0..) |_, i| {
        k_work = k_values;
        k_work[i] += eps;
        const plus = try windowedAttentionTinyLoss(&ctx, q_values[0..], k_work[0..], v_values[0..], kv_head_for_head[0..], scale_value, window);
        k_work[i] -= 2 * eps;
        const minus = try windowedAttentionTinyLoss(&ctx, q_values[0..], k_work[0..], v_values[0..], kv_head_for_head[0..], scale_value, window);
        const expected = (plus - minus) / (2 * eps);
        try std.testing.expectApproxEqAbs(expected, gk.asRawTensor().dataConst()[i], 2e-2);
    }
    var v_work = v_values;
    for (v_values, 0..) |_, i| {
        v_work = v_values;
        v_work[i] += eps;
        const plus = try windowedAttentionTinyLoss(&ctx, q_values[0..], k_values[0..], v_work[0..], kv_head_for_head[0..], scale_value, window);
        v_work[i] -= 2 * eps;
        const minus = try windowedAttentionTinyLoss(&ctx, q_values[0..], k_values[0..], v_work[0..], kv_head_for_head[0..], scale_value, window);
        const expected = (plus - minus) / (2 * eps);
        try std.testing.expectApproxEqAbs(expected, gv.asRawTensor().dataConst()[i], 2e-2);
    }
}

fn windowedAttentionTinyLoss(
    ctx: *ExecContext,
    q_values: []const f32,
    k_values: []const f32,
    v_values: []const f32,
    kv_head_for_head: []const usize,
    scale_value: f32,
    window: usize,
) !f32 {
    var q = try Tensor(.{ .seq, .head, .d }).fromSlice(ctx, .{ 4, 1, 2 }, q_values);
    defer q.deinit();
    var k = try Tensor(.{ .seq, .kv_head, .d }).fromSlice(ctx, .{ 4, 1, 2 }, k_values);
    defer k.deinit();
    var v = try Tensor(.{ .seq, .kv_head, .d }).fromSlice(ctx, .{ 4, 1, 2 }, v_values);
    defer v.deinit();
    var y = try q.groupedAttention(ctx, &k, &v, kv_head_for_head, .out, scale_value, .{ .window = window });
    defer y.deinit();
    var loss = try y.sumAll(ctx);
    defer loss.deinit();
    return loss.item();
}

test "tagged autograd f16 KV attention propagates q gradient" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Exactly representable in f16 so the f32 reference sees identical K/V.
    const q_values = [_]f32{ 0.25, -0.5, 0.125, 0.75 };
    const k_f16 = [_]f16{ 0.5, -0.25, 0.375, 0.625 };
    const v_f16 = [_]f16{ 0.75, -0.125, 0.25, 0.5 };
    const k_f32 = [_]f32{ 0.5, -0.25, 0.375, 0.625 };
    const v_f32 = [_]f32{ 0.75, -0.125, 0.25, 0.5 };
    const kv_head_for_head = [_]usize{0};
    const scale_value: f32 = 0.6;

    var q = try Tensor(.{ .seq, .head, .d }).variableFromSlice(&ctx, .{ 2, 1, 2 }, &q_values);
    defer q.deinit();
    var k16 = try Tensor(.{ .dtype = .f16, .tags = .{ .seq, .kv_head, .d } }).fromSlice(&ctx, .{ 2, 1, 2 }, &k_f16);
    defer k16.deinit();
    var v16 = try Tensor(.{ .dtype = .f16, .tags = .{ .seq, .kv_head, .d } }).fromSlice(&ctx, .{ 2, 1, 2 }, &v_f16);
    defer v16.deinit();

    var y = try q.groupedAttention(&ctx, &k16, &v16, kv_head_for_head[0..], .out, scale_value, .{});
    defer y.deinit();
    try std.testing.expect(y.requiresGrad());
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    var gq = (try q.grad(&ctx)).?;
    defer gq.deinit();

    var q_ref = try Tensor(.{ .seq, .head, .d }).variableFromSlice(&ctx, .{ 2, 1, 2 }, &q_values);
    defer q_ref.deinit();
    var k32 = try Tensor(.{ .seq, .kv_head, .d }).fromSlice(&ctx, .{ 2, 1, 2 }, &k_f32);
    defer k32.deinit();
    var v32 = try Tensor(.{ .seq, .kv_head, .d }).fromSlice(&ctx, .{ 2, 1, 2 }, &v_f32);
    defer v32.deinit();
    var y_ref = try q_ref.groupedAttention(&ctx, &k32, &v32, kv_head_for_head[0..], .out, scale_value, .{});
    defer y_ref.deinit();
    var loss_ref = try y_ref.sumAll(&ctx);
    defer loss_ref.deinit();
    try loss_ref.backward(&ctx);
    var gq_ref = (try q_ref.grad(&ctx)).?;
    defer gq_ref.deinit();

    try expectCloseSlices(y_ref.asRawTensor().dataConst(), y.asRawTensor().dataConst(), 1e-6);
    try expectCloseSlices(gq_ref.asRawTensor().dataConst(), gq.asRawTensor().dataConst(), 1e-6);

    // Windowed f16-KV fallback against the windowed f32 reference.
    var q_w = try Tensor(.{ .seq, .head, .d }).variableFromSlice(&ctx, .{ 2, 1, 2 }, &q_values);
    defer q_w.deinit();
    var y_w = try q_w.groupedAttention(&ctx, &k16, &v16, kv_head_for_head[0..], .out, scale_value, .{ .window = 1 });
    defer y_w.deinit();
    var loss_w = try y_w.sumAll(&ctx);
    defer loss_w.deinit();
    try loss_w.backward(&ctx);
    var gq_w = (try q_w.grad(&ctx)).?;
    defer gq_w.deinit();

    var q_wref = try Tensor(.{ .seq, .head, .d }).variableFromSlice(&ctx, .{ 2, 1, 2 }, &q_values);
    defer q_wref.deinit();
    var y_wref = try q_wref.groupedAttention(&ctx, &k32, &v32, kv_head_for_head[0..], .out, scale_value, .{ .window = 1 });
    defer y_wref.deinit();
    var loss_wref = try y_wref.sumAll(&ctx);
    defer loss_wref.deinit();
    try loss_wref.backward(&ctx);
    var gq_wref = (try q_wref.grad(&ctx)).?;
    defer gq_wref.deinit();

    try expectCloseSlices(y_wref.asRawTensor().dataConst(), y_w.asRawTensor().dataConst(), 1e-6);
    try expectCloseSlices(gq_wref.asRawTensor().dataConst(), gq_w.asRawTensor().dataConst(), 1e-6);
}

test "tagged autograd fused rmsNormMulRope matches unfused composition" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const x_values = [_]f32{ 1, 2, 3, 4, -2, 0.5, 1.5, -1 };
    const w_values = [_]f32{ 0.5, -1.5, 2, 0.75 };
    const positions = [_]i32{ 0, 3 };

    var table = try ctx.prepareRopeTable(&positions, 4, 10000.0, false);
    defer table.deinit();

    var x = try Tensor(.{ .seq, .d }).variableFromSlice(&ctx, .{ 2, 4 }, &x_values);
    defer x.deinit();
    var w = try Tensor(.{.d}).variableFromSlice(&ctx, .{4}, &w_values);
    defer w.deinit();
    var fused = try x.rmsNormMulRopeHalfPrepared(&ctx, .seq, .d, &w, 1e-5, &table);
    defer fused.deinit();
    try std.testing.expect(fused.requiresGrad());
    var fused_loss = try fused.sumAll(&ctx);
    defer fused_loss.deinit();
    try fused_loss.backward(&ctx);
    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    var gw = (try w.grad(&ctx)).?;
    defer gw.deinit();

    var x_ref = try Tensor(.{ .seq, .d }).variableFromSlice(&ctx, .{ 2, 4 }, &x_values);
    defer x_ref.deinit();
    var w_ref = try Tensor(.{.d}).variableFromSlice(&ctx, .{4}, &w_values);
    defer w_ref.deinit();
    var normed = try x_ref.rmsNormMul(&ctx, .d, &w_ref, 1e-5);
    defer normed.deinit();
    var ref_y = try normed.rope(&ctx, .seq, .d, &table, .half);
    defer ref_y.deinit();
    var ref_loss = try ref_y.sumAll(&ctx);
    defer ref_loss.deinit();
    try ref_loss.backward(&ctx);
    var gx_ref = (try x_ref.grad(&ctx)).?;
    defer gx_ref.deinit();
    var gw_ref = (try w_ref.grad(&ctx)).?;
    defer gw_ref.deinit();

    try expectCloseSlices(ref_y.asRawTensor().dataConst(), fused.asRawTensor().dataConst(), 1e-5);
    try expectCloseSlices(gx_ref.asRawTensor().dataConst(), gx.asRawTensor().dataConst(), 1e-5);
    try expectCloseSlices(gw_ref.asRawTensor().dataConst(), gw.asRawTensor().dataConst(), 1e-5);
}

test "tagged autograd prepared rope backward honors freq factors" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const x_values = [_]f32{ 1, 2, 3, 4, -2, 0.5, 1.5, -1 };
    const positions = [_]i32{ 1, 2 };
    const freq_factors = [_]f32{ 0.5, 2.0 };

    var table = try ctx.prepareRopeTableFactors(&positions, 4, 100.0, false, &freq_factors);
    defer table.deinit();

    var x = try Tensor(.{ .seq, .d }).variableFromSlice(&ctx, .{ 2, 4 }, &x_values);
    defer x.deinit();
    var y = try x.rope(&ctx, .seq, .d, &table, .half);
    defer y.deinit();
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();

    const eps: f32 = 1e-3;
    var x_work = x_values;
    for (x_values, 0..) |_, i| {
        x_work = x_values;
        x_work[i] += eps;
        const plus = try preparedRopeFactorsLoss(&ctx, x_work[0..], &table);
        x_work[i] -= 2 * eps;
        const minus = try preparedRopeFactorsLoss(&ctx, x_work[0..], &table);
        const expected = (plus - minus) / (2 * eps);
        try std.testing.expectApproxEqAbs(expected, gx.asRawTensor().dataConst()[i], 5e-3);
    }
}

fn preparedRopeFactorsLoss(ctx: *ExecContext, x_values: []const f32, table: *const exec_mod.RopeTable) !f32 {
    var x = try Tensor(.{ .seq, .d }).fromSlice(ctx, .{ 2, 4 }, x_values);
    defer x.deinit();
    var y = try x.rope(ctx, .seq, .d, table, .half);
    defer y.deinit();
    var loss = try y.sumAll(ctx);
    defer loss.deinit();
    return loss.item();
}

test "exec scope owns differentiable intermediates through backward" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ 1, 2, 3 });
    defer x.deinit();

    // Two steps with per-iteration scopes: no keep, no defer, no leaks.
    for (0..2) |_| {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const y = try x.mul(&ctx, &x);
        const z = try y.mul(&ctx, &x);
        const loss = try z.sumAll(&ctx);
        try loss.backward(&ctx);

        var gx = (try x.grad(&ctx)).?; // gradients stay caller-owned
        defer gx.deinit();
        try std.testing.expectEqualSlices(f32, &.{ 3, 12, 27 }, try gx.dataConst());
        x.zeroGrad();
    }
}

test "exec scope adopts no-grad op results (constants, argmax, topK)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var c = try Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ 2, 4 }, &.{ 1, 5, 2, 3, -1, 2, 4, 0 });
    defer c.deinit();

    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    const doubled = try c.add(&ctx, &c);
    // Index outputs are i64 typed constants: caller-owned even under the
    // scope (the typed-constant ownership rule) — values stay scope-owned.
    var arg = try doubled.argmax(&ctx, .d);
    defer arg.deinit();
    try std.testing.expectEqualSlices(i64, &.{ 1, 2 }, try arg.dataConst());
    var top = try doubled.topK(&ctx, .d, 2, .k);
    defer top.indices.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 10, 6, 8, 4 }, try top.values.dataConst());
    try std.testing.expectEqualSlices(i64, &.{ 1, 3, 2, 1 }, try top.indices.dataConst());
}

test "nested exec scopes release only their suffix" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{.d}).variableFromSlice(&ctx, .{2}, &.{ 2, 3 });
    defer x.deinit();

    const outer = ctx.openExecScope();
    defer ctx.closeExecScope(outer);
    const a = try x.mul(&ctx, &x); // owned by the outer scope

    {
        const inner = ctx.openExecScope();
        defer ctx.closeExecScope(inner);
        const b = try a.mul(&ctx, &x); // owned by the inner scope
        try std.testing.expectEqualSlices(f32, &.{ 8, 27 }, try b.dataConst());
    }

    // `a` survives the inner close and is still usable.
    const c2 = try a.add(&ctx, &a);
    try std.testing.expectEqualSlices(f32, &.{ 8, 18 }, try c2.dataConst());
}

// This test documents WHY exec scopes are a training tool and not a
// replacement for deinit-ASAP in inference code: with deinit-ASAP a chain of
// same-shaped ops recycles ~2 pooled buffers (O(1) working set, warm
// addresses), while a held scope keeps every intermediate live until close
// (O(N) working set, cold addresses). See the note at the end of
// docs/MEMORY-MODEL.md §5.
test "exec scope holds buffers until close; deinit-ASAP recycles through the pool" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    const side: usize = 64;
    const chain_len: usize = 16;
    const data = try allocator.alloc(f32, side * side);
    defer allocator.free(data);
    @memset(data, 0.5);

    // Variant A: deinit-ASAP (the inference idiom) — bounded working set.
    var peak_asap: usize = 0;
    {
        var ctx: ExecContext = undefined;
        ctx.init(allocator);
        defer ctx.deinit();
        var c = try Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ side, side }, data);
        defer c.deinit();
        const base = ctx.rt.buffers.outstandingBuffers();
        var cur = try c.add(&ctx, &c);
        for (1..chain_len) |_| {
            const next = try cur.add(&ctx, &c);
            peak_asap = @max(peak_asap, ctx.rt.buffers.outstandingBuffers() - base);
            cur.deinit(); // previous intermediate returns to the pool immediately
            cur = next;
        }
        cur.deinit();
    }
    try std.testing.expect(peak_asap <= 2);

    // Variant B: one exec scope held to the end — every intermediate lives.
    var peak_scope: usize = 0;
    {
        var ctx: ExecContext = undefined;
        ctx.init(allocator);
        defer ctx.deinit();
        var c = try Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ side, side }, data);
        defer c.deinit();
        const base = ctx.rt.buffers.outstandingBuffers();
        const scope = ctx.openExecScope();
        var cur = try c.add(&ctx, &c);
        for (1..chain_len) |_| {
            cur = try cur.add(&ctx, &c);
            peak_scope = @max(peak_scope, ctx.rt.buffers.outstandingBuffers() - base);
        }
        ctx.closeExecScope(scope);
        try std.testing.expectEqual(base, ctx.rt.buffers.outstandingBuffers());
    }
    try std.testing.expectEqual(chain_len, peak_scope);
}

// The write-once property: engine-style forward code (defer-deinit everywhere,
// ctx.replace for the residual stream) is inference code when no scope is
// open, and training code when one is — deinit on scope-owned results is a
// safe no-op (arena semantics), so neither double-frees nor leaks.
fn engineStyleForward(ctx: *ExecContext, w: *const Tensor(.{ .out, .in }), x0: *const Tensor(.{ .batch, .in })) !Tensor(.{}) {
    var h = try x0.dot(ctx, w, .in);
    defer h.deinit();
    var x = try h.withTags(ctx, .{ .batch, .in });
    defer x.deinit();
    for (0..3) |_| {
        // residual-style carry: old released by replace (no-op when scope-owned)
        x = try ctx.replace(x, blockStep(ctx, w, &x));
    }
    var sq = try x.mul(ctx, &x);
    defer sq.deinit();
    return sq.sumAll(ctx);
}

fn blockStep(ctx: *ExecContext, w: *const Tensor(.{ .out, .in }), x: *const Tensor(.{ .batch, .in })) !Tensor(.{ .batch, .in }) {
    var z = try x.dot(ctx, w, .in);
    defer z.deinit();
    var a = try z.tanh(ctx);
    defer a.deinit();
    var renamed = try a.withTags(ctx, .{ .batch, .in });
    defer renamed.deinit();
    return renamed.add(ctx, x);
}

test "scope-owned deinit is a no-op: the same engine-style forward runs unscoped and scoped" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x0 = try Tensor(.{ .batch, .in }).fromSlice(&ctx, .{ 2, 2 }, &.{ 0.5, -0.25, 0.125, 0.75 });
    defer x0.deinit();

    // Inference mode: no scope, every defer is a real release (deinit-ASAP).
    var inference_loss: f32 = 0;
    {
        var w = try Tensor(.{ .out, .in }).fromSlice(&ctx, .{ 2, 2 }, &.{ 0.4, -0.3, 0.2, 0.6 });
        defer w.deinit();
        var loss = try engineStyleForward(&ctx, &w, &x0);
        defer loss.deinit();
        inference_loss = try loss.item();
    }

    // Training mode: SAME code under a scope — defers no-op on scope-owned
    // results, the graph survives to backward, grads flow.
    {
        var w = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 2, 2 }, &.{ 0.4, -0.3, 0.2, 0.6 });
        defer w.deinit();
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        var loss = try engineStyleForward(&ctx, &w, &x0);
        defer loss.deinit(); // no-op: scope-owned
        try std.testing.expectApproxEqAbs(inference_loss, try loss.item(), 1e-6);
        try loss.backward(&ctx);
        var gw = (try w.grad(&ctx)).?;
        defer gw.deinit();
        for (try gw.dataConst()) |g| try std.testing.expect(g != 0);
        w.zeroGrad();
    }
}

test "tagged autograd dropout regenerates the mask and matches mul-by-mask gradients" {
    const rng = @import("../rng.zig");

    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const len = 64;
    const seed: u64 = 0xd20b;
    var prng = std.Random.DefaultPrng.init(0x5eed);
    const random = prng.random();
    var x_data: [len]f32 = undefined;
    var w_data: [len]f32 = undefined;
    for (&x_data) |*value| value.* = random.floatNorm(f32) + 0.25;
    for (&w_data) |*value| value.* = random.floatNorm(f32);

    var x = try Tensor(.{ .batch, .d }).variableFromSlice(&ctx, .{ 8, 8 }, &x_data);
    defer x.deinit();
    var w = try Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ 8, 8 }, &w_data);
    defer w.deinit();

    for ([_]f32{ 0.25, 0.5 }) |p| {
        const scale = 1.0 / (1.0 - p);
        // Test-side mask from the same counter-based stream: m[i] = scale if
        // the 53-bit uniform of rng.at(seed, i) is < 1-p, else 0 — so
        // dropout(x) must equal x .* m exactly, forward and backward.
        var mask_data: [len]f32 = undefined;
        for (&mask_data, 0..) |*value, i| {
            const uniform = @as(f64, @floatFromInt(rng.at(seed, i) >> 11)) * 0x1.0p-53;
            value.* = if (uniform < 1.0 - @as(f64, p)) scale else 0;
        }
        var mask = try Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ 8, 8 }, &mask_data);
        defer mask.deinit();

        // Forward: kept positions scaled exactly by 1/(1-p), zeros elsewhere.
        var y = try x.dropout(&ctx, p, seed);
        defer y.deinit();
        var kept: usize = 0;
        for (y.asRawTensor().dataConst(), x_data, mask_data) |out_value, in_value, m| {
            if (m != 0) {
                try std.testing.expectEqual(in_value * scale, out_value);
                kept += 1;
            } else {
                try std.testing.expectEqual(@as(f32, 0), out_value);
            }
        }
        try std.testing.expect(kept > 0 and kept < len);

        // Same seed -> bitwise identical output; different seed -> different.
        var y_same = try x.dropout(&ctx, p, seed);
        defer y_same.deinit();
        try std.testing.expectEqualSlices(f32, y.asRawTensor().dataConst(), y_same.asRawTensor().dataConst());
        var y_other = try x.dropout(&ctx, p, seed + 1);
        defer y_other.deinit();
        try std.testing.expect(!std.mem.eql(f32, y.asRawTensor().dataConst(), y_other.asRawTensor().dataConst()));

        // Gradients equal the x .* mask composition bitwise (non-trivial
        // upstream gradient via the constant weights).
        const dropout_grad = grad: {
            var weighted = try y.mul(&ctx, &w);
            defer weighted.deinit();
            var loss = try weighted.sumAll(&ctx);
            defer loss.deinit();
            try loss.backward(&ctx);
            var g = (try x.grad(&ctx)).?;
            defer g.deinit();
            break :grad (try allocator.dupe(f32, try g.dataConst()));
        };
        defer allocator.free(dropout_grad);
        x.zeroGrad();

        const mask_grad = grad: {
            var masked = try x.mul(&ctx, &mask);
            defer masked.deinit();
            var weighted = try masked.mul(&ctx, &w);
            defer weighted.deinit();
            var loss = try weighted.sumAll(&ctx);
            defer loss.deinit();
            try loss.backward(&ctx);
            var g = (try x.grad(&ctx)).?;
            defer g.deinit();
            break :grad (try allocator.dupe(f32, try g.dataConst()));
        };
        defer allocator.free(mask_grad);
        x.zeroGrad();

        try std.testing.expectEqualSlices(f32, mask_grad, dropout_grad);
    }

    // p == 0: identity view (no copy), bitwise-equal data, gradient flows.
    {
        var y = try x.dropout(&ctx, 0, seed);
        defer y.deinit();
        try std.testing.expect(y.asRawTensor().buffer == x.asRawTensor().buffer);
        try std.testing.expectEqualSlices(f32, &x_data, y.asRawTensor().dataConst());
        var loss = try y.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);
        var g = (try x.grad(&ctx)).?;
        defer g.deinit();
        for (try g.dataConst()) |value| try std.testing.expectEqual(@as(f32, 1), value);
        x.zeroGrad();
    }

    try std.testing.expectError(error.InvalidShape, x.dropout(&ctx, 1.0, seed));
    try std.testing.expectError(error.InvalidShape, x.dropout(&ctx, -0.5, seed));
}

test "public tensor borrowed slice aliases caller-owned data as no-grad constant" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var values = [_]f32{ 1, 2, 3, 4 };
    var x = try Tensor(.{ .row, .col }).fromBorrowedSlice(&ctx, .{ 2, 2 }, values[0..]);
    defer x.deinit();

    try std.testing.expect(!x.requiresGrad());
    values[2] = 30;
    try std.testing.expectEqual(@as(f32, 30), (try x.dataConst())[2]);
}

test "public Tensor empty/zeros/ones/full/scalar constructors (f32)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const T = Tensor(.{ .batch, .d });

    var z = try T.zeros(&ctx, .{ 2, 3 });
    defer z.deinit();
    try std.testing.expect(!z.requiresGrad());
    try std.testing.expectEqual(@as(usize, 2), z.dim(.batch));
    try std.testing.expectEqual(@as(usize, 3), z.dim(.d));
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 0, 0, 0, 0 }, z.asRawTensor().dataConst());

    var o = try T.ones(&ctx, .{ 2, 3 });
    defer o.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 1, 1, 1, 1 }, o.asRawTensor().dataConst());

    var f = try T.full(&ctx, .{ 2, 3 }, 3.0);
    defer f.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 3, 3, 3, 3, 3, 3 }, f.asRawTensor().dataConst());

    var e = try T.empty(&ctx, .{ 2, 3 });
    defer e.deinit();
    try std.testing.expectEqual(@as(usize, 6), e.asRawTensor().dataConst().len);

    const S = Tensor(.{.d});
    var s = try S.scalar(&ctx, 5.0);
    defer s.deinit();
    try std.testing.expectEqual(@as(f32, 5), s.asRawTensor().dataConst()[0]);
}

test "public Tensor zeros/ones/empty typed equivalents build the right shape (f16)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const H = Tensor(.{ .dtype = .f16, .tags = .{ .batch, .d } });
    try std.testing.expect(H.dtype == .f16);

    var z = try H.zeros(&ctx, .{ 2, 2 });
    defer z.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, z.value.shape.slice());

    var o = try H.ones(&ctx, .{ 3, 1 });
    defer o.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 1 }, o.value.shape.slice());

    var e = try H.empty(&ctx, .{ 2, 2 });
    defer e.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, e.value.shape.slice());
}

test "public Tensor fromBorrowedConstSlice wraps const data zero-copy, no @constCast" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const T = Tensor(.{ .batch, .d });
    const data = [_]f32{ 1, 2, 3, 4, 5, 6 };

    // No @constCast at the call site — that is the point of the API.
    var t = try T.fromBorrowedConstSlice(&ctx, .{ 2, 3 }, &data);
    defer t.deinit();

    try std.testing.expect(!t.requiresGrad());
    try std.testing.expectEqual(@as(usize, 2), t.dim(.batch));
    try std.testing.expectEqualSlices(f32, &data, t.asRawTensor().dataConst());
    // Zero-copy: the tensor view aliases the source buffer (no copy).
    try std.testing.expectEqual(@intFromPtr(&data), @intFromPtr(t.asRawTensor().dataConst().ptr));
}

test "public Tensor no-grad matmul wrappers match the ctx kernels" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    // matmul2D (NN): [2,3]·[3,2] -> [2,2]
    var a = try Tensor(.{ .m, .k }).fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer a.deinit();
    var b = try Tensor(.{ .k, .n }).fromSlice(&ctx, .{ 3, 2 }, &.{ 7, 8, 9, 10, 11, 12 });
    defer b.deinit();
    var got_nn = try a.matmul(&ctx, b, .plain, .{ .m, .n });
    defer got_nn.deinit();
    var want_nn = try ctx.matmul2D(a.asRawTensor(), b.asRawTensor());
    defer want_nn.deinit();
    try std.testing.expectEqualSlices(f32, want_nn.dataConst(), got_nn.asRawTensor().dataConst());

    // matmulTransB: [2,3]·[2,3]ᵀ -> [2,2]
    var bt = try Tensor(.{ .n, .k }).fromSlice(&ctx, .{ 2, 3 }, &.{ 12, 11, 10, 9, 8, 7 });
    defer bt.deinit();
    var got_tb = try a.matmul(&ctx, bt, .trans_b, .{ .m, .n });
    defer got_tb.deinit();
    var want_tb = try ctx.matmulTransB(a.asRawTensor(), bt.asRawTensor());
    defer want_tb.deinit();
    try std.testing.expectEqualSlices(f32, want_tb.dataConst(), got_tb.asRawTensor().dataConst());

    // bmm: [2,2,3]·[2,3,2] -> [2,2,2]
    var ba = try Tensor(.{ .batch, .m, .k }).fromSlice(&ctx, .{ 2, 2, 3 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
    defer ba.deinit();
    var bb = try Tensor(.{ .batch, .k, .n }).fromSlice(&ctx, .{ 2, 3, 2 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
    defer bb.deinit();
    var got_bmm = try ba.matmul(&ctx, bb, .plain, .{ .batch, .m, .n });
    defer got_bmm.deinit();
    var want_bmm = try ctx.bmm(ba.asRawTensor(), bb.asRawTensor());
    defer want_bmm.deinit();
    try std.testing.expectEqualSlices(f32, want_bmm.dataConst(), got_bmm.asRawTensor().dataConst());

    // bmmTransA: [2,3,2]ᵀ·[2,3,2] -> [2,2,2]
    var bta = try Tensor(.{ .batch, .k, .m }).fromSlice(&ctx, .{ 2, 3, 2 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
    defer bta.deinit();
    var got_ta = try bta.matmul(&ctx, bb, .trans_a, .{ .batch, .m, .n });
    defer got_ta.deinit();
    var want_ta = try ctx.bmmTransA(bta.asRawTensor(), bb.asRawTensor());
    defer want_ta.deinit();
    try std.testing.expectEqualSlices(f32, want_ta.dataConst(), got_ta.asRawTensor().dataConst());

    // bmmTransB: [2,2,3]·[2,2,3]ᵀ -> [2,2,2]
    var btb = try Tensor(.{ .batch, .n, .k }).fromSlice(&ctx, .{ 2, 2, 3 }, &.{ 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 });
    defer btb.deinit();
    var got_tbb = try ba.matmul(&ctx, btb, .trans_b, .{ .batch, .m, .n });
    defer got_tbb.deinit();
    var want_tbb = try ctx.bmmTransB(ba.asRawTensor(), btb.asRawTensor());
    defer want_tbb.deinit();
    try std.testing.expectEqualSlices(f32, want_tbb.dataConst(), got_tbb.asRawTensor().dataConst());
}

test "public Tensor BMM supports multi-axis broadcasted batch dims" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const M = 2;
    const K = 3;
    const N = 2;
    const A0 = 2;
    const B1 = 3;

    var a_data: [A0 * 1 * M * K]f32 = undefined;
    for (&a_data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i + 1));
    var a_ta_data: [A0 * 1 * K * M]f32 = undefined;
    for (&a_ta_data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i + 1));
    var b_data: [1 * B1 * K * N]f32 = undefined;
    for (&b_data, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i % 9) + 1));
    var b_tb_data: [1 * B1 * N * K]f32 = undefined;
    for (&b_tb_data, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i % 7) + 1));

    const T4 = Tensor(4);
    var a = try T4.fromSlice(&ctx, .{ A0, 1, M, K }, &a_data);
    defer a.deinit();
    var a_ta = try T4.fromSlice(&ctx, .{ A0, 1, K, M }, &a_ta_data);
    defer a_ta.deinit();
    var b = try T4.fromSlice(&ctx, .{ 1, B1, K, N }, &b_data);
    defer b.deinit();
    var b_tb = try T4.fromSlice(&ctx, .{ 1, B1, N, K }, &b_tb_data);
    defer b_tb.deinit();

    var want_plain: [A0 * B1 * M * N]f32 = undefined;
    var want_ta: [A0 * B1 * M * N]f32 = undefined;
    var want_tb: [A0 * B1 * M * N]f32 = undefined;
    for (0..A0) |batch_a| {
        for (0..B1) |batch_b| {
            for (0..M) |mi| {
                for (0..N) |ni| {
                    var plain: f32 = 0;
                    var trans_a: f32 = 0;
                    var trans_b: f32 = 0;
                    for (0..K) |ki| {
                        plain += a_data[((batch_a * M + mi) * K) + ki] * b_data[((batch_b * K + ki) * N) + ni];
                        trans_a += a_ta_data[((batch_a * K + ki) * M) + mi] * b_data[((batch_b * K + ki) * N) + ni];
                        trans_b += a_data[((batch_a * M + mi) * K) + ki] * b_tb_data[((batch_b * N + ni) * K) + ki];
                    }
                    const out_i = (((batch_a * B1 + batch_b) * M + mi) * N) + ni;
                    want_plain[out_i] = plain;
                    want_ta[out_i] = trans_a;
                    want_tb[out_i] = trans_b;
                }
            }
        }
    }

    var got_plain = try a.matmul(&ctx, b, .plain, 4);
    defer got_plain.deinit();
    try std.testing.expectEqualSlices(usize, &.{ A0, B1, M, N }, got_plain.shape()[0..]);
    try std.testing.expectEqualSlices(f32, &want_plain, got_plain.asRawTensor().dataConst());

    var got_ta = try a_ta.matmul(&ctx, b, .trans_a, 4);
    defer got_ta.deinit();
    try std.testing.expectEqualSlices(usize, &.{ A0, B1, M, N }, got_ta.shape()[0..]);
    try std.testing.expectEqualSlices(f32, &want_ta, got_ta.asRawTensor().dataConst());

    var got_tb = try a.matmul(&ctx, b_tb, .trans_b, 4);
    defer got_tb.deinit();
    try std.testing.expectEqualSlices(usize, &.{ A0, B1, M, N }, got_tb.shape()[0..]);
    try std.testing.expectEqualSlices(f32, &want_tb, got_tb.asRawTensor().dataConst());

    var bad_a_data: [2 * 2 * M * K]f32 = undefined;
    @memset(&bad_a_data, 1);
    var bad_b_data: [3 * K * N]f32 = undefined;
    @memset(&bad_b_data, 1);
    var bad_a = try T4.fromSlice(&ctx, .{ 2, 2, M, K }, &bad_a_data);
    defer bad_a.deinit();
    var bad_b = try Tensor(3).fromSlice(&ctx, .{ 3, K, N }, &bad_b_data);
    defer bad_b.deinit();
    try std.testing.expectError(error.ShapeMismatch, bad_a.matmul(&ctx, bad_b, .plain, 4));
}

test "public Tensor BMM gradients reduce broadcasted batch axes" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const M = 2;
    const K = 3;
    const N = 2;
    const A0 = 2;
    const B1 = 3;

    var a_data: [A0 * M * K]f32 = undefined;
    for (&a_data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i + 1)) * 0.25;
    var a_ta_data: [A0 * K * M]f32 = undefined;
    for (&a_ta_data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i + 1)) * 0.2;
    var b_data: [B1 * K * N]f32 = undefined;
    for (&b_data, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i % 7) + 1)) * 0.125;
    var b_tb_data: [B1 * N * K]f32 = undefined;
    for (&b_tb_data, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i % 5) + 1)) * 0.1;

    const T4 = Tensor(4);

    var a = try T4.variableFromSlice(&ctx, .{ A0, 1, M, K }, &a_data);
    defer a.deinit();
    var b = try T4.variableFromSlice(&ctx, .{ 1, B1, K, N }, &b_data);
    defer b.deinit();
    var y = try a.matmul(&ctx, b, .plain, 4);
    defer y.deinit();
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var ga = (try a.grad(&ctx)).?;
    defer ga.deinit();
    var gb = (try b.grad(&ctx)).?;
    defer gb.deinit();

    var expected_a: [A0 * M * K]f32 = undefined;
    for (0..A0) |batch_a| {
        for (0..M) |mi| {
            for (0..K) |ki| {
                var total: f32 = 0;
                for (0..B1) |batch_b| {
                    for (0..N) |ni| total += b_data[((batch_b * K + ki) * N) + ni];
                }
                expected_a[(batch_a * M + mi) * K + ki] = total;
            }
        }
    }
    var expected_b: [B1 * K * N]f32 = undefined;
    for (0..B1) |batch_b| {
        for (0..K) |ki| {
            for (0..N) |ni| {
                var total: f32 = 0;
                for (0..A0) |batch_a| {
                    for (0..M) |mi| total += a_data[(batch_a * M + mi) * K + ki];
                }
                expected_b[((batch_b * K + ki) * N) + ni] = total;
            }
        }
    }
    try expectCloseSlices(&expected_a, ga.asRawTensor().dataConst(), 1e-5);
    try expectCloseSlices(&expected_b, gb.asRawTensor().dataConst(), 1e-5);

    var a_ta = try T4.variableFromSlice(&ctx, .{ A0, 1, K, M }, &a_ta_data);
    defer a_ta.deinit();
    var b_ta = try T4.variableFromSlice(&ctx, .{ 1, B1, K, N }, &b_data);
    defer b_ta.deinit();
    var y_ta = try a_ta.matmul(&ctx, b_ta, .trans_a, 4);
    defer y_ta.deinit();
    var loss_ta = try y_ta.sumAll(&ctx);
    defer loss_ta.deinit();
    try loss_ta.backward(&ctx);

    var ga_ta = (try a_ta.grad(&ctx)).?;
    defer ga_ta.deinit();
    var gb_ta = (try b_ta.grad(&ctx)).?;
    defer gb_ta.deinit();

    var expected_a_ta: [A0 * K * M]f32 = undefined;
    for (0..A0) |batch_a| {
        for (0..K) |ki| {
            for (0..M) |mi| {
                var total: f32 = 0;
                for (0..B1) |batch_b| {
                    for (0..N) |ni| total += b_data[((batch_b * K + ki) * N) + ni];
                }
                expected_a_ta[(batch_a * K + ki) * M + mi] = total;
            }
        }
    }
    var expected_b_ta: [B1 * K * N]f32 = undefined;
    for (0..B1) |batch_b| {
        for (0..K) |ki| {
            for (0..N) |ni| {
                var total: f32 = 0;
                for (0..A0) |batch_a| {
                    for (0..M) |mi| total += a_ta_data[(batch_a * K + ki) * M + mi];
                }
                expected_b_ta[((batch_b * K + ki) * N) + ni] = total;
            }
        }
    }
    try expectCloseSlices(&expected_a_ta, ga_ta.asRawTensor().dataConst(), 1e-5);
    try expectCloseSlices(&expected_b_ta, gb_ta.asRawTensor().dataConst(), 1e-5);

    var a_tb = try T4.variableFromSlice(&ctx, .{ A0, 1, M, K }, &a_data);
    defer a_tb.deinit();
    var b_tb = try T4.variableFromSlice(&ctx, .{ 1, B1, N, K }, &b_tb_data);
    defer b_tb.deinit();
    var y_tb = try a_tb.matmul(&ctx, b_tb, .trans_b, 4);
    defer y_tb.deinit();
    var loss_tb = try y_tb.sumAll(&ctx);
    defer loss_tb.deinit();
    try loss_tb.backward(&ctx);

    var ga_tb = (try a_tb.grad(&ctx)).?;
    defer ga_tb.deinit();
    var gb_tb = (try b_tb.grad(&ctx)).?;
    defer gb_tb.deinit();

    var expected_a_tb: [A0 * M * K]f32 = undefined;
    for (0..A0) |batch_a| {
        for (0..M) |mi| {
            for (0..K) |ki| {
                var total: f32 = 0;
                for (0..B1) |batch_b| {
                    for (0..N) |ni| total += b_tb_data[((batch_b * N + ni) * K) + ki];
                }
                expected_a_tb[(batch_a * M + mi) * K + ki] = total;
            }
        }
    }
    var expected_b_tb: [B1 * N * K]f32 = undefined;
    for (0..B1) |batch_b| {
        for (0..N) |ni| {
            for (0..K) |ki| {
                var total: f32 = 0;
                for (0..A0) |batch_a| {
                    for (0..M) |mi| total += a_data[(batch_a * M + mi) * K + ki];
                }
                expected_b_tb[((batch_b * N + ni) * K) + ki] = total;
            }
        }
    }
    try expectCloseSlices(&expected_a_tb, ga_tb.asRawTensor().dataConst(), 1e-5);
    try expectCloseSlices(&expected_b_tb, gb_tb.asRawTensor().dataConst(), 1e-5);
}

test "public Tensor biasAdd / addAxisVectorInPlace / addScaledInPlace" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const M = Tensor(.{ .batch, .n });
    const bias = [_]f32{ 10, 20, 30 };

    // out-of-place biasAdd: [2,3] + bias[3] along .n; source unchanged.
    var x = try M.fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();
    var y = try x.biasAdd(&ctx, &bias, .n);
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 11, 22, 33, 14, 25, 36 }, y.asRawTensor().dataConst());
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6 }, x.asRawTensor().dataConst()); // unchanged

    // in-place addAxisVectorInPlace mutates x to match the out-of-place result.
    try x.addAxisVectorInPlace(&ctx, &bias, .n);
    try std.testing.expectEqualSlices(f32, &.{ 11, 22, 33, 14, 25, 36 }, x.asRawTensor().dataConst());

    // in-place addScaledInPlace (self += 0.5·other) matches ctx.addScaledInPlace.
    const S = Tensor(.{ .r, .c });
    var a = try S.fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer a.deinit();
    var b = try S.fromSlice(&ctx, .{ 2, 2 }, &.{ 10, 20, 30, 40 });
    defer b.deinit();
    var raw = try a.asRawTensor().clone(ctx.allocator);
    defer raw.deinit();
    try a.addScaledInPlace(&ctx, b, 0.5);
    try ctx.addScaledInPlace(&raw, b.asRawTensor(), 0.5);
    try std.testing.expectEqualSlices(f32, &.{ 6, 12, 18, 24 }, a.asRawTensor().dataConst());
    try std.testing.expectEqualSlices(f32, raw.dataConst(), a.asRawTensor().dataConst());
}

test "public Tensor scalar convenience ops (addScalar/subScalar/divScalar/powScalar/log1p) values" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const T = Tensor(.{.d});
    var x = try T.fromSlice(&ctx, .{4}, &.{ 1, 2, 4, 8 });
    defer x.deinit();

    var a = try x.addScalar(&ctx, 10);
    defer a.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 11, 12, 14, 18 }, a.asRawTensor().dataConst());

    var s = try x.subScalar(&ctx, 1);
    defer s.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0, 1, 3, 7 }, s.asRawTensor().dataConst());

    var d = try x.divScalar(&ctx, 2);
    defer d.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0.5, 1, 2, 4 }, d.asRawTensor().dataConst());

    var p = try x.powScalar(&ctx, 2);
    defer p.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 4, 16, 64 }, p.asRawTensor().dataConst());

    var l = try x.log1p(&ctx);
    defer l.deinit();
    for (l.asRawTensor().dataConst(), [_]f32{ 1, 2, 4, 8 }) |got, xv| {
        try std.testing.expectApproxEqAbs(@log(1 + xv), got, 1e-6);
    }
}

test "public Tensor conv2d facade matches ctx.conv2d (with and without bias)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    // input [H=3, W=3, Cin=1], weight [Cout=1, kH=2, kW=2, Cin/g=1] -> [oH=2, oW=2, Cout=1]
    var inp = try Tensor(.{ .h, .w, .cin }).fromSlice(&ctx, .{ 3, 3, 1 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 });
    defer inp.deinit();
    var w = try Tensor(.{ .cout, .kh, .kw, .cinpg }).fromSlice(&ctx, .{ 1, 2, 2, 1 }, &.{ 1, 2, 3, 4 });
    defer w.deinit();

    // no bias
    var got = try inp.conv2d(&ctx, w, null, .{ 1, 1 }, .{ 0, 0 }, 1, .{ .oh, .ow, .cout });
    defer got.deinit();
    var want = try ctx.conv2d(inp.asRawTensor(), w.asRawTensor(), null, .{ 1, 1 }, .{ 0, 0 }, 1);
    defer want.deinit();
    try std.testing.expectEqualSlices(f32, want.dataConst(), got.asRawTensor().dataConst());

    // with a [Cout] bias
    var bias = try Tensor(.{.cout}).fromSlice(&ctx, .{1}, &.{10});
    defer bias.deinit();
    var got_b = try inp.conv2d(&ctx, w, bias, .{ 1, 1 }, .{ 0, 0 }, 1, .{ .oh, .ow, .cout });
    defer got_b.deinit();
    var want_b = try ctx.conv2d(inp.asRawTensor(), w.asRawTensor(), bias.asRawTensor(), .{ 1, 1 }, .{ 0, 0 }, 1);
    defer want_b.deinit();
    try std.testing.expectEqualSlices(f32, want_b.dataConst(), got_b.asRawTensor().dataConst());
}

test "public Tensor where / maskedFill / zeroSlice / zeroRows values" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const T = Tensor(.{.d});
    var x = try T.fromSlice(&ctx, .{4}, &.{ 1, 2, 3, 4 });
    defer x.deinit();
    var y = try T.fromSlice(&ctx, .{4}, &.{ 10, 20, 30, 40 });
    defer y.deinit();

    // where: cond ? x : y
    var cond = try T.fromSlice(&ctx, .{4}, &.{ 1, 0, 1, 0 });
    defer cond.deinit();
    var w = try x.where(&ctx, cond, y);
    defer w.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 20, 3, 40 }, w.asRawTensor().dataConst());

    // maskedFill: mask ? value : x
    var mask = try T.fromSlice(&ctx, .{4}, &.{ 0, 1, 0, 1 });
    defer mask.deinit();
    var mf = try x.maskedFill(&ctx, mask, 99);
    defer mf.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 99, 3, 99 }, mf.asRawTensor().dataConst());

    // zeroSlice / zeroRows vs the ctx kernels
    const M = Tensor(.{ .batch, .n });
    var m = try M.fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer m.deinit();
    var zs = try m.zeroSlice(&ctx, .n, 1, 1);
    defer zs.deinit();
    var zs_want = try ctx.zeroSliceAxisRank(2, m.asRawTensor(), 1, 1, 1);
    defer zs_want.deinit();
    try std.testing.expectEqualSlices(f32, zs_want.dataConst(), zs.asRawTensor().dataConst());

    var zr = try m.zeroRows(&ctx, .batch, &.{1});
    defer zr.deinit();
    var zr_want = try ctx.zeroRowsAxisRank(2, m.asRawTensor(), 0, &.{1});
    defer zr_want.deinit();
    try std.testing.expectEqualSlices(f32, zr_want.dataConst(), zr.asRawTensor().dataConst());
}

test "public Tensor rank/axis ops match the ctx *AxisRank kernels" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    // softmax-with-scale over the last axis of [H,Q,K]=[2,2,2]
    var s = try Tensor(.{ .h, .q, .k }).fromSlice(&ctx, .{ 2, 2, 2 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer s.deinit();
    var sm = try s.softmax(&ctx, .k, .{ .scale = 0.5 });
    defer sm.deinit();
    var sm_want = try ctx.softmaxExtAxisRank(3, s.asRawTensor(), 2, .{ .scale = 0.5 });
    defer sm_want.deinit();
    try std.testing.expectEqualSlices(f32, sm_want.dataConst(), sm.asRawTensor().dataConst());

    // affine LayerNorm over .c of [T,C]=[2,3]
    var ln = try Tensor(.{ .t, .c }).fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer ln.deinit();
    var wln = try Tensor(.{.c}).fromSlice(&ctx, .{3}, &.{ 2, 1, 0.5 });
    defer wln.deinit();
    var bln = try Tensor(.{.c}).fromSlice(&ctx, .{3}, &.{ 0.1, -0.1, 0 });
    defer bln.deinit();
    var lno = try ln.layerNorm(&ctx, .c, 1e-5, .{ .weight = wln, .bias = bln });
    defer lno.deinit();
    var lno_want = try ctx.layerNormAffineAxisRank(2, ln.asRawTensor(), wln.asRawTensor(), bln.asRawTensor(), 1, 1e-5);
    defer lno_want.deinit();
    try std.testing.expectEqualSlices(f32, lno_want.dataConst(), lno.asRawTensor().dataConst());

    // split-glu over .d of [T, 2C]=[2,4]
    var g = try Tensor(.{ .t, .d }).fromSlice(&ctx, .{ 2, 4 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer g.deinit();
    var gg = try g.splitGated(&ctx, .glu, .d, .d);
    defer gg.deinit();
    var gg_want = try ctx.splitGluAxisRank(2, g.asRawTensor(), 1);
    defer gg_want.deinit();
    try std.testing.expectEqualSlices(f32, gg_want.dataConst(), gg.asRawTensor().dataConst());

    // causal depthwise conv1d: input [time=4, channel=2], kernel [channel=2, taps=3]
    var cin = try Tensor(.{ .time, .channel }).fromSlice(&ctx, .{ 4, 2 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer cin.deinit();
    var ker = try Tensor(.{ .channel, .taps }).fromSlice(&ctx, .{ 2, 3 }, &.{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 });
    defer ker.deinit();
    var cc = try cin.causalDepthwiseConv1d(&ctx, .time, .channel, .taps, &ker, null);
    defer cc.deinit();
    var cc_want = try ctx.causalDepthwiseConv1dAxisRank(2, cin.asRawTensor(), ker.asRawTensor(), 0, 1, null);
    defer cc_want.deinit();
    try std.testing.expectEqualSlices(f32, cc_want.dataConst(), cc.asRawTensor().dataConst());

    // silu / relu (existing facade methods) match the ctx kernels (no-grad path)
    var u = try Tensor(.{.d}).fromSlice(&ctx, .{4}, &.{ -2, -1, 1, 2 });
    defer u.deinit();
    var re = try u.relu(&ctx);
    defer re.deinit();
    var re_want = try ctx.relu(u.asRawTensor());
    defer re_want.deinit();
    try std.testing.expectEqualSlices(f32, re_want.dataConst(), re.asRawTensor().dataConst());
    var si = try u.silu(&ctx);
    defer si.deinit();
    var si_want = try ctx.silu(u.asRawTensor());
    defer si_want.deinit();
    for (si_want.dataConst(), si.asRawTensor().dataConst()) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, 1e-6);
    }
}

test "public Tensor rank/axis ops preserve autograd when a VJP exists" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var s = try Tensor(.{ .batch, .d }).variableFromSlice(&ctx, .{ 1, 3 }, &.{ 1, 2, 3 });
    defer s.deinit();
    var weights = try Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ 1, 3 }, &.{ 0.5, -1, 2 });
    defer weights.deinit();
    var sm = try s.softmax(&ctx, .d, .{ .scale = 0.5 });
    defer sm.deinit();
    var weighted = try sm.mul(&ctx, &weights);
    defer weighted.deinit();
    var softmax_loss = try weighted.sumAll(&ctx);
    defer softmax_loss.deinit();
    try softmax_loss.backward(&ctx);

    var expected_probs: [3]f32 = undefined;
    var expected_grad: [3]f32 = undefined;
    expectedSoftmaxExtWeighted(.{ 1, 2, 3 }, .{ 0, 0, 0 }, .{ 0.5, -1, 2 }, 0.5, 1, null, expected_probs[0..], expected_grad[0..]);
    var gs = (try s.grad(&ctx)).?;
    defer gs.deinit();
    try expectCloseSlices(&expected_grad, gs.asRawTensor().dataConst(), 1e-6);

    var ln = try Tensor(.{ .t, .c }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer ln.deinit();
    var wln = try Tensor(.{.c}).variableFromSlice(&ctx, .{3}, &.{ 2, 1, 0.5 });
    defer wln.deinit();
    var bln = try Tensor(.{.c}).variableFromSlice(&ctx, .{3}, &.{ 0.1, -0.1, 0 });
    defer bln.deinit();
    var lno = try ln.layerNorm(&ctx, .c, 1e-5, .{ .weight = &wln, .bias = &bln });
    defer lno.deinit();
    var ln_loss = try lno.sumAll(&ctx);
    defer ln_loss.deinit();
    try ln_loss.backward(&ctx);

    var gln = (try ln.grad(&ctx)).?;
    defer gln.deinit();
    var gwln = (try wln.grad(&ctx)).?;
    defer gwln.deinit();
    var gbln = (try bln.grad(&ctx)).?;
    defer gbln.deinit();
    try std.testing.expectEqual(@as(usize, 6), gln.asRawTensor().dataConst().len);
    try std.testing.expectEqual(@as(usize, 3), gwln.asRawTensor().dataConst().len);
    try std.testing.expectEqual(@as(usize, 3), gbln.asRawTensor().dataConst().len);

    var glu_input = try Tensor(.{ .t, .d }).variableFromSlice(&ctx, .{ 1, 4 }, &.{ 1, 2, 3, 4 });
    defer glu_input.deinit();
    var glu = try glu_input.splitGated(&ctx, .glu, .d, .d);
    defer glu.deinit();
    var glu_loss = try glu.sumAll(&ctx);
    defer glu_loss.deinit();
    try glu_loss.backward(&ctx);

    var gglu = (try glu_input.grad(&ctx)).?;
    defer gglu.deinit();
    try std.testing.expectEqual(@as(usize, 4), gglu.asRawTensor().dataConst().len);

    var conv_input = try Tensor(.{ .time, .channel }).variableFromSlice(&ctx, .{ 4, 2 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer conv_input.deinit();
    var conv_kernel = try Tensor(.{ .channel, .taps }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 });
    defer conv_kernel.deinit();
    var conv = try conv_input.causalDepthwiseConv1d(&ctx, .time, .channel, .taps, &conv_kernel, null);
    defer conv.deinit();
    var conv_loss = try conv.sumAll(&ctx);
    defer conv_loss.deinit();
    try conv_loss.backward(&ctx);

    var gconv_input = (try conv_input.grad(&ctx)).?;
    defer gconv_input.deinit();
    var gconv_kernel = (try conv_kernel.grad(&ctx)).?;
    defer gconv_kernel.deinit();
    try std.testing.expectEqual(@as(usize, 8), gconv_input.asRawTensor().dataConst().len);
    try std.testing.expectEqual(@as(usize, 6), gconv_kernel.asRawTensor().dataConst().len);

    var a = try Tensor(.{ .m, .k }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer a.deinit();
    var b = try Tensor(.{ .k, .n }).variableFromSlice(&ctx, .{ 3, 2 }, &.{ 7, 8, 9, 10, 11, 12 });
    defer b.deinit();
    var mm = try a.matmul(&ctx, &b, .plain, .{ .m, .n });
    defer mm.deinit();
    var mm_loss = try mm.sumAll(&ctx);
    defer mm_loss.deinit();
    try mm_loss.backward(&ctx);
    var ga = (try a.grad(&ctx)).?;
    defer ga.deinit();
    var gb = (try b.grad(&ctx)).?;
    defer gb.deinit();
    try expectCloseSlices(&.{ 15, 19, 23, 15, 19, 23 }, ga.asRawTensor().dataConst(), 1e-6);
    try expectCloseSlices(&.{ 5, 5, 7, 7, 9, 9 }, gb.asRawTensor().dataConst(), 1e-6);

    var at = try Tensor(.{ .m, .k }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer at.deinit();
    var bt = try Tensor(.{ .n, .k }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 7, 9, 11, 8, 10, 12 });
    defer bt.deinit();
    var mmt = try at.matmul(&ctx, &bt, .trans_b, .{ .m, .n });
    defer mmt.deinit();
    var mmt_loss = try mmt.sumAll(&ctx);
    defer mmt_loss.deinit();
    try mmt_loss.backward(&ctx);
    var gat = (try at.grad(&ctx)).?;
    defer gat.deinit();
    var gbt = (try bt.grad(&ctx)).?;
    defer gbt.deinit();
    try expectCloseSlices(&.{ 15, 19, 23, 15, 19, 23 }, gat.asRawTensor().dataConst(), 1e-6);
    try expectCloseSlices(&.{ 5, 7, 9, 5, 7, 9 }, gbt.asRawTensor().dataConst(), 1e-6);

    var bias_x = try Tensor(.{ .batch, .n }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer bias_x.deinit();
    var biased = try bias_x.biasAdd(&ctx, &.{ 10, 20, 30 }, .n);
    defer biased.deinit();
    var bias_loss = try biased.sumAll(&ctx);
    defer bias_loss.deinit();
    try bias_loss.backward(&ctx);
    var gbias_x = (try bias_x.grad(&ctx)).?;
    defer gbias_x.deinit();
    try expectCloseSlices(&.{ 1, 1, 1, 1, 1, 1 }, gbias_x.asRawTensor().dataConst(), 1e-6);

    var zs_x = try Tensor(.{ .batch, .n }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer zs_x.deinit();
    var zs = try zs_x.zeroSlice(&ctx, .n, 1, 1);
    defer zs.deinit();
    var zs_loss = try zs.sumAll(&ctx);
    defer zs_loss.deinit();
    try zs_loss.backward(&ctx);
    var gzs = (try zs_x.grad(&ctx)).?;
    defer gzs.deinit();
    try expectCloseSlices(&.{ 1, 0, 1, 1, 0, 1 }, gzs.asRawTensor().dataConst(), 1e-6);

    var zr_x = try Tensor(.{ .batch, .n }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer zr_x.deinit();
    var zr = try zr_x.zeroRows(&ctx, .batch, &.{0});
    defer zr.deinit();
    var zr_loss = try zr.sumAll(&ctx);
    defer zr_loss.deinit();
    try zr_loss.backward(&ctx);
    var gzr = (try zr_x.grad(&ctx)).?;
    defer gzr.deinit();
    try expectCloseSlices(&.{ 0, 0, 0, 1, 1, 1 }, gzr.asRawTensor().dataConst(), 1e-6);
}

test "public Tensor explicit no-grad helpers reject trainable inputs" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var bias_x = try Tensor(.{ .batch, .n }).variableFromSlice(&ctx, .{ 1, 3 }, &.{ 1, 2, 3 });
    defer bias_x.deinit();
    try std.testing.expectError(error.UnsupportedGradient, bias_x.addAxisVectorInPlace(&ctx, &.{ 1, 2, 3 }, .n));
    try std.testing.expectError(error.UnsupportedGradient, bias_x.addAxisVectorUnaryInPlace(&ctx, .relu, &.{ 1, 2, 3 }, .n));

    var add_dst = try Tensor(.{.d}).fromSlice(&ctx, .{3}, &.{ 1, 2, 3 });
    defer add_dst.deinit();
    var add_src = try Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ 4, 5, 6 });
    defer add_src.deinit();
    try std.testing.expectError(error.UnsupportedGradient, add_dst.addScaledInPlace(&ctx, &add_src, 0.5));

    // conv2d is now differentiable (Conv2dBackward) — its VJP is covered by the
    // gradcheck in ag/gradcheck_tests.zig, so it is no longer a no-grad-only op.
}

// --- omnivoice op set: conv1d / convTranspose1d / snake / groupNorm facades
// (forward parity + FD gradchecks) + elu / geluErf unary methods ---

/// Deterministic smooth fill for the FD gradcheck fixtures (values in
/// roughly [-0.75, 0.75], no two operands alike thanks to `phase`).
fn fdFillPattern(values: []f32, phase: f32) void {
    for (values, 0..) |*v, i| {
        const x = @as(f32, @floatFromInt(i)) + phase;
        v.* = @sin(x * 0.7) * 0.5 + @cos(x * 0.3) * 0.25;
    }
}

/// f64-accumulated weighted sum — the FD side of `loss = sum(y * coef)`.
fn fdWeightedSum(y: []const f32, coef: []const f32) f32 {
    var acc: f64 = 0;
    for (y, coef) |value, c| acc += @as(f64, value) * @as(f64, c);
    return @floatCast(acc);
}

/// Central-difference check of `grad` against `lossAt` over every element of
/// `values` (mutated in place and restored).
fn fdCheckGrad(
    values: []f32,
    grad: []const f32,
    eps: f32,
    tol: f32,
    context: anytype,
    comptime lossAt: fn (@TypeOf(context)) anyerror!f32,
) !void {
    try std.testing.expectEqual(values.len, grad.len);
    for (0..values.len) |i| {
        const original = values[i];
        values[i] = original + eps;
        const plus = try lossAt(context);
        values[i] = original - eps;
        const minus = try lossAt(context);
        values[i] = original;
        const expected = (plus - minus) / (2 * eps);
        try std.testing.expectApproxEqAbs(expected, grad[i], tol);
    }
}

test "public Tensor conv1d facade matches ctx.conv1dAxisRank" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var x = try Tensor(.{ .time, .cin }).fromSlice(&ctx, .{ 4, 1 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();
    var w = try Tensor(.{ .tap, .cin, .cout }).fromSlice(&ctx, .{ 3, 1, 1 }, &.{ 1, 2, 3 });
    defer w.deinit();

    var y = try x.conv1d(&ctx, .time, .cin, .tap, .cout, &w, 1, 1, 1, 1);
    defer y.deinit();
    // PyTorch: F.conv1d([1,2,3,4], [1,2,3], padding=1) = [8, 14, 20, 11].
    try expectCloseSlices(&.{ 8, 14, 20, 11 }, y.asRawTensor().dataConst(), 1e-6);
}

const Conv1dFdContext = struct {
    ctx: *ExecContext,
    x_vals: []f32,
    w_vals: []f32,
    coef: []const f32,
    seq: usize,
    in_ch: usize,
    ipg: usize,
    out_ch: usize,
    taps: usize,
    stride: usize,
    pad: usize,
    dilation: usize,
    groups: usize,
};

fn conv1dFdLoss(c: Conv1dFdContext) anyerror!f32 {
    var x = try c.ctx.fromSlice(&.{ c.seq, c.in_ch }, c.x_vals);
    defer x.deinit();
    var w = try c.ctx.fromSlice(&.{ c.taps, c.ipg, c.out_ch }, c.w_vals);
    defer w.deinit();
    var y = try c.ctx.conv1dAxisRank(2, &x, &w, 0, 1, c.stride, c.pad, c.dilation, c.groups);
    defer y.deinit();
    return fdWeightedSum(y.dataConst(), c.coef);
}

test "tagged autograd conv1d matches finite differences across stride/pad/dilation/groups" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // {stride, pad, dilation, groups, taps}: plain, strided+padded, dilated
    // with long pad, the grouped k=8 p=64 shape, the stride>1 AND dilation>1
    // interaction (exercises the backward-input divisibility skip), and a
    // grouped case with in_per_group > 1.
    const configs = [_][5]usize{
        .{ 1, 0, 1, 1, 3 },
        .{ 2, 3, 1, 1, 3 },
        .{ 1, 6, 3, 1, 5 },
        .{ 1, 64, 1, 4, 8 },
        .{ 2, 3, 3, 1, 3 },
        .{ 3, 2, 2, 2, 3 },
    };
    const seq: usize = 10;
    for (configs) |cfg| {
        const stride = cfg[0];
        const pad = cfg[1];
        const dilation = cfg[2];
        const groups = cfg[3];
        const taps = cfg[4];
        const in_ch: usize = switch (groups) {
            4 => 4,
            2 => 4,
            else => 2,
        };
        const out_ch: usize = switch (groups) {
            4 => 4,
            2 => 6,
            else => 3,
        };
        const ipg = in_ch / groups;
        const out_len = (seq + 2 * pad - (dilation * (taps - 1) + 1)) / stride + 1;

        const x_vals = try allocator.alloc(f32, seq * in_ch);
        defer allocator.free(x_vals);
        fdFillPattern(x_vals, 0.3);
        const w_vals = try allocator.alloc(f32, taps * ipg * out_ch);
        defer allocator.free(w_vals);
        fdFillPattern(w_vals, 1.7);
        const coef = try allocator.alloc(f32, out_len * out_ch);
        defer allocator.free(coef);
        fdFillPattern(coef, 2.9);

        var x = try Tensor(.{ .time, .cin }).variable(&ctx, try ctx.fromSlice(&.{ seq, in_ch }, x_vals));
        defer x.deinit();
        var w = try Tensor(.{ .tap, .cin, .cout }).variable(&ctx, try ctx.fromSlice(&.{ taps, ipg, out_ch }, w_vals));
        defer w.deinit();
        var coef_t = try Tensor(.{ .time, .cout }).fromTensor(&ctx, try ctx.fromSlice(&.{ out_len, out_ch }, coef));
        defer coef_t.deinit();

        var y = try x.conv1d(&ctx, .time, .cin, .tap, .cout, &w, stride, pad, dilation, groups);
        defer y.deinit();
        var weighted = try y.mul(&ctx, &coef_t);
        defer weighted.deinit();
        var loss = try weighted.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);

        var gx = (try x.grad(&ctx)).?;
        defer gx.deinit();
        var gw = (try w.grad(&ctx)).?;
        defer gw.deinit();

        const fd_ctx = Conv1dFdContext{
            .ctx = &ctx,
            .x_vals = x_vals,
            .w_vals = w_vals,
            .coef = coef,
            .seq = seq,
            .in_ch = in_ch,
            .ipg = ipg,
            .out_ch = out_ch,
            .taps = taps,
            .stride = stride,
            .pad = pad,
            .dilation = dilation,
            .groups = groups,
        };
        try fdCheckGrad(x_vals, gx.asRawTensor().dataConst(), 1e-2, 1e-2, fd_ctx, conv1dFdLoss);
        try fdCheckGrad(w_vals, gw.asRawTensor().dataConst(), 1e-2, 1e-2, fd_ctx, conv1dFdLoss);
    }
}

test "public Tensor convTranspose1d facade upsamples with bias" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    // t_in=2, IC=1, OC=1, K=2, stride=2, pad=0, output_pad=1: y (before bias)
    // = [x0*w_k0, x0*w_k1, x1*w_k0, x1*w_k1, 0] with weight2 rows [(oc*K + k), IC].
    var x = try Tensor(.{ .time, .cin }).fromSlice(&ctx, .{ 2, 1 }, &.{ 1, 2 });
    defer x.deinit();
    var w2 = try Tensor(.{ .kout, .cin }).fromSlice(&ctx, .{ 2, 1 }, &.{ 10, 20 });
    defer w2.deinit();
    var bias = try Tensor(.{.cout}).fromSlice(&ctx, .{1}, &.{100});
    defer bias.deinit();

    var y = try x.convTranspose1d(&ctx, .time, .cin, .kout, .cout, &w2, &bias, 1, 2, 2, 0, 1);
    defer y.deinit();
    try expectCloseSlices(&.{ 110, 120, 120, 140, 100 }, y.asRawTensor().dataConst(), 1e-6);
}

const ConvTranspose1dFdContext = struct {
    ctx: *ExecContext,
    x_vals: []f32,
    w_vals: []f32,
    b_vals: ?[]f32,
    coef: []const f32,
    t_in: usize,
    in_ch: usize,
    out_ch: usize,
    taps: usize,
    stride: usize,
    pad: usize,
    output_pad: usize,
};

fn convTranspose1dFdLoss(c: ConvTranspose1dFdContext) anyerror!f32 {
    var x = try c.ctx.fromSlice(&.{ c.t_in, c.in_ch }, c.x_vals);
    defer x.deinit();
    var w2 = try c.ctx.fromSlice(&.{ c.taps * c.out_ch, c.in_ch }, c.w_vals);
    defer w2.deinit();
    var bias: ?RawTensor = if (c.b_vals) |bv| try c.ctx.fromSlice(&.{c.out_ch}, bv) else null;
    defer if (bias) |*b| b.deinit();
    var y = try c.ctx.convTranspose1d(&x, &w2, if (bias) |*b| b else null, c.out_ch, c.taps, c.stride, c.pad, c.output_pad);
    defer y.deinit();
    return fdWeightedSum(y.dataConst(), c.coef);
}

test "tagged autograd convTranspose1d matches finite differences at DAC configs" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // {stride, taps, pad, output_pad}: two of the DAC decoder configs.
    const configs = [_][4]usize{
        .{ 8, 16, 4, 0 },
        .{ 5, 10, 3, 1 },
    };
    const t_in: usize = 3;
    const in_ch: usize = 2;
    const out_ch: usize = 2;
    for (configs) |cfg| {
        const stride = cfg[0];
        const taps = cfg[1];
        const pad = cfg[2];
        const output_pad = cfg[3];
        const out_len = (t_in - 1) * stride + taps - 2 * pad + output_pad;

        for ([_]bool{ true, false }) |with_bias| {
            const x_vals = try allocator.alloc(f32, t_in * in_ch);
            defer allocator.free(x_vals);
            fdFillPattern(x_vals, 0.9);
            const w_vals = try allocator.alloc(f32, taps * out_ch * in_ch);
            defer allocator.free(w_vals);
            fdFillPattern(w_vals, 4.2);
            const b_vals = try allocator.alloc(f32, out_ch);
            defer allocator.free(b_vals);
            fdFillPattern(b_vals, 6.1);
            const coef = try allocator.alloc(f32, out_len * out_ch);
            defer allocator.free(coef);
            fdFillPattern(coef, 7.4);

            var x = try Tensor(.{ .time, .cin }).variable(&ctx, try ctx.fromSlice(&.{ t_in, in_ch }, x_vals));
            defer x.deinit();
            var w2 = try Tensor(.{ .kout, .cin }).variable(&ctx, try ctx.fromSlice(&.{ taps * out_ch, in_ch }, w_vals));
            defer w2.deinit();
            var bias: ?Tensor(.{.cout}) = if (with_bias) try Tensor(.{.cout}).variable(&ctx, try ctx.fromSlice(&.{out_ch}, b_vals)) else null;
            defer if (bias) |*b| b.deinit();
            var coef_t = try Tensor(.{ .time, .cout }).fromTensor(&ctx, try ctx.fromSlice(&.{ out_len, out_ch }, coef));
            defer coef_t.deinit();

            var y = try x.convTranspose1d(&ctx, .time, .cin, .kout, .cout, &w2, if (bias) |*b| b else null, out_ch, taps, stride, pad, output_pad);
            defer y.deinit();
            var weighted = try y.mul(&ctx, &coef_t);
            defer weighted.deinit();
            var loss = try weighted.sumAll(&ctx);
            defer loss.deinit();
            try loss.backward(&ctx);

            var gx = (try x.grad(&ctx)).?;
            defer gx.deinit();
            var gw = (try w2.grad(&ctx)).?;
            defer gw.deinit();

            const fd_ctx = ConvTranspose1dFdContext{
                .ctx = &ctx,
                .x_vals = x_vals,
                .w_vals = w_vals,
                .b_vals = if (with_bias) b_vals else null,
                .coef = coef,
                .t_in = t_in,
                .in_ch = in_ch,
                .out_ch = out_ch,
                .taps = taps,
                .stride = stride,
                .pad = pad,
                .output_pad = output_pad,
            };
            try fdCheckGrad(x_vals, gx.asRawTensor().dataConst(), 1e-2, 1e-2, fd_ctx, convTranspose1dFdLoss);
            try fdCheckGrad(w_vals, gw.asRawTensor().dataConst(), 1e-2, 1e-2, fd_ctx, convTranspose1dFdLoss);
            if (with_bias) {
                var gb = (try bias.?.grad(&ctx)).?;
                defer gb.deinit();
                // The bias broadcast onto ALL output rows including the
                // output_pad ones — the FD loss sees exactly that.
                try fdCheckGrad(b_vals, gb.asRawTensor().dataConst(), 1e-2, 1e-2, fd_ctx, convTranspose1dFdLoss);
            }
        }
    }
}

test "public Tensor snake facade applies per-channel activation" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var x = try Tensor(.{ .time, .ch }).fromSlice(&ctx, .{ 2, 2 }, &.{ 0.5, -1.0, 2.0, 0.0 });
    defer x.deinit();
    var alpha = try Tensor(.{.ch}).fromSlice(&ctx, .{2}, &.{ 1.0, 2.0 });
    defer alpha.deinit();
    var inv_b = try Tensor(.{.ch}).fromSlice(&ctx, .{2}, &.{ 1.0, 0.5 });
    defer inv_b.deinit();

    var y = try x.snake(&ctx, .ch, &alpha, &inv_b);
    defer y.deinit();
    const s05 = @sin(@as(f32, 0.5));
    const sm2 = @sin(@as(f32, -2.0));
    const s4 = @sin(@as(f32, 2.0));
    try expectCloseSlices(&.{
        0.5 + s05 * s05,
        -1.0 + 0.5 * (sm2 * sm2),
        2.0 + s4 * s4,
        0.0,
    }, y.asRawTensor().dataConst(), 1e-6);
}

const SnakeFdContext = struct {
    ctx: *ExecContext,
    x_vals: []f32,
    a_vals: []f32,
    ib_vals: []f32,
    coef: []const f32,
    rows: usize,
    cols: usize,
};

fn snakeFdLoss(c: SnakeFdContext) anyerror!f32 {
    var x = try c.ctx.fromSlice(&.{ c.rows, c.cols }, c.x_vals);
    defer x.deinit();
    var alpha = try c.ctx.fromSlice(&.{c.cols}, c.a_vals);
    defer alpha.deinit();
    var inv_b = try c.ctx.fromSlice(&.{c.cols}, c.ib_vals);
    defer inv_b.deinit();
    var y = try c.ctx.snakeRows(&x, &alpha, &inv_b);
    defer y.deinit();
    return fdWeightedSum(y.dataConst(), c.coef);
}

test "tagged autograd snake matches finite differences for input alpha and inv_b" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // cols=13: not a multiple of any SIMD width.
    const rows: usize = 3;
    const cols: usize = 13;

    const x_vals = try allocator.alloc(f32, rows * cols);
    defer allocator.free(x_vals);
    fdFillPattern(x_vals, 0.5);
    const a_vals = try allocator.alloc(f32, cols);
    defer allocator.free(a_vals);
    fdFillPattern(a_vals, 3.3);
    for (a_vals) |*a| a.* = @abs(a.*) + 0.25;
    const ib_vals = try allocator.alloc(f32, cols);
    defer allocator.free(ib_vals);
    for (ib_vals, a_vals) |*ib, a| ib.* = 1.0 / (a + 1e-9);
    const coef = try allocator.alloc(f32, rows * cols);
    defer allocator.free(coef);
    fdFillPattern(coef, 5.8);

    var x = try Tensor(.{ .time, .ch }).variable(&ctx, try ctx.fromSlice(&.{ rows, cols }, x_vals));
    defer x.deinit();
    var alpha = try Tensor(.{.ch}).variable(&ctx, try ctx.fromSlice(&.{cols}, a_vals));
    defer alpha.deinit();
    var inv_b = try Tensor(.{.ch}).variable(&ctx, try ctx.fromSlice(&.{cols}, ib_vals));
    defer inv_b.deinit();
    var coef_t = try Tensor(.{ .time, .ch }).fromTensor(&ctx, try ctx.fromSlice(&.{ rows, cols }, coef));
    defer coef_t.deinit();

    var y = try x.snake(&ctx, .ch, &alpha, &inv_b);
    defer y.deinit();
    var weighted = try y.mul(&ctx, &coef_t);
    defer weighted.deinit();
    var loss = try weighted.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    var ga = (try alpha.grad(&ctx)).?;
    defer ga.deinit();
    var gib = (try inv_b.grad(&ctx)).?;
    defer gib.deinit();

    const fd_ctx = SnakeFdContext{
        .ctx = &ctx,
        .x_vals = x_vals,
        .a_vals = a_vals,
        .ib_vals = ib_vals,
        .coef = coef,
        .rows = rows,
        .cols = cols,
    };
    try fdCheckGrad(x_vals, gx.asRawTensor().dataConst(), 1e-2, 1e-2, fd_ctx, snakeFdLoss);
    try fdCheckGrad(a_vals, ga.asRawTensor().dataConst(), 1e-2, 1e-2, fd_ctx, snakeFdLoss);
    try fdCheckGrad(ib_vals, gib.asRawTensor().dataConst(), 1e-2, 1e-2, fd_ctx, snakeFdLoss);
}

test "public Tensor groupNorm facade normalizes per group" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const eps: f32 = 1e-5;
    var x = try Tensor(.{ .time, .ch }).fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();

    // G=C: per-channel InstanceNorm over time; col0 {1,3} and col1 {2,4} both
    // have biased var 1.
    var wt = try Tensor(.{.ch}).fromSlice(&ctx, .{2}, &.{ 2.0, 3.0 });
    defer wt.deinit();
    var bt = try Tensor(.{.ch}).fromSlice(&ctx, .{2}, &.{ 10.0, 20.0 });
    defer bt.deinit();
    var y = try x.groupNorm(&ctx, .ch, 2, eps, &wt, &bt);
    defer y.deinit();
    const inv = 1.0 / @sqrt(@as(f32, 1.0) + eps);
    try expectCloseSlices(&.{ -inv * 2 + 10, -inv * 3 + 20, inv * 2 + 10, inv * 3 + 20 }, y.asRawTensor().dataConst(), 1e-5);

    // No-affine arm.
    var y_plain = try x.groupNorm(&ctx, .ch, 2, eps, null, null);
    defer y_plain.deinit();
    try expectCloseSlices(&.{ -inv, -inv, inv, inv }, y_plain.asRawTensor().dataConst(), 1e-6);
}

const GroupNormFdContext = struct {
    ctx: *ExecContext,
    x_vals: []f32,
    w_vals: ?[]f32,
    b_vals: ?[]f32,
    coef: []const f32,
    rows: usize,
    cols: usize,
    groups: usize,
    eps: f32,
};

fn groupNormFdLoss(c: GroupNormFdContext) anyerror!f32 {
    var x = try c.ctx.fromSlice(&.{ c.rows, c.cols }, c.x_vals);
    defer x.deinit();
    var weight: ?RawTensor = if (c.w_vals) |wv| try c.ctx.fromSlice(&.{c.cols}, wv) else null;
    defer if (weight) |*w| w.deinit();
    var bias: ?RawTensor = if (c.b_vals) |bv| try c.ctx.fromSlice(&.{c.cols}, bv) else null;
    defer if (bias) |*b| b.deinit();
    var y = try c.ctx.groupNormAxisRank(&x, c.groups, c.eps, if (weight) |*w| w else null, if (bias) |*b| b else null);
    defer y.deinit();
    return fdWeightedSum(y.dataConst(), c.coef);
}

test "tagged autograd groupNorm matches finite differences across group configurations" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const rows: usize = 3;
    const cols: usize = 8;
    const eps: f32 = 1e-5;

    // {groups, affine}: G=1, G=C, G=4 with affine, G=4 without.
    const configs = [_]struct { groups: usize, affine: bool }{
        .{ .groups = 1, .affine = false },
        .{ .groups = cols, .affine = false },
        .{ .groups = 4, .affine = true },
        .{ .groups = 4, .affine = false },
    };
    for (configs) |cfg| {
        const x_vals = try allocator.alloc(f32, rows * cols);
        defer allocator.free(x_vals);
        fdFillPattern(x_vals, 1.1);
        const w_vals = try allocator.alloc(f32, cols);
        defer allocator.free(w_vals);
        fdFillPattern(w_vals, 2.4);
        for (w_vals) |*wv| wv.* += 1.0; // keep the affine weight away from 0
        const b_vals = try allocator.alloc(f32, cols);
        defer allocator.free(b_vals);
        fdFillPattern(b_vals, 3.6);
        const coef = try allocator.alloc(f32, rows * cols);
        defer allocator.free(coef);
        fdFillPattern(coef, 4.7);

        var x = try Tensor(.{ .time, .ch }).variable(&ctx, try ctx.fromSlice(&.{ rows, cols }, x_vals));
        defer x.deinit();
        var weight: ?Tensor(.{.ch}) = if (cfg.affine) try Tensor(.{.ch}).variable(&ctx, try ctx.fromSlice(&.{cols}, w_vals)) else null;
        defer if (weight) |*w| w.deinit();
        var bias: ?Tensor(.{.ch}) = if (cfg.affine) try Tensor(.{.ch}).variable(&ctx, try ctx.fromSlice(&.{cols}, b_vals)) else null;
        defer if (bias) |*b| b.deinit();
        var coef_t = try Tensor(.{ .time, .ch }).fromTensor(&ctx, try ctx.fromSlice(&.{ rows, cols }, coef));
        defer coef_t.deinit();

        var y = try x.groupNorm(&ctx, .ch, cfg.groups, eps, if (weight) |*w| w else null, if (bias) |*b| b else null);
        defer y.deinit();
        var weighted = try y.mul(&ctx, &coef_t);
        defer weighted.deinit();
        var loss = try weighted.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);

        var gx = (try x.grad(&ctx)).?;
        defer gx.deinit();

        const fd_ctx = GroupNormFdContext{
            .ctx = &ctx,
            .x_vals = x_vals,
            .w_vals = if (cfg.affine) w_vals else null,
            .b_vals = if (cfg.affine) b_vals else null,
            .coef = coef,
            .rows = rows,
            .cols = cols,
            .groups = cfg.groups,
            .eps = eps,
        };
        try fdCheckGrad(x_vals, gx.asRawTensor().dataConst(), 1e-2, 1e-2, fd_ctx, groupNormFdLoss);
        if (cfg.affine) {
            var gw = (try weight.?.grad(&ctx)).?;
            defer gw.deinit();
            var gb = (try bias.?.grad(&ctx)).?;
            defer gb.deinit();
            try fdCheckGrad(w_vals, gw.asRawTensor().dataConst(), 1e-2, 1e-2, fd_ctx, groupNormFdLoss);
            try fdCheckGrad(b_vals, gb.asRawTensor().dataConst(), 1e-2, 1e-2, fd_ctx, groupNormFdLoss);
        }
    }
}

test "tagged autograd differentiates elu and geluErf" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{.d}).variableFromSlice(&ctx, .{2}, &.{ -1.0, 1.25 });
    defer x.deinit();

    var elu_x = try x.elu(&ctx);
    defer elu_x.deinit();
    var gelu_x = try x.geluErf(&ctx);
    defer gelu_x.deinit();

    // Forward values: elu(-1) = expm1(-1); geluErf(x) = x*Phi(x) with the
    // standard normal CDF: 1.25*Phi(1.25) = 1.1179378, -1*Phi(-1) = -0.15865526.
    try expectCloseSlices(&.{ -0.6321206, 1.25 }, elu_x.asRawTensor().dataConst(), 1e-6);
    try expectCloseSlices(&.{ -0.15865526, 1.1179378 }, gelu_x.asRawTensor().dataConst(), 1e-5);

    var total = try elu_x.add(&ctx, &gelu_x);
    defer total.deinit();
    var loss = try total.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var grad = (try x.grad(&ctx)).?;
    defer grad.deinit();
    // elu' = exp(x) for x <= 0, 1 for x > 0;
    // geluErf' = 0.5*(1+erf(x/sqrt(2))) + x*exp(-x*x/2)/sqrt(2*pi).
    const x0: f32 = -1.0;
    const x1: f32 = 1.25;
    const expected = [_]f32{
        @exp(x0) + testGeluErfDerivative(x0),
        1.0 + testGeluErfDerivative(x1),
    };
    try expectCloseSlices(&expected, grad.asRawTensor().dataConst(), 1e-5);

    // Finite-difference cross-check of the analytic geluErf derivative.
    inline for (.{ x0, x1 }) |point| {
        const h: f32 = 1e-3;
        const fd = (geluErfValue(point + h) - geluErfValue(point - h)) / (2 * h);
        try std.testing.expectApproxEqAbs(fd, testGeluErfDerivative(point), 1e-3);
    }
}

fn geluErfValue(value: f32) f32 {
    return 0.5 * value * (1 + backend_mod.ops.erff(value * 0.70710678118654752440084436210484));
}

fn testGeluErfDerivative(value: f32) f32 {
    const inv_sqrt_2pi: f32 = 0.3989422804014327;
    const cdf = 0.5 * (1 + backend_mod.ops.erff(value * 0.70710678118654752440084436210484));
    return cdf + value * @exp(-0.5 * value * value) * inv_sqrt_2pi;
}

test "public Tensor mseLoss/huberLoss/bceLoss/klDivLoss values and two-parent gradients" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const T = Tensor(.{ .batch, .d });
    var x = try T.variableFromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();
    var t = try T.variableFromSlice(&ctx, .{ 2, 2 }, &.{ 0.5, 2.5, 1.5, 4 });
    defer t.deinit();

    // torch F.mse_loss: mean((x - t)^2) with d = {0.5, -0.5, 1.5, 0}.
    var mse = try x.mseLoss(&ctx, &t, .{});
    defer mse.deinit();
    try std.testing.expectApproxEqAbs(0.6875, try mse.item(), 1e-6);
    try mse.backward(&ctx);
    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    var gt = (try t.grad(&ctx)).?;
    defer gt.deinit();
    // d/dx = 2·d/4 = {0.25, -0.25, 0.75, 0}; d/dt = -d/dx.
    try expectCloseSlices(&.{ 0.25, -0.25, 0.75, 0 }, gx.asRawTensor().dataConst(), 1e-6);
    try expectCloseSlices(&.{ -0.25, 0.25, -0.75, 0 }, gt.asRawTensor().dataConst(), 1e-6);

    // `.none` keeps the input tags/shape (per-element losses).
    var mse_none = try x.mseLoss(&ctx, &t, .{ .reduction = .none });
    defer mse_none.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, mse_none.asRawTensor().shape.slice());
    try expectCloseSlices(&.{ 0.25, 0.25, 2.25, 0 }, try mse_none.dataConst(), 1e-6);

    // torch F.huber_loss(delta=1): d = {0.5, -0.5, 1.5, 0} ->
    // {0.125, 0.125, 1·(1.5 - 0.5), 0}; sum = 1.25.
    var huber = try x.huberLoss(&ctx, &t, .{ .reduction = .sum });
    defer huber.deinit();
    try std.testing.expectApproxEqAbs(1.25, try huber.item(), 1e-6);

    // torch F.binary_cross_entropy_with_logits on logits {0, 2} vs
    // targets {0, 1}: (ln 2 + log1p(e^-2))/2 = 0.4100375958014592.
    const V = Tensor(.{.d});
    var logits = try V.variableFromSlice(&ctx, .{2}, &.{ 0, 2 });
    defer logits.deinit();
    var bce_target = try V.fromSlice(&ctx, .{2}, &.{ 0, 1 });
    defer bce_target.deinit();
    var bce = try logits.bceLoss(&ctx, &bce_target, .{ .from_logits = true });
    defer bce.deinit();
    try std.testing.expectApproxEqAbs(0.4100375958014592, try bce.item(), 1e-6);
    try bce.backward(&ctx);
    var g_logits = (try logits.grad(&ctx)).?;
    defer g_logits.deinit();
    // d/dx = (sigmoid(x) - y)/2 = {0.25, -0.059601461}.
    try expectCloseSlices(&.{ 0.25, -0.05960146101105877 }, g_logits.asRawTensor().dataConst(), 1e-6);

    // torch F.kl_div(x, t, reduction='sum'): x = ln{0.7, 0.3}, t = {0.5, 0.5}:
    // sum t·(ln t - x) = 0.5·(ln 0.5 - ln 0.7) + 0.5·(ln 0.5 - ln 0.3).
    var logp = try V.variableFromSlice(&ctx, .{2}, &.{ -0.35667494393873245, -1.2039728043259361 });
    defer logp.deinit();
    var kl_target = try V.fromSlice(&ctx, .{2}, &.{ 0.5, 0.5 });
    defer kl_target.deinit();
    var kl = try logp.klDivLoss(&ctx, &kl_target, .{ .reduction = .sum });
    defer kl.deinit();
    try std.testing.expectApproxEqAbs(0.08717669357238897, try kl.item(), 1e-6);
    try kl.backward(&ctx);
    var g_logp = (try logp.grad(&ctx)).?;
    defer g_logp.deinit();
    // d/dx = -t.
    try expectCloseSlices(&.{ -0.5, -0.5 }, g_logp.asRawTensor().dataConst(), 1e-6);
}

test "public Tensor nllLoss composes -logp[label] with reductions and gradient" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const T = Tensor(.{ .pos, .class });
    const logp = [_]f32{ -1, -2, -0.5, -0.3, -1.2, -2.3 };
    const labels = [_]usize{ 2, 0 };

    // No-grad values work unscoped (the intermediates are freed eagerly).
    var c = try T.fromSlice(&ctx, .{ 2, 3 }, &logp);
    defer c.deinit();

    // torch F.nll_loss on log-probs: mean of {-x[0,2], -x[1,0]} = (0.5 + 0.3)/2.
    var mean_loss = try c.nllLoss(&ctx, .class, &labels, .mean);
    defer mean_loss.deinit();
    try std.testing.expectApproxEqAbs(0.4, try mean_loss.item(), 1e-6);

    var sum_loss = try c.nllLoss(&ctx, .class, &labels, .sum);
    defer sum_loss.deinit();
    try std.testing.expectApproxEqAbs(0.8, try sum_loss.item(), 1e-6);

    // `.none` removes the class tag: per-position losses {0.5, 0.3}.
    var none_loss = try c.nllLoss(&ctx, .class, &labels, .none);
    defer none_loss.deinit();
    try std.testing.expectEqualSlices(usize, &.{2}, none_loss.asRawTensor().shape.slice());
    try expectCloseSlices(&.{ 0.5, 0.3 }, try none_loss.dataConst(), 1e-6);

    // Validation: label count and range.
    try std.testing.expectError(error.InvalidDataLength, c.nllLoss(&ctx, .class, &.{2}, .mean));
    try std.testing.expectError(error.IndexOutOfBounds, c.nllLoss(&ctx, .class, &.{ 3, 0 }, .mean));

    // Grad tracking without an exec scope is a LOUD error (the composed
    // intermediates would dangle) — the training pattern below is scoped.
    var x = try T.variableFromSlice(&ctx, .{ 2, 3 }, &logp);
    defer x.deinit();
    try std.testing.expectError(error.ActiveExecScopeRequired, x.nllLoss(&ctx, .class, &labels, .mean));

    // mean gradient: -onehot/positions.
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        var loss = try x.nllLoss(&ctx, .class, &labels, .mean);
        defer loss.deinit();
        try std.testing.expectApproxEqAbs(0.4, try loss.item(), 1e-6);
        try loss.backward(&ctx);
    }
    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try expectCloseSlices(&.{ 0, 0, -0.5, -0.5, 0, 0 }, gx.asRawTensor().dataConst(), 1e-6);
}

test "public Tensor l2Normalize and cosineSimilarity compositions" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    // y = x·rsqrt(sum(x²) + eps): {3, 4} -> {0.6, 0.8} (eps = 0).
    const V = Tensor(.{.d});
    var v = try V.fromSlice(&ctx, .{2}, &.{ 3, 4 });
    defer v.deinit();
    var vn = try v.l2Normalize(&ctx, .d, 0);
    defer vn.deinit();
    try expectCloseSlices(&.{ 0.6, 0.8 }, try vn.dataConst(), 1e-6);

    // Per-row normalization along the tag axis of a rank-2 tensor.
    const M = Tensor(.{ .row, .d });
    var m = try M.fromSlice(&ctx, .{ 2, 2 }, &.{ 3, 4, 6, 8 });
    defer m.deinit();
    var mn = try m.l2Normalize(&ctx, .d, 0);
    defer mn.deinit();
    try expectCloseSlices(&.{ 0.6, 0.8, 0.6, 0.8 }, try mn.dataConst(), 1e-6);

    // cos({1,0,1,0}, {1,1,0,0}) = 1/(√2·√2) = 0.5 (torch F.cosine_similarity).
    const W = Tensor(.{.d});
    var a = try W.fromSlice(&ctx, .{4}, &.{ 1, 0, 1, 0 });
    defer a.deinit();
    var b = try W.fromSlice(&ctx, .{4}, &.{ 1, 1, 0, 0 });
    defer b.deinit();
    var cos = try a.cosineSimilarity(&ctx, &b, .d, 1e-8);
    defer cos.deinit();
    try std.testing.expectApproxEqAbs(0.5, try cos.item(), 1e-6);

    // Zero vector: the eps clamp keeps the quotient finite -> similarity 0.
    var zero = try W.fromSlice(&ctx, .{4}, &.{ 0, 0, 0, 0 });
    defer zero.deinit();
    var cos_zero = try zero.cosineSimilarity(&ctx, &b, .d, 1e-8);
    defer cos_zero.deinit();
    try std.testing.expectApproxEqAbs(0.0, try cos_zero.item(), 1e-6);

    // Rank-2 reduces the tag away: rows {3,4}·{3,4} colinear -> 1,
    // {1,0}·{0,1} orthogonal -> 0.
    var p = try M.fromSlice(&ctx, .{ 2, 2 }, &.{ 3, 4, 1, 0 });
    defer p.deinit();
    var q = try M.fromSlice(&ctx, .{ 2, 2 }, &.{ 3, 4, 0, 1 });
    defer q.deinit();
    var cos_rows = try p.cosineSimilarity(&ctx, &q, .d, 1e-8);
    defer cos_rows.deinit();
    try std.testing.expectEqualSlices(usize, &.{2}, cos_rows.asRawTensor().shape.slice());
    try expectCloseSlices(&.{ 1, 0 }, try cos_rows.dataConst(), 1e-6);

    // Grad tracking without an exec scope is a LOUD error for these
    // compositions (gradcheck covers the scoped gradient path).
    var vx = try V.variableFromSlice(&ctx, .{2}, &.{ 3, 4 });
    defer vx.deinit();
    try std.testing.expectError(error.ActiveExecScopeRequired, vx.l2Normalize(&ctx, .d, 1e-6));
    var wx = try W.variableFromSlice(&ctx, .{4}, &.{ 1, 0, 1, 0 });
    defer wx.deinit();
    try std.testing.expectError(error.ActiveExecScopeRequired, wx.cosineSimilarity(&ctx, &b, .d, 1e-8));
}

// --- comparison/logical masks + shape/scan/sort ops --------------------------

test "public Tensor comparison and logical ops produce constant 0/1 masks" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const V = Tensor(.{.d});
    // Variables in, constant masks out: comparisons are non-differentiable
    // (the argmax precedent), so downstream where/maskedFill treat them as
    // plain masks.
    var a = try V.variableFromSlice(&ctx, .{4}, &.{ 1, 2, 3, 4 });
    defer a.deinit();
    var b = try V.variableFromSlice(&ctx, .{4}, &.{ 1, 5, 2, 4 });
    defer b.deinit();

    // torch.lt(a, b).float() = {0, 1, 0, 0}; torch.ge = {1, 0, 1, 1}.
    var lt = try a.compare(&ctx, .lt, &b);
    defer lt.deinit();
    try std.testing.expect(!lt.requiresGrad());
    try expectCloseSlices(&.{ 0, 1, 0, 0 }, try lt.dataConst(), 0);
    var ge = try a.compare(&ctx, .ge, &b);
    defer ge.deinit();
    try expectCloseSlices(&.{ 1, 0, 1, 1 }, try ge.dataConst(), 0);

    // torch.eq(a, 4).float() and torch.le(a, 2).float().
    var eq4 = try a.compare(&ctx, .eq, 4);
    defer eq4.deinit();
    try std.testing.expect(!eq4.requiresGrad());
    try expectCloseSlices(&.{ 0, 0, 0, 1 }, try eq4.dataConst(), 0);
    var le2 = try a.compare(&ctx, .le, 2);
    defer le2.deinit();
    try expectCloseSlices(&.{ 1, 1, 0, 0 }, try le2.dataConst(), 0);

    // Logical combinators over != 0 truthiness.
    var both = try le2.logicalAnd(&ctx, &eq4);
    defer both.deinit();
    try std.testing.expect(!both.requiresGrad());
    try expectCloseSlices(&.{ 0, 0, 0, 0 }, try both.dataConst(), 0);
    var either = try le2.logicalOr(&ctx, &eq4);
    defer either.deinit();
    try expectCloseSlices(&.{ 1, 1, 0, 1 }, try either.dataConst(), 0);
    var one_of = try le2.logicalXor(&ctx, &eq4);
    defer one_of.deinit();
    try expectCloseSlices(&.{ 1, 1, 0, 1 }, try one_of.dataConst(), 0);
    var neither = try either.logicalNot(&ctx);
    defer neither.deinit();
    try expectCloseSlices(&.{ 0, 0, 1, 0 }, try neither.dataConst(), 0);

    // The mask feeds where/maskedFill directly: where(le2, a, b).
    var picked = try a.where(&ctx, le2, &b);
    defer picked.deinit();
    try expectCloseSlices(&.{ 1, 2, 2, 4 }, try picked.dataConst(), 0);
}

test "public Tensor maskedSelect gathers masked elements and routes gradients" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const M = Tensor(.{ .row, .col });
    var mask = try M.fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 0, 0, 1 });
    defer mask.deinit();

    // torch.masked_select: row-major selected elements {1, 4}. No-grad
    // composition works unscoped (intermediates are freed eagerly).
    var c = try M.fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer c.deinit();
    var picked = try c.maskedSelect(&ctx, mask, .m);
    defer picked.deinit();
    try std.testing.expectEqualSlices(usize, &.{2}, picked.asRawTensor().shape.slice());
    try expectCloseSlices(&.{ 1, 4 }, try picked.dataConst(), 0);

    // Empty selection is a loud but RECOVERABLE error (zero-size tensors are
    // unrepresentable): the dedicated EmptySelection, distinct from the shape
    // errors, so a no-match outcome is catchable apart from caller bugs.
    var none = try M.fromSlice(&ctx, .{ 2, 2 }, &.{ 0, 0, 0, 0 });
    defer none.deinit();
    try std.testing.expectError(error.EmptySelection, c.maskedSelect(&ctx, none, .m));
    var misshapen = try M.fromSlice(&ctx, .{ 1, 4 }, &.{ 0, 0, 0, 1 });
    defer misshapen.deinit();
    try std.testing.expectError(error.ShapeMismatch, c.maskedSelect(&ctx, misshapen, .m));

    // Grad tracking without an exec scope is a LOUD error (composed op).
    var x = try M.variableFromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();
    try std.testing.expectError(error.ActiveExecScopeRequired, x.maskedSelect(&ctx, mask, .m));

    // Scoped gradient: d(sum of selected)/dx = the mask itself.
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        var y = try x.maskedSelect(&ctx, mask, .m);
        defer y.deinit();
        var loss = try y.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);
    }
    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try expectCloseSlices(&.{ 1, 0, 0, 1 }, gx.asRawTensor().dataConst(), 0);
}

test "public Tensor stack unbindInto flip roll repeatAxis shape compositions" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const V = Tensor(.{.d});
    var a = try V.fromSlice(&ctx, .{2}, &.{ 1, 2 });
    defer a.deinit();
    var b = try V.fromSlice(&ctx, .{2}, &.{ 3, 4 });
    defer b.deinit();

    // torch.stack([a, b], dim=0) -> [[1,2],[3,4]] tagged {s, d}.
    var stacked = try a.stack(&ctx, .s, 0, &.{&b});
    defer stacked.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, stacked.asRawTensor().shape.slice());
    try expectCloseSlices(&.{ 1, 2, 3, 4 }, try stacked.dataConst(), 0);

    // dim=1 insertion: torch.stack([a, b], dim=1) -> [[1,3],[2,4]].
    var stacked1 = try a.stack(&ctx, .s, 1, &.{&b});
    defer stacked1.deinit();
    try expectCloseSlices(&.{ 1, 3, 2, 4 }, try stacked1.dataConst(), 0);

    // torch.unbind(stacked, dim=0) -> ({1,2}, {3,4}); caller owns the outs.
    var parts: [2]Tensor(.{.d}) = undefined;
    try stacked.unbindInto(&ctx, .s, &parts);
    defer for (&parts) |*part| part.deinit();
    try expectCloseSlices(&.{ 1, 2 }, try parts[0].dataConst(), 0);
    try expectCloseSlices(&.{ 3, 4 }, try parts[1].dataConst(), 0);
    var wrong: [3]Tensor(.{.d}) = undefined;
    try std.testing.expectError(error.InvalidShape, stacked.unbindInto(&ctx, .s, &wrong));

    // torch.flip / torch.roll on one dim.
    var seq = try V.fromSlice(&ctx, .{4}, &.{ 1, 2, 3, 4 });
    defer seq.deinit();
    var flipped = try seq.flip(&ctx, .d);
    defer flipped.deinit();
    try expectCloseSlices(&.{ 4, 3, 2, 1 }, try flipped.dataConst(), 0);
    var rolled = try seq.roll(&ctx, .d, 1);
    defer rolled.deinit();
    try expectCloseSlices(&.{ 4, 1, 2, 3 }, try rolled.dataConst(), 0);
    var rolled_back = try seq.roll(&ctx, .d, -1);
    defer rolled_back.deinit();
    try expectCloseSlices(&.{ 2, 3, 4, 1 }, try rolled_back.dataConst(), 0);
    var rolled_far = try seq.roll(&ctx, .d, 5);
    defer rolled_far.deinit();
    try expectCloseSlices(&.{ 4, 1, 2, 3 }, try rolled_far.dataConst(), 0);

    // x.repeat(2) on one dim; n == 1 is an identity view, n == 0 is an error.
    var doubled = try a.repeatAxis(&ctx, .d, 2);
    defer doubled.deinit();
    try expectCloseSlices(&.{ 1, 2, 1, 2 }, try doubled.dataConst(), 0);
    var once = try a.repeatAxis(&ctx, .d, 1);
    defer once.deinit();
    try expectCloseSlices(&.{ 1, 2 }, try once.dataConst(), 0);
    try std.testing.expectError(error.InvalidShape, a.repeatAxis(&ctx, .d, 0));

    // Grad tracking without an exec scope is a LOUD error for the two
    // compositions with function-local intermediates (stack, unbindInto);
    // gradcheck covers their scoped gradient paths.
    var xv = try V.variableFromSlice(&ctx, .{2}, &.{ 1, 2 });
    defer xv.deinit();
    try std.testing.expectError(error.ActiveExecScopeRequired, xv.stack(&ctx, .s, 0, &.{&b}));
    var sv = try Tensor(.{ .s, .d }).variableFromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer sv.deinit();
    var vparts: [2]Tensor(.{.d}) = undefined;
    try std.testing.expectError(error.ActiveExecScopeRequired, sv.unbindInto(&ctx, .s, &vparts));
}

test "public Tensor cumsum values and reversed-cumsum gradient" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const V = Tensor(.{.d});
    var x = try V.variableFromSlice(&ctx, .{3}, &.{ 1, 2, 3 });
    defer x.deinit();

    // torch.cumsum(x, 0) = {1, 3, 6}.
    var y = try x.cumsum(&ctx, .d);
    defer y.deinit();
    try expectCloseSlices(&.{ 1, 3, 6 }, try y.dataConst(), 0);

    // d(sum y)/dx = suffix counts {3, 2, 1} (reversed cumsum of ones).
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try expectCloseSlices(&.{ 3, 2, 1 }, gx.asRawTensor().dataConst(), 0);

    // Rank-2 leading-axis scan: torch.cumsum(m, 0).
    const M = Tensor(.{ .row, .col });
    var m = try M.fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer m.deinit();
    var mc = try m.cumsum(&ctx, .row);
    defer mc.deinit();
    try expectCloseSlices(&.{ 1, 2, 4, 6 }, try mc.dataConst(), 0);
}

test "public Tensor pad values and narrowed gradient" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const V = Tensor(.{.d});
    var x = try V.variableFromSlice(&ctx, .{2}, &.{ 1, 2 });
    defer x.deinit();

    // torch F.pad(x, (1, 2), value=9) = {9, 1, 2, 9, 9}.
    var y = try x.pad(&ctx, .d, 1, 2, 9);
    defer y.deinit();
    try std.testing.expectEqualSlices(usize, &.{5}, y.asRawTensor().shape.slice());
    try expectCloseSlices(&.{ 9, 1, 2, 9, 9 }, try y.dataConst(), 0);

    // Weighted sum: d(Σ w·y)/dx picks w at the body offset -> {2, 3}.
    var w = try V.fromSlice(&ctx, .{5}, &.{ 1, 2, 3, 4, 5 });
    defer w.deinit();
    var z = try y.mul(&ctx, &w);
    defer z.deinit();
    var loss = try z.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try expectCloseSlices(&.{ 2, 3 }, gx.asRawTensor().dataConst(), 0);
}

test "public Tensor sort and argsort values, constant indices, scatter gradient" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const V = Tensor(.{.d});
    var x = try V.variableFromSlice(&ctx, .{3}, &.{ 3, 1, 2 });
    defer x.deinit();

    // torch.sort(x): values {1, 2, 3}, indices {1, 2, 0}.
    var sorted = try x.sort(&ctx, .d, false);
    defer sorted.deinit();
    try expectCloseSlices(&.{ 1, 2, 3 }, try sorted.values.dataConst(), 0);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 0 }, try sorted.indices.dataConst());
    try std.testing.expect(!sorted.indices.requiresGrad());

    // torch.argsort(x, descending=True) = {0, 2, 1}; no grad.
    var order = try x.argsort(&ctx, .d, true);
    defer order.deinit();
    try std.testing.expectEqualSlices(i64, &.{ 0, 2, 1 }, try order.dataConst());
    try std.testing.expect(!order.requiresGrad());

    // Weighted sum over the sorted values: w = {1, 2, 3} routes back through
    // the permutation -> gx = {3, 1, 2} (x[0]=3 landed in output slot 2, ...).
    var w = try V.fromSlice(&ctx, .{3}, &.{ 1, 2, 3 });
    defer w.deinit();
    var z = try sorted.values.mul(&ctx, &w);
    defer z.deinit();
    var loss = try z.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try expectCloseSlices(&.{ 3, 1, 2 }, gx.asRawTensor().dataConst(), 0);
}

// ---------------- dotTernarySte (BitNet b1.58 STE linear) ----------------

const ternary_k = dtype_mod.qk_k_block_size;

fn ternaryTestFill(values: []f32, modulus: usize, scale: f32) void {
    const half: i64 = @intCast(modulus / 2);
    for (values, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(@as(i64, @intCast(i % modulus)) - half)) * scale;
    }
}

test "public f32 Tensor dotTernarySte forward matches manual encode plus kernel" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const m = 3;
    const n = 4;
    var x_values: [m * ternary_k]f32 = undefined;
    ternaryTestFill(&x_values, 11, 0.125);
    var w_values: [n * ternary_k]f32 = undefined;
    ternaryTestFill(&w_values, 13, 0.21);

    var x = try Tensor(.{ .batch, .in }).fromSlice(&ctx, .{ m, ternary_k }, &x_values);
    defer x.deinit();
    var w = try Tensor(.{ .out, .in }).fromSlice(&ctx, .{ n, ternary_k }, &w_values);
    defer w.deinit();

    var y = try x.dotTernarySte(&ctx, &w, .in);
    defer y.deinit();
    try std.testing.expectEqualSlices(usize, &.{ m, n }, y.asRawTensor().shape.slice());

    var rhs = try backend_mod.quantized_matmul.quantizedMatmulRhsTQ2_0FromF32Absmean(allocator, ternary_k, n, &w_values);
    defer rhs.deinit();
    var expected: [m * n]f32 = undefined;
    backend_mod.quantized_matmul.matmulTQ2_0F32RhsRange(&expected, &x_values, &rhs, m, n, 0, m);
    try std.testing.expectEqualSlices(f32, &expected, y.asRawTensor().dataConst());
}

fn ternarySteDxLoss(ctx: *ExecContext, x: *const Tensor(.{ .batch, .in })) !Tensor(.{}) {
    var w_values: [2 * ternary_k]f32 = undefined;
    ternaryTestFill(&w_values, 7, 0.35);
    var w = try Tensor(.{ .out, .in }).fromSlice(ctx, .{ 2, ternary_k }, &w_values);
    defer w.deinit();
    var y = try x.dotTernarySte(ctx, &w, .in);
    defer y.deinit();
    return y.sumAll(ctx);
}

test "public f32 Tensor dotTernarySte dx passes gradcheck against the quantized weight" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x_values: [ternary_k]f32 = undefined;
    ternaryTestFill(&x_values, 11, 0.05);
    var x = try Tensor(.{ .batch, .in }).variableFromSlice(&ctx, .{ 1, ternary_k }, &x_values);
    defer x.deinit();

    // The op is linear in x, so central differences are exact up to f32
    // evaluation noise; the wide eps keeps the cancellation error small.
    const result = try gradcheck_mod.gradcheck(&ctx, ternarySteDxLoss, .{&x}, .{ .eps = 1e-2 });
    try std.testing.expectEqual(@as(usize, ternary_k), result.checked);
}

test "public f32 Tensor dotTernarySte weight grad is the plain matmul VJP (STE identity)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const m = 2;
    const n = 3;
    var x_values: [m * ternary_k]f32 = undefined;
    ternaryTestFill(&x_values, 9, 0.2);
    var w_values: [n * ternary_k]f32 = undefined;
    ternaryTestFill(&w_values, 13, 0.4);

    var x = try Tensor(.{ .batch, .in }).variableFromSlice(&ctx, .{ m, ternary_k }, &x_values);
    defer x.deinit();
    var w = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ n, ternary_k }, &w_values);
    defer w.deinit();

    var y = try x.dotTernarySte(&ctx, &w, .in);
    defer y.deinit();
    // Scale y by a fixed non-uniform constant before the sum so gy = c with
    // every entry distinct: a permuted/transposed gy in either VJP cannot
    // cancel out the way it would under the uniform gy of a plain sumAll.
    const c_values = [m * n]f32{ 2, -1, 0.5, 3, 1.5, -2 };
    var c = try Tensor(.{ .batch, .out }).fromSlice(&ctx, .{ m, n }, &c_values);
    defer c.deinit();
    var weighted = try y.mul(&ctx, &c);
    defer weighted.deinit();
    var loss = try weighted.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    // dW is the plain trans_b matmul VJP with gy = c:
    // dW[o][i] = sum_r gy[r][o]·x[r][i] — NOT clipped/masked even where |w|
    // rounds outside {-1, 0, +1}.
    var gw = (try w.grad(&ctx)).?;
    defer gw.deinit();
    const gw_data = gw.asRawTensor().dataConst();
    for (0..n) |o| {
        for (0..ternary_k) |i| {
            var expected: f32 = 0;
            for (0..m) |r| expected += c_values[r * n + o] * x_values[r * ternary_k + i];
            try std.testing.expectApproxEqAbs(expected, gw_data[o * ternary_k + i], 1e-5);
        }
    }

    // dx flows through the QUANTIZED weight:
    // dx[r][i] = sum_o gy[r][o]·dequant(W_q)[o][i].
    var rhs = try backend_mod.quantized_matmul.quantizedMatmulRhsTQ2_0FromF32Absmean(allocator, ternary_k, n, &w_values);
    defer rhs.deinit();
    var w_dequant: [n * ternary_k]f32 = undefined;
    for (0..n) |o| {
        try backend_mod.quantized_matmul.dequantizeRowTQ2_0Into(w_dequant[o * ternary_k ..][0..ternary_k], rhs.columnBlocks(o));
    }
    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    const gx_data = gx.asRawTensor().dataConst();
    for (0..m) |r| {
        for (0..ternary_k) |i| {
            var expected: f32 = 0;
            for (0..n) |o| expected += c_values[r * n + o] * w_dequant[o * ternary_k + i];
            try std.testing.expectApproxEqAbs(expected, gx_data[r * ternary_k + i], 1e-5);
        }
    }
}

test "public f32 Tensor dotTernarySte constant weight gets no grad and no error" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x_values: [ternary_k]f32 = undefined;
    ternaryTestFill(&x_values, 11, 0.1);
    var w_values: [2 * ternary_k]f32 = undefined;
    ternaryTestFill(&w_values, 7, 0.3);

    var x = try Tensor(.{ .batch, .in }).variableFromSlice(&ctx, .{ 1, ternary_k }, &x_values);
    defer x.deinit();
    var w = try Tensor(.{ .out, .in }).fromSlice(&ctx, .{ 2, ternary_k }, &w_values);
    defer w.deinit();
    try std.testing.expect(!w.requiresGrad());

    var y = try x.dotTernarySte(&ctx, &w, .in);
    defer y.deinit();
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expect((try w.grad(&ctx)) == null);
}

test "public f32 Tensor dotTernarySte works under exec scope" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const m = 2;
    const n = 3;
    var x_values: [m * ternary_k]f32 = undefined;
    ternaryTestFill(&x_values, 9, 0.2);
    var w_values: [n * ternary_k]f32 = undefined;
    ternaryTestFill(&w_values, 13, 0.4);

    var x = try Tensor(.{ .batch, .in }).variableFromSlice(&ctx, .{ m, ternary_k }, &x_values);
    defer x.deinit();
    var w = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ n, ternary_k }, &w_values);
    defer w.deinit();

    // dx flows through the dequantized weight; precompute it once.
    var rhs = try backend_mod.quantized_matmul.quantizedMatmulRhsTQ2_0FromF32Absmean(allocator, ternary_k, n, &w_values);
    defer rhs.deinit();
    var w_dequant: [n * ternary_k]f32 = undefined;
    for (0..n) |o| {
        try backend_mod.quantized_matmul.dequantizeRowTQ2_0Into(w_dequant[o * ternary_k ..][0..ternary_k], rhs.columnBlocks(o));
    }

    // Two steps with per-iteration scopes: the op's hand-inlined finishOp
    // tail (reserveScopeSlot + adoptIntoScope + scope_owned) runs with the
    // scope active, and the scope close must free y/loss exactly once.
    for (0..2) |_| {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const y = try x.dotTernarySte(&ctx, &w, .in);
        const loss = try y.sumAll(&ctx);
        try loss.backward(&ctx);

        // gy = ones[m, n]: dW[o][i] = sum_r x[r][i] (STE identity), and
        // dx[r][i] = sum_o dequant(W_q)[o][i].
        var gw = (try w.grad(&ctx)).?;
        defer gw.deinit();
        const gw_data = gw.asRawTensor().dataConst();
        for (0..n) |o| {
            for (0..ternary_k) |i| {
                var expected: f32 = 0;
                for (0..m) |r| expected += x_values[r * ternary_k + i];
                try std.testing.expectApproxEqAbs(expected, gw_data[o * ternary_k + i], 1e-5);
            }
        }
        var gx = (try x.grad(&ctx)).?;
        defer gx.deinit();
        const gx_data = gx.asRawTensor().dataConst();
        for (0..m) |r| {
            for (0..ternary_k) |i| {
                var expected: f32 = 0;
                for (0..n) |o| expected += w_dequant[o * ternary_k + i];
                try std.testing.expectApproxEqAbs(expected, gx_data[r * ternary_k + i], 1e-5);
            }
        }
        x.zeroGrad();
        w.zeroGrad();
    }
}

test "public f32 Tensor dotTernarySte no-grad result under exec scope is scope-owned" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const m = 2;
    const n = 3;
    var x_values: [m * ternary_k]f32 = undefined;
    ternaryTestFill(&x_values, 11, 0.125);
    var w_values: [n * ternary_k]f32 = undefined;
    ternaryTestFill(&w_values, 13, 0.21);

    var x = try Tensor(.{ .batch, .in }).fromSlice(&ctx, .{ m, ternary_k }, &x_values);
    defer x.deinit();
    var w = try Tensor(.{ .out, .in }).fromSlice(&ctx, .{ n, ternary_k }, &w_values);
    defer w.deinit();

    var rhs = try backend_mod.quantized_matmul.quantizedMatmulRhsTQ2_0FromF32Absmean(allocator, ternary_k, n, &w_values);
    defer rhs.deinit();
    var expected: [m * n]f32 = undefined;
    backend_mod.quantized_matmul.matmulTQ2_0F32RhsRange(&expected, &x_values, &rhs, m, n, 0, m);

    // Constant weight + no-grad x inside a scope: the no-grad branch frees
    // the encoded rhs and the result is adopted by the scope (a borrow — no
    // deinit here; closeExecScope must free it exactly once).
    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    const y = try x.dotTernarySte(&ctx, &w, .in);
    try std.testing.expect(!y.requiresGrad());
    try std.testing.expectEqualSlices(f32, &expected, y.asRawTensor().dataConst());
}

test "public f32 Tensor dotTernarySte rejects a contract dim that is not a 256 multiple" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const k = 128;
    var x_values: [k]f32 = undefined;
    ternaryTestFill(&x_values, 11, 0.1);
    var w_values: [2 * k]f32 = undefined;
    ternaryTestFill(&w_values, 7, 0.3);

    var x = try Tensor(.{ .batch, .in }).fromSlice(&ctx, .{ 1, k }, &x_values);
    defer x.deinit();
    var w = try Tensor(.{ .out, .in }).fromSlice(&ctx, .{ 2, k }, &w_values);
    defer w.deinit();

    try std.testing.expectError(error.TernaryContractDimNotBlockAligned, x.dotTernarySte(&ctx, &w, .in));
}

test "public Tensor maxPool2d avgPool2d upsample2xNearest prelu channelAffine forward values" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const Map = Tensor(.{ .h, .w, .c });
    const Chan = Tensor(.{.c});

    // 4x4x1 fixture.
    var x = try Map.fromSlice(&ctx, .{ 4, 4, 1 }, &.{
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16,
    });
    defer x.deinit();

    // max 2x2 s2 p0: window maxima.
    var mp = try x.maxPool2d(&ctx, .{ 2, 2 }, .{ 2, 2 }, .{ 0, 0 });
    defer mp.deinit();
    try expectCloseSlices(&.{ 6, 8, 14, 16 }, mp.asRawTensor().dataConst(), 0);

    // max 3x3 s2 p1: −inf border (out-of-range taps skipped).
    var mp3 = try x.maxPool2d(&ctx, .{ 3, 3 }, .{ 2, 2 }, .{ 1, 1 });
    defer mp3.deinit();
    try expectCloseSlices(&.{ 6, 8, 14, 16 }, mp3.asRawTensor().dataConst(), 0);

    // avg 2x2 s2 p1: corner windows hold ONE valid tap (count excludes pad).
    var ap = try x.avgPool2d(&ctx, .{ 2, 2 }, .{ 2, 2 }, .{ 1, 1 });
    defer ap.deinit();
    try expectCloseSlices(&.{ 1, 2.5, 4, 7, 8.5, 10, 13, 14.5, 16 }, ap.asRawTensor().dataConst(), 0);

    // avg 2x2 s2 p0: plain window means.
    var ap0 = try x.avgPool2d(&ctx, .{ 2, 2 }, .{ 2, 2 }, .{ 0, 0 });
    defer ap0.deinit();
    try expectCloseSlices(&.{ 3.5, 5.5, 11.5, 13.5 }, ap0.asRawTensor().dataConst(), 0);

    // upsample 2x on 1x2x2: each pixel becomes a 2x2 block.
    var u = try Map.fromSlice(&ctx, .{ 1, 2, 2 }, &.{ 1, 2, 3, 4 });
    defer u.deinit();
    var up = try u.upsample2xNearest(&ctx);
    defer up.deinit();
    try expectCloseSlices(&.{ 1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4 }, up.asRawTensor().dataConst(), 0);

    // prelu: negative lanes scale by the channel alpha.
    var px = try Map.fromSlice(&ctx, .{ 1, 2, 2 }, &.{ 2, -2, -4, 4 });
    defer px.deinit();
    var alpha = try Chan.fromSlice(&ctx, .{2}, &.{ 0.5, 0.25 });
    defer alpha.deinit();
    var py = try px.prelu(&ctx, &alpha);
    defer py.deinit();
    try expectCloseSlices(&.{ 2, -0.5, -2, 4 }, py.asRawTensor().dataConst(), 0);

    // channelAffine: y = x*s + t per channel.
    var s = try Chan.fromSlice(&ctx, .{2}, &.{ 2, -1 });
    defer s.deinit();
    var t = try Chan.fromSlice(&ctx, .{2}, &.{ 1, 0.5 });
    defer t.deinit();
    var ay = try px.channelAffine(&ctx, &s, &t);
    defer ay.deinit();
    try expectCloseSlices(&.{ 5, 2.5, -7, -3.5 }, ay.asRawTensor().dataConst(), 0);
}

test "conv2d winograd route matches the direct kernel (3x3 s1, pad 0/1, odd shapes)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Pin the Winograd route ON so the check is meaningful on every build
    // config (BLAS builds default the route off).
    const exec_conv = @import("../exec/conv.zig");
    exec_conv.setWinogradForTest(true);
    defer exec_conv.setWinogradForTest(null);

    const cases = [_][5]usize{
        // h, w, cin, cout, pad — even/odd spatial exercise full and partial
        // tiles; min(oh,ow) >= 14 shapes take the F4 tier, the rest F2.
        .{ 8, 8, 8, 8, 1 },
        .{ 5, 7, 17, 5, 1 },
        .{ 4, 4, 4, 8, 0 },
        .{ 9, 5, 8, 3, 1 },
        .{ 16, 16, 8, 8, 1 },
        .{ 17, 15, 12, 5, 1 },
        .{ 18, 14, 8, 8, 0 },
    };
    var seed: u64 = 1;
    for (cases) |cs| {
        const h = cs[0];
        const w = cs[1];
        const cin = cs[2];
        const cout = cs[3];
        const p = cs[4];
        const oh = h + 2 * p - 2;
        const ow = w + 2 * p - 2;

        const xd = try allocator.alloc(f32, h * w * cin);
        defer allocator.free(xd);
        const wd = try allocator.alloc(f32, cout * 9 * cin);
        defer allocator.free(wd);
        const bd = try allocator.alloc(f32, cout);
        defer allocator.free(bd);
        const rng_mod = @import("../rng.zig");
        rng_mod.gaussianFill(seed, xd, 1.0);
        rng_mod.gaussianFill(seed + 1, wd, 0.5);
        rng_mod.gaussianFill(seed + 2, bd, 0.5);
        seed += 3;

        // Facade conv2d — the eligible shape takes the Winograd route.
        var x = try Tensor(.{ .h, .w, .c }).fromSlice(&ctx, .{ h, w, cin }, xd);
        defer x.deinit();
        var wt = try Tensor(.{ .oc, .kh, .kw, .c }).fromSlice(&ctx, .{ cout, 3, 3, cin }, wd);
        defer wt.deinit();
        var bt = try Tensor(.{.oc}).fromSlice(&ctx, .{cout}, bd);
        defer bt.deinit();
        var y = try x.conv2d(&ctx, &wt, &bt, .{ 1, 1 }, .{ p, p }, 1, .{ .h, .w, .c });
        defer y.deinit();

        // Reference: the scalar backend's independent direct kernel.
        var xr = try RawTensor.fromSlice(allocator, &[_]usize{ h, w, cin }, xd);
        defer xr.deinit();
        var wr = try RawTensor.fromSlice(allocator, &[_]usize{ cout, 3, 3, cin }, wd);
        defer wr.deinit();
        var expected = try RawTensor.zeros(allocator, &[_]usize{ oh, ow, cout });
        defer expected.deinit();
        backend_mod.scalar_impl.conv2dIntoWithConfig(&expected, &xr, &wr, bd, .{
            .h = h,
            .w = w,
            .cin = cin,
            .oh = oh,
            .ow = ow,
            .cout = cout,
            .kh = 3,
            .kw = 3,
            .stride_h = 1,
            .stride_w = 1,
            .pad_h = p,
            .pad_w = p,
            .groups = 1,
        }, .{});

        // Winograd reassociates the 3x3 reduction: ~1e-6 relative vs direct.
        const yd = try y.dataConst();
        const ed = expected.dataConst();
        try std.testing.expectEqual(ed.len, yd.len);
        for (ed, yd) |e, a| {
            const tol = 1e-4 * @max(@as(f32, 1.0), @abs(e));
            try std.testing.expect(@abs(e - a) <= tol);
        }
    }
}

test "conv2dPrepared matches conv2d bitwise (winograd F2/F4 tiers, odd tails, cin gate, stride-2 fallback)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Pin the Winograd route ON so preparation is exercised on every build
    // config (BLAS builds default the route off).
    const exec_conv = @import("../exec/conv.zig");
    exec_conv.setWinogradForTest(true);
    defer exec_conv.setWinogradForTest(null);

    const cases = [_][6]usize{
        // h, w, cin, cout, pad, stride — F2-tier small maps (even/odd, full
        // and partial tiles), F4-tier large maps (min(oh,ow) >= 14, cin <=
        // 56), a large map with cin > 56 (F4-ineligible at prepare AND call
        // time: the prepared set carries f2 only and serves the call), and a
        // stride-2 call where the prepared planes are inert (non-Winograd
        // route, identical code path).
        .{ 8, 8, 8, 8, 1, 1 },
        .{ 5, 7, 17, 5, 1, 1 },
        .{ 9, 5, 8, 3, 0, 1 },
        .{ 16, 16, 8, 8, 1, 1 },
        .{ 17, 15, 12, 5, 1, 1 },
        .{ 16, 16, 60, 5, 1, 1 },
        .{ 9, 9, 8, 4, 1, 2 },
    };
    var seed: u64 = 101;
    for (cases) |cs| {
        const h = cs[0];
        const w = cs[1];
        const cin = cs[2];
        const cout = cs[3];
        const p = cs[4];
        const s = cs[5];

        const xd = try allocator.alloc(f32, h * w * cin);
        defer allocator.free(xd);
        const wd = try allocator.alloc(f32, cout * 9 * cin);
        defer allocator.free(wd);
        const bd = try allocator.alloc(f32, cout);
        defer allocator.free(bd);
        const rng_mod = @import("../rng.zig");
        rng_mod.gaussianFill(seed, xd, 1.0);
        rng_mod.gaussianFill(seed + 1, wd, 0.5);
        rng_mod.gaussianFill(seed + 2, bd, 0.5);
        seed += 3;

        var x = try Tensor(.{ .h, .w, .c }).fromSlice(&ctx, .{ h, w, cin }, xd);
        defer x.deinit();
        var wt = try Tensor(.{ .oc, .kh, .kw, .c }).fromSlice(&ctx, .{ cout, 3, 3, cin }, wd);
        defer wt.deinit();
        var bt = try Tensor(.{.oc}).fromSlice(&ctx, .{cout}, bd);
        defer bt.deinit();

        var prep = try wt.prepareConv2dWeights(&ctx);
        defer prep.deinit();
        // 3x3 with cin >= 4 and the route pinned on: F2 always prepared, F4
        // iff cin passes the max-cin gate (default 56); f4 => f2.
        try std.testing.expect(prep.f2 != null);
        try std.testing.expect((prep.f4 != null) == (cin <= 56));

        var y_ref = try x.conv2d(&ctx, &wt, &bt, .{ s, s }, .{ p, p }, 1, .{ .h, .w, .c });
        defer y_ref.deinit();
        var y_prep = try x.conv2dPrepared(&ctx, &wt, &prep, &bt, .{ s, s }, .{ p, p }, 1, .{ .h, .w, .c });
        defer y_prep.deinit();
        try std.testing.expectEqualSlices(f32, try y_ref.dataConst(), try y_prep.dataConst());

        var yr_ref = try x.conv2dRelu(&ctx, &wt, &bt, .{ s, s }, .{ p, p }, 1, .{ .h, .w, .c });
        defer yr_ref.deinit();
        var yr_prep = try x.conv2dPreparedRelu(&ctx, &wt, &prep, &bt, .{ s, s }, .{ p, p }, 1, .{ .h, .w, .c });
        defer yr_prep.deinit();
        try std.testing.expectEqualSlices(f32, try yr_ref.dataConst(), try yr_prep.dataConst());
    }
}

test "conv2dPrepared: 1x1 and .empty preparations are inert; grad operands are rejected" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const exec_conv = @import("../exec/conv.zig");
    exec_conv.setWinogradForTest(true);
    defer exec_conv.setWinogradForTest(null);

    const rng_mod = @import("../rng.zig");
    const xd = try allocator.alloc(f32, 8 * 8 * 8);
    defer allocator.free(xd);
    rng_mod.gaussianFill(11, xd, 1.0);
    var x = try Tensor(.{ .h, .w, .c }).fromSlice(&ctx, .{ 8, 8, 8 }, xd);
    defer x.deinit();

    // 1x1 weight: preparation is naturally `.empty` (not Winograd-shaped)
    // and the call takes the pointwise fast path — bitwise equal.
    const w1d = try allocator.alloc(f32, 4 * 8);
    defer allocator.free(w1d);
    rng_mod.gaussianFill(12, w1d, 0.5);
    var w1 = try Tensor(.{ .oc, .kh, .kw, .c }).fromSlice(&ctx, .{ 4, 1, 1, 8 }, w1d);
    defer w1.deinit();
    var prep1 = try w1.prepareConv2dWeights(&ctx);
    defer prep1.deinit();
    try std.testing.expect(prep1.f2 == null and prep1.f4 == null);
    var y1_ref = try x.conv2d(&ctx, &w1, null, .{ 1, 1 }, .{ 0, 0 }, 1, .{ .h, .w, .c });
    defer y1_ref.deinit();
    var y1_prep = try x.conv2dPrepared(&ctx, &w1, &prep1, null, .{ 1, 1 }, .{ 0, 0 }, 1, .{ .h, .w, .c });
    defer y1_prep.deinit();
    try std.testing.expectEqualSlices(f32, try y1_ref.dataConst(), try y1_prep.dataConst());

    // Explicit `.empty` on a Winograd-eligible call: per-call transform
    // fallback, bitwise equal to the unprepared conv.
    const w3d = try allocator.alloc(f32, 4 * 9 * 8);
    defer allocator.free(w3d);
    rng_mod.gaussianFill(13, w3d, 0.5);
    var w3 = try Tensor(.{ .oc, .kh, .kw, .c }).fromSlice(&ctx, .{ 4, 3, 3, 8 }, w3d);
    defer w3.deinit();
    const empty = exec_mod.ExecContext.PreparedConvWeights.empty;
    var y3_ref = try x.conv2d(&ctx, &w3, null, .{ 1, 1 }, .{ 1, 1 }, 1, .{ .h, .w, .c });
    defer y3_ref.deinit();
    var y3_empty = try x.conv2dPrepared(&ctx, &w3, &empty, null, .{ 1, 1 }, .{ 1, 1 }, 1, .{ .h, .w, .c });
    defer y3_empty.deinit();
    try std.testing.expectEqualSlices(f32, try y3_ref.dataConst(), try y3_empty.dataConst());

    // Grad guards (the dotPacked policy): a grad-carrying weight cannot be
    // prepared, and no operand of conv2dPrepared may require grad.
    var wg = try Tensor(.{ .oc, .kh, .kw, .c }).variableFromSlice(&ctx, .{ 4, 3, 3, 8 }, w3d);
    defer wg.deinit();
    try std.testing.expectError(error.GradientPreparedConv2dUnsupported, wg.prepareConv2dWeights(&ctx));
    var xg = try Tensor(.{ .h, .w, .c }).variableFromSlice(&ctx, .{ 8, 8, 8 }, xd);
    defer xg.deinit();
    try std.testing.expectError(
        error.GradientPreparedConv2dUnsupported,
        xg.conv2dPrepared(&ctx, &w3, &empty, null, .{ 1, 1 }, .{ 1, 1 }, 1, .{ .h, .w, .c }),
    );
    try std.testing.expectError(
        error.GradientPreparedConv2dUnsupported,
        x.conv2dPreparedRelu(&ctx, &wg, &empty, null, .{ 1, 1 }, .{ 1, 1 }, 1, .{ .h, .w, .c }),
    );
}

test "public f32 Tensor einsum with an f16 constant RHS matches the f32 path and keeps LHS gradients" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // .s is summed away, so the LHS gradient exercises the broadcast-expand
    // path of the const-RHS einsum backward.
    var x = try Tensor(.{ .s, .i, .k }).variableFromSlice(&ctx, .{ 2, 2, 3 }, &.{
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    });
    defer x.deinit();
    const w_data = [_]f32{ 1, -2, 0.5, 3, -1, 2 };
    const w16_data = [_]f16{ 1, -2, 0.5, 3, -1, 2 };
    var w16 = try Tensor(.{ .dtype = .f16, .tags = .{ .j, .k } }).fromSlice(&ctx, .{ 2, 3 }, &w16_data);
    defer w16.deinit();
    var w32 = try Tensor(.{ .j, .k }).variableFromSlice(&ctx, .{ 2, 3 }, &w_data);
    defer w32.deinit();

    var y16 = try x.einsum(&ctx, &w16, .{ .i, .j });
    defer y16.deinit();
    var y32 = try x.einsum(&ctx, &w32, .{ .i, .j });
    defer y32.deinit();
    try std.testing.expectEqualSlices(f32, y32.asRawTensor().dataConst(), y16.asRawTensor().dataConst());

    var loss16 = try y16.sumAll(&ctx);
    defer loss16.deinit();
    try loss16.backward(&ctx);
    var gx16 = (try x.grad(&ctx)).?;
    defer gx16.deinit();
    const gx16_data = try allocator.dupe(f32, gx16.asRawTensor().dataConst());
    defer allocator.free(gx16_data);
    x.zeroGrad();

    var loss32 = try y32.sumAll(&ctx);
    defer loss32.deinit();
    try loss32.backward(&ctx);
    var gx32 = (try x.grad(&ctx)).?;
    defer gx32.deinit();
    try std.testing.expectEqualSlices(f32, gx32.asRawTensor().dataConst(), gx16_data);
}

test "public f32 Tensor einsum under noGrad returns a constant" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try Tensor(.{ .m, .k }).variableFromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer a.deinit();
    var b = try Tensor(.{ .k, .n }).variableFromSlice(&ctx, .{ 2, 2 }, &.{ 5, 6, 7, 8 });
    defer b.deinit();

    var scope = control.noGrad();
    defer scope.close();
    var y = try a.einsum(&ctx, &b, .{ .m, .n });
    defer y.deinit();
    try std.testing.expect(y.grad_state == null);
    try std.testing.expectEqualSlices(f32, &.{ 19, 22, 43, 50 }, y.asRawTensor().dataConst());
}

test "public f32 Tensor dot with an f16 RHS and seven shared batch axes compiles and backpropagates" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Rank-8 operands whose equation has 7 batch axes: the contraction has
    // no rank-(batch+2) bmm representation, so the lowering must collapse
    // the batch group (regression: the delegated backward record used to
    // compile-error here while the forward was legal).
    const BatchTags = .{ .b1, .b2, .b3, .b4, .b5, .b6, .b7, .k };
    var x = try Tensor(BatchTags).variableFromSlice(&ctx, .{ 2, 1, 2, 1, 2, 1, 1, 3 }, &.{
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    });
    defer x.deinit();
    const w_data = [_]f16{ 1, -1, 2, 0.5, -2, 1, 3, -0.5, 1, -1, 0.5, 2, 1, -2, 0.5, 1, -1, 2, 0.5, 1, -0.5, 2, 1, -1 };
    var w = try Tensor(.{ .dtype = .f16, .tags = BatchTags }).fromSlice(&ctx, .{ 2, 1, 2, 1, 2, 1, 1, 3 }, &w_data);
    defer w.deinit();

    var y = try x.dot(&ctx, &w, .k);
    defer y.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 1, 2, 1, 2, 1, 1 }, y.asRawTensor().shape.slice());

    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 1, 2, 1, 2, 1, 1, 3 }, gx.asRawTensor().shape.slice());
    // dL/dx = w widened: spot-check the first batch's row.
    try std.testing.expectEqualSlices(f32, &.{ 1, -1, 2 }, gx.asRawTensor().dataConst()[0..3]);
}

test "public f32 Tensor einsum with a bf16 constant RHS matches the f32 path" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .i, .k }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();
    const w_data = [_]f32{ 1, -2, 0.5, 3, -1, 2 };
    var wb = try Tensor(.{ .dtype = .bf16, .tags = .{ .j, .k } }).fromSlice(&ctx, .{ 2, 3 }, &.{
        dtype_mod.f32ToBf16(1),
        dtype_mod.f32ToBf16(-2),
        dtype_mod.f32ToBf16(0.5),
        dtype_mod.f32ToBf16(3),
        dtype_mod.f32ToBf16(-1),
        dtype_mod.f32ToBf16(2),
    });
    defer wb.deinit();
    var w32 = try Tensor(.{ .j, .k }).variableFromSlice(&ctx, .{ 2, 3 }, &w_data);
    defer w32.deinit();

    var yb = try x.einsum(&ctx, &wb, .{ .i, .j });
    defer yb.deinit();
    var y32 = try x.einsum(&ctx, &w32, .{ .i, .j });
    defer y32.deinit();
    try std.testing.expectEqualSlices(f32, y32.asRawTensor().dataConst(), yb.asRawTensor().dataConst());

    var loss = try yb.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, gx.asRawTensor().shape.slice());
}

test "public Tensor contiguous borrows contiguous layouts and materializes strided views" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const M = Tensor(.{ .row, .col });
    var c = try M.fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer c.deinit();

    // Already contiguous: zero-copy alias of the same storage bytes.
    var cc = try c.contiguous(&ctx);
    defer cc.deinit();
    try std.testing.expectEqual((try c.dataConst()).ptr, (try cc.dataConst()).ptr);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6 }, try cc.dataConst());

    // Strided view: dataConst is a loud error on the view, and contiguous
    // returns an owned copy in logical order.
    var t = try c.permuteTo(&ctx, .{ .col, .row });
    defer t.deinit();
    try std.testing.expectError(error.UnsupportedView, t.dataConst());
    var tc = try t.contiguous(&ctx);
    defer tc.deinit();
    try std.testing.expect((try tc.dataConst()).ptr != (try c.dataConst()).ptr);
    try std.testing.expectEqualSlices(f32, &.{ 1, 4, 2, 5, 3, 6 }, try tc.dataConst());

    // Differentiable identity through a strided source view: gradient lands
    // in the source's own layout.
    var x = try M.variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();
    var xt = try x.permuteTo(&ctx, .{ .col, .row });
    defer xt.deinit();
    var xc = try xt.contiguous(&ctx);
    defer xc.deinit();
    try std.testing.expectError(error.MutableDataRequiresNoGrad, xc.data());
    var loss = try xc.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try expectCloseSlices(&.{ 1, 1, 1, 1, 1, 1 }, try gx.dataConst(), 0);

    // Already-contiguous grad case: identity node, gradient passes through.
    var v = try M.variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer v.deinit();
    var vc = try v.contiguous(&ctx);
    defer vc.deinit();
    var loss2 = try vc.sumAll(&ctx);
    defer loss2.deinit();
    try loss2.backward(&ctx);
    var gv = (try v.grad(&ctx)).?;
    defer gv.deinit();
    try expectCloseSlices(&.{ 1, 1, 1, 1, 1, 1 }, try gv.dataConst(), 0);
}

test "public Tensor maskedScatter scatters rank-1 values and routes gradients" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const V = Tensor(.{.d});
    var field = try V.fromSlice(&ctx, .{4}, &.{ 10, 10, 10, 10 });
    defer field.deinit();
    var mask = try V.fromSlice(&ctx, .{4}, &.{ 0, 1, 0, 1 });
    defer mask.deinit();
    var vals = try Tensor(.{.nz}).fromSlice(&ctx, .{2}, &.{ 3, 7 });
    defer vals.deinit();

    // torch masked_scatter with an exact-count contract. No-grad composition
    // works unscoped.
    var out = try field.maskedScatter(&ctx, mask, .nz, &vals);
    defer out.deinit();
    try expectCloseSlices(&.{ 10, 3, 10, 7 }, try out.dataConst(), 0);

    // Inverse pairing: maskedSelect(maskedScatter(f, m, v)) == v.
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        var back = try out.maskedSelect(&ctx, mask, .nz);
        defer back.deinit();
        try expectCloseSlices(&.{ 3, 7 }, try back.dataConst(), 0);
    }

    // Count and shape contracts are loud errors; the empty-selection case
    // gets the dedicated (recoverable) EmptySelection, as in maskedSelect.
    var none = try V.fromSlice(&ctx, .{4}, &.{ 0, 0, 0, 0 });
    defer none.deinit();
    try std.testing.expectError(error.EmptySelection, field.maskedScatter(&ctx, none, .nz, &vals));
    var short = try Tensor(.{.nz}).fromSlice(&ctx, .{1}, &.{3});
    defer short.deinit();
    try std.testing.expectError(error.InvalidShape, field.maskedScatter(&ctx, mask, .nz, &short));

    // Grad tracking without an exec scope is a LOUD error (composed op).
    var xf = try V.variableFromSlice(&ctx, .{4}, &.{ 10, 10, 10, 10 });
    defer xf.deinit();
    try std.testing.expectError(error.ActiveExecScopeRequired, xf.maskedScatter(&ctx, mask, .nz, &vals));

    // Scoped gradients with a nontrivial upstream gradient: weight the output
    // by w = {2, 3, 5, 7} so d_field = w·(1-mask) and d_vals = w at the
    // selected row-major positions.
    var xv = try Tensor(.{.nz}).variableFromSlice(&ctx, .{2}, &.{ 3, 7 });
    defer xv.deinit();
    var w = try V.fromSlice(&ctx, .{4}, &.{ 2, 3, 5, 7 });
    defer w.deinit();
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        var y = try xf.maskedScatter(&ctx, mask, .nz, &xv);
        defer y.deinit();
        var weighted = try y.mul(&ctx, &w);
        defer weighted.deinit();
        var loss = try weighted.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);
    }
    var gf = (try xf.grad(&ctx)).?;
    defer gf.deinit();
    try expectCloseSlices(&.{ 2, 0, 5, 0 }, try gf.dataConst(), 0);
    var gv = (try xv.grad(&ctx)).?;
    defer gv.deinit();
    try expectCloseSlices(&.{ 3, 7 }, try gv.dataConst(), 0);
}

test "public Tensor rollBy rotates per-section and shiftBy fills dropped positions" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const M = Tensor(.{ .batch, .seq });
    var x = try M.fromSlice(&ctx, .{ 2, 4 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer x.deinit();

    // Per-batch shifts {1, -1} with roll's sign convention.
    var rolled = try x.rollBy(&ctx, .seq, &.{ 1, -1 });
    defer rolled.deinit();
    try expectCloseSlices(&.{ 4, 1, 2, 3, 6, 7, 8, 5 }, try rolled.dataConst(), 0);

    // Offsets length must match the section count.
    try std.testing.expectError(error.InvalidShape, x.rollBy(&ctx, .seq, &.{1}));

    // Rank-1 scalar-offset compatibility: rollBy == roll.
    var v = try Tensor(.{.d}).fromSlice(&ctx, .{4}, &.{ 1, 2, 3, 4 });
    defer v.deinit();
    var r1 = try v.rollBy(&ctx, .d, &.{1});
    defer r1.deinit();
    var r2 = try v.roll(&ctx, .d, 1);
    defer r2.deinit();
    try std.testing.expectEqualSlices(f32, try r2.dataConst(), try r1.dataConst());

    // shiftBy: same offsets, non-circular, fill = 0.
    var shifted = try x.shiftBy(&ctx, .seq, &.{ 1, -1 }, 0);
    defer shifted.deinit();
    try expectCloseSlices(&.{ 0, 1, 2, 3, 6, 7, 8, 0 }, try shifted.dataConst(), 0);

    // Sections along a non-innermost axis: roll the .batch column sections.
    var colroll = try x.rollBy(&ctx, .batch, &.{ 1, 0, 1, 0 });
    defer colroll.deinit();
    try expectCloseSlices(&.{ 5, 2, 7, 4, 1, 6, 3, 8 }, try colroll.dataConst(), 0);

    // Gradients: rollBy is a permutation (all-ones); shiftBy zeroes the
    // positions shifted out of the axis.
    var xr = try M.variableFromSlice(&ctx, .{ 2, 4 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer xr.deinit();
    try std.testing.expectError(error.ActiveExecScopeRequired, xr.rollBy(&ctx, .seq, &.{ 1, -1 }));
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        var y = try xr.rollBy(&ctx, .seq, &.{ 1, -1 });
        defer y.deinit();
        var loss = try y.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);
    }
    var gr = (try xr.grad(&ctx)).?;
    defer gr.deinit();
    try expectCloseSlices(&.{ 1, 1, 1, 1, 1, 1, 1, 1 }, try gr.dataConst(), 0);

    var xs = try M.variableFromSlice(&ctx, .{ 2, 4 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer xs.deinit();
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        var y = try xs.shiftBy(&ctx, .seq, &.{ 1, -1 }, 0.5);
        defer y.deinit();
        var loss = try y.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);
    }
    // shift +1 drops source j=3 (batch 0); shift -1 drops source j=0 (batch 1).
    var gs = (try xs.grad(&ctx)).?;
    defer gs.deinit();
    try expectCloseSlices(&.{ 1, 1, 1, 0, 0, 1, 1, 1 }, try gs.dataConst(), 0);
}

test "public Tensor rollBy operates on strided views" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var x = try Tensor(.{ .row, .col }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();

    // Transposed view {col, row}: sections along .row are the 3 columns.
    var gx_owned: Tensor(.{ .row, .col }) = undefined;
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        var t = try x.permuteTo(&ctx, .{ .col, .row });
        defer t.deinit();
        var y = try t.rollBy(&ctx, .row, &.{ 1, 0, 1 });
        defer y.deinit();
        // t = {{1,4},{2,5},{3,6}}; rows of columns 0 and 2 swap.
        try expectCloseSlices(&.{ 4, 1, 2, 5, 6, 3 }, try y.dataConst(), 0);
        var loss = try y.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);
        gx_owned = (try x.grad(&ctx)).?;
    }
    defer gx_owned.deinit();
    try expectCloseSlices(&.{ 1, 1, 1, 1, 1, 1 }, try gx_owned.dataConst(), 0);
}

test "public Tensor zeroPad2d pads named axes by the (left, right, top, bottom) spec" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const M = Tensor(.{ .h, .w });
    var x = try M.fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();

    // (left=1, right=2, top=1, bottom=0): left/right grow the width axis,
    // top/bottom the height axis.
    var padded = try x.zeroPad2d(&ctx, .h, .w, .{ 1, 2, 1, 0 });
    defer padded.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 6 }, padded.asRawTensor().shape.slice());
    try expectCloseSlices(&.{
        0, 0, 0, 0, 0, 0,
        0, 1, 2, 3, 0, 0,
        0, 4, 5, 6, 0, 0,
    }, try padded.dataConst(), 0);

    // An integer pads all four sides.
    var q = try M.fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer q.deinit();
    var uniform = try q.zeroPad2d(&ctx, .h, .w, 1);
    defer uniform.deinit();
    try expectCloseSlices(&.{
        0, 0, 0, 0,
        0, 1, 2, 0,
        0, 3, 4, 0,
        0, 0, 0, 0,
    }, try uniform.dataConst(), 0);

    // Leading axes pass through (channel-first image layout).
    const C = Tensor(.{ .c, .h, .w });
    var xc = try C.fromSlice(&ctx, .{ 2, 2, 2 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer xc.deinit();
    var pc = try xc.zeroPad2d(&ctx, .h, .w, .{ 0, 1, 1, 0 });
    defer pc.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 3, 3 }, pc.asRawTensor().shape.slice());
    try expectCloseSlices(&.{
        0, 0, 0, 1, 2, 0, 3, 4, 0,
        0, 0, 0, 5, 6, 0, 7, 8, 0,
    }, try pc.dataConst(), 0);

    // constantPad2d carries the fill value.
    var s = try M.fromSlice(&ctx, .{ 1, 1 }, &.{5});
    defer s.deinit();
    var filled = try s.constantPad2d(&ctx, .h, .w, 1, 9);
    defer filled.deinit();
    try expectCloseSlices(&.{ 9, 9, 9, 9, 5, 9, 9, 9, 9 }, try filled.dataConst(), 0);
}

test "public Tensor constantPad2d crops on negative padding" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const M = Tensor(.{ .h, .w });
    var x = try M.fromSlice(&ctx, .{ 3, 4 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
    defer x.deinit();

    // (-1, 1, 0, -1): crop one column left, pad one right, crop one row bottom.
    var out = try x.zeroPad2d(&ctx, .h, .w, .{ -1, 1, 0, -1 });
    defer out.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 4 }, out.asRawTensor().shape.slice());
    try expectCloseSlices(&.{ 2, 3, 4, 0, 6, 7, 8, 0 }, try out.dataConst(), 0);

    // Mixed signs on ONE axis: pad left 2, crop right 1.
    var y = try M.fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer y.deinit();
    var mixed = try y.zeroPad2d(&ctx, .h, .w, .{ 2, -1, 0, 0 });
    defer mixed.deinit();
    try expectCloseSlices(&.{ 0, 0, 1, 2, 0, 0, 4, 5 }, try mixed.dataConst(), 0);

    // Crop-only padding still returns a regular contiguous tensor, never
    // a strided view.
    var crop_only = try x.zeroPad2d(&ctx, .h, .w, .{ 0, -1, -1, 0 });
    defer crop_only.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, crop_only.asRawTensor().shape.slice());
    try expectCloseSlices(&.{ 5, 6, 7, 9, 10, 11 }, try crop_only.dataConst(), 0);

    // Cropping an axis away entirely is a loud error.
    try std.testing.expectError(error.InvalidShape, x.zeroPad2d(&ctx, .h, .w, .{ 0, 0, -2, -1 }));
}

test "public Tensor zeroPad2d routes gradients to the interior only" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const M = Tensor(.{ .h, .w });
    var x = try M.variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();

    // Grad tracking without an exec scope is a LOUD error (composed op).
    try std.testing.expectError(error.ActiveExecScopeRequired, x.zeroPad2d(&ctx, .h, .w, 1));

    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        var y = try x.zeroPad2d(&ctx, .h, .w, .{ 1, 2, 1, 0 });
        defer y.deinit();
        var loss = try y.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);
    }
    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try expectCloseSlices(&.{ 1, 1, 1, 1, 1, 1 }, try gx.dataConst(), 0);

    // Cropped source positions receive zero gradient.
    var z = try M.variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer z.deinit();
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        var y = try z.zeroPad2d(&ctx, .h, .w, .{ 0, -1, 1, 0 });
        defer y.deinit();
        var loss = try y.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);
    }
    var gz = (try z.grad(&ctx)).?;
    defer gz.deinit();
    try expectCloseSlices(&.{ 1, 1, 0, 1, 1, 0 }, try gz.dataConst(), 0);

    // All-zero padding is the identity; gradient passes through, scoped.
    var w = try M.variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer w.deinit();
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        var y = try w.zeroPad2d(&ctx, .h, .w, 0);
        defer y.deinit();
        try expectCloseSlices(&.{ 1, 2, 3, 4, 5, 6 }, try y.dataConst(), 0);
        var loss = try y.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);
    }
    var gw = (try w.grad(&ctx)).?;
    defer gw.deinit();
    try expectCloseSlices(&.{ 1, 1, 1, 1, 1, 1 }, try gw.dataConst(), 0);
}

test "public Tensor *Like constructors take shape from the instance" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const M = Tensor(.{ .row, .col });
    var x = try M.variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();

    // Fresh no-grad constants with x's shape; the grad state does not
    // carry over, so mutable data access works on emptyLike storage.
    var e = try x.emptyLike(&ctx);
    defer e.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, e.asRawTensor().shape.slice());
    try std.testing.expect(!e.requiresGrad());
    @memset(try e.data(), 7);
    try expectCloseSlices(&.{ 7, 7, 7, 7, 7, 7 }, try e.dataConst(), 0);

    var z = try x.zerosLike(&ctx);
    defer z.deinit();
    try expectCloseSlices(&.{ 0, 0, 0, 0, 0, 0 }, try z.dataConst(), 0);

    var o = try x.onesLike(&ctx);
    defer o.deinit();
    try expectCloseSlices(&.{ 1, 1, 1, 1, 1, 1 }, try o.dataConst(), 0);

    var m = try x.fullLike(&ctx, -std.math.inf(f32));
    defer m.deinit();
    for (try m.dataConst()) |v| try std.testing.expect(std.math.isNegativeInf(v));

    // A strided view contributes its LOGICAL shape.
    var t = try x.permuteTo(&ctx, .{ .col, .row });
    defer t.deinit();
    var zt = try t.zerosLike(&ctx);
    defer zt.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, zt.asRawTensor().shape.slice());

    // Typed constant branch: same sugar, dtype preserved.
    const I = Tensor(.{ .dtype = .i32, .tags = .{.d} });
    var xi = try I.fromSlice(&ctx, .{4}, &.{ 5, 6, 7, 8 });
    defer xi.deinit();
    var zi = try xi.zerosLike(&ctx);
    defer zi.deinit();
    try std.testing.expectEqualSlices(i32, &.{ 0, 0, 0, 0 }, try zi.dataConst());
    var oi = try xi.onesLike(&ctx);
    defer oi.deinit();
    try std.testing.expectEqualSlices(i32, &.{ 1, 1, 1, 1 }, try oi.dataConst());
}

test "public Tensor isnan isinf isfinite produce constant masks" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const inf = std.math.inf(f32);
    const V = Tensor(.{.d});
    var x = try V.fromSlice(&ctx, .{5}, &.{ 1, inf, -inf, std.math.nan(f32), 0 });
    defer x.deinit();

    var nan_mask = try x.isnan(&ctx);
    defer nan_mask.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 0, 1, 0 }, try nan_mask.dataConst());
    var inf_mask = try x.isinf(&ctx);
    defer inf_mask.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0, 1, 1, 0, 0 }, try inf_mask.dataConst());
    var finite_mask = try x.isfinite(&ctx);
    defer finite_mask.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 0, 0, 0, 1 }, try finite_mask.dataConst());

    // Masks are constants even off a grad-tracked source, and need no scope.
    var xv = try V.variableFromSlice(&ctx, .{5}, &.{ 1, inf, -inf, std.math.nan(f32), 0 });
    defer xv.deinit();
    var m = try xv.isfinite(&ctx);
    defer m.deinit();
    try std.testing.expect(!m.requiresGrad());
}

test "public Tensor any all reductions follow torch truthiness" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const M = Tensor(.{ .row, .col });
    // Rows: {0, 0, 0} → any 0, all 0; {1, 0, 2} → any 1, all 0;
    // {3, NaN, -1} → NaN is truthy (torch.any/all): any 1, all 1.
    var x = try M.fromSlice(&ctx, .{ 3, 3 }, &.{ 0, 0, 0, 1, 0, 2, 3, std.math.nan(f32), -1 });
    defer x.deinit();

    var any_row = try x.any(&ctx, .col);
    defer any_row.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0, 1, 1 }, try any_row.dataConst());
    var all_row = try x.all(&ctx, .col);
    defer all_row.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 1 }, try all_row.dataConst());

    var any_scalar = try x.anyAll(&ctx);
    defer any_scalar.deinit();
    try std.testing.expectEqual(@as(f32, 1), try any_scalar.item());
    var all_scalar = try x.allAll(&ctx);
    defer all_scalar.deinit();
    try std.testing.expectEqual(@as(f32, 0), try all_scalar.item());
}

test "public Tensor maximum minimum route gradients with even tie split" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const V = Tensor(.{.d});
    var a = try V.variableFromSlice(&ctx, .{4}, &.{ 1, 5, 2, 2 });
    defer a.deinit();
    var b = try V.variableFromSlice(&ctx, .{4}, &.{ 3, 4, 2, -1 });
    defer b.deinit();

    var hi = try a.maximum(&ctx, &b);
    defer hi.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 3, 5, 2, 2 }, try hi.dataConst());
    var lo = try a.minimum(&ctx, &b);
    defer lo.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 4, 2, -1 }, try lo.dataConst());

    var loss = try hi.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    var ga = (try a.grad(&ctx)).?;
    defer ga.deinit();
    try expectCloseSlices(&.{ 0, 1, 0.5, 1 }, try ga.dataConst(), 0);
    var gb = (try b.grad(&ctx)).?;
    defer gb.deinit();
    try expectCloseSlices(&.{ 1, 0, 0.5, 0 }, try gb.dataConst(), 0);

    // NaN in either operand propagates (torch.maximum, NOT IEEE maxNum).
    var with_nan = try V.fromSlice(&ctx, .{4}, &.{ std.math.nan(f32), 1, 2, 3 });
    defer with_nan.deinit();
    var c = try V.fromSlice(&ctx, .{4}, &.{ 0, 1, 2, 3 });
    defer c.deinit();
    var nan_out = try c.maximum(&ctx, &with_nan);
    defer nan_out.deinit();
    try std.testing.expect(std.math.isNan((try nan_out.dataConst())[0]));
}

test "public Tensor pow with tensor exponent and both gradients" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const V = Tensor(.{.d});
    var base = try V.variableFromSlice(&ctx, .{2}, &.{ 2, 3 });
    defer base.deinit();
    var expo = try V.variableFromSlice(&ctx, .{2}, &.{ 3, 2 });
    defer expo.deinit();

    var y = try base.pow(&ctx, &expo);
    defer y.deinit();
    try expectCloseSlices(&.{ 8, 9 }, try y.dataConst(), 1e-6);

    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    var gb = (try base.grad(&ctx)).?;
    defer gb.deinit();
    // b·a^(b-1): {3·4, 2·3} = {12, 6}.
    try expectCloseSlices(&.{ 12, 6 }, try gb.dataConst(), 1e-5);
    var ge = (try expo.grad(&ctx)).?;
    defer ge.deinit();
    // ln(a)·a^b: {8·ln2, 9·ln3}.
    try expectCloseSlices(&.{ 8 * 0.6931472, 9 * 1.0986123 }, try ge.dataConst(), 1e-4);
}

test "public Tensor reshape reinterprets row-major with view-or-materialize" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const M = Tensor(.{ .row, .col });
    var x = try M.fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();

    // Contiguous 2x3 → 3x2 under new tags: same row-major data.
    var y = try x.reshape(&ctx, .{ .a, .b }, .{ 3, 2 });
    defer y.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, y.asRawTensor().shape.slice());
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6 }, try y.dataConst());

    // Rank-1 target degenerates to flatten; element-count mismatch is loud.
    var flat = try x.reshape(&ctx, .{.n}, .{6});
    defer flat.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6 }, try flat.dataConst());
    try std.testing.expectError(error.InvalidShape, x.reshape(&ctx, .{.n}, .{5}));

    // Non-contiguous source (transpose view) materializes: reshape reads
    // the LOGICAL row-major order, torch.reshape semantics.
    var xt = try x.transpose(&ctx, .{ .col, .row }); // 3x2 view: {1,4},{2,5},{3,6}
    defer xt.deinit();
    var zt = try xt.reshape(&ctx, .{.n}, .{6});
    defer zt.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 4, 2, 5, 3, 6 }, try zt.dataConst());

    // Gradient flows through the composed views (scope-required).
    var xv = try M.variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer xv.deinit();
    try std.testing.expectError(error.ActiveExecScopeRequired, xv.reshape(&ctx, .{ .a, .b }, .{ 3, 2 }));
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        var yv = try xv.reshape(&ctx, .{ .a, .b }, .{ 3, 2 });
        defer yv.deinit();
        var w = try Tensor(.{ .a, .b }).fromSlice(&ctx, .{ 3, 2 }, &.{ 1, 2, 3, 4, 5, 6 });
        defer w.deinit();
        var weighted = try yv.mul(&ctx, &w);
        defer weighted.deinit();
        var loss = try weighted.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);
    }
    var gx = (try xv.grad(&ctx)).?;
    defer gx.deinit();
    try expectCloseSlices(&.{ 1, 2, 3, 4, 5, 6 }, try gx.dataConst(), 0);
}

test "public Tensor logsumexp and logSoftmax match the analytic form" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const M = Tensor(.{ .row, .col });
    var x = try M.fromSlice(&ctx, .{ 2, 2 }, &.{ 0, @log(@as(f32, 3)), 1, 1 });
    defer x.deinit();

    // Row 0: log(1 + 3) = log 4; row 1: log(2e) = 1 + log 2.
    var lse = try x.logsumexp(&ctx, .col);
    defer lse.deinit();
    try expectCloseSlices(&.{ @log(@as(f32, 4)), 1 + @log(@as(f32, 2)) }, try lse.dataConst(), 1e-6);

    var lsm = try x.logSoftmax(&ctx, .col);
    defer lsm.deinit();
    // Row 0 probabilities {0.25, 0.75}; row 1 {0.5, 0.5}.
    try expectCloseSlices(&.{ @log(@as(f32, 0.25)), @log(@as(f32, 0.75)), @log(@as(f32, 0.5)), @log(@as(f32, 0.5)) }, try lsm.dataConst(), 1e-6);

    // Non-finite rows follow torch: all -inf → -inf, a +inf entry → +inf.
    const inf = std.math.inf(f32);
    var edge = try M.fromSlice(&ctx, .{ 2, 2 }, &.{ -inf, -inf, 0, inf });
    defer edge.deinit();
    var edge_lse = try edge.logsumexp(&ctx, .col);
    defer edge_lse.deinit();
    try std.testing.expect(std.math.isNegativeInf((try edge_lse.dataConst())[0]));
    try std.testing.expect(std.math.isPositiveInf((try edge_lse.dataConst())[1]));

    // d(logsumexp)/dx is the softmax (the shift's gradient cancels).
    var xv = try M.variableFromSlice(&ctx, .{ 2, 2 }, &.{ 0, @log(@as(f32, 3)), 1, 1 });
    defer xv.deinit();
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        var lv = try xv.logsumexp(&ctx, .col);
        defer lv.deinit();
        var loss = try lv.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);
    }
    var gx = (try xv.grad(&ctx)).?;
    defer gx.deinit();
    try expectCloseSlices(&.{ 0.25, 0.75, 0.5, 0.5 }, try gx.dataConst(), 1e-6);
}

test "public Tensor norm variants and the l2 gradient" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const M = Tensor(.{ .row, .col });
    var x = try M.fromSlice(&ctx, .{ 2, 2 }, &.{ 3, -4, 1, -1 });
    defer x.deinit();

    var l1 = try x.norm(&ctx, .col, .l1);
    defer l1.deinit();
    try expectCloseSlices(&.{ 7, 2 }, try l1.dataConst(), 0);
    var l2 = try x.norm(&ctx, .col, .l2);
    defer l2.deinit();
    try expectCloseSlices(&.{ 5, std.math.sqrt2 }, try l2.dataConst(), 1e-6);
    var linf = try x.norm(&ctx, .col, .inf);
    defer linf.deinit();
    try expectCloseSlices(&.{ 4, 1 }, try linf.dataConst(), 0);

    var total = try x.normAll(&ctx, .l1);
    defer total.deinit();
    try std.testing.expectEqual(@as(f32, 9), try total.item());

    // l2 gradient is x/‖x‖ (scope-required composed op).
    var xv = try Tensor(.{.d}).variableFromSlice(&ctx, .{2}, &.{ 3, -4 });
    defer xv.deinit();
    try std.testing.expectError(error.ActiveExecScopeRequired, xv.norm(&ctx, .d, .l2));
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        var n = try xv.norm(&ctx, .d, .l2);
        defer n.deinit();
        var loss = try n.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);
    }
    var gx = (try xv.grad(&ctx)).?;
    defer gx.deinit();
    try expectCloseSlices(&.{ 0.6, -0.8 }, try gx.dataConst(), 1e-6);
}

test "public Tensor floor ceil round sign reciprocal" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const V = Tensor(.{.d});
    var x = try V.fromSlice(&ctx, .{6}, &.{ -1.5, -0.5, 0.5, 1.5, 2.5, 2.3 });
    defer x.deinit();

    var fl = try x.floor(&ctx);
    defer fl.deinit();
    try std.testing.expectEqualSlices(f32, &.{ -2, -1, 0, 1, 2, 2 }, try fl.dataConst());
    var ce = try x.ceil(&ctx);
    defer ce.deinit();
    try std.testing.expectEqualSlices(f32, &.{ -1, 0, 1, 2, 3, 3 }, try ce.dataConst());
    // Round-half-to-even (torch.round): ties go to the even neighbor.
    var ro = try x.round(&ctx);
    defer ro.deinit();
    try std.testing.expectEqualSlices(f32, &.{ -2, -0.0, 0, 2, 2, 2 }, try ro.dataConst());
    var sg = try x.sign(&ctx);
    defer sg.deinit();
    try std.testing.expectEqualSlices(f32, &.{ -1, -1, 1, 1, 1, 1 }, try sg.dataConst());

    // sign preserves ±0 and propagates NaN; round passes big/NaN through.
    var edge = try V.fromSlice(&ctx, .{4}, &.{ 0.0, -0.0, std.math.nan(f32), 8388609.0 });
    defer edge.deinit();
    var sg_edge = try edge.sign(&ctx);
    defer sg_edge.deinit();
    const sge = try sg_edge.dataConst();
    try std.testing.expectEqual(@as(f32, 0), sge[0]);
    try std.testing.expect(std.math.signbit(sge[1]));
    try std.testing.expect(std.math.isNan(sge[2]));
    var ro_edge = try edge.round(&ctx);
    defer ro_edge.deinit();
    try std.testing.expectEqual(@as(f32, 8388609.0), (try ro_edge.dataConst())[3]);

    var r = try V.fromSlice(&ctx, .{3}, &.{ 2, -4, 0.5 });
    defer r.deinit();
    var rec = try r.reciprocal(&ctx);
    defer rec.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0.5, -0.25, 2 }, try rec.dataConst());

    // Gradients: piecewise-constant ops get exact zero; reciprocal -1/x².
    var xv = try V.variableFromSlice(&ctx, .{2}, &.{ 1.4, -2.6 });
    defer xv.deinit();
    var yv = try xv.round(&ctx);
    defer yv.deinit();
    var loss = try yv.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    var gx = (try xv.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0, 0 }, try gx.dataConst());

    var rv = try V.variableFromSlice(&ctx, .{2}, &.{ 2, -4 });
    defer rv.deinit();
    var ry = try rv.reciprocal(&ctx);
    defer ry.deinit();
    var rloss = try ry.sumAll(&ctx);
    defer rloss.deinit();
    try rloss.backward(&ctx);
    var gr = (try rv.grad(&ctx)).?;
    defer gr.deinit();
    try expectCloseSlices(&.{ -0.25, -0.0625 }, try gr.dataConst(), 1e-7);
}

test "public Tensor arange linspace oneHot constructors" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const V = Tensor(.{.d});
    var a = try V.arange(&ctx, 0, 5, 1);
    defer a.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0, 1, 2, 3, 4 }, try a.dataConst());
    var b = try V.arange(&ctx, 1, 2.5, 0.5); // exclusive end, torch.arange
    defer b.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 1.5, 2 }, try b.dataConst());
    var c = try V.arange(&ctx, 3, 0, -1); // negative step
    defer c.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 3, 2, 1 }, try c.dataConst());
    try std.testing.expectError(error.InvalidShape, V.arange(&ctx, 0, 5, 0));
    try std.testing.expectError(error.InvalidShape, V.arange(&ctx, 5, 0, 1)); // empty range

    var l = try V.linspace(&ctx, 0, 1, 5);
    defer l.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0, 0.25, 0.5, 0.75, 1 }, try l.dataConst());
    var one = try V.linspace(&ctx, 7, 9, 1);
    defer one.deinit();
    try std.testing.expectEqualSlices(f32, &.{7}, try one.dataConst());
    // The endpoint is pinned exactly even when the stride rounds.
    var pinned = try V.linspace(&ctx, 0, 1, 3);
    defer pinned.deinit();
    try std.testing.expectEqual(@as(f32, 1), (try pinned.dataConst())[2]);
    try std.testing.expectError(error.InvalidShape, V.linspace(&ctx, 0, 1, 0));

    const M = Tensor(.{ .pos, .class });
    var oh = try M.oneHot(&ctx, &.{ 2, 0, 1 }, 3);
    defer oh.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 1, 1, 0, 0, 0, 1, 0 }, try oh.dataConst());
    try std.testing.expectError(error.IndexOutOfBounds, M.oneHot(&ctx, &.{3}, 3));
    try std.testing.expectError(error.InvalidShape, M.oneHot(&ctx, &.{}, 3));
}

test "public Tensor rand randn uniform normal bernoulli ride the seed stream" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();
    const rng_mod = @import("../rng.zig");

    const M = Tensor(.{ .row, .col });
    // Same seed → identical tensor (the §6.8 checkpoint contract);
    // different seed → different values. Constructors are no-grad consts.
    var ua = try M.rand(&ctx, .{ 4, 8 }, 42);
    defer ua.deinit();
    var ub = try M.rand(&ctx, .{ 4, 8 }, 42);
    defer ub.deinit();
    try std.testing.expectEqualSlices(f32, try ua.dataConst(), try ub.dataConst());
    try std.testing.expect(!ua.requiresGrad());
    var uc = try M.rand(&ctx, .{ 4, 8 }, 43);
    defer uc.deinit();
    try std.testing.expect(!std.mem.eql(f32, try ua.dataConst(), try uc.dataConst()));
    for (try ua.dataConst()) |v| try std.testing.expect(v >= 0 and v < 1);

    // rand/uniform/randn/normal reproduce the documented rng.zig mappings.
    var expected: [32]f32 = undefined;
    rng_mod.uniformFill(42, &expected, 0, 1);
    try std.testing.expectEqualSlices(f32, &expected, try ua.dataConst());
    var un = try M.uniform(&ctx, .{ 4, 8 }, 7, -2, 3);
    defer un.deinit();
    rng_mod.uniformFill(7, &expected, -2, 3);
    try std.testing.expectEqualSlices(f32, &expected, try un.dataConst());
    for (try un.dataConst()) |v| try std.testing.expect(v >= -2 and v < 3);
    var g = try M.randn(&ctx, .{ 4, 8 }, 11);
    defer g.deinit();
    rng_mod.normalFill(11, &expected, 0, 1);
    try std.testing.expectEqualSlices(f32, &expected, try g.dataConst());
    var n = try M.normal(&ctx, .{ 4, 8 }, 11, 10, 0.5);
    defer n.deinit();
    rng_mod.normalFill(11, &expected, 10, 0.5);
    try std.testing.expectEqualSlices(f32, &expected, try n.dataConst());

    // bernoulli: 1 iff the [0,1) uniform draw at (seed, i) is below p.
    var bern = try M.bernoulli(&ctx, .{ 4, 8 }, 42, 0.4);
    defer bern.deinit();
    rng_mod.uniformFill(42, &expected, 0, 1);
    for (try bern.dataConst(), expected) |got, draw| {
        try std.testing.expectEqual(@as(f32, if (draw < 0.4) 1 else 0), got);
    }
    try std.testing.expectError(error.InvalidShape, M.bernoulli(&ctx, .{ 4, 8 }, 1, 1.5));
}

test "public Tensor sliceStep strided view with exact gradient" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const V = Tensor(.{.d});
    var x = try V.fromSlice(&ctx, .{6}, &.{ 0, 1, 2, 3, 4, 5 });
    defer x.deinit();
    var stepped = try x.sliceStep(&ctx, .d, 1, 3, 2); // x[1::2] → {1, 3, 5}
    defer stepped.deinit();
    var stepped_mat = try stepped.materialize(&ctx);
    defer stepped_mat.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 3, 5 }, try stepped_mat.dataConst());
    // It is a VIEW: aliases the source buffer.
    try std.testing.expect(stepped.asRawTensor().buffer == x.asRawTensor().buffer);
    try std.testing.expectError(error.InvalidShape, x.sliceStep(&ctx, .d, 0, 4, 2)); // last lands at 6
    try std.testing.expectError(error.InvalidShape, x.sliceStep(&ctx, .d, 0, 1, 0));
    try std.testing.expectError(error.InvalidShape, x.sliceStep(&ctx, .d, 0, 0, 1));

    // Axis steps compose per-axis on higher ranks.
    const M = Tensor(.{ .row, .col });
    var m = try M.fromSlice(&ctx, .{ 2, 4 }, &.{ 0, 1, 2, 3, 4, 5, 6, 7 });
    defer m.deinit();
    var cols = try m.sliceStep(&ctx, .col, 0, 2, 3); // columns 0 and 3
    defer cols.deinit();
    var cols_mat = try cols.materialize(&ctx);
    defer cols_mat.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0, 3, 4, 7 }, try cols_mat.dataConst());

    // Gradient scatters into the stepped positions, zero elsewhere.
    var xv = try V.variableFromSlice(&ctx, .{6}, &.{ 0, 1, 2, 3, 4, 5 });
    defer xv.deinit();
    var sv = try xv.sliceStep(&ctx, .d, 1, 3, 2);
    defer sv.deinit();
    var w = try V.fromSlice(&ctx, .{3}, &.{ 10, 20, 30 });
    defer w.deinit();
    var weighted = try sv.mul(&ctx, &w);
    defer weighted.deinit();
    var loss = try weighted.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    var gx = (try xv.grad(&ctx)).?;
    defer gx.deinit();
    try expectCloseSlices(&.{ 0, 10, 0, 20, 0, 30 }, try gx.dataConst(), 0);
}

test "public Tensor diagonal diag trace" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const M = Tensor(.{ .row, .col });
    var x = try M.fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();
    // Rectangular: diagonal length is min(2, 3) = 2.
    var d = try x.diagonal(&ctx, .row, .col, .k);
    defer d.deinit();
    var d_mat = try d.materialize(&ctx);
    defer d_mat.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 5 }, try d_mat.dataConst());

    // Batched rank-3: both tags removed, diagonal appended last.
    const B = Tensor(.{ .b, .i, .j });
    var bx = try B.fromSlice(&ctx, .{ 2, 2, 2 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer bx.deinit();
    var bd = try bx.diagonal(&ctx, .i, .j, .k); // Tensor(.{ .b, .k })
    defer bd.deinit();
    var bd_mat = try bd.materialize(&ctx);
    defer bd_mat.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 4, 5, 8 }, try bd_mat.dataConst());

    // trace = sum of the diagonal; gradient is the identity scatter.
    var sq = try M.variableFromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer sq.deinit();
    try std.testing.expectError(error.ActiveExecScopeRequired, sq.trace(&ctx, .row, .col));
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        var t = try sq.trace(&ctx, .row, .col);
        defer t.deinit();
        try std.testing.expectEqual(@as(f32, 5), try t.item());
        try t.backward(&ctx);
    }
    var gsq = (try sq.grad(&ctx)).?;
    defer gsq.deinit();
    try expectCloseSlices(&.{ 1, 0, 0, 1 }, try gsq.dataConst(), 0);

    // diag embeds a vector; gradient extracts the diagonal back.
    var v = try Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ 7, 8, 9 });
    defer v.deinit();
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        var dm = try v.diag(&ctx, .{ .row, .col });
        defer dm.deinit();
        try std.testing.expectEqualSlices(f32, &.{ 7, 0, 0, 0, 8, 0, 0, 0, 9 }, try dm.dataConst());
        var w2 = try M.fromSlice(&ctx, .{ 3, 3 }, &.{ 2, 1, 1, 1, 3, 1, 1, 1, 4 });
        defer w2.deinit();
        var weighted = try dm.mul(&ctx, &w2);
        defer weighted.deinit();
        var loss = try weighted.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);
    }
    var gv = (try v.grad(&ctx)).?;
    defer gv.deinit();
    try expectCloseSlices(&.{ 2, 3, 4 }, try gv.dataConst(), 0);
}

test "public Tensor nonzero returns host indices and indexAdd accumulates" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();
    const alloc = gpa.allocator();

    const M = Tensor(.{ .row, .col });
    var x = try M.fromSlice(&ctx, .{ 2, 3 }, &.{ 0, 1.5, 0, -2, 0, std.math.nan(f32) });
    defer x.deinit();
    const hits = try x.nonzero(alloc);
    defer alloc.free(hits);
    try std.testing.expectEqualSlices(usize, &.{ 1, 3, 5 }, hits);

    // No match → empty host slice (no zero-size tensor involved).
    var zeros_t = try M.zeros(&ctx, .{ 2, 3 });
    defer zeros_t.deinit();
    const none = try zeros_t.nonzero(alloc);
    defer alloc.free(none);
    try std.testing.expectEqual(@as(usize, 0), none.len);

    // indexAdd: accumulates (duplicate indices allowed), grads to both.
    const V = Tensor(.{ .n, .d });
    var base = try V.variableFromSlice(&ctx, .{ 3, 2 }, &.{ 1, 1, 1, 1, 1, 1 });
    defer base.deinit();
    var update = try V.variableFromSlice(&ctx, .{ 3, 2 }, &.{ 10, 20, 30, 40, 50, 60 });
    defer update.deinit();
    var out = try base.indexAdd(&ctx, .n, &.{ 2, 0, 2 }, &update);
    defer out.deinit();
    try expectCloseSlices(&.{ 31, 41, 1, 1, 61, 81 }, try out.dataConst(), 0);

    var w = try V.fromSlice(&ctx, .{ 3, 2 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer w.deinit();
    var weighted = try out.mul(&ctx, &w);
    defer weighted.deinit();
    var loss = try weighted.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    var gb = (try base.grad(&ctx)).?;
    defer gb.deinit();
    try expectCloseSlices(&.{ 1, 2, 3, 4, 5, 6 }, try gb.dataConst(), 0);
    var gu = (try update.grad(&ctx)).?;
    defer gu.deinit();
    // Update rows gather their scattered position's gradient: rows 2, 0, 2.
    try expectCloseSlices(&.{ 5, 6, 1, 2, 5, 6 }, try gu.dataConst(), 0);
}

test "public Tensor prod and cumprod with torch zero-handling gradients" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const M = Tensor(.{ .row, .col });
    var x = try M.fromSlice(&ctx, .{ 2, 3 }, &.{ 2, 3, 4, 1, 0, 5 });
    defer x.deinit();
    var p = try x.prod(&ctx, .col);
    defer p.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 24, 0 }, try p.dataConst());
    var cp = try x.cumprod(&ctx, .col);
    defer cp.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 2, 6, 24, 1, 0, 0 }, try cp.dataConst());

    // prod gradients: zero-free row g·(prod/x_i); a single zero routes the
    // whole gradient to the zero slot; two zeros kill the row.
    var xv = try M.variableFromSlice(&ctx, .{ 3, 3 }, &.{ 2, 3, 4, 1, 0, 5, 0, 6, 0 });
    defer xv.deinit();
    var pv = try xv.prod(&ctx, .col);
    defer pv.deinit();
    var loss = try pv.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    var gx = (try xv.grad(&ctx)).?;
    defer gx.deinit();
    try expectCloseSlices(&.{ 12, 8, 6, 0, 5, 0, 0, 0, 0 }, try gx.dataConst(), 1e-6);

    // cumprod gradient, zero-free closed form: d/dx_i Σ_j y_j.
    var cv = try Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ 2, 3, 4 });
    defer cv.deinit();
    var cy = try cv.cumprod(&ctx, .d); // {2, 6, 24}
    defer cy.deinit();
    var closs = try cy.sumAll(&ctx);
    defer closs.deinit();
    try closs.backward(&ctx);
    var gc = (try cv.grad(&ctx)).?;
    defer gc.deinit();
    // d/dx0 = 1 + x1 + x1x2 = 16; d/dx1 = x0(1 + x2) = 10; d/dx2 = x0x1 = 6.
    try expectCloseSlices(&.{ 16, 10, 6 }, try gc.dataConst(), 1e-6);

    // cumprod gradient with a zero (exact O(n²) fallback):
    // x = {2, 0, 4}: y = {2, 0, 0}; d/dx0 = 1, d/dx1 = 2(1+4) = 10, d/dx2 = 0.
    var zv = try Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ 2, 0, 4 });
    defer zv.deinit();
    var zy = try zv.cumprod(&ctx, .d);
    defer zy.deinit();
    var zloss = try zy.sumAll(&ctx);
    defer zloss.deinit();
    try zloss.backward(&ctx);
    var gz = (try zv.grad(&ctx)).?;
    defer gz.deinit();
    try expectCloseSlices(&.{ 1, 10, 0 }, try gz.dataConst(), 1e-6);
}

test "public Tensor takeAlongAxis pairs with argsort and routes gradients" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const M = Tensor(.{ .row, .col });
    var x = try M.fromSlice(&ctx, .{ 2, 3 }, &.{ 10, 20, 30, 40, 50, 60 });
    defer x.deinit();
    // Per-row index tensors, i64 (the argmax/topK/sort index convention).
    const I = Tensor(.{ .dtype = .i64, .tags = .{ .row, .col } });
    var idx = try I.fromSlice(&ctx, .{ 2, 2 }, &.{ 2, 0, 1, 1 });
    defer idx.deinit();
    var picked = try x.takeAlongAxis(&ctx, .col, &idx);
    defer picked.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 30, 10, 50, 50 }, try picked.dataConst());
    // Out-of-range indices are loud.
    var bad = try I.fromSlice(&ctx, .{ 2, 2 }, &.{ 3, 0, 1, 1 });
    defer bad.deinit();
    try std.testing.expectError(error.IndexOutOfBounds, x.takeAlongAxis(&ctx, .col, &bad));

    // Gradient: duplicate reads accumulate into the same source slot.
    var xv = try M.variableFromSlice(&ctx, .{ 2, 3 }, &.{ 10, 20, 30, 40, 50, 60 });
    defer xv.deinit();
    var pv = try xv.takeAlongAxis(&ctx, .col, &idx);
    defer pv.deinit();
    var loss = try pv.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    var gx = (try xv.grad(&ctx)).?;
    defer gx.deinit();
    try expectCloseSlices(&.{ 1, 0, 1, 0, 2, 0 }, try gx.dataConst(), 0);
}

test "public Tensor scatterAdd accumulates and scatter overwrites deterministically" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const M = Tensor(.{ .row, .col });
    var base = try M.variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 1, 1, 1, 1, 1 });
    defer base.deinit();
    var src = try M.variableFromSlice(&ctx, .{ 2, 2 }, &.{ 10, 20, 30, 40 });
    defer src.deinit();
    var idx = try Tensor(.{ .dtype = .i64, .tags = .{ .row, .col } }).fromSlice(&ctx, .{ 2, 2 }, &.{ 0, 0, 2, 1 });
    defer idx.deinit();

    // scatter_add: row 0 gets 10+20 at col 0 (duplicates accumulate).
    var added = try base.scatterAdd(&ctx, .col, &idx, &src);
    defer added.deinit();
    try expectCloseSlices(&.{ 31, 1, 1, 1, 41, 31 }, try added.dataConst(), 0);

    // scatter: overwrite, duplicates resolve to the LAST row-major write.
    var written = try base.scatter(&ctx, .col, &idx, &src);
    defer written.deinit();
    try expectCloseSlices(&.{ 20, 1, 1, 1, 40, 30 }, try written.dataConst(), 0);

    // Gradients for scatterAdd: base identity; src gathers its slots.
    var w = try M.fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer w.deinit();
    var weighted = try added.mul(&ctx, &w);
    defer weighted.deinit();
    var loss = try weighted.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    var gb = (try base.grad(&ctx)).?;
    defer gb.deinit();
    try expectCloseSlices(&.{ 1, 2, 3, 4, 5, 6 }, try gb.dataConst(), 0);
    var gs = (try src.grad(&ctx)).?;
    defer gs.deinit();
    try expectCloseSlices(&.{ 1, 1, 6, 5 }, try gs.dataConst(), 0);

    // Gradients for scatter: base zeroed at written slots.
    var base2 = try M.variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 1, 1, 1, 1, 1 });
    defer base2.deinit();
    var src2 = try M.variableFromSlice(&ctx, .{ 2, 2 }, &.{ 10, 20, 30, 40 });
    defer src2.deinit();
    var written2 = try base2.scatter(&ctx, .col, &idx, &src2);
    defer written2.deinit();
    var weighted2 = try written2.mul(&ctx, &w);
    defer weighted2.deinit();
    var loss2 = try weighted2.sumAll(&ctx);
    defer loss2.deinit();
    try loss2.backward(&ctx);
    var gb2 = (try base2.grad(&ctx)).?;
    defer gb2.deinit();
    try expectCloseSlices(&.{ 0, 2, 3, 4, 0, 0 }, try gb2.dataConst(), 0);
    var gs2 = (try src2.grad(&ctx)).?;
    defer gs2.deinit();
    // torch formula: every writer reads its slot's gradient (dups share).
    try expectCloseSlices(&.{ 1, 1, 6, 5 }, try gs2.dataConst(), 0);
}


test "public Tensor scan kernels match the serial reference under either -Dvector-scan" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();
    const build_options = @import("build_options");
    const rng_mod = @import("../rng.zig");

    // 19 columns: crosses the 8-lane register-scan boundary + scalar tail.
    const rows = 5;
    const cols = 19;
    var values: [rows * cols]f32 = undefined;
    rng_mod.uniformFill(3, &values, 0.5, 1.5);

    const M = Tensor(.{ .row, .col });
    var x = try M.fromSlice(&ctx, .{ rows, cols }, &values);
    defer x.deinit();

    // Serial references computed in-test.
    var ref_sum: [rows * cols]f32 = undefined;
    var ref_prod: [rows * cols]f32 = undefined;
    var ref_sum_rows: [rows * cols]f32 = undefined; // scan along .row (non-last axis)
    for (0..rows) |r| {
        var acc_s: f32 = 0;
        var acc_p: f32 = 1;
        for (0..cols) |c| {
            acc_s += values[r * cols + c];
            ref_sum[r * cols + c] = acc_s;
            acc_p *= values[r * cols + c];
            ref_prod[r * cols + c] = acc_p;
        }
    }
    for (0..cols) |c| {
        var acc: f32 = 0;
        for (0..rows) |r| {
            acc += values[r * cols + c];
            ref_sum_rows[r * cols + c] = acc;
        }
    }

    var cs = try x.cumsum(&ctx, .col);
    defer cs.deinit();
    var cp = try x.cumprod(&ctx, .col);
    defer cp.deinit();
    var cr = try x.cumsum(&ctx, .row);
    defer cr.deinit();

    if (build_options.vector_scan) {
        // Last-axis register scan reassociates: last-ulp class only.
        for (try cs.dataConst(), ref_sum) |got, want| {
            try std.testing.expectApproxEqRel(want, got, 1e-6);
        }
        for (try cp.dataConst(), ref_prod) |got, want| {
            try std.testing.expectApproxEqRel(want, got, 1e-6);
        }
    } else {
        try std.testing.expectEqualSlices(f32, &ref_sum, try cs.dataConst());
        try std.testing.expectEqualSlices(f32, &ref_prod, try cp.dataConst());
    }
    // Non-last-axis strips are independent lanes: bitwise identical to
    // serial under BOTH configs.
    try std.testing.expectEqualSlices(f32, &ref_sum_rows, try cr.dataConst());

    // The reverse (suffix) scan — cumsum's VJP — under the same gating:
    // d(sum of w·cumsum(x)) / dx_i = Σ_{j >= i} w_j.
    var xv = try Tensor(.{.d}).variableFromSlice(&ctx, .{cols}, values[0..cols]);
    defer xv.deinit();
    var w: [cols]f32 = undefined;
    rng_mod.uniformFill(9, &w, 0.5, 1.5);
    var wt = try Tensor(.{.d}).fromSlice(&ctx, .{cols}, &w);
    defer wt.deinit();
    var y = try xv.cumsum(&ctx, .d);
    defer y.deinit();
    var weighted = try y.mul(&ctx, &wt);
    defer weighted.deinit();
    var loss = try weighted.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    var gx = (try xv.grad(&ctx)).?;
    defer gx.deinit();
    var ref_grad: [cols]f32 = undefined;
    var suffix: f32 = 0;
    var i: usize = cols;
    while (i > 0) {
        i -= 1;
        suffix += w[i];
        ref_grad[i] = suffix;
    }
    if (build_options.vector_scan) {
        for (try gx.dataConst(), ref_grad) |got, want| {
            try std.testing.expectApproxEqRel(want, got, 1e-6);
        }
    } else {
        try std.testing.expectEqualSlices(f32, &ref_grad, try gx.dataConst());
    }
}


test "16-bit variables receive f32 gradients through dot and einsum" {
    @setEvalBranchQuota(1_000_000);
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .batch, .in }).fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();
    var w32 = try Tensor(.{ .out, .in }).fromSlice(&ctx, .{ 2, 2 }, &.{ 0.5, -1.5, 2, 3 });
    defer w32.deinit();

    inline for (.{ DType.f16, DType.bf16 }) |float_dtype| {
        var w_source = try w32.to(&ctx, float_dtype);
        defer w_source.deinit();
        var w = try @TypeOf(w_source).variable(&ctx, try w_source.asRawTensor().cloneView());
        defer w.deinit();
        try std.testing.expect(w.requiresGrad());

        var y = try x.dot(&ctx, &w, .in);
        defer y.deinit();
        try std.testing.expect(y.requiresGrad());
        var loss = try y.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);

        // gy = 1 everywhere, so dW[o, i] = sum_b x[b, i] regardless of the
        // 16-bit forward rounding — exact in f32.
        var wg = (try w.grad(&ctx)).?;
        defer wg.deinit();
        comptime std.debug.assert(@TypeOf(wg).dtype == .f32);
        try std.testing.expectEqualSlices(f32, &.{ 4, 6, 4, 6 }, try wg.dataConst());

        w.zeroGrad();
        try std.testing.expect((try w.grad(&ctx)) == null);

        var y2 = try x.einsum(&ctx, &w, .{ .batch, .out });
        defer y2.deinit();
        var loss2 = try y2.sumAll(&ctx);
        defer loss2.deinit();
        try loss2.backward(&ctx);
        var wg2 = (try w.grad(&ctx)).?;
        defer wg2.deinit();
        try std.testing.expectEqualSlices(f32, &.{ 4, 6, 4, 6 }, try wg2.dataConst());
    }
}

test "f32 master weights train through the differentiable narrow" {
    @setEvalBranchQuota(1_000_000);
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .batch, .in }).fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();

    inline for (.{ DType.f16, DType.bf16 }) |float_dtype| {
        var w32 = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 2, 2 }, &.{ 0.5, -1.5, 2, 3 });
        defer w32.deinit();

        var h = try w32.to(&ctx, float_dtype);
        defer h.deinit();
        try std.testing.expect(h.requiresGrad());

        var y = try x.dot(&ctx, &h, .in);
        defer y.deinit();
        var loss = try y.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);

        // The cast backward is the identity on the f32 gradient.
        var wg = (try w32.grad(&ctx)).?;
        defer wg.deinit();
        try std.testing.expectEqualSlices(f32, &.{ 4, 6, 4, 6 }, try wg.dataConst());
    }
}

test "grad-carrying narrow is scope-owned inside an exec scope" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .batch, .in }).fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();
    var w32 = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 2, 2 }, &.{ 0.5, -1.5, 2, 3 });
    defer w32.deinit();

    const mark = ctx.openExecScope();
    var h = try w32.to(&ctx, .bf16);
    defer h.deinit(); // borrow: no-op, the scope owns value + node
    try std.testing.expect(h.scope_owned);
    var y = try x.dot(&ctx, &h, .in);
    defer y.deinit();
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    ctx.closeExecScope(mark);

    var wg = (try w32.grad(&ctx)).?;
    defer wg.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 4, 6, 4, 6 }, try wg.dataConst());
}

test "typed forward ops reject grad-requiring operands" {
    @setEvalBranchQuota(1_000_000);
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const W = Tensor(.{ .dtype = .bf16, .tags = .{ .batch, .d } });
    var w = try W.variableFromSlice(&ctx, .{ 2, 2 }, &.{
        dtype_mod.f32ToBf16(1),
        dtype_mod.f32ToBf16(2),
        dtype_mod.f32ToBf16(3),
        dtype_mod.f32ToBf16(4),
    });
    defer w.deinit();

    try std.testing.expectError(error.UnsupportedGradient, w.gelu(&ctx));
    try std.testing.expectError(error.UnsupportedGradient, w.add(&ctx, &w));
    try std.testing.expectError(error.UnsupportedGradient, w.flatten(&ctx, .flat));
    try std.testing.expectError(error.UnsupportedGradient, w.sum(&ctx, .d));

    // The detached view is a constant again: the whole forward set works.
    var frozen = try w.detach(&ctx);
    defer frozen.deinit();
    try std.testing.expect(!frozen.requiresGrad());
    var activated = try frozen.gelu(&ctx);
    defer activated.deinit();
}
