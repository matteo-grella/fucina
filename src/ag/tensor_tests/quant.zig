//! Quantized weights on the public Tensor: block dtype surface, dequant and
//! row gathers, quantized and packed RHS dot dispatch, fused packed
//! splitSwiGlu/geglu/rmsNormMul paths, and the dotTernarySte STE linear.

const std = @import("std");
const backend_mod = @import("../../backend.zig");
const dtype_mod = @import("../../dtype.zig");
const exec_mod = @import("../../exec.zig");
const control = @import("../control.zig");
const core = @import("../core.zig");
const ag_tensor = @import("../tensor.zig");
const gradcheck_mod = @import("../gradcheck.zig");

const DType = dtype_mod.DType;
const ExecContext = exec_mod.ExecContext;
const GradState = core.GradState;
const Tensor = ag_tensor.Tensor;
const RawTensor = @import("../../tensor.zig").Tensor;

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

test "quant RHS dot under gradients: GPU dispatch matches the CPU kernels" {
    // The gradient-aware GPU policy: a grads-on quant dot may take the
    // backend's quant GPU route. This pins its values and LHS gradient
    // against the CPU quant kernels (the disable-scope route). Shape is
    // chosen to clear the Metal q8_0 dense gate (m >= 32, m*n*k >= 2^30);
    // on CPU-only builds both runs take the same kernels and the compare
    // is exact. Optimized-only: the shape is sized for the dispatch gate,
    // not for Debug throughput.
    if (@import("builtin").mode == .Debug) return error.SkipZigTest;
    const allocator = std.heap.smp_allocator;

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const m = 512;
    const k = 2048;
    const n = 1024;
    const X = Tensor(.{ .dtype = .f32, .tags = .{ .batch, .in } });
    const W = Tensor(.{ .dtype = .q8_0, .tags = .{ .out, .in } });

    const x_values = try allocator.alloc(f32, m * k);
    defer allocator.free(x_values);
    var rng: u32 = 0x2F6E2B1;
    for (x_values) |*v| {
        rng = rng *% 1664525 +% 1013904223;
        v.* = (@as(f32, @floatFromInt(rng >> 9)) / (1 << 23) - 0.5) * 0.1;
    }
    const blocks = try allocator.alloc(dtype_mod.BlockQ8_0, n * k / dtype_mod.q8_0_block_size);
    defer allocator.free(blocks);
    for (blocks, 0..) |*block, bi| {
        block.d = f16TestBits(0.01);
        for (&block.qs, 0..) |*q, qi| q.* = @intCast(@as(i32, @intCast((bi * 31 + qi * 7) % 255)) - 127);
    }

    const Run = struct {
        out: []f32,
        grad: []f32,
        fn free(self: *@This(), a: std.mem.Allocator) void {
            a.free(self.out);
            a.free(self.grad);
        }
    };
    var runs: [2]Run = undefined;
    var runs_built: usize = 0;
    defer for (runs[0..runs_built]) |*run| run.free(allocator);
    for (&runs, 0..) |*run, ri| {
        var disable: ?control.QuantDotGpuDisabledScope = if (ri == 1) control.disableQuantDotGpu() else null;
        defer if (disable) |*scope| scope.close();

        var x = try X.variableFromSlice(&ctx, .{ m, k }, x_values);
        defer x.deinit();
        var w = try W.fromBlocks(&ctx, .{ n, k }, blocks);
        defer w.deinit();
        var y = try x.dot(&ctx, &w, .in);
        defer y.deinit();
        const out = try allocator.dupe(f32, y.asRawTensor().dataConst());
        errdefer allocator.free(out);
        var loss = try y.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);
        var grad = (try x.grad(&ctx)) orelse return error.MissingGrad;
        defer grad.deinit();
        run.* = .{ .out = out, .grad = try allocator.dupe(f32, try grad.dataConst()) };
        runs_built = ri + 1;
    }

    const Rel = struct {
        fn max(a_values: []const f32, b_values: []const f32) f32 {
            var max_abs: f32 = 0;
            var max_ref: f32 = 0;
            for (a_values, b_values) |a, b| {
                max_abs = @max(max_abs, @abs(a - b));
                max_ref = @max(max_ref, @abs(b));
            }
            return if (max_ref == 0) max_abs else max_abs / max_ref;
        }
    };
    const out_rel = Rel.max(runs[0].out, runs[1].out);
    const grad_rel = Rel.max(runs[0].grad, runs[1].grad);
    std.debug.print("quant-dot grads-on GPU parity: out max rel {d}, grad max rel {d}\n", .{ out_rel, grad_rel });
    // Per-ELEMENT forward compare (a summed scalar cancels to near zero
    // here and amplifies kernel noise): the routes may differ by the
    // activation-quantization gap — the CPU q8 kernels quantize
    // activations, a GPU kernel may keep them f32 — the same scale the
    // widened-cache train tests document. The LHS gradient is
    // route-independent: dgrad contracts gy with the transiently widened
    // f32 weight on both runs.
    try std.testing.expect(out_rel < 5e-3);
    try std.testing.expect(grad_rel < 1e-4);
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

test "public dense packed RHS supports f32 f16 and bf16 weights" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const m = 7;
    const n = 5;
    const k = 11;
    const X = Tensor(.{ .batch, .in });
    var x_values: [m * k]f32 = undefined;
    for (&x_values, 0..) |*v, i| {
        const q: i32 = @as(i32, @intCast(i % 17)) - 8;
        v.* = @as(f32, @floatFromInt(q)) / 7;
    }
    var x = try X.fromSlice(&ctx, .{ m, k }, &x_values);
    defer x.deinit();

    var w_values: [n * k]f32 = undefined;
    for (&w_values, 0..) |*v, i| {
        const q: i32 = @as(i32, @intCast((i * 3) % 19)) - 9;
        v.* = @as(f32, @floatFromInt(q)) / 11;
    }
    var expected: [m * n]f32 = undefined;
    for (0..m) |i| for (0..n) |j| {
        var sum: f32 = 0;
        for (0..k) |p| sum += x_values[i * k + p] * w_values[j * k + p];
        expected[i * n + j] = sum;
    };

    const W32 = Tensor(.{ .out, .in });
    var w32 = try W32.fromSlice(&ctx, .{ n, k }, &w_values);
    defer w32.deinit();
    var p32 = try w32.packRhs(&ctx);
    defer p32.deinit();
    var y32 = try x.dotPacked(&ctx, &p32, .in, .out);
    defer y32.deinit();
    for (try y32.dataConst(), expected) |got, want| try std.testing.expectApproxEqAbs(want, got, 2e-5);

    var w16_values: [n * k]f16 = undefined;
    for (&w16_values, w_values) |*dst, src| dst.* = @floatCast(src);
    const W16 = Tensor(.{ .dtype = .f16, .tags = .{ .out, .in } });
    var w16 = try W16.fromSlice(&ctx, .{ n, k }, &w16_values);
    defer w16.deinit();
    var p16 = try w16.packRhs(&ctx);
    defer p16.deinit();
    var y16 = try x.dotPacked(&ctx, &p16, .in, .out);
    defer y16.deinit();
    for (try y16.dataConst(), 0..) |got, index| {
        const i = index / n;
        const j = index % n;
        var want: f32 = 0;
        for (0..k) |p| want += x_values[i * k + p] * @as(f32, @floatCast(w16_values[j * k + p]));
        try std.testing.expectApproxEqAbs(want, got, 2e-5);
    }

    var wb_values: [n * k]u16 = undefined;
    for (&wb_values, w_values) |*dst, src| dst.* = dtype_mod.f32ToBf16(src);
    const WB = Tensor(.{ .dtype = .bf16, .tags = .{ .out, .in } });
    var wb = try WB.fromSlice(&ctx, .{ n, k }, &wb_values);
    defer wb.deinit();
    var pb = try wb.packRhs(&ctx);
    defer pb.deinit();
    var yb = try x.dotPacked(&ctx, &pb, .in, .out);
    defer yb.deinit();
    for (try yb.dataConst(), 0..) |got, index| {
        const i = index / n;
        const j = index % n;
        var want: f32 = 0;
        for (0..k) |p| want += x_values[i * k + p] * dtype_mod.bf16ToF32(wb_values[j * k + p]);
        try std.testing.expectApproxEqAbs(want, got, 2e-5);
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

    // m=1: the decode reroute (fused f32 row + plain-lhs x4 kernel);
    // m=13: LHS quantization scratch fits the 512-block stack fast path;
    // m=1027: 257 row groups x 2 blocks/row = 514 blocks, crossing into the
    // pooled ScratchLease fallback.
    for ([_]usize{ 1, 13, 1027 }) |m| {
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

fn f16TestBits(value: f32) u16 {
    const h: f16 = @floatCast(value);
    return @bitCast(h);
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

    var rhs = try backend_mod.quantized_matmul.ternary.quantizedMatmulRhsTQ2_0FromF32Absmean(allocator, ternary_k, n, &w_values);
    defer rhs.deinit();
    var expected: [m * n]f32 = undefined;
    backend_mod.quantized_matmul.ternary.matmulTQ2_0F32RhsRange(&expected, &x_values, &rhs, m, n, 0, m);
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
    var rhs = try backend_mod.quantized_matmul.ternary.quantizedMatmulRhsTQ2_0FromF32Absmean(allocator, ternary_k, n, &w_values);
    defer rhs.deinit();
    var w_dequant: [n * ternary_k]f32 = undefined;
    for (0..n) |o| {
        try backend_mod.quantized_matmul.cold.dequantizeRowTQ2_0Into(w_dequant[o * ternary_k ..][0..ternary_k], rhs.columnBlocks(o));
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
    var rhs = try backend_mod.quantized_matmul.ternary.quantizedMatmulRhsTQ2_0FromF32Absmean(allocator, ternary_k, n, &w_values);
    defer rhs.deinit();
    var w_dequant: [n * ternary_k]f32 = undefined;
    for (0..n) |o| {
        try backend_mod.quantized_matmul.cold.dequantizeRowTQ2_0Into(w_dequant[o * ternary_k ..][0..ternary_k], rhs.columnBlocks(o));
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

    var rhs = try backend_mod.quantized_matmul.ternary.quantizedMatmulRhsTQ2_0FromF32Absmean(allocator, ternary_k, n, &w_values);
    defer rhs.deinit();
    var expected: [m * n]f32 = undefined;
    backend_mod.quantized_matmul.ternary.matmulTQ2_0F32RhsRange(&expected, &x_values, &rhs, m, n, 0, m);

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
