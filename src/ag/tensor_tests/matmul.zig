//! Matmul and contraction: dot and einsum with gradients, einsumMany chains,
//! BMM broadcasting, addDot, no-grad matmul wrappers, and f16/bf16 constant
//! RHS paths.

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

const util = @import("util.zig");
const expectCloseSlices = util.expectCloseSlices;

test "public addDot matches dot + add in values and gradients" {
    // Values: bit-exact on the vector accumulate kernels (k below the BLAS
    // floor); the BLAS beta=1 shape pins to a relative tolerance (any two
    // BLAS entry points may differ in the last ulp). Gradients: bit-exact on
    // both shapes — the base branch is the seed itself, and the operand
    // branches run the identical einsum contractions the composed pair runs.
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const Base = Tensor(.{ .row, .col });
    const Left = Tensor(.{ .row, .inner });
    const Right = Tensor(.{ .inner, .col });

    inline for (.{ .{ 13, 17, 9, true }, .{ 32, 32, 32, false } }) |case| {
        const m = case[0];
        const n = case[1];
        const k = case[2];
        const exact_values = case[3];

        const base_values = try allocator.alloc(f32, m * n);
        defer allocator.free(base_values);
        const left_values = try allocator.alloc(f32, m * k);
        defer allocator.free(left_values);
        const right_values = try allocator.alloc(f32, k * n);
        defer allocator.free(right_values);
        for (base_values, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 7) % 19)) - 9)) * 0.25;
        for (left_values, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 11)) - 5)) * 0.125;
        for (right_values, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 3) % 13)) - 6)) * 0.0625;

        // Composed reference: dot then add, its own variables.
        var base_ref = try Base.variableFromSlice(&ctx, .{ m, n }, base_values);
        defer base_ref.deinit();
        var left_ref = try Left.variableFromSlice(&ctx, .{ m, k }, left_values);
        defer left_ref.deinit();
        var right_ref = try Right.variableFromSlice(&ctx, .{ k, n }, right_values);
        defer right_ref.deinit();
        var product = try left_ref.dot(&ctx, &right_ref, .inner);
        defer product.deinit();
        var expected = try base_ref.add(&ctx, &product);
        defer expected.deinit();
        var expected_loss = try expected.sumAll(&ctx);
        defer expected_loss.deinit();
        try expected_loss.backward(&ctx);

        // Fused addDot on fresh variables.
        var base_f = try Base.variableFromSlice(&ctx, .{ m, n }, base_values);
        defer base_f.deinit();
        var left_f = try Left.variableFromSlice(&ctx, .{ m, k }, left_values);
        defer left_f.deinit();
        var right_f = try Right.variableFromSlice(&ctx, .{ k, n }, right_values);
        defer right_f.deinit();
        var fused = try base_f.addDot(&ctx, left_f, right_f, .inner);
        defer fused.deinit();
        var fused_loss = try fused.sumAll(&ctx);
        defer fused_loss.deinit();
        try fused_loss.backward(&ctx);

        for (expected.asRawTensor().dataConst(), fused.asRawTensor().dataConst()) |e, actual| {
            if (exact_values) {
                try std.testing.expectEqual(e, actual);
            } else {
                try std.testing.expectApproxEqRel(e, actual, 1e-5);
            }
        }

        inline for (.{ .{ &base_ref, &base_f }, .{ &left_ref, &left_f }, .{ &right_ref, &right_f } }) |pair| {
            var g_ref = (try pair[0].grad(&ctx)).?;
            defer g_ref.deinit();
            var g_fused = (try pair[1].grad(&ctx)).?;
            defer g_fused.deinit();
            try std.testing.expectEqualSlices(f32, g_ref.asRawTensor().dataConst(), g_fused.asRawTensor().dataConst());
        }
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

test "tagged autograd dot contracts by tag and propagates gradients" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try Tensor(.{ .batch, .m, .k }).variable(
        &ctx,
        try ctx.fromSlice(.f32, &.{ 2, 2, 3 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }),
    );
    defer a.deinit();
    var b = try Tensor(.{ .batch, .k, .n }).variable(
        &ctx,
        try ctx.fromSlice(.f32, &.{ 2, 3, 2 }, &.{ 1, 10, 2, 20, 3, 30, 4, 40, 5, 50, 6, 60 }),
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
        try ctx.fromSlice(.f32, &.{ 2, 2, 3 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }),
    );
    defer a.deinit();
    var b = try Tensor(.{ .n, .batch, .k }).variable(
        &ctx,
        try ctx.fromSlice(.f32, &.{ 2, 2, 3 }, &.{ 1, 2, 3, 4, 5, 6, 10, 20, 30, 40, 50, 60 }),
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
    var want_nn = try ctx.matmul(.f32, .plain, a.asRawTensor(), b.asRawTensor());
    defer want_nn.deinit();
    try std.testing.expectEqualSlices(f32, want_nn.dataConst(), got_nn.asRawTensor().dataConst());

    // matmulTransB: [2,3]·[2,3]ᵀ -> [2,2]
    var bt = try Tensor(.{ .n, .k }).fromSlice(&ctx, .{ 2, 3 }, &.{ 12, 11, 10, 9, 8, 7 });
    defer bt.deinit();
    var got_tb = try a.matmul(&ctx, bt, .trans_b, .{ .m, .n });
    defer got_tb.deinit();
    var want_tb = try ctx.matmul(.f32, .trans_b, a.asRawTensor(), bt.asRawTensor());
    defer want_tb.deinit();
    try std.testing.expectEqualSlices(f32, want_tb.dataConst(), got_tb.asRawTensor().dataConst());

    // bmm: [2,2,3]·[2,3,2] -> [2,2,2]
    var ba = try Tensor(.{ .batch, .m, .k }).fromSlice(&ctx, .{ 2, 2, 3 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
    defer ba.deinit();
    var bb = try Tensor(.{ .batch, .k, .n }).fromSlice(&ctx, .{ 2, 3, 2 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
    defer bb.deinit();
    var got_bmm = try ba.matmul(&ctx, bb, .plain, .{ .batch, .m, .n });
    defer got_bmm.deinit();
    var want_bmm = try ctx.bmm(.f32, .plain, ba.asRawTensor(), bb.asRawTensor());
    defer want_bmm.deinit();
    try std.testing.expectEqualSlices(f32, want_bmm.dataConst(), got_bmm.asRawTensor().dataConst());

    // bmmTransA: [2,3,2]ᵀ·[2,3,2] -> [2,2,2]
    var bta = try Tensor(.{ .batch, .k, .m }).fromSlice(&ctx, .{ 2, 3, 2 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
    defer bta.deinit();
    var got_ta = try bta.matmul(&ctx, bb, .trans_a, .{ .batch, .m, .n });
    defer got_ta.deinit();
    var want_ta = try ctx.bmm(.f32, .trans_a, bta.asRawTensor(), bb.asRawTensor());
    defer want_ta.deinit();
    try std.testing.expectEqualSlices(f32, want_ta.dataConst(), got_ta.asRawTensor().dataConst());

    // bmmTransB: [2,2,3]·[2,2,3]ᵀ -> [2,2,2]
    var btb = try Tensor(.{ .batch, .n, .k }).fromSlice(&ctx, .{ 2, 2, 3 }, &.{ 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 });
    defer btb.deinit();
    var got_tbb = try ba.matmul(&ctx, btb, .trans_b, .{ .batch, .m, .n });
    defer got_tbb.deinit();
    var want_tbb = try ctx.bmm(.f32, .trans_b, ba.asRawTensor(), btb.asRawTensor());
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
