//! RoPE on the public Tensor: half and interleaved modes, partial spans,
//! ggml context-shift composition, prepared tables with freq factors, and
//! the fused rmsNormMulRope paths.

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

test "tagged partial rope interleaved_tail rotates the trailing span only" {
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

    var y = try x.rope(&ctx, .seq, .d, &table, .interleaved_tail);
    defer y.deinit();

    // Position 0 is the identity; position 1 rotates the TRAILING 4 dims as
    // adjacent pairs — (3,4) by angle 1, (5,6) by angle 0.01 — with the
    // leading 2 dims untouched. Bit-identical to the hand loop
    // `a*c - b*s / a*s + b*c` over tail[2i], tail[2i+1].
    const c0 = @cos(@as(f32, 1));
    const s0 = @sin(@as(f32, 1));
    const c1 = @cos(@as(f32, 0.01));
    const s1 = @sin(@as(f32, 0.01));
    const expected_y = [_]f32{
        1, 2, 3,               4,               5,               6,
        1, 2, 3 * c0 - 4 * s0, 3 * s0 + 4 * c0, 5 * c1 - 6 * s1, 5 * s1 + 6 * c1,
    };
    try expectCloseSlices(&expected_y, y.asRawTensor().dataConst(), 1e-6);

    // The backward inverts the tail rotation and passes the head through.
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    var grad = (try x.grad(&ctx)).?;
    defer grad.deinit();
    const expected_grad = [_]f32{
        1, 1, 1,       1,       1,       1,
        1, 1, c0 + s0, c0 - s0, c1 + s1, c1 - s1,
    };
    try expectCloseSlices(&expected_grad, grad.asRawTensor().dataConst(), 1e-6);
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
