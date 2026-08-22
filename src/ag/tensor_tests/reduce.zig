//! Reductions by tag: broadcast-gradient reduction, permuted named-axis
//! reductions, and the masked reduction family (mask= on sum/mean/max/min,
//! empty-lane identities, finite-difference gradcheck).

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

// ---------------------------------------------------------------------------
// Masked reductions (Fortran's `mask=` on sum/mean/maxval/minval)
// ---------------------------------------------------------------------------

test "an all-true mask reproduces the unmasked reduction bitwise" {
    const alloc = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    // 3x64: a last-axis reduction long enough to exercise the SIMD row kernel,
    // which is the whole point of the scratch-gather design. Bitwise equality
    // is the property that proves the masked path did not silently change the
    // accumulation order.
    const X = Tensor(.{ .row, .col });
    const M = Tensor(.{ .dtype = .bool, .tags = .{ .row, .col } });
    var values: [3 * 64]f32 = undefined;
    for (&values, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 17)) * 0.3735 - 2.5;

    var x = try X.fromSlice(&ctx, .{ 3, 64 }, &values);
    defer x.deinit();
    var all_true = try M.fromSlice(&ctx, .{ 3, 64 }, &([_]bool{true} ** (3 * 64)));
    defer all_true.deinit();

    var plain_sum = try x.sum(&ctx, .col, .{});
    defer plain_sum.deinit();
    var masked_sum = try x.sum(&ctx, .col, .{ .mask = &all_true });
    defer masked_sum.deinit();
    try std.testing.expectEqualSlices(f32, try plain_sum.dataConst(), try masked_sum.dataConst());

    var plain_mean = try x.mean(&ctx, .col, .{});
    defer plain_mean.deinit();
    var masked_mean = try x.mean(&ctx, .col, .{ .mask = &all_true });
    defer masked_mean.deinit();
    try std.testing.expectEqualSlices(f32, try plain_mean.dataConst(), try masked_mean.dataConst());

    var plain_max = try x.max(&ctx, .col, .{});
    defer plain_max.deinit();
    var masked_max = try x.max(&ctx, .col, .{ .mask = &all_true });
    defer masked_max.deinit();
    try std.testing.expectEqualSlices(f32, try plain_max.dataConst(), try masked_max.dataConst());

    var plain_min = try x.min(&ctx, .col, .{});
    defer plain_min.deinit();
    var masked_min = try x.min(&ctx, .col, .{ .mask = &all_true });
    defer masked_min.deinit();
    try std.testing.expectEqualSlices(f32, try plain_min.dataConst(), try masked_min.dataConst());
}

test "empty opts forward to the unmasked reduction" {
    const alloc = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try Tensor(.{ .row, .col }).fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();

    var s = try x.sum(&ctx, .col, .{});
    defer s.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 3, 7 }, try s.dataConst());
    var m = try x.mean(&ctx, .col, .{});
    defer m.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1.5, 3.5 }, try m.dataConst());
    var mx = try x.max(&ctx, .col, .{});
    defer mx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 2, 4 }, try mx.dataConst());
}

test "masked reductions restrict the reduction to the selected elements" {
    const alloc = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const M = Tensor(.{ .dtype = .bool, .tags = .{ .row, .col } });
    var x = try Tensor(.{ .row, .col }).fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();
    // row 0 keeps {1, 3}, row 1 keeps {5}
    var mask = try M.fromSlice(&ctx, .{ 2, 3 }, &.{ true, false, true, false, true, false });
    defer mask.deinit();

    var s = try x.sum(&ctx, .col, .{ .mask = &mask });
    defer s.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 4, 5 }, try s.dataConst());

    // The mean divides by the SELECTED count (2 and 1), not by the axis length.
    var m = try x.mean(&ctx, .col, .{ .mask = &mask });
    defer m.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 2, 5 }, try m.dataConst());

    var mx = try x.max(&ctx, .col, .{ .mask = &mask });
    defer mx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 3, 5 }, try mx.dataConst());

    var mn = try x.min(&ctx, .col, .{ .mask = &mask });
    defer mn.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 5 }, try mn.dataConst());
}

test "masked sum is bitwise equal to the composed maskedFill reference" {
    const alloc = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const X = Tensor(.{ .row, .col });
    var values: [4 * 9]f32 = undefined;
    for (&values, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i * 7) % 23)) - 11.0;
    var x = try X.fromSlice(&ctx, .{ 4, 9 }, &values);
    defer x.deinit();

    // keep the strictly-positive entries
    var keep = try x.compare(&ctx, .gt, 0);
    defer keep.deinit();
    var drop = try keep.logicalNot(&ctx);
    defer drop.deinit();

    var fused = try x.sum(&ctx, .col, .{ .mask = &keep });
    defer fused.deinit();

    // The status-quo spelling this op replaces: zero the excluded entries into
    // a full-size temporary, then reduce it.
    var zeroed = try x.maskedFill(&ctx, &drop, 0);
    defer zeroed.deinit();
    var composed = try zeroed.sum(&ctx, .col, .{});
    defer composed.deinit();

    // Bitwise, not approximate: the fused kernel substitutes the identity for
    // an excluded element rather than skipping it, so it performs exactly the
    // composition's additions in exactly its order. That is what makes
    // replacing the composed spelling a safe migration.
    try std.testing.expectEqualSlices(f32, try composed.dataConst(), try fused.dataConst());
}

test "a lane that selects nothing takes the identity, and empty overrides it" {
    const alloc = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const M = Tensor(.{ .dtype = .bool, .tags = .{ .row, .col } });
    var x = try Tensor(.{ .row, .col }).fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();
    // row 0 selects nothing, row 1 keeps {3}
    var mask = try M.fromSlice(&ctx, .{ 2, 2 }, &.{ false, false, true, false });
    defer mask.deinit();

    // A sum has an identity, so the empty lane is 0 — no EmptySelection error.
    var s = try x.sum(&ctx, .col, .{ .mask = &mask });
    defer s.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0, 3 }, try s.dataConst());

    // A mean has none: 0/0 is NaN unless the caller supplies a sentinel.
    var m = try x.mean(&ctx, .col, .{ .mask = &mask });
    defer m.deinit();
    try std.testing.expect(std.math.isNan((try m.dataConst())[0]));
    try std.testing.expectEqual(@as(f32, 3), (try m.dataConst())[1]);

    var m_zero = try x.mean(&ctx, .col, .{ .mask = &mask, .empty = 0 });
    defer m_zero.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0, 3 }, try m_zero.dataConst());

    // maxval of an empty selection is Fortran's -HUGE (the identity seed).
    var mx = try x.max(&ctx, .col, .{ .mask = &mask });
    defer mx.deinit();
    try std.testing.expect(std.math.isNegativeInf((try mx.dataConst())[0]));
    var mx_sentinel = try x.max(&ctx, .col, .{ .mask = &mask, .empty = -1 });
    defer mx_sentinel.deinit();
    try std.testing.expectEqualSlices(f32, &.{ -1, 3 }, try mx_sentinel.dataConst());

    var s_sentinel = try x.sum(&ctx, .col, .{ .mask = &mask, .empty = -7 });
    defer s_sentinel.deinit();
    try std.testing.expectEqualSlices(f32, &.{ -7, 3 }, try s_sentinel.dataConst());
}

test "masked reductions work on a non-last axis" {
    const alloc = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const M = Tensor(.{ .dtype = .bool, .tags = .{ .row, .col } });
    var x = try Tensor(.{ .row, .col }).fromSlice(&ctx, .{ 3, 2 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();
    // col 0 keeps rows {0, 2} -> {1, 5}; col 1 keeps row {1} -> {4}
    var mask = try M.fromSlice(&ctx, .{ 3, 2 }, &.{ true, false, false, true, true, false });
    defer mask.deinit();

    var s = try x.sum(&ctx, .row, .{ .mask = &mask });
    defer s.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 6, 4 }, try s.dataConst());

    var m = try x.mean(&ctx, .row, .{ .mask = &mask });
    defer m.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 3, 4 }, try m.dataConst());

    var mx = try x.max(&ctx, .row, .{ .mask = &mask });
    defer mx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 5, 4 }, try mx.dataConst());
}

test "a float mask is read by truthiness like where and maskedFill" {
    const alloc = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try Tensor(.{ .row, .col }).fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();
    var gate = try Tensor(.{ .row, .col }).fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 0, 0, 2.5 });
    defer gate.deinit();

    var s = try x.sum(&ctx, .col, .{ .mask = &gate });
    defer s.deinit();
    // Truthiness, not weighting: 2.5 selects, it does not scale.
    try std.testing.expectEqualSlices(f32, &.{ 1, 4 }, try s.dataConst());
}

test "masked sum sends gradient only to the selected elements" {
    const alloc = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const M = Tensor(.{ .dtype = .bool, .tags = .{ .row, .col } });
    var x = try Tensor(.{ .row, .col }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();
    var mask = try M.fromSlice(&ctx, .{ 2, 3 }, &.{ true, false, true, false, true, false });
    defer mask.deinit();

    var s = try x.sum(&ctx, .col, .{ .mask = &mask });
    defer s.deinit();
    var loss = try s.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 0, 1, 0, 1, 0 }, try gx.dataConst());
}

test "masked mean divides the gradient by the selected count" {
    const alloc = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const M = Tensor(.{ .dtype = .bool, .tags = .{ .row, .col } });
    var x = try Tensor(.{ .row, .col }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();
    // row 0 selects 2 elements, row 1 selects 1
    var mask = try M.fromSlice(&ctx, .{ 2, 3 }, &.{ true, false, true, false, true, false });
    defer mask.deinit();

    var m = try x.mean(&ctx, .col, .{ .mask = &mask });
    defer m.deinit();
    var loss = try m.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0.5, 0, 0.5, 0, 1, 0 }, try gx.dataConst());
}

test "an empty masked-mean lane produces no gradient" {
    const alloc = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const M = Tensor(.{ .dtype = .bool, .tags = .{ .row, .col } });
    var x = try Tensor(.{ .row, .col }).variableFromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();
    var mask = try M.fromSlice(&ctx, .{ 2, 2 }, &.{ false, false, true, true });
    defer mask.deinit();

    var m = try x.mean(&ctx, .col, .{ .mask = &mask, .empty = 0 });
    defer m.deinit();
    var loss = try m.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    // Row 0 produced the `empty` constant, so nothing flows back into it.
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 0.5, 0.5 }, try gx.dataConst());
}

test "masked extremum routes gradient to the first selected winner and skips empty lanes" {
    const alloc = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const M = Tensor(.{ .dtype = .bool, .tags = .{ .row, .col } });
    var x = try Tensor(.{ .row, .col }).variableFromSlice(&ctx, .{ 3, 3 }, &.{
        9, 1, 5, // masked out of the 9: the winner must be 5
        2, 2, 2, // tie among selected: first selected position wins
        1, 2, 3, // selects nothing
    });
    defer x.deinit();
    var mask = try M.fromSlice(&ctx, .{ 3, 3 }, &.{
        false, true,  true,
        false, true,  true,
        false, false, false,
    });
    defer mask.deinit();

    var mx = try x.max(&ctx, .col, .{ .mask = &mask, .empty = 0 });
    defer mx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 5, 2, 0 }, try mx.dataConst());

    var loss = try mx.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{
        0, 0, 1, // the 9 is masked out; the 5 wins
        0, 1, 0, // first SELECTED position of the tie
        0, 0, 0, // empty lane: nothing participated
    }, try gx.dataConst());
}

/// The mask a gradcheck loss closes over. `gradcheck` takes a bare function
/// plus a tuple of differentiable inputs, and a bool mask is neither, so the
/// test parks it here for the loss bodies to read.
var masked_reduce_gradcheck_mask: ?*const Tensor(.{ .dtype = .bool, .tags = .{ .row, .col } }) = null;

fn maskedSumGradcheckLoss(c: *ExecContext, input: *const Tensor(.{ .row, .col })) !Tensor(.{}) {
    var s = try input.sum(c, .col, .{ .mask = masked_reduce_gradcheck_mask.? });
    defer s.deinit();
    return s.sumAll(c);
}

fn maskedMeanGradcheckLoss(c: *ExecContext, input: *const Tensor(.{ .row, .col })) !Tensor(.{}) {
    var m = try input.mean(c, .col, .{ .mask = masked_reduce_gradcheck_mask.? });
    defer m.deinit();
    return m.sumAll(c);
}

test "masked reductions gradcheck against finite differences" {
    const alloc = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const X = Tensor(.{ .row, .col });
    const M = Tensor(.{ .dtype = .bool, .tags = .{ .row, .col } });
    var mask = try M.fromSlice(&ctx, .{ 2, 4 }, &.{
        true, false, true,  true,
        true, true,  false, true,
    });
    defer mask.deinit();
    masked_reduce_gradcheck_mask = &mask;
    defer masked_reduce_gradcheck_mask = null;

    var x = try X.variableFromSlice(&ctx, .{ 2, 4 }, &.{ 0.5, -1.25, 2.0, 0.75, -0.5, 1.5, 3.25, -2.0 });
    defer x.deinit();

    // gradcheck errors on a mismatch, so reaching the assertions is the pass;
    // `checked` proves it actually walked all 8 entries rather than nothing.
    const sum_result = try gradcheck_mod.gradcheck(&ctx, maskedSumGradcheckLoss, .{&x}, .{});
    try std.testing.expectEqual(@as(usize, 8), sum_result.checked);
    const mean_result = try gradcheck_mod.gradcheck(&ctx, maskedMeanGradcheckLoss, .{&x}, .{});
    try std.testing.expectEqual(@as(usize, 8), mean_result.checked);
}
