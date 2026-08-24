//! Structural ops: axis insert/squeeze/split/merge, transpose and permute
//! views, concat and narrow, stack/flip/roll, reshape, and contiguous
//! materialization, with gradient routing through views.

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

    var x = try Tensor(.{ .batch, .flat }).variable(&ctx, try ctx.fromSlice(.f32, &.{ 2, 6 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }));
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

    var x = try Tensor(.{.singleton}).variable(&ctx, try ctx.fromSlice(.f32, &.{1}, &.{2}));
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

    var x = try Tensor(.{}).variable(&ctx, try ctx.scalar(.f32, 2));
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
    try std.testing.expect(c.isContiguous());
    var cc = try c.contiguous(&ctx);
    defer cc.deinit();
    try std.testing.expectEqual((try c.dataConst()).ptr, (try cc.dataConst()).ptr);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6 }, try cc.dataConst());

    // Strided view: dataConst is a loud error on the view, and contiguous
    // returns an owned copy in logical order.
    var t = try c.permuteTo(&ctx, .{ .col, .row });
    defer t.deinit();
    try std.testing.expect(!t.isContiguous());
    try std.testing.expectError(error.UnsupportedView, t.dataConst());
    var tc = try t.contiguous(&ctx);
    defer tc.deinit();
    try std.testing.expect(tc.isContiguous());
    try std.testing.expect((try tc.dataConst()).ptr != (try c.dataConst()).ptr);
    try std.testing.expectEqualSlices(f32, &.{ 1, 4, 2, 5, 3, 6 }, try tc.dataConst());

    // The predicate is uniform across dtype branches.
    var ti = try Tensor(.{ .dtype = .i64, .tags = .{ .row, .col } }).fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer ti.deinit();
    try std.testing.expect(ti.isContiguous());
    var tit = try ti.permuteTo(&ctx, .{ .col, .row });
    defer tit.deinit();
    try std.testing.expect(!tit.isContiguous());
    var th = try Tensor(.{ .dtype = .f16, .tags = .{ .row, .col } }).fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer th.deinit();
    try std.testing.expect(th.isContiguous());
    var tht = try th.permuteTo(&ctx, .{ .col, .row });
    defer tht.deinit();
    try std.testing.expect(!tht.isContiguous());

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
