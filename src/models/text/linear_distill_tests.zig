const std = @import("std");
const fucina = @import("fucina");
const linear_distill = @import("linear_distill.zig");

const ExecContext = fucina.ExecContext;
const Tensor = fucina.Tensor;

fn gcLoss(ctx: *ExecContext, x: *const Tensor(.{ .row, .k }), w: *const Tensor(.{ .class, .k })) !Tensor(.{}) {
    return linear_distill.linearDistill(ctx, x, w, &.{ 2, 2, 0, 3 }, &.{ 1, 4, 0, 7 }, &.{ 0.6, 0.3, 0.9, 0.5 }, .{ .loss_scale = 0.7 });
}

test "fused linearDistill matches the composed sparse-gather route" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const rows = 5;
    const in_dim = 6;
    const classes = 9;
    var prng = std.Random.DefaultPrng.init(0xd157111);
    const random = prng.random();
    var x_data: [rows * in_dim]f32 = undefined;
    for (&x_data) |*value| value.* = random.floatNorm(f32);
    var w_data: [classes * in_dim]f32 = undefined;
    for (&w_data) |*value| value.* = random.floatNorm(f32) * 0.5;
    // Rows 1 and 4 are unsupervised; row 2 carries two entries.
    const t_rows = [_]usize{ 2, 2, 0, 3 };
    const t_classes = [_]usize{ 1, 4, 0, 7 };
    const t_probs = [_]f32{ 0.6, 0.3, 0.9, 0.5 };
    const scale: f32 = 0.7;

    // Composed reference: full logits -> logSoftmax -> flat gather ->
    // prob-weighted mean.
    var loss_ref: f32 = undefined;
    var gx_ref: [rows * in_dim]f32 = undefined;
    var gw_ref: [classes * in_dim]f32 = undefined;
    {
        var x = try Tensor(.{ .row, .k }).variableFromSlice(&ctx, .{ rows, in_dim }, &x_data);
        defer x.deinit();
        var w = try Tensor(.{ .class, .k }).variableFromSlice(&ctx, .{ classes, in_dim }, &w_data);
        defer w.deinit();
        var logits = try x.dot(&ctx, &w, .k);
        defer logits.deinit();
        var logq = try logits.logSoftmax(&ctx, .class);
        defer logq.deinit();
        var flat = try logq.flatten(&ctx, .flat);
        defer flat.deinit();
        var flat_indices: [t_rows.len]usize = undefined;
        var neg_weights: [t_rows.len]f32 = undefined;
        for (&flat_indices, &neg_weights, t_rows, t_classes, t_probs) |*idx, *nw, r, c, p| {
            idx.* = r * classes + c;
            nw.* = -p;
        }
        var picked = try flat.gather(&ctx, .flat, &flat_indices, .entry);
        defer picked.deinit();
        var weights = try Tensor(.{.entry}).fromSlice(&ctx, .{t_rows.len}, &neg_weights);
        defer weights.deinit();
        var weighted = try picked.mul(&ctx, &weights);
        defer weighted.deinit();
        var reduced = try weighted.mean(&ctx, .entry, .{});
        defer reduced.deinit();
        var loss = try reduced.scale(&ctx, scale);
        defer loss.deinit();
        try loss.backward(&ctx);
        loss_ref = try loss.item();
        var gx = (try x.grad(&ctx)).?;
        defer gx.deinit();
        @memcpy(&gx_ref, try gx.dataConst());
        var gw = (try w.grad(&ctx)).?;
        defer gw.deinit();
        @memcpy(&gw_ref, try gw.dataConst());
    }

    // Fused op: one node, selected-row logits internal to the record.
    var x = try Tensor(.{ .row, .k }).variableFromSlice(&ctx, .{ rows, in_dim }, &x_data);
    defer x.deinit();
    var w = try Tensor(.{ .class, .k }).variableFromSlice(&ctx, .{ classes, in_dim }, &w_data);
    defer w.deinit();
    var loss = try linear_distill.linearDistill(&ctx, &x, &w, &t_rows, &t_classes, &t_probs, .{ .loss_scale = scale });
    defer loss.deinit();
    try loss.backward(&ctx);
    try std.testing.expectApproxEqAbs(loss_ref, try loss.item(), 1e-6);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    for (try gx.dataConst(), gx_ref) |got, want| {
        try std.testing.expect(@abs(got - want) <= 1e-6 + 1e-4 * @abs(want));
    }
    var gw = (try w.grad(&ctx)).?;
    defer gw.deinit();
    for (try gw.dataConst(), gw_ref) |got, want| {
        try std.testing.expect(@abs(got - want) <= 1e-6 + 1e-4 * @abs(want));
    }

    // Finite-difference check of the fused VJP end to end (both operands).
    const result = try fucina.gradcheck(&ctx, gcLoss, .{ &x, &w }, .{});
    try std.testing.expect(result.checked == rows * in_dim + classes * in_dim);
}

test "fused linearDistill without gradients releases its record state" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .row, .k }).fromSlice(&ctx, .{ 2, 3 }, &.{ 0.1, -0.2, 0.3, 0.4, 0.5, -0.6 });
    defer x.deinit();
    var w = try Tensor(.{ .class, .k }).fromSlice(&ctx, .{ 4, 3 }, &.{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1 });
    defer w.deinit();
    var loss = try linear_distill.linearDistill(&ctx, &x, &w, &.{ 0, 1 }, &.{ 3, 2 }, &.{ 1, 0.5 }, .{ .reduction = .sum });
    defer loss.deinit();
    try std.testing.expect(!loss.requiresGrad());
    try std.testing.expect(std.math.isFinite(try loss.item()));
}
