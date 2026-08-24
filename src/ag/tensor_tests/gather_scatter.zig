//! Index-driven ops: gather, setSlice/setRows, maskedSelect/maskedScatter,
//! indexSelect, nonzero/indexAdd, takeAlongAxis, and the scatter variants,
//! with scatter-add gradient routing.

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

test "tagged autograd gathers embedding rows and scatter-adds gradients" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var table = try Tensor(.{ .vocab, .d }).variable(
        &ctx,
        try ctx.fromSlice(.f32, &.{ 4, 2 }, &.{ 1, 10, 2, 20, 3, 30, 4, 40 }),
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

test "public Tensor indexSelect gathers rows by an i64 index tensor" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const V = Tensor(.{.d});
    const I = Tensor(.{ .dtype = .i64, .tags = .{.g} });
    var x = try V.fromSlice(&ctx, .{3}, &.{ 10, 20, 30 });
    defer x.deinit();
    var idx = try I.fromSlice(&ctx, .{3}, &.{ 2, 0, 2 });
    defer idx.deinit();

    // Values match the host-slice gather; the axis is retagged .g.
    var picked = try x.indexSelect(&ctx, .d, &idx, .g); // Tensor(.{ .g })
    defer picked.deinit();
    try std.testing.expectEqual(@as(usize, 3), picked.dim(.g));
    try std.testing.expectEqualSlices(f32, &.{ 30, 10, 30 }, try picked.dataConst());

    // Out-of-range and negative entries error (no wrapping).
    var bad = try I.fromSlice(&ctx, .{1}, &.{3});
    defer bad.deinit();
    try std.testing.expectError(error.IndexOutOfBounds, x.indexSelect(&ctx, .d, &bad, .g));
    var neg = try I.fromSlice(&ctx, .{1}, &.{-1});
    defer neg.deinit();
    try std.testing.expectError(error.IndexOutOfBounds, x.indexSelect(&ctx, .d, &neg, .g));

    // Gradient: single delegated op — no scope needed; duplicate index
    // reads accumulate their gradients (the gather scatter-add adjoint).
    var xv = try V.variableFromSlice(&ctx, .{3}, &.{ 10, 20, 30 });
    defer xv.deinit();
    {
        var y = try xv.indexSelect(&ctx, .d, &idx, .g);
        defer y.deinit();
        var w = try Tensor(.{.g}).fromSlice(&ctx, .{3}, &.{ 1, 2, 4 });
        defer w.deinit();
        var weighted = try y.mul(&ctx, &w);
        defer weighted.deinit();
        var loss = try weighted.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);
    }
    var gx = (try xv.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 2, 0, 5 }, try gx.dataConst());
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
