//! Slicing and padding: slice/select/sliceStep, pad and the 2D pad variants,
//! diagonal/diag/trace, and diagEmbed, with exact gradients.

const std = @import("std");
const backend_mod = @import("../../backend.zig");
const dtype_mod = @import("../../dtype.zig");
const exec_mod = @import("../../exec.zig");
const control = @import("../control.zig");
const core = @import("../core.zig");
const ag_tensor = @import("../tensor.zig");

const DType = dtype_mod.DType;
const ExecContext = exec_mod.ExecContext;
const GradState = core.GradState;
const Tensor = ag_tensor.Tensor;
const RawTensor = @import("../../tensor.zig").Tensor;

const util = @import("util.zig");
const expectCloseSlices = util.expectCloseSlices;

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

test "public Tensor select removes the axis (torch.select)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const M = Tensor(.{ .row, .col });
    var x = try M.fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();

    // Row select: a contiguous zero-copy view; the .row tag is removed.
    var r = try x.select(&ctx, .row, 1); // Tensor(.{ .col })
    defer r.deinit();
    try std.testing.expectEqual(@as(usize, 3), r.dim(.col));
    try std.testing.expectEqualSlices(f32, &.{ 4, 5, 6 }, try r.dataConst());

    // Column select with a negative index (torch convention: -1 = last).
    var c = try x.select(&ctx, .col, -1); // Tensor(.{ .row })
    defer c.deinit();
    var c_mat = try c.materialize(&ctx);
    defer c_mat.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 3, 6 }, try c_mat.dataConst());

    // Rank-1 select degenerates to a scalar.
    var s = try r.select(&ctx, .col, 0); // Tensor(.{})
    defer s.deinit();
    try std.testing.expectEqual(@as(f32, 4), try s.item());

    // Out-of-range indices error on both sides.
    try std.testing.expectError(error.IndexOutOfBounds, x.select(&ctx, .row, 2));
    try std.testing.expectError(error.IndexOutOfBounds, x.select(&ctx, .row, -3));

    // Gradient: exact scatter — unselected positions receive zero.
    var xv = try M.variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer xv.deinit();
    try std.testing.expectError(error.ActiveExecScopeRequired, xv.select(&ctx, .col, 1));
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        var mid = try xv.select(&ctx, .col, 1); // Tensor(.{ .row }) = {2, 5}
        defer mid.deinit();
        var w = try Tensor(.{.row}).fromSlice(&ctx, .{2}, &.{ 10, 100 });
        defer w.deinit();
        var weighted = try mid.mul(&ctx, &w);
        defer weighted.deinit();
        var loss = try weighted.sumAll(&ctx);
        defer loss.deinit();
        try std.testing.expectEqual(@as(f32, 520), try loss.item());
        try loss.backward(&ctx);
    }
    var gx = (try xv.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0, 10, 0, 0, 100, 0 }, try gx.dataConst());
}

test "public Tensor slice composes multi-axis ranges (torch basic indexing)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const M = Tensor(.{ .row, .col });
    var x = try M.fromSlice(&ctx, .{ 3, 4 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
    defer x.deinit();

    // x[1:, 1:-1] — negative end counts from the axis end.
    var inner = try x.slice(&ctx, .{ .row = .{ .start = 1 }, .col = .{ .start = 1, .end = -1 } });
    defer inner.deinit();
    try std.testing.expectEqual(@as(usize, 2), inner.dim(.row));
    try std.testing.expectEqual(@as(usize, 2), inner.dim(.col));
    var inner_mat = try inner.materialize(&ctx);
    defer inner_mat.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 6, 7, 10, 11 }, try inner_mat.dataConst());

    // x[:, ::2] — stepped axis; x[-1:] — negative start; out-of-range clamps.
    var stepped = try x.slice(&ctx, .{ .col = .{ .step = 2 } });
    defer stepped.deinit();
    var stepped_mat = try stepped.materialize(&ctx);
    defer stepped_mat.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 3, 5, 7, 9, 11 }, try stepped_mat.dataConst());
    var last = try x.slice(&ctx, .{ .row = .{ .start = -1 } });
    defer last.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 9, 10, 11, 12 }, try last.dataConst());
    var clamped = try x.slice(&ctx, .{ .row = .{ .start = -100, .end = 100 } });
    defer clamped.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, try clamped.dataConst());

    // A typed fucina.SliceRange value works in place of the literal.
    var typed = try x.slice(&ctx, .{ .row = ag_tensor.SliceRange{ .start = 1, .end = 2 } });
    defer typed.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 5, 6, 7, 8 }, try typed.dataConst());

    // step == 0 and empty results are InvalidShape (no zero-size tensors).
    try std.testing.expectError(error.InvalidShape, x.slice(&ctx, .{ .col = .{ .step = 0 } }));
    try std.testing.expectError(error.InvalidShape, x.slice(&ctx, .{ .row = .{ .start = 2, .end = 2 } }));

    // Gradient: multi-axis slicing needs a scope; the scatter is exact.
    var xv = try M.variableFromSlice(&ctx, .{ 3, 4 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
    defer xv.deinit();
    try std.testing.expectError(
        error.ActiveExecScopeRequired,
        xv.slice(&ctx, .{ .row = .{ .start = 1 }, .col = .{ .end = 2 } }),
    );
    {
        // A single sliced axis is one narrow — no scope required.
        var single = try xv.slice(&ctx, .{ .row = .{ .start = 2 } });
        defer single.deinit();
        try std.testing.expectEqualSlices(f32, &.{ 9, 10, 11, 12 }, try single.dataConst());
    }
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        var block = try xv.slice(&ctx, .{ .row = .{ .start = 1 }, .col = .{ .end = 2, .step = 1 } });
        defer block.deinit();
        var loss = try block.sumAll(&ctx); // rows 1..3, cols 0..2: 5+6+9+10
        defer loss.deinit();
        try std.testing.expectEqual(@as(f32, 30), try loss.item());
        try loss.backward(&ctx);
    }
    var gx = (try xv.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0 }, try gx.dataConst());
}

test "public Tensor diagEmbed embeds batched diagonals with exact extraction gradient" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const V = Tensor(.{ .b, .d });
    var v = try V.variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer v.deinit();
    try std.testing.expectError(error.ActiveExecScopeRequired, v.diagEmbed(&ctx, .d, .{ .row, .col }));
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        var m = try v.diagEmbed(&ctx, .d, .{ .row, .col }); // Tensor(.{ .b, .row, .col })
        defer m.deinit();
        var m_mat = try m.materialize(&ctx);
        defer m_mat.deinit();
        try expectCloseSlices(&.{
            1, 0, 0, 0, 2, 0, 0, 0, 3,
            4, 0, 0, 0, 5, 0, 0, 0, 6,
        }, try m_mat.dataConst(), 0);
        // Weight the plane so the gradient shows the exact diagonal extraction.
        var w = try Tensor(.{ .row, .col }).fromSlice(&ctx, .{ 3, 3 }, &.{ 2, 9, 9, 9, 3, 9, 9, 9, 4 });
        defer w.deinit();
        var weighted = try m.mul(&ctx, &w);
        defer weighted.deinit();
        var loss = try weighted.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);
    }
    var gv = (try v.grad(&ctx)).?;
    defer gv.deinit();
    try expectCloseSlices(&.{ 2, 3, 4, 2, 3, 4 }, try gv.dataConst(), 0);

    // No-grad rank-1 use works unscoped and matches diag.
    var plain = try Tensor(.{.d}).fromSlice(&ctx, .{2}, &.{ 7, 8 });
    defer plain.deinit();
    var pm = try plain.diagEmbed(&ctx, .d, .{ .row, .col });
    defer pm.deinit();
    var pm_mat = try pm.materialize(&ctx);
    defer pm_mat.deinit();
    try expectCloseSlices(&.{ 7, 0, 0, 8 }, try pm_mat.dataConst(), 0);
}
