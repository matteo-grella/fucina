//! Statistics and ordering: variance, standardizeAxis, max/min extremum
//! gradient routing, argmax/topK reductions and sampling helpers, topK
//! backward, and sort/argsort.

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

    var m = try x.max(&ctx, .d, .{});
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
    var mn = try x2.min(&ctx, .d, .{});
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
