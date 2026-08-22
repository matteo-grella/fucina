//! Cumulative and scan ops: cumsum, prod/cumprod, the scan kernels under
//! either -Dvector-scan setting, and linearRecurrence with its reverse-scan
//! VJP.

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

test "public Tensor scan kernels match the serial reference under either -Dvector-scan" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();
    const build_options = @import("build_options");
    const rng_mod = @import("../../rng.zig");

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

test "public Tensor linearRecurrence scans lanes with tag-broadcast decay" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    // Scalar decay over a rank-1 time axis: h_t = 0.5·h_{t-1} + 1.
    const V = Tensor(.{.t});
    var ones_b = try V.fromSlice(&ctx, .{4}, &.{ 1, 1, 1, 1 });
    defer ones_b.deinit();
    var half = try Tensor(.{}).fromSlice(&ctx, .{1}, &.{0.5});
    defer half.deinit();
    var geo = try ones_b.linearRecurrence(&ctx, .t, &half, .{});
    defer geo.deinit();
    try expectCloseSlices(&.{ 1, 1.5, 1.75, 1.875 }, try geo.dataConst(), 0);

    // decay == 1 is cumsum, bitwise (same serial add order).
    var vals = try V.fromSlice(&ctx, .{5}, &.{ 0.1, 2.3, -1.7, 0.9, 4.2 });
    defer vals.deinit();
    var one = try Tensor(.{}).fromSlice(&ctx, .{1}, &.{1});
    defer one.deinit();
    var lr = try vals.linearRecurrence(&ctx, .t, &one, .{});
    defer lr.deinit();
    var cs = try vals.cumsum(&ctx, .t);
    defer cs.deinit();
    try std.testing.expectEqualSlices(f32, try cs.dataConst(), try lr.dataConst());

    // Rank-3 [t, h, d] with a [t, h] decay (broadcast over d): manual
    // serial reference per lane.
    const T = 4;
    const H = 2;
    const D = 3;
    var b_vals: [T * H * D]f32 = undefined;
    for (&b_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 7)) * 0.25 - 0.5;
    var a_vals: [T * H]f32 = undefined;
    for (&a_vals, 0..) |*v, i| v.* = 0.1 + @as(f32, @floatFromInt(i % 5)) * 0.2;
    var b3 = try Tensor(.{ .t, .h, .d }).fromSlice(&ctx, .{ T, H, D }, &b_vals);
    defer b3.deinit();
    var a2 = try Tensor(.{ .t, .h }).fromSlice(&ctx, .{ T, H }, &a_vals);
    defer a2.deinit();
    var h3 = try b3.linearRecurrence(&ctx, .t, &a2, .{});
    defer h3.deinit();
    var expected: [T * H * D]f32 = undefined;
    for (0..H) |hi| {
        for (0..D) |di| {
            var acc: f32 = 0;
            for (0..T) |t| {
                const off = (t * H + hi) * D + di;
                acc = a_vals[t * H + hi] * acc + b_vals[off];
                expected[off] = acc;
            }
        }
    }
    try std.testing.expectEqualSlices(f32, &expected, try h3.dataConst());

    // Static per-lane decay (no time tag): [d] over [t, d].
    var b2 = try Tensor(.{ .t, .d }).fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 1, 1, 1 });
    defer b2.deinit();
    var ad = try Tensor(.{.d}).fromSlice(&ctx, .{2}, &.{ 0, 0.5 });
    defer ad.deinit();
    var hd = try b2.linearRecurrence(&ctx, .t, &ad, .{});
    defer hd.deinit();
    try expectCloseSlices(&.{ 1, 1, 1, 1.5 }, try hd.dataConst(), 0);

    // Chunked streaming: scanning the tail with `initial` = the head's
    // last step reproduces the full scan bitwise.
    var head = try b3.narrow(&ctx, .t, 0, 2);
    defer head.deinit();
    var tail = try b3.narrow(&ctx, .t, 2, 2);
    defer tail.deinit();
    var a_head = try a2.narrow(&ctx, .t, 0, 2);
    defer a_head.deinit();
    var a_tail = try a2.narrow(&ctx, .t, 2, 2);
    defer a_tail.deinit();
    var h_head = try head.linearRecurrence(&ctx, .t, &a_head, .{});
    defer h_head.deinit();
    var carry = try h_head.select(&ctx, .t, 1); // [h, d] last head step
    defer carry.deinit();
    var h_tail = try tail.linearRecurrence(&ctx, .t, &a_tail, .{ .initial = &carry });
    defer h_tail.deinit();
    var h_full_tail = try h3.narrow(&ctx, .t, 2, 2);
    defer h_full_tail.deinit();
    var h_full_tail_mat = try h_full_tail.materialize(&ctx);
    defer h_full_tail_mat.deinit();
    try std.testing.expectEqualSlices(f32, try h_full_tail_mat.dataConst(), try h_tail.dataConst());

    // Wrong-shaped initial is a recoverable error.
    var bad_init = try Tensor(.{ .h, .d }).fromSlice(&ctx, .{ H, D + 1 }, &([_]f32{0} ** (H * (D + 1))));
    defer bad_init.deinit();
    try std.testing.expectError(error.ShapeMismatch, b3.linearRecurrence(&ctx, .t, &a2, .{ .initial = &bad_init }));
}

test "public Tensor linearRecurrence gradients follow the reverse-scan VJP" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    // Hand-computed: b = {1,2,3}, scalar decay 0.5, initial = {2}.
    // h = {2, 3, 4.5}; gh = {1.75, 1.5, 1};
    // ga = Σ gh_t·h_{t-1} = 1.75·2 + 1.5·2 + 1·3 = 9.5; ginit = 0.5·gh_0.
    var b = try Tensor(.{.t}).variableFromSlice(&ctx, .{3}, &.{ 1, 2, 3 });
    defer b.deinit();
    var a = try Tensor(.{}).variableFromSlice(&ctx, .{1}, &.{0.5});
    defer a.deinit();
    var h0 = try Tensor(.{}).variableFromSlice(&ctx, .{1}, &.{2});
    defer h0.deinit();
    var h = try b.linearRecurrence(&ctx, .t, &a, .{ .initial = &h0 });
    defer h.deinit();
    try expectCloseSlices(&.{ 2, 3, 4.5 }, try h.dataConst(), 0);
    var loss = try h.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    var gb = (try b.grad(&ctx)).?;
    defer gb.deinit();
    try expectCloseSlices(&.{ 1.75, 1.5, 1 }, try gb.dataConst(), 0);
    var ga = (try a.grad(&ctx)).?;
    defer ga.deinit();
    try expectCloseSlices(&.{9.5}, try ga.dataConst(), 0);
    var gh0 = (try h0.grad(&ctx)).?;
    defer gh0.deinit();
    try expectCloseSlices(&.{0.875}, try gh0.dataConst(), 0);
}

fn linRecGradcheckLoss(ctx: *ExecContext, b: *const Tensor(.{ .t, .d }), a: *const Tensor(.{.d}), h0: *const Tensor(.{.d})) !Tensor(.{}) {
    var h = try b.linearRecurrence(ctx, .t, a, .{ .initial = h0 });
    defer h.deinit();
    var w = try Tensor(.{ .t, .d }).fromSlice(ctx, .{ 3, 2 }, &.{ 1, -2, 3, 0.5, -1, 2 });
    defer w.deinit();
    var z = try h.mul(ctx, &w);
    defer z.deinit();
    return z.sumAll(ctx);
}

test "public Tensor linearRecurrence passes gradcheck for input decay and initial" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var b = try Tensor(.{ .t, .d }).variableFromSlice(&ctx, .{ 3, 2 }, &.{ 0.4, -1.2, 2.1, 0.3, -0.7, 1.6 });
    defer b.deinit();
    var a = try Tensor(.{.d}).variableFromSlice(&ctx, .{2}, &.{ 0.8, -0.6 });
    defer a.deinit();
    var h0 = try Tensor(.{.d}).variableFromSlice(&ctx, .{2}, &.{ 1.5, -0.9 });
    defer h0.deinit();
    const result = try gradcheck_mod.gradcheck(&ctx, linRecGradcheckLoss, .{ &b, &a, &h0 }, .{});
    try std.testing.expectEqual(@as(usize, 10), result.checked);
}

test "public Tensor linearRecurrence wide lanes match the serial reference under either -Dvector-scan" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    // 19 lanes: one full 8-wide strip + remainder on the vector-scan build;
    // pure serial otherwise — the results must be bitwise identical.
    const T = 5;
    const D = 19;
    var b_vals: [T * D]f32 = undefined;
    for (&b_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i * 7) % 11)) * 0.3 - 1.1;
    var b = try Tensor(.{ .t, .d }).variableFromSlice(&ctx, .{ T, D }, &b_vals);
    defer b.deinit();

    // Contiguous decay lanes ([d] decay) and broadcast lanes ([t] decay).
    var ad_vals: [D]f32 = undefined;
    for (&ad_vals, 0..) |*v, i| v.* = 0.05 + @as(f32, @floatFromInt(i % 4)) * 0.2;
    var ad = try Tensor(.{.d}).variableFromSlice(&ctx, .{D}, &ad_vals);
    defer ad.deinit();
    var at_vals: [T]f32 = .{ 0.9, -0.3, 0.5, 0.1, -0.8 };
    var at = try Tensor(.{.t}).fromSlice(&ctx, .{T}, &at_vals);
    defer at.deinit();
    var init_vals: [D]f32 = undefined;
    for (&init_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.1 - 0.7;
    var h0 = try Tensor(.{.d}).variableFromSlice(&ctx, .{D}, &init_vals);
    defer h0.deinit();

    var hd = try b.linearRecurrence(&ctx, .t, &ad, .{ .initial = &h0 });
    defer hd.deinit();
    var ht = try b.linearRecurrence(&ctx, .t, &at, .{});
    defer ht.deinit();

    var expected_d: [T * D]f32 = undefined;
    var expected_t: [T * D]f32 = undefined;
    for (0..D) |di| {
        var acc_d: f32 = init_vals[di];
        var acc_t: f32 = 0;
        for (0..T) |t| {
            acc_d = ad_vals[di] * acc_d + b_vals[t * D + di];
            expected_d[t * D + di] = acc_d;
            acc_t = at_vals[t] * acc_t + b_vals[t * D + di];
            expected_t[t * D + di] = acc_t;
        }
    }
    try std.testing.expectEqualSlices(f32, &expected_d, try hd.dataConst());
    try std.testing.expectEqualSlices(f32, &expected_t, try ht.dataConst());

    // Backward over the wide lanes (exercises the vectorized reverse arm):
    // gb, da, and dinitial against a manual reverse reference.
    var loss = try hd.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    var gh: [D]f32 = undefined;
    var exp_gb: [T * D]f32 = undefined;
    var exp_ga: [D]f32 = .{0} ** D;
    for (0..D) |di| {
        var acc: f32 = 0;
        var t: usize = T;
        while (t > 0) {
            t -= 1;
            acc = if (t + 1 == T) 1 else ad_vals[di] * acc + 1;
            exp_gb[t * D + di] = acc;
            const h_prev = if (t > 0) expected_d[(t - 1) * D + di] else init_vals[di];
            exp_ga[di] += acc * h_prev;
        }
        gh[di] = acc;
    }
    var gb = (try b.grad(&ctx)).?;
    defer gb.deinit();
    try std.testing.expectEqualSlices(f32, &exp_gb, try gb.dataConst());
    var ga = (try ad.grad(&ctx)).?;
    defer ga.deinit();
    // The broadcast reduction sums the per-t contributions; roundoff-level
    // reassociation is possible, so compare with a tolerance.
    for (try ga.dataConst(), exp_ga) |got, exp| try std.testing.expectApproxEqAbs(exp, got, 1e-5);
    var gh0 = (try h0.grad(&ctx)).?;
    defer gh0.deinit();
    for (try gh0.dataConst(), 0..) |got, di| try std.testing.expectApproxEqAbs(ad_vals[di] * gh[di], got, 0);
}
