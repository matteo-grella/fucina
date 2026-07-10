//! Behavioral tests for the optimizer module (`optim.zig`): routing, update
//! semantics, convergence through the public autograd facade, Newton-Schulz
//! properties, and exact checkpoint resumption. Reference-parity ("golden")
//! tests against the PyTorch/apollo_torch implementations live alongside.
const std = @import("std");
const exec_mod = @import("exec.zig");
const ag = @import("ag.zig");
const optim = @import("optim.zig");
const dtype_mod = @import("dtype.zig");
const parallel = @import("parallel.zig");

const ExecContext = exec_mod.ExecContext;
const Tensor = ag.Tensor;

fn expectCloseSlices(expected: []const f32, actual: []const f32, tolerance: f32) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |e, a| {
        try std.testing.expectApproxEqAbs(e, a, tolerance);
    }
}

test "optim sumSquares is thread-count invariant and tracks the f64 reference" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(0x5175a1e5);
    const random = prng.random();
    // Crosses the parallel threshold and ends mid-chunk with a scalar tail.
    const len = (1 << 18) + 12_347;
    const values = try allocator.alloc(f32, len);
    defer allocator.free(values);
    for (values) |*value| value.* = random.floatNorm(f32);

    var naive: f64 = 0;
    for (values) |value| naive += @as(f64, value) * value;

    const base = try optim.sumSquares(&ctx, values);
    // The chunked reduction re-associates the scalar chain; agreement stays
    // at f64-roundoff scale for f32 inputs of this size.
    try std.testing.expectApproxEqRel(naive, base, 1e-12);

    // The value must not depend on how many workers split the chunk grid
    // (task_count = 1 exercises the degenerate pooled split; the serial
    // fallback arm walks the same grid in the same order by construction).
    const saved_threads = parallel.cpuThreadCount(parallel.vector_max_threads);
    defer parallel.setMaxThreads(saved_threads);
    for ([_]usize{ 1, 2, 3, 5 }) |threads| {
        parallel.setMaxThreads(threads);
        try std.testing.expectEqual(base, try optim.sumSquares(&ctx, values));
    }
}

test "optim AdamW single step matches the PyTorch formula" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const w0 = [_]f32{ 1.0, -2.0, 0.5, 3.0 };
    var w = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 2, 2 }, &w0);
    defer w.deinit();
    // Loss = sum(w * c) so that dL/dw = c exactly.
    const c_values = [_]f32{ 0.5, -1.0, 2.0, 0.25 };
    var c = try Tensor(.{ .out, .in }).fromSlice(&ctx, .{ 2, 2 }, &c_values);
    defer c.deinit();

    const config = optim.AdamWConfig{ .lr = 0.1, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-8, .weight_decay = 0.1 };
    var opt = optim.AdamW.init(allocator, config);
    defer opt.deinit();
    try opt.addParam(&w);

    {
        var prod = try w.mul(&ctx, &c);
        defer prod.deinit();
        var loss = try prod.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);
    }
    try opt.step(&ctx);
    opt.zeroGrad();

    // Independent f64 reference of the same algorithm (t = 1):
    //   p *= 1 - lr*wd; m = (1-b1)*g; v = (1-b2)*g^2
    //   p -= (lr/(1-b1)) * m / (sqrt(v)/sqrt(1-b2) + eps)
    const data = try w.dataConst();
    for (w0, c_values, data) |p0, g, actual| {
        const lr: f64 = 0.1;
        const decayed = @as(f64, p0) * (1.0 - lr * 0.1);
        const m = (1.0 - 0.9) * @as(f64, g);
        const v = (1.0 - 0.999) * @as(f64, g) * g;
        const denom = @sqrt(v) / @sqrt(1.0 - 0.999) + 1e-8;
        const expected = decayed - (lr / (1.0 - 0.9)) * (m / denom);
        try std.testing.expectApproxEqAbs(@as(f32, @floatCast(expected)), actual, 1e-6);
    }
}

test "optim Adam single step uses coupled PyTorch weight decay" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const w0 = [_]f32{ 1.0, -2.0, 0.5, 3.0 };
    var w = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 2, 2 }, &w0);
    defer w.deinit();
    const c_values = [_]f32{ 0.5, -1.0, 2.0, 0.25 };
    var c = try Tensor(.{ .out, .in }).fromSlice(&ctx, .{ 2, 2 }, &c_values);
    defer c.deinit();

    const config = optim.AdamConfig{ .lr = 0.1, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-8, .weight_decay = 0.1 };
    var opt = optim.Adam.init(allocator, config);
    defer opt.deinit();
    try opt.addParam(&w);

    {
        var prod = try w.mul(&ctx, &c);
        defer prod.deinit();
        var loss = try prod.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);
    }
    try opt.step(&ctx);
    opt.zeroGrad();

    const data = try w.dataConst();
    for (w0, c_values, data) |p0, raw_g, actual| {
        const lr: f64 = 0.1;
        const g = @as(f64, raw_g) + 0.1 * @as(f64, p0);
        const m = (1.0 - 0.9) * g;
        const v = (1.0 - 0.999) * g * g;
        const denom = @sqrt(v) / @sqrt(1.0 - 0.999) + 1e-8;
        const expected = @as(f64, p0) - (lr / (1.0 - 0.9)) * (m / denom);
        try std.testing.expectApproxEqAbs(@as(f32, @floatCast(expected)), actual, 1e-6);
    }
}

test "optim AdamW converges on a quadratic through the facade" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var w = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, -1, 2, 0.5, -2, 1.5 });
    defer w.deinit();
    var target = try Tensor(.{ .out, .in }).fromSlice(&ctx, .{ 2, 3 }, &.{ -0.5, 0.25, 1, -1, 0.75, 0 });
    defer target.deinit();

    var opt = optim.AdamW.init(allocator, .{ .lr = 0.05, .weight_decay = 0 });
    defer opt.deinit();
    try opt.addParam(&w);

    var last_loss: f32 = std.math.inf(f32);
    for (0..300) |_| {
        var diff = try w.sub(&ctx, &target);
        defer diff.deinit();
        var sq = try diff.mul(&ctx, &diff);
        defer sq.deinit();
        var loss = try sq.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);
        try opt.step(&ctx);
        opt.zeroGrad();
        last_loss = try loss.item();
    }
    try std.testing.expect(last_loss < 1e-4);
}

test "optim Muon Newton-Schulz preserves diagonal structure and lands near unit singular values" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // For a diagonal input, every NS iterate stays diagonal, so the output's
    // diagonal entries ARE its singular values (up to sign). The tuned quintic
    // does not converge them to exactly 1; Keller documents ~(0.5, 1.5) for
    // well-scaled inputs. Signs must be preserved (O ~ U*V^T = sign(d)).
    var u = try ctx.fromSliceRank(2, .{ 2, 2 }, &.{ 0.8, 0, 0, -1.6 });
    defer u.deinit();
    var o = try optim.newtonSchulz5(&ctx, &u, 5);
    defer o.deinit();
    const od = o.dataConst();
    try std.testing.expectApproxEqAbs(@as(f32, 0), od[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), od[2], 1e-6);
    try std.testing.expect(od[0] > 0.3 and od[0] < 1.7);
    try std.testing.expect(od[3] < -0.3 and od[3] > -1.7);

    // Tall input exercises the transpose trick; output shape must match input.
    var tall = try ctx.fromSliceRank(2, .{ 3, 2 }, &.{ 0.5, 0.1, -0.2, 0.4, 0.3, -0.6 });
    defer tall.deinit();
    var ot = try optim.newtonSchulz5(&ctx, &tall, 5);
    defer ot.deinit();
    try std.testing.expectEqual(@as(usize, 3), ot.shape.at(0));
    try std.testing.expectEqual(@as(usize, 2), ot.shape.at(1));
}

test "optim Muon routes by rank and converges with its AdamW fallback" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var w = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 3, 2 }, &.{ 1, -1, 0.5, 2, -1.5, 0.25 });
    defer w.deinit();
    var b = try Tensor(.{.out}).variableFromSlice(&ctx, .{3}, &.{ 0.5, -0.5, 1 });
    defer b.deinit();
    var w_target = try Tensor(.{ .out, .in }).fromSlice(&ctx, .{ 3, 2 }, &.{ -0.25, 0.5, 1, -0.75, 0.3, -1 });
    defer w_target.deinit();
    var b_target = try Tensor(.{.out}).fromSlice(&ctx, .{3}, &.{ -0.2, 0.4, 0.1 });
    defer b_target.deinit();

    var opt = optim.Muon.init(allocator, .{
        .lr = 0.02,
        .weight_decay = 0,
        .fallback = .{ .lr = 0.05, .weight_decay = 0 },
    });
    defer opt.deinit();
    try opt.addParam(&w); // rank 2 -> Muon
    try opt.addParam(&b); // rank 1 -> AdamW fallback
    try std.testing.expectEqual(@as(usize, 1), opt.slots.items.len);
    try std.testing.expectEqual(@as(usize, 1), opt.fallback.slots.items.len);

    for (0..400) |_| {
        var wd = try w.sub(&ctx, &w_target);
        defer wd.deinit();
        var wsq = try wd.mul(&ctx, &wd);
        defer wsq.deinit();
        var wl = try wsq.sumAll(&ctx);
        defer wl.deinit();
        var bd = try b.sub(&ctx, &b_target);
        defer bd.deinit();
        var bsq = try bd.mul(&ctx, &bd);
        defer bsq.deinit();
        var bl = try bsq.sumAll(&ctx);
        defer bl.deinit();
        var loss = try wl.add(&ctx, &bl);
        defer loss.deinit();
        try loss.backward(&ctx);
        try opt.step(&ctx);
        opt.zeroGrad();
    }

    const wd_final = try w.dataConst();
    const wt_final = try w_target.dataConst();
    for (wd_final, wt_final) |actual, expected| {
        try std.testing.expectApproxEqAbs(expected, actual, 0.08);
    }
    const bd_final = try b.dataConst();
    const bt_final = try b_target.dataConst();
    for (bd_final, bt_final) |actual, expected| {
        try std.testing.expectApproxEqAbs(expected, actual, 0.05);
    }
}

test "optim Apollo converges on a quadratic (channel and mini variants)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    inline for (.{ optim.ApolloConfig{ .lr = 0.02, .rank = 2, .update_proj_gap = 50 }, blk: {
        var mini = optim.ApolloConfig.mini();
        mini.lr = 0.02;
        mini.scale = 4;
        mini.update_proj_gap = 50;
        break :blk mini;
    } }) |config| {
        var w = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 4, 3 }, &.{
            1, -1, 2, 0.5, -2, 1.5, 0.25, 0.75, -0.5, 1.25, -0.25, 0.1,
        });
        defer w.deinit();
        var b = try Tensor(.{.out}).variableFromSlice(&ctx, .{4}, &.{ 0.5, -0.5, 1, -1 });
        defer b.deinit();
        var w_target = try Tensor(.{ .out, .in }).fromSlice(&ctx, .{ 4, 3 }, &.{
            -0.5, 0.25, 1, -1, 0.75, 0, 0.6, -0.4, 0.2, -0.1, 0.9, -0.7,
        });
        defer w_target.deinit();
        var b_target = try Tensor(.{.out}).fromSlice(&ctx, .{4}, &.{ -0.2, 0.4, 0.1, 0.3 });
        defer b_target.deinit();

        var opt = optim.Apollo.init(allocator, config);
        defer opt.deinit();
        try opt.addParam(&w); // rank 2 -> APOLLO path
        try opt.addParam(&b); // rank 1 -> HF-AdamW fallback
        try std.testing.expectEqual(@as(usize, 1), opt.slots.items.len);
        try std.testing.expectEqual(@as(usize, 1), opt.fallback_slots.items.len);

        for (0..600) |_| {
            var wd = try w.sub(&ctx, &w_target);
            defer wd.deinit();
            var wsq = try wd.mul(&ctx, &wd);
            defer wsq.deinit();
            var wl = try wsq.sumAll(&ctx);
            defer wl.deinit();
            var bdiff = try b.sub(&ctx, &b_target);
            defer bdiff.deinit();
            var bsq = try bdiff.mul(&ctx, &bdiff);
            defer bsq.deinit();
            var bl = try bsq.sumAll(&ctx);
            defer bl.deinit();
            var loss = try wl.add(&ctx, &bl);
            defer loss.deinit();
            try loss.backward(&ctx);
            try opt.step(&ctx);
            opt.zeroGrad();
        }

        const w_final = try w.dataConst();
        const wt = try w_target.dataConst();
        for (w_final, wt) |actual, expected| {
            try std.testing.expectApproxEqAbs(expected, actual, 0.1);
        }
        const b_final = try b.dataConst();
        const bt = try b_target.dataConst();
        for (b_final, bt) |actual, expected| {
            try std.testing.expectApproxEqAbs(expected, actual, 0.05);
        }
    }
}

test "optim saveTensors/loadTensors roundtrip is exact" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const w_values = [_]f32{ 1.5, -2.25, 0.125, 3.75, -0.875, 2.5 };
    const b_values = [_]f32{ 0.0625, -1.125 };
    var w = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 2, 3 }, &w_values);
    defer w.deinit();
    var b = try Tensor(.{.out}).variableFromSlice(&ctx, .{2}, &b_values);
    defer b.deinit();

    var buf: [4096]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try optim.saveTensors(&writer, .{ &w, &b });
    const written = writer.buffered();

    // Scribble over the params, then restore.
    var w2 = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 0, 0, 0, 0, 0, 0 });
    defer w2.deinit();
    var b2 = try Tensor(.{.out}).variableFromSlice(&ctx, .{2}, &.{ 9, 9 });
    defer b2.deinit();
    var reader = std.Io.Reader.fixed(written);
    try optim.loadTensors(allocator, &reader, .{ &w2, &b2 });
    try std.testing.expectEqualSlices(f32, &w_values, try w2.dataConst());
    try std.testing.expectEqualSlices(f32, &b_values, try b2.dataConst());

    // Shape mismatch is rejected.
    var w3 = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 3, 2 }, &.{ 0, 0, 0, 0, 0, 0 });
    defer w3.deinit();
    var reader2 = std.Io.Reader.fixed(written);
    try std.testing.expectError(optim.OptimError.CheckpointShapeMismatch, optim.loadTensors(allocator, &reader2, .{ &w3, &b2 }));
}

test "optim saveStateDict/loadStateDict roundtrips f32, f16, and bf16 byte-exactly" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const w_values = [_]f32{ 1.5, -2.25, 0.125, 3.75, -0.875, 2.5 };
    var w = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 2, 3 }, &w_values);
    defer w.deinit();
    const h_values = [_]f16{ 0.5, -1.25, 2, -3.5 };
    var h = try Tensor(.{ .dtype = .f16, .tags = .{ .out, .in } }).fromSlice(&ctx, .{ 2, 2 }, &h_values);
    defer h.deinit();
    const g_values = [_]u16{ dtype_mod.f32ToBf16(0.5), dtype_mod.f32ToBf16(-1), dtype_mod.f32ToBf16(2) };
    var g = try Tensor(.{ .dtype = .bf16, .tags = .{.out} }).fromSlice(&ctx, .{3}, &g_values);
    defer g.deinit();

    var buf: [4096]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    const save_entries = [_]optim.NamedTensor{
        try optim.NamedTensor.of("w", &w),
        try optim.NamedTensor.of("h", &h),
        try optim.NamedTensor.of("g", &g),
    };
    try optim.saveStateDict(allocator, &writer, &save_entries);
    const written = writer.buffered();

    var w2 = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 0, 0, 0, 0, 0, 0 });
    defer w2.deinit();
    var h2 = try Tensor(.{ .dtype = .f16, .tags = .{ .out, .in } }).fromSlice(&ctx, .{ 2, 2 }, &.{ 9, 9, 9, 9 });
    defer h2.deinit();
    var g2 = try Tensor(.{ .dtype = .bf16, .tags = .{.out} }).fromSlice(&ctx, .{3}, &.{ 0, 0, 0 });
    defer g2.deinit();
    var reader = std.Io.Reader.fixed(written);
    const load_entries = [_]optim.NamedTensorMut{
        try optim.NamedTensorMut.of("w", &w2),
        try optim.NamedTensorMut.of("h", &h2),
        try optim.NamedTensorMut.of("g", &g2),
    };
    try optim.loadStateDict(allocator, &reader, &load_entries, .{});
    try std.testing.expectEqualSlices(f32, &w_values, try w2.dataConst());
    try std.testing.expectEqualSlices(f16, &h_values, try h2.dataConst());
    try std.testing.expectEqualSlices(u16, &g_values, try g2.dataConst());
}

test "optim loadStateDict matches entries by name regardless of order" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const a_values = [_]f32{ 1, 2, 3, 4 };
    const b_values = [_]f32{ -1, -2, -3 };
    const c_values = [_]f32{ 0.5, -0.5 };
    var a = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 2, 2 }, &a_values);
    defer a.deinit();
    var b = try Tensor(.{.out}).variableFromSlice(&ctx, .{3}, &b_values);
    defer b.deinit();
    var c = try Tensor(.{.out}).variableFromSlice(&ctx, .{2}, &c_values);
    defer c.deinit();

    var buf: [4096]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    const save_entries = [_]optim.NamedTensor{
        try optim.NamedTensor.of("a", &a),
        try optim.NamedTensor.of("b", &b),
        try optim.NamedTensor.of("c", &c),
    };
    try optim.saveStateDict(allocator, &writer, &save_entries);
    const written = writer.buffered();

    var a2 = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 2, 2 }, &.{ 0, 0, 0, 0 });
    defer a2.deinit();
    var b2 = try Tensor(.{.out}).variableFromSlice(&ctx, .{3}, &.{ 0, 0, 0 });
    defer b2.deinit();
    var c2 = try Tensor(.{.out}).variableFromSlice(&ctx, .{2}, &.{ 0, 0 });
    defer c2.deinit();
    var reader = std.Io.Reader.fixed(written);
    const load_entries = [_]optim.NamedTensorMut{
        try optim.NamedTensorMut.of("c", &c2),
        try optim.NamedTensorMut.of("a", &a2),
        try optim.NamedTensorMut.of("b", &b2),
    };
    try optim.loadStateDict(allocator, &reader, &load_entries, .{});
    try std.testing.expectEqualSlices(f32, &a_values, try a2.dataConst());
    try std.testing.expectEqualSlices(f32, &b_values, try b2.dataConst());
    try std.testing.expectEqualSlices(f32, &c_values, try c2.dataConst());
}

test "optim loadStateDict strict and non-strict semantics" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const a_values = [_]f32{ 1, 2, 3, 4 };
    const x_values = [_]f32{ 7, 8 };
    var a = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 2, 2 }, &a_values);
    defer a.deinit();
    var x = try Tensor(.{.out}).variableFromSlice(&ctx, .{2}, &x_values);
    defer x.deinit();

    // File contains [a, x].
    var buf: [4096]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    const both_entries = [_]optim.NamedTensor{
        try optim.NamedTensor.of("a", &a),
        try optim.NamedTensor.of("x", &x),
    };
    try optim.saveStateDict(allocator, &writer, &both_entries);
    const both = writer.buffered();

    // File contains only [a].
    var buf_a: [4096]u8 = undefined;
    var writer_a = std.Io.Writer.fixed(&buf_a);
    const a_entries = [_]optim.NamedTensor{try optim.NamedTensor.of("a", &a)};
    try optim.saveStateDict(allocator, &writer_a, &a_entries);
    const only_a = writer_a.buffered();

    var a2 = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 2, 2 }, &.{ 0, 0, 0, 0 });
    defer a2.deinit();
    var x2 = try Tensor(.{.out}).variableFromSlice(&ctx, .{2}, &.{ 0, 0 });
    defer x2.deinit();

    // Strict: a stream name not in the provided list errors.
    {
        var reader = std.Io.Reader.fixed(both);
        const entries = [_]optim.NamedTensorMut{try optim.NamedTensorMut.of("a", &a2)};
        try std.testing.expectError(optim.OptimError.CheckpointUnknownName, optim.loadStateDict(allocator, &reader, &entries, .{}));
    }
    // Strict: a provided entry missing from the stream errors.
    {
        var reader = std.Io.Reader.fixed(only_a);
        const entries = [_]optim.NamedTensorMut{
            try optim.NamedTensorMut.of("a", &a2),
            try optim.NamedTensorMut.of("x", &x2),
        };
        try std.testing.expectError(optim.OptimError.CheckpointMissingEntry, optim.loadStateDict(allocator, &reader, &entries, .{}));
    }
    // Non-strict: unknown stream entries are skipped, the rest is loaded.
    {
        var reader = std.Io.Reader.fixed(both);
        const entries = [_]optim.NamedTensorMut{try optim.NamedTensorMut.of("a", &a2)};
        try optim.loadStateDict(allocator, &reader, &entries, .{ .strict = false });
        try std.testing.expectEqualSlices(f32, &a_values, try a2.dataConst());
    }
}

test "optim state dict rejects duplicate/empty names, shape and dtype mismatches" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer a.deinit();
    var b = try Tensor(.{.out}).variableFromSlice(&ctx, .{2}, &.{ 0.5, -0.5 });
    defer b.deinit();

    var buf: [4096]u8 = undefined;
    // Duplicate name at save.
    {
        var writer = std.Io.Writer.fixed(&buf);
        const entries = [_]optim.NamedTensor{
            try optim.NamedTensor.of("x", &a),
            try optim.NamedTensor.of("x", &b),
        };
        try std.testing.expectError(optim.OptimError.CheckpointDuplicateName, optim.saveStateDict(allocator, &writer, &entries));
    }
    // Empty name at save.
    {
        var writer = std.Io.Writer.fixed(&buf);
        const entries = [_]optim.NamedTensor{try optim.NamedTensor.of("", &a)};
        try std.testing.expectError(optim.OptimError.CheckpointInvalidName, optim.saveStateDict(allocator, &writer, &entries));
    }

    var writer = std.Io.Writer.fixed(&buf);
    const save_entries = [_]optim.NamedTensor{try optim.NamedTensor.of("a", &a)};
    try optim.saveStateDict(allocator, &writer, &save_entries);
    const written = writer.buffered();

    // Shape mismatch on load.
    {
        var wrong_shape = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 3, 2 }, &.{ 0, 0, 0, 0, 0, 0 });
        defer wrong_shape.deinit();
        var reader = std.Io.Reader.fixed(written);
        const entries = [_]optim.NamedTensorMut{try optim.NamedTensorMut.of("a", &wrong_shape)};
        try std.testing.expectError(optim.OptimError.CheckpointShapeMismatch, optim.loadStateDict(allocator, &reader, &entries, .{}));
    }
    // Dtype mismatch on load (same shape, f16 destination for f32 data).
    {
        var wrong_dtype = try Tensor(.{ .dtype = .f16, .tags = .{ .out, .in } }).fromSlice(&ctx, .{ 2, 3 }, &.{ 0, 0, 0, 0, 0, 0 });
        defer wrong_dtype.deinit();
        var reader = std.Io.Reader.fixed(written);
        const entries = [_]optim.NamedTensorMut{try optim.NamedTensorMut.of("a", &wrong_dtype)};
        try std.testing.expectError(optim.OptimError.CheckpointDtypeMismatch, optim.loadStateDict(allocator, &reader, &entries, .{}));
    }
}

fn quadraticStep(ctx: *ExecContext, w: anytype, target: anytype, opt: anytype) !void {
    var diff = try w.sub(ctx, target);
    defer diff.deinit();
    var sq = try diff.mul(ctx, &diff);
    defer sq.deinit();
    var loss = try sq.sumAll(ctx);
    defer loss.deinit();
    try loss.backward(ctx);
    try opt.step(ctx);
    opt.zeroGrad();
}

test "optim checkpoint resume is bit-exact for Adam, AdamW, Muon, and Apollo" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const w0 = [_]f32{ 1, -1, 2, 0.5, -2, 1.5, 0.25, 0.75 };
    const t0 = [_]f32{ -0.5, 0.25, 1, -1, 0.75, 0, 0.6, -0.4 };

    inline for (.{ "adam", "adamw", "muon", "apollo" }) |kind| {
        var w = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 4, 2 }, &w0);
        defer w.deinit();
        var target = try Tensor(.{ .out, .in }).fromSlice(&ctx, .{ 4, 2 }, &t0);
        defer target.deinit();

        var opt = if (comptime std.mem.eql(u8, kind, "adam"))
            optim.Adam.init(allocator, .{ .lr = 0.05 })
        else if (comptime std.mem.eql(u8, kind, "adamw"))
            optim.AdamW.init(allocator, .{ .lr = 0.05 })
        else if (comptime std.mem.eql(u8, kind, "muon"))
            optim.Muon.init(allocator, .{ .lr = 0.02, .weight_decay = 0.01 })
        else
            // gap=3 puts a projection-regeneration boundary inside the resumed
            // segment, proving P is reconstructed deterministically.
            optim.Apollo.init(allocator, .{ .lr = 0.02, .rank = 2, .update_proj_gap = 3 });
        defer opt.deinit();
        try opt.addParam(&w);

        for (0..5) |_| try quadraticStep(&ctx, &w, &target, &opt);

        var buf: [16384]u8 = undefined;
        var writer = std.Io.Writer.fixed(&buf);
        try optim.saveTensors(&writer, .{&w});
        try opt.saveState(&writer);
        const snapshot = writer.buffered();

        for (0..5) |_| try quadraticStep(&ctx, &w, &target, &opt);
        var final_a: [8]f32 = undefined;
        @memcpy(&final_a, try w.dataConst());

        var reader = std.Io.Reader.fixed(snapshot);
        try optim.loadTensors(allocator, &reader, .{&w});
        try opt.loadState(&reader);
        for (0..5) |_| try quadraticStep(&ctx, &w, &target, &opt);

        try std.testing.expectEqualSlices(f32, &final_a, try w.dataConst());
    }
}

test "optim v3 named state resumes bit-exactly under permuted registration" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const w1_0 = [_]f32{ 1, -1, 2, 0.5, -2, 1.5, 0.25, 0.75 };
    const w1_t = [_]f32{ -0.5, 0.25, 1, -1, 0.75, 0, 0.6, -0.4 };
    const w2_0 = [_]f32{ 0.5, -1.5, 2, -0.25, 1, 0.75 };
    const w2_t = [_]f32{ -0.8, 0.6, -1.1, 0.9, -0.4, 0.3 };
    const b0 = [_]f32{ 0.5, -0.5, 1 };
    const b_t = [_]f32{ -0.2, 0.4, 0.1 };

    inline for (.{ "adam", "adamw", "muon", "apollo" }) |kind| {
        // w1 (4x2, tall) and w2 (2x3, wide) share a slot list in every
        // optimizer; their different shapes guarantee a positional load
        // would fail, so a passing resume proves name matching. b routes to
        // the Muon/Apollo fallback list.
        var w1 = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 4, 2 }, &w1_0);
        defer w1.deinit();
        var w2 = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 2, 3 }, &w2_0);
        defer w2.deinit();
        var b = try Tensor(.{.out}).variableFromSlice(&ctx, .{3}, &b0);
        defer b.deinit();
        var w1_target = try Tensor(.{ .out, .in }).fromSlice(&ctx, .{ 4, 2 }, &w1_t);
        defer w1_target.deinit();
        var w2_target = try Tensor(.{ .out, .in }).fromSlice(&ctx, .{ 2, 3 }, &w2_t);
        defer w2_target.deinit();
        var b_target = try Tensor(.{.out}).fromSlice(&ctx, .{3}, &b_t);
        defer b_target.deinit();

        var opt = if (comptime std.mem.eql(u8, kind, "adam"))
            optim.Adam.init(allocator, .{ .lr = 0.05 })
        else if (comptime std.mem.eql(u8, kind, "adamw"))
            optim.AdamW.init(allocator, .{ .lr = 0.05 })
        else if (comptime std.mem.eql(u8, kind, "muon"))
            optim.Muon.init(allocator, .{ .lr = 0.02, .weight_decay = 0.01 })
        else
            // gap=3 puts a projection-regeneration boundary inside the
            // resumed segment, proving P is reconstructed deterministically
            // from the per-name restored seeds.
            optim.Apollo.init(allocator, .{ .lr = 0.02, .rank = 2, .update_proj_gap = 3 });
        defer opt.deinit();
        try opt.addParamNamed(&w1, "w1");
        try opt.addParamNamed(&w2, "w2");
        try opt.addParamNamed(&b, "bias");

        for (0..4) |_| {
            try quadraticStep(&ctx, &w1, &w1_target, &opt);
            try quadraticStep(&ctx, &w2, &w2_target, &opt);
            try quadraticStep(&ctx, &b, &b_target, &opt);
        }

        var buf: [16384]u8 = undefined;
        var writer = std.Io.Writer.fixed(&buf);
        const save_entries = [_]optim.NamedTensor{
            try optim.NamedTensor.of("w1", &w1),
            try optim.NamedTensor.of("w2", &w2),
            try optim.NamedTensor.of("bias", &b),
        };
        try optim.saveStateDict(allocator, &writer, &save_entries);
        try opt.saveState(&writer);
        const snapshot = writer.buffered();

        for (0..4) |_| {
            try quadraticStep(&ctx, &w1, &w1_target, &opt);
            try quadraticStep(&ctx, &w2, &w2_target, &opt);
            try quadraticStep(&ctx, &b, &b_target, &opt);
        }
        var final_w1: [8]f32 = undefined;
        @memcpy(&final_w1, try w1.dataConst());
        var final_w2: [6]f32 = undefined;
        @memcpy(&final_w2, try w2.dataConst());
        var final_b: [3]f32 = undefined;
        @memcpy(&final_b, try b.dataConst());

        // Fresh optimizer registering the SAME names in REVERSED order; only
        // opt2 steps from here on (cross-instance duplicate registration is
        // the caller's responsibility, and opt is left idle).
        var opt2 = if (comptime std.mem.eql(u8, kind, "adam"))
            optim.Adam.init(allocator, .{ .lr = 0.05 })
        else if (comptime std.mem.eql(u8, kind, "adamw"))
            optim.AdamW.init(allocator, .{ .lr = 0.05 })
        else if (comptime std.mem.eql(u8, kind, "muon"))
            optim.Muon.init(allocator, .{ .lr = 0.02, .weight_decay = 0.01 })
        else
            optim.Apollo.init(allocator, .{ .lr = 0.02, .rank = 2, .update_proj_gap = 3 });
        defer opt2.deinit();
        try opt2.addParamNamed(&b, "bias");
        try opt2.addParamNamed(&w2, "w2");
        try opt2.addParamNamed(&w1, "w1");

        var reader = std.Io.Reader.fixed(snapshot);
        const load_entries = [_]optim.NamedTensorMut{
            try optim.NamedTensorMut.of("bias", &b),
            try optim.NamedTensorMut.of("w1", &w1),
            try optim.NamedTensorMut.of("w2", &w2),
        };
        try optim.loadStateDict(allocator, &reader, &load_entries, .{});
        try opt2.loadState(&reader);
        for (0..4) |_| {
            try quadraticStep(&ctx, &w1, &w1_target, &opt2);
            try quadraticStep(&ctx, &w2, &w2_target, &opt2);
            try quadraticStep(&ctx, &b, &b_target, &opt2);
        }

        try std.testing.expectEqualSlices(f32, &final_w1, try w1.dataConst());
        try std.testing.expectEqualSlices(f32, &final_w2, try w2.dataConst());
        try std.testing.expectEqualSlices(f32, &final_b, try b.dataConst());
    }
}

test "optim v3 save rejects name collisions; load rejects unknown and missing names" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer a.deinit();
    var b = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 2, 2 }, &.{ 5, 6, 7, 8 });
    defer b.deinit();
    var c = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 2, 2 }, &.{ 9, 10, 11, 12 });
    defer c.deinit();

    var buf: [16384]u8 = undefined;
    // An explicit name colliding with an unnamed param's auto-name is caught
    // at save time, before any byte is written.
    {
        var opt = optim.AdamW.init(allocator, .{});
        defer opt.deinit();
        try opt.addParam(&a); // auto-name "param0"
        try opt.addParamNamed(&b, "param0");
        var writer = std.Io.Writer.fixed(&buf);
        try std.testing.expectError(optim.OptimError.CheckpointDuplicateName, opt.saveState(&writer));
    }

    var writer = std.Io.Writer.fixed(&buf);
    {
        var opt = optim.AdamW.init(allocator, .{});
        defer opt.deinit();
        try opt.addParamNamed(&a, "x");
        try opt.addParamNamed(&b, "y");
        try opt.saveState(&writer);
    }
    const snapshot = writer.buffered();

    // A record name with no matching registered param errors.
    {
        var opt = optim.AdamW.init(allocator, .{});
        defer opt.deinit();
        try opt.addParamNamed(&a, "x");
        try opt.addParamNamed(&b, "z");
        var reader = std.Io.Reader.fixed(snapshot);
        try std.testing.expectError(optim.OptimError.CheckpointUnknownName, opt.loadState(&reader));
    }
    // A registered param with no record errors after the stream.
    {
        var opt = optim.AdamW.init(allocator, .{});
        defer opt.deinit();
        try opt.addParamNamed(&a, "x");
        try opt.addParamNamed(&b, "y");
        try opt.addParamNamed(&c, "w");
        var reader = std.Io.Reader.fixed(snapshot);
        try std.testing.expectError(optim.OptimError.CheckpointMissingEntry, opt.loadState(&reader));
    }
}

// ---------------------------------------------------------------------------
// Reference-parity ("golden") tests. Expected values were generated with the
// actual reference implementations on CPU float32 by
// tools/gen_optim_goldens.py (venv recipe in its header; it prints the Zig
// constants to paste below):
//   - AdamW: torch.optim.AdamW (torch 2.12).
//   - Muon: Keller Jordan's muon.py update transcribed verbatim, Newton-Schulz
//     in float32 (the reference's bfloat16 cast is a GPU-throughput choice; the
//     f32 CPU port is strictly more accurate, so the golden run matches f32).
//   - APOLLO: the official apollo_torch rank-path math with a FIXED injected
//     projection matrix, so the test compares optimizer math rather than RNG
//     streams (the projection only needs to be i.i.d. N(0, 1/rank)).
// ---------------------------------------------------------------------------

/// Per-step gradients delivered through the real autograd path:
/// loss = sum(w * c) gives dL/dw = c exactly.
fn applyGrad(ctx: *ExecContext, w: anytype, comptime tags: anytype, shape: anytype, c_values: []const f32, opt: anytype) !void {
    var c = try Tensor(tags).fromSlice(ctx, shape, c_values);
    defer c.deinit();
    var prod = try w.mul(ctx, &c);
    defer prod.deinit();
    var loss = try prod.sumAll(ctx);
    defer loss.deinit();
    try loss.backward(ctx);
    try opt.step(ctx);
    opt.zeroGrad();
}

test "optim golden: AdamW matches torch.optim.AdamW over 3 steps" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var w = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1.0, -2.0, 0.5, 3.0, -0.25, 1.5 });
    defer w.deinit();
    var opt = optim.AdamW.init(allocator, .{ .lr = 0.1, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-8, .weight_decay = 0.1 });
    defer opt.deinit();
    try opt.addParam(&w);

    const grads = [_][6]f32{
        .{ 0.5, -1.0, 2.0, 0.25, -0.75, 1.25 },
        .{ -0.3, 0.8, -1.2, 0.6, 0.1, -0.9 },
        .{ 1.1, 0.2, -0.4, -1.3, 0.7, 0.05 },
    };
    for (&grads) |*g| try applyGrad(&ctx, &w, .{ .out, .in }, .{ 2, 3 }, g, &opt);

    const expected = [_]f32{ 0.789949954, -1.84225929, 0.364398748, 2.74388266, -0.0998689979, 1.33619833 };
    try expectCloseSlices(&expected, try w.dataConst(), 2e-6);
}

test "optim golden: Muon matches the Keller reference over 3 steps (both scales)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const w0 = [_]f32{ 1.0, -2.0, 0.5, 3.0, -0.25, 1.5 };
    const grads = [_][6]f32{
        .{ 0.5, -1.0, 2.0, 0.25, -0.75, 1.25 },
        .{ -0.3, 0.8, -1.2, 0.6, 0.1, -0.9 },
        .{ 1.1, 0.2, -0.4, -1.3, 0.7, 0.05 },
    };
    const cases = .{
        .{ optim.MuonScale.spectral, [_]f32{ 0.969309151, -1.99795902, 0.490155995, 3.00016975, -0.226596415, 1.49175036 } },
        .{ optim.MuonScale.match_rms_adamw, [_]f32{ 0.990889192, -1.99856234, 0.497000635, 2.9987576, -0.24327293, 1.49702132 } },
    };
    inline for (cases) |case| {
        // 3x2 is tall, so the Newton-Schulz transpose trick is exercised.
        var w = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 3, 2 }, &w0);
        defer w.deinit();
        var opt = optim.Muon.init(allocator, .{ .lr = 0.02, .weight_decay = 0.01, .scale = case[0] });
        defer opt.deinit();
        try opt.addParam(&w);
        for (&grads) |*g| try applyGrad(&ctx, &w, .{ .out, .in }, .{ 3, 2 }, g, &opt);
        const expected: [6]f32 = case[1];
        try expectCloseSlices(&expected, try w.dataConst(), 1e-4);
    }
}

test "optim golden: APOLLO matches the apollo_torch reference (tall, wide, mini)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Tall 4x3, rank 2, channel scaling, wd=0.1.
    {
        var w = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 4, 3 }, &.{
            1.0, -2.0, 0.5, 3.0, -0.25, 1.5, 0.75, -1.25, 2.25, -0.5, 0.3, -1.8,
        });
        defer w.deinit();
        var opt = optim.Apollo.init(allocator, .{
            .lr = 0.02,
            .rank = 2,
            .weight_decay = 0.1,
            .update_proj_gap = 1000,
        });
        defer opt.deinit();
        try opt.addParam(&w);
        // Inject the fixed projection used by the golden run (tall: P is (rank, cols)).
        opt.slots.items[0].proj = try ctx.fromSliceRank(2, .{ 2, 3 }, &.{ 0.4, -0.7, 0.2, -0.3, 0.6, 0.9 });
        opt.slots.items[0].proj_chunk = 0;

        const grads = [_][12]f32{
            .{ 0.5, -1.0, 2.0, 0.25, -0.75, 1.25, -0.6, 0.45, -0.15, 0.9, -1.1, 0.7 },
            .{ -0.3, 0.8, -1.2, 0.6, 0.1, -0.9, 1.05, -0.35, 0.55, -0.85, 0.2, 0.4 },
            .{ 1.1, 0.2, -0.4, -1.3, 0.7, 0.05, -0.95, 0.65, -0.25, 0.15, -0.45, 1.35 },
        };
        for (&grads) |*g| try applyGrad(&ctx, &w, .{ .out, .in }, .{ 4, 3 }, g, &opt);

        const expected = [_]f32{
            0.981491923, -1.9768182,  0.472314239, 2.97066045,   -0.232518956, 1.47144246,
            0.767329931, -1.26503956, 2.23753452,  -0.506211758, 0.325036258,  -1.82804072,
        };
        try expectCloseSlices(&expected, try w.dataConst(), 1e-5);
    }

    // Wide 3x5, rank 2, channel scaling, wd=0.
    {
        var w = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 3, 5 }, &.{
            0.5, -1.5, 2.0, -0.25, 1.0, 0.75, -0.5, 1.25, -2.0, 0.3, -0.8, 0.6, -1.1, 0.9, -0.4,
        });
        defer w.deinit();
        var opt = optim.Apollo.init(allocator, .{
            .lr = 0.02,
            .rank = 2,
            .weight_decay = 0,
            .update_proj_gap = 1000,
        });
        defer opt.deinit();
        try opt.addParam(&w);
        // Wide: P is (rows, rank).
        opt.slots.items[0].proj = try ctx.fromSliceRank(2, .{ 3, 2 }, &.{ 0.3, -0.5, 0.8, 0.1, -0.6, 0.4 });
        opt.slots.items[0].proj_chunk = 0;

        const grads = [_][15]f32{
            .{ 0.2, -0.9, 1.3, 0.45, -0.7, -0.55, 0.85, -0.2, 0.65, -1.15, 0.95, -0.35, 0.5, -0.6, 0.1 },
            .{ -0.4, 0.7, -1.0, 0.55, 0.15, 0.8, -1.2, 0.35, -0.65, 0.25, -0.9, 1.05, -0.15, 0.45, -0.75 },
        };
        for (&grads) |*g| try applyGrad(&ctx, &w, .{ .out, .in }, .{ 3, 5 }, g, &opt);

        const expected = [_]f32{
            0.497191012,  -1.47246993, 1.92589998,  -0.277290583, 1.015185,
            0.759963334,  -0.52053839, 1.26076066,  -1.99838519,  0.324922234,
            -0.820804715, 0.602822661, -1.12926483, 0.902770698,  -0.397193015,
        };
        try expectCloseSlices(&expected, try w.dataConst(), 1e-5);
    }

    // APOLLO-Mini: rank 1, tensor scaling, scale=128.
    {
        var w = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 4, 3 }, &.{
            1.0, -2.0, 0.5, 3.0, -0.25, 1.5, 0.75, -1.25, 2.25, -0.5, 0.3, -1.8,
        });
        defer w.deinit();
        var config = optim.ApolloConfig.mini();
        config.lr = 0.005;
        config.update_proj_gap = 1000;
        var opt = optim.Apollo.init(allocator, config);
        defer opt.deinit();
        try opt.addParam(&w);
        opt.slots.items[0].proj = try ctx.fromSliceRank(2, .{ 1, 3 }, &.{ 0.7, -0.2, 0.5 });
        opt.slots.items[0].proj_chunk = 0;

        const grads = [_][12]f32{
            .{ 0.5, -1.0, 2.0, 0.25, -0.75, 1.25, -0.6, 0.45, -0.15, 0.9, -1.1, 0.7 },
            .{ -0.3, 0.8, -1.2, 0.6, 0.1, -0.9, 1.05, -0.35, 0.55, -0.85, 0.2, 0.4 },
            .{ 1.1, 0.2, -0.4, -1.3, 0.7, 0.05, -0.95, 0.65, -0.25, 0.15, -0.45, 1.35 },
        };
        for (&grads) |*g| try applyGrad(&ctx, &w, .{ .out, .in }, .{ 4, 3 }, g, &opt);

        const expected = [_]f32{
            0.958371639, -1.9791553,  0.445997328, 2.99946189,   -0.231866777, 1.46383381,
            0.770040989, -1.27703333, 2.24645305,  -0.522493541, 0.359566301,  -1.87906826,
        };
        try expectCloseSlices(&expected, try w.dataConst(), 1e-5);
    }
}

test "optim golden: AdamW tiny gradients pin the eps placement" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // With grads O(1e-5), sqrt(v)/sqrt(bc2) is comparable to eps: the wrong
    // placement (sqrt(v)+eps)/sqrt(bc2) diverges from torch by ~1e-3 here,
    // 100x the tolerance.
    var w = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 2, 2 }, &.{ 1.0, -2.0, 0.5, 3.0 });
    defer w.deinit();
    var opt = optim.AdamW.init(allocator, .{ .lr = 0.1, .weight_decay = 0 });
    defer opt.deinit();
    try opt.addParam(&w);

    const grads = [_][4]f32{
        .{ 2e-5, -1e-5, 3e-5, -4e-5 },
        .{ -1e-5, 2.5e-5, -2e-5, 1.5e-5 },
    };
    for (&grads) |*g| try applyGrad(&ctx, &w, .{ .out, .in }, .{ 2, 2 }, g, &opt);

    const expected = [_]f32{ 0.873433113, -1.94429815, 0.385586947, 3.13655877 };
    try expectCloseSlices(&expected, try w.dataConst(), 1e-5);
}

test "optim golden: Muon wide matrix exercises the non-transposed Newton-Schulz branch" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var w = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1.0, -2.0, 0.5, 3.0, -0.25, 1.5 });
    defer w.deinit();
    var opt = optim.Muon.init(allocator, .{ .lr = 0.02, .weight_decay = 0.01 });
    defer opt.deinit();
    try opt.addParam(&w);

    const grads = [_][6]f32{
        .{ 0.5, -1.0, 2.0, 0.25, -0.75, 1.25 },
        .{ -0.3, 0.8, -1.2, 0.6, 0.1, -0.9 },
        .{ 1.1, 0.2, -0.4, -1.3, 0.7, 0.05 },
    };
    for (&grads) |*g| try applyGrad(&ctx, &w, .{ .out, .in }, .{ 2, 3 }, g, &opt);

    const expected = [_]f32{ 0.980552793, -2.02351117, 0.499660254, 3.00021386, -0.242180541, 1.50000536 };
    try expectCloseSlices(&expected, try w.dataConst(), 1e-4);
}

test "optim rejects duplicate parameter registration" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var w = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer w.deinit();

    var adamw = optim.AdamW.init(allocator, .{});
    defer adamw.deinit();
    try adamw.addParam(&w);
    try std.testing.expectError(optim.OptimError.DuplicateParam, adamw.addParam(&w));

    var muon = optim.Muon.init(allocator, .{});
    defer muon.deinit();
    try muon.addParam(&w);
    try std.testing.expectError(optim.OptimError.DuplicateParam, muon.addFallbackParam(&w));

    var apollo = optim.Apollo.init(allocator, .{ .rank = 2 });
    defer apollo.deinit();
    try apollo.addFallbackParam(&w);
    try std.testing.expectError(optim.OptimError.DuplicateParam, apollo.addParam(&w));
}

test "optim golden: SGD matches torch.optim.SGD (nesterov, dampening, plain)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const w0 = [_]f32{ 1.0, -2.0, 0.5, 3.0, -0.25, 1.5 };
    const grads = [_][6]f32{
        .{ 0.5, -1.0, 2.0, 0.25, -0.75, 1.25 },
        .{ -0.3, 0.8, -1.2, 0.6, 0.1, -0.9 },
        .{ 1.1, 0.2, -0.4, -1.3, 0.7, 0.05 },
    };

    const cases = .{
        .{ optim.SgdConfig{ .lr = 0.1, .momentum = 0.9, .weight_decay = 0.05, .nesterov = true }, 3, [_]f32{ 0.662476242, -1.83479154, 0.201548651, 2.8810966, -0.145868853, 1.24946809 } },
        .{ optim.SgdConfig{ .lr = 0.1, .momentum = 0.9, .dampening = 0.1 }, 3, [_]f32{ 0.816799998, -1.88380003, 0.199200019, 2.94665003, -0.126849994, 1.31064999 } },
        .{ optim.SgdConfig{ .lr = 0.1 }, 2, [_]f32{ 0.980000019, -1.98000002, 0.420000017, 2.91499996, -0.185000002, 1.46500003 } },
    };
    inline for (cases) |case| {
        var w = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 2, 3 }, &w0);
        defer w.deinit();
        var opt = optim.SGD.init(allocator, case[0]);
        defer opt.deinit();
        try opt.addParam(&w);
        for (grads[0..case[1]]) |*g| try applyGrad(&ctx, &w, .{ .out, .in }, .{ 2, 3 }, g, &opt);
        const expected: [6]f32 = case[2];
        try expectCloseSlices(&expected, try w.dataConst(), 2e-6);
    }
}

test "optim golden: global grad clipping across an OptimizerSet matches torch clip_grad_norm_" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Two params in two optimizer instances ("param groups"); torch clipped
    // them JOINTLY, so only a correct global two-phase clip reproduces it.
    var a = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1.0, -2.0, 0.5, 3.0, -0.25, 1.5 });
    defer a.deinit();
    var b = try Tensor(.{.out}).variableFromSlice(&ctx, .{2}, &.{ 0.5, -1.0 });
    defer b.deinit();

    var opt_a = optim.SGD.init(allocator, .{ .lr = 0.1, .momentum = 0.9 });
    defer opt_a.deinit();
    try opt_a.addParam(&a);
    var opt_b = optim.SGD.init(allocator, .{ .lr = 0.1, .momentum = 0.9 });
    defer opt_b.deinit();
    try opt_b.addParam(&b);

    var set = optim.OptimizerSet.init(allocator);
    defer set.deinit();
    try set.add(&opt_a);
    try set.add(&opt_b);

    const a_grads = [_][6]f32{
        .{ 0.5, -1.0, 2.0, 0.25, -0.75, 1.25 },
        .{ -0.3, 0.8, -1.2, 0.6, 0.1, -0.9 },
    };
    const b_grads = [_][2]f32{ .{ 0.3, -0.6 }, .{ -0.2, 0.4 } };
    const expected_norms = [_]f32{ 2.8084693, 1.88414443 };

    for (a_grads, b_grads, expected_norms) |grad_a, grad_b, expected_norm| {
        var ca = try Tensor(.{ .out, .in }).fromSlice(&ctx, .{ 2, 3 }, &grad_a);
        defer ca.deinit();
        var cb = try Tensor(.{.out}).fromSlice(&ctx, .{2}, &grad_b);
        defer cb.deinit();
        var pa = try a.mul(&ctx, &ca);
        defer pa.deinit();
        var la = try pa.sumAll(&ctx);
        defer la.deinit();
        var pb = try b.mul(&ctx, &cb);
        defer pb.deinit();
        var lb = try pb.sumAll(&ctx);
        defer lb.deinit();
        var loss = try la.add(&ctx, &lb);
        defer loss.deinit();
        try loss.backward(&ctx);

        const norm = try set.clipGradNorm(&ctx, 0.5);
        try std.testing.expectApproxEqAbs(expected_norm, norm, 1e-5);
        try set.step(&ctx);
        set.zeroGrad();
    }

    const a_expected = [_]f32{ 0.991048038, -1.98740351, 0.464192212, 2.97562122, -0.227284044, 1.48160064 };
    const b_expected = [_]f32{ 0.495159566, -0.990319133 };
    try expectCloseSlices(&a_expected, try a.dataConst(), 2e-6);
    try expectCloseSlices(&b_expected, try b.dataConst(), 2e-6);
}

test "optim LrSchedule rescales attached lrs from their bases" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var muon = optim.Muon.init(allocator, .{ .lr = 0.02, .fallback = .{ .lr = 0.004 } });
    defer muon.deinit();

    var sched = optim.LrSchedule.init(allocator);
    defer sched.deinit();
    try sched.attach(&muon.config.lr);
    try sched.attach(&muon.fallback.config.lr);

    sched.apply(0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.01), muon.config.lr, 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.002), muon.fallback.config.lr, 1e-7);
    // Factors always rescale from the captured base, not the current value.
    sched.apply(1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.02), muon.config.lr, 1e-7);

    // warmupCosineFactor: linear ramp, peak at end of warmup, min at the end.
    try std.testing.expectApproxEqAbs(@as(f64, 0.1), optim.warmupCosineFactor(0, 100, 10, 0.1), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), optim.warmupCosineFactor(10, 100, 10, 0.1), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.1), optim.warmupCosineFactor(100, 100, 10, 0.1), 1e-9);
    const mid = optim.warmupCosineFactor(55, 100, 10, 0.1);
    try std.testing.expectApproxEqAbs(@as(f64, 0.55), mid, 1e-9);
}

test "optim OptimizerSet checkpoint roundtrip is exact across members" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var w = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 2, 2 }, &.{ 1, -1, 2, 0.5 });
    defer w.deinit();
    var b = try Tensor(.{.out}).variableFromSlice(&ctx, .{2}, &.{ 0.5, -0.5 });
    defer b.deinit();
    var w_target = try Tensor(.{ .out, .in }).fromSlice(&ctx, .{ 2, 2 }, &.{ 0, 0, 0, 0 });
    defer w_target.deinit();
    var b_target = try Tensor(.{.out}).fromSlice(&ctx, .{2}, &.{ 0, 0 });
    defer b_target.deinit();

    var opt_w = optim.AdamW.init(allocator, .{ .lr = 0.05 });
    defer opt_w.deinit();
    try opt_w.addParam(&w);
    var opt_b = optim.SGD.init(allocator, .{ .lr = 0.05, .momentum = 0.9 });
    defer opt_b.deinit();
    try opt_b.addParam(&b);

    var set = optim.OptimizerSet.init(allocator);
    defer set.deinit();
    try set.add(&opt_w);
    try set.add(&opt_b);

    for (0..4) |_| {
        try quadraticStep(&ctx, &w, &w_target, &set);
        try quadraticStep(&ctx, &b, &b_target, &set);
    }

    var buf: [16384]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try optim.saveTensors(&writer, .{ &w, &b });
    try set.saveState(&writer);
    const snapshot = writer.buffered();

    for (0..4) |_| {
        try quadraticStep(&ctx, &w, &w_target, &set);
        try quadraticStep(&ctx, &b, &b_target, &set);
    }
    var final_w: [4]f32 = undefined;
    @memcpy(&final_w, try w.dataConst());
    var final_b: [2]f32 = undefined;
    @memcpy(&final_b, try b.dataConst());

    var reader = std.Io.Reader.fixed(snapshot);
    try optim.loadTensors(allocator, &reader, .{ &w, &b });
    try set.loadState(&reader);
    for (0..4) |_| {
        try quadraticStep(&ctx, &w, &w_target, &set);
        try quadraticStep(&ctx, &b, &b_target, &set);
    }
    try std.testing.expectEqualSlices(f32, &final_w, try w.dataConst());
    try std.testing.expectEqualSlices(f32, &final_b, try b.dataConst());
}

test "optim loadTensors is transactional: a truncated stream leaves prior tensors unmutated" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var w = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer w.deinit();
    var b = try Tensor(.{.out}).variableFromSlice(&ctx, .{3}, &.{ 5, 6, 7 });
    defer b.deinit();

    var buf: [4096]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try optim.saveTensors(&writer, .{ &w, &b });
    // Truncate mid-second-record: drop 2 bytes of b's data so the first tensor
    // reads in full but the second's data is short.
    const truncated = writer.buffered()[0 .. writer.buffered().len - 2];

    var w2 = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 2, 2 }, &.{ 9, 9, 9, 9 });
    defer w2.deinit();
    var b2 = try Tensor(.{.out}).variableFromSlice(&ctx, .{3}, &.{ 9, 9, 9 });
    defer b2.deinit();
    var reader = std.Io.Reader.fixed(truncated);
    if (optim.loadTensors(allocator, &reader, .{ &w2, &b2 })) |_| {
        return error.TestExpectedTruncationError;
    } else |_| {}
    // Transactional: w2 (the first, fully-read tensor) is byte-unchanged.
    try std.testing.expectEqualSlices(f32, &[_]f32{ 9, 9, 9, 9 }, try w2.dataConst());
}

test "optim AdamW.loadState is transactional: a truncated record leaves prior slot unmutated" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try Tensor(.{.out}).variableFromSlice(&ctx, .{2}, &.{ 1, 2 });
    defer a.deinit();
    var b = try Tensor(.{.out}).variableFromSlice(&ctx, .{2}, &.{ 3, 4 });
    defer b.deinit();

    var opt = optim.AdamW.init(allocator, .{ .lr = 0.1 });
    defer opt.deinit();
    try opt.addParamNamed(&a, "a");
    try opt.addParamNamed(&b, "b");
    // Give both slots non-trivial state directly (slots are in registration order).
    opt.slots.items[0].step = 7;
    @memcpy(opt.slots.items[0].m.f32, &[_]f32{ 0.1, 0.2 });
    @memcpy(opt.slots.items[0].v.f32, &[_]f32{ 0.3, 0.4 });
    opt.slots.items[1].step = 9;
    @memcpy(opt.slots.items[1].m.f32, &[_]f32{ 0.5, 0.6 });
    @memcpy(opt.slots.items[1].v.f32, &[_]f32{ 0.7, 0.8 });

    var buf: [4096]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try opt.saveState(&writer);
    // Truncate mid-second-record (slot "b"): "a" reads in full, "b" is short.
    const truncated = writer.buffered()[0 .. writer.buffered().len - 2];

    // Fresh optimizer; sentinel its first slot ("a") before the load.
    var opt2 = optim.AdamW.init(allocator, .{ .lr = 0.1 });
    defer opt2.deinit();
    try opt2.addParamNamed(&a, "a");
    try opt2.addParamNamed(&b, "b");
    opt2.slots.items[0].step = 123;
    @memcpy(opt2.slots.items[0].m.f32, &[_]f32{ 8, 8 });
    @memcpy(opt2.slots.items[0].v.f32, &[_]f32{ 8, 8 });

    var reader = std.Io.Reader.fixed(truncated);
    if (opt2.loadState(&reader)) |_| {
        return error.TestExpectedTruncationError;
    } else |_| {}
    // Transactional: slot "a" (read OK before "b" truncated) is byte-unchanged.
    try std.testing.expectEqual(@as(u64, 123), opt2.slots.items[0].step);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 8, 8 }, opt2.slots.items[0].m.f32);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 8, 8 }, opt2.slots.items[0].v.f32);
}

test "optim OptimizerSet rejects the same variable registered into two member optimizers" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var shared = try Tensor(.{.out}).variableFromSlice(&ctx, .{2}, &.{ 1, 2 });
    defer shared.deinit();
    var w = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer w.deinit();

    // Group 1 (AdamW) owns `w` and `shared`.
    var g1 = optim.AdamW.init(allocator, .{ .lr = 0.1 });
    defer g1.deinit();
    try g1.addParam(&w);
    try g1.addParam(&shared);
    // The per-instance dup guard still fires (registering shared into g1 twice).
    try std.testing.expectError(optim.OptimError.DuplicateParam, g1.addParam(&shared));

    // Group 2 (SGD) also registers `shared` — legal per-instance; the
    // cross-instance check is the OptimizerSet's responsibility.
    var g2 = optim.SGD.init(allocator, .{ .lr = 0.1 });
    defer g2.deinit();
    try g2.addParam(&shared);

    var set = optim.OptimizerSet.init(allocator);
    defer set.deinit();
    try set.add(&g1); // ok: registers w + shared
    // g2 re-registers `shared`, already owned by g1 -> cross-instance duplicate.
    try std.testing.expectError(optim.OptimError.DuplicateParam, set.add(&g2));
}

// ---------------------------------------------------------------------------
// bf16 optimizer state (`StateDType`). Step math stays f32 — bf16 buffers are
// widened on read and narrowed (round-to-nearest-even) on write via
// dtype.bf16ToF32/f32ToBf16 — so each kernel is pinned EXACTLY against an
// in-test reference performing the identical widen -> f32 recurrence ->
// narrow sequence. The torch goldens cannot cover this: torch.optim keeps
// state in the PARAM dtype, so there is no f32-param/bf16-state oracle.
// Param lengths are 9-10 elements so both the 8-lane hand-vectorized kernel
// body AND the scalar tail are exercised and proven bit-identical.
// ---------------------------------------------------------------------------

const bf16ToF32 = dtype_mod.bf16ToF32;
const f32ToBf16 = dtype_mod.f32ToBf16;

test "optim bf16 state: AdamW kernel matches the widen-narrow reference exactly" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const w0 = [_]f32{ 1.0, -2.0, 0.5, 3.0, -0.25, 1.5, 0.75, -1.25, 2.25, -0.5 };
    const grads = [_][10]f32{
        .{ 0.5, -1.0, 2.0, 0.25, -0.75, 1.25, -0.6, 0.45, -0.15, 0.9 },
        .{ -0.3, 0.8, -1.2, 0.6, 0.1, -0.9, 1.05, -0.35, 0.55, -0.85 },
        .{ 1.1, 0.2, -0.4, -1.3, 0.7, 0.05, -0.95, 0.65, -0.25, 0.15 },
        .{ -0.7, 0.4, 0.9, -0.2, 0.35, -1.15, 0.8, -0.05, 0.3, -0.45 },
    };
    const lr: f32 = 0.1;
    const wd: f32 = 0.1;
    const beta1: f32 = 0.9;
    const beta2: f32 = 0.999;
    const eps: f32 = 1e-8;

    // Both bf16 arms: m-only (the recommended config) and m+v.
    inline for (.{ false, true }) |v_bf16| {
        var w = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 2, 5 }, &w0);
        defer w.deinit();
        var opt = optim.AdamW.init(allocator, .{
            .lr = lr,
            .weight_decay = wd,
            .state_dtype = .bf16,
            .second_moment_dtype = if (v_bf16) .bf16 else .f32,
        });
        defer opt.deinit();
        try opt.addParam(&w);
        for (&grads) |*g| try applyGrad(&ctx, &w, .{ .out, .in }, .{ 2, 5 }, g, &opt);

        // Identical scalar prep (f64 rounded once to f32) + element loop,
        // with the same widen/narrow points; the parameter update uses the
        // just-computed (pre-narrow) f32 moments.
        var p_ref: [10]f32 = w0;
        var m_bits = [_]u16{0} ** 10;
        var v_f32 = [_]f32{0} ** 10;
        var v_bits = [_]u16{0} ** 10;
        for (&grads, 1..) |*g, step| {
            const t: f64 = @floatFromInt(step);
            const bc1 = 1 - std.math.pow(f64, beta1, t);
            const bc2s: f32 = @floatCast(@sqrt(1 - std.math.pow(f64, beta2, t)));
            const keep: f32 = @floatCast(1.0 - @as(f64, lr) * @as(f64, wd));
            const one_minus_b1: f32 = @floatCast(1.0 - @as(f64, beta1));
            const one_minus_b2: f32 = @floatCast(1.0 - @as(f64, beta2));
            const step_size: f32 = @floatCast(@as(f64, lr) / bc1);
            for (0..10) |i| {
                const gi = g[i];
                const decayed = p_ref[i] * keep;
                const m0 = bf16ToF32(m_bits[i]);
                const v0 = if (v_bf16) bf16ToF32(v_bits[i]) else v_f32[i];
                const m1 = m0 + one_minus_b1 * (gi - m0);
                const v1 = beta2 * v0 + one_minus_b2 * gi * gi;
                m_bits[i] = f32ToBf16(m1);
                if (v_bf16) {
                    v_bits[i] = f32ToBf16(v1);
                } else {
                    v_f32[i] = v1;
                }
                p_ref[i] = decayed - step_size * (m1 / (@sqrt(v1) / bc2s + eps));
            }
        }
        try std.testing.expectEqualSlices(f32, &p_ref, try w.dataConst());
        try std.testing.expectEqualSlices(u16, &m_bits, opt.slots.items[0].m.bf16);
        if (v_bf16) {
            try std.testing.expectEqualSlices(u16, &v_bits, opt.slots.items[0].v.bf16);
        } else {
            try std.testing.expectEqualSlices(f32, &v_f32, opt.slots.items[0].v.f32);
        }
    }
}

test "optim bf16 state: Adam kernel matches the widen-narrow reference exactly" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const w0 = [_]f32{ 1.0, -2.0, 0.5, 3.0, -0.25, 1.5, 0.75, -1.25, 2.25 };
    const grads = [_][9]f32{
        .{ 0.5, -1.0, 2.0, 0.25, -0.75, 1.25, -0.6, 0.45, -0.15 },
        .{ -0.3, 0.8, -1.2, 0.6, 0.1, -0.9, 1.05, -0.35, 0.55 },
        .{ 1.1, 0.2, -0.4, -1.3, 0.7, 0.05, -0.95, 0.65, -0.25 },
    };
    const lr: f32 = 0.1;
    const wd: f32 = 0.1; // coupled L2: exercises the g += wd*p path
    const beta1: f32 = 0.9;
    const beta2: f32 = 0.999;
    const eps: f32 = 1e-8;

    var w = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 3, 3 }, &w0);
    defer w.deinit();
    var opt = optim.Adam.init(allocator, .{
        .lr = lr,
        .weight_decay = wd,
        .state_dtype = .bf16,
        .second_moment_dtype = .bf16,
    });
    defer opt.deinit();
    try opt.addParam(&w);
    for (&grads) |*g| try applyGrad(&ctx, &w, .{ .out, .in }, .{ 3, 3 }, g, &opt);

    var p_ref: [9]f32 = w0;
    var m_bits = [_]u16{0} ** 9;
    var v_bits = [_]u16{0} ** 9;
    for (&grads, 1..) |*g, step| {
        const t: f64 = @floatFromInt(step);
        const bc1 = 1 - std.math.pow(f64, beta1, t);
        const bc2s: f32 = @floatCast(@sqrt(1 - std.math.pow(f64, beta2, t)));
        const one_minus_b1: f32 = @floatCast(1.0 - @as(f64, beta1));
        const one_minus_b2: f32 = @floatCast(1.0 - @as(f64, beta2));
        const step_size: f32 = @floatCast(@as(f64, lr) / bc1);
        for (0..9) |i| {
            const gi = g[i] + wd * p_ref[i];
            const m0 = bf16ToF32(m_bits[i]);
            const v0 = bf16ToF32(v_bits[i]);
            const m1 = m0 + one_minus_b1 * (gi - m0);
            const v1 = beta2 * v0 + one_minus_b2 * gi * gi;
            m_bits[i] = f32ToBf16(m1);
            v_bits[i] = f32ToBf16(v1);
            p_ref[i] -= step_size * (m1 / (@sqrt(v1) / bc2s + eps));
        }
    }
    try std.testing.expectEqualSlices(f32, &p_ref, try w.dataConst());
    try std.testing.expectEqualSlices(u16, &m_bits, opt.slots.items[0].m.bf16);
    try std.testing.expectEqualSlices(u16, &v_bits, opt.slots.items[0].v.bf16);
}

test "optim bf16 state: SGD momentum kernel matches the widen-narrow reference exactly" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const w0 = [_]f32{ 1.0, -2.0, 0.5, 3.0, -0.25, 1.5, 0.75, -1.25, 2.25, -0.5 };
    const grads = [_][10]f32{
        .{ 0.5, -1.0, 2.0, 0.25, -0.75, 1.25, -0.6, 0.45, -0.15, 0.9 },
        .{ -0.3, 0.8, -1.2, 0.6, 0.1, -0.9, 1.05, -0.35, 0.55, -0.85 },
        .{ 1.1, 0.2, -0.4, -1.3, 0.7, 0.05, -0.95, 0.65, -0.25, 0.15 },
    };
    const lr: f32 = 0.1;
    const momentum: f32 = 0.9;
    const wd: f32 = 0.05;

    var w = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 2, 5 }, &w0);
    defer w.deinit();
    var opt = optim.SGD.init(allocator, .{
        .lr = lr,
        .momentum = momentum,
        .weight_decay = wd,
        .nesterov = true,
        .state_dtype = .bf16,
    });
    defer opt.deinit();
    try opt.addParam(&w);
    for (&grads) |*g| try applyGrad(&ctx, &w, .{ .out, .in }, .{ 2, 5 }, g, &opt);

    // First step: buf = clone of the DECAYED gradient, stored narrowed; the
    // nesterov blend and the parameter update use the pre-narrow f32 value.
    var p_ref: [10]f32 = w0;
    var buf_bits = [_]u16{0} ** 10;
    const one_minus_damp: f32 = @floatCast(1.0 - @as(f64, @as(f32, 0)));
    for (&grads, 1..) |*g, step| {
        for (0..10) |i| {
            var gi = g[i];
            gi += wd * p_ref[i];
            const b1 = if (step == 1)
                gi
            else
                momentum * bf16ToF32(buf_bits[i]) + one_minus_damp * gi;
            buf_bits[i] = f32ToBf16(b1);
            gi = gi + momentum * b1; // nesterov
            p_ref[i] -= lr * gi;
        }
    }
    try std.testing.expectEqualSlices(f32, &p_ref, try w.dataConst());
    try std.testing.expectEqualSlices(u16, &buf_bits, opt.slots.items[0].buf.bf16);
}

test "optim bf16 state: Muon momentum kernel matches the widen-narrow reference exactly" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const w0 = [_]f32{ 1.0, -2.0, 0.5, 3.0, -0.25, 1.5, 0.75, -1.25, 2.25, -0.5 };
    const grads = [_][10]f32{
        .{ 0.5, -1.0, 2.0, 0.25, -0.75, 1.25, -0.6, 0.45, -0.15, 0.9 },
        .{ -0.3, 0.8, -1.2, 0.6, 0.1, -0.9, 1.05, -0.35, 0.55, -0.85 },
        .{ 1.1, 0.2, -0.4, -1.3, 0.7, 0.05, -0.95, 0.65, -0.25, 0.15 },
    };
    const lr: f32 = 0.02;
    const wd: f32 = 0.01;
    const beta: f32 = 0.95; // default momentum; nesterov = true (default)

    // 5x2 is tall, so the Newton-Schulz transpose trick is exercised.
    var w = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 5, 2 }, &w0);
    defer w.deinit();
    var opt = optim.Muon.init(allocator, .{ .lr = lr, .weight_decay = wd, .state_dtype = .bf16 });
    defer opt.deinit();
    try opt.addParam(&w);
    for (&grads) |*g| try applyGrad(&ctx, &w, .{ .out, .in }, .{ 5, 2 }, g, &opt);

    // Reference: the identical momentum recurrence with the same
    // widen/narrow points, then the REAL Newton-Schulz (pub) on the same u —
    // bit-identical inputs give bit-identical orthogonalizations in-process.
    var p_ref: [10]f32 = w0;
    var m_bits = [_]u16{0} ** 10;
    for (&grads) |*g| {
        var u_vals: [10]f32 = undefined;
        for (0..10) |i| {
            const gi = g[i];
            const lerp_w = 1 - beta;
            const m0 = bf16ToF32(m_bits[i]);
            const m1 = if (lerp_w < 0.5) m0 + lerp_w * (gi - m0) else gi - (gi - m0) * beta;
            m_bits[i] = f32ToBf16(m1);
            u_vals[i] = if (beta < 0.5) gi + beta * (m1 - gi) else m1 - (m1 - gi) * lerp_w;
        }
        var u = try ctx.fromSliceRank(2, .{ 5, 2 }, &u_vals);
        defer u.deinit();
        var ortho = try optim.newtonSchulz5(&ctx, &u, 5);
        defer ortho.deinit();
        const rows_f: f32 = 5;
        const cols_f: f32 = 2;
        const lr_eff = lr * @sqrt(@max(1, rows_f / cols_f));
        const keep: f32 = @floatCast(1.0 - @as(f64, lr) * @as(f64, wd));
        for (p_ref[0..], ortho.dataConst()) |*pi, oi| {
            pi.* = pi.* * keep - lr_eff * oi;
        }
    }
    try std.testing.expectEqualSlices(f32, &p_ref, try w.dataConst());
    try std.testing.expectEqualSlices(u16, &m_bits, opt.slots.items[0].momentum.bf16);
}

test "optim bf16 state: AdamW converges on a quadratic" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var w = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, -1, 2, 0.5, -2, 1.5 });
    defer w.deinit();
    var target = try Tensor(.{ .out, .in }).fromSlice(&ctx, .{ 2, 3 }, &.{ -0.5, 0.25, 1, -1, 0.75, 0 });
    defer target.deinit();

    var opt = optim.AdamW.init(allocator, .{
        .lr = 0.05,
        .weight_decay = 0,
        .state_dtype = .bf16,
        .second_moment_dtype = .bf16,
    });
    defer opt.deinit();
    try opt.addParam(&w);

    var last_loss: f32 = std.math.inf(f32);
    for (0..300) |_| {
        var diff = try w.sub(&ctx, &target);
        defer diff.deinit();
        var sq = try diff.mul(&ctx, &diff);
        defer sq.deinit();
        var loss = try sq.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);
        try opt.step(&ctx);
        opt.zeroGrad();
        last_loss = try loss.item();
    }
    // Looser than the f32 test's 1e-4: bf16 v quantizes the denominator.
    try std.testing.expect(last_loss < 1e-3);
}

test "optim bf16 state checkpoint resume is bit-exact (AdamW, Adam, Muon, SGD)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const w0 = [_]f32{ 1, -1, 2, 0.5, -2, 1.5, 0.25, 0.75 };
    const t0 = [_]f32{ -0.5, 0.25, 1, -1, 0.75, 0, 0.6, -0.4 };
    const b0 = [_]f32{ 0.5, -0.5, 1 };
    const b_t = [_]f32{ -0.2, 0.4, 0.1 };

    inline for (.{ "adam", "adamw", "muon", "sgd" }) |kind| {
        var w = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 4, 2 }, &w0);
        defer w.deinit();
        var target = try Tensor(.{ .out, .in }).fromSlice(&ctx, .{ 4, 2 }, &t0);
        defer target.deinit();
        // b routes to Muon's AdamW fallback (bf16 there too, so the nested
        // FZA4 frame is exercised inside FZM4); the elementwise optimizers
        // just treat it as a second slot.
        var b = try Tensor(.{.out}).variableFromSlice(&ctx, .{3}, &b0);
        defer b.deinit();
        var b_target = try Tensor(.{.out}).fromSlice(&ctx, .{3}, &b_t);
        defer b_target.deinit();

        var opt = if (comptime std.mem.eql(u8, kind, "adam"))
            optim.Adam.init(allocator, .{ .lr = 0.05, .state_dtype = .bf16, .second_moment_dtype = .bf16 })
        else if (comptime std.mem.eql(u8, kind, "adamw"))
            optim.AdamW.init(allocator, .{ .lr = 0.05, .state_dtype = .bf16, .second_moment_dtype = .bf16 })
        else if (comptime std.mem.eql(u8, kind, "muon"))
            optim.Muon.init(allocator, .{
                .lr = 0.02,
                .weight_decay = 0.01,
                .state_dtype = .bf16,
                .fallback = .{ .lr = 3e-4, .beta1 = 0.9, .beta2 = 0.95, .eps = 1e-10, .weight_decay = 0, .state_dtype = .bf16 },
            })
        else
            optim.SGD.init(allocator, .{ .lr = 0.05, .momentum = 0.9, .state_dtype = .bf16 });
        defer opt.deinit();
        try opt.addParam(&w);
        try opt.addParam(&b);

        for (0..5) |_| {
            try quadraticStep(&ctx, &w, &target, &opt);
            try quadraticStep(&ctx, &b, &b_target, &opt);
        }

        var buf: [16384]u8 = undefined;
        var writer = std.Io.Writer.fixed(&buf);
        try optim.saveTensors(&writer, .{ &w, &b });
        try opt.saveState(&writer);
        const snapshot = writer.buffered();

        for (0..5) |_| {
            try quadraticStep(&ctx, &w, &target, &opt);
            try quadraticStep(&ctx, &b, &b_target, &opt);
        }
        var final_w: [8]f32 = undefined;
        @memcpy(&final_w, try w.dataConst());
        var final_b: [3]f32 = undefined;
        @memcpy(&final_b, try b.dataConst());

        var reader = std.Io.Reader.fixed(snapshot);
        try optim.loadTensors(allocator, &reader, .{ &w, &b });
        try opt.loadState(&reader);
        for (0..5) |_| {
            try quadraticStep(&ctx, &w, &target, &opt);
            try quadraticStep(&ctx, &b, &b_target, &opt);
        }

        try std.testing.expectEqualSlices(f32, &final_w, try w.dataConst());
        try std.testing.expectEqualSlices(f32, &final_b, try b.dataConst());
    }
}

test "optim bf16 state v4 named resume is bit-exact under permuted registration" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const w1_0 = [_]f32{ 1, -1, 2, 0.5, -2, 1.5, 0.25, 0.75 };
    const w1_t = [_]f32{ -0.5, 0.25, 1, -1, 0.75, 0, 0.6, -0.4 };
    const w2_0 = [_]f32{ 0.5, -1.5, 2, -0.25, 1, 0.75 };
    const w2_t = [_]f32{ -0.8, 0.6, -1.1, 0.9, -0.4, 0.3 };
    const b0 = [_]f32{ 0.5, -0.5, 1 };
    const b_t = [_]f32{ -0.2, 0.4, 0.1 };

    inline for (.{ "adamw", "muon" }) |kind| {
        var w1 = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 4, 2 }, &w1_0);
        defer w1.deinit();
        var w2 = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 2, 3 }, &w2_0);
        defer w2.deinit();
        var b = try Tensor(.{.out}).variableFromSlice(&ctx, .{3}, &b0);
        defer b.deinit();
        var w1_target = try Tensor(.{ .out, .in }).fromSlice(&ctx, .{ 4, 2 }, &w1_t);
        defer w1_target.deinit();
        var w2_target = try Tensor(.{ .out, .in }).fromSlice(&ctx, .{ 2, 3 }, &w2_t);
        defer w2_target.deinit();
        var b_target = try Tensor(.{.out}).fromSlice(&ctx, .{3}, &b_t);
        defer b_target.deinit();

        var opt = if (comptime std.mem.eql(u8, kind, "adamw"))
            optim.AdamW.init(allocator, .{ .lr = 0.05, .state_dtype = .bf16, .second_moment_dtype = .bf16 })
        else
            optim.Muon.init(allocator, .{
                .lr = 0.02,
                .weight_decay = 0.01,
                .state_dtype = .bf16,
                .fallback = .{ .lr = 3e-4, .beta1 = 0.9, .beta2 = 0.95, .eps = 1e-10, .weight_decay = 0, .state_dtype = .bf16 },
            });
        defer opt.deinit();
        try opt.addParamNamed(&w1, "w1");
        try opt.addParamNamed(&w2, "w2");
        try opt.addParamNamed(&b, "bias");

        for (0..4) |_| {
            try quadraticStep(&ctx, &w1, &w1_target, &opt);
            try quadraticStep(&ctx, &w2, &w2_target, &opt);
            try quadraticStep(&ctx, &b, &b_target, &opt);
        }

        var buf: [16384]u8 = undefined;
        var writer = std.Io.Writer.fixed(&buf);
        const save_entries = [_]optim.NamedTensor{
            try optim.NamedTensor.of("w1", &w1),
            try optim.NamedTensor.of("w2", &w2),
            try optim.NamedTensor.of("bias", &b),
        };
        try optim.saveStateDict(allocator, &writer, &save_entries);
        try opt.saveState(&writer);
        const snapshot = writer.buffered();

        for (0..4) |_| {
            try quadraticStep(&ctx, &w1, &w1_target, &opt);
            try quadraticStep(&ctx, &w2, &w2_target, &opt);
            try quadraticStep(&ctx, &b, &b_target, &opt);
        }
        var final_w1: [8]f32 = undefined;
        @memcpy(&final_w1, try w1.dataConst());
        var final_w2: [6]f32 = undefined;
        @memcpy(&final_w2, try w2.dataConst());
        var final_b: [3]f32 = undefined;
        @memcpy(&final_b, try b.dataConst());

        // Fresh optimizer registering the SAME names in REVERSED order.
        var opt2 = if (comptime std.mem.eql(u8, kind, "adamw"))
            optim.AdamW.init(allocator, .{ .lr = 0.05, .state_dtype = .bf16, .second_moment_dtype = .bf16 })
        else
            optim.Muon.init(allocator, .{
                .lr = 0.02,
                .weight_decay = 0.01,
                .state_dtype = .bf16,
                .fallback = .{ .lr = 3e-4, .beta1 = 0.9, .beta2 = 0.95, .eps = 1e-10, .weight_decay = 0, .state_dtype = .bf16 },
            });
        defer opt2.deinit();
        try opt2.addParamNamed(&b, "bias");
        try opt2.addParamNamed(&w2, "w2");
        try opt2.addParamNamed(&w1, "w1");

        var reader = std.Io.Reader.fixed(snapshot);
        const load_entries = [_]optim.NamedTensorMut{
            try optim.NamedTensorMut.of("bias", &b),
            try optim.NamedTensorMut.of("w1", &w1),
            try optim.NamedTensorMut.of("w2", &w2),
        };
        try optim.loadStateDict(allocator, &reader, &load_entries, .{});
        try opt2.loadState(&reader);
        for (0..4) |_| {
            try quadraticStep(&ctx, &w1, &w1_target, &opt2);
            try quadraticStep(&ctx, &w2, &w2_target, &opt2);
            try quadraticStep(&ctx, &b, &b_target, &opt2);
        }

        try std.testing.expectEqualSlices(f32, &final_w1, try w1.dataConst());
        try std.testing.expectEqualSlices(f32, &final_w2, try w2.dataConst());
        try std.testing.expectEqualSlices(f32, &final_b, try b.dataConst());
    }
}

test "optim checkpoint cross-dtype loads are rejected without conversion" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var w = try Tensor(.{.out}).variableFromSlice(&ctx, .{2}, &.{ 1, 2 });
    defer w.deinit();

    var buf: [4096]u8 = undefined;

    // f32-configured save writes the v3 magic...
    var writer_v3 = std.Io.Writer.fixed(&buf);
    {
        var opt = optim.AdamW.init(allocator, .{ .lr = 0.1 });
        defer opt.deinit();
        try opt.addParamNamed(&w, "w");
        try opt.saveState(&writer_v3);
    }
    const snapshot_v3 = writer_v3.buffered();
    try std.testing.expectEqualSlices(u8, "FZA3", snapshot_v3[0..4]);

    // ...which a bf16-configured optimizer must refuse (v3 implies f32).
    {
        var opt = optim.AdamW.init(allocator, .{ .lr = 0.1, .state_dtype = .bf16 });
        defer opt.deinit();
        try opt.addParamNamed(&w, "w");
        var reader = std.Io.Reader.fixed(snapshot_v3);
        try std.testing.expectError(optim.OptimError.CheckpointDtypeMismatch, opt.loadState(&reader));
    }

    // bf16-configured save writes the v4 magic...
    var buf_v4: [4096]u8 = undefined;
    var writer_v4 = std.Io.Writer.fixed(&buf_v4);
    {
        var opt = optim.AdamW.init(allocator, .{ .lr = 0.1, .state_dtype = .bf16 });
        defer opt.deinit();
        try opt.addParamNamed(&w, "w");
        try opt.saveState(&writer_v4);
    }
    const snapshot_v4 = writer_v4.buffered();
    try std.testing.expectEqualSlices(u8, "FZA4", snapshot_v4[0..4]);

    // ...which an f32-configured optimizer must refuse...
    {
        var opt = optim.AdamW.init(allocator, .{ .lr = 0.1 });
        defer opt.deinit();
        try opt.addParamNamed(&w, "w");
        var reader = std.Io.Reader.fixed(snapshot_v4);
        try std.testing.expectError(optim.OptimError.CheckpointDtypeMismatch, opt.loadState(&reader));
    }
    // ...and so must a bf16-m one whose v dtype differs (m bf16 + v bf16).
    {
        var opt = optim.AdamW.init(allocator, .{ .lr = 0.1, .state_dtype = .bf16, .second_moment_dtype = .bf16 });
        defer opt.deinit();
        try opt.addParamNamed(&w, "w");
        var reader = std.Io.Reader.fixed(snapshot_v4);
        try std.testing.expectError(optim.OptimError.CheckpointDtypeMismatch, opt.loadState(&reader));
    }

    // SGD without momentum has no state buffers: state_dtype is inert and
    // the frame stays v3 regardless.
    var writer_sgd = std.Io.Writer.fixed(&buf);
    {
        var opt = optim.SGD.init(allocator, .{ .lr = 0.1, .state_dtype = .bf16 });
        defer opt.deinit();
        try opt.addParamNamed(&w, "w");
        try opt.saveState(&writer_sgd);
    }
    try std.testing.expectEqualSlices(u8, "FZS3", writer_sgd.buffered()[0..4]);
}

test "optim v4 loadState is transactional: a truncated record leaves prior slot unmutated" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try Tensor(.{.out}).variableFromSlice(&ctx, .{2}, &.{ 1, 2 });
    defer a.deinit();
    var b = try Tensor(.{.out}).variableFromSlice(&ctx, .{2}, &.{ 3, 4 });
    defer b.deinit();

    const config = optim.AdamWConfig{ .lr = 0.1, .state_dtype = .bf16, .second_moment_dtype = .bf16 };
    var opt = optim.AdamW.init(allocator, config);
    defer opt.deinit();
    try opt.addParamNamed(&a, "a");
    try opt.addParamNamed(&b, "b");
    opt.slots.items[0].step = 7;
    @memcpy(opt.slots.items[0].m.bf16, &[_]u16{ f32ToBf16(0.1), f32ToBf16(0.2) });
    @memcpy(opt.slots.items[0].v.bf16, &[_]u16{ f32ToBf16(0.3), f32ToBf16(0.4) });
    opt.slots.items[1].step = 9;
    @memcpy(opt.slots.items[1].m.bf16, &[_]u16{ f32ToBf16(0.5), f32ToBf16(0.6) });
    @memcpy(opt.slots.items[1].v.bf16, &[_]u16{ f32ToBf16(0.7), f32ToBf16(0.8) });

    var buf: [4096]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try opt.saveState(&writer);
    // Truncate mid-second-record (slot "b"): "a" reads in full, "b" is short.
    const truncated = writer.buffered()[0 .. writer.buffered().len - 2];

    var opt2 = optim.AdamW.init(allocator, config);
    defer opt2.deinit();
    try opt2.addParamNamed(&a, "a");
    try opt2.addParamNamed(&b, "b");
    opt2.slots.items[0].step = 123;
    const sentinel = f32ToBf16(8);
    @memcpy(opt2.slots.items[0].m.bf16, &[_]u16{ sentinel, sentinel });
    @memcpy(opt2.slots.items[0].v.bf16, &[_]u16{ sentinel, sentinel });

    var reader = std.Io.Reader.fixed(truncated);
    if (opt2.loadState(&reader)) |_| {
        return error.TestExpectedTruncationError;
    } else |_| {}
    // Transactional: slot "a" (read OK before "b" truncated) is byte-unchanged.
    try std.testing.expectEqual(@as(u64, 123), opt2.slots.items[0].step);
    try std.testing.expectEqualSlices(u16, &[_]u16{ sentinel, sentinel }, opt2.slots.items[0].m.bf16);
    try std.testing.expectEqualSlices(u16, &[_]u16{ sentinel, sentinel }, opt2.slots.items[0].v.bf16);
}

test "optim state frames: all-f32 writes byte-identical v3; bf16 writes dtype-tagged v4" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try Tensor(.{.out}).variableFromSlice(&ctx, .{2}, &.{ 1, 2 });
    defer a.deinit();

    // Default (all-f32) config: the frame must be byte-for-byte the v3 wire
    // format — no tags, raw f32 state — so pre-bf16 builds keep reading it.
    {
        var opt = optim.AdamW.init(allocator, .{ .lr = 0.1 });
        defer opt.deinit();
        try opt.addParamNamed(&a, "a");
        opt.slots.items[0].step = 3;
        @memcpy(opt.slots.items[0].m.f32, &[_]f32{ 0.25, -0.5 });
        @memcpy(opt.slots.items[0].v.f32, &[_]f32{ 0.125, 2.0 });

        var buf: [256]u8 = undefined;
        var writer = std.Io.Writer.fixed(&buf);
        try opt.saveState(&writer);

        var exp_buf: [256]u8 = undefined;
        var exp = std.Io.Writer.fixed(&exp_buf);
        try exp.writeAll("FZA3");
        try exp.writeInt(u32, 1, .little); // slot count
        try exp.writeInt(u16, 1, .little); // name length
        try exp.writeAll("a");
        try exp.writeInt(u64, 2, .little); // rows
        try exp.writeInt(u64, 1, .little); // cols
        try exp.writeInt(u64, 3, .little); // step
        try exp.writeAll(std.mem.sliceAsBytes(&[_]f32{ 0.25, -0.5 }));
        try exp.writeAll(std.mem.sliceAsBytes(&[_]f32{ 0.125, 2.0 }));
        try std.testing.expectEqualSlices(u8, exp.buffered(), writer.buffered());
    }

    // bf16 m + f32 v: v4 frame with one u8 StateDType tag per state buffer.
    {
        var opt = optim.AdamW.init(allocator, .{ .lr = 0.1, .state_dtype = .bf16 });
        defer opt.deinit();
        try opt.addParamNamed(&a, "a");
        opt.slots.items[0].step = 3;
        @memcpy(opt.slots.items[0].m.bf16, &[_]u16{ f32ToBf16(0.25), f32ToBf16(-0.5) });
        @memcpy(opt.slots.items[0].v.f32, &[_]f32{ 0.125, 2.0 });

        var buf: [256]u8 = undefined;
        var writer = std.Io.Writer.fixed(&buf);
        try opt.saveState(&writer);

        var exp_buf: [256]u8 = undefined;
        var exp = std.Io.Writer.fixed(&exp_buf);
        try exp.writeAll("FZA4");
        try exp.writeInt(u32, 1, .little);
        try exp.writeInt(u16, 1, .little);
        try exp.writeAll("a");
        try exp.writeInt(u64, 2, .little);
        try exp.writeInt(u64, 1, .little);
        try exp.writeInt(u64, 3, .little);
        try exp.writeInt(u8, 1, .little); // StateDType.bf16 wire tag
        try exp.writeAll(std.mem.sliceAsBytes(&[_]u16{ f32ToBf16(0.25), f32ToBf16(-0.5) }));
        try exp.writeInt(u8, 0, .little); // StateDType.f32 wire tag
        try exp.writeAll(std.mem.sliceAsBytes(&[_]f32{ 0.125, 2.0 }));
        try std.testing.expectEqualSlices(u8, exp.buffered(), writer.buffered());

        // And the v4 frame round-trips into an identically-configured optimizer.
        var opt2 = optim.AdamW.init(allocator, .{ .lr = 0.1, .state_dtype = .bf16 });
        defer opt2.deinit();
        try opt2.addParamNamed(&a, "a");
        var reader = std.Io.Reader.fixed(writer.buffered());
        try opt2.loadState(&reader);
        try std.testing.expectEqual(@as(u64, 3), opt2.slots.items[0].step);
        try std.testing.expectEqualSlices(u16, opt.slots.items[0].m.bf16, opt2.slots.items[0].m.bf16);
        try std.testing.expectEqualSlices(f32, opt.slots.items[0].v.f32, opt2.slots.items[0].v.f32);
    }
}

// ---------------------------------------------------------------------------
// Gradient accumulation (docs/TRAINING.md §4 "Gradient accumulation"): big-batch
// equivalence, run-to-run determinism, and clip ordering — on a tiny
// batch-axis MLP (the LLM trainers' loss is single-sequence, so a literal
// "one 4x batch" comparison is only expressible here).
// ---------------------------------------------------------------------------

const rng = @import("rng.zig");

const AccumMlp = struct {
    w1: Tensor(.{ .h1, .in }),
    w2: Tensor(.{ .class, .h1 }),

    const in_dim = 4;
    const hidden = 8;
    const n_class = 3;
    const batch = 8;

    fn init(ctx: *ExecContext, seed: u64) !AccumMlp {
        var w1_buf: [hidden * in_dim]f32 = undefined;
        rng.uniformFill(rng.at(seed, 0), &w1_buf, -0.6, 0.6);
        var w2_buf: [n_class * hidden]f32 = undefined;
        rng.uniformFill(rng.at(seed, 1), &w2_buf, -0.6, 0.6);
        var w1 = try Tensor(.{ .h1, .in }).variableFromSlice(ctx, .{ hidden, in_dim }, &w1_buf);
        errdefer w1.deinit();
        var w2 = try Tensor(.{ .class, .h1 }).variableFromSlice(ctx, .{ n_class, hidden }, &w2_buf);
        errdefer w2.deinit();
        return .{ .w1 = w1, .w2 = w2 };
    }

    fn deinit(self: *AccumMlp) void {
        self.w2.deinit();
        self.w1.deinit();
        self.* = undefined;
    }
};

fn accumInputs() [AccumMlp.batch * AccumMlp.in_dim]f32 {
    var xs: [AccumMlp.batch * AccumMlp.in_dim]f32 = undefined;
    rng.uniformFill(rng.at(0xACC0DA7A, 0), &xs, -1, 1);
    return xs;
}

const accum_labels = [AccumMlp.batch]usize{ 0, 1, 2, 0, 2, 1, 0, 1 };

/// One micro-batch pass under its own exec scope: forward over `rows` samples
/// starting at `first`, mean CE, loss scaled by `loss_scale` before backward
/// (the canonical loss-side normalization arm). Returns the scaled loss.
fn accumLossBackward(
    ctx: *ExecContext,
    model: *const AccumMlp,
    xs: []const f32,
    labels: []const usize,
    first: usize,
    rows: usize,
    loss_scale: f32,
) !f32 {
    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    var x = try Tensor(.{ .batch, .in }).fromSlice(ctx, .{ rows, AccumMlp.in_dim }, xs[first * AccumMlp.in_dim ..][0 .. rows * AccumMlp.in_dim]);
    defer x.deinit();
    const h = try x.dot(ctx, &model.w1, .in);
    const a = try h.tanh(ctx);
    const logits = try a.dot(ctx, &model.w2, .h1);
    const ce = try logits.crossEntropy(ctx, .class, labels[first .. first + rows]);
    if (loss_scale == 1) {
        try ce.backward(ctx);
        return ce.item();
    }
    const scaled = try ce.scale(ctx, loss_scale);
    try scaled.backward(ctx);
    return scaled.item();
}

/// Caller-owned f32 copy of a param's accumulated gradient.
fn accumGradDupe(allocator: std.mem.Allocator, ctx: *ExecContext, t: anytype) ![]f32 {
    var g = (try t.grad(ctx)) orelse return error.MissingGrad;
    defer g.deinit();
    return allocator.dupe(f32, try g.dataConst());
}

fn expectRelCloseSlices(expected: []const f32, actual: []const f32, rtol: f32, atol: f32) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |e, a| {
        const diff = @abs(e - a);
        const tol = @max(rtol * @max(@abs(e), @abs(a)), atol);
        try std.testing.expect(diff <= tol);
    }
}

const AccumWindowResult = struct {
    g_w1: []f32,
    g_w2: []f32,
    p_w1: []f32,
    p_w2: []f32,

    fn deinit(self: *AccumWindowResult, allocator: std.mem.Allocator) void {
        allocator.free(self.p_w2);
        allocator.free(self.p_w1);
        allocator.free(self.g_w2);
        allocator.free(self.g_w1);
        self.* = undefined;
    }
};

/// One full accumulation window from a fresh (seed-determined, bitwise
/// identical) start: 4 quarter-batch backwards with loss_scale 1/4, grad
/// capture, one AdamW step, param capture.
fn runAccumWindow(ctx: *ExecContext, allocator: std.mem.Allocator, xs: []const f32) !AccumWindowResult {
    var model = try AccumMlp.init(ctx, 0xB16B);
    defer model.deinit();
    var opt = optim.AdamW.init(allocator, .{ .lr = 0.05, .weight_decay = 0.01 });
    defer opt.deinit();
    try opt.addParam(&model.w1);
    try opt.addParam(&model.w2);

    for (0..4) |k| {
        _ = try accumLossBackward(ctx, &model, xs, &accum_labels, k * 2, 2, 0.25);
    }
    const g_w1 = try accumGradDupe(allocator, ctx, &model.w1);
    errdefer allocator.free(g_w1);
    const g_w2 = try accumGradDupe(allocator, ctx, &model.w2);
    errdefer allocator.free(g_w2);
    try opt.step(ctx);
    opt.zeroGrad();
    const p_w1 = try allocator.dupe(f32, try model.w1.dataConst());
    errdefer allocator.free(p_w1);
    const p_w2 = try allocator.dupe(f32, try model.w2.dataConst());
    errdefer allocator.free(p_w2);
    return .{ .g_w1 = g_w1, .g_w2 = g_w2, .p_w1 = p_w1, .p_w2 = p_w2 };
}

test "optim gradient accumulation matches the full batch and is run-to-run deterministic" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const xs = accumInputs();

    // Full batch: one mean-CE backward over all 8 rows.
    var full = try AccumMlp.init(&ctx, 0xB16B);
    defer full.deinit();
    _ = try accumLossBackward(&ctx, &full, &xs, &accum_labels, 0, AccumMlp.batch, 1);
    const full_w1 = try accumGradDupe(allocator, &ctx, &full.w1);
    defer allocator.free(full_w1);
    const full_w2 = try accumGradDupe(allocator, &ctx, &full.w2);
    defer allocator.free(full_w2);

    // 4x quarter-batch accumulation with loss_scale = 1/4: equal micro-batch
    // sizes make mean-of-means mathematically identical to the full-batch
    // mean; fp summation order differs, hence tolerance, not bitwise.
    var acc = try AccumMlp.init(&ctx, 0xB16B);
    defer acc.deinit();
    for (0..4) |k| {
        _ = try accumLossBackward(&ctx, &acc, &xs, &accum_labels, k * 2, 2, 0.25);
    }
    const acc_w1 = try accumGradDupe(allocator, &ctx, &acc.w1);
    defer allocator.free(acc_w1);
    const acc_w2 = try accumGradDupe(allocator, &ctx, &acc.w2);
    defer allocator.free(acc_w2);
    try expectRelCloseSlices(full_w1, acc_w1, 1e-5, 1e-7);
    try expectRelCloseSlices(full_w2, acc_w2, 1e-5, 1e-7);

    // Exact determinism: the same window from a bitwise-identical start
    // reproduces the accumulated grads AND the post-step params bitwise.
    var run_a = try runAccumWindow(&ctx, allocator, &xs);
    defer run_a.deinit(allocator);
    var run_b = try runAccumWindow(&ctx, allocator, &xs);
    defer run_b.deinit(allocator);
    try std.testing.expectEqualSlices(f32, run_a.g_w1, run_b.g_w1);
    try std.testing.expectEqualSlices(f32, run_a.g_w2, run_b.g_w2);
    try std.testing.expectEqualSlices(f32, run_a.p_w1, run_b.p_w1);
    try std.testing.expectEqualSlices(f32, run_a.p_w2, run_b.p_w2);
}

test "optim clip once after the accumulation window equals clipping the summed grads; mid-window clip differs" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const xs = accumInputs();
    var model = try AccumMlp.init(&ctx, 0xC11F);
    defer model.deinit();

    var opt = optim.AdamW.init(allocator, .{ .lr = 0.05, .weight_decay = 0 });
    defer opt.deinit();
    try opt.addParam(&model.w1);
    try opt.addParam(&model.w2);
    var set = optim.OptimizerSet.init(allocator);
    defer set.deinit();
    try set.add(&opt);

    const max_norm = 0.01; // far below the grad norms: every clip triggers
    const rows = 4;
    const b1_first = 0;
    const b2_first = 4;
    // No optimizer step anywhere below: the params never move, so each
    // micro-batch reproduces the identical gradient in every phase.

    // Reference: run each micro-batch alone, sum g1 + g2 manually (the same
    // elementwise addition order accumulation performs), install the sums
    // into the GradStates, clip ONCE through the production path.
    _ = try accumLossBackward(&ctx, &model, &xs, &accum_labels, b1_first, rows, 0.5);
    var g1_w1 = (try model.w1.grad(&ctx)) orelse return error.MissingGrad;
    defer g1_w1.deinit();
    var g1_w2 = (try model.w2.grad(&ctx)) orelse return error.MissingGrad;
    defer g1_w2.deinit();
    set.zeroGrad();
    _ = try accumLossBackward(&ctx, &model, &xs, &accum_labels, b2_first, rows, 0.5);
    var g2_w1 = (try model.w1.grad(&ctx)) orelse return error.MissingGrad;
    defer g2_w1.deinit();
    var g2_w2 = (try model.w2.grad(&ctx)) orelse return error.MissingGrad;
    defer g2_w2.deinit();
    set.zeroGrad();

    var sum_w1 = try g1_w1.add(&ctx, &g2_w1);
    defer sum_w1.deinit();
    var sum_w2 = try g1_w2.add(&ctx, &g2_w2);
    defer sum_w2.deinit();
    model.w1.grad_state.?.setGrad(try sum_w1.value.clone(allocator));
    model.w2.grad_state.?.setGrad(try sum_w2.value.clone(allocator));
    const ref_norm = try set.clipGradNorm(&ctx, max_norm);
    try std.testing.expect(ref_norm > max_norm);
    const ref_w1 = try accumGradDupe(allocator, &ctx, &model.w1);
    defer allocator.free(ref_w1);
    const ref_w2 = try accumGradDupe(allocator, &ctx, &model.w2);
    defer allocator.free(ref_w2);
    set.zeroGrad();

    // Correct ordering: accumulate both micro-batches, clip ONCE at the end.
    // Bitwise-equal to the reference: accumulation performed the identical
    // g1 + g2 addition, and the same clip ran on the same numbers.
    _ = try accumLossBackward(&ctx, &model, &xs, &accum_labels, b1_first, rows, 0.5);
    _ = try accumLossBackward(&ctx, &model, &xs, &accum_labels, b2_first, rows, 0.5);
    const acc_norm = try set.clipGradNorm(&ctx, max_norm);
    try std.testing.expect(acc_norm > max_norm);
    const acc_w1 = try accumGradDupe(allocator, &ctx, &model.w1);
    defer allocator.free(acc_w1);
    const acc_w2 = try accumGradDupe(allocator, &ctx, &model.w2);
    defer allocator.free(acc_w2);
    try std.testing.expectEqualSlices(f32, ref_w1, acc_w1);
    try std.testing.expectEqualSlices(f32, ref_w2, acc_w2);
    set.zeroGrad();

    // Wrong ordering: clipping mid-window rescales the partial sum, so the
    // final grads must NOT match the reference.
    _ = try accumLossBackward(&ctx, &model, &xs, &accum_labels, b1_first, rows, 0.5);
    const mid_norm = try set.clipGradNorm(&ctx, max_norm); // clips g1 alone
    try std.testing.expect(mid_norm > max_norm);
    _ = try accumLossBackward(&ctx, &model, &xs, &accum_labels, b2_first, rows, 0.5);
    _ = try set.clipGradNorm(&ctx, max_norm);
    const wrong_w1 = try accumGradDupe(allocator, &ctx, &model.w1);
    defer allocator.free(wrong_w1);
    const wrong_w2 = try accumGradDupe(allocator, &ctx, &model.w2);
    defer allocator.free(wrong_w2);
    try std.testing.expect(!std.mem.eql(f32, ref_w1, wrong_w1) or !std.mem.eql(f32, ref_w2, wrong_w2));
    set.zeroGrad();
}
