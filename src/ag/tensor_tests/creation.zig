//! Constructors: empty/zeros/ones/full/scalar and *Like forms, borrowed
//! slices, arange/linspace/oneHot, the seeded random families, multinomial,
//! and the mask/diagonal builders (bandMask, tril/triu/bandPart).

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

test "public tensor borrowed slice aliases caller-owned data as no-grad constant" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var values = [_]f32{ 1, 2, 3, 4 };
    var x = try Tensor(.{ .row, .col }).fromBorrowedSlice(&ctx, .{ 2, 2 }, values[0..]);
    defer x.deinit();

    try std.testing.expect(!x.requiresGrad());
    values[2] = 30;
    try std.testing.expectEqual(@as(f32, 30), (try x.dataConst())[2]);
}

test "public Tensor empty/zeros/ones/full/scalar constructors (f32)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const T = Tensor(.{ .batch, .d });

    var z = try T.zeros(&ctx, .{ 2, 3 });
    defer z.deinit();
    try std.testing.expect(!z.requiresGrad());
    try std.testing.expectEqual(@as(usize, 2), z.dim(.batch));
    try std.testing.expectEqual(@as(usize, 3), z.dim(.d));
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 0, 0, 0, 0 }, z.asRawTensor().dataConst());

    var o = try T.ones(&ctx, .{ 2, 3 });
    defer o.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 1, 1, 1, 1 }, o.asRawTensor().dataConst());

    var f = try T.full(&ctx, .{ 2, 3 }, 3.0);
    defer f.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 3, 3, 3, 3, 3, 3 }, f.asRawTensor().dataConst());

    var e = try T.empty(&ctx, .{ 2, 3 });
    defer e.deinit();
    try std.testing.expectEqual(@as(usize, 6), e.asRawTensor().dataConst().len);

    const S = Tensor(.{.d});
    var s = try S.scalar(&ctx, 5.0);
    defer s.deinit();
    try std.testing.expectEqual(@as(f32, 5), s.asRawTensor().dataConst()[0]);
}

test "public Tensor zeros/ones/empty typed equivalents build the right shape (f16)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const H = Tensor(.{ .dtype = .f16, .tags = .{ .batch, .d } });
    try std.testing.expect(H.dtype == .f16);

    var z = try H.zeros(&ctx, .{ 2, 2 });
    defer z.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, z.value.shape.slice());

    var o = try H.ones(&ctx, .{ 3, 1 });
    defer o.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 1 }, o.value.shape.slice());

    var e = try H.empty(&ctx, .{ 2, 2 });
    defer e.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, e.value.shape.slice());
}

test "public Tensor fromBorrowedConstSlice wraps const data zero-copy, no @constCast" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const T = Tensor(.{ .batch, .d });
    const data = [_]f32{ 1, 2, 3, 4, 5, 6 };

    // No @constCast at the call site — that is the point of the API.
    var t = try T.fromBorrowedConstSlice(&ctx, .{ 2, 3 }, &data);
    defer t.deinit();

    try std.testing.expect(!t.requiresGrad());
    try std.testing.expectEqual(@as(usize, 2), t.dim(.batch));
    try std.testing.expectEqualSlices(f32, &data, t.asRawTensor().dataConst());
    // Zero-copy: the tensor view aliases the source buffer (no copy).
    try std.testing.expectEqual(@intFromPtr(&data), @intFromPtr(t.asRawTensor().dataConst().ptr));
}

test "public Tensor *Like constructors take shape from the instance" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const M = Tensor(.{ .row, .col });
    var x = try M.variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();

    // Fresh no-grad constants with x's shape; the grad state does not
    // carry over, so mutable data access works on emptyLike storage.
    var e = try x.emptyLike(&ctx);
    defer e.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, e.asRawTensor().shape.slice());
    try std.testing.expect(!e.requiresGrad());
    @memset(try e.data(), 7);
    try expectCloseSlices(&.{ 7, 7, 7, 7, 7, 7 }, try e.dataConst(), 0);

    var z = try x.zerosLike(&ctx);
    defer z.deinit();
    try expectCloseSlices(&.{ 0, 0, 0, 0, 0, 0 }, try z.dataConst(), 0);

    var o = try x.onesLike(&ctx);
    defer o.deinit();
    try expectCloseSlices(&.{ 1, 1, 1, 1, 1, 1 }, try o.dataConst(), 0);

    var m = try x.fullLike(&ctx, -std.math.inf(f32));
    defer m.deinit();
    for (try m.dataConst()) |v| try std.testing.expect(std.math.isNegativeInf(v));

    // A strided view contributes its LOGICAL shape.
    var t = try x.permuteTo(&ctx, .{ .col, .row });
    defer t.deinit();
    var zt = try t.zerosLike(&ctx);
    defer zt.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, zt.asRawTensor().shape.slice());

    // Typed constant branch: same sugar, dtype preserved.
    const I = Tensor(.{ .dtype = .i32, .tags = .{.d} });
    var xi = try I.fromSlice(&ctx, .{4}, &.{ 5, 6, 7, 8 });
    defer xi.deinit();
    var zi = try xi.zerosLike(&ctx);
    defer zi.deinit();
    try std.testing.expectEqualSlices(i32, &.{ 0, 0, 0, 0 }, try zi.dataConst());
    var oi = try xi.onesLike(&ctx);
    defer oi.deinit();
    try std.testing.expectEqualSlices(i32, &.{ 1, 1, 1, 1 }, try oi.dataConst());
}

test "public Tensor arange linspace oneHot constructors" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const V = Tensor(.{.d});
    var a = try V.arange(&ctx, 0, 5, 1);
    defer a.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0, 1, 2, 3, 4 }, try a.dataConst());
    var b = try V.arange(&ctx, 1, 2.5, 0.5); // exclusive end, torch.arange
    defer b.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 1.5, 2 }, try b.dataConst());
    var c = try V.arange(&ctx, 3, 0, -1); // negative step
    defer c.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 3, 2, 1 }, try c.dataConst());
    try std.testing.expectError(error.InvalidShape, V.arange(&ctx, 0, 5, 0));
    try std.testing.expectError(error.InvalidShape, V.arange(&ctx, 5, 0, 1)); // empty range

    var l = try V.linspace(&ctx, 0, 1, 5);
    defer l.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0, 0.25, 0.5, 0.75, 1 }, try l.dataConst());
    var one = try V.linspace(&ctx, 7, 9, 1);
    defer one.deinit();
    try std.testing.expectEqualSlices(f32, &.{7}, try one.dataConst());
    // The endpoint is pinned exactly even when the stride rounds.
    var pinned = try V.linspace(&ctx, 0, 1, 3);
    defer pinned.deinit();
    try std.testing.expectEqual(@as(f32, 1), (try pinned.dataConst())[2]);
    try std.testing.expectError(error.InvalidShape, V.linspace(&ctx, 0, 1, 0));

    const M = Tensor(.{ .pos, .class });
    var oh = try M.oneHot(&ctx, &.{ 2, 0, 1 }, 3);
    defer oh.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 1, 1, 0, 0, 0, 1, 0 }, try oh.dataConst());
    try std.testing.expectError(error.IndexOutOfBounds, M.oneHot(&ctx, &.{3}, 3));
    try std.testing.expectError(error.InvalidShape, M.oneHot(&ctx, &.{}, 3));
}

test "public Tensor rand randn uniform normal bernoulli ride the seed stream" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();
    const rng_mod = @import("../../rng.zig");

    const M = Tensor(.{ .row, .col });
    // Same seed → identical tensor (the §6.8 checkpoint contract);
    // different seed → different values. Constructors are no-grad consts.
    var ua = try M.rand(&ctx, .{ 4, 8 }, 42);
    defer ua.deinit();
    var ub = try M.rand(&ctx, .{ 4, 8 }, 42);
    defer ub.deinit();
    try std.testing.expectEqualSlices(f32, try ua.dataConst(), try ub.dataConst());
    try std.testing.expect(!ua.requiresGrad());
    var uc = try M.rand(&ctx, .{ 4, 8 }, 43);
    defer uc.deinit();
    try std.testing.expect(!std.mem.eql(f32, try ua.dataConst(), try uc.dataConst()));
    for (try ua.dataConst()) |v| try std.testing.expect(v >= 0 and v < 1);

    // rand/uniform/randn/normal reproduce the documented rng.zig mappings.
    var expected: [32]f32 = undefined;
    rng_mod.uniformFill(42, &expected, 0, 1);
    try std.testing.expectEqualSlices(f32, &expected, try ua.dataConst());
    var un = try M.uniform(&ctx, .{ 4, 8 }, 7, -2, 3);
    defer un.deinit();
    rng_mod.uniformFill(7, &expected, -2, 3);
    try std.testing.expectEqualSlices(f32, &expected, try un.dataConst());
    for (try un.dataConst()) |v| try std.testing.expect(v >= -2 and v < 3);
    var g = try M.randn(&ctx, .{ 4, 8 }, 11);
    defer g.deinit();
    rng_mod.normalFill(11, &expected, 0, 1);
    try std.testing.expectEqualSlices(f32, &expected, try g.dataConst());
    var n = try M.normal(&ctx, .{ 4, 8 }, 11, 10, 0.5);
    defer n.deinit();
    rng_mod.normalFill(11, &expected, 10, 0.5);
    try std.testing.expectEqualSlices(f32, &expected, try n.dataConst());

    // bernoulli: 1 iff the [0,1) uniform draw at (seed, i) is below p.
    var bern = try M.bernoulli(&ctx, .{ 4, 8 }, 42, 0.4);
    defer bern.deinit();
    rng_mod.uniformFill(42, &expected, 0, 1);
    for (try bern.dataConst(), expected) |got, draw| {
        try std.testing.expectEqual(@as(f32, if (draw < 0.4) 1 else 0), got);
    }
    try std.testing.expectError(error.InvalidShape, M.bernoulli(&ctx, .{ 4, 8 }, 1, 1.5));
}

test "public Tensor gumbel and eye constructors" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();
    const rng_mod = @import("../../rng.zig");

    const M = Tensor(.{ .row, .col });
    // gumbel reproduces the documented rng.gumbelFill mapping; no-grad const.
    var g = try M.gumbel(&ctx, .{ 4, 8 }, 42);
    defer g.deinit();
    var expected: [32]f32 = undefined;
    rng_mod.gumbelFill(42, &expected);
    try std.testing.expectEqualSlices(f32, &expected, try g.dataConst());
    try std.testing.expect(!g.requiresGrad());
    for (try g.dataConst()) |v| try std.testing.expect(std.math.isFinite(v));
    var g2 = try M.gumbel(&ctx, .{ 4, 8 }, 42);
    defer g2.deinit();
    try std.testing.expectEqualSlices(f32, try g.dataConst(), try g2.dataConst());

    // eye: ones on the main diagonal, zeros elsewhere.
    var identity = try M.eye(&ctx, 3);
    defer identity.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 0, 0, 0, 1, 0, 0, 0, 1 }, try identity.dataConst());
    try std.testing.expect(!identity.requiresGrad());
    try std.testing.expectError(error.InvalidShape, M.eye(&ctx, 0));
}

test "public Tensor randint randperm ride the seed stream" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();
    const rng_mod = @import("../../rng.zig");

    const I = Tensor(.{ .dtype = .i64, .tags = .{ .row, .col } });
    var r = try I.randint(&ctx, .{ 4, 8 }, 42, -3, 5);
    defer r.deinit();
    var expected: [32]i64 = undefined;
    rng_mod.randintFill(42, &expected, -3, 5);
    try std.testing.expectEqualSlices(i64, &expected, try r.dataConst());
    for (try r.dataConst()) |v| try std.testing.expect(v >= -3 and v < 5);
    try std.testing.expect(!r.requiresGrad());
    try std.testing.expectError(error.InvalidShape, I.randint(&ctx, .{ 4, 8 }, 1, 5, 5));

    // randperm: a permutation of 0..n-1, deterministic per seed.
    const P = Tensor(.{ .dtype = .i64, .tags = .{.idx} });
    var p = try P.randperm(&ctx, 17, 7);
    defer p.deinit();
    var seen = [_]bool{false} ** 17;
    for (try p.dataConst()) |v| {
        const i: usize = @intCast(v);
        try std.testing.expect(i < 17);
        try std.testing.expect(!seen[i]);
        seen[i] = true;
    }
    var p2 = try P.randperm(&ctx, 17, 7);
    defer p2.deinit();
    try std.testing.expectEqualSlices(i64, try p.dataConst(), try p2.dataConst());
    var p3 = try P.randperm(&ctx, 17, 8);
    defer p3.deinit();
    try std.testing.expect(!std.mem.eql(i64, try p.dataConst(), try p3.dataConst()));
    try std.testing.expectError(error.InvalidShape, P.randperm(&ctx, 0, 7));
}

test "public Tensor multinomial samples the seed stream with and without replacement" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const M = Tensor(.{ .row, .class });
    // Zero-weight classes are never drawn: all mass on class 1 per row.
    var certain = try M.fromSlice(&ctx, .{ 2, 3 }, &.{ 0, 5, 0, 0, 2, 0 });
    defer certain.deinit();
    var picks = try certain.multinomial(&ctx, .class, .sample, 4, 42, true);
    defer picks.deinit();
    try std.testing.expectEqualSlices(i64, &.{ 1, 1, 1, 1, 1, 1, 1, 1 }, try picks.dataConst());
    try std.testing.expect(!picks.requiresGrad());

    // Deterministic per seed; indices always in range.
    var mixed = try M.fromSlice(&ctx, .{ 2, 4 }, &.{ 1, 2, 3, 4, 4, 3, 2, 1 });
    defer mixed.deinit();
    var a = try mixed.multinomial(&ctx, .class, .sample, 8, 7, true);
    defer a.deinit();
    var b = try mixed.multinomial(&ctx, .class, .sample, 8, 7, true);
    defer b.deinit();
    try std.testing.expectEqualSlices(i64, try a.dataConst(), try b.dataConst());
    for (try a.dataConst()) |v| try std.testing.expect(v >= 0 and v < 4);

    // Without replacement: each row's draws are distinct; exhausting the
    // classes exactly yields a permutation of them.
    var wo = try mixed.multinomial(&ctx, .class, .sample, 4, 11, false);
    defer wo.deinit();
    const wo_data = try wo.dataConst();
    for (0..2) |row| {
        var seen = [_]bool{false} ** 4;
        for (wo_data[row * 4 ..][0..4]) |v| {
            const i: usize = @intCast(v);
            try std.testing.expect(!seen[i]);
            seen[i] = true;
        }
    }
    // num_samples beyond the class count (or the nonzero classes) errors.
    try std.testing.expectError(error.InvalidShape, mixed.multinomial(&ctx, .class, .sample, 5, 1, false));
    try std.testing.expectError(error.InvalidShape, certain.multinomial(&ctx, .class, .sample, 2, 1, false));
    try std.testing.expectError(error.InvalidShape, mixed.multinomial(&ctx, .class, .sample, 0, 1, true));

    // Invalid distributions: negative, NaN, non-positive total.
    var neg = try M.fromSlice(&ctx, .{ 2, 3 }, &.{ 1, -1, 1, 1, 1, 1 });
    defer neg.deinit();
    try std.testing.expectError(error.InvalidShape, neg.multinomial(&ctx, .class, .sample, 1, 1, true));
    var nan_row = try M.fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 1, 1, std.math.nan(f32), 1, 1 });
    defer nan_row.deinit();
    try std.testing.expectError(error.InvalidShape, nan_row.multinomial(&ctx, .class, .sample, 1, 1, true));
    var zero_row = try M.fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 1, 1, 0, 0, 0 });
    defer zero_row.deinit();
    try std.testing.expectError(error.InvalidShape, zero_row.multinomial(&ctx, .class, .sample, 1, 1, true));

    // Rank-1 surface: the class tag is replaced by the sample tag.
    var v = try Tensor(.{.class}).fromSlice(&ctx, .{3}, &.{ 0, 0, 2 });
    defer v.deinit();
    var vp = try v.multinomial(&ctx, .class, .sample, 3, 5, true);
    defer vp.deinit();
    try std.testing.expectEqualSlices(i64, &.{ 2, 2, 2 }, try vp.dataConst());
}

test "public Tensor bandMask builds causal window and triangular keep-sets" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const B = Tensor(.{ .dtype = .bool, .tags = .{ .q, .k } });
    // Causal: keep j <= i.
    var causal = try B.bandMask(&ctx, .{ 3, 3 }, null, 0);
    defer causal.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true, true, false, true, true, true }, try causal.dataConst());

    // Sliding window of width 2 (see one step back), causal.
    var window = try B.bandMask(&ctx, .{ 3, 3 }, 1, 0);
    defer window.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true, true, false, false, true, true }, try window.dataConst());

    // triu(1) keep-set on a rectangle: strictly above the diagonal.
    var upper = try B.bandMask(&ctx, .{ 2, 3 }, -1, null);
    defer upper.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, false, false, true }, try upper.dataConst());

    // Contradictory bounds: all-false, not an error.
    var empty_band = try B.bandMask(&ctx, .{ 2, 2 }, -1, -1);
    defer empty_band.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false }, try empty_band.dataConst());

    // maskedFill consumes it directly (the attention-mask idiom).
    var scores = try Tensor(.{ .q, .k }).fromSlice(&ctx, .{ 3, 3 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 });
    defer scores.deinit();
    var not_causal = try causal.logicalNot(&ctx);
    defer not_causal.deinit();
    var masked = try scores.maskedFill(&ctx, &not_causal, -1e30);
    defer masked.deinit();
    try expectCloseSlices(&.{ 1, -1e30, -1e30, 4, 5, -1e30, 7, 8, 9 }, try masked.dataConst(), 0);
}

test "public Tensor tril triu bandPart with exact mask gradients" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const M = Tensor(.{ .row, .col });
    var x = try M.variableFromSlice(&ctx, .{ 3, 3 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 });
    defer x.deinit();

    var lo = try x.tril(&ctx, .row, .col, 0);
    defer lo.deinit();
    try expectCloseSlices(&.{ 1, 0, 0, 4, 5, 0, 7, 8, 9 }, try lo.dataConst(), 0);
    var loss = try lo.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try expectCloseSlices(&.{ 1, 0, 0, 1, 1, 0, 1, 1, 1 }, try gx.dataConst(), 0);

    // Offsets follow torch: tril(-1) strictly below, triu(1) strictly above.
    var strict_lo = try x.tril(&ctx, .row, .col, -1);
    defer strict_lo.deinit();
    try expectCloseSlices(&.{ 0, 0, 0, 4, 0, 0, 7, 8, 0 }, try strict_lo.dataConst(), 0);
    var up = try x.triu(&ctx, .row, .col, 1);
    defer up.deinit();
    try expectCloseSlices(&.{ 0, 2, 3, 0, 0, 6, 0, 0, 0 }, try up.dataConst(), 0);

    // bandPart keeps a general band; rectangular shape.
    const R = Tensor(.{ .row, .col });
    var rect = try R.fromSlice(&ctx, .{ 2, 4 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer rect.deinit();
    var band = try rect.bandPart(&ctx, .row, .col, 0, 1);
    defer band.deinit();
    try expectCloseSlices(&.{ 1, 2, 0, 0, 0, 6, 7, 0 }, try band.dataConst(), 0);

    // Batched rank-3: remaining axes pass through.
    const T3 = Tensor(.{ .b, .row, .col });
    var bx = try T3.fromSlice(&ctx, .{ 2, 2, 2 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer bx.deinit();
    var btl = try bx.tril(&ctx, .row, .col, 0);
    defer btl.deinit();
    try expectCloseSlices(&.{ 1, 0, 3, 4, 5, 0, 7, 8 }, try btl.dataConst(), 0);
}
