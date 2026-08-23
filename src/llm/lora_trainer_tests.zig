//! Tests for the family-independent LoRA trainer scaffolding
//! (`lora_trainer.zig`): target selection, adapter-set shape, the A/B tuple
//! layout, and the dropout seed stream. The family trainers' own suites
//! cover the forward passes these feed.

const std = @import("std");
const fucina = @import("fucina");
const lora_trainer = @import("lora_trainer.zig");

const ExecContext = fucina.ExecContext;
const Targets = lora_trainer.Targets;

const qv: Targets = .{}; // the default selection: q and v
const all: Targets = .{ .q = true, .k = true, .v = true, .o = true, .gate = true, .up = true, .down = true };
const none: Targets = .{ .q = false, .v = false };

test "target selection drives n_enabled and the enabled predicate" {
    const QV = lora_trainer.AdapterSet(qv, 0);
    try std.testing.expectEqual(@as(usize, 2), QV.n_enabled);
    try std.testing.expect(QV.enabled(0)); // q
    try std.testing.expect(!QV.enabled(1)); // k
    try std.testing.expect(QV.enabled(2)); // v
    try std.testing.expect(!QV.enabled(6)); // down

    try std.testing.expectEqual(@as(usize, 7), lora_trainer.AdapterSet(all, 0).n_enabled);
    try std.testing.expectEqual(@as(usize, 0), lora_trainer.AdapterSet(none, 0).n_enabled);
}

test "disabled targets cost nothing: their LayerAdapters field is void" {
    const QV = lora_trainer.AdapterSet(qv, 0);
    const fields = @typeInfo(QV.LayerAdapters).@"struct".fields;
    try std.testing.expectEqual(@as(usize, 7), fields.len);
    inline for (fields) |f| {
        const is_void = f.type == void;
        const is_enabled = @field(qv, f.name);
        try std.testing.expectEqual(is_enabled, !is_void);
    }
}

test "abIndex packs only enabled targets, two slots each, in target order" {
    const QV = lora_trainer.AdapterSet(qv, 0);
    try std.testing.expectEqual(@as(usize, 0), comptime QV.abIndex(0)); // q -> slots 0,1
    try std.testing.expectEqual(@as(usize, 2), comptime QV.abIndex(2)); // v -> slots 2,3
    try std.testing.expectEqual(@as(usize, 2 * QV.n_enabled), QV.ab_ptr_types.len);

    const All = lora_trainer.AdapterSet(all, 0);
    try std.testing.expectEqual(@as(usize, 0), comptime All.abIndex(0));
    try std.testing.expectEqual(@as(usize, 6), comptime All.abIndex(3)); // o is the fourth
    try std.testing.expectEqual(@as(usize, 12), comptime All.abIndex(6)); // down is the seventh
}

test "adapters build at the shapes TargetDims gives, and round-trip through abTuple" {
    var ctx: ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();

    const QV = lora_trainer.AdapterSet(qv, 0);
    var dims: lora_trainer.TargetDims = undefined;
    for (&dims) |*d| d.* = .{ 8, 8 };
    dims[0] = .{ 16, 24 }; // q: [hidden, q_dim]
    dims[2] = .{ 16, 8 }; // v: [hidden, kv_dim]

    var ads: QV.LayerAdapters = undefined;
    try QV.initLayerAdapters(&ctx, &ads, dims, .{ .rank = 4, .alpha = 8 }, 0xfeed, 3);
    defer QV.deinitLayerAdaptersPartial(&ads, QV.n_enabled);

    const abs = QV.abTuple(&ads);
    try std.testing.expectEqual(@as(usize, 4), abs.len);
    // A is [rank, in], B is [out, rank] for each enabled target.
    try std.testing.expectEqual(@as(usize, 16), abs[0].dim(.embed));
    try std.testing.expectEqual(@as(usize, 24), abs[1].dim(.q));
    try std.testing.expectEqual(@as(usize, 16), abs[2].dim(.embed));
    try std.testing.expectEqual(@as(usize, 8), abs[3].dim(.v));
}

test "a failed adapter build tears down only what was built" {
    var ctx: ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();

    // The second enabled target (v) gets an output dim below the rank, which
    // its adapter rejects; the first (q) must not leak. The testing allocator
    // is the assertion: a leak fails the test.
    const QV = lora_trainer.AdapterSet(qv, 0);
    var dims: lora_trainer.TargetDims = undefined;
    for (&dims) |*d| d.* = .{ 16, 16 };
    dims[2] = .{ 16, 2 };

    var ads: QV.LayerAdapters = undefined;
    try std.testing.expectError(
        error.InvalidRank,
        QV.initLayerAdapters(&ctx, &ads, dims, .{ .rank = 4, .alpha = 8 }, 1, 0),
    );
}

test "dropout seeds: null on eval, per-target on train, distinct across step/layer/family" {
    const A = lora_trainer.AdapterSet(qv, 0xAAAA_AAAA);
    const B = lora_trainer.AdapterSet(qv, 0xBBBB_BBBB); // a second family's domain

    // Eval (null step) draws nothing, so it never advances the noise.
    for (A.layerSeeds(7, 4, null, 0)) |s| try std.testing.expect(s == null);

    const s00 = A.layerSeeds(7, 4, 0, 0);
    const s01 = A.layerSeeds(7, 4, 0, 1);
    const s10 = A.layerSeeds(7, 4, 1, 0);
    for (s00) |s| try std.testing.expect(s != null);

    // Same inputs, same seeds: the stream is a pure function of them.
    try std.testing.expectEqualSlices(?u64, &s00, &A.layerSeeds(7, 4, 0, 0));

    // Distinct per target, per layer, per step, and per family domain.
    try std.testing.expect(s00[0].? != s00[2].?);
    try std.testing.expect(s00[0].? != s01[0].?);
    try std.testing.expect(s00[0].? != s10[0].?);
    try std.testing.expect(s00[0].? != B.layerSeeds(7, 4, 0, 0)[0].?);
}
