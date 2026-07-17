//! Multi-cartridge composition tests (Cartridges at Scale, arXiv 2606.04557)
//! over the synthetic tiny model: a single-part composition must reproduce
//! the single-cartridge forward bitwise, a two-part composition built from
//! one full capture must match the real prefill of the concatenated prefix
//! (the exact composition oracle: concat order, summed position offset, and
//! the end-aligned kernel in one shot), gradients must reach EVERY part,
//! serving through `writeComposedToCache` must match the trainer's composed
//! eval, and malformed compositions must be rejected.

const std = @import("std");
const fucina = @import("fucina");
const qwen3_train = @import("train.zig");
const cartridge = @import("../cartridge.zig");
const kv_cache = @import("../kv_cache.zig");
const scaffolding = @import("train_tests.zig");

const ExecContext = fucina.ExecContext;
const optim = fucina.optim;

const CartridgeTrainer = qwen3_train.Trainer(.{ .q = false, .v = false });
const no_lora = fucina.lora.Config{ .rank = 1, .alpha = 1 };

const full_tokens = [_]usize{ 14, 6, 25, 0, 13, 1, 26, 22, 5, 29, 33, 2 };
const p_a = 4;
const p_b = 4;
const prefix_len = p_a + p_b;
const suffix = full_tokens[prefix_len..];

/// Two cartridges from ONE capture of the concatenated prefix: part A holds
/// the rows of tokens 0..p_a-1 (positions 0..p_a-1), part B the rows of
/// tokens p_a..p_a+p_b-1 (positions p_a..) — so their composition IS the
/// real prefill's KV state, the exact oracle for the composed forward.
const SplitCartridges = struct {
    a: cartridge.Cartridge,
    b: cartridge.Cartridge,

    fn init(ctx: *ExecContext, allocator: std.mem.Allocator, trainer: anytype) !SplitCartridges {
        var cap = try trainer.captureKv(ctx, full_tokens[0..prefix_len]);
        defer cap.deinit();
        const cfg = trainer.model.config;
        const row = cfg.num_key_value_heads * cfg.head_dim;
        const n_layers = cap.k_rows.len;

        const k_slices = try allocator.alloc([]const f32, n_layers);
        defer allocator.free(k_slices);
        const v_slices = try allocator.alloc([]const f32, n_layers);
        defer allocator.free(v_slices);

        for (k_slices, v_slices, cap.k_rows, cap.v_rows) |*k, *v, k_full, v_full| {
            k.* = k_full[0 .. p_a * row];
            v.* = v_full[0 .. p_a * row];
        }
        var a = try cartridge.Cartridge.initFromRows(ctx, allocator, 1, p_a, cfg.num_key_value_heads, cfg.head_dim, k_slices, v_slices);
        errdefer a.deinit();

        for (k_slices, v_slices, cap.k_rows, cap.v_rows) |*k, *v, k_full, v_full| {
            k.* = k_full[p_a * row ..];
            v.* = v_full[p_a * row ..];
        }
        const b = try cartridge.Cartridge.initFromRows(ctx, allocator, 1, p_b, cfg.num_key_value_heads, cfg.head_dim, k_slices, v_slices);
        return .{ .a = a, .b = b };
    }

    fn deinit(self: *SplitCartridges) void {
        self.a.deinit();
        self.b.deinit();
    }
};

test "single-part composition is bitwise the single-cartridge forward" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var model = try scaffolding.buildTinyModel(&ctx, 3);
    defer model.deinit();
    var trainer = try CartridgeTrainer.init(&ctx, &model, no_lora, 7);
    defer trainer.deinit();

    var cart = try trainer.initCartridge(&ctx, full_tokens[0..prefix_len], 1);
    defer cart.deinit();

    var single = try trainer.evalLogitsExt(&ctx, suffix, .{ .cartridge = &cart });
    defer single.deinit();
    var composed = try trainer.evalLogitsExt(&ctx, suffix, .{ .cartridges = &.{&cart} });
    defer composed.deinit();

    // Same concat inputs, same position offset, same kernels: bitwise.
    try std.testing.expectEqualSlices(f32, try single.dataConst(), try composed.dataConst());
}

test "two-part composition matches the real prefill of the concatenated prefix" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try scaffolding.buildTinyModel(&ctx, 3);
    defer model.deinit();
    var trainer = try CartridgeTrainer.init(&ctx, &model, no_lora, 7);
    defer trainer.deinit();

    var teacher = try trainer.evalLogits(&ctx, &full_tokens);
    defer teacher.deinit();

    var split = try SplitCartridges.init(&ctx, allocator, &trainer);
    defer split.deinit();
    try std.testing.expectEqual(@as(usize, prefix_len), cartridge.composedP(&.{ &split.a, &split.b }));

    var student = try trainer.evalLogitsExt(&ctx, suffix, .{ .cartridges = &.{ &split.a, &split.b } });
    defer student.deinit();

    const vocab = model.config.vocab_size;
    const teacher_data = try teacher.dataConst();
    const student_data = try student.dataConst();
    try std.testing.expectEqual(@as(usize, suffix.len * vocab), student_data.len);
    for (student_data, teacher_data[prefix_len * vocab ..], 0..) |got, want, i| {
        const tol = 1e-5 + 1e-4 * @abs(want);
        if (@abs(got - want) > tol) {
            std.debug.print("composed prefill-equivalence mismatch at {d}: want {d} got {d}\n", .{ i, want, got });
            return error.ComposedEquivalenceMismatch;
        }
    }
}

test "composed distillation routes gradients into every part (sinks frozen)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try scaffolding.buildTinyModel(&ctx, 3);
    defer model.deinit();
    var trainer = try CartridgeTrainer.init(&ctx, &model, no_lora, 7);
    defer trainer.deinit();

    var split = try SplitCartridges.init(&ctx, allocator, &trainer);
    defer split.deinit();

    // Teacher top-k targets from the real-prefill forward, packed to
    // student-local positions.
    var teacher = try trainer.evalLogits(&ctx, &full_tokens);
    defer teacher.deinit();
    const vocab = model.config.vocab_size;
    const teacher_data = try teacher.dataConst();
    var builder = cartridge.TargetsBuilder.init(allocator);
    defer builder.deinit();
    for (1..suffix.len) |j| {
        const row = teacher_data[(prefix_len + j - 1) * vocab ..][0..vocab];
        try builder.appendRow(j, row, 5, 0.99);
    }

    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        var loss = try trainer.distillLossExt(&ctx, suffix, .{ .cartridges = &.{ &split.a, &split.b } }, builder.targets(), .{});
        defer loss.deinit();
        try loss.backward(&ctx);
    }

    // Both parts' trainable rows received gradients; sinks stay constants.
    inline for (.{ &split.a, &split.b }) |part| {
        try std.testing.expect(!part.layers[0].k_sink.?.requiresGrad());
        inline for (.{ "k", "v" }) |field| {
            var grad = (try @field(part.layers[0], field).grad(&ctx)) orelse return error.MissingComposedGrad;
            defer grad.deinit();
            var nonzero = false;
            for (grad.asRawTensor().dataConst()) |g| nonzero = nonzero or (g != 0);
            try std.testing.expect(nonzero);
        }
        part.zeroGrad();
    }
}

test "served composition in a KvCache matches the trainer's composed eval" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try scaffolding.buildTinyModel(&ctx, 3);
    defer model.deinit();
    var trainer = try CartridgeTrainer.init(&ctx, &model, no_lora, 7);
    defer trainer.deinit();

    var split = try SplitCartridges.init(&ctx, allocator, &trainer);
    defer split.deinit();

    var reference = try trainer.evalLogitsExt(&ctx, suffix, .{ .cartridges = &.{ &split.a, &split.b } });
    defer reference.deinit();

    var cache = try kv_cache.KvCache.init(
        &ctx,
        model.config.num_layers,
        model.config.num_key_value_heads,
        model.config.head_dim,
        prefix_len + suffix.len + 4,
    );
    defer cache.deinit();
    try cartridge.writeComposedToCache(&ctx, &.{ &split.a, &split.b }, &cache);
    try std.testing.expectEqual(@as(usize, prefix_len), cache.len);

    var served = try model.forwardStepAllLogits(&ctx, &cache, suffix, cache.len);
    defer served.deinit();
    try std.testing.expectEqual(@as(usize, prefix_len + suffix.len), cache.len);

    // f16 cache dtype + f16-KV kernels: approximate but tight (the same
    // envelope as the single-cartridge serving test).
    const want = try reference.dataConst();
    const got = try served.dataConst();
    try std.testing.expectEqual(want.len, got.len);
    for (want, got, 0..) |w, g, i| {
        const tol = 5e-2 + 5e-3 * @abs(w);
        if (@abs(w - g) > tol) {
            std.debug.print("composed serving mismatch at {d}: trainer {d} served {d}\n", .{ i, w, g });
            return error.ComposedServingMismatch;
        }
    }

    // A non-empty cache must be rejected.
    try std.testing.expectError(
        cartridge.Error.InvalidCartridge,
        cartridge.writeComposedToCache(&ctx, &.{ &split.a, &split.b }, &cache),
    );
}

test "packed composed forward matches per-sequence composed forwards" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try scaffolding.buildTinyModel(&ctx, 3);
    defer model.deinit();
    var trainer = try CartridgeTrainer.init(&ctx, &model, no_lora, 7);
    defer trainer.deinit();

    var split = try SplitCartridges.init(&ctx, allocator, &trainer);
    defer split.deinit();
    const parts: []const *const cartridge.Cartridge = &.{ &split.a, &split.b };

    const s1 = full_tokens[8..10];
    const s2 = full_tokens[10..12];
    var one = try trainer.evalLogitsExt(&ctx, s1, .{ .cartridges = parts });
    defer one.deinit();
    var two = try trainer.evalLogitsExt(&ctx, s2, .{ .cartridges = parts });
    defer two.deinit();

    var packed_logits = try trainer.evalLogitsExt(&ctx, full_tokens[8..12], .{
        .cartridges = parts,
        .packed_segments = &.{ s1.len, s2.len },
    });
    defer packed_logits.deinit();
    defer trainer.freeTransientRope();

    const vocab = model.config.vocab_size;
    const got = try packed_logits.dataConst();
    const want_one = try one.dataConst();
    const want_two = try two.dataConst();
    for (got[0 .. s1.len * vocab], want_one, 0..) |g, w, i| {
        if (@abs(g - w) > 1e-5 + 1e-4 * @abs(w)) {
            std.debug.print("packed composed seg1 mismatch at {d}: want {d} got {d}\n", .{ i, w, g });
            return error.PackedComposedMismatch;
        }
    }
    for (got[s1.len * vocab ..], want_two, 0..) |g, w, i| {
        if (@abs(g - w) > 1e-5 + 1e-4 * @abs(w)) {
            std.debug.print("packed composed seg2 mismatch at {d}: want {d} got {d}\n", .{ i, w, g });
            return error.PackedComposedMismatch;
        }
    }
}

test "composed forward rejects malformed part lists" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try scaffolding.buildTinyModel(&ctx, 3);
    defer model.deinit();
    var trainer = try CartridgeTrainer.init(&ctx, &model, no_lora, 7);
    defer trainer.deinit();

    var good = try trainer.initCartridge(&ctx, full_tokens[0..p_a], 1);
    defer good.deinit();

    // Empty part list.
    try std.testing.expectError(
        qwen3_train.Error.InvalidCartridge,
        trainer.evalLogitsExt(&ctx, suffix, .{ .cartridges = &.{} }),
    );

    // Both the single and the composed field set.
    try std.testing.expectError(
        qwen3_train.Error.InvalidCartridge,
        trainer.evalLogitsExt(&ctx, suffix, .{ .cartridge = &good, .cartridges = &.{&good} }),
    );

    // A part with foreign KV geometry.
    var bad = try cartridge.Cartridge.initRandom(
        &ctx,
        allocator,
        model.config.num_layers,
        1,
        p_b,
        model.config.num_key_value_heads,
        model.config.head_dim / 2,
        1,
        0.02,
    );
    defer bad.deinit();
    try std.testing.expectError(
        qwen3_train.Error.InvalidCartridge,
        trainer.evalLogitsExt(&ctx, suffix, .{ .cartridges = &.{ &good, &bad } }),
    );
    try std.testing.expectError(
        cartridge.Error.InvalidCartridge,
        cartridge.validateComposition(&.{ &good, &bad }),
    );

    // Composition under checkpointed layers.
    trainer.checkpoint_layers = true;
    defer trainer.checkpoint_layers = false;
    try std.testing.expectError(
        qwen3_train.Error.CartridgeCheckpointUnsupported,
        trainer.evalLogitsExt(&ctx, suffix, .{ .cartridges = &.{&good} }),
    );
}
