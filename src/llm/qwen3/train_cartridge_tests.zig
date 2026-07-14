//! Cartridge-on-qwen3 trainer tests over the synthetic tiny model: the
//! init-equals-prefill equivalence gate (an untrained corpus-token cartridge
//! must reproduce the real prefill's logits), distillation training smoke
//! (teacher top-k targets pull a random-init cartridge's loss down through
//! the frozen stack), and the state-dict roundtrip.

const std = @import("std");
const fucina = @import("fucina");
const qwen3_train = @import("train.zig");
const cartridge = @import("../cartridge.zig");
const kv_cache = @import("../kv_cache.zig");
const scaffolding = @import("train_tests.zig");

const ExecContext = fucina.ExecContext;
const optim = fucina.optim;

/// No LoRA targets: the base model is fully frozen and the cartridge rows
/// are the only trainable parameters.
const CartridgeTrainer = qwen3_train.Trainer(.{ .q = false, .v = false });
const no_lora = fucina.lora.Config{ .rank = 1, .alpha = 1 };

const full_tokens = [_]usize{ 14, 6, 25, 0, 13, 1, 26, 22, 5, 29, 33, 2 };
const prefix_len = 5;
const suffix = full_tokens[prefix_len..];

test "corpus-init cartridge is logit-equivalent to a real prefill" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var model = try scaffolding.buildTinyModel(&ctx, 3);
    defer model.deinit();
    var trainer = try CartridgeTrainer.init(&ctx, &model, no_lora, 7);
    defer trainer.deinit();

    var teacher = try trainer.evalLogits(&ctx, &full_tokens);
    defer teacher.deinit();

    // Capture the model's own K/V rows for the prefix; zero training steps.
    var cart = try trainer.initCartridge(&ctx, full_tokens[0..prefix_len], 1);
    defer cart.deinit();
    try std.testing.expectEqual(@as(usize, prefix_len), cart.p);
    try std.testing.expectEqual(@as(usize, 1), cart.frozen_prefix);

    // Suffix behind the cartridge (positions p..) must match the suffix
    // rows of the full-sequence forward: the trained-KV parameterization
    // starts EXACTLY at the real prefill.
    var student = try trainer.evalLogitsExt(&ctx, suffix, .{ .cartridge = &cart });
    defer student.deinit();

    const vocab = model.config.vocab_size;
    const teacher_data = try teacher.dataConst();
    const student_data = try student.dataConst();
    try std.testing.expectEqual(@as(usize, suffix.len * vocab), student_data.len);
    for (student_data, teacher_data[prefix_len * vocab ..], 0..) |got, want, i| {
        const tol = 1e-5 + 1e-4 * @abs(want);
        if (@abs(got - want) > tol) {
            std.debug.print("prefill-equivalence mismatch at {d}: want {d} got {d}\n", .{ i, want, got });
            return error.PrefillEquivalenceMismatch;
        }
    }
}

test "distillation pulls a random cartridge toward the teacher (sink frozen)" {
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

    // Teacher = the same frozen model WITH the prefix really in context;
    // its top-k rows over the suffix are the distillation targets
    // (positions are student-packed: token at full position p+j sits at
    // student position j and is predicted from teacher row p+j-1).
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

    // Random-vector init (the paper's worst baseline) so the loss has room
    // to fall; the base model stays frozen throughout.
    var cart = try cartridge.Cartridge.initRandom(
        &ctx,
        allocator,
        model.config.num_layers,
        1,
        prefix_len,
        model.config.num_key_value_heads,
        model.config.head_dim,
        123,
        0.05,
    );
    defer cart.deinit();

    var opt = optim.AdamW.init(allocator, .{ .lr = 2e-2, .weight_decay = 0 });
    defer opt.deinit();
    try cart.registerParams(&opt);

    const sink_before = try allocator.dupe(f32, try cart.layers[0].k_sink.?.dataConst());
    defer allocator.free(sink_before);
    const k_before = try allocator.dupe(f32, try cart.layers[0].k.dataConst());
    defer allocator.free(k_before);

    var first: f32 = 0;
    var last: f32 = 0;
    for (0..8) |step_i| {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        var loss = try trainer.distillLoss(&ctx, suffix, &cart, builder.targets(), null, .{});
        defer loss.deinit();
        try loss.backward(&ctx);
        const value = try loss.item();
        if (step_i == 0) first = value;
        last = value;
        try opt.step(&ctx);
        opt.zeroGrad();
    }
    try std.testing.expect(last < first * 0.9);

    // The attention sink never moves; the trainable rows do.
    try std.testing.expectEqualSlices(f32, sink_before, try cart.layers[0].k_sink.?.dataConst());
    try std.testing.expect(!std.mem.eql(f32, k_before, try cart.layers[0].k.dataConst()));
}

test "cartridge state roundtrips through the safetensors dict" {
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

    var cart = try trainer.initCartridge(&ctx, full_tokens[0..prefix_len], 1);
    defer cart.deinit();

    var buf: [64 * 1024]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try cart.saveState(&writer);
    const written = writer.buffered();

    // Same geometry, different values: the load must overwrite everything,
    // frozen sinks included.
    var fresh = try cartridge.Cartridge.initRandom(
        &ctx,
        allocator,
        model.config.num_layers,
        1,
        prefix_len,
        model.config.num_key_value_heads,
        model.config.head_dim,
        999,
        0.02,
    );
    defer fresh.deinit();
    var reader = std.Io.Reader.fixed(written);
    try fresh.loadState(&reader);

    for (cart.layers, fresh.layers) |*want, *got| {
        try std.testing.expectEqualSlices(f32, try want.k_sink.?.dataConst(), try got.k_sink.?.dataConst());
        try std.testing.expectEqualSlices(f32, try want.v_sink.?.dataConst(), try got.v_sink.?.dataConst());
        try std.testing.expectEqualSlices(f32, try want.k.dataConst(), try got.k.dataConst());
        try std.testing.expectEqualSlices(f32, try want.v.dataConst(), try got.v.dataConst());
    }

    // Geometry-blind reload: initFromStateDict recovers layer count, p,
    // frozen prefix, and kv dims from the safetensors header alone.
    var rebuilt = try cartridge.Cartridge.initFromStateDict(&ctx, allocator, written);
    defer rebuilt.deinit();
    try std.testing.expectEqual(cart.p, rebuilt.p);
    try std.testing.expectEqual(cart.frozen_prefix, rebuilt.frozen_prefix);
    try std.testing.expectEqual(cart.layers.len, rebuilt.layers.len);
    for (cart.layers, rebuilt.layers) |*want, *got| {
        try std.testing.expectEqualSlices(f32, try want.k_sink.?.dataConst(), try got.k_sink.?.dataConst());
        try std.testing.expectEqualSlices(f32, try want.v.dataConst(), try got.v.dataConst());
    }
}

test "served cartridge in a KvCache matches the trainer's cartridge eval" {
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

    var reference = try trainer.evalLogitsExt(&ctx, suffix, .{ .cartridge = &cart });
    defer reference.deinit();

    // Serve: preload the cartridge into an empty inference cache, then run
    // the production decode path over the suffix at positions p...
    var cache = try kv_cache.KvCache.init(
        &ctx,
        model.config.num_layers,
        model.config.num_key_value_heads,
        model.config.head_dim,
        prefix_len + suffix.len + 4,
    );
    defer cache.deinit();
    try cart.writeToCache(&ctx, &cache);
    try std.testing.expectEqual(@as(usize, prefix_len), cache.len);

    var served = try model.forwardStepAllLogits(&ctx, &cache, suffix, cache.len);
    defer served.deinit();
    try std.testing.expectEqual(@as(usize, prefix_len + suffix.len), cache.len);

    // The inference path stores the prefix in the cache dtype (f16) and runs
    // the f16-KV attention kernels, so parity is approximate — but tight.
    const want = try reference.dataConst();
    const got = try served.dataConst();
    try std.testing.expectEqual(want.len, got.len);
    for (want, got, 0..) |w, g, i| {
        const tol = 5e-2 + 5e-3 * @abs(w);
        if (@abs(w - g) > tol) {
            std.debug.print("serving mismatch at {d}: trainer {d} served {d}\n", .{ i, w, g });
            return error.ServingParityMismatch;
        }
    }

    // A non-empty cache must be rejected (the cartridge IS the prefix).
    try std.testing.expectError(cartridge.Error.InvalidCartridge, cart.writeToCache(&ctx, &cache));
}

test "evalLogitsRows matches the corresponding full-logits rows" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var model = try scaffolding.buildTinyModel(&ctx, 3);
    defer model.deinit();
    var trainer = try CartridgeTrainer.init(&ctx, &model, no_lora, 7);
    defer trainer.deinit();

    var full = try trainer.evalLogits(&ctx, &full_tokens);
    defer full.deinit();
    const rows = [_]usize{ 2, 7, 11 };
    var picked = try trainer.evalLogitsRows(&ctx, &full_tokens, &rows, .{});
    defer picked.deinit();

    const vocab = model.config.vocab_size;
    const full_data = try full.dataConst();
    const picked_data = try picked.dataConst();
    try std.testing.expectEqual(@as(usize, rows.len * vocab), picked_data.len);
    for (rows, 0..) |row, i| {
        try std.testing.expectEqualSlices(
            f32,
            full_data[row * vocab ..][0..vocab],
            picked_data[i * vocab ..][0..vocab],
        );
    }
}

test "cartridge forward rejects geometry and checkpoint misuse" {
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

    // Wrong kv geometry (head_dim halved).
    var bad = try cartridge.Cartridge.initRandom(
        &ctx,
        allocator,
        model.config.num_layers,
        1,
        prefix_len,
        model.config.num_key_value_heads,
        model.config.head_dim / 2,
        1,
        0.02,
    );
    defer bad.deinit();
    try std.testing.expectError(
        qwen3_train.Error.InvalidCartridge,
        trainer.evalLogitsExt(&ctx, suffix, .{ .cartridge = &bad }),
    );

    // Checkpointed layers cannot host a cartridge (or a capture).
    var good = try trainer.initCartridge(&ctx, full_tokens[0..prefix_len], 1);
    defer good.deinit();
    trainer.checkpoint_layers = true;
    defer trainer.checkpoint_layers = false;
    try std.testing.expectError(
        qwen3_train.Error.CartridgeCheckpointUnsupported,
        trainer.evalLogitsExt(&ctx, suffix, .{ .cartridge = &good }),
    );
    try std.testing.expectError(
        qwen3_train.Error.CartridgeCheckpointUnsupported,
        trainer.captureKv(&ctx, full_tokens[0..prefix_len]),
    );
}

test "packed forward matches per-sequence forwards (no cartridge)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var model = try scaffolding.buildTinyModel(&ctx, 3);
    defer model.deinit();
    var trainer = try CartridgeTrainer.init(&ctx, &model, no_lora, 7);
    defer trainer.deinit();

    const s1 = full_tokens[0..5];
    const s2 = full_tokens[5..12];
    var one = try trainer.evalLogits(&ctx, s1);
    defer one.deinit();
    var two = try trainer.evalLogits(&ctx, s2);
    defer two.deinit();

    var packed_logits = try trainer.evalLogitsExt(&ctx, &full_tokens, .{ .packed_segments = &.{ s1.len, s2.len } });
    defer packed_logits.deinit();
    defer trainer.freeTransientRope();

    const vocab = model.config.vocab_size;
    const got = try packed_logits.dataConst();
    const want_one = try one.dataConst();
    const want_two = try two.dataConst();
    try std.testing.expectEqual(@as(usize, full_tokens.len * vocab), got.len);
    for (got[0 .. s1.len * vocab], want_one, 0..) |g, w, i| {
        if (@abs(g - w) > 1e-5 + 1e-4 * @abs(w)) {
            std.debug.print("packed seg1 mismatch at {d}: want {d} got {d}\n", .{ i, w, g });
            return error.PackedMismatch;
        }
    }
    for (got[s1.len * vocab ..], want_two, 0..) |g, w, i| {
        if (@abs(g - w) > 1e-5 + 1e-4 * @abs(w)) {
            std.debug.print("packed seg2 mismatch at {d}: want {d} got {d}\n", .{ i, w, g });
            return error.PackedMismatch;
        }
    }
}

test "packed forward matches per-sequence forwards behind a cartridge" {
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

    // Two independent "conversations": each must see the shared prefix plus
    // its OWN causal rows at RoPE positions p.. — never the other segment.
    const s1 = full_tokens[5..9];
    const s2 = full_tokens[9..12];
    var one = try trainer.evalLogitsExt(&ctx, s1, .{ .cartridge = &cart });
    defer one.deinit();
    var two = try trainer.evalLogitsExt(&ctx, s2, .{ .cartridge = &cart });
    defer two.deinit();

    const packed_tokens = full_tokens[5..12];
    var packed_logits = try trainer.evalLogitsExt(&ctx, packed_tokens, .{
        .cartridge = &cart,
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
            std.debug.print("packed cart seg1 mismatch at {d}: want {d} got {d}\n", .{ i, w, g });
            return error.PackedMismatch;
        }
    }
    for (got[s1.len * vocab ..], want_two, 0..) |g, w, i| {
        if (@abs(g - w) > 1e-5 + 1e-4 * @abs(w)) {
            std.debug.print("packed cart seg2 mismatch at {d}: want {d} got {d}\n", .{ i, w, g });
            return error.PackedMismatch;
        }
    }
}

test "fused distill tail matches the composed logits tail (loss + cartridge grads)" {
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
    const vocab = model.config.vocab_size;
    const teacher_data = try teacher.dataConst();
    var builder = cartridge.TargetsBuilder.init(allocator);
    defer builder.deinit();
    for (1..suffix.len) |j| {
        const row = teacher_data[(prefix_len + j - 1) * vocab ..][0..vocab];
        try builder.appendRow(j, row, 5, 0.99);
    }

    // Same cartridge values for both arms; grads compared leaf by leaf.
    const arms = [2]bool{ false, true };
    var losses: [2]f32 = undefined;
    var grads: [2][]f32 = undefined;
    defer for (&grads) |g| allocator.free(g);
    for (arms, 0..) |fused, arm_i| {
        qwen3_train.setFusedDistill(fused);
        var cart = try cartridge.Cartridge.initRandom(
            &ctx,
            allocator,
            model.config.num_layers,
            1,
            prefix_len,
            model.config.num_key_value_heads,
            model.config.head_dim,
            123,
            0.05,
        );
        defer cart.deinit();
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        var loss = try trainer.distillLoss(&ctx, suffix, &cart, builder.targets(), null, .{ .loss_scale = 0.5 });
        defer loss.deinit();
        try loss.backward(&ctx);
        losses[arm_i] = try loss.item();
        var gk = (try cart.layers[0].k.grad(&ctx)).?;
        defer gk.deinit();
        grads[arm_i] = try allocator.dupe(f32, try gk.dataConst());
    }
    qwen3_train.setFusedDistill(null);

    try std.testing.expectApproxEqAbs(losses[0], losses[1], 1e-5);
    try std.testing.expectEqual(grads[0].len, grads[1].len);
    for (grads[0], grads[1]) |want, got| {
        try std.testing.expect(@abs(got - want) <= 1e-6 + 1e-3 * @abs(want));
    }
}

test "packed distillation equals accumulated per-conversation gradients" {
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

    var cart = try trainer.initCartridge(&ctx, full_tokens[0..prefix_len], 1);
    defer cart.deinit();

    const s1 = full_tokens[5..9]; // 4 tokens, 2 supervised entries
    const s2 = full_tokens[9..12]; // 3 tokens, 3 supervised entries
    const t1 = cartridge.DistillTargets{
        .positions = &.{ 1, 3 },
        .tokens = &.{ 2, 30 },
        .logprobs = &.{ -0.35667494, -1.20397280 },
    };
    const t2 = cartridge.DistillTargets{
        .positions = &.{ 1, 2, 2 },
        .tokens = &.{ 7, 11, 40 },
        .logprobs = &.{ -0.10536052, -0.69314718, -1.60943791 },
    };
    const n_total: f32 = 5;

    // Packed: one forward/backward, targets at packed positions, mean over
    // all five entries.
    const packed_targets = cartridge.DistillTargets{
        .positions = &.{ 1, 3, 1 + s1.len, 2 + s1.len, 2 + s1.len },
        .tokens = &.{ 2, 30, 7, 11, 40 },
        .logprobs = &.{ -0.35667494, -1.20397280, -0.10536052, -0.69314718, -1.60943791 },
    };
    var packed_loss_value: f32 = 0;
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        var loss = try trainer.distillLoss(&ctx, full_tokens[5..12], &cart, packed_targets, &.{ s1.len, s2.len }, .{});
        defer loss.deinit();
        try loss.backward(&ctx);
        packed_loss_value = try loss.item();
    }
    trainer.freeTransientRope();
    var packed_gk = (try cart.layers[0].k.grad(&ctx)).?;
    defer packed_gk.deinit();
    var packed_gv = (try cart.layers[0].v.grad(&ctx)).?;
    defer packed_gv.deinit();
    cart.zeroGrad();

    // Sequential: two forwards/backwards, .sum reduction scaled by 1/5 —
    // exactly the packed mean; gradients accumulate across the two passes.
    var seq_loss_value: f32 = 0;
    inline for (.{ .{ s1, t1 }, .{ s2, t2 } }) |pair| {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        var loss = try trainer.distillLoss(&ctx, pair[0], &cart, pair[1], null, .{
            .reduction = .sum,
            .loss_scale = 1.0 / n_total,
        });
        defer loss.deinit();
        try loss.backward(&ctx);
        seq_loss_value += try loss.item();
    }
    var seq_gk = (try cart.layers[0].k.grad(&ctx)).?;
    defer seq_gk.deinit();
    var seq_gv = (try cart.layers[0].v.grad(&ctx)).?;
    defer seq_gv.deinit();
    cart.zeroGrad();

    try std.testing.expect(@abs(packed_loss_value - seq_loss_value) <= 1e-5 + 1e-4 * @abs(seq_loss_value));
    const pk = packed_gk.asRawTensor().dataConst();
    const sk = seq_gk.asRawTensor().dataConst();
    const pv = packed_gv.asRawTensor().dataConst();
    const sv = seq_gv.asRawTensor().dataConst();
    for (pk, sk, 0..) |g, w, i| {
        if (@abs(g - w) > 1e-6 + 1e-3 * @abs(w)) {
            std.debug.print("packed grad-k mismatch at {d}: want {d} got {d}\n", .{ i, w, g });
            return error.PackedGradMismatch;
        }
    }
    for (pv, sv, 0..) |g, w, i| {
        if (@abs(g - w) > 1e-6 + 1e-3 * @abs(w)) {
            std.debug.print("packed grad-v mismatch at {d}: want {d} got {d}\n", .{ i, w, g });
            return error.PackedGradMismatch;
        }
    }
}

test "packed forward rejects malformed segment specs" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var model = try scaffolding.buildTinyModel(&ctx, 3);
    defer model.deinit();
    var trainer = try CartridgeTrainer.init(&ctx, &model, no_lora, 7);
    defer trainer.deinit();

    // Lengths must be nonzero and sum to the token count.
    try std.testing.expectError(
        qwen3_train.Error.InvalidPacking,
        trainer.evalLogitsExt(&ctx, &full_tokens, .{ .packed_segments = &.{ 5, 5 } }),
    );
    try std.testing.expectError(
        qwen3_train.Error.InvalidPacking,
        trainer.evalLogitsExt(&ctx, &full_tokens, .{ .packed_segments = &.{ 0, 12 } }),
    );
}

test "forwardStepBatchSpans matches per-stream forwards (ragged batch)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var model = try scaffolding.buildTinyModel(&ctx, 3);
    defer model.deinit();
    const cfg = model.config;

    // Two streams with different cache prefixes and different span lengths.
    const prompt_a = full_tokens[0..4];
    const prompt_b = full_tokens[4..7];
    const span_a = full_tokens[7..10]; // 3 tokens
    const span_b = full_tokens[10..12]; // 2 tokens

    var cache_a = try kv_cache.KvCache.init(&ctx, cfg.num_layers, cfg.num_key_value_heads, cfg.head_dim, 32);
    defer cache_a.deinit();
    var cache_b = try kv_cache.KvCache.init(&ctx, cfg.num_layers, cfg.num_key_value_heads, cfg.head_dim, 32);
    defer cache_b.deinit();
    {
        var la = try model.forwardStep(&ctx, &cache_a, prompt_a, 0);
        la.deinit();
        var lb = try model.forwardStep(&ctx, &cache_b, prompt_b, 0);
        lb.deinit();
    }

    var span_tokens: [5]usize = undefined;
    @memcpy(span_tokens[0..3], span_a);
    @memcpy(span_tokens[3..5], span_b);
    var batch = try model.forwardStepBatchSpans(&ctx, &.{ &cache_a, &cache_b }, &span_tokens, &.{ 3, 2 });
    defer batch.deinit();
    try std.testing.expectEqual(prompt_a.len + span_a.len, cache_a.len);
    try std.testing.expectEqual(prompt_b.len + span_b.len, cache_b.len);

    // Reference: each stream alone through the standard span forward.
    var ref_cache_a = try kv_cache.KvCache.init(&ctx, cfg.num_layers, cfg.num_key_value_heads, cfg.head_dim, 32);
    defer ref_cache_a.deinit();
    var ref_cache_b = try kv_cache.KvCache.init(&ctx, cfg.num_layers, cfg.num_key_value_heads, cfg.head_dim, 32);
    defer ref_cache_b.deinit();
    {
        var la = try model.forwardStep(&ctx, &ref_cache_a, prompt_a, 0);
        la.deinit();
        var lb = try model.forwardStep(&ctx, &ref_cache_b, prompt_b, 0);
        lb.deinit();
    }
    var ref_a = try model.forwardStepAllLogits(&ctx, &ref_cache_a, span_a, ref_cache_a.len);
    defer ref_a.deinit();
    var ref_b = try model.forwardStepAllLogits(&ctx, &ref_cache_b, span_b, ref_cache_b.len);
    defer ref_b.deinit();

    const vocab = cfg.vocab_size;
    const got = try batch.dataConst();
    const want_a = try ref_a.dataConst();
    const want_b = try ref_b.dataConst();
    try std.testing.expectEqual(@as(usize, 5 * vocab), got.len);
    for (got[0 .. 3 * vocab], want_a, 0..) |g, w, i| {
        if (@abs(g - w) > 1e-4 + 1e-3 * @abs(w)) {
            std.debug.print("spans stream-a mismatch at {d}: want {d} got {d}\n", .{ i, w, g });
            return error.SpansMismatch;
        }
    }
    for (got[3 * vocab ..], want_b, 0..) |g, w, i| {
        if (@abs(g - w) > 1e-4 + 1e-3 * @abs(w)) {
            std.debug.print("spans stream-b mismatch at {d}: want {d} got {d}\n", .{ i, w, g });
            return error.SpansMismatch;
        }
    }
}
