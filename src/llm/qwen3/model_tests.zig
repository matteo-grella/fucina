//! Behavioral tests for the qwen3 model's batched multi-sequence decode
//! (`forwardStepBatch`): per-stream logits parity vs sequential
//! `forwardStep` — bitwise, since the batch stays below the m-dependent
//! kernel thresholds — over ragged stream positions, for both the f16 and
//! q8_0 cache arms, plus the input-validation error paths. Force-imported
//! by `model.zig`'s test block.

const std = @import("std");
const fucina = @import("fucina");
const qwen3 = @import("model.zig");
const kv_cache = @import("../kv_cache.zig");
const scaffolding = @import("train_tests.zig");

const ExecContext = fucina.ExecContext;
const KvCache = kv_cache.KvCache;

/// Real-qwen3-shaped tiny geometry (adjacent 2:1 GQA — the pair attention
/// arm) with a q8_0-compatible `head_dim % 32 == 0`; the general attention
/// arm is covered bitwise at the exec level (attention_tests.zig).
const wide_config = qwen3.Config{
    .vocab_size = 64,
    .hidden_size = 32,
    .intermediate_size = 64,
    .num_layers = 2,
    .num_attention_heads = 4,
    .num_key_value_heads = 2,
    .head_dim = 32,
    .rms_norm_eps = 1e-6,
    .rope_theta = 10_000,
};

fn argmaxRow(row: []const f32) usize {
    var best: usize = 0;
    for (row, 0..) |x, i| {
        if (x > row[best]) best = i;
    }
    return best;
}

/// Ragged prompts, then `steps` lockstep decode steps: every batch logits
/// row must be BITWISE identical to the same stream decoded alone through
/// `forwardStep`, and the caches must advance in lockstep.
fn checkBatchDecodeParity(dtype: kv_cache.KvDtype) !void {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try scaffolding.buildTinyModelWithConfig(&ctx, wide_config, 0xB47C);
    defer model.deinit();
    const cfg = model.config;

    const prompts = [_][]const usize{
        &.{ 5, 9, 13 },
        &.{ 2, 33 },
        &.{ 60, 21, 8, 42, 17 },
    };
    const n = prompts.len;
    const capacity = 32;
    const steps = 4;

    var batch_caches: [n]KvCache = undefined;
    var seq_caches: [n]KvCache = undefined;
    var inited: usize = 0;
    defer for (0..inited) |i| {
        batch_caches[i].deinit();
        seq_caches[i].deinit();
    };
    for (0..n) |i| {
        batch_caches[i] = try KvCache.initWithDtype(&ctx, cfg.num_layers, cfg.num_key_value_heads, cfg.head_dim, capacity, dtype);
        errdefer batch_caches[i].deinit();
        seq_caches[i] = try KvCache.initWithDtype(&ctx, cfg.num_layers, cfg.num_key_value_heads, cfg.head_dim, capacity, dtype);
        inited += 1;
    }

    // Identical per-stream prefills on both sides seed identical caches.
    var tokens: [n]usize = undefined;
    for (prompts, 0..) |prompt, i| {
        var batch_logits = try model.forwardStep(&ctx, &batch_caches[i], prompt, 0);
        defer batch_logits.deinit();
        var seq_logits = try model.forwardStep(&ctx, &seq_caches[i], prompt, 0);
        defer seq_logits.deinit();
        try std.testing.expectEqualSlices(f32, try seq_logits.dataConst(), try batch_logits.dataConst());
        tokens[i] = argmaxRow(try seq_logits.dataConst());
    }

    var cache_ptrs: [n]*KvCache = undefined;
    for (&cache_ptrs, &batch_caches) |*ptr, *cache| ptr.* = cache;

    for (0..steps) |_| {
        var batch_logits = try model.forwardStepBatch(&ctx, &cache_ptrs, &tokens);
        defer batch_logits.deinit();
        const batch_rows = try batch_logits.dataConst();
        try std.testing.expectEqual(n, batch_logits.dim(.seq));

        for (0..n) |i| {
            var single = [_]usize{tokens[i]};
            var seq_logits = try model.forwardStep(&ctx, &seq_caches[i], &single, seq_caches[i].len);
            defer seq_logits.deinit();
            const seq_row = try seq_logits.dataConst();
            const batch_row = batch_rows[i * cfg.vocab_size ..][0..cfg.vocab_size];
            try std.testing.expectEqualSlices(f32, seq_row, batch_row);
            try std.testing.expectEqual(seq_caches[i].len, batch_caches[i].len);
            tokens[i] = argmaxRow(seq_row);
        }
    }
}

test "forwardStepBatch rows == sequential forwardStep (f16 cache, ragged positions)" {
    try checkBatchDecodeParity(.f16);
}

test "forwardStepBatch rows == sequential forwardStep (q8_0 cache, ragged positions)" {
    try checkBatchDecodeParity(.q8_0);
}

test "forwardStepBatch validates its cache batch" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try scaffolding.buildTinyModelWithConfig(&ctx, wide_config, 0xB47C);
    defer model.deinit();
    const cfg = model.config;

    var cache_a = try KvCache.initWithDtype(&ctx, cfg.num_layers, cfg.num_key_value_heads, cfg.head_dim, 4, .f16);
    defer cache_a.deinit();
    var cache_b = try KvCache.initWithDtype(&ctx, cfg.num_layers, cfg.num_key_value_heads, cfg.head_dim, 4, .q8_0);
    defer cache_b.deinit();

    // Batch/token count mismatch and the empty batch.
    var one_cache = [_]*KvCache{&cache_a};
    try std.testing.expectError(qwen3.Error.InvalidSequenceLength, model.forwardStepBatch(&ctx, &one_cache, &.{ 1, 2 }));
    try std.testing.expectError(qwen3.Error.InvalidSequenceLength, model.forwardStepBatch(&ctx, &.{}, &.{}));

    // Mixed cache dtypes and a duplicated cache.
    var mixed = [_]*KvCache{ &cache_a, &cache_b };
    try std.testing.expectError(qwen3.Error.MismatchedKvCaches, model.forwardStepBatch(&ctx, &mixed, &.{ 1, 2 }));
    var duplicated = [_]*KvCache{ &cache_a, &cache_a };
    try std.testing.expectError(qwen3.Error.MismatchedKvCaches, model.forwardStepBatch(&ctx, &duplicated, &.{ 1, 2 }));

    // A full cache overflows before any compute.
    var full = try KvCache.initWithDtype(&ctx, cfg.num_layers, cfg.num_key_value_heads, cfg.head_dim, 2, .f16);
    defer full.deinit();
    var prefill = try model.forwardStep(&ctx, &full, &.{ 1, 2 }, 0);
    prefill.deinit();
    var full_batch = [_]*KvCache{&full};
    try std.testing.expectError(kv_cache.Error.KvCacheOverflow, model.forwardStepBatch(&ctx, &full_batch, &.{3}));
}
