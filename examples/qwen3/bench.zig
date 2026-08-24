//! Timing primitives of the Qwen3 runner: the prefill+decode pass every
//! generation/bench mode is built on, the mean±std stat line, the forward
//! profile printers, and the `--spec-bench` verify-economics probe.
const std = @import("std");
const fucina = @import("fucina");
const models = @import("fucina_models");
const util = @import("util.zig");

const nowNs = util.nowNs;
const seconds = util.seconds;
const millis = util.millis;
const millisI128 = util.millisI128;

pub const PassResult = struct { prefill_ns: i96, decode_ns: i96, decode_steps: usize, produced: usize };

/// One prefill+decode pass into the (reset) cache: prefill `tokens`, then decode
/// greedily up to `max_new`/`stop_token`, timing prefill and decode separately.
pub fn benchOnePass(
    io: std.Io,
    ctx: *fucina.ExecContext,
    model: *const models.qwen3.model.Model,
    cache: *models.text.kv_cache.KvCache,
    tokens: []const usize,
    out: []usize,
    history: []usize,
    sampler_cfg: models.text.sampler.Config,
    max_new: usize,
    stop_token: ?usize,
    profile_prefill: bool,
    profile_decode: bool,
    prefill_profile: ?*models.qwen3.model.ForwardProfile,
    decode_profile: ?*models.qwen3.model.ForwardProfile,
    processor: ?models.text.sampler.LogitProcessor,
) !PassResult {
    cache.reset();
    var sampler = models.text.sampler.Sampler.init(sampler_cfg);
    sampler.processor = processor;
    if (processor) |p| try p.reset(); // fresh grammar state per pass
    @memcpy(history[0..tokens.len], tokens);
    var hist_len = tokens.len;

    const prefill_start = nowNs(io);
    var logits = if (profile_prefill)
        try model.forwardStepProfiled(ctx, io, cache, tokens, 0, prefill_profile.?)
    else
        try model.forwardStep(ctx, cache, tokens, 0);
    const prefill_ns = nowNs(io) - prefill_start;
    out[0] = try sampler.next(ctx, &logits, history[0..hist_len]);
    history[hist_len] = out[0];
    hist_len += 1;
    var produced: usize = 1;

    const decode_start = nowNs(io);
    while (produced < max_new) : (produced += 1) {
        if (stop_token) |stop| if (out[produced - 1] == stop) break;
        const fresh = if (profile_decode)
            try model.forwardStepProfiled(ctx, io, cache, out[produced - 1 ..][0..1], cache.len(), decode_profile.?)
        else
            try model.forwardStep(ctx, cache, out[produced - 1 ..][0..1], cache.len());
        logits.deinit();
        logits = fresh;
        out[produced] = try sampler.next(ctx, &logits, history[0..hist_len]);
        history[hist_len] = out[produced];
        hist_len += 1;
    }
    const decode_ns = nowNs(io) - decode_start;
    logits.deinit();
    return .{ .prefill_ns = prefill_ns, .decode_ns = decode_ns, .decode_steps = produced - 1, .produced = produced };
}

pub fn printBenchStat(stdout: anytype, label: []const u8, vals: []const f64) !void {
    var min: f64 = vals[0];
    var max: f64 = vals[0];
    var sum: f64 = 0;
    for (vals) |v| {
        if (v < min) min = v;
        if (v > max) max = v;
        sum += v;
    }
    const mean = sum / @as(f64, @floatFromInt(vals.len));
    var ss: f64 = 0;
    for (vals) |v| {
        const d = v - mean;
        ss += d * d;
    }
    const std_dev = @sqrt(ss / @as(f64, @floatFromInt(vals.len)));
    try stdout.print("{s}: {d:.2} ± {d:.2} tok/s  (min {d:.2}, max {d:.2})\n", .{ label, mean, std_dev, min, max });
}

pub fn printProfile(stdout: anytype, profile: *const models.qwen3.model.ForwardProfile, repeat: usize) !void {
    try printProfileLabeled(stdout, "profile avg ms", profile, @floatFromInt(repeat));
}

pub fn printProfileLabeled(stdout: anytype, label: []const u8, profile: *const models.qwen3.model.ForwardProfile, denom: f64) !void {
    try stdout.print("{s}:", .{label});
    try stdout.print(" attn_prep={d:.3}", .{millisI128(profile.attn_prep_ns) / denom});
    try stdout.print(" qkv={d:.3}", .{millisI128(profile.qkv_ns) / denom});
    try stdout.print(" qk_norm_rope={d:.3}", .{millisI128(profile.qk_norm_rope_ns) / denom});
    try stdout.print(" attention={d:.3}", .{millisI128(profile.attention_ns) / denom});
    try stdout.print(" attn_out={d:.3}", .{millisI128(profile.attn_out_ns) / denom});
    try stdout.print(" attn_residual={d:.3}", .{millisI128(profile.attn_residual_ns) / denom});
    try stdout.print(" ffn_prep={d:.3}", .{millisI128(profile.ffn_prep_ns) / denom});
    try stdout.print(" router={d:.3}", .{millisI128(profile.router_ns) / denom});
    try stdout.print(" gate_up={d:.3}", .{millisI128(profile.gate_up_ns) / denom});
    try stdout.print(" swiglu={d:.3}", .{millisI128(profile.swiglu_ns) / denom});
    try stdout.print(" down={d:.3}", .{millisI128(profile.down_ns) / denom});
    try stdout.print(" ffn_residual={d:.3}", .{millisI128(profile.ffn_residual_ns) / denom});
    try stdout.print(" final={d:.3}", .{millisI128(profile.final_ns) / denom});
    try stdout.print(" layers={d}\n", .{profile.layers});

    const mb = profile.moe_batch;
    if (mb.batches > 0) {
        const batch_denom: f64 = @floatFromInt(mb.batches);
        try stdout.print("moe batch avg ms:", .{});
        try stdout.print(" total={d:.3}", .{millisI128(mb.total_ns) / denom});
        try stdout.print(" alloc={d:.3}", .{millisI128(mb.alloc_ns) / denom});
        try stdout.print(" count_sort={d:.3}", .{millisI128(mb.count_sort_ns) / denom});
        try stdout.print(" expert_wall={d:.3}", .{millisI128(mb.expert_wall_ns) / denom});
        try stdout.print(" scatter={d:.3}", .{millisI128(mb.scatter_ns) / denom});
        try stdout.print(" task_sum_gather_q={d:.3}", .{millisI128(mb.gather_quant_ns) / denom});
        try stdout.print(" task_sum_gate_up={d:.3}", .{millisI128(mb.gate_up_ns) / denom});
        try stdout.print(" task_sum_swiglu_q={d:.3}", .{millisI128(mb.swiglu_requant_ns) / denom});
        try stdout.print(" task_sum_down={d:.3}", .{millisI128(mb.down_ns) / denom});
        try stdout.print(
            " batches/pass={d:.1} pairs/batch={d:.1} active_experts/batch={d:.1} max_m={d}\n",
            .{
                @as(f64, @floatFromInt(mb.batches)) / denom,
                @as(f64, @floatFromInt(mb.pairs)) / batch_denom,
                @as(f64, @floatFromInt(mb.active_experts)) / batch_denom,
                mb.max_expert_m,
            },
        );
    }
}

/// Hidden `--spec-bench` mode: the speculative-decoding verify-economics
/// probe. After prefilling the prompt, measure (best-of-`reps`) the cost of
/// ONE batched k-row verify pass (`forwardStepAllLogits`) against k
/// sequential single-token decode steps, rewinding the cache with
/// `truncate()` between runs — exactly the state dance the speculative
/// decoder performs. The ratio quantifies what acceptance rate speculation
/// needs to win: a verify pass that costs `eq` single steps pays off once
/// it commits more than `eq` tokens on average.
pub fn runSpecBench(
    io: std.Io,
    stdout: anytype,
    ctx: *fucina.ExecContext,
    model: *const models.qwen3.model.Model,
    tokens: []const usize,
    load_ns: i96,
    cache_type: models.text.kv_cache.KvDtype,
    reps: usize,
) !void {
    const cfg = model.config;
    const ks = [_]usize{ 2, 4, 8, 16 };
    const max_k = ks[ks.len - 1];
    const capacity = tokens.len + max_k + 1;
    var cache = try models.text.kv_cache.KvCache.initWithDtype(ctx, cfg.num_layers, cfg.num_key_value_heads, cfg.head_dim, capacity, cache_type);
    defer cache.deinit();

    // Prefill once; every measurement runs at this depth and rewinds to it.
    var prefill = try model.forwardStep(ctx, &cache, tokens, 0);
    prefill.deinit();
    const base = cache.len();

    // Continuation token values don't affect the cost; cycle the prompt.
    var cont: [max_k]usize = undefined;
    for (&cont, 0..) |*t, i| t.* = tokens[i % tokens.len];

    try stdout.print("load: {d:.3} s\n", .{seconds(load_ns)});
    try stdout.print("spec verify economics: prompt {d} tok (base kv len {d}), best of {d} reps, kv {s}\n", .{ tokens.len, base, reps, @tagName(cache.dtype) });
    try stdout.print("  k | batch-k verify ms | k x single ms | batch/k-single | verify = N single steps\n", .{});

    // Warmup both paths (prime buffers/threads), then measure.
    {
        var warm = try model.forwardStepAllLogits(ctx, &cache, cont[0..max_k], cache.len());
        warm.deinit();
        cache.truncate(base);
        var warm2 = try model.forwardStep(ctx, &cache, cont[0..1], cache.len());
        warm2.deinit();
        cache.truncate(base);
    }

    for (ks) |k| {
        var batch_best: i96 = std.math.maxInt(i96);
        var single_best: i96 = std.math.maxInt(i96);
        for (0..reps) |_| {
            const t0 = nowNs(io);
            var logits = try model.forwardStepAllLogits(ctx, &cache, cont[0..k], cache.len());
            logits.deinit();
            const dt = nowNs(io) - t0;
            cache.truncate(base);
            batch_best = @min(batch_best, dt);
        }
        for (0..reps) |_| {
            const t0 = nowNs(io);
            for (0..k) |i| {
                var logits = try model.forwardStep(ctx, &cache, cont[i..][0..1], cache.len());
                logits.deinit();
            }
            const dt = nowNs(io) - t0;
            cache.truncate(base);
            single_best = @min(single_best, dt);
        }
        const batch_ms = millis(batch_best);
        const single_ms = millis(single_best);
        const single_one = single_ms / @as(f64, @floatFromInt(k));
        try stdout.print(" {d:>2} | {d:>17.3} | {d:>13.3} | {d:>14.3} | {d:>6.2}\n", .{
            k,
            batch_ms,
            single_ms,
            batch_ms / single_ms,
            batch_ms / single_one,
        });
    }
}
