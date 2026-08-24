//! Numerical-parity harnesses of the Qwen3 runner: `--verify-batch` (batched
//! verify vs sequential decode, the speculative byte-identity contract),
//! `--verify-cache` (KV cache vs re-prefill), and `--compare-logits`
//! (dumped-logits diff against a reference file).
const std = @import("std");
const fucina = @import("fucina");
const llm = @import("fucina_llm");
const util = @import("util.zig");

const seconds = util.seconds;

pub fn compareLogits(io: std.Io, allocator: std.mem.Allocator, stdout: anytype, path: []const u8, values: []const f32) !void {
    const reference = try readF32File(io, allocator, path);
    defer allocator.free(reference);
    if (reference.len != values.len) return error.CompareLogitsShapeMismatch;

    var max_abs: f64 = 0;
    var sum_abs: f64 = 0;
    var sum_sq: f64 = 0;
    var value_top: usize = 0;
    var reference_top: usize = 0;
    for (values, reference, 0..) |value, ref, i| {
        const diff: f64 = @floatCast(value - ref);
        const abs_diff = @abs(diff);
        max_abs = @max(max_abs, abs_diff);
        sum_abs += abs_diff;
        sum_sq += diff * diff;
        if (value > values[value_top]) value_top = i;
        if (ref > reference[reference_top]) reference_top = i;
    }

    const n: f64 = @floatFromInt(values.len);
    try stdout.print(
        "compare logits: max_abs={d:.6} mean_abs={d:.6} rms={d:.6} top={d}:{d:.4} ref_top={d}:{d:.4} aligned={}\n",
        .{
            max_abs,
            sum_abs / n,
            @sqrt(sum_sq / n),
            value_top,
            values[value_top],
            reference_top,
            reference[reference_top],
            value_top == reference_top,
        },
    );
}

const CacheCompare = struct {
    cache_top: usize,
    ref_top: usize,
    aligned: bool,
    max_abs: f64,
    // A misaligned argmax is "benign" when the two candidates' logit gap is
    // within the floating-point drift (2x the max per-element abs diff): the
    // flip is fully explained by m=1-vs-m=L kernel reassociation, not a cache
    // bug. A structural bug would diverge far beyond the drift.
    benign: bool,
};

/// --verify-batch=N: are batched verify logits bit-identical to sequential
/// decode? Prefill the prompt, greedy-decode N tokens one forwardStep at a
/// time (recording every step's FULL logit row), then for a sweep of batch
/// sizes m re-prefill a fresh cache and score the same committed tokens in
/// ONE forwardStepAllLogits pass — reporting, per m, the first row that is
/// not bitwise equal to its sequential twin, the max |diff|, and whether
/// any greedy top-1 flips. The sweep runs kernel-pinned (the speculative
/// verify configuration, expected all-bitwise) and unpinned (the x4 batch
/// kernels legally drift at m >= 4). A top-1 flip here is exactly a
/// committed-token divergence in --spec, so this harness guards the
/// byte-identity contract (speculative/core.zig).
pub fn runVerifyBatch(
    allocator: std.mem.Allocator,
    stdout: anytype,
    ctx: *fucina.ExecContext,
    model: *const llm.qwen3.model.Model,
    tokens: []const usize,
    load_ns: i96,
    max_steps: usize,
    cache_type: llm.kv_cache.KvDtype,
) !void {
    const cfg = model.config;
    const vocab = cfg.vocab_size;
    const capacity = tokens.len + max_steps + 1;
    const pos0 = tokens.len;

    // Sequential reference arm: the exact per-token decode numerics.
    const ref = try allocator.alloc(f32, max_steps * vocab);
    defer allocator.free(ref);
    const committed = try allocator.alloc(usize, max_steps + 1);
    defer allocator.free(committed);
    {
        var cache = try llm.kv_cache.KvCache.initWithDtype(ctx, cfg.num_layers, cfg.num_key_value_heads, cfg.head_dim, capacity, cache_type);
        defer cache.deinit();
        var prefill_logits = try model.forwardStep(ctx, &cache, tokens, 0);
        defer prefill_logits.deinit();
        committed[0] = argmaxSlice(try prefill_logits.dataConst());
        for (0..max_steps) |i| {
            var logits = try model.forwardStep(ctx, &cache, committed[i .. i + 1], cache.len());
            defer logits.deinit();
            const row = try logits.dataConst();
            @memcpy(ref[i * vocab ..][0..vocab], row);
            committed[i + 1] = argmaxSlice(row);
        }
    }
    try stdout.print("load: {d:.3} s\n", .{seconds(load_ns)});
    try stdout.print("verify batch vs sequential ({d} prompt tokens, {d} steps):\n", .{ tokens.len, max_steps });

    const sweep = [_]usize{ 2, 3, 4, 5, 8, 12, 16, 24, 32 };
    for ([_]bool{ true, false }) |pinned| {
        for (sweep) |m| {
            if (m > max_steps) continue;
            var cache = try llm.kv_cache.KvCache.initWithDtype(ctx, cfg.num_layers, cfg.num_key_value_heads, cfg.head_dim, capacity, cache_type);
            defer cache.deinit();
            var prefill_logits = try model.forwardStep(ctx, &cache, tokens, 0);
            prefill_logits.deinit();
            ctx.pinRowwiseKernels(pinned);
            var all = blk: {
                defer ctx.pinRowwiseKernels(false);
                break :blk try model.forwardStepAllLogits(ctx, &cache, committed[0..m], pos0);
            };
            defer all.deinit();
            const rows = try all.dataConst();

            var first_diff: ?usize = null;
            var bitwise_rows: usize = 0;
            var max_abs: f64 = 0;
            var flip_row: ?usize = null;
            for (0..m) |i| {
                const got = rows[i * vocab ..][0..vocab];
                const want = ref[i * vocab ..][0..vocab];
                if (std.mem.eql(u8, std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(want))) {
                    bitwise_rows += 1;
                    continue;
                }
                if (first_diff == null) first_diff = i;
                const cmp = compareTopAndDiff(got, want);
                max_abs = @max(max_abs, cmp.max_abs);
                if (!cmp.aligned and flip_row == null) flip_row = i;
            }
            if (first_diff == null) {
                try stdout.print("  m={d:>2} {s}: all {d} rows BITWISE", .{ m, if (pinned) "pinned  " else "unpinned", m });
            } else if (flip_row) |fr| {
                try stdout.print("  m={d:>2} {s}: bitwise {d}/{d}, first diff row {d}, max_abs {e:.3}, TOP-1 FLIP at row {d}", .{ m, if (pinned) "pinned  " else "unpinned", bitwise_rows, m, first_diff.?, max_abs, fr });
            } else {
                try stdout.print("  m={d:>2} {s}: bitwise {d}/{d}, first diff row {d}, max_abs {e:.3}, top-1 stable", .{ m, if (pinned) "pinned  " else "unpinned", bitwise_rows, m, first_diff.?, max_abs });
            }

            // Continuation probe: the batch also WROTE m cache rows; one
            // more sequential step on top of them must reproduce the
            // sequential arm's next logits, else the batch leaves numerically
            // different KV state behind (invisible to the row comparison
            // above, poisonous to every later token).
            if (m < max_steps) {
                var cont = try model.forwardStep(ctx, &cache, committed[m .. m + 1], pos0 + m);
                defer cont.deinit();
                const got = try cont.dataConst();
                const want = ref[m * vocab ..][0..vocab];
                if (std.mem.eql(u8, std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(want))) {
                    try stdout.print("; continuation BITWISE\n", .{});
                } else {
                    const cmp = compareTopAndDiff(got, want);
                    try stdout.print("; continuation DIFFERS (max_abs {e:.3}, top {d} vs {d})\n", .{ cmp.max_abs, cmp.cache_top, cmp.ref_top });
                }
            } else {
                try stdout.print("\n", .{});
            }
        }
    }

    // Truncate-replay probe (the speculative verify's exact cache dance):
    // batch [correct token, m-1 GARBAGE drafts], truncate back to one
    // committed row, then walk the true continuation sequentially — every
    // step must be bitwise vs the sequential arm, else rejected drafts
    // leave residue behind the truncation point.
    {
        const m: usize = @min(9, max_steps);
        var cache = try llm.kv_cache.KvCache.initWithDtype(ctx, cfg.num_layers, cfg.num_key_value_heads, cfg.head_dim, capacity, cache_type);
        defer cache.deinit();
        var prefill_logits = try model.forwardStep(ctx, &cache, tokens, 0);
        prefill_logits.deinit();
        var verify_buf: [9]usize = undefined;
        verify_buf[0] = committed[0];
        for (verify_buf[1..m]) |*t| t.* = 0; // deliberately wrong drafts
        ctx.pinRowwiseKernels(true);
        var all = blk: {
            defer ctx.pinRowwiseKernels(false);
            break :blk try model.forwardStepAllLogits(ctx, &cache, verify_buf[0..m], pos0);
        };
        all.deinit();
        cache.truncate(pos0 + 1);
        var ok = true;
        for (1..max_steps) |i| {
            var logits = try model.forwardStep(ctx, &cache, committed[i .. i + 1], cache.len());
            defer logits.deinit();
            const got = try logits.dataConst();
            const want = ref[i * vocab ..][0..vocab];
            if (!std.mem.eql(u8, std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(want))) {
                const cmp = compareTopAndDiff(got, want);
                try stdout.print("  truncate-replay: step {d} DIFFERS (max_abs {e:.3}, top {d} vs {d})\n", .{ i, cmp.max_abs, cmp.cache_top, cmp.ref_top });
                ok = false;
                break;
            }
        }
        if (ok) try stdout.print("  truncate-replay: all {d} steps BITWISE after garbage-draft truncation\n", .{max_steps - 1});
    }
}

fn argmaxSlice(values: []const f32) usize {
    var best: usize = 0;
    for (values, 0..) |v, i| {
        if (v > values[best]) best = i;
    }
    return best;
}

pub fn runVerifyCache(
    allocator: std.mem.Allocator,
    stdout: anytype,
    ctx: *fucina.ExecContext,
    model: *const llm.qwen3.model.Model,
    tokens: []const usize,
    load_ns: i96,
    max_steps: usize,
    cache_type: llm.kv_cache.KvDtype,
) !void {
    const cfg = model.config;
    const capacity = tokens.len + max_steps;
    var cache = try llm.kv_cache.KvCache.initWithDtype(ctx, cfg.num_layers, cfg.num_key_value_heads, cfg.head_dim, capacity, cache_type);
    defer cache.deinit();
    try util.printCacheInfo(stdout, &cache);

    const seq = try allocator.alloc(usize, capacity);
    defer allocator.free(seq);
    @memcpy(seq[0..tokens.len], tokens);
    var seq_len = tokens.len;

    var cache_logits = try model.forwardStep(ctx, &cache, tokens, 0);
    defer cache_logits.deinit();
    var ref_logits = try model.forwardLastLogits(ctx, seq[0..seq_len]);
    defer ref_logits.deinit();

    try stdout.print("load: {d:.3} s\n", .{seconds(load_ns)});
    try stdout.print("verify cache vs re-prefill ({d} prompt tokens, {d} steps):\n", .{ tokens.len, max_steps });

    var aligned_steps: usize = 0;
    var ok = true;
    var max_diff_overall: f64 = 0;
    var step: usize = 0;
    while (true) : (step += 1) {
        const cmp = compareTopAndDiff(try cache_logits.dataConst(), try ref_logits.dataConst());
        max_diff_overall = @max(max_diff_overall, cmp.max_abs);
        if (cmp.aligned) aligned_steps += 1;
        // f16 KV is lossy vs the f32 re-prefill reference, so per-step logits
        // differ by the f16 rounding error; correctness here means every greedy
        // divergence is a drift near-tie. A non-benign divergence (gap far beyond
        // the per-element diff) would signal a real cache bug.
        if (!cmp.benign) ok = false;
        const note = if (cmp.aligned) "" else if (cmp.benign) " (near-tie)" else " (DIVERGED)";
        try stdout.print("  step {d}: cache_top={d} ref_top={d} aligned={} max_abs={d:.6}{s}\n", .{ step, cmp.cache_top, cmp.ref_top, cmp.aligned, cmp.max_abs, note });

        if (step == max_steps or seq_len == capacity) break;
        seq[seq_len] = cmp.cache_top;
        seq_len += 1;

        // Allocate-then-swap so an error never leaves a deinit'd tensor live
        // under the function-scope defers.
        const fresh_cache = try model.forwardStep(ctx, &cache, seq[seq_len - 1 ..][0..1], cache.len());
        cache_logits.deinit();
        cache_logits = fresh_cache;
        const fresh_ref = try model.forwardLastLogits(ctx, seq[0..seq_len]);
        ref_logits.deinit();
        ref_logits = fresh_ref;
    }
    try stdout.print("verify: {s} (top-aligned {d}/{d} steps, max_abs={d:.6}; divergences within drift are expected)\n", .{ if (ok) "PASS" else "FAIL", aligned_steps, step + 1, max_diff_overall });
}

fn compareTopAndDiff(cache_values: []const f32, ref_values: []const f32) CacheCompare {
    var cache_top: usize = 0;
    var ref_top: usize = 0;
    var max_abs: f64 = 0;
    for (cache_values, ref_values, 0..) |cache_value, ref_value, i| {
        if (cache_value > cache_values[cache_top]) cache_top = i;
        if (ref_value > ref_values[ref_top]) ref_top = i;
        const diff = @abs(@as(f64, cache_value) - @as(f64, ref_value));
        max_abs = @max(max_abs, diff);
    }
    const aligned = cache_top == ref_top;
    const gap = @as(f64, ref_values[ref_top]) - @as(f64, ref_values[cache_top]);
    return .{
        .cache_top = cache_top,
        .ref_top = ref_top,
        .aligned = aligned,
        .max_abs = max_abs,
        .benign = aligned or gap < 2 * max_abs,
    };
}

fn readF32File(io: std.Io, allocator: std.mem.Allocator, path: []const u8) ![]f32 {
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);

    const stat = try file.stat(io);
    if (stat.kind != .file) return error.IsDir;
    if (stat.size % @sizeOf(f32) != 0) return error.InvalidLogitsFile;

    const byte_len: usize = @intCast(stat.size);
    const bytes = try allocator.alloc(u8, byte_len);
    defer allocator.free(bytes);

    var read_len: usize = 0;
    while (read_len < bytes.len) {
        const n = try file.readStreaming(io, &.{bytes[read_len..]});
        if (n == 0) return error.EndOfStream;
        read_len += n;
    }

    const out = try allocator.alloc(f32, byte_len / @sizeOf(f32));
    errdefer allocator.free(out);
    for (out, 0..) |*dst, i| {
        const bits = std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little);
        dst.* = @bitCast(bits);
    }
    return out;
}
