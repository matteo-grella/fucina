//! Behavioral tests for the KV cache module (`kv_cache.zig`): cached vs. full
//! causal attention parity (f16 and q8_0), append overflow/shape-mismatch
//! rejection, per-layer head_dim sizing, q8_0 quantization round-trip bounds,
//! block-alignment enforcement, and the speculative-decode truncate/re-append
//! rewind contract.
const std = @import("std");
const fucina = @import("fucina");

const ExecContext = fucina.ExecContext;

const kv_cache = @import("kv_cache.zig");
const KvTensor = kv_cache.KvTensor;
const KvInput = kv_cache.KvInput;
const KvDtype = kv_cache.KvDtype;
const Error = kv_cache.Error;
const KvCache = kv_cache.KvCache;

test "cached decode matches full causal attention per position (f16)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // GQA shape that hits the production adjacent-kv-head-pair attention path
    // (heads == kv_heads * 2), like Qwen3-0.6B's 16/8 heads.
    const S = 5;
    const H = 4;
    const KV = 2;
    const D = 4;
    const scale = 1.0 / @sqrt(@as(f32, D));
    const kv_head_for_head = [_]usize{ 0, 0, 1, 1 };

    var q_values: [S * H * D]f32 = undefined;
    for (&q_values, 0..) |*value, i| value.* = @sin(@as(f32, @floatFromInt(i)) * 0.3) * 1.5;
    var k_values: [S * KV * D]f32 = undefined;
    for (&k_values, 0..) |*value, i| value.* = @cos(@as(f32, @floatFromInt(i)) * 0.21) - 0.4;
    var v_values: [S * KV * D]f32 = undefined;
    for (&v_values, 0..) |*value, i| value.* = @sin(@as(f32, @floatFromInt(i)) * 0.17 + 1.0);

    const QTensor = fucina.Tensor(.{ .seq, .head, .d });
    var q_all = try QTensor.fromSlice(&ctx, .{ S, H, D }, &q_values);
    defer q_all.deinit();
    var k_all = try KvInput.fromSlice(&ctx, .{ S, KV, D }, &k_values);
    defer k_all.deinit();
    var v_all = try KvInput.fromSlice(&ctx, .{ S, KV, D }, &v_values);
    defer v_all.deinit();

    // Reference: full causal attention over the SAME f16 K/V. The cache append
    // applies the identical f32->f16 cast, so the incremental decode output must
    // be bit-identical to the matching row of this reference — this isolates the
    // cache mechanism (append offset, source_offset causal mask) from f16 error.
    var k_all_f16 = try ctx.castTyped(.f32, .f16, k_all.asRawTensor());
    defer k_all_f16.deinit();
    var v_all_f16 = try ctx.castTyped(.f32, .f16, v_all.asRawTensor());
    defer v_all_f16.deinit();
    var full = try ctx.groupedCausalAttentionF16Kv(q_all.asRawTensor(), &k_all_f16, &v_all_f16, &kv_head_for_head, scale);
    defer full.deinit();
    const full_data = full.dataConst();

    // The f16 kernel must also compute the *right* attention: close to the f32
    // result, within f16 precision (guards against a widening/indexing bug that
    // is self-consistent but wrong).
    var full_f32 = try q_all.groupedAttention(&ctx, &k_all, &v_all, &kv_head_for_head, .attn, scale, .{});
    defer full_f32.deinit();
    for (full_data, try full_f32.dataConst()) |got, want| {
        try std.testing.expectApproxEqAbs(want, got, 1e-2);
    }

    var cache = try KvCache.init(&ctx, 1, KV, D, S);
    defer cache.deinit();

    for (0..S) |pos| {
        var q_row = try q_all.narrow(&ctx, .seq, pos, 1);
        defer q_row.deinit();
        var k_row = try k_all.narrow(&ctx, .seq, pos, 1);
        defer k_row.deinit();
        var v_row = try v_all.narrow(&ctx, .seq, pos, 1);
        defer v_row.deinit();

        try cache.appendLayer(&ctx, 0, &k_row, &v_row);
        const cached_len = cache.len + 1;
        var k_view = try cache.k[0].narrow(&ctx, .seq, 0, cached_len);
        defer k_view.deinit();
        var v_view = try cache.v[0].narrow(&ctx, .seq, 0, cached_len);
        defer v_view.deinit();

        var step = try ctx.groupedCausalAttentionF16Kv(q_row.asRawTensor(), k_view.asRawTensor(), v_view.asRawTensor(), &kv_head_for_head, scale);
        defer step.deinit();
        cache.advance(1);

        try std.testing.expectEqualSlices(f32, full_data[pos * H * D ..][0 .. H * D], step.dataConst());
    }
    try std.testing.expectEqual(@as(usize, S), cache.len);
}

test "appendLayer rejects overflow and shape mismatch" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var cache = try KvCache.init(&ctx, 1, 2, 4, 2);
    defer cache.deinit();

    var rows = try KvInput.fromSlice(&ctx, .{ 1, 2, 4 }, &[_]f32{1} ** 8);
    defer rows.deinit();
    var wrong = try KvInput.fromSlice(&ctx, .{ 1, 3, 4 }, &[_]f32{1} ** 12);
    defer wrong.deinit();

    try std.testing.expectError(Error.KvCacheShapeMismatch, cache.appendLayer(&ctx, 0, &rows, &wrong));

    try cache.appendLayer(&ctx, 0, &rows, &rows);
    cache.advance(1);
    try cache.appendLayer(&ctx, 0, &rows, &rows);
    cache.advance(1);
    try std.testing.expectError(Error.KvCacheOverflow, cache.appendLayer(&ctx, 0, &rows, &rows));
}

test "KvCache per-layer head_dim sizes each layer independently" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const KV = 2;
    // Gemma-shaped: interleaved head_dims (e.g. SWA 4 vs global 8) per layer.
    const kv_heads = [_]usize{ KV, KV };
    const head_dims = [_]usize{ 4, 8 };
    var cache = try KvCache.initPerLayer(&ctx, &kv_heads, &head_dims, 3);
    defer cache.deinit();

    try std.testing.expectEqual(@as(usize, 2), cache.k.len);
    try std.testing.expectEqual(@as(usize, 4), cache.head_dim[0]);
    try std.testing.expectEqual(@as(usize, 8), cache.head_dim[1]);
    try std.testing.expectEqual(@as(usize, 4), cache.k[0].dim(.d));
    try std.testing.expectEqual(@as(usize, 8), cache.k[1].dim(.d));

    var row0 = try KvInput.fromSlice(&ctx, .{ 1, KV, 4 }, &[_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer row0.deinit();
    var row1 = try KvInput.fromSlice(&ctx, .{ 1, KV, 8 }, &([_]f32{3} ** 16));
    defer row1.deinit();

    // A layer rejects rows sized for a different layer's head_dim.
    try std.testing.expectError(Error.KvCacheShapeMismatch, cache.appendLayer(&ctx, 1, &row0, &row0));

    try cache.appendLayer(&ctx, 0, &row0, &row0);
    try cache.appendLayer(&ctx, 1, &row1, &row1);
    cache.advance(1);
    try std.testing.expectEqual(@as(usize, 1), cache.len);

    var k0 = try cache.k[0].narrow(&ctx, .seq, 0, 1);
    defer k0.deinit();
    const k0d = k0.value.dataConst();
    try std.testing.expectEqual(@as(usize, KV * 4), k0d.len);
    try std.testing.expectApproxEqAbs(@as(f32, 1), @as(f32, @floatCast(k0d[0])), 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 8), @as(f32, @floatCast(k0d[7])), 1e-3);
}

test "q8_0 cache append round-trips within Q8_0 quantization error" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const S = 4;
    const KV = 2;
    const D = 64; // multiple of 32: two blocks per (position, head) row segment
    var cache = try KvCache.initWithDtype(&ctx, 1, KV, D, S, .q8_0);
    defer cache.deinit();
    try std.testing.expectEqual(KvDtype.q8_0, cache.dtype);
    // 34 bytes per 32 elements, K+V: the memory contract behind the flag.
    try std.testing.expectEqual(@as(usize, 2 * S * KV * (D / 32) * 34), cache.byteSize());

    // K appended from a contiguous tensor; V from a strided rank-3 view (the
    // fused-QKV split shape) to exercise the per-row-span quantize path.
    var k_values: [S * KV * D]f32 = undefined;
    for (&k_values, 0..) |*value, i| value.* = @sin(@as(f32, @floatFromInt(i)) * 0.37) * 3.0 - 0.5;
    var wide_values: [S * 2 * KV * D]f32 = undefined;
    for (&wide_values, 0..) |*value, i| value.* = @cos(@as(f32, @floatFromInt(i)) * 0.23) * 2.0;
    const Wide = fucina.Tensor(.{ .seq, .kv_head, .d });
    var k_rows = try KvInput.fromSlice(&ctx, .{ S, KV, D }, &k_values);
    defer k_rows.deinit();
    var wide = try Wide.fromSlice(&ctx, .{ S, 2 * KV, D }, &wide_values);
    defer wide.deinit();
    var v_rows = try wide.narrow(&ctx, .kv_head, KV, KV);
    defer v_rows.deinit();

    try cache.appendLayer(&ctx, 0, &k_rows, &v_rows);
    cache.advance(S);

    // Gather the f32 the strided V view actually holds.
    var v_values: [S * KV * D]f32 = undefined;
    for (0..S) |s| for (0..KV) |h| for (0..D) |d| {
        v_values[(s * KV + h) * D + d] = wide_values[(s * 2 * KV + KV + h) * D + d];
    };

    var deq: [S * KV * D]f32 = undefined;
    inline for (.{ .{ cache.kBlocks(0, S), k_values }, .{ cache.vBlocks(0, S), v_values } }) |pair| {
        const blocks, const source = pair;
        try ctx.dequantizeQ8_0RowsInto(&deq, blocks);
        var sq_err: f64 = 0;
        var sq_step: f64 = 0;
        for (0..S * KV * (D / 32)) |block_i| {
            var amax: f32 = 0;
            for (source[block_i * 32 ..][0..32]) |value| amax = @max(amax, @abs(value));
            // Round-to-nearest error <= step/2, plus the f16 rounding of the
            // scale itself (~2^-11 relative): bound by half a step + slack.
            const step = amax / 127.0;
            const bound = step * 0.5 + amax * 1e-3;
            for (source[block_i * 32 ..][0..32], deq[block_i * 32 ..][0..32]) |want, got| {
                try std.testing.expect(@abs(want - got) <= bound);
                sq_err += @as(f64, want - got) * @as(f64, want - got);
            }
            sq_step += @as(f64, step) * @as(f64, step) * 32;
        }
        // RMSE of round-to-nearest uniform quantization is step/sqrt(12) ~
        // 0.29 step; a layout/indexing bug would be O(amax) instead.
        try std.testing.expect(@sqrt(sq_err) <= 0.6 * @sqrt(sq_step));
    }
}

test "q8_0 cache rejects head_dim not divisible by the block size" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    try std.testing.expectError(
        Error.KvCacheHeadDimNotBlockAligned,
        KvCache.initWithDtype(&ctx, 1, 2, 48, 4, .q8_0),
    );
    var ok = try KvCache.initWithDtype(&ctx, 1, 2, 48, 4, .f16);
    ok.deinit();
}

/// Fills a q8_0 cache from deterministic pseudo-random K/V, then checks the
/// q8_0 attention entry against the f32 kernel run on the *dequantized*
/// cache: that isolates kernel correctness (row addressing, scratch dequant,
/// dispatch) from quantization loss — both kernels see numerically identical
/// K/V rows and run the same inner loops, so results must match exactly.
/// Also sanity-checks against f32 attention on the original (pre-quant) K/V
/// within a loose bound justified by Q8_0's error (<= ~0.4% of each block's
/// max; unit-scale data and softmax keep the output error well under 5e-2).
fn expectQ8AttentionParity(
    ctx: *ExecContext,
    comptime S: usize,
    comptime H: usize,
    comptime KV: usize,
    comptime D: usize,
    kv_head_for_head: []const usize,
    window: usize,
) !void {
    const scale = 1.0 / @sqrt(@as(f32, D));
    const allocator = ctx.allocator;

    const q_values = try allocator.alloc(f32, S * H * D);
    defer allocator.free(q_values);
    for (q_values, 0..) |*value, i| value.* = @sin(@as(f32, @floatFromInt(i)) * 0.3) * 1.5;
    const k_values = try allocator.alloc(f32, S * KV * D);
    defer allocator.free(k_values);
    for (k_values, 0..) |*value, i| value.* = @cos(@as(f32, @floatFromInt(i)) * 0.21) - 0.4;
    const v_values = try allocator.alloc(f32, S * KV * D);
    defer allocator.free(v_values);
    for (v_values, 0..) |*value, i| value.* = @sin(@as(f32, @floatFromInt(i)) * 0.17 + 1.0);

    const QTensor = fucina.Tensor(.{ .seq, .head, .d });
    var q_all = try QTensor.fromSlice(ctx, .{ S, H, D }, q_values);
    defer q_all.deinit();
    var k_all = try KvInput.fromSlice(ctx, .{ S, KV, D }, k_values);
    defer k_all.deinit();
    var v_all = try KvInput.fromSlice(ctx, .{ S, KV, D }, v_values);
    defer v_all.deinit();

    var cache = try KvCache.initWithDtype(ctx, 1, KV, D, S, .q8_0);
    defer cache.deinit();
    try cache.appendLayer(ctx, 0, &k_all, &v_all);
    cache.advance(S);

    // q8_0 kernel on the quantized cache.
    var got = if (window == 0)
        try ctx.groupedCausalAttentionQ8Kv(q_all.asRawTensor(), cache.kBlocks(0, S), cache.vBlocks(0, S), S, KV, kv_head_for_head, scale)
    else
        try ctx.groupedCausalAttentionQ8KvWindowed(q_all.asRawTensor(), cache.kBlocks(0, S), cache.vBlocks(0, S), S, KV, kv_head_for_head, scale, window);
    defer got.deinit();

    // Tight reference: f32 kernel on the dequantized cache.
    const k_deq = try allocator.alloc(f32, S * KV * D);
    defer allocator.free(k_deq);
    try ctx.dequantizeQ8_0RowsInto(k_deq, cache.kBlocks(0, S));
    const v_deq = try allocator.alloc(f32, S * KV * D);
    defer allocator.free(v_deq);
    try ctx.dequantizeQ8_0RowsInto(v_deq, cache.vBlocks(0, S));
    var k_deq_t = try KvInput.fromSlice(ctx, .{ S, KV, D }, k_deq);
    defer k_deq_t.deinit();
    var v_deq_t = try KvInput.fromSlice(ctx, .{ S, KV, D }, v_deq);
    defer v_deq_t.deinit();
    var want = if (window == 0)
        try ctx.groupedCausalAttention(q_all.asRawTensor(), k_deq_t.asRawTensor(), v_deq_t.asRawTensor(), kv_head_for_head, scale)
    else
        try ctx.groupedCausalAttentionWindowed(q_all.asRawTensor(), k_deq_t.asRawTensor(), v_deq_t.asRawTensor(), kv_head_for_head, scale, window);
    defer want.deinit();
    if (S >= 48) {
        // Tiled kernel (long prefill): still the dequant-scratch path with
        // f32 query rows — bit-exact vs the f32 kernel on the dequantized
        // cache, exactly as before.
        try std.testing.expectEqualSlices(f32, want.dataConst(), got.dataConst());
    } else {
        // Per-query kernels: the integer score path quantizes the QUERY row
        // to q8_0 too (q8xq8 sdot straight on the cached blocks), so the
        // f32-query reference is approached, not reproduced — the only new
        // error is the query's q8 rounding. The integer path's own
        // exactness is pinned by the primitive tests (pair == single) and
        // the incremental-decode == full-cache test below.
        for (want.dataConst(), got.dataConst()) |want_value, got_value| {
            try std.testing.expectApproxEqAbs(want_value, got_value, 2e-2);
        }
    }

    // Lossy sanity bound vs the original f32 K/V.
    var full = if (window == 0)
        try ctx.groupedCausalAttention(q_all.asRawTensor(), k_all.asRawTensor(), v_all.asRawTensor(), kv_head_for_head, scale)
    else
        try ctx.groupedCausalAttentionWindowed(q_all.asRawTensor(), k_all.asRawTensor(), v_all.asRawTensor(), kv_head_for_head, scale, window);
    defer full.deinit();
    for (full.dataConst(), got.dataConst()) |want_value, got_value| {
        try std.testing.expectApproxEqAbs(want_value, got_value, 5e-2);
    }
}

test "q8_0 attention tracks f32 attention on the dequantized cache (integer score path; tiled stays bit-exact)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Adjacent-pair GQA mapping (the production Qwen3 path).
    try expectQ8AttentionParity(&ctx, 5, 4, 2, 32, &.{ 0, 0, 1, 1 }, 0);
    // Non-pairable mapping: the general per-head kernel.
    try expectQ8AttentionParity(&ctx, 5, 4, 2, 32, &.{ 0, 1, 0, 1 }, 0);
    // Sliding-window variants of both.
    try expectQ8AttentionParity(&ctx, 6, 4, 2, 32, &.{ 0, 0, 1, 1 }, 3);
    try expectQ8AttentionParity(&ctx, 6, 4, 2, 32, &.{ 0, 1, 0, 1 }, 3);
    // q_seq >= 48: the query-tiled kernel (pair and general grouping).
    try expectQ8AttentionParity(&ctx, 64, 4, 2, 64, &.{ 0, 0, 1, 1 }, 0);
    try expectQ8AttentionParity(&ctx, 64, 4, 2, 64, &.{ 0, 1, 0, 1 }, 0);
    // Tiled + window.
    try expectQ8AttentionParity(&ctx, 64, 4, 2, 64, &.{ 0, 0, 1, 1 }, 17);
}

test "cached decode matches full causal attention per position (q8_0)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const S = 5;
    const H = 4;
    const KV = 2;
    const D = 32;
    const scale = 1.0 / @sqrt(@as(f32, D));
    const kv_head_for_head = [_]usize{ 0, 0, 1, 1 };

    var q_values: [S * H * D]f32 = undefined;
    for (&q_values, 0..) |*value, i| value.* = @sin(@as(f32, @floatFromInt(i)) * 0.3) * 1.5;
    var k_values: [S * KV * D]f32 = undefined;
    for (&k_values, 0..) |*value, i| value.* = @cos(@as(f32, @floatFromInt(i)) * 0.21) - 0.4;
    var v_values: [S * KV * D]f32 = undefined;
    for (&v_values, 0..) |*value, i| value.* = @sin(@as(f32, @floatFromInt(i)) * 0.17 + 1.0);

    const QTensor = fucina.Tensor(.{ .seq, .head, .d });
    var q_all = try QTensor.fromSlice(&ctx, .{ S, H, D }, &q_values);
    defer q_all.deinit();
    var k_all = try KvInput.fromSlice(&ctx, .{ S, KV, D }, &k_values);
    defer k_all.deinit();
    var v_all = try KvInput.fromSlice(&ctx, .{ S, KV, D }, &v_values);
    defer v_all.deinit();

    // Reference: q8_0 attention over the fully-populated cache. Incremental
    // decode quantizes the identical rows at the identical offsets, so each
    // step must reproduce the matching reference row bit-for-bit — isolating
    // the cache mechanism from quantization error (bounded separately above).
    var full_cache = try KvCache.initWithDtype(&ctx, 1, KV, D, S, .q8_0);
    defer full_cache.deinit();
    try full_cache.appendLayer(&ctx, 0, &k_all, &v_all);
    full_cache.advance(S);
    var full = try ctx.groupedCausalAttentionQ8Kv(q_all.asRawTensor(), full_cache.kBlocks(0, S), full_cache.vBlocks(0, S), S, KV, &kv_head_for_head, scale);
    defer full.deinit();
    const full_data = full.dataConst();

    var cache = try KvCache.initWithDtype(&ctx, 1, KV, D, S, .q8_0);
    defer cache.deinit();

    for (0..S) |pos| {
        var q_row = try q_all.narrow(&ctx, .seq, pos, 1);
        defer q_row.deinit();
        var k_row = try k_all.narrow(&ctx, .seq, pos, 1);
        defer k_row.deinit();
        var v_row = try v_all.narrow(&ctx, .seq, pos, 1);
        defer v_row.deinit();

        try cache.appendLayer(&ctx, 0, &k_row, &v_row);
        const cached_len = cache.len + 1;
        var step = try ctx.groupedCausalAttentionQ8Kv(q_row.asRawTensor(), cache.kBlocks(0, cached_len), cache.vBlocks(0, cached_len), cached_len, KV, &kv_head_for_head, scale);
        defer step.deinit();
        cache.advance(1);

        try std.testing.expectEqualSlices(f32, full_data[pos * H * D ..][0 .. H * D], step.dataConst());
    }
    try std.testing.expectEqual(@as(usize, S), cache.len);
}

/// Shared truncate contract check: append a 3-position prefix + 2 rejected
/// positions, truncate back to the prefix, re-append 2 DIFFERENT positions,
/// and demand attention over the result is bitwise equal to a fresh cache that
/// only ever saw prefix + replacement — proving truncation leaves no trace of
/// the rejected rows (the speculative-decoding rewind contract).
fn expectTruncateReappendParity(ctx: *ExecContext, dtype: KvDtype) !void {
    const S = 5; // final sequence: 3 prefix + 2 replacement positions
    const H = 4;
    const KV = 2;
    const D = 32;
    const scale = 1.0 / @sqrt(@as(f32, D));
    const kv_head_for_head = [_]usize{ 0, 0, 1, 1 };

    var q_values: [S * H * D]f32 = undefined;
    for (&q_values, 0..) |*value, i| value.* = @sin(@as(f32, @floatFromInt(i)) * 0.3) * 1.5;
    var k_values: [S * KV * D]f32 = undefined;
    for (&k_values, 0..) |*value, i| value.* = @cos(@as(f32, @floatFromInt(i)) * 0.21) - 0.4;
    var v_values: [S * KV * D]f32 = undefined;
    for (&v_values, 0..) |*value, i| value.* = @sin(@as(f32, @floatFromInt(i)) * 0.17 + 1.0);
    // Rejected rows: distinct values, so stale data would corrupt the compare.
    var bad_values: [2 * KV * D]f32 = undefined;
    for (&bad_values, 0..) |*value, i| value.* = @cos(@as(f32, @floatFromInt(i)) * 0.91) * 7.0;

    const QTensor = fucina.Tensor(.{ .seq, .head, .d });
    var q_all = try QTensor.fromSlice(ctx, .{ S, H, D }, &q_values);
    defer q_all.deinit();
    var k_all = try KvInput.fromSlice(ctx, .{ S, KV, D }, &k_values);
    defer k_all.deinit();
    var v_all = try KvInput.fromSlice(ctx, .{ S, KV, D }, &v_values);
    defer v_all.deinit();
    var bad = try KvInput.fromSlice(ctx, .{ 2, KV, D }, &bad_values);
    defer bad.deinit();

    var k_prefix = try k_all.narrow(ctx, .seq, 0, 3);
    defer k_prefix.deinit();
    var v_prefix = try v_all.narrow(ctx, .seq, 0, 3);
    defer v_prefix.deinit();
    var k_tail = try k_all.narrow(ctx, .seq, 3, 2);
    defer k_tail.deinit();
    var v_tail = try v_all.narrow(ctx, .seq, 3, 2);
    defer v_tail.deinit();

    // Cache under test: prefix, then rejected rows, truncate, then the tail.
    var cache = try KvCache.initWithDtype(ctx, 1, KV, D, S, dtype);
    defer cache.deinit();
    try cache.appendLayer(ctx, 0, &k_prefix, &v_prefix);
    cache.advance(3);
    try cache.appendLayer(ctx, 0, &bad, &bad);
    cache.advance(2);
    try std.testing.expectEqual(@as(usize, S), cache.len);
    cache.truncate(3);
    try std.testing.expectEqual(@as(usize, 3), cache.len);
    cache.truncate(S + 10); // clamp: never grows
    try std.testing.expectEqual(@as(usize, 3), cache.len);
    try cache.appendLayer(ctx, 0, &k_tail, &v_tail);
    cache.advance(2);

    // Reference cache: only ever saw the final sequence.
    var fresh = try KvCache.initWithDtype(ctx, 1, KV, D, S, dtype);
    defer fresh.deinit();
    try fresh.appendLayer(ctx, 0, &k_all, &v_all);
    fresh.advance(S);

    switch (dtype) {
        .f16 => {
            var k_view = try cache.k[0].narrow(ctx, .seq, 0, S);
            defer k_view.deinit();
            var v_view = try cache.v[0].narrow(ctx, .seq, 0, S);
            defer v_view.deinit();
            var got = try ctx.groupedCausalAttentionF16Kv(q_all.asRawTensor(), k_view.asRawTensor(), v_view.asRawTensor(), &kv_head_for_head, scale);
            defer got.deinit();

            var k_ref = try fresh.k[0].narrow(ctx, .seq, 0, S);
            defer k_ref.deinit();
            var v_ref = try fresh.v[0].narrow(ctx, .seq, 0, S);
            defer v_ref.deinit();
            var want = try ctx.groupedCausalAttentionF16Kv(q_all.asRawTensor(), k_ref.asRawTensor(), v_ref.asRawTensor(), &kv_head_for_head, scale);
            defer want.deinit();
            try std.testing.expectEqualSlices(f32, want.dataConst(), got.dataConst());
        },
        .q8_0 => {
            var got = try ctx.groupedCausalAttentionQ8Kv(q_all.asRawTensor(), cache.kBlocks(0, S), cache.vBlocks(0, S), S, KV, &kv_head_for_head, scale);
            defer got.deinit();
            var want = try ctx.groupedCausalAttentionQ8Kv(q_all.asRawTensor(), fresh.kBlocks(0, S), fresh.vBlocks(0, S), S, KV, &kv_head_for_head, scale);
            defer want.deinit();
            try std.testing.expectEqualSlices(f32, want.dataConst(), got.dataConst());
        },
    }
}

test "truncate + re-append matches a fresh cache bitwise (f16 and q8_0)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    try expectTruncateReappendParity(&ctx, .f16);
    try expectTruncateReappendParity(&ctx, .q8_0);
}

test "KvCache.init fills a uniform per-layer head_dim" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var cache = try KvCache.init(&ctx, 3, 2, 4, 5);
    defer cache.deinit();
    try std.testing.expectEqual(@as(usize, 3), cache.head_dim.len);
    for (cache.head_dim) |hd| try std.testing.expectEqual(@as(usize, 4), hd);
}
