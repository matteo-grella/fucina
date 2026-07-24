const std = @import("std");
const fucina = @import("fucina");

const Allocator = std.mem.Allocator;
const ExecContext = fucina.ExecContext;
const q8_0_block_size = fucina.q8_0_block_size;

/// Per-layer post-RoPE key/value store for autoregressive decode.
///
/// K and V are kept as f16 `[capacity, kv_heads, head_dim]` contiguous tensors —
/// the exact `[.seq, .kv_head, .d]` layout the f16 attention kernel consumes —
/// so the active prefix `[0..len]` is a zero-copy narrow that feeds attention
/// directly. f16 halves the cache footprint and the per-step bandwidth (the
/// kernel widens to f32 in-register), matching llama.cpp's default cache type.
/// K is stored *after* RoPE (V has no RoPE), so past positions are never
/// re-rotated.
///
/// Opt-in q8_0 mode (llama.cpp's `--cache-type-k/v q8_0`) stores each
/// (position, kv_head) row as `head_dim/32` BlockQ8_0 instead: 34 bytes per 32
/// elements, ~halving f16's footprint/bandwidth again at a small quantization
/// loss. q8_0 layers are raw block slices (not tensors); attention consumes
/// them via `kBlocks`/`vBlocks` + `groupedAttention`'s q8_0-block KV arm —
/// decode runs the integer q8xq8 score path straight on the blocks (the
/// query row quantizes once per head; see `exec/attention.zig`), so the
/// halved bytes translate into decode speed instead of a dequant tax.
pub const KvTensor = fucina.Tensor(.{ .dtype = .f16, .tags = .{ .seq, .kv_head, .d } });

/// The f32 K/V rows handed to `appendLayer` (post-RoPE K, raw V); converted to
/// the cache dtype on write.
pub const KvInput = fucina.Tensor(.{ .seq, .kv_head, .d });

/// Storage dtype of the cached K/V rows.
pub const KvDtype = enum { f16, q8_0 };

pub const Error = error{
    KvCacheOverflow,
    KvCacheShapeMismatch,
    /// q8_0 packs 32 elements per block; a head_dim that is not a multiple of
    /// 32 would straddle heads/positions. Use the f16 cache for such models.
    KvCacheHeadDimNotBlockAligned,
};

pub const KvCache = struct {
    allocator: Allocator,
    dtype: KvDtype,
    // f16 mode: one [capacity, kv_heads, head_dim] f16 tensor per layer.
    // Empty in q8_0 mode.
    k: []KvTensor,
    v: []KvTensor,
    // q8_0 mode: capacity * kv_heads * head_dim/32 blocks per layer, laid out
    // [capacity, kv_heads, head_dim/32]. Empty in f16 mode.
    k_q8: [][]fucina.BlockQ8_0,
    v_q8: [][]fucina.BlockQ8_0,
    len: usize,
    capacity: usize,
    // Per-layer KV-head count and head_dim. Most models share one value across
    // layers, but Gemma 4 interleaves local-SWA layers (kv_heads 8, head_dim 256)
    // with global layers (kv_heads 2, head_dim 512), so each layer's K/V slot is
    // sized independently. `init` fills both uniformly.
    kv_heads: []usize,
    head_dim: []usize,

    pub fn init(
        ctx: *ExecContext,
        num_layers: usize,
        kv_heads: usize,
        head_dim: usize,
        capacity: usize,
    ) !KvCache {
        return initWithDtype(ctx, num_layers, kv_heads, head_dim, capacity, .f16);
    }

    pub fn initWithDtype(
        ctx: *ExecContext,
        num_layers: usize,
        kv_heads: usize,
        head_dim: usize,
        capacity: usize,
        dtype: KvDtype,
    ) !KvCache {
        const allocator = ctx.allocator;
        const kv_heads_arr = try allocator.alloc(usize, num_layers);
        defer allocator.free(kv_heads_arr);
        @memset(kv_heads_arr, kv_heads);
        const head_dims = try allocator.alloc(usize, num_layers);
        defer allocator.free(head_dims);
        @memset(head_dims, head_dim);
        return initPerLayerWithDtype(ctx, kv_heads_arr, head_dims, capacity, dtype);
    }

    /// Per-layer variant: one `kv_heads`/`head_dim` per layer. The K/V attention
    /// kernels read both from the tensor shape, so no kernel change is needed —
    /// only the per-layer slot sizing here.
    pub fn initPerLayer(
        ctx: *ExecContext,
        kv_heads_per_layer: []const usize,
        head_dims: []const usize,
        capacity: usize,
    ) !KvCache {
        return initPerLayerWithDtype(ctx, kv_heads_per_layer, head_dims, capacity, .f16);
    }

    pub fn initPerLayerWithDtype(
        ctx: *ExecContext,
        kv_heads_per_layer: []const usize,
        head_dims: []const usize,
        capacity: usize,
        dtype: KvDtype,
    ) !KvCache {
        const allocator = ctx.allocator;
        const num_layers = head_dims.len;
        std.debug.assert(kv_heads_per_layer.len == num_layers);
        if (dtype == .q8_0) for (head_dims) |head_dim| {
            if (head_dim == 0 or head_dim % q8_0_block_size != 0) return Error.KvCacheHeadDimNotBlockAligned;
        };
        const kv_heads = try allocator.dupe(usize, kv_heads_per_layer);
        errdefer allocator.free(kv_heads);
        const head_dim = try allocator.dupe(usize, head_dims);
        errdefer allocator.free(head_dim);

        var cache: KvCache = .{
            .allocator = allocator,
            .dtype = dtype,
            .k = &.{},
            .v = &.{},
            .k_q8 = &.{},
            .v_q8 = &.{},
            .len = 0,
            .capacity = capacity,
            .kv_heads = kv_heads,
            .head_dim = head_dim,
        };

        switch (dtype) {
            .f16 => {
                const k = try allocator.alloc(KvTensor, num_layers);
                errdefer allocator.free(k);
                const v = try allocator.alloc(KvTensor, num_layers);
                errdefer allocator.free(v);

                var initialized: usize = 0;
                errdefer for (0..initialized) |i| {
                    k[i].deinit();
                    v[i].deinit();
                };
                for (0..num_layers) |i| {
                    k[i] = try makeLayer(ctx, kv_heads_per_layer[i], head_dims[i], capacity);
                    errdefer k[i].deinit();
                    v[i] = try makeLayer(ctx, kv_heads_per_layer[i], head_dims[i], capacity);
                    initialized += 1;
                }
                cache.k = k;
                cache.v = v;
            },
            .q8_0 => {
                const k_q8 = try allocator.alloc([]fucina.BlockQ8_0, num_layers);
                errdefer allocator.free(k_q8);
                const v_q8 = try allocator.alloc([]fucina.BlockQ8_0, num_layers);
                errdefer allocator.free(v_q8);

                var initialized: usize = 0;
                errdefer for (0..initialized) |i| {
                    allocator.free(k_q8[i]);
                    allocator.free(v_q8[i]);
                };
                for (0..num_layers) |i| {
                    const layer_blocks = capacity * kv_heads_per_layer[i] * (head_dims[i] / q8_0_block_size);
                    k_q8[i] = try allocator.alloc(fucina.BlockQ8_0, layer_blocks);
                    errdefer allocator.free(k_q8[i]);
                    v_q8[i] = try allocator.alloc(fucina.BlockQ8_0, layer_blocks);
                    initialized += 1;
                }
                cache.k_q8 = k_q8;
                cache.v_q8 = v_q8;
            },
        }

        return cache;
    }

    fn makeLayer(ctx: *ExecContext, kv_heads: usize, head_dim: usize, capacity: usize) !KvTensor {
        var raw = try ctx.emptyRankTyped(.f16, 3, .{ capacity, kv_heads, head_dim });
        errdefer raw.deinit();
        return KvTensor.fromTensor(ctx, raw);
    }

    pub fn deinit(self: *KvCache) void {
        for (self.k) |*layer| layer.deinit();
        for (self.v) |*layer| layer.deinit();
        self.allocator.free(self.k);
        self.allocator.free(self.v);
        for (self.k_q8) |layer| self.allocator.free(layer);
        for (self.v_q8) |layer| self.allocator.free(layer);
        self.allocator.free(self.k_q8);
        self.allocator.free(self.v_q8);
        self.allocator.free(self.kv_heads);
        self.allocator.free(self.head_dim);
        self.* = undefined;
    }

    /// Drop all cached positions; buffers are retained for reuse.
    pub fn reset(self: *KvCache) void {
        self.len = 0;
    }

    /// Total bytes of K+V storage across layers, from the actual allocations
    /// (f16: 2 bytes/element; q8_0: 34 bytes per 32 elements).
    pub fn byteSize(self: *const KvCache) usize {
        var total: usize = 0;
        switch (self.dtype) {
            .f16 => for (self.k, self.v) |*k_layer, *v_layer| {
                total += (k_layer.value.len() + v_layer.value.len()) * @sizeOf(f16);
            },
            .q8_0 => for (self.k_q8, self.v_q8) |k_layer, v_layer| {
                total += (k_layer.len + v_layer.len) * @sizeOf(fucina.BlockQ8_0);
            },
        }
        return total;
    }

    /// q8_0 mode: layer `layer_i`'s cached K blocks for the first `len`
    /// positions, laid out [len, kv_heads, head_dim/32] — the shape
    /// `groupedAttention`'s q8_0-block KV arm consumes.
    pub fn kBlocks(self: *const KvCache, layer_i: usize, len: usize) []const fucina.BlockQ8_0 {
        return self.k_q8[layer_i][0 .. len * self.layerRowBlocks(layer_i)];
    }

    /// q8_0 mode: as `kBlocks`, for V.
    pub fn vBlocks(self: *const KvCache, layer_i: usize, len: usize) []const fucina.BlockQ8_0 {
        return self.v_q8[layer_i][0 .. len * self.layerRowBlocks(layer_i)];
    }

    /// f16 mode: layer `layer_i`'s cached K rows for the first `len`
    /// positions as a raw `[len, kv_heads, head_dim]` f16 slice — the
    /// per-stream span `groupedAttention`'s multi-stream KV arm consumes.
    pub fn kSlice(self: *const KvCache, layer_i: usize, len: usize) ![]const f16 {
        return (try self.k[layer_i].dataConst())[0 .. len * self.layerRowElems(layer_i)];
    }

    /// f16 mode: as `kSlice`, for V.
    pub fn vSlice(self: *const KvCache, layer_i: usize, len: usize) ![]const f16 {
        return (try self.v[layer_i].dataConst())[0 .. len * self.layerRowElems(layer_i)];
    }

    fn layerRowElems(self: *const KvCache, layer_i: usize) usize {
        return self.kv_heads[layer_i] * self.head_dim[layer_i];
    }

    fn layerRowBlocks(self: *const KvCache, layer_i: usize) usize {
        return self.kv_heads[layer_i] * (self.head_dim[layer_i] / q8_0_block_size);
    }

    /// Convert the new tokens' f32 K/V to the cache dtype and copy them into
    /// layer `layer_i` at offset `len`. Does not advance `len` — every layer
    /// appends at the same base; call `advance` once after all layers for a
    /// step have been written.
    pub fn appendLayer(
        self: *KvCache,
        ctx: *ExecContext,
        layer_i: usize,
        k_rows: *const KvInput,
        v_rows: *const KvInput,
    ) !void {
        const m = k_rows.dim(.seq);
        const head_dim = self.head_dim[layer_i];
        const kv_heads = self.kv_heads[layer_i];
        if (k_rows.dim(.kv_head) != kv_heads or k_rows.dim(.d) != head_dim) return Error.KvCacheShapeMismatch;
        if (v_rows.dim(.seq) != m or v_rows.dim(.kv_head) != kv_heads or v_rows.dim(.d) != head_dim) return Error.KvCacheShapeMismatch;
        if (self.len + m > self.capacity) return Error.KvCacheOverflow;

        switch (self.dtype) {
            .f16 => {
                const row = kv_heads * head_dim;
                const start = self.len * row;
                const span = m * row;
                // Cast straight into the cache slot: one pass, no temporaries. K is
                // contiguous; V is a split view of the fused QKV row, walked as
                // per-row contiguous spans (previously castTyped materialized an f32
                // copy of V, allocated an unpooled f16 temp for each, and memcpy'd).
                const k_slot = try self.k[layer_i].data();
                const v_slot = try self.v[layer_i].data();
                try ctx.castF32RowsToF16Into(k_rows.asRawTensor(), k_slot[start..][0..span]);
                try ctx.castF32RowsToF16Into(v_rows.asRawTensor(), v_slot[start..][0..span]);
            },
            .q8_0 => {
                // Quantize straight into the cache's block storage — same
                // one-pass, allocation-free shape as the f16 cast. head_dim is
                // a multiple of 32 (checked at init), so block boundaries
                // align with (position, kv_head) row segments.
                const row_blocks = kv_heads * (head_dim / q8_0_block_size);
                const start = self.len * row_blocks;
                const span = m * row_blocks;
                try ctx.quantizeF32RowsToQ8_0Into(k_rows.asRawTensor(), self.k_q8[layer_i][start..][0..span]);
                try ctx.quantizeF32RowsToQ8_0Into(v_rows.asRawTensor(), self.v_q8[layer_i][start..][0..span]);
            },
        }
    }

    pub fn advance(self: *KvCache, m: usize) void {
        self.len += m;
    }

    /// Rewind the cache to its first `keep_len` positions (clamp: a `keep_len`
    /// at or above `len` is a no-op). Decrementing `len` is sufficient for BOTH
    /// storage modes: the buffers are pre-allocated at `capacity` and never
    /// shrink, each position occupies whole per-(position, kv_head) rows — f16
    /// rows of `head_dim` halves; q8_0 rows of `head_dim/32` BlockQ8_0 (init
    /// enforces `head_dim % 32 == 0`, so no block ever straddles positions) —
    /// and every reader (`k`/`v` narrows, `kBlocks`/`vBlocks`) as well as
    /// `appendLayer` (which writes at offset `len`) addresses rows strictly
    /// from `len`: the next append simply overwrites the abandoned rows.
    /// Speculative decoding uses this to drop rejected draft positions.
    pub fn truncate(self: *KvCache, keep_len: usize) void {
        if (keep_len < self.len) self.len = keep_len;
    }
};

test {
    _ = @import("kv_cache_tests.zig");
}
