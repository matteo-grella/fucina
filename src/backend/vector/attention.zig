//! The attention seam: the Task payloads of the grouped-causal attention
//! family, the `run*Task` pool adapters the exec dispatch hands to
//! `Pool.parallelChunks` (these dispatches carry per-part scratch, so they
//! stay on the substrate), the tile constants, and the KV-layout helpers.
//! The kernel bodies live in `attention/kernels.zig` beneath this file and
//! are reached through `backend.kernels` only.
const std = @import("std");
const dtype_mod = @import("../../dtype.zig");
const kernels = @import("attention/kernels.zig");

const DType = dtype_mod.DType;

pub fn kvDtypeOf(comptime KvElem: type) DType {
    return if (KvElem == f32) .f32 else .f16;
}

pub const BlockQ8_0 = dtype_mod.BlockQ8_0;

pub const q8_0_block_size = dtype_mod.q8_0_block_size;

/// Head dims the per-query q8_0 attention kernels support: the per-task
/// dequant scratch row is this many f32s on the stack (Gemma's widest global
/// head is 512). Validated at the q8_0 arms of `groupedAttention`.
pub const attention_q8_max_d: usize = 512;

pub fn GroupedCausalAttentionTask(comptime KvElem: type) type {
    return struct {
        q_data: []const f32,
        k_data: []const KvElem,
        v_data: []const KvElem,
        out_data: []f32,
        kv_head_for_head: []const usize,
        q_seq: usize,
        kv_seq: usize,
        source_offset: usize,
        heads: usize,
        d: usize,
        kv_heads: usize,
        scale_value: f32,
        // Sliding-window attention: 0 = full causal; else a query at absolute
        // position `p` attends keys in `[max(0, p-window+1), p]` (local SWA).
        window: usize,
        // false = bidirectional: every query attends ALL kv_seq keys (block
        // diffusion canvas attention). Requires window == 0 — a canvas window
        // is realized by narrowing the K/V views instead.
        causal: bool = true,
        // Optional additive f32 bias, row-contiguous [q_seq, kv_seq]: added
        // to the SCALED score before the softmax — score(query_i, source_i)
        // = dot * scale_value + bias[query_i * kv_seq + source_i]. Null = no
        // bias; non-null only via the bidirectional biased entry today.
        bias: ?[]const f32 = null,
        head_start: usize,
        head_end: usize,
        scores: []f32,
        // Optional per-(head, query) softmax statistics out: {max, sum_exp}
        // interleaved f32 pairs at (head_i * q_seq + query_i) * 2. Each
        // kernel records the normalizer IT used (two-pass max/sum here and
        // on the pair kernel; the tiled kernel's online-rescaled running
        // pair), so a backward fed with them reconstructs that kernel's
        // probabilities. Disjoint writes across tasks.
        stats: ?[]f32 = null,
    };
}

pub fn GroupedCausalAttentionPairTask(comptime KvElem: type) type {
    return struct {
        q_data: []const f32,
        k_data: []const KvElem,
        v_data: []const KvElem,
        out_data: []f32,
        q_seq: usize,
        kv_seq: usize,
        source_offset: usize,
        heads: usize,
        d: usize,
        kv_heads: usize,
        scale_value: f32,
        // Sliding-window attention: 0 = full causal (see GroupedCausalAttentionTask).
        window: usize,
        // false = bidirectional (see GroupedCausalAttentionTask).
        causal: bool = true,
        // Optional additive [q_seq, kv_seq] bias (see GroupedCausalAttentionTask).
        bias: ?[]const f32 = null,
        kv_head_start: usize,
        kv_head_end: usize,
        scores: []f32,
        // Optional {max, sum_exp} stats out (see GroupedCausalAttentionTask).
        stats: ?[]f32 = null,
    };
}

/// Query rows in flight per tile in the query-tiled attention forward:
/// queries per tile = `attention_tile_rows / head_group` (head_group = 2 on
/// the adjacent-pair GQA path shares each K/V row across the pair as well).
/// With `attention_key_block` keys per inner step the score microkernel keeps
/// rows x keys = 16 @Vector(4, f32) dot accumulators plus both operand rows
/// in registers — the NEON file's sweet spot; 8 rows would spill.
/// Picked by benchmark on M1 Max (see docs/BENCHMARK.md protocol).
pub const attention_tile_rows: usize = 4;

/// Keys processed per inner step. Blocking keys amortizes the q-row reloads
/// of the score pass and the accumulator load/stores of the value pass (the
/// L1-port bottleneck of a 1-key-at-a-time online-softmax kernel) and means
/// one running-max check per block instead of per key.
pub const attention_key_block: usize = 4;

/// The tiled kernel keeps per-row output accumulators on the stack
/// (`rows * attention_tile_max_d` floats, 4 KiB at 4 rows x 256): L1-resident
/// and allocation-free. Heads wider than this stay on the per-query kernels.
pub const attention_tile_max_d: usize = 256;

/// Prefill length at which the tiled kernel takes over. Decode (q_seq == 1)
/// and short prefill stay on the per-query kernels above, whose results
/// remain bit-identical. Tuned by benchmark: below this the K/V re-stream
/// fits in cache and the per-query kernels' simpler inner loop wins.
pub const attention_tiled_min_q_seq: usize = 48;

pub fn GroupedCausalAttentionTiledTask(comptime KvElem: type) type {
    return struct {
        q_data: []const f32,
        k_data: []const KvElem,
        v_data: []const KvElem,
        out_data: []f32,
        // Only read on the general (head_group == 1) path; the pair path maps
        // head_unit -> kv head implicitly like GroupedCausalAttentionPairTask.
        kv_head_for_head: []const usize,
        q_seq: usize,
        kv_seq: usize,
        source_offset: usize,
        heads: usize,
        d: usize,
        kv_heads: usize,
        scale_value: f32,
        // Sliding-window attention: 0 = full causal (see GroupedCausalAttentionTask).
        window: usize,
        // false = bidirectional (see GroupedCausalAttentionTask).
        causal: bool = true,
        // Optional additive [q_seq, kv_seq] bias (see GroupedCausalAttentionTask).
        bias: ?[]const f32 = null,
        n_tiles: usize,
        // Work items are flattened head-major: `head_unit * n_tiles + tile_i`.
        work_start: usize,
        work_end: usize,
        // Optional {max, sum_exp} stats out (see GroupedCausalAttentionTask);
        // this kernel records its online-rescaled running pair.
        stats: ?[]f32 = null,
    };
}

pub const GroupedCausalAttentionBackwardTask = struct {
    q_data: []const f32,
    k_data: []const f32,
    v_data: []const f32,
    gy_data: []const f32,
    q_grad: ?[]f32,
    k_grad: ?[]f32,
    v_grad: ?[]f32,
    kv_head_for_head: []const usize,
    q_seq: usize,
    kv_seq: usize,
    source_offset: usize,
    heads: usize,
    d: usize,
    kv_heads: usize,
    scale_value: f32,
    window: usize,
    // false = bidirectional (see GroupedCausalAttentionTask).
    causal: bool = true,
    kv_head_start: usize,
    kv_head_end: usize,
    scores: []f32,
    dprob: []f32,
};

pub fn runGroupedCausalAttentionTask(comptime KvElem: type) fn (*const GroupedCausalAttentionTask(KvElem)) void {
    return struct {
        fn run(task: *const GroupedCausalAttentionTask(KvElem)) void {
            kernels.groupedCausalAttentionHeads(KvElem, task.*);
        }
    }.run;
}

pub fn runGroupedCausalAttentionPairTask(comptime KvElem: type) fn (*const GroupedCausalAttentionPairTask(KvElem)) void {
    return struct {
        fn run(task: *const GroupedCausalAttentionPairTask(KvElem)) void {
            kernels.groupedCausalAttentionHeadPairs(KvElem, task.*);
        }
    }.run;
}

pub fn runGroupedCausalAttentionTiledTask(comptime KvElem: type, comptime head_group: usize) fn (*const GroupedCausalAttentionTiledTask(KvElem)) void {
    return struct {
        fn run(task: *const GroupedCausalAttentionTiledTask(KvElem)) void {
            kernels.groupedCausalAttentionQueryTiles(KvElem, head_group, task.*);
        }
    }.run;
}

pub fn runGroupedCausalAttentionBackwardTask(task: *const GroupedCausalAttentionBackwardTask) void {
    kernels.groupedCausalAttentionBackwardKvHeads(task.*);
}

/// Keys scanned by one query tile — the load-balance weight of a work item in
/// the tiled attention split (late tiles attend more keys under causal
/// masking, so equal-count task ranges would be lopsided). Bidirectional
/// tiles all scan the full key range, so their weights are uniform.
pub fn attentionTileKeyCount(q_tile: usize, q_seq: usize, kv_seq: usize, source_offset: usize, window: usize, causal: bool, tile_i: usize) u64 {
    if (!causal) return kv_seq;
    const q0 = tile_i * q_tile;
    const rows_active = @min(q_tile, q_seq - q0);
    const p_first = source_offset + q0;
    const active_last = p_first + rows_active;
    const lo_first = if (window == 0) 0 else (p_first + 1) -| window;
    return active_last - lo_first;
}

/// Panel rows per backward q-tile: the P and dS panels are
/// `attention_bwd_tile_rows * kv_seq` floats each per task, so a tile's
/// score/probability rows stay cache-resident across all five backward
/// contractions instead of round-tripping full `[group_rows, kv_seq]`
/// panels through memory five times (the retired GEMM route).
pub const attention_bwd_tile_rows: usize = 16;

pub const GroupedCausalAttentionBackwardTiledTask = struct {
    q_data: []const f32,
    k_data: []const f32,
    v_data: []const f32,
    gy_data: []const f32,
    /// Forward-saved {max, sum_exp} per (head, query), or null to recompute.
    stats: ?[]const f32,
    q_grad: ?[]f32,
    /// dK/dV accumulation targets. Partial mode (shared kv heads): contiguous
    /// `[heads][kv_seq][d]` planes indexed by GLOBAL head, reduced in fixed
    /// head order after the pool joins. Direct mode (one query head per kv
    /// head): the `[kv_seq, kv_heads, d]` gradient tensors themselves.
    dk_target: ?[]f32,
    dv_target: ?[]f32,
    partial_mode: bool,
    kv_head_for_head: []const usize,
    q_seq: usize,
    kv_seq: usize,
    source_offset: usize,
    heads: usize,
    d: usize,
    kv_heads: usize,
    scale_value: f32,
    window: usize,
    causal: bool,
    /// Per-task scratch: 2 panels of `attention_bwd_tile_rows * kv_seq`.
    scratch: []f32,
    head_start: usize,
    head_end: usize,
};

pub fn runGroupedCausalAttentionBackwardTiledTask(task: *const GroupedCausalAttentionBackwardTiledTask) void {
    kernels.groupedCausalAttentionBackwardTiles(task.*);
}

/// Wider q-tiles for the BLAS-strip route: the sgemm strips need enough
/// rows to reach the GEMM engine's efficient regime; the panels stay
/// L2-resident (`2 * 64 * kv_seq` floats per task).
pub const attention_bwd_blas_tile_rows: usize = 64;

pub fn runGroupedCausalAttentionBackwardBlasTiledTask(task: *const GroupedCausalAttentionBackwardTiledTask) void {
    kernels.groupedCausalAttentionBackwardBlasTiles(task.*);
}

/// Fixed-head-order reduction of the per-head dK/dV planes into the
/// `[kv_seq, kv_heads, d]` gradients — bitwise identical for any task count
/// (each source row sums its group's heads in ascending head order).
pub const AttentionBackwardReduceTask = struct {
    partials: []const f32,
    grad: []f32,
    kv_head_for_head: []const usize,
    kv_seq: usize,
    d: usize,
    kv_heads: usize,
};

pub fn hasAdjacentKvHeadPairs(kv_head_for_head: []const usize, heads: usize, kv_heads: usize) bool {
    if (heads != kv_heads * 2) return false;
    for (0..kv_heads) |kv_head_i| {
        if (kv_head_for_head[kv_head_i * 2] != kv_head_i) return false;
        if (kv_head_for_head[kv_head_i * 2 + 1] != kv_head_i) return false;
    }
    return true;
}

/// One (stream × head-unit) work range of the ragged multi-stream decode
/// attention (the multi-stream `KvView` arms): query row `s` of `q_data`
/// attends the leading `lens[s]` cached rows of stream `s`'s K/V slices.
/// Work items are stream-major (`stream_i * n_units + unit_i`); each item
/// runs ONE head unit of ONE stream through the per-query decode kernels,
/// so per-stream results are bit-identical to N single-stream calls.
pub fn GroupedCausalAttentionMultiTask(comptime KvElem: type) type {
    return struct {
        q_data: []const f32,
        out_data: []f32,
        ks: []const []const KvElem,
        vs: []const []const KvElem,
        lens: []const usize,
        kv_head_for_head: []const usize,
        heads: usize,
        d: usize,
        kv_heads: usize,
        scale_value: f32,
        /// Head units per stream: kv-head pairs on the pair path, single
        /// query heads on the general path.
        n_units: usize,
        work_start: usize,
        work_end: usize,
        /// Softmax scratch, `2 * max(lens)` floats (the pair kernel needs
        /// two rows; the general path uses the first `lens[s]`).
        scores: []f32,
    };
}

pub fn runGroupedCausalAttentionMultiTask(comptime KvElem: type, comptime pair: bool) fn (*const GroupedCausalAttentionMultiTask(KvElem)) void {
    return struct {
        fn run(task: *const GroupedCausalAttentionMultiTask(KvElem)) void {
            kernels.groupedCausalAttentionMultiUnits(KvElem, pair, task.*);
        }
    }.run;
}

/// Elements per cached (position) row: `kv_heads * d` f16/f32 lanes, or
/// `kv_heads * d/32` BlockQ8_0.
pub fn kvRowElems(comptime KvElem: type, kv_heads: usize, d: usize) usize {
    return if (KvElem == BlockQ8_0) kv_heads * (d / q8_0_block_size) else kv_heads * d;
}
