//! The accelerator seam: the one module outside the providers that names
//! `gpu_impl`. Every entry here makes the offload decision (capability,
//! work gate, layout preconditions) and dispatches; a refusal returns
//! `false`/`null` and the caller runs its CPU path. Exec, weights and the
//! facade talk to this file, never to a provider, so a provider is a
//! plug-in behind one contract (`gpu_provider.zig`) and the rest of the
//! library carries no GPU branches of its own. With `-Dgpu=none` every
//! entry folds to its refusal at comptime.

const std = @import("std");
const gpu_provider = @import("gpu_provider.zig");
const gpu = @import("gpu.zig").impl;
const tensor = @import("../tensor.zig");
const dtype_mod = @import("../dtype.zig");

const Tensor = tensor.Tensor;
const DType = dtype_mod.DType;

/// A provider is selected (`-Dgpu=metal|cuda`).
pub const enabled = gpu.enabled;
/// The provider has quantized GEMM kernels at all.
pub const has_quant_gemm = gpu.has_quant_gemm;

pub const deviceName = gpu.deviceName;
pub const traceEnabled = gpu.traceEnabled;
pub const traceReset = gpu.traceReset;
pub const traceDump = gpu.traceDump;

/// Device-resident storage for weights the provider may keep on the device
/// (the resident-RHS caches). `null` when nothing is resident.
pub const allocResidentBytes = gpu.allocResidentBytes;
pub const freeResidentBytes = gpu.freeResidentBytes;

// Test hooks (the provider's gates, pinned by the model tests).
pub const deviceAvailableForTest = gpu.deviceAvailableForTest;
pub const setMinWorkQMoeForTest = gpu.setMinWorkQMoeForTest;
pub const qmoeMinFillForTest = gpu.qmoeMinFillForTest;
pub const setQmoeMinFillForTest = gpu.setQmoeMinFillForTest;

// ---------------------------------------------------------------------------
// Quantized GEMM
// ---------------------------------------------------------------------------

/// The quantized RHS layouts an accelerator may take — the one format
/// vocabulary (`gpu_provider.QuantFormat`; `tq2_0_folded` is the ternary
/// folded-plane layout, `weights/ptqtp.zig`). A provider's kernel-side
/// integer for a format is its private ABI (`abiValue`), never seen here.
pub const QuantFormat = gpu_provider.QuantFormat;

/// The provider has a kernel for `fmt` — THE per-format capability
/// predicate (`abiValue(fmt) != null`); every layer above asks this, never
/// a re-spelled format list.
pub fn supportsQuant(comptime fmt: QuantFormat) bool {
    if (!enabled) return false;
    return gpu.abiValue(fmt) != null;
}

/// `supportsQuant` for a runtime format.
pub fn supportsQuantAt(fmt: QuantFormat) bool {
    return switch (fmt) {
        inline else => |f| comptime supportsQuant(f),
    };
}

/// `supportsQuant` over a storage dtype (false for non-quantized dtypes).
pub fn supportsQuantDType(comptime dt: DType) bool {
    const fmt = QuantFormat.fromDType(dt) orelse return false;
    return supportsQuant(fmt);
}

/// Which CPU path the caller falls back to; the providers gate the two
/// differently (the packed CPU kernels are faster, so the device must win
/// by more).
pub const QuantGemmArm = enum { blocks, panels };

fn quantWork(m: usize, n: usize, k: usize) u64 {
    const mm: u64 = @intCast(m);
    const nn: u64 = @intCast(n);
    const kk: u64 = @intCast(k);
    const mn = std.math.mul(u64, mm, nn) catch return std.math.maxInt(u64);
    return std.math.mul(u64, mn, kk) catch std.math.maxInt(u64);
}

/// The offload decision for `input[m,k] · rhs[n,k]ᵀ` with a quantized RHS:
/// a prefill arm (`m >= 32`, the provider's dense-quant gate for `arm`) or
/// a decode arm (`m <= 8`, the provider's decode-GEMV gate; the folded
/// ternary layout has no decode kernel), plus the layout preconditions.
pub fn quantGemmAccepts(comptime fmt: QuantFormat, m: usize, n: usize, k: usize, input_contiguous: bool, comptime arm: QuantGemmArm) bool {
    if (comptime !enabled) return false;
    if (comptime !supportsQuant(fmt)) return false;
    const work = quantWork(m, n, k);
    const prefill_arm = m >= 32 and switch (arm) {
        .panels => gpu.shouldUseGpuDenseQuantPacked(fmt, work),
        .blocks => gpu.shouldUseGpuDenseQuant(fmt, work),
    };
    const decode_arm = fmt != .tq2_0_folded and m <= 8 and gpu.shouldUseGpuQuantDecode(fmt, m, n, k);
    const k_mult = comptime fmt.kMultiple();
    return (prefill_arm or decode_arm) and k % k_mult == 0 and n % 4 == 0 and input_contiguous;
}

/// Run the quantized GEMM the caller asked `quantGemmAccepts` about, into
/// `out[m,n]` (f32). A cacheable RHS goes through the async resident path;
/// otherwise the rows are dispatched synchronously in chunks. `false` when
/// the provider declined (the caller frees `out` and runs the CPU path).
pub fn gemmQuant(comptime fmt: QuantFormat, rhs_bytes: []const u8, cacheable: bool, nb01: usize, input: *const Tensor, out: *Tensor, m: usize, n: usize, k: usize) bool {
    if (comptime !enabled) return false;
    if (comptime !supportsQuant(fmt)) return false;
    var req: gpu_provider.QuantGemmRequest = .{
        .format = fmt,
        .rhs = rhs_bytes,
        .rhs_cacheable = cacheable,
        .nb01 = nb01,
        .m = m,
        .n = n,
        .k = k,
    };
    if (cacheable and gpu.gemmQuantNtAsync(req, input, out)) return true;

    const in_data = input.dataConst();
    const in_elems = std.math.mul(usize, m, k) catch return false;
    if (in_data.len != in_elems) return false;
    const max_rows_per_dispatch = 2048;
    const n_chunks = (m + max_rows_per_dispatch - 1) / max_rows_per_dispatch;
    const rows_per = (m + n_chunks - 1) / n_chunks;
    var row0: usize = 0;
    while (row0 < m) : (row0 += rows_per) {
        const rows = @min(rows_per, m - row0);
        req.m = rows;
        if (!gpu.gemmQuantNt(req, in_data[row0 * k .. (row0 + rows) * k], out.data()[row0 * n .. (row0 + rows) * n])) return false;
    }
    return true;
}

/// The offload decision for `batch_count` GEMMs sharing one `input[m,k]`
/// against `batch_count` quantized RHS slabs (`nb02` bytes apart).
pub fn quantGemmSharedInputAccepts(comptime fmt: QuantFormat, batch_count: usize, m: usize, n: usize, k: usize, input_contiguous: bool) bool {
    if (comptime !enabled) return false;
    if (comptime !supportsQuant(fmt)) return false;
    if (batch_count == 0) return false;
    const per_work = quantWork(m, n, k);
    const work = std.math.mul(u64, per_work, @as(u64, @intCast(batch_count))) catch std.math.maxInt(u64);
    const k_mult = comptime fmt.kMultiple();
    return m >= 32 and k % k_mult == 0 and n % 4 == 0 and input_contiguous and gpu.shouldUseGpuDenseQuant(fmt, work);
}

/// Run the shared-input batch into `out[batch_count * m, n]`.
pub fn gemmQuantSharedInput(comptime fmt: QuantFormat, rhs_bytes: []const u8, cacheable: bool, nb01: usize, nb02: usize, input: *const Tensor, out: *Tensor, batch_count: usize, m: usize, n: usize, k: usize) bool {
    if (comptime !enabled) return false;
    if (comptime !supportsQuant(fmt)) return false;
    const req: gpu_provider.QuantGemmRequest = .{
        .format = fmt,
        .rhs = rhs_bytes,
        .rhs_cacheable = cacheable,
        .nb01 = nb01,
        .nb02 = nb02,
        .batch = batch_count,
        .m = m,
        .n = n,
        .k = k,
    };
    if (cacheable and gpu.gemmQuantNtAsync(req, input, out)) return true;
    const in_data = input.dataConst();
    const in_elems = std.math.mul(usize, m, k) catch return false;
    if (m > 2048 or in_data.len != in_elems) return false;
    return gpu.gemmQuantNtSharedABatch(req, in_data, out.data());
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

const attn_prefill_max_heads = 64;
const attn_prefill_max_head_dim = 256;

/// The fused f16-KV prefill attention (grouped heads, window, causal). The
/// head map is built here from `kv_head_for_head`, or as pairs when
/// `head_group == 2`. `false` when the provider declines or the shape is
/// outside the kernel's limits.
pub fn attnPrefillF16(
    q: []const f32,
    k: []const f16,
    v: []const f16,
    out: []f32,
    kv_head_for_head: []const usize,
    head_group: usize,
    q_seq: usize,
    kv_seq: usize,
    heads: usize,
    kv_heads: usize,
    d: usize,
    source_offset: usize,
    scale: f32,
    window: usize,
    causal: bool,
    has_bias: bool,
) bool {
    if (comptime !enabled) return false;
    if (has_bias or d > attn_prefill_max_head_dim or heads > attn_prefill_max_heads) return false;
    if (!gpu.shouldUseGpuAttn(q_seq, kv_seq, heads, d)) return false;
    var map_buf: [attn_prefill_max_heads]i32 = undefined;
    for (0..heads) |h| {
        map_buf[h] = if (head_group == 2) @intCast(h / 2) else @intCast(kv_head_for_head[h]);
    }
    return gpu.attnPrefillF16(q, k, v, out, map_buf[0..heads], .{
        .q_seq = q_seq,
        .kv_seq = kv_seq,
        .heads = heads,
        .kv_heads = kv_heads,
        .d = d,
        .window = window,
        .causal = causal,
        .scale = scale,
        .source_offset = source_offset,
    });
}

/// The attention forward with row statistics (the training path). Only the
/// regular grouped layout (`kv_head = head / heads_per_kv`) and no bias.
pub fn attentionFwd(
    comptime KvElem: type,
    q: []const f32,
    k: []const KvElem,
    v: []const KvElem,
    out: []f32,
    stats: ?[]f32,
    kv_head_for_head: []const usize,
    q_seq: usize,
    kv_seq: usize,
    heads: usize,
    kv_heads: usize,
    d: usize,
    window: usize,
    causal: bool,
    scale: f32,
    has_bias: bool,
) bool {
    if (comptime !(enabled and gpu.has_attention_fwd)) return false;
    if (comptime (KvElem != f32 and KvElem != f16)) return false;
    if (has_bias) return false;
    if (kv_heads == 0 or heads % kv_heads != 0) return false;
    const heads_per_kv = heads / kv_heads;
    for (kv_head_for_head, 0..) |kv_head_i, head_i| {
        if (kv_head_i != head_i / heads_per_kv) return false;
    }
    if (!gpu.shouldUseGpuAttentionFwd(q_seq, kv_seq, heads, d)) return false;
    const req: gpu_provider.AttentionRequest = .{
        .q_seq = q_seq,
        .kv_seq = kv_seq,
        .heads = heads,
        .kv_heads = kv_heads,
        .d = d,
        .window = @min(window, kv_seq),
        .causal = causal,
        .scale = scale,
        .heads_per_kv = heads_per_kv,
    };
    return if (comptime KvElem == f32)
        gpu.attentionFwdF32(q, k, v, out, stats, req)
    else
        gpu.attentionFwdF16Kv(q, k, v, out, stats, req);
}

// ---------------------------------------------------------------------------
// Grouped quantized MoE
// ---------------------------------------------------------------------------

/// One grouped-MoE tile (32 rows of one expert group); crosses to the
/// kernel side verbatim.
pub const QMMTile = gpu_provider.QMMTile;

/// The offload decision for a grouped-MoE batch: enough tile fill and enough
/// work for the provider's gates.
pub fn qmoeAccepts(n_pairs: usize, n_tiles: usize, work: u64) bool {
    if (comptime !enabled) return false;
    return gpu.qmoeFillAcceptable(n_pairs, n_tiles) and gpu.shouldUseGpuQMoe(work);
}

/// One grouped-MoE batch on the device: holds the provider's staging
/// buffers (`stage.in`/`stage.out`, host-visible) and its lock from `begin`
/// to `end`. The caller fills `stage.in`, runs the grouped GEMMs, reads
/// `stage.out`.
pub const QMoeSession = struct {
    stage: gpu_provider.QMoeStage,

    pub fn begin(in_bytes: usize, out_bytes: usize) ?QMoeSession {
        if (comptime !enabled) return null;
        gpu.qmoe_lock.lock();
        const stage = gpu.qmoeStage(in_bytes, out_bytes) orelse {
            gpu.qmoe_lock.unlock();
            return null;
        };
        return .{ .stage = stage };
    }

    /// `stage.out[tile rows, n] = stage.in[tile rows, k] · rhs[expert][n,k]ᵀ`
    /// for every tile. `false` when the provider declined.
    pub fn gemmGrouped(self: *const QMoeSession, fmt: QuantFormat, rhs_bytes: []const u8, cacheable: bool, nb01: usize, nb02: usize, n: usize, k: usize, tiles: []const QMMTile) bool {
        _ = self;
        if (comptime !enabled) return false;
        if (!supportsQuantAt(fmt)) return false;
        return gpu.gemmQGroupedNt(.{
            .format = fmt,
            .rhs = rhs_bytes,
            .rhs_cacheable = cacheable,
            .nb01 = nb01,
            .nb02 = nb02,
            .m = 0, // rows live in the tile table
            .n = n,
            .k = k,
        }, tiles);
    }

    pub fn end(self: *QMoeSession) void {
        if (comptime !enabled) return;
        gpu.qmoe_lock.unlock();
        self.* = undefined;
    }
};

// ---------------------------------------------------------------------------
// Evolution-strategies flat kernels (device-resident parameter slabs)
// ---------------------------------------------------------------------------

pub const FlatDType = gpu_provider.FlatDType;
pub const flatPerturb = gpu.flatPerturb;
pub const flatWeightedUpdate = gpu.flatWeightedUpdate;
pub const flatAnchorDecay = gpu.flatAnchorDecay;
