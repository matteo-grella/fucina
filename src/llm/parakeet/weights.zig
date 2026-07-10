//! Session-lifetime weight cache for the Parakeet model. The hot inference path
//! borrows stable GGUF f16/quantized bytes directly, promotes reused Q6_K/Q8_0
//! bytes into Metal-resident storage when profitable, and falls back to cached
//! packed weights for small quantized batches or grad-requiring tensors. F32 BLAS
//! caches decoded weights when explicitly enabled.
//!
//! Lazy, by-name: works with partial synthetic GGUFs (unit tests) and the real
//! model alike. Pointers into the cache are stable (the value is a heap pointer),
//! so a returned `*const LinearWeight` survives later inserts.
const std = @import("std");
const builtin = @import("builtin");
const fucina = @import("fucina");
const gguf = fucina.gguf;
const weights = @import("../weights.zig");

const DType = fucina.DType;
const ExecContext = fucina.ExecContext;
const Allocator = std.mem.Allocator;
// RAW (allowed): the dispatch-batched q6 helper produces one raw output that is
// immediately wrapped as a Tensor. Ordinary per-weight borrowed RHS dispatch
// lives in src/llm/weights.zig.
const RawTensor = fucina.internal.RawTensor;

/// Tagged facade for the linear bridge: a raw `[T,in]` activation is wrapped as
/// `(.seq,.in)` and `linearSeq` returns `(.seq,.out)`.
const SeqIn = fucina.Tensor(.{ .seq, .in });

const LinearShape = struct {
    in: usize,
    out: usize,
    info2: gguf.TensorInfo,
};

const Tensor2 = fucina.Tensor(2);

pub const ParakeetWeights = struct {
    ctx: *ExecContext,
    file: *const gguf.File,
    allocator: Allocator,
    cache: std.StringHashMapUnmanaged(*weights.LinearWeight) = .empty,
    f32_cache: std.StringHashMapUnmanaged(*weights.WeightF32) = .empty,
    qkv_cache: std.AutoHashMapUnmanaged(usize, *weights.QuantByteStack) = .empty,
    pos_cache: ?*weights.QuantByteStack = null,
    seq_once: std.StringHashMapUnmanaged(void) = .empty,
    prefer_f32_blas: bool = false,
    resident: weights.ResidentByteRegistry,

    pub fn init(ctx: *ExecContext, file: *const gguf.File) ParakeetWeights {
        return .{ .ctx = ctx, .file = file, .allocator = ctx.allocator, .resident = weights.ResidentByteRegistry.init(ctx.allocator) };
    }

    pub fn deinit(self: *ParakeetWeights) void {
        var it = self.cache.iterator();
        while (it.next()) |e| {
            e.value_ptr.*.deinit();
            self.allocator.destroy(e.value_ptr.*);
            self.allocator.free(e.key_ptr.*);
        }
        self.cache.deinit(self.allocator);
        var f32_it = self.f32_cache.iterator();
        while (f32_it.next()) |e| {
            e.value_ptr.*.deinit();
            self.allocator.destroy(e.value_ptr.*);
            self.allocator.free(e.key_ptr.*);
        }
        self.f32_cache.deinit(self.allocator);
        var qkv_it = self.qkv_cache.iterator();
        while (qkv_it.next()) |e| {
            e.value_ptr.*.deinit(self.allocator);
            self.allocator.destroy(e.value_ptr.*);
        }
        self.qkv_cache.deinit(self.allocator);
        var once_it = self.seq_once.iterator();
        while (once_it.next()) |e| self.allocator.free(e.key_ptr.*);
        self.seq_once.deinit(self.allocator);
        if (self.pos_cache) |p| {
            p.deinit(self.allocator);
            self.allocator.destroy(p);
        }
        self.resident.deinit();
        self.* = undefined;
    }

    pub fn enableF32Blas(self: *ParakeetWeights) void {
        self.prefer_f32_blas = true;
    }

    /// Lazily load + pack the linear weight `w_name` as a `[out, in]` matrix.
    /// Derives `in`/`out` from the GGUF info: a 2-D nn.Linear weight is ggml
    /// `[in, out]`; the conv pointwise weights are ggml `[1, in, out]` (a 1×1
    /// conv) — the leading unit dim is squeezed (same flat layout). Cached by
    /// name; the returned pointer is stable across later inserts.
    pub fn getLinear(self: *ParakeetWeights, w_name: []const u8) !*const weights.LinearWeight {
        if (self.cache.get(w_name)) |p| return p;
        const wi = try self.file.get(w_name);
        const shape = try linearShape(wi);

        const lw = try self.allocator.create(weights.LinearWeight);
        errdefer self.allocator.destroy(lw);
        lw.* = try weights.LinearWeight.load(self.ctx, &shape.info2, shape.out, shape.in);
        errdefer lw.deinit();
        const key = try self.allocator.dupe(u8, w_name);
        errdefer self.allocator.free(key);
        try self.cache.put(self.allocator, key, lw);
        return lw;
    }

    pub fn getLinearF32(self: *ParakeetWeights, w_name: []const u8) !*const weights.WeightF32 {
        if (self.f32_cache.get(w_name)) |p| return p;
        const wi = try self.file.get(w_name);
        if (!canCacheF32(wi.ggml_type)) return weights.Error.UnsupportedWeightType;
        const shape = try linearShape(wi);
        const len = try std.math.mul(usize, shape.out, shape.in);
        gguf.prefetch(wi.data);

        const values = try self.allocator.alloc(f32, len);
        defer self.allocator.free(values);
        try gguf.decodeF32(wi.ggml_type, wi.data, values);

        const wt = try self.allocator.create(weights.WeightF32);
        errdefer self.allocator.destroy(wt);
        wt.* = try weights.WeightF32.fromSlice(self.ctx, .{ shape.out, shape.in }, values);
        errdefer wt.deinit();
        const key = try self.allocator.dupe(u8, w_name);
        errdefer self.allocator.free(key);
        try self.f32_cache.put(self.allocator, key, wt);
        return wt;
    }

    /// `y[seq,out] = x[seq,in] @ W(`w_name`)ᵀ` (no bias — callers add it). The
    /// Tensor-first boundary for the encoder/subsampling/CTC/joint linears:
    /// a `(.seq,.in)` activation in, a `(.seq,.out)` activation out. Keeps the
    /// three perf arms — f32 BLAS (`matmul` .trans_b, grad-preserving), borrowed-mmap
    /// f16/quantized (shared LLM weight helpers; Exec chooses Metal vs CPU for
    /// quantized RHS), and packed `linearSeq` (grad-preserving). Grad-requiring
    /// inputs skip the no-grad borrowed quantized seam and take a grad-preserving
    /// arm (Parakeet inference never requires grad, so the fast borrowed path is
    /// always taken in practice).
    pub fn linear(self: *ParakeetWeights, w_name: []const u8, x: *const SeqIn) !fucina.Tensor(.{ .seq, .out }) {
        if (try self.shouldUseF32Blas(w_name, x)) {
            const wt = try self.getLinearF32(w_name);
            return x.matmul(self.ctx, wt, .trans_b, .{ .seq, .out });
        }
        if (!x.requiresGrad()) {
            if (try self.linearBorrowedCold(w_name, x)) |out| return out;
        }
        const lw = try self.getLinear(w_name);
        return lw.linearSeq(self.ctx, x, .in, .out);
    }

    /// Tensor-first linear over the encoder/subsampling `.{._0,._1}` carry shape:
    /// retags `.{._0,._1}`→`.{.seq,.in}` for `linear` (a grad-preserving view —
    /// `withTags` retains the buffer), adds the optional row bias in place, then
    /// retags the result back to `.{._0,._1}`. `bias` is the already-resolved slice
    /// (callers resolve it via f32Data) so this stays free of gguf helpers. Shared
    /// by `encoder.linearWT` and `subsampling` (which cannot import the encoder).
    pub fn linearD(self: *ParakeetWeights, w_name: []const u8, bias: ?[]const f32, x: *const fucina.Tensor(2)) !fucina.Tensor(2) {
        var x_in = try x.withTags(self.ctx, .{ .seq, .in });
        defer x_in.deinit();
        var y = try self.linear(w_name, &x_in);
        errdefer y.deinit();
        if (bias) |b| try y.addAxisVectorInPlace(self.ctx, b, .out);
        const out = try y.withTags(self.ctx, 2);
        y.deinit(); // `out` retains the buffer
        return out;
    }

    /// Output-stacked q/k/v projection for the q6_k no-grad inference path. This
    /// is not a general fused op: it is exactly one larger linear with rows
    /// `[q; k; v]`, then callers read column bands out of the resulting `[T,3D]`
    /// Tensor. Unsupported dtypes, grad-requiring inputs, or small batches return
    /// null and use the existing separate linears.
    pub fn linearQkvD(
        self: *ParakeetWeights,
        cache_key: usize,
        q_name: []const u8,
        k_name: []const u8,
        v_name: []const u8,
        q_bias: ?[]const f32,
        k_bias: ?[]const f32,
        v_bias: ?[]const f32,
        x: *const Tensor2,
    ) !?Tensor2 {
        if (x.requiresGrad()) return null;
        const x_raw = x.asRawTensor();
        const xv = try x_raw.rankView(2);
        if (xv.shape[0] < 16) return null;
        const qkv = (try self.getQkvWeightQ6(cache_key, q_name, k_name, v_name)) orelse return null;
        if (xv.shape[1] != qkv.in) return weights.Error.InvalidWeightShape;

        const blocks = try quantBlockSlice(fucina.BlockQ6_K, qkv.data);
        const rhs_lifetime: fucina.RhsLifetime = if (qkv.device_owned) .stable_process else .transient;
        var raw_out = try self.ctx.matmul2DWithQuantizedBlocksRhsOptions(.q6_k, x_raw, blocks, qkv.totalOutRows(), qkv.in, .{ .rhs_lifetime = rhs_lifetime });
        errdefer raw_out.deinit();
        try addQkvBias(&raw_out, qkv.out, q_bias, k_bias, v_bias);
        return try Tensor2.constant(self.ctx, raw_out);
    }

    /// Dispatch-batched `linear_pos` for all offline encoder layers. The weights
    /// stay as separate expert entries in the grouped quant kernel; this is a
    /// one-command eager batch, not a graph/fused public op. Result layout is
    /// `[layer_count * P, D]`, so layer `il` starts at row `il * P`.
    pub fn linearPosAllD(self: *ParakeetWeights, layer_count: usize, pos_emb: *const Tensor2) !?Tensor2 {
        if (pos_emb.requiresGrad()) return null;
        const pos_raw = pos_emb.asRawTensor();
        const pv = try pos_raw.rankView(2);
        if (pv.shape[0] < 32) return null;
        const batched = (try self.getLinearPosBatchQ6(layer_count)) orelse return null;
        if (pv.shape[1] != batched.in) return weights.Error.InvalidWeightShape;
        var raw_out = (try self.ctx.denseQuantMatmulGpuSharedInputBatch(
            .q6_k,
            batched.data,
            .stable_process,
            batched.bytesPerRow(),
            batched.bytes_per_weight,
            pos_raw,
            batched.count,
            pv.shape[0],
            batched.out,
            batched.in,
        )) orelse return null;
        errdefer raw_out.deinit();
        return try Tensor2.constant(self.ctx, raw_out);
    }

    fn linearBorrowedCold(self: *ParakeetWeights, w_name: []const u8, x: *const SeqIn) !?fucina.Tensor(.{ .seq, .out }) {
        const wi = try self.file.get(w_name);
        const shape = try linearShape(wi);
        // f16: borrow the mmap'd weight for EVERY batch size. The packed path
        // (getLinear → LinearWeight.linearSeq) runs the identical `matmulTransB2DWithF16Rhs`
        // kernel but first COPIES the full weight (loadDenseF16Weight) — pure overhead
        // that dominated streaming wall-clock on the 0.6B encoder (the one-time pack was
        // ~30% of a short clip's time). Numerics are bit-identical (same kernel, same
        // f16 values + [out,in] layout), so this is a free win for streaming/decode.
        if (wi.ggml_type == .f16) {
            return try weights.linearSeqBorrowedF16(self.ctx, x, wi.data, .{ shape.out, shape.in }, .in, .out);
        }
        // x86: the packed path's VNNI arms beat the direct-block borrow at
        // parakeet's offline m≈93 (i9-13950HX 8P, ReleaseFast, tdt_ctc-110m,
        // speech.wav, warmed best-of-20 / cold best-of-7): q8_0 wins packed in
        // BOTH regimes (steady 256→117 ms, cold 320→230) → always packed.
        // q5_k/q6_k split: packed wins steady (195→130 / 198→142) but the
        // one-time x4 pack loses cold (240→300 / 260→300) → borrow direct-block
        // on each weight's FIRST seq-batch use (a cold single shot never packs),
        // pack from the second use on (steady-state gets the VNNI kernels).
        // aarch64 keeps the direct-block borrow (measured faster cold on M1).
        if (comptime builtin.cpu.arch == .x86_64) {
            if (wi.ggml_type != .q5_k and wi.ggml_type != .q6_k) return null;
            if (x.dim(.seq) < 16) return null;
            if (!try self.firstSeqUse(w_name)) return null;
        }
        // Quantized: the packed path enables a faster small-batch GEMV, so keep the
        // prefill threshold and fall through to packed (linearSeq) for small batches.
        if (x.dim(.seq) < 16) return null;
        return switch (wi.ggml_type) {
            .q8_0 => try self.linearBorrowedQuantizedTyped(.q8_0, wi, shape, x),
            .q5_k => try self.linearBorrowedQuantizedTyped(.q5_k, wi, shape, x),
            .q6_k => try self.linearBorrowedQuantizedTyped(.q6_k, wi, shape, x),
            else => null,
        };
    }

    /// True exactly once per weight name: its first seq-batch (m>=16) use in
    /// this session. Drives the x86 k-quant cold/steady routing split in
    /// `linearBorrowedCold`.
    fn firstSeqUse(self: *ParakeetWeights, w_name: []const u8) !bool {
        if (self.seq_once.contains(w_name)) return false;
        const key = try self.allocator.dupe(u8, w_name);
        errdefer self.allocator.free(key);
        try self.seq_once.put(self.allocator, key, {});
        return true;
    }

    fn linearBorrowedQuantizedTyped(
        self: *ParakeetWeights,
        comptime dtype: DType,
        wi: *const gguf.TensorInfo,
        shape: LinearShape,
        x: *const SeqIn,
    ) !fucina.Tensor(.{ .seq, .out }) {
        // Residency: q6_k offloads at parakeet sizes (the 2²² dense gate)
        // and q8_0 can clear the QMOE gate on long clips, so back both with
        // device-owned bytes from the shared LLM registry — Metal keeps them
        // resident instead of re-wiring the pageable mmap every dispatch.
        // q5_k has no GPU kernel, so it keeps the borrowed mmap (no wasted
        // device memory). Only device-owned bytes may be cached by address:
        // promotion can fail (or be compiled out), and a plain mmap dies with
        // the session, so cacheability derives from whether promotion happened.
        const bytes = if (comptime (dtype == .q6_k or dtype == .q8_0)) self.resident.bytes(wi.data) else wi.data;
        const rhs_lifetime: fucina.RhsLifetime = if (bytes.ptr != wi.data.ptr) .stable_process else .transient;
        return weights.linearSeqBorrowedQuantized(dtype, self.ctx, x, bytes, .{ shape.out, shape.in }, .{
            .rhs_lifetime = rhs_lifetime,
        }, .in, .out);
    }

    fn getQkvWeightQ6(self: *ParakeetWeights, cache_key: usize, q_name: []const u8, k_name: []const u8, v_name: []const u8) !?*const weights.QuantByteStack {
        if (self.qkv_cache.get(cache_key)) |cached| return cached;
        const qi = try self.file.get(q_name);
        const ki = try self.file.get(k_name);
        const vi = try self.file.get(v_name);
        if (qi.ggml_type != .q6_k or ki.ggml_type != .q6_k or vi.ggml_type != .q6_k) return null;
        const qs = try linearShape(qi);
        const ks = try linearShape(ki);
        const vs = try linearShape(vi);
        if (qs.in != ks.in or qs.in != vs.in or qs.out != ks.out or qs.out != vs.out) return null;

        const parts = [_]weights.QuantByteStackPart{
            .{ .data = qi.data, .in = qs.in, .out = qs.out },
            .{ .data = ki.data, .in = ks.in, .out = ks.out },
            .{ .data = vi.data, .in = vs.in, .out = vs.out },
        };
        var stack = (try weights.makeQuantByteStack(.q6_k, self.allocator, &parts, .{
            .prefer_device = true,
            .require_device = false,
        })) orelse return null;
        errdefer stack.deinit(self.allocator);
        try self.qkv_cache.ensureUnusedCapacity(self.allocator, 1);
        const qw = try self.allocator.create(weights.QuantByteStack);
        errdefer self.allocator.destroy(qw);
        qw.* = stack;
        self.qkv_cache.putAssumeCapacityNoClobber(cache_key, qw);
        return qw;
    }

    fn getLinearPosBatchQ6(self: *ParakeetWeights, layer_count: usize) !?*const weights.QuantByteStack {
        if (self.pos_cache) |cached| {
            if (cached.count == layer_count) return cached;
            return null;
        }
        if (layer_count == 0) return null;

        var name_buf: [160]u8 = undefined;
        const first = try self.file.get(posName(&name_buf, 0));
        if (first.ggml_type != .q6_k) return null;
        const first_shape = try linearShape(first);
        if (first.data.len == 0 or first.data.len % first_shape.out != 0) return null;
        const bytes_per_weight = first.data.len;

        var parts = try self.allocator.alloc(weights.QuantByteStackPart, layer_count);
        defer self.allocator.free(parts);
        parts[0] = .{ .data = first.data, .in = first_shape.in, .out = first_shape.out };
        for (1..layer_count) |il| {
            const wi = try self.file.get(posName(&name_buf, il));
            if (wi.ggml_type != .q6_k) return null;
            const shape = try linearShape(wi);
            if (shape.in != first_shape.in or shape.out != first_shape.out or wi.data.len != bytes_per_weight) return null;
            parts[il] = .{ .data = wi.data, .in = shape.in, .out = shape.out };
        }

        var stack = (try weights.makeQuantByteStack(.q6_k, self.allocator, parts, .{
            .prefer_device = true,
            .require_device = true,
        })) orelse return null;
        errdefer stack.deinit(self.allocator);

        const batched = try self.allocator.create(weights.QuantByteStack);
        errdefer self.allocator.destroy(batched);
        batched.* = stack;
        self.pos_cache = batched;
        return batched;
    }

    fn shouldUseF32Blas(self: *ParakeetWeights, w_name: []const u8, x: *const SeqIn) !bool {
        if (!self.prefer_f32_blas) return false;
        if (x.dim(.seq) < 16) return false;
        const wi = try self.file.get(w_name);
        return canCacheF32(wi.ggml_type);
    }
};

fn posName(buf: []u8, il: usize) []const u8 {
    return std.fmt.bufPrint(buf, "encoder.layers.{d}.self_attn.linear_pos.weight", .{il}) catch unreachable;
}

fn linearShape(wi: *const gguf.TensorInfo) !LinearShape {
    var in: usize = undefined;
    var out: usize = undefined;
    if (wi.n_dims == 2) {
        in = wi.dims[0];
        out = wi.dims[1];
    } else if (wi.n_dims == 3 and wi.dims[0] == 1) {
        in = wi.dims[1];
        out = wi.dims[2];
    } else {
        return weights.Error.InvalidWeightShape;
    }
    var info2 = wi.*;
    info2.n_dims = 2;
    info2.dims[0] = in;
    info2.dims[1] = out;
    return .{ .in = in, .out = out, .info2 = info2 };
}

fn addQkvBias(raw_out: *RawTensor, d: usize, q_bias: ?[]const f32, k_bias: ?[]const f32, v_bias: ?[]const f32) !void {
    const view = try raw_out.rankView(2);
    if (view.dim(1) != 3 * d) return weights.Error.InvalidWeightShape;
    inline for (.{ q_bias, k_bias, v_bias }, 0..) |maybe_bias, band| {
        if (maybe_bias) |bias| {
            if (bias.len != d) return weights.Error.InvalidWeightShape;
            const data = raw_out.data();
            for (0..view.dim(0)) |row| {
                const dst = data[row * 3 * d + band * d ..][0..d];
                for (dst, bias) |*y, b| y.* += b;
            }
        }
    }
}

fn canCacheF32(ggml_type: gguf.GgmlType) bool {
    return switch (ggml_type) {
        .f16, .bf16, .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q4_k, .q5_k, .q6_k => true,
        else => false,
    };
}

fn quantBlockSlice(comptime Elem: type, bytes: []const u8) ![]const Elem {
    if (bytes.len % @sizeOf(Elem) != 0) return weights.Error.InvalidWeightShape;
    if (@intFromPtr(bytes.ptr) % @alignOf(Elem) != 0) return weights.Error.InvalidWeightShape;
    const aligned: []align(@alignOf(Elem)) const u8 = @alignCast(bytes);
    return std.mem.bytesAsSlice(Elem, aligned);
}

/// Borrow a contiguous f32 GGUF tensor's bytes as `[]const f32`. mmap'd GGUF
/// bytes are untrusted: a non-4-aligned start (e.g. a hostile `info.offset` or
/// `general.alignment`) or a non-multiple-of-4 length would be illegal behaviour
/// at the `@alignCast` under ReleaseFast — guard both and return an error. The
/// single home for the four parakeet f32 borrow helpers (loader/encoder/
/// subsampling/decoder) that used to `@ptrCast(@alignCast(...))` unguarded.
pub fn borrowF32(bytes: []const u8) ![]const f32 {
    if (bytes.len % @sizeOf(f32) != 0) return weights.Error.InvalidWeightShape;
    if (@intFromPtr(bytes.ptr) % @alignOf(f32) != 0) return weights.Error.InvalidWeightShape;
    const aligned: []align(@alignOf(f32)) const u8 = @alignCast(bytes);
    return std.mem.bytesAsSlice(f32, aligned);
}

test {
    _ = @import("weights_tests.zig");
}

// Inline (file-private symbol, per the sibling-tests policy): `firstSeqUse`
// drives the x86 k-quant cold/steady routing split — it must be true exactly
// once per weight name and must own its keys (callers pass stack-formatted
// names; deinit under the testing allocator is the leak check).
test "firstSeqUse: true exactly once per name; keys duped and freed" {
    var ctx: ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();
    var pw: ParakeetWeights = .{
        .ctx = &ctx,
        .file = undefined,
        .allocator = std.testing.allocator,
        .resident = weights.ResidentByteRegistry.init(std.testing.allocator),
    };
    defer pw.deinit();

    var name_buf: [32]u8 = undefined;
    const n1 = try std.fmt.bufPrint(&name_buf, "enc.{d}.ffn1.weight", .{1});
    try std.testing.expect(try pw.firstSeqUse(n1));
    @memset(&name_buf, 0xAA); // clobber the caller buffer: the map must hold its own copy
    const n2 = try std.fmt.bufPrint(&name_buf, "enc.{d}.ffn1.weight", .{1});
    try std.testing.expect(!try pw.firstSeqUse(n2));
    try std.testing.expect(try pw.firstSeqUse("enc.2.ffn1.weight"));
}
