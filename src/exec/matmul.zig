//! Dense contractions: `dot`/`matmul` and the transpose variants, batched
//! (`bmm*`), the packed dense-RHS arms, the f16/bf16 TransB streams, and
//! the eager GPU GEMM dispatch with its accelerator-fence bookkeeping.
//! Domain module: every op receives an explicit `*ExecContext`.
const std = @import("std");
const build_options = @import("build_options");
const accelerator = @import("../accelerator.zig");
const backend_mod = @import("../backend.zig");
const kernels = backend_mod.kernels;
const dtype_mod = @import("../dtype.zig");
const parallel = @import("../parallel.zig");
const storage_mod = @import("../storage.zig");
const tuning = @import("../tuning.zig");
const tensor = @import("../tensor.zig");

const exec_convert = @import("convert.zig");
const ensureForwardFloatMath = dtype_mod.requireForwardFloatMath;
const ExecContext = @import("../exec.zig").ExecContext;

const DType = tensor.DType;
const Tensor = tensor.Tensor;

const ops = backend_mod.ops;

/// Which operand is stored transposed (the one enum from the facade down to
/// the GPU providers; `backend/ops.zig`).
pub const MatmulKind = ops.MatmulKind;

pub const Matmul2DShape = struct {
    m: usize,
    k: usize,
    n: usize,
};

pub fn analyzeMatmul2D(comptime kind: MatmulKind, a: anytype, b: anytype) !Matmul2DShape {
    const av = try a.rankView(2);
    const bv = try b.rankView(2);

    var m: usize = undefined;
    var k_a: usize = undefined;
    var k_b: usize = undefined;
    var n: usize = undefined;
    switch (kind) {
        .plain => {
            m = av.dim(0);
            k_a = av.dim(1);
            k_b = bv.dim(0);
            n = bv.dim(1);
        },
        .trans_a => {
            k_a = av.dim(0);
            m = av.dim(1);
            k_b = bv.dim(0);
            n = bv.dim(1);
        },
        .trans_b => {
            m = av.dim(0);
            k_a = av.dim(1);
            n = bv.dim(0);
            k_b = bv.dim(1);
        },
    }
    if (k_a != k_b) return tensor.TensorError.ShapeMismatch;
    return .{ .m = m, .k = k_a, .n = n };
}

pub const BmmKind = MatmulKind;
const max_bmm_batch_rank = tensor.max_rank - 2;

pub const BmmBatchMode = enum {
    compact,
    broadcast,
};

pub const BmmShape = struct {
    num_batches: usize,
    batch_dims_len: u8,
    batch_dims_buf: [max_bmm_batch_rank]usize,
    m: usize,
    k: usize,
    n: usize,
    batch_mode: BmmBatchMode,
    compact_a_stride: usize, // valid only for .compact: 0 if A is shared, matrix elements otherwise
    compact_b_stride: usize, // valid only for .compact: 0 if B is shared, matrix elements otherwise
    a_broadcast_strides_buf: [max_bmm_batch_rank]usize,
    b_broadcast_strides_buf: [max_bmm_batch_rank]usize,

    pub fn batchDims(self: *const BmmShape) []const usize {
        return self.batch_dims_buf[0..self.batch_dims_len];
    }

    pub fn aBroadcastStrides(self: *const BmmShape) []const usize {
        return self.a_broadcast_strides_buf[0..self.batch_dims_len];
    }

    pub fn bBroadcastStrides(self: *const BmmShape) []const usize {
        return self.b_broadcast_strides_buf[0..self.batch_dims_len];
    }
};

pub fn analyzeBmm(comptime kind: BmmKind, a: *const Tensor, b: *const Tensor) !BmmShape {
    const a_rank = a.shape.len;
    const b_rank = b.shape.len;
    if (a_rank < 2 or b_rank < 2) return tensor.TensorError.InvalidShape;

    const a_inner = a.shape.slice()[a_rank - 2 ..];
    const b_inner = b.shape.slice()[b_rank - 2 ..];

    var m: usize = undefined;
    var k_a: usize = undefined;
    var k_b: usize = undefined;
    var n: usize = undefined;
    switch (kind) {
        .plain => {
            m = a_inner[0];
            k_a = a_inner[1];
            k_b = b_inner[0];
            n = b_inner[1];
        },
        .trans_a => {
            k_a = a_inner[0];
            m = a_inner[1];
            k_b = b_inner[0];
            n = b_inner[1];
        },
        .trans_b => {
            m = a_inner[0];
            k_a = a_inner[1];
            n = b_inner[0];
            k_b = b_inner[1];
        },
    }
    if (k_a != k_b) return tensor.TensorError.ShapeMismatch;

    const a_batch = a.shape.slice()[0 .. a_rank - 2];
    const b_batch = b.shape.slice()[0 .. b_rank - 2];

    // Strict 2-D on both sides is not a batched operation; the user should
    // call matmul/matmulTransA/matmulTransB instead.
    if (a_batch.len == 0 and b_batch.len == 0) return tensor.TensorError.InvalidShape;

    const dims_len_usize = @max(a_batch.len, b_batch.len);
    if (dims_len_usize > max_bmm_batch_rank) return tensor.TensorError.InvalidShape;

    var num: usize = 1;
    var dims_buf: [max_bmm_batch_rank]usize = undefined;
    for (0..dims_len_usize) |i| {
        const a_dim = alignedBatchDim(a_batch, dims_len_usize, i);
        const b_dim = alignedBatchDim(b_batch, dims_len_usize, i);
        const out_dim = if (a_dim == b_dim)
            a_dim
        else if (a_dim == 1)
            b_dim
        else if (b_dim == 1)
            a_dim
        else
            return tensor.TensorError.ShapeMismatch;
        dims_buf[i] = out_dim;
        num = try std.math.mul(usize, num, out_dim);
    }

    const dims_len: u8 = @intCast(dims_len_usize);
    const a_matrix_elems = if (kind == .trans_a) k_a * m else m * k_a;
    const b_matrix_elems = if (kind == .trans_b) n * k_b else k_b * n;

    var a_broadcast_strides_buf = [_]usize{0} ** max_bmm_batch_rank;
    var b_broadcast_strides_buf = [_]usize{0} ** max_bmm_batch_rank;
    writeBroadcastBatchStrides(a_batch, dims_buf[0..dims_len_usize], a_matrix_elems, &a_broadcast_strides_buf);
    writeBroadcastBatchStrides(b_batch, dims_buf[0..dims_len_usize], b_matrix_elems, &b_broadcast_strides_buf);

    const a_range_stride = compactBatchRangeStride(a_batch, dims_buf[0..dims_len_usize], a_matrix_elems);
    const b_range_stride = compactBatchRangeStride(b_batch, dims_buf[0..dims_len_usize], b_matrix_elems);
    const batch_mode: BmmBatchMode = if (a_range_stride == null or b_range_stride == null) .broadcast else .compact;

    return .{
        .num_batches = num,
        .batch_dims_len = dims_len,
        .batch_dims_buf = dims_buf,
        .m = m,
        .k = k_a,
        .n = n,
        .batch_mode = batch_mode,
        .compact_a_stride = a_range_stride orelse 0,
        .compact_b_stride = b_range_stride orelse 0,
        .a_broadcast_strides_buf = a_broadcast_strides_buf,
        .b_broadcast_strides_buf = b_broadcast_strides_buf,
    };
}

fn alignedBatchDim(batch: []const usize, out_len: usize, out_index: usize) usize {
    const prefix = out_len - batch.len;
    if (out_index < prefix) return 1;
    return batch[out_index - prefix];
}

fn writeBroadcastBatchStrides(
    source_batch: []const usize,
    out_batch: []const usize,
    matrix_elems: usize,
    out: *[max_bmm_batch_rank]usize,
) void {
    var source_strides: [max_bmm_batch_rank]usize = undefined;
    var stride = matrix_elems;
    var source_index = source_batch.len;
    while (source_index > 0) {
        source_index -= 1;
        source_strides[source_index] = stride;
        stride *= source_batch[source_index];
    }

    const prefix = out_batch.len - source_batch.len;
    for (out_batch, 0..) |_, out_index| {
        if (out_index < prefix) {
            out[out_index] = 0;
            continue;
        }
        const source_dim = out_index - prefix;
        out[out_index] = if (source_batch[source_dim] == 1) 0 else source_strides[source_dim];
    }
}

fn compactBatchRangeStride(source_batch: []const usize, out_batch: []const usize, matrix_elems: usize) ?usize {
    if (source_batch.len == out_batch.len and std.mem.eql(usize, source_batch, out_batch)) return matrix_elems;

    const prefix = out_batch.len - source_batch.len;
    for (out_batch, 0..) |_, out_index| {
        if (out_index < prefix) continue;
        if (source_batch[out_index - prefix] != 1) return null;
    }
    return 0;
}

pub const BatchTensorView = struct {
    borrowed: Tensor,

    pub fn constPtr(self: *const BatchTensorView) *const Tensor {
        return &self.borrowed;
    }

    pub fn ptr(self: *BatchTensorView) *Tensor {
        return &self.borrowed;
    }
};

pub fn batchTensorView(t: *const Tensor, offset_in_elems: usize) BatchTensorView {
    return .{
        .borrowed = .{
            .buffer = t.buffer,
            .shape = t.shape,
            .strides = t.strides,
            .offset = t.offset + offset_in_elems,
        },
    };
}

/// Full dot product of two same-shape tensors, a scalar tensor result.
pub fn dot(
    self: *ExecContext,
    comptime dtype: DType,
    a: *const tensor.TensorOf(dtype),
    b: *const tensor.TensorOf(dtype),
) !tensor.TensorOf(dtype_mod.outputDType(.matmul, dtype)) {
    comptime ensureForwardFloatMath(dtype);
    const output_dtype = comptime dtype_mod.outputDType(.matmul, dtype);

    var aa = try self.prepareContiguous(dtype, a);
    defer aa.deinit();
    var bb = try self.prepareContiguous(dtype, b);
    defer bb.deinit();

    const ap = aa.tensor();
    const bp = bb.tensor();
    try tensor.requireSameShape(ap, bp);

    var out = try self.scalar(output_dtype, dtype_mod.zero(output_dtype));
    errdefer out.deinit();
    self.enableNativeVectorPoolForWork(ap.len(), parallel.vector_elementwise_len_threshold);
    try kernels.dot(self.pc(), dtype, &out, ap, bp);
    return out;
}

fn matmul2DF32(self: *ExecContext, comptime kind: MatmulKind, a: *const Tensor, b: *const Tensor) !Tensor {
    const info = try analyzeMatmul2D(kind, a, b);

    var aa = try self.prepareContiguous(.f32, a);
    defer aa.deinit();
    var bb = try self.prepareContiguous(.f32, b);
    defer bb.deinit();

    const ap = aa.tensor();
    const bp = bb.tensor();

    var out = try self.empty(.f32, .{ info.m, info.n });
    errdefer out.deinit();
    self.enableNativeMatmulPoolForWork(.f32, info.m, info.n, info.k);
    kernels.gemm(self.pc(), .{ .kind = kind }, &out, ap, bp, info.m, info.n, info.k);
    return out;
}

/// out = base + a·b for 2-D operands. `base` is copied into the result
/// buffer and the accumulate GEMM (BLAS beta=1, or the vector accumulate
/// kernels) adds the product in place — the addmm/residual pattern with no
/// intermediate product tensor and no separate elementwise add pass.
/// `base + a @ b` in one accumulating GEMM (the addmm form, f32).
pub fn matmulAdd(self: *ExecContext, a: *const Tensor, b: *const Tensor, base: *const Tensor) !Tensor {
    const info = try analyzeMatmul2D(.plain, a, b);
    const basev = try base.rankView(2);
    if (basev.dim(0) != info.m or basev.dim(1) != info.n) return tensor.TensorError.ShapeMismatch;

    var aa = try self.prepareContiguous(.f32, a);
    defer aa.deinit();
    var bb = try self.prepareContiguous(.f32, b);
    defer bb.deinit();

    var out = try self.materialize(.f32, base);
    errdefer out.deinit();
    self.enableNativeMatmulPoolForWork(.f32, info.m, info.n, info.k);
    kernels.gemm(self.pc(), .{ .accumulate = true }, &out, aa.tensor(), bb.tensor(), info.m, info.n, info.k);
    return out;
}

/// Strict 2-D `a[m,k] @ b[k,n]`.
/// Rank-2 matmul over `kind` (`.plain` a·b, `.trans_a` aᵀ·b, `.trans_b`
/// a·bᵀ). f32 runs the dense GEMM; a typed `.plain` runs the typed GEMM
/// (f32 accumulation, `outputDType(.matmul, dtype)` store); every other
/// typed case follows the `.widened` policy.
pub fn matmul(
    self: *ExecContext,
    comptime dtype: DType,
    comptime kind: MatmulKind,
    a: *const tensor.TensorOf(dtype),
    b: *const tensor.TensorOf(dtype),
) !tensor.TensorOf(dtype_mod.outputDType(.matmul, dtype)) {
    if (comptime dtype == .f32) return matmul2DF32(self, kind, a, b);
    comptime ensureForwardFloatMath(dtype);
    const output_dtype = comptime dtype_mod.outputDType(.matmul, dtype);
    if (comptime kind != .plain) {
        const compute = comptime ExecContext.widenedCompute(dtype, "matmul");
        var aa = try self.prepareAs(dtype, compute, a);
        defer aa.deinit();
        var bb = try self.prepareAs(dtype, compute, b);
        defer bb.deinit();
        var out = try matmul2DF32(self, kind, aa.tensor(), bb.tensor());
        errdefer out.deinit();
        return self.storeAs(compute, output_dtype, out);
    }

    const info = try analyzeMatmul2D(.plain, a, b);

    var aa = try self.prepareContiguous(dtype, a);
    defer aa.deinit();
    var bb = try self.prepareContiguous(dtype, b);
    defer bb.deinit();

    var out = try self.empty(output_dtype, .{ info.m, info.n });
    errdefer out.deinit();
    self.enableNativeMatmulPoolForWork(dtype, info.m, info.n, info.k);
    kernels.gemm(self.pc(), ops.Gemm.typed(dtype), &out, aa.tensor(), bb.tensor(), info.m, info.n, info.k);
    return out;
}

pub fn packDenseMatmulRhs(self: *ExecContext, comptime dtype: DType, rhs: *const tensor.TensorOf(dtype)) !backend_mod.PackedDenseRhs {
    _ = try rhs.rankView(2);
    var rr = try self.prepareContiguous(dtype, rhs);
    defer rr.deinit();
    return kernels.packDenseRhs(dtype, self.allocator(), rr.tensor());
}

// ---------------------------------------------------------------------------
// CPU f32 weight shadow (FUCINA_CPU_F32_SHADOW=1; CPU builds only).
//
// MEASURED (bench-f16gemm, M1 Max + Accelerate, idle): at m >= ~32 the BLAS
// f32 route over a pre-widened RHS beats the 16-bit streaming kernels
// 1.5-2.5x at the Qwen3 projection shapes (m=64: ~2.5x across the board),
// while decode (m < 32) stays with the streaming kernels — half the bytes
// per weight is what decode speed is. The shadow is a widen-ONCE f32 copy
// attached to the 16-bit weight's storage in its `HostShadow` slot: created
// on the first eligible GEMM, it lives exactly as long as the weight buffer
// and costs +4 bytes/weight resident — which is why it is opt-in.
// Mutation-safe by lifetime only for weights that are not trained in
// place; 16-bit TRAINING should leave the flag off (the streaming kernels
// read the live bytes). GPU builds never take this route: they offload
// these shapes anyway.
//
// FUCINA_CPU_F32_SHADOW_MIN_M overrides the m >= 32 crossover.
// ---------------------------------------------------------------------------

/// The shadow route's crossover for this runtime, or null when the route is
/// off. Per-context `Overrides` win; otherwise the process table decides.
fn cpuShadowMinM(ctx: *const ExecContext) ?u64 {
    if (!tuning.resolve(&ctx.rt.tuning, "cpu_f32_shadow")) return null;
    return tuning.resolve(&ctx.rt.tuning, "cpu_f32_shadow_min_m");
}

const CpuShadow = struct {
    shadow: storage_mod.HostShadow,
    buffer: *storage_mod.Buffer,

    fn destroy(ctx: *anyopaque) void {
        const self: *CpuShadow = @ptrCast(@alignCast(ctx));
        self.buffer.release();
        std.heap.smp_allocator.destroy(self);
    }
};

/// Get-or-create the f32 shadow of a contiguous 16-bit weight buffer.
/// Returns a borrowed pointer valid while the weight buffer lives (the
/// HostShadow holds the owning reference). Loser of a concurrent first-touch
/// race frees its copy and adopts the winner's (the storageWrap dance).
fn cpuShadowBuffer(comptime dtype: DType, b: anytype) ?*storage_mod.Buffer {
    if (b.buffer.hostShadow()) |shadow| {
        const cached: *CpuShadow = @ptrCast(@alignCast(shadow.ctx));
        return cached.buffer;
    }
    const elems = b.buffer.data.len;
    const shadow = storage_mod.Buffer.create(std.heap.smp_allocator, elems) catch return null;
    switch (comptime dtype) {
        .f16 => for (shadow.data, b.buffer.data) |*dst, src| {
            dst.* = @floatCast(src);
        },
        .bf16 => for (shadow.data, b.buffer.data) |*dst, src| {
            dst.* = @bitCast(@as(u32, src) << 16);
        },
        else => comptime unreachable,
    }
    const created = std.heap.smp_allocator.create(CpuShadow) catch {
        shadow.release();
        return null;
    };
    created.* = .{
        .shadow = .{ .ctx = created, .destroy_fn = CpuShadow.destroy },
        .buffer = shadow,
    };
    if (b.buffer.setHostShadow(&created.shadow)) return shadow;
    created.shadow.destroy();
    const winner = b.buffer.hostShadow() orelse return null;
    const cached: *CpuShadow = @ptrCast(@alignCast(winner.ctx));
    return cached.buffer;
}

/// The shadow route shared by both 16-bit arms: f32 A (never cast) x the
/// widened RHS through the ordinary f32 TransB entry, which takes the BLAS
/// arm at these shapes.
fn matmulTransB2DViaShadow(
    self: *ExecContext,
    comptime dtype: DType,
    a_contig: *const Tensor,
    b_contig: anytype,
    m: usize,
    n: usize,
    k: usize,
) ?Tensor {
    // The shadow mirrors the WHOLE buffer at offset 0; weight tensors own
    // their buffer outright. Anything else falls back to streaming.
    if (b_contig.offset != 0) return null;
    const shadow = cpuShadowBuffer(dtype, b_contig) orelse return null;
    shadow.retain();
    var b32 = Tensor.fromOwnedBuffer(shadow, &.{ n, k }) catch {
        shadow.release();
        return null;
    };
    defer b32.deinit();
    var out = self.empty(.f32, .{ m, n }) catch return null;
    self.enableNativeMatmulPoolForWork(.f32, m, n, k);
    kernels.gemm(self.pc(), .{ .kind = .trans_b }, &out, a_contig, &b32, m, n, k);
    return out;
}

/// Mixed-precision `a[m,k] x b[n,k]^T -> f32 [m,n]` over a 16-bit weight
/// (`dtype` is `.f16` or `.bf16`; f32 accumulation). The f16 arm casts the
/// LHS to f16 for the f16-operand streaming kernel; the bf16 arm keeps the
/// LHS f32 (the kernel widens the bf16 RHS in-register), so only
/// contiguity is prepared. Deliberately no default BLAS arm: sgemm would
/// need both operands widened to f32, and a PER-CALL RHS widen alone costs
/// an order of magnitude more than the streaming kernels' whole GEMM at
/// LLM shapes (bench-f16gemm: lm-head 4.6 ms pooled vs ~50 ms of widen); a
/// cached widened copy is unsound when 16-bit weights are trained in
/// place, which is why the shadow arm below is opt-in.
/// The mixed-precision GEMM `a[m,k] · b[n,k]ᵀ` with an f32 `a` and a
/// 16-bit `b` (the weight layout): the kernels widen in register and
/// accumulate in f32; the result is f32.
pub fn matmulHalfRhs(self: *ExecContext, comptime dtype: DType, a: *const Tensor, b: *const tensor.TensorOf(dtype)) !Tensor {
    comptime if (dtype != .f16 and dtype != .bf16) @compileError("matmulHalfRhs: the RHS dtype must be .f16 or .bf16");
    const av = try a.rankView(2);
    const bv = try b.rankView(2);
    const m = av.dim(0);
    const k = av.dim(1);
    const n = bv.dim(0);
    if (k != bv.dim(1)) return tensor.TensorError.ShapeMismatch;

    var aa = try self.prepareContiguous(.f32, a);
    defer aa.deinit();
    var bb = try self.prepareContiguous(dtype, b);
    defer bb.deinit();

    // Opt-in cached-shadow BLAS arm for prefill-shaped GEMMs (see the
    // FUCINA_CPU_F32_SHADOW block above); the per-call-widen objection
    // does not apply to a widen-once copy (the bf16 widen is a pure bit
    // shift, exact).
    if (comptime !build_options.use_gpu) {
        if (cpuShadowMinM(self)) |min_m| {
            if (m >= min_m) {
                if (matmulTransB2DViaShadow(self, dtype, aa.tensor(), bb.tensor(), m, n, k)) |out| return out;
            }
        }
    }

    var out = try self.empty(.f32, .{ m, n });
    errdefer out.deinit();
    self.enableNativeMatmulPoolForWork(dtype, m, n, k);
    switch (comptime dtype) {
        .f16 => {
            var a16 = try exec_convert.cast(self, .f32, .f16, aa.tensor());
            defer a16.deinit();
            kernels.gemm(self.pc(), .{ .kind = .trans_b, .a = .f16, .b = .f16 }, &out, &a16, bb.tensor(), m, n, k);
        },
        .bf16 => kernels.gemm(self.pc(), .{ .kind = .trans_b, .b = .bf16 }, &out, aa.tensor(), bb.tensor(), m, n, k),
        else => unreachable,
    }
    return out;
}

/// Batched matrix multiplication. Supports:
///   - Full batched:    a=[..., M, K] @ b=[..., K, N] -> [..., M, N]
///                      Leading batch dims may match exactly or broadcast.
///   - Broadcast RHS:   a=[..., M, K] @ b=[K, N]      -> [..., M, N]
///                      Single fused 2-D GEMM via reshape, no per-batch loop.
///   - Broadcast LHS:   a=[M, K]      @ b=[..., K, N] -> [..., M, N]
/// General multi-axis broadcast never materializes expanded tensors; the
/// runtime computes per-output-batch source offsets and preserves the exact
/// and shared-operand fast paths. Strict 2-D inputs must use matmul/matmul2D.
/// Batched matmul over the leading (broadcast) axes and `kind`. One f32
/// kernel set; 16-bit inputs follow the `.widened` policy.
pub fn bmm(
    self: *ExecContext,
    comptime dtype: DType,
    comptime kind: BmmKind,
    a: *const tensor.TensorOf(dtype),
    b: *const tensor.TensorOf(dtype),
) !tensor.TensorOf(dtype_mod.outputDType(.matmul, dtype)) {
    if (comptime dtype == .f32) return bmmF32(self, kind, a, b);
    const compute = comptime ExecContext.widenedCompute(dtype, "bmm");
    var aa = try self.prepareAs(dtype, compute, a);
    defer aa.deinit();
    var bb = try self.prepareAs(dtype, compute, b);
    defer bb.deinit();
    var out = try bmmF32(self, kind, aa.tensor(), bb.tensor());
    errdefer out.deinit();
    return self.storeAs(compute, comptime dtype_mod.outputDType(.matmul, dtype), out);
}

fn bmmF32(self: *ExecContext, comptime kind: BmmKind, a: *const Tensor, b: *const Tensor) !Tensor {
    const info = try analyzeBmm(kind, a, b);

    var out_buf: [tensor.max_rank]usize = undefined;
    @memcpy(out_buf[0..info.batch_dims_len], info.batchDims());
    out_buf[info.batch_dims_len] = info.m;
    out_buf[info.batch_dims_len + 1] = info.n;
    const out_shape = out_buf[0 .. info.batch_dims_len + 2];

    if (info.batch_mode == .compact and a.isContiguous() and b.isContiguous() and info.compact_b_stride == 0 and
        (kind == .plain or kind == .trans_b))
    {
        return bmmFastPathSharedB(self, kind, a, b, info, out_shape);
    }

    return bmmLoop(self, kind, a, b, info, out_shape);
}

fn bmmFastPathSharedB(
    self: *ExecContext,
    comptime kind: BmmKind,
    a: *const Tensor,
    b: *const Tensor,
    info: BmmShape,
    out_shape: []const usize,
) !Tensor {
    const fused_m = info.num_batches * info.m;

    var a_2d = try a.reshape(&.{ fused_m, info.k });
    defer a_2d.deinit();

    var out_2d = try self.empty(.f32, .{ fused_m, info.n });
    errdefer out_2d.deinit();

    self.enableNativeMatmulPoolForWork(.f32, fused_m, info.n, info.k);
    kernels.gemm(self.pc(), .{ .kind = kind }, &out_2d, &a_2d, b, fused_m, info.n, info.k);

    const result = try out_2d.reshape(out_shape);
    out_2d.deinit();
    return result;
}

fn bmmLoop(
    self: *ExecContext,
    kind: BmmKind,
    a: *const Tensor,
    b: *const Tensor,
    info: BmmShape,
    out_shape: []const usize,
) !Tensor {
    var aa = try self.prepareContiguous(.f32, a);
    defer aa.deinit();
    var bb = try self.prepareContiguous(.f32, b);
    defer bb.deinit();

    const ap = aa.tensor();
    const bp = bb.tensor();

    var out = try self.empty(.f32, out_shape);
    errdefer out.deinit();

    const stride_c = info.m * info.n;

    const per_batch_work = std.math.mul(usize, info.m, info.n) catch std.math.maxInt(usize);
    const per_batch_flops = std.math.mul(usize, per_batch_work, info.k) catch std.math.maxInt(usize);
    const total_work = std.math.mul(usize, per_batch_flops, info.num_batches) catch std.math.maxInt(usize);
    self.enableNativeVectorPoolForWork(total_work, parallel.vector_matmul_work_threshold);

    if (info.batch_mode == .broadcast) {
        bmmBroadcastDispatchRange(self.pc(), kind, ap, bp, &out, info, stride_c, 0, info.num_batches);
    } else {
        bmmDispatchRange(self.pc(), kind, ap, bp, &out, info, stride_c, 0, info.num_batches);
    }

    return out;
}

fn bmmDispatchRange(
    pc: backend_mod.ParallelConfig,
    kind: BmmKind,
    a: *const Tensor,
    b: *const Tensor,
    out: *Tensor,
    info: BmmShape,
    stride_c: usize,
    start: usize,
    count: usize,
) void {
    if (count == 0) return;

    const a_view = batchTensorView(a, start * info.compact_a_stride);
    const b_view = batchTensorView(b, start * info.compact_b_stride);
    var out_view = batchTensorView(out, start * stride_c);

    switch (kind) {
        inline else => |k| kernels.gemmBatched(
            pc,
            k,
            out_view.ptr(),
            a_view.constPtr(),
            b_view.constPtr(),
            info.m,
            info.n,
            info.k,
            count,
            info.compact_a_stride,
            info.compact_b_stride,
            stride_c,
        ),
    }
}

fn bmmBroadcastDispatchRange(
    pc: backend_mod.ParallelConfig,
    kind: BmmKind,
    a: *const Tensor,
    b: *const Tensor,
    out: *Tensor,
    info: BmmShape,
    stride_c: usize,
    start: usize,
    count: usize,
) void {
    if (count == 0) return;

    for (start..start + count) |batch| {
        const a_view = batchTensorView(a, batchOffsetForLinear(&info, batch, info.aBroadcastStrides()));
        const b_view = batchTensorView(b, batchOffsetForLinear(&info, batch, info.bBroadcastStrides()));
        var out_view = batchTensorView(out, batch * stride_c);

        switch (kind) {
            inline else => |k| kernels.gemm(pc, .{ .kind = k }, out_view.ptr(), a_view.constPtr(), b_view.constPtr(), info.m, info.n, info.k),
        }
    }
}

fn batchOffsetForLinear(info: *const BmmShape, linear: usize, strides: []const usize) usize {
    var remaining = linear;
    var offset: usize = 0;
    var dim = info.batch_dims_len;
    while (dim > 0) {
        dim -= 1;
        const coord = remaining % info.batch_dims_buf[dim];
        remaining /= info.batch_dims_buf[dim];
        offset += coord * strides[dim];
    }
    return offset;
}
