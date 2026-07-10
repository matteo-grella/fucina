const std = @import("std");
const backend_mod = @import("../backend.zig");
const dtype_mod = @import("../dtype.zig");
const parallel = @import("../parallel.zig");
const tensor = @import("../tensor.zig");
const thread = @import("../thread.zig");

const exec_convert = @import("convert.zig");
const Runtime = @import("runtime.zig").Runtime;

const DType = tensor.DType;
const Tensor = tensor.Tensor;

pub const MatmulKind = enum { plain, trans_a, trans_b };

pub const Matmul2DShape = struct {
    m: usize,
    k: usize,
    n: usize,
};

pub fn analyzeMatmul2D(kind: MatmulKind, a: anytype, b: anytype) !Matmul2DShape {
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

pub fn analyzeBmm(kind: BmmKind, a: *const Tensor, b: *const Tensor) !BmmShape {
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

pub fn dot(self: *Runtime, a: *const Tensor, b: *const Tensor) !Tensor {
    var aa = try self.prepareContiguous(a);
    defer aa.deinit();
    var bb = try self.prepareContiguous(b);
    defer bb.deinit();

    const ap = aa.tensor();
    const bp = bb.tensor();
    try tensor.requireSameShape(ap, bp);

    var out = try self.scalar(0);
    errdefer out.deinit();
    self.enableNativeVectorPoolForWork(ap.len(), parallel.vector_elementwise_len_threshold);
    try self.backend.dotInto(&out, ap, bp);
    return out;
}

pub fn dotTyped(
    self: *Runtime,
    comptime dtype: DType,
    a: *const tensor.TensorOf(dtype),
    b: *const tensor.TensorOf(dtype),
) !tensor.TensorOf(dtype_mod.outputDType(.matmul, dtype)) {
    if (comptime dtype == .f32) return dot(self, a, b);
    comptime ensureForwardFloatMath(dtype);
    const compute_dtype = comptime dtype_mod.computeDType(.matmul, dtype);
    const output_dtype = comptime dtype_mod.outputDType(.matmul, dtype);

    var aa = try self.prepareContiguousTyped(dtype, a);
    defer aa.deinit();
    var bb = try self.prepareContiguousTyped(dtype, b);
    defer bb.deinit();

    const ap = aa.tensor();
    const bp = bb.tensor();
    try tensor.requireSameShapeOf(dtype, ap, bp);

    _ = compute_dtype;
    var out = try scalarTyped(self, output_dtype, dtype_mod.zero(output_dtype));
    errdefer out.deinit();
    self.enableNativeVectorPoolForWork(ap.len(), parallel.vector_elementwise_len_threshold);
    try self.backend.dotIntoTyped(dtype, &out, ap, bp);
    return out;
}

pub fn matmul2DDispatch(self: *Runtime, kind: MatmulKind, a: *const Tensor, b: *const Tensor) !Tensor {
    const info = try analyzeMatmul2D(kind, a, b);

    var aa = try self.prepareContiguous(a);
    defer aa.deinit();
    var bb = try self.prepareContiguous(b);
    defer bb.deinit();

    const ap = aa.tensor();
    const bp = bb.tensor();

    var out = try self.emptyRank(2, .{ info.m, info.n });
    errdefer out.deinit();
    self.enableNativeMatmulPoolForWork(info.m, info.n, info.k);
    switch (kind) {
        .plain => self.backend.matmul2DIntoUnchecked(&out, ap, bp, info.m, info.n, info.k),
        .trans_a => self.backend.matmulTransA2DIntoUnchecked(&out, ap, bp, info.m, info.n, info.k),
        .trans_b => self.backend.matmulTransB2DIntoUnchecked(&out, ap, bp, info.m, info.n, info.k),
    }
    return out;
}

pub fn matmul2DTyped(
    self: *Runtime,
    comptime dtype: DType,
    a: *const tensor.TensorOf(dtype),
    b: *const tensor.TensorOf(dtype),
) !tensor.TensorOf(dtype_mod.outputDType(.matmul, dtype)) {
    if (comptime dtype == .f32) return matmul2DDispatch(self, .plain, a, b);
    comptime ensureForwardFloatMath(dtype);
    const output_dtype = comptime dtype_mod.outputDType(.matmul, dtype);

    const info = try analyzeMatmul2D(.plain, a, b);

    var aa = try self.prepareContiguousTyped(dtype, a);
    defer aa.deinit();
    var bb = try self.prepareContiguousTyped(dtype, b);
    defer bb.deinit();

    var out = try self.emptyRankTyped(output_dtype, 2, .{ info.m, info.n });
    errdefer out.deinit();
    self.enableNativeTypedMatmulPoolForWork(info.m, info.n, info.k);
    self.backend.matmul2DIntoUncheckedTyped(dtype, &out, aa.tensor(), bb.tensor(), info.m, info.n, info.k);
    return out;
}

pub fn packMatmulRhsTyped(self: *Runtime, comptime dtype: DType, rhs: *const tensor.TensorOf(dtype)) !backend_mod.PackedMatmulRhsFor(dtype) {
    _ = try rhs.rankView(2);
    var rr = try self.prepareContiguousTyped(dtype, rhs);
    defer rr.deinit();
    return self.backend.packMatmulRhsTyped(dtype, self.allocator, rr.tensor());
}

pub fn matmul2DWithPackedRhsTyped(
    self: *Runtime,
    comptime dtype: DType,
    a: *const tensor.TensorOf(dtype),
    rhs: *const backend_mod.PackedMatmulRhsFor(dtype),
) !tensor.TensorOf(dtype_mod.outputDType(.matmul, dtype)) {
    comptime ensureForwardFloatMath(dtype);
    const output_dtype = comptime dtype_mod.outputDType(.matmul, dtype);

    const av = try a.rankView(2);
    const m = av.dim(0);
    const k = av.dim(1);
    if (k != rhs.k) return tensor.TensorError.ShapeMismatch;

    var aa = try self.prepareContiguousTyped(dtype, a);
    defer aa.deinit();

    var out = try self.emptyRankTyped(output_dtype, 2, .{ m, rhs.n });
    errdefer out.deinit();
    self.enableNativeTypedMatmulPoolForWork(m, rhs.n, k);
    try self.backend.matmul2DIntoUncheckedPackedRhsTyped(dtype, self.allocator, &out, aa.tensor(), rhs, m, rhs.n, k);
    return out;
}

pub fn matmulTransB2DWithF16Rhs(self: *Runtime, a: *const Tensor, b: *const tensor.TensorOf(.f16)) !Tensor {
    const av = try a.rankView(2);
    const bv = try b.rankView(2);
    const m = av.dim(0);
    const k = av.dim(1);
    const n = bv.dim(0);
    if (k != bv.dim(1)) return tensor.TensorError.ShapeMismatch;

    var aa_f32 = try self.prepareContiguous(a);
    defer aa_f32.deinit();
    var aa = try exec_convert.castTyped(self, .f32, .f16, aa_f32.tensor());
    defer aa.deinit();
    var bb = try self.prepareContiguousTyped(.f16, b);
    defer bb.deinit();

    var out = try self.emptyRank(2, .{ m, n });
    errdefer out.deinit();
    // Deliberately no BLAS arm here: sgemm would need both operands
    // widened to f32, and the RHS widen alone costs an order of magnitude
    // more than the streaming f16 kernels' whole GEMM at LLM shapes
    // (bench-f16gemm: lm-head 4.6 ms pooled vs ~50 ms of widen); a cached
    // widened copy is unsound because f16 weights may be trained in place.
    self.enableNativeTypedMatmulPoolForWork(m, n, k);
    self.backend.matmulTransB2DIntoUncheckedF16Operands(&out, &aa, bb.tensor(), m, n, k);
    return out;
}

pub fn matmulTransB2DWithBf16Rhs(self: *Runtime, a: *const Tensor, b: *const tensor.TensorOf(.bf16)) !Tensor {
    const av = try a.rankView(2);
    const bv = try b.rankView(2);
    const m = av.dim(0);
    const k = av.dim(1);
    const n = bv.dim(0);
    if (k != bv.dim(1)) return tensor.TensorError.ShapeMismatch;

    var aa = try self.prepareContiguous(a);
    defer aa.deinit();
    var bb = try self.prepareContiguousTyped(.bf16, b);
    defer bb.deinit();

    var out = try self.emptyRank(2, .{ m, n });
    errdefer out.deinit();
    self.enableNativeTypedMatmulPoolForWork(m, n, k);
    self.backend.matmulTransB2DIntoUncheckedBf16Rhs(&out, aa.tensor(), bb.tensor(), m, n, k);
    return out;
}

pub fn bmmDispatch(self: *Runtime, kind: BmmKind, a: *const Tensor, b: *const Tensor) !Tensor {
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
    self: *Runtime,
    kind: BmmKind,
    a: *const Tensor,
    b: *const Tensor,
    info: BmmShape,
    out_shape: []const usize,
) !Tensor {
    const fused_m = info.num_batches * info.m;

    var a_2d = try a.reshape(&.{ fused_m, info.k });
    defer a_2d.deinit();

    var out_2d = try self.emptyRank(2, .{ fused_m, info.n });
    errdefer out_2d.deinit();

    self.enableNativeMatmulPoolForWork(fused_m, info.n, info.k);
    switch (kind) {
        .plain => self.backend.matmul2DIntoUnchecked(&out_2d, &a_2d, b, fused_m, info.n, info.k),
        .trans_b => self.backend.matmulTransB2DIntoUnchecked(&out_2d, &a_2d, b, fused_m, info.n, info.k),
        .trans_a => unreachable,
    }

    const result = try out_2d.reshape(out_shape);
    out_2d.deinit();
    return result;
}

fn bmmLoop(
    self: *Runtime,
    kind: BmmKind,
    a: *const Tensor,
    b: *const Tensor,
    info: BmmShape,
    out_shape: []const usize,
) !Tensor {
    var aa = try self.prepareContiguous(a);
    defer aa.deinit();
    var bb = try self.prepareContiguous(b);
    defer bb.deinit();

    const ap = aa.tensor();
    const bp = bb.tensor();

    var out = try self.empty(out_shape);
    errdefer out.deinit();

    const stride_c = info.m * info.n;

    const per_batch_work = std.math.mul(usize, info.m, info.n) catch std.math.maxInt(usize);
    const per_batch_flops = std.math.mul(usize, per_batch_work, info.k) catch std.math.maxInt(usize);
    const total_work = std.math.mul(usize, per_batch_flops, info.num_batches) catch std.math.maxInt(usize);
    self.enableNativeVectorPoolForWork(total_work, parallel.vector_matmul_work_threshold);

    if (info.num_batches > 1 and total_work >= parallel.bmm_loop_work_threshold) {
        try bmmLoopParallel(self, kind, ap, bp, &out, info, stride_c);
    } else if (info.batch_mode == .broadcast) {
        bmmBroadcastDispatchRange(&self.backend, kind, ap, bp, &out, info, stride_c, 0, info.num_batches);
    } else {
        bmmDispatchRange(&self.backend, kind, ap, bp, &out, info, stride_c, 0, info.num_batches);
    }

    return out;
}

fn bmmDispatchRange(
    backend: *backend_mod.Backend,
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
        .plain => backend.matmulBatched2DIntoUnchecked(
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
        .trans_a => backend.matmulBatchedTransA2DIntoUnchecked(
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
        .trans_b => backend.matmulBatchedTransB2DIntoUnchecked(
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
    backend: *backend_mod.Backend,
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
            .plain => backend.matmul2DIntoUnchecked(out_view.ptr(), a_view.constPtr(), b_view.constPtr(), info.m, info.n, info.k),
            .trans_a => backend.matmulTransA2DIntoUnchecked(out_view.ptr(), a_view.constPtr(), b_view.constPtr(), info.m, info.n, info.k),
            .trans_b => backend.matmulTransB2DIntoUnchecked(out_view.ptr(), a_view.constPtr(), b_view.constPtr(), info.m, info.n, info.k),
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

fn bmmLoopParallel(
    self: *Runtime,
    kind: BmmKind,
    a: *const Tensor,
    b: *const Tensor,
    out: *Tensor,
    info: BmmShape,
    stride_c: usize,
) !void {
    const pool = self.tryWorkPool() catch {
        if (info.batch_mode == .broadcast) {
            bmmBroadcastDispatchRange(&self.backend, kind, a, b, out, info, stride_c, 0, info.num_batches);
        } else {
            bmmDispatchRange(&self.backend, kind, a, b, out, info, stride_c, 0, info.num_batches);
        }
        return;
    };
    var wait_group: thread.WaitGroup = .{};

    const target_chunks = @min(
        @min(info.num_batches, parallel.cpuThreadCount(parallel.bmm_loop_max_chunks)),
        parallel.bmm_loop_max_chunks,
    );
    const chunk_size = (info.num_batches + target_chunks - 1) / target_chunks;

    var tasks: [parallel.bmm_loop_max_chunks]BmmChunkTask = undefined;
    var dispatched: usize = 0;

    var start: usize = 0;
    while (start < info.num_batches) : (start += chunk_size) {
        const count = @min(chunk_size, info.num_batches - start);
        tasks[dispatched] = .{
            .backend = &self.backend,
            .kind = kind,
            .a = a,
            .b = b,
            .out = out,
            .info = info,
            .stride_c = stride_c,
            .start = start,
            .count = count,
        };
        dispatched += 1;
    }

    for (tasks[1..dispatched]) |*task| {
        _ = pool.spawnWg(&wait_group, runBmmChunkTask, .{task});
    }
    runBmmChunkTask(&tasks[0]);
    pool.waitAndWork(&wait_group);
}

const BmmChunkTask = struct {
    backend: *backend_mod.Backend,
    kind: BmmKind,
    a: *const Tensor,
    b: *const Tensor,
    out: *Tensor,
    info: BmmShape,
    stride_c: usize,
    start: usize,
    count: usize,
};

fn runBmmChunkTask(task: *BmmChunkTask) void {
    if (task.info.batch_mode == .broadcast) {
        bmmBroadcastDispatchRange(
            task.backend,
            task.kind,
            task.a,
            task.b,
            task.out,
            task.info,
            task.stride_c,
            task.start,
            task.count,
        );
    } else {
        bmmDispatchRange(
            task.backend,
            task.kind,
            task.a,
            task.b,
            task.out,
            task.info,
            task.stride_c,
            task.start,
            task.count,
        );
    }
}

fn ensureForwardFloatMath(comptime dtype: DType) void {
    if (!dtype_mod.supportsForwardFloatMath(dtype)) {
        @compileError("forward math is currently supported only for floating dtypes");
    }
}

fn scalarTyped(self: *Runtime, comptime dtype: DType, value: dtype_mod.Scalar(dtype)) !tensor.TensorOf(dtype) {
    var out = try self.emptyRankTyped(dtype, 1, .{1});
    out.data()[0] = value;
    return out;
}
