const std = @import("std");
const backend_mod = @import("../backend.zig");
const kernels = backend_mod.kernels;
const dtype_mod = @import("../dtype.zig");
const parallel = @import("../parallel.zig");
const tensor = @import("../tensor.zig");

const exec_buffer_pool = @import("buffer_pool.zig");
const exec_row_ops = @import("row_ops.zig");
const ExecContext = @import("../exec.zig").ExecContext;

const DType = tensor.DType;
const Tensor = tensor.Tensor;

const FusedActQuantTask = exec_row_ops.FusedActQuantTask;
const SplitSwiGluQuantQ8_0x4Task = exec_row_ops.SplitSwiGluQuantQ8_0x4Task;
const runSplitSwiGluQuantQ8_0x4Task = exec_row_ops.runSplitSwiGluQuantQ8_0x4Task;

fn checkedTensorProduct(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch tensor.TensorError.InvalidDataLength;
}

pub const RhsLifetime = enum {
    /// Ordinary tensor/temporary storage. The backend may still use the GPU,
    /// but it must not cache an address-keyed wrap beyond this dispatch.
    transient,
    /// Caller guarantees the RHS bytes stay mapped at the same address for the
    /// process lifetime, or are registered device-resident storage
    /// (`internal.gpu.allocResidentBytes`) whose owner evicts cached wraps via
    /// `freeResidentBytes` before freeing. A backend may cache address-keyed
    /// wraps.
    stable_process,

    pub fn isCacheable(self: RhsLifetime) bool {
        return self == .stable_process;
    }
};

pub const QuantizedMatmulOptions = struct {
    /// Let Exec try backend-specific accelerators before the CPU quant kernels.
    /// Public autograd callers pass false for trainable inputs so the training
    /// path stays CPU unless a gradient-aware GPU policy is added deliberately.
    allow_gpu: bool = true,
    /// Lifetime guarantee for the quantized RHS bytes. This is about storage
    /// stability, not whether the operand is a model weight.
    rhs_lifetime: RhsLifetime = .transient,
};

pub fn dequantizeTensorTyped(self: *ExecContext, comptime dtype: DType, x: *const tensor.TensorOf(dtype)) !Tensor {
    comptime if (!dtype_mod.isBlockQuantized(dtype)) @compileError("dequantizeTensorTyped requires a block-quantized dtype");
    const view = try x.rankView(2);
    var out = try self.emptyRankTyped(.f32, 2, .{ view.dim(0), view.dim(1) });
    errdefer out.deinit();
    try backend_mod.quantized_matmul.dequantizeTensorInto(dtype, &out, x);
    return out;
}

pub fn getRowsQuantizedTyped(self: *ExecContext, comptime dtype: DType, table: *const tensor.TensorOf(dtype), indices: []const usize) !Tensor {
    comptime if (!dtype_mod.isBlockQuantized(dtype)) @compileError("getRowsQuantizedTyped requires a block-quantized dtype");
    if (indices.len == 0) return tensor.TensorError.InvalidShape;
    const view = try table.rankView(2);
    var out = try self.emptyRankTyped(.f32, 2, .{ indices.len, view.dim(1) });
    errdefer out.deinit();
    try backend_mod.quantized_matmul.getRowsTensorInto(dtype, &out, table, indices);
    return out;
}

/// Kernel-pinned batch fallback (see `ExecContext.pin_rowwise_kernels`): run
/// a batched entry as independent single-row calls of the SAME entry, so
/// every row's numerics are exactly the m == 1 numerics regardless of
/// which kernels the backend would pick for the batch shape — including
/// a GPU backend, whose row calls take its own m == 1 path. `ctx` is a
/// small capture struct with `fn call(ctx, *ExecContext, *const Tensor) !Tensor`
/// invoking the entry on one row. Callers gate on m > 1, so the first
/// row always materializes the output. The `call` methods declare
/// `anyerror` on purpose: entry -> pinnedRowwise -> call -> entry is a
/// cycle, and one explicit error set breaks the inferred-set dependency
/// loop the compiler otherwise rejects.
fn pinnedRowwise(self: *ExecContext, a: *const Tensor, ctx: anytype) !Tensor {
    const av = try a.rankView(2);
    const m = av.dim(0);
    const cols = av.dim(1);
    var aa = try self.prepareContiguousTyped(.f32, a);
    defer aa.deinit();
    const input = aa.tensor().dataConst();
    var row = try self.emptyRankTyped(.f32, 2, .{ 1, cols });
    defer row.deinit();
    var out: ?Tensor = null;
    errdefer if (out) |*o| o.deinit();
    var n: usize = 0;
    for (0..m) |r| {
        @memcpy(row.data(), input[r * cols ..][0..cols]);
        var row_out = try ctx.call(self, &row);
        defer row_out.deinit();
        if (out == null) {
            n = (try row_out.rankView(2)).dim(1);
            out = try self.emptyRankTyped(.f32, 2, .{ m, n });
        }
        @memcpy(out.?.data()[r * n ..][0..n], row_out.dataConst());
    }
    return out.?;
}

/// f32 activations [m, k] x block-quantized RHS weights stored as [n, k]
/// row blocks -> f32 [m, n]. This is the public Tensor-backed path; GGUF
/// loading will populate these block-quantized tensors directly.
pub fn matmul2DWithQuantizedTensorRhs(
    self: *ExecContext,
    comptime rhs_dtype: DType,
    a: *const Tensor,
    rhs: *const tensor.TensorOf(rhs_dtype),
) !Tensor {
    return matmul2DWithQuantizedTensorRhsOptions(self, rhs_dtype, a, rhs, .{});
}

pub fn matmul2DWithQuantizedTensorRhsOptions(
    self: *ExecContext,
    comptime rhs_dtype: DType,
    a: *const Tensor,
    rhs: *const tensor.TensorOf(rhs_dtype),
    options: QuantizedMatmulOptions,
) !Tensor {
    comptime if (!dtype_mod.supportsQuantizedMatmulRhs(rhs_dtype)) @compileError("RHS dtype does not support quantized matmul");

    const av = try a.rankView(2);
    const rv = try rhs.rankView(2);
    const m = av.dim(0);
    const k = av.dim(1);
    const n = rv.dim(0);
    if (k != rv.dim(1)) return tensor.TensorError.ShapeMismatch;
    if (!rhs.isContiguous()) return tensor.TensorError.UnsupportedView;
    if (self.pin_rowwise_kernels and m > 1) return pinnedRowwise(self, a, struct {
        rhs: *const tensor.TensorOf(rhs_dtype),
        options: QuantizedMatmulOptions,
        fn call(c: @This(), ctx: *ExecContext, row: *const Tensor) anyerror!Tensor {
            return matmul2DWithQuantizedTensorRhsOptions(ctx, rhs_dtype, row, c.rhs, c.options);
        }
    }{ .rhs = rhs, .options = options });

    var aa = try self.prepareContiguousTyped(.f32, a);
    defer aa.deinit();

    const blocks = try rhs.dataConstChecked();
    const blocks_per_row = try backend_mod.quantized_matmul.blockCountForDType(rhs_dtype, k);

    if (options.allow_gpu) {
        if (try denseQuantMatmulGpuForBlocks(self, rhs_dtype, std.mem.sliceAsBytes(blocks), options.rhs_lifetime, n, aa.tensor(), m, k)) |gpu_out| {
            return gpu_out;
        }
    }

    var out = try self.emptyRankTyped(.f32, 2, .{ m, n });
    errdefer out.deinit();
    self.enableNativeTypedMatmulPoolForWork(m, n, k);

    switch (rhs_dtype) {
        .q1_0 => try matmul2DWithQuantizedRowsTensorRhs(self, backend_mod.QuantizedMatmulRhsQ1_0, "ggml_q1_0", &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .q2_0 => try matmul2DWithQuantizedRowsTensorRhs(self, backend_mod.QuantizedMatmulRhsQ2_0, "ggml_q2_0", &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .q4_0 => try matmul2DWithQuantizedRowsTensorRhs(self, backend_mod.QuantizedMatmulRhsQ4_0, "ggml_q4_0", &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .q4_1 => try matmul2DWithQuantizedRowsTensorRhs(self, backend_mod.QuantizedMatmulRhsQ4_1, "ggml_q4_1", &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .q5_0 => try matmul2DWithQuantizedRowsTensorRhs(self, backend_mod.QuantizedMatmulRhsQ5_0, "ggml_q5_0", &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .q5_1 => try matmul2DWithQuantizedRowsTensorRhs(self, backend_mod.QuantizedMatmulRhsQ5_1, "ggml_q5_1", &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .q8_0 => try matmul2DWithQuantizedRowsTensorRhs(self, backend_mod.QuantizedMatmulRhsQ8_0, "ggml_q8_0", &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .q2_k => try matmul2DWithQuantizedKTensorRhs(self, backend_mod.QuantizedMatmulRhsQ2_K, "ggml_q2_k", &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .q3_k => try matmul2DWithQuantizedKTensorRhs(self, backend_mod.QuantizedMatmulRhsQ3_K, "ggml_q3_k", &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .q4_k => try matmul2DWithQuantizedKTensorRhs(self, backend_mod.QuantizedMatmulRhsQ4_K, "ggml_q4_k", &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .q5_k => try matmul2DWithQuantizedKTensorRhs(self, backend_mod.QuantizedMatmulRhsQ5_K, "ggml_q5_k", &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .q6_k => try matmul2DWithQuantizedKTensorRhs(self, backend_mod.QuantizedMatmulRhsQ6_K, "ggml_q6_k", &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .iq1_s => try matmul2DWithQuantizedRowsTensorRhs(self, backend_mod.QuantizedMatmulRhsIQ1_S, "ggml_iq1_s", &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .iq1_m => try matmul2DWithQuantizedRowsTensorRhs(self, backend_mod.QuantizedMatmulRhsIQ1_M, "ggml_iq1_m", &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .iq2_xxs => try matmul2DWithQuantizedRowsTensorRhs(self, backend_mod.QuantizedMatmulRhsIQ2_XXS, "ggml_iq2_xxs", &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .iq2_xs => try matmul2DWithQuantizedRowsTensorRhs(self, backend_mod.QuantizedMatmulRhsIQ2_XS, "ggml_iq2_xs", &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .iq2_s => try matmul2DWithQuantizedRowsTensorRhs(self, backend_mod.QuantizedMatmulRhsIQ2_S, "ggml_iq2_s", &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .iq3_xxs => try matmul2DWithQuantizedRowsTensorRhs(self, backend_mod.QuantizedMatmulRhsIQ3_XXS, "ggml_iq3_xxs", &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .iq3_s => try matmul2DWithQuantizedRowsTensorRhs(self, backend_mod.QuantizedMatmulRhsIQ3_S, "ggml_iq3_s", &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .iq4_nl => try matmul2DWithQuantizedRowsTensorRhs(self, backend_mod.QuantizedMatmulRhsIQ4_NL, "ggml_iq4_nl", &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .iq4_xs => try matmul2DWithQuantizedRowsTensorRhs(self, backend_mod.QuantizedMatmulRhsIQ4_XS, "ggml_iq4_xs", &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .tq1_0 => try matmul2DWithQuantizedRowsTensorRhs(self, backend_mod.QuantizedMatmulRhsTQ1_0, "ggml_tq1_0", &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .tq2_0 => try matmul2DWithQuantizedRowsTensorRhs(self, backend_mod.QuantizedMatmulRhsTQ2_0, "ggml_tq2_0", &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .mxfp4 => try matmul2DWithQuantizedRowsTensorRhs(self, backend_mod.QuantizedMatmulRhsMXFP4, "ggml_mxfp4", &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .nvfp4 => try matmul2DWithQuantizedRowsTensorRhs(self, backend_mod.QuantizedMatmulRhsNVFP4, "ggml_nvfp4", &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        else => @compileError("supported quantized matmul RHS dtype is missing a dispatch prong"),
    }

    return out;
}

pub fn matmul2DWithQuantizedBlocksRhs(
    self: *ExecContext,
    comptime rhs_dtype: DType,
    a: *const Tensor,
    blocks: []const dtype_mod.Storage(rhs_dtype),
    n: usize,
    k: usize,
) !Tensor {
    return matmul2DWithQuantizedBlocksRhsOptions(self, rhs_dtype, a, blocks, n, k, .{});
}

pub fn matmul2DWithQuantizedBlocksRhsOptions(
    self: *ExecContext,
    comptime rhs_dtype: DType,
    a: *const Tensor,
    blocks: []const dtype_mod.Storage(rhs_dtype),
    n: usize,
    k: usize,
    options: QuantizedMatmulOptions,
) !Tensor {
    comptime if (!dtype_mod.supportsQuantizedMatmulRhs(rhs_dtype)) @compileError("RHS dtype does not support quantized matmul");

    const av = try a.rankView(2);
    const m = av.dim(0);
    if (av.dim(1) != k) return tensor.TensorError.ShapeMismatch;
    const blocks_per_row = try backend_mod.quantized_matmul.blockCountForDType(rhs_dtype, k);
    if (blocks.len != try checkedTensorProduct(n, blocks_per_row)) return tensor.TensorError.InvalidDataLength;
    if (self.pin_rowwise_kernels and m > 1) return pinnedRowwise(self, a, struct {
        blocks: []const dtype_mod.Storage(rhs_dtype),
        n: usize,
        k: usize,
        options: QuantizedMatmulOptions,
        fn call(c: @This(), ctx: *ExecContext, row: *const Tensor) anyerror!Tensor {
            return matmul2DWithQuantizedBlocksRhsOptions(ctx, rhs_dtype, row, c.blocks, c.n, c.k, c.options);
        }
    }{ .blocks = blocks, .n = n, .k = k, .options = options });

    var aa = try self.prepareContiguousTyped(.f32, a);
    defer aa.deinit();

    if (options.allow_gpu) {
        if (try denseQuantMatmulGpuForBlocks(self, rhs_dtype, std.mem.sliceAsBytes(blocks), options.rhs_lifetime, n, aa.tensor(), m, k)) |out| {
            return out;
        }
    }

    var out = try self.emptyRankTyped(.f32, 2, .{ m, n });
    errdefer out.deinit();
    self.enableNativeTypedMatmulPoolForWork(m, n, k);

    switch (rhs_dtype) {
        .q8_0 => try matmul2DWithQuantizedRowsTensorRhs(self, backend_mod.QuantizedMatmulRhsQ8_0, "ggml_q8_0", &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .q4_k => try matmul2DWithQuantizedKTensorRhs(self, backend_mod.QuantizedMatmulRhsQ4_K, "ggml_q4_k", &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .q5_k => try matmul2DWithQuantizedKTensorRhs(self, backend_mod.QuantizedMatmulRhsQ5_K, "ggml_q5_k", &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .q6_k => try matmul2DWithQuantizedKTensorRhs(self, backend_mod.QuantizedMatmulRhsQ6_K, "ggml_q6_k", &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        else => @compileError("direct quantized-block RHS matmul currently supports q8_0/q4_k/q5_k/q6_k"),
    }

    return out;
}

fn matmul2DWithQuantizedRowsTensorRhs(
    self: *ExecContext,
    comptime Rhs: type,
    comptime union_field: []const u8,
    out: *Tensor,
    a: *const Tensor,
    blocks: anytype,
    m: usize,
    n: usize,
    k: usize,
    blocks_per_row: usize,
) !void {
    // Stack wrapper borrowing caller-owned blocks (allocator = null, never
    // deinit'd here; the matmul path never mutates blocks, so the @constCast
    // over containers whose block slice is mutable is sound). Row containers
    // without the ?Allocator borrow pattern still take the legacy owning
    // shape over @constCast'd blocks.
    const Rows = @FieldType(Rhs, "rows");
    const qrhs = Rhs{
        .rows = if (comptime @typeInfo(@FieldType(Rows, "allocator")) == .optional)
            .{ .allocator = null, .blocks = @constCast(blocks), .rows = n, .cols = k, .blocks_per_row = blocks_per_row }
        else
            .{ .allocator = self.allocator, .blocks = @constCast(blocks), .rows = n, .cols = k, .blocks_per_row = blocks_per_row },
        .k = k,
        .n = n,
    };
    try kernels.matmul2DQuantizedRhs(self.pc(), self.allocator, out, a, @unionInit(backend_mod.AnyQuantizedMatmulRhs, union_field, &qrhs), m, n, k);
}

fn matmul2DWithQuantizedKTensorRhs(
    self: *ExecContext,
    comptime Rhs: type,
    comptime union_field: []const u8,
    out: *Tensor,
    a: *const Tensor,
    blocks: anytype,
    m: usize,
    n: usize,
    k: usize,
    blocks_per_row: usize,
) !void {
    // Stack wrapper borrowing caller-owned blocks (never deinit'd here):
    // Q4_K/Q5_K/Q6_K take them as-is; q2_k/q3_k containers lack the ?Allocator
    // borrow pattern and keep the legacy owning shape.
    const qrhs = if (comptime @typeInfo(@FieldType(Rhs, "allocator")) == .optional)
        Rhs{ .allocator = null, .blocks = blocks, .k = k, .n = n, .blocks_per_column = blocks_per_row }
    else
        Rhs{ .allocator = self.allocator, .blocks = @constCast(blocks), .k = k, .n = n, .blocks_per_column = blocks_per_row };
    try kernels.matmul2DQuantizedRhs(self.pc(), self.allocator, out, a, @unionInit(backend_mod.AnyQuantizedMatmulRhs, union_field, &qrhs), m, n, k);
}

pub fn packMatmulRhsQ8_0x4(self: *ExecContext, rhs: *const tensor.TensorOf(.q8_0)) !backend_mod.QuantizedMatmulRhsQ8_0x4 {
    const view = try rhs.rankView(2);
    if (!rhs.isContiguous()) return tensor.TensorError.UnsupportedView;
    const n = view.dim(0);
    const k = view.dim(1);
    const blocks_per_row = try backend_mod.quantized_matmul.blockCountForDType(.q8_0, k);
    return backend_mod.quantized_matmul.q8_0.packMatmulRhsQ8_0x4(self.allocator, rhs.dataConst(), n, k, blocks_per_row);
}

pub fn packMatmulRhsQ6_Kx4(self: *ExecContext, rhs: *const tensor.TensorOf(.q6_k)) !backend_mod.QuantizedMatmulRhsQ6_Kx4 {
    const view = try rhs.rankView(2);
    if (!rhs.isContiguous()) return tensor.TensorError.UnsupportedView;
    const n = view.dim(0);
    const k = view.dim(1);
    const blocks_per_row = try backend_mod.quantized_matmul.blockCountForDType(.q6_k, k);
    return backend_mod.quantized_matmul.q6_k.packMatmulRhsQ6_Kx4(self.allocator, rhs.dataConst(), n, k, blocks_per_row);
}

pub fn packMatmulRhsQ4_Kx4(self: *ExecContext, rhs: *const tensor.TensorOf(.q4_k)) !backend_mod.QuantizedMatmulRhsQ4_Kx4 {
    const view = try rhs.rankView(2);
    if (!rhs.isContiguous()) return tensor.TensorError.UnsupportedView;
    const n = view.dim(0);
    const k = view.dim(1);
    const blocks_per_row = try backend_mod.quantized_matmul.blockCountForDType(.q4_k, k);
    return backend_mod.quantized_matmul.q4_k.packMatmulRhsQ4_Kx4(self.allocator, rhs.dataConst(), n, k, blocks_per_row);
}

pub fn packMatmulRhsQ4_Kx8(self: *ExecContext, rhs: *const tensor.TensorOf(.q4_k)) !backend_mod.QuantizedMatmulRhsQ4_Kx8 {
    const view = try rhs.rankView(2);
    if (!rhs.isContiguous()) return tensor.TensorError.UnsupportedView;
    const n = view.dim(0);
    const k = view.dim(1);
    const blocks_per_row = try backend_mod.quantized_matmul.blockCountForDType(.q4_k, k);
    return backend_mod.quantized_matmul.q4_k.packMatmulRhsQ4_Kx8(self.allocator, rhs.dataConst(), n, k, blocks_per_row);
}

pub fn packMatmulRhsQ4_Kx2Mmla(self: *ExecContext, rhs: *const tensor.TensorOf(.q4_k)) !backend_mod.QuantizedMatmulRhsQ4_Kx2Mmla {
    const view = try rhs.rankView(2);
    if (!rhs.isContiguous()) return tensor.TensorError.UnsupportedView;
    const n = view.dim(0);
    const k = view.dim(1);
    const blocks_per_row = try backend_mod.quantized_matmul.blockCountForDType(.q4_k, k);
    return backend_mod.quantized_matmul.q4_k.packMatmulRhsQ4_Kx2Mmla(self.allocator, rhs.dataConst(), n, k, blocks_per_row);
}

pub fn packMatmulRhsQ5_Kx8(self: *ExecContext, rhs: *const tensor.TensorOf(.q5_k)) !backend_mod.QuantizedMatmulRhsQ5_Kx8 {
    const view = try rhs.rankView(2);
    if (!rhs.isContiguous()) return tensor.TensorError.UnsupportedView;
    const n = view.dim(0);
    const k = view.dim(1);
    const blocks_per_row = try backend_mod.quantized_matmul.blockCountForDType(.q5_k, k);
    return backend_mod.quantized_matmul.q5_k.packMatmulRhsQ5_Kx8(self.allocator, rhs.dataConst(), n, k, blocks_per_row);
}

pub fn matmul2DWithPackedQ8_0x4Rhs(self: *ExecContext, a: *const Tensor, rhs: *const backend_mod.QuantizedMatmulRhsQ8_0x4) !Tensor {
    const av = try a.rankView(2);
    const m = av.dim(0);
    const k = av.dim(1);
    if (k != rhs.k) return tensor.TensorError.ShapeMismatch;
    if (self.pin_rowwise_kernels and m > 1) return pinnedRowwise(self, a, struct {
        rhs: *const backend_mod.QuantizedMatmulRhsQ8_0x4,
        fn call(c: @This(), ctx: *ExecContext, row: *const Tensor) anyerror!Tensor {
            return matmul2DWithPackedQ8_0x4Rhs(ctx, row, c.rhs);
        }
    }{ .rhs = rhs });

    var aa = try self.prepareContiguousTyped(.f32, a);
    defer aa.deinit();

    var out = try self.emptyRankTyped(.f32, 2, .{ m, rhs.n });
    errdefer out.deinit();
    self.enableNativeTypedMatmulPoolForWork(m, rhs.n, k);
    try kernels.matmul2DQuantizedRhsQ8_0x4(self.pc(), self.allocator, &out, aa.tensor(), rhs, m, rhs.n, k);
    return out;
}

pub fn splitSwiGluMatmul2DWithPackedQ8_0x4Rhs(
    self: *ExecContext,
    gate_up: *const Tensor,
    rhs: *const backend_mod.QuantizedMatmulRhsQ8_0x4,
) !Tensor {
    const gv = try gate_up.rankView(2);
    const m = gv.dim(0);
    const axis_dim = gv.dim(1);
    if (axis_dim % 2 != 0) return tensor.TensorError.InvalidShape;
    const k = axis_dim / 2;
    if (k != rhs.k) return tensor.TensorError.ShapeMismatch;
    if (self.pin_rowwise_kernels and m > 1) return pinnedRowwise(self, gate_up, struct {
        rhs: *const backend_mod.QuantizedMatmulRhsQ8_0x4,
        fn call(c: @This(), ctx: *ExecContext, row: *const Tensor) anyerror!Tensor {
            return splitSwiGluMatmul2DWithPackedQ8_0x4Rhs(ctx, row, c.rhs);
        }
    }{ .rhs = rhs });

    var gg = try self.prepareContiguousTyped(.f32, gate_up);
    defer gg.deinit();

    // Decode (m == 1): the lane-packed kernel pads the single row to four
    // sdot lanes and pays four-row compute for one row of output — measured
    // at roughly half the weight-stream GB/s of the plain-lhs x4 kernel
    // (bench-q8gemv). Materialize the fused SwiGLU row and take the
    // plain-lhs route instead.
    if (m == 1) {
        var fused = try self.emptyRankTyped(.f32, 2, .{ 1, k });
        defer fused.deinit();
        backend_mod.quantized_matmul.q8_0.splitSwiGluRowInto(fused.data(), gg.tensor().dataConst(), k);
        var row_out = try self.emptyRankTyped(.f32, 2, .{ 1, rhs.n });
        errdefer row_out.deinit();
        self.enableNativeTypedMatmulPoolForWork(1, rhs.n, k);
        try kernels.matmul2DQuantizedRhsQ8_0x4(self.pc(), self.allocator, &row_out, &fused, rhs, 1, rhs.n, k);
        return row_out;
    }

    var out = try self.emptyRankTyped(.f32, 2, .{ m, rhs.n });
    errdefer out.deinit();

    const blocks_per_row = try backend_mod.quantized_matmul.q8k.q8_0BlockCount(k);
    const block_count = ((m + 3) / 4) * blocks_per_row;
    var stack_blocks: [512]backend_mod.quantized_matmul.BlockQ8_0x4 = undefined;
    var qlhs_lease: ?exec_buffer_pool.ScratchLease(backend_mod.quantized_matmul.BlockQ8_0x4) = null;
    defer if (qlhs_lease) |*lease| lease.release();
    const qlhs_blocks = if (block_count <= stack_blocks.len)
        stack_blocks[0..block_count]
    else blk: {
        qlhs_lease = try self.buffers.acquireScratch(backend_mod.quantized_matmul.BlockQ8_0x4, block_count);
        break :blk qlhs_lease.?.items;
    };

    const input = gg.tensor().dataConst();
    const row_groups = (m + 3) / 4;
    const base: SplitSwiGluQuantQ8_0x4Task = .{
        .input = input,
        .blocks = qlhs_blocks,
        .rows = m,
        .cols = k,
        .blocks_per_row = blocks_per_row,
        .row_group_start = 0,
        .row_group_end = row_groups,
    };
    const pooled = m * k >= parallel.vector_elementwise_len_threshold / 8 and
        self.dispatchRange(SplitSwiGluQuantQ8_0x4Task, "row_group_start", "row_group_end", base, row_groups, runSplitSwiGluQuantQ8_0x4Task);
    if (!pooled) backend_mod.quantized_matmul.q8_0.quantizeSplitSwiGluRowsQ8_0x4PaddedGroupsInto(
        qlhs_blocks,
        input,
        m,
        k,
        blocks_per_row,
        0,
        row_groups,
    );
    self.enableNativeTypedMatmulPoolForWork(m, rhs.n, k);
    if (m % 4 == 0) {
        try kernels.matmul2DPackedQ8_0x4LhsRhs(self.pc(), &out, qlhs_blocks, rhs, m, rhs.n, k);
    } else {
        try kernels.matmul2DPackedPaddedQ8_0x4LhsRhs(self.pc(), &out, qlhs_blocks, rhs, m, rhs.n, k);
    }
    return out;
}

fn fusedActQuantDispatch(self: *ExecContext, comptime TaskT: type, base: TaskT, row_groups: usize, scratch: []f32) void {
    const cols = base.cols;
    if (base.rows * cols >= parallel.vector_elementwise_len_threshold / 8) {
        if (self.workPool()) |pool| {
            const task_count = @min(parallel.cpuThreadCount(parallel.vector_max_threads), row_groups);
            var tasks: [parallel.vector_max_threads]TaskT = undefined;
            for (0..task_count) |task_i| {
                tasks[task_i] = base;
                tasks[task_i].scratch = scratch[task_i * 4 * cols ..][0 .. 4 * cols];
                tasks[task_i].row_group_start = task_i * row_groups / task_count;
                tasks[task_i].row_group_end = (task_i + 1) * row_groups / task_count;
            }
            pool.parallelChunks(TaskT, tasks[0..task_count], TaskT.run);
            return;
        }
    }
    var serial = base;
    serial.scratch = scratch[0 .. 4 * cols];
    serial.row_group_start = 0;
    serial.row_group_end = row_groups;
    TaskT.run(&serial);
}

const KQuantFusedRhsKind = enum { q4_kx8, q5_kx8, q6_kx4 };

fn splitSwiGluMatmulKQuantImpl(self: *ExecContext, comptime kind: KQuantFusedRhsKind, gate_up: *const Tensor, rhs: anytype) !Tensor {
    const qm = backend_mod.quantized_matmul;
    const gv = try gate_up.rankView(2);
    const m = gv.dim(0);
    const axis_dim = gv.dim(1);
    if (axis_dim % 2 != 0) return tensor.TensorError.InvalidShape;
    const k = axis_dim / 2;
    if (k != rhs.k) return tensor.TensorError.ShapeMismatch;
    const blocks_per_row = try qm.q8k.qkBlockCount(k);
    const n = rhs.n;

    var gg = try self.prepareContiguousTyped(.f32, gate_up);
    defer gg.deinit();
    const input = gg.tensor().dataConst();

    var out = try self.emptyRankTyped(.f32, 2, .{ m, n });
    errdefer out.deinit();
    const out_data = out.data();

    // Pinned mode forces the per-row tail kernels for every row: they are
    // the m == 1 dispatch, so the batch stays bit-identical to sequential
    // decode (see ExecContext.pin_rowwise_kernels).
    const use_x4 = !self.pin_rowwise_kernels and switch (kind) {
        .q4_kx8 => m % 4 == 0 or m >= 64 or (m >= 4 and m < 32),
        .q5_kx8 => m % 4 == 0 or m >= 128,
        .q6_kx4 => false,
    };
    const pad_x4 = kind == .q4_kx8;
    const prefix_rows = if (!use_x4) 0 else if (pad_x4) m else m - m % 4;

    const scratch_storage = try self.buffers.acquire(parallel.vector_max_threads * 4 * k);
    defer scratch_storage.release();
    const scratch = scratch_storage.data[0 .. parallel.vector_max_threads * 4 * k];

    self.enableNativeTypedMatmulPoolForWork(m, n, k);

    if (prefix_rows > 0) {
        const row_groups = if (pad_x4) (prefix_rows + 3) / 4 else prefix_rows / 4;
        var qlhs_x4_lease = try self.buffers.acquireScratch(qm.BlockQ8_Kx4, try checkedTensorProduct(row_groups, blocks_per_row));
        defer qlhs_x4_lease.release();
        const qlhs_x4 = qlhs_x4_lease.items;
        const TaskT = FusedActQuantTask(.split_swiglu, .q8_kx4);
        fusedActQuantDispatch(self, TaskT, .{
            .gate = input,
            .up = &.{},
            .scratch = &.{},
            .rows = prefix_rows,
            .cols = k,
            .blocks_per_row = blocks_per_row,
            .row_group_start = 0,
            .row_group_end = row_groups,
            .x4_blocks = qlhs_x4,
        }, row_groups, scratch);
        switch (kind) {
            .q4_kx8 => kernels.matmulPackedQ4_Kx8Q8_Kx4Slice(self.pc(), out_data[0 .. prefix_rows * n], qlhs_x4, rhs, prefix_rows, n, k),
            .q5_kx8 => kernels.matmulPackedQ5_Kx8Q8_Kx4Slice(self.pc(), out_data[0 .. prefix_rows * n], qlhs_x4, rhs, prefix_rows, n, k),
            .q6_kx4 => unreachable,
        }
    }

    if (prefix_rows < m) {
        const tail_rows = m - prefix_rows;
        const tail_groups = (tail_rows + 3) / 4;
        var qlhs_rows_lease = try self.buffers.acquireScratch(qm.BlockQ8_K, try checkedTensorProduct(tail_rows, blocks_per_row));
        defer qlhs_rows_lease.release();
        const qlhs_rows = qlhs_rows_lease.items;
        const TaskT = FusedActQuantTask(.split_swiglu, .q8_k_rows);
        fusedActQuantDispatch(self, TaskT, .{
            .gate = input[prefix_rows * axis_dim ..],
            .up = &.{},
            .scratch = &.{},
            .rows = tail_rows,
            .cols = k,
            .blocks_per_row = blocks_per_row,
            .row_group_start = 0,
            .row_group_end = tail_groups,
            .row_blocks = qlhs_rows,
        }, tail_groups, scratch);
        const tail_out = out_data[prefix_rows * n ..][0 .. tail_rows * n];
        switch (kind) {
            .q4_kx8 => kernels.matmulPackedQ4_Kx8RowsSlice(self.pc(), tail_out, qlhs_rows, rhs, tail_rows, n, k),
            .q5_kx8 => kernels.matmulPackedQ5_Kx8RowsSlice(self.pc(), tail_out, qlhs_rows, rhs, tail_rows, n, k),
            .q6_kx4 => kernels.matmulPackedQ6_Kx4RowsSlice(self.pc(), tail_out, qlhs_rows, rhs, tail_rows, n, k),
        }
    }
    return out;
}

pub fn splitSwiGluMatmul2DWithPackedQ4_Kx8Rhs(self: *ExecContext, gate_up: *const Tensor, rhs: *const backend_mod.QuantizedMatmulRhsQ4_Kx8) !Tensor {
    return splitSwiGluMatmulKQuantImpl(self, .q4_kx8, gate_up, rhs);
}

pub fn splitSwiGluMatmul2DWithPackedQ5_Kx8Rhs(self: *ExecContext, gate_up: *const Tensor, rhs: *const backend_mod.QuantizedMatmulRhsQ5_Kx8) !Tensor {
    return splitSwiGluMatmulKQuantImpl(self, .q5_kx8, gate_up, rhs);
}

pub fn splitSwiGluMatmul2DWithPackedQ6_Kx4Rhs(self: *ExecContext, gate_up: *const Tensor, rhs: *const backend_mod.QuantizedMatmulRhsQ6_Kx4) !Tensor {
    return splitSwiGluMatmulKQuantImpl(self, .q6_kx4, gate_up, rhs);
}

fn rmsNormMulMatmulKQuantImpl(self: *ExecContext, comptime kind: KQuantFusedRhsKind, x: *const Tensor, norm_weights: *const Tensor, eps: f32, rhs: anytype) !Tensor {
    const qm = backend_mod.quantized_matmul;
    const xv = try x.rankView(2);
    const m = xv.dim(0);
    const k = xv.dim(1);
    if (k != rhs.k) return tensor.TensorError.ShapeMismatch;
    const wv = try norm_weights.rankView(1);
    if (wv.dim(0) != k) return tensor.TensorError.ShapeMismatch;
    const blocks_per_row = try qm.q8k.qkBlockCount(k);
    const n = rhs.n;

    var xx = try self.prepareContiguousTyped(.f32, x);
    defer xx.deinit();
    const input = xx.tensor().dataConst();
    var ww = try self.prepareContiguousTyped(.f32, norm_weights);
    defer ww.deinit();
    const weights = ww.tensor().dataConst();
    const inv_cols: f32 = 1.0 / @as(f32, @floatFromInt(k));

    var out = try self.emptyRankTyped(.f32, 2, .{ m, n });
    errdefer out.deinit();
    const out_data = out.data();

    // Pinned mode forces the per-row tail kernels for every row (see the
    // matching gate in splitSwiGluMatmulKQuantImpl).
    const use_x4 = !self.pin_rowwise_kernels and switch (kind) {
        .q4_kx8 => m % 4 == 0 or m >= 64 or (m >= 4 and m < 32),
        .q5_kx8 => m % 4 == 0 or m >= 128,
        .q6_kx4 => false,
    };
    const pad_x4 = kind == .q4_kx8;
    const prefix_rows = if (!use_x4) 0 else if (pad_x4) m else m - m % 4;

    const scratch_storage = try self.buffers.acquire(parallel.vector_max_threads * 4 * k);
    defer scratch_storage.release();
    const scratch = scratch_storage.data[0 .. parallel.vector_max_threads * 4 * k];

    self.enableNativeTypedMatmulPoolForWork(m, n, k);

    if (prefix_rows > 0) {
        const row_groups = if (pad_x4) (prefix_rows + 3) / 4 else prefix_rows / 4;
        var qlhs_x4_lease = try self.buffers.acquireScratch(qm.BlockQ8_Kx4, try checkedTensorProduct(row_groups, blocks_per_row));
        defer qlhs_x4_lease.release();
        const qlhs_x4 = qlhs_x4_lease.items;
        const TaskT = FusedActQuantTask(.rms_norm_mul, .q8_kx4);
        fusedActQuantDispatch(self, TaskT, .{
            .gate = input,
            .up = weights,
            .scratch = &.{},
            .rows = prefix_rows,
            .cols = k,
            .blocks_per_row = blocks_per_row,
            .eps = eps,
            .inv_cols = inv_cols,
            .rows_kernel = m * k >= parallel.row_kernel_len_threshold,
            .row_group_start = 0,
            .row_group_end = row_groups,
            .x4_blocks = qlhs_x4,
        }, row_groups, scratch);
        switch (kind) {
            .q4_kx8 => kernels.matmulPackedQ4_Kx8Q8_Kx4Slice(self.pc(), out_data[0 .. prefix_rows * n], qlhs_x4, rhs, prefix_rows, n, k),
            .q5_kx8 => kernels.matmulPackedQ5_Kx8Q8_Kx4Slice(self.pc(), out_data[0 .. prefix_rows * n], qlhs_x4, rhs, prefix_rows, n, k),
            .q6_kx4 => unreachable,
        }
    }

    if (prefix_rows < m) {
        const tail_rows = m - prefix_rows;
        const tail_groups = (tail_rows + 3) / 4;
        var qlhs_rows_lease = try self.buffers.acquireScratch(qm.BlockQ8_K, try checkedTensorProduct(tail_rows, blocks_per_row));
        defer qlhs_rows_lease.release();
        const qlhs_rows = qlhs_rows_lease.items;
        const TaskT = FusedActQuantTask(.rms_norm_mul, .q8_k_rows);
        fusedActQuantDispatch(self, TaskT, .{
            .gate = input[prefix_rows * k ..],
            .up = weights,
            .scratch = &.{},
            .rows = tail_rows,
            .cols = k,
            .blocks_per_row = blocks_per_row,
            .eps = eps,
            .inv_cols = inv_cols,
            .rows_kernel = m * k >= parallel.row_kernel_len_threshold,
            .row_group_start = 0,
            .row_group_end = tail_groups,
            .row_blocks = qlhs_rows,
        }, tail_groups, scratch);
        const tail_out = out_data[prefix_rows * n ..][0 .. tail_rows * n];
        switch (kind) {
            .q4_kx8 => kernels.matmulPackedQ4_Kx8RowsSlice(self.pc(), tail_out, qlhs_rows, rhs, tail_rows, n, k),
            .q5_kx8 => kernels.matmulPackedQ5_Kx8RowsSlice(self.pc(), tail_out, qlhs_rows, rhs, tail_rows, n, k),
            .q6_kx4 => kernels.matmulPackedQ6_Kx4RowsSlice(self.pc(), tail_out, qlhs_rows, rhs, tail_rows, n, k),
        }
    }
    return out;
}

pub fn rmsNormMulMatmul2DWithPackedQ4_Kx8Rhs(self: *ExecContext, x: *const Tensor, norm_weights: *const Tensor, eps: f32, rhs: *const backend_mod.QuantizedMatmulRhsQ4_Kx8) !Tensor {
    return rmsNormMulMatmulKQuantImpl(self, .q4_kx8, x, norm_weights, eps, rhs);
}

pub fn rmsNormMulMatmul2DWithPackedQ5_Kx8Rhs(self: *ExecContext, x: *const Tensor, norm_weights: *const Tensor, eps: f32, rhs: *const backend_mod.QuantizedMatmulRhsQ5_Kx8) !Tensor {
    return rmsNormMulMatmulKQuantImpl(self, .q5_kx8, x, norm_weights, eps, rhs);
}

pub fn rmsNormMulMatmul2DWithPackedQ6_Kx4Rhs(self: *ExecContext, x: *const Tensor, norm_weights: *const Tensor, eps: f32, rhs: *const backend_mod.QuantizedMatmulRhsQ6_Kx4) !Tensor {
    return rmsNormMulMatmulKQuantImpl(self, .q6_kx4, x, norm_weights, eps, rhs);
}

/// Fused rmsNormMul + Q8_0x4 LHS quantize + packed GEMM: normalizes the
/// PRE-norm rows into task-private scratch with the exact kernels the
/// unfused dispatch uses, then quantizes — matches the unfused pair to f32
/// roundoff (see rmsNormMulDotPacked), no [m, k] normalized tensor.
pub fn rmsNormMulMatmul2DWithPackedQ8_0x4Rhs(self: *ExecContext, x: *const Tensor, norm_weights: *const Tensor, eps: f32, rhs: *const backend_mod.QuantizedMatmulRhsQ8_0x4) !Tensor {
    const qm = backend_mod.quantized_matmul;
    const xv = try x.rankView(2);
    const m = xv.dim(0);
    const k = xv.dim(1);
    if (k != rhs.k) return tensor.TensorError.ShapeMismatch;
    if (self.pin_rowwise_kernels and m > 1) return pinnedRowwise(self, x, struct {
        norm_weights: *const Tensor,
        eps: f32,
        rhs: *const backend_mod.QuantizedMatmulRhsQ8_0x4,
        fn call(c: @This(), ctx: *ExecContext, row: *const Tensor) anyerror!Tensor {
            return rmsNormMulMatmul2DWithPackedQ8_0x4Rhs(ctx, row, c.norm_weights, c.eps, c.rhs);
        }
    }{ .norm_weights = norm_weights, .eps = eps, .rhs = rhs });
    const wv = try norm_weights.rankView(1);
    if (wv.dim(0) != k) return tensor.TensorError.ShapeMismatch;
    const blocks_per_row = try qm.q8k.q8_0BlockCount(k);
    const n = rhs.n;

    var xx = try self.prepareContiguousTyped(.f32, x);
    defer xx.deinit();
    var ww = try self.prepareContiguousTyped(.f32, norm_weights);
    defer ww.deinit();

    var out = try self.emptyRankTyped(.f32, 2, .{ m, n });
    errdefer out.deinit();

    const row_groups = (m + 3) / 4;
    var qlhs_lease = try self.buffers.acquireScratch(qm.BlockQ8_0x4, try checkedTensorProduct(row_groups, blocks_per_row));
    defer qlhs_lease.release();
    const qlhs = qlhs_lease.items;

    const scratch_storage = try self.buffers.acquire(parallel.vector_max_threads * 4 * k);
    defer scratch_storage.release();
    const scratch = scratch_storage.data[0 .. parallel.vector_max_threads * 4 * k];

    const TaskT = FusedActQuantTask(.rms_norm_mul, .q8_0x4);
    fusedActQuantDispatch(self, TaskT, .{
        .gate = xx.tensor().dataConst(),
        .up = ww.tensor().dataConst(),
        .scratch = &.{},
        .rows = m,
        .cols = k,
        .blocks_per_row = blocks_per_row,
        .eps = eps,
        .inv_cols = 1.0 / @as(f32, @floatFromInt(k)),
        .rows_kernel = m * k >= parallel.row_kernel_len_threshold,
        .row_group_start = 0,
        .row_group_end = row_groups,
        .q8_0x4_blocks = qlhs,
    }, row_groups, scratch);

    self.enableNativeTypedMatmulPoolForWork(m, n, k);
    if (m % 4 == 0) {
        try kernels.matmul2DPackedQ8_0x4LhsRhs(self.pc(), &out, qlhs, rhs, m, n, k);
    } else {
        try kernels.matmul2DPackedPaddedQ8_0x4LhsRhs(self.pc(), &out, qlhs, rhs, m, n, k);
    }
    return out;
}

/// Fused GeGLU (`up * geluQuant(gate)`, ggml f16-LUT semantics) + Q8_0 LHS
/// quantization + packed Q8_0x4 GEMM, for separate gate/up projections.
/// Bit-identical to unary(.gelu_quant) + mul + the packed dot, without
/// materializing the activation tensors.
pub fn gegluQuantMatmul2DWithPackedQ8_0x4Rhs(self: *ExecContext, gate: *const Tensor, up: *const Tensor, rhs: *const backend_mod.QuantizedMatmulRhsQ8_0x4) anyerror!Tensor {
    const qm = backend_mod.quantized_matmul;
    const gv = try gate.rankView(2);
    const uv = try up.rankView(2);
    const m = gv.dim(0);
    const k = gv.dim(1);
    if (uv.dim(0) != m or uv.dim(1) != k) return tensor.TensorError.ShapeMismatch;
    if (k != rhs.k) return tensor.TensorError.ShapeMismatch;
    if (self.pin_rowwise_kernels and m > 1) {
        // Two row-batched inputs: the shared pinnedRowwise helper carries
        // one, so loop both here — same contract, each row runs the
        // m == 1 entry.
        var gg_pin = try self.prepareContiguousTyped(.f32, gate);
        defer gg_pin.deinit();
        var uu_pin = try self.prepareContiguousTyped(.f32, up);
        defer uu_pin.deinit();
        const g_in = gg_pin.tensor().dataConst();
        const u_in = uu_pin.tensor().dataConst();
        var g_row = try self.emptyRankTyped(.f32, 2, .{ 1, k });
        defer g_row.deinit();
        var u_row = try self.emptyRankTyped(.f32, 2, .{ 1, k });
        defer u_row.deinit();
        var out = try self.emptyRankTyped(.f32, 2, .{ m, rhs.n });
        errdefer out.deinit();
        for (0..m) |r| {
            @memcpy(g_row.data(), g_in[r * k ..][0..k]);
            @memcpy(u_row.data(), u_in[r * k ..][0..k]);
            var row_out = try gegluQuantMatmul2DWithPackedQ8_0x4Rhs(self, &g_row, &u_row, rhs);
            defer row_out.deinit();
            @memcpy(out.data()[r * rhs.n ..][0..rhs.n], row_out.dataConst());
        }
        return out;
    }
    const blocks_per_row = try qm.q8k.q8_0BlockCount(k);
    const n = rhs.n;

    var gg = try self.prepareContiguousTyped(.f32, gate);
    defer gg.deinit();
    var uu = try self.prepareContiguousTyped(.f32, up);
    defer uu.deinit();

    var out = try self.emptyRankTyped(.f32, 2, .{ m, n });
    errdefer out.deinit();

    const row_groups = (m + 3) / 4;
    var qlhs_lease = try self.buffers.acquireScratch(qm.BlockQ8_0x4, try checkedTensorProduct(row_groups, blocks_per_row));
    defer qlhs_lease.release();
    const qlhs = qlhs_lease.items;

    const scratch_storage = try self.buffers.acquire(parallel.vector_max_threads * 4 * k);
    defer scratch_storage.release();
    const scratch = scratch_storage.data[0 .. parallel.vector_max_threads * 4 * k];

    const TaskT = FusedActQuantTask(.geglu_quant, .q8_0x4);
    fusedActQuantDispatch(self, TaskT, .{
        .gate = gg.tensor().dataConst(),
        .up = uu.tensor().dataConst(),
        .scratch = &.{},
        .rows = m,
        .cols = k,
        .blocks_per_row = blocks_per_row,
        .row_group_start = 0,
        .row_group_end = row_groups,
        .q8_0x4_blocks = qlhs,
    }, row_groups, scratch);

    self.enableNativeTypedMatmulPoolForWork(m, n, k);
    if (m % 4 == 0) {
        try kernels.matmul2DPackedQ8_0x4LhsRhs(self.pc(), &out, qlhs, rhs, m, n, k);
    } else {
        try kernels.matmul2DPackedPaddedQ8_0x4LhsRhs(self.pc(), &out, qlhs, rhs, m, n, k);
    }
    return out;
}

pub fn matmul2DWithPackedQ6_Kx4Rhs(self: *ExecContext, a: *const Tensor, rhs: *const backend_mod.QuantizedMatmulRhsQ6_Kx4) !Tensor {
    const av = try a.rankView(2);
    const m = av.dim(0);
    const k = av.dim(1);
    if (k != rhs.k) return tensor.TensorError.ShapeMismatch;
    if (self.pin_rowwise_kernels and m > 1) return pinnedRowwise(self, a, struct {
        rhs: *const backend_mod.QuantizedMatmulRhsQ6_Kx4,
        fn call(c: @This(), ctx: *ExecContext, row: *const Tensor) anyerror!Tensor {
            return matmul2DWithPackedQ6_Kx4Rhs(ctx, row, c.rhs);
        }
    }{ .rhs = rhs });

    var aa = try self.prepareContiguousTyped(.f32, a);
    defer aa.deinit();

    var out = try self.emptyRankTyped(.f32, 2, .{ m, rhs.n });
    errdefer out.deinit();
    self.enableNativeTypedMatmulPoolForWork(m, rhs.n, k);
    try kernels.matmul2DQuantizedRhsQ6_Kx4(self.pc(), self.allocator, &out, aa.tensor(), rhs, m, rhs.n, k);
    return out;
}

pub fn matmul2DWithPackedQ4_Kx4Rhs(self: *ExecContext, a: *const Tensor, rhs: *const backend_mod.QuantizedMatmulRhsQ4_Kx4) !Tensor {
    const av = try a.rankView(2);
    const m = av.dim(0);
    const k = av.dim(1);
    if (k != rhs.k) return tensor.TensorError.ShapeMismatch;
    if (self.pin_rowwise_kernels and m > 1) return pinnedRowwise(self, a, struct {
        rhs: *const backend_mod.QuantizedMatmulRhsQ4_Kx4,
        fn call(c: @This(), ctx: *ExecContext, row: *const Tensor) anyerror!Tensor {
            return matmul2DWithPackedQ4_Kx4Rhs(ctx, row, c.rhs);
        }
    }{ .rhs = rhs });

    var aa = try self.prepareContiguousTyped(.f32, a);
    defer aa.deinit();

    var out = try self.emptyRankTyped(.f32, 2, .{ m, rhs.n });
    errdefer out.deinit();
    self.enableNativeTypedMatmulPoolForWork(m, rhs.n, k);
    try kernels.matmul2DQuantizedRhsQ4_Kx4(self.pc(), self.allocator, &out, aa.tensor(), rhs, m, rhs.n, k);
    return out;
}

pub fn matmul2DWithPackedQ4_Kx8Rhs(self: *ExecContext, a: *const Tensor, rhs: *const backend_mod.QuantizedMatmulRhsQ4_Kx8) !Tensor {
    const av = try a.rankView(2);
    const m = av.dim(0);
    const k = av.dim(1);
    if (k != rhs.k) return tensor.TensorError.ShapeMismatch;
    if (self.pin_rowwise_kernels and m > 1) return pinnedRowwise(self, a, struct {
        rhs: *const backend_mod.QuantizedMatmulRhsQ4_Kx8,
        fn call(c: @This(), ctx: *ExecContext, row: *const Tensor) anyerror!Tensor {
            return matmul2DWithPackedQ4_Kx8Rhs(ctx, row, c.rhs);
        }
    }{ .rhs = rhs });

    var aa = try self.prepareContiguousTyped(.f32, a);
    defer aa.deinit();

    var out = try self.emptyRankTyped(.f32, 2, .{ m, rhs.n });
    errdefer out.deinit();
    self.enableNativeTypedMatmulPoolForWork(m, rhs.n, k);
    try kernels.matmul2DQuantizedRhsQ4_Kx8(self.pc(), self.allocator, &out, aa.tensor(), rhs, m, rhs.n, k);
    return out;
}

pub fn matmul2DWithPackedQ4_Kx2MmlaRhs(self: *ExecContext, a: *const Tensor, rhs: *const backend_mod.QuantizedMatmulRhsQ4_Kx2Mmla) !Tensor {
    const av = try a.rankView(2);
    const m = av.dim(0);
    const k = av.dim(1);
    if (k != rhs.k) return tensor.TensorError.ShapeMismatch;
    if (self.pin_rowwise_kernels and m > 1) return pinnedRowwise(self, a, struct {
        rhs: *const backend_mod.QuantizedMatmulRhsQ4_Kx2Mmla,
        fn call(c: @This(), ctx: *ExecContext, row: *const Tensor) anyerror!Tensor {
            return matmul2DWithPackedQ4_Kx2MmlaRhs(ctx, row, c.rhs);
        }
    }{ .rhs = rhs });

    var aa = try self.prepareContiguousTyped(.f32, a);
    defer aa.deinit();

    var out = try self.emptyRankTyped(.f32, 2, .{ m, rhs.n });
    errdefer out.deinit();
    self.enableNativeTypedMatmulPoolForWork(m, rhs.n, k);
    try kernels.matmul2DQuantizedRhsQ4_Kx2Mmla(self.pc(), self.allocator, &out, aa.tensor(), rhs, m, rhs.n, k);
    return out;
}

pub fn matmul2DWithPackedQ5_Kx8Rhs(self: *ExecContext, a: *const Tensor, rhs: *const backend_mod.QuantizedMatmulRhsQ5_Kx8) !Tensor {
    const av = try a.rankView(2);
    const m = av.dim(0);
    const k = av.dim(1);
    if (k != rhs.k) return tensor.TensorError.ShapeMismatch;
    if (self.pin_rowwise_kernels and m > 1) return pinnedRowwise(self, a, struct {
        rhs: *const backend_mod.QuantizedMatmulRhsQ5_Kx8,
        fn call(c: @This(), ctx: *ExecContext, row: *const Tensor) anyerror!Tensor {
            return matmul2DWithPackedQ5_Kx8Rhs(ctx, row, c.rhs);
        }
    }{ .rhs = rhs });

    var aa = try self.prepareContiguousTyped(.f32, a);
    defer aa.deinit();

    var out = try self.emptyRankTyped(.f32, 2, .{ m, rhs.n });
    errdefer out.deinit();
    self.enableNativeTypedMatmulPoolForWork(m, rhs.n, k);
    try kernels.matmul2DQuantizedRhsQ5_Kx8(self.pc(), self.allocator, &out, aa.tensor(), rhs, m, rhs.n, k);
    return out;
}

/// GPU arm of a DENSE quantized linear: `out[m,n] = in[m,k] · dequant(W)ᵀ`
/// via the vendored ggml dequant-in-kernel Metal GEMM (`gemmQuantNt`), with
/// the RHS raw quantized blocks (`rhs_bytes`, row stride `nb01`). Returns the
/// f32 result, or `null` whenever the GPU did not run — below the work gate,
/// shape unsupported (k % blocksize, n % 4, m in [32, 2048]), non-contiguous
/// input, GPU disabled, or dispatch failure — so the caller falls through to
/// the CPU packed path, never-a-loss. dtype must be q4_k/q6_k/q8_0 (the
/// formats the Metal kernel dequantizes). The whole body is comptime-elided
/// on non-gpu builds.
pub fn denseQuantMatmulGpu(
    self: *ExecContext,
    comptime dtype: DType,
    rhs_bytes: []const u8,
    rhs_lifetime: RhsLifetime,
    nb01: usize,
    input: *const Tensor,
    m: usize,
    n: usize,
    k: usize,
) !?Tensor {
    return denseQuantMatmulGpuImpl(self, dtype, rhs_bytes, rhs_lifetime, nb01, input, m, n, k, true);
}

fn denseQuantMatmulGpuImpl(
    self: *ExecContext,
    comptime dtype: DType,
    rhs_bytes: []const u8,
    rhs_lifetime: RhsLifetime,
    nb01: usize,
    input: *const Tensor,
    m: usize,
    n: usize,
    k: usize,
    cpu_fallback_packed: bool,
) !?Tensor {
    if (comptime backend_mod.gpu_impl.enabled) {
        const gpu = backend_mod.gpu_impl;
        if (comptime dtype == .q5_k and !gpu.has_q5_k_quant) return null;
        if (comptime dtype == .tq2_0 and !gpu.has_tq2_0_quant) return null;
        const fmt: gpu.QFormat = comptime switch (dtype) {
            .q4_k => .q4_k,
            .q5_k => .q5_k,
            .q6_k => .q6_k,
            .q8_0 => .q8_0,
            .tq2_0 => .tq2_0,
            else => @compileError("denseQuantMatmulGpu supports q4_k/q5_k/q6_k/q8_0/tq2_0 only"),
        };
        const work = quantMatmulWork(m, n, k);
        // Prefill arm: m >= 32 behind the relevant CPU-competitor gate. Stable
        // weights first try the direct-storage async entry (Metal admits up to
        // 8192 rows per command-data tile table; CUDA grows its slot table).
        // Transient/longer fallbacks retain balanced <=2048-row blocking
        // chunks, so prompts never silently lose the whole quant offload.
        // Decode arm (m <= 8, provider opt-in): bytes-bound GEMV — the work
        // gate never passes at decode shapes, so the provider's own decode
        // gate decides instead.
        const prefill_arm = m >= 32 and if (cpu_fallback_packed)
            gpu.shouldUseGpuDenseQuantPacked(fmt, work)
        else
            gpu.shouldUseGpuDenseQuant(fmt, work);
        const decode_arm = m <= 8 and gpu.shouldUseGpuQuantDecode(fmt, m, n, k);
        if ((prefill_arm or decode_arm) and k % fmt.kMultiple() == 0 and n % 4 == 0 and
            input.isContiguous())
        {
            var out = try self.emptyRank(2, .{ m, n });
            errdefer out.deinit();
            // Stable GGUF/model weights admit true eager-async dispatch:
            // providers bind tensor storage directly and attach completion to
            // `out`. Transient byte slices retain the blocking path because an
            // asynchronous command cannot outlive an unowned RHS borrow.
            if (rhs_lifetime.isCacheable() and gpu.gemmQuantNtAsync(
                fmt,
                rhs_bytes,
                true,
                nb01,
                0,
                input,
                &out,
                1,
                m,
                n,
                k,
            )) return out;

            const in_data = input.dataConst();
            const in_elems = std.math.mul(usize, m, k) catch return null;
            if (in_data.len == in_elems) {
                const max_rows_per_dispatch = 2048;
                const n_chunks = (m + max_rows_per_dispatch - 1) / max_rows_per_dispatch;
                const rows_per = (m + n_chunks - 1) / n_chunks;
                var ok = true;
                var row0: usize = 0;
                while (row0 < m) : (row0 += rows_per) {
                    const rows = @min(rows_per, m - row0);
                    if (!gpu.gemmQuantNt(
                        fmt,
                        rhs_bytes,
                        rhs_lifetime.isCacheable(),
                        nb01,
                        in_data[row0 * k .. (row0 + rows) * k],
                        out.data()[row0 * n .. (row0 + rows) * n],
                        rows,
                        n,
                        k,
                    )) {
                        ok = false;
                        break;
                    }
                }
                if (ok) return out;
            }
            out.deinit();
        }
    }
    return null;
}

/// GPU dispatch for the folded tie-fitted PTQTP pack (BlockTQ2_0Folded row
/// bytes, docs/PTQTP.md): the provider's dedicated folded format, one
/// command per linear instead of one per plane. Mirrors the dense-quant
/// prefill arm (stable-RHS async first, balanced blocking chunks as the
/// fallback); null = CPU path. Pruned on providers without the kernel.
pub fn foldedTernaryMatmulGpu(
    self: *ExecContext,
    rhs_bytes: []const u8,
    rhs_lifetime: RhsLifetime,
    nb01: usize,
    input: *const Tensor,
    m: usize,
    n: usize,
    k: usize,
) !?Tensor {
    if (comptime backend_mod.gpu_impl.enabled) {
        const gpu = backend_mod.gpu_impl;
        if (comptime !gpu.has_tq2_0_folded_quant) return null;
        const fmt: gpu.QFormat = .tq2_0_folded;
        const work = quantMatmulWork(m, n, k);
        const prefill_arm = m >= 32 and gpu.shouldUseGpuDenseQuantPacked(fmt, work);
        if (prefill_arm and k % fmt.kMultiple() == 0 and n % 4 == 0 and input.isContiguous()) {
            var out = try self.emptyRank(2, .{ m, n });
            errdefer out.deinit();
            if (rhs_lifetime.isCacheable() and gpu.gemmQuantNtAsync(
                fmt,
                rhs_bytes,
                true,
                nb01,
                0,
                input,
                &out,
                1,
                m,
                n,
                k,
            )) return out;

            const in_data = input.dataConst();
            const in_elems = std.math.mul(usize, m, k) catch return null;
            if (in_data.len == in_elems) {
                const max_rows_per_dispatch = 2048;
                const n_chunks = (m + max_rows_per_dispatch - 1) / max_rows_per_dispatch;
                const rows_per = (m + n_chunks - 1) / n_chunks;
                var ok = true;
                var row0: usize = 0;
                while (row0 < m) : (row0 += rows_per) {
                    const rows = @min(rows_per, m - row0);
                    if (!gpu.gemmQuantNt(
                        fmt,
                        rhs_bytes,
                        rhs_lifetime.isCacheable(),
                        nb01,
                        in_data[row0 * k .. (row0 + rows) * k],
                        out.data()[row0 * n .. (row0 + rows) * n],
                        rows,
                        n,
                        k,
                    )) {
                        ok = false;
                        break;
                    }
                }
                if (ok) return out;
            }
            out.deinit();
        }
    }
    return null;
}

pub fn denseQuantMatmulGpuSharedInputBatch(
    self: *ExecContext,
    comptime dtype: DType,
    rhs_bytes: []const u8,
    rhs_lifetime: RhsLifetime,
    nb01: usize,
    nb02: usize,
    input: *const Tensor,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) !?Tensor {
    if (comptime backend_mod.gpu_impl.enabled) {
        const gpu = backend_mod.gpu_impl;
        if (comptime dtype == .q5_k and !gpu.has_q5_k_quant) return null;
        if (comptime dtype == .tq2_0 and !gpu.has_tq2_0_quant) return null;
        const fmt: gpu.QFormat = comptime switch (dtype) {
            .q4_k => .q4_k,
            .q5_k => .q5_k,
            .q6_k => .q6_k,
            .q8_0 => .q8_0,
            .tq2_0 => .tq2_0,
            else => @compileError("denseQuantMatmulGpuSharedInputBatch supports q4_k/q5_k/q6_k/q8_0/tq2_0 only"),
        };
        if (batch_count == 0) return null;
        const per_work = quantMatmulWork(m, n, k);
        const work = std.math.mul(u64, per_work, @as(u64, @intCast(batch_count))) catch std.math.maxInt(u64);
        const rows_total = std.math.mul(usize, batch_count, m) catch return null;
        if (m >= 32 and k % fmt.kMultiple() == 0 and n % 4 == 0 and
            input.isContiguous() and
            gpu.shouldUseGpuDenseQuant(fmt, work))
        {
            var out = try self.emptyRank(2, .{ rows_total, n });
            errdefer out.deinit();
            if (rhs_lifetime.isCacheable() and gpu.gemmQuantNtAsync(
                fmt,
                rhs_bytes,
                true,
                nb01,
                nb02,
                input,
                &out,
                batch_count,
                m,
                n,
                k,
            )) return out;

            const in_data = input.dataConst();
            const in_elems = std.math.mul(usize, m, k) catch return null;
            if (m <= 2048 and in_data.len == in_elems) {
                if (gpu.gemmQuantNtSharedABatch(fmt, rhs_bytes, rhs_lifetime.isCacheable(), nb01, nb02, in_data, out.data(), batch_count, m, n, k)) {
                    return out;
                }
            }
            out.deinit();
        }
    }
    return null;
}

fn denseQuantMatmulGpuForBlocks(
    self: *ExecContext,
    comptime dtype: DType,
    rhs_bytes: []const u8,
    rhs_lifetime: RhsLifetime,
    n: usize,
    input: *const Tensor,
    m: usize,
    k: usize,
) !?Tensor {
    if (n == 0) return null;
    if (comptime dtype == .q5_k and !backend_mod.gpu_impl.has_q5_k_quant) return null;
    switch (dtype) {
        .q4_k, .q5_k, .q6_k, .q8_0 => {},
        else => return null,
    }
    const nb01 = std.math.divExact(usize, rhs_bytes.len, n) catch return null;
    return denseQuantMatmulGpuImpl(self, dtype, rhs_bytes, rhs_lifetime, nb01, input, m, n, k, false);
}

fn quantMatmulWork(m: usize, n: usize, k: usize) u64 {
    const mm: u64 = @intCast(m);
    const nn: u64 = @intCast(n);
    const kk: u64 = @intCast(k);
    const mn = std.math.mul(u64, mm, nn) catch return std.math.maxInt(u64);
    return std.math.mul(u64, mn, kk) catch std.math.maxInt(u64);
}
