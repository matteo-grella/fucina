//! Quantized matmul dispatch: dequantize/getRows, the tensor-RHS and
//! packed-RHS matmul entries, RHS pack preparation, the fused
//! activation+quantize+GEMM arms (`splitSwiGlu`/`rmsNormMul`/`geglu` over
//! the Q8_0x4 and K-quant packs), and the GPU dense-quant entries. Domain
//! module: every op receives an explicit `*ExecContext`.
const std = @import("std");
const backend_mod = @import("../backend.zig");
const offload = backend_mod.offload;
const kernels = backend_mod.kernels;
const dtype_mod = @import("../dtype.zig");
const parallel = @import("../parallel.zig");
const tensor = @import("../tensor.zig");

const exec_buffer_pool = @import("buffer_pool.zig");
const exec_row_ops = @import("row_ops.zig");
const ExecContext = @import("../exec.zig").ExecContext;

const DType = tensor.DType;

/// Storage dtype of a packed-RHS container (value or pointer), for the
/// matmul pool gate.
fn packedRhsDType(comptime Rhs: type) DType {
    return switch (@typeInfo(Rhs)) {
        .pointer => |info| info.child.dtype,
        else => Rhs.dtype,
    };
}

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

pub fn dequantizeTensor(self: *ExecContext, comptime dtype: DType, x: *const tensor.TensorOf(dtype)) !Tensor {
    comptime if (!dtype_mod.isBlockQuantized(dtype)) @compileError("dequantizeTensor requires a block-quantized dtype");
    const view = try x.rankView(2);
    var out = try self.empty(.f32, .{ view.dim(0), view.dim(1) });
    errdefer out.deinit();
    try backend_mod.quantized_matmul.dequantizeTensorInto(dtype, &out, x);
    return out;
}

pub fn getRowsQuantized(self: *ExecContext, comptime dtype: DType, table: *const tensor.TensorOf(dtype), indices: []const usize) !Tensor {
    comptime if (!dtype_mod.isBlockQuantized(dtype)) @compileError("getRowsQuantized requires a block-quantized dtype");
    if (indices.len == 0) return tensor.TensorError.InvalidShape;
    const view = try table.rankView(2);
    var out = try self.empty(.f32, .{ indices.len, view.dim(1) });
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
    var aa = try self.prepareContiguous(.f32, a);
    defer aa.deinit();
    const input = aa.tensor().dataConst();
    var row = try self.empty(.f32, .{ 1, cols });
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
            out = try self.empty(.f32, .{ m, n });
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
            return matmul2DWithQuantizedTensorRhs(ctx, rhs_dtype, row, c.rhs, c.options);
        }
    }{ .rhs = rhs, .options = options });

    var aa = try self.prepareContiguous(.f32, a);
    defer aa.deinit();

    const blocks = try rhs.dataConstChecked();
    const blocks_per_row = try backend_mod.quantized_matmul.blockCountForDType(rhs_dtype, k);

    if (options.allow_gpu) {
        if (try tryQuantGemmForBlocks(self, rhs_dtype, std.mem.sliceAsBytes(blocks), options.rhs_lifetime, n, aa.tensor(), m, k)) |gpu_out| {
            return gpu_out;
        }
    }

    var out = try self.empty(.f32, .{ m, n });
    errdefer out.deinit();
    self.enableNativeMatmulPoolForWork(rhs_dtype, m, n, k);

    switch (rhs_dtype) {
        .q1_0 => try matmul2DWithQuantizedRowsTensorRhs(self, backend_mod.QuantizedMatmulRhsQ1_0, &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .q2_0 => try matmul2DWithQuantizedRowsTensorRhs(self, backend_mod.QuantizedMatmulRhsQ2_0, &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .q4_0 => try matmul2DWithQuantizedRowsTensorRhs(self, backend_mod.QuantizedMatmulRhsQ4_0, &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .q4_1 => try matmul2DWithQuantizedRowsTensorRhs(self, backend_mod.QuantizedMatmulRhsQ4_1, &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .q5_0 => try matmul2DWithQuantizedRowsTensorRhs(self, backend_mod.QuantizedMatmulRhsQ5_0, &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .q5_1 => try matmul2DWithQuantizedRowsTensorRhs(self, backend_mod.QuantizedMatmulRhsQ5_1, &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .q8_0 => try matmul2DWithQuantizedRowsTensorRhs(self, backend_mod.QuantizedMatmulRhsQ8_0, &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .q2_k => try matmul2DWithQuantizedKTensorRhs(self, backend_mod.QuantizedMatmulRhsQ2_K, &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .q3_k => try matmul2DWithQuantizedKTensorRhs(self, backend_mod.QuantizedMatmulRhsQ3_K, &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .q4_k => try matmul2DWithQuantizedKTensorRhs(self, backend_mod.QuantizedMatmulRhsQ4_K, &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .q5_k => try matmul2DWithQuantizedKTensorRhs(self, backend_mod.QuantizedMatmulRhsQ5_K, &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .q6_k => try matmul2DWithQuantizedKTensorRhs(self, backend_mod.QuantizedMatmulRhsQ6_K, &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .iq1_s => try matmul2DWithQuantizedRowsTensorRhs(self, backend_mod.QuantizedMatmulRhsIQ1_S, &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .iq1_m => try matmul2DWithQuantizedRowsTensorRhs(self, backend_mod.QuantizedMatmulRhsIQ1_M, &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .iq2_xxs => try matmul2DWithQuantizedRowsTensorRhs(self, backend_mod.QuantizedMatmulRhsIQ2_XXS, &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .iq2_xs => try matmul2DWithQuantizedRowsTensorRhs(self, backend_mod.QuantizedMatmulRhsIQ2_XS, &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .iq2_s => try matmul2DWithQuantizedRowsTensorRhs(self, backend_mod.QuantizedMatmulRhsIQ2_S, &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .iq3_xxs => try matmul2DWithQuantizedRowsTensorRhs(self, backend_mod.QuantizedMatmulRhsIQ3_XXS, &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .iq3_s => try matmul2DWithQuantizedRowsTensorRhs(self, backend_mod.QuantizedMatmulRhsIQ3_S, &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .iq4_nl => try matmul2DWithQuantizedRowsTensorRhs(self, backend_mod.QuantizedMatmulRhsIQ4_NL, &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .iq4_xs => try matmul2DWithQuantizedRowsTensorRhs(self, backend_mod.QuantizedMatmulRhsIQ4_XS, &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .tq1_0 => try matmul2DWithQuantizedRowsTensorRhs(self, backend_mod.QuantizedMatmulRhsTQ1_0, &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .tq2_0 => try matmul2DWithQuantizedRowsTensorRhs(self, backend_mod.QuantizedMatmulRhsTQ2_0, &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .mxfp4 => try matmul2DWithQuantizedRowsTensorRhs(self, backend_mod.QuantizedMatmulRhsMXFP4, &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .nvfp4 => try matmul2DWithQuantizedRowsTensorRhs(self, backend_mod.QuantizedMatmulRhsNVFP4, &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
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
            return matmul2DWithQuantizedBlocksRhs(ctx, rhs_dtype, row, c.blocks, c.n, c.k, c.options);
        }
    }{ .blocks = blocks, .n = n, .k = k, .options = options });

    var aa = try self.prepareContiguous(.f32, a);
    defer aa.deinit();

    if (options.allow_gpu) {
        if (try tryQuantGemmForBlocks(self, rhs_dtype, std.mem.sliceAsBytes(blocks), options.rhs_lifetime, n, aa.tensor(), m, k)) |out| {
            return out;
        }
    }

    var out = try self.empty(.f32, .{ m, n });
    errdefer out.deinit();
    self.enableNativeMatmulPoolForWork(rhs_dtype, m, n, k);

    switch (rhs_dtype) {
        .q8_0 => try matmul2DWithQuantizedRowsTensorRhs(self, backend_mod.QuantizedMatmulRhsQ8_0, &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .q4_k => try matmul2DWithQuantizedKTensorRhs(self, backend_mod.QuantizedMatmulRhsQ4_K, &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .q5_k => try matmul2DWithQuantizedKTensorRhs(self, backend_mod.QuantizedMatmulRhsQ5_K, &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        .q6_k => try matmul2DWithQuantizedKTensorRhs(self, backend_mod.QuantizedMatmulRhsQ6_K, &out, aa.tensor(), blocks, m, n, k, blocks_per_row),
        else => @compileError("direct quantized-block RHS matmul currently supports q8_0/q4_k/q5_k/q6_k"),
    }

    return out;
}

fn matmul2DWithQuantizedRowsTensorRhs(
    self: *ExecContext,
    comptime Rhs: type,
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
    try kernels.matmul2DQuantizedRhs(self.pc(), self.allocator, out, a, @unionInit(backend_mod.AnyQuantizedMatmulRhs, @tagName(Rhs.dtype), &qrhs), m, n, k);
}

fn matmul2DWithQuantizedKTensorRhs(
    self: *ExecContext,
    comptime Rhs: type,
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
    try kernels.matmul2DQuantizedRhs(self.pc(), self.allocator, out, a, @unionInit(backend_mod.AnyQuantizedMatmulRhs, @tagName(Rhs.dtype), &qrhs), m, n, k);
}

/// Pack a block-quantized [n, k] weight tensor into the ISA-best packed
/// matmul RHS container for its dtype (`backend.PackedRhsFor(dt)`); the
/// Q4_K choice between x8 and x2Mmla stays comptime. The per-format
/// packers live in the backend's `quant/<fmt>.zig` children. Dense f32,
/// f16, and bf16 weights take `packDenseMatmulRhs` (the f32 panel).
pub fn packMatmulRhs(self: *ExecContext, comptime dt: DType, rhs: *const tensor.TensorOf(dt)) !backend_mod.PackedRhsFor(dt) {
    return packMatmulRhsAs(self, backend_mod.PackedRhsFor(dt), rhs);
}

/// Container-addressed variant of `packMatmulRhs`: pack into a specific
/// RHS container type (the explicit-layout escape hatch, e.g. the Q4_K x8
/// pack on an smmla target).
pub fn packMatmulRhsAs(self: *ExecContext, comptime Rhs: type, rhs: *const tensor.TensorOf(Rhs.dtype)) !Rhs {
    const view = try rhs.rankView(2);
    if (!rhs.isContiguous()) return tensor.TensorError.UnsupportedView;
    const n = view.dim(0);
    const k = view.dim(1);
    const blocks_per_row = try backend_mod.quantized_matmul.blockCountForDType(Rhs.dtype, k);
    return backend_mod.quantized_matmul.packRhsAs(Rhs, self.allocator, rhs.dataConst(), n, k, blocks_per_row);
}

/// The two packed-RHS arms of `matmulPacked`, selected by the container
/// type: the f32 output-row panel (`packDenseMatmulRhs`) or a quantized
/// lane pack (`backend.PackedRhsFor(dt)`, `packMatmulRhs`/`packMatmulRhsAs`).
const PackedArm = enum { dense, quant };

fn packedArm(comptime RhsContainer: type) PackedArm {
    if (RhsContainer == backend_mod.PackedDenseRhs) return .dense;
    if (!@hasDecl(RhsContainer, "dtype")) @compileError("matmulPacked: not a packed RHS container: " ++ @typeName(RhsContainer));
    if (dtype_mod.isBlockQuantized(RhsContainer.dtype)) return .quant;
    @compileError("matmulPacked: not a packed RHS container: " ++ @typeName(RhsContainer));
}

/// The output type of `matmulPacked(a, rhs)`: f32 for both arms, which take
/// an f32 LHS.
pub fn MatmulPackedOutput(comptime Lhs: type, comptime Rhs: type) type {
    const lhs_dtype = @typeInfo(Lhs).pointer.child.dtype;
    const RhsContainer = @typeInfo(Rhs).pointer.child;
    _ = packedArm(RhsContainer);
    if (lhs_dtype != .f32) @compileError("matmulPacked: " ++ @typeName(RhsContainer) ++ " takes an f32 LHS");
    return Tensor;
}

/// Activations [m, k] x a pre-packed RHS -> [m, n]. `rhs` points at a
/// packed RHS container and its type selects the backend kernel at
/// comptime (`MatmulPackedOutput` names the arms and their LHS/output
/// dtypes). The quantized arm honors `pinRowwiseKernels`.
pub fn matmulPacked(self: *ExecContext, a: anytype, rhs: anytype) !MatmulPackedOutput(@TypeOf(a), @TypeOf(rhs)) {
    const RhsContainer = @typeInfo(@TypeOf(rhs)).pointer.child;
    const av = try a.rankView(2);
    const m = av.dim(0);
    const k = av.dim(1);
    if (k != rhs.k) return tensor.TensorError.ShapeMismatch;

    switch (comptime packedArm(RhsContainer)) {
        .dense => {
            var aa = try self.prepareContiguous(.f32, a);
            defer aa.deinit();
            var out = try self.empty(.f32, .{ m, rhs.n });
            errdefer out.deinit();
            self.enableNativeMatmulPoolForWork(.f32, m, rhs.n, k);
            try kernels.matmulPacked(self.pc(), self.allocator, &out, aa.tensor(), rhs, m, rhs.n, k);
            return out;
        },
        .quant => {
            if (self.pin_rowwise_kernels and m > 1) return pinnedRowwise(self, a, struct {
                rhs: @TypeOf(rhs),
                fn call(c: @This(), ctx: *ExecContext, row: *const Tensor) anyerror!Tensor {
                    return matmulPacked(ctx, row, c.rhs);
                }
            }{ .rhs = rhs });

            var aa = try self.prepareContiguous(.f32, a);
            defer aa.deinit();
            var out = try self.empty(.f32, .{ m, rhs.n });
            errdefer out.deinit();
            self.enableNativeMatmulPoolForWork(RhsContainer.dtype, m, rhs.n, k);
            try kernels.matmulPacked(self.pc(), self.allocator, &out, aa.tensor(), rhs, m, rhs.n, k);
            return out;
        },
    }
}

/// `matmulPacked` into caller-supplied contiguous storage: the buffer-reuse
/// form, dense-panel arm only. Accelerator builds are synchronized before
/// return because callers may consume the borrowed host slice directly
/// rather than crossing a later Tensor visibility boundary.
pub fn matmulPackedInto(self: *ExecContext, out: *Tensor, a: *const Tensor, rhs: *const backend_mod.PackedDenseRhs) !void {
    const av = try a.rankView(2);
    const ov = try out.rankView(2);
    const m = av.dim(0);
    const k = av.dim(1);
    if (k != rhs.k or ov.dim(0) != m or ov.dim(1) != rhs.n) return tensor.TensorError.ShapeMismatch;
    if (!out.isContiguous()) return tensor.TensorError.UnsupportedView;

    var aa = try self.prepareContiguous(.f32, a);
    defer aa.deinit();
    self.enableNativeMatmulPoolForWork(.f32, m, rhs.n, k);
    try kernels.matmulPacked(self.pc(), self.allocator, out, aa.tensor(), rhs, m, rhs.n, k);
    _ = try out.dataConstChecked();
}

/// Fused split-SwiGLU + packed down GEMM: the packed RHS container type
/// selects the per-format implementation at comptime.
pub fn splitSwiGluMatmulPacked(self: *ExecContext, gate_up: *const Tensor, rhs: anytype) !Tensor {
    return switch (comptime kQuantFusedKind(@TypeOf(rhs.*), "splitSwiGluMatmulPacked")) {
        .q8_0x4 => splitSwiGluMatmulQ8_0x4Impl(self, gate_up, rhs),
        .q4_kx8 => splitSwiGluMatmulKQuantImpl(self, .q4_kx8, gate_up, rhs),
        .q5_kx8 => splitSwiGluMatmulKQuantImpl(self, .q5_kx8, gate_up, rhs),
        .q6_kx4 => splitSwiGluMatmulKQuantImpl(self, .q6_kx4, gate_up, rhs),
    };
}

/// The packed containers with fused activation kernels; every other
/// container type is a compile error at the dispatch site.
const FusedRhsKind = enum { q8_0x4, q4_kx8, q5_kx8, q6_kx4 };

fn kQuantFusedKind(comptime Rhs: type, comptime op_name: []const u8) FusedRhsKind {
    return if (Rhs == backend_mod.QuantizedMatmulRhsQ8_0x4)
        .q8_0x4
    else if (Rhs == backend_mod.QuantizedMatmulRhsQ4_Kx8)
        .q4_kx8
    else if (Rhs == backend_mod.QuantizedMatmulRhsQ5_Kx8)
        .q5_kx8
    else if (Rhs == backend_mod.QuantizedMatmulRhsQ6_Kx4)
        .q6_kx4
    else
        @compileError(op_name ++ ": no fused kernel for packed RHS " ++ @typeName(Rhs));
}

fn splitSwiGluMatmulQ8_0x4Impl(
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
            return splitSwiGluMatmulQ8_0x4Impl(ctx, row, c.rhs);
        }
    }{ .rhs = rhs });

    var gg = try self.prepareContiguous(.f32, gate_up);
    defer gg.deinit();

    // Decode (m == 1): the lane-packed kernel pads the single row to four
    // sdot lanes and pays four-row compute for one row of output — measured
    // at roughly half the weight-stream GB/s of the plain-lhs x4 kernel
    // (bench-q8gemv). Materialize the fused SwiGLU row and take the
    // plain-lhs route instead.
    if (m == 1) {
        var fused = try self.empty(.f32, .{ 1, k });
        defer fused.deinit();
        backend_mod.quantized_matmul.q8_0.splitSwiGluRowInto(fused.data(), gg.tensor().dataConst(), k);
        var row_out = try self.empty(.f32, .{ 1, rhs.n });
        errdefer row_out.deinit();
        self.enableNativeMatmulPoolForWork(comptime packedRhsDType(@TypeOf(rhs)), 1, rhs.n, k);
        try kernels.matmulPacked(self.pc(), self.allocator, &row_out, &fused, rhs, 1, rhs.n, k);
        return row_out;
    }

    var out = try self.empty(.f32, .{ m, rhs.n });
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
    const pooled = m * k >= parallel.fused_chain_len_threshold and
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
    self.enableNativeMatmulPoolForWork(comptime packedRhsDType(@TypeOf(rhs)), m, rhs.n, k);
    if (m % 4 == 0) {
        try kernels.matmul2DPackedQ8_0x4LhsRhs(self.pc(), &out, qlhs_blocks, rhs, m, rhs.n, k);
    } else {
        try kernels.matmul2DPackedPaddedQ8_0x4LhsRhs(self.pc(), &out, qlhs_blocks, rhs, m, rhs.n, k);
    }
    return out;
}

fn fusedActQuantDispatch(self: *ExecContext, comptime TaskT: type, base: TaskT, row_groups: usize, scratch: []f32) void {
    const cols = base.cols;
    if (base.rows * cols >= parallel.fused_chain_len_threshold) {
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
    const gv = try gate_up.rankView(2);
    const m = gv.dim(0);
    const axis_dim = gv.dim(1);
    if (axis_dim % 2 != 0) return tensor.TensorError.InvalidShape;
    const k = axis_dim / 2;
    if (k != rhs.k) return tensor.TensorError.ShapeMismatch;

    var gg = try self.prepareContiguous(.f32, gate_up);
    defer gg.deinit();
    return fusedKQuantGemm(self, kind, .split_swiglu, rhs, gg.tensor().dataConst(), axis_dim, m, k, .{});
}

/// Act-specific task fields the fused K-quant engine threads through to
/// `FusedActQuantTask`; fields an act does not use keep the Task defaults.
const FusedKQuantExtras = struct {
    up: []const f32 = &.{},
    eps: f32 = 0,
    inv_cols: f32 = 0,
    rows_kernel: bool = true,
};

/// The shared engine of the fused activation+quantize+K-quant-GEMM entries
/// (`splitSwiGluMatmulKQuantImpl`, `rmsNormMulMatmulKQuantImpl`): the
/// x4-batch policy, the Q8_Kx4 prefix arm, the per-row Q8_K tail arm,
/// scratch and pool setup, and the packed GEMM per stage. Callers validate
/// shapes and prepare `input` (contiguous f32 rows, `input_row_stride`
/// elements apart); the act picks the task family.
fn fusedKQuantGemm(
    self: *ExecContext,
    comptime kind: KQuantFusedRhsKind,
    comptime act: exec_row_ops.FusedActKind,
    rhs: anytype,
    input: []const f32,
    input_row_stride: usize,
    m: usize,
    k: usize,
    extras: FusedKQuantExtras,
) !Tensor {
    const qm = backend_mod.quantized_matmul;
    const blocks_per_row = try qm.q8k.qkBlockCount(k);
    const n = rhs.n;

    var out = try self.empty(.f32, .{ m, n });
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

    self.enableNativeMatmulPoolForWork(comptime packedRhsDType(@TypeOf(rhs)), m, n, k);

    if (prefix_rows > 0) {
        const row_groups = if (pad_x4) (prefix_rows + 3) / 4 else prefix_rows / 4;
        var qlhs_x4_lease = try self.buffers.acquireScratch(qm.BlockQ8_Kx4, try checkedTensorProduct(row_groups, blocks_per_row));
        defer qlhs_x4_lease.release();
        const qlhs_x4 = qlhs_x4_lease.items;
        const TaskT = FusedActQuantTask(act, .q8_kx4);
        fusedActQuantDispatch(self, TaskT, .{
            .gate = input,
            .up = extras.up,
            .scratch = &.{},
            .rows = prefix_rows,
            .cols = k,
            .blocks_per_row = blocks_per_row,
            .eps = extras.eps,
            .inv_cols = extras.inv_cols,
            .rows_kernel = extras.rows_kernel,
            .row_group_start = 0,
            .row_group_end = row_groups,
            .x4_blocks = qlhs_x4,
        }, row_groups, scratch);
        // Only the x4-capable kinds reach this stage (use_x4 is false for
        // .q6_kx4, so its prefix is always empty).
        if (comptime kind == .q6_kx4) unreachable else kernels.matmulPackedSlice(self.pc(), out_data[0 .. prefix_rows * n], qlhs_x4, rhs, prefix_rows, n, k);
    }

    if (prefix_rows < m) {
        const tail_rows = m - prefix_rows;
        const tail_groups = (tail_rows + 3) / 4;
        var qlhs_rows_lease = try self.buffers.acquireScratch(dtype_mod.BlockQ8_K, try checkedTensorProduct(tail_rows, blocks_per_row));
        defer qlhs_rows_lease.release();
        const qlhs_rows = qlhs_rows_lease.items;
        const TaskT = FusedActQuantTask(act, .q8_k_rows);
        fusedActQuantDispatch(self, TaskT, .{
            .gate = input[prefix_rows * input_row_stride ..],
            .up = extras.up,
            .scratch = &.{},
            .rows = tail_rows,
            .cols = k,
            .blocks_per_row = blocks_per_row,
            .eps = extras.eps,
            .inv_cols = extras.inv_cols,
            .rows_kernel = extras.rows_kernel,
            .row_group_start = 0,
            .row_group_end = tail_groups,
            .row_blocks = qlhs_rows,
        }, tail_groups, scratch);
        const tail_out = out_data[prefix_rows * n ..][0 .. tail_rows * n];
        kernels.matmulPackedSlice(self.pc(), tail_out, qlhs_rows, rhs, tail_rows, n, k);
    }
    return out;
}

/// Fused rmsNormMul + packed GEMM: the packed RHS container type selects
/// the per-format implementation at comptime.
pub fn rmsNormMulMatmulPacked(self: *ExecContext, x: *const Tensor, norm_weights: *const Tensor, eps: f32, rhs: anytype) !Tensor {
    return switch (comptime kQuantFusedKind(@TypeOf(rhs.*), "rmsNormMulMatmulPacked")) {
        .q8_0x4 => rmsNormMulMatmulQ8_0x4Impl(self, x, norm_weights, eps, rhs),
        .q4_kx8 => rmsNormMulMatmulKQuantImpl(self, .q4_kx8, x, norm_weights, eps, rhs),
        .q5_kx8 => rmsNormMulMatmulKQuantImpl(self, .q5_kx8, x, norm_weights, eps, rhs),
        .q6_kx4 => rmsNormMulMatmulKQuantImpl(self, .q6_kx4, x, norm_weights, eps, rhs),
    };
}

fn rmsNormMulMatmulKQuantImpl(self: *ExecContext, comptime kind: KQuantFusedRhsKind, x: *const Tensor, norm_weights: *const Tensor, eps: f32, rhs: anytype) !Tensor {
    const xv = try x.rankView(2);
    const m = xv.dim(0);
    const k = xv.dim(1);
    if (k != rhs.k) return tensor.TensorError.ShapeMismatch;
    const wv = try norm_weights.rankView(1);
    if (wv.dim(0) != k) return tensor.TensorError.ShapeMismatch;

    var xx = try self.prepareContiguous(.f32, x);
    defer xx.deinit();
    var ww = try self.prepareContiguous(.f32, norm_weights);
    defer ww.deinit();
    return fusedKQuantGemm(self, kind, .rms_norm_mul, rhs, xx.tensor().dataConst(), k, m, k, .{
        .up = ww.tensor().dataConst(),
        .eps = eps,
        .inv_cols = 1.0 / @as(f32, @floatFromInt(k)),
        .rows_kernel = m * k >= parallel.row_kernel_len_threshold,
    });
}

/// Fused rmsNormMul + Q8_0x4 LHS quantize + packed GEMM: normalizes the
/// PRE-norm rows into task-private scratch with the exact kernels the
/// unfused dispatch uses, then quantizes — matches the unfused pair to f32
/// roundoff (see rmsNormMulDotPacked), no [m, k] normalized tensor.
fn rmsNormMulMatmulQ8_0x4Impl(self: *ExecContext, x: *const Tensor, norm_weights: *const Tensor, eps: f32, rhs: *const backend_mod.QuantizedMatmulRhsQ8_0x4) !Tensor {
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
            return rmsNormMulMatmulQ8_0x4Impl(ctx, row, c.norm_weights, c.eps, c.rhs);
        }
    }{ .norm_weights = norm_weights, .eps = eps, .rhs = rhs });
    const wv = try norm_weights.rankView(1);
    if (wv.dim(0) != k) return tensor.TensorError.ShapeMismatch;
    const blocks_per_row = try qm.q8k.q8_0BlockCount(k);
    const n = rhs.n;

    var xx = try self.prepareContiguous(.f32, x);
    defer xx.deinit();
    var ww = try self.prepareContiguous(.f32, norm_weights);
    defer ww.deinit();

    var out = try self.empty(.f32, .{ m, n });
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

    self.enableNativeMatmulPoolForWork(comptime packedRhsDType(@TypeOf(rhs)), m, n, k);
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
pub fn gegluQuantMatmulPacked(self: *ExecContext, gate: *const Tensor, up: *const Tensor, rhs: *const backend_mod.QuantizedMatmulRhsQ8_0x4) anyerror!Tensor {
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
        var gg_pin = try self.prepareContiguous(.f32, gate);
        defer gg_pin.deinit();
        var uu_pin = try self.prepareContiguous(.f32, up);
        defer uu_pin.deinit();
        const g_in = gg_pin.tensor().dataConst();
        const u_in = uu_pin.tensor().dataConst();
        var g_row = try self.empty(.f32, .{ 1, k });
        defer g_row.deinit();
        var u_row = try self.empty(.f32, .{ 1, k });
        defer u_row.deinit();
        var out = try self.empty(.f32, .{ m, rhs.n });
        errdefer out.deinit();
        for (0..m) |r| {
            @memcpy(g_row.data(), g_in[r * k ..][0..k]);
            @memcpy(u_row.data(), u_in[r * k ..][0..k]);
            var row_out = try gegluQuantMatmulPacked(self, &g_row, &u_row, rhs);
            defer row_out.deinit();
            @memcpy(out.data()[r * rhs.n ..][0..rhs.n], row_out.dataConst());
        }
        return out;
    }
    const blocks_per_row = try qm.q8k.q8_0BlockCount(k);
    const n = rhs.n;

    var gg = try self.prepareContiguous(.f32, gate);
    defer gg.deinit();
    var uu = try self.prepareContiguous(.f32, up);
    defer uu.deinit();

    var out = try self.empty(.f32, .{ m, n });
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

    self.enableNativeMatmulPoolForWork(comptime packedRhsDType(@TypeOf(rhs)), m, n, k);
    if (m % 4 == 0) {
        try kernels.matmul2DPackedQ8_0x4LhsRhs(self.pc(), &out, qlhs, rhs, m, n, k);
    } else {
        try kernels.matmul2DPackedPaddedQ8_0x4LhsRhs(self.pc(), &out, qlhs, rhs, m, n, k);
    }
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
/// Try the accelerator for `input[m,k] · rhs[n,k]ᵀ` with a quantized-bytes
/// RHS (`nb01` bytes per RHS row). `null` when it declines: the caller runs
/// its CPU path. The gates assume the caller's CPU fallback is the packed
/// panel kernel set.
pub fn tryMatmulQuantRhs(
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
    const fmt = comptime offload.QuantFormat.fromDType(dtype) orelse @compileError("tryMatmulQuantRhs supports q4_k/q5_k/q6_k/q8_0/tq2_0 only");
    return tryQuantGemm(self, fmt, rhs_bytes, rhs_lifetime, nb01, input, m, n, k, .panels);
}

/// `tryMatmulQuantRhs` for the folded ternary layout (`weights/ptqtp.zig`):
/// one dispatch over the folded planes, prefill shapes only.
pub fn tryMatmulTernaryFolded(
    self: *ExecContext,
    rhs_bytes: []const u8,
    rhs_lifetime: RhsLifetime,
    nb01: usize,
    input: *const Tensor,
    m: usize,
    n: usize,
    k: usize,
) !?Tensor {
    return tryQuantGemm(self, .tq2_0_folded, rhs_bytes, rhs_lifetime, nb01, input, m, n, k, .panels);
}

fn tryQuantGemm(
    self: *ExecContext,
    comptime fmt: offload.QuantFormat,
    rhs_bytes: []const u8,
    rhs_lifetime: RhsLifetime,
    nb01: usize,
    input: *const Tensor,
    m: usize,
    n: usize,
    k: usize,
    comptime arm: offload.QuantGemmArm,
) !?Tensor {
    if (!offload.quantGemmAccepts(fmt, m, n, k, input.isContiguous(), arm)) return null;
    var out = try self.empty(.f32, .{ m, n });
    errdefer out.deinit();
    if (offload.gemmQuant(fmt, rhs_bytes, rhs_lifetime.isCacheable(), nb01, input, &out, m, n, k)) return out;
    out.deinit();
    return null;
}

/// Try the accelerator for `batch_count` GEMMs that share one `input[m,k]`
/// against `batch_count` quantized RHS slabs (`nb02` bytes apart); the
/// result is `[batch_count * m, n]`. `null` when it declines.
pub fn tryMatmulQuantRhsSharedInput(
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
    const fmt = comptime offload.QuantFormat.fromDType(dtype) orelse @compileError("tryMatmulQuantRhsSharedInput supports q4_k/q5_k/q6_k/q8_0/tq2_0 only");
    if (!offload.quantGemmSharedInputAccepts(fmt, batch_count, m, n, k, input.isContiguous())) return null;
    const rows_total = std.math.mul(usize, batch_count, m) catch return null;
    var out = try self.empty(.f32, .{ rows_total, n });
    errdefer out.deinit();
    if (offload.gemmQuantSharedInput(fmt, rhs_bytes, rhs_lifetime.isCacheable(), nb01, nb02, input, &out, batch_count, m, n, k)) return out;
    out.deinit();
    return null;
}

/// The blocks-RHS entries' offload attempt (the CPU fallback is the
/// block kernels, which the gates account for).
fn tryQuantGemmForBlocks(
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
    if (comptime (dtype != .q4_k and dtype != .q5_k and dtype != .q6_k and dtype != .q8_0)) return null;
    const fmt = comptime offload.QuantFormat.fromDType(dtype).?;
    const nb01 = std.math.divExact(usize, rhs_bytes.len, n) catch return null;
    return tryQuantGemm(self, fmt, rhs_bytes, rhs_lifetime, nb01, input, m, n, k, .blocks);
}

fn quantMatmulWork(m: usize, n: usize, k: usize) u64 {
    const mm: u64 = @intCast(m);
    const nn: u64 = @intCast(n);
    const kk: u64 = @intCast(k);
    const mn = std.math.mul(u64, mm, nn) catch return std.math.maxInt(u64);
    return std.math.mul(u64, mn, kk) catch std.math.maxInt(u64);
}
