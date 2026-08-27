//! Quantized matmul dispatch: dequantize/getRows, the one
//! `matmulQuant`/`matmulQuantInto` request entry, RHS container
//! preparation (compact borrows and lane packs), the fused
//! activation+quantize+GEMM arms (`split_swiglu`/`rms_norm_mul`/
//! `geglu_quant` over the Q8_0x4 and K-quant packs), and the GPU
//! dense-quant entries. Domain module: every op receives an explicit
//! `*ExecContext`.
const std = @import("std");
const backend_mod = @import("../backend.zig");
const offload = backend_mod.offload;
const kernels = backend_mod.kernels;
const dtype_mod = @import("../dtype.zig");
const parallel = @import("../parallel.zig");
const tensor = @import("../tensor.zig");

const exec_buffer_pool = @import("buffer_pool.zig");
const exec_row_ops = backend_mod.rows;
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

pub const Placement = enum { auto, cpu };
pub const Numerics = enum { batched, rowwise };

/// One quantized matmul request at the exec seam: `prologue` names the
/// fused activation applied to the LHS rows before quantization (null =
/// plain activations), `placement` whether the accelerator may be
/// consulted (`.auto` asks `offload`, exactly as the dedicated entries
/// did; a decline falls through to the CPU kernels), `rhs_lifetime` the
/// RHS storage guarantee for address-keyed GPU caches, and `numerics` the
/// batch policy: `.rowwise` pins every row to the m == 1 kernels, also
/// forced context-wide by an open `pinRowwiseNumerics` scope (the K-quant
/// fused engine pins by forcing its per-row tail kernel instead of
/// looping).
pub const QuantMatmul = struct {
    prologue: ?exec_row_ops.FusedActKind = null,
    placement: Placement = .auto,
    rhs_lifetime: RhsLifetime = .transient,
    numerics: Numerics = .batched,
};

/// The activation operand of `matmulQuant`: plain rows, or the operands
/// of a fused prologue. `.split_swiglu` reads the fused gate|up rows
/// (width 2k) through `.plain`; `.rms_norm_mul` takes `.rms_norm` (the
/// pre-norm rows, the [k] norm weight, eps); `.geglu_quant` takes
/// `.gate_up` (separate gate and up rows).
pub const Lhs = union(enum) {
    plain: *const Tensor,
    rms_norm: struct { x: *const Tensor, weight: *const Tensor, eps: f32 },
    gate_up: struct { gate: *const Tensor, up: *const Tensor },
};

/// RHS families `matmulQuant` serves, derived from the container type:
/// compact `.rows` containers (block kernels via the backend union entry,
/// GPU dequant-in-kernel candidates), lane-packed containers (the packed
/// kernel set and the fused prologues), and the dense f32 panel.
const RhsClass = enum { compact, lane_packed, dense };

fn rhsClass(comptime Rhs: type) RhsClass {
    if (Rhs == backend_mod.PackedDenseRhs) return .dense;
    if (@hasDecl(Rhs, "pack")) return if (Rhs.pack == .rows) .compact else .lane_packed;
    @compileError("matmulQuant: not a packed or compact RHS container: " ++ @typeName(Rhs));
}

/// The compact `.rows` container of a weight dtype (the backend's map):
/// the RHS type `compactMatmulRhs`/`compactMatmulRhsFromBlocks` build and
/// `matmulQuant`'s compact arm consumes.
pub fn CompactRhsFor(comptime dt: DType) type {
    return @typeInfo(backend_mod.ops.RhsOf(backend_mod.ops.QuantGemm.rowsFor(dt))).pointer.child;
}

/// Stack container borrowing caller-owned blocks (never deinit'd here;
/// the matmul path never mutates blocks, so the @constCast over
/// containers whose block slice is mutable is sound). Containers without
/// the ?Allocator borrow pattern keep the legacy owning shape.
fn compactFromBlocks(comptime Rhs: type, allocator: std.mem.Allocator, blocks: anytype, n: usize, k: usize, blocks_per_row: usize) Rhs {
    if (comptime @hasField(Rhs, "rows")) {
        const Rows = @FieldType(Rhs, "rows");
        return .{
            .rows = if (comptime @typeInfo(@FieldType(Rows, "allocator")) == .optional)
                .{ .allocator = null, .blocks = @constCast(blocks), .rows = n, .cols = k, .blocks_per_row = blocks_per_row }
            else
                .{ .allocator = allocator, .blocks = @constCast(blocks), .rows = n, .cols = k, .blocks_per_row = blocks_per_row },
            .k = k,
            .n = n,
        };
    }
    return if (comptime @typeInfo(@FieldType(Rhs, "allocator")) == .optional)
        .{ .allocator = null, .blocks = blocks, .k = k, .n = n, .blocks_per_column = blocks_per_row }
    else
        .{ .allocator = allocator, .blocks = @constCast(blocks), .k = k, .n = n, .blocks_per_column = blocks_per_row };
}

/// The raw block bytes behind a compact container (the GPU attempt's RHS).
fn compactBlocksBytes(rhs: anytype) []const u8 {
    return if (comptime @hasField(@typeInfo(@TypeOf(rhs)).pointer.child, "rows"))
        std.mem.sliceAsBytes(rhs.rows.blocks)
    else
        std.mem.sliceAsBytes(rhs.blocks);
}

/// The primary LHS tensor of a request (the one whose rows set m).
fn lhsPrimary(lhs: Lhs) *const Tensor {
    return switch (lhs) {
        .plain => |x| x,
        .rms_norm => |r| r.x,
        .gate_up => |g| g.gate,
    };
}

/// Validated [m, k] geometry of the request against `rhs.k` (`k` is the
/// GEMM inner dim; `.split_swiglu` reads 2k-wide fused rows), including
/// the prologue-operand pairing.
fn lhsGeometry(lhs: Lhs, comptime opts: QuantMatmul, rhs_k: usize) !struct { m: usize, k: usize } {
    const av = try lhsPrimary(lhs).rankView(2);
    const m = av.dim(0);
    var k = av.dim(1);
    if (comptime opts.prologue == .split_swiglu) {
        if (k % 2 != 0) return tensor.TensorError.InvalidShape;
        k /= 2;
    }
    if (k != rhs_k) return tensor.TensorError.ShapeMismatch;
    switch (lhs) {
        .plain => if (comptime opts.prologue != null and opts.prologue != .split_swiglu) return tensor.TensorError.InvalidShape,
        .rms_norm => |r| {
            if (comptime opts.prologue != .rms_norm_mul) return tensor.TensorError.InvalidShape;
            const wv = try r.weight.rankView(1);
            if (wv.dim(0) != k) return tensor.TensorError.ShapeMismatch;
        },
        .gate_up => |g| {
            if (comptime opts.prologue != .geglu_quant) return tensor.TensorError.InvalidShape;
            const uv = try g.up.rankView(2);
            if (uv.dim(0) != m or uv.dim(1) != k) return tensor.TensorError.ShapeMismatch;
        },
    }
    return .{ .m = m, .k = k };
}

/// The one quantized matmul entry: `lhs` (with `opts.prologue`'s fused
/// activation) x a packed or compact RHS container -> f32 [m, rhs.n]. The
/// body is one row-pinned fallback, one fused prologue (prepare + fused
/// activate+quantize), one accelerator attempt, and one backend call.
pub fn matmulQuant(self: *ExecContext, lhs: Lhs, rhs: anytype, comptime opts: QuantMatmul) !Tensor {
    const geo = try lhsGeometry(lhs, opts, rhs.k);
    var out = try self.empty(.f32, .{ geo.m, rhs.n });
    errdefer out.deinit();
    try matmulQuantImpl(self, &out, lhs, rhs, opts, geo.m, geo.k);
    return out;
}

/// `matmulQuant` into caller-supplied contiguous storage (the
/// buffer-reuse form). Accelerator builds are synchronized before return
/// because callers may consume the borrowed host slice directly.
pub fn matmulQuantInto(self: *ExecContext, out: *Tensor, lhs: Lhs, rhs: anytype, comptime opts: QuantMatmul) !void {
    const geo = try lhsGeometry(lhs, opts, rhs.k);
    const ov = try out.rankView(2);
    if (ov.dim(0) != geo.m or ov.dim(1) != rhs.n) return tensor.TensorError.ShapeMismatch;
    if (!out.isContiguous()) return tensor.TensorError.UnsupportedView;
    try matmulQuantImpl(self, out, lhs, rhs, opts, geo.m, geo.k);
    _ = try out.dataConstChecked();
}

/// True when the request pins batches through the K-quant fused engine's
/// per-row tail kernel rather than the row loop.
fn pinsViaTailKernel(comptime Rhs: type, comptime opts: QuantMatmul) bool {
    if (opts.prologue == null) return false;
    return rhsClass(Rhs) == .lane_packed and kQuantFusedKind(Rhs, "matmulQuant") != .q8_0x4;
}

fn matmulQuantImpl(self: *ExecContext, out: *Tensor, lhs: Lhs, rhs: anytype, comptime opts: QuantMatmul, m: usize, k: usize) !void {
    const Rhs = @typeInfo(@TypeOf(rhs)).pointer.child;
    comptime if (opts.prologue != null and rhsClass(Rhs) != .lane_packed)
        @compileError("matmulQuant: the fused prologues serve the lane-packed containers, not " ++ @typeName(Rhs));
    // The one row-pinned fallback (see `QuantMatmul.numerics` and
    // `ExecContext.pinRowwiseNumerics`). The dense panel never pins; the K-quant
    // fused engine pins inside its body.
    if (comptime rhsClass(Rhs) != .dense and !pinsViaTailKernel(Rhs, opts)) {
        if ((self.rowwiseNumericsPinned() or opts.numerics == .rowwise) and m > 1)
            return rowwisePinned(self, out, lhs, rhs, opts, m, k);
    }
    return matmulQuantBody(self, out, lhs, rhs, opts, m, k);
}

/// Kernel-pinned batch fallback: prepare each input once, then feed the
/// SAME body one row at a time, so every row's numerics are exactly the
/// m == 1 numerics regardless of which kernels the backend would pick for
/// the batch shape — including a GPU backend, whose row calls take its
/// own m == 1 path. Calls `matmulQuantBody` (not the pinned entry), so
/// the error sets stay inferred — no `anyerror` cycle-breaker.
fn rowwisePinned(self: *ExecContext, out: *Tensor, lhs: Lhs, rhs: anytype, comptime opts: QuantMatmul, m: usize, k: usize) !void {
    const n = rhs.n;
    const out_data = out.data();
    var row_out = try self.empty(.f32, .{ 1, n });
    defer row_out.deinit();
    switch (lhs) {
        .plain => |x| {
            const width = if (comptime opts.prologue == .split_swiglu) 2 * k else k;
            var aa = try self.prepareContiguous(.f32, x);
            defer aa.deinit();
            const input = aa.tensor().dataConst();
            var row = try self.empty(.f32, .{ 1, width });
            defer row.deinit();
            for (0..m) |r| {
                @memcpy(row.data(), input[r * width ..][0..width]);
                try matmulQuantBody(self, &row_out, .{ .plain = &row }, rhs, opts, 1, k);
                @memcpy(out_data[r * n ..][0..n], row_out.dataConst());
            }
        },
        .rms_norm => |rn| {
            var xx = try self.prepareContiguous(.f32, rn.x);
            defer xx.deinit();
            const input = xx.tensor().dataConst();
            var row = try self.empty(.f32, .{ 1, k });
            defer row.deinit();
            for (0..m) |r| {
                @memcpy(row.data(), input[r * k ..][0..k]);
                try matmulQuantBody(self, &row_out, .{ .rms_norm = .{ .x = &row, .weight = rn.weight, .eps = rn.eps } }, rhs, opts, 1, k);
                @memcpy(out_data[r * n ..][0..n], row_out.dataConst());
            }
        },
        .gate_up => |gu| {
            var gg = try self.prepareContiguous(.f32, gu.gate);
            defer gg.deinit();
            var uu = try self.prepareContiguous(.f32, gu.up);
            defer uu.deinit();
            const g_in = gg.tensor().dataConst();
            const u_in = uu.tensor().dataConst();
            var g_row = try self.empty(.f32, .{ 1, k });
            defer g_row.deinit();
            var u_row = try self.empty(.f32, .{ 1, k });
            defer u_row.deinit();
            for (0..m) |r| {
                @memcpy(g_row.data(), g_in[r * k ..][0..k]);
                @memcpy(u_row.data(), u_in[r * k ..][0..k]);
                try matmulQuantBody(self, &row_out, .{ .gate_up = .{ .gate = &g_row, .up = &u_row } }, rhs, opts, 1, k);
                @memcpy(out_data[r * n ..][0..n], row_out.dataConst());
            }
        },
    }
}

/// The formats the compact-RHS accelerator attempt serves (the dequant-
/// in-kernel GEMM over raw row blocks).
fn compactGpuFormat(comptime dt: DType) ?offload.QuantFormat {
    return switch (dt) {
        .q4_k, .q5_k, .q6_k, .q8_0 => offload.QuantFormat.fromDType(dt),
        else => null,
    };
}

/// The one accelerator attempt: `offload`'s accept gates, then the
/// dequant-in-kernel GEMM into `out`. False = declined or failed; the
/// caller runs its CPU path (never-a-loss).
fn quantGemmAttempt(comptime fmt: offload.QuantFormat, out: *Tensor, rhs_bytes: []const u8, rhs_lifetime: RhsLifetime, nb01: usize, input: *const Tensor, m: usize, n: usize, k: usize, comptime arm: offload.QuantGemmArm) bool {
    if (!offload.quantGemmAccepts(fmt, m, n, k, input.isContiguous(), arm)) return false;
    return offload.gemmQuant(fmt, rhs_bytes, rhs_lifetime.isCacheable(), nb01, input, out, m, n, k);
}

fn matmulQuantBody(self: *ExecContext, out: *Tensor, lhs: Lhs, rhs: anytype, comptime opts: QuantMatmul, m: usize, k: usize) !void {
    const Rhs = @typeInfo(@TypeOf(rhs)).pointer.child;
    const n = rhs.n;
    if (comptime opts.prologue) |act| {
        const kind = comptime kQuantFusedKind(Rhs, "matmulQuant");
        if (comptime kind == .q8_0x4) return fusedQ8_0x4(self, out, act, lhs, rhs, m, k);
        comptime if (act == .geglu_quant) @compileError("matmulQuant: the geglu_quant prologue serves the Q8_0x4 pack only");
        const kk: KQuantFusedRhsKind = comptime switch (kind) {
            .q4_kx8 => .q4_kx8,
            .q5_kx8 => .q5_kx8,
            .q6_kx4 => .q6_kx4,
            else => unreachable,
        };
        const pinned = self.rowwiseNumericsPinned() or opts.numerics == .rowwise;
        switch (comptime act) {
            .split_swiglu => {
                const gate_up = switch (lhs) {
                    .plain => |x| x,
                    else => return tensor.TensorError.InvalidShape,
                };
                var gg = try self.prepareContiguous(.f32, gate_up);
                defer gg.deinit();
                return fusedKQuantGemm(self, out, kk, act, rhs, gg.tensor().dataConst(), 2 * k, m, k, .{}, pinned);
            },
            .rms_norm_mul => {
                const rn = switch (lhs) {
                    .rms_norm => |r| r,
                    else => return tensor.TensorError.InvalidShape,
                };
                var xx = try self.prepareContiguous(.f32, rn.x);
                defer xx.deinit();
                var ww = try self.prepareContiguous(.f32, rn.weight);
                defer ww.deinit();
                return fusedKQuantGemm(self, out, kk, act, rhs, xx.tensor().dataConst(), k, m, k, .{
                    .up = ww.tensor().dataConst(),
                    .eps = rn.eps,
                    .inv_cols = 1.0 / @as(f32, @floatFromInt(k)),
                    .rows_kernel = m * k >= parallel.row_kernel_len_threshold,
                }, pinned);
            },
            .geglu_quant => comptime unreachable,
        }
    }
    const a = switch (lhs) {
        .plain => |x| x,
        else => return tensor.TensorError.InvalidShape,
    };
    var aa = try self.prepareContiguous(.f32, a);
    defer aa.deinit();
    switch (comptime rhsClass(Rhs)) {
        .compact => {
            if (comptime opts.placement == .auto and compactGpuFormat(Rhs.dtype) != null) {
                const bytes = compactBlocksBytes(rhs);
                if (std.math.divExact(usize, bytes.len, @max(n, 1)) catch null) |nb01| {
                    if (n != 0 and quantGemmAttempt(compactGpuFormat(Rhs.dtype).?, out, bytes, opts.rhs_lifetime, nb01, aa.tensor(), m, n, k, .blocks)) return;
                }
            }
            self.enableNativeMatmulPoolForWork(Rhs.dtype, m, n, k);
            try kernels.matmul2DQuantizedRhs(self.pc(), self.allocator(), out, aa.tensor(), @unionInit(backend_mod.AnyQuantizedMatmulRhs, @tagName(Rhs.dtype), rhs), m, n, k);
        },
        .dense => {
            self.enableNativeMatmulPoolForWork(.f32, m, n, k);
            try kernels.matmulPacked(self.pc(), self.allocator(), out, aa.tensor(), rhs, m, n, k);
        },
        .lane_packed => {
            self.enableNativeMatmulPoolForWork(Rhs.dtype, m, n, k);
            try kernels.matmulPacked(self.pc(), self.allocator(), out, aa.tensor(), rhs, m, n, k);
        },
    }
}

/// Borrowing compact `.rows` container over a block-quantized weight
/// TENSOR's blocks, for `matmulQuant`'s compact arm:
/// `ctx.matmulQuant(.{ .plain = a }, &compact, .{ ... })`. The container
/// borrows the tensor's blocks (its deinit is not required and must not
/// outlive them); the tensor must be contiguous `[n, k]`.
pub fn compactMatmulRhs(self: *ExecContext, comptime dt: DType, rhs: *const tensor.TensorOf(dt)) !CompactRhsFor(dt) {
    comptime if (!dtype_mod.supportsQuantizedMatmulRhs(dt)) @compileError("RHS dtype does not support quantized matmul");
    const rv = try rhs.rankView(2);
    if (!rhs.isContiguous()) return tensor.TensorError.UnsupportedView;
    return compactMatmulRhsFromBlocks(self, dt, try rhs.dataConstChecked(), rv.dim(0), rv.dim(1));
}

/// `compactMatmulRhs` over a raw block slice (`[n, k]` row blocks, e.g.
/// borrowed GGUF bytes). The container borrows `blocks`.
pub fn compactMatmulRhsFromBlocks(
    self: *ExecContext,
    comptime dt: DType,
    blocks: []const dtype_mod.Storage(dt),
    n: usize,
    k: usize,
) !CompactRhsFor(dt) {
    comptime if (!dtype_mod.supportsQuantizedMatmulRhs(dt)) @compileError("RHS dtype does not support quantized matmul");
    const blocks_per_row = try backend_mod.quantized_matmul.blockCountForDType(dt, k);
    if (blocks.len != try checkedTensorProduct(n, blocks_per_row)) return tensor.TensorError.InvalidDataLength;
    return compactFromBlocks(CompactRhsFor(dt), self.allocator(), blocks, n, k, blocks_per_row);
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
    return backend_mod.quantized_matmul.packRhsAs(Rhs, self.allocator(), rhs.dataConst(), n, k, blocks_per_row);
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

/// Act-specific task fields the fused engines thread through to
/// `FusedActQuantTask`; fields an act does not use keep the Task defaults.
const FusedExtras = struct {
    up: []const f32 = &.{},
    eps: f32 = 0,
    inv_cols: f32 = 0,
    rows_kernel: bool = true,
};

/// The shared engine of the fused activation+quantize+K-quant-GEMM arms
/// (`matmulQuant` with a K-quant lane pack): the x4-batch policy, the
/// Q8_Kx4 prefix arm, the per-row Q8_K tail arm, scratch and pool setup,
/// and the packed GEMM per stage. The caller validates shapes and
/// prepares `input` (contiguous f32 rows, `input_row_stride` elements
/// apart); the act picks the task family.
fn fusedKQuantGemm(
    self: *ExecContext,
    out: *Tensor,
    comptime kind: KQuantFusedRhsKind,
    comptime act: exec_row_ops.FusedActKind,
    rhs: anytype,
    input: []const f32,
    input_row_stride: usize,
    m: usize,
    k: usize,
    extras: FusedExtras,
    pinned: bool,
) !void {
    const qm = backend_mod.quantized_matmul;
    const blocks_per_row = try qm.blockCountForDType(.q8_k, k);
    const n = rhs.n;
    const out_data = out.data();

    // Pinned mode forces the per-row tail kernels for every row: they are
    // the m == 1 dispatch, so the batch stays bit-identical to sequential
    // decode (see QuantMatmul.numerics / ExecContext.pinRowwiseNumerics).
    const use_x4 = !pinned and switch (kind) {
        .q4_kx8 => m % 4 == 0 or m >= 64 or (m >= 4 and m < 32),
        .q5_kx8 => m % 4 == 0 or m >= 128,
        .q6_kx4 => false,
    };
    const pad_x4 = kind == .q4_kx8;
    const prefix_rows = if (!use_x4) 0 else if (pad_x4) m else m - m % 4;

    const scratch_storage = try self.rt.buffers.acquire(parallel.vector_max_threads * 4 * k);
    defer scratch_storage.release();
    const scratch = scratch_storage.data[0 .. parallel.vector_max_threads * 4 * k];

    self.enableNativeMatmulPoolForWork(comptime packedRhsDType(@TypeOf(rhs)), m, n, k);

    if (prefix_rows > 0) {
        const row_groups = if (pad_x4) (prefix_rows + 3) / 4 else prefix_rows / 4;
        var qlhs_x4_lease = try self.rt.buffers.acquireScratch(qm.BlockQ8_Kx4, try checkedTensorProduct(row_groups, blocks_per_row));
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
        var qlhs_rows_lease = try self.rt.buffers.acquireScratch(dtype_mod.BlockQ8_K, try checkedTensorProduct(tail_rows, blocks_per_row));
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
}

/// The one fused activation + Q8_0x4-quantize + packed GEMM body: the act
/// picks the input resolution and fused quantizer (split_swiglu keeps its
/// dedicated fused-quantizer task; rms_norm_mul and geglu_quant share the
/// `FusedActQuantTask` stage); storage and the packed GEMM tail are
/// shared.
fn fusedQ8_0x4(self: *ExecContext, out: *Tensor, comptime act: exec_row_ops.FusedActKind, lhs: Lhs, rhs: anytype, m: usize, k: usize) !void {
    const qm = backend_mod.quantized_matmul;
    const blocks_per_row = try qm.blockCountForDType(.q8_0, k);
    const n = rhs.n;
    switch (comptime act) {
        .split_swiglu => {
            const gate_up = switch (lhs) {
                .plain => |x| x,
                else => return tensor.TensorError.InvalidShape,
            };
            var gg = try self.prepareContiguous(.f32, gate_up);
            defer gg.deinit();

            // Decode (m == 1): the lane-packed kernel pads the single row to
            // four sdot lanes and pays four-row compute for one row of output
            // — measured at roughly half the weight-stream GB/s of the
            // plain-lhs x4 kernel (bench-q8gemv). Materialize the fused
            // SwiGLU row and take the plain-lhs route instead.
            if (m == 1) {
                var fused = try self.empty(.f32, .{ 1, k });
                defer fused.deinit();
                kernels.splitSwiGluRowInto(fused.data(), gg.tensor().dataConst(), k);
                self.enableNativeMatmulPoolForWork(.q8_0, 1, n, k);
                return kernels.matmulPacked(self.pc(), self.allocator(), out, &fused, rhs, 1, n, k);
            }

            const block_count = ((m + 3) / 4) * blocks_per_row;
            var stack_blocks: [512]qm.BlockQ8_0x4 = undefined;
            var qlhs_lease: ?exec_buffer_pool.ScratchLease(qm.BlockQ8_0x4) = null;
            defer if (qlhs_lease) |*lease| lease.release();
            const qlhs_blocks = if (block_count <= stack_blocks.len)
                stack_blocks[0..block_count]
            else blk: {
                qlhs_lease = try self.rt.buffers.acquireScratch(qm.BlockQ8_0x4, block_count);
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
            if (!pooled) kernels.quantizeSplitSwiGluRowsQ8_0x4PaddedGroupsInto(
                qlhs_blocks,
                input,
                m,
                k,
                blocks_per_row,
                0,
                row_groups,
            );
            try packedQ8_0x4Tail(self, out, qlhs_blocks, rhs, m, n, k);
        },
        .rms_norm_mul => {
            const rn = switch (lhs) {
                .rms_norm => |r| r,
                else => return tensor.TensorError.InvalidShape,
            };
            var xx = try self.prepareContiguous(.f32, rn.x);
            defer xx.deinit();
            var ww = try self.prepareContiguous(.f32, rn.weight);
            defer ww.deinit();
            try fusedQ8_0x4Pipeline(self, act, xx.tensor().dataConst(), ww.tensor().dataConst(), .{
                .eps = rn.eps,
                .inv_cols = 1.0 / @as(f32, @floatFromInt(k)),
                .rows_kernel = m * k >= parallel.row_kernel_len_threshold,
            }, out, rhs, m, k, blocks_per_row);
        },
        .geglu_quant => {
            const gu = switch (lhs) {
                .gate_up => |g| g,
                else => return tensor.TensorError.InvalidShape,
            };
            var gg = try self.prepareContiguous(.f32, gu.gate);
            defer gg.deinit();
            var uu = try self.prepareContiguous(.f32, gu.up);
            defer uu.deinit();
            try fusedQ8_0x4Pipeline(self, act, gg.tensor().dataConst(), uu.tensor().dataConst(), .{}, out, rhs, m, k, blocks_per_row);
        },
    }
}

/// The shared FusedActQuantTask(q8_0x4) stage + the packed GEMM tail
/// (activate up to 4 rows into task-private scratch with the exact
/// unfused kernels, quantize with the exact unfused packers — results
/// stay bit-identical, no [m, k] activation tensor).
fn fusedQ8_0x4Pipeline(self: *ExecContext, comptime act: exec_row_ops.FusedActKind, gate_data: []const f32, up_data: []const f32, extras: FusedExtras, out: *Tensor, rhs: anytype, m: usize, k: usize, blocks_per_row: usize) !void {
    const qm = backend_mod.quantized_matmul;
    const n = rhs.n;
    const row_groups = (m + 3) / 4;
    var qlhs_lease = try self.rt.buffers.acquireScratch(qm.BlockQ8_0x4, try checkedTensorProduct(row_groups, blocks_per_row));
    defer qlhs_lease.release();
    const qlhs = qlhs_lease.items;

    const scratch_storage = try self.rt.buffers.acquire(parallel.vector_max_threads * 4 * k);
    defer scratch_storage.release();
    const scratch = scratch_storage.data[0 .. parallel.vector_max_threads * 4 * k];

    const TaskT = FusedActQuantTask(act, .q8_0x4);
    fusedActQuantDispatch(self, TaskT, .{
        .gate = gate_data,
        .up = up_data,
        .scratch = &.{},
        .rows = m,
        .cols = k,
        .blocks_per_row = blocks_per_row,
        .eps = extras.eps,
        .inv_cols = extras.inv_cols,
        .rows_kernel = extras.rows_kernel,
        .row_group_start = 0,
        .row_group_end = row_groups,
        .q8_0x4_blocks = qlhs,
    }, row_groups, scratch);
    try packedQ8_0x4Tail(self, out, qlhs, rhs, m, n, k);
}

/// The shared packed-Q8_0x4 GEMM tail of the fused arms.
fn packedQ8_0x4Tail(self: *ExecContext, out: *Tensor, qlhs: []const backend_mod.quantized_matmul.BlockQ8_0x4, rhs: anytype, m: usize, n: usize, k: usize) !void {
    self.enableNativeMatmulPoolForWork(.q8_0, m, n, k);
    if (m % 4 == 0) {
        try kernels.matmul2DPackedQ8_0x4LhsRhs(self.pc(), out, qlhs, rhs, m, n, k);
    } else {
        try kernels.matmul2DPackedPaddedQ8_0x4LhsRhs(self.pc(), out, qlhs, rhs, m, n, k);
    }
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
    var out = try self.empty(.f32, .{ m, n });
    errdefer out.deinit();
    if (quantGemmAttempt(fmt, &out, rhs_bytes, rhs_lifetime, nb01, input, m, n, k, arm)) return out;
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

fn quantMatmulWork(m: usize, n: usize, k: usize) u64 {
    const mm: u64 = @intCast(m);
    const nn: u64 = @intCast(n);
    const kk: u64 = @intCast(k);
    const mn = std.math.mul(u64, mm, nn) catch return std.math.maxInt(u64);
    return std.math.mul(u64, mn, kk) catch std.math.maxInt(u64);
}
