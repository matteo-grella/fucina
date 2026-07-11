const std = @import("std");
const backend_mod = @import("backend.zig");
const backend_ops = backend_mod.ops;
const dtype_mod = @import("dtype.zig");
const tensor = @import("tensor.zig");
const thread = @import("thread.zig");

const exec_attention = @import("exec/attention.zig");
const exec_moe = @import("exec/moe.zig");
const exec_moe_chain = @import("exec/moe_chain.zig");
const exec_matmul = @import("exec/matmul.zig");
const exec_elementwise = @import("exec/elementwise.zig");
const exec_quant_matmul = @import("exec/quant_matmul.zig");
const exec_buffer_pool = @import("exec/buffer_pool.zig");
const exec_runtime = @import("exec/runtime.zig");
const exec_convert = @import("exec/convert.zig");
const exec_rope = @import("exec/rope.zig");
const exec_softmax = @import("exec/softmax.zig");
const exec_loss = @import("exec/loss.zig");
const exec_reduce = @import("exec/reduce.zig");
const exec_topk = @import("exec/topk.zig");
const exec_stats = @import("exec/stats.zig");
const exec_gather_scatter = @import("exec/gather_scatter.zig");
const exec_norm = @import("exec/norm.zig");
const exec_conv = @import("exec/conv.zig");
const exec_pool = @import("exec/pool.zig");

const Allocator = std.mem.Allocator;
const Runtime = exec_runtime.Runtime;
const Backend = backend_mod.Backend;
const DType = tensor.DType;
const Tensor = tensor.Tensor;

pub const MoeBatchProfile = exec_moe.MoeBatchProfile;

pub const PackedMatmulFormat = backend_mod.PackedMatmulFormat;
pub const parallel_dot_backward_branches = Backend.kind == .native and backend_mod.native_uses_blas;
pub const PackedMatmulRhsFor = backend_mod.PackedMatmulRhsFor;
pub const QuantizedMatmulFormat = backend_mod.QuantizedMatmulFormat;
pub const QuantizedMatmulRhsQ4_Kx4 = backend_mod.QuantizedMatmulRhsQ4_Kx4;
pub const QuantizedMatmulRhsQ4_Kx8 = backend_mod.QuantizedMatmulRhsQ4_Kx8;
pub const QuantizedMatmulRhsQ4_Kx2Mmla = backend_mod.QuantizedMatmulRhsQ4_Kx2Mmla;
pub const QuantizedMatmulRhsQ5_Kx8 = backend_mod.QuantizedMatmulRhsQ5_Kx8;
pub const RhsLifetime = exec_quant_matmul.RhsLifetime;

pub const LayoutClass = enum {
    contiguous,
    scalar,
    tail_broadcast,
    arbitrary,
};

pub const UnaryOp = backend_ops.UnaryOp;
pub const GatedOp = backend_ops.GatedOp;
pub const ElementwiseOp = backend_ops.ElementwiseOp;
pub const CompareOp = backend_ops.CompareOp;

/// RoPE mode + precomputed sin/cos table. Defined in `exec/rope.zig`;
/// re-exported here for the autograd VJP params (`ag/*` name `exec.RopeTable`).
pub const RopeMode = exec_rope.RopeMode;
pub const RopeTable = exec_rope.RopeTable;
pub const RopeTheta = exec_rope.RopeTheta;

pub const SoftmaxExtOptions = exec_softmax.SoftmaxExtOptions;

/// Vector-norm order for the facade `norm`/`normAll` compositions:
/// `.l1` = Σ|x|, `.l2` = sqrt(Σ x²), `.inf` = max|x|.
pub const NormOrder = enum { l1, l2, inf };

pub const Reduction = exec_loss.Reduction;
pub const CrossEntropyOptions = exec_loss.CrossEntropyOptions;
pub const LinearCrossEntropyGrads = exec_loss.LinearCrossEntropyGrads;
pub const MseOptions = exec_loss.MseOptions;
pub const HuberOptions = exec_loss.HuberOptions;
pub const BceOptions = exec_loss.BceOptions;
pub const KlDivOptions = exec_loss.KlDivOptions;
pub const LossWrt = exec_loss.LossWrt;

pub const TopKResult = exec_stats.TopKResult;

pub const RouterTopKOptions = exec_topk.RouterTopKOptions;

pub const StandardizeAccumulation = exec_stats.StandardizeAccumulation;
pub const StandardizeEpsMode = exec_stats.StandardizeEpsMode;
pub const StandardizeOptions = exec_stats.StandardizeOptions;

pub const GroupedCausalAttentionBackwardResult = exec_attention.GroupedCausalAttentionBackwardResult;

pub const LayerNormAffineBackwardResult = exec_norm.LayerNormAffineBackwardResult;
pub const GroupNormBackwardResult = exec_norm.GroupNormBackwardResult;
pub const SnakeBackwardParamsResult = exec_elementwise.SnakeBackwardParamsResult;

const BlockQ8_0 = exec_attention.BlockQ8_0;

const MoeDecodeScratch = exec_moe.MoeDecodeScratch;
pub const MatmulKind = exec_matmul.MatmulKind;
pub const BmmKind = exec_matmul.BmmKind;
pub const BmmBatchMode = exec_matmul.BmmBatchMode;
pub const BmmShape = exec_matmul.BmmShape;
pub const QuantizedMatmulOptions = exec_quant_matmul.QuantizedMatmulOptions;
const tailBroadcastInfo = exec_elementwise.tailBroadcastInfo;

/// Reusable transient-buffer pool. Defined in the `exec/buffer_pool.zig` leaf;
/// re-exported here so `exec.BufferPool` stays reachable and the `Runtime`
/// `buffers` field can name it.
pub const BufferPool = exec_buffer_pool.BufferPool;

pub const ExecContext = struct {
    /// Generic runtime substrate (allocator, backend, buffer pool, worker
    /// team, exec-scope stack, tensor allocation primitives). All substrate
    /// methods below forward here; domain modules receive `&self.rt`.
    rt: Runtime,
    /// MoE-decode scratch state. Domain state, not substrate: owned by the
    /// facade and passed to the MoE ops rather than living on `Runtime`.
    moe_scratch: MoeDecodeScratch = .{},
    /// Cached copy of `rt.allocator` (a fat pointer into `rt`'s stable
    /// ThreadSafeAllocator, immutable after init). Lets the MoE decode path
    /// and the still-inline domain wrappers keep `self.allocator`.
    allocator: Allocator,

    pub const ScopeNodeDestroy = exec_runtime.ScopeNodeDestroy;
    pub const ExecScope = exec_runtime.ExecScope;

    pub fn init(self: *ExecContext, allocator: Allocator) void {
        self.rt.init(allocator);
        self.allocator = self.rt.allocator;
        self.moe_scratch = .{};
    }

    pub fn deinit(self: *ExecContext) void {
        self.moe_scratch.deinit(self.allocator);
        self.rt.deinit();
        self.* = undefined;
    }

    // ------------------------------------------------------------------
    // Exec scopes: implicit ownership of EXECUTION artifacts — the tensor
    // values ops produce plus a type-erased per-op payload released through
    // a registered destructor. Exec deliberately knows nothing about
    // autograd types; the ag facade stores its backward nodes in that
    // payload. The user scopes the execution; that, in turn, is what
    // enables autograd on top. (Hence "exec scope", not "graph" anything:
    // there is no graph object in this runtime.)
    //
    // While a scope is open, every tensor RETURNED BY A FACADE OP is owned
    // by the innermost scope (the ag facade adopts it here); the value the
    // caller receives is a borrow — never deinit it, never use it after the
    // scope closes. Tensors created explicitly (variable/constant/fromSlice)
    // and fetched gradients (grad/gradView) stay caller-owned. This is what
    // makes training forward passes look like inference code: intermediates
    // between the parameters and the loss must outlive backward() because
    // GradStates are single-owner (see docs/TRAINING.md), and the scope holds
    // them so the user doesn't have to.
    //
    // Scopes nest with strict stack discipline (close in reverse order) and
    // are not thread-safe — open/close/ops on one ctx from one thread, like
    // every other ctx mutation.
    // ------------------------------------------------------------------

    pub fn execScopeActive(self: *const ExecContext) bool {
        return self.rt.execScopeActive();
    }

    /// Open a scope; close it with `closeExecScope(mark)` (typically `defer`).
    pub fn openExecScope(self: *ExecContext) ExecScope {
        return self.rt.openExecScope();
    }

    /// Release every tensor adopted since `mark`, newest first. Only safe
    /// once no backward() over tensors adopted in the scope is pending.
    pub fn closeExecScope(self: *ExecContext, mark: ExecScope) void {
        self.rt.closeExecScope(mark);
    }

    /// Two-phase adoption so op construction can stay infallible after its
    /// "consumes the value on success" point: reserve BEFORE building the
    /// result, adopt (cannot fail) after.
    pub fn reserveScopeSlot(self: *ExecContext) !void {
        return self.rt.reserveScopeSlot();
    }

    pub fn adoptScopeValueAssumeCapacity(self: *ExecContext, value: Tensor, node: ?*anyopaque, destroy_node: ScopeNodeDestroy) void {
        self.rt.adoptScopeValueAssumeCapacity(value, node, destroy_node);
    }

    pub fn adoptScopeNodeAssumeCapacity(self: *ExecContext, node: *anyopaque, destroy_node: ScopeNodeDestroy) void {
        self.rt.adoptScopeNodeAssumeCapacity(node, destroy_node);
    }

    pub fn tryWorkPool(self: *ExecContext) !*thread.Pool {
        return self.rt.tryWorkPool();
    }

    pub fn workPool(self: *ExecContext) ?*thread.Pool {
        return self.rt.workPool();
    }

    pub fn dotBackwardWorker(self: *ExecContext) ?*thread.OneShotWorker {
        return self.rt.dotBackwardWorker();
    }

    pub fn classify(_: *const ExecContext, x: *const Tensor) LayoutClass {
        if (x.isScalar()) return .scalar;
        if (x.isContiguous()) return .contiguous;
        if (tailBroadcastInfo(x) != null) return .tail_broadcast;
        return .arbitrary;
    }

    pub fn broadcastTo(self: *ExecContext, x: *const Tensor, shape: []const usize) !Tensor {
        _ = self;
        return x.broadcastTo(shape);
    }

    pub fn broadcastToRank(self: *ExecContext, comptime rank: usize, x: *const Tensor, shape: [rank]usize) !Tensor {
        _ = self;
        return x.broadcastToRank(rank, shape);
    }

    pub fn empty(self: *ExecContext, shape: []const usize) !Tensor {
        return self.rt.empty(shape);
    }

    pub fn emptyRank(self: *ExecContext, comptime rank: usize, shape: [rank]usize) !Tensor {
        return self.rt.emptyRank(rank, shape);
    }

    fn emptyTyped(self: *ExecContext, comptime dtype: DType, shape: []const usize) !tensor.TensorOf(dtype) {
        return self.rt.emptyTyped(dtype, shape);
    }

    pub fn emptyRankTyped(self: *ExecContext, comptime dtype: DType, comptime rank: usize, shape: [rank]usize) !tensor.TensorOf(dtype) {
        return self.rt.emptyRankTyped(dtype, rank, shape);
    }

    pub fn zeros(self: *ExecContext, shape: []const usize) !Tensor {
        return self.rt.zeros(shape);
    }

    pub fn zerosTyped(self: *ExecContext, comptime dtype: DType, shape: []const usize) !tensor.TensorOf(dtype) {
        return self.rt.zerosTyped(dtype, shape);
    }

    fn zerosRankTyped(self: *ExecContext, comptime dtype: DType, comptime rank: usize, shape: [rank]usize) !tensor.TensorOf(dtype) {
        return self.rt.zerosRankTyped(dtype, rank, shape);
    }

    pub fn ones(self: *ExecContext, shape: []const usize) !Tensor {
        return self.rt.ones(shape);
    }

    pub fn onesRank(self: *ExecContext, comptime rank: usize, shape: [rank]usize) !Tensor {
        return self.rt.onesRank(rank, shape);
    }

    pub fn onesTyped(self: *ExecContext, comptime dtype: DType, shape: []const usize) !tensor.TensorOf(dtype) {
        return self.rt.onesTyped(dtype, shape);
    }

    pub fn onesRankTyped(self: *ExecContext, comptime dtype: DType, comptime rank: usize, shape: [rank]usize) !tensor.TensorOf(dtype) {
        return self.rt.onesRankTyped(dtype, rank, shape);
    }

    pub fn full(self: *ExecContext, shape: []const usize, value: f32) !Tensor {
        return self.rt.full(shape, value);
    }

    pub fn fullTyped(self: *ExecContext, comptime dtype: DType, shape: []const usize, value: dtype_mod.Scalar(dtype)) !tensor.TensorOf(dtype) {
        return self.rt.fullTyped(dtype, shape, value);
    }

    pub fn scalar(self: *ExecContext, value: f32) !Tensor {
        return self.rt.scalar(value);
    }

    pub fn fromSlice(self: *ExecContext, shape: []const usize, values: []const f32) !Tensor {
        return self.rt.fromSlice(shape, values);
    }

    pub fn fromSliceRank(self: *ExecContext, comptime rank: usize, shape: [rank]usize, values: []const f32) !Tensor {
        return self.rt.fromSliceRank(rank, shape, values);
    }

    pub fn fromBorrowedSliceRank(self: *ExecContext, comptime rank: usize, shape: [rank]usize, values: []f32) !Tensor {
        return self.rt.fromBorrowedSliceRank(rank, shape, values);
    }

    pub fn fromSliceTyped(self: *ExecContext, comptime dtype: DType, shape: []const usize, values: []const dtype_mod.Scalar(dtype)) !tensor.TensorOf(dtype) {
        return self.rt.fromSliceTyped(dtype, shape, values);
    }

    pub fn fromSliceRankTyped(
        self: *ExecContext,
        comptime dtype: DType,
        comptime rank: usize,
        shape: [rank]usize,
        values: []const dtype_mod.Scalar(dtype),
    ) !tensor.TensorOf(dtype) {
        return self.rt.fromSliceRankTyped(dtype, rank, shape, values);
    }

    pub fn fromBorrowedSliceRankTyped(
        self: *ExecContext,
        comptime dtype: DType,
        comptime rank: usize,
        shape: [rank]usize,
        values: []dtype_mod.Scalar(dtype),
    ) !tensor.TensorOf(dtype) {
        return self.rt.fromBorrowedSliceRankTyped(dtype, rank, shape, values);
    }

    pub fn fromStorageSliceRankTyped(
        self: *ExecContext,
        comptime dtype: DType,
        comptime rank: usize,
        shape: [rank]usize,
        values: []const dtype_mod.Storage(dtype),
    ) !tensor.TensorOf(dtype) {
        return self.rt.fromStorageSliceRankTyped(dtype, rank, shape, values);
    }

    pub fn fromBorrowedStorageSliceRankTyped(
        self: *ExecContext,
        comptime dtype: DType,
        comptime rank: usize,
        shape: [rank]usize,
        values: []dtype_mod.Storage(dtype),
    ) !tensor.TensorOf(dtype) {
        return self.rt.fromBorrowedStorageSliceRankTyped(dtype, rank, shape, values);
    }

    // Lifetime helper for the "carried value" pattern: deinitializes `old` and
    // returns the freshly computed `new_value`, advancing an accumulator (e.g. a
    // transformer residual stream) in one statement instead of the
    // create/deinit/reassign dance:
    //
    //     x = try ctx.replace(x, x.add(ctx, &delta));
    //
    // `new_value` is evaluated by the caller before `replace` runs; if it is an
    // error, `old` is left untouched and the error propagates, so the caller's
    // `errdefer old.deinit()` still frees it exactly once. On success `old` is
    // released (one ref) and the new value returned. Generic over any owned
    // value with a `deinit` method (tagged tensors, projection structs, ...).
    /// Swap a carried tensor for the result of a block call, e.g.
    /// `x = try ctx.replace(x, attentionBlock(ctx, ..., &x, ...));`.
    /// `new_value` is an error union on purpose: on error the old tensor is
    /// NOT consumed (the caller's binding and defers stay valid) and the
    /// error propagates; on success the old tensor is released and the new
    /// one returned for rebinding. Inside an exec scope the release is a
    /// safe no-op for scope-owned op results (their `deinit` is a no-op —
    /// the scope owns them), so the same forward code is also training-safe.
    pub fn replace(self: *ExecContext, old: anytype, new_value: anytype) @TypeOf(new_value) {
        _ = self;
        comptime {
            const ret_info = @typeInfo(@TypeOf(new_value));
            if (ret_info != .error_union or ret_info.error_union.payload != @TypeOf(old)) {
                @compileError("ctx.replace expects new_value of type E!" ++ @typeName(@TypeOf(old)));
            }
        }
        const value = try new_value;
        var owned = old;
        owned.deinit();
        return value;
    }

    pub fn materialize(self: *ExecContext, x: *const Tensor) !Tensor {
        return self.rt.materialize(x);
    }

    pub fn materializeTyped(self: *ExecContext, comptime dtype: DType, x: *const tensor.TensorOf(dtype)) !tensor.TensorOf(dtype) {
        return self.rt.materializeTyped(dtype, x);
    }

    pub fn clone(self: *ExecContext, x: *const Tensor) !Tensor {
        return self.rt.clone(x);
    }

    fn cloneTyped(self: *ExecContext, comptime dtype: DType, x: *const tensor.TensorOf(dtype)) !tensor.TensorOf(dtype) {
        return self.rt.cloneTyped(dtype, x);
    }

    pub fn add(self: *ExecContext, a: *const Tensor, b: *const Tensor) !Tensor {
        return exec_elementwise.add(&self.rt, a, b);
    }

    pub fn sub(self: *ExecContext, a: *const Tensor, b: *const Tensor) !Tensor {
        return exec_elementwise.sub(&self.rt, a, b);
    }

    pub fn mul(self: *ExecContext, a: *const Tensor, b: *const Tensor) !Tensor {
        return exec_elementwise.mul(&self.rt, a, b);
    }

    pub fn div(self: *ExecContext, a: *const Tensor, b: *const Tensor) !Tensor {
        return exec_elementwise.div(&self.rt, a, b);
    }

    pub fn addRank(self: *ExecContext, comptime rank: usize, a: *const Tensor, b: *const Tensor) !Tensor {
        return exec_elementwise.addRank(&self.rt, rank, a, b);
    }

    pub fn addRankTyped(
        self: *ExecContext,
        comptime dtype: DType,
        comptime rank: usize,
        a: *const tensor.TensorOf(dtype),
        b: *const tensor.TensorOf(dtype),
    ) !tensor.TensorOf(dtype_mod.outputDType(.pointwise, dtype)) {
        return exec_elementwise.addRankTyped(&self.rt, dtype, rank, a, b);
    }

    pub fn subRank(self: *ExecContext, comptime rank: usize, a: *const Tensor, b: *const Tensor) !Tensor {
        return exec_elementwise.subRank(&self.rt, rank, a, b);
    }

    pub fn subRankTyped(
        self: *ExecContext,
        comptime dtype: DType,
        comptime rank: usize,
        a: *const tensor.TensorOf(dtype),
        b: *const tensor.TensorOf(dtype),
    ) !tensor.TensorOf(dtype_mod.outputDType(.pointwise, dtype)) {
        return exec_elementwise.subRankTyped(&self.rt, dtype, rank, a, b);
    }

    pub fn mulRank(self: *ExecContext, comptime rank: usize, a: *const Tensor, b: *const Tensor) !Tensor {
        return exec_elementwise.mulRank(&self.rt, rank, a, b);
    }

    pub fn mulRankTyped(
        self: *ExecContext,
        comptime dtype: DType,
        comptime rank: usize,
        a: *const tensor.TensorOf(dtype),
        b: *const tensor.TensorOf(dtype),
    ) !tensor.TensorOf(dtype_mod.outputDType(.pointwise, dtype)) {
        return exec_elementwise.mulRankTyped(&self.rt, dtype, rank, a, b);
    }

    pub fn divRank(self: *ExecContext, comptime rank: usize, a: *const Tensor, b: *const Tensor) !Tensor {
        return exec_elementwise.divRank(&self.rt, rank, a, b);
    }

    pub fn divRankTyped(
        self: *ExecContext,
        comptime dtype: DType,
        comptime rank: usize,
        a: *const tensor.TensorOf(dtype),
        b: *const tensor.TensorOf(dtype),
    ) !tensor.TensorOf(dtype_mod.outputDType(.pointwise, dtype)) {
        return exec_elementwise.divRankTyped(&self.rt, dtype, rank, a, b);
    }

    pub fn maxRank(self: *ExecContext, comptime rank: usize, a: *const Tensor, b: *const Tensor) !Tensor {
        return exec_elementwise.maxRank(&self.rt, rank, a, b);
    }

    pub fn minRank(self: *ExecContext, comptime rank: usize, a: *const Tensor, b: *const Tensor) !Tensor {
        return exec_elementwise.minRank(&self.rt, rank, a, b);
    }

    pub fn gatedRank(self: *ExecContext, comptime rank: usize, comptime op: GatedOp, a: *const Tensor, b: *const Tensor) !Tensor {
        return exec_elementwise.gatedRank(&self.rt, rank, op, a, b);
    }

    pub fn gluRank(self: *ExecContext, comptime rank: usize, a: *const Tensor, b: *const Tensor) !Tensor {
        return exec_elementwise.gluRank(&self.rt, rank, a, b);
    }

    pub fn swigluRank(self: *ExecContext, comptime rank: usize, a: *const Tensor, b: *const Tensor) !Tensor {
        return exec_elementwise.swigluRank(&self.rt, rank, a, b);
    }

    pub fn gegluRank(self: *ExecContext, comptime rank: usize, a: *const Tensor, b: *const Tensor) !Tensor {
        return exec_elementwise.gegluRank(&self.rt, rank, a, b);
    }

    pub fn splitSwiGluAxisRank(self: *ExecContext, comptime rank: usize, x: *const Tensor, comptime axis: usize) !Tensor {
        return exec_elementwise.splitSwiGluAxisRank(&self.rt, rank, x, axis);
    }

    /// Transformer-XL relative-shift ("skew"): a rank-3
    /// relative-score tensor `bd[H,Tq,P]` → `out[H,Tq,Tk]` with
    /// `out[h,qi,kj] = bd[h, qi, kj + (Tq-1) - qi]` — the closed form of the NeMo
    /// relpos pad/reshape/view remap (and parakeet.cpp's skew). `P` must be
    /// `>= Tk + Tq - 1`. Differentiable via `RelposShiftBackward` (scatter VJP).
    pub fn relposShiftRank3(self: *ExecContext, bd: *const Tensor, t_k: usize) !Tensor {
        return exec_gather_scatter.relposShiftRank3(&self.rt, bd, t_k);
    }

    pub fn splitGluAxisRank(self: *ExecContext, comptime rank: usize, x: *const Tensor, comptime axis: usize) !Tensor {
        return exec_elementwise.splitGluAxisRank(&self.rt, rank, x, axis);
    }

    pub fn splitSwiGluBackwardAxisRank(self: *ExecContext, comptime rank: usize, x: *const Tensor, gy: *const Tensor, comptime axis: usize) !Tensor {
        return exec_elementwise.splitSwiGluBackwardAxisRank(&self.rt, rank, x, gy, axis);
    }

    pub fn splitGluBackwardAxisRank(self: *ExecContext, comptime rank: usize, x: *const Tensor, gy: *const Tensor, comptime axis: usize) !Tensor {
        return exec_elementwise.splitGluBackwardAxisRank(&self.rt, rank, x, gy, axis);
    }

    pub fn addInPlace(self: *ExecContext, target: *Tensor, other: *const Tensor) !void {
        return exec_elementwise.addInPlace(&self.rt, target, other);
    }

    pub fn subInPlace(self: *ExecContext, target: *Tensor, other: *const Tensor) !void {
        return exec_elementwise.subInPlace(&self.rt, target, other);
    }

    pub fn mulInPlace(self: *ExecContext, target: *Tensor, other: *const Tensor) !void {
        return exec_elementwise.mulInPlace(&self.rt, target, other);
    }

    pub fn divInPlace(self: *ExecContext, target: *Tensor, other: *const Tensor) !void {
        return exec_elementwise.divInPlace(&self.rt, target, other);
    }

    pub fn takeAdd(self: *ExecContext, target: *Tensor, other: *const Tensor) !Tensor {
        return exec_elementwise.takeAdd(&self.rt, target, other);
    }

    pub fn takeSub(self: *ExecContext, target: *Tensor, other: *const Tensor) !Tensor {
        return exec_elementwise.takeSub(&self.rt, target, other);
    }

    pub fn takeMul(self: *ExecContext, target: *Tensor, other: *const Tensor) !Tensor {
        return exec_elementwise.takeMul(&self.rt, target, other);
    }

    pub fn takeDiv(self: *ExecContext, target: *Tensor, other: *const Tensor) !Tensor {
        return exec_elementwise.takeDiv(&self.rt, target, other);
    }

    pub fn takeScale(self: *ExecContext, target: *Tensor, scalar_value: f32) !Tensor {
        return exec_elementwise.takeScale(&self.rt, target, scalar_value);
    }

    pub fn takeRelu(self: *ExecContext, target: *Tensor) !Tensor {
        return exec_elementwise.takeRelu(&self.rt, target);
    }

    pub fn takeSilu(self: *ExecContext, target: *Tensor) !Tensor {
        return exec_elementwise.takeSilu(&self.rt, target);
    }

    pub fn scale(self: *ExecContext, x: *const Tensor, scalar_value: f32) !Tensor {
        return exec_elementwise.scale(&self.rt, x, scalar_value);
    }

    pub fn addScalar(self: *ExecContext, x: *const Tensor, scalar_value: f32) !Tensor {
        return exec_elementwise.addScalar(&self.rt, x, scalar_value);
    }

    pub fn powScalar(self: *ExecContext, x: *const Tensor, exponent: f32) !Tensor {
        return exec_elementwise.powScalar(&self.rt, x, exponent);
    }

    pub fn where(self: *ExecContext, x: *const Tensor, cond: *const Tensor, y: *const Tensor) !Tensor {
        return exec_elementwise.where(&self.rt, x, cond, y);
    }

    pub fn maskedFill(self: *ExecContext, x: *const Tensor, mask: *const Tensor, value: f32) !Tensor {
        return exec_elementwise.maskedFill(&self.rt, x, mask, value);
    }

    pub fn compare(self: *ExecContext, comptime op: CompareOp, a: *const Tensor, b: *const Tensor) !Tensor {
        return exec_elementwise.compare(&self.rt, op, a, b);
    }

    pub fn compareScalar(self: *ExecContext, comptime op: CompareOp, x: *const Tensor, scalar_value: f32) !Tensor {
        return exec_elementwise.compareScalar(&self.rt, op, x, scalar_value);
    }

    pub fn logicalAnd(self: *ExecContext, a: *const Tensor, b: *const Tensor) !Tensor {
        return exec_elementwise.logicalAnd(&self.rt, a, b);
    }

    pub fn logicalOr(self: *ExecContext, a: *const Tensor, b: *const Tensor) !Tensor {
        return exec_elementwise.logicalOr(&self.rt, a, b);
    }

    pub fn logicalXor(self: *ExecContext, a: *const Tensor, b: *const Tensor) !Tensor {
        return exec_elementwise.logicalXor(&self.rt, a, b);
    }

    pub fn logicalNot(self: *ExecContext, x: *const Tensor) !Tensor {
        return exec_elementwise.logicalNot(&self.rt, x);
    }

    pub fn addScaledInPlace(self: *ExecContext, target: *Tensor, source: *const Tensor, scalar_value: f32) !void {
        return exec_elementwise.addScaledInPlace(&self.rt, target, source, scalar_value);
    }

    pub fn addAxisVectorInPlaceRank(self: *ExecContext, comptime rank: usize, target: *Tensor, row_vector: []const f32, comptime axis: usize) !void {
        return exec_elementwise.addAxisVectorInPlaceRank(&self.rt, rank, target, row_vector, axis);
    }

    pub fn addAxisVectorUnaryInPlaceRank(self: *ExecContext, comptime rank: usize, comptime op: ?UnaryOp, target: *Tensor, row_vector: []const f32, comptime axis: usize) !void {
        return exec_elementwise.addAxisVectorUnaryInPlaceRank(&self.rt, rank, op, target, row_vector, axis);
    }

    pub fn dropoutForward(self: *ExecContext, x: *const Tensor, p: f32, seed: u64) !Tensor {
        return exec_elementwise.dropoutForward(&self.rt, x, p, seed);
    }

    pub fn dropoutBackward(self: *ExecContext, gy: *const Tensor, p: f32, seed: u64) !Tensor {
        return exec_elementwise.dropoutBackward(&self.rt, gy, p, seed);
    }

    pub fn castTyped(
        self: *ExecContext,
        comptime source_dtype: DType,
        comptime target_dtype: DType,
        x: *const tensor.TensorOf(source_dtype),
    ) !tensor.TensorOf(target_dtype) {
        return exec_convert.castTyped(&self.rt, source_dtype, target_dtype, x);
    }

    /// Cast an f32 tensor into a caller-owned f16 slice in logical row-major
    /// order without allocating: the KV-cache append path. Supports contiguous
    /// sources (one SIMD pass) and rank-3 views whose two inner axes are
    /// contiguous (a `{seq, kv_head, d}` split of a fused QKV row), walked as
    /// per-row spans. Anything else is UnsupportedView — extend deliberately
    /// rather than silently gathering.
    pub fn castF32RowsToF16Into(self: *ExecContext, x: *const tensor.Tensor, dst: []f16) !void {
        return exec_convert.castF32RowsToF16Into(&self.rt, x, dst);
    }

    pub fn quantizeF32RowsToQ8_0Into(self: *ExecContext, x: *const tensor.Tensor, dst: []BlockQ8_0) !void {
        return exec_convert.quantizeF32RowsToQ8_0Into(&self.rt, x, dst);
    }

    pub fn dequantizeQ8_0RowsInto(self: *ExecContext, dst: []f32, blocks: []const BlockQ8_0) !void {
        return exec_convert.dequantizeQ8_0RowsInto(&self.rt, dst, blocks);
    }

    pub fn scaleTyped(
        self: *ExecContext,
        comptime dtype: DType,
        x: *const tensor.TensorOf(dtype),
        scalar_value: dtype_mod.Accumulator(dtype),
    ) !tensor.TensorOf(dtype_mod.outputDType(.pointwise, dtype)) {
        return exec_convert.scaleTyped(&self.rt, dtype, x, scalar_value);
    }

    pub fn causalDepthwiseConv1dAxisRank(
        self: *ExecContext,
        comptime rank: usize,
        input: *const Tensor,
        kernel: *const Tensor,
        comptime time_axis: usize,
        comptime channel_axis: usize,
        state: ?[]const f32,
    ) !Tensor {
        return exec_conv.causalDepthwiseConv1dAxisRank(&self.rt, rank, input, kernel, time_axis, channel_axis, state);
    }

    pub fn causalDepthwiseConv1dBackwardInputAxisRank(
        self: *ExecContext,
        comptime rank: usize,
        gy: *const Tensor,
        kernel: *const Tensor,
        comptime time_axis: usize,
        comptime channel_axis: usize,
    ) !Tensor {
        return exec_conv.causalDepthwiseConv1dBackwardInputAxisRank(&self.rt, rank, gy, kernel, time_axis, channel_axis);
    }

    pub fn causalDepthwiseConv1dBackwardKernelAxisRank(
        self: *ExecContext,
        comptime rank: usize,
        input: *const Tensor,
        gy: *const Tensor,
        comptime time_axis: usize,
        comptime channel_axis: usize,
        taps: usize,
        state: ?[]const f32,
    ) !Tensor {
        return exec_conv.causalDepthwiseConv1dBackwardKernelAxisRank(&self.rt, rank, input, gy, time_axis, channel_axis, taps, state);
    }

    pub fn causalConv1dAxisRank(
        self: *ExecContext,
        comptime rank: usize,
        input: *const Tensor,
        weight: *const Tensor,
        comptime time_axis: usize,
        comptime channel_axis: usize,
        dilation: usize,
        state: ?[]const f32,
    ) !Tensor {
        return exec_conv.causalConv1dAxisRank(&self.rt, rank, input, weight, time_axis, channel_axis, dilation, state);
    }

    pub fn conv2d(
        self: *ExecContext,
        input: *const Tensor,
        weight: *const Tensor,
        bias: ?*const Tensor,
        stride: [2]usize,
        pad: [2]usize,
        groups: usize,
    ) !Tensor {
        return exec_conv.conv2d(&self.rt, input, weight, bias, stride, pad, groups);
    }

    /// conv2d with the relu fused into the epilogue (identical values to
    /// conv2d followed by relu; zero extra passes on the Winograd route).
    /// Inference-path op — the facade guards the differentiable composition.
    pub fn conv2dRelu(
        self: *ExecContext,
        input: *const Tensor,
        weight: *const Tensor,
        bias: ?*const Tensor,
        stride: [2]usize,
        pad: [2]usize,
        groups: usize,
    ) !Tensor {
        return exec_conv.conv2dExt(&self.rt, input, weight, bias, stride, pad, groups, true);
    }

    /// Load-time prepared Winograd conv weight planes; `.empty` is valid and
    /// inert on every conv route. See `exec/conv.zig PreparedConvWeights`.
    pub const PreparedConvWeights = exec_conv.PreparedConvWeights;

    /// Build the Winograd weight-transform planes for a conv2d weight
    /// (`[Cout, kH, kW, Cin/groups]`) once, at load time; `.empty` when the
    /// weight can never take the Winograd route. See
    /// `exec/conv.zig prepareConv2dWeights`.
    pub fn prepareConv2dWeights(self: *ExecContext, weight: *const Tensor) !PreparedConvWeights {
        return exec_conv.prepareConv2dWeights(&self.rt, weight);
    }

    /// conv2d against load-time prepared Winograd weight planes: bitwise
    /// identical values to `conv2d`, minus the per-call weight transform on
    /// the Winograd route. Every other route ignores `prepared`.
    pub fn conv2dPrepared(
        self: *ExecContext,
        input: *const Tensor,
        weight: *const Tensor,
        prepared: *const PreparedConvWeights,
        bias: ?*const Tensor,
        stride: [2]usize,
        pad: [2]usize,
        groups: usize,
    ) !Tensor {
        return exec_conv.conv2dPreparedExt(&self.rt, input, weight, prepared, bias, stride, pad, groups, false);
    }

    /// `conv2dPrepared` with the relu fused into the epilogue (identical
    /// values to `conv2dPrepared` followed by relu; see `conv2dRelu`).
    pub fn conv2dPreparedRelu(
        self: *ExecContext,
        input: *const Tensor,
        weight: *const Tensor,
        prepared: *const PreparedConvWeights,
        bias: ?*const Tensor,
        stride: [2]usize,
        pad: [2]usize,
        groups: usize,
    ) !Tensor {
        return exec_conv.conv2dPreparedExt(&self.rt, input, weight, prepared, bias, stride, pad, groups, true);
    }

    pub fn conv2dBackwardInput(self: *ExecContext, gy: *const Tensor, weight: *const Tensor, in_h: usize, in_w: usize, stride: [2]usize, pad: [2]usize, groups: usize) !Tensor {
        return exec_conv.conv2dBackwardInput(&self.rt, gy, weight, in_h, in_w, stride, pad, groups);
    }

    pub fn conv2dBackwardWeight(self: *ExecContext, input: *const Tensor, gy: *const Tensor, kh: usize, kw: usize, stride: [2]usize, pad: [2]usize, groups: usize) !Tensor {
        return exec_conv.conv2dBackwardWeight(&self.rt, input, gy, kh, kw, stride, pad, groups);
    }

    /// 2-D max pool, channel-last rank-3 `[H,W,C]` → `[OH,OW,C]` (zero-pad
    /// border reads as −inf). See `exec/pool.zig`.
    pub fn maxPool2d(self: *ExecContext, input: *const Tensor, kernel: [2]usize, stride: [2]usize, pad: [2]usize) !Tensor {
        return exec_pool.pool2d(&self.rt, .max, input, kernel, stride, pad);
    }

    /// 2-D average pool, channel-last rank-3; averages the valid taps only
    /// (ONNX `count_include_pad=0`). See `exec/pool.zig`.
    pub fn avgPool2d(self: *ExecContext, input: *const Tensor, kernel: [2]usize, stride: [2]usize, pad: [2]usize) !Tensor {
        return exec_pool.pool2d(&self.rt, .avg, input, kernel, stride, pad);
    }

    pub fn maxPool2dBackward(self: *ExecContext, input: *const Tensor, gy: *const Tensor, kernel: [2]usize, stride: [2]usize, pad: [2]usize) !Tensor {
        return exec_pool.maxPool2dBackward(&self.rt, input, gy, kernel, stride, pad);
    }

    pub fn avgPool2dBackward(self: *ExecContext, gy: *const Tensor, in_h: usize, in_w: usize, kernel: [2]usize, stride: [2]usize, pad: [2]usize) !Tensor {
        return exec_pool.avgPool2dBackward(&self.rt, gy, in_h, in_w, kernel, stride, pad);
    }

    /// 2× nearest-neighbour upsample, channel-last rank-3 `[H,W,C]` → `[2H,2W,C]`.
    pub fn upsample2xNearest(self: *ExecContext, input: *const Tensor) !Tensor {
        return exec_pool.upsample2xNearest(&self.rt, input);
    }

    pub fn upsample2xNearestBackward(self: *ExecContext, gy: *const Tensor) !Tensor {
        return exec_pool.upsample2xNearestBackward(&self.rt, gy);
    }

    /// Per-channel PReLU (channel axis innermost): `y = x > 0 ? x : α[c]·x`.
    pub fn preluChannels(self: *ExecContext, x: *const Tensor, alpha: *const Tensor) !Tensor {
        return exec_elementwise.preluChannels(&self.rt, x, alpha);
    }

    pub fn preluChannelsBackwardInput(self: *ExecContext, gy: *const Tensor, x: *const Tensor, alpha: *const Tensor) !Tensor {
        return exec_elementwise.preluChannelsBackwardInput(&self.rt, gy, x, alpha);
    }

    pub fn preluChannelsBackwardAlpha(self: *ExecContext, gy: *const Tensor, x: *const Tensor, channels: usize) !Tensor {
        return exec_elementwise.preluChannelsBackwardAlpha(&self.rt, gy, x, channels);
    }

    /// Per-channel affine (frozen-stats inference BatchNorm) in one pass:
    /// `y = x·scale[c] + shift[c]`, channel axis innermost; null `shift_vec`
    /// degrades to the per-channel scale.
    pub fn channelAffine(self: *ExecContext, x: *const Tensor, scale_vec: *const Tensor, shift_vec: ?*const Tensor) !Tensor {
        return exec_elementwise.channelAffine(&self.rt, x, scale_vec, shift_vec);
    }

    pub fn causalConv1dBackwardInputAxisRank(
        self: *ExecContext,
        comptime rank: usize,
        gy: *const Tensor,
        weight: *const Tensor,
        comptime time_axis: usize,
        comptime channel_axis: usize,
        dilation: usize,
    ) !Tensor {
        return exec_conv.causalConv1dBackwardInputAxisRank(&self.rt, rank, gy, weight, time_axis, channel_axis, dilation);
    }

    pub fn causalConv1dBackwardWeightAxisRank(
        self: *ExecContext,
        comptime rank: usize,
        input: *const Tensor,
        gy: *const Tensor,
        comptime time_axis: usize,
        comptime channel_axis: usize,
        taps: usize,
        dilation: usize,
        state: ?[]const f32,
    ) !Tensor {
        return exec_conv.causalConv1dBackwardWeightAxisRank(&self.rt, rank, input, gy, time_axis, channel_axis, taps, dilation, state);
    }

    pub fn groupedCausalConv1dAxisRank(
        self: *ExecContext,
        comptime rank: usize,
        input: *const Tensor,
        weight: *const Tensor,
        comptime time_axis: usize,
        comptime channel_axis: usize,
        dilation: usize,
        groups: usize,
        state: ?[]const f32,
    ) !Tensor {
        return exec_conv.groupedCausalConv1dAxisRank(&self.rt, rank, input, weight, time_axis, channel_axis, dilation, groups, state);
    }

    pub fn groupedCausalConv1dBackwardInputAxisRank(
        self: *ExecContext,
        comptime rank: usize,
        gy: *const Tensor,
        weight: *const Tensor,
        comptime time_axis: usize,
        comptime channel_axis: usize,
        dilation: usize,
        groups: usize,
    ) !Tensor {
        return exec_conv.groupedCausalConv1dBackwardInputAxisRank(&self.rt, rank, gy, weight, time_axis, channel_axis, dilation, groups);
    }

    pub fn groupedCausalConv1dBackwardWeightAxisRank(
        self: *ExecContext,
        comptime rank: usize,
        input: *const Tensor,
        gy: *const Tensor,
        comptime time_axis: usize,
        comptime channel_axis: usize,
        taps: usize,
        dilation: usize,
        groups: usize,
        state: ?[]const f32,
    ) !Tensor {
        return exec_conv.groupedCausalConv1dBackwardWeightAxisRank(&self.rt, rank, input, gy, time_axis, channel_axis, taps, dilation, groups, state);
    }

    pub fn conv1dAxisRank(
        self: *ExecContext,
        comptime rank: usize,
        input: *const Tensor,
        weight: *const Tensor,
        comptime time_axis: usize,
        comptime channel_axis: usize,
        stride: usize,
        pad: usize,
        dilation: usize,
        groups: usize,
    ) !Tensor {
        return exec_conv.conv1dAxisRank(&self.rt, rank, input, weight, time_axis, channel_axis, stride, pad, dilation, groups);
    }

    pub fn conv1dBackwardInputAxisRank(
        self: *ExecContext,
        comptime rank: usize,
        gy: *const Tensor,
        weight: *const Tensor,
        comptime time_axis: usize,
        comptime channel_axis: usize,
        seq: usize,
        stride: usize,
        pad: usize,
        dilation: usize,
        groups: usize,
    ) !Tensor {
        return exec_conv.conv1dBackwardInputAxisRank(&self.rt, rank, gy, weight, time_axis, channel_axis, seq, stride, pad, dilation, groups);
    }

    pub fn conv1dBackwardWeightAxisRank(
        self: *ExecContext,
        comptime rank: usize,
        input: *const Tensor,
        gy: *const Tensor,
        comptime time_axis: usize,
        comptime channel_axis: usize,
        taps: usize,
        stride: usize,
        pad: usize,
        dilation: usize,
        groups: usize,
    ) !Tensor {
        return exec_conv.conv1dBackwardWeightAxisRank(&self.rt, rank, input, gy, time_axis, channel_axis, taps, stride, pad, dilation, groups);
    }

    pub fn col2im1dAxisRank(
        self: *ExecContext,
        col: *const Tensor,
        out_channels: usize,
        taps: usize,
        stride: usize,
        pad: usize,
        output_pad: usize,
    ) !Tensor {
        return exec_conv.col2im1dAxisRank(&self.rt, col, out_channels, taps, stride, pad, output_pad);
    }

    pub fn col2im1dBackwardAxisRank(
        self: *ExecContext,
        gy: *const Tensor,
        t_in: usize,
        out_channels: usize,
        taps: usize,
        stride: usize,
        pad: usize,
    ) !Tensor {
        return exec_conv.col2im1dBackwardAxisRank(&self.rt, gy, t_in, out_channels, taps, stride, pad);
    }

    pub fn convTranspose1d(
        self: *ExecContext,
        input: *const Tensor,
        weight2: *const Tensor,
        bias: ?*const Tensor,
        out_channels: usize,
        taps: usize,
        stride: usize,
        pad: usize,
        output_pad: usize,
    ) !Tensor {
        return exec_conv.convTranspose1d(&self.rt, input, weight2, bias, out_channels, taps, stride, pad, output_pad);
    }

    pub fn unary(self: *ExecContext, comptime op: UnaryOp, x: *const Tensor) !Tensor {
        return exec_elementwise.unary(&self.rt, op, x);
    }

    pub fn snakeRows(self: *ExecContext, x: *const Tensor, alpha: *const Tensor, inv_b: *const Tensor) !Tensor {
        return exec_elementwise.snakeRows(&self.rt, x, alpha, inv_b);
    }

    pub fn snakeRowsBackwardInput(self: *ExecContext, x: *const Tensor, gy: *const Tensor, alpha: *const Tensor, inv_b: *const Tensor) !Tensor {
        return exec_elementwise.snakeRowsBackwardInput(&self.rt, x, gy, alpha, inv_b);
    }

    pub fn snakeRowsBackwardParams(self: *ExecContext, x: *const Tensor, gy: *const Tensor, alpha: *const Tensor, inv_b: *const Tensor) !SnakeBackwardParamsResult {
        return exec_elementwise.snakeRowsBackwardParams(&self.rt, x, gy, alpha, inv_b);
    }

    pub fn relu(self: *ExecContext, x: *const Tensor) !Tensor {
        return exec_elementwise.relu(&self.rt, x);
    }

    pub fn leakyRelu(self: *ExecContext, x: *const Tensor, negative_slope: f32) !Tensor {
        return exec_elementwise.leakyRelu(&self.rt, x, negative_slope);
    }

    pub fn exp(self: *ExecContext, x: *const Tensor) !Tensor {
        return exec_elementwise.exp(&self.rt, x);
    }

    pub fn sqrt(self: *ExecContext, x: *const Tensor) !Tensor {
        return exec_elementwise.sqrt(&self.rt, x);
    }

    pub fn rsqrt(self: *ExecContext, x: *const Tensor) !Tensor {
        return exec_elementwise.rsqrt(&self.rt, x);
    }

    pub fn sigmoid(self: *ExecContext, x: *const Tensor) !Tensor {
        return exec_elementwise.sigmoid(&self.rt, x);
    }

    pub fn silu(self: *ExecContext, x: *const Tensor) !Tensor {
        return exec_elementwise.silu(&self.rt, x);
    }

    pub fn log(self: *ExecContext, x: *const Tensor) !Tensor {
        return exec_elementwise.log(&self.rt, x);
    }

    pub fn neg(self: *ExecContext, x: *const Tensor) !Tensor {
        return exec_elementwise.neg(&self.rt, x);
    }

    pub fn abs(self: *ExecContext, x: *const Tensor) !Tensor {
        return exec_elementwise.abs(&self.rt, x);
    }

    pub fn sin(self: *ExecContext, x: *const Tensor) !Tensor {
        return exec_elementwise.sin(&self.rt, x);
    }

    pub fn cos(self: *ExecContext, x: *const Tensor) !Tensor {
        return exec_elementwise.cos(&self.rt, x);
    }

    pub fn tanh(self: *ExecContext, x: *const Tensor) !Tensor {
        return exec_elementwise.tanh(&self.rt, x);
    }

    pub fn gelu(self: *ExecContext, x: *const Tensor) !Tensor {
        return exec_elementwise.gelu(&self.rt, x);
    }

    pub fn quickGelu(self: *ExecContext, x: *const Tensor) !Tensor {
        return exec_elementwise.quickGelu(&self.rt, x);
    }

    pub fn clamp(self: *ExecContext, x: *const Tensor, min_value: f32, max_value: f32) !Tensor {
        return exec_elementwise.clamp(&self.rt, x, min_value, max_value);
    }

    pub fn sum(self: *ExecContext, x: *const Tensor) !Tensor {
        return exec_reduce.sum(&self.rt, x);
    }

    pub fn sumTyped(self: *ExecContext, comptime dtype: DType, x: *const tensor.TensorOf(dtype)) !tensor.TensorOf(dtype_mod.outputDType(.reduction, dtype)) {
        return exec_reduce.sumTyped(&self.rt, dtype, x);
    }

    pub fn sumAxisRank(self: *ExecContext, comptime rank: usize, x: *const Tensor, comptime axis: usize) !Tensor {
        return exec_reduce.sumAxisRank(&self.rt, rank, x, axis);
    }

    pub fn sumAxisRankTyped(
        self: *ExecContext,
        comptime dtype: DType,
        comptime rank: usize,
        x: *const tensor.TensorOf(dtype),
        comptime axis: usize,
    ) !tensor.TensorOf(dtype_mod.outputDType(.reduction, dtype)) {
        return exec_reduce.sumAxisRankTyped(&self.rt, dtype, rank, x, axis);
    }

    pub fn cumsumAxisRank(self: *ExecContext, comptime rank: usize, x: *const Tensor, comptime axis: usize) !Tensor {
        return exec_reduce.cumsumAxisRank(&self.rt, rank, x, axis);
    }

    pub fn prodAxisRank(self: *ExecContext, comptime rank: usize, x: *const Tensor, comptime axis: usize) !Tensor {
        return exec_reduce.prodAxisRank(&self.rt, rank, x, axis);
    }

    pub fn cumprodAxisRank(self: *ExecContext, comptime rank: usize, x: *const Tensor, comptime axis: usize) !Tensor {
        return exec_reduce.cumprodAxisRank(&self.rt, rank, x, axis);
    }

    pub fn cumsumReverseAxisRank(self: *ExecContext, comptime rank: usize, x: *const Tensor, comptime axis: usize) !Tensor {
        return exec_reduce.cumsumReverseAxisRank(&self.rt, rank, x, axis);
    }

    pub fn meanAxisRank(self: *ExecContext, comptime rank: usize, x: *const Tensor, comptime axis: usize) !Tensor {
        return exec_reduce.meanAxisRank(&self.rt, rank, x, axis);
    }

    pub fn meanAxisRankTyped(
        self: *ExecContext,
        comptime dtype: DType,
        comptime rank: usize,
        x: *const tensor.TensorOf(dtype),
        comptime axis: usize,
    ) !tensor.TensorOf(dtype_mod.outputDType(.reduction, dtype)) {
        return exec_reduce.meanAxisRankTyped(&self.rt, dtype, rank, x, axis);
    }

    pub fn narrowAxisRank(self: *ExecContext, comptime rank: usize, x: *const Tensor, comptime axis: usize, start: usize, length: usize) !Tensor {
        return exec_gather_scatter.narrowAxisRank(&self.rt, rank, x, axis, start, length);
    }

    pub fn narrowAxisRankTyped(
        self: *ExecContext,
        comptime dtype: DType,
        comptime rank: usize,
        x: *const tensor.TensorOf(dtype),
        comptime axis: usize,
        start: usize,
        length: usize,
    ) !tensor.TensorOf(dtype) {
        return exec_gather_scatter.narrowAxisRankTyped(&self.rt, dtype, rank, x, axis, start, length);
    }

    pub fn concatAxisRank(self: *ExecContext, comptime rank: usize, inputs: []const *const Tensor, comptime axis: usize) !Tensor {
        return exec_gather_scatter.concatAxisRank(&self.rt, rank, inputs, axis);
    }

    pub fn concatAxisRankTyped(
        self: *ExecContext,
        comptime dtype: DType,
        comptime rank: usize,
        inputs: []const *const tensor.TensorOf(dtype),
        comptime axis: usize,
    ) !tensor.TensorOf(dtype) {
        return exec_gather_scatter.concatAxisRankTyped(&self.rt, dtype, rank, inputs, axis);
    }

    pub fn concatQuantizedRowsTyped(
        self: *ExecContext,
        comptime dtype: DType,
        inputs: []const *const tensor.TensorOf(dtype),
    ) !tensor.TensorOf(dtype) {
        return exec_gather_scatter.concatQuantizedRowsTyped(&self.rt, dtype, inputs);
    }

    pub fn padAxisRank(self: *ExecContext, comptime rank: usize, x: *const Tensor, comptime axis: usize, before: usize, after: usize, fill: f32) !Tensor {
        return exec_gather_scatter.padAxisRank(&self.rt, rank, x, axis, before, after, fill);
    }

    pub fn gatherAxisRank(self: *ExecContext, comptime rank: usize, x: *const Tensor, comptime axis: usize, indices: []const usize) !Tensor {
        return exec_gather_scatter.gatherAxisRank(&self.rt, rank, x, axis, indices);
    }

    pub fn gatherAxisRankTyped(
        self: *ExecContext,
        comptime dtype: DType,
        comptime rank: usize,
        x: *const tensor.TensorOf(dtype),
        comptime axis: usize,
        indices: []const usize,
    ) !tensor.TensorOf(dtype) {
        return exec_gather_scatter.gatherAxisRankTyped(&self.rt, dtype, rank, x, axis, indices);
    }

    pub fn setSliceAxisRank(self: *ExecContext, comptime rank: usize, base: *const Tensor, update: *const Tensor, comptime axis: usize, start: usize) !Tensor {
        return exec_gather_scatter.setSliceAxisRank(&self.rt, rank, base, update, axis, start);
    }

    pub fn setSliceAxisRankTyped(
        self: *ExecContext,
        comptime dtype: DType,
        comptime rank: usize,
        base: *const tensor.TensorOf(dtype),
        update: *const tensor.TensorOf(dtype),
        comptime axis: usize,
        start: usize,
    ) !tensor.TensorOf(dtype) {
        return exec_gather_scatter.setSliceAxisRankTyped(&self.rt, dtype, rank, base, update, axis, start);
    }

    pub fn setRowsAxisRank(self: *ExecContext, comptime rank: usize, base: *const Tensor, update: *const Tensor, comptime axis: usize, indices: []const usize) !Tensor {
        return exec_gather_scatter.setRowsAxisRank(&self.rt, rank, base, update, axis, indices);
    }

    pub fn setRowsAxisRankTyped(
        self: *ExecContext,
        comptime dtype: DType,
        comptime rank: usize,
        base: *const tensor.TensorOf(dtype),
        update: *const tensor.TensorOf(dtype),
        comptime axis: usize,
        indices: []const usize,
    ) !tensor.TensorOf(dtype) {
        return exec_gather_scatter.setRowsAxisRankTyped(&self.rt, dtype, rank, base, update, axis, indices);
    }

    pub fn zeroSliceAxisRank(self: *ExecContext, comptime rank: usize, x: *const Tensor, comptime axis: usize, start: usize, length: usize) !Tensor {
        return exec_gather_scatter.zeroSliceAxisRank(&self.rt, rank, x, axis, start, length);
    }

    pub fn zeroRowsAxisRank(self: *ExecContext, comptime rank: usize, x: *const Tensor, comptime axis: usize, indices: []const usize) !Tensor {
        return exec_gather_scatter.zeroRowsAxisRank(&self.rt, rank, x, axis, indices);
    }

    pub fn sliceGradientAxisRank(self: *ExecContext, comptime rank: usize, grad: *const Tensor, source_shape: [rank]usize, comptime axis: usize, start: usize) !Tensor {
        return exec_gather_scatter.sliceGradientAxisRank(&self.rt, rank, grad, source_shape, axis, start);
    }

    pub fn argmaxAxisRank(self: *ExecContext, comptime rank: usize, x: *const Tensor, comptime axis: usize) !Tensor {
        return exec_stats.argmaxAxisRank(&self.rt, rank, x, axis);
    }

    pub fn maxAxisRank(self: *ExecContext, comptime rank: usize, x: *const Tensor, comptime axis: usize) !TopKResult {
        return exec_stats.maxAxisRank(&self.rt, rank, x, axis);
    }

    pub fn minAxisRank(self: *ExecContext, comptime rank: usize, x: *const Tensor, comptime axis: usize) !TopKResult {
        return exec_stats.minAxisRank(&self.rt, rank, x, axis);
    }

    pub fn varAxisRank(self: *ExecContext, comptime rank: usize, x: *const Tensor, comptime axis: usize, ddof: u1) !Tensor {
        return exec_stats.varAxisRank(&self.rt, rank, x, axis, ddof);
    }

    pub fn standardizeAxisRank(
        self: *ExecContext,
        comptime rank: usize,
        x: *const Tensor,
        comptime axis: usize,
        options: StandardizeOptions,
    ) !Tensor {
        return exec_stats.standardizeAxisRank(&self.rt, rank, x, axis, options);
    }

    pub fn standardizeAxisValidPrefixRank(
        self: *ExecContext,
        comptime rank: usize,
        x: *const Tensor,
        comptime axis: usize,
        valid_len: usize,
        options: StandardizeOptions,
    ) !Tensor {
        return exec_stats.standardizeAxisValidPrefixRank(&self.rt, rank, x, axis, valid_len, options);
    }

    pub fn standardizeBackwardAxisRank(
        self: *ExecContext,
        comptime rank: usize,
        x: *const Tensor,
        gy: *const Tensor,
        comptime axis: usize,
        valid_len: ?usize,
        options: StandardizeOptions,
    ) !Tensor {
        return exec_stats.standardizeBackwardAxisRank(&self.rt, rank, x, gy, axis, valid_len, options);
    }

    pub fn topKAxisRank(self: *ExecContext, comptime rank: usize, x: *const Tensor, comptime axis: usize, k: usize) !TopKResult {
        return exec_stats.topKAxisRank(&self.rt, rank, x, axis, k);
    }

    pub fn sortAxisRank(self: *ExecContext, comptime rank: usize, x: *const Tensor, comptime axis: usize, descending: bool) !TopKResult {
        return exec_stats.sortAxisRank(&self.rt, rank, x, axis, descending);
    }

    /// Router softmax over rank-2 `[row, expert]` logits, followed by top-k expert
    /// selection per row. `probs` is per-row scratch of length >= expert count;
    /// `selected` and `weights` are row-major `[row, k]` outputs.
    pub fn routerTopK(
        self: *ExecContext,
        logits: *const Tensor,
        k: usize,
        options: RouterTopKOptions,
        selected: []usize,
        weights: []f32,
    ) !void {
        return exec_topk.routerTopK(&self.rt, logits, k, options, selected, weights);
    }

    pub fn scatterAddAxisRank(
        self: *ExecContext,
        comptime rank: usize,
        grad: *const Tensor,
        source_shape: [rank]usize,
        comptime axis: usize,
        indices: []const usize,
    ) !Tensor {
        return exec_gather_scatter.scatterAddAxisRank(&self.rt, rank, grad, source_shape, axis, indices);
    }

    pub fn takeAlongAxisRank(
        self: *ExecContext,
        comptime rank: usize,
        x: *const Tensor,
        comptime axis: usize,
        indices: []const usize,
        out_axis_len: usize,
    ) !Tensor {
        return exec_gather_scatter.takeAlongAxisRank(&self.rt, rank, x, axis, indices, out_axis_len);
    }

    pub fn scatterAddAlongAxisRank(
        self: *ExecContext,
        comptime rank: usize,
        base: *const Tensor,
        src: *const Tensor,
        comptime axis: usize,
        indices: []const usize,
    ) !Tensor {
        return exec_gather_scatter.scatterAddAlongAxisRank(&self.rt, rank, base, src, axis, indices);
    }

    pub fn scatterAlongAxisRank(
        self: *ExecContext,
        comptime rank: usize,
        base: *const Tensor,
        src: *const Tensor,
        comptime axis: usize,
        indices: []const usize,
    ) !Tensor {
        return exec_gather_scatter.scatterAlongAxisRank(&self.rt, rank, base, src, axis, indices);
    }

    pub fn softmaxAxisRank(self: *ExecContext, comptime rank: usize, x: *const Tensor, comptime axis: usize) !Tensor {
        return exec_softmax.softmaxAxisRank(&self.rt, rank, x, axis);
    }

    pub fn logsumexpAxisRank(self: *ExecContext, comptime rank: usize, x: *const Tensor, comptime axis: usize) !Tensor {
        return exec_softmax.logsumexpAxisRank(&self.rt, rank, x, axis);
    }

    pub fn logSoftmaxAxisRank(self: *ExecContext, comptime rank: usize, x: *const Tensor, comptime axis: usize) !Tensor {
        return exec_softmax.logSoftmaxAxisRank(&self.rt, rank, x, axis);
    }

    pub fn softmaxExtAxisRank(self: *ExecContext, comptime rank: usize, x: *const Tensor, comptime axis: usize, options: SoftmaxExtOptions) !Tensor {
        return exec_softmax.softmaxExtAxisRank(&self.rt, rank, x, axis, options);
    }

    pub fn softmaxBackwardAxisRank(self: *ExecContext, comptime rank: usize, y: *const Tensor, gy: *const Tensor, comptime axis: usize) !Tensor {
        return exec_softmax.softmaxBackwardAxisRank(&self.rt, rank, y, gy, axis);
    }

    pub fn softmaxExtBackwardAxisRank(self: *ExecContext, comptime rank: usize, y: *const Tensor, gy: *const Tensor, comptime axis: usize, scale_value: f32) !Tensor {
        return exec_softmax.softmaxExtBackwardAxisRank(&self.rt, rank, y, gy, axis, scale_value);
    }

    pub fn rmsNormAxisRank(self: *ExecContext, comptime rank: usize, x: *const Tensor, comptime axis: usize, eps: f32) !Tensor {
        return exec_norm.rmsNormAxisRank(&self.rt, rank, x, axis, eps);
    }

    pub fn rmsNormMulAxisRank(self: *ExecContext, comptime rank: usize, x: *const Tensor, weight: *const Tensor, comptime axis: usize, eps: f32) !Tensor {
        return exec_norm.rmsNormMulAxisRank(&self.rt, rank, x, weight, axis, eps);
    }

    pub fn rmsNormMulAddAxisRank(self: *ExecContext, comptime rank: usize, x: *const Tensor, weight: *const Tensor, residual: *const Tensor, comptime axis: usize, eps: f32) !Tensor {
        return exec_norm.rmsNormMulAddAxisRank(&self.rt, rank, x, weight, residual, axis, eps);
    }

    pub fn rmsNormMulBackwardInputAxisRank(
        self: *ExecContext,
        comptime rank: usize,
        x: *const Tensor,
        weight: *const Tensor,
        gy: *const Tensor,
        comptime axis: usize,
        eps: f32,
    ) !Tensor {
        return exec_norm.rmsNormMulBackwardInputAxisRank(&self.rt, rank, x, weight, gy, axis, eps);
    }

    pub fn rmsNormMulBackwardWeightAxisRank(
        self: *ExecContext,
        comptime rank: usize,
        x: *const Tensor,
        gy: *const Tensor,
        comptime axis: usize,
        eps: f32,
    ) !Tensor {
        return exec_norm.rmsNormMulBackwardWeightAxisRank(&self.rt, rank, x, gy, axis, eps);
    }

    pub fn rmsNormMulRopeAxisRankWithTable(
        self: *ExecContext,
        comptime rank: usize,
        x: *const Tensor,
        weight: *const Tensor,
        comptime position_axis: usize,
        comptime feature_axis: usize,
        eps: f32,
        table: *const RopeTable,
        comptime mode: RopeMode,
    ) !Tensor {
        return exec_norm.rmsNormMulRopeAxisRankWithTable(&self.rt, rank, x, weight, position_axis, feature_axis, eps, table, mode);
    }

    pub fn rmsNormBackwardAxisRank(self: *ExecContext, comptime rank: usize, x: *const Tensor, gy: *const Tensor, comptime axis: usize, eps: f32) !Tensor {
        return exec_norm.rmsNormBackwardAxisRank(&self.rt, rank, x, gy, axis, eps);
    }

    pub fn layerNormAffineRows(
        self: *ExecContext,
        input: []const f32,
        rows: usize,
        cols: usize,
        weight: []const f32,
        bias: []const f32,
        eps: f32,
    ) !Tensor {
        return exec_norm.layerNormAffineRows(&self.rt, input, rows, cols, weight, bias, eps);
    }

    pub fn layerNormAxisRank(self: *ExecContext, comptime rank: usize, x: *const Tensor, comptime axis: usize, eps: f32) !Tensor {
        return exec_norm.layerNormAxisRank(&self.rt, rank, x, axis, eps);
    }

    pub fn groupNormAxisRank(self: *ExecContext, x: *const Tensor, groups: usize, eps: f32, weight: ?*const Tensor, bias: ?*const Tensor) !Tensor {
        return exec_norm.groupNormAxisRank(&self.rt, x, groups, eps, weight, bias);
    }

    pub fn groupNormBackwardAxisRank(
        self: *ExecContext,
        x: *const Tensor,
        gy: *const Tensor,
        groups: usize,
        eps: f32,
        weight: ?*const Tensor,
        need_input: bool,
        need_weight: bool,
        need_bias: bool,
    ) !GroupNormBackwardResult {
        return exec_norm.groupNormBackwardAxisRank(&self.rt, x, gy, groups, eps, weight, need_input, need_weight, need_bias);
    }

    pub fn layerNormAffineAxisRank(
        self: *ExecContext,
        comptime rank: usize,
        x: *const Tensor,
        weight: *const Tensor,
        bias: *const Tensor,
        comptime axis: usize,
        eps: f32,
    ) !Tensor {
        return exec_norm.layerNormAffineAxisRank(&self.rt, rank, x, weight, bias, axis, eps);
    }

    pub fn layerNormBackwardAxisRank(self: *ExecContext, comptime rank: usize, x: *const Tensor, gy: *const Tensor, comptime axis: usize, eps: f32) !Tensor {
        return exec_norm.layerNormBackwardAxisRank(&self.rt, rank, x, gy, axis, eps);
    }

    pub fn layerNormAffineBackwardAxisRank(
        self: *ExecContext,
        comptime rank: usize,
        x: *const Tensor,
        weight: *const Tensor,
        gy: *const Tensor,
        comptime axis: usize,
        eps: f32,
        need_input: bool,
        need_weight: bool,
        need_bias: bool,
    ) !LayerNormAffineBackwardResult {
        return exec_norm.layerNormAffineBackwardAxisRank(&self.rt, rank, x, weight, gy, axis, eps, need_input, need_weight, need_bias);
    }

    pub fn crossEntropyLossAxisRank(self: *ExecContext, comptime rank: usize, logits: *const Tensor, comptime axis: usize, labels: []const usize) !Tensor {
        return exec_loss.crossEntropyLossAxisRank(&self.rt, rank, logits, axis, labels);
    }

    pub fn crossEntropyLossExAxisRank(
        self: *ExecContext,
        comptime rank: usize,
        logits: *const Tensor,
        comptime axis: usize,
        labels: []const usize,
        options: CrossEntropyOptions,
    ) !Tensor {
        return exec_loss.crossEntropyLossExAxisRank(&self.rt, rank, logits, axis, labels, options);
    }

    pub fn crossEntropyLossExStatsAxisRank(
        self: *ExecContext,
        comptime rank: usize,
        logits: *const Tensor,
        comptime axis: usize,
        labels: []const usize,
        options: CrossEntropyOptions,
        row_stats: ?[]f32,
    ) !Tensor {
        return exec_loss.crossEntropyLossExStatsAxisRank(&self.rt, rank, logits, axis, labels, options, row_stats);
    }

    pub fn crossEntropyBackwardAxisRank(
        self: *ExecContext,
        comptime rank: usize,
        logits: *const Tensor,
        comptime axis: usize,
        labels: []const usize,
        scale_value: f32,
    ) !Tensor {
        return exec_loss.crossEntropyBackwardAxisRank(&self.rt, rank, logits, axis, labels, scale_value);
    }

    pub fn crossEntropyBackwardExAxisRank(
        self: *ExecContext,
        comptime rank: usize,
        logits: *const Tensor,
        comptime axis: usize,
        labels: []const usize,
        options: CrossEntropyOptions,
        scale_value: f32,
        per_row_scale: ?[]const f32,
    ) !Tensor {
        return exec_loss.crossEntropyBackwardExAxisRank(&self.rt, rank, logits, axis, labels, options, scale_value, per_row_scale);
    }

    pub fn crossEntropyBackwardExStatsAxisRank(
        self: *ExecContext,
        comptime rank: usize,
        logits: *const Tensor,
        comptime axis: usize,
        labels: []const usize,
        options: CrossEntropyOptions,
        scale_value: f32,
        per_row_scale: ?[]const f32,
        row_stats: ?[]const f32,
    ) !Tensor {
        return exec_loss.crossEntropyBackwardExStatsAxisRank(&self.rt, rank, logits, axis, labels, options, scale_value, per_row_scale, row_stats);
    }

    pub fn crossEntropyBackwardExUpstreamAxisRank(
        self: *ExecContext,
        comptime rank: usize,
        logits: *const Tensor,
        comptime axis: usize,
        labels: []const usize,
        options: CrossEntropyOptions,
        gy: *const Tensor,
    ) !Tensor {
        return exec_loss.crossEntropyBackwardExUpstreamAxisRank(&self.rt, rank, logits, axis, labels, options, gy);
    }

    pub fn crossEntropyBackwardExUpstreamStatsAxisRank(
        self: *ExecContext,
        comptime rank: usize,
        logits: *const Tensor,
        comptime axis: usize,
        labels: []const usize,
        options: CrossEntropyOptions,
        gy: *const Tensor,
        row_stats: ?[]const f32,
    ) !Tensor {
        return exec_loss.crossEntropyBackwardExUpstreamStatsAxisRank(&self.rt, rank, logits, axis, labels, options, gy, row_stats);
    }

    /// DESTRUCTIVE in `logits` — see `exec_loss.linearCrossEntropyBackwardUpstream`.
    pub fn linearCrossEntropyBackwardUpstream(
        self: *ExecContext,
        x: *const Tensor,
        weight: *const Tensor,
        logits: *Tensor,
        labels: []const usize,
        options: CrossEntropyOptions,
        gy: *const Tensor,
        row_stats: []const f32,
        need_x: bool,
        need_weight: bool,
    ) !LinearCrossEntropyGrads {
        return exec_loss.linearCrossEntropyBackwardUpstream(&self.rt, x, weight, logits, labels, options, gy, row_stats, need_x, need_weight);
    }

    pub fn mseLoss(self: *ExecContext, input: *const Tensor, target: *const Tensor, options: MseOptions) !Tensor {
        return exec_loss.mseLoss(&self.rt, input, target, options);
    }

    pub fn mseBackwardUpstream(self: *ExecContext, input: *const Tensor, target: *const Tensor, options: MseOptions, gy: *const Tensor, wrt: LossWrt) !Tensor {
        return exec_loss.mseBackwardUpstream(&self.rt, input, target, options, gy, wrt);
    }

    pub fn huberLoss(self: *ExecContext, input: *const Tensor, target: *const Tensor, options: HuberOptions) !Tensor {
        return exec_loss.huberLoss(&self.rt, input, target, options);
    }

    pub fn huberBackwardUpstream(self: *ExecContext, input: *const Tensor, target: *const Tensor, options: HuberOptions, gy: *const Tensor, wrt: LossWrt) !Tensor {
        return exec_loss.huberBackwardUpstream(&self.rt, input, target, options, gy, wrt);
    }

    pub fn bceLoss(self: *ExecContext, input: *const Tensor, target: *const Tensor, options: BceOptions) !Tensor {
        return exec_loss.bceLoss(&self.rt, input, target, options);
    }

    pub fn bceBackwardUpstream(self: *ExecContext, input: *const Tensor, target: *const Tensor, options: BceOptions, gy: *const Tensor, wrt: LossWrt) !Tensor {
        return exec_loss.bceBackwardUpstream(&self.rt, input, target, options, gy, wrt);
    }

    pub fn klDivLoss(self: *ExecContext, input: *const Tensor, target: *const Tensor, options: KlDivOptions) !Tensor {
        return exec_loss.klDivLoss(&self.rt, input, target, options);
    }

    pub fn klDivBackwardUpstream(self: *ExecContext, input: *const Tensor, target: *const Tensor, options: KlDivOptions, gy: *const Tensor, wrt: LossWrt) !Tensor {
        return exec_loss.klDivBackwardUpstream(&self.rt, input, target, options, gy, wrt);
    }

    pub fn ropeAxisRank(
        self: *ExecContext,
        comptime rank: usize,
        x: *const Tensor,
        comptime position_axis: usize,
        comptime feature_axis: usize,
        positions: []const i32,
        theta_base: f32,
        comptime mode: RopeMode,
        comptime inverse: bool,
    ) !Tensor {
        return exec_rope.ropeAxisRank(&self.rt, rank, x, position_axis, feature_axis, positions, theta_base, mode, inverse);
    }

    pub fn prepareRopeTable(self: *ExecContext, positions: []const i32, feature_dim: usize, theta_base: f32, inverse: bool) !RopeTable {
        return exec_rope.prepareRopeTable(&self.rt, positions, feature_dim, theta_base, inverse);
    }

    pub fn prepareRopeTableFactors(
        self: *ExecContext,
        positions: []const i32,
        feature_dim: usize,
        theta_base: f32,
        inverse: bool,
        freq_factors: ?[]const f32,
    ) !RopeTable {
        return exec_rope.prepareRopeTableFactors(&self.rt, positions, feature_dim, theta_base, inverse, freq_factors);
    }

    pub fn ropeAxisRankWithTable(
        self: *ExecContext,
        comptime rank: usize,
        x: *const Tensor,
        comptime position_axis: usize,
        comptime feature_axis: usize,
        table: *const RopeTable,
        comptime mode: RopeMode,
    ) !Tensor {
        return exec_rope.ropeAxisRankWithTable(&self.rt, rank, x, position_axis, feature_axis, table, mode);
    }

    pub fn ropePartialAxisRank(
        self: *ExecContext,
        comptime rank: usize,
        x: *const Tensor,
        comptime position_axis: usize,
        comptime feature_axis: usize,
        rotary_dim: usize,
        positions: []const i32,
        theta_base: f32,
        comptime mode: RopeMode,
        comptime inverse: bool,
    ) !Tensor {
        return exec_rope.ropePartialAxisRank(&self.rt, rank, x, position_axis, feature_axis, rotary_dim, positions, theta_base, mode, inverse);
    }

    pub fn ropePartialAxisRankWithTable(
        self: *ExecContext,
        comptime rank: usize,
        x: *const Tensor,
        comptime position_axis: usize,
        comptime feature_axis: usize,
        table: *const RopeTable,
        comptime mode: RopeMode,
    ) !Tensor {
        return exec_rope.ropePartialAxisRankWithTable(&self.rt, rank, x, position_axis, feature_axis, table, mode);
    }

    pub fn groupedCausalAttention(
        self: *ExecContext,
        q: *const Tensor,
        k: *const Tensor,
        v: *const Tensor,
        kv_head_for_head: []const usize,
        scale_value: f32,
    ) !Tensor {
        return exec_attention.groupedCausalAttention(&self.rt, q, k, v, kv_head_for_head, scale_value);
    }

    /// As `groupedCausalAttention` with a sliding-window `window` (0 = full
    /// causal; else a query at absolute position `p` attends only keys in
    /// `[max(0, p-window+1), p]`). Used by Gemma's local SWA layers.
    pub fn groupedCausalAttentionWindowed(
        self: *ExecContext,
        q: *const Tensor,
        k: *const Tensor,
        v: *const Tensor,
        kv_head_for_head: []const usize,
        scale_value: f32,
        window: usize,
    ) !Tensor {
        return exec_attention.groupedCausalAttentionWindowed(&self.rt, q, k, v, kv_head_for_head, scale_value, window);
    }

    /// Bidirectional (non-causal) grouped attention: every query row attends
    /// EVERY key row.
    pub fn groupedBidirectionalAttention(
        self: *ExecContext,
        q: *const Tensor,
        k: *const Tensor,
        v: *const Tensor,
        kv_head_for_head: []const usize,
        scale_value: f32,
    ) !Tensor {
        return exec_attention.groupedBidirectionalAttention(&self.rt, q, k, v, kv_head_for_head, scale_value);
    }

    /// As `groupedBidirectionalAttention` with an additive f32 `bias`
    /// `[q_seq, kv_seq]` added to the scaled pre-softmax scores (an additive
    /// soft bias, ggml_soft_max_ext mask semantics — NOT -inf masking).
    /// OmniVoice's uncond CFG row.
    pub fn groupedBidirectionalAttentionBiased(
        self: *ExecContext,
        q: *const Tensor,
        k: *const Tensor,
        v: *const Tensor,
        kv_head_for_head: []const usize,
        scale_value: f32,
        bias: *const Tensor,
    ) !Tensor {
        return exec_attention.groupedBidirectionalAttentionBiased(&self.rt, q, k, v, kv_head_for_head, scale_value, bias);
    }

    /// As the f32 grouped attention forwards (`causal`/`window` select the
    /// variant) additionally recording per-(head, query) softmax
    /// {max, sum_exp} statistics (heads * q_seq * 2 interleaved f32) — the
    /// output is BITWISE identical to the stats-less entries; the stats feed
    /// `groupedCausalAttentionBackward`.
    pub fn groupedCausalAttentionStatsOut(
        self: *ExecContext,
        q: *const Tensor,
        k: *const Tensor,
        v: *const Tensor,
        kv_head_for_head: []const usize,
        scale_value: f32,
        window: usize,
        causal: bool,
        stats: []f32,
    ) !Tensor {
        return exec_attention.groupedCausalAttentionStatsOut(&self.rt, q, k, v, kv_head_for_head, scale_value, window, causal, stats);
    }

    /// `stats` (optional): forward-saved {max, sum_exp} pairs from
    /// `groupedCausalAttentionStatsOut` (heads * q_seq * 2), consumed by the
    /// GEMM route (FUCINA_NO_ATTN_BWD_STATS pins the recompute route).
    pub fn groupedCausalAttentionBackward(
        self: *ExecContext,
        q: *const Tensor,
        k: *const Tensor,
        v: *const Tensor,
        gy: *const Tensor,
        kv_head_for_head: []const usize,
        scale_value: f32,
        window: usize,
        causal: bool,
        stats: ?[]const f32,
        out: ?*const Tensor,
        need_q: bool,
        need_k: bool,
        need_v: bool,
    ) !GroupedCausalAttentionBackwardResult {
        return exec_attention.groupedCausalAttentionBackward(&self.rt, q, k, v, gy, kv_head_for_head, scale_value, window, causal, stats, out, need_q, need_k, need_v);
    }

    /// Same as `groupedCausalAttention` but the cached K/V are f16 (decode KV
    /// cache): half the bandwidth, widened to f32 in the kernel. Q and the
    /// output stay f32.
    pub fn groupedCausalAttentionF16Kv(
        self: *ExecContext,
        q: *const Tensor,
        k: *const tensor.TensorOf(.f16),
        v: *const tensor.TensorOf(.f16),
        kv_head_for_head: []const usize,
        scale_value: f32,
    ) !Tensor {
        return exec_attention.groupedCausalAttentionF16Kv(&self.rt, q, k, v, kv_head_for_head, scale_value);
    }

    /// f16-KV bidirectional attention (see `groupedBidirectionalAttention`):
    /// the block-diffusion canvas pass over a prefix+canvas f16 KV cache.
    pub fn groupedBidirectionalAttentionF16Kv(
        self: *ExecContext,
        q: *const Tensor,
        k: *const tensor.TensorOf(.f16),
        v: *const tensor.TensorOf(.f16),
        kv_head_for_head: []const usize,
        scale_value: f32,
    ) !Tensor {
        return exec_attention.groupedBidirectionalAttentionF16Kv(&self.rt, q, k, v, kv_head_for_head, scale_value);
    }

    /// f16-KV decode attention with a sliding `window` (see
    /// `groupedCausalAttentionWindowed`).
    pub fn groupedCausalAttentionF16KvWindowed(
        self: *ExecContext,
        q: *const Tensor,
        k: *const tensor.TensorOf(.f16),
        v: *const tensor.TensorOf(.f16),
        kv_head_for_head: []const usize,
        scale_value: f32,
        window: usize,
    ) !Tensor {
        return exec_attention.groupedCausalAttentionF16KvWindowed(&self.rt, q, k, v, kv_head_for_head, scale_value, window);
    }

    /// Same as `groupedCausalAttention` but the cached K/V are q8_0 blocks
    /// (the quantized decode KV cache). Q and the output stay f32.
    pub fn groupedCausalAttentionQ8Kv(
        self: *ExecContext,
        q: *const Tensor,
        k_blocks: []const BlockQ8_0,
        v_blocks: []const BlockQ8_0,
        kv_seq: usize,
        kv_heads: usize,
        kv_head_for_head: []const usize,
        scale_value: f32,
    ) !Tensor {
        return exec_attention.groupedCausalAttentionQ8Kv(&self.rt, q, k_blocks, v_blocks, kv_seq, kv_heads, kv_head_for_head, scale_value);
    }

    /// q8_0-KV attention with a sliding `window` (see
    /// `groupedCausalAttentionWindowed`).
    pub fn groupedCausalAttentionQ8KvWindowed(
        self: *ExecContext,
        q: *const Tensor,
        k_blocks: []const BlockQ8_0,
        v_blocks: []const BlockQ8_0,
        kv_seq: usize,
        kv_heads: usize,
        kv_head_for_head: []const usize,
        scale_value: f32,
        window: usize,
    ) !Tensor {
        return exec_attention.groupedCausalAttentionQ8KvWindowed(&self.rt, q, k_blocks, v_blocks, kv_seq, kv_heads, kv_head_for_head, scale_value, window);
    }

    /// Ragged multi-stream decode attention over per-stream f16 KV caches
    /// (batch-N decode): query row `s` of `q` `[n_streams, heads, d]`
    /// attends all `lens[s]` cached positions of `ks[s]`/`vs[s]`. Runs the
    /// same per-query kernels as m=1 decode — per-stream results are
    /// bit-identical to N single-stream `groupedCausalAttentionF16Kv` calls.
    pub fn groupedCausalAttentionMultiF16Kv(
        self: *ExecContext,
        q: *const Tensor,
        ks: []const []const f16,
        vs: []const []const f16,
        lens: []const usize,
        kv_heads: usize,
        kv_head_for_head: []const usize,
        scale_value: f32,
    ) !Tensor {
        return exec_attention.groupedCausalAttentionMultiF16Kv(&self.rt, q, ks, vs, lens, kv_heads, kv_head_for_head, scale_value);
    }

    /// As `groupedCausalAttentionMultiF16Kv` for q8_0 caches (per-stream
    /// `kBlocks`/`vBlocks` slices).
    pub fn groupedCausalAttentionMultiQ8Kv(
        self: *ExecContext,
        q: *const Tensor,
        ks: []const []const BlockQ8_0,
        vs: []const []const BlockQ8_0,
        lens: []const usize,
        kv_heads: usize,
        kv_head_for_head: []const usize,
        scale_value: f32,
    ) !Tensor {
        return exec_attention.groupedCausalAttentionMultiQ8Kv(&self.rt, q, ks, vs, lens, kv_heads, kv_head_for_head, scale_value);
    }

    pub fn dot(self: *ExecContext, a: *const Tensor, b: *const Tensor) !Tensor {
        return exec_matmul.dot(&self.rt, a, b);
    }

    pub fn dotTyped(self: *ExecContext, comptime dtype: DType, a: *const tensor.TensorOf(dtype), b: *const tensor.TensorOf(dtype)) !tensor.TensorOf(dtype_mod.outputDType(.matmul, dtype)) {
        return exec_matmul.dotTyped(&self.rt, dtype, a, b);
    }

    pub fn matmul(self: *ExecContext, a: *const Tensor, b: *const Tensor) !Tensor {
        return self.matmul2D(a, b);
    }

    pub fn matmulTyped(self: *ExecContext, comptime dtype: DType, a: *const tensor.TensorOf(dtype), b: *const tensor.TensorOf(dtype)) !tensor.TensorOf(dtype_mod.outputDType(.matmul, dtype)) {
        return self.matmul2DTyped(dtype, a, b);
    }

    pub fn matmul2D(self: *ExecContext, a: *const Tensor, b: *const Tensor) !Tensor {
        return exec_matmul.matmul2DDispatch(&self.rt, .plain, a, b);
    }

    pub fn matmul2DTyped(self: *ExecContext, comptime dtype: DType, a: *const tensor.TensorOf(dtype), b: *const tensor.TensorOf(dtype)) !tensor.TensorOf(dtype_mod.outputDType(.matmul, dtype)) {
        return exec_matmul.matmul2DTyped(&self.rt, dtype, a, b);
    }

    pub fn packMatmulRhsTyped(self: *ExecContext, comptime dtype: DType, rhs: *const tensor.TensorOf(dtype)) !backend_mod.PackedMatmulRhsFor(dtype) {
        return exec_matmul.packMatmulRhsTyped(&self.rt, dtype, rhs);
    }

    pub fn matmul2DWithPackedRhsTyped(
        self: *ExecContext,
        comptime dtype: DType,
        a: *const tensor.TensorOf(dtype),
        rhs: *const backend_mod.PackedMatmulRhsFor(dtype),
    ) !tensor.TensorOf(dtype_mod.outputDType(.matmul, dtype)) {
        return exec_matmul.matmul2DWithPackedRhsTyped(&self.rt, dtype, a, rhs);
    }

    pub fn dequantizeTensorTyped(self: *ExecContext, comptime dtype: DType, x: *const tensor.TensorOf(dtype)) !Tensor {
        return exec_quant_matmul.dequantizeTensorTyped(&self.rt, dtype, x);
    }

    pub fn getRowsQuantizedTyped(self: *ExecContext, comptime dtype: DType, table: *const tensor.TensorOf(dtype), indices: []const usize) !Tensor {
        return exec_quant_matmul.getRowsQuantizedTyped(&self.rt, dtype, table, indices);
    }

    // f32 activations [m, k] x block-quantized RHS weights stored as [n, k]
    // row blocks -> f32 [m, n]. This is the public Tensor-backed path; GGUF
    // loading will populate these block-quantized tensors directly.
    pub fn matmul2DWithQuantizedTensorRhs(
        self: *ExecContext,
        comptime rhs_dtype: DType,
        a: *const Tensor,
        rhs: *const tensor.TensorOf(rhs_dtype),
    ) !Tensor {
        return exec_quant_matmul.matmul2DWithQuantizedTensorRhs(&self.rt, rhs_dtype, a, rhs);
    }

    pub fn matmul2DWithQuantizedTensorRhsOptions(
        self: *ExecContext,
        comptime rhs_dtype: DType,
        a: *const Tensor,
        rhs: *const tensor.TensorOf(rhs_dtype),
        options: QuantizedMatmulOptions,
    ) !Tensor {
        return exec_quant_matmul.matmul2DWithQuantizedTensorRhsOptions(&self.rt, rhs_dtype, a, rhs, options);
    }

    pub fn matmul2DWithQuantizedBlocksRhs(
        self: *ExecContext,
        comptime rhs_dtype: DType,
        a: *const Tensor,
        blocks: []const dtype_mod.Storage(rhs_dtype),
        n: usize,
        k: usize,
    ) !Tensor {
        return exec_quant_matmul.matmul2DWithQuantizedBlocksRhs(&self.rt, rhs_dtype, a, blocks, n, k);
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
        return exec_quant_matmul.matmul2DWithQuantizedBlocksRhsOptions(&self.rt, rhs_dtype, a, blocks, n, k, options);
    }

    pub fn packMatmulRhsQ8_0x4(self: *ExecContext, rhs: *const tensor.TensorOf(.q8_0)) !backend_mod.QuantizedMatmulRhsQ8_0x4 {
        return exec_quant_matmul.packMatmulRhsQ8_0x4(&self.rt, rhs);
    }

    pub fn packMatmulRhsQ6_Kx4(self: *ExecContext, rhs: *const tensor.TensorOf(.q6_k)) !backend_mod.QuantizedMatmulRhsQ6_Kx4 {
        return exec_quant_matmul.packMatmulRhsQ6_Kx4(&self.rt, rhs);
    }

    pub fn packMatmulRhsQ4_Kx4(self: *ExecContext, rhs: *const tensor.TensorOf(.q4_k)) !backend_mod.QuantizedMatmulRhsQ4_Kx4 {
        return exec_quant_matmul.packMatmulRhsQ4_Kx4(&self.rt, rhs);
    }

    pub fn packMatmulRhsQ4_Kx8(self: *ExecContext, rhs: *const tensor.TensorOf(.q4_k)) !backend_mod.QuantizedMatmulRhsQ4_Kx8 {
        return exec_quant_matmul.packMatmulRhsQ4_Kx8(&self.rt, rhs);
    }

    pub fn packMatmulRhsQ4_Kx2Mmla(self: *ExecContext, rhs: *const tensor.TensorOf(.q4_k)) !backend_mod.QuantizedMatmulRhsQ4_Kx2Mmla {
        return exec_quant_matmul.packMatmulRhsQ4_Kx2Mmla(&self.rt, rhs);
    }

    pub fn packMatmulRhsQ5_Kx8(self: *ExecContext, rhs: *const tensor.TensorOf(.q5_k)) !backend_mod.QuantizedMatmulRhsQ5_Kx8 {
        return exec_quant_matmul.packMatmulRhsQ5_Kx8(&self.rt, rhs);
    }

    pub fn matmul2DWithPackedQ8_0x4Rhs(self: *ExecContext, a: *const Tensor, rhs: *const backend_mod.QuantizedMatmulRhsQ8_0x4) !Tensor {
        return exec_quant_matmul.matmul2DWithPackedQ8_0x4Rhs(&self.rt, a, rhs);
    }

    pub fn rmsNormMulMatmul2DWithPackedQ8_0x4Rhs(self: *ExecContext, x: *const Tensor, norm_weights: *const Tensor, eps: f32, rhs: *const backend_mod.QuantizedMatmulRhsQ8_0x4) !Tensor {
        return exec_quant_matmul.rmsNormMulMatmul2DWithPackedQ8_0x4Rhs(&self.rt, x, norm_weights, eps, rhs);
    }

    pub fn rmsNormMulMatmul2DWithPackedQ4_Kx8Rhs(self: *ExecContext, x: *const Tensor, norm_weights: *const Tensor, eps: f32, rhs: *const backend_mod.QuantizedMatmulRhsQ4_Kx8) !Tensor {
        return exec_quant_matmul.rmsNormMulMatmul2DWithPackedQ4_Kx8Rhs(&self.rt, x, norm_weights, eps, rhs);
    }

    pub fn rmsNormMulMatmul2DWithPackedQ5_Kx8Rhs(self: *ExecContext, x: *const Tensor, norm_weights: *const Tensor, eps: f32, rhs: *const backend_mod.QuantizedMatmulRhsQ5_Kx8) !Tensor {
        return exec_quant_matmul.rmsNormMulMatmul2DWithPackedQ5_Kx8Rhs(&self.rt, x, norm_weights, eps, rhs);
    }

    pub fn rmsNormMulMatmul2DWithPackedQ6_Kx4Rhs(self: *ExecContext, x: *const Tensor, norm_weights: *const Tensor, eps: f32, rhs: *const backend_mod.QuantizedMatmulRhsQ6_Kx4) !Tensor {
        return exec_quant_matmul.rmsNormMulMatmul2DWithPackedQ6_Kx4Rhs(&self.rt, x, norm_weights, eps, rhs);
    }

    pub fn splitSwiGluMatmul2DWithPackedQ8_0x4Rhs(
        self: *ExecContext,
        gate_up: *const Tensor,
        rhs: *const backend_mod.QuantizedMatmulRhsQ8_0x4,
    ) !Tensor {
        return exec_quant_matmul.splitSwiGluMatmul2DWithPackedQ8_0x4Rhs(&self.rt, gate_up, rhs);
    }

    pub fn splitSwiGluMatmul2DWithPackedQ4_Kx8Rhs(self: *ExecContext, gate_up: *const Tensor, rhs: *const backend_mod.QuantizedMatmulRhsQ4_Kx8) !Tensor {
        return exec_quant_matmul.splitSwiGluMatmul2DWithPackedQ4_Kx8Rhs(&self.rt, gate_up, rhs);
    }

    pub fn splitSwiGluMatmul2DWithPackedQ5_Kx8Rhs(self: *ExecContext, gate_up: *const Tensor, rhs: *const backend_mod.QuantizedMatmulRhsQ5_Kx8) !Tensor {
        return exec_quant_matmul.splitSwiGluMatmul2DWithPackedQ5_Kx8Rhs(&self.rt, gate_up, rhs);
    }

    pub fn splitSwiGluMatmul2DWithPackedQ6_Kx4Rhs(self: *ExecContext, gate_up: *const Tensor, rhs: *const backend_mod.QuantizedMatmulRhsQ6_Kx4) !Tensor {
        return exec_quant_matmul.splitSwiGluMatmul2DWithPackedQ6_Kx4Rhs(&self.rt, gate_up, rhs);
    }

    /// Fused GeGLU (`up * geluQuant(gate)`, ggml f16-LUT semantics) + Q8_0 LHS
    /// quantization + packed Q8_0x4 GEMM, for separate gate/up projections.
    /// Bit-identical to unary(.gelu_quant) + mul + the packed dot, without
    /// materializing the activation tensors.
    pub fn gegluQuantMatmul2DWithPackedQ8_0x4Rhs(self: *ExecContext, gate: *const Tensor, up: *const Tensor, rhs: *const backend_mod.QuantizedMatmulRhsQ8_0x4) !Tensor {
        return exec_quant_matmul.gegluQuantMatmul2DWithPackedQ8_0x4Rhs(&self.rt, gate, up, rhs);
    }

    pub fn matmul2DWithPackedQ6_Kx4Rhs(self: *ExecContext, a: *const Tensor, rhs: *const backend_mod.QuantizedMatmulRhsQ6_Kx4) !Tensor {
        return exec_quant_matmul.matmul2DWithPackedQ6_Kx4Rhs(&self.rt, a, rhs);
    }

    pub fn matmul2DWithPackedQ4_Kx4Rhs(self: *ExecContext, a: *const Tensor, rhs: *const backend_mod.QuantizedMatmulRhsQ4_Kx4) !Tensor {
        return exec_quant_matmul.matmul2DWithPackedQ4_Kx4Rhs(&self.rt, a, rhs);
    }

    pub fn matmul2DWithPackedQ4_Kx8Rhs(self: *ExecContext, a: *const Tensor, rhs: *const backend_mod.QuantizedMatmulRhsQ4_Kx8) !Tensor {
        return exec_quant_matmul.matmul2DWithPackedQ4_Kx8Rhs(&self.rt, a, rhs);
    }

    pub fn matmul2DWithPackedQ4_Kx2MmlaRhs(self: *ExecContext, a: *const Tensor, rhs: *const backend_mod.QuantizedMatmulRhsQ4_Kx2Mmla) !Tensor {
        return exec_quant_matmul.matmul2DWithPackedQ4_Kx2MmlaRhs(&self.rt, a, rhs);
    }

    pub fn matmul2DWithPackedQ5_Kx8Rhs(self: *ExecContext, a: *const Tensor, rhs: *const backend_mod.QuantizedMatmulRhsQ5_Kx8) !Tensor {
        return exec_quant_matmul.matmul2DWithPackedQ5_Kx8Rhs(&self.rt, a, rhs);
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
        return exec_quant_matmul.denseQuantMatmulGpu(&self.rt, dtype, rhs_bytes, rhs_lifetime, nb01, input, m, n, k);
    }

    /// One-command GPU batch for same-shape dense quantized linears that share
    /// the same f32 activation matrix. Internal eager helper; callers still get
    /// a normal f32 Tensor shaped `[batch_count*m, n]`.
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
        return exec_quant_matmul.denseQuantMatmulGpuSharedInputBatch(&self.rt, dtype, rhs_bytes, rhs_lifetime, nb01, nb02, input, batch_count, m, n, k);
    }

    /// A Mixture-of-Experts projection: all experts of one layer's gate/up/down
    /// stacked into a single RHS buffer. The implementation lives in exec/moe.zig;
    /// this alias preserves the public ExecContext.MoeRhs surface.
    pub const MoeRhs = exec_moe.MoeRhs;

    /// Shared batched-MoE scheduling scaffolding (route plan, phase-chain
    /// machinery, chunk helpers, profile timers). Lives in exec/moe_chain.zig;
    /// exposed as an ExecContext decl so the gemma MoE engines at the llm
    /// layer reach the exact same types through the `fucina` root.
    pub const moe_chain = exec_moe_chain;

    pub fn lockMoeDecodeScratch(self: *ExecContext) void {
        exec_moe.lockMoeDecodeScratch(&self.moe_scratch);
    }

    pub fn unlockMoeDecodeScratch(self: *ExecContext) void {
        exec_moe.unlockMoeDecodeScratch(&self.moe_scratch);
    }

    pub fn MoeDecodeScratchView(comptime QgBlock: type, comptime Task: type) type {
        return exec_moe.MoeDecodeScratchView(QgBlock, Task);
    }

    pub fn MoeDecodeChainScratchView(comptime QgBlock: type, comptime State: type, comptime Task: type) type {
        return exec_moe.MoeDecodeChainScratchView(QgBlock, State, Task);
    }

    pub fn carveMoeDecodeScratch(
        self: *ExecContext,
        comptime QgBlock: type,
        comptime Task: type,
        hidden_blocks: usize,
        top_k: usize,
        out_pe: usize,
        hidden: usize,
        blocks_per_g: usize,
    ) !MoeDecodeScratchView(QgBlock, Task) {
        return exec_moe.carveMoeDecodeScratch(&self.rt, &self.moe_scratch, QgBlock, Task, hidden_blocks, top_k, out_pe, hidden, blocks_per_g);
    }

    pub fn carveMoeDecodeChainScratch(
        self: *ExecContext,
        comptime QgBlock: type,
        comptime State: type,
        comptime Task: type,
        hidden_blocks: usize,
        top_k: usize,
        out_pe: usize,
        hidden: usize,
        blocks_per_g: usize,
        task_count: usize,
    ) !MoeDecodeChainScratchView(QgBlock, State, Task) {
        return exec_moe.carveMoeDecodeChainScratch(&self.rt, &self.moe_scratch, QgBlock, State, Task, hidden_blocks, top_k, out_pe, hidden, blocks_per_g, task_count);
    }

    /// Fused MoE FFN for a single token: route-weighted sum over the selected
    /// experts of down(SwiGLU(gate(x), up(x))).
    pub fn moeExpertFfn(
        self: *ExecContext,
        x: *const Tensor,
        gate: *const MoeRhs,
        up: *const MoeRhs,
        down: *const MoeRhs,
        selected: []const usize,
        weights: []const f32,
        out_pe: usize,
        act: GatedOp,
        io: ?std.Io,
        profile: ?*MoeBatchProfile,
    ) !Tensor {
        return exec_moe.moeExpertFfn(&self.rt, &self.moe_scratch, x, gate, up, down, selected, weights, out_pe, act, io, profile);
    }

    /// Batched-prefill MoE FFN over `seq > 1` tokens.
    pub fn moeExpertFfnBatch(
        self: *ExecContext,
        x: *const Tensor,
        gate: *const MoeRhs,
        up: *const MoeRhs,
        down: *const MoeRhs,
        selected: []const usize,
        weights: []const f32,
        top_k: usize,
        out_pe: usize,
        act: GatedOp,
        io: ?std.Io,
        profile: ?*MoeBatchProfile,
    ) !Tensor {
        return exec_moe.moeExpertFfnBatch(&self.rt, x, gate, up, down, selected, weights, top_k, out_pe, act, io, profile);
    }

    pub fn matmulTransA(self: *ExecContext, a: *const Tensor, b: *const Tensor) !Tensor {
        return exec_matmul.matmul2DDispatch(&self.rt, .trans_a, a, b);
    }

    pub fn matmulTransB(self: *ExecContext, a: *const Tensor, b: *const Tensor) !Tensor {
        return exec_matmul.matmul2DDispatch(&self.rt, .trans_b, a, b);
    }

    pub fn matmulTransB2DWithF16Rhs(self: *ExecContext, a: *const Tensor, b: *const tensor.TensorOf(.f16)) !Tensor {
        return exec_matmul.matmulTransB2DWithF16Rhs(&self.rt, a, b);
    }

    // Mixed-precision twin of matmulTransB2DWithF16Rhs for bf16 weights. The
    // LHS stays f32 (no cast: the kernel widens the bf16 RHS in-register and
    // accumulates in f32), so only contiguity is prepared here.
    pub fn matmulTransB2DWithBf16Rhs(self: *ExecContext, a: *const Tensor, b: *const tensor.TensorOf(.bf16)) !Tensor {
        return exec_matmul.matmulTransB2DWithBf16Rhs(&self.rt, a, b);
    }

    // Batched matrix multiplication. Supports:
    //   - Full batched:    a=[..., M, K] @ b=[..., K, N] -> [..., M, N]
    //                      Leading batch dims may match exactly or broadcast.
    //   - Broadcast RHS:   a=[..., M, K] @ b=[K, N]      -> [..., M, N]
    //                      Single fused 2-D GEMM via reshape, no per-batch loop.
    //   - Broadcast LHS:   a=[M, K]      @ b=[..., K, N] -> [..., M, N]
    // General multi-axis broadcast never materializes expanded tensors; the
    // runtime computes per-output-batch source offsets and preserves the exact
    // and shared-operand fast paths. Strict 2-D inputs must use matmul/matmul2D.
    pub fn bmm(self: *ExecContext, a: *const Tensor, b: *const Tensor) !Tensor {
        return exec_matmul.bmmDispatch(&self.rt, .plain, a, b);
    }

    // Batched matmul with implicit transpose of the per-batch A:
    //   a=[..., K, M] @ b=[..., K, N] -> [..., M, N]
    // Used by autograd to compute dB = A^T @ dY in batched form. Shares the
    // dispatch logic with bmm.
    pub fn bmmTransA(self: *ExecContext, a: *const Tensor, b: *const Tensor) !Tensor {
        return exec_matmul.bmmDispatch(&self.rt, .trans_a, a, b);
    }

    // Batched matmul with implicit transpose of the per-batch B:
    //   a=[..., M, K] @ b=[..., N, K] -> [..., M, N]
    // Used by autograd to compute dA = dY @ B^T in batched form. The shared-B
    // fast path (broadcast RHS) also applies here.
    pub fn bmmTransB(self: *ExecContext, a: *const Tensor, b: *const Tensor) !Tensor {
        return exec_matmul.bmmDispatch(&self.rt, .trans_b, a, b);
    }

    pub fn reduceBroadcast(self: *ExecContext, x: *const Tensor, target_shape: []const usize) !Tensor {
        return exec_elementwise.reduceBroadcast(&self.rt, x, target_shape);
    }
};

test {
    _ = @import("exec_tests.zig");
}
