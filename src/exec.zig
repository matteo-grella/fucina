//! `ExecContext`: the eager execution runtime. One struct carries the
//! substrate state (allocator, worker team, buffer pool, exec-scope
//! stack, MoE decode scratch) and every op as a method. Bodies live in
//! `exec/`: `exec/runtime.zig` holds the lifecycle, scope, pool, and tensor
//! allocation primitives; `exec/<domain>.zig` holds the ops, each taking
//! `*ExecContext` first. The struct body below is the alias registry (`pub
//! const add = exec_elementwise.add;`) grouped by domain; the section banners
//! name the file each group resolves to. Layer stack: docs/ARCHITECTURE.md.
const std = @import("std");
const backend_mod = @import("backend.zig");
const backend_ops = backend_mod.ops;
const fpenv = @import("fpenv.zig");
const tensor = @import("tensor.zig");
const thread = @import("thread.zig");
const tuning = @import("tuning.zig");

const exec_attention = @import("exec/attention.zig");
const exec_moe = @import("exec/moe.zig");
const exec_moe_chain = @import("exec/moe_chain.zig");
const exec_moe_gu = @import("exec/moe_gu.zig");
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
const Tensor = tensor.Tensor;

pub const MoeBatchProfile = exec_moe.MoeBatchProfile;
pub const delta_attention = @import("exec/delta_attention.zig");
pub const expert_store = @import("exec/expert_store.zig");

pub const parallel_dot_backward_branches = backend_mod.active_kind == .native and backend_mod.native_uses_blas;
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
pub const LinearDistillOptions = exec_loss.LinearDistillOptions;
pub const LinearDistillForward = exec_loss.LinearDistillForward;
pub const MseOptions = exec_loss.MseOptions;
pub const HuberOptions = exec_loss.HuberOptions;
pub const BceOptions = exec_loss.BceOptions;
pub const KlDivOptions = exec_loss.KlDivOptions;
pub const LossWrt = exec_loss.LossWrt;

pub const TopKResult = exec_stats.TopKResult;
pub const LogicalOp = exec_elementwise.LogicalOp;
pub const IntBitwiseOp = exec_elementwise.IntBitwiseOp;

pub const RouterTopKOptions = exec_topk.RouterTopKOptions;

/// Fake-quantization round trips (FP8-E4M3 / FP4-E2M1 microscaling groups,
/// Hadamard rotation, f16 round trip): quantization-aware inference numerics
/// over host slices — see `exec/fakequant.zig`.
pub const fakequant = @import("exec/fakequant.zig");

pub const StandardizeAccumulation = exec_stats.StandardizeAccumulation;
pub const StandardizeEpsMode = exec_stats.StandardizeEpsMode;
pub const StandardizeOptions = exec_stats.StandardizeOptions;

pub const GroupedCausalAttentionBackwardResult = exec_attention.GroupedCausalAttentionBackwardResult;

/// Masked-mean forward result: the per-lane means plus the per-lane counts of
/// selected elements (the mean's divisor, which the VJP reuses).
pub const MaskedMeanResult = exec_reduce.MaskedMeanResult;
pub const LayerNormAffineBackwardResult = exec_norm.LayerNormAffineBackwardResult;
pub const GroupNormBackwardResult = exec_norm.GroupNormBackwardResult;
pub const SnakeBackwardParamsResult = exec_elementwise.SnakeBackwardParamsResult;

const MoeDecodeScratch = exec_moe.MoeDecodeScratch;
pub const MatmulKind = exec_matmul.MatmulKind;
pub const BmmKind = exec_matmul.BmmKind;
pub const BmmBatchMode = exec_matmul.BmmBatchMode;
pub const BmmShape = exec_matmul.BmmShape;
pub const QuantizedMatmulOptions = exec_quant_matmul.QuantizedMatmulOptions;
const tailBroadcastInfo = exec_elementwise.tailBroadcastInfo;

/// Reusable transient-buffer pool. Defined in the `exec/buffer_pool.zig` leaf;
/// re-exported here so `exec.BufferPool` stays reachable and the `buffers`
/// field below can name it.
pub const BufferPool = exec_buffer_pool.BufferPool;

pub const ExecContext = struct {
    thread_safe_allocator: thread.ThreadSafeAllocator,
    allocator: Allocator,
    /// The worker team as published to kernel dispatch (`pc` snapshots it).
    /// Atomic: kernels may dispatch on other threads (dot-backward's
    /// `OneShotWorker`) while a lazy `tryWorkPool` retry publishes the pool;
    /// release/acquire so a racing first observer also sees `Pool.init`'s
    /// writes.
    parallel_pool: std.atomic.Value(?*thread.Pool) = .init(null),
    buffers: BufferPool,
    /// Per-context tuning overrides (`setTuning`); every field
    /// null = follow the process-wide gates (see src/tuning.zig).
    tuning: tuning.Overrides = .{},
    work_pool: thread.Pool,
    work_pool_ready: bool = false,
    work_pool_mutex: thread.Mutex = .{},
    dot_backward_worker: thread.OneShotWorker,
    dot_backward_worker_ready: bool = false,
    dot_backward_worker_mutex: thread.Mutex = .{},
    scope_entries: std.ArrayList(exec_runtime.ScopeEntry) = .empty,
    scope_depth: usize = 0,
    /// Speculation-verify kernel pinning (toggle through
    /// `pinRowwiseKernels`). While set, every batched
    /// quant-matmul entry reproduces the m == 1 kernel numerics exactly:
    /// the packed/plain entries run as independent single-row calls of
    /// themselves, the fused K-quant entries keep their per-row tail
    /// kernels for every row, and the batched MoE op skips the
    /// lane-packed Q8_Kx4 kernels. A verify batch then produces logits
    /// bit-identical to sequential decode — the property that keeps deep
    /// speculative drafting lossless. Batch matmul throughput is
    /// deliberately sacrificed while pinned (verify batches are small,
    /// and on streamed MoE the expert-fetch amortization — the part that
    /// pays — is preserved).
    pin_rowwise_kernels: bool = false,
    /// The IEEE floating-point environment observed when this context was
    /// created, or null where the target does not expose it. Every numeric
    /// contract the context goes on to honor (backend parity tolerances,
    /// thread-count invariance, checkpoint reproducibility) is stated under
    /// this environment; `checkFloatEnvironment` is how a caller confirms it
    /// still holds after code outside our control has run on the thread.
    fp_env_at_init: ?fpenv.Environment = null,
    /// Grow-only MoE-decode scratch (`exec/moe.zig`): carved by the
    /// single-row MoE entries under its own mutex.
    moe_scratch: MoeDecodeScratch = .{},

    pub const ScopeNodeDestroy = exec_runtime.ScopeNodeDestroy;
    pub const ExecScope = exec_runtime.ExecScope;
    pub const PreparedTensor = exec_runtime.PreparedTensor;
    pub const PreparedTensorOf = exec_runtime.PreparedTensorOf;

    pub const init = exec_runtime.init;
    pub const deinit = exec_runtime.deinit;
    pub const pinRowwiseKernels = exec_runtime.pinRowwiseKernels;
    pub const setTuning = exec_runtime.setTuning;
    pub const checkFloatEnvironment = exec_runtime.checkFloatEnvironment;
    pub const floatEnvironmentAtInit = exec_runtime.floatEnvironmentAtInit;

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
    pub const execScopeActive = exec_runtime.execScopeActive;
    pub const openExecScope = exec_runtime.openExecScope;
    pub const closeExecScope = exec_runtime.closeExecScope;
    pub const reserveScopeSlot = exec_runtime.reserveScopeSlot;
    pub const adoptScopeValueAssumeCapacity = exec_runtime.adoptScopeValueAssumeCapacity;
    pub const adoptScopeNodeAssumeCapacity = exec_runtime.adoptScopeNodeAssumeCapacity;
    pub const tryWorkPool = exec_runtime.tryWorkPool;
    pub const workPool = exec_runtime.workPool;
    pub const pc = exec_runtime.pc;
    pub const dotBackwardWorker = exec_runtime.dotBackwardWorker;
    pub const dispatchRange = exec_runtime.dispatchRange;
    pub const dispatchRangeCapped = exec_runtime.dispatchRangeCapped;

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

    pub const empty = exec_runtime.empty;
    pub const emptyRank = exec_runtime.emptyRank;
    pub const emptyTyped = exec_runtime.emptyTyped;
    pub const emptyRankTyped = exec_runtime.emptyRankTyped;
    pub const zeros = exec_runtime.zeros;
    pub const zerosRank = exec_runtime.zerosRank;
    pub const zerosTyped = exec_runtime.zerosTyped;
    pub const zerosRankTyped = exec_runtime.zerosRankTyped;
    pub const ones = exec_runtime.ones;
    pub const onesRank = exec_runtime.onesRank;
    pub const onesTyped = exec_runtime.onesTyped;
    pub const onesRankTyped = exec_runtime.onesRankTyped;
    pub const full = exec_runtime.full;
    pub const fullTyped = exec_runtime.fullTyped;
    pub const scalar = exec_runtime.scalar;
    pub const scalarTyped = exec_runtime.scalarTyped;
    pub const fromSlice = exec_runtime.fromSlice;
    pub const fromSliceRank = exec_runtime.fromSliceRank;
    pub const fromBorrowedSliceRank = exec_runtime.fromBorrowedSliceRank;
    pub const fromSliceTyped = exec_runtime.fromSliceTyped;
    pub const fromSliceRankTyped = exec_runtime.fromSliceRankTyped;
    pub const fromBorrowedSliceRankTyped = exec_runtime.fromBorrowedSliceRankTyped;
    pub const fromStorageSliceRankTyped = exec_runtime.fromStorageSliceRankTyped;
    pub const fromBorrowedStorageSliceRankTyped = exec_runtime.fromBorrowedStorageSliceRankTyped;
    pub const replace = exec_runtime.replace;
    pub const materialize = exec_runtime.materialize;
    pub const materializeTyped = exec_runtime.materializeTyped;
    pub const clone = exec_runtime.clone;
    pub const cloneTyped = exec_runtime.cloneTyped;
    pub const prepareContiguous = exec_runtime.prepareContiguous;
    pub const prepareContiguousTyped = exec_runtime.prepareContiguousTyped;
    pub const enableNativeVectorPoolForWork = exec_runtime.enableNativeVectorPoolForWork;
    pub const enableNativeMatmulPoolForWork = exec_runtime.enableNativeMatmulPoolForWork;
    pub const enableNativeTypedMatmulPoolForWork = exec_runtime.enableNativeTypedMatmulPoolForWork;

    // ----------------------------------------------------------------------
    // elementwise: pointwise arithmetic, activations, masks, casts (exec/elementwise.zig)
    // ----------------------------------------------------------------------
    pub const add = exec_elementwise.add;
    pub const sub = exec_elementwise.sub;
    pub const mul = exec_elementwise.mul;
    pub const div = exec_elementwise.div;
    pub const addRank = exec_elementwise.addRank;
    pub const addRankTyped = exec_elementwise.addRankTyped;
    pub const subRank = exec_elementwise.subRank;
    pub const subRankTyped = exec_elementwise.subRankTyped;
    pub const mulRank = exec_elementwise.mulRank;
    pub const mulRankTyped = exec_elementwise.mulRankTyped;
    pub const divRank = exec_elementwise.divRank;
    pub const divRankTyped = exec_elementwise.divRankTyped;
    pub const maxRank = exec_elementwise.maxRank;
    pub const maxRankTyped = exec_elementwise.maxRankTyped;
    pub const minRankTyped = exec_elementwise.minRankTyped;
    pub const divTruncRankTyped = exec_elementwise.divTruncRankTyped;
    pub const divFloorRankTyped = exec_elementwise.divFloorRankTyped;
    pub const remRankTyped = exec_elementwise.remRankTyped;
    pub const modRankTyped = exec_elementwise.modRankTyped;
    pub const bitwiseRankTyped = exec_elementwise.intBitwiseRankTyped;
    pub const minRank = exec_elementwise.minRank;
    pub const gatedRank = exec_elementwise.gatedRank;
    pub const gluRank = exec_elementwise.gluRank;
    pub const swigluRank = exec_elementwise.swigluRank;
    pub const gegluRank = exec_elementwise.gegluRank;
    pub const splitSwiGluAxisRank = exec_elementwise.splitSwiGluAxisRank;
    pub const relposShiftRank3 = exec_gather_scatter.relposShiftRank3;

    // ----------------------------------------------------------------------
    // elementwise: pointwise arithmetic, activations, masks, casts (exec/elementwise.zig)  [continued]
    // ----------------------------------------------------------------------
    pub const splitGluAxisRank = exec_elementwise.splitGluAxisRank;
    pub const splitSwiGluBackwardAxisRank = exec_elementwise.splitSwiGluBackwardAxisRank;
    pub const splitGluBackwardAxisRank = exec_elementwise.splitGluBackwardAxisRank;
    pub const addInPlace = exec_elementwise.addInPlace;
    pub const subInPlace = exec_elementwise.subInPlace;
    pub const mulInPlace = exec_elementwise.mulInPlace;
    pub const divInPlace = exec_elementwise.divInPlace;
    pub const takeAdd = exec_elementwise.takeAdd;
    pub const takeSub = exec_elementwise.takeSub;
    pub const takeMul = exec_elementwise.takeMul;
    pub const takeDiv = exec_elementwise.takeDiv;
    pub const takeScale = exec_elementwise.takeScale;
    pub const takeRelu = exec_elementwise.takeRelu;
    pub const takeSilu = exec_elementwise.takeSilu;
    pub const scale = exec_elementwise.scale;
    pub const addScalar = exec_elementwise.addScalar;
    pub const powScalar = exec_elementwise.powScalar;
    pub const where = exec_elementwise.where;
    pub const whereTyped = exec_elementwise.whereTyped;
    pub const maskedFill = exec_elementwise.maskedFill;
    pub const maskedFillTyped = exec_elementwise.maskedFillTyped;
    pub const compare = exec_elementwise.compare;
    pub const compareScalar = exec_elementwise.compareScalar;
    pub const compareIntTyped = exec_elementwise.compareIntTyped;
    pub const compareIntScalarTyped = exec_elementwise.compareIntScalarTyped;
    pub const logicalTyped = exec_elementwise.logicalTyped;
    pub const logicalNotTyped = exec_elementwise.logicalNotTyped;
    pub const addScaledInPlace = exec_elementwise.addScaledInPlace;
    pub const addAxisVectorInPlaceRank = exec_elementwise.addAxisVectorInPlaceRank;
    pub const addAxisVectorUnaryInPlaceRank = exec_elementwise.addAxisVectorUnaryInPlaceRank;
    pub const dropoutForward = exec_elementwise.dropoutForward;
    pub const dropoutBackward = exec_elementwise.dropoutBackward;

    // ----------------------------------------------------------------------
    // convert: dtype conversion and quantize/dequantize round trips (exec/convert.zig)
    // ----------------------------------------------------------------------
    pub const castTyped = exec_convert.castTyped;
    pub const castF32RowsToF16Into = exec_convert.castF32RowsToF16Into;
    pub const quantizeF32RowsToQ8_0Into = exec_convert.quantizeF32RowsToQ8_0Into;
    pub const dequantizeQ8_0RowsInto = exec_convert.dequantizeQ8_0RowsInto;
    pub const scaleTyped = exec_convert.scaleTyped;

    // ----------------------------------------------------------------------
    // conv: 1-D/2-D convolutions, im2col/col2im, Winograd (exec/conv.zig)
    // ----------------------------------------------------------------------
    pub const causalDepthwiseConv1dAxisRank = exec_conv.causalDepthwiseConv1dAxisRank;
    pub const causalDepthwiseConv1dBackwardInputAxisRank = exec_conv.causalDepthwiseConv1dBackwardInputAxisRank;
    pub const causalDepthwiseConv1dBackwardKernelAxisRank = exec_conv.causalDepthwiseConv1dBackwardKernelAxisRank;
    pub const causalConv1dAxisRank = exec_conv.causalConv1dAxisRank;
    pub const conv2d = exec_conv.conv2d;
    pub const conv2dRelu = exec_conv.conv2dRelu;

    /// Load-time prepared Winograd conv weight planes; `.empty` is valid and
    /// inert on every conv route. See `exec/conv.zig PreparedConvWeights`.
    pub const PreparedConvWeights = exec_conv.PreparedConvWeights;

    pub const prepareConv2dWeights = exec_conv.prepareConv2dWeights;
    pub const conv2dPrepared = exec_conv.conv2dPrepared;
    pub const conv2dPreparedRelu = exec_conv.conv2dPreparedRelu;
    pub const conv2dBackwardInput = exec_conv.conv2dBackwardInput;
    pub const conv2dBackwardWeight = exec_conv.conv2dBackwardWeight;
    pub const unfold = exec_conv.unfold;
    pub const fold = exec_conv.fold;

    // ----------------------------------------------------------------------
    // pool: 2-D pooling and upsampling (exec/pool.zig)
    // ----------------------------------------------------------------------
    pub const maxPool2d = exec_pool.maxPool2d;
    pub const avgPool2d = exec_pool.avgPool2d;
    pub const maxPool2dBackward = exec_pool.maxPool2dBackward;
    pub const avgPool2dBackward = exec_pool.avgPool2dBackward;
    pub const upsample2xNearest = exec_pool.upsample2xNearest;
    pub const upsample2xNearestBackward = exec_pool.upsample2xNearestBackward;

    // ----------------------------------------------------------------------
    // elementwise: pointwise arithmetic, activations, masks, casts (exec/elementwise.zig)  [continued]
    // ----------------------------------------------------------------------
    pub const preluChannels = exec_elementwise.preluChannels;
    pub const preluChannelsBackwardInput = exec_elementwise.preluChannelsBackwardInput;
    pub const preluChannelsBackwardAlpha = exec_elementwise.preluChannelsBackwardAlpha;
    pub const channelAffine = exec_elementwise.channelAffine;

    // ----------------------------------------------------------------------
    // conv: 1-D/2-D convolutions, im2col/col2im, Winograd (exec/conv.zig)  [continued]
    // ----------------------------------------------------------------------
    pub const causalConv1dBackwardInputAxisRank = exec_conv.causalConv1dBackwardInputAxisRank;
    pub const causalConv1dBackwardWeightAxisRank = exec_conv.causalConv1dBackwardWeightAxisRank;
    pub const groupedCausalConv1dAxisRank = exec_conv.groupedCausalConv1dAxisRank;
    pub const groupedCausalConv1dBackwardInputAxisRank = exec_conv.groupedCausalConv1dBackwardInputAxisRank;
    pub const groupedCausalConv1dBackwardWeightAxisRank = exec_conv.groupedCausalConv1dBackwardWeightAxisRank;
    pub const conv1dAxisRank = exec_conv.conv1dAxisRank;
    pub const conv1dBackwardInputAxisRank = exec_conv.conv1dBackwardInputAxisRank;
    pub const conv1dBackwardWeightAxisRank = exec_conv.conv1dBackwardWeightAxisRank;
    pub const col2im1dAxisRank = exec_conv.col2im1dAxisRank;
    pub const col2im1dBackwardAxisRank = exec_conv.col2im1dBackwardAxisRank;
    pub const convTranspose1d = exec_conv.convTranspose1d;

    // ----------------------------------------------------------------------
    // elementwise: pointwise arithmetic, activations, masks, casts (exec/elementwise.zig)  [continued]
    // ----------------------------------------------------------------------
    pub const unary = exec_elementwise.unary;
    pub const snakeRows = exec_elementwise.snakeRows;
    pub const snakeRowsBackwardInput = exec_elementwise.snakeRowsBackwardInput;
    pub const snakeRowsBackwardParams = exec_elementwise.snakeRowsBackwardParams;
    pub const relu = exec_elementwise.relu;
    pub const leakyRelu = exec_elementwise.leakyRelu;
    pub const exp = exec_elementwise.exp;
    pub const sqrt = exec_elementwise.sqrt;
    pub const rsqrt = exec_elementwise.rsqrt;
    pub const sigmoid = exec_elementwise.sigmoid;
    pub const silu = exec_elementwise.silu;
    pub const log = exec_elementwise.log;
    pub const neg = exec_elementwise.neg;
    pub const abs = exec_elementwise.abs;
    pub const sin = exec_elementwise.sin;
    pub const cos = exec_elementwise.cos;
    pub const tanh = exec_elementwise.tanh;
    pub const gelu = exec_elementwise.gelu;
    pub const quickGelu = exec_elementwise.quickGelu;
    pub const clamp = exec_elementwise.clamp;

    // ----------------------------------------------------------------------
    // reduce: sums, products, means, argmin/argmax, scans (exec/reduce.zig)
    // ----------------------------------------------------------------------
    pub const sum = exec_reduce.sum;
    pub const sumTyped = exec_reduce.sumTyped;
    pub const sumAxisRank = exec_reduce.sumAxisRank;
    pub const sumAxisRankTyped = exec_reduce.sumAxisRankTyped;
    pub const cumsumAxisRank = exec_reduce.cumsumAxisRank;
    pub const prodAxisRank = exec_reduce.prodAxisRank;
    pub const cumprodAxisRank = exec_reduce.cumprodAxisRank;
    pub const cumsumReverseAxisRank = exec_reduce.cumsumReverseAxisRank;
    pub const segmentSumAxisRank = exec_reduce.segmentSumAxisRank;
    pub const segmentBroadcastAxisRank = exec_reduce.segmentBroadcastAxisRank;
    pub const linearRecurrenceAxisRank = exec_reduce.linearRecurrenceAxisRank;
    pub const linearRecurrenceBackwardAxisRank = exec_reduce.linearRecurrenceBackwardAxisRank;
    pub const meanAxisRank = exec_reduce.meanAxisRank;
    pub const sumMaskedAxisRank = exec_reduce.sumMaskedAxisRank;
    pub const meanMaskedAxisRank = exec_reduce.meanMaskedAxisRank;
    pub const meanAxisRankTyped = exec_reduce.meanAxisRankTyped;

    // ----------------------------------------------------------------------
    // gather/scatter: indexing, embedding lookups, strided views (exec/gather_scatter.zig)
    // ----------------------------------------------------------------------
    pub const narrowAxisRank = exec_gather_scatter.narrowAxisRank;
    pub const narrowAxisRankTyped = exec_gather_scatter.narrowAxisRankTyped;
    pub const concatAxisRank = exec_gather_scatter.concatAxisRank;
    pub const concatAxisRankTyped = exec_gather_scatter.concatAxisRankTyped;
    pub const concatQuantizedRowsTyped = exec_gather_scatter.concatQuantizedRowsTyped;
    pub const padAxisRank = exec_gather_scatter.padAxisRank;
    pub const gatherAxisRank = exec_gather_scatter.gatherAxisRank;
    pub const gatherAxisRankTyped = exec_gather_scatter.gatherAxisRankTyped;
    pub const setSliceAxisRank = exec_gather_scatter.setSliceAxisRank;
    pub const setSliceAxisRankTyped = exec_gather_scatter.setSliceAxisRankTyped;
    pub const setRowsAxisRank = exec_gather_scatter.setRowsAxisRank;
    pub const setRowsAxisRankTyped = exec_gather_scatter.setRowsAxisRankTyped;
    pub const zeroSliceAxisRank = exec_gather_scatter.zeroSliceAxisRank;
    pub const zeroRowsAxisRank = exec_gather_scatter.zeroRowsAxisRank;
    pub const sliceGradientAxisRank = exec_gather_scatter.sliceGradientAxisRank;

    // ----------------------------------------------------------------------
    // stats: normalization statistics, standardize, moments (exec/stats.zig)
    // ----------------------------------------------------------------------
    pub const argmaxAxisRank = exec_stats.argmaxAxisRank;
    pub const maxAxisRank = exec_stats.maxAxisRank;
    pub const minAxisRank = exec_stats.minAxisRank;
    pub const maxMaskedAxisRank = exec_stats.maxMaskedAxisRank;
    pub const minMaskedAxisRank = exec_stats.minMaskedAxisRank;
    pub const varAxisRank = exec_stats.varAxisRank;
    pub const standardizeAxisRank = exec_stats.standardizeAxisRank;
    pub const standardizeAxisValidPrefixRank = exec_stats.standardizeAxisValidPrefixRank;
    pub const standardizeBackwardAxisRank = exec_stats.standardizeBackwardAxisRank;
    pub const topKAxisRank = exec_stats.topKAxisRank;
    pub const sortAxisRank = exec_stats.sortAxisRank;
    pub const routerTopK = exec_topk.routerTopK;

    // ----------------------------------------------------------------------
    // gather/scatter: indexing, embedding lookups, strided views (exec/gather_scatter.zig)  [continued]
    // ----------------------------------------------------------------------
    pub const scatterAddAxisRank = exec_gather_scatter.scatterAddAxisRank;
    pub const takeAlongAxisRank = exec_gather_scatter.takeAlongAxisRank;
    pub const scatterAddAlongAxisRank = exec_gather_scatter.scatterAddAlongAxisRank;
    pub const scatterAlongAxisRank = exec_gather_scatter.scatterAlongAxisRank;

    // ----------------------------------------------------------------------
    // softmax family (exec/softmax.zig)
    // ----------------------------------------------------------------------
    pub const softmaxAxisRank = exec_softmax.softmaxAxisRank;
    pub const logsumexpAxisRank = exec_softmax.logsumexpAxisRank;
    pub const logSoftmaxAxisRank = exec_softmax.logSoftmaxAxisRank;
    pub const softmaxExtAxisRank = exec_softmax.softmaxExtAxisRank;
    pub const softmaxBackwardAxisRank = exec_softmax.softmaxBackwardAxisRank;
    pub const softmaxExtBackwardAxisRank = exec_softmax.softmaxExtBackwardAxisRank;

    // ----------------------------------------------------------------------
    // norm: RMS/layer/group normalization and their fused arms (exec/norm.zig)
    // ----------------------------------------------------------------------
    pub const rmsNormAxisRank = exec_norm.rmsNormAxisRank;
    pub const rmsNormMulAxisRank = exec_norm.rmsNormMulAxisRank;
    pub const rmsNormMulAddAxisRank = exec_norm.rmsNormMulAddAxisRank;
    pub const rmsNormMulBackwardInputAxisRank = exec_norm.rmsNormMulBackwardInputAxisRank;
    pub const rmsNormMulBackwardWeightAxisRank = exec_norm.rmsNormMulBackwardWeightAxisRank;
    pub const rmsNormMulRopeAxisRankWithTable = exec_norm.rmsNormMulRopeAxisRankWithTable;
    pub const rmsNormBackwardAxisRank = exec_norm.rmsNormBackwardAxisRank;
    pub const layerNormAffineRows = exec_norm.layerNormAffineRows;
    pub const layerNormAxisRank = exec_norm.layerNormAxisRank;
    pub const groupNormAxisRank = exec_norm.groupNormAxisRank;
    pub const groupNormBackwardAxisRank = exec_norm.groupNormBackwardAxisRank;
    pub const layerNormAffineAxisRank = exec_norm.layerNormAffineAxisRank;
    pub const layerNormBackwardAxisRank = exec_norm.layerNormBackwardAxisRank;
    pub const layerNormAffineBackwardAxisRank = exec_norm.layerNormAffineBackwardAxisRank;

    // ----------------------------------------------------------------------
    // loss: cross-entropy family and reductions (exec/loss.zig)
    // ----------------------------------------------------------------------
    pub const crossEntropyLossAxisRank = exec_loss.crossEntropyLossAxisRank;
    pub const crossEntropyLossExAxisRank = exec_loss.crossEntropyLossExAxisRank;
    pub const crossEntropyLossExStatsAxisRank = exec_loss.crossEntropyLossExStatsAxisRank;
    pub const crossEntropyBackwardAxisRank = exec_loss.crossEntropyBackwardAxisRank;
    pub const crossEntropyBackwardExAxisRank = exec_loss.crossEntropyBackwardExAxisRank;
    pub const crossEntropyBackwardExStatsAxisRank = exec_loss.crossEntropyBackwardExStatsAxisRank;
    pub const crossEntropyBackwardExUpstreamAxisRank = exec_loss.crossEntropyBackwardExUpstreamAxisRank;
    pub const crossEntropyBackwardExUpstreamStatsAxisRank = exec_loss.crossEntropyBackwardExUpstreamStatsAxisRank;
    pub const linearCrossEntropyBackwardUpstream = exec_loss.linearCrossEntropyBackwardUpstream;
    pub const linearDistillLossStats = exec_loss.linearDistillLossStats;
    pub const linearDistillBackwardUpstream = exec_loss.linearDistillBackwardUpstream;
    pub const mseLoss = exec_loss.mseLoss;
    pub const mseBackwardUpstream = exec_loss.mseBackwardUpstream;
    pub const huberLoss = exec_loss.huberLoss;
    pub const huberBackwardUpstream = exec_loss.huberBackwardUpstream;
    pub const bceLoss = exec_loss.bceLoss;
    pub const bceBackwardUpstream = exec_loss.bceBackwardUpstream;
    pub const klDivLoss = exec_loss.klDivLoss;
    pub const klDivBackwardUpstream = exec_loss.klDivBackwardUpstream;

    // ----------------------------------------------------------------------
    // rope: rotary tables and their fused application (exec/rope.zig)
    // ----------------------------------------------------------------------
    pub const ropeAxisRank = exec_rope.ropeAxisRank;
    pub const prepareRopeTable = exec_rope.prepareRopeTable;
    pub const prepareRopeTableFactors = exec_rope.prepareRopeTableFactors;
    pub const prepareRopeTableRange = exec_rope.prepareRopeTableRange;
    pub const prepareRopeTableFactorsRange = exec_rope.prepareRopeTableFactorsRange;
    pub const prepareRopeTableInvFreqsF64 = exec_rope.prepareRopeTableInvFreqsF64;
    pub const yarnBlendInvFreqsF64 = exec_rope.yarnBlendInvFreqsF64;
    pub const ropeAxisRankWithTable = exec_rope.ropeAxisRankWithTable;
    pub const ropePartialAxisRank = exec_rope.ropePartialAxisRank;
    pub const ropePartialAxisRankWithTable = exec_rope.ropePartialAxisRankWithTable;

    // ----------------------------------------------------------------------
    // attention: the fused forward/backward kernels (exec/attention.zig)
    // ----------------------------------------------------------------------
    pub const groupedCausalAttention = exec_attention.groupedCausalAttention;
    pub const groupedCausalAttentionWindowed = exec_attention.groupedCausalAttentionWindowed;
    pub const groupedBidirectionalAttention = exec_attention.groupedBidirectionalAttention;
    pub const groupedBidirectionalAttentionBiased = exec_attention.groupedBidirectionalAttentionBiased;
    pub const groupedCausalAttentionStatsOut = exec_attention.groupedCausalAttentionStatsOut;
    pub const groupedCausalAttentionBackward = exec_attention.groupedCausalAttentionBackward;
    pub const groupedCausalAttentionF16Kv = exec_attention.groupedCausalAttentionF16Kv;
    pub const groupedBidirectionalAttentionF16Kv = exec_attention.groupedBidirectionalAttentionF16Kv;
    pub const groupedCausalAttentionF16KvWindowed = exec_attention.groupedCausalAttentionF16KvWindowed;
    pub const groupedCausalAttentionQ8Kv = exec_attention.groupedCausalAttentionQ8Kv;
    pub const groupedCausalAttentionQ8KvWindowed = exec_attention.groupedCausalAttentionQ8KvWindowed;
    pub const groupedCausalAttentionMultiF16Kv = exec_attention.groupedCausalAttentionMultiF16Kv;
    pub const groupedCausalAttentionMultiQ8Kv = exec_attention.groupedCausalAttentionMultiQ8Kv;
    pub const dot = exec_matmul.dot;
    pub const dotTyped = exec_matmul.dotTyped;
    pub const matmul = exec_matmul.matmul2D;
    pub const matmulTyped = exec_matmul.matmul2DTyped;
    pub const matmul2D = exec_matmul.matmul2D;
    pub const matmul2DAdd = exec_matmul.matmul2DAdd;
    pub const kdaRecurrent = delta_attention.kdaRecurrent;

    // ----------------------------------------------------------------------
    // matmul: dense contractions, batched and packed (exec/matmul.zig)
    // ----------------------------------------------------------------------
    pub const matmul2DTyped = exec_matmul.matmul2DTyped;
    pub const packMatmulRhsTyped = exec_matmul.packMatmulRhsTyped;
    pub const packDenseMatmulRhsTyped = exec_matmul.packDenseMatmulRhsTyped;
    pub const matmul2DWithPackedDenseRhs = exec_matmul.matmul2DWithPackedDenseRhs;
    pub const matmul2DWithPackedDenseRhsInto = exec_matmul.matmul2DWithPackedDenseRhsInto;
    pub const matmul2DWithPackedRhsTyped = exec_matmul.matmul2DWithPackedRhsTyped;

    // ----------------------------------------------------------------------
    // quantized matmul: per-format dense and decode arms (exec/quant_matmul.zig)
    // ----------------------------------------------------------------------
    pub const dequantizeTensorTyped = exec_quant_matmul.dequantizeTensorTyped;
    pub const getRowsQuantizedTyped = exec_quant_matmul.getRowsQuantizedTyped;
    pub const matmul2DWithQuantizedTensorRhs = exec_quant_matmul.matmul2DWithQuantizedTensorRhs;
    pub const matmul2DWithQuantizedTensorRhsOptions = exec_quant_matmul.matmul2DWithQuantizedTensorRhsOptions;
    pub const matmul2DWithQuantizedBlocksRhs = exec_quant_matmul.matmul2DWithQuantizedBlocksRhs;
    pub const matmul2DWithQuantizedBlocksRhsOptions = exec_quant_matmul.matmul2DWithQuantizedBlocksRhsOptions;
    pub const packMatmulRhsQ8_0x4 = exec_quant_matmul.packMatmulRhsQ8_0x4;
    pub const packMatmulRhsQ6_Kx4 = exec_quant_matmul.packMatmulRhsQ6_Kx4;
    pub const packMatmulRhsQ4_Kx4 = exec_quant_matmul.packMatmulRhsQ4_Kx4;
    pub const packMatmulRhsQ4_Kx8 = exec_quant_matmul.packMatmulRhsQ4_Kx8;
    pub const packMatmulRhsQ4_Kx2Mmla = exec_quant_matmul.packMatmulRhsQ4_Kx2Mmla;
    pub const packMatmulRhsQ5_Kx8 = exec_quant_matmul.packMatmulRhsQ5_Kx8;
    pub const matmul2DWithPackedQ8_0x4Rhs = exec_quant_matmul.matmul2DWithPackedQ8_0x4Rhs;
    pub const rmsNormMulMatmul2DWithPackedQ8_0x4Rhs = exec_quant_matmul.rmsNormMulMatmul2DWithPackedQ8_0x4Rhs;
    pub const rmsNormMulMatmul2DWithPackedQ4_Kx8Rhs = exec_quant_matmul.rmsNormMulMatmul2DWithPackedQ4_Kx8Rhs;
    pub const rmsNormMulMatmul2DWithPackedQ5_Kx8Rhs = exec_quant_matmul.rmsNormMulMatmul2DWithPackedQ5_Kx8Rhs;
    pub const rmsNormMulMatmul2DWithPackedQ6_Kx4Rhs = exec_quant_matmul.rmsNormMulMatmul2DWithPackedQ6_Kx4Rhs;
    pub const splitSwiGluMatmul2DWithPackedQ8_0x4Rhs = exec_quant_matmul.splitSwiGluMatmul2DWithPackedQ8_0x4Rhs;
    pub const splitSwiGluMatmul2DWithPackedQ4_Kx8Rhs = exec_quant_matmul.splitSwiGluMatmul2DWithPackedQ4_Kx8Rhs;
    pub const splitSwiGluMatmul2DWithPackedQ5_Kx8Rhs = exec_quant_matmul.splitSwiGluMatmul2DWithPackedQ5_Kx8Rhs;
    pub const splitSwiGluMatmul2DWithPackedQ6_Kx4Rhs = exec_quant_matmul.splitSwiGluMatmul2DWithPackedQ6_Kx4Rhs;
    pub const gegluQuantMatmul2DWithPackedQ8_0x4Rhs = exec_quant_matmul.gegluQuantMatmul2DWithPackedQ8_0x4Rhs;
    pub const matmul2DWithPackedQ6_Kx4Rhs = exec_quant_matmul.matmul2DWithPackedQ6_Kx4Rhs;
    pub const matmul2DWithPackedQ4_Kx4Rhs = exec_quant_matmul.matmul2DWithPackedQ4_Kx4Rhs;
    pub const matmul2DWithPackedQ4_Kx8Rhs = exec_quant_matmul.matmul2DWithPackedQ4_Kx8Rhs;
    pub const matmul2DWithPackedQ4_Kx2MmlaRhs = exec_quant_matmul.matmul2DWithPackedQ4_Kx2MmlaRhs;
    pub const matmul2DWithPackedQ5_Kx8Rhs = exec_quant_matmul.matmul2DWithPackedQ5_Kx8Rhs;
    pub const denseQuantMatmulGpu = exec_quant_matmul.denseQuantMatmulGpu;
    pub const foldedTernaryMatmulGpu = exec_quant_matmul.foldedTernaryMatmulGpu;
    pub const denseQuantMatmulGpuSharedInputBatch = exec_quant_matmul.denseQuantMatmulGpuSharedInputBatch;

    /// A Mixture-of-Experts projection: all experts of one layer's gate/up/down
    /// stacked into a single RHS buffer. The implementation lives in exec/moe.zig;
    /// this alias preserves the public ExecContext.MoeRhs surface.
    pub const MoeRhs = exec_moe.MoeRhs;

    /// Shared batched-MoE scheduling scaffolding (route plan, phase-chain
    /// machinery, chunk helpers, profile timers). Lives in exec/moe_chain.zig;
    /// exposed as an ExecContext decl so the gemma MoE engines at the llm
    /// layer reach the exact same types through the `fucina` root.
    pub const moe_chain = exec_moe_chain;

    /// Fused gate|up MoE expert kernels over raw GGUF stack layouts
    /// (`exec/moe_gu.zig`): the packed Q6_Kx4/Q8_0x4 arms, the raw
    /// Q6_K/Q4_K block arm, and the GPU batch path. The gemma family's
    /// tagged wrappers live in `llm/gemma/moe.zig`.
    pub const moe_gu = exec_moe_gu;

    // ----------------------------------------------------------------------
    // MoE gate/up: the fused gated-FFN expert kernels (exec/moe_gu.zig)
    // ----------------------------------------------------------------------
    pub const moeGuDecodePacked = exec_moe_gu.decodePacked;
    pub const moeGuBatchPacked = exec_moe_gu.batchPacked;
    pub const moeGuDecodeRaw = exec_moe_gu.decodeRaw;
    pub const moeGuBatchRaw = exec_moe_gu.batchRaw;

    // ----------------------------------------------------------------------
    // MoE: expert routing and the batched expert chains (exec/moe.zig)
    // ----------------------------------------------------------------------
    pub const lockMoeDecodeScratch = exec_moe.lockMoeDecodeScratch;
    pub const unlockMoeDecodeScratch = exec_moe.unlockMoeDecodeScratch;
    pub const MoeDecodeScratchView = exec_moe.MoeDecodeScratchView;
    pub const MoeDecodeChainScratchView = exec_moe.MoeDecodeChainScratchView;
    pub const carveMoeDecodeScratch = exec_moe.carveMoeDecodeScratch;
    pub const carveMoeDecodeChainScratch = exec_moe.carveMoeDecodeChainScratch;
    pub const moeExpertFfn = exec_moe.moeExpertFfn;
    pub const moeExpertFfnBatch = exec_moe.moeExpertFfnBatch;

    // ----------------------------------------------------------------------
    // matmul: dense contractions, batched and packed (exec/matmul.zig)  [continued]
    // ----------------------------------------------------------------------
    pub const matmulTransA = exec_matmul.matmulTransA;
    pub const matmulTransB = exec_matmul.matmulTransB;
    pub const matmulTransB2DWithF16Rhs = exec_matmul.matmulTransB2DWithF16Rhs;
    pub const matmulTransB2DWithBf16Rhs = exec_matmul.matmulTransB2DWithBf16Rhs;
    pub const bmm = exec_matmul.bmm;
    pub const bmmTransA = exec_matmul.bmmTransA;
    pub const bmmTransB = exec_matmul.bmmTransB;
    pub const reduceBroadcast = exec_elementwise.reduceBroadcast;
};

test {
    _ = @import("exec_tests.zig");
    _ = @import("exec/conv_tests.zig");
    _ = @import("exec/convert_tests.zig");
    _ = @import("exec/elementwise_tests.zig");
    _ = @import("exec/gather_scatter_tests.zig");
    _ = @import("exec/matmul_tests.zig");
    _ = @import("exec/moe_tests.zig");
    _ = @import("exec/norm_tests.zig");
    _ = @import("exec/reduce_tests.zig");
    _ = @import("exec/rope_tests.zig");
    _ = @import("exec/row_ops_tests.zig");
    _ = @import("exec/softmax_tests.zig");
    _ = @import("exec/stats_tests.zig");
    _ = @import("exec/delta_attention.zig");
    _ = @import("exec/expert_store.zig");
}
