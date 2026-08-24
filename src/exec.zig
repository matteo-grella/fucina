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
pub const RopeTableSpec = exec_rope.RopeTableSpec;
pub const RopePositions = exec_rope.RopePositions;
pub const RopeFreqs = exec_rope.RopeFreqs;
pub const RopeTheta = exec_rope.RopeTheta;

pub const SoftmaxExtOptions = exec_softmax.SoftmaxExtOptions;

/// Vector-norm order for the facade `norm`/`normAll` compositions:
/// `.l1` = Σ|x|, `.l2` = sqrt(Σ x²), `.inf` = max|x|.
pub const NormOrder = enum { l1, l2, inf };

pub const Reduction = exec_loss.Reduction;
pub const CrossEntropyOptions = exec_loss.CrossEntropyOptions;
pub const CrossEntropyUpstream = exec_loss.CrossEntropyUpstream;
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

pub const AttentionMask = exec_attention.AttentionMask;
pub const KvView = exec_attention.KvView;
pub const AttentionOptions = exec_attention.AttentionOptions;
pub const GroupedCausalAttentionBackwardResult = exec_attention.GroupedCausalAttentionBackwardResult;

/// Masked-mean forward result: the per-lane means plus the per-lane counts of
/// selected elements (the mean's divisor, which the VJP reuses).
pub const MaskedMeanResult = exec_reduce.MaskedMeanResult;
pub const AffineOptions = exec_norm.AffineOptions;
pub const AffineSlices = exec_norm.AffineSlices;
pub const AffineBackwardOptions = exec_norm.AffineBackwardOptions;
pub const AffineBackwardResult = exec_norm.AffineBackwardResult;
pub const RmsNormOptions = exec_norm.RmsNormOptions;
pub const RmsNormBackwardOptions = exec_norm.RmsNormBackwardOptions;
pub const RmsNormBackwardResult = exec_norm.RmsNormBackwardResult;
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

/// The eager execution runtime. PINNED after `init`: `allocator` embeds a
/// pointer to this context's own `thread_safe_allocator` field, so an
/// initialized context must never be copied or moved — keep it in a stable
/// stack frame or heap-allocate it, and hand out `*ExecContext` (every op
/// already takes the pointer). Field defaults below are the initial state;
/// `init` sets only what a fresh context must compute (see
/// exec/runtime.zig).
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
    pub const parallelMap = exec_runtime.parallelMap;

    pub fn classify(_: *const ExecContext, x: *const Tensor) LayoutClass {
        if (x.isScalar()) return .scalar;
        if (x.isContiguous()) return .contiguous;
        if (tailBroadcastInfo(x) != null) return .tail_broadcast;
        return .arbitrary;
    }

    pub const broadcastTo = exec_runtime.broadcastTo;

    pub const empty = exec_runtime.empty;
    pub const zeros = exec_runtime.zeros;
    pub const ones = exec_runtime.ones;
    pub const full = exec_runtime.full;
    pub const scalar = exec_runtime.scalar;
    pub const fromSlice = exec_runtime.fromSlice;
    pub const fromBorrowedSlice = exec_runtime.fromBorrowedSlice;
    pub const fromStorageSlice = exec_runtime.fromStorageSlice;
    pub const fromBorrowedStorageSlice = exec_runtime.fromBorrowedStorageSlice;
    pub const replace = exec_runtime.replace;
    pub const materialize = exec_runtime.materialize;
    pub const clone = exec_runtime.clone;
    pub const prepareContiguous = exec_runtime.prepareContiguous;
    pub const enableNativeVectorPoolForWork = exec_runtime.enableNativeVectorPoolForWork;
    pub const enableNativeMatmulPoolForWork = exec_runtime.enableNativeMatmulPoolForWork;

    // ----------------------------------------------------------------------
    // elementwise: pointwise arithmetic, activations, masks (exec/elementwise.zig)
    // ----------------------------------------------------------------------
    pub const elementwise = exec_elementwise.elementwise;
    pub const add = exec_elementwise.add;
    pub const sub = exec_elementwise.sub;
    pub const mul = exec_elementwise.mul;
    pub const div = exec_elementwise.div;
    pub const max = exec_elementwise.max;
    pub const min = exec_elementwise.min;
    pub const divTrunc = exec_elementwise.divTrunc;
    pub const divFloor = exec_elementwise.divFloor;
    pub const rem = exec_elementwise.rem;
    pub const mod = exec_elementwise.mod;
    pub const bitwise = exec_elementwise.bitwise;
    pub const gated = exec_elementwise.gated;
    pub const glu = exec_elementwise.glu;
    pub const swiglu = exec_elementwise.swiglu;
    pub const geglu = exec_elementwise.geglu;
    pub const splitSwiGlu = exec_elementwise.splitSwiGlu;
    pub const splitGlu = exec_elementwise.splitGlu;
    pub const splitSwiGluBackward = exec_elementwise.splitSwiGluBackward;
    pub const splitGluBackward = exec_elementwise.splitGluBackward;
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
    pub const maskedFill = exec_elementwise.maskedFill;
    pub const compare = exec_elementwise.compare;
    pub const compareScalar = exec_elementwise.compareScalar;
    pub const logical = exec_elementwise.logical;
    pub const logicalNot = exec_elementwise.logicalNot;
    pub const addScaledInPlace = exec_elementwise.addScaledInPlace;
    pub const addAxisVectorInPlace = exec_elementwise.addAxisVectorInPlace;
    pub const addAxisVectorUnaryInPlace = exec_elementwise.addAxisVectorUnaryInPlace;
    pub const dropoutForward = exec_elementwise.dropoutForward;
    pub const dropoutBackward = exec_elementwise.dropoutBackward;
    pub const preluChannels = exec_elementwise.preluChannels;
    pub const preluChannelsBackwardInput = exec_elementwise.preluChannelsBackwardInput;
    pub const preluChannelsBackwardAlpha = exec_elementwise.preluChannelsBackwardAlpha;
    pub const channelAffine = exec_elementwise.channelAffine;
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
    pub const reduceBroadcast = exec_elementwise.reduceBroadcast;

    // ----------------------------------------------------------------------
    // convert: dtype conversion and quantize/dequantize round trips (exec/convert.zig)
    // ----------------------------------------------------------------------
    pub const cast = exec_convert.cast;
    pub const castF32RowsToF16Into = exec_convert.castF32RowsToF16Into;
    pub const quantizeF32RowsToQ8_0Into = exec_convert.quantizeF32RowsToQ8_0Into;
    pub const dequantizeQ8_0RowsInto = exec_convert.dequantizeQ8_0RowsInto;

    // ----------------------------------------------------------------------
    // conv: 1-D/2-D convolutions, im2col/col2im, Winograd (exec/conv.zig)
    // ----------------------------------------------------------------------
    pub const causalDepthwiseConv1d = exec_conv.causalDepthwiseConv1d;
    pub const causalDepthwiseConv1dBackwardInput = exec_conv.causalDepthwiseConv1dBackwardInput;
    pub const causalDepthwiseConv1dBackwardKernel = exec_conv.causalDepthwiseConv1dBackwardKernel;
    pub const causalConv1d = exec_conv.causalConv1d;
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
    pub const causalConv1dBackwardInput = exec_conv.causalConv1dBackwardInput;
    pub const causalConv1dBackwardWeight = exec_conv.causalConv1dBackwardWeight;
    pub const groupedCausalConv1d = exec_conv.groupedCausalConv1d;
    pub const groupedCausalConv1dBackwardInput = exec_conv.groupedCausalConv1dBackwardInput;
    pub const groupedCausalConv1dBackwardWeight = exec_conv.groupedCausalConv1dBackwardWeight;
    pub const conv1d = exec_conv.conv1d;
    pub const conv1dBackwardInput = exec_conv.conv1dBackwardInput;
    pub const conv1dBackwardWeight = exec_conv.conv1dBackwardWeight;
    pub const col2im1d = exec_conv.col2im1d;
    pub const col2im1dBackward = exec_conv.col2im1dBackward;
    pub const convTranspose1d = exec_conv.convTranspose1d;

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
    // reduce: sums, products, means, scans, recurrences (exec/reduce.zig)
    // ----------------------------------------------------------------------
    pub const sum = exec_reduce.sum;
    pub const sumAxis = exec_reduce.sumAxis;
    pub const cumsum = exec_reduce.cumsum;
    pub const prod = exec_reduce.prod;
    pub const cumprod = exec_reduce.cumprod;
    pub const cumsumReverse = exec_reduce.cumsumReverse;
    pub const segmentSum = exec_reduce.segmentSum;
    pub const segmentBroadcast = exec_reduce.segmentBroadcast;
    pub const linearRecurrence = exec_reduce.linearRecurrence;
    pub const linearRecurrenceBackward = exec_reduce.linearRecurrenceBackward;
    pub const meanAxis = exec_reduce.meanAxis;
    pub const sumMasked = exec_reduce.sumMasked;
    pub const meanMasked = exec_reduce.meanMasked;

    // ----------------------------------------------------------------------
    // gather/scatter: indexing, embedding lookups, strided views (exec/gather_scatter.zig)
    // ----------------------------------------------------------------------
    pub const relposShift = exec_gather_scatter.relposShift;
    pub const narrowAxis = exec_gather_scatter.narrowAxis;
    pub const concatAxis = exec_gather_scatter.concatAxis;
    pub const concatQuantizedRows = exec_gather_scatter.concatQuantizedRows;
    pub const pad = exec_gather_scatter.pad;
    pub const gatherAxis = exec_gather_scatter.gatherAxis;
    pub const setSliceAxis = exec_gather_scatter.setSliceAxis;
    pub const setRows = exec_gather_scatter.setRows;
    pub const zeroSlice = exec_gather_scatter.zeroSlice;
    pub const zeroRows = exec_gather_scatter.zeroRows;
    pub const sliceGradient = exec_gather_scatter.sliceGradient;
    pub const scatterAdd = exec_gather_scatter.scatterAdd;
    pub const takeAlong = exec_gather_scatter.takeAlong;
    pub const scatterAddAlong = exec_gather_scatter.scatterAddAlong;
    pub const scatterAlong = exec_gather_scatter.scatterAlong;

    // ----------------------------------------------------------------------
    // stats: extrema, moments, standardize, top-k, sort (exec/stats.zig)
    // ----------------------------------------------------------------------
    pub const argmax = exec_stats.argmax;
    pub const maxAxis = exec_stats.maxAxis;
    pub const minAxis = exec_stats.minAxis;
    pub const maxMasked = exec_stats.maxMasked;
    pub const minMasked = exec_stats.minMasked;
    pub const varAxis = exec_stats.varAxis;
    pub const standardize = exec_stats.standardize;
    pub const standardizeValidPrefix = exec_stats.standardizeValidPrefix;
    pub const standardizeBackward = exec_stats.standardizeBackward;
    pub const topK = exec_stats.topK;
    pub const sort = exec_stats.sort;

    // ----------------------------------------------------------------------
    // router top-k: the MoE routing selector (exec/topk.zig)
    // ----------------------------------------------------------------------
    pub const routerTopK = exec_topk.routerTopK;

    // ----------------------------------------------------------------------
    // softmax family (exec/softmax.zig)
    // ----------------------------------------------------------------------
    pub const softmax = exec_softmax.softmax;
    pub const logsumexp = exec_softmax.logsumexp;
    pub const logSoftmax = exec_softmax.logSoftmax;
    pub const softmaxExt = exec_softmax.softmaxExt;
    pub const softmaxBackward = exec_softmax.softmaxBackward;

    // ----------------------------------------------------------------------
    // norm: RMS/layer/group normalization and their fused arms (exec/norm.zig)
    // ----------------------------------------------------------------------
    pub const rmsNorm = exec_norm.rmsNorm;
    pub const rmsNormBackward = exec_norm.rmsNormBackward;
    pub const rmsNormMulRopeWithTable = exec_norm.rmsNormMulRopeWithTable;
    pub const layerNorm = exec_norm.layerNorm;
    pub const layerNormRows = exec_norm.layerNormRows;
    pub const layerNormBackward = exec_norm.layerNormBackward;
    pub const groupNorm = exec_norm.groupNorm;
    pub const groupNormBackward = exec_norm.groupNormBackward;

    // ----------------------------------------------------------------------
    // loss: cross-entropy family and reductions (exec/loss.zig)
    // ----------------------------------------------------------------------
    pub const crossEntropyLoss = exec_loss.crossEntropyLoss;
    pub const crossEntropyBackward = exec_loss.crossEntropyBackward;
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
    pub const rope = exec_rope.rope;
    pub const prepareRopeTable = exec_rope.prepareRopeTable;
    pub const yarnBlendInvFreqsF64 = exec_rope.yarnBlendInvFreqsF64;
    pub const ropeWithTable = exec_rope.ropeWithTable;

    // ----------------------------------------------------------------------
    // attention: the fused forward/backward kernels (exec/attention.zig)
    // ----------------------------------------------------------------------
    pub const groupedAttention = exec_attention.groupedAttention;
    pub const groupedAttentionBackward = exec_attention.groupedAttentionBackward;

    // ----------------------------------------------------------------------
    // delta attention: the gated-DeltaNet recurrence (exec/delta_attention.zig)
    // ----------------------------------------------------------------------
    pub const kdaRecurrent = delta_attention.kdaRecurrent;

    // ----------------------------------------------------------------------
    // matmul: dense contractions, batched and packed (exec/matmul.zig)
    // ----------------------------------------------------------------------
    pub const dot = exec_matmul.dot;
    pub const matmul = exec_matmul.matmul;
    pub const matmul2DAdd = exec_matmul.matmul2DAdd;
    pub const packDenseMatmulRhs = exec_matmul.packDenseMatmulRhs;
    pub const matmulTransA = exec_matmul.matmulTransA;
    pub const matmulTransB = exec_matmul.matmulTransB;
    pub const matmulTransB2DWithHalfRhs = exec_matmul.matmulTransB2DWithHalfRhs;
    pub const bmm = exec_matmul.bmm;
    pub const bmmTransA = exec_matmul.bmmTransA;
    pub const bmmTransB = exec_matmul.bmmTransB;

    // ----------------------------------------------------------------------
    // quantized matmul: dense, packed and fused arms (exec/quant_matmul.zig)
    // ----------------------------------------------------------------------
    pub const dequantizeTensor = exec_quant_matmul.dequantizeTensor;
    pub const getRowsQuantized = exec_quant_matmul.getRowsQuantized;
    pub const matmul2DWithQuantizedTensorRhs = exec_quant_matmul.matmul2DWithQuantizedTensorRhs;
    pub const matmul2DWithQuantizedBlocksRhs = exec_quant_matmul.matmul2DWithQuantizedBlocksRhs;
    pub const packMatmulRhs = exec_quant_matmul.packMatmulRhs;
    pub const packMatmulRhsAs = exec_quant_matmul.packMatmulRhsAs;
    pub const matmulPacked = exec_quant_matmul.matmulPacked;
    pub const matmulPackedInto = exec_quant_matmul.matmulPackedInto;
    pub const rmsNormMulMatmulPacked = exec_quant_matmul.rmsNormMulMatmulPacked;
    pub const splitSwiGluMatmulPacked = exec_quant_matmul.splitSwiGluMatmulPacked;
    pub const gegluQuantMatmulPacked = exec_quant_matmul.gegluQuantMatmulPacked;
    pub const denseQuantMatmulGpu = exec_quant_matmul.denseQuantMatmulGpu;
    pub const foldedTernaryMatmulGpu = exec_quant_matmul.foldedTernaryMatmulGpu;
    pub const denseQuantMatmulGpuSharedInputBatch = exec_quant_matmul.denseQuantMatmulGpuSharedInputBatch;

    // ----------------------------------------------------------------------
    // MoE: routing scratch, expert chains, fused gate|up kernels (exec/moe.zig, exec/moe_chain.zig, exec/moe_gu.zig)
    // ----------------------------------------------------------------------
    /// A Mixture-of-Experts projection: all experts of one layer's gate/up/down
    /// stacked into a single RHS buffer. The implementation lives in exec/moe.zig;
    /// this alias preserves the public ExecContext.MoeRhs surface.
    pub const MoeRhs = exec_moe.MoeRhs;
    /// Shared batched-MoE scheduling scaffolding (route plan, phase-chain
    /// machinery, chunk helpers, profile timers). Lives in exec/moe_chain.zig;
    /// exposed as an ExecContext decl so the gemma MoE engines at the models
    /// layer reach the exact same types through the `fucina` root.
    pub const moe_chain = exec_moe_chain;
    /// Fused gate|up MoE expert kernels over raw GGUF stack layouts
    /// (`exec/moe_gu.zig`): the packed Q6_Kx4/Q8_0x4 arms, the raw
    /// Q6_K/Q4_K block arm, and the GPU batch path. The gemma family's
    /// tagged wrappers live in `models/gemma/moe.zig`.
    pub const moe_gu = exec_moe_gu;
    pub const moeGuDecodePacked = exec_moe_gu.decodePacked;
    pub const moeGuBatchPacked = exec_moe_gu.batchPacked;
    pub const moeGuDecodeRaw = exec_moe_gu.decodeRaw;
    pub const moeGuBatchRaw = exec_moe_gu.batchRaw;
    pub const lockMoeDecodeScratch = exec_moe.lockMoeDecodeScratch;
    pub const unlockMoeDecodeScratch = exec_moe.unlockMoeDecodeScratch;
    pub const MoeDecodeScratchView = exec_moe.MoeDecodeScratchView;
    pub const MoeDecodeChainScratchView = exec_moe.MoeDecodeChainScratchView;
    pub const carveMoeDecodeScratch = exec_moe.carveMoeDecodeScratch;
    pub const carveMoeDecodeChainScratch = exec_moe.carveMoeDecodeChainScratch;
    pub const moeExpertFfn = exec_moe.moeExpertFfn;
    pub const moeExpertFfnBatch = exec_moe.moeExpertFfnBatch;
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
}
