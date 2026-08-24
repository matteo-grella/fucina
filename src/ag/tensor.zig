//! The public tensor: `Tensor(spec)` names one comptime-fixed struct type
//! per (dtype, normalized tag list). Four branches share one set of
//! method mixins under `tensor/` and differ only in which methods they
//! alias, so the method set of every dtype is fixed at compile time and
//! `@hasDecl` is the contract:
//!
//! - f32 (`FloatTensor`): the differentiable tensor, every op;
//! - f16/bf16/f64 (`TypedFloatTensor`): forward math, views, and 16-bit
//!   autograd leaves (f32 gradients);
//! - integers, bool, f8 (`TypedScalarTensor`): constants with views,
//!   casts, and integer/mask math;
//! - block-quantized (`QuantizedTensor`): inference constants (dequantize,
//!   row gather, packed matmul RHS).
//!
//! Spellings that normalize to the same (dtype, tags) are the same type:
//! the dispatcher normalizes before instantiating a branch.

const std = @import("std");
const tensor_mod = @import("../tensor.zig");
const dtype_mod = @import("../dtype.zig");
const exec_mod = @import("../exec.zig");
const backend_mod = @import("../backend.zig");
const core = @import("core.zig");
const tags_mod = @import("../tags.zig");
const tag_ops = @import("../tag_ops.zig");

const RawTensor = tensor_mod.Tensor;
const DType = tensor_mod.DType;
const BlockQ8_0 = dtype_mod.BlockQ8_0;
const ExecContext = exec_mod.ExecContext;
const GradState = core.GradState;
const Tag = tags_mod.Tag;
const normalizeTags = tags_mod.normalizeTags;
const dtypeFromSpec = tags_mod.dtypeFromSpec;
const validateUniqueTags = tags_mod.validateUniqueTags;
const rawRank = tags_mod.rawRank;
const replaceTag = tags_mod.replaceTag;
const tagsEqual = tags_mod.tagsEqual;
const validateTensorRank = tag_ops.validateTensorRank;

const ag_file = @This();

const plumbing = @import("tensor/plumbing.zig").Mod(ag_file);
pub const einsumMany = plumbing.einsumMany;
const typed_constant = @import("tensor/typed_constant.zig").Mod(ag_file);

/// Input counts covered by the stack fast path for `concat`/`stack` metadata
/// temporaries (input pointers, backward parents/sizes); larger input counts
/// fall back to a heap allocation.
pub const concat_inline_inputs = 16;

/// Per-axis range for `slice` (torch/numpy basic slicing, positive steps):
/// `start`/`end` count from the end of the axis when negative and clamp to
/// it, `end = null` means the axis dim, `step` must be >= 1. Negative
/// steps are deliberately unsupported (torch rejects them in basic
/// indexing too; strides are unsigned, so a reversed view is not
/// representable; compose `flip`).
pub const SliceRange = struct {
    start: isize = 0,
    end: ?isize = null,
    step: usize = 1,
};

pub fn TopKResult(comptime tags_spec: anytype) type {
    const result_tags = normalizeTags(tags_spec);
    return struct {
        values: Tensor(result_tags),
        /// Source positions along the reduced/sorted axis: a constant i64
        /// tensor (exact for any axis length, torch's index dtype). Like
        /// every typed-constant result it is caller-owned even under an
        /// exec scope.
        indices: Tensor(.{ .dtype = .i64, .tags = result_tags }),

        pub fn deinit(self: *@This()) void {
            self.values.deinit();
            self.indices.deinit();
            self.* = undefined;
        }
    };
}

/// The public tensor type for a spec: a tag tuple (`.{ .batch, .d }`), a
/// numeric rank (`2`, generating `._0, ._1`), or a dtype struct
/// (`.{ .dtype = .i64, .tags = ... }` / `.{ .dtype = .f16, .rank = 2 }`).
/// The spec is normalized to (dtype, tag list) before the branch is
/// instantiated, so every spelling of the same tensor names the same type.
pub fn Tensor(comptime spec: anytype) type {
    const tensor_dtype = comptime dtypeFromSpec(spec);
    const tags = comptime normalizeTags(spec);
    if (comptime tensor_dtype == .f32) return FloatTensor(tags);
    if (comptime dtype_mod.isBlockQuantized(tensor_dtype)) return QuantizedTensor(tags, tensor_dtype);
    if (comptime dtype_mod.supportsForwardFloatMath(tensor_dtype)) return TypedFloatTensor(tags, tensor_dtype);
    return TypedScalarTensor(tags, tensor_dtype);
}

fn validateSpecTags(comptime tags: anytype) void {
    validateUniqueTags(tags);
    if (tags.len > tensor_mod.max_rank) @compileError("too many tensor tags");
}

/// The differentiable f32 branch.
fn FloatTensor(comptime tags: anytype) type {
    comptime validateSpecTags(tags);

    return struct {
        pub const axis_tags = tags;
        pub const tag_count = tags.len;
        pub const tensor_rank = rawRank(tags.len);
        pub const dtype = DType.f32;
        /// The enclosing module, handed to the method mixins so they need
        /// no upward import (keeps the production import graph acyclic;
        /// see arch-check).
        pub const ag_root = ag_file;

        value: RawTensor,
        grad_state: ?*GradState = null,
        /// True when an exec scope owns this tensor: the struct is a borrow
        /// and `deinit` is a safe no-op (arena-allocator semantics; the
        /// scope releases value and node at closeExecScope). Lets the same
        /// defer-deinit forward code run scoped (training) and unscoped
        /// (inference).
        scope_owned: bool = false,

        const Self = @This();

        // ---- common: lifetime, raw access, tag/shape queries ----
        const common = @import("tensor/common.zig").Ops(Self);
        pub const deinit = common.deinit;
        pub const asRawTensor = common.asRawTensor;
        pub const item = common.item;
        pub const data = common.data;
        pub const dataConst = common.dataConst;
        pub const copyTo = common.copyTo;
        pub const requiresGrad = common.requiresGrad;
        pub const axis = common.axis;
        pub const hasTag = common.hasTag;
        pub const dim = common.dim;
        pub const shape = common.shape;
        pub const isContiguous = common.isContiguous;

        // ---- autograd: leaves, gradients, backward ----
        const autograd_ops = @import("tensor/autograd.zig").Ops(Self);
        pub const variable = autograd_ops.variable;
        pub const variableFromSlice = autograd_ops.variableFromSlice;
        pub const zeroGrad = autograd_ops.zeroGrad;
        pub const grad = autograd_ops.grad;
        pub const gradView = autograd_ops.gradView;
        pub const backward = autograd_ops.backward;
        pub const backwardWithGrad = autograd_ops.backwardWithGrad;

        // ---- views: zero-copy views and data movement ----
        const views = @import("tensor/views.zig").Ops(Self);
        pub const materialize = views.materialize;
        pub const contiguous = views.contiguous;
        pub const detach = views.detach;
        pub const withTags = views.withTags;
        pub const viewWithStrides = views.viewWithStrides;
        pub const alignTo = views.alignTo;
        pub const permuteTo = views.permuteTo;
        pub const transpose = views.transpose;
        pub const insertAxis = views.insertAxis;
        pub const squeeze = views.squeeze;
        pub const split = views.split;
        pub const merge = views.merge;
        pub const reshape = views.reshape;
        pub const broadcastTo = views.broadcastTo;
        pub const flatten = views.flatten;
        pub const flip = views.flip;
        pub const roll = views.roll;
        pub const rollBy = views.rollBy;
        pub const narrow = views.narrow;
        pub const select = views.select;
        pub const sliceStep = views.sliceStep;
        pub const slice = views.slice;
        pub const gather = views.gather;
        pub const setSlice = views.setSlice;
        pub const setRows = views.setRows;
        pub const concat = views.concat;
        pub const stack = views.stack;
        pub const unbindInto = views.unbindInto;
        pub const repeatAxis = views.repeatAxis;

        // ---- creation: constructors and fills ----
        const creation_ops = @import("tensor/float/creation.zig").Ops(Self);
        pub const constant = creation_ops.constant;
        pub const fromTensor = creation_ops.fromTensor;
        pub const fromSlice = creation_ops.fromSlice;
        pub const fromBorrowedSlice = creation_ops.fromBorrowedSlice;
        pub const fromBorrowedConstSlice = creation_ops.fromBorrowedConstSlice;
        pub const empty = creation_ops.empty;
        pub const zeros = creation_ops.zeros;
        pub const ones = creation_ops.ones;
        pub const full = creation_ops.full;
        pub const scalar = creation_ops.scalar;
        pub const arange = creation_ops.arange;
        pub const linspace = creation_ops.linspace;
        pub const oneHot = creation_ops.oneHot;
        pub const rand = creation_ops.rand;
        pub const uniform = creation_ops.uniform;
        pub const randn = creation_ops.randn;
        pub const normal = creation_ops.normal;
        pub const bernoulli = creation_ops.bernoulli;
        pub const gumbel = creation_ops.gumbel;
        pub const eye = creation_ops.eye;
        pub const emptyLike = creation_ops.emptyLike;
        pub const zerosLike = creation_ops.zerosLike;
        pub const onesLike = creation_ops.onesLike;
        pub const fullLike = creation_ops.fullLike;

        // ---- matmul: contractions (dot/einsum, packed and ternary-STE RHS) ----
        const matmul_ops = @import("tensor/float/matmul.zig").Ops(Self);
        pub const matmul = matmul_ops.matmul;
        pub const dot = matmul_ops.dot;
        pub const addDot = matmul_ops.addDot;
        pub const einsum = matmul_ops.einsum;
        pub const dotTernarySte = matmul_ops.dotTernarySte;
        pub const dotPacked = matmul_ops.dotPacked;
        pub const packRhs = matmul_ops.packRhs;
        pub const rmsNormMulDotPacked = matmul_ops.rmsNormMulDotPacked;
        pub const splitSwiGluDotPacked = matmul_ops.splitSwiGluDotPacked;
        pub const gegluQuantDotPacked = matmul_ops.gegluQuantDotPacked;

        // ---- elementwise: pointwise arithmetic, activations, masks, casts ----
        const elementwise_ops = @import("tensor/float/elementwise.zig").Ops(Self);
        pub const addAxisVectorInPlace = elementwise_ops.addAxisVectorInPlace;
        pub const addAxisVectorUnaryInPlace = elementwise_ops.addAxisVectorUnaryInPlace;
        pub const addScaledInPlace = elementwise_ops.addScaledInPlace;
        pub const biasAdd = elementwise_ops.biasAdd;
        pub const where = elementwise_ops.where;
        pub const maskedFill = elementwise_ops.maskedFill;
        pub const compare = elementwise_ops.compare;
        pub const logicalAnd = elementwise_ops.logicalAnd;
        pub const logicalOr = elementwise_ops.logicalOr;
        pub const logicalXor = elementwise_ops.logicalXor;
        pub const logicalNot = elementwise_ops.logicalNot;
        pub const isnan = elementwise_ops.isnan;
        pub const isinf = elementwise_ops.isinf;
        pub const isfinite = elementwise_ops.isfinite;
        pub const prelu = elementwise_ops.prelu;
        pub const channelAffine = elementwise_ops.channelAffine;
        pub const to = elementwise_ops.to;
        pub const add = elementwise_ops.add;
        pub const sub = elementwise_ops.sub;
        pub const mul = elementwise_ops.mul;
        pub const div = elementwise_ops.div;
        pub const scale = elementwise_ops.scale;
        pub const takeAddNoGrad = elementwise_ops.takeAddNoGrad;
        pub const takeScaleNoGrad = elementwise_ops.takeScaleNoGrad;
        pub const addScalar = elementwise_ops.addScalar;
        pub const subScalar = elementwise_ops.subScalar;
        pub const divScalar = elementwise_ops.divScalar;
        pub const powScalar = elementwise_ops.powScalar;
        pub const log1p = elementwise_ops.log1p;
        pub const dropout = elementwise_ops.dropout;
        pub const snake = elementwise_ops.snake;
        pub const gated = elementwise_ops.gated;
        pub const glu = elementwise_ops.glu;
        pub const swiglu = elementwise_ops.swiglu;
        pub const geglu = elementwise_ops.geglu;
        pub const situ = elementwise_ops.situ;
        pub const splitGated = elementwise_ops.splitGated;
        pub const unary = elementwise_ops.unary;
        pub const elementalUnary = elementwise_ops.elementalUnary;
        pub const elementalBinary = elementwise_ops.elementalBinary;
        pub const maximum = elementwise_ops.maximum;
        pub const minimum = elementwise_ops.minimum;
        pub const pow = elementwise_ops.pow;
        pub const relu = elementwise_ops.relu;
        pub const leakyRelu = elementwise_ops.leakyRelu;
        pub const exp = elementwise_ops.exp;
        pub const sqrt = elementwise_ops.sqrt;
        pub const rsqrt = elementwise_ops.rsqrt;
        pub const sigmoid = elementwise_ops.sigmoid;
        pub const silu = elementwise_ops.silu;
        pub const log = elementwise_ops.log;
        pub const neg = elementwise_ops.neg;
        pub const abs = elementwise_ops.abs;
        pub const sin = elementwise_ops.sin;
        pub const cos = elementwise_ops.cos;
        pub const tanh = elementwise_ops.tanh;
        pub const fastTanh = elementwise_ops.fastTanh;
        pub const softcap30 = elementwise_ops.softcap30;
        pub const softcap15 = elementwise_ops.softcap15;
        pub const gelu = elementwise_ops.gelu;
        pub const quickGelu = elementwise_ops.quickGelu;
        pub const elu = elementwise_ops.elu;
        pub const geluErf = elementwise_ops.geluErf;
        pub const erf = elementwise_ops.erf;
        pub const floor = elementwise_ops.floor;
        pub const ceil = elementwise_ops.ceil;
        pub const round = elementwise_ops.round;
        pub const sign = elementwise_ops.sign;
        pub const reciprocal = elementwise_ops.reciprocal;
        pub const clamp = elementwise_ops.clamp;
        pub const clampMin = elementwise_ops.clampMin;
        pub const clampMax = elementwise_ops.clampMax;

        // ---- conv: convolutions and unfold/fold ----
        const conv_ops = @import("tensor/float/conv.zig").Ops(Self);
        pub const conv2d = conv_ops.conv2d;
        pub const conv2dRelu = conv_ops.conv2dRelu;
        pub const prepareConv2dWeights = conv_ops.prepareConv2dWeights;
        pub const conv2dPrepared = conv_ops.conv2dPrepared;
        pub const conv2dPreparedRelu = conv_ops.conv2dPreparedRelu;
        pub const unfold = conv_ops.unfold;
        pub const fold = conv_ops.fold;
        pub const causalDepthwiseConv1d = conv_ops.causalDepthwiseConv1d;
        pub const causalConv1d = conv_ops.causalConv1d;
        pub const groupedCausalConv1d = conv_ops.groupedCausalConv1d;
        pub const conv1d = conv_ops.conv1d;
        pub const convTranspose1d = conv_ops.convTranspose1d;

        // ---- pool: pooling and upsampling ----
        const pool_ops = @import("tensor/float/pool.zig").Ops(Self);
        pub const maxPool2d = pool_ops.maxPool2d;
        pub const avgPool2d = pool_ops.avgPool2d;
        pub const upsample2xNearest = pool_ops.upsample2xNearest;

        // ---- gather_scatter: indexed reads and writes beyond the views ----
        const gather_scatter_ops = @import("tensor/float/gather_scatter.zig").Ops(Self);
        pub const zeroSlice = gather_scatter_ops.zeroSlice;
        pub const zeroRows = gather_scatter_ops.zeroRows;
        pub const relposShift = gather_scatter_ops.relposShift;
        pub const indexSelect = gather_scatter_ops.indexSelect;
        pub const maskedSelect = gather_scatter_ops.maskedSelect;
        pub const nonzero = gather_scatter_ops.nonzero;
        pub const maskedScatter = gather_scatter_ops.maskedScatter;
        pub const indexAdd = gather_scatter_ops.indexAdd;
        pub const takeAlongAxis = gather_scatter_ops.takeAlongAxis;
        pub const scatterAdd = gather_scatter_ops.scatterAdd;
        pub const scatter = gather_scatter_ops.scatter;

        // ---- reduce: reductions, scans, linear recurrence ----
        const reduce_ops = @import("tensor/float/reduce.zig").Ops(Self);
        pub const any = reduce_ops.any;
        pub const all = reduce_ops.all;
        pub const anyAll = reduce_ops.anyAll;
        pub const allAll = reduce_ops.allAll;
        pub const sum = reduce_ops.sum;
        pub const mean = reduce_ops.mean;
        pub const cumsum = reduce_ops.cumsum;
        pub const segmentSum = reduce_ops.segmentSum;
        pub const linearRecurrence = reduce_ops.linearRecurrence;
        pub const prod = reduce_ops.prod;
        pub const cumprod = reduce_ops.cumprod;
        pub const sumAll = reduce_ops.sumAll;
        pub const sumMany = reduce_ops.sumMany;

        // ---- stats: variance/standardize, extrema, argmax, multinomial ----
        const stats_ops = @import("tensor/float/stats.zig").Ops(Self);
        pub const variance = stats_ops.variance;
        pub const standardizeAxis = stats_ops.standardizeAxis;
        pub const argmax = stats_ops.argmax;
        pub const multinomial = stats_ops.multinomial;
        pub const max = stats_ops.max;
        pub const min = stats_ops.min;

        // ---- topk: top-k, sort, router top-k ----
        const topk_ops = @import("tensor/float/topk.zig").Ops(Self);
        pub const topK = topk_ops.topK;
        pub const sort = topk_ops.sort;
        pub const argsort = topk_ops.argsort;
        pub const routerTopK = topk_ops.routerTopK;

        // ---- shape: the f32-only structural ops (fills, diagonals, bands) ----
        const shape_ops = @import("tensor/float/shape.zig").Ops(Self);
        pub const shiftBy = shape_ops.shiftBy;
        pub const diagonal = shape_ops.diagonal;
        pub const trace = shape_ops.trace;
        pub const diag = shape_ops.diag;
        pub const bandPart = shape_ops.bandPart;
        pub const tril = shape_ops.tril;
        pub const triu = shape_ops.triu;
        pub const diagEmbed = shape_ops.diagEmbed;
        pub const pad = shape_ops.pad;
        pub const zeroPad2d = shape_ops.zeroPad2d;
        pub const constantPad2d = shape_ops.constantPad2d;

        // ---- softmax: softmax family ----
        const softmax_ops = @import("tensor/float/softmax.zig").Ops(Self);
        pub const logsumexp = softmax_ops.logsumexp;
        pub const logSoftmax = softmax_ops.logSoftmax;
        pub const softmax = softmax_ops.softmax;

        // ---- norm: normalization ops and norms ----
        const norm_ops = @import("tensor/float/norm.zig").Ops(Self);
        pub const groupNorm = norm_ops.groupNorm;
        pub const rmsNorm = norm_ops.rmsNorm;
        pub const rmsNormMul = norm_ops.rmsNormMul;
        pub const rmsNormMulAdd = norm_ops.rmsNormMulAdd;
        pub const rmsNormMulRopeHalfPrepared = norm_ops.rmsNormMulRopeHalfPrepared;
        pub const layerNorm = norm_ops.layerNorm;
        pub const l2Normalize = norm_ops.l2Normalize;
        pub const norm = norm_ops.norm;
        pub const normAll = norm_ops.normAll;
        pub const cosineSimilarity = norm_ops.cosineSimilarity;

        // ---- loss: loss heads ----
        const loss_ops = @import("tensor/float/loss.zig").Ops(Self);
        pub const crossEntropy = loss_ops.crossEntropy;
        pub const linearCrossEntropy = loss_ops.linearCrossEntropy;
        pub const linearDistill = loss_ops.linearDistill;
        pub const mseLoss = loss_ops.mseLoss;
        pub const huberLoss = loss_ops.huberLoss;
        pub const bceLoss = loss_ops.bceLoss;
        pub const klDivLoss = loss_ops.klDivLoss;
        pub const nllLoss = loss_ops.nllLoss;

        // ---- rope: rotary position embedding ----
        const rope_ops = @import("tensor/float/rope.zig").Ops(Self);
        pub const rope = rope_ops.rope;

        // ---- attention: fused grouped causal attention ----
        const attention_ops = @import("tensor/float/attention.zig").Ops(Self);
        pub const groupedAttention = attention_ops.groupedAttention;
    };
}

/// The typed float branch (f16/bf16/f64): forward math over the stored
/// dtype (f16/bf16 widen through f32 where no native kernel exists), every
/// view, and, on f16/bf16, trainable leaves with f32 gradients.
fn TypedFloatTensor(comptime tags: anytype, comptime tensor_dtype: DType) type {
    comptime validateSpecTags(tags);

    return struct {
        pub const axis_tags = tags;
        pub const tag_count = tags.len;
        pub const tensor_rank = rawRank(tags.len);
        pub const dtype = tensor_dtype;
        pub const ag_root = ag_file;

        value: tensor_mod.TensorOf(tensor_dtype),
        grad_state: ?*GradState = null,
        scope_owned: bool = false,

        const Self = @This();

        // ---- common ----
        const common = @import("tensor/common.zig").Ops(Self);
        pub const deinit = common.deinit;
        pub const asRawTensor = common.asRawTensor;
        pub const item = common.item;
        pub const data = common.data;
        pub const dataConst = common.dataConst;
        pub const copyTo = common.copyTo;
        pub const requiresGrad = common.requiresGrad;
        pub const axis = common.axis;
        pub const hasTag = common.hasTag;
        pub const dim = common.dim;
        pub const shape = common.shape;
        pub const isContiguous = common.isContiguous;

        // ---- autograd leaves (f16/bf16): no backward, a 16-bit tensor is never a loss ----
        const autograd_ops = @import("tensor/autograd.zig").Ops(Self);
        pub const variable = autograd_ops.variable;
        pub const variableFromSlice = autograd_ops.variableFromSlice;
        pub const zeroGrad = autograd_ops.zeroGrad;
        pub const grad = autograd_ops.grad;
        pub const gradView = autograd_ops.gradView;

        // ---- views ----
        const views = @import("tensor/views.zig").Ops(Self);
        pub const materialize = views.materialize;
        pub const contiguous = views.contiguous;
        pub const detach = views.detach;
        pub const withTags = views.withTags;
        pub const viewWithStrides = views.viewWithStrides;
        pub const alignTo = views.alignTo;
        pub const permuteTo = views.permuteTo;
        pub const transpose = views.transpose;
        pub const insertAxis = views.insertAxis;
        pub const squeeze = views.squeeze;
        pub const split = views.split;
        pub const merge = views.merge;
        pub const reshape = views.reshape;
        pub const broadcastTo = views.broadcastTo;
        pub const flatten = views.flatten;
        pub const flip = views.flip;
        pub const roll = views.roll;
        pub const rollBy = views.rollBy;
        pub const narrow = views.narrow;
        pub const select = views.select;
        pub const sliceStep = views.sliceStep;
        pub const slice = views.slice;
        pub const gather = views.gather;
        pub const setSlice = views.setSlice;
        pub const setRows = views.setRows;
        pub const concat = views.concat;
        pub const stack = views.stack;
        pub const unbindInto = views.unbindInto;
        pub const repeatAxis = views.repeatAxis;

        // ---- creation ----
        const base = typed_constant.TypedConstantBase(Self, tags, tensor_dtype);
        pub const constant = base.constant;
        pub const fromTensor = base.fromTensor;
        pub const fromSlice = base.fromSlice;
        pub const fromBorrowedConstSlice = base.fromBorrowedConstSlice;
        pub const empty = base.empty;
        pub const zeros = base.zeros;
        pub const ones = base.ones;
        pub const emptyLike = base.emptyLike;
        pub const zerosLike = base.zerosLike;
        pub const onesLike = base.onesLike;

        /// Snapshot this rank-2 f16/bf16 `[out, contract]` weight as f32
        /// output-row panels for a FloatTensor `dotPacked`. Widening happens
        /// once here; the returned resource is caller-owned and no-grad.
        pub fn packRhs(self: *const Self, ctx: *ExecContext) !PackedRhs(tensor_dtype) {
            comptime {
                if (tag_count != 2) @compileError("packRhs requires a rank-2 tensor");
                if (tensor_dtype != .f16 and tensor_dtype != .bf16)
                    @compileError("dense packRhs supports f32, f16, and bf16 weights");
            }
            if (self.requiresGrad()) return error.GradientPackedMatmulUnsupported;
            return ctx.packDenseMatmulRhs(tensor_dtype, self.asRawTensor());
        }

        // ---- native typed math (every typed float dtype) ----
        pub const to = typed_constant.typedConstantTo;
        pub const add = typed_constant.typedConstantAdd;
        pub const sub = typed_constant.typedConstantSub;
        pub const mul = typed_constant.typedConstantMul;
        pub const div = typed_constant.typedConstantDiv;
        pub const sum = typed_constant.typedConstantSum;
        pub const mean = typed_constant.typedConstantMean;
        pub const sumAll = typed_constant.typedConstantSumAll;
        pub const dot = typed_constant.typedConstantDot;
        pub const scale = typed_constant.typedConstantScale;
        pub const divScalar = typed_constant.typedConstantDivScalar;

        // ---- widened forward math (f16/bf16 only: f32 compute, one final round) ----
        pub const unary = typed_constant.typedConstantUnary;
        pub const relu = typed_constant.TypedUnaryMethod(.relu).call;
        pub const exp = typed_constant.TypedUnaryMethod(.exp).call;
        pub const sqrt = typed_constant.TypedUnaryMethod(.sqrt).call;
        pub const rsqrt = typed_constant.TypedUnaryMethod(.rsqrt).call;
        pub const sigmoid = typed_constant.TypedUnaryMethod(.sigmoid).call;
        pub const silu = typed_constant.TypedUnaryMethod(.silu).call;
        pub const log = typed_constant.TypedUnaryMethod(.log).call;
        pub const log1p = typed_constant.TypedUnaryMethod(.log1p).call;
        pub const neg = typed_constant.TypedUnaryMethod(.neg).call;
        pub const abs = typed_constant.TypedUnaryMethod(.abs).call;
        pub const sin = typed_constant.TypedUnaryMethod(.sin).call;
        pub const cos = typed_constant.TypedUnaryMethod(.cos).call;
        pub const tanh = typed_constant.TypedUnaryMethod(.tanh).call;
        pub const fastTanh = typed_constant.TypedUnaryMethod(.fast_tanh).call;
        pub const softcap30 = typed_constant.TypedUnaryMethod(.softcap_30).call;
        pub const softcap15 = typed_constant.TypedUnaryMethod(.softcap_15).call;
        pub const gelu = typed_constant.TypedUnaryMethod(.gelu).call;
        pub const quickGelu = typed_constant.TypedUnaryMethod(.quick_gelu).call;
        pub const elu = typed_constant.TypedUnaryMethod(.elu).call;
        pub const geluErf = typed_constant.TypedUnaryMethod(.gelu_erf).call;
        pub const erf = typed_constant.TypedUnaryMethod(.erf).call;
        pub const floor = typed_constant.TypedUnaryMethod(.floor).call;
        pub const ceil = typed_constant.TypedUnaryMethod(.ceil).call;
        pub const round = typed_constant.TypedUnaryMethod(.round).call;
        pub const sign = typed_constant.TypedUnaryMethod(.sign).call;
        pub const reciprocal = typed_constant.TypedUnaryMethod(.reciprocal).call;
        pub const leakyRelu = typed_constant.typedConstantLeakyRelu;
        pub const clamp = typed_constant.typedConstantClamp;
        pub const addScalar = typed_constant.typedConstantAddScalar;
        pub const subScalar = typed_constant.typedConstantSubScalar;
        pub const powScalar = typed_constant.typedConstantPowScalar;
        pub const maximum = typed_constant.typedConstantMaximum;
        pub const minimum = typed_constant.typedConstantMinimum;
        pub const gated = typed_constant.typedConstantGated;
        pub const glu = typed_constant.typedConstantGlu;
        pub const swiglu = typed_constant.typedConstantSwiglu;
        pub const geglu = typed_constant.typedConstantGeglu;
        pub const softmax = typed_constant.typedConstantSoftmax;
        pub const logSoftmax = typed_constant.typedConstantLogSoftmax;
        pub const rmsNorm = typed_constant.typedConstantRmsNorm;
        pub const rmsNormMul = typed_constant.typedConstantRmsNormMul;
        pub const layerNorm = typed_constant.typedConstantLayerNorm;
        pub const cumsum = typed_constant.typedConstantCumsum;
        pub const cumprod = typed_constant.typedConstantCumprod;
        pub const where = typed_constant.typedConstantWhere;
        pub const maskedFill = typed_constant.typedConstantMaskedFill;
        pub const compare = typed_constant.typedConstantCompare;
        pub const pad = typed_constant.typedConstantPad;
        pub const einsum = typed_constant.typedConstantEinsum;

        // ---- widened reductions (f16/bf16 only; f32 result per the dtype policy) ----
        pub const max = typed_constant.typedConstantMax;
        pub const min = typed_constant.typedConstantMin;
        pub const argmax = typed_constant.typedConstantArgmax;
        pub const prod = typed_constant.typedConstantProd;
        pub const variance = typed_constant.typedConstantVariance;
        pub const logsumexp = typed_constant.typedConstantLogsumexp;
    };
}

/// The typed scalar branch (integers, bool, the f8 storage floats):
/// constants with every view, scalar casts, and the integer/mask math.
fn TypedScalarTensor(comptime tags: anytype, comptime tensor_dtype: DType) type {
    comptime validateSpecTags(tags);

    return struct {
        pub const axis_tags = tags;
        pub const tag_count = tags.len;
        pub const tensor_rank = rawRank(tags.len);
        pub const dtype = tensor_dtype;
        pub const ag_root = ag_file;

        value: tensor_mod.TensorOf(tensor_dtype),

        const Self = @This();

        // ---- common ----
        const common = @import("tensor/common.zig").Ops(Self);
        pub const deinit = common.deinit;
        pub const asRawTensor = common.asRawTensor;
        pub const item = common.item;
        pub const data = common.data;
        pub const dataConst = common.dataConst;
        pub const copyTo = common.copyTo;
        pub const requiresGrad = common.requiresGrad;
        pub const axis = common.axis;
        pub const hasTag = common.hasTag;
        pub const dim = common.dim;
        pub const shape = common.shape;
        pub const isContiguous = common.isContiguous;

        // ---- views ----
        const views = @import("tensor/views.zig").Ops(Self);
        pub const materialize = views.materialize;
        pub const contiguous = views.contiguous;
        pub const detach = views.detach;
        pub const withTags = views.withTags;
        pub const viewWithStrides = views.viewWithStrides;
        pub const alignTo = views.alignTo;
        pub const permuteTo = views.permuteTo;
        pub const transpose = views.transpose;
        pub const insertAxis = views.insertAxis;
        pub const squeeze = views.squeeze;
        pub const split = views.split;
        pub const merge = views.merge;
        pub const reshape = views.reshape;
        pub const broadcastTo = views.broadcastTo;
        pub const flatten = views.flatten;
        pub const flip = views.flip;
        pub const roll = views.roll;
        pub const rollBy = views.rollBy;
        pub const narrow = views.narrow;
        pub const select = views.select;
        pub const sliceStep = views.sliceStep;
        pub const slice = views.slice;
        pub const gather = views.gather;
        pub const setSlice = views.setSlice;
        pub const setRows = views.setRows;
        pub const concat = views.concat;
        pub const stack = views.stack;
        pub const unbindInto = views.unbindInto;
        pub const repeatAxis = views.repeatAxis;

        // ---- creation ----
        const base = typed_constant.TypedConstantBase(Self, tags, tensor_dtype);
        pub const constant = base.constant;
        pub const fromTensor = base.fromTensor;
        pub const fromSlice = base.fromSlice;
        pub const fromBorrowedConstSlice = base.fromBorrowedConstSlice;
        pub const empty = base.empty;
        pub const zeros = base.zeros;
        pub const ones = base.ones;
        pub const emptyLike = base.emptyLike;
        pub const zerosLike = base.zerosLike;
        pub const onesLike = base.onesLike;
        pub const randint = base.randint;
        pub const randperm = base.randperm;
        pub const bandMask = base.bandMask;

        // ---- integer forward math (docs/reference/04-tensor-operations.md): wrapping
        // two's-complement pointwise, explicit division/remainder, bitwise
        // combinators, i64-returning reductions, and scalar casts. On
        // `.bool` the arithmetic entries are compile errors; only `to` and
        // the counting `sum`/`sumAll` apply. ----
        pub const to = typed_constant.typedConstantTo;
        pub const add = typed_constant.typedConstantAdd;
        pub const sub = typed_constant.typedConstantSub;
        pub const mul = typed_constant.typedConstantMul;
        pub const maximum = typed_constant.typedConstantMaximum;
        pub const minimum = typed_constant.typedConstantMinimum;
        pub const divTrunc = typed_constant.typedConstantDivTrunc;
        pub const divFloor = typed_constant.typedConstantDivFloor;
        pub const rem = typed_constant.typedConstantRem;
        pub const mod = typed_constant.typedConstantMod;
        pub const bitAnd = typed_constant.typedConstantBitAnd;
        pub const bitOr = typed_constant.typedConstantBitOr;
        pub const bitXor = typed_constant.typedConstantBitXor;
        pub const sum = typed_constant.typedConstantSum;
        pub const sumAll = typed_constant.typedConstantSumAll;

        // ---- masks: integer `compare` is exact at any magnitude; the logical
        // combinators live on the `.bool` branch. ----
        pub const compare = typed_constant.typedConstantCompare;
        pub const logicalAnd = typed_constant.typedConstantLogicalAnd;
        pub const logicalOr = typed_constant.typedConstantLogicalOr;
        pub const logicalXor = typed_constant.typedConstantLogicalXor;
        pub const logicalNot = typed_constant.typedConstantLogicalNot;
    };
}

/// The block-quantized branch: inference constants. Blocks pack the last
/// axis, so the view set is the block-safe subset (tag rename, row concat,
/// materialize) plus dequantization, row gather, and packed matmul RHS.
fn QuantizedTensor(comptime tags: anytype, comptime tensor_dtype: DType) type {
    comptime validateSpecTags(tags);

    const RawTypedTensor = tensor_mod.TensorOf(tensor_dtype);
    const Elem = dtype_mod.Storage(tensor_dtype);

    return struct {
        pub const axis_tags = tags;
        pub const tag_count = tags.len;
        pub const tensor_rank = rawRank(tags.len);
        pub const dtype = tensor_dtype;
        pub const ag_root = ag_file;

        value: RawTypedTensor,

        const Self = @This();

        // ---- common ----
        const common = @import("tensor/common.zig").Ops(Self);
        pub const deinit = common.deinit;
        pub const asRawTensor = common.asRawTensor;
        pub const data = common.data;
        pub const dataConst = common.dataConst;
        pub const copyTo = common.copyTo;
        pub const requiresGrad = common.requiresGrad;
        pub const axis = common.axis;
        pub const hasTag = common.hasTag;
        pub const dim = common.dim;
        pub const shape = common.shape;
        pub const isContiguous = common.isContiguous;

        /// Consumes `value` on success; on error, ownership stays with the caller.
        pub fn constant(ctx: *ExecContext, value: RawTypedTensor) !Self {
            _ = ctx;
            var v = value;
            try validateTensorRank(tensor_dtype, tags, &v);
            return .{ .value = v };
        }

        pub fn fromTensor(ctx: *ExecContext, value: RawTypedTensor) !Self {
            return try Self.constant(ctx, value);
        }

        pub fn fromBlocks(ctx: *ExecContext, raw_shape: [tensor_rank]usize, values: []const Elem) !Self {
            var value = try ctx.fromStorageSlice(tensor_dtype, raw_shape, values);
            errdefer value.deinit();
            return try Self.constant(ctx, value);
        }

        pub fn fromStorageSlice(ctx: *ExecContext, raw_shape: [tensor_rank]usize, values: []const Elem) !Self {
            return Self.fromBlocks(ctx, raw_shape, values);
        }

        pub fn fromBorrowedBlocks(ctx: *ExecContext, raw_shape: [tensor_rank]usize, values: []Elem) !Self {
            var value = try ctx.fromBorrowedStorageSlice(tensor_dtype, raw_shape, values);
            errdefer value.deinit();
            return try Self.constant(ctx, value);
        }

        pub fn withTags(self: *const Self, ctx: *ExecContext, comptime new_tags_spec: anytype) !Tensor(.{ .dtype = tensor_dtype, .tags = normalizeTags(new_tags_spec) }) {
            const new_tags = normalizeTags(new_tags_spec);
            comptime {
                validateUniqueTags(new_tags);
                if (new_tags.len != tag_count) @compileError("withTags requires the same rank");
            }
            var value = try self.asRawTensor().cloneView();
            errdefer value.deinit();
            return Tensor(.{ .dtype = tensor_dtype, .tags = new_tags }).fromTensor(ctx, value);
        }

        pub fn to(self: *const Self, ctx: *ExecContext, comptime target_dtype: DType) !Tensor(.{ .dtype = target_dtype, .tags = tags }) {
            comptime if (target_dtype != .f32) @compileError("block-quantized tensors can currently only be converted to f32");
            var value = try ctx.dequantizeTensor(tensor_dtype, self.asRawTensor());
            errdefer value.deinit();
            return Tensor(.{ .dtype = target_dtype, .tags = tags }).fromTensor(ctx, value);
        }

        pub fn materialize(self: *const Self, ctx: *ExecContext) !Self {
            var value = try ctx.materialize(tensor_dtype, self.asRawTensor());
            errdefer value.deinit();
            return Self.fromTensor(ctx, value);
        }

        pub fn concat(self: *const Self, ctx: *ExecContext, comptime tag: Tag, others: []const *const Self) !Self {
            comptime {
                if (tag_count != 2) @compileError("block-quantized concat currently requires a rank-2 tensor");
                if (axis(tag) != 0) @compileError("block-quantized concat currently supports the row axis only");
            }

            var raw_inputs = try ctx.allocator.alloc(*const RawTypedTensor, others.len + 1);
            defer ctx.allocator.free(raw_inputs);

            raw_inputs[0] = self.asRawTensor();
            for (others, 0..) |other, i| raw_inputs[i + 1] = other.asRawTensor();

            var value = try ctx.concatQuantizedRows(tensor_dtype, raw_inputs);
            errdefer value.deinit();
            return Self.fromTensor(ctx, value);
        }

        /// Pack this rank-2 quantized weight into the ISA-best packed matmul
        /// RHS layout for its dtype (see `PackedRhs`): q8_0->x4, q6_k->x4,
        /// q5_k->x8, q4_k->x2mmla on aarch64+i8mm targets else x8. Use
        /// `packRhsAs` to force a specific container instead.
        pub fn packRhs(self: *const Self, ctx: *ExecContext) !PackedRhs(tensor_dtype) {
            comptime if (tag_count != 2) @compileError("packRhs requires a rank-2 tensor");
            return ctx.packMatmulRhs(tensor_dtype, self.asRawTensor());
        }

        /// Explicit-layout escape hatch over `packRhs`: pack into a specific
        /// container type (`fucina.quant.QuantizedMatmulRhsQ4_Kx8`, ...),
        /// comptime-validated against the tensor dtype. Needed e.g. to
        /// exercise the fused x8 kernels on hardware where `packRhs` would
        /// select x2mmla, at the cost of the ISA-best kernel.
        pub fn packRhsAs(self: *const Self, ctx: *ExecContext, comptime Rhs: type) !Rhs {
            comptime {
                if (tag_count != 2) @compileError("packRhsAs requires a rank-2 tensor");
                if (Rhs == backend_mod.PackedDenseRhs) @compileError("packRhsAs(PackedDenseRhs) requires an f32/f16/bf16 tensor; call its packRhs method");
                if (Rhs == backend_mod.QuantizedMatmulRhsQ4_Kx4) @compileError("packRhsAs: the Q4_Kx4 pack has no facade entry (kernel-comparison surface below the facade)");
                if (!isPackedRhsType(Rhs)) @compileError("packRhsAs: " ++ @typeName(Rhs) ++ " is not a packed matmul RHS");
                if (tensor_dtype != Rhs.dtype) @compileError("packRhsAs(" ++ @typeName(Rhs) ++ ") requires a ." ++ @tagName(Rhs.dtype) ++ " tensor");
            }
            return ctx.packMatmulRhsAs(Rhs, self.asRawTensor());
        }

        pub fn getRows(
            self: *const Self,
            ctx: *ExecContext,
            comptime tag: Tag,
            indices: []const usize,
            comptime out_tag: Tag,
        ) !Tensor(.{ .dtype = .f32, .tags = replaceTag(tags, tag, out_tag) }) {
            comptime {
                if (tag_count != 2) @compileError("quantized getRows currently requires a rank-2 tensor");
                if (axis(tag) != 0) @compileError("quantized getRows gathers rows from the first axis");
            }
            const result_tags = replaceTag(tags, tag, out_tag);
            var value = try ctx.getRowsQuantized(tensor_dtype, self.asRawTensor(), indices);
            errdefer value.deinit();
            return Tensor(.{ .dtype = .f32, .tags = result_tags }).fromTensor(ctx, value);
        }
    };
}

/// Packed matmul RHS container type for a weight dtype: the backend's
/// `PackedRhsFor` map (dense f32/f16/bf16 share the f32 output-row panel;
/// block-quantized weights select the ISA-best lane pack). This is the
/// return type of `packRhs`; model code stores packed weights as
/// `fucina.PackedRhs(dtype)` fields.
pub const PackedRhs = backend_mod.PackedRhsFor;

/// Comptime tag guard for rank-1 norm weight/bias operands: tagged tensors
/// must carry the normalized axis tag; numeric-tag `Tensor(1)` values (`._0`,
/// Parakeet's raw weights) fall through to the kernel's runtime length check.
pub fn normParamTagCheck(comptime Obj: type, comptime op_name: []const u8, comptime param_name: []const u8, comptime tag: Tag) void {
    if (Obj.axis_tags.len != 1)
        @compileError(op_name ++ ": ." ++ param_name ++ " must be a rank-1 [" ++ @tagName(tag) ++ "] tensor");
    if (Obj.axis_tags[0] != tag and Obj.axis_tags[0] != ._0)
        @compileError(op_name ++ ": ." ++ param_name ++ " tag ." ++ @tagName(Obj.axis_tags[0]) ++ " does not match the normalized axis ." ++ @tagName(tag));
}

pub const AttentionMask = exec_mod.AttentionMask;

pub const AttentionKvRepr = enum { f32_kv, f16_kv, q8_kv, multi_f16_kv, multi_q8_kv };

/// Comptime KV-representation of a `groupedAttention` k/v argument. Slice
/// shapes are classified before tensor objects (a `[]const BlockQ8_0` is a
/// pointer type too); anything else is a curated @compileError.
pub fn attentionKvRepr(comptime T: type, comptime which: []const u8) AttentionKvRepr {
    const info = @typeInfo(T);
    if (info == .pointer and info.pointer.size == .slice) {
        const Elem = info.pointer.child;
        if (Elem == BlockQ8_0) return .q8_kv;
        if (Elem == []const f16) return .multi_f16_kv;
        if (Elem == []const BlockQ8_0) return .multi_q8_kv;
    }
    if (info == .pointer and info.pointer.size == .one) {
        const Obj = info.pointer.child;
        if (@typeInfo(Obj) == .@"struct" and @hasDecl(Obj, "axis_tags") and @hasDecl(Obj, "dtype")) {
            if (comptime tagsEqual(Obj.axis_tags, .{ .seq, .kv_head, .d })) {
                if (Obj.dtype == .f32) return .f32_kv;
                if (Obj.dtype == .f16) return .f16_kv;
            }
            @compileError("groupedAttention: " ++ which ++ " must be an f32/f16 tensor tagged .{ .seq, .kv_head, .d }; got " ++ @typeName(Obj));
        }
    }
    @compileError("groupedAttention: unsupported " ++ which ++ " type " ++ @typeName(T) ++
        " (want *Tensor(.{ .seq, .kv_head, .d }) f32/f16, []const BlockQ8_0, []const []const f16, or []const []const BlockQ8_0)");
}

/// Comptime container type of a `*const <packed RHS>` argument, with a
/// curated @compileError (naming `op_name` and the offending type) for
/// anything else. Packed RHS containers are the structs `PackedRhs(dt)`
/// maps to, plus the explicit-layout packs (`packRhsAs`); each carries
/// `pub const dtype`, and the Q4_K x8 / x2mmla split is the container type.
pub fn packedRhsType(comptime T: type, comptime op_name: []const u8) type {
    const info = @typeInfo(T);
    if (info != .pointer or info.pointer.size != .one)
        @compileError(op_name ++ " expects a pointer to a packed matmul RHS (e.g. *const PackedRhs(.f32)); got " ++ @typeName(T));
    const Rhs = info.pointer.child;
    if (!isPackedRhsType(Rhs))
        @compileError(op_name ++ ": " ++ @typeName(Rhs) ++ " is not a packed matmul RHS");
    return Rhs;
}

/// The packed RHS containers the facade dispatches on.
pub fn isPackedRhsType(comptime Rhs: type) bool {
    return Rhs == backend_mod.PackedDenseRhs or
        Rhs == backend_mod.QuantizedMatmulRhsQ8_0x4 or
        Rhs == backend_mod.QuantizedMatmulRhsQ6_Kx4 or
        Rhs == backend_mod.QuantizedMatmulRhsQ5_Kx8 or
        Rhs == backend_mod.QuantizedMatmulRhsQ4_Kx8 or
        Rhs == backend_mod.QuantizedMatmulRhsQ4_Kx2Mmla or
        Rhs == backend_mod.QuantizedMatmulRhsQ4_Kx4;
}

test {
    _ = @import("tensor_tests/attention.zig");
    _ = @import("tensor_tests/control.zig");
    _ = @import("tensor_tests/conv.zig");
    _ = @import("tensor_tests/conv2d.zig");
    _ = @import("tensor_tests/creation.zig");
    _ = @import("tensor_tests/elementwise.zig");
    _ = @import("tensor_tests/facade.zig");
    _ = @import("tensor_tests/gather_scatter.zig");
    _ = @import("tensor_tests/indexing.zig");
    _ = @import("tensor_tests/integer.zig");
    _ = @import("tensor_tests/loss.zig");
    _ = @import("tensor_tests/matmul.zig");
    _ = @import("tensor_tests/norm.zig");
    _ = @import("tensor_tests/quant.zig");
    _ = @import("tensor_tests/reduce.zig");
    _ = @import("tensor_tests/rope.zig");
    _ = @import("tensor_tests/scan.zig");
    _ = @import("tensor_tests/shape.zig");
    _ = @import("tensor_tests/softmax.zig");
    _ = @import("tensor_tests/stats.zig");
    _ = @import("tensor_tests/typed.zig");
}
