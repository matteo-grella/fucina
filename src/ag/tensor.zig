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

/// Every `pub` decl of `Mixin` is aliased on `Self`, except the named
/// ones: an entry added to a mixin without its alias line is a compile
/// error here, not a silently thinner branch. The `except` lists below
/// are the complete statement of where the branches differ.
fn assertAliased(comptime Self: type, comptime Mixin: type, comptime except: []const []const u8) void {
    comptime {
        @setEvalBranchQuota(20_000);
        for (@typeInfo(Mixin).@"struct".decls) |decl| {
            var skip = false;
            for (except) |name| {
                if (std.mem.eql(u8, name, decl.name)) skip = true;
            }
            if (!skip and !@hasDecl(Self, decl.name)) {
                @compileError(@typeName(Self) ++ " does not alias " ++ @typeName(Mixin) ++ "." ++ decl.name);
            }
        }
    }
}

/// The named `pub` decls of `Mixin` are aliased on `Self`: the positive
/// form for a branch that takes a small subset of a mixin.
fn assertAliasedSubset(comptime Self: type, comptime Mixin: type, comptime names: []const []const u8) void {
    comptime {
        for (names) |name| {
            if (!@hasDecl(Mixin, name)) @compileError(@typeName(Mixin) ++ " has no decl " ++ name);
            if (!@hasDecl(Self, name)) @compileError(@typeName(Self) ++ " does not alias " ++ @typeName(Mixin) ++ "." ++ name);
        }
    }
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
        const elementwise_ops = @import("tensor/elementwise.zig").Ops(Self);
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
        pub const softcap = elementwise_ops.softcap;
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

        comptime {
            assertAliased(Self, common, &.{});
            assertAliased(Self, autograd_ops, &.{});
            assertAliased(Self, views, &.{});
            assertAliased(Self, creation_ops, &.{});
            assertAliased(Self, matmul_ops, &.{});
            assertAliased(Self, elementwise_ops, &.{});
            assertAliased(Self, conv_ops, &.{});
            assertAliased(Self, pool_ops, &.{});
            assertAliased(Self, gather_scatter_ops, &.{});
            assertAliased(Self, reduce_ops, &.{});
            assertAliased(Self, stats_ops, &.{});
            assertAliased(Self, topk_ops, &.{});
            assertAliased(Self, shape_ops, &.{});
            assertAliased(Self, softmax_ops, &.{});
            assertAliased(Self, norm_ops, &.{});
            assertAliased(Self, loss_ops, &.{});
            assertAliased(Self, rope_ops, &.{});
            assertAliased(Self, attention_ops, &.{});
        }
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
        const creation_ops = @import("tensor/typed/creation.zig").Ops(Self);
        pub const constant = creation_ops.constant;
        pub const fromTensor = creation_ops.fromTensor;
        pub const fromSlice = creation_ops.fromSlice;
        pub const fromBorrowedConstSlice = creation_ops.fromBorrowedConstSlice;
        pub const empty = creation_ops.empty;
        pub const zeros = creation_ops.zeros;
        pub const ones = creation_ops.ones;
        pub const emptyLike = creation_ops.emptyLike;
        pub const zerosLike = creation_ops.zerosLike;
        pub const onesLike = creation_ops.onesLike;

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

        // ---- math over the stored dtype (every typed float dtype) ----
        const math_ops = @import("tensor/typed/math.zig").Ops(Self);
        pub const to = math_ops.to;
        pub const sum = math_ops.sum;
        pub const mean = math_ops.mean;
        pub const sumAll = math_ops.sumAll;
        pub const dot = math_ops.dot;

        // ---- elementwise: the shared float mixin (f32 kernels or typed kernels per the dtype policy; constants here) ----
        const elementwise_ops = @import("tensor/elementwise.zig").Ops(Self);
        pub const add = elementwise_ops.add;
        pub const sub = elementwise_ops.sub;
        pub const mul = elementwise_ops.mul;
        pub const div = elementwise_ops.div;
        pub const scale = elementwise_ops.scale;
        pub const divScalar = elementwise_ops.divScalar;
        pub const addScalar = elementwise_ops.addScalar;
        pub const subScalar = elementwise_ops.subScalar;
        pub const powScalar = elementwise_ops.powScalar;
        pub const maximum = elementwise_ops.maximum;
        pub const minimum = elementwise_ops.minimum;
        pub const where = elementwise_ops.where;
        pub const maskedFill = elementwise_ops.maskedFill;
        pub const gated = elementwise_ops.gated;
        pub const glu = elementwise_ops.glu;
        pub const swiglu = elementwise_ops.swiglu;
        pub const geglu = elementwise_ops.geglu;
        pub const situ = elementwise_ops.situ;
        pub const splitGated = elementwise_ops.splitGated;
        pub const unary = elementwise_ops.unary;
        pub const relu = elementwise_ops.relu;
        pub const leakyRelu = elementwise_ops.leakyRelu;
        pub const exp = elementwise_ops.exp;
        pub const sqrt = elementwise_ops.sqrt;
        pub const rsqrt = elementwise_ops.rsqrt;
        pub const sigmoid = elementwise_ops.sigmoid;
        pub const silu = elementwise_ops.silu;
        pub const log = elementwise_ops.log;
        pub const log1p = elementwise_ops.log1p;
        pub const neg = elementwise_ops.neg;
        pub const abs = elementwise_ops.abs;
        pub const sin = elementwise_ops.sin;
        pub const cos = elementwise_ops.cos;
        pub const tanh = elementwise_ops.tanh;
        pub const fastTanh = elementwise_ops.fastTanh;
        pub const softcap = elementwise_ops.softcap;
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
        pub const compare = elementwise_ops.compare;
        pub const logicalAnd = elementwise_ops.logicalAnd;
        pub const logicalOr = elementwise_ops.logicalOr;
        pub const logicalXor = elementwise_ops.logicalXor;
        pub const logicalNot = elementwise_ops.logicalNot;
        pub const isnan = elementwise_ops.isnan;
        pub const isinf = elementwise_ops.isinf;
        pub const isfinite = elementwise_ops.isfinite;

        // ---- softmax, scans, reductions, pad: the shared float mixins (f32 kernels through the widened policy; constants here) ----
        const softmax_ops = @import("tensor/float/softmax.zig").Ops(Self);
        pub const softmax = softmax_ops.softmax;
        pub const logSoftmax = softmax_ops.logSoftmax;
        pub const logsumexp = softmax_ops.logsumexp;
        const reduce_ops = @import("tensor/float/reduce.zig").Ops(Self);
        pub const cumsum = reduce_ops.cumsum;
        pub const cumprod = reduce_ops.cumprod;
        pub const prod = reduce_ops.prod;
        const stats_ops = @import("tensor/float/stats.zig").Ops(Self);
        pub const variance = stats_ops.variance;
        pub const argmax = stats_ops.argmax;
        pub const max = stats_ops.max;
        pub const min = stats_ops.min;
        const shape_ops = @import("tensor/float/shape.zig").Ops(Self);
        pub const pad = shape_ops.pad;

        const norm_ops = @import("tensor/float/norm.zig").Ops(Self);
        pub const rmsNorm = norm_ops.rmsNorm;
        pub const rmsNormMul = norm_ops.rmsNormMul;
        pub const rmsNormMulAdd = norm_ops.rmsNormMulAdd;
        pub const layerNorm = norm_ops.layerNorm;

        // ---- matmul: the shared float mixin (typed GEMM for the plain kind, the widened policy otherwise; constants here) ----
        const matmul_ops = @import("tensor/float/matmul.zig").Ops(Self);
        pub const matmul = matmul_ops.matmul;
        pub const einsum = matmul_ops.einsum;

        comptime {
            assertAliased(Self, common, &.{});
            // A 16-bit tensor is never a loss: no backward entry points.
            assertAliased(Self, autograd_ops, &.{ "backward", "backwardWithGrad" });
            assertAliased(Self, views, &.{});
            // The i64 seed streams and the .bool band mask are scalar-branch constructors.
            assertAliased(Self, creation_ops, &.{ "randint", "randperm", "bandMask" });
            // Float comparison comes from the shared elementwise mixin (math_ops.compare is the exact integer one).
            assertAliased(Self, math_ops, &.{"compare"});
            // `dot` and the packed/ternary dots keep their typed and f32 forms (math_ops.dot, the inline packRhs).
            assertAliased(Self, matmul_ops, &.{ "dot", "addDot", "dotTernarySte", "dotPacked", "packRhs", "rmsNormMulDotPacked", "splitSwiGluDotPacked", "gegluQuantDotPacked" });
            assertAliased(Self, softmax_ops, &.{});
            // groupNorm, the fused rms-norm+rope kernel and the vector norms stay f32-only.
            assertAliased(Self, norm_ops, &.{ "groupNorm", "rmsNormMulRopeHalfPrepared", "l2Normalize", "norm", "normAll", "cosineSimilarity" });
            // The typed reductions (`sum`, `mean`, `sumAll`) come from math_ops; the
            // masked, segmented and recurrence arms are f32-only.
            assertAliased(Self, reduce_ops, &.{ "any", "all", "anyAll", "allAll", "sum", "mean", "segmentSum", "linearRecurrence", "sumAll", "sumMany" });
            assertAliased(Self, stats_ops, &.{ "standardizeAxis", "multinomial" });
            // The diagonal/band family and the 2-d pads stay f32-only.
            assertAliased(Self, shape_ops, &.{ "shiftBy", "diagonal", "trace", "diag", "bandPart", "tril", "triu", "diagEmbed", "zeroPad2d", "constantPad2d" });
            // The rest of the elementwise mixin is f32-only: in-place updates on
            // f32 storage, the graph-side cast and consuming ops, dropout, the
            // channel ops, the elemental escape hatch and its `pow`.
            assertAliased(Self, elementwise_ops, &.{ "addAxisVectorInPlace", "addAxisVectorUnaryInPlace", "addScaledInPlace", "biasAdd", "prelu", "channelAffine", "to", "takeAddNoGrad", "takeScaleNoGrad", "dropout", "snake", "elementalUnary", "elementalBinary", "pow" });
        }
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
        const creation_ops = @import("tensor/typed/creation.zig").Ops(Self);
        pub const constant = creation_ops.constant;
        pub const fromTensor = creation_ops.fromTensor;
        pub const fromSlice = creation_ops.fromSlice;
        pub const fromBorrowedConstSlice = creation_ops.fromBorrowedConstSlice;
        pub const empty = creation_ops.empty;
        pub const zeros = creation_ops.zeros;
        pub const ones = creation_ops.ones;
        pub const emptyLike = creation_ops.emptyLike;
        pub const zerosLike = creation_ops.zerosLike;
        pub const onesLike = creation_ops.onesLike;
        pub const randint = creation_ops.randint;
        pub const randperm = creation_ops.randperm;
        pub const bandMask = creation_ops.bandMask;

        // ---- integer forward math (docs/reference/04-tensor-operations.md): wrapping
        // two's-complement pointwise, explicit division/remainder, bitwise
        // combinators, i64-returning reductions, and scalar casts. On
        // `.bool` the arithmetic entries are compile errors; only `to` and
        // the counting `sum`/`sumAll` apply. ----
        const math_ops = @import("tensor/typed/math.zig").Ops(Self);
        pub const to = math_ops.to;
        pub const sum = math_ops.sum;
        pub const sumAll = math_ops.sumAll;
        pub const compare = math_ops.compare;
        // The integer arithmetic subset of the shared elementwise mixin
        // (wrapping two's complement; `div` is explicit, see intDiv).
        const elementwise_ops = @import("tensor/elementwise.zig").Ops(Self);
        pub const add = elementwise_ops.add;
        pub const sub = elementwise_ops.sub;
        pub const mul = elementwise_ops.mul;
        pub const maximum = elementwise_ops.maximum;
        pub const minimum = elementwise_ops.minimum;
        const int_ops = @import("tensor/typed/int.zig").Ops(Self);
        pub const divTrunc = int_ops.divTrunc;
        pub const divFloor = int_ops.divFloor;
        pub const rem = int_ops.rem;
        pub const mod = int_ops.mod;
        pub const bitAnd = int_ops.bitAnd;
        pub const bitOr = int_ops.bitOr;
        pub const bitXor = int_ops.bitXor;

        // ---- masks: the logical combinators live on the `.bool` branch ----
        pub const logicalAnd = int_ops.logicalAnd;
        pub const logicalOr = int_ops.logicalOr;
        pub const logicalXor = int_ops.logicalXor;
        pub const logicalNot = int_ops.logicalNot;

        comptime {
            assertAliased(Self, common, &.{});
            assertAliased(Self, views, &.{});
            assertAliased(Self, creation_ops, &.{});
            // Integer division is explicit (divTrunc/divFloor); mean, dot, and
            // scaling are float ops.
            assertAliased(Self, math_ops, &.{ "mean", "dot" });
            assertAliasedSubset(Self, elementwise_ops, &.{ "add", "sub", "mul", "maximum", "minimum" });
            assertAliased(Self, int_ops, &.{});
        }
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

        comptime {
            // Block dtypes have no scalar element to read out.
            assertAliased(Self, common, &.{"item"});
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
