const std = @import("std");
const tensor_mod = @import("../tensor.zig");
const dtype_mod = @import("../dtype.zig");
const exec_mod = @import("../exec.zig");
const backend_mod = @import("../backend.zig");
const core = @import("core.zig");
const tags_mod = @import("../tags.zig");
const backward = @import("backward.zig");

const RawTensor = tensor_mod.Tensor;
const DType = tensor_mod.DType;
const BlockQ8_0 = dtype_mod.BlockQ8_0;
const TensorError = tensor_mod.TensorError;
const ExecContext = exec_mod.ExecContext;
const GradState = core.GradState;
const Tag = tags_mod.Tag;
const normalizeTags = tags_mod.normalizeTags;
const dtypeFromSpec = tags_mod.dtypeFromSpec;
const validateUniqueTags = tags_mod.validateUniqueTags;
const rawRank = tags_mod.rawRank;
const tagIndex = tags_mod.tagIndex;
const tagIndexOrCompileError = tags_mod.tagIndexOrCompileError;
const tagsEqual = tags_mod.tagsEqual;

const ag_file = @This();

const plumbing = @import("tensor/plumbing.zig").Mod(ag_file);
const finishNoGrad = plumbing.finishNoGrad;
pub const einsumMany = plumbing.einsumMany;
const typed_constant = @import("tensor/typed_constant.zig").Mod(ag_file);
const QuantizedConstantTensor = typed_constant.QuantizedConstantTensor;
const TypedConstantTensor = typed_constant.TypedConstantTensor;

/// Input counts covered by the stack fast path for `concat`/`stack` metadata
/// temporaries (input pointers, backward parents/sizes); larger input counts
/// fall back to a heap allocation.
pub const concat_inline_inputs = 16;

/// Per-axis range for `slice` (torch/numpy basic slicing, positive steps):
/// `start`/`end` count from the end of the axis when negative and clamp to
/// it, `end = null` means the axis dim, `step` must be >= 1. Negative
/// steps are deliberately unsupported (torch rejects them in basic
/// indexing too; strides are unsigned, so a reversed view is not
/// representable — compose `flip`).
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

pub fn Tensor(comptime tags_spec: anytype) type {
    const tensor_dtype = dtypeFromSpec(tags_spec);
    if (comptime tensor_dtype == .f32) return FloatTensor(tags_spec);
    if (comptime dtype_mod.isBlockQuantized(tensor_dtype)) return QuantizedConstantTensor(tags_spec, tensor_dtype);
    return TypedConstantTensor(tags_spec, tensor_dtype);
}

fn FloatTensor(comptime tags_spec: anytype) type {
    const tags = normalizeTags(tags_spec);
    comptime validateUniqueTags(tags);
    const tag_rank = tags.len;
    if (tag_rank > tensor_mod.max_rank) @compileError("too many tensor tags");

    return struct {
        pub const axis_tags = tags;
        pub const tag_count = tag_rank;
        pub const tensor_rank = rawRank(tag_rank);
        pub const dtype = DType.f32;
        /// The enclosing module, handed to the float/ method mixins and
        /// the Mod-parameterized bands so they need no upward import
        /// (keeps the production import graph acyclic; see arch-check).
        pub const ag_root = ag_file;

        value: RawTensor,
        grad_state: ?*GradState = null,
        /// True when an exec scope owns this tensor: the struct is a borrow
        /// and `deinit` is a safe no-op (arena-allocator semantics — the
        /// scope releases value and node at closeExecScope). Lets the same
        /// defer-deinit forward code run scoped (training) and unscoped
        /// (inference).
        scope_owned: bool = false,

        const Self = @This();
        pub fn deinit(self: *Self) void {
            if (self.scope_owned) return; // borrow: the exec scope owns value + node
            self.value.deinit();
            if (self.grad_state) |state| state.deinit();
            self.* = undefined;
        }

        pub fn asRawTensor(self: *const Self) *const RawTensor {
            return &self.value;
        }

        pub fn item(self: *const Self) !f32 {
            if (!self.value.isScalar()) return TensorError.InvalidShape;
            return (try self.value.dataConstChecked())[0];
        }

        pub fn data(self: *Self) ![]f32 {
            if (self.requiresGrad()) return error.MutableDataRequiresNoGrad;
            return self.value.dataChecked();
        }

        pub fn dataConst(self: *const Self) ![]const f32 {
            return self.value.dataConstChecked();
        }

        pub fn detach(self: *const Self, ctx: *ExecContext) !Self {
            var value = try self.value.cloneView();
            errdefer value.deinit();
            return finishNoGrad(tags, ctx, value);
        }

        pub fn copyTo(self: *const Self, dst: []f32) !void {
            return self.value.copyTo(dst);
        }

        pub fn requiresGrad(self: *const Self) bool {
            return self.grad_state != null;
        }

        /// Drop the accumulated gradient (no-op for constants). Training loops
        /// call this between steps so gradients don't accumulate across them.
        pub fn zeroGrad(self: *const Self) void {
            if (self.grad_state) |state| state.zeroGrad();
        }

        pub fn backward(self: *const Self, ctx: *ExecContext) !void {
            const state = self.grad_state orelse return error.NoGradientGraph;
            return core.backwardGradOne(ctx, state, &self.value);
        }

        /// As `backward`, but with an explicit output gradient instead of
        /// the implicit scalar 1: the way to run backward from a non-scalar
        /// output (scalar outputs may take one too). `grad_output` is
        /// same-tagged and must match `self`'s shape
        /// (`error.ShapeMismatch`); it is read as a value — its own gradient
        /// state, if any, is ignored — and replaces any gradient already
        /// accumulated on `self`.
        pub fn backwardWithGrad(self: *const Self, ctx: *ExecContext, grad_output: *const Self) !void {
            const state = self.grad_state orelse return error.NoGradientGraph;
            // Checked here too so the error exit leaves `self`'s accumulated
            // gradient untouched (the engine re-checks after setGrad).
            if (state.backward_done) return core.AgError.BackwardAlreadyRun;
            if (!std.mem.eql(usize, self.value.shape.slice(), grad_output.value.shape.slice())) {
                return TensorError.ShapeMismatch;
            }
            state.setGrad(try grad_output.value.cloneView());
            return core.backwardGradOne(ctx, state, &self.value);
        }

        pub fn grad(self: *const Self, ctx: *ExecContext) !?Self {
            const state = self.grad_state orelse return null;
            var value = (try state.gradClone(ctx.allocator)) orelse return null;
            errdefer value.deinit();
            const out = try Self.constant(ctx, value);
            return out;
        }

        pub fn gradView(self: *const Self, ctx: *ExecContext) !?Self {
            const state = self.grad_state orelse return null;
            var value = (try state.gradView()) orelse return null;
            errdefer value.deinit();
            const out = try Self.constant(ctx, value);
            return out;
        }

        pub fn axis(comptime tag: Tag) usize {
            return tagIndexOrCompileError(tags, tag);
        }

        pub fn hasTag(comptime tag: Tag) bool {
            return comptime tagIndex(tags, tag) != null;
        }

        pub fn dim(self: *const Self, comptime tag: Tag) usize {
            return self.asRawTensor().shape.at(axis(tag));
        }

        pub fn shape(self: *const Self) [tensor_rank]usize {
            var out: [tensor_rank]usize = undefined;
            inline for (0..tensor_rank) |i| {
                out[i] = self.asRawTensor().shape.at(i);
            }
            return out;
        }

        /// True when the storage is dense row-major in logical order
        /// (innermost stride 1) — the layout `data`/`dataConst` require;
        /// false for strided views (permutes, broadcasts, inner narrows).
        pub fn isContiguous(self: *const Self) bool {
            return self.value.isContiguous();
        }

        // ---- creation: constructors and fills ----
        const creation_ops = @import("tensor/float/creation.zig").Ops(Self);
        pub const variable = creation_ops.variable;
        pub const variableFromSlice = creation_ops.variableFromSlice;
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

        // ---- gather_scatter: indexed reads and writes ----
        const gather_scatter_ops = @import("tensor/float/gather_scatter.zig").Ops(Self);
        pub const zeroSlice = gather_scatter_ops.zeroSlice;
        pub const zeroRows = gather_scatter_ops.zeroRows;
        pub const relposShift = gather_scatter_ops.relposShift;
        pub const gather = gather_scatter_ops.gather;
        pub const indexSelect = gather_scatter_ops.indexSelect;
        pub const maskedSelect = gather_scatter_ops.maskedSelect;
        pub const nonzero = gather_scatter_ops.nonzero;
        pub const maskedScatter = gather_scatter_ops.maskedScatter;
        pub const setSlice = gather_scatter_ops.setSlice;
        pub const setRows = gather_scatter_ops.setRows;
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

        // ---- shape: views, reshapes, slicing, concat/stack, padding ----
        const shape_ops = @import("tensor/float/shape.zig").Ops(Self);
        pub const materialize = shape_ops.materialize;
        pub const contiguous = shape_ops.contiguous;
        pub const withTags = shape_ops.withTags;
        pub const viewWithStrides = shape_ops.viewWithStrides;
        pub const alignTo = shape_ops.alignTo;
        pub const permuteTo = shape_ops.permuteTo;
        pub const transpose = shape_ops.transpose;
        pub const insertAxis = shape_ops.insertAxis;
        pub const squeeze = shape_ops.squeeze;
        pub const split = shape_ops.split;
        pub const merge = shape_ops.merge;
        pub const reshape = shape_ops.reshape;
        pub const broadcastTo = shape_ops.broadcastTo;
        pub const flatten = shape_ops.flatten;
        pub const flip = shape_ops.flip;
        pub const roll = shape_ops.roll;
        pub const rollBy = shape_ops.rollBy;
        pub const shiftBy = shape_ops.shiftBy;
        pub const narrow = shape_ops.narrow;
        pub const select = shape_ops.select;
        pub const sliceStep = shape_ops.sliceStep;
        pub const slice = shape_ops.slice;
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
        pub const concat = shape_ops.concat;
        pub const stack = shape_ops.stack;
        pub const unbindInto = shape_ops.unbindInto;
        pub const repeatAxis = shape_ops.repeatAxis;

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
