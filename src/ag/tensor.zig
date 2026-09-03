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
//! Which branch has what is ONE positive statement: the capability table
//! (`Caps`/`caps`). The alias lines in the branches are grouped by
//! capability, each group is guarded by `requireCap` against the table,
//! and `auditMixin` closes the loop at comptime — every mixin decl is
//! either aliased on the branch or belongs to a decl-group the dtype's
//! row does not grant, whose `why` documents the absence.
//!
//! Spellings that normalize to the same (dtype, tags) are the same type:
//! the dispatcher normalizes before instantiating a branch.

const std = @import("std");
const tensor_mod = @import("../tensor.zig");
const dtype_mod = @import("../dtype.zig");
const exec_mod = @import("../exec.zig");
const backend_mod = @import("../backend.zig");
const core = @import("core.zig");
const AgError = core.AgError;
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

/// Options for `variance`: `ddof` 0 = biased population estimator (the
/// LayerNorm convention), 1 = Bessel-corrected (the torch.var default).
pub const VarianceOptions = struct {
    ddof: u1 = 1,
};

pub fn TopKResult(comptime tags_spec: anytype) type {
    const result_tags = normalizeTags(tags_spec);
    return struct {
        values: Tensor(result_tags),
        /// Source positions along the reduced/sorted axis: a constant i64
        /// tensor (exact for any axis length, torch's index dtype); a
        /// borrow under an exec scope like every op result.
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
    if (tags.len > tensor_mod.max_rank) @compileError(tensor_mod.too_many_tags_msg);
}

// ---------------------------------------------------------------------------
// The capability table
// ---------------------------------------------------------------------------

/// The positive capability table: one boolean per decl-group of the shared
/// mixins (plus the branch-local groups at the end), one row per dtype
/// (`caps`). A granted capability means "the decls exist on the branch";
/// individual ops may still refuse narrower dtype subsets with curated
/// comptime errors in their bodies (bool arithmetic, the f64 leaf
/// constructors, the f64 dense `packRhs`). The `Group` tables below map
/// every mixin decl to its capability and carry the reason each absence
/// exists.
pub const Caps = struct {
    // tensor/common.zig
    /// deinit/asRawTensor/data/dataConst/copyTo/requiresGrad and the
    /// tag/shape queries.
    lifetime: bool = false,
    /// item: scalar read-out (block dtypes have no per-element scalar).
    scalar_item: bool = false,
    // tensor/autograd.zig
    /// variable/variableFromSlice/zeroGrad/grad/gradView: the trainable-
    /// leaf surface. On f64 the decls exist as curated comptime refusals
    /// (gradients are always f32).
    leaf_decls: bool = false,
    /// backward/backwardWithGrad: only an f32 tensor can be a loss root.
    backward: bool = false,
    /// Not an alias group: the branch carries a live `?*GradState` field
    /// (f32/f16/bf16). Every other dtype stores no gradient pointer.
    grad_slot: bool = false,
    // tensor/views.zig
    /// The zero-copy views and data movement, every scalar dtype.
    views: bool = false,
    // tensor/float/creation.zig and tensor/typed/creation.zig
    /// The f32 constructors and random fills.
    float_creation: bool = false,
    /// The typed constant constructors (constant..onesLike).
    typed_constants: bool = false,
    /// randint/randperm (i64 seed streams) and the .bool bandMask.
    int_fills: bool = false,
    // tensor/float/matmul.zig and tensor/typed/math.zig
    /// matmul and einsum (f32 kernels; the widened policy on 16-bit).
    contraction: bool = false,
    /// dot and the packed/ternary dot family (f32-led contractions).
    float_dots: bool = false,
    /// to and the counting sum/sumAll at the stored dtype.
    typed_math_core: bool = false,
    /// The typed mean and the typed same-dtype dot.
    typed_float_math: bool = false,
    /// The exact integer compare (float compare is elementwise's).
    exact_compare: bool = false,
    // tensor/elementwise.zig and tensor/typed/int.zig
    /// add/sub/mul/maximum/minimum (wrapping on integers).
    arith_core: bool = false,
    /// The float pointwise family (unary/gated/scalar/mask/compare ops).
    float_pointwise: bool = false,
    /// f32-only graph-side elementwise: in-place updates, biasAdd, the
    /// cast, the consuming no-grad helpers, dropout, channel ops, the
    /// elemental escape hatch and its pow.
    float_graph_ops: bool = false,
    /// divTrunc/divFloor/rem/mod and the bitwise combinators.
    int_arith: bool = false,
    /// The .bool mask combinators logicalAnd/Or/Xor/Not.
    mask_logic: bool = false,
    // tensor/float/<domain>.zig
    /// Convolutions and unfold/fold.
    conv: bool = false,
    /// Pooling and upsampling.
    pool: bool = false,
    /// The index-tensor gathers/scatters beyond the views.
    gather_scatter: bool = false,
    /// cumsum/cumprod/prod.
    scans: bool = false,
    /// The masked/segmented/recurrence reductions and sumMany.
    reduce_full: bool = false,
    /// variance/argmax/max/min.
    stats_core: bool = false,
    /// standardizeAxis and multinomial.
    stats_full: bool = false,
    /// topK/sort/argsort/routerTopK.
    topk: bool = false,
    /// pad.
    pad: bool = false,
    /// The fills, diagonal/band family and the 2-d pads.
    shape_bands: bool = false,
    /// softmax/logSoftmax/logsumexp.
    softmax_family: bool = false,
    /// rmsNorm/rmsNormMul/rmsNormMulAdd/layerNorm.
    norms_core: bool = false,
    /// groupNorm, the fused rms-norm+rope kernel and the vector norms.
    norms_full: bool = false,
    /// The loss heads.
    loss: bool = false,
    /// Rotary position embedding.
    rope: bool = false,
    /// Fused grouped causal attention.
    attention: bool = false,
    // branch-local (inline methods, not mixin-backed)
    /// The dense f32-panel packRhs over f16/bf16 weights (refuses f64).
    dense_pack_rhs: bool = false,
    /// The block-quantized constant surface (dequantize, row gather,
    /// packed matmul RHS, block-safe views).
    quant_ops: bool = false,
};

/// The rows of the capability table: what each dtype's branch has.
pub fn caps(comptime tensor_dtype: DType) Caps {
    // f32: the differentiable branch, every capability of the float world.
    if (tensor_dtype == .f32) return .{
        .lifetime = true,
        .scalar_item = true,
        .leaf_decls = true,
        .backward = true,
        .grad_slot = true,
        .views = true,
        .float_creation = true,
        .contraction = true,
        .float_dots = true,
        .arith_core = true,
        .float_pointwise = true,
        .float_graph_ops = true,
        .conv = true,
        .pool = true,
        .gather_scatter = true,
        .scans = true,
        .reduce_full = true,
        .stats_core = true,
        .stats_full = true,
        .topk = true,
        .pad = true,
        .shape_bands = true,
        .softmax_family = true,
        .norms_core = true,
        .norms_full = true,
        .loss = true,
        .rope = true,
        .attention = true,
    };
    // Block-quantized: inference constants.
    if (dtype_mod.isBlockQuantized(tensor_dtype)) return .{
        .lifetime = true,
        .quant_ops = true,
    };
    // Typed float (f16/bf16/f64): forward float math; a live gradient
    // slot on the 16-bit leaves only (f64 training is unsupported).
    if (dtype_mod.supportsForwardFloatMath(tensor_dtype)) return .{
        .lifetime = true,
        .scalar_item = true,
        .leaf_decls = true,
        .grad_slot = tensor_dtype != .f64,
        .views = true,
        .typed_constants = true,
        .typed_math_core = true,
        .typed_float_math = true,
        .contraction = true,
        .dense_pack_rhs = true,
        .arith_core = true,
        .float_pointwise = true,
        .scans = true,
        .stats_core = true,
        .pad = true,
        .softmax_family = true,
        .norms_core = true,
    };
    // Typed scalar (integers, bool, the f8 storage floats): constants
    // with views, casts, and the integer/mask math.
    return .{
        .lifetime = true,
        .scalar_item = true,
        .views = true,
        .typed_constants = true,
        .int_fills = true,
        .typed_math_core = true,
        .exact_compare = true,
        .arith_core = true,
        .int_arith = true,
        .mask_logic = true,
    };
}

const Cap = std.meta.FieldEnum(Caps);

/// One decl-group of a shared mixin: the capability granting it, the decl
/// names it covers (`null` = every mixin decl not named by another group
/// of the same mixin), and why the rows without the capability lack them.
const Group = struct {
    cap: Cap,
    decls: ?[]const []const u8,
    why: []const u8,
};

const common_caps: []const Group = &.{
    .{ .cap = .scalar_item, .decls = &.{"item"}, .why = "item: block dtypes have no per-element scalar; dequantize with to(.f32) first" },
    .{ .cap = .lifetime, .decls = null, .why = "lifetime, raw access and the tag/shape queries exist on every branch" },
};

const autograd_caps: []const Group = &.{
    .{ .cap = .backward, .decls = &.{ "backward", "backwardWithGrad" }, .why = "backward: f32 only — losses are f32; 16-bit tensors are leaves, never losses" },
    .{ .cap = .leaf_decls, .decls = null, .why = "trainable leaves are f32/f16/bf16 with f32 gradients; integer, bool, f8 and block-quantized tensors are constants" },
};

const views_caps: []const Group = &.{
    .{ .cap = .views, .decls = null, .why = "block layouts pack the last axis: the quantized branch keeps only the block-safe inline set (withTags, materialize, row concat)" },
};

const float_creation_caps: []const Group = &.{
    .{ .cap = .float_creation, .decls = null, .why = "the f32 constructors and random fills; typed branches build constants from the typed constructor set" },
};

const typed_creation_caps: []const Group = &.{
    .{ .cap = .int_fills, .decls = &.{ "randint", "randperm", "bandMask" }, .why = "randint/randperm are i64 seed streams and bandMask is a .bool mask: scalar-branch constructors" },
    .{ .cap = .typed_constants, .decls = null, .why = "the typed constant constructors (the f32 branch spells them in its float creation set)" },
};

const matmul_caps: []const Group = &.{
    .{ .cap = .contraction, .decls = &.{ "matmul", "einsum" }, .why = "matmul/einsum: float branches only (f32 kernels, or the widened policy on 16-bit); cast integer tensors with to() first" },
    .{ .cap = .float_dots, .decls = null, .why = "dot and the packed/ternary dot family are f32-led contractions (typed branches keep the typed dot and the inline dense packRhs)" },
};

const typed_math_caps: []const Group = &.{
    .{ .cap = .typed_float_math, .decls = &.{ "mean", "dot" }, .why = "mean divides and dot contracts in float: typed float branches only; cast integer tensors with to() first" },
    .{ .cap = .exact_compare, .decls = &.{"compare"}, .why = "the exact integer compare (float branches take the shared elementwise compare)" },
    .{ .cap = .typed_math_core, .decls = null, .why = "to and the counting sum/sumAll exist on every typed branch" },
};

const int_caps: []const Group = &.{
    .{ .cap = .mask_logic, .decls = &.{ "logicalAnd", "logicalOr", "logicalXor", "logicalNot" }, .why = "the .bool mask combinators (float branches take the shared elementwise ones)" },
    .{ .cap = .int_arith, .decls = null, .why = "explicit integer division/remainder and the bitwise ops: integer branches only (float division is div)" },
};

const elementwise_caps: []const Group = &.{
    .{ .cap = .arith_core, .decls = &.{ "add", "sub", "mul", "maximum", "minimum" }, .why = "the shared arithmetic core: float branches and the wrapping integer set (.bool refuses in the op)" },
    .{ .cap = .float_graph_ops, .decls = &.{ "addAxisVectorInPlace", "addAxisVectorUnaryInPlace", "addScaledInPlace", "biasAdd", "prelu", "channelAffine", "to", "takeAddNoGrad", "takeScaleNoGrad", "dropout", "snake", "elementalUnary", "elementalBinary", "pow" }, .why = "f32 only: in-place updates on f32 storage, the graph-side cast and the consuming no-grad helpers, dropout, the channel ops, the elemental escape hatch and its pow" },
    .{ .cap = .float_pointwise, .decls = null, .why = "float pointwise math: float branches only; cast with to() first" },
};

const conv_caps: []const Group = &.{
    .{ .cap = .conv, .decls = null, .why = "convolutions and unfold/fold are f32 only; cast with to() first" },
};

const pool_caps: []const Group = &.{
    .{ .cap = .pool, .decls = null, .why = "pooling and upsampling are f32 only; cast with to() first" },
};

const gather_scatter_caps: []const Group = &.{
    .{ .cap = .gather_scatter, .decls = null, .why = "the index-tensor gathers/scatters are f32 only; cast with to() first" },
};

const reduce_caps: []const Group = &.{
    .{ .cap = .scans, .decls = &.{ "cumsum", "cumprod", "prod" }, .why = "scans and prod: float branches only; cast with to() first" },
    .{ .cap = .reduce_full, .decls = null, .why = "the masked, segmented and recurrence reductions and sumMany are f32 only (typed branches take sum/mean/sumAll at the stored dtype from the typed math set)" },
};

const stats_caps: []const Group = &.{
    .{ .cap = .stats_core, .decls = &.{ "variance", "argmax", "max", "min" }, .why = "variance/argmax/extrema: float branches only; cast with to() first" },
    .{ .cap = .stats_full, .decls = null, .why = "standardizeAxis and multinomial are f32 only" },
};

const topk_caps: []const Group = &.{
    .{ .cap = .topk, .decls = null, .why = "top-k and sort are f32 only; cast with to() first" },
};

const shape_caps: []const Group = &.{
    .{ .cap = .pad, .decls = &.{"pad"}, .why = "pad: float branches only; cast with to() first" },
    .{ .cap = .shape_bands, .decls = null, .why = "the fills, diagonal/band family and the 2-d pads are f32 only" },
};

const softmax_caps: []const Group = &.{
    .{ .cap = .softmax_family, .decls = null, .why = "softmax family: float tensors only (f32/f16/bf16/f64); cast with to() first" },
};

const norm_caps: []const Group = &.{
    .{ .cap = .norms_core, .decls = &.{ "rmsNorm", "rmsNormMul", "rmsNormMulAdd", "layerNorm" }, .why = "the core norms: float branches only; cast with to() first" },
    .{ .cap = .norms_full, .decls = null, .why = "groupNorm, the fused rms-norm+rope kernel and the vector norms are f32 only" },
};

const loss_caps: []const Group = &.{
    .{ .cap = .loss, .decls = null, .why = "the loss heads are f32 only (losses are f32)" },
};

const rope_caps: []const Group = &.{
    .{ .cap = .rope, .decls = null, .why = "rope is f32 only" },
};

const attention_caps: []const Group = &.{
    .{ .cap = .attention, .decls = null, .why = "groupedAttention is f32 only (the KV representations are argument types)" },
};

/// Branch-local groups (inline methods, not mixin-backed): named here so
/// `requireCap` guards and `capWhy` cover them; never passed to
/// `auditMixin`.
const local_caps: []const Group = &.{
    .{ .cap = .grad_slot, .decls = &.{"grad_state"}, .why = "a live f32 gradient slot exists on f32/f16/bf16 only (f64 training is unsupported — gradients are always f32)" },
    .{ .cap = .dense_pack_rhs, .decls = &.{"packRhs"}, .why = "the dense f32-panel packRhs over f16/bf16 weights (the op refuses f64)" },
    .{ .cap = .quant_ops, .decls = &.{ "constant", "fromTensor", "fromBlocks", "fromStorageSlice", "fromBorrowedBlocks", "withTags", "to", "materialize", "concat", "packRhs", "packRhsAs", "getRows" }, .why = "the block-quantized constant surface (dequantize, row gather, packed matmul RHS)" },
};

const all_groups = common_caps ++ autograd_caps ++ views_caps ++ float_creation_caps ++
    typed_creation_caps ++ matmul_caps ++ typed_math_caps ++ int_caps ++ elementwise_caps ++
    conv_caps ++ pool_caps ++ gather_scatter_caps ++ reduce_caps ++ stats_caps ++ topk_caps ++
    shape_caps ++ softmax_caps ++ norm_caps ++ loss_caps ++ rope_caps ++ attention_caps ++
    local_caps;

fn capWhy(comptime cap: Cap) []const u8 {
    comptime {
        for (all_groups) |group| {
            if (group.cap == cap) return group.why;
        }
        return "see the capability table";
    }
}

/// The guard in front of an alias group: the branch spells the group out,
/// so the dtype's row must grant it.
fn requireCap(comptime tensor_dtype: DType, comptime cap: Cap) void {
    comptime {
        if (!@field(caps(tensor_dtype), @tagName(cap))) {
            @compileError("the ." ++ @tagName(tensor_dtype) ++ " branch aliases ." ++ @tagName(cap) ++
                " but caps(." ++ @tagName(tensor_dtype) ++ ") does not grant it — " ++ capWhy(cap));
        }
    }
}

/// The completeness check, driven by the table (it replaces the old
/// hand-written except lists): every pub decl of `Mixin` must be aliased
/// on `Self` when the dtype's row grants its group; a decl whose group is
/// not granted is documented negative space (the group's `why`). A decl
/// added to a mixin without its alias line is a compile error naming op,
/// dtype and capability, not a silently thinner branch.
fn auditMixin(comptime Self: type, comptime Mixin: type, comptime groups: []const Group) void {
    comptime {
        @setEvalBranchQuota(200_000);
        for (groups) |group| {
            if (group.decls) |names| for (names) |name| {
                if (!@hasDecl(Mixin, name))
                    @compileError("capability ." ++ @tagName(group.cap) ++ " names " ++ name ++ ", which " ++ @typeName(Mixin) ++ " does not declare");
            };
        }
        const row = caps(Self.dtype);
        for (@typeInfo(Mixin).@"struct".decls) |decl| {
            const group = declGroup(groups, decl.name);
            if (!@field(row, @tagName(group.cap))) continue;
            if (!@hasDecl(Self, decl.name)) {
                @compileError(@typeName(Self) ++ " grants ." ++ @tagName(group.cap) ++ " but does not alias " ++ @typeName(Mixin) ++ "." ++ decl.name);
            }
        }
    }
}

fn declGroup(comptime groups: []const Group, comptime name: []const u8) Group {
    comptime {
        var rest: ?Group = null;
        for (groups) |group| {
            if (group.decls) |names| {
                for (names) |n| {
                    if (std.mem.eql(u8, n, name)) return group;
                }
            } else rest = group;
        }
        if (rest) |group| return group;
        @compileError("the capability table names no group for decl " ++ name);
    }
}

/// The gradient slot: a live `?*GradState` where the row grants
/// `.grad_slot` (f32/f16/bf16), `void` — no field storage, nothing to
/// release — elsewhere. f64 keeps the leaf DECLS (curated comptime
/// refusals in tensor/autograd.zig) but carries no slot: an f64 tensor
/// can never require gradients.
fn GradSlot(comptime tensor_dtype: DType) type {
    return if (caps(tensor_dtype).grad_slot) ?*GradState else void;
}

fn gradSlotDefault(comptime tensor_dtype: DType) GradSlot(tensor_dtype) {
    if (comptime caps(tensor_dtype).grad_slot) return null;
    return {};
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
        /// True when an exec scope holds this result's references: the
        /// struct is a borrow and `deinit` is a safe no-op for value and
        /// node alike (arena semantics; the scope releases at
        /// closeExecScope). Every branch carries the flag, so the same
        /// defer-deinit forward code runs scoped (training) and unscoped
        /// (inference) whatever the dtype.
        scope_owned: bool = false,

        const Self = @This();

        // ---- common: lifetime, raw access, tag/shape queries (.lifetime, .scalar_item) ----
        const common = @import("tensor/common.zig").Ops(Self);
        comptime {
            requireCap(dtype, .lifetime);
            requireCap(dtype, .scalar_item);
        }
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

        // ---- autograd: trainable leaves and gradient read-out (.leaf_decls) ----
        const autograd_ops = @import("tensor/autograd.zig").Ops(Self);
        comptime {
            requireCap(dtype, .leaf_decls);
        }
        pub const variable = autograd_ops.variable;
        pub const variableFromSlice = autograd_ops.variableFromSlice;
        pub const zeroGrad = autograd_ops.zeroGrad;
        pub const grad = autograd_ops.grad;
        pub const gradView = autograd_ops.gradView;

        // ---- autograd: backward entry points (.backward) ----
        comptime {
            requireCap(dtype, .backward);
        }
        pub const backward = autograd_ops.backward;
        pub const backwardWithGrad = autograd_ops.backwardWithGrad;

        // ---- views: zero-copy views and data movement (.views) ----
        const views = @import("tensor/views.zig").Ops(Self);
        comptime {
            requireCap(dtype, .views);
        }
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

        // ---- creation: constructors and fills (.float_creation) ----
        const creation_ops = @import("tensor/float/creation.zig").Ops(Self);
        comptime {
            requireCap(dtype, .float_creation);
        }
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

        // ---- matmul: the caller-named contractions (.contraction) ----
        const matmul_ops = @import("tensor/float/matmul.zig").Ops(Self);
        comptime {
            requireCap(dtype, .contraction);
        }
        pub const matmul = matmul_ops.matmul;
        pub const einsum = matmul_ops.einsum;

        // ---- matmul: dot and the packed/ternary RHS family (.float_dots) ----
        comptime {
            requireCap(dtype, .float_dots);
        }
        pub const dot = matmul_ops.dot;
        pub const addDot = matmul_ops.addDot;
        pub const dotTernarySte = matmul_ops.dotTernarySte;
        pub const dotPacked = matmul_ops.dotPacked;
        pub const packRhs = matmul_ops.packRhs;
        pub const rmsNormMulDotPacked = matmul_ops.rmsNormMulDotPacked;
        pub const splitSwiGluDotPacked = matmul_ops.splitSwiGluDotPacked;
        pub const gegluQuantDotPacked = matmul_ops.gegluQuantDotPacked;

        // ---- elementwise: the shared arithmetic core (.arith_core) ----
        const elementwise_ops = @import("tensor/elementwise.zig").Ops(Self);
        comptime {
            requireCap(dtype, .arith_core);
        }
        pub const add = elementwise_ops.add;
        pub const sub = elementwise_ops.sub;
        pub const mul = elementwise_ops.mul;
        pub const maximum = elementwise_ops.maximum;
        pub const minimum = elementwise_ops.minimum;

        // ---- elementwise: the float pointwise family (.float_pointwise) ----
        comptime {
            requireCap(dtype, .float_pointwise);
        }
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
        pub const div = elementwise_ops.div;
        pub const scale = elementwise_ops.scale;
        pub const addScalar = elementwise_ops.addScalar;
        pub const subScalar = elementwise_ops.subScalar;
        pub const divScalar = elementwise_ops.divScalar;
        pub const powScalar = elementwise_ops.powScalar;
        pub const log1p = elementwise_ops.log1p;
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

        // ---- elementwise: the f32 graph-side ops (.float_graph_ops) ----
        comptime {
            requireCap(dtype, .float_graph_ops);
        }
        pub const addAxisVectorInPlace = elementwise_ops.addAxisVectorInPlace;
        pub const addAxisVectorUnaryInPlace = elementwise_ops.addAxisVectorUnaryInPlace;
        pub const addScaledInPlace = elementwise_ops.addScaledInPlace;
        pub const biasAdd = elementwise_ops.biasAdd;
        pub const prelu = elementwise_ops.prelu;
        pub const channelAffine = elementwise_ops.channelAffine;
        pub const to = elementwise_ops.to;
        pub const takeAddNoGrad = elementwise_ops.takeAddNoGrad;
        pub const takeScaleNoGrad = elementwise_ops.takeScaleNoGrad;
        pub const dropout = elementwise_ops.dropout;
        pub const snake = elementwise_ops.snake;
        pub const elementalUnary = elementwise_ops.elementalUnary;
        pub const elementalBinary = elementwise_ops.elementalBinary;
        pub const pow = elementwise_ops.pow;

        // ---- conv: convolutions and unfold/fold (.conv) ----
        const conv_ops = @import("tensor/float/conv.zig").Ops(Self);
        comptime {
            requireCap(dtype, .conv);
        }
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

        // ---- pool: pooling and upsampling (.pool) ----
        const pool_ops = @import("tensor/float/pool.zig").Ops(Self);
        comptime {
            requireCap(dtype, .pool);
        }
        pub const maxPool2d = pool_ops.maxPool2d;
        pub const avgPool2d = pool_ops.avgPool2d;
        pub const upsample2xNearest = pool_ops.upsample2xNearest;

        // ---- gather_scatter: indexed reads and writes beyond the views (.gather_scatter) ----
        const gather_scatter_ops = @import("tensor/float/gather_scatter.zig").Ops(Self);
        comptime {
            requireCap(dtype, .gather_scatter);
        }
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

        // ---- reduce: scans (.scans) ----
        const reduce_ops = @import("tensor/float/reduce.zig").Ops(Self);
        comptime {
            requireCap(dtype, .scans);
        }
        pub const cumsum = reduce_ops.cumsum;
        pub const cumprod = reduce_ops.cumprod;
        pub const prod = reduce_ops.prod;

        // ---- reduce: the full reduction set (.reduce_full) ----
        comptime {
            requireCap(dtype, .reduce_full);
        }
        pub const any = reduce_ops.any;
        pub const all = reduce_ops.all;
        pub const anyAll = reduce_ops.anyAll;
        pub const allAll = reduce_ops.allAll;
        pub const sum = reduce_ops.sum;
        pub const mean = reduce_ops.mean;
        pub const segmentSum = reduce_ops.segmentSum;
        pub const linearRecurrence = reduce_ops.linearRecurrence;
        pub const sumAll = reduce_ops.sumAll;
        pub const sumMany = reduce_ops.sumMany;

        // ---- stats: variance, argmax, extrema (.stats_core) ----
        const stats_ops = @import("tensor/float/stats.zig").Ops(Self);
        comptime {
            requireCap(dtype, .stats_core);
        }
        pub const variance = stats_ops.variance;
        pub const argmax = stats_ops.argmax;
        pub const max = stats_ops.max;
        pub const min = stats_ops.min;

        // ---- stats: standardize and multinomial (.stats_full) ----
        comptime {
            requireCap(dtype, .stats_full);
        }
        pub const standardizeAxis = stats_ops.standardizeAxis;
        pub const multinomial = stats_ops.multinomial;

        // ---- topk: top-k, sort, router top-k (.topk) ----
        const topk_ops = @import("tensor/float/topk.zig").Ops(Self);
        comptime {
            requireCap(dtype, .topk);
        }
        pub const topK = topk_ops.topK;
        pub const sort = topk_ops.sort;
        pub const argsort = topk_ops.argsort;
        pub const routerTopK = topk_ops.routerTopK;

        // ---- shape: pad (.pad) ----
        const shape_ops = @import("tensor/float/shape.zig").Ops(Self);
        comptime {
            requireCap(dtype, .pad);
        }
        pub const pad = shape_ops.pad;

        // ---- shape: fills, diagonals, bands (.shape_bands) ----
        comptime {
            requireCap(dtype, .shape_bands);
        }
        pub const shiftBy = shape_ops.shiftBy;
        pub const diagonal = shape_ops.diagonal;
        pub const trace = shape_ops.trace;
        pub const diag = shape_ops.diag;
        pub const bandPart = shape_ops.bandPart;
        pub const tril = shape_ops.tril;
        pub const triu = shape_ops.triu;
        pub const diagEmbed = shape_ops.diagEmbed;
        pub const zeroPad2d = shape_ops.zeroPad2d;
        pub const constantPad2d = shape_ops.constantPad2d;

        // ---- softmax: softmax family (.softmax_family) ----
        const softmax_ops = @import("tensor/float/softmax.zig").Ops(Self);
        comptime {
            requireCap(dtype, .softmax_family);
        }
        pub const logsumexp = softmax_ops.logsumexp;
        pub const logSoftmax = softmax_ops.logSoftmax;
        pub const softmax = softmax_ops.softmax;

        // ---- norm: the core norms (.norms_core) ----
        const norm_ops = @import("tensor/float/norm.zig").Ops(Self);
        comptime {
            requireCap(dtype, .norms_core);
        }
        pub const rmsNorm = norm_ops.rmsNorm;
        pub const rmsNormMul = norm_ops.rmsNormMul;
        pub const rmsNormMulAdd = norm_ops.rmsNormMulAdd;
        pub const layerNorm = norm_ops.layerNorm;

        // ---- norm: groupNorm, fused norm+rope, vector norms (.norms_full) ----
        comptime {
            requireCap(dtype, .norms_full);
        }
        pub const groupNorm = norm_ops.groupNorm;
        pub const GroupNormOptions = norm_ops.GroupNormOptions;
        pub const rmsNormMulRopeHalfPrepared = norm_ops.rmsNormMulRopeHalfPrepared;
        pub const l2Normalize = norm_ops.l2Normalize;
        pub const norm = norm_ops.norm;
        pub const normAll = norm_ops.normAll;
        pub const cosineSimilarity = norm_ops.cosineSimilarity;

        // ---- loss: loss heads (.loss) ----
        const loss_ops = @import("tensor/float/loss.zig").Ops(Self);
        comptime {
            requireCap(dtype, .loss);
        }
        pub const crossEntropy = loss_ops.crossEntropy;
        pub const linearCrossEntropy = loss_ops.linearCrossEntropy;
        pub const mseLoss = loss_ops.mseLoss;
        pub const huberLoss = loss_ops.huberLoss;
        pub const bceLoss = loss_ops.bceLoss;
        pub const klDivLoss = loss_ops.klDivLoss;
        pub const nllLoss = loss_ops.nllLoss;

        // ---- rope: rotary position embedding (.rope) ----
        const rope_ops = @import("tensor/float/rope.zig").Ops(Self);
        comptime {
            requireCap(dtype, .rope);
        }
        pub const rope = rope_ops.rope;

        // ---- attention: fused grouped causal attention (.attention) ----
        const attention_ops = @import("tensor/float/attention.zig").Ops(Self);
        comptime {
            requireCap(dtype, .attention);
        }
        pub const groupedAttention = attention_ops.groupedAttention;

        comptime {
            auditMixin(Self, common, common_caps);
            auditMixin(Self, autograd_ops, autograd_caps);
            auditMixin(Self, views, views_caps);
            auditMixin(Self, creation_ops, float_creation_caps);
            auditMixin(Self, matmul_ops, matmul_caps);
            auditMixin(Self, elementwise_ops, elementwise_caps);
            auditMixin(Self, conv_ops, conv_caps);
            auditMixin(Self, pool_ops, pool_caps);
            auditMixin(Self, gather_scatter_ops, gather_scatter_caps);
            auditMixin(Self, reduce_ops, reduce_caps);
            auditMixin(Self, stats_ops, stats_caps);
            auditMixin(Self, topk_ops, topk_caps);
            auditMixin(Self, shape_ops, shape_caps);
            auditMixin(Self, softmax_ops, softmax_caps);
            auditMixin(Self, norm_ops, norm_caps);
            auditMixin(Self, loss_ops, loss_caps);
            auditMixin(Self, rope_ops, rope_caps);
            auditMixin(Self, attention_ops, attention_caps);
        }
    };
}

/// The typed float branch (f16/bf16/f64): forward math over the stored
/// dtype (f16/bf16 widen through f32 where no native kernel exists), every
/// view, and, on f16/bf16, trainable leaves with f32 gradients (f64 keeps
/// the leaf decls as curated comptime refusals and carries no gradient
/// slot).
fn TypedFloatTensor(comptime tags: anytype, comptime tensor_dtype: DType) type {
    comptime validateSpecTags(tags);

    return struct {
        pub const axis_tags = tags;
        pub const tag_count = tags.len;
        pub const tensor_rank = rawRank(tags.len);
        pub const dtype = tensor_dtype;
        pub const ag_root = ag_file;

        value: tensor_mod.TensorOf(tensor_dtype),
        /// The gradient slot where `caps(dtype).grad_slot` grants it
        /// (f16/bf16); `void` on f64 — no dead pointer field.
        grad_state: GradSlot(tensor_dtype) = gradSlotDefault(tensor_dtype),
        scope_owned: bool = false,

        const Self = @This();

        // ---- common (.lifetime, .scalar_item) ----
        const common = @import("tensor/common.zig").Ops(Self);
        comptime {
            requireCap(tensor_dtype, .lifetime);
            requireCap(tensor_dtype, .scalar_item);
        }
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

        // ---- autograd leaves (.leaf_decls): f16/bf16 train with f32
        // gradients; on f64 the constructors are comptime refusals and no
        // backward entry points exist anywhere on this branch (a 16-bit
        // tensor is never a loss) ----
        const autograd_ops = @import("tensor/autograd.zig").Ops(Self);
        comptime {
            requireCap(tensor_dtype, .leaf_decls);
        }
        pub const variable = autograd_ops.variable;
        pub const variableFromSlice = autograd_ops.variableFromSlice;
        pub const zeroGrad = autograd_ops.zeroGrad;
        pub const grad = autograd_ops.grad;
        pub const gradView = autograd_ops.gradView;

        // ---- views (.views) ----
        const views = @import("tensor/views.zig").Ops(Self);
        comptime {
            requireCap(tensor_dtype, .views);
        }
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

        // ---- creation (.typed_constants) ----
        const creation_ops = @import("tensor/typed/creation.zig").Ops(Self);
        comptime {
            requireCap(tensor_dtype, .typed_constants);
        }
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

        // ---- the dense f32-panel weight snapshot (.dense_pack_rhs) ----
        comptime {
            requireCap(tensor_dtype, .dense_pack_rhs);
        }

        /// Snapshot this rank-2 f16/bf16 `[out, contract]` weight as f32
        /// output-row panels for a FloatTensor `dotPacked`. Widening happens
        /// once here; the returned resource is caller-owned and no-grad.
        pub fn packRhs(self: *const Self, ctx: *ExecContext) !PackedRhs(tensor_dtype) {
            comptime {
                if (tag_count != 2) @compileError("packRhs requires a rank-2 tensor");
                if (tensor_dtype != .f16 and tensor_dtype != .bf16)
                    @compileError("dense packRhs supports f32, f16, and bf16 weights");
            }
            if (self.requiresGrad()) return AgError.UnsupportedGradient;
            return ctx.packDenseMatmulRhs(tensor_dtype, self.asRawTensor());
        }

        // ---- math over the stored dtype (.typed_math_core, .typed_float_math) ----
        const math_ops = @import("tensor/typed/math.zig").Ops(Self);
        comptime {
            requireCap(tensor_dtype, .typed_math_core);
            requireCap(tensor_dtype, .typed_float_math);
        }
        pub const to = math_ops.to;
        pub const sum = math_ops.sum;
        pub const sumAll = math_ops.sumAll;
        pub const mean = math_ops.mean;
        pub const dot = math_ops.dot;

        // ---- elementwise: the shared arithmetic core (.arith_core) ----
        const elementwise_ops = @import("tensor/elementwise.zig").Ops(Self);
        comptime {
            requireCap(tensor_dtype, .arith_core);
        }
        pub const add = elementwise_ops.add;
        pub const sub = elementwise_ops.sub;
        pub const mul = elementwise_ops.mul;
        pub const maximum = elementwise_ops.maximum;
        pub const minimum = elementwise_ops.minimum;

        // ---- elementwise: the float pointwise family (.float_pointwise)
        // (f32 kernels or typed kernels per the dtype policy; constants here) ----
        comptime {
            requireCap(tensor_dtype, .float_pointwise);
        }
        pub const div = elementwise_ops.div;
        pub const scale = elementwise_ops.scale;
        pub const divScalar = elementwise_ops.divScalar;
        pub const addScalar = elementwise_ops.addScalar;
        pub const subScalar = elementwise_ops.subScalar;
        pub const powScalar = elementwise_ops.powScalar;
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

        // ---- softmax family (.softmax_family), scans (.scans), stats
        // (.stats_core), pad (.pad), norms (.norms_core): the shared float
        // mixins (f32 kernels through the widened policy; constants here) ----
        const softmax_ops = @import("tensor/float/softmax.zig").Ops(Self);
        comptime {
            requireCap(tensor_dtype, .softmax_family);
        }
        pub const softmax = softmax_ops.softmax;
        pub const logSoftmax = softmax_ops.logSoftmax;
        pub const logsumexp = softmax_ops.logsumexp;

        const reduce_ops = @import("tensor/float/reduce.zig").Ops(Self);
        comptime {
            requireCap(tensor_dtype, .scans);
        }
        pub const cumsum = reduce_ops.cumsum;
        pub const cumprod = reduce_ops.cumprod;
        pub const prod = reduce_ops.prod;

        const stats_ops = @import("tensor/float/stats.zig").Ops(Self);
        comptime {
            requireCap(tensor_dtype, .stats_core);
        }
        pub const variance = stats_ops.variance;
        pub const argmax = stats_ops.argmax;
        pub const max = stats_ops.max;
        pub const min = stats_ops.min;

        const shape_ops = @import("tensor/float/shape.zig").Ops(Self);
        comptime {
            requireCap(tensor_dtype, .pad);
        }
        pub const pad = shape_ops.pad;

        const norm_ops = @import("tensor/float/norm.zig").Ops(Self);
        comptime {
            requireCap(tensor_dtype, .norms_core);
        }
        pub const rmsNorm = norm_ops.rmsNorm;
        pub const rmsNormMul = norm_ops.rmsNormMul;
        pub const rmsNormMulAdd = norm_ops.rmsNormMulAdd;
        pub const layerNorm = norm_ops.layerNorm;

        // ---- matmul: the caller-named contractions (.contraction)
        // (typed GEMM for the plain kind, the widened policy otherwise) ----
        const matmul_ops = @import("tensor/float/matmul.zig").Ops(Self);
        comptime {
            requireCap(tensor_dtype, .contraction);
        }
        pub const matmul = matmul_ops.matmul;
        pub const einsum = matmul_ops.einsum;

        comptime {
            auditMixin(Self, common, common_caps);
            auditMixin(Self, autograd_ops, autograd_caps);
            auditMixin(Self, views, views_caps);
            auditMixin(Self, creation_ops, typed_creation_caps);
            auditMixin(Self, math_ops, typed_math_caps);
            auditMixin(Self, elementwise_ops, elementwise_caps);
            auditMixin(Self, softmax_ops, softmax_caps);
            auditMixin(Self, reduce_ops, reduce_caps);
            auditMixin(Self, stats_ops, stats_caps);
            auditMixin(Self, shape_ops, shape_caps);
            auditMixin(Self, norm_ops, norm_caps);
            auditMixin(Self, matmul_ops, matmul_caps);
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
        scope_owned: bool = false,

        const Self = @This();

        // ---- common (.lifetime, .scalar_item) ----
        const common = @import("tensor/common.zig").Ops(Self);
        comptime {
            requireCap(tensor_dtype, .lifetime);
            requireCap(tensor_dtype, .scalar_item);
        }
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

        // ---- views (.views) ----
        const views = @import("tensor/views.zig").Ops(Self);
        comptime {
            requireCap(tensor_dtype, .views);
        }
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

        // ---- creation (.typed_constants) ----
        const creation_ops = @import("tensor/typed/creation.zig").Ops(Self);
        comptime {
            requireCap(tensor_dtype, .typed_constants);
        }
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

        // ---- creation: seed streams and the band mask (.int_fills) ----
        comptime {
            requireCap(tensor_dtype, .int_fills);
        }
        pub const randint = creation_ops.randint;
        pub const randperm = creation_ops.randperm;
        pub const bandMask = creation_ops.bandMask;

        // ---- integer forward math (docs/reference/04-tensor-operations.md): wrapping
        // two's-complement pointwise, explicit division/remainder, bitwise
        // combinators, i64-returning reductions, and scalar casts. On
        // `.bool` the arithmetic entries are compile errors; only `to` and
        // the counting `sum`/`sumAll` apply. ----

        // ---- casts and the counting reductions (.typed_math_core), the
        // exact integer compare (.exact_compare) ----
        const math_ops = @import("tensor/typed/math.zig").Ops(Self);
        comptime {
            requireCap(tensor_dtype, .typed_math_core);
            requireCap(tensor_dtype, .exact_compare);
        }
        pub const to = math_ops.to;
        pub const sum = math_ops.sum;
        pub const sumAll = math_ops.sumAll;
        pub const compare = math_ops.compare;

        // ---- the wrapping arithmetic subset of the shared elementwise
        // mixin (.arith_core; `div` is explicit, see intDiv) ----
        const elementwise_ops = @import("tensor/elementwise.zig").Ops(Self);
        comptime {
            requireCap(tensor_dtype, .arith_core);
        }
        pub const add = elementwise_ops.add;
        pub const sub = elementwise_ops.sub;
        pub const mul = elementwise_ops.mul;
        pub const maximum = elementwise_ops.maximum;
        pub const minimum = elementwise_ops.minimum;

        // ---- explicit division and the bitwise combinators (.int_arith) ----
        const int_ops = @import("tensor/typed/int.zig").Ops(Self);
        comptime {
            requireCap(tensor_dtype, .int_arith);
        }
        pub const divTrunc = int_ops.divTrunc;
        pub const divFloor = int_ops.divFloor;
        pub const rem = int_ops.rem;
        pub const mod = int_ops.mod;
        pub const bitAnd = int_ops.bitAnd;
        pub const bitOr = int_ops.bitOr;
        pub const bitXor = int_ops.bitXor;

        // ---- masks: the logical combinators live on the `.bool` branch (.mask_logic) ----
        comptime {
            requireCap(tensor_dtype, .mask_logic);
        }
        pub const logicalAnd = int_ops.logicalAnd;
        pub const logicalOr = int_ops.logicalOr;
        pub const logicalXor = int_ops.logicalXor;
        pub const logicalNot = int_ops.logicalNot;

        comptime {
            auditMixin(Self, common, common_caps);
            auditMixin(Self, views, views_caps);
            auditMixin(Self, creation_ops, typed_creation_caps);
            auditMixin(Self, math_ops, typed_math_caps);
            auditMixin(Self, elementwise_ops, elementwise_caps);
            auditMixin(Self, int_ops, int_caps);
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
        scope_owned: bool = false,

        const Self = @This();

        // ---- common (.lifetime; no .scalar_item: blocks have no scalar) ----
        const common = @import("tensor/common.zig").Ops(Self);
        comptime {
            requireCap(tensor_dtype, .lifetime);
        }
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

        // ---- the block-quantized constant surface (.quant_ops) ----
        comptime {
            requireCap(tensor_dtype, .quant_ops);
        }

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
            return plumbing.finishTyped(Tensor(.{ .dtype = tensor_dtype, .tags = new_tags }), ctx, value);
        }

        pub fn to(self: *const Self, ctx: *ExecContext, comptime target_dtype: DType) !Tensor(.{ .dtype = target_dtype, .tags = tags }) {
            comptime if (target_dtype != .f32) @compileError("block-quantized tensors can currently only be converted to f32");
            var value = try ctx.dequantizeTensor(tensor_dtype, self.asRawTensor());
            errdefer value.deinit();
            return plumbing.finishTyped(Tensor(.{ .dtype = target_dtype, .tags = tags }), ctx, value);
        }

        pub fn materialize(self: *const Self, ctx: *ExecContext) !Self {
            var value = try ctx.materialize(tensor_dtype, self.asRawTensor());
            errdefer value.deinit();
            return plumbing.finishTyped(Self, ctx, value);
        }

        pub fn concat(self: *const Self, ctx: *ExecContext, comptime tag: Tag, others: []const *const Self) !Self {
            comptime {
                if (tag_count != 2) @compileError("block-quantized concat currently requires a rank-2 tensor");
                if (axis(tag) != 0) @compileError("block-quantized concat currently supports the row axis only");
            }

            var raw_inputs = try ctx.allocator().alloc(*const RawTypedTensor, others.len + 1);
            defer ctx.allocator().free(raw_inputs);

            raw_inputs[0] = self.asRawTensor();
            for (others, 0..) |other, i| raw_inputs[i + 1] = other.asRawTensor();

            var value = try ctx.concatQuantizedRows(tensor_dtype, raw_inputs);
            errdefer value.deinit();
            return plumbing.finishTyped(Self, ctx, value);
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
                if (Rhs == backend_mod.quant.QuantizedMatmulRhsQ4_Kx4) @compileError("packRhsAs: the Q4_Kx4 pack has no facade entry (kernel-comparison surface below the facade)");
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
            return plumbing.finishTyped(Tensor(.{ .dtype = .f32, .tags = result_tags }), ctx, value);
        }

        comptime {
            auditMixin(Self, common, common_caps);
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

/// The packed RHS containers the facade dispatches on: the dense f32
/// panel, and every quantized container whose `pack` is a lane interleave
/// (the compact `.rows` containers are the quantized `dot`'s operand, not
/// `dotPacked`'s). Derived from the container, never a list.
pub fn isPackedRhsType(comptime Rhs: type) bool {
    if (Rhs == backend_mod.PackedDenseRhs) return true;
    if (@typeInfo(Rhs) != .@"struct" or !@hasDecl(Rhs, "pack") or !@hasDecl(Rhs, "dtype")) return false;
    return Rhs.pack != .rows;
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
    _ = @import("tensor_tests/ownership.zig");
    _ = @import("tensor_tests/quant.zig");
    _ = @import("tensor_tests/reduce.zig");
    _ = @import("tensor_tests/rope.zig");
    _ = @import("tensor_tests/scan.zig");
    _ = @import("tensor_tests/shape.zig");
    _ = @import("tensor_tests/softmax.zig");
    _ = @import("tensor_tests/stats.zig");
    _ = @import("tensor_tests/typed.zig");
}
