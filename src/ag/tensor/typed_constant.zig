//! The typed-constant tensor band: non-f32 `Tensor(...)` object
//! types (integer/half/quantized storage, no-grad constants) and
//! the `typedConstant*` op implementations they alias.

const std = @import("std");
const tensor_mod = @import("../../tensor.zig");
const dtype_mod = @import("../../dtype.zig");
const exec_mod = @import("../../exec.zig");
const backend_mod = @import("../../backend.zig");
const tag_ops = @import("../../tagged.zig");
const control = @import("../control.zig");
const core = @import("../core.zig");
const tags_mod = @import("../../tags.zig");
const backward = @import("../backward.zig");
const rng = @import("../../rng.zig");

const RawTensor = tensor_mod.Tensor;
const DType = tensor_mod.DType;
const TensorError = tensor_mod.TensorError;
const Scalar = tensor_mod.Scalar;
const ExecContext = exec_mod.ExecContext;
const UnaryOp = exec_mod.UnaryOp;
const GatedOp = exec_mod.GatedOp;
const GradState = core.GradState;
const Tag = tags_mod.Tag;
const normalizeTags = tags_mod.normalizeTags;
const validateUniqueTags = tags_mod.validateUniqueTags;
const validateSameTagSet = tags_mod.validateSameTagSet;
const rawRank = tags_mod.rawRank;
const tagIndex = tags_mod.tagIndex;
const tagIndexOrCompileError = tags_mod.tagIndexOrCompileError;
const identityAxes = tags_mod.identityAxes;
const alignAxes = tags_mod.alignAxes;
const insertAxes = tags_mod.insertAxes;
const squeezeAxes = tags_mod.squeezeAxes;
const removeTag = tags_mod.removeTag;
const replaceTag = tags_mod.replaceTag;
const pointwiseResultTags = tags_mod.pointwiseResultTags;
const dotResultTags = tags_mod.dotResultTags;
const insertTagAt = tags_mod.insertTagAt;
const splitTags = tags_mod.splitTags;
const mergeTags = tags_mod.mergeTags;
const broadcastTensorToOf = tag_ops.broadcastTensorToOf;
const pointwiseShapeOf = tag_ops.pointwiseShapeOf;
const validateTensorRankOf = tag_ops.validateTensorRankOf;
const PointwiseOp = backward.PointwiseOp;
const CastBackward = backward.CastBackward;

pub fn Mod(comptime ag_tensor: type) type {
    return struct {
        const Tensor = ag_tensor.Tensor;
        const PackedRhs = ag_tensor.PackedRhs;
        const concat_inline_inputs = ag_tensor.concat_inline_inputs;

        const plumbing = @import("plumbing.zig").Mod(ag_tensor);
        const pointwise = plumbing.pointwise;
        const gatedPointwise = plumbing.gatedPointwise;
        const finishOp = plumbing.finishOp;
        const finishNoGrad = plumbing.finishNoGrad;
        const axisViewTensorOf = plumbing.axisViewTensorOf;
        const typedDotRaw = plumbing.typedDotRaw;
        const TensorObject = plumbing.TensorObject;
        const tensorObjectPtrFrom = plumbing.tensorObjectPtrFrom;

        pub fn QuantizedConstantTensor(comptime tags_spec: anytype, comptime tensor_dtype: DType) type {
            const tags = normalizeTags(tags_spec);
            comptime validateUniqueTags(tags);
            const tag_rank = tags.len;
            if (tag_rank > tensor_mod.max_rank) @compileError("too many tensor tags");

            const RawTypedTensor = tensor_mod.TensorOf(tensor_dtype);
            const Elem = dtype_mod.Storage(tensor_dtype);

            return struct {
                pub const axis_tags = tags;
                pub const tag_count = tag_rank;
                pub const tensor_rank = rawRank(tag_rank);
                pub const dtype = tensor_dtype;

                value: RawTypedTensor,

                const Self = @This();

                /// Consumes `value` on success; on error, ownership stays with the caller.
                pub fn constant(ctx: *ExecContext, value: RawTypedTensor) !Self {
                    _ = ctx;
                    var v = value;
                    try validateTensorRankOf(tensor_dtype, tags, &v);
                    return .{ .value = v };
                }

                pub fn fromTensor(ctx: *ExecContext, value: RawTypedTensor) !Self {
                    return try Self.constant(ctx, value);
                }

                pub fn fromBlocks(ctx: *ExecContext, raw_shape: [tensor_rank]usize, values: []const Elem) !Self {
                    var value = try ctx.fromStorageSliceRankTyped(tensor_dtype, tensor_rank, raw_shape, values);
                    errdefer value.deinit();
                    return try Self.constant(ctx, value);
                }

                pub fn fromStorageSlice(ctx: *ExecContext, raw_shape: [tensor_rank]usize, values: []const Elem) !Self {
                    return Self.fromBlocks(ctx, raw_shape, values);
                }

                pub fn fromBorrowedBlocks(ctx: *ExecContext, raw_shape: [tensor_rank]usize, values: []Elem) !Self {
                    var value = try ctx.fromBorrowedStorageSliceRankTyped(tensor_dtype, tensor_rank, raw_shape, values);
                    errdefer value.deinit();
                    return try Self.constant(ctx, value);
                }

                pub fn deinit(self: *Self) void {
                    self.value.deinit();
                    self.* = undefined;
                }

                pub fn asRawTensor(self: *const Self) *const RawTypedTensor {
                    return &self.value;
                }

                pub fn data(self: *Self) ![]Elem {
                    return self.value.dataChecked();
                }

                pub fn dataConst(self: *const Self) ![]const Elem {
                    return self.value.dataConstChecked();
                }

                pub fn copyTo(self: *const Self, dst: []Elem) !void {
                    return self.value.copyTo(dst);
                }

                pub fn requiresGrad(_: *const Self) bool {
                    return false;
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

                pub fn withTags(self: *const Self, ctx: *ExecContext, comptime new_tags_spec: anytype) !Tensor(.{ .dtype = tensor_dtype, .tags = normalizeTags(new_tags_spec) }) {
                    const new_tags = normalizeTags(new_tags_spec);
                    comptime {
                        validateUniqueTags(new_tags);
                        if (new_tags.len != tag_rank) @compileError("withTags requires the same rank");
                    }
                    var value = try self.asRawTensor().cloneView();
                    errdefer value.deinit();
                    return Tensor(.{ .dtype = tensor_dtype, .tags = new_tags }).fromTensor(ctx, value);
                }

                pub fn to(self: *const Self, ctx: *ExecContext, comptime target_dtype: DType) !Tensor(.{ .dtype = target_dtype, .tags = tags }) {
                    comptime if (target_dtype != .f32) @compileError("block-quantized tensors can currently only be converted to f32");
                    var value = try ctx.dequantizeTensorTyped(tensor_dtype, self.asRawTensor());
                    errdefer value.deinit();
                    return Tensor(.{ .dtype = target_dtype, .tags = tags }).fromTensor(ctx, value);
                }

                pub fn isContiguous(self: *const Self) bool {
                    return self.value.isContiguous();
                }

                pub fn materialize(self: *const Self, ctx: *ExecContext) !Self {
                    var value = try ctx.materializeTyped(tensor_dtype, self.asRawTensor());
                    errdefer value.deinit();
                    return Self.fromTensor(ctx, value);
                }

                pub fn concat(self: *const Self, ctx: *ExecContext, comptime tag: Tag, others: []const *const Self) !Self {
                    comptime {
                        if (tag_rank != 2) @compileError("block-quantized concat currently requires a rank-2 tensor");
                        if (axis(tag) != 0) @compileError("block-quantized concat currently supports the row axis only");
                    }

                    var raw_inputs = try ctx.allocator.alloc(*const tensor_mod.TensorOf(tensor_dtype), others.len + 1);
                    defer ctx.allocator.free(raw_inputs);

                    raw_inputs[0] = self.asRawTensor();
                    for (others, 0..) |other, i| raw_inputs[i + 1] = other.asRawTensor();

                    var value = try ctx.concatQuantizedRowsTyped(tensor_dtype, raw_inputs);
                    errdefer value.deinit();
                    return Self.fromTensor(ctx, value);
                }

                /// Pack this rank-2 quantized weight into the ISA-best packed matmul
                /// RHS layout for its dtype (see `PackedRhs`): q8_0→x4, q6_k→x4,
                /// q5_k→x8, q4_k→x2mmla on aarch64+i8mm targets else x8. Use
                /// `packRhsLayout` to force a specific layout instead.
                pub fn packRhs(self: *const Self, ctx: *ExecContext) !PackedRhs(tensor_dtype) {
                    comptime if (tag_rank != 2) @compileError("packRhs requires a rank-2 tensor");
                    return switch (comptime tensor_dtype) {
                        .q8_0 => ctx.packMatmulRhsQ8_0x4(self.asRawTensor()),
                        .q6_k => ctx.packMatmulRhsQ6_Kx4(self.asRawTensor()),
                        .q5_k => ctx.packMatmulRhsQ5_Kx8(self.asRawTensor()),
                        .q4_k => if (comptime backend_mod.supports_q4_k_mmla)
                            ctx.packMatmulRhsQ4_Kx2Mmla(self.asRawTensor())
                        else
                            ctx.packMatmulRhsQ4_Kx8(self.asRawTensor()),
                        else => @compileError("packRhs: no packed matmul RHS layout for a ." ++ @tagName(tensor_dtype) ++ " tensor"),
                    };
                }

                /// Explicit-layout escape hatch over `packRhs`: force a specific packed
                /// layout, comptime-validated against the tensor dtype. Needed e.g. to
                /// exercise the fused x8 kernels on hardware where `packRhs` would
                /// select x2mmla, at the cost of the ISA-best kernel.
                pub fn packRhsLayout(self: *const Self, ctx: *ExecContext, comptime layout: backend_mod.PackedRhsLayout) !backend_mod.PackedRhsFor(layout) {
                    comptime {
                        if (tag_rank != 2) @compileError("packRhsLayout requires a rank-2 tensor");
                        const want: DType = switch (layout) {
                            .dense_f32 => @compileError("packRhsLayout(.dense_f32) requires an f32/f16/bf16 tensor; call its packRhs method"),
                            .q8_0x4 => .q8_0,
                            .q6_kx4 => .q6_k,
                            .q4_kx8, .q4_kx2mmla => .q4_k,
                            .q5_kx8 => .q5_k,
                            .q4_kx4 => @compileError("packRhsLayout: the Q4_Kx4 layout has no facade entry (kernel-comparison surface below the facade)"),
                        };
                        if (tensor_dtype != want) @compileError("packRhsLayout(." ++ @tagName(layout) ++ ") requires a ." ++ @tagName(want) ++ " tensor");
                    }
                    return switch (comptime layout) {
                        .dense_f32 => unreachable,
                        .q8_0x4 => ctx.packMatmulRhsQ8_0x4(self.asRawTensor()),
                        .q6_kx4 => ctx.packMatmulRhsQ6_Kx4(self.asRawTensor()),
                        .q4_kx8 => ctx.packMatmulRhsQ4_Kx8(self.asRawTensor()),
                        .q4_kx2mmla => ctx.packMatmulRhsQ4_Kx2Mmla(self.asRawTensor()),
                        .q5_kx8 => ctx.packMatmulRhsQ5_Kx8(self.asRawTensor()),
                        .q4_kx4 => unreachable, // rejected by the comptime block above
                    };
                }

                pub fn getRows(
                    self: *const Self,
                    ctx: *ExecContext,
                    comptime tag: Tag,
                    indices: []const usize,
                    comptime out_tag: Tag,
                ) !Tensor(.{ .dtype = .f32, .tags = replaceTag(tags, tag, out_tag) }) {
                    comptime {
                        if (tag_rank != 2) @compileError("quantized getRows currently requires a rank-2 tensor");
                        if (axis(tag) != 0) @compileError("quantized getRows gathers rows from the first axis");
                    }
                    const result_tags = replaceTag(tags, tag, out_tag);
                    var value = try ctx.getRowsQuantizedTyped(tensor_dtype, self.asRawTensor(), indices);
                    errdefer value.deinit();
                    return Tensor(.{ .dtype = .f32, .tags = result_tags }).fromTensor(ctx, value);
                }
            };
        }

        pub fn TypedConstantTensor(comptime tags_spec: anytype, comptime tensor_dtype: DType) type {
            if (comptime dtype_mod.supportsForwardFloatMath(tensor_dtype)) {
                return TypedFloatConstantTensor(tags_spec, tensor_dtype);
            }
            return TypedScalarConstantTensor(tags_spec, tensor_dtype);
        }

        /// Shared constructor/accessor tier for the two typed-constant branches,
        /// extending the `typedConstant*` shared-fn pattern below to everything the
        /// branches duplicated verbatim: the branches differ only in which MATH decls
        /// they add (`TypedFloatConstantTensor` layers to/add/.../dot on top).
        pub fn TypedConstantBase(comptime SelfT: type, comptime tags: anytype, comptime tensor_dtype: DType) type {
            const tensor_rank = rawRank(tags.len);
            const RawTypedTensor = tensor_mod.TensorOf(tensor_dtype);
            const Elem = Scalar(tensor_dtype);

            return struct {
                /// Consumes `value` on success; on error, ownership stays with the caller.
                pub fn constant(ctx: *ExecContext, value: RawTypedTensor) !SelfT {
                    _ = ctx;
                    var v = value;
                    try validateTensorRankOf(tensor_dtype, tags, &v);
                    return .{ .value = v };
                }

                pub fn fromTensor(ctx: *ExecContext, value: RawTypedTensor) !SelfT {
                    return try @This().constant(ctx, value);
                }

                pub fn fromSlice(ctx: *ExecContext, raw_shape: [tensor_rank]usize, values: []const Elem) !SelfT {
                    var value = try ctx.fromSliceRankTyped(tensor_dtype, tensor_rank, raw_shape, values);
                    errdefer value.deinit();
                    return try @This().constant(ctx, value);
                }

                /// Zero-copy wrap caller-owned READ-ONLY typed storage as a no-grad
                /// constant tensor without `@constCast` at the call site. Read-only
                /// borrow: `values` must outlive the tensor and must not be mutated
                /// (see the f32 `fromBorrowedConstSlice` contract).
                pub fn fromBorrowedConstSlice(ctx: *ExecContext, raw_shape: [tensor_rank]usize, values: []const Elem) !SelfT {
                    var value = try ctx.fromBorrowedSliceRankTyped(tensor_dtype, tensor_rank, raw_shape, @constCast(values));
                    errdefer value.deinit();
                    return try @This().constant(ctx, value);
                }

                /// Allocate an uninitialized no-grad typed tensor of the tag-implied rank.
                pub fn empty(ctx: *ExecContext, raw_shape: [tensor_rank]usize) !SelfT {
                    var value = try ctx.emptyRankTyped(tensor_dtype, tensor_rank, raw_shape);
                    errdefer value.deinit();
                    return try @This().constant(ctx, value);
                }

                /// Allocate a zero-filled no-grad typed tensor.
                pub fn zeros(ctx: *ExecContext, raw_shape: [tensor_rank]usize) !SelfT {
                    var value = try ctx.zerosTyped(tensor_dtype, &raw_shape);
                    errdefer value.deinit();
                    return try @This().constant(ctx, value);
                }

                /// Allocate a one-filled no-grad typed tensor.
                pub fn ones(ctx: *ExecContext, raw_shape: [tensor_rank]usize) !SelfT {
                    var value = try ctx.onesTyped(tensor_dtype, &raw_shape);
                    errdefer value.deinit();
                    return try @This().constant(ctx, value);
                }

                /// No-grad tensor of uniform integer draws in `[low, high)`
                /// (torch.randint) from the deterministic counter-based stream at
                /// `seed` (§6.8): element i is a pure function of `(seed, i)` via
                /// the widening multiply-shift map (`fucina.rng.randintFill`).
                /// i64-only — the repo-wide index dtype; cast with `to` for
                /// narrower integers. `low >= high` is `InvalidShape`.
                pub fn randint(ctx: *ExecContext, raw_shape: [tensor_rank]usize, seed: u64, low: i64, high: i64) !SelfT {
                    comptime if (tensor_dtype != .i64) @compileError("randint is i64-only (the repo-wide index dtype); cast the result with to()");
                    if (low >= high) return TensorError.InvalidShape;
                    var value = try ctx.emptyRankTyped(tensor_dtype, tensor_rank, raw_shape);
                    errdefer value.deinit();
                    rng.randintFill(seed, value.data(), low, high);
                    return try @This().constant(ctx, value);
                }

                /// Rank-1 no-grad random permutation of `{0, …, n-1}`
                /// (torch.randperm) as i64: Fisher–Yates driven by the
                /// counter-based stream at `seed` (`fucina.rng.randpermFill`;
                /// same seed, same permutation). `n == 0` is `InvalidShape`
                /// (zero-size tensors are not representable).
                pub fn randperm(ctx: *ExecContext, n: usize, seed: u64) !SelfT {
                    comptime {
                        if (tensor_dtype != .i64) @compileError("randperm is i64-only (the repo-wide index dtype)");
                        if (tags.len != 1) @compileError("randperm builds a rank-1 tensor; use a single-tag Tensor type");
                    }
                    if (n == 0) return TensorError.InvalidShape;
                    var value = try ctx.emptyRankTyped(tensor_dtype, tensor_rank, .{n});
                    errdefer value.deinit();
                    rng.randpermFill(seed, value.data());
                    return try @This().constant(ctx, value);
                }

                /// Rank-2 no-grad band mask (the attention-mask constructor), on
                /// the `.bool` branch only: element `(i, j)` is true iff
                /// `i - j <= lower` and `j - i <= upper`; a null bound is
                /// unbounded on that side. Bounds are signed. Causal keep-set =
                /// `(null, 0)`; sliding window of width W = `(W - 1, 0)`; the
                /// tril(k) keep-set = `(null, k)`; triu(k) = `(-k, null)`. Feed
                /// it to `where`/`maskedFill` (broadcast with `broadcastTo` for
                /// batched scores), or cast with `to(.f32)` for the mask-multiply
                /// idiom. Contradictory bounds yield an all-false mask, not an
                /// error.
                pub fn bandMask(ctx: *ExecContext, raw_shape: [tensor_rank]usize, lower: ?i64, upper: ?i64) !SelfT {
                    comptime {
                        if (tensor_dtype != .bool) @compileError("bandMask is a .bool mask constructor; use a .bool Tensor type");
                        if (tags.len != 2) @compileError("bandMask builds a rank-2 [row, col] mask; use a two-tag Tensor type");
                    }
                    var value = try ctx.emptyRankTyped(tensor_dtype, tensor_rank, raw_shape);
                    errdefer value.deinit();
                    const out = value.data();
                    const cols = raw_shape[1];
                    for (0..raw_shape[0]) |i| {
                        for (0..cols) |j| {
                            const d = @as(i64, @intCast(j)) - @as(i64, @intCast(i)); // j - i
                            const in_lower = if (lower) |l| -d <= l else true;
                            const in_upper = if (upper) |u| d <= u else true;
                            out[i * cols + j] = in_lower and in_upper;
                        }
                    }
                    return try @This().constant(ctx, value);
                }

                /// `empty` with `self`'s shape (same dtype and tags via `SelfT`).
                pub fn emptyLike(self: *const SelfT, ctx: *ExecContext) !SelfT {
                    return SelfT.empty(ctx, self.shape());
                }

                /// `zeros` with `self`'s shape.
                pub fn zerosLike(self: *const SelfT, ctx: *ExecContext) !SelfT {
                    return SelfT.zeros(ctx, self.shape());
                }

                /// `ones` with `self`'s shape.
                pub fn onesLike(self: *const SelfT, ctx: *ExecContext) !SelfT {
                    return SelfT.ones(ctx, self.shape());
                }

                pub fn item(self: *const SelfT) !Elem {
                    if (!self.value.isScalar()) return TensorError.InvalidShape;
                    return (try self.value.dataConstChecked())[0];
                }

                pub fn data(self: *SelfT) ![]Elem {
                    return self.value.dataChecked();
                }

                pub fn dataConst(self: *const SelfT) ![]const Elem {
                    return self.value.dataConstChecked();
                }

                pub fn copyTo(self: *const SelfT, dst: []Elem) !void {
                    return self.value.copyTo(dst);
                }

                pub fn axis(comptime tag: Tag) usize {
                    return tagIndexOrCompileError(tags, tag);
                }

                pub fn hasTag(comptime tag: Tag) bool {
                    return comptime tagIndex(tags, tag) != null;
                }
            };
        }

        pub fn TypedScalarConstantTensor(comptime tags_spec: anytype, comptime tensor_dtype: DType) type {
            const tags = normalizeTags(tags_spec);
            comptime validateUniqueTags(tags);
            const tag_rank = tags.len;
            if (tag_rank > tensor_mod.max_rank) @compileError("too many tensor tags");

            const RawTypedTensor = tensor_mod.TensorOf(tensor_dtype);

            return struct {
                pub const axis_tags = tags;
                pub const tag_count = tag_rank;
                pub const tensor_rank = rawRank(tag_rank);
                pub const dtype = tensor_dtype;

                value: RawTypedTensor,
                const Self = @This();
                const base = TypedConstantBase(Self, tags, tensor_dtype);

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
                pub const item = base.item;
                pub const data = base.data;
                pub const dataConst = base.dataConst;
                pub const copyTo = base.copyTo;
                pub const axis = base.axis;
                pub const hasTag = base.hasTag;

                pub const deinit = typedConstantDeinit;
                pub const asRawTensor = typedConstantAsRawTensor;
                pub const requiresGrad = typedConstantRequiresGrad;

                pub const dim = typedConstantDim;
                pub const shape = typedConstantShape;
                pub const isContiguous = typedConstantIsContiguous;
                pub const materialize = typedConstantMaterialize;
                pub const withTags = typedConstantWithTags;
                pub const alignTo = typedConstantAlignTo;
                pub const permuteTo = typedConstantPermuteTo;
                pub const transpose = typedConstantTranspose;
                pub const insertAxis = typedConstantInsertAxis;
                pub const squeeze = typedConstantSqueeze;
                pub const broadcastTo = typedConstantBroadcastTo;
                pub const gather = typedConstantGather;
                pub const narrow = typedConstantNarrow;
                pub const concat = typedConstantConcat;
                pub const setSlice = typedConstantSetSlice;
                pub const setRows = typedConstantSetRows;

                // Integer forward math (§4.19): wrapping two's-complement
                // pointwise, explicit division/remainder, bitwise combinators,
                // i64-returning reductions, and scalar casts. On `.bool` the
                // arithmetic entries are compile errors — only `to` and the
                // counting `sum`/`sumAll` apply.
                pub const to = typedConstantTo;
                pub const add = typedConstantAdd;
                pub const sub = typedConstantSub;
                pub const mul = typedConstantMul;
                pub const maximum = typedConstantMaximum;
                pub const minimum = typedConstantMinimum;
                pub const divTrunc = typedConstantDivTrunc;
                pub const divFloor = typedConstantDivFloor;
                pub const rem = typedConstantRem;
                pub const mod = typedConstantMod;
                pub const bitAnd = typedConstantBitAnd;
                pub const bitOr = typedConstantBitOr;
                pub const bitXor = typedConstantBitXor;
                pub const sum = typedConstantSum;
                pub const sumAll = typedConstantSumAll;

                // Masks (§4.6): integer `compare` is exact at any magnitude; the
                // logical combinators live on the `.bool` branch.
                pub const compare = typedConstantCompare;
                pub const logicalAnd = typedConstantLogicalAnd;
                pub const logicalOr = typedConstantLogicalOr;
                pub const logicalXor = typedConstantLogicalXor;
                pub const logicalNot = typedConstantLogicalNot;
            };
        }

        pub fn TypedFloatConstantTensor(comptime tags_spec: anytype, comptime tensor_dtype: DType) type {
            const tags = normalizeTags(tags_spec);
            comptime validateUniqueTags(tags);
            const tag_rank = tags.len;
            if (tag_rank > tensor_mod.max_rank) @compileError("too many tensor tags");

            const RawTypedTensor = tensor_mod.TensorOf(tensor_dtype);

            return struct {
                pub const axis_tags = tags;
                pub const tag_count = tag_rank;
                pub const tensor_rank = rawRank(tag_rank);
                pub const dtype = tensor_dtype;

                value: RawTypedTensor,
                grad_state: ?*GradState = null,
                scope_owned: bool = false,
                const Self = @This();
                const base = TypedConstantBase(Self, tags, tensor_dtype);

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
                pub const item = base.item;
                pub const data = base.data;
                pub const dataConst = base.dataConst;
                pub const copyTo = base.copyTo;
                pub const axis = base.axis;
                pub const hasTag = base.hasTag;

                /// Trainable 16-bit leaf: the VALUE is stored in this dtype, the
                /// accumulated gradient is ALWAYS f32 (there is no 16-bit gradient
                /// anywhere). f16/bf16 only — f64 training is unsupported.
                pub fn variable(ctx: *ExecContext, value: RawTypedTensor) !Self {
                    comptime requireHalfFloatGrad(tensor_dtype, "variable");
                    var v = value;
                    try validateTensorRankOf(tensor_dtype, tags, &v);
                    const state = try GradState.leaf(ctx.allocator);
                    errdefer state.deinit();
                    return .{ .value = v, .grad_state = state };
                }

                pub fn variableFromSlice(ctx: *ExecContext, raw_shape: [tensor_rank]usize, values: []const Scalar(tensor_dtype)) !Self {
                    comptime requireHalfFloatGrad(tensor_dtype, "variableFromSlice");
                    var value = try ctx.fromSliceRankTyped(tensor_dtype, tensor_rank, raw_shape, values);
                    errdefer value.deinit();
                    return try Self.variable(ctx, value);
                }

                pub fn requiresGrad(self: *const Self) bool {
                    return self.grad_state != null;
                }

                pub fn zeroGrad(self: *const Self) void {
                    if (self.grad_state) |state| state.zeroGrad();
                }

                /// The accumulated gradient as an owned f32 constant (null before
                /// backward). The gradient of a 16-bit tensor is f32 by contract.
                pub fn grad(self: *const Self, ctx: *ExecContext) !?Tensor(.{ .dtype = .f32, .tags = tags }) {
                    comptime requireHalfFloatGrad(tensor_dtype, "grad");
                    const state = self.grad_state orelse return null;
                    var value = (try state.gradClone(ctx.allocator)) orelse return null;
                    errdefer value.deinit();
                    return try Tensor(.{ .dtype = .f32, .tags = tags }).constant(ctx, value);
                }

                /// Aliasing f32 view of the accumulated gradient (see `grad`).
                pub fn gradView(self: *const Self, ctx: *ExecContext) !?Tensor(.{ .dtype = .f32, .tags = tags }) {
                    comptime requireHalfFloatGrad(tensor_dtype, "gradView");
                    const state = self.grad_state orelse return null;
                    var value = (try state.gradView()) orelse return null;
                    errdefer value.deinit();
                    return try Tensor(.{ .dtype = .f32, .tags = tags }).constant(ctx, value);
                }

                /// No-grad view of the same storage (caller-owned constant).
                pub fn detach(self: *const Self, ctx: *ExecContext) !Self {
                    var value = try self.value.cloneView();
                    errdefer value.deinit();
                    return Self.fromTensor(ctx, value);
                }

                pub fn deinit(self: *Self) void {
                    if (self.scope_owned) return; // borrow: the exec scope owns value + node
                    self.value.deinit();
                    if (self.grad_state) |state| state.deinit();
                    self.* = undefined;
                }

                pub const asRawTensor = typedConstantAsRawTensor;

                pub const dim = typedConstantDim;
                pub const shape = typedConstantShape;
                pub const isContiguous = typedConstantIsContiguous;
                pub const materialize = typedConstantMaterialize;
                pub const withTags = typedConstantWithTags;
                pub const alignTo = typedConstantAlignTo;
                pub const permuteTo = typedConstantPermuteTo;
                pub const transpose = typedConstantTranspose;
                pub const insertAxis = typedConstantInsertAxis;
                pub const squeeze = typedConstantSqueeze;
                pub const broadcastTo = typedConstantBroadcastTo;
                pub const gather = typedConstantGather;
                pub const narrow = typedConstantNarrow;
                pub const concat = typedConstantConcat;
                pub const setSlice = typedConstantSetSlice;
                pub const setRows = typedConstantSetRows;

                pub const to = typedConstantTo;
                pub const add = typedConstantAdd;
                pub const sub = typedConstantSub;
                pub const mul = typedConstantMul;
                pub const div = typedConstantDiv;
                pub const sum = typedConstantSum;
                pub const mean = typedConstantMean;
                pub const sumAll = typedConstantSumAll;
                pub const dot = typedConstantDot;

                /// Snapshot this rank-2 f16/bf16 `[out, contract]` weight as f32
                /// output-row panels for a FloatTensor `dotPacked`. Widening happens
                /// once here; the returned resource is caller-owned and no-grad.
                pub fn packRhs(self: *const Self, ctx: *ExecContext) !PackedRhs(tensor_dtype) {
                    comptime {
                        if (tag_rank != 2) @compileError("packRhs requires a rank-2 tensor");
                        if (tensor_dtype != .f16 and tensor_dtype != .bf16)
                            @compileError("dense packRhs supports f32, f16, and bf16 weights");
                    }
                    if (self.requiresGrad()) return error.GradientPackedMatmulUnsupported;
                    return ctx.packDenseMatmulRhsTyped(tensor_dtype, self.asRawTensor());
                }

                // Structural ops (views / data movement; every typed float dtype).
                pub const split = typedConstantSplit;
                pub const merge = typedConstantMerge;
                pub const flatten = typedConstantFlatten;
                pub const reshape = typedConstantReshape;
                pub const sliceStep = typedConstantSliceStep;
                pub const flip = typedConstantFlip;
                pub const roll = typedConstantRoll;
                pub const stack = typedConstantStack;
                pub const repeatAxis = typedConstantRepeatAxis;
                pub const scale = typedConstantScale;
                pub const divScalar = typedConstantDivScalar;

                // Widened forward math (f16/bf16 only: f32 compute, one final round).
                pub const unary = typedConstantUnary;
                pub const relu = TypedUnaryMethod(.relu).call;
                pub const exp = TypedUnaryMethod(.exp).call;
                pub const sqrt = TypedUnaryMethod(.sqrt).call;
                pub const rsqrt = TypedUnaryMethod(.rsqrt).call;
                pub const sigmoid = TypedUnaryMethod(.sigmoid).call;
                pub const silu = TypedUnaryMethod(.silu).call;
                pub const log = TypedUnaryMethod(.log).call;
                pub const log1p = TypedUnaryMethod(.log1p).call;
                pub const neg = TypedUnaryMethod(.neg).call;
                pub const abs = TypedUnaryMethod(.abs).call;
                pub const sin = TypedUnaryMethod(.sin).call;
                pub const cos = TypedUnaryMethod(.cos).call;
                pub const tanh = TypedUnaryMethod(.tanh).call;
                pub const fastTanh = TypedUnaryMethod(.fast_tanh).call;
                pub const softcap30 = TypedUnaryMethod(.softcap_30).call;
                pub const softcap15 = TypedUnaryMethod(.softcap_15).call;
                pub const gelu = TypedUnaryMethod(.gelu).call;
                pub const quickGelu = TypedUnaryMethod(.quick_gelu).call;
                pub const elu = TypedUnaryMethod(.elu).call;
                pub const geluErf = TypedUnaryMethod(.gelu_erf).call;
                pub const erf = TypedUnaryMethod(.erf).call;
                pub const floor = TypedUnaryMethod(.floor).call;
                pub const ceil = TypedUnaryMethod(.ceil).call;
                pub const round = TypedUnaryMethod(.round).call;
                pub const sign = TypedUnaryMethod(.sign).call;
                pub const reciprocal = TypedUnaryMethod(.reciprocal).call;
                pub const leakyRelu = typedConstantLeakyRelu;
                pub const clamp = typedConstantClamp;
                pub const addScalar = typedConstantAddScalar;
                pub const subScalar = typedConstantSubScalar;
                pub const powScalar = typedConstantPowScalar;
                pub const maximum = typedConstantMaximum;
                pub const minimum = typedConstantMinimum;
                pub const gated = typedConstantGated;
                pub const glu = typedConstantGlu;
                pub const swiglu = typedConstantSwiglu;
                pub const geglu = typedConstantGeglu;
                pub const softmax = typedConstantSoftmax;
                pub const logSoftmax = typedConstantLogSoftmax;
                pub const rmsNorm = typedConstantRmsNorm;
                pub const rmsNormMul = typedConstantRmsNormMul;
                pub const layerNorm = typedConstantLayerNorm;
                pub const cumsum = typedConstantCumsum;
                pub const cumprod = typedConstantCumprod;
                pub const where = typedConstantWhere;
                pub const maskedFill = typedConstantMaskedFill;
                pub const compare = typedConstantCompare;
                pub const pad = typedConstantPad;
                pub const einsum = typedConstantEinsum;

                // Widened reductions (f16/bf16 only; f32 result per §8.3).
                pub const max = typedConstantMax;
                pub const min = typedConstantMin;
                pub const argmax = typedConstantArgmax;
                pub const prod = typedConstantProd;
                pub const variance = typedConstantVariance;
                pub const logsumexp = typedConstantLogsumexp;
            };
        }

        pub fn typedConstantDeinit(self: anytype) void {
            self.value.deinit();
            self.* = undefined;
        }

        pub fn typedConstantAsRawTensor(self: anytype) *const tensor_mod.TensorOf(TensorObject(@TypeOf(self)).dtype) {
            return &self.value;
        }

        pub fn typedConstantRequiresGrad(_: anytype) bool {
            return false;
        }

        pub fn typedConstantDim(self: anytype, comptime tag: Tag) usize {
            const Self = TensorObject(@TypeOf(self));
            return self.asRawTensor().shape.at(Self.axis(tag));
        }

        pub fn typedConstantShape(self: anytype) [TensorObject(@TypeOf(self)).tensor_rank]usize {
            const Self = TensorObject(@TypeOf(self));
            var out: [Self.tensor_rank]usize = undefined;
            inline for (0..Self.tensor_rank) |i| {
                out[i] = self.asRawTensor().shape.at(i);
            }
            return out;
        }

        pub fn typedConstantIsContiguous(self: anytype) bool {
            return self.value.isContiguous();
        }

        pub fn typedConstantMaterialize(self: anytype, ctx: *ExecContext) !TensorObject(@TypeOf(self)) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            var value = try ctx.materializeTyped(Self.dtype, self.asRawTensor());
            errdefer value.deinit();
            return Self.fromTensor(ctx, value);
        }

        pub fn typedConstantWithTags(self: anytype, ctx: *ExecContext, comptime new_tags_spec: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = normalizeTags(new_tags_spec) }) {
            const Self = TensorObject(@TypeOf(self));
            const new_tags = normalizeTags(new_tags_spec);
            comptime {
                validateUniqueTags(new_tags);
                if (new_tags.len != Self.tag_count) @compileError("withTags requires the same rank");
            }
            return typedConstantAxisView(self, ctx, identityAxes(Self.tag_count), new_tags);
        }

        pub fn typedConstantAlignTo(self: anytype, ctx: *ExecContext, comptime target_tags_spec: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = normalizeTags(target_tags_spec) }) {
            const Self = TensorObject(@TypeOf(self));
            const target_tags = normalizeTags(target_tags_spec);
            return typedConstantAxisView(self, ctx, alignAxes(Self.axis_tags, target_tags), target_tags);
        }

        pub fn typedConstantPermuteTo(self: anytype, ctx: *ExecContext, comptime target_tags_spec: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = normalizeTags(target_tags_spec) }) {
            const Self = TensorObject(@TypeOf(self));
            const target_tags = normalizeTags(target_tags_spec);
            comptime validateSameTagSet(Self.axis_tags, target_tags);
            return typedConstantAxisView(self, ctx, alignAxes(Self.axis_tags, target_tags), target_tags);
        }

        pub fn typedConstantTranspose(self: anytype, ctx: *ExecContext, comptime target_tags_spec: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = normalizeTags(target_tags_spec) }) {
            return typedConstantPermuteTo(self, ctx, target_tags_spec);
        }

        pub fn typedConstantInsertAxis(self: anytype, ctx: *ExecContext, comptime tag: Tag, comptime axis_index: usize) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = insertTagAt(TensorObject(@TypeOf(self)).axis_tags, tag, axis_index) }) {
            const Self = TensorObject(@TypeOf(self));
            const result_tags = insertTagAt(Self.axis_tags, tag, axis_index);
            return typedConstantAxisView(self, ctx, insertAxes(Self.tag_count, axis_index), result_tags);
        }

        pub fn typedConstantSqueeze(self: anytype, ctx: *ExecContext, comptime tag: Tag) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = removeTag(TensorObject(@TypeOf(self)).axis_tags, tag) }) {
            const Self = TensorObject(@TypeOf(self));
            const result_tags = removeTag(Self.axis_tags, tag);
            const axis_index = comptime tagIndexOrCompileError(Self.axis_tags, tag);
            if (self.asRawTensor().shape.at(axis_index) != 1) return TensorError.InvalidShape;
            return typedConstantAxisView(self, ctx, squeezeAxes(Self.tag_count, axis_index), result_tags);
        }

        pub fn typedConstantBroadcastTo(
            self: anytype,
            ctx: *ExecContext,
            comptime target_tags_spec: anytype,
            target_shape: [normalizeTags(target_tags_spec).len]usize,
        ) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = normalizeTags(target_tags_spec) }) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            const target_tags = normalizeTags(target_tags_spec);
            var value = try broadcastTensorToOf(Self.dtype, Self.axis_tags, self.asRawTensor(), target_tags, target_shape);
            errdefer value.deinit();
            return Tensor(.{ .dtype = Self.dtype, .tags = target_tags }).fromTensor(ctx, value);
        }

        pub fn typedConstantGather(
            self: anytype,
            ctx: *ExecContext,
            comptime tag: Tag,
            indices: []const usize,
            comptime out_tag: Tag,
        ) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = replaceTag(TensorObject(@TypeOf(self)).axis_tags, tag, out_tag) }) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            const result_tags = replaceTag(Self.axis_tags, tag, out_tag);
            var value = try ctx.gatherAxisRankTyped(Self.dtype, Self.tag_count, self.asRawTensor(), Self.axis(tag), indices);
            errdefer value.deinit();
            return Tensor(.{ .dtype = Self.dtype, .tags = result_tags }).fromTensor(ctx, value);
        }

        pub fn typedConstantNarrow(self: anytype, ctx: *ExecContext, comptime tag: Tag, start: usize, length: usize) !TensorObject(@TypeOf(self)) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            var value = try ctx.narrowAxisRankTyped(Self.dtype, Self.tag_count, self.asRawTensor(), Self.axis(tag), start, length);
            errdefer value.deinit();
            return Self.fromTensor(ctx, value);
        }

        pub fn typedConstantConcat(self: anytype, ctx: *ExecContext, comptime tag: Tag, others: []const *const TensorObject(@TypeOf(self))) !TensorObject(@TypeOf(self)) {
            try typedRequireNoGrad(self);
            for (others) |other_item| try typedRequireNoGrad(other_item);
            const Self = TensorObject(@TypeOf(self));
            const RawTypedTensor = tensor_mod.TensorOf(Self.dtype);
            var raw_inputs = try ctx.allocator.alloc(*const RawTypedTensor, others.len + 1);
            defer ctx.allocator.free(raw_inputs);

            raw_inputs[0] = self.asRawTensor();
            for (others, 0..) |other, i| raw_inputs[i + 1] = other.asRawTensor();

            var value = try ctx.concatAxisRankTyped(Self.dtype, Self.tag_count, raw_inputs, Self.axis(tag));
            errdefer value.deinit();
            return Self.fromTensor(ctx, value);
        }

        pub fn typedConstantSetSlice(self: anytype, ctx: *ExecContext, comptime tag: Tag, start: usize, update: *const TensorObject(@TypeOf(self))) !TensorObject(@TypeOf(self)) {
            try typedRequireNoGrad(self);
            try typedRequireNoGrad(update);
            const Self = TensorObject(@TypeOf(self));
            var value = try ctx.setSliceAxisRankTyped(Self.dtype, Self.tag_count, self.asRawTensor(), update.asRawTensor(), Self.axis(tag), start);
            errdefer value.deinit();
            return Self.fromTensor(ctx, value);
        }

        pub fn typedConstantSetRows(self: anytype, ctx: *ExecContext, comptime tag: Tag, indices: []const usize, update: *const TensorObject(@TypeOf(self))) !TensorObject(@TypeOf(self)) {
            try typedRequireNoGrad(self);
            try typedRequireNoGrad(update);
            const Self = TensorObject(@TypeOf(self));
            var value = try ctx.setRowsAxisRankTyped(Self.dtype, Self.tag_count, self.asRawTensor(), update.asRawTensor(), Self.axis(tag), indices);
            errdefer value.deinit();
            return Self.fromTensor(ctx, value);
        }

        pub fn typedConstantAxisView(self: anytype, ctx: *ExecContext, comptime axes: anytype, comptime target_tags: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = target_tags }) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            var value = try axisViewTensorOf(Self.dtype, self.asRawTensor(), axes, target_tags);
            errdefer value.deinit();
            return Tensor(.{ .dtype = Self.dtype, .tags = target_tags }).fromTensor(ctx, value);
        }

        pub fn typedConstantTo(self: anytype, ctx: *ExecContext, comptime target_dtype: DType) !Tensor(.{ .dtype = target_dtype, .tags = TensorObject(@TypeOf(self)).axis_tags }) {
            const Self = TensorObject(@TypeOf(self));
            if (comptime target_dtype != .f32) {
                if (self.requiresGrad()) return error.GradientCastUnsupported;
            }
            var value = try ctx.castTyped(Self.dtype, target_dtype, self.asRawTensor());
            errdefer value.deinit();
            if (comptime target_dtype == .f32) {
                if (comptime @hasField(Self, "grad_state")) {
                    // Differentiable widen: the f32 result joins the f32 graph and
                    // the 16-bit source receives the upstream gradient unchanged.
                    return finishOp(Self.axis_tags, ctx, value, self.requiresGrad(), CastBackward(Self.axis_tags), .{ ctx.allocator, self.grad_state });
                }
                // Integer/bool sources are grad-free: a plain f32 constant.
                return finishNoGrad(Self.axis_tags, ctx, value);
            }
            return Tensor(.{ .dtype = target_dtype, .tags = Self.axis_tags }).fromTensor(ctx, value);
        }

        pub fn typedConstantAdd(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = dtype_mod.outputDType(.pointwise, TensorObject(@TypeOf(self)).dtype), .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
            const Self = TensorObject(@TypeOf(self));
            return typedPointwise(Self.dtype, .add, self, ctx, other);
        }

        pub fn typedConstantSub(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = dtype_mod.outputDType(.pointwise, TensorObject(@TypeOf(self)).dtype), .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
            const Self = TensorObject(@TypeOf(self));
            return typedPointwise(Self.dtype, .sub, self, ctx, other);
        }

        pub fn typedConstantMul(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = dtype_mod.outputDType(.pointwise, TensorObject(@TypeOf(self)).dtype), .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
            const Self = TensorObject(@TypeOf(self));
            return typedPointwise(Self.dtype, .mul, self, ctx, other);
        }

        pub fn typedConstantDiv(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = dtype_mod.outputDType(.pointwise, TensorObject(@TypeOf(self)).dtype), .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
            const Self = TensorObject(@TypeOf(self));
            return typedPointwise(Self.dtype, .div, self, ctx, other);
        }

        pub fn typedConstantSum(self: anytype, ctx: *ExecContext, comptime tag: Tag) !Tensor(.{ .dtype = dtype_mod.outputDType(.reduction, TensorObject(@TypeOf(self)).dtype), .tags = removeTag(TensorObject(@TypeOf(self)).axis_tags, tag) }) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            const result_tags = removeTag(Self.axis_tags, tag);
            var value = try ctx.sumAxisRankTyped(Self.dtype, Self.tag_count, self.asRawTensor(), Self.axis(tag));
            errdefer value.deinit();
            return Tensor(.{ .dtype = dtype_mod.outputDType(.reduction, Self.dtype), .tags = result_tags }).fromTensor(ctx, value);
        }

        pub fn typedConstantMean(self: anytype, ctx: *ExecContext, comptime tag: Tag) !Tensor(.{ .dtype = dtype_mod.outputDType(.reduction, TensorObject(@TypeOf(self)).dtype), .tags = removeTag(TensorObject(@TypeOf(self)).axis_tags, tag) }) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            const result_tags = removeTag(Self.axis_tags, tag);
            var value = try ctx.meanAxisRankTyped(Self.dtype, Self.tag_count, self.asRawTensor(), Self.axis(tag));
            errdefer value.deinit();
            return Tensor(.{ .dtype = dtype_mod.outputDType(.reduction, Self.dtype), .tags = result_tags }).fromTensor(ctx, value);
        }

        pub fn typedConstantSumAll(self: anytype, ctx: *ExecContext) !Tensor(.{ .dtype = dtype_mod.outputDType(.reduction, TensorObject(@TypeOf(self)).dtype), .tags = .{} }) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            var value = try ctx.sumTyped(Self.dtype, self.asRawTensor());
            errdefer value.deinit();
            return Tensor(.{ .dtype = dtype_mod.outputDType(.reduction, Self.dtype), .tags = .{} }).fromTensor(ctx, value);
        }

        pub fn typedConstantDot(self: anytype, ctx: *ExecContext, other: anytype, comptime contract_tag: Tag) !Tensor(.{ .dtype = dtype_mod.outputDType(.matmul, TensorObject(@TypeOf(self)).dtype), .tags = dotResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags, contract_tag) }) {
            const Self = TensorObject(@TypeOf(self));
            return typedDot(Self.dtype, self, ctx, other, contract_tag);
        }

        // Widened typed-float ops: forward coverage for ops with no native typed
        // kernel. The input widens to f32, the f32 exec kernel runs, and the result
        // narrows ONCE on store — f32 accumulation with a single final round, the
        // §8.3 policy. f64 is excluded at comptime: f64 math must stay f64, and
        // rounding it through f32 would silently lose precision.
        pub fn requireWidenedTypedFloat(comptime tensor_dtype: DType, comptime what: []const u8) void {
            if (tensor_dtype != .f16 and tensor_dtype != .bf16) {
                @compileError(what ++ " on the typed float branch is f16/bf16 only (it computes through f32; f64 must not round through f32 — cast explicitly)");
            }
        }

        pub fn requireHalfFloatGrad(comptime tensor_dtype: DType, comptime what: []const u8) void {
            if (tensor_dtype != .f16 and tensor_dtype != .bf16) {
                @compileError(what ++ " requires an f16/bf16 tensor (gradients are always f32; f64 training is unsupported)");
            }
        }

        /// Typed forward ops are no-grad: a grad-requiring operand would silently
        /// drop its graph, so it is rejected instead (`to(.f32)` is the trained
        /// path; the differentiable typed entries are `to` and the mixed-RHS
        /// `dot`/`einsum`).
        pub fn typedRequireNoGrad(operand: anytype) !void {
            const Operand = TensorObject(@TypeOf(operand));
            if (comptime @hasField(Operand, "grad_state")) {
                if (operand.grad_state != null) return error.UnsupportedGradient;
            }
        }

        /// Scope payload for a grad-carrying 16-bit result: the exec-scope slot
        /// holds f32 values only, so the typed value travels inside the type-erased
        /// node payload with a destructor that frees value + graph node together.
        pub fn TypedScopePayload(comptime tensor_dtype: DType) type {
            return struct {
                allocator: std.mem.Allocator,
                value: tensor_mod.TensorOf(tensor_dtype),
                state: *GradState,

                fn destroy(ptr: *anyopaque) void {
                    const payload: *@This() = @ptrCast(@alignCast(ptr));
                    payload.value.deinit();
                    payload.state.deinit();
                    payload.allocator.destroy(payload);
                }
            };
        }

        /// `finishOp` for a differentiable op whose RESULT is 16-bit (today: the
        /// f32 → f16/bf16 cast). Same contract as `finishOp`: consumes `value` on
        /// success; under an active exec scope the result is a scope-owned borrow.
        pub fn typedFinishOp(
            comptime tensor_dtype: DType,
            comptime result_tags: anytype,
            ctx: *ExecContext,
            value: tensor_mod.TensorOf(tensor_dtype),
            wants_grad: bool,
            comptime BackwardType: type,
            create_args: anytype,
        ) !Tensor(.{ .dtype = tensor_dtype, .tags = result_tags }) {
            const OutT = Tensor(.{ .dtype = tensor_dtype, .tags = result_tags });
            if (!wants_grad or !control.isGradEnabled()) {
                return OutT.fromTensor(ctx, value);
            }
            if (ctx.execScopeActive()) {
                try ctx.reserveScopeSlot();
                const payload = try ctx.allocator.create(TypedScopePayload(tensor_dtype));
                errdefer ctx.allocator.destroy(payload);
                const state = try core.createNode(BackwardType, create_args);
                payload.* = .{ .allocator = ctx.allocator, .value = value, .state = state };
                ctx.adoptScopeNodeAssumeCapacity(payload, TypedScopePayload(tensor_dtype).destroy);
                return .{ .value = value, .grad_state = state, .scope_owned = true };
            }
            const state = try core.createNode(BackwardType, create_args);
            return .{ .value = value, .grad_state = state };
        }

        /// Shared tail of the widened ops: narrow the f32 kernel result back to
        /// `tensor_dtype` and wrap it as a typed constant.
        pub fn typedFromWidened(comptime tensor_dtype: DType, comptime result_tags: anytype, ctx: *ExecContext, wide_value: *const RawTensor) !Tensor(.{ .dtype = tensor_dtype, .tags = result_tags }) {
            var value = try ctx.castTyped(.f32, tensor_dtype, wide_value);
            errdefer value.deinit();
            return Tensor(.{ .dtype = tensor_dtype, .tags = result_tags }).fromTensor(ctx, value);
        }

        pub fn typedConstantUnary(self: anytype, ctx: *ExecContext, comptime op: UnaryOp) !TensorObject(@TypeOf(self)) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            comptime requireWidenedTypedFloat(Self.dtype, "unary");
            var wide = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
            defer wide.deinit();
            var wide_value = try ctx.unary(op, &wide);
            defer wide_value.deinit();
            return typedFromWidened(Self.dtype, Self.axis_tags, ctx, &wide_value);
        }

        pub fn TypedUnaryMethod(comptime op: UnaryOp) type {
            return struct {
                fn call(self: anytype, ctx: *ExecContext) !TensorObject(@TypeOf(self)) {
                    return typedConstantUnary(self, ctx, op);
                }
            };
        }

        pub fn typedConstantLeakyRelu(self: anytype, ctx: *ExecContext, negative_slope: f32) !TensorObject(@TypeOf(self)) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            comptime requireWidenedTypedFloat(Self.dtype, "leakyRelu");
            var wide = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
            defer wide.deinit();
            var wide_value = try ctx.leakyRelu(&wide, negative_slope);
            defer wide_value.deinit();
            return typedFromWidened(Self.dtype, Self.axis_tags, ctx, &wide_value);
        }

        pub fn typedConstantClamp(self: anytype, ctx: *ExecContext, min_value: f32, max_value: f32) !TensorObject(@TypeOf(self)) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            comptime requireWidenedTypedFloat(Self.dtype, "clamp");
            var wide = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
            defer wide.deinit();
            var wide_value = try ctx.clamp(&wide, min_value, max_value);
            defer wide_value.deinit();
            return typedFromWidened(Self.dtype, Self.axis_tags, ctx, &wide_value);
        }

        pub fn typedConstantScale(self: anytype, ctx: *ExecContext, scalar_value: dtype_mod.Accumulator(TensorObject(@TypeOf(self)).dtype)) !TensorObject(@TypeOf(self)) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            var value = try ctx.scaleTyped(Self.dtype, self.asRawTensor(), scalar_value);
            errdefer value.deinit();
            return Self.fromTensor(ctx, value);
        }

        pub fn typedConstantAddScalar(self: anytype, ctx: *ExecContext, scalar_value: f32) !TensorObject(@TypeOf(self)) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            comptime requireWidenedTypedFloat(Self.dtype, "addScalar");
            var wide = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
            defer wide.deinit();
            var wide_value = try ctx.addScalar(&wide, scalar_value);
            defer wide_value.deinit();
            return typedFromWidened(Self.dtype, Self.axis_tags, ctx, &wide_value);
        }

        pub fn typedConstantSubScalar(self: anytype, ctx: *ExecContext, scalar_value: f32) !TensorObject(@TypeOf(self)) {
            return typedConstantAddScalar(self, ctx, -scalar_value);
        }

        pub fn typedConstantDivScalar(self: anytype, ctx: *ExecContext, scalar_value: dtype_mod.Accumulator(TensorObject(@TypeOf(self)).dtype)) !TensorObject(@TypeOf(self)) {
            return typedConstantScale(self, ctx, 1.0 / scalar_value);
        }

        pub fn typedConstantPowScalar(self: anytype, ctx: *ExecContext, exponent: f32) !TensorObject(@TypeOf(self)) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            comptime requireWidenedTypedFloat(Self.dtype, "powScalar");
            var wide = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
            defer wide.deinit();
            var wide_value = try ctx.powScalar(&wide, exponent);
            defer wide_value.deinit();
            return typedFromWidened(Self.dtype, Self.axis_tags, ctx, &wide_value);
        }

        /// Widened binary pointwise (`maximum`/`minimum`): both operands widen,
        /// the f32 tag-broadcast kernel runs, the result narrows.
        pub fn typedWidenedPointwise(
            comptime op: PointwiseOp,
            self: anytype,
            ctx: *ExecContext,
            other: anytype,
        ) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
            try typedRequireNoGrad(self);
            try typedRequireNoGrad(other);
            const Self = TensorObject(@TypeOf(self));
            const Other = TensorObject(@TypeOf(other));
            comptime requireWidenedTypedFloat(Self.dtype, "maximum/minimum");
            if (Other.dtype != Self.dtype) @compileError("typed pointwise requires matching dtypes; cast explicitly");
            const result_tags = pointwiseResultTags(Self.axis_tags, Other.axis_tags);
            const other_ptr = tensorObjectPtrFrom(@TypeOf(other), &other);
            var wide_left = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
            defer wide_left.deinit();
            var wide_right = try ctx.castTyped(Self.dtype, .f32, other_ptr.asRawTensor());
            defer wide_right.deinit();
            var wide_value = try tag_ops.pointwise(op, Self.axis_tags, &wide_left, ctx, Other.axis_tags, &wide_right);
            defer wide_value.deinit();
            return typedFromWidened(Self.dtype, result_tags, ctx, &wide_value);
        }

        pub fn typedConstantMaximum(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
            const Self = TensorObject(@TypeOf(self));
            if (comptime dtype_mod.supportsIntMath(Self.dtype)) return typedPointwise(Self.dtype, .max, self, ctx, other);
            return typedWidenedPointwise(.max, self, ctx, other);
        }

        pub fn typedConstantMinimum(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
            const Self = TensorObject(@TypeOf(self));
            if (comptime dtype_mod.supportsIntMath(Self.dtype)) return typedPointwise(Self.dtype, .min, self, ctx, other);
            return typedWidenedPointwise(.min, self, ctx, other);
        }

        /// Explicit integer division with the standard tag-broadcast rule:
        /// `.trunc` rounds toward zero, `.floor` toward negative infinity; a zero
        /// divisor is `error.DivisionByZero`; minInt/-1 wraps (the +% contract).
        pub fn typedIntDiv(
            comptime mode: enum { trunc, floor },
            self: anytype,
            ctx: *ExecContext,
            other: anytype,
        ) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
            const Self = TensorObject(@TypeOf(self));
            const Other = TensorObject(@TypeOf(other));
            comptime {
                if (!dtype_mod.supportsIntMath(Self.dtype)) @compileError("divTrunc/divFloor are integer ops; floats use div");
            }
            if (Other.dtype != Self.dtype) @compileError("typed pointwise requires matching dtypes; cast explicitly");
            const left_tags = Self.axis_tags;
            const right_tags = Other.axis_tags;
            const left = tensorObjectPtrFrom(@TypeOf(self), &self);
            const right = tensorObjectPtrFrom(@TypeOf(other), &other);
            const result_tags = pointwiseResultTags(left_tags, right_tags);
            const result_shape = try pointwiseShapeOf(Self.dtype, result_tags, left_tags, left.asRawTensor(), right_tags, right.asRawTensor());

            var left_view = try broadcastTensorToOf(Self.dtype, left_tags, left.asRawTensor(), result_tags, result_shape);
            defer left_view.deinit();
            var right_view = try broadcastTensorToOf(Self.dtype, right_tags, right.asRawTensor(), result_tags, result_shape);
            defer right_view.deinit();

            var value = switch (mode) {
                .trunc => try ctx.divTruncRankTyped(Self.dtype, rawRank(result_tags.len), &left_view, &right_view),
                .floor => try ctx.divFloorRankTyped(Self.dtype, rawRank(result_tags.len), &left_view, &right_view),
            };
            errdefer value.deinit();
            return Tensor(.{ .dtype = Self.dtype, .tags = result_tags }).fromTensor(ctx, value);
        }

        pub fn typedConstantDivTrunc(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
            return typedIntDiv(.trunc, self, ctx, other);
        }

        pub fn typedConstantDivFloor(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
            return typedIntDiv(.floor, self, ctx, other);
        }

        /// Explicit integer remainder with the standard tag-broadcast rule:
        /// `rem` pairs `divTrunc` (sign of the dividend, C `%`), `mod` pairs
        /// `divFloor` (sign of the divisor, Python/numpy `%` and Zig's `@mod`).
        /// A zero divisor is `error.DivisionByZero`; minInt % -1 is 0.
        pub fn typedIntMod(
            comptime mode: enum { rem, mod },
            self: anytype,
            ctx: *ExecContext,
            other: anytype,
        ) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
            const Self = TensorObject(@TypeOf(self));
            const Other = TensorObject(@TypeOf(other));
            comptime {
                if (!dtype_mod.supportsIntMath(Self.dtype)) @compileError("rem/mod are integer ops; no float counterpart is defined");
            }
            if (Other.dtype != Self.dtype) @compileError("typed pointwise requires matching dtypes; cast explicitly");
            const left_tags = Self.axis_tags;
            const right_tags = Other.axis_tags;
            const left = tensorObjectPtrFrom(@TypeOf(self), &self);
            const right = tensorObjectPtrFrom(@TypeOf(other), &other);
            const result_tags = pointwiseResultTags(left_tags, right_tags);
            const result_shape = try pointwiseShapeOf(Self.dtype, result_tags, left_tags, left.asRawTensor(), right_tags, right.asRawTensor());

            var left_view = try broadcastTensorToOf(Self.dtype, left_tags, left.asRawTensor(), result_tags, result_shape);
            defer left_view.deinit();
            var right_view = try broadcastTensorToOf(Self.dtype, right_tags, right.asRawTensor(), result_tags, result_shape);
            defer right_view.deinit();

            var value = switch (mode) {
                .rem => try ctx.remRankTyped(Self.dtype, rawRank(result_tags.len), &left_view, &right_view),
                .mod => try ctx.modRankTyped(Self.dtype, rawRank(result_tags.len), &left_view, &right_view),
            };
            errdefer value.deinit();
            return Tensor(.{ .dtype = Self.dtype, .tags = result_tags }).fromTensor(ctx, value);
        }

        pub fn typedConstantRem(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
            return typedIntMod(.rem, self, ctx, other);
        }

        pub fn typedConstantMod(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
            return typedIntMod(.mod, self, ctx, other);
        }

        /// Bitwise combinators on two's-complement bit patterns, with the standard
        /// tag-broadcast rule. Integer dtypes only — `.bool` masks use the
        /// truthiness `logicalAnd/Or/Xor`, and floats have no bit-pattern algebra.
        pub fn typedIntBitwise(
            comptime op: exec_mod.IntBitwiseOp,
            self: anytype,
            ctx: *ExecContext,
            other: anytype,
        ) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
            const Self = TensorObject(@TypeOf(self));
            const Other = TensorObject(@TypeOf(other));
            comptime {
                if (!dtype_mod.supportsIntMath(Self.dtype)) @compileError("bitAnd/bitOr/bitXor are integer ops; `.bool` masks use logicalAnd/Or/Xor");
            }
            if (Other.dtype != Self.dtype) @compileError("typed pointwise requires matching dtypes; cast explicitly");
            const left_tags = Self.axis_tags;
            const right_tags = Other.axis_tags;
            const left = tensorObjectPtrFrom(@TypeOf(self), &self);
            const right = tensorObjectPtrFrom(@TypeOf(other), &other);
            const result_tags = pointwiseResultTags(left_tags, right_tags);
            const result_shape = try pointwiseShapeOf(Self.dtype, result_tags, left_tags, left.asRawTensor(), right_tags, right.asRawTensor());

            var left_view = try broadcastTensorToOf(Self.dtype, left_tags, left.asRawTensor(), result_tags, result_shape);
            defer left_view.deinit();
            var right_view = try broadcastTensorToOf(Self.dtype, right_tags, right.asRawTensor(), result_tags, result_shape);
            defer right_view.deinit();

            var value = try ctx.bitwiseRankTyped(Self.dtype, rawRank(result_tags.len), op, &left_view, &right_view);
            errdefer value.deinit();
            return Tensor(.{ .dtype = Self.dtype, .tags = result_tags }).fromTensor(ctx, value);
        }

        pub fn typedConstantBitAnd(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
            return typedIntBitwise(.b_and, self, ctx, other);
        }

        pub fn typedConstantBitOr(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
            return typedIntBitwise(.b_or, self, ctx, other);
        }

        pub fn typedConstantBitXor(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
            return typedIntBitwise(.b_xor, self, ctx, other);
        }

        /// Logical ops on the `.bool` branch (the mask combinators): `.bool`
        /// output; `other` may be `.bool` or float (truthiness).
        pub fn typedLogicalBinary(comptime op: exec_mod.LogicalOp, self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = .bool, .tags = TensorObject(@TypeOf(self)).axis_tags }) {
            const Self = TensorObject(@TypeOf(self));
            comptime {
                if (Self.dtype != .bool) @compileError("logical ops on the typed branch are .bool-only; cast explicitly");
            }
            const Other = TensorObject(@TypeOf(other));
            comptime {
                if (Other.dtype != .bool and !dtype_mod.supportsForwardFloatMath(Other.dtype))
                    @compileError("logical ops take .bool or float operands; cast integer masks explicitly");
            }
            const other_ptr = tensorObjectPtrFrom(@TypeOf(other), &other);
            var value = try ctx.logicalTyped(op, .bool, Other.dtype, self.asRawTensor(), other_ptr.asRawTensor());
            errdefer value.deinit();
            return Tensor(.{ .dtype = .bool, .tags = Self.axis_tags }).fromTensor(ctx, value);
        }

        pub fn typedConstantLogicalAnd(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = .bool, .tags = TensorObject(@TypeOf(self)).axis_tags }) {
            return typedLogicalBinary(.l_and, self, ctx, other);
        }

        pub fn typedConstantLogicalOr(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = .bool, .tags = TensorObject(@TypeOf(self)).axis_tags }) {
            return typedLogicalBinary(.l_or, self, ctx, other);
        }

        pub fn typedConstantLogicalXor(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = .bool, .tags = TensorObject(@TypeOf(self)).axis_tags }) {
            return typedLogicalBinary(.l_xor, self, ctx, other);
        }

        pub fn typedConstantLogicalNot(self: anytype, ctx: *ExecContext) !Tensor(.{ .dtype = .bool, .tags = TensorObject(@TypeOf(self)).axis_tags }) {
            const Self = TensorObject(@TypeOf(self));
            comptime {
                if (Self.dtype != .bool) @compileError("logical ops on the typed branch are .bool-only; cast explicitly");
            }
            var value = try ctx.logicalNotTyped(.bool, self.asRawTensor());
            errdefer value.deinit();
            return Tensor(.{ .dtype = .bool, .tags = Self.axis_tags }).fromTensor(ctx, value);
        }

        pub fn typedConstantGated(
            self: anytype,
            ctx: *ExecContext,
            other: anytype,
            comptime op: GatedOp,
        ) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
            try typedRequireNoGrad(self);
            try typedRequireNoGrad(other);
            const Self = TensorObject(@TypeOf(self));
            const Other = TensorObject(@TypeOf(other));
            comptime requireWidenedTypedFloat(Self.dtype, "gated");
            if (Other.dtype != Self.dtype) @compileError("typed gated requires matching dtypes; cast explicitly");
            const result_tags = pointwiseResultTags(Self.axis_tags, Other.axis_tags);
            const other_ptr = tensorObjectPtrFrom(@TypeOf(other), &other);
            var wide_left = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
            defer wide_left.deinit();
            var wide_right = try ctx.castTyped(Self.dtype, .f32, other_ptr.asRawTensor());
            defer wide_right.deinit();
            var wide_value = try tag_ops.gatedPointwise(op, Self.axis_tags, &wide_left, ctx, Other.axis_tags, &wide_right);
            defer wide_value.deinit();
            return typedFromWidened(Self.dtype, result_tags, ctx, &wide_value);
        }

        pub fn typedConstantGlu(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
            return typedConstantGated(self, ctx, other, .glu);
        }

        pub fn typedConstantSwiglu(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
            return typedConstantGated(self, ctx, other, .swiglu);
        }

        pub fn typedConstantGeglu(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
            return typedConstantGated(self, ctx, other, .geglu);
        }

        pub fn typedConstantSoftmax(self: anytype, ctx: *ExecContext, comptime tag: Tag, options: anytype) !TensorObject(@TypeOf(self)) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            comptime requireWidenedTypedFloat(Self.dtype, "softmax");
            comptime {
                const Options = @TypeOf(options);
                if (@typeInfo(Options) != .@"struct" or @typeInfo(Options).@"struct".fields.len != 0) {
                    @compileError("typed softmax supports only plain .{} options; cast to f32 for the ext path (mask/sinks/causal/scale)");
                }
            }
            var wide = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
            defer wide.deinit();
            var wide_value = try ctx.softmaxAxisRank(Self.tag_count, &wide, Self.axis(tag));
            defer wide_value.deinit();
            return typedFromWidened(Self.dtype, Self.axis_tags, ctx, &wide_value);
        }

        pub fn typedConstantLogSoftmax(self: anytype, ctx: *ExecContext, comptime tag: Tag) !TensorObject(@TypeOf(self)) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            comptime requireWidenedTypedFloat(Self.dtype, "logSoftmax");
            var wide = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
            defer wide.deinit();
            var wide_value = try ctx.logSoftmaxAxisRank(Self.tag_count, &wide, Self.axis(tag));
            defer wide_value.deinit();
            return typedFromWidened(Self.dtype, Self.axis_tags, ctx, &wide_value);
        }

        pub fn typedConstantRmsNorm(self: anytype, ctx: *ExecContext, comptime tag: Tag, eps: f32) !TensorObject(@TypeOf(self)) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            comptime requireWidenedTypedFloat(Self.dtype, "rmsNorm");
            var wide = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
            defer wide.deinit();
            var wide_value = try ctx.rmsNormAxisRank(Self.tag_count, &wide, Self.axis(tag), eps);
            defer wide_value.deinit();
            return typedFromWidened(Self.dtype, Self.axis_tags, ctx, &wide_value);
        }

        pub fn typedConstantRmsNormMul(
            self: anytype,
            ctx: *ExecContext,
            comptime tag: Tag,
            weight: *const Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = .{tag} }),
            eps: f32,
        ) !TensorObject(@TypeOf(self)) {
            try typedRequireNoGrad(self);
            try typedRequireNoGrad(weight);
            const Self = TensorObject(@TypeOf(self));
            comptime requireWidenedTypedFloat(Self.dtype, "rmsNormMul");
            var wide = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
            defer wide.deinit();
            var wide_weight = try ctx.castTyped(Self.dtype, .f32, weight.asRawTensor());
            defer wide_weight.deinit();
            var wide_value = try ctx.rmsNormMulAxisRank(Self.tag_count, &wide, &wide_weight, Self.axis(tag), eps);
            defer wide_value.deinit();
            return typedFromWidened(Self.dtype, Self.axis_tags, ctx, &wide_value);
        }

        pub fn typedConstantLayerNorm(self: anytype, ctx: *ExecContext, comptime tag: Tag, eps: f32, options: anytype) !TensorObject(@TypeOf(self)) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            comptime requireWidenedTypedFloat(Self.dtype, "layerNorm");
            comptime {
                const Options = @TypeOf(options);
                if (@typeInfo(Options) != .@"struct" or @typeInfo(Options).@"struct".fields.len != 0) {
                    @compileError("typed layerNorm supports only plain .{} options; cast to f32 for the affine path");
                }
            }
            var wide = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
            defer wide.deinit();
            var wide_value = try ctx.layerNormAxisRank(Self.tag_count, &wide, Self.axis(tag), eps);
            defer wide_value.deinit();
            return typedFromWidened(Self.dtype, Self.axis_tags, ctx, &wide_value);
        }

        // Widened reductions return f32 like the native typed sum/mean (§8.3:
        // reductions on 16-bit floats keep the accumulator dtype).
        pub fn typedConstantLogsumexp(self: anytype, ctx: *ExecContext, comptime tag: Tag) !Tensor(.{ .dtype = .f32, .tags = removeTag(TensorObject(@TypeOf(self)).axis_tags, tag) }) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            comptime requireWidenedTypedFloat(Self.dtype, "logsumexp");
            const result_tags = removeTag(Self.axis_tags, tag);
            var wide = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
            defer wide.deinit();
            var value = try ctx.logsumexpAxisRank(Self.tag_count, &wide, Self.axis(tag));
            errdefer value.deinit();
            return Tensor(.{ .dtype = .f32, .tags = result_tags }).fromTensor(ctx, value);
        }

        pub fn typedConstantMax(self: anytype, ctx: *ExecContext, comptime tag: Tag) !Tensor(.{ .dtype = .f32, .tags = removeTag(TensorObject(@TypeOf(self)).axis_tags, tag) }) {
            return typedConstantExtremum(self, ctx, tag, .max);
        }

        pub fn typedConstantMin(self: anytype, ctx: *ExecContext, comptime tag: Tag) !Tensor(.{ .dtype = .f32, .tags = removeTag(TensorObject(@TypeOf(self)).axis_tags, tag) }) {
            return typedConstantExtremum(self, ctx, tag, .min);
        }

        pub fn typedConstantExtremum(self: anytype, ctx: *ExecContext, comptime tag: Tag, comptime op: enum { max, min }) !Tensor(.{ .dtype = .f32, .tags = removeTag(TensorObject(@TypeOf(self)).axis_tags, tag) }) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            comptime requireWidenedTypedFloat(Self.dtype, "max/min");
            const result_tags = removeTag(Self.axis_tags, tag);
            var wide = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
            defer wide.deinit();
            var raw = switch (op) {
                .max => try ctx.maxAxisRank(Self.tag_count, &wide, Self.axis(tag)),
                .min => try ctx.minAxisRank(Self.tag_count, &wide, Self.axis(tag)),
            };
            raw.indices.deinit();
            errdefer raw.values.deinit();
            return Tensor(.{ .dtype = .f32, .tags = result_tags }).fromTensor(ctx, raw.values);
        }

        pub fn typedConstantArgmax(self: anytype, ctx: *ExecContext, comptime tag: Tag) !Tensor(.{ .dtype = .i64, .tags = removeTag(TensorObject(@TypeOf(self)).axis_tags, tag) }) {
            const Self = TensorObject(@TypeOf(self));
            comptime requireWidenedTypedFloat(Self.dtype, "argmax");
            const result_tags = removeTag(Self.axis_tags, tag);
            var wide = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
            defer wide.deinit();
            var value = try ctx.argmaxAxisRank(Self.tag_count, &wide, Self.axis(tag));
            errdefer value.deinit();
            return Tensor(.{ .dtype = .i64, .tags = result_tags }).fromTensor(ctx, value);
        }

        pub fn typedConstantProd(self: anytype, ctx: *ExecContext, comptime tag: Tag) !Tensor(.{ .dtype = .f32, .tags = removeTag(TensorObject(@TypeOf(self)).axis_tags, tag) }) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            comptime requireWidenedTypedFloat(Self.dtype, "prod");
            const result_tags = removeTag(Self.axis_tags, tag);
            var wide = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
            defer wide.deinit();
            var value = try ctx.prodAxisRank(Self.tag_count, &wide, Self.axis(tag));
            errdefer value.deinit();
            return Tensor(.{ .dtype = .f32, .tags = result_tags }).fromTensor(ctx, value);
        }

        pub fn typedConstantVariance(self: anytype, ctx: *ExecContext, comptime tag: Tag, ddof: u1) !Tensor(.{ .dtype = .f32, .tags = removeTag(TensorObject(@TypeOf(self)).axis_tags, tag) }) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            comptime requireWidenedTypedFloat(Self.dtype, "variance");
            const result_tags = removeTag(Self.axis_tags, tag);
            var wide = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
            defer wide.deinit();
            var value = try ctx.varAxisRank(Self.tag_count, &wide, Self.axis(tag), ddof);
            errdefer value.deinit();
            return Tensor(.{ .dtype = .f32, .tags = result_tags }).fromTensor(ctx, value);
        }

        pub fn typedConstantCumsum(self: anytype, ctx: *ExecContext, comptime tag: Tag) !TensorObject(@TypeOf(self)) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            comptime requireWidenedTypedFloat(Self.dtype, "cumsum");
            var wide = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
            defer wide.deinit();
            var wide_value = try ctx.cumsumAxisRank(Self.tag_count, &wide, Self.axis(tag));
            defer wide_value.deinit();
            return typedFromWidened(Self.dtype, Self.axis_tags, ctx, &wide_value);
        }

        pub fn typedConstantCumprod(self: anytype, ctx: *ExecContext, comptime tag: Tag) !TensorObject(@TypeOf(self)) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            comptime requireWidenedTypedFloat(Self.dtype, "cumprod");
            var wide = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
            defer wide.deinit();
            var wide_value = try ctx.cumprodAxisRank(Self.tag_count, &wide, Self.axis(tag));
            defer wide_value.deinit();
            return typedFromWidened(Self.dtype, Self.axis_tags, ctx, &wide_value);
        }

        pub fn typedConstantWhere(self: anytype, ctx: *ExecContext, cond: anytype, other: anytype) !TensorObject(@TypeOf(self)) {
            try typedRequireNoGrad(self);
            try typedRequireNoGrad(other);
            const Self = TensorObject(@TypeOf(self));
            comptime requireWidenedTypedFloat(Self.dtype, "where");
            const Cond = TensorObject(@TypeOf(cond));
            const Other = TensorObject(@TypeOf(other));
            if (comptime Cond.dtype != .bool and Cond.dtype != Self.dtype) @compileError("typed where takes a .bool or same-dtype condition; cast explicitly");
            if (comptime Other.dtype != Self.dtype) @compileError("typed where requires matching dtypes; cast explicitly");
            const cond_ptr = tensorObjectPtrFrom(@TypeOf(cond), &cond);
            const other_ptr = tensorObjectPtrFrom(@TypeOf(other), &other);
            var wide = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
            defer wide.deinit();
            var wide_other = try ctx.castTyped(Self.dtype, .f32, other_ptr.asRawTensor());
            defer wide_other.deinit();
            var wide_value = try ctx.whereTyped(Cond.dtype, &wide, cond_ptr.asRawTensor(), &wide_other);
            defer wide_value.deinit();
            return typedFromWidened(Self.dtype, Self.axis_tags, ctx, &wide_value);
        }

        pub fn typedConstantMaskedFill(self: anytype, ctx: *ExecContext, mask: anytype, value: f32) !TensorObject(@TypeOf(self)) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            comptime requireWidenedTypedFloat(Self.dtype, "maskedFill");
            const Mask = TensorObject(@TypeOf(mask));
            if (comptime Mask.dtype != .bool and Mask.dtype != Self.dtype) @compileError("typed maskedFill takes a .bool or same-dtype mask; cast explicitly");
            const mask_ptr = tensorObjectPtrFrom(@TypeOf(mask), &mask);
            var wide = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
            defer wide.deinit();
            var wide_value = try ctx.maskedFillTyped(Mask.dtype, &wide, mask_ptr.asRawTensor(), value);
            defer wide_value.deinit();
            return typedFromWidened(Self.dtype, Self.axis_tags, ctx, &wide_value);
        }

        /// Comparison on the typed branches: `.bool` result everywhere (torch's
        /// comparison dtype). f16/bf16 compare through f32 (the widening seam);
        /// integers compare natively (exact at any magnitude).
        pub fn typedConstantCompare(self: anytype, ctx: *ExecContext, comptime op: exec_mod.CompareOp, other: anytype) !Tensor(.{ .dtype = .bool, .tags = TensorObject(@TypeOf(self)).axis_tags }) {
            const Self = TensorObject(@TypeOf(self));
            const BoolT = Tensor(.{ .dtype = .bool, .tags = Self.axis_tags });
            const OtherT = @TypeOf(other);
            if (comptime dtype_mod.supportsIntMath(Self.dtype)) {
                if (comptime (OtherT == comptime_int or @typeInfo(OtherT) == .int)) {
                    var value = try ctx.compareIntScalarTyped(Self.dtype, op, self.asRawTensor(), @intCast(other));
                    errdefer value.deinit();
                    return BoolT.fromTensor(ctx, value);
                }
                const Other = TensorObject(@TypeOf(other));
                if (comptime Other.dtype != Self.dtype) @compileError("typed compare requires matching dtypes; cast explicitly");
                const other_ptr = tensorObjectPtrFrom(@TypeOf(other), &other);
                var value = try ctx.compareIntTyped(Self.dtype, op, self.asRawTensor(), other_ptr.asRawTensor());
                errdefer value.deinit();
                return BoolT.fromTensor(ctx, value);
            }
            comptime requireWidenedTypedFloat(Self.dtype, "compare");
            var wide = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
            defer wide.deinit();
            if (comptime (OtherT == comptime_float or OtherT == comptime_int or @typeInfo(OtherT) == .float or @typeInfo(OtherT) == .int)) {
                var value = try ctx.compareScalar(op, &wide, other);
                errdefer value.deinit();
                return BoolT.fromTensor(ctx, value);
            }
            const Other = TensorObject(@TypeOf(other));
            if (comptime Other.dtype != Self.dtype) @compileError("typed compare requires matching dtypes; cast explicitly");
            const other_ptr = tensorObjectPtrFrom(@TypeOf(other), &other);
            var wide_other = try ctx.castTyped(Self.dtype, .f32, other_ptr.asRawTensor());
            defer wide_other.deinit();
            var value = try ctx.compare(op, &wide, &wide_other);
            errdefer value.deinit();
            return BoolT.fromTensor(ctx, value);
        }

        /// Widened einsum: both operands widen to f32 and the f32 GEMM lowering
        /// runs (f32 accumulation); the result narrows to the input dtype per the
        /// §8.3 matmul policy — the same contract as the typed `dot`.
        pub fn typedConstantEinsum(self: anytype, ctx: *ExecContext, other: anytype, comptime out_tags: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = normalizeTags(out_tags) }) {
            try typedRequireNoGrad(self);
            try typedRequireNoGrad(other);
            const Self = TensorObject(@TypeOf(self));
            const Other = TensorObject(@TypeOf(other));
            comptime requireWidenedTypedFloat(Self.dtype, "einsum");
            if (comptime Other.dtype != Self.dtype) @compileError("typed einsum requires matching dtypes; cast explicitly");
            const result_tags = comptime normalizeTags(out_tags);
            const other_ptr = tensorObjectPtrFrom(@TypeOf(other), &other);
            var wide_left = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
            defer wide_left.deinit();
            var wide_right = try ctx.castTyped(Self.dtype, .f32, other_ptr.asRawTensor());
            defer wide_right.deinit();
            var wide_value = try tag_ops.taggedEinsum(Self.axis_tags, &wide_left, ctx, Other.axis_tags, &wide_right, result_tags);
            defer wide_value.deinit();
            return typedFromWidened(Self.dtype, result_tags, ctx, &wide_value);
        }

        pub fn typedConstantPad(self: anytype, ctx: *ExecContext, comptime tag: Tag, before: usize, after: usize, fill: f32) !TensorObject(@TypeOf(self)) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            comptime requireWidenedTypedFloat(Self.dtype, "pad");
            var wide = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
            defer wide.deinit();
            var wide_value = try ctx.padAxisRank(Self.tag_count, &wide, Self.axis(tag), before, after, fill);
            defer wide_value.deinit();
            return typedFromWidened(Self.dtype, Self.axis_tags, ctx, &wide_value);
        }

        // Typed structural ops: pure views / data movement, valid for every typed
        // float dtype (f64 included — nothing rounds through f32).
        pub fn typedConstantSplit(
            self: anytype,
            ctx: *ExecContext,
            comptime tag: Tag,
            comptime split_tags_spec: anytype,
            split_shape: [normalizeTags(split_tags_spec).len]usize,
        ) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = splitTags(TensorObject(@TypeOf(self)).axis_tags, tag, normalizeTags(split_tags_spec)) }) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            const split_tags = normalizeTags(split_tags_spec);
            const result_tags = splitTags(Self.axis_tags, tag, split_tags);
            var value = try tag_ops.splitAxisViewOf(Self.dtype, Self.axis_tags, self.asRawTensor(), tag, split_tags, split_shape);
            errdefer value.deinit();
            return Tensor(.{ .dtype = Self.dtype, .tags = result_tags }).fromTensor(ctx, value);
        }

        pub fn typedConstantMerge(self: anytype, ctx: *ExecContext, comptime out_tag: Tag, comptime merge_tags_spec: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = mergeTags(TensorObject(@TypeOf(self)).axis_tags, out_tag, normalizeTags(merge_tags_spec)) }) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            const merge_tags = normalizeTags(merge_tags_spec);
            const result_tags = mergeTags(Self.axis_tags, out_tag, merge_tags);
            var value = try tag_ops.mergeAxesViewOf(Self.dtype, Self.axis_tags, self.asRawTensor(), out_tag, merge_tags);
            errdefer value.deinit();
            return Tensor(.{ .dtype = Self.dtype, .tags = result_tags }).fromTensor(ctx, value);
        }

        pub fn typedConstantFlatten(self: anytype, ctx: *ExecContext, comptime out_tag: Tag) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = .{out_tag} }) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            var value = try tag_ops.flattenTensorOf(Self.dtype, ctx, self.asRawTensor());
            errdefer value.deinit();
            return Tensor(.{ .dtype = Self.dtype, .tags = .{out_tag} }).fromTensor(ctx, value);
        }

        pub fn typedConstantReshape(
            self: anytype,
            ctx: *ExecContext,
            comptime new_tags_spec: anytype,
            new_shape: [normalizeTags(new_tags_spec).len]usize,
        ) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = normalizeTags(new_tags_spec) }) {
            const new_tags = comptime normalizeTags(new_tags_spec);
            if (comptime new_tags.len == 1) {
                if (self.asRawTensor().len() != new_shape[0]) return TensorError.InvalidShape;
                return typedConstantFlatten(self, ctx, new_tags[0]);
            }
            var flat = try typedConstantFlatten(self, ctx, new_tags[0]);
            defer flat.deinit();
            return typedConstantSplit(&flat, ctx, new_tags[0], new_tags_spec, new_shape);
        }

        pub fn typedConstantSliceStep(self: anytype, ctx: *ExecContext, comptime tag: Tag, start: usize, length: usize, step: usize) !TensorObject(@TypeOf(self)) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            const slice_axis = comptime Self.axis(tag);
            const raw = self.asRawTensor();
            const axis_dim = raw.shape.at(slice_axis);
            if (step == 0 or length == 0) return TensorError.InvalidShape;
            if (start >= axis_dim or start + (length - 1) * step >= axis_dim) return TensorError.InvalidShape;
            var new_shape: [Self.tensor_rank]usize = undefined;
            var new_strides: [Self.tensor_rank]usize = undefined;
            inline for (0..Self.tensor_rank) |i| {
                new_shape[i] = raw.shape.at(i);
                new_strides[i] = raw.strides.at(i);
            }
            new_shape[slice_axis] = length;
            new_strides[slice_axis] = raw.strides.at(slice_axis) * step;
            var value = try raw.viewWithStridesOffset(&new_shape, &new_strides, start * raw.strides.at(slice_axis));
            errdefer value.deinit();
            return Self.fromTensor(ctx, value);
        }

        pub fn typedConstantFlip(self: anytype, ctx: *ExecContext, comptime tag: Tag) !TensorObject(@TypeOf(self)) {
            const Self = TensorObject(@TypeOf(self));
            const n = self.asRawTensor().shape.at(Self.axis(tag));
            const indices = try ctx.allocator.alloc(usize, n);
            defer ctx.allocator.free(indices);
            for (indices, 0..) |*index, i| index.* = n - 1 - i;
            return typedConstantGather(self, ctx, tag, indices, tag);
        }

        pub fn typedConstantRoll(self: anytype, ctx: *ExecContext, comptime tag: Tag, shift: isize) !TensorObject(@TypeOf(self)) {
            const Self = TensorObject(@TypeOf(self));
            const n = self.asRawTensor().shape.at(Self.axis(tag));
            const indices = try ctx.allocator.alloc(usize, n);
            defer ctx.allocator.free(indices);
            // out[i] = x[(i - shift) mod n]; s = shift mod n in [0, n).
            const s: usize = @intCast(@mod(shift, @as(isize, @intCast(n))));
            for (indices, 0..) |*index, i| index.* = (i + n - s) % n;
            return typedConstantGather(self, ctx, tag, indices, tag);
        }

        pub fn typedConstantStack(
            self: anytype,
            ctx: *ExecContext,
            comptime new_tag: Tag,
            comptime axis_index: usize,
            others: []const *const TensorObject(@TypeOf(self)),
        ) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = insertTagAt(TensorObject(@TypeOf(self)).axis_tags, new_tag, axis_index) }) {
            const Self = TensorObject(@TypeOf(self));
            const Expanded = Tensor(.{ .dtype = Self.dtype, .tags = insertTagAt(Self.axis_tags, new_tag, axis_index) });
            var expanded = try ctx.allocator.alloc(Expanded, others.len + 1);
            defer ctx.allocator.free(expanded);
            var created: usize = 0;
            defer for (expanded[0..created]) |*view| view.deinit();

            expanded[0] = try typedConstantInsertAxis(self, ctx, new_tag, axis_index);
            created = 1;
            for (others) |other| {
                expanded[created] = try typedConstantInsertAxis(other, ctx, new_tag, axis_index);
                created += 1;
            }

            var ptrs_stack: [concat_inline_inputs]*const Expanded = undefined;
            const ptrs = if (others.len <= ptrs_stack.len)
                ptrs_stack[0..others.len]
            else
                try ctx.allocator.alloc(*const Expanded, others.len);
            defer if (others.len > ptrs_stack.len) ctx.allocator.free(ptrs);
            for (ptrs, expanded[1..]) |*ptr, *view| ptr.* = view;
            return typedConstantConcat(&expanded[0], ctx, new_tag, ptrs);
        }

        pub fn typedConstantRepeatAxis(self: anytype, ctx: *ExecContext, comptime tag: Tag, n: usize) !TensorObject(@TypeOf(self)) {
            const Self = TensorObject(@TypeOf(self));
            if (n == 0) return TensorError.InvalidShape;
            if (n == 1) return typedConstantWithTags(self, ctx, Self.axis_tags);
            const ptrs = try ctx.allocator.alloc(*const Self, n - 1);
            defer ctx.allocator.free(ptrs);
            const self_ptr = tensorObjectPtrFrom(@TypeOf(self), &self);
            for (ptrs) |*ptr| ptr.* = self_ptr;
            return typedConstantConcat(self_ptr, ctx, tag, ptrs);
        }

        pub fn typedPointwise(
            comptime tensor_dtype: DType,
            comptime op: PointwiseOp,
            self: anytype,
            ctx: *ExecContext,
            other: anytype,
        ) !Tensor(.{ .dtype = dtype_mod.outputDType(.pointwise, tensor_dtype), .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
            try typedRequireNoGrad(self);
            try typedRequireNoGrad(other);
            const SelfTensor = TensorObject(@TypeOf(self));
            const left_tags = SelfTensor.axis_tags;
            const Other = TensorObject(@TypeOf(other));
            if (Other.dtype != tensor_dtype) @compileError("typed pointwise requires matching dtypes; cast explicitly");
            if (comptime tensor_dtype == .bool) @compileError("bool tensors have no pointwise arithmetic; cast with to() first");
            const right_tags = Other.axis_tags;
            const left = tensorObjectPtrFrom(@TypeOf(self), &self);
            const right = tensorObjectPtrFrom(@TypeOf(other), &other);
            const result_tags = pointwiseResultTags(left_tags, right_tags);
            const result_shape = try pointwiseShapeOf(tensor_dtype, result_tags, left_tags, left.asRawTensor(), right_tags, right.asRawTensor());

            var left_view = try broadcastTensorToOf(tensor_dtype, left_tags, left.asRawTensor(), result_tags, result_shape);
            defer left_view.deinit();
            var right_view = try broadcastTensorToOf(tensor_dtype, right_tags, right.asRawTensor(), result_tags, result_shape);
            defer right_view.deinit();

            var value = switch (op) {
                .add => try ctx.addRankTyped(tensor_dtype, rawRank(result_tags.len), &left_view, &right_view),
                .sub => try ctx.subRankTyped(tensor_dtype, rawRank(result_tags.len), &left_view, &right_view),
                .mul => try ctx.mulRankTyped(tensor_dtype, rawRank(result_tags.len), &left_view, &right_view),
                .div => if (comptime dtype_mod.supportsIntMath(tensor_dtype))
                    @compileError("integer `div` is explicit: use divTrunc/divFloor (torch's `/` promotes to float; Fucina keeps promotion explicit)")
                else
                    try ctx.divRankTyped(tensor_dtype, rawRank(result_tags.len), &left_view, &right_view),
                .max => if (comptime dtype_mod.supportsIntMath(tensor_dtype))
                    try ctx.maxRankTyped(tensor_dtype, rawRank(result_tags.len), &left_view, &right_view)
                else
                    @compileError("float typed maximum/minimum widen through f32 (the f16/bf16 facade entries)"),
                .min => if (comptime dtype_mod.supportsIntMath(tensor_dtype))
                    try ctx.minRankTyped(tensor_dtype, rawRank(result_tags.len), &left_view, &right_view)
                else
                    @compileError("float typed maximum/minimum widen through f32 (the f16/bf16 facade entries)"),
            };
            errdefer value.deinit();
            return Tensor(.{ .dtype = dtype_mod.outputDType(.pointwise, tensor_dtype), .tags = result_tags }).fromTensor(ctx, value);
        }

        pub fn typedDot(
            comptime tensor_dtype: DType,
            self: anytype,
            ctx: *ExecContext,
            other: anytype,
            comptime contract_tag: Tag,
        ) !Tensor(.{ .dtype = dtype_mod.outputDType(.matmul, tensor_dtype), .tags = dotResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags, contract_tag) }) {
            try typedRequireNoGrad(self);
            try typedRequireNoGrad(other);
            const SelfTensor = TensorObject(@TypeOf(self));
            const left_tags = SelfTensor.axis_tags;
            const Other = TensorObject(@TypeOf(other));
            if (Other.dtype != tensor_dtype) @compileError("typed dot requires matching dtypes; cast explicitly");
            const right_tags = Other.axis_tags;
            const result_tags = dotResultTags(left_tags, right_tags, contract_tag);
            const left = tensorObjectPtrFrom(@TypeOf(self), &self);
            const right = tensorObjectPtrFrom(@TypeOf(other), &other);

            var value = try typedDotRaw(tensor_dtype, left_tags, left.asRawTensor(), ctx, right_tags, right.asRawTensor(), contract_tag);
            errdefer value.deinit();
            return Tensor(.{ .dtype = dtype_mod.outputDType(.matmul, tensor_dtype), .tags = result_tags }).fromTensor(ctx, value);
        }
    };
}
