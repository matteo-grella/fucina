//! Shared op plumbing for the ag tensor bands: result
//! finishing (grad wiring + scope adoption), pointwise/dot
//! dispatch helpers, einsum folding, and small validators.

const std = @import("std");
const tensor_mod = @import("../../tensor.zig");
const dtype_mod = @import("../../dtype.zig");
const exec_mod = @import("../../exec.zig");
const tag_ops = @import("../../tag_ops.zig");
const control = @import("../control.zig");
const core = @import("../core.zig");
const tags_mod = @import("../../tags.zig");
const backward_common = @import("../backward/common.zig");
const backward_elementwise = @import("../backward/elementwise.zig");

const RawTensor = tensor_mod.Tensor;
const DType = tensor_mod.DType;
const ExecContext = exec_mod.ExecContext;
const GatedOp = exec_mod.GatedOp;
const GradState = core.GradState;
const Tag = tags_mod.Tag;
const inserted_axis = tags_mod.inserted_axis;
const normalizeTags = tags_mod.normalizeTags;
const rawRank = tags_mod.rawRank;
const removeTags = tags_mod.removeTags;
const pointwiseResultTags = tags_mod.pointwiseResultTags;
const tagsEqual = tags_mod.tagsEqual;
const pointwiseShape = tag_ops.pointwiseShape;
const validateTensorRank = tag_ops.validateTensorRank;
const PointwiseOp = backward_common.PointwiseOp;
const PointwiseBackward = backward_elementwise.PointwiseBackward;
const GatedBackward = backward_elementwise.GatedBackward;

pub fn Mod(comptime ag_tensor: type) type {
    return struct {
        const Tensor = ag_tensor.Tensor;
        pub const rawShapeArray = backward_common.rawShapeArray;
        pub const rawShapeArrayOf = backward_common.rawShapeArrayOf;
        pub const cloneInverseRopeTable = backward_common.cloneInverseRopeTable;

        /// The result of a binary op between `Left` and `Right`: the tag
        /// broadcast rule over the left operand's dtype (the operands must
        /// share it; cast explicitly).
        fn BinaryOut(comptime Left: type, comptime Right: type) type {
            const left_dtype = TensorObject(Left).dtype;
            if (TensorObject(Right).dtype != left_dtype) @compileError("pointwise operands must share a dtype; cast explicitly with to()");
            if (left_dtype == .bool) @compileError("bool tensors have no pointwise arithmetic; cast with to() first");
            return Tensor(.{ .dtype = left_dtype, .tags = pointwiseResultTags(TensorObject(Left).axis_tags, TensorObject(Right).axis_tags) });
        }

        /// The dtype-generic tail of a binary op: f32 builds the VJP
        /// record; a typed result is a caller-owned constant and a
        /// grad-requiring operand is rejected.
        /// True when this op must record a backward node: an operand wants
        /// gradients and grad mode is on. Every op tail checks it BEFORE
        /// taking the views its record would save, so the no-grad path
        /// (inference) clones nothing.
        pub fn recordsGrad(wants_grad: bool) bool {
            return wants_grad and control.isGradEnabled();
        }

        /// The typed-branch tail of a dtype-generic op: a grad-requiring
        /// operand is rejected (the typed branches never record), otherwise
        /// the value is a no-grad constant. Consumes `value` on success; on
        /// error it stays with the caller.
        pub fn finishTypedNoGrad(comptime OutT: type, ctx: *ExecContext, value: tensor_mod.TensorOf(OutT.dtype), wants_grad: bool) !OutT {
            if (wants_grad) return error.UnsupportedGradient;
            return finishTyped(OutT, ctx, value);
        }

        /// The tail of every op result that carries no graph, of any dtype:
        /// wrap as a constant and, under an exec scope, adopt it like every
        /// result. Consumes `value` on success; on error it stays with the
        /// caller.
        pub fn finishTyped(comptime OutT: type, ctx: *ExecContext, value: tensor_mod.TensorOf(OutT.dtype)) !OutT {
            var out = try OutT.fromTensor(ctx, value);
            if (ctx.execScopeActive()) try adoptResult(ctx, &out);
            return out;
        }

        /// Hand a result to the innermost scope: the scope takes over the
        /// handle's references (the value buffer and, when present, the
        /// graph node) and the handle becomes a borrow whose `deinit` is a
        /// no-op. All or nothing: on failure the handle still owns
        /// everything (`ExecContext.adopt`).
        pub fn adoptResult(ctx: *ExecContext, t: anytype) !void {
            const T = @TypeOf(t.*);
            var entries: [2]ExecContext.ScopeEntry = undefined;
            var count: usize = 1;
            entries[0] = ExecContext.bufferEntry(T.dtype, t.value.buffer);
            if (comptime hasGradSlot(T)) {
                if (t.grad_state) |state| {
                    entries[1] = core.scopeEntry(state);
                    count = 2;
                }
            }
            try ctx.adopt(entries[0..count]);
            t.scope_owned = true;
        }

        /// The grad-carrying tail shared by `finishOp` and `typedFinishOp`:
        /// allocate the node, adopt its address and the value into an open
        /// scope (the one fallible step after the record was built), then
        /// move the record in. On error nothing was consumed: `value` and the
        /// record's resources stay with the caller.
        pub fn finishWithRecord(comptime OutT: type, ctx: *ExecContext, value: tensor_mod.TensorOf(OutT.dtype), record: anytype) !OutT {
            const node = try core.allocNode(ctx.allocator(), @TypeOf(record));
            errdefer ctx.allocator().destroy(node);
            var out = OutT{ .value = value, .grad_state = &node.state };
            if (ctx.execScopeActive()) try adoptResult(ctx, &out);
            _ = core.initNode(node, ctx.allocator(), record);
            return out;
        }

        /// True when `T` carries a live gradient slot: a non-void
        /// `grad_state` field (f32/f16/bf16; on f64 the field is `void`).
        pub fn hasGradSlot(comptime T: type) bool {
            return @hasField(T, "grad_state") and @FieldType(T, "grad_state") != void;
        }

        /// The operand's gradient state, null on the branches without one.
        fn gradStateOf(t: anytype) ?*GradState {
            if (comptime hasGradSlot(@TypeOf(t.*))) return t.grad_state;
            return null;
        }

        pub fn pointwise(comptime op: PointwiseOp, self: anytype, ctx: *ExecContext, other: anytype) !BinaryOut(@TypeOf(self), @TypeOf(other)) {
            const OutT = BinaryOut(@TypeOf(self), @TypeOf(other));
            const dtype = OutT.dtype;
            const SelfTensor = TensorObject(@TypeOf(self));
            const left_tags = SelfTensor.axis_tags;
            const Other = TensorObject(@TypeOf(other));
            const right_tags = Other.axis_tags;
            const left = tensorObjectPtrFrom(@TypeOf(self), &self);
            const right = tensorObjectPtrFrom(@TypeOf(other), &other);
            const result_tags = pointwiseResultTags(left_tags, right_tags);
            const left_tensor = left.asRawTensor();
            const right_tensor = right.asRawTensor();
            _ = try pointwiseShape(dtype, result_tags, left_tags, left_tensor, right_tags, right_tensor);
            const wants_grad = left.requiresGrad() or right.requiresGrad();

            if (comptime tagsEqual(left_tags, right_tags)) {
                if (std.mem.eql(usize, left_tensor.shape.slice(), right_tensor.shape.slice())) {
                    var value = try ctx.elementwise(dtype, comptime tag_ops.elementwiseOp(op), left_tensor, right_tensor);
                    errdefer value.deinit();
                    if (comptime OutT.dtype != .f32) return finishTypedNoGrad(OutT, ctx, value, wants_grad);
                    if (!recordsGrad(wants_grad)) return finishNoGrad(OutT.axis_tags, ctx, value);
                    return finishPointwise(op, OutT, left_tags, right_tags, result_tags, ctx, value, gradStateOf(left), gradStateOf(right), left_tensor, right_tensor);
                }
            }

            var value = try tag_ops.pointwise(dtype, op, left_tags, left_tensor, ctx, right_tags, right_tensor);
            errdefer value.deinit();
            if (comptime OutT.dtype != .f32) return finishTypedNoGrad(OutT, ctx, value, wants_grad);
            if (!recordsGrad(wants_grad)) return finishNoGrad(OutT.axis_tags, ctx, value);
            return finishPointwise(op, OutT, left_tags, right_tags, result_tags, ctx, value, gradStateOf(left), gradStateOf(right), left_tensor, right_tensor);
        }

        pub fn gatedPointwise(comptime op: GatedOp, self: anytype, ctx: *ExecContext, other: anytype) !BinaryOut(@TypeOf(self), @TypeOf(other)) {
            const OutT = BinaryOut(@TypeOf(self), @TypeOf(other));
            const dtype = OutT.dtype;
            const SelfTensor = TensorObject(@TypeOf(self));
            const left_tags = SelfTensor.axis_tags;
            const Other = TensorObject(@TypeOf(other));
            const right_tags = Other.axis_tags;
            const left = tensorObjectPtrFrom(@TypeOf(self), &self);
            const right = tensorObjectPtrFrom(@TypeOf(other), &other);
            const result_tags = pointwiseResultTags(left_tags, right_tags);
            const left_tensor = left.asRawTensor();
            const right_tensor = right.asRawTensor();
            _ = try pointwiseShape(dtype, result_tags, left_tags, left_tensor, right_tags, right_tensor);
            const wants_grad = left.requiresGrad() or right.requiresGrad();

            if (comptime tagsEqual(left_tags, right_tags)) {
                if (std.mem.eql(usize, left_tensor.shape.slice(), right_tensor.shape.slice())) {
                    var value = try ctx.gated(dtype, rawRank(result_tags.len), op, left_tensor, right_tensor);
                    errdefer value.deinit();
                    if (comptime OutT.dtype != .f32) return finishTypedNoGrad(OutT, ctx, value, wants_grad);
                    if (!recordsGrad(wants_grad)) return finishNoGrad(OutT.axis_tags, ctx, value);
                    return finishGated(op, OutT, left_tags, right_tags, result_tags, ctx, value, gradStateOf(left), gradStateOf(right), left_tensor, right_tensor);
                }
            }

            var value = try tag_ops.gatedPointwise(dtype, op, left_tags, left_tensor, ctx, right_tags, right_tensor);
            errdefer value.deinit();
            if (comptime OutT.dtype != .f32) return finishTypedNoGrad(OutT, ctx, value, wants_grad);
            if (!recordsGrad(wants_grad)) return finishNoGrad(OutT.axis_tags, ctx, value);
            return finishGated(op, OutT, left_tags, right_tags, result_tags, ctx, value, gradStateOf(left), gradStateOf(right), left_tensor, right_tensor);
        }

        /// The record tail of `pointwise`, spelled once for the same-shape
        /// and the broadcast path: the operand views the op's VJP reads
        /// (mul/div/max/min) and the finished op. Consumes `value` on
        /// success; on error it stays with the caller.
        fn finishPointwise(comptime op: PointwiseOp, comptime OutT: type, comptime left_tags: anytype, comptime right_tags: anytype, comptime result_tags: anytype, ctx: *ExecContext, value: RawTensor, left_parent: ?*GradState, right_parent: ?*GradState, left_tensor: *const RawTensor, right_tensor: *const RawTensor) !OutT {
            const Record = PointwiseBackward(op, left_tags, right_tags, result_tags);
            const saves = comptime (op == .mul or op == .div or op == .max or op == .min);
            var saved_left: ?RawTensor = if (saves) try left_tensor.cloneView() else null;
            errdefer if (saved_left) |*v| v.deinit();
            var saved_right: ?RawTensor = if (saves) try right_tensor.cloneView() else null;
            errdefer if (saved_right) |*v| v.deinit();
            return finishOp(OutT.axis_tags, ctx, value, Record{
                .parents = .{ left_parent, right_parent },
                .left_shape = rawShapeArray(left_tags, left_tensor),
                .right_shape = rawShapeArray(right_tags, right_tensor),
                .left_value = saved_left,
                .right_value = saved_right,
            });
        }

        /// The record tail of `gatedPointwise`, spelled once for both paths.
        fn finishGated(comptime op: GatedOp, comptime OutT: type, comptime left_tags: anytype, comptime right_tags: anytype, comptime result_tags: anytype, ctx: *ExecContext, value: RawTensor, left_parent: ?*GradState, right_parent: ?*GradState, left_tensor: *const RawTensor, right_tensor: *const RawTensor) !OutT {
            const Record = GatedBackward(op, left_tags, right_tags, result_tags);
            var saved_left = try left_tensor.cloneView();
            errdefer saved_left.deinit();
            var saved_right = try right_tensor.cloneView();
            errdefer saved_right.deinit();
            var v = value;
            return finishOp(OutT.axis_tags, ctx, value, Record{
                .parents = .{ left_parent, right_parent },
                .left_shape = rawShapeArray(left_tags, left_tensor),
                .right_shape = rawShapeArray(right_tags, right_tensor),
                .result_shape = rawShapeArray(result_tags, (&v)),
                .left_value = saved_left,
                .right_value = saved_right,
            });
        }

        /// Per-position {max, sum_exp} buffer for the stats-saving forwards
        /// (cross-entropy, grouped attention): allocated exactly when `finishOp`
        /// will create the backward node (wants_grad AND grad mode on — keep the
        /// conditions in sync), so the node always receives real statistics. The
        /// caller frees it; the node dupes.
        pub fn rowStatsAlloc(ctx: *ExecContext, wants_grad: bool, position_count: usize) !?[]f32 {
            if (!recordsGrad(wants_grad)) return null;
            return try ctx.allocator().alloc(f32, 2 * position_count);
        }

        /// Shared tail of every differentiable op once `recordsGrad` said yes:
        /// attach the backward node built from `record` (a typed struct literal
        /// the op filled, saved views included) through `core.createNode`, one
        /// allocation holding the GradState header and the record. Every
        /// fallible step precedes the node allocation, and the allocation is
        /// the last one, so on error the record's resources and `value` stay
        /// with the caller (its errdefers), never double-released.
        ///
        /// While an exec scope is open on `ctx` (ExecContext.openExecScope), the
        /// result is adopted by the scope and the caller receives a borrow
        /// (`finishWithRecord`).
        pub fn finishOp(
            comptime result_tags: anytype,
            ctx: *ExecContext,
            value: RawTensor,
            record: anytype,
        ) !Tensor(result_tags) {
            var owned_value = value;
            try validateTensorRank(.f32, normalizeTags(result_tags), &owned_value);
            return finishWithRecord(Tensor(result_tags), ctx, owned_value, record);
        }

        /// N-ary einsum: contracts two or more f32 tensors (values or pointers, in a
        /// tuple) down to `out_tags` by a comptime left-fold of the binary `einsum`.
        /// Each intermediate keeps exactly the tags still needed by the remaining
        /// operands or the output, in group-nested order, so classic chains (e.g. a
        /// LoRA delta `x[s,i]·A[r,i]·B[o,r] -> [s,o]`) stay on the direct GEMM paths.
        /// Contraction order is the operand order — order the tuple so early
        /// intermediates stay small. Gradients flow through every operand; the
        /// intermediates are released on return and the graph keeps them alive
        /// (each consumer record holds a reference to its operands' states).
        pub fn einsumMany(ctx: *ExecContext, comptime out_tags: anytype, operands: anytype) !Tensor(normalizeTags(out_tags)) {
            const OperandsT = @TypeOf(operands);
            const operand_count = comptime @typeInfo(OperandsT).@"struct".fields.len;
            comptime {
                if (operand_count < 2) @compileError("einsumMany requires at least two operands");
            }
            const first = tensorObjectPtrFrom(@TypeOf(operands[0]), &operands[0]);
            return einsumManyFold(ctx, out_tags, first, operands, 1);
        }

        pub fn einsumManyFold(ctx: *ExecContext, comptime out_tags: anytype, acc: anytype, operands: anytype, comptime i: usize) !Tensor(normalizeTags(out_tags)) {
            const operand_count = comptime @typeInfo(@TypeOf(operands)).@"struct".fields.len;
            if (comptime i == operand_count - 1) {
                return acc.einsum(ctx, operands[i], comptime normalizeTags(out_tags));
            } else {
                const acc_tags = comptime TensorObject(@TypeOf(acc)).axis_tags;
                const op_tags = comptime TensorObject(@TypeOf(operands[i])).axis_tags;
                const needed = comptime einsumManyNeededTags(@TypeOf(operands), out_tags, i + 1);
                const keep = comptime einsumManyKeepTags(acc_tags, op_tags, needed);
                var next = try acc.einsum(ctx, operands[i], keep);
                defer next.deinit();
                return einsumManyFold(ctx, out_tags, &next, operands, i + 1);
            }
        }

        /// Tags an intermediate must keep: every tag of the pair still needed by the
        /// remaining operands or the output, in group-nested (batch, then acc-free,
        /// then operand-free) order so the next contraction stays direct-lowerable.
        pub inline fn einsumManyKeepTags(comptime acc_tags: anytype, comptime op_tags: anytype, comptime needed: anytype) []const Tag {
            comptime {
                const shared = tags_mod.intersectTags(acc_tags, op_tags);
                const op_shared = tags_mod.intersectTags(op_tags, acc_tags);
                return tags_mod.intersectTags(shared, needed) ++
                    tags_mod.intersectTags(removeTags(acc_tags, shared), needed) ++
                    tags_mod.intersectTags(removeTags(op_tags, op_shared), needed);
            }
        }

        /// Union of the output tags and every remaining operand's tags, used purely
        /// as a membership set (may exceed the tensor rank limit, unlike
        /// `pointwiseResultTags`).
        pub inline fn einsumManyNeededTags(comptime OperandsT: type, comptime out_tags: anytype, comptime from: usize) []const Tag {
            comptime {
                const fields = @typeInfo(OperandsT).@"struct".fields;
                if (from == fields.len) {
                    const final = normalizeTags(out_tags);
                    return &final;
                }
                const rest = einsumManyNeededTags(OperandsT, out_tags, from + 1);
                return tags_mod.unionTags(rest, TensorObject(fields[from].type).axis_tags);
            }
        }

        /// Typed forward ops are no-grad: a grad-requiring operand would silently
        /// drop its graph, so it is rejected instead (`to(.f32)` is the trained
        /// path; the differentiable typed entries are `to` and the mixed-RHS
        /// `dot`/`einsum`).
        pub fn typedRequireNoGrad(operand: anytype) !void {
            if (operand.requiresGrad()) return error.UnsupportedGradient;
        }

        /// `finishOp` for a differentiable op whose RESULT is 16-bit (today: the
        /// f32 -> f16/bf16 cast), once `recordsGrad` said yes. Same contract as
        /// `finishOp`: consumes `value` on success; under an exec scope the
        /// result is a borrow like any other (the 16-bit buffer and the node
        /// are two ordinary scope entries).
        pub fn typedFinishOp(
            comptime tensor_dtype: DType,
            comptime result_tags: anytype,
            ctx: *ExecContext,
            value: tensor_mod.TensorOf(tensor_dtype),
            record: anytype,
        ) !Tensor(.{ .dtype = tensor_dtype, .tags = result_tags }) {
            return finishWithRecord(Tensor(.{ .dtype = tensor_dtype, .tags = result_tags }), ctx, value, record);
        }

        /// Shared no-grad tail: wrap as a constant and, when an exec scope is
        /// open, adopt it. Same value-ownership contract as `fromTensor`.
        pub fn finishNoGrad(comptime result_tags: anytype, ctx: *ExecContext, value: RawTensor) !Tensor(result_tags) {
            return finishTyped(Tensor(result_tags), ctx, value);
        }

        pub fn axisViewTensor(source: *const RawTensor, comptime axes: anytype, comptime target_tags: anytype) !RawTensor {
            return axisViewTensorOf(.f32, source, axes, target_tags);
        }

        pub fn axisViewTensorOf(
            comptime tensor_dtype: DType,
            source: *const tensor_mod.TensorOf(tensor_dtype),
            comptime axes: anytype,
            comptime target_tags: anytype,
        ) !tensor_mod.TensorOf(tensor_dtype) {
            if (comptime target_tags.len == 0) {
                return source.viewWithStrides(&.{1}, &.{1});
            }

            var shape: [target_tags.len]usize = undefined;
            var strides: [target_tags.len]usize = undefined;
            inline for (axes, 0..) |axis, i| {
                if (axis == inserted_axis) {
                    shape[i] = 1;
                    strides[i] = 0;
                } else {
                    shape[i] = source.shape.at(axis);
                    strides[i] = source.strides.at(axis);
                }
            }

            return source.viewWithStrides(shape[0..], strides[0..]);
        }

        pub fn TensorObject(comptime T: type) type {
            return switch (@typeInfo(T)) {
                .pointer => |ptr| ptr.child,
                else => T,
            };
        }

        /// Comptime whitelist for an `anytype` options struct: a misspelled
        /// field is a compile error naming the op and the field. These option
        /// sets stay `anytype` (instead of a typed struct) because they carry
        /// caller-typed tensor operands (a mask, affine weights, an initial
        /// state, each keeping its own tensor type) or comptime tag fields,
        /// which one runtime struct type cannot hold.
        pub fn validateOptionFields(
            comptime op_name: []const u8,
            comptime Opts: type,
            comptime allowed: []const []const u8,
            comptime example: []const u8,
        ) void {
            const info = @typeInfo(Opts);
            if (info != .@"struct") @compileError(op_name ++ ": options must be a struct literal, e.g. " ++ example);
            inline for (info.@"struct".fields) |field| {
                var known = false;
                for (allowed) |name| {
                    if (std.mem.eql(u8, field.name, name)) known = true;
                }
                if (!known) @compileError(op_name ++ ": unknown option ." ++ field.name);
            }
        }

        /// Comptime whitelist for the masked-reduction option struct. A misspelled
        /// field is a compile error, never a silently-unmasked reduction — the
        /// `groupedAttention` opts discipline.
        pub fn validateMaskedReduceOptions(comptime Opts: type) void {
            const info = @typeInfo(Opts);
            if (info != .@"struct") @compileError("masked-reduction options must be a struct literal, e.g. .{ .mask = &m }");
            inline for (info.@"struct".fields) |field| {
                if (!std.mem.eql(u8, field.name, "mask") and !std.mem.eql(u8, field.name, "empty")) {
                    @compileError("unknown masked-reduction option '" ++ field.name ++ "'; expected .mask or .empty");
                }
            }
        }

        /// The mask contract shared with `where`/`maskedFill`: `.bool` or a float read
        /// by truthiness. Integer masks stay a compile error so a token-id tensor is
        /// never mistaken for a mask.
        pub fn validateMaskType(comptime Mask: type, comptime op_name: []const u8) void {
            if (Mask.dtype != .bool and !dtype_mod.supportsForwardFloatMath(Mask.dtype)) {
                @compileError(op_name ++ " takes a .bool or float mask; cast integer masks explicitly");
            }
        }

        /// `opts.empty` when present, else null (each op falls back to its own
        /// identity: 0 for a sum, ±inf for an extremum, NaN for a mean).
        pub fn maskedReduceEmpty(opts: anytype) ?f32 {
            if (comptime @hasField(@TypeOf(opts), "empty")) return @as(f32, opts.empty);
            return null;
        }

        pub fn tensorObjectPtrFrom(comptime T: type, value: *const T) *const TensorObject(T) {
            return switch (@typeInfo(T)) {
                .pointer => value.*,
                else => value,
            };
        }

        /// Normalize `constantPad2d`'s padding spec: an integer pads all four
        /// sides; a 4-tuple/array is `(left, right, top, bottom)`.
        pub fn padding2dValues(padding: anytype) [4]isize {
            const P = @TypeOf(padding);
            const info = @typeInfo(P);
            if (comptime (P == comptime_int or info == .int)) {
                const p: isize = @intCast(padding);
                return .{ p, p, p, p };
            }
            if (comptime ((info == .@"struct" and info.@"struct".is_tuple and info.@"struct".fields.len == 4) or
                (info == .array and info.array.len == 4)))
            {
                return .{ @intCast(padding[0]), @intCast(padding[1]), @intCast(padding[2]), @intCast(padding[3]) };
            }
            @compileError("constantPad2d: padding must be an integer or a 4-tuple/array (left, right, top, bottom), got " ++ @typeName(P));
        }
    };
}
