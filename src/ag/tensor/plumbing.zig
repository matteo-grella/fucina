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
const TensorError = tensor_mod.TensorError;
const ExecContext = exec_mod.ExecContext;
const GatedOp = exec_mod.GatedOp;
const GradState = core.GradState;
const Tag = tags_mod.Tag;
const inserted_axis = tags_mod.inserted_axis;
const normalizeTags = tags_mod.normalizeTags;
const rawRank = tags_mod.rawRank;
const removeTags = tags_mod.removeTags;
const pointwiseResultTags = tags_mod.pointwiseResultTags;
const dotResultTags = tags_mod.dotResultTags;
const tagsEqual = tags_mod.tagsEqual;
const dotLeftOrder = tags_mod.dotLeftOrder;
const dotRightOrder = tags_mod.dotRightOrder;
const dotRightTransBOrder = tags_mod.dotRightTransBOrder;
const dotBatchLen = tags_mod.dotBatchLen;
const dotLeftFreeLen = tags_mod.dotLeftFreeLen;
const dotRightFreeLen = tags_mod.dotRightFreeLen;
const alignTensorTo = tag_ops.alignTensorTo;
const contiguousForReshape = tag_ops.contiguousForReshape;
const dotResultShape = tag_ops.dotResultShape;
const pointwiseShape = tag_ops.pointwiseShape;
const productRange = tag_ops.productRange;
const validateTensorRank = tag_ops.validateTensorRank;
const PointwiseOp = backward_common.PointwiseOp;
const PointwiseBackward = backward_elementwise.PointwiseBackward;
const GatedBackward = backward_elementwise.GatedBackward;

pub fn Mod(comptime ag_tensor: type) type {
    return struct {
        const Tensor = ag_tensor.Tensor;

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
        /// The operand's gradient state, null on the branches without one.
        fn gradStateOf(t: anytype) ?*GradState {
            if (comptime @hasField(@TypeOf(t.*), "grad_state")) return t.grad_state;
            return null;
        }

        fn finishBinary(comptime OutT: type, ctx: *ExecContext, value: anytype, wants_grad: bool, comptime Backward: type, create_args: anytype) !OutT {
            if (comptime OutT.dtype == .f32) return finishOp(OutT.axis_tags, ctx, value, wants_grad, Backward, create_args);
            if (wants_grad) return error.UnsupportedGradient;
            return OutT.fromTensor(ctx, value);
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
                    var value = switch (op) {
                        .add => try ctx.add(dtype, rawRank(result_tags.len), left_tensor, right_tensor),
                        .sub => try ctx.sub(dtype, rawRank(result_tags.len), left_tensor, right_tensor),
                        .mul => try ctx.mul(dtype, rawRank(result_tags.len), left_tensor, right_tensor),
                        .div => try ctx.div(dtype, rawRank(result_tags.len), left_tensor, right_tensor),
                        .max => try ctx.max(dtype, rawRank(result_tags.len), left_tensor, right_tensor),
                        .min => try ctx.min(dtype, rawRank(result_tags.len), left_tensor, right_tensor),
                    };
                    errdefer value.deinit();
                    return finishBinary(OutT, ctx, value, wants_grad, PointwiseBackward(op, left_tags, right_tags, result_tags), .{ ctx.allocator, gradStateOf(left), gradStateOf(right), left_tensor, right_tensor });
                }
            }

            var value = try tag_ops.pointwise(dtype, op, left_tags, left_tensor, ctx, right_tags, right_tensor);
            errdefer value.deinit();
            return finishBinary(OutT, ctx, value, wants_grad, PointwiseBackward(op, left_tags, right_tags, result_tags), .{ ctx.allocator, gradStateOf(left), gradStateOf(right), left_tensor, right_tensor });
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
                    return finishBinary(OutT, ctx, value, wants_grad, GatedBackward(op, left_tags, right_tags, result_tags), .{ ctx.allocator, gradStateOf(left), gradStateOf(right), left_tensor, right_tensor, &value });
                }
            }

            var value = try tag_ops.gatedPointwise(dtype, op, left_tags, left_tensor, ctx, right_tags, right_tensor);
            errdefer value.deinit();
            return finishBinary(OutT, ctx, value, wants_grad, GatedBackward(op, left_tags, right_tags, result_tags), .{ ctx.allocator, gradStateOf(left), gradStateOf(right), left_tensor, right_tensor, &value });
        }

        /// Per-position {max, sum_exp} buffer for the stats-saving forwards
        /// (cross-entropy, grouped attention): allocated exactly when `finishOp`
        /// will create the backward node (wants_grad AND grad mode on — keep the
        /// conditions in sync), so the node always receives real statistics. The
        /// caller frees it; the node dupes.
        pub fn rowStatsAlloc(ctx: *ExecContext, wants_grad: bool, position_count: usize) !?[]f32 {
            if (!wants_grad or !control.isGradEnabled()) return null;
            return try ctx.allocator.alloc(f32, 2 * position_count);
        }

        /// Shared tail of every differentiable op: wrap `value` as a no-grad tensor
        /// when no operand needs gradients, otherwise attach the backward node built
        /// by `core.createNode(BackwardType, create_args)` — one allocation holding
        /// the GradState header and the typed record. On error, ownership of `value`
        /// stays with the caller (same contract as `fromTensor` and
        /// `finishWithBackward`).
        ///
        /// While an exec scope is open on `ctx` (ExecContext.openExecScope), the result
        /// is adopted by the scope and the caller receives a borrow. The scope slot
        /// is reserved BEFORE construction so adoption itself cannot fail after the
        /// value has been consumed.
        pub fn finishOp(
            comptime result_tags: anytype,
            ctx: *ExecContext,
            value: RawTensor,
            wants_grad: bool,
            comptime BackwardType: type,
            create_args: anytype,
        ) !Tensor(result_tags) {
            if (!wants_grad or !control.isGradEnabled()) return finishNoGrad(result_tags, ctx, value);
            if (ctx.execScopeActive()) try ctx.reserveScopeSlot();
            const state = try core.createNode(BackwardType, create_args);
            var out = try finishWithBackward(result_tags, value, state);
            if (ctx.execScopeActive()) {
                adoptIntoScope(ctx, &out);
                out.scope_owned = true;
            }
            return out;
        }

        /// Guard for facade-level COMPOSED differentiable ops (nllLoss, l2Normalize,
        /// cosineSimilarity): their intermediate graph nodes are function-local, so
        /// when gradients are tracked only an active exec scope can own them until
        /// backward (GradState is single-owner — unscoped deinit of a grad-carrying
        /// intermediate would dangle the downstream operand pointers). Loud error
        /// instead of undefined behavior; no-grad composition works unscoped.
        pub fn requireScopeForComposedGrad(ctx: *ExecContext, wants_grad: bool) !void {
            if (wants_grad and control.isGradEnabled() and !ctx.execScopeActive()) {
                return error.ActiveExecScopeRequired;
            }
        }

        /// N-ary einsum: contracts two or more f32 tensors (values or pointers, in a
        /// tuple) down to `out_tags` by a comptime left-fold of the binary `einsum`.
        /// Each intermediate keeps exactly the tags still needed by the remaining
        /// operands or the output, in group-nested order, so classic chains (e.g. a
        /// LoRA delta `x[s,i]·A[r,i]·B[o,r] -> [s,o]`) stay on the direct GEMM paths.
        /// Contraction order is the operand order — order the tuple so early
        /// intermediates stay small. Gradients flow through every operand; as with
        /// other composed facade ops, tracking gradients requires an active exec
        /// scope to own the intermediates (`error.ActiveExecScopeRequired`).
        pub fn einsumMany(ctx: *ExecContext, comptime out_tags: anytype, operands: anytype) !Tensor(normalizeTags(out_tags)) {
            const OperandsT = @TypeOf(operands);
            const operand_count = comptime @typeInfo(OperandsT).@"struct".fields.len;
            comptime {
                if (operand_count < 2) @compileError("einsumMany requires at least two operands");
            }
            var wants_grad = false;
            inline for (0..operand_count) |i| {
                const ptr = tensorObjectPtrFrom(@TypeOf(operands[i]), &operands[i]);
                if (ptr.requiresGrad()) wants_grad = true;
            }
            try requireScopeForComposedGrad(ctx, wants_grad);
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
        pub fn einsumManyKeepTags(comptime acc_tags: anytype, comptime op_tags: anytype, comptime needed: anytype) [einsumManyKeepLen(acc_tags, op_tags, needed)]Tag {
            const shared = tags_mod.intersectTags(acc_tags, op_tags);
            const op_shared = tags_mod.intersectTags(op_tags, acc_tags);
            return tags_mod.intersectTags(shared, needed) ++
                tags_mod.intersectTags(removeTags(acc_tags, shared), needed) ++
                tags_mod.intersectTags(removeTags(op_tags, op_shared), needed);
        }

        pub fn einsumManyKeepLen(comptime acc_tags: anytype, comptime op_tags: anytype, comptime needed: anytype) usize {
            const shared = tags_mod.intersectTags(acc_tags, op_tags);
            const op_shared = tags_mod.intersectTags(op_tags, acc_tags);
            return tags_mod.intersectTagsLen(shared, needed) +
                tags_mod.intersectTagsLen(removeTags(acc_tags, shared), needed) +
                tags_mod.intersectTagsLen(removeTags(op_tags, op_shared), needed);
        }

        /// Union of the output tags and every remaining operand's tags, used purely
        /// as a membership set (may exceed the tensor rank limit, unlike
        /// `pointwiseResultTags`).
        pub fn einsumManyNeededTags(comptime OperandsT: type, comptime out_tags: anytype, comptime from: usize) [einsumManyNeededLen(OperandsT, out_tags, from)]Tag {
            const fields = @typeInfo(OperandsT).@"struct".fields;
            if (comptime from == fields.len) {
                return normalizeTags(out_tags);
            } else {
                const rest = einsumManyNeededTags(OperandsT, out_tags, from + 1);
                return tags_mod.unionTags(rest, TensorObject(fields[from].type).axis_tags);
            }
        }

        pub fn einsumManyNeededLen(comptime OperandsT: type, comptime out_tags: anytype, comptime from: usize) usize {
            const fields = @typeInfo(OperandsT).@"struct".fields;
            if (comptime from == fields.len) {
                return normalizeTags(out_tags).len;
            } else {
                const rest = einsumManyNeededTags(OperandsT, out_tags, from + 1);
                return tags_mod.unionTagsLen(rest, TensorObject(fields[from].type).axis_tags);
            }
        }

        /// Typed forward ops are no-grad: a grad-requiring operand would silently
        /// drop its graph, so it is rejected instead (`to(.f32)` is the trained
        /// path; the differentiable typed entries are `to` and the mixed-RHS
        /// `dot`/`einsum`).
        pub fn typedRequireNoGrad(operand: anytype) !void {
            if (operand.requiresGrad()) return error.UnsupportedGradient;
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
        /// f32 -> f16/bf16 cast). Same contract as `finishOp`: consumes `value` on
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

        /// Shared no-grad tail: wrap as a constant and, when an exec scope is open,
        /// hand ownership to the scope. Same value-ownership contract as `fromTensor`.
        pub fn finishNoGrad(comptime result_tags: anytype, ctx: *ExecContext, value: RawTensor) !Tensor(result_tags) {
            if (ctx.execScopeActive()) try ctx.reserveScopeSlot();
            var out = try Tensor(result_tags).fromTensor(ctx, value);
            if (ctx.execScopeActive()) {
                adoptIntoScope(ctx, &out);
                out.scope_owned = true;
            }
            return out;
        }

        pub fn adoptIntoScope(ctx: *ExecContext, t: anytype) void {
            ctx.adoptScopeValueAssumeCapacity(
                t.value,
                if (t.grad_state) |state| @ptrCast(state) else null,
                destroyGradStateOpaque,
            );
        }

        pub fn destroyGradStateOpaque(ptr: *anyopaque) void {
            const state: *GradState = @ptrCast(@alignCast(ptr));
            state.deinit();
        }

        /// Consumes `value` and `state` on success. On error, ownership of `value`
        /// stays with the caller (every call site holds an `errdefer value.deinit()`),
        /// while `state` — a co-allocated node the caller cannot reach — is destroyed
        /// here.
        pub fn finishWithBackward(comptime tags: anytype, value: RawTensor, state: *GradState) !Tensor(tags) {
            errdefer state.deinit();
            var owned_value = value;
            try validateTensorRank(.f32, normalizeTags(tags), &owned_value);
            return .{ .value = owned_value, .grad_state = state };
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

        pub fn typedDotRaw(
            comptime tensor_dtype: DType,
            comptime left_tags: anytype,
            left: *const tensor_mod.TensorOf(tensor_dtype),
            ctx: *ExecContext,
            comptime right_tags: anytype,
            right: *const tensor_mod.TensorOf(tensor_dtype),
            comptime contract_tag: Tag,
        ) !tensor_mod.TensorOf(dtype_mod.outputDType(.matmul, tensor_dtype)) {
            const output_dtype = comptime dtype_mod.outputDType(.matmul, tensor_dtype);
            const result_shape = try dotResultShape(tensor_dtype, tensor_dtype, left_tags, left, right_tags, right, contract_tag);
            const batch_rank = comptime dotBatchLen(left_tags, right_tags, contract_tag);
            const left_free_rank = comptime dotLeftFreeLen(left_tags, right_tags, contract_tag);
            const right_free_rank = comptime dotRightFreeLen(left_tags, right_tags, contract_tag);

            const left_order = dotLeftOrder(left_tags, right_tags, contract_tag);
            var left_aligned = try alignTensorTo(tensor_dtype, left_tags, left, left_order);
            defer left_aligned.deinit();

            const right_order = dotRightOrder(left_tags, right_tags, contract_tag);
            var right_aligned = try alignTensorTo(tensor_dtype, right_tags, right, right_order);
            defer right_aligned.deinit();

            const m = productRange(tensor_dtype, &left_aligned, batch_rank, left_free_rank);
            const k = left_aligned.shape.at(batch_rank + left_free_rank);
            const n = productRange(tensor_dtype, &right_aligned, batch_rank + 1, right_free_rank);

            var left_ready = try contiguousForReshape(tensor_dtype, ctx, &left_aligned);
            defer left_ready.deinit();
            var right_ready = try contiguousForReshape(tensor_dtype, ctx, &right_aligned);
            defer right_ready.deinit();

            if (comptime batch_rank == 0 and left_free_rank == 0 and right_free_rank == 0) {
                var left_vector = try left_ready.reshape(&.{k});
                defer left_vector.deinit();
                var right_vector = try right_ready.reshape(&.{k});
                defer right_vector.deinit();
                return ctx.dot(tensor_dtype, &left_vector, &right_vector);
            }

            if (comptime batch_rank != 0) {
                const num_batches = productRange(tensor_dtype, &left_aligned, 0, batch_rank);
                var left_batched = try left_ready.reshape(&.{ num_batches, m, k });
                defer left_batched.deinit();
                var right_batched = try right_ready.reshape(&.{ num_batches, k, n });
                defer right_batched.deinit();

                var out = try ctx.empty(output_dtype, result_shape);
                errdefer out.deinit();

                const left_batch_len = m * k;
                const right_batch_len = k * n;
                const out_batch_len = m * n;
                for (0..num_batches) |batch| {
                    var left_matrix = try left_batched.viewWithStridesOffset(&.{ m, k }, &.{ k, 1 }, batch * left_batch_len);
                    defer left_matrix.deinit();
                    var right_matrix = try right_batched.viewWithStridesOffset(&.{ k, n }, &.{ n, 1 }, batch * right_batch_len);
                    defer right_matrix.deinit();
                    var product = try ctx.matmul(tensor_dtype, &left_matrix, &right_matrix);
                    defer product.deinit();
                    @memcpy(out.data()[batch * out_batch_len ..][0..out_batch_len], product.dataConst());
                }
                return out;
            }

            var left_matrix = try left_ready.reshape(&.{ m, k });
            defer left_matrix.deinit();
            var right_matrix = try right_ready.reshape(&.{ k, n });
            defer right_matrix.deinit();
            var matmul = try ctx.matmul(tensor_dtype, &left_matrix, &right_matrix);
            errdefer matmul.deinit();

            if (std.mem.eql(usize, matmul.shape.slice(), result_shape[0..])) return matmul;
            const reshaped = try matmul.reshape(result_shape[0..]);
            matmul.deinit();
            return reshaped;
        }

        pub fn quantizedRhsDotRaw(
            comptime rhs_dtype: DType,
            comptime left_tags: anytype,
            left: *const RawTensor,
            ctx: *ExecContext,
            comptime right_tags: anytype,
            right: *const tensor_mod.TensorOf(rhs_dtype),
            comptime contract_tag: Tag,
            allow_gpu: bool,
        ) !RawTensor {
            comptime if (!dtype_mod.isBlockQuantized(rhs_dtype)) @compileError("quantizedRhsDotRaw requires a block-quantized RHS dtype");
            comptime if (!dtype_mod.supportsQuantizedMatmulRhs(rhs_dtype)) @compileError("RHS dtype does not support quantized matmul");

            const result_shape = try dotResultShape(.f32, rhs_dtype, left_tags, left, right_tags, right, contract_tag);
            const batch_rank = comptime dotBatchLen(left_tags, right_tags, contract_tag);
            if (comptime batch_rank != 0) @compileError("quantized RHS dot does not support shared batch tags yet");

            const left_free_rank = comptime dotLeftFreeLen(left_tags, right_tags, contract_tag);
            const right_free_rank = comptime dotRightFreeLen(left_tags, right_tags, contract_tag);
            if (comptime right_free_rank != 1) @compileError("quantized RHS dot requires one RHS free axis");

            const expected_right_order = dotRightTransBOrder(left_tags, right_tags, contract_tag);
            comptime if (!tagsEqual(right_tags, expected_right_order)) {
                @compileError("quantized RHS dot requires RHS storage order [free, contract], e.g. weight tags {.out, .in}");
            };

            const left_order = dotLeftOrder(left_tags, right_tags, contract_tag);
            var left_aligned = try alignTensorTo(.f32, left_tags, left, left_order);
            defer left_aligned.deinit();

            const m = productRange(.f32, &left_aligned, batch_rank, left_free_rank);
            const k = left_aligned.shape.at(batch_rank + left_free_rank);
            if (right.shape.at(1) != k) return TensorError.ShapeMismatch;

            var left_ready = try contiguousForReshape(.f32, ctx, &left_aligned);
            defer left_ready.deinit();

            var left_matrix = try left_ready.reshape(&.{ m, k });
            defer left_matrix.deinit();
            var matmul = try ctx.matmul2DWithQuantizedTensorRhs(rhs_dtype, &left_matrix, right, .{ .allow_gpu = allow_gpu });
            errdefer matmul.deinit();

            if (std.mem.eql(usize, matmul.shape.slice(), result_shape[0..])) return matmul;
            const reshaped = try matmul.reshape(result_shape[0..]);
            matmul.deinit();
            return reshaped;
        }

        /// The f16/bf16 constant-RHS lowering `dot` shares with the quantized
        /// arms: the NT fast path runs the half-precision TransB kernel when the
        /// contraction is a plain 2-D `[m,k]x[n,k]` (no batch axes, one right
        /// free axis, right already in TransB order); everything else widens the
        /// RHS to f32 once and takes the typed dot.
        pub fn halfRhsDotRaw(
            comptime rhs_dtype: DType,
            comptime left_tags: anytype,
            left: *const RawTensor,
            ctx: *ExecContext,
            comptime right_tags: anytype,
            right: *const tensor_mod.TensorOf(rhs_dtype),
            comptime contract_tag: Tag,
        ) !RawTensor {
            comptime std.debug.assert(rhs_dtype == .f16 or rhs_dtype == .bf16);
            const result_shape = try dotResultShape(.f32, rhs_dtype, left_tags, left, right_tags, right, contract_tag);
            const batch_rank = comptime dotBatchLen(left_tags, right_tags, contract_tag);
            const left_free_rank = comptime dotLeftFreeLen(left_tags, right_tags, contract_tag);
            const right_free_rank = comptime dotRightFreeLen(left_tags, right_tags, contract_tag);

            const expected_right_order = dotRightTransBOrder(left_tags, right_tags, contract_tag);
            if (comptime batch_rank == 0 and right_free_rank == 1 and tagsEqual(right_tags, expected_right_order)) {
                const left_order = dotLeftOrder(left_tags, right_tags, contract_tag);
                var left_aligned = try alignTensorTo(.f32, left_tags, left, left_order);
                defer left_aligned.deinit();

                const m = productRange(.f32, &left_aligned, batch_rank, left_free_rank);
                const k = left_aligned.shape.at(batch_rank + left_free_rank);
                if (right.shape.at(1) != k) return TensorError.ShapeMismatch;

                var left_ready = try contiguousForReshape(.f32, ctx, &left_aligned);
                defer left_ready.deinit();
                var right_ready = try contiguousForReshape(rhs_dtype, ctx, right);
                defer right_ready.deinit();

                var left_matrix = try left_ready.reshape(&.{ m, k });
                defer left_matrix.deinit();
                var right_matrix = try right_ready.reshape(&.{ right.shape.at(0), k });
                defer right_matrix.deinit();

                var matmul = try ctx.matmulTransB2DWithHalfRhs(rhs_dtype, &left_matrix, &right_matrix);
                errdefer matmul.deinit();

                if (std.mem.eql(usize, matmul.shape.slice(), result_shape[0..])) return matmul;
                const reshaped = try matmul.reshape(result_shape[0..]);
                matmul.deinit();
                return reshaped;
            }

            var right_f32 = try ctx.cast(rhs_dtype, .f32, right);
            defer right_f32.deinit();
            return typedDotRaw(.f32, left_tags, left, ctx, right_tags, &right_f32, contract_tag);
        }

        pub fn TensorObject(comptime T: type) type {
            return switch (@typeInfo(T)) {
                .pointer => |ptr| ptr.child,
                else => T,
            };
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
