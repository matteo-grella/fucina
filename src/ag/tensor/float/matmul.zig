//! f32 tensor methods: contractions: matmul/dot/einsum, packed and ternary-STE RHS. A mixin over the ag
//! FloatTensor struct; aliased back onto it in ../../tensor.zig.

const std = @import("std");
const tensor_mod = @import("../../../tensor.zig");
const dtype_mod = @import("../../../dtype.zig");
const exec_mod = @import("../../../exec.zig");
const backend_mod = @import("../../../backend.zig");
const parallel = @import("../../../parallel.zig");
const tag_ops = @import("../../../tag_ops.zig");
const control = @import("../../control.zig");
const core = @import("../../core.zig");
const tags_mod = @import("../../../tags.zig");
const backward_matmul = @import("../../backward/matmul.zig");

const TensorError = tensor_mod.TensorError;
const RawTensor = tensor_mod.Tensor;
const ExecContext = exec_mod.ExecContext;
const Tag = tags_mod.Tag;
const normalizeTags = tags_mod.normalizeTags;
const tagIndexOrCompileError = tags_mod.tagIndexOrCompileError;
const replaceTag = tags_mod.replaceTag;
const dotResultTags = tags_mod.dotResultTags;
const tagsEqual = tags_mod.tagsEqual;
const dotLeftOrder = tags_mod.dotLeftOrder;
const dotRightTransBOrder = tags_mod.dotRightTransBOrder;
const dotBatchTags = tags_mod.dotBatchTags;
const dotLeftFreeTags = tags_mod.dotLeftFreeTags;
const dotRightFreeTags = tags_mod.dotRightFreeTags;
const alignTensorTo = tag_ops.alignTensorTo;
const contiguousForReshape = tag_ops.contiguousForReshape;
const dotResultShape = tag_ops.dotResultShape;
const productRange = tag_ops.productRange;
const Matmul2DBackward = backward_matmul.Matmul2DBackward;
const BmmBackward = backward_matmul.BmmBackward;
const DotBackward = backward_matmul.DotBackward;
const AddDotBackward = backward_matmul.AddDotBackward;
const EinsumBackward = backward_matmul.EinsumBackward;
const ConstRhsDotBackward = backward_matmul.ConstRhsDotBackward;
const ConstRhsEinsumBackward = backward_matmul.ConstRhsEinsumBackward;
const TernarySteDotBackward = backward_matmul.TernarySteDotBackward;

pub fn Ops(comptime Self: type) type {
    return struct {
        const tags = Self.axis_tags;
        const tag_rank = Self.tag_count;
        const tag_count = Self.tag_count;
        const axis = Self.axis;
        const ag_tensor = Self.ag_root;
        const Tensor = ag_tensor.Tensor;
        const PackedRhs = ag_tensor.PackedRhs;
        const packedRhsType = ag_tensor.packedRhsType;
        const plumbing = @import("../plumbing.zig").Mod(ag_tensor);
        const recordsGrad = plumbing.recordsGrad;
        const finishTypedNoGrad = plumbing.finishTypedNoGrad;
        const rawShapeArray = plumbing.rawShapeArray;
        const rawShapeArrayOf = plumbing.rawShapeArrayOf;
        const cloneInverseRopeTable = plumbing.cloneInverseRopeTable;
        const finishOp = plumbing.finishOp;
        const dtype = Self.dtype;
        /// The f32 branch is the differentiable one; every other dtype takes
        /// the constant tail.
        const differentiable = dtype == .f32;
        const matmul_dtype = dtype_mod.outputDType(.matmul, dtype);
        const finishNoGrad = plumbing.finishNoGrad;
        const TensorObject = plumbing.TensorObject;
        const tensorObjectPtrFrom = plumbing.tensorObjectPtrFrom;

        /// Explicit matmul over caller-named result axes, comptime-routed on
        /// operand rank: BOTH operands rank-2 → the 2-D GEMM entries
        /// (`.plain`: `[m,k]·[k,n] -> [m,n]`; `.trans_b`: `[m,k]·[n,k]ᵀ ->
        /// [m,n]`); anything else → the batched bmm entries with stride-0
        /// BROADCAST leading batch axes (`[...,m,k]·[...,k,n] -> [...,m,n]`
        /// etc.) — mixed-rank operands broadcast rather than error.
        /// `.trans_a` (`[...,k,m]ᵀ·[...,k,n]`) exists only on the batched
        /// path: rank-2 Aᵀ·B has no backward record — use `dot`, whose tag
        /// algebra reaches the 2-D trans-A kernel. f32 only, full two-operand
        /// grads; unlike `dot` there is no materialize fallback — the
        /// operands' storage order IS the kernel layout. (Distinct from the
        /// strictly-2-D-plain `ExecContext.matmul` at the exec layer.)
        pub fn matmul(self: *const Self, ctx: *ExecContext, other: anytype, comptime kind: exec_mod.MatmulKind, comptime out_tags: anytype) !Tensor(.{ .dtype = matmul_dtype, .tags = out_tags }) {
            const other_ptr = tensorObjectPtrFrom(@TypeOf(other), &other);
            const other_rank = comptime TensorObject(@TypeOf(other)).axis_tags.len;
            comptime {
                if (TensorObject(@TypeOf(other)).dtype != dtype) @compileError("matmul requires matching dtypes; cast explicitly (dot takes a 16-bit or quantized RHS)");
            }
            if (comptime (tag_rank == 2 and other_rank == 2)) {
                comptime if (kind == .trans_a) {
                    @compileError("matmul: rank-2 .trans_a has no backward record — use `dot` (its tag algebra reaches the 2-D trans-A kernel)");
                };
                var value = try ctx.matmul(dtype, kind, self.asRawTensor(), other_ptr.asRawTensor());
                errdefer value.deinit();
                if (comptime !differentiable) return finishTypedNoGrad(Tensor(.{ .dtype = matmul_dtype, .tags = out_tags }), ctx, value, self.requiresGrad() or other_ptr.requiresGrad());
                if (!recordsGrad(self.requiresGrad() or other_ptr.requiresGrad())) return finishNoGrad(out_tags, ctx, value);
                const Record = Matmul2DBackward(kind == .trans_b);
                var saved_left = try self.asRawTensor().cloneView();
                errdefer saved_left.deinit();
                var saved_right = try other_ptr.asRawTensor().cloneView();
                errdefer saved_right.deinit();
                return finishOp(out_tags, ctx, value, Record{
                    .parents = .{ self.grad_state, other_ptr.grad_state },
                    .left = saved_left,
                    .right = saved_right,
                });
            }
            var value = try ctx.bmm(dtype, kind, self.asRawTensor(), other_ptr.asRawTensor());
            errdefer value.deinit();
            if (comptime !differentiable) return finishTypedNoGrad(Tensor(.{ .dtype = matmul_dtype, .tags = out_tags }), ctx, value, self.requiresGrad() or other_ptr.requiresGrad());
            if (!recordsGrad(self.requiresGrad() or other_ptr.requiresGrad())) return finishNoGrad(out_tags, ctx, value);
            const Record = BmmBackward(kind);
            var saved_left = try self.asRawTensor().cloneView();
            errdefer saved_left.deinit();
            var saved_right = try other_ptr.asRawTensor().cloneView();
            errdefer saved_right.deinit();
            return finishOp(out_tags, ctx, value, Record{
                .parents = .{ self.grad_state, other_ptr.grad_state },
                .left = saved_left,
                .right = saved_right,
            });
        }

        pub fn dot(self: *const Self, ctx: *ExecContext, other: anytype, comptime contract_tag: Tag) !Tensor(dotResultTags(tags, TensorObject(@TypeOf(other)).axis_tags, contract_tag)) {
            const Other = TensorObject(@TypeOf(other));
            const other_tags = Other.axis_tags;
            const other_ptr = tensorObjectPtrFrom(@TypeOf(other), &other);
            const result_tags = dotResultTags(tags, other_tags, contract_tag);
            if (comptime dtype_mod.isBlockQuantized(Other.dtype)) {
                // Gradient-aware GPU policy: the quant GPU dot is legal
                // under gradients — the VJP never reads the forward
                // kernel's internals (dgrad contracts gy with a
                // transiently widened f32 weight; the quant RHS is const,
                // so there is no wgrad), and the GPU and CPU quant
                // kernels sit within serving-parity tolerance of each
                // other. Backend gates own the economics per memory
                // system (Metal: size; CUDA: residency-aware floors).
                // Checkpoint blocks still pin BOTH passes to the CPU
                // kernels through the context's disable scope, keeping
                // the recompute bitwise against pass 1.
                const allow_gpu = ctx.quantDotGpuEnabled();
                var value = try tag_ops.contract(.{ .quant_rhs = Other.dtype }, tags, self.asRawTensor(), ctx, other_tags, other_ptr.asRawTensor(), result_tags, .{ .allow_gpu = allow_gpu });
                errdefer value.deinit();
                if (!recordsGrad(self.requiresGrad())) return finishNoGrad(result_tags, ctx, value);
                const Record = ConstRhsDotBackward(Other.dtype, tags, other_tags, contract_tag);
                var saved_right = try other_ptr.asRawTensor().cloneView();
                errdefer saved_right.deinit();
                var saved_left: ?RawTensor = if (null != null) try self.asRawTensor().cloneView() else null;
                errdefer if (saved_left) |*v| v.deinit();
                return finishOp(result_tags, ctx, value, Record{
                    .parents = .{ self.grad_state, null },
                    .left_shape = rawShapeArray(tags, self.asRawTensor()),
                    .right_shape = rawShapeArrayOf(Other.dtype, other_tags, other_ptr.asRawTensor()),
                    .right_value = saved_right,
                    .left_value = saved_left,
                });
            }
            if (comptime Other.dtype == .f16) {
                var value = try tag_ops.contract(.{ .half_rhs = .f16 }, tags, self.asRawTensor(), ctx, other_tags, other_ptr.asRawTensor(), result_tags, .{});
                errdefer value.deinit();
                if (!recordsGrad(self.requiresGrad() or other_ptr.requiresGrad())) return finishNoGrad(result_tags, ctx, value);
                const Record = ConstRhsDotBackward(.f16, tags, other_tags, contract_tag);
                var saved_right = try other_ptr.asRawTensor().cloneView();
                errdefer saved_right.deinit();
                var saved_left: ?RawTensor = if (other_ptr.grad_state != null) try self.asRawTensor().cloneView() else null;
                errdefer if (saved_left) |*v| v.deinit();
                return finishOp(result_tags, ctx, value, Record{
                    .parents = .{ self.grad_state, other_ptr.grad_state },
                    .left_shape = rawShapeArray(tags, self.asRawTensor()),
                    .right_shape = rawShapeArrayOf(.f16, other_tags, other_ptr.asRawTensor()),
                    .right_value = saved_right,
                    .left_value = saved_left,
                });
            }
            if (comptime Other.dtype == .bf16) {
                var value = try tag_ops.contract(.{ .half_rhs = .bf16 }, tags, self.asRawTensor(), ctx, other_tags, other_ptr.asRawTensor(), result_tags, .{});
                errdefer value.deinit();
                if (!recordsGrad(self.requiresGrad() or other_ptr.requiresGrad())) return finishNoGrad(result_tags, ctx, value);
                const Record = ConstRhsDotBackward(.bf16, tags, other_tags, contract_tag);
                var saved_right = try other_ptr.asRawTensor().cloneView();
                errdefer saved_right.deinit();
                var saved_left: ?RawTensor = if (other_ptr.grad_state != null) try self.asRawTensor().cloneView() else null;
                errdefer if (saved_left) |*v| v.deinit();
                return finishOp(result_tags, ctx, value, Record{
                    .parents = .{ self.grad_state, other_ptr.grad_state },
                    .left_shape = rawShapeArray(tags, self.asRawTensor()),
                    .right_shape = rawShapeArrayOf(.bf16, other_tags, other_ptr.asRawTensor()),
                    .right_value = saved_right,
                    .left_value = saved_left,
                });
            }
            var value = try tag_ops.contract(.f32, tags, self.asRawTensor(), ctx, other_tags, other_ptr.asRawTensor(), result_tags, .{});
            errdefer value.deinit();
            if (!recordsGrad(self.requiresGrad() or other_ptr.requiresGrad())) return finishNoGrad(result_tags, ctx, value);
            const Record = DotBackward(tags, other_tags, contract_tag);
            var saved_left = try self.asRawTensor().cloneView();
            errdefer saved_left.deinit();
            var saved_right = try other_ptr.asRawTensor().cloneView();
            errdefer saved_right.deinit();
            return finishOp(result_tags, ctx, value, Record{
                .parents = .{ self.grad_state, other_ptr.grad_state },
                .left_shape = rawShapeArray(tags, self.asRawTensor()),
                .right_shape = rawShapeArray(other_tags, other_ptr.asRawTensor()),
                .estimated_work = Record.einsumBackwardWorkEstimate(self.grad_state, other_ptr.grad_state, self.asRawTensor(), other_ptr.asRawTensor()),
                .left_value = saved_left,
                .right_value = saved_right,
            });
        }

        /// `self + a·b` in one op (torch's addmm). The product accumulates
        /// directly into a copy of `self` — the BLAS beta=1 route or the
        /// vector accumulate kernels — so the residual/projection pattern
        /// costs one GEMM with no intermediate product tensor and no
        /// separate add pass. On the vector kernels the values are
        /// bit-identical to `a.dot(b)` then `self.add` (the finalized
        /// accumulator joins the addend with the same single f32 add); the
        /// BLAS beta=1 route may differ from the composed pair in the last
        /// ulp, exactly as any two BLAS entry points may.
        ///
        /// Supported form (compile-checked): f32 operands, `a` tagged
        /// [rows, contract], `b` tagged [contract, cols], `self` tagged
        /// exactly [rows, cols]. Differentiable in all three operands.
        pub fn addDot(self: *const Self, ctx: *ExecContext, a: anytype, b: anytype, comptime contract_tag: Tag) !Self {
            const Left = TensorObject(@TypeOf(a));
            const Right = TensorObject(@TypeOf(b));
            const left_tags = Left.axis_tags;
            const right_tags = Right.axis_tags;
            comptime {
                if (Left.dtype != .f32 or Right.dtype != .f32)
                    @compileError("addDot: f32 operands only — compose dot + add for other RHS dtypes");
                if (left_tags.len != 2 or right_tags.len != 2 or tags.len != 2)
                    @compileError("addDot: rank-2 addmm form only ([rows, contract] · [contract, cols] into [rows, cols])");
                if (left_tags[1] != contract_tag or right_tags[0] != contract_tag)
                    @compileError("addDot: the contract tag must be a's last and b's first axis");
                if (tags[0] != left_tags[0] or tags[1] != right_tags[1])
                    @compileError("addDot: self must be tagged [a's rows, b's cols]");
            }
            const a_ptr = tensorObjectPtrFrom(@TypeOf(a), &a);
            const b_ptr = tensorObjectPtrFrom(@TypeOf(b), &b);
            var value = try ctx.matmulAdd(a_ptr.asRawTensor(), b_ptr.asRawTensor(), self.asRawTensor());
            errdefer value.deinit();
            const wants_grad = self.requiresGrad() or a_ptr.requiresGrad() or b_ptr.requiresGrad();
            if (!recordsGrad(wants_grad)) return finishNoGrad(tags, ctx, value);
            const Record = AddDotBackward(tags, left_tags, right_tags, contract_tag);
            var saved_left = try a_ptr.asRawTensor().cloneView();
            errdefer saved_left.deinit();
            var saved_right = try b_ptr.asRawTensor().cloneView();
            errdefer saved_right.deinit();
            return finishOp(tags, ctx, value, Record{
                .parents = .{ self.grad_state, a_ptr.grad_state, b_ptr.grad_state },
                .estimated_work = Record.workEstimate(a_ptr.grad_state, b_ptr.grad_state, a_ptr.asRawTensor(), b_ptr.asRawTensor()),
                .left_value = saved_left,
                .right_value = saved_right,
            });
        }

        /// Multi-index tagged contraction (einsum). `out_tags` is the whole
        /// equation: `result[out_tags] = Σ over every tag not in out_tags of
        /// self ⊙ other`. Shared tags are batch axes when kept and contraction
        /// axes when dropped; operand-private tags are free axes when kept and
        /// summed away when dropped. Result axis order is exactly `out_tags`;
        /// every output tag must exist in an operand (compile error) and every
        /// shared dim must match (`ShapeMismatch`). Generalizes `dot` to any
        /// number of contraction, batch, and free axes in one differentiable
        /// operation that lowers onto the same matmul/bmm kernels; both
        /// operand gradients are einsums themselves (no pointwise fallback).
        /// An f16/bf16 `other` is widened to f32 once per call (forward and
        /// backward); a constant 16-bit RHS routes gradient to `self` only,
        /// while a grad-requiring 16-bit RHS variable also receives its own
        /// f32 gradient. Quantized RHS stays dot-only.
        pub fn einsum(self: *const Self, ctx: *ExecContext, other: anytype, comptime out_tags: anytype) !Tensor(.{ .dtype = matmul_dtype, .tags = normalizeTags(out_tags) }) {
            const Other = TensorObject(@TypeOf(other));
            comptime {
                if (dtype_mod.isBlockQuantized(Other.dtype))
                    @compileError("einsum does not take a quantized RHS; use dot, whose packed kernels require the [free, contract] weight layout");
                if (Other.dtype != .f32 and Other.dtype != .f16 and Other.dtype != .bf16)
                    @compileError("einsum requires f32 operands (f16/bf16 RHS runs as a widened constant)");
            }
            const other_tags = Other.axis_tags;
            const other_ptr = tensorObjectPtrFrom(@TypeOf(other), &other);
            const result_tags = comptime normalizeTags(out_tags);
            if (comptime !differentiable) {
                // The 16-bit branch: same-dtype operands through the widened
                // f32 lowering, a constant of the matmul output dtype.
                comptime if (Other.dtype != dtype) @compileError("einsum on a 16-bit tensor requires a same-dtype RHS; cast explicitly");
                var value = try tag_ops.taggedEinsum(dtype, tags, self.asRawTensor(), ctx, other_tags, other_ptr.asRawTensor(), result_tags);
                errdefer value.deinit();
                return finishTypedNoGrad(Tensor(.{ .dtype = matmul_dtype, .tags = result_tags }), ctx, value, self.requiresGrad() or other_ptr.requiresGrad());
            }
            if (comptime (Other.dtype == .f16 or Other.dtype == .bf16)) {
                // Mixed-precision RHS: widen once per call and run the f32
                // lowering. A constant RHS routes gradient to `self` only; a
                // grad-requiring 16-bit RHS variable also receives its f32
                // gradient (gradients are always f32).
                var right_f32 = try ctx.cast(Other.dtype, .f32, other_ptr.asRawTensor());
                defer right_f32.deinit();
                var value = try tag_ops.taggedEinsum(.f32, tags, self.asRawTensor(), ctx, other_tags, &right_f32, result_tags);
                errdefer value.deinit();
                if (!recordsGrad(self.requiresGrad() or other_ptr.requiresGrad())) return finishNoGrad(result_tags, ctx, value);
                const Record = ConstRhsEinsumBackward(Other.dtype, tags, other_tags, result_tags);
                var saved_right = try other_ptr.asRawTensor().cloneView();
                errdefer saved_right.deinit();
                var saved_left: ?RawTensor = if (other_ptr.grad_state != null) try self.asRawTensor().cloneView() else null;
                errdefer if (saved_left) |*v| v.deinit();
                return finishOp(result_tags, ctx, value, Record{
                    .parents = .{ self.grad_state, other_ptr.grad_state },
                    .left_shape = rawShapeArray(tags, self.asRawTensor()),
                    .right_shape = rawShapeArrayOf(Other.dtype, other_tags, other_ptr.asRawTensor()),
                    .right_value = saved_right,
                    .left_value = saved_left,
                });
            }
            var value = try tag_ops.taggedEinsum(.f32, tags, self.asRawTensor(), ctx, other_tags, other_ptr.asRawTensor(), result_tags);
            errdefer value.deinit();
            if (!recordsGrad(self.requiresGrad() or other_ptr.requiresGrad())) return finishNoGrad(result_tags, ctx, value);
            const Record = EinsumBackward(tags, other_tags, result_tags);
            var saved_left = try self.asRawTensor().cloneView();
            errdefer saved_left.deinit();
            var saved_right = try other_ptr.asRawTensor().cloneView();
            errdefer saved_right.deinit();
            return finishOp(result_tags, ctx, value, Record{
                .parents = .{ self.grad_state, other_ptr.grad_state },
                .left_shape = rawShapeArray(tags, self.asRawTensor()),
                .right_shape = rawShapeArray(other_tags, other_ptr.asRawTensor()),
                .estimated_work = Record.einsumBackwardWorkEstimate(self.grad_state, other_ptr.grad_state, self.asRawTensor(), other_ptr.asRawTensor()),
                .left_value = saved_left,
                .right_value = saved_right,
            });
        }

        /// Trainable ternary linear (BitNet b1.58 straight-through estimator):
        /// every forward encodes the f32 latent `weight` (tags `{.out, .in}`,
        /// per-tensor absmean scale, round-clip to {-1, 0, +1}) to TQ2_0 and
        /// contracts `self` (`[..., in]`) against it with the mul-free f32
        /// kernel. Backward: dx flows through the QUANTIZED weight
        /// (dequantize-then-matmul); dW is the straight-through estimate —
        /// the plain matmul VJP as if the forward had been `x @ Wᵀ` with the
        /// latent weight (identity through the quantizer, no clip/mask).
        /// The contract dim must be a multiple of 256 (the TQ2_0 block size);
        /// anything else fails with `error.TernaryContractDimNotBlockAligned`.
        pub fn dotTernarySte(self: *const Self, ctx: *ExecContext, weight: anytype, comptime contract_tag: Tag) !Tensor(dotResultTags(tags, TensorObject(@TypeOf(weight)).axis_tags, contract_tag)) {
            const Weight = TensorObject(@TypeOf(weight));
            const weight_tags = Weight.axis_tags;
            const result_tags = dotResultTags(tags, weight_tags, contract_tag);
            comptime {
                if (Weight.dtype != .f32) @compileError("dotTernarySte requires an f32 latent weight");
                if (dotBatchTags(tags, weight_tags, contract_tag).len != 0) @compileError("dotTernarySte does not support shared batch tags");
                if (dotRightFreeTags(tags, weight_tags, contract_tag).len != 1) @compileError("dotTernarySte requires one weight free axis");
                if (!tagsEqual(weight_tags, dotRightTransBOrder(tags, weight_tags, contract_tag)))
                    @compileError("dotTernarySte requires weight storage order [free, contract], e.g. weight tags {.out, .in}");
                if (tagIndexOrCompileError(tags, contract_tag) != tags.len - 1)
                    @compileError("dotTernarySte requires lhs storage order [..., contract]");
            }
            const weight_ptr = tensorObjectPtrFrom(@TypeOf(weight), &weight);
            const weight_raw = weight_ptr.asRawTensor();

            const result_shape = try dotResultShape(.f32, .f32, tags, self.asRawTensor(), weight_tags, weight_raw, contract_tag);
            const n = weight_raw.shape.at(0);
            const k = weight_raw.shape.at(1);
            if (self.asRawTensor().shape.at(tag_rank - 1) != k) return TensorError.ShapeMismatch;
            if (k == 0 or k % dtype_mod.qk_k_block_size != 0) return error.TernaryContractDimNotBlockAligned;

            const left_free_rank = comptime dotLeftFreeTags(tags, weight_tags, contract_tag).len;
            var left_aligned = try alignTensorTo(.f32, tags, self.asRawTensor(), dotLeftOrder(tags, weight_tags, contract_tag));
            defer left_aligned.deinit();
            const m = productRange(.f32, &left_aligned, 0, left_free_rank);
            var left_ready = try contiguousForReshape(.f32, ctx, &left_aligned);
            defer left_ready.deinit();
            var left_matrix = try left_ready.reshape(&.{ m, k });
            defer left_matrix.deinit();

            var weight_ready = try contiguousForReshape(.f32, ctx, weight_raw);
            defer weight_ready.deinit();

            // Per-tensor absmean scale + round-clip encode of the latent weight.
            var rhs = try backend_mod.quantized_matmul.ternary.quantizedMatmulRhsTQ2_0FromF32Absmean(ctx.allocator, k, n, weight_ready.dataConst());
            var rhs_owned = true;
            errdefer if (rhs_owned) rhs.deinit();

            var value = forward: {
                var product = try ctx.empty(.f32, .{ m, n });
                errdefer product.deinit();
                const work = parallel.saturatedMul3(m, n, k);
                const config: backend_mod.vector_impl.ParallelConfig =
                    if (work >= parallel.vector_matmul_work_threshold) .{ .pool = ctx.workPool() } else .{};
                // Deliberately the vector kernel on BOTH backend kinds
                // (including -Dbackend=scalar): the mul-free f32 kernel is
                // pure fixed-order @Vector bitwise ops + adds, bitwise-
                // identical on every target by construction, so the scalar
                // leg exercises the same numerics (the quant matmuls instead
                // select their scalar accumulators through backend/isa.zig).
                backend_mod.vector_impl.matmul_quant.matmul2DTQ2_0F32RhsInto(config, product.data(), left_matrix.dataConst(), &rhs, m, n, k);
                if (std.mem.eql(usize, product.shape.slice(), result_shape[0..])) break :forward product;
                const reshaped = try product.reshape(result_shape[0..]);
                product.deinit();
                break :forward reshaped;
            };
            errdefer value.deinit();

            // The encoded rhs is a non-refcounted resource: the record
            // literal moves it into the node, and `finishOp` cannot fail
            // once the node exists, so the site's errdefer covers exactly
            // the failures before the hand-off.
            if (!recordsGrad(self.requiresGrad() or weight_ptr.requiresGrad())) {
                rhs_owned = false;
                rhs.deinit();
                return finishNoGrad(result_tags, ctx, value);
            }
            const Record = TernarySteDotBackward(tags);
            var saved_left = try left_matrix.cloneView();
            errdefer saved_left.deinit();
            const out = try finishOp(result_tags, ctx, value, Record{
                .parents = .{ self.grad_state, weight_ptr.grad_state },
                .left = saved_left,
                .left_shape = rawShapeArray(tags, self.asRawTensor()),
                .estimated_work = Record.workEstimate(self.grad_state, weight_ptr.grad_state, &left_matrix, rhs.n),
                .rhs = rhs,
            });
            rhs_owned = false; // owned by the node from here on
            return out;
        }

        /// Packed-RHS matmul: `out[free, out_tag] = self[free, contract_tag] · rhsᵀ`.
        /// `rhs` points to one of the packed RHS containers produced by
        /// `packRhs`/`packRhsAs` (dense f32 panels or a quantized lane
        /// pack); the container type is comptime-dispatched from the pointer.
        /// No gradient support: packed resources live outside the graph.
        pub fn dotPacked(
            self: *const Self,
            ctx: *ExecContext,
            rhs: anytype,
            comptime contract_tag: Tag,
            comptime out_tag: Tag,
        ) !Tensor(replaceTag(tags, contract_tag, out_tag)) {
            const Rhs = comptime packedRhsType(@TypeOf(rhs), "dotPacked");
            comptime {
                if (tag_rank != 2) @compileError("dotPacked (" ++ @typeName(Rhs) ++ " RHS) currently requires a rank-2 lhs");
                if (axis(contract_tag) != 1) @compileError("dotPacked (" ++ @typeName(Rhs) ++ " RHS) requires lhs storage order [free, contract]");
            }
            comptime if (Rhs == backend_mod.QuantizedMatmulRhsQ4_Kx4)
                @compileError("dotPacked: the Q4_Kx4 pack has no facade entry (kernel-comparison surface below the facade); pack q4_k with packRhs (x2mmla/x8) instead");
            if (self.requiresGrad()) return if (Rhs == backend_mod.PackedDenseRhs)
                error.GradientPackedMatmulUnsupported
            else
                error.GradientQuantizedMatmulUnsupported;
            var value = try ctx.matmulPacked(self.asRawTensor(), rhs);
            errdefer value.deinit();
            return finishNoGrad(replaceTag(tags, contract_tag, out_tag), ctx, value);
        }

        /// Snapshot this rank-2 f32 `[out, contract]` weight into load-time
        /// output-row panels for `dotPacked`. The resource is no-grad and must
        /// be deinitialized by its owner.
        pub fn packRhs(self: *const Self, ctx: *ExecContext) !PackedRhs(.f32) {
            comptime if (tag_rank != 2) @compileError("packRhs requires a rank-2 tensor");
            if (self.requiresGrad()) return error.GradientPackedMatmulUnsupported;
            return ctx.packDenseMatmulRhs(.f32, self.asRawTensor());
        }

        /// Fused rmsNormMul + packed GEMM: computes
        /// `rmsNormMul(self, norm_weight) · rhsᵀ` without materializing the
        /// normalized tensor — the kernel normalizes up to 4 rows into
        /// task-private scratch with the exact `rmsNormMulRows` kernel and
        /// quantizes with the fused packers — results match rmsNormMul +
        /// dotPacked to f32 roundoff (<= 1 ulp observed; the packed matmul's
        /// internal LHS quantizer arrangement may differ in the last ulp,
        /// the splitSwiGluDotPacked precedent).
        /// `self` is the PRE-norm rank-2 [free, contract] input;
        /// `norm_weight` is the [contract] scale row. Fused kernels exist
        /// for q8_0x4 / q4_kx8 / q5_kx8 / q6_kx4 only; q4_kx2mmla falls
        /// through at the caller like splitSwiGluDotPacked. No gradient
        /// support (same policy as `dotPacked`).
        pub fn rmsNormMulDotPacked(
            self: *const Self,
            ctx: *ExecContext,
            norm_weight: anytype,
            eps: f32,
            rhs: anytype,
            comptime contract_tag: Tag,
            comptime out_tag: Tag,
        ) !Tensor(replaceTag(tags, contract_tag, out_tag)) {
            const Rhs = comptime packedRhsType(@TypeOf(rhs), "rmsNormMulDotPacked");
            comptime {
                if (tag_rank != 2) @compileError("rmsNormMulDotPacked (" ++ @typeName(Rhs) ++ " RHS) currently requires a rank-2 lhs");
                if (axis(contract_tag) != 1) @compileError("rmsNormMulDotPacked (" ++ @typeName(Rhs) ++ " RHS) requires lhs storage order [free, contract]");
            }
            comptime {
                if (Rhs == backend_mod.PackedDenseRhs)
                    @compileError("rmsNormMulDotPacked: dense packed RHS has no fused norm kernel; use rmsNormMul + dotPacked");
                if (Rhs == backend_mod.QuantizedMatmulRhsQ4_Kx2Mmla)
                    @compileError("rmsNormMulDotPacked: no fused MMLA kernel exists; use the unfused path (rmsNormMul + dotPacked)");
                if (Rhs == backend_mod.QuantizedMatmulRhsQ4_Kx4)
                    @compileError("rmsNormMulDotPacked: the Q4_Kx4 pack has no facade entry (kernel-comparison surface below the facade)");
            }
            const weight_ptr = tensorObjectPtrFrom(@TypeOf(norm_weight), &norm_weight);
            if (self.requiresGrad() or weight_ptr.requiresGrad()) return error.GradientQuantizedMatmulUnsupported;
            var value = try ctx.rmsNormMulMatmulPacked(self.asRawTensor(), weight_ptr.asRawTensor(), eps, rhs);
            errdefer value.deinit();
            return finishNoGrad(replaceTag(tags, contract_tag, out_tag), ctx, value);
        }

        /// Fused split-SwiGLU + packed down GEMM: `self` is a fused rank-2
        /// `[free, split_tag]` gate_up activation; computes
        /// `swiglu(split(self)) · rhsᵀ` without materializing the gated tensor.
        /// Fused kernels exist for q8_0x4 / q4_kx8 / q5_kx8 / q6_kx4 only;
        /// q4_kx2mmla is a deliberate compile error — callers keep their
        /// `comptime !supports_q4_k_mmla` guard so MMLA targets fall through
        /// to the unfused path (splitSwiGlu + dotPacked).
        /// No gradient support (same policy as `dotPacked`).
        pub fn splitSwiGluDotPacked(
            self: *const Self,
            ctx: *ExecContext,
            rhs: anytype,
            comptime split_tag: Tag,
            comptime out_tag: Tag,
        ) !Tensor(replaceTag(tags, split_tag, out_tag)) {
            const Rhs = comptime packedRhsType(@TypeOf(rhs), "splitSwiGluDotPacked");
            comptime {
                if (tag_rank != 2) @compileError("splitSwiGluDotPacked (" ++ @typeName(Rhs) ++ " RHS) currently requires a rank-2 lhs");
                if (axis(split_tag) != 1) @compileError("splitSwiGluDotPacked (" ++ @typeName(Rhs) ++ " RHS) requires lhs storage order [free, fused]");
            }
            comptime {
                if (Rhs == backend_mod.PackedDenseRhs)
                    @compileError("splitSwiGluDotPacked: dense packed RHS has no fused SwiGLU kernel; use splitSwiGlu + dotPacked");
                if (Rhs == backend_mod.QuantizedMatmulRhsQ4_Kx2Mmla)
                    @compileError("splitSwiGluDotPacked: no fused MMLA kernel exists; on aarch64+i8mm targets use the unfused path (splitSwiGlu + dotPacked)");
                if (Rhs == backend_mod.QuantizedMatmulRhsQ4_Kx4)
                    @compileError("splitSwiGluDotPacked: the Q4_Kx4 pack has no facade entry (kernel-comparison surface below the facade)");
            }
            if (self.requiresGrad()) return error.GradientQuantizedMatmulUnsupported;
            var value = try ctx.splitSwiGluMatmulPacked(self.asRawTensor(), rhs);
            errdefer value.deinit();
            return finishNoGrad(replaceTag(tags, split_tag, out_tag), ctx, value);
        }

        /// Fused GeGLU + down projection for separate gate/up activations:
        /// `self` is the gate, `up` the up projection (same tags); the result is
        /// `(up * geluQuant(gate)) @ rhs` without materializing the gated tensor.
        /// Only the q8_0x4 packed layout has a fused geglu kernel today.
        /// No gradient support (same policy as `dotPacked`, checked on both
        /// `self` and `up`).
        pub fn gegluQuantDotPacked(
            self: *const Self,
            ctx: *ExecContext,
            up: *const Self,
            rhs: anytype,
            comptime in_tag: Tag,
            comptime out_tag: Tag,
        ) !Tensor(replaceTag(tags, in_tag, out_tag)) {
            const Rhs = comptime packedRhsType(@TypeOf(rhs), "gegluQuantDotPacked");
            comptime {
                if (tag_rank != 2) @compileError("gegluQuantDotPacked (" ++ @typeName(Rhs) ++ " RHS) currently requires a rank-2 lhs");
                if (axis(in_tag) != 1) @compileError("gegluQuantDotPacked (" ++ @typeName(Rhs) ++ " RHS) requires lhs storage order [free, contract]");
            }
            comptime if (Rhs != backend_mod.QuantizedMatmulRhsQ8_0x4)
                @compileError("gegluQuantDotPacked: no fused geglu kernel for packed RHS " ++ @typeName(Rhs));
            if (self.requiresGrad() or up.requiresGrad()) return error.GradientQuantizedMatmulUnsupported;
            var value = try ctx.gegluQuantMatmulPacked(self.asRawTensor(), up.asRawTensor(), rhs);
            errdefer value.deinit();
            return finishNoGrad(replaceTag(tags, in_tag, out_tag), ctx, value);
        }
    };
}
