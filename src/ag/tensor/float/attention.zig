//! f32 tensor methods: fused grouped causal attention. A mixin over the ag
//! FloatTensor struct; aliased back onto it in ../../tensor.zig.

const std = @import("std");
const tensor_mod = @import("../../../tensor.zig");
const dtype_mod = @import("../../../dtype.zig");
const exec_mod = @import("../../../exec.zig");
const backend_mod = @import("../../../backend.zig");
const parallel = @import("../../../parallel.zig");
const tag_ops = @import("../../../tagged.zig");
const control = @import("../../control.zig");
const core = @import("../../core.zig");
const tags_mod = @import("../../../tags.zig");
const backward = @import("../../backward.zig");
const elemental = @import("../../elemental.zig");
const rng = @import("../../../rng.zig");

const RawTensor = tensor_mod.Tensor;
const DType = tensor_mod.DType;
const BlockQ8_0 = dtype_mod.BlockQ8_0;
const TensorError = tensor_mod.TensorError;
const Scalar = tensor_mod.Scalar;
const ExecContext = exec_mod.ExecContext;
const SoftmaxExtOptions = exec_mod.SoftmaxExtOptions;
const UnaryOp = exec_mod.UnaryOp;
const GatedOp = exec_mod.GatedOp;
const RopeMode = exec_mod.RopeMode;
const GradState = core.GradState;
const Tag = tags_mod.Tag;
const inserted_axis = tags_mod.inserted_axis;
const normalizeTags = tags_mod.normalizeTags;
const dtypeFromSpec = tags_mod.dtypeFromSpec;
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
const removeTags = tags_mod.removeTags;
const replaceTag = tags_mod.replaceTag;
const pointwiseResultTags = tags_mod.pointwiseResultTags;
const dotResultTags = tags_mod.dotResultTags;
const insertTagAt = tags_mod.insertTagAt;
const splitTags = tags_mod.splitTags;
const mergeTags = tags_mod.mergeTags;
const tagsEqual = tags_mod.tagsEqual;
const dotLeftOrder = tags_mod.dotLeftOrder;
const dotRightOrder = tags_mod.dotRightOrder;
const dotRightTransBOrder = tags_mod.dotRightTransBOrder;
const dotBatchLen = tags_mod.dotBatchLen;
const dotLeftFreeLen = tags_mod.dotLeftFreeLen;
const dotRightFreeLen = tags_mod.dotRightFreeLen;
const alignTensorToOf = tag_ops.alignTensorToOf;
const broadcastTensorTo = tag_ops.broadcastTensorTo;
const broadcastTensorToOf = tag_ops.broadcastTensorToOf;
const contiguousForReshapeOf = tag_ops.contiguousForReshapeOf;
const dotResultShapeOf = tag_ops.dotResultShapeOf;
const pointwiseShape = tag_ops.pointwiseShape;
const pointwiseShapeOf = tag_ops.pointwiseShapeOf;
const productRangeOf = tag_ops.productRangeOf;
const validateTensorRank = tag_ops.validateTensorRank;
const validateTensorRankOf = tag_ops.validateTensorRankOf;
const PointwiseOp = backward.PointwiseOp;
const PointwiseBackward = backward.PointwiseBackward;
const CastBackward = backward.CastBackward;
const IdentityBackward = backward.IdentityBackward;
const Matmul2DBackward = backward.Matmul2DBackward;
const BmmBackward = backward.BmmBackward;
const ReluBackward = backward.ReluBackward;
const Conv2dBackward = backward.Conv2dBackward;
const UnfoldBackward = backward.UnfoldBackward;
const FoldBackward = backward.FoldBackward;
const MaxPool2dBackward = backward.MaxPool2dBackward;
const AvgPool2dBackward = backward.AvgPool2dBackward;
const Upsample2xNearestBackward = backward.Upsample2xNearestBackward;
const PreluChannelsBackward = backward.PreluChannelsBackward;
const ChannelAffineBackward = backward.ChannelAffineBackward;
const RelposShiftBackward = backward.RelposShiftBackward;
const LeakyReluBackward = backward.LeakyReluBackward;
const UnaryBackward = backward.UnaryBackward;
const unaryUsesOutput = backward.unaryUsesOutput;
const ScaleBackward = backward.ScaleBackward;
const AddScalarBackward = backward.AddScalarBackward;
const PowScalarBackward = backward.PowScalarBackward;
const MaskedFillBackward = backward.MaskedFillBackward;
const WhereBackward = backward.WhereBackward;
const DropoutBackward = backward.DropoutBackward;
const ClampBackward = backward.ClampBackward;
const GatedBackward = backward.GatedBackward;
const SplitSwiGluBackward = backward.SplitSwiGluBackward;
const SplitGluBackward = backward.SplitGluBackward;
const SumBackward = backward.SumBackward;
const MeanBackward = backward.MeanBackward;
const MaskedSumBackward = backward.MaskedSumBackward;
const MaskedMeanBackward = backward.MaskedMeanBackward;
const MaskedMinMaxBackward = backward.MaskedMinMaxBackward;
const VarBackward = backward.VarBackward;
const StandardizeBackward = backward.StandardizeBackward;
const BroadcastBackward = backward.BroadcastBackward;
const GatherBackward = backward.GatherBackward;
const TopKBackward = backward.TopKBackward;
const MinMaxBackward = backward.MinMaxBackward;
const NarrowBackward = backward.NarrowBackward;
const ConcatBackward = backward.ConcatBackward;
const CumsumBackward = backward.CumsumBackward;
const SegmentSumBackward = backward.SegmentSumBackward;
const LinearRecurrenceBackward = backward.LinearRecurrenceBackward;
const PadBackward = backward.PadBackward;
const SetSliceBackward = backward.SetSliceBackward;
const SetRowsBackward = backward.SetRowsBackward;
const IndexAddBackward = backward.IndexAddBackward;
const ProdBackward = backward.ProdBackward;
const CumprodBackward = backward.CumprodBackward;
const TakeAlongBackward = backward.TakeAlongBackward;
const LogsumexpBackward = backward.LogsumexpBackward;
const LogSoftmaxBackward = backward.LogSoftmaxBackward;
const ScatterAlongBackward = backward.ScatterAlongBackward;
const ZeroSliceBackward = backward.ZeroSliceBackward;
const ZeroRowsBackward = backward.ZeroRowsBackward;
const SoftmaxBackward = backward.SoftmaxBackward;
const SoftmaxExtBackward = backward.SoftmaxExtBackward;
const RmsNormBackward = backward.RmsNormBackward;
const RmsNormMulBackward = backward.RmsNormMulBackward;
const RmsNormMulAddBackward = backward.RmsNormMulAddBackward;
const RmsNormMulRopeBackward = backward.RmsNormMulRopeBackward;
const LayerNormBackward = backward.LayerNormBackward;
const LayerNormAffineBackward = backward.LayerNormAffineBackward;
const CrossEntropyBackward = backward.CrossEntropyBackward;
const CrossEntropyExtBackward = backward.CrossEntropyExtBackward;
const LinearCrossEntropyBackward = backward.LinearCrossEntropyBackward;
const LinearDistillBackward = backward.LinearDistillBackward;
const MseLossBackward = backward.MseLossBackward;
const HuberLossBackward = backward.HuberLossBackward;
const BceLossBackward = backward.BceLossBackward;
const KlDivLossBackward = backward.KlDivLossBackward;
const RopeBackward = backward.RopeBackward;
const RopeTableBackward = backward.RopeTableBackward;
const ReshapeBackward = backward.ReshapeBackward;
const AxisViewBackward = backward.AxisViewBackward;
const StridedViewBackward = backward.StridedViewBackward;
const CausalDepthwiseConv1dBackward = backward.CausalDepthwiseConv1dBackward;
const CausalConv1dBackward = backward.CausalConv1dBackward;
const GroupedCausalConv1dBackward = backward.GroupedCausalConv1dBackward;
const Conv1dBackward = backward.Conv1dBackward;
const ConvTranspose1dBackward = backward.ConvTranspose1dBackward;
const SnakeBackward = backward.SnakeBackward;
const GroupNormBackward = backward.GroupNormBackward;
const GroupedCausalAttentionBackward = backward.GroupedCausalAttentionBackward;
const DotBackward = backward.DotBackward;
const AddDotBackward = backward.AddDotBackward;
const EinsumBackward = backward.EinsumBackward;
const ConstRhsDotBackward = backward.ConstRhsDotBackward;
const ConstRhsEinsumBackward = backward.ConstRhsEinsumBackward;
const TernarySteDotBackward = backward.TernarySteDotBackward;

pub fn Ops(comptime Self: type) type {
    return struct {
        const tags = Self.axis_tags;
        const tag_rank = Self.tag_count;
        const tensor_rank = Self.tensor_rank;
        const tag_count = Self.tag_count;
        const axis = Self.axis;
        const ag_tensor = Self.ag_root;
        const Tensor = ag_tensor.Tensor;
        const TopKResult = ag_tensor.TopKResult;
        const PackedRhs = ag_tensor.PackedRhs;
        const SliceRange = ag_tensor.SliceRange;
        const concat_inline_inputs = ag_tensor.concat_inline_inputs;
        const AttentionMask = ag_tensor.AttentionMask;
        const AttentionKvRepr = ag_tensor.AttentionKvRepr;
        const normParamTagCheck = ag_tensor.normParamTagCheck;
        const attentionKvRepr = ag_tensor.attentionKvRepr;
        const packedRhsLayout = ag_tensor.packedRhsLayout;
        const plumbing = @import("../plumbing.zig").Mod(ag_tensor);
        const pointwise = plumbing.pointwise;
        const gatedPointwise = plumbing.gatedPointwise;
        const rowStatsAlloc = plumbing.rowStatsAlloc;
        const finishOp = plumbing.finishOp;
        const requireScopeForComposedGrad = plumbing.requireScopeForComposedGrad;
        const einsumMany = plumbing.einsumMany;
        const finishNoGrad = plumbing.finishNoGrad;
        const adoptIntoScope = plumbing.adoptIntoScope;
        const destroyGradStateOpaque = plumbing.destroyGradStateOpaque;
        const finishWithBackward = plumbing.finishWithBackward;
        const axisViewTensor = plumbing.axisViewTensor;
        const axisViewTensorOf = plumbing.axisViewTensorOf;
        const typedDotRaw = plumbing.typedDotRaw;
        const quantizedRhsDotRaw = plumbing.quantizedRhsDotRaw;
        const halfRhsDotRaw = plumbing.halfRhsDotRaw;
        const TensorObject = plumbing.TensorObject;
        const validateMaskedReduceOptions = plumbing.validateMaskedReduceOptions;
        const validateMaskType = plumbing.validateMaskType;
        const maskedReduceEmpty = plumbing.maskedReduceEmpty;
        const tensorObjectPtrFrom = plumbing.tensorObjectPtrFrom;
        const padding2dValues = plumbing.padding2dValues;
        const typed_constant = @import("../typed_constant.zig").Mod(ag_tensor);
        const QuantizedConstantTensor = typed_constant.QuantizedConstantTensor;
        const TypedConstantTensor = typed_constant.TypedConstantTensor;
        const typedFinishOp = typed_constant.typedFinishOp;

        /// Grouped (GQA) attention over every KV representation, `q` = `self`
        /// with tags `.{ .seq, .head, .d }`. The KV representation is
        /// comptime-dispatched from `@TypeOf(k)`:
        ///
        ///   - `*Tensor(.{ .seq, .kv_head, .d })` (f32)        → f32 kernel
        ///   - f16 Tensor with the same tags                   → f16 kernel (decode KV cache)
        ///   - `[]const BlockQ8_0`                             → q8_0 raw-block cache,
        ///     layout `[kv_seq, kv_heads, d/32]` (see `ExecContext.groupedCausalAttentionQ8Kv`)
        ///   - `[]const []const f16`                           → ragged multi-stream decode
        ///   - `[]const []const BlockQ8_0`                     → ragged multi-stream decode (q8_0)
        ///
        /// `opts` is a comptime-validated struct literal; unknown fields are
        /// compile errors:
        ///
        ///   - `.mask = .causal` (default) | `.bidirectional` — f32/f16 KV only
        ///     (bidirectional = every query attends every key: the
        ///     block-diffusion canvas / OmniVoice encoder attention; its SWA
        ///     reach is realized by narrowing the K/V views — no
        ///     bidirectional window exists by design).
        ///   - `.window = w` — runtime sliding window, 0 = full causal
        ///     (query p attends [max(0, p-w+1), p]); causal only.
        ///   - `.bias = &b` — rank-2 `[q_seq, kv_seq]` additive f32 bias added
        ///     to the scaled scores pre-softmax (ggml_soft_max_ext semantics,
        ///     NOT -inf masking); bidirectional + f32 KV only.
        ///   - `.kv_seq = n, .kv_heads = h` — REQUIRED for the q8_0-block
        ///     repr (raw blocks carry no shape).
        ///   - `.lens = lens, .kv_heads = h` — REQUIRED for the multi-stream
        ///     reprs. There, q's `.seq` tag is reinterpreted as the STREAM
        ///     axis: exactly one query row per stream, row `s` attending all
        ///     `lens[s]` cached positions of `k[s]`/`v[s]`; per-stream
        ///     results are bit-identical to N single-stream f16/q8 calls.
        ///
        /// Gradient matrix (unchanged from the former per-variant entries;
        /// no longer readable off the name):
        ///
        ///   - f32 KV: full q/k/v backward (windowed re-masks to the window).
        ///   - f16 KV: q-grad only — K/V are cache constants, widened to f32
        ///     once and run through the f32 kernel + backward.
        ///   - q8_0 KV: q-grad only, causal only — the cache is dequantized
        ///     to f32 once (no bidirectional q8 path exists).
        ///   - `.bias` present: inference-only; ANY grad-requiring operand
        ///     (q, k, v, or the bias) returns `error.UnsupportedGradient` —
        ///     the shared backward re-derives the softmax without a bias.
        ///   - multi-stream: inference-only (`error.UnsupportedGradient`).
        pub fn groupedAttention(
            self: *const Self,
            ctx: *ExecContext,
            k: anytype,
            v: anytype,
            kv_head_for_head: []const usize,
            comptime out_tag: Tag,
            scale_value: f32,
            opts: anytype,
        ) !Tensor(.{ .seq, out_tag }) {
            comptime if (!tagsEqual(tags, .{ .seq, .head, .d })) {
                @compileError("groupedAttention requires q tags .{ .seq, .head, .d }");
            };
            const repr = comptime attentionKvRepr(@TypeOf(k), "k");
            comptime {
                const v_repr = attentionKvRepr(@TypeOf(v), "v");
                if (repr != v_repr) @compileError("groupedAttention: k is a ." ++ @tagName(repr) ++ " KV but v is a ." ++ @tagName(v_repr) ++ " KV");
            }
            const O = @TypeOf(opts);
            comptime if (@typeInfo(O) != .@"struct") {
                @compileError("groupedAttention: opts must be a struct literal, e.g. .{} or .{ .window = w }");
            };
            const mask: AttentionMask = comptime if (@hasField(O, "mask")) opts.mask else .causal;
            comptime {
                // Field whitelist per KV repr: a misspelled option (`.windw`)
                // must be a compile error, never silently-full-causal attention.
                const allowed: []const []const u8 = switch (repr) {
                    .f32_kv => &.{ "mask", "window", "bias" },
                    .f16_kv => &.{ "mask", "window" },
                    .q8_kv => &.{ "window", "kv_seq", "kv_heads" },
                    .multi_f16_kv, .multi_q8_kv => &.{ "lens", "kv_heads" },
                };
                for (@typeInfo(O).@"struct".fields) |field| {
                    var known = false;
                    for (allowed) |name| {
                        if (std.mem.eql(u8, field.name, name)) known = true;
                    }
                    if (!known) @compileError("groupedAttention: unknown option ." ++ field.name ++ " for the ." ++ @tagName(repr) ++ " KV representation");
                }
                if (mask == .bidirectional and @hasField(O, "window")) {
                    @compileError("groupedAttention: no bidirectional windowed kernel exists — realize SWA reach by narrowing the K/V views");
                }
                if (@hasField(O, "bias") and mask != .bidirectional) {
                    @compileError("groupedAttention: .bias requires .mask = .bidirectional (the only additive-bias kernel)");
                }
                switch (repr) {
                    .q8_kv => if (!@hasField(O, "kv_seq") or !@hasField(O, "kv_heads")) {
                        @compileError("groupedAttention: the q8_0-block KV repr requires .kv_seq and .kv_heads (raw blocks carry no shape; layout [kv_seq, kv_heads, d/32])");
                    },
                    .multi_f16_kv, .multi_q8_kv => if (!@hasField(O, "lens") or !@hasField(O, "kv_heads")) {
                        @compileError("groupedAttention: the multi-stream KV reprs require .lens and .kv_heads (q's .seq tag is the stream axis)");
                    },
                    else => {},
                }
            }
            switch (comptime repr) {
                .f32_kv => {
                    if (comptime @hasField(O, "bias")) {
                        const bias = opts.bias;
                        const bias_ptr = tensorObjectPtrFrom(@TypeOf(bias), &bias);
                        comptime if (TensorObject(@TypeOf(bias)).axis_tags.len != 2) {
                            @compileError("groupedAttention: .bias must be a rank-2 [q_seq, kv_seq] tensor");
                        };
                        if (self.requiresGrad() or k.requiresGrad() or v.requiresGrad() or bias_ptr.requiresGrad()) {
                            return error.UnsupportedGradient;
                        }
                        var value = try ctx.groupedBidirectionalAttentionBiased(self.asRawTensor(), k.asRawTensor(), v.asRawTensor(), kv_head_for_head, scale_value, bias_ptr.asRawTensor());
                        errdefer value.deinit();
                        return finishNoGrad(.{ .seq, out_tag }, ctx, value);
                    }
                    const window: usize = if (comptime @hasField(O, "window")) opts.window else 0;
                    const wants_grad = self.requiresGrad() or k.requiresGrad() or v.requiresGrad();
                    // Forward-saved softmax {max, sum_exp} for the VJP's
                    // one-pass probability rebuild; the stats entry's output
                    // is bitwise identical to the stats-less ones.
                    const row_stats = try rowStatsAlloc(ctx, wants_grad, kv_head_for_head.len * self.asRawTensor().shape.at(0));
                    defer if (row_stats) |stats| ctx.allocator.free(stats);
                    var value = try if (row_stats) |stats|
                        ctx.groupedCausalAttentionStatsOut(self.asRawTensor(), k.asRawTensor(), v.asRawTensor(), kv_head_for_head, scale_value, window, mask == .causal, stats)
                    else switch (comptime mask) {
                        .causal => if (comptime @hasField(O, "window"))
                            ctx.groupedCausalAttentionWindowed(self.asRawTensor(), k.asRawTensor(), v.asRawTensor(), kv_head_for_head, scale_value, opts.window)
                        else
                            ctx.groupedCausalAttention(self.asRawTensor(), k.asRawTensor(), v.asRawTensor(), kv_head_for_head, scale_value),
                        .bidirectional => ctx.groupedBidirectionalAttention(self.asRawTensor(), k.asRawTensor(), v.asRawTensor(), kv_head_for_head, scale_value),
                    };
                    errdefer value.deinit();
                    return finishOp(.{ .seq, out_tag }, ctx, value, wants_grad, GroupedCausalAttentionBackward, .{
                        ctx.allocator,
                        self.grad_state,
                        k.grad_state,
                        v.grad_state,
                        self.asRawTensor(),
                        k.asRawTensor(),
                        v.asRawTensor(),
                        kv_head_for_head,
                        scale_value,
                        window,
                        mask == .causal,
                        row_stats orelse &[_]f32{},
                        &value,
                    });
                },
                .f16_kv => {
                    const window: usize = if (comptime @hasField(O, "window")) opts.window else 0;
                    if (self.requiresGrad()) {
                        return f16KvAttentionWithGrad(self, ctx, k, v, kv_head_for_head, out_tag, scale_value, window, mask == .causal);
                    }
                    var value = try switch (comptime mask) {
                        .causal => if (comptime @hasField(O, "window"))
                            ctx.groupedCausalAttentionF16KvWindowed(self.asRawTensor(), k.asRawTensor(), v.asRawTensor(), kv_head_for_head, scale_value, opts.window)
                        else
                            ctx.groupedCausalAttentionF16Kv(self.asRawTensor(), k.asRawTensor(), v.asRawTensor(), kv_head_for_head, scale_value),
                        .bidirectional => ctx.groupedBidirectionalAttentionF16Kv(self.asRawTensor(), k.asRawTensor(), v.asRawTensor(), kv_head_for_head, scale_value),
                    };
                    errdefer value.deinit();
                    return finishNoGrad(.{ .seq, out_tag }, ctx, value);
                },
                .q8_kv => {
                    if (self.requiresGrad()) {
                        const window: usize = if (comptime @hasField(O, "window")) opts.window else 0;
                        return q8KvAttentionWithGrad(self, ctx, k, v, opts.kv_seq, opts.kv_heads, kv_head_for_head, out_tag, scale_value, window);
                    }
                    var value = if (comptime @hasField(O, "window"))
                        try ctx.groupedCausalAttentionQ8KvWindowed(self.asRawTensor(), k, v, opts.kv_seq, opts.kv_heads, kv_head_for_head, scale_value, opts.window)
                    else
                        try ctx.groupedCausalAttentionQ8Kv(self.asRawTensor(), k, v, opts.kv_seq, opts.kv_heads, kv_head_for_head, scale_value);
                    errdefer value.deinit();
                    return finishNoGrad(.{ .seq, out_tag }, ctx, value);
                },
                .multi_f16_kv => {
                    if (self.requiresGrad()) return error.UnsupportedGradient;
                    var value = try ctx.groupedCausalAttentionMultiF16Kv(self.asRawTensor(), k, v, opts.lens, opts.kv_heads, kv_head_for_head, scale_value);
                    errdefer value.deinit();
                    return finishNoGrad(.{ .seq, out_tag }, ctx, value);
                },
                .multi_q8_kv => {
                    if (self.requiresGrad()) return error.UnsupportedGradient;
                    var value = try ctx.groupedCausalAttentionMultiQ8Kv(self.asRawTensor(), k, v, opts.lens, opts.kv_heads, kv_head_for_head, scale_value);
                    errdefer value.deinit();
                    return finishNoGrad(.{ .seq, out_tag }, ctx, value);
                },
            }
        }

        /// Gradient path for q8_0-KV attention: dequantize the constant
        /// cache to f32 once, then the f32 kernel + backward (q-grad only),
        /// mirroring `f16KvAttentionWithGrad`.
        fn q8KvAttentionWithGrad(
            self: *const Self,
            ctx: *ExecContext,
            k_blocks: []const BlockQ8_0,
            v_blocks: []const BlockQ8_0,
            kv_seq: usize,
            kv_heads: usize,
            kv_head_for_head: []const usize,
            comptime out_tag: Tag,
            scale_value: f32,
            window: usize,
        ) !Tensor(.{ .seq, out_tag }) {
            const block_size = dtype_mod.q8_0_block_size;
            if (kv_seq * kv_heads == 0 or k_blocks.len % (kv_seq * kv_heads) != 0) return TensorError.InvalidShape;
            if (v_blocks.len != k_blocks.len) return TensorError.InvalidShape;
            const d = (k_blocks.len / (kv_seq * kv_heads)) * block_size;
            var k32 = try ctx.emptyRank(3, .{ kv_seq, kv_heads, d });
            defer k32.deinit();
            try ctx.dequantizeQ8_0RowsInto(k32.data(), k_blocks);
            var v32 = try ctx.emptyRank(3, .{ kv_seq, kv_heads, d });
            defer v32.deinit();
            try ctx.dequantizeQ8_0RowsInto(v32.data(), v_blocks);
            const row_stats = try rowStatsAlloc(ctx, true, kv_head_for_head.len * self.asRawTensor().shape.at(0));
            defer if (row_stats) |stats| ctx.allocator.free(stats);
            var value = if (row_stats) |stats|
                try ctx.groupedCausalAttentionStatsOut(self.asRawTensor(), &k32, &v32, kv_head_for_head, scale_value, window, true, stats)
            else if (window == 0)
                try ctx.groupedCausalAttention(self.asRawTensor(), &k32, &v32, kv_head_for_head, scale_value)
            else
                try ctx.groupedCausalAttentionWindowed(self.asRawTensor(), &k32, &v32, kv_head_for_head, scale_value, window);
            errdefer value.deinit();
            return finishOp(.{ .seq, out_tag }, ctx, value, true, GroupedCausalAttentionBackward, .{
                ctx.allocator,
                self.grad_state,
                null,
                null,
                self.asRawTensor(),
                &k32,
                &v32,
                kv_head_for_head,
                scale_value,
                window,
                true,
                row_stats orelse &[_]f32{},
                &value,
            });
        }
        // (q8KvAttentionWithGrad above stays causal-only: no bidirectional
        // q8_0-cache user exists — the diffusion canvas runs on f16 KV.)

        /// Gradient path for f16-KV attention: K/V are constants (the cache),
        /// so only q-grad flows. Widens K/V to f32 once and runs the f32
        /// kernel + backward; the f16 fast path stays grad-free.
        fn f16KvAttentionWithGrad(
            self: *const Self,
            ctx: *ExecContext,
            k: *const Tensor(.{ .dtype = .f16, .tags = .{ .seq, .kv_head, .d } }),
            v: *const Tensor(.{ .dtype = .f16, .tags = .{ .seq, .kv_head, .d } }),
            kv_head_for_head: []const usize,
            comptime out_tag: Tag,
            scale_value: f32,
            window: usize,
            causal: bool,
        ) !Tensor(.{ .seq, out_tag }) {
            var k32 = try ctx.castTyped(.f16, .f32, k.asRawTensor());
            defer k32.deinit();
            var v32 = try ctx.castTyped(.f16, .f32, v.asRawTensor());
            defer v32.deinit();
            const row_stats = try rowStatsAlloc(ctx, true, kv_head_for_head.len * self.asRawTensor().shape.at(0));
            defer if (row_stats) |stats| ctx.allocator.free(stats);
            var value = if (row_stats) |stats|
                try ctx.groupedCausalAttentionStatsOut(self.asRawTensor(), &k32, &v32, kv_head_for_head, scale_value, window, causal, stats)
            else if (!causal)
                try ctx.groupedBidirectionalAttention(self.asRawTensor(), &k32, &v32, kv_head_for_head, scale_value)
            else if (window == 0)
                try ctx.groupedCausalAttention(self.asRawTensor(), &k32, &v32, kv_head_for_head, scale_value)
            else
                try ctx.groupedCausalAttentionWindowed(self.asRawTensor(), &k32, &v32, kv_head_for_head, scale_value, window);
            errdefer value.deinit();
            return finishOp(.{ .seq, out_tag }, ctx, value, true, GroupedCausalAttentionBackward, .{
                ctx.allocator,
                self.grad_state,
                null,
                null,
                self.asRawTensor(),
                &k32,
                &v32,
                kv_head_for_head,
                scale_value,
                window,
                causal,
                row_stats orelse &[_]f32{},
                &value,
            });
        }
    };
}
