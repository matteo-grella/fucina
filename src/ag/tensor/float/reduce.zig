//! f32 tensor methods: reductions, scans, linear recurrence. A mixin over the ag
//! FloatTensor struct; aliased back onto it in ../../tensor.zig.

const std = @import("std");
const tensor_mod = @import("../../../tensor.zig");
const exec_mod = @import("../../../exec.zig");
const tag_ops = @import("../../../tagged.zig");
const core = @import("../../core.zig");
const tags_mod = @import("../../../tags.zig");
const backward = @import("../../backward.zig");

const RawTensor = tensor_mod.Tensor;
const TensorError = tensor_mod.TensorError;
const ExecContext = exec_mod.ExecContext;
const GradState = core.GradState;
const Tag = tags_mod.Tag;
const normalizeTags = tags_mod.normalizeTags;
const tagIndex = tags_mod.tagIndex;
const removeTag = tags_mod.removeTag;
const removeTags = tags_mod.removeTags;
const tagsEqual = tags_mod.tagsEqual;
const broadcastTensorTo = tag_ops.broadcastTensorTo;
const SumBackward = backward.SumBackward;
const MeanBackward = backward.MeanBackward;
const MaskedSumBackward = backward.MaskedSumBackward;
const MaskedMeanBackward = backward.MaskedMeanBackward;
const MaskedMinMaxBackward = backward.MaskedMinMaxBackward;
const CumsumBackward = backward.CumsumBackward;
const SegmentSumBackward = backward.SegmentSumBackward;
const LinearRecurrenceBackward = backward.LinearRecurrenceBackward;
const ProdBackward = backward.ProdBackward;
const CumprodBackward = backward.CumprodBackward;

pub fn Ops(comptime Self: type) type {
    return struct {
        const tags = Self.axis_tags;
        const tag_rank = Self.tag_count;
        const tag_count = Self.tag_count;
        const axis = Self.axis;
        const ag_tensor = Self.ag_root;
        const Tensor = ag_tensor.Tensor;
        const plumbing = @import("../plumbing.zig").Mod(ag_tensor);
        const finishOp = plumbing.finishOp;
        const TensorObject = plumbing.TensorObject;
        const validateMaskedReduceOptions = plumbing.validateMaskedReduceOptions;
        const validateMaskType = plumbing.validateMaskType;
        const maskedReduceEmpty = plumbing.maskedReduceEmpty;
        const tensorObjectPtrFrom = plumbing.tensorObjectPtrFrom;

        /// `.bool`, true where ANY element along `tag` is truthy (`!= 0`;
        /// NaN is truthy, the torch.any convention), with `tag` removed.
        /// Non-differentiable constant mask (compare → i64 count →
        /// compare), unscoped-safe.
        pub fn any(self: *const Self, ctx: *ExecContext, comptime tag: Tag) !Tensor(.{ .dtype = .bool, .tags = removeTag(tags, tag) }) {
            var truthy = try self.compare(ctx, .ne, 0);
            defer truthy.deinit();
            var count = try truthy.sum(ctx, tag, .{});
            defer count.deinit();
            return count.compare(ctx, .ge, 1);
        }

        /// `.bool`, true where EVERY element along `tag` is truthy (see
        /// `any`), with `tag` removed. Counts the zero entries and tests
        /// the count against 1. Non-differentiable constant mask,
        /// unscoped-safe.
        pub fn all(self: *const Self, ctx: *ExecContext, comptime tag: Tag) !Tensor(.{ .dtype = .bool, .tags = removeTag(tags, tag) }) {
            var zero = try self.compare(ctx, .eq, 0);
            defer zero.deinit();
            var count = try zero.sum(ctx, tag, .{});
            defer count.deinit();
            return count.compare(ctx, .lt, 1);
        }

        /// Scalar `any` over every element (torch.any with no dim); `.bool`.
        pub fn anyAll(self: *const Self, ctx: *ExecContext) !Tensor(.{ .dtype = .bool, .tags = .{} }) {
            var truthy = try self.compare(ctx, .ne, 0);
            defer truthy.deinit();
            var count = try truthy.sumAll(ctx);
            defer count.deinit();
            return count.compare(ctx, .ge, 1);
        }

        /// Scalar `all` over every element (torch.all with no dim); `.bool`.
        pub fn allAll(self: *const Self, ctx: *ExecContext) !Tensor(.{ .dtype = .bool, .tags = .{} }) {
            var zero = try self.compare(ctx, .eq, 0);
            defer zero.deinit();
            var count = try zero.sumAll(ctx);
            defer count.deinit();
            return count.compare(ctx, .lt, 1);
        }

        fn sumUnmasked(self: *const Self, ctx: *ExecContext, comptime tag: Tag) !Tensor(removeTag(tags, tag)) {
            const result_tags = removeTag(tags, tag);
            var value = try ctx.sumAxisRank(tag_rank, self.asRawTensor(), axis(tag));
            errdefer value.deinit();
            return finishOp(result_tags, ctx, value, self.requiresGrad(), SumBackward(tags, result_tags), .{ ctx.allocator, self.grad_state, &self.value });
        }

        fn meanUnmasked(self: *const Self, ctx: *ExecContext, comptime tag: Tag) !Tensor(removeTag(tags, tag)) {
            const result_tags = removeTag(tags, tag);
            const reduce_axis = comptime axis(tag);
            var value = try ctx.meanAxisRank(tag_rank, self.asRawTensor(), reduce_axis);
            errdefer value.deinit();
            return finishOp(result_tags, ctx, value, self.requiresGrad(), MeanBackward(tags, result_tags, reduce_axis), .{ ctx.allocator, self.grad_state, &self.value });
        }

        /// Sum over `tag` (torch.sum); with a mask, restricted to the
        /// elements the mask selects — the `sum(a, dim, mask)` of the
        /// Fortran intrinsic set.
        ///
        /// `opts` is `.{}` (plain sum), `.{ .mask = &m }`, or
        /// `.{ .mask = &m, .empty = v }`; any other field is a compile error.
        /// The mask follows the `where`/`maskedFill` convention: `.bool` or a
        /// float read by truthiness (`!= 0`), `self`'s exact tags and shape,
        /// non-grad. Compose `broadcastTo` for a smaller mask.
        ///
        /// The masked arm's point is fusion. The composed spelling
        /// (`maskedFill` then sum) materializes a full f32 copy of the input
        /// and walks it twice; this accumulates straight out of the source
        /// through the same SIMD kernel the plain sum uses, so an ALL-TRUE
        /// mask reproduces the plain sum bitwise.
        ///
        /// A lane whose mask selects nothing yields `empty orelse 0` — the
        /// operation's identity, which is Fortran's answer and the reason a
        /// masked reduction has no `EmptySelection` error the way
        /// `maskedSelect` does. Differentiable in `self`: an excluded element
        /// contributed nothing, so it receives nothing back.
        pub fn sum(self: *const Self, ctx: *ExecContext, comptime tag: Tag, opts: anytype) !Tensor(removeTag(tags, tag)) {
            comptime validateMaskedReduceOptions(@TypeOf(opts));
            if (comptime !@hasField(@TypeOf(opts), "mask")) return sumUnmasked(self, ctx, tag);

            const Mask = TensorObject(@TypeOf(opts.mask));
            comptime validateMaskType(Mask, "sum");
            const result_tags = removeTag(tags, tag);
            var value = try ctx.sumMaskedAxisRank(Mask.dtype, tag_rank, self.asRawTensor(), opts.mask.asRawTensor(), axis(tag), maskedReduceEmpty(opts));
            errdefer value.deinit();
            return finishOp(result_tags, ctx, value, self.requiresGrad(), MaskedSumBackward(tags, result_tags, Mask.dtype), .{ ctx.allocator, self.grad_state, &self.value, opts.mask.asRawTensor() });
        }

        /// Mean over `tag` (torch.mean); with a mask, the masked sum divided
        /// by the per-lane count of SELECTED elements, not by the axis
        /// length — padding-masked pooling in one op.
        ///
        /// `opts` and the mask contract are `sum`'s. An all-true mask
        /// reproduces the plain mean bitwise.
        ///
        /// A masked lane selecting nothing yields `empty orelse NaN`: unlike
        /// a sum, a mean has no identity to fall back on (0/0), so the caller
        /// either supplies a sentinel or gets the IEEE answer. Such a lane's
        /// gradient is zero — it produced a constant, not a function of the
        /// data.
        pub fn mean(self: *const Self, ctx: *ExecContext, comptime tag: Tag, opts: anytype) !Tensor(removeTag(tags, tag)) {
            comptime validateMaskedReduceOptions(@TypeOf(opts));
            if (comptime !@hasField(@TypeOf(opts), "mask")) return meanUnmasked(self, ctx, tag);

            const Mask = TensorObject(@TypeOf(opts.mask));
            comptime validateMaskType(Mask, "mean");
            const result_tags = removeTag(tags, tag);
            var raw = try ctx.meanMaskedAxisRank(Mask.dtype, tag_rank, self.asRawTensor(), opts.mask.asRawTensor(), axis(tag), maskedReduceEmpty(opts));
            var raw_values: ?RawTensor = raw.values;
            errdefer if (raw_values) |*value| value.deinit();
            defer raw.counts.deinit();
            const out = try finishOp(result_tags, ctx, raw_values.?, self.requiresGrad(), MaskedMeanBackward(tags, result_tags, Mask.dtype), .{ ctx.allocator, self.grad_state, &self.value, opts.mask.asRawTensor(), &raw.counts });
            raw_values = null;
            return out;
        }

        /// Cumulative sum along `tag` (torch.cumsum), preserving shape:
        /// `y[..., i, ...] = Σ_{j <= i} x[..., j, ...]`. Differentiable:
        /// the gradient is the reversed cumulative (suffix) sum of the
        /// upstream gradient. Both passes are serial per row — bitwise
        /// deterministic for any thread count.
        pub fn cumsum(self: *const Self, ctx: *ExecContext, comptime tag: Tag) !Self {
            const scan_axis = comptime axis(tag);
            var value = try ctx.cumsumAxisRank(tag_rank, self.asRawTensor(), scan_axis);
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad(), CumsumBackward(tags, scan_axis), .{ ctx.allocator, self.grad_state });
        }

        /// Segmented sum along `tag`: contiguous index ranges
        /// `[offsets[i], offsets[i+1])` collapse to one output row each, so
        /// the axis size becomes `offsets.len - 1` (`offsets[0] == 0`,
        /// nondecreasing, last entry == the axis size; empty segments
        /// produce zero rows). The sorted-contiguous form of
        /// torch.segment_reduce / JAX segment_sum. Serial per segment in
        /// axis order — bitwise deterministic for any thread count.
        /// Differentiable: the VJP broadcasts each segment's gradient row
        /// back over that segment's input rows.
        pub fn segmentSum(self: *const Self, ctx: *ExecContext, comptime tag: Tag, offsets: []const usize) !Self {
            const seg_axis = comptime axis(tag);
            const n = self.dim(tag);
            var value = try ctx.segmentSumAxisRank(tag_rank, self.asRawTensor(), seg_axis, offsets);
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad(), SegmentSumBackward(tags, seg_axis), .{ ctx.allocator, self.grad_state, offsets, n });
        }

        /// First-order linear recurrence along `time_tag` — the
        /// associative-scan primitive of the SSM / linear-attention family:
        /// `h_t = a_t ⊙ h_{t-1} + b_t` where `self` supplies `b` (and the
        /// result shape), `decay` supplies `a`, and each lane (every fixed
        /// choice of the non-time axes) scans independently. `decay`
        /// broadcasts BY TAG like the pointwise ops: its tags must be a
        /// subset of `self`'s, missing or size-1 axes broadcast (a decay
        /// without the time tag is a static per-lane decay; a `[time]`-only
        /// decay is a shared schedule; state-shaped tags give
        /// Mamba/GDN-style diagonal or outer-product decay states). The
        /// broadcast is a zero-stride view read in place — never
        /// materialized. `options` is `.{}` or
        /// `.{ .initial = &h0 }` where `h0` carries `self`'s tags minus
        /// `time_tag` (exact shape, `ShapeMismatch` otherwise) and supplies
        /// `h_{-1}` per lane — the streaming-decode seam: pass the previous
        /// chunk's last time step to continue a sequence.
        ///
        /// Determinism (the `cumsum` contract): one serial pass per lane in
        /// time order, each step evaluated multiply-then-add — bitwise
        /// deterministic for any thread count. With `-Dvector-scan`,
        /// non-last time axes vectorize across independent lanes bitwise
        /// IDENTICALLY; the last axis always stays serial per lane (an
        /// in-register form would reassociate the recurrence).
        ///
        /// Differentiable in `self`, `decay`, and `initial`: the VJP is one
        /// reverse scan (`gh_t = a_{t+1}·gh_{t+1} + gy_t`), with the decay
        /// gradient `gh_t·h_{t-1}` reduced back over the broadcast axes and
        /// `initial` receiving `a_0·gh_0`.
        pub fn linearRecurrence(self: *const Self, ctx: *ExecContext, comptime time_tag: Tag, decay: anytype, options: anytype) !Self {
            const Options = @TypeOf(options);
            comptime {
                if (@typeInfo(Options) != .@"struct") @compileError("linearRecurrence: options must be a struct literal, e.g. .{} or .{ .initial = &h0 }");
                for (@typeInfo(Options).@"struct".fields) |field| {
                    if (!std.mem.eql(u8, field.name, "initial"))
                        @compileError("linearRecurrence: unknown option ." ++ field.name);
                }
            }
            const Decay = TensorObject(@TypeOf(decay));
            comptime {
                for (Decay.axis_tags) |t| {
                    if (tagIndex(tags, t) == null)
                        @compileError("linearRecurrence: decay tags must be a subset of the input tags");
                }
            }
            const time_axis = comptime axis(time_tag);
            const decay_ptr = tensorObjectPtrFrom(@TypeOf(decay), &decay);

            var tagged_shape: [tag_count]usize = undefined;
            inline for (tags, 0..) |t, i| tagged_shape[i] = self.dim(t);
            var a_view = try tag_ops.broadcastTensorTo(Decay.axis_tags, decay_ptr.asRawTensor(), tags, tagged_shape);
            defer a_view.deinit();

            var any_grad = self.requiresGrad() or decay_ptr.requiresGrad();
            var initial_raw: ?*const RawTensor = null;
            var initial_grad: ?*GradState = null;
            if (comptime @hasField(Options, "initial")) {
                const init_ptr = tensorObjectPtrFrom(@TypeOf(options.initial), &options.initial);
                const InitT = TensorObject(@TypeOf(options.initial));
                comptime {
                    if (!tagsEqual(InitT.axis_tags, removeTag(tags, time_tag)))
                        @compileError("linearRecurrence: initial must carry the input tags minus the time tag");
                }
                inline for (comptime removeTag(tags, time_tag)) |t| {
                    if (init_ptr.dim(t) != self.dim(t)) return TensorError.ShapeMismatch;
                }
                initial_raw = init_ptr.asRawTensor();
                initial_grad = init_ptr.grad_state;
                any_grad = any_grad or init_ptr.requiresGrad();
            }

            var value = try ctx.linearRecurrenceAxisRank(tag_rank, self.asRawTensor(), &a_view, time_axis, initial_raw);
            errdefer value.deinit();
            return finishOp(tags, ctx, value, any_grad, LinearRecurrenceBackward(tags, Decay.axis_tags, time_axis), .{ ctx.allocator, self.grad_state, decay_ptr.grad_state, initial_grad, &a_view, &value, initial_raw, decay_ptr.asRawTensor() });
        }

        /// Product along `tag` (torch.prod over a dim), the tag removed.
        /// Serial per row (the `cumsum` determinism contract).
        /// Differentiable with torch's zero-handling: zero-free rows get
        /// `g·(Π x)/x_i`; exactly one zero routes the whole gradient to the
        /// zero slot; two or more zeros kill the row's gradient.
        pub fn prod(self: *const Self, ctx: *ExecContext, comptime tag: Tag) !Tensor(removeTag(tags, tag)) {
            const result_tags = removeTag(tags, tag);
            const reduce_axis = comptime axis(tag);
            var value = try ctx.prodAxisRank(tag_rank, self.asRawTensor(), reduce_axis);
            errdefer value.deinit();
            return finishOp(result_tags, ctx, value, self.requiresGrad(), ProdBackward(tags, reduce_axis), .{ ctx.allocator, self.grad_state, &self.value });
        }

        /// Inclusive running product along `tag` (torch.cumprod),
        /// shape-preserving; serial per row (the `cumsum` determinism
        /// contract). Differentiable: zero-free rows use the O(n)
        /// reverse-scan closed form; rows containing a zero fall back to
        /// the exact division-free O(n²) expansion (torch semantics).
        pub fn cumprod(self: *const Self, ctx: *ExecContext, comptime tag: Tag) !Self {
            const scan_axis = comptime axis(tag);
            var value = try ctx.cumprodAxisRank(tag_rank, self.asRawTensor(), scan_axis);
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad(), CumprodBackward(tags, scan_axis), .{ ctx.allocator, self.grad_state, &self.value, &value });
        }

        pub fn sumAll(self: *const Self, ctx: *ExecContext) !Tensor(.{}) {
            var value = try ctx.sum(self.asRawTensor());
            errdefer value.deinit();
            return finishOp(.{}, ctx, value, self.requiresGrad(), SumBackward(tags, .{}), .{ ctx.allocator, self.grad_state, &self.value });
        }

        pub fn sumMany(self: *const Self, ctx: *ExecContext, comptime reduce_tags_spec: anytype) !Tensor(removeTags(tags, normalizeTags(reduce_tags_spec))) {
            const reduce_tags = normalizeTags(reduce_tags_spec);
            const result_tags = removeTags(tags, reduce_tags);
            var value = try tag_ops.sumManyTensor(tags, self.asRawTensor(), ctx, reduce_tags);
            errdefer value.deinit();
            return finishOp(result_tags, ctx, value, self.requiresGrad(), SumBackward(tags, result_tags), .{ ctx.allocator, self.grad_state, &self.value });
        }
    };
}
