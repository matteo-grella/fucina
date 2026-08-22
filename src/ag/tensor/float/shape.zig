//! f32 tensor methods: views, reshapes, slicing, concat/stack, padding. A mixin over the ag
//! FloatTensor struct; aliased back onto it in ../../tensor.zig.

const std = @import("std");
const tensor_mod = @import("../../../tensor.zig");
const exec_mod = @import("../../../exec.zig");
const tag_ops = @import("../../../tag_ops.zig");
const control = @import("../../control.zig");
const core = @import("../../core.zig");
const tags_mod = @import("../../../tags.zig");
const backward = @import("../../backward.zig");

const RawTensor = tensor_mod.Tensor;
const TensorError = tensor_mod.TensorError;
const ExecContext = exec_mod.ExecContext;
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
const insertTagAt = tags_mod.insertTagAt;
const splitTags = tags_mod.splitTags;
const mergeTags = tags_mod.mergeTags;
const broadcastTensorTo = tag_ops.broadcastTensorTo;
const IdentityBackward = backward.IdentityBackward;
const BroadcastBackward = backward.BroadcastBackward;
const NarrowBackward = backward.NarrowBackward;
const ConcatBackward = backward.ConcatBackward;
const PadBackward = backward.PadBackward;
const ReshapeBackward = backward.ReshapeBackward;
const AxisViewBackward = backward.AxisViewBackward;
const StridedViewBackward = backward.StridedViewBackward;

pub fn Ops(comptime Self: type) type {
    return struct {
        const tags = Self.axis_tags;
        const tag_rank = Self.tag_count;
        const tensor_rank = Self.tensor_rank;
        const tag_count = Self.tag_count;
        const axis = Self.axis;
        const ag_tensor = Self.ag_root;
        const Tensor = ag_tensor.Tensor;
        const SliceRange = ag_tensor.SliceRange;
        const concat_inline_inputs = ag_tensor.concat_inline_inputs;
        const plumbing = @import("../plumbing.zig").Mod(ag_tensor);
        const finishOp = plumbing.finishOp;
        const requireScopeForComposedGrad = plumbing.requireScopeForComposedGrad;
        const finishNoGrad = plumbing.finishNoGrad;
        const axisViewTensor = plumbing.axisViewTensor;
        const padding2dValues = plumbing.padding2dValues;

        pub fn materialize(self: *const Self, ctx: *ExecContext) !Self {
            var value = try ctx.materialize(self.asRawTensor());
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad(), StridedViewBackward(tags, tags), .{ ctx.allocator, self.grad_state, &self.value, &value });
        }

        /// Borrow-if-contiguous materialize: an already-contiguous tensor
        /// returns a zero-copy retained view of the same storage (linked to
        /// the graph through an identity backward — `GradState` is
        /// single-owner, so the handle carries its own state rather than
        /// aliasing `self`'s); a strided view returns `materialize(ctx)`.
        /// Either way the result is owned by the caller (always `deinit` it;
        /// refcounted storage keeps the view case safe past the source's
        /// deinit), contiguous, and safe for `data`/`dataConst` access
        /// (`data` still rejects grad-carrying tensors — the autograd
        /// mutation invariant). The torch.contiguous aliasing caveat
        /// carries over: the already-contiguous case ALIASES `self`'s bytes
        /// (in-place mutation of either is visible through both) while the
        /// strided case is an independent snapshot — treat the result as
        /// read-only where the distinction matters, or use `materialize`
        /// when a guaranteed copy is wanted.
        pub fn contiguous(self: *const Self, ctx: *ExecContext) !Self {
            if (!self.isContiguous()) return self.materialize(ctx);
            var value = try self.value.cloneView();
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad(), IdentityBackward(tags), .{ ctx.allocator, self.grad_state });
        }

        pub fn withTags(self: *const Self, ctx: *ExecContext, comptime new_tags_spec: anytype) !Tensor(normalizeTags(new_tags_spec)) {
            const new_tags = normalizeTags(new_tags_spec);
            comptime {
                validateUniqueTags(new_tags);
                if (new_tags.len != tag_rank) @compileError("withTags requires the same rank");
            }
            return axisView(self, ctx, identityAxes(tag_rank), new_tags);
        }

        /// No-copy view with an explicit raw shape/stride layout and a new public
        /// tag set. Use this for audited structural views that cannot be expressed
        /// as pure tag permutations, insertions, or squeezes.
        pub fn viewWithStrides(
            self: *const Self,
            ctx: *ExecContext,
            comptime new_tags_spec: anytype,
            raw_shape: [rawRank(normalizeTags(new_tags_spec).len)]usize,
            raw_strides: [rawRank(normalizeTags(new_tags_spec).len)]usize,
        ) !Tensor(normalizeTags(new_tags_spec)) {
            const new_tags = normalizeTags(new_tags_spec);
            comptime validateUniqueTags(new_tags);
            var value = try self.value.viewWithStrides(raw_shape[0..], raw_strides[0..]);
            errdefer value.deinit();
            return finishOp(new_tags, ctx, value, self.requiresGrad(), StridedViewBackward(tags, new_tags), .{ ctx.allocator, self.grad_state, &self.value, &value });
        }

        pub fn alignTo(self: *const Self, ctx: *ExecContext, comptime target_tags_spec: anytype) !Tensor(normalizeTags(target_tags_spec)) {
            const target_tags = normalizeTags(target_tags_spec);
            return axisView(self, ctx, alignAxes(tags, target_tags), target_tags);
        }

        pub fn permuteTo(self: *const Self, ctx: *ExecContext, comptime target_tags_spec: anytype) !Tensor(normalizeTags(target_tags_spec)) {
            const target_tags = normalizeTags(target_tags_spec);
            comptime validateSameTagSet(tags, target_tags);
            return axisView(self, ctx, alignAxes(tags, target_tags), target_tags);
        }

        pub fn transpose(self: *const Self, ctx: *ExecContext, comptime target_tags_spec: anytype) !Tensor(normalizeTags(target_tags_spec)) {
            return self.permuteTo(ctx, target_tags_spec);
        }

        pub fn insertAxis(self: *const Self, ctx: *ExecContext, comptime tag: Tag, comptime axis_index: usize) !Tensor(insertTagAt(tags, tag, axis_index)) {
            const result_tags = insertTagAt(tags, tag, axis_index);
            return axisView(self, ctx, insertAxes(tag_rank, axis_index), result_tags);
        }

        pub fn squeeze(self: *const Self, ctx: *ExecContext, comptime tag: Tag) !Tensor(removeTag(tags, tag)) {
            const result_tags = removeTag(tags, tag);
            const axis_index = comptime tagIndexOrCompileError(tags, tag);
            if (self.asRawTensor().shape.at(axis_index) != 1) return TensorError.InvalidShape;
            return axisView(self, ctx, squeezeAxes(tag_rank, axis_index), result_tags);
        }

        pub fn split(
            self: *const Self,
            ctx: *ExecContext,
            comptime tag: Tag,
            comptime split_tags_spec: anytype,
            split_shape: [normalizeTags(split_tags_spec).len]usize,
        ) !Tensor(splitTags(tags, tag, normalizeTags(split_tags_spec))) {
            const split_tags = normalizeTags(split_tags_spec);
            const result_tags = splitTags(tags, tag, split_tags);
            var value = try tag_ops.splitAxisView(tags, self.asRawTensor(), tag, split_tags, split_shape);
            errdefer value.deinit();
            return finishOp(result_tags, ctx, value, self.requiresGrad(), StridedViewBackward(tags, result_tags), .{ ctx.allocator, self.grad_state, &self.value, &value });
        }

        pub fn merge(self: *const Self, ctx: *ExecContext, comptime out_tag: Tag, comptime merge_tags_spec: anytype) !Tensor(mergeTags(tags, out_tag, normalizeTags(merge_tags_spec))) {
            const merge_tags = normalizeTags(merge_tags_spec);
            const result_tags = mergeTags(tags, out_tag, merge_tags);
            var value = try tag_ops.mergeAxesView(tags, self.asRawTensor(), out_tag, merge_tags);
            errdefer value.deinit();
            return finishOp(result_tags, ctx, value, self.requiresGrad(), StridedViewBackward(tags, result_tags), .{ ctx.allocator, self.grad_state, &self.value, &value });
        }

        /// Arbitrary row-major reinterpretation to `new_tags_spec` /
        /// `new_shape` (torch.reshape): the element count must match
        /// (`InvalidShape` otherwise). View-or-materialize like torch — a
        /// contiguous source stays a zero-copy view; a non-contiguous one
        /// materializes first (the `flatten` rule). Composed flatten →
        /// split, so gradients come from the existing exact view records;
        /// when the target rank is > 1 and gradients are tracked this
        /// requires an active exec scope (see `nllLoss`); a rank-1 target
        /// degenerates to plain `flatten` (no scope needed).
        pub fn reshape(
            self: *const Self,
            ctx: *ExecContext,
            comptime new_tags_spec: anytype,
            new_shape: [normalizeTags(new_tags_spec).len]usize,
        ) !Tensor(normalizeTags(new_tags_spec)) {
            const new_tags = comptime normalizeTags(new_tags_spec);
            if (comptime new_tags.len == 1) {
                if (self.asRawTensor().len() != new_shape[0]) return TensorError.InvalidShape;
                return self.flatten(ctx, new_tags[0]);
            }
            try requireScopeForComposedGrad(ctx, self.requiresGrad());
            var flat = try self.flatten(ctx, new_tags[0]);
            defer flat.deinit();
            return flat.split(ctx, new_tags[0], new_tags_spec, new_shape);
        }

        pub fn broadcastTo(
            self: *const Self,
            ctx: *ExecContext,
            comptime target_tags_spec: anytype,
            target_shape: [normalizeTags(target_tags_spec).len]usize,
        ) !Tensor(normalizeTags(target_tags_spec)) {
            const target_tags = normalizeTags(target_tags_spec);
            var value = try broadcastTensorTo(tags, self.asRawTensor(), target_tags, target_shape);
            errdefer value.deinit();
            return finishOp(target_tags, ctx, value, self.requiresGrad(), BroadcastBackward(tags, target_tags), .{ ctx.allocator, self.grad_state, &self.value });
        }

        pub fn flatten(self: *const Self, ctx: *ExecContext, comptime out_tag: Tag) !Tensor(.{out_tag}) {
            var value = try tag_ops.flattenTensor(ctx, self.asRawTensor());
            errdefer value.deinit();
            return finishOp(.{out_tag}, ctx, value, self.requiresGrad(), ReshapeBackward, .{ ctx.allocator, self.grad_state, &self.value });
        }

        /// Reverse the order of `tag` (torch.flip on one dim): a gather with
        /// the reversed index permutation, so the value is a copy and
        /// GatherBackward routes the gradient exactly (a permutation
        /// scatters 1-to-1).
        pub fn flip(self: *const Self, ctx: *ExecContext, comptime tag: Tag) !Self {
            const n = self.asRawTensor().shape.at(axis(tag));
            const indices = try ctx.allocator.alloc(usize, n);
            defer ctx.allocator.free(indices);
            for (indices, 0..) |*index, i| index.* = n - 1 - i;
            return self.gather(ctx, tag, indices, tag);
        }

        /// Rotate `tag` by `shift` positions (torch.roll on one dim): the
        /// element at index i moves to index `(i + shift) mod n`; elements
        /// shifted past the end re-enter at the front. Negative shifts roll
        /// the other way. Implemented as a gather with the rotated index
        /// permutation (exact gradient, like `flip`).
        pub fn roll(self: *const Self, ctx: *ExecContext, comptime tag: Tag, shift: isize) !Self {
            const n = self.asRawTensor().shape.at(axis(tag));
            const indices = try ctx.allocator.alloc(usize, n);
            defer ctx.allocator.free(indices);
            // out[i] = x[(i - shift) mod n]; s = shift mod n in [0, n).
            const s: usize = @intCast(@mod(shift, @as(isize, @intCast(n))));
            for (indices, 0..) |*index, i| index.* = (i + n - s) % n;
            return self.gather(ctx, tag, indices, tag);
        }

        /// `roll` with one shift per section: every section (the sub-vector
        /// obtained by fixing all axes except `tag`) rotates by its own
        /// offset, with `roll`'s sign convention
        /// (`out[..., j, ...] = self[..., (j - shift) mod n, ...]`).
        /// `offsets` holds one shift per section, indexed row-major over the
        /// remaining axes in `self`'s tag order (host-side control data,
        /// like `gather` indices) — `offsets.len` must equal
        /// `numel / dim(tag)` or the call errors with `InvalidShape`; for a
        /// rank-1 tensor that is a single element and `rollBy` matches
        /// `roll`. A per-section permutation, composed flatten + gather +
        /// split, so the gradient is exact (the inverse per-section roll).
        /// When gradients are tracked this requires an active exec scope
        /// (see `maskedSelect`).
        pub fn rollBy(self: *const Self, ctx: *ExecContext, comptime tag: Tag, offsets: []const isize) !Self {
            const roll_axis = comptime axis(tag);
            try requireScopeForComposedGrad(ctx, self.requiresGrad());
            const raw = self.asRawTensor();
            const n = raw.shape.at(roll_axis);
            var inner: usize = 1;
            inline for (roll_axis + 1..tensor_rank) |i| inner *= raw.shape.at(i);
            const total = raw.len();
            const sections = total / n;
            if (offsets.len != sections) return TensorError.InvalidShape;

            // Normalize each section's shift once: s in [0, n) with
            // out[j] = x[(j - s) mod n], matching `roll`.
            const normalized = try ctx.allocator.alloc(usize, sections);
            defer ctx.allocator.free(normalized);
            for (normalized, offsets) |*s, shift| s.* = @intCast(@mod(shift, @as(isize, @intCast(n))));

            const indices = try ctx.allocator.alloc(usize, total);
            defer ctx.allocator.free(indices);
            const section_stride = n * inner;
            for (indices, 0..) |*index, i| {
                const outer = i / section_stride;
                const j = (i % section_stride) / inner;
                const inner_pos = i % inner;
                const s = normalized[outer * inner + inner_pos];
                index.* = outer * section_stride + ((j + n - s) % n) * inner + inner_pos;
            }

            var flat = try self.flatten(ctx, tag);
            defer flat.deinit();
            var rolled = try flat.gather(ctx, tag, indices, tag);
            defer rolled.deinit();
            return rolled.split(ctx, tag, tags, self.shape());
        }

        /// Non-circular `rollBy`: same per-section offsets and sign
        /// convention, but positions shifted in from outside the axis hold
        /// the constant `fill` instead of wrapping
        /// (`out[..., j, ...] = self[..., j - shift, ...]` when in bounds,
        /// else `fill`). `fill` is a constant and receives no gradient;
        /// source positions shifted out of the axis receive zero gradient
        /// (composed gather + maskedFill — the fill mask zeroes their
        /// upstream gradient before the gather scatters it back). Same
        /// `offsets` layout, `InvalidShape`, and exec-scope rules as
        /// `rollBy`.
        pub fn shiftBy(self: *const Self, ctx: *ExecContext, comptime tag: Tag, offsets: []const isize, fill: f32) !Self {
            const shift_axis = comptime axis(tag);
            try requireScopeForComposedGrad(ctx, self.requiresGrad());
            const raw = self.asRawTensor();
            const n = raw.shape.at(shift_axis);
            var inner: usize = 1;
            inline for (shift_axis + 1..tensor_rank) |i| inner *= raw.shape.at(i);
            const total = raw.len();
            const sections = total / n;
            if (offsets.len != sections) return TensorError.InvalidShape;

            const indices = try ctx.allocator.alloc(usize, total);
            defer ctx.allocator.free(indices);
            var mask = try Self.empty(ctx, self.shape());
            defer mask.deinit();
            const fill_mask = try mask.data();
            const section_stride = n * inner;
            const n_signed: isize = @intCast(n);
            for (indices, fill_mask, 0..) |*index, *fm, i| {
                const outer = i / section_stride;
                const j = (i % section_stride) / inner;
                const inner_pos = i % inner;
                const shift = offsets[outer * inner + inner_pos];
                const src_j = @as(isize, @intCast(j)) - shift;
                const section_base = outer * section_stride + inner_pos;
                if (src_j >= 0 and src_j < n_signed) {
                    index.* = section_base + @as(usize, @intCast(src_j)) * inner;
                    fm.* = 0;
                } else {
                    // Placeholder source; maskedFill overwrites the value and
                    // zeroes its gradient contribution.
                    index.* = section_base;
                    fm.* = 1;
                }
            }

            var flat = try self.flatten(ctx, tag);
            defer flat.deinit();
            var shifted_flat = try flat.gather(ctx, tag, indices, tag);
            defer shifted_flat.deinit();
            var shifted = try shifted_flat.split(ctx, tag, tags, self.shape());
            defer shifted.deinit();
            return shifted.maskedFill(ctx, mask, fill);
        }

        pub fn narrow(self: *const Self, ctx: *ExecContext, comptime tag: Tag, start: usize, length: usize) !Self {
            const slice_axis = comptime axis(tag);
            var value = try ctx.narrowAxisRank(tag_rank, self.asRawTensor(), slice_axis, start, length);
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad(), NarrowBackward(tags, slice_axis), .{ ctx.allocator, self.grad_state, &self.value, start });
        }

        /// Select one position of `tag` and remove the axis (torch.select /
        /// `x[i]`): the single-slice sibling of `unbindInto`. `index`
        /// counts from the end when negative (torch convention); out of
        /// range errors with `IndexOutOfBounds`. Composed narrow → squeeze,
        /// so the value is a zero-copy view aliasing the selected row and
        /// the gradient is the exact scatter (every unselected position
        /// receives zero). When gradients are tracked this requires an
        /// active exec scope (see `nllLoss`); errors with
        /// `ActiveExecScopeRequired` otherwise.
        pub fn select(self: *const Self, ctx: *ExecContext, comptime tag: Tag, index: isize) !Tensor(removeTag(tags, tag)) {
            try requireScopeForComposedGrad(ctx, self.requiresGrad());
            const n: isize = @intCast(self.asRawTensor().shape.at(comptime axis(tag)));
            const shifted = if (index < 0) index +| n else index;
            if (shifted < 0 or shifted >= n) return TensorError.IndexOutOfBounds;
            var row = try self.narrow(ctx, tag, @intCast(shifted), 1);
            defer row.deinit();
            return row.squeeze(ctx, tag);
        }

        /// `narrow` with a step (torch basic slicing `x[start::step]` along
        /// one axis): `length` elements at `start, start+step, …`.
        /// `step == 0`, `length == 0` (zero-size tensors are not
        /// representable), or a last element at or past `dim(tag)` error
        /// with `InvalidShape`. On a no-grad tensor the result is a
        /// zero-copy strided **view** (retains and aliases the source
        /// buffer, like `narrow`); under gradients it lowers to `gather`
        /// over the stepped indices — a copy with the exact scatter-add
        /// record, the `flip`/`roll` precedent — so skipped positions
        /// receive zero gradient.
        pub fn sliceStep(self: *const Self, ctx: *ExecContext, comptime tag: Tag, start: usize, length: usize, step: usize) !Self {
            const slice_axis = comptime axis(tag);
            const raw = self.asRawTensor();
            const axis_dim = raw.shape.at(slice_axis);
            if (step == 0 or length == 0) return TensorError.InvalidShape;
            if (start >= axis_dim or start + (length - 1) * step >= axis_dim) return TensorError.InvalidShape;
            if (self.requiresGrad()) {
                const indices = try ctx.allocator.alloc(usize, length);
                defer ctx.allocator.free(indices);
                for (indices, 0..) |*index, i| index.* = start + i * step;
                return self.gather(ctx, tag, indices, tag);
            }
            var new_shape: [tensor_rank]usize = undefined;
            var new_strides: [tensor_rank]usize = undefined;
            inline for (0..tensor_rank) |i| {
                new_shape[i] = raw.shape.at(i);
                new_strides[i] = raw.strides.at(i);
            }
            new_shape[slice_axis] = length;
            new_strides[slice_axis] = raw.strides.at(slice_axis) * step;
            var value = try self.value.viewWithStridesOffset(&new_shape, &new_strides, start * raw.strides.at(slice_axis));
            errdefer value.deinit();
            return finishNoGrad(tags, ctx, value);
        }

        /// Multi-axis basic slicing (torch/numpy `x[1:-1, ::2]`, positive
        /// steps): `spec` is a struct literal naming the tags to slice —
        /// e.g. `.{ .h = .{ .start = 1, .end = -1 }, .w = .{ .step = 2 } }`
        /// — each field a `fucina.SliceRange`-shaped range (`start`/`end`/
        /// `step`, each optional); axes not named pass through whole, and a
        /// field naming a tag not on the tensor is a compile error. torch
        /// bounds semantics: negatives count from the end, `end = null`
        /// means the axis dim, out-of-range bounds clamp; `step == 0` or an
        /// empty result error with `InvalidShape` (zero-size tensors are
        /// not representable). Negative steps are deliberately unsupported
        /// (torch rejects them in basic indexing too; strides are unsigned,
        /// so a reversed view is not representable — compose `flip`).
        /// Lowered to per-axis `narrow`/`sliceStep` in tag order: with
        /// step-1 ranges the no-grad value is a zero-copy view and the
        /// gradient is the exact per-axis scatter; stepped axes follow the
        /// `sliceStep` contract (view no-grad, gather copy under
        /// gradients). Slicing more than one axis with gradients tracked
        /// requires an active exec scope (see `nllLoss`); errors with
        /// `ActiveExecScopeRequired` otherwise.
        pub fn slice(self: *const Self, ctx: *ExecContext, spec: anytype) !Self {
            const Spec = @TypeOf(spec);
            comptime {
                if (@typeInfo(Spec) != .@"struct")
                    @compileError("slice expects a per-tag range struct, e.g. .{ .h = .{ .start = 1 } }");
                const fields = @typeInfo(Spec).@"struct".fields;
                if (fields.len == 0) @compileError("slice requires at least one tag range");
                for (fields) |field| {
                    var known = false;
                    for (tags) |t| {
                        if (std.mem.eql(u8, @tagName(t), field.name)) known = true;
                    }
                    if (!known) @compileError("slice range names a tag not on this tensor: ." ++ field.name);
                }
            }
            if (comptime @typeInfo(Spec).@"struct".fields.len > 1)
                try requireScopeForComposedGrad(ctx, self.requiresGrad());
            var current: ?Self = null;
            errdefer if (current) |*c| c.deinit();
            inline for (tags) |tag| {
                if (comptime @hasField(Spec, @tagName(tag))) {
                    const range = sliceRangeOf(@field(spec, @tagName(tag)));
                    if (range.step == 0) return TensorError.InvalidShape;
                    const source: *const Self = if (current) |*c| c else self;
                    const dim_len = source.asRawTensor().shape.at(comptime axis(tag));
                    const start = clampSliceBound(range.start, dim_len);
                    const stop = if (range.end) |e| clampSliceBound(e, dim_len) else dim_len;
                    if (stop <= start) return TensorError.InvalidShape;
                    const next = if (range.step == 1)
                        try source.narrow(ctx, tag, start, stop - start)
                    else
                        try source.sliceStep(ctx, tag, start, (stop - start - 1) / range.step + 1, range.step);
                    if (current) |*c| c.deinit();
                    current = next;
                }
            }
            return current.?;
        }

        /// Normalize a `slice` range literal to `SliceRange`: unknown
        /// fields are compile errors, omitted fields take the defaults.
        fn sliceRangeOf(range_spec: anytype) SliceRange {
            const F = @TypeOf(range_spec);
            if (comptime F == SliceRange) return range_spec;
            comptime {
                if (@typeInfo(F) != .@"struct")
                    @compileError("slice range must be a struct literal with start/end/step fields");
                for (@typeInfo(F).@"struct".fields) |field| {
                    if (!std.mem.eql(u8, field.name, "start") and
                        !std.mem.eql(u8, field.name, "end") and
                        !std.mem.eql(u8, field.name, "step"))
                        @compileError("unknown slice range field ." ++ field.name ++ " (start/end/step)");
                }
            }
            var out: SliceRange = .{};
            if (comptime @hasField(F, "start")) out.start = range_spec.start;
            if (comptime @hasField(F, "end")) out.end = range_spec.end;
            if (comptime @hasField(F, "step")) out.step = range_spec.step;
            return out;
        }

        /// torch bound normalization: negatives count from the end of the
        /// axis, then clamp to `[0, dim_len]`.
        fn clampSliceBound(bound: isize, dim_len: usize) usize {
            const n: isize = @intCast(dim_len);
            const shifted = if (bound < 0) bound +| n else bound;
            if (shifted < 0) return 0;
            return @min(@as(usize, @intCast(shifted)), dim_len);
        }

        /// Main diagonal over the (`tag_a`, `tag_b`) plane (torch.diagonal,
        /// offset 0): a zero-copy strided **view** of length
        /// `min(dim(tag_a), dim(tag_b))` — element i is `self[.., i, .., i,
        /// ..]`. Both tags are removed and the diagonal axis is appended
        /// LAST as `out_tag` (the torch axis order). Works at any rank
        /// carrying both tags; differentiable (strided-view scatter —
        /// off-diagonal positions receive zero gradient).
        pub fn diagonal(self: *const Self, ctx: *ExecContext, comptime tag_a: Tag, comptime tag_b: Tag, comptime out_tag: Tag) !Tensor(insertTagAt(removeTag(removeTag(tags, tag_a), tag_b), out_tag, removeTag(removeTag(tags, tag_a), tag_b).len)) {
            comptime if (tag_a == tag_b) @compileError("diagonal requires two distinct tags");
            const base_tags = comptime removeTag(removeTag(tags, tag_a), tag_b);
            const result_tags = comptime insertTagAt(base_tags, out_tag, base_tags.len);
            const axis_a = comptime axis(tag_a);
            const axis_b = comptime axis(tag_b);
            const raw = self.asRawTensor();
            const diag_len = @min(raw.shape.at(axis_a), raw.shape.at(axis_b));
            var new_shape: [tensor_rank - 1]usize = undefined;
            var new_strides: [tensor_rank - 1]usize = undefined;
            var write: usize = 0;
            inline for (0..tensor_rank) |i| {
                if (i != axis_a and i != axis_b) {
                    new_shape[write] = raw.shape.at(i);
                    new_strides[write] = raw.strides.at(i);
                    write += 1;
                }
            }
            new_shape[tensor_rank - 2] = diag_len;
            new_strides[tensor_rank - 2] = raw.strides.at(axis_a) + raw.strides.at(axis_b);
            var value = try self.value.viewWithStridesOffset(&new_shape, &new_strides, 0);
            errdefer value.deinit();
            return finishOp(result_tags, ctx, value, self.requiresGrad(), StridedViewBackward(tags, result_tags), .{ ctx.allocator, self.grad_state, &self.value, &value });
        }

        /// Sum of the main diagonal over the (`tag_a`, `tag_b`) plane
        /// (torch.trace generalized to named axes): composed diagonal →
        /// sum, so the gradient is the exact identity-matrix scatter. When
        /// gradients are tracked this requires an active exec scope (see
        /// `nllLoss`); errors with `ActiveExecScopeRequired` otherwise.
        pub fn trace(self: *const Self, ctx: *ExecContext, comptime tag_a: Tag, comptime tag_b: Tag) !Tensor(removeTag(removeTag(tags, tag_a), tag_b)) {
            try requireScopeForComposedGrad(ctx, self.requiresGrad());
            var diag_view = try self.diagonal(ctx, tag_a, tag_b, tag_a);
            defer diag_view.deinit();
            return diag_view.sum(ctx, tag_a, .{});
        }

        /// Embed a rank-1 tensor as the main diagonal of an `[n, n]` matrix
        /// (torch.diag on a vector): zeros elsewhere. `out_tags_spec` names
        /// the two result axes. Composed zeros → setRows (flat diagonal
        /// positions) → reshape, so the gradient is the exact
        /// diagonal-extract; scope-required under gradients (see
        /// `nllLoss`).
        pub fn diag(self: *const Self, ctx: *ExecContext, comptime out_tags_spec: anytype) !Tensor(normalizeTags(out_tags_spec)) {
            comptime if (tag_count != 1) @compileError("diag embeds a rank-1 tensor as a matrix diagonal; for extraction use diagonal");
            const out_tags = comptime normalizeTags(out_tags_spec);
            comptime if (out_tags.len != 2) @compileError("diag builds a rank-2 [n, n] tensor: pass exactly two tags");
            try requireScopeForComposedGrad(ctx, self.requiresGrad());
            const n = self.asRawTensor().shape.at(0);
            var base = try Self.zeros(ctx, .{n * n});
            defer base.deinit();
            const indices = try ctx.allocator.alloc(usize, n);
            defer ctx.allocator.free(indices);
            for (indices, 0..) |*index, i| index.* = i * (n + 1);
            var filled = try base.setRows(ctx, tags[0], indices, self);
            defer filled.deinit();
            return filled.reshape(ctx, out_tags_spec, .{ n, n });
        }

        /// Keep the band `i - j <= lower and j - i <= upper` of the
        /// (`row_tag`, `col_tag`) plane and zero everything outside
        /// (tf.linalg.band_part with signed, nullable bounds; `tril`/`triu`
        /// are the triangular special cases). A null bound is unbounded on
        /// that side. Works at any rank carrying both tags — remaining
        /// axes pass through. Implemented as a constant 0/1 band-plane
        /// multiply (the `bandMask` keep-set cast to f32), so the VJP is
        /// the exact same-band mask on the upstream gradient.
        /// Differentiable in `self`.
        pub fn bandPart(self: *const Self, ctx: *ExecContext, comptime row_tag: Tag, comptime col_tag: Tag, lower: ?i64, upper: ?i64) !Self {
            comptime if (row_tag == col_tag) @compileError("bandPart requires two distinct tags");
            const MaskT = Tensor(.{ .dtype = .bool, .tags = .{ row_tag, col_tag } });
            var keep = try MaskT.bandMask(ctx, .{ self.dim(row_tag), self.dim(col_tag) }, lower, upper);
            defer keep.deinit();
            var plane = try keep.to(ctx, .f32);
            defer plane.deinit();
            return self.mul(ctx, &plane);
        }

        /// Lower-triangular part over the (`row_tag`, `col_tag`) plane
        /// (torch.tril with a diagonal offset, batched over any remaining
        /// tags): elements with `col - row > offset` become 0.
        /// `bandPart(null, offset)`; differentiable in `self` with the
        /// exact triangle-mask VJP.
        pub fn tril(self: *const Self, ctx: *ExecContext, comptime row_tag: Tag, comptime col_tag: Tag, offset: i64) !Self {
            return self.bandPart(ctx, row_tag, col_tag, null, offset);
        }

        /// Upper-triangular part over the (`row_tag`, `col_tag`) plane
        /// (torch.triu with a diagonal offset): elements with
        /// `col - row < offset` become 0. `bandPart(-offset, null)`; see
        /// `tril`.
        pub fn triu(self: *const Self, ctx: *ExecContext, comptime row_tag: Tag, comptime col_tag: Tag, offset: i64) !Self {
            return self.bandPart(ctx, row_tag, col_tag, -offset, null);
        }

        /// Embed the `vec_tag` axis as the main diagonal of a square
        /// matrix plane (torch.diag_embed generalized to named axes):
        /// `out[…, i, j] = self[…, i] · [i == j]`, with `vec_tag` renamed
        /// to `out_tags[0]` (the row axis) in place and `out_tags[1]` (the
        /// column axis) appended LAST. Works at any rank carrying
        /// `vec_tag`; the remaining axes pass through. Composed rename →
        /// broadcast-multiply against an `eye` constant, so the gradient
        /// is the exact diagonal extraction; scope-required under
        /// gradients (see `nllLoss`).
        pub fn diagEmbed(
            self: *const Self,
            ctx: *ExecContext,
            comptime vec_tag: Tag,
            comptime out_tags_spec: anytype,
        ) !Tensor(pointwiseResultTags(replaceTag(tags, vec_tag, normalizeTags(out_tags_spec)[0]), normalizeTags(out_tags_spec))) {
            const out_tags = comptime normalizeTags(out_tags_spec);
            comptime {
                if (out_tags.len != 2) @compileError("diagEmbed names exactly two result axes: pass .{ row_tag, col_tag }");
                if (out_tags[0] == out_tags[1]) @compileError("diagEmbed requires two distinct result tags");
                const rest = removeTag(tags, vec_tag);
                if (tagIndex(rest, out_tags[0]) != null or tagIndex(rest, out_tags[1]) != null)
                    @compileError("diagEmbed result tags must not collide with the remaining input tags");
            }
            try requireScopeForComposedGrad(ctx, self.requiresGrad());
            var renamed = try self.withTags(ctx, replaceTag(tags, vec_tag, out_tags[0]));
            defer renamed.deinit();
            var identity = try Tensor(.{ out_tags[0], out_tags[1] }).eye(ctx, self.dim(vec_tag));
            defer identity.deinit();
            return renamed.mul(ctx, &identity);
        }

        /// Constant padding along `tag` (torch F.pad, mode='constant', one
        /// dim): the axis grows by `before + after`, the body sits at offset
        /// `before`, pad positions hold `fill`. Differentiable: the gradient
        /// is the narrow of the upstream gradient at offset `before` (pad
        /// positions are constants and drop their gradient).
        pub fn pad(self: *const Self, ctx: *ExecContext, comptime tag: Tag, before: usize, after: usize, fill: f32) !Self {
            const pad_axis = comptime axis(tag);
            var value = try ctx.padAxisRank(tag_rank, self.asRawTensor(), pad_axis, before, after, fill);
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad(), PadBackward(tags, pad_axis), .{ ctx.allocator, self.grad_state, &self.value, before });
        }

        /// Zero-pad the two named axes: `constantPad2d` with fill 0.
        pub fn zeroPad2d(self: *const Self, ctx: *ExecContext, comptime h_tag: Tag, comptime w_tag: Tag, padding: anytype) !Self {
            return self.constantPad2d(ctx, h_tag, w_tag, padding, 0);
        }

        /// Constant-pad two named axes in one call. `padding` is either an
        /// integer, padding all four sides by that amount, or a
        /// 4-tuple/array `(left, right, top, bottom)`: left/right grow
        /// `w_tag`, top/bottom grow `h_tag`. A NEGATIVE entry crops that
        /// side instead of padding it; cropping an axis to zero size or
        /// below errors with `InvalidShape` (zero-size tensors are not
        /// representable). Any rank carrying both tags works —
        /// the remaining axes pass through. The result is always an owned
        /// regular tensor: an all-zero padding is the contiguous identity,
        /// and a crop-only result materializes rather than returning a
        /// strided view. Differentiable in `self`: pad positions are
        /// constants and drop their gradient, cropped source positions
        /// receive zero gradient — composed narrow + pad per axis, so the
        /// gradients come from the existing exact records. When gradients
        /// are tracked this requires an active exec scope (see
        /// `maskedSelect`), even for paddings that degenerate to fewer ops.
        pub fn constantPad2d(self: *const Self, ctx: *ExecContext, comptime h_tag: Tag, comptime w_tag: Tag, padding: anytype, fill: f32) !Self {
            comptime if (h_tag == w_tag) @compileError("constantPad2d: h_tag and w_tag must be distinct");
            try requireScopeForComposedGrad(ctx, self.requiresGrad());
            const p = padding2dValues(padding);

            var cur: Self = undefined;
            var owned = false;
            var last_was_pad = false;
            errdefer if (owned) cur.deinit();
            inline for (0..2) |axis_i| {
                const tag = comptime if (axis_i == 0) h_tag else w_tag;
                const before: isize = if (axis_i == 0) p[2] else p[0];
                const after: isize = if (axis_i == 0) p[3] else p[1];
                if (before < 0 or after < 0) {
                    const src: *const Self = if (owned) &cur else self;
                    const remaining = @as(isize, @intCast(src.dim(tag))) + @min(before, 0) + @min(after, 0);
                    if (remaining <= 0) return TensorError.InvalidShape;
                    const next = try src.narrow(ctx, tag, @intCast(@max(-before, 0)), @intCast(remaining));
                    if (owned) cur.deinit();
                    cur = next;
                    owned = true;
                    last_was_pad = false;
                }
                if (before > 0 or after > 0) {
                    const src: *const Self = if (owned) &cur else self;
                    const next = try src.pad(ctx, tag, @intCast(@max(before, 0)), @intCast(@max(after, 0)), fill);
                    if (owned) cur.deinit();
                    cur = next;
                    owned = true;
                    last_was_pad = true;
                }
            }
            // The result contract is a regular tensor in every case: an
            // all-zero padding is the contiguous identity, and a crop-only
            // result (narrow is a strided view) materializes so
            // `dataConst` works like any padded output.
            if (!owned) return self.contiguous(ctx);
            if (!last_was_pad) {
                const out = try cur.contiguous(ctx);
                cur.deinit();
                owned = false;
                return out;
            }
            return cur;
        }

        pub fn concat(self: *const Self, ctx: *ExecContext, comptime tag: Tag, others: []const *const Self) !Self {
            var any_grad = self.requiresGrad();
            for (others) |other| any_grad = any_grad or other.requiresGrad();

            const input_count = others.len + 1;
            var raw_inputs_stack: [concat_inline_inputs]*const RawTensor = undefined;
            const raw_inputs = if (input_count <= raw_inputs_stack.len)
                raw_inputs_stack[0..input_count]
            else
                try ctx.allocator.alloc(*const RawTensor, input_count);
            defer if (input_count > raw_inputs_stack.len) ctx.allocator.free(raw_inputs);
            raw_inputs[0] = self.asRawTensor();
            for (others, raw_inputs[1..]) |other, *raw| raw.* = other.asRawTensor();

            // Backward metadata is only materialized when finishOp will attach
            // a backward record (same gate finishOp itself applies); its
            // no-grad branch never reads create_args, so the empty slices are
            // never touched there. ConcatBackward copies both slices when it
            // is constructed, so stack-backed temporaries are safe on the grad
            // path.
            const track_grad = any_grad and control.isGradEnabled();
            var parents_stack: [concat_inline_inputs]?*GradState = undefined;
            var sizes_stack: [concat_inline_inputs]usize = undefined;
            var parents: []?*GradState = parents_stack[0..0];
            var sizes: []usize = sizes_stack[0..0];
            const metadata_on_heap = track_grad and input_count > parents_stack.len;
            defer if (metadata_on_heap) {
                ctx.allocator.free(parents);
                ctx.allocator.free(sizes);
            };
            if (track_grad) {
                if (metadata_on_heap) {
                    parents = try ctx.allocator.alloc(?*GradState, input_count);
                    sizes = try ctx.allocator.alloc(usize, input_count);
                } else {
                    parents = parents_stack[0..input_count];
                    sizes = sizes_stack[0..input_count];
                }
                parents[0] = self.grad_state;
                sizes[0] = self.asRawTensor().shape.at(axis(tag));
                for (others, parents[1..], sizes[1..]) |other, *parent, *size| {
                    parent.* = other.grad_state;
                    size.* = other.asRawTensor().shape.at(axis(tag));
                }
            }

            const concat_axis = comptime axis(tag);
            var value = try ctx.concatAxisRank(tag_rank, raw_inputs, concat_axis);
            errdefer value.deinit();
            return finishOp(tags, ctx, value, any_grad, ConcatBackward(tags, concat_axis), .{ ctx.allocator, parents, sizes });
        }

        /// Stack `self` and `others` along a NEW axis tagged `new_tag`
        /// inserted at `axis_index` (torch.stack): composed as insertAxis on
        /// every input + concat, so the result is differentiable in ALL
        /// inputs through the multi-parent ConcatBackward. When gradients
        /// are tracked this requires an active exec scope (the inserted-axis
        /// intermediates are function-local graph nodes — see `nllLoss`);
        /// errors with `ActiveExecScopeRequired` otherwise.
        pub fn stack(
            self: *const Self,
            ctx: *ExecContext,
            comptime new_tag: Tag,
            comptime axis_index: usize,
            others: []const *const Self,
        ) !Tensor(insertTagAt(tags, new_tag, axis_index)) {
            var any_grad = self.requiresGrad();
            for (others) |other| any_grad = any_grad or other.requiresGrad();
            try requireScopeForComposedGrad(ctx, any_grad);

            const Expanded = Tensor(insertTagAt(tags, new_tag, axis_index));
            var expanded = try ctx.allocator.alloc(Expanded, others.len + 1);
            defer ctx.allocator.free(expanded);
            var created: usize = 0;
            // The inserted-axis views are composition temporaries: deinit is
            // a no-op under an exec scope (scope-owned) and a real release
            // in the unscoped no-grad arm.
            defer for (expanded[0..created]) |*view| view.deinit();

            expanded[0] = try self.insertAxis(ctx, new_tag, axis_index);
            created = 1;
            for (others) |other| {
                expanded[created] = try other.insertAxis(ctx, new_tag, axis_index);
                created += 1;
            }

            var ptrs_stack: [concat_inline_inputs]*const Expanded = undefined;
            const ptrs = if (others.len <= ptrs_stack.len)
                ptrs_stack[0..others.len]
            else
                try ctx.allocator.alloc(*const Expanded, others.len);
            defer if (others.len > ptrs_stack.len) ctx.allocator.free(ptrs);
            for (ptrs, expanded[1..]) |*ptr, *view| ptr.* = view;
            return expanded[0].concat(ctx, new_tag, ptrs);
        }

        /// Unbind `tag` (torch.unbind): fill the caller-provided `out` slice
        /// with the `dim(tag)` slices of `self`, each with `tag` removed
        /// (composed narrow + squeeze; differentiable per entry). `out.len`
        /// must equal `dim(tag)`. The CALLER owns every filled tensor and
        /// deinits each (under an exec scope they are scope-owned borrows and
        /// deinit is a no-op); on error, entries filled so far have already
        /// been released. When gradients are tracked this requires an active
        /// exec scope (see `nllLoss`); errors with `ActiveExecScopeRequired`
        /// otherwise.
        pub fn unbindInto(self: *const Self, ctx: *ExecContext, comptime tag: Tag, out: []Tensor(removeTag(tags, tag))) !void {
            try requireScopeForComposedGrad(ctx, self.requiresGrad());
            if (out.len != self.asRawTensor().shape.at(axis(tag))) return TensorError.InvalidShape;
            var filled: usize = 0;
            errdefer for (out[0..filled]) |*entry| entry.deinit();
            for (out, 0..) |*entry, i| {
                var sliced = try self.narrow(ctx, tag, i, 1);
                defer sliced.deinit();
                entry.* = try sliced.squeeze(ctx, tag);
                filled += 1;
            }
        }

        /// Repeat (tile) `tag` n times (torch.repeat on one dim): the axis
        /// length becomes `n·dim(tag)` with n back-to-back copies of `self`.
        /// `n == 1` returns a zero-copy identity view; `n > 1` is a concat
        /// of `self` with itself (one multi-parent node — gradients from all
        /// n copies accumulate into `self`). Errors with `InvalidShape` for
        /// `n == 0` (zero-size tensors are not representable).
        pub fn repeatAxis(self: *const Self, ctx: *ExecContext, comptime tag: Tag, n: usize) !Self {
            if (n == 0) return TensorError.InvalidShape;
            if (n == 1) return self.withTags(ctx, tags);
            const ptrs = try ctx.allocator.alloc(*const Self, n - 1);
            defer ctx.allocator.free(ptrs);
            for (ptrs) |*ptr| ptr.* = self;
            return self.concat(ctx, tag, ptrs);
        }

        fn axisView(self: *const Self, ctx: *ExecContext, comptime axes: anytype, comptime target_tags: anytype) !Tensor(target_tags) {
            var value = try axisViewTensor(self.asRawTensor(), axes, target_tags);
            errdefer value.deinit();
            return finishOp(target_tags, ctx, value, self.requiresGrad(), AxisViewBackward(tags, axes), .{ ctx.allocator, self.grad_state, &self.value });
        }
    };
}
