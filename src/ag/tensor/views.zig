//! Views and data movement for every scalar-dtype `Tensor(...)` branch,
//! written once over `Self.dtype`: the zero-copy views (tag renames,
//! permutes, inserted and squeezed axes, split/merge, narrow, strided
//! slices), the copying movers (gather, concat, stack, flip, roll,
//! setSlice, setRows), and the materialize/contiguous/detach trio. On the
//! f32 branch every op is differentiable (the VJP attaches through
//! `finishOp`). On the typed branches the same ops are no-grad constants:
//! a grad-requiring operand is `error.UnsupportedGradient` (the
//! typed-branch rule, so a 16-bit leaf never silently drops its graph).
//! Every result, typed or not, is adopted by an open exec scope. A mixin
//! over the tensor struct; aliased back onto it in ../tensor.zig.

const std = @import("std");
const tensor_mod = @import("../../tensor.zig");
const exec_mod = @import("../../exec.zig");
const tag_ops = @import("../../tag_ops.zig");
const control = @import("../control.zig");
const core = @import("../core.zig");
const tags_mod = @import("../../tags.zig");
const backward_elementwise = @import("../backward/elementwise.zig");
const backward_gather_scatter = @import("../backward/gather_scatter.zig");
const backward_shape = @import("../backward/shape.zig");

const TensorError = tensor_mod.TensorError;
const ExecContext = exec_mod.ExecContext;
const GradState = core.GradState;
const Tag = tags_mod.Tag;
const normalizeTags = tags_mod.normalizeTags;
const validateUniqueTags = tags_mod.validateUniqueTags;
const validateSameTagSet = tags_mod.validateSameTagSet;
const rawRank = tags_mod.rawRank;
const tagIndexOrCompileError = tags_mod.tagIndexOrCompileError;
const identityAxes = tags_mod.identityAxes;
const alignAxes = tags_mod.alignAxes;
const insertAxes = tags_mod.insertAxes;
const squeezeAxes = tags_mod.squeezeAxes;
const removeTag = tags_mod.removeTag;
const replaceTag = tags_mod.replaceTag;
const insertTagAt = tags_mod.insertTagAt;
const splitTags = tags_mod.splitTags;
const mergeTags = tags_mod.mergeTags;
const IdentityBackward = backward_elementwise.IdentityBackward;
const BroadcastBackward = backward_shape.BroadcastBackward;
const NarrowBackward = backward_shape.NarrowBackward;
const ConcatBackward = backward_shape.ConcatBackward;
const ReshapeBackward = backward_shape.ReshapeBackward;
const AxisViewBackward = backward_shape.AxisViewBackward;
const StridedViewBackward = backward_shape.StridedViewBackward;
const GatherBackward = backward_gather_scatter.GatherBackward;
const SetSliceBackward = backward_gather_scatter.SetSliceBackward;
const SetRowsBackward = backward_gather_scatter.SetRowsBackward;

pub fn Ops(comptime Self: type) type {
    return struct {
        const dtype = Self.dtype;
        const tags = Self.axis_tags;
        const tag_rank = Self.tag_count;
        const tensor_rank = Self.tensor_rank;
        const axis = Self.axis;
        const ag_tensor = Self.ag_root;
        const Tensor = ag_tensor.Tensor;
        const SliceRange = ag_tensor.SliceRange;
        const concat_inline_inputs = ag_tensor.concat_inline_inputs;
        const plumbing = @import("plumbing.zig").Mod(ag_tensor);
        const recordsGrad = plumbing.recordsGrad;
        const finishTypedNoGrad = plumbing.finishTypedNoGrad;
        const finishNoGrad = plumbing.finishNoGrad;
        const finishOp = plumbing.finishOp;
        const rawShapeArray = plumbing.rawShapeArray;
        const rawShapeArrayOf = plumbing.rawShapeArrayOf;
        const cloneInverseRopeTable = plumbing.cloneInverseRopeTable;
        const RawT = tensor_mod.TensorOf(dtype);
        /// The f32 branch is the differentiable one; every other dtype takes
        /// the constant tail.
        const differentiable = dtype == .f32;

        /// The result type of a view with `result_tags`: same dtype as
        /// `Self` (for f32 this is `Self` whenever the tags are unchanged).
        fn Out(comptime result_tags: anytype) type {
            return Tensor(.{ .dtype = dtype, .tags = result_tags });
        }

        /// The operand's gradient state, null on the branches without one.
        fn gradStateOf(t: anytype) ?*GradState {
            if (comptime @hasField(@TypeOf(t.*), "grad_state")) return t.grad_state;
            return null;
        }

        /// The no-grad tail: a constant, adopted by an open exec scope
        /// whatever the dtype.
        fn finishConstant(comptime result_tags: anytype, ctx: *ExecContext, value: RawT) !Out(result_tags) {
            return plumbing.finishTyped(Out(result_tags), ctx, value);
        }

        pub fn materialize(self: *const Self, ctx: *ExecContext) !Self {
            var value = try ctx.materialize(dtype, self.asRawTensor());
            errdefer value.deinit();
            if (comptime !differentiable) return finishTypedNoGrad(Out(tags), ctx, value, self.requiresGrad());
            if (!recordsGrad(self.requiresGrad())) return finishNoGrad(tags, ctx, value);
            const Record = StridedViewBackward(tags, tags);
            return finishOp(tags, ctx, value, Record.of(gradStateOf(self), &self.value, &value));
        }

        /// Borrow-if-contiguous materialize: an already-contiguous tensor
        /// returns a zero-copy retained view of the same storage (linked to
        /// the graph through an identity backward, so the handle carries a
        /// state of its own rather than sharing `self`'s); a strided view
        /// returns `materialize(ctx)`.
        /// Either way the result is owned by the caller (always `deinit` it;
        /// refcounted storage keeps the view case safe past the source's
        /// deinit), contiguous, and safe for `data`/`dataConst` access
        /// (`data` still rejects grad-carrying tensors, the autograd
        /// mutation invariant). The torch.contiguous aliasing caveat
        /// carries over: the already-contiguous case ALIASES `self`'s bytes
        /// (in-place mutation of either is visible through both) while the
        /// strided case is an independent snapshot; treat the result as
        /// read-only where the distinction matters, or use `materialize`
        /// when a guaranteed copy is wanted.
        pub fn contiguous(self: *const Self, ctx: *ExecContext) !Self {
            if (!self.isContiguous()) return self.materialize(ctx);
            var value = try self.value.cloneView();
            errdefer value.deinit();
            if (comptime !differentiable) return finishTypedNoGrad(Out(tags), ctx, value, self.requiresGrad());
            if (!recordsGrad(self.requiresGrad())) return finishNoGrad(tags, ctx, value);
            const Record = IdentityBackward(tags);
            return finishOp(tags, ctx, value, Record{ .parents = .{gradStateOf(self)} });
        }

        /// No-grad view of the same storage.
        pub fn detach(self: *const Self, ctx: *ExecContext) !Self {
            var value = try self.value.cloneView();
            errdefer value.deinit();
            return finishConstant(tags, ctx, value);
        }

        pub fn withTags(self: *const Self, ctx: *ExecContext, comptime new_tags_spec: anytype) !Out(normalizeTags(new_tags_spec)) {
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
        ) !Out(normalizeTags(new_tags_spec)) {
            const new_tags = normalizeTags(new_tags_spec);
            comptime validateUniqueTags(new_tags);
            var value = try self.value.viewWithStrides(raw_shape[0..], raw_strides[0..]);
            errdefer value.deinit();
            if (comptime !differentiable) return finishTypedNoGrad(Out(new_tags), ctx, value, self.requiresGrad());
            if (!recordsGrad(self.requiresGrad())) return finishNoGrad(new_tags, ctx, value);
            const Record = StridedViewBackward(tags, new_tags);
            return finishOp(new_tags, ctx, value, Record.of(gradStateOf(self), &self.value, &value));
        }

        pub fn alignTo(self: *const Self, ctx: *ExecContext, comptime target_tags_spec: anytype) !Out(normalizeTags(target_tags_spec)) {
            const target_tags = normalizeTags(target_tags_spec);
            return axisView(self, ctx, alignAxes(tags, target_tags), target_tags);
        }

        pub fn permuteTo(self: *const Self, ctx: *ExecContext, comptime target_tags_spec: anytype) !Out(normalizeTags(target_tags_spec)) {
            const target_tags = normalizeTags(target_tags_spec);
            comptime validateSameTagSet(tags, target_tags);
            return axisView(self, ctx, alignAxes(tags, target_tags), target_tags);
        }

        pub fn transpose(self: *const Self, ctx: *ExecContext, comptime target_tags_spec: anytype) !Out(normalizeTags(target_tags_spec)) {
            return self.permuteTo(ctx, target_tags_spec);
        }

        pub fn insertAxis(self: *const Self, ctx: *ExecContext, comptime tag: Tag, comptime axis_index: usize) !Out(insertTagAt(tags, tag, axis_index)) {
            const result_tags = insertTagAt(tags, tag, axis_index);
            return axisView(self, ctx, insertAxes(tag_rank, axis_index), result_tags);
        }

        pub fn squeeze(self: *const Self, ctx: *ExecContext, comptime tag: Tag) !Out(removeTag(tags, tag)) {
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
        ) !Out(splitTags(tags, tag, normalizeTags(split_tags_spec))) {
            const split_tags = normalizeTags(split_tags_spec);
            const result_tags = splitTags(tags, tag, split_tags);
            var value = try tag_ops.splitAxisView(dtype, tags, self.asRawTensor(), tag, split_tags, split_shape);
            errdefer value.deinit();
            if (comptime !differentiable) return finishTypedNoGrad(Out(result_tags), ctx, value, self.requiresGrad());
            if (!recordsGrad(self.requiresGrad())) return finishNoGrad(result_tags, ctx, value);
            const Record = StridedViewBackward(tags, result_tags);
            return finishOp(result_tags, ctx, value, Record.of(gradStateOf(self), &self.value, &value));
        }

        pub fn merge(self: *const Self, ctx: *ExecContext, comptime out_tag: Tag, comptime merge_tags_spec: anytype) !Out(mergeTags(tags, out_tag, normalizeTags(merge_tags_spec))) {
            const merge_tags = normalizeTags(merge_tags_spec);
            const result_tags = mergeTags(tags, out_tag, merge_tags);
            var value = try tag_ops.mergeAxesView(dtype, tags, self.asRawTensor(), out_tag, merge_tags);
            errdefer value.deinit();
            if (comptime !differentiable) return finishTypedNoGrad(Out(result_tags), ctx, value, self.requiresGrad());
            if (!recordsGrad(self.requiresGrad())) return finishNoGrad(result_tags, ctx, value);
            const Record = StridedViewBackward(tags, result_tags);
            return finishOp(result_tags, ctx, value, Record.of(gradStateOf(self), &self.value, &value));
        }

        /// Arbitrary row-major reinterpretation to `new_tags_spec` /
        /// `new_shape` (torch.reshape): the element count must match
        /// (`InvalidShape` otherwise). View-or-materialize like torch: a
        /// contiguous source stays a zero-copy view; a non-contiguous one
        /// materializes first (the `flatten` rule). Composed flatten ->
        /// split, so gradients come from the existing exact view records;
        /// a rank-1 target degenerates to plain `flatten`.
        pub fn reshape(
            self: *const Self,
            ctx: *ExecContext,
            comptime new_tags_spec: anytype,
            new_shape: [normalizeTags(new_tags_spec).len]usize,
        ) !Out(normalizeTags(new_tags_spec)) {
            const new_tags = comptime normalizeTags(new_tags_spec);
            if (comptime new_tags.len == 1) {
                if (self.asRawTensor().len() != new_shape[0]) return TensorError.InvalidShape;
                return self.flatten(ctx, new_tags[0]);
            }
            var flat = try self.flatten(ctx, new_tags[0]);
            defer flat.deinit();
            return flat.split(ctx, new_tags[0], new_tags_spec, new_shape);
        }

        pub fn broadcastTo(
            self: *const Self,
            ctx: *ExecContext,
            comptime target_tags_spec: anytype,
            target_shape: [normalizeTags(target_tags_spec).len]usize,
        ) !Out(normalizeTags(target_tags_spec)) {
            const target_tags = normalizeTags(target_tags_spec);
            var value = try tag_ops.broadcastTensorTo(dtype, tags, self.asRawTensor(), target_tags, target_shape);
            errdefer value.deinit();
            if (comptime !differentiable) return finishTypedNoGrad(Out(target_tags), ctx, value, self.requiresGrad());
            if (!recordsGrad(self.requiresGrad())) return finishNoGrad(target_tags, ctx, value);
            const Record = BroadcastBackward(tags, target_tags);
            return finishOp(target_tags, ctx, value, Record{ .parents = .{gradStateOf(self)}, .source_shape = rawShapeArray(tags, (&self.value)) });
        }

        pub fn flatten(self: *const Self, ctx: *ExecContext, comptime out_tag: Tag) !Out(.{out_tag}) {
            var value = try tag_ops.flattenTensor(dtype, ctx, self.asRawTensor());
            errdefer value.deinit();
            if (comptime !differentiable) return finishTypedNoGrad(Out(.{out_tag}), ctx, value, self.requiresGrad());
            if (!recordsGrad(self.requiresGrad())) return finishNoGrad(.{out_tag}, ctx, value);
            const owned_shape = try ctx.allocator.dupe(usize, (&self.value).shape.slice());
            errdefer ctx.allocator.free(owned_shape);
            return finishOp(.{out_tag}, ctx, value, ReshapeBackward{ .parents = .{gradStateOf(self)}, .source_shape = owned_shape });
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
        /// like `gather` indices); `offsets.len` must equal
        /// `numel / dim(tag)` or the call errors with `InvalidShape`; for a
        /// rank-1 tensor that is a single element and `rollBy` matches
        /// `roll`. A per-section permutation, composed flatten + gather +
        /// split, so the gradient is exact (the inverse per-section roll).
        pub fn rollBy(self: *const Self, ctx: *ExecContext, comptime tag: Tag, offsets: []const isize) !Self {
            const roll_axis = comptime axis(tag);
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

        pub fn narrow(self: *const Self, ctx: *ExecContext, comptime tag: Tag, start: usize, length: usize) !Self {
            const slice_axis = comptime axis(tag);
            var value = try ctx.narrowAxis(dtype, tag_rank, self.asRawTensor(), slice_axis, start, length);
            errdefer value.deinit();
            if (comptime !differentiable) return finishTypedNoGrad(Out(tags), ctx, value, self.requiresGrad());
            if (!recordsGrad(self.requiresGrad())) return finishNoGrad(tags, ctx, value);
            const Record = NarrowBackward(tags, slice_axis);
            return finishOp(tags, ctx, value, Record{
                .parents = .{gradStateOf(self)},
                .source_shape = rawShapeArray(tags, (&self.value)),
                .start = start,
            });
        }

        /// Select one position of `tag` and remove the axis (torch.select /
        /// `x[i]`): the single-slice sibling of `unbindInto`. `index`
        /// counts from the end when negative (torch convention); out of
        /// range errors with `IndexOutOfBounds`. Composed narrow -> squeeze,
        /// so the value is a zero-copy view aliasing the selected row and
        /// the gradient is the exact scatter (every unselected position
        /// receives zero).
        pub fn select(self: *const Self, ctx: *ExecContext, comptime tag: Tag, index: isize) !Out(removeTag(tags, tag)) {
            const n: isize = @intCast(self.asRawTensor().shape.at(comptime axis(tag)));
            const shifted = if (index < 0) index +| n else index;
            if (shifted < 0 or shifted >= n) return TensorError.IndexOutOfBounds;
            var row = try self.narrow(ctx, tag, @intCast(shifted), 1);
            defer row.deinit();
            return row.squeeze(ctx, tag);
        }

        /// `narrow` with a step (torch basic slicing `x[start::step]` along
        /// one axis): `length` elements at `start, start+step, ...`.
        /// `step == 0`, `length == 0` (zero-size tensors are not
        /// representable), or a last element at or past `dim(tag)` error
        /// with `InvalidShape`. On a no-grad tensor the result is a
        /// zero-copy strided **view** (retains and aliases the source
        /// buffer, like `narrow`); under gradients it lowers to `gather`
        /// over the stepped indices, a copy with the exact scatter-add
        /// record (the `flip`/`roll` precedent), so skipped positions
        /// receive zero gradient.
        pub fn sliceStep(self: *const Self, ctx: *ExecContext, comptime tag: Tag, start: usize, length: usize, step: usize) !Self {
            const slice_axis = comptime axis(tag);
            const raw = self.asRawTensor();
            const axis_dim = raw.shape.at(slice_axis);
            if (step == 0 or length == 0) return TensorError.InvalidShape;
            if (start >= axis_dim or start + (length - 1) * step >= axis_dim) return TensorError.InvalidShape;
            if (self.requiresGrad()) {
                if (comptime !differentiable) return error.UnsupportedGradient;
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
            return finishConstant(tags, ctx, value);
        }

        /// Multi-axis basic slicing (torch/numpy `x[1:-1, ::2]`, positive
        /// steps): `spec` is a struct literal naming the tags to slice, e.g.
        /// `.{ .h = .{ .start = 1, .end = -1 }, .w = .{ .step = 2 } }`, each
        /// field a `fucina.SliceRange`-shaped range (`start`/`end`/`step`,
        /// each optional); axes not named pass through whole, and a field
        /// naming a tag not on the tensor is a compile error. torch bounds
        /// semantics: negatives count from the end, `end = null` means the
        /// axis dim, out-of-range bounds clamp; `step == 0` or an empty
        /// result error with `InvalidShape` (zero-size tensors are not
        /// representable). Negative steps are deliberately unsupported
        /// (torch rejects them in basic indexing too; strides are unsigned,
        /// so a reversed view is not representable; compose `flip`).
        /// Lowered to per-axis `narrow`/`sliceStep` in tag order: with
        /// step-1 ranges the no-grad value is a zero-copy view and the
        /// gradient is the exact per-axis scatter; stepped axes follow the
        /// `sliceStep` contract (view no-grad, gather copy under gradients).
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

        pub fn gather(
            self: *const Self,
            ctx: *ExecContext,
            comptime tag: Tag,
            indices: []const usize,
            comptime out_tag: Tag,
        ) !Out(replaceTag(tags, tag, out_tag)) {
            const result_tags = replaceTag(tags, tag, out_tag);
            const gather_axis = comptime axis(tag);
            var value = try ctx.gatherAxis(dtype, tag_rank, self.asRawTensor(), gather_axis, indices);
            errdefer value.deinit();
            if (comptime !differentiable) return finishTypedNoGrad(Out(result_tags), ctx, value, self.requiresGrad());
            if (!recordsGrad(self.requiresGrad())) return finishNoGrad(result_tags, ctx, value);
            const Record = GatherBackward(tags, gather_axis);
            const owned_indices = try ctx.allocator.dupe(usize, indices);
            errdefer ctx.allocator.free(owned_indices);
            return finishOp(result_tags, ctx, value, Record{
                .parents = .{gradStateOf(self)},
                .source_shape = rawShapeArray(tags, (&self.value)),
                .estimated_work = if (gradStateOf(self) != null) (&self.value).len() else 0,
                .indices = owned_indices,
            });
        }

        pub fn setSlice(self: *const Self, ctx: *ExecContext, comptime tag: Tag, start: usize, update: *const Self) !Self {
            const slice_axis = comptime axis(tag);
            var value = try ctx.setSliceAxis(dtype, tag_rank, self.asRawTensor(), update.asRawTensor(), slice_axis, start);
            errdefer value.deinit();
            if (comptime !differentiable) return finishTypedNoGrad(Out(tags), ctx, value, self.requiresGrad() or update.requiresGrad());
            if (!recordsGrad(self.requiresGrad() or update.requiresGrad())) return finishNoGrad(tags, ctx, value);
            const Record = SetSliceBackward(tags, slice_axis);
            return finishOp(tags, ctx, value, Record{
                .parents = .{ gradStateOf(self), gradStateOf(update) },
                .update_shape = rawShapeArray(tags, update.asRawTensor()),
                .start = start,
            });
        }

        pub fn setRows(self: *const Self, ctx: *ExecContext, comptime tag: Tag, indices: []const usize, update: *const Self) !Self {
            const rows_axis = comptime axis(tag);
            var value = try ctx.setRows(dtype, tag_rank, self.asRawTensor(), update.asRawTensor(), rows_axis, indices);
            errdefer value.deinit();
            if (comptime !differentiable) return finishTypedNoGrad(Out(tags), ctx, value, self.requiresGrad() or update.requiresGrad());
            if (!recordsGrad(self.requiresGrad() or update.requiresGrad())) return finishNoGrad(tags, ctx, value);
            const Record = SetRowsBackward(tags, rows_axis);
            const owned_indices = try ctx.allocator.dupe(usize, indices);
            errdefer ctx.allocator.free(owned_indices);
            return finishOp(tags, ctx, value, Record{ .parents = .{ gradStateOf(self), gradStateOf(update) }, .indices = owned_indices });
        }

        pub fn concat(self: *const Self, ctx: *ExecContext, comptime tag: Tag, others: []const *const Self) !Self {
            var any_grad = self.requiresGrad();
            for (others) |other| any_grad = any_grad or other.requiresGrad();

            const input_count = others.len + 1;
            var raw_inputs_stack: [concat_inline_inputs]*const RawT = undefined;
            const raw_inputs = if (input_count <= raw_inputs_stack.len)
                raw_inputs_stack[0..input_count]
            else
                try ctx.allocator.alloc(*const RawT, input_count);
            defer if (input_count > raw_inputs_stack.len) ctx.allocator.free(raw_inputs);
            raw_inputs[0] = self.asRawTensor();
            for (others, raw_inputs[1..]) |other, *raw| raw.* = other.asRawTensor();

            const concat_axis = comptime axis(tag);
            var value = try ctx.concatAxis(dtype, tag_rank, raw_inputs, concat_axis);
            errdefer value.deinit();
            if (comptime !differentiable) return finishTypedNoGrad(Out(tags), ctx, value, any_grad);
            if (!recordsGrad(any_grad)) return finishNoGrad(tags, ctx, value);
            // The record owns one parent slot and one axis size per input.
            const Record = ConcatBackward(tags, concat_axis);
            const owned_parents = try ctx.allocator.alloc(?*GradState, input_count);
            errdefer ctx.allocator.free(owned_parents);
            const owned_sizes = try ctx.allocator.alloc(usize, input_count);
            errdefer ctx.allocator.free(owned_sizes);
            owned_parents[0] = gradStateOf(self);
            owned_sizes[0] = self.asRawTensor().shape.at(concat_axis);
            for (others, owned_parents[1..], owned_sizes[1..]) |other, *parent, *size| {
                parent.* = gradStateOf(other);
                size.* = other.asRawTensor().shape.at(concat_axis);
            }
            return finishOp(tags, ctx, value, Record{ .parents = owned_parents, .sizes = owned_sizes });
        }

        /// Stack `self` and `others` along a NEW axis tagged `new_tag`
        /// inserted at `axis_index` (torch.stack): composed as insertAxis on
        /// every input + concat, so the result is differentiable in ALL
        /// inputs through the multi-parent ConcatBackward.
        pub fn stack(
            self: *const Self,
            ctx: *ExecContext,
            comptime new_tag: Tag,
            comptime axis_index: usize,
            others: []const *const Self,
        ) !Out(insertTagAt(tags, new_tag, axis_index)) {
            var any_grad = self.requiresGrad();
            for (others) |other| any_grad = any_grad or other.requiresGrad();

            const Expanded = Out(insertTagAt(tags, new_tag, axis_index));
            var expanded = try ctx.allocator.alloc(Expanded, others.len + 1);
            defer ctx.allocator.free(expanded);
            var created: usize = 0;
            // The inserted-axis views are composition temporaries: releasing
            // their handles is always safe (the concat record retains their
            // graph states; a scope-owned handle's deinit is a no-op).
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
        /// been released.
        pub fn unbindInto(self: *const Self, ctx: *ExecContext, comptime tag: Tag, out: []Out(removeTag(tags, tag))) !void {
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
        /// length becomes `n * dim(tag)` with n back-to-back copies of `self`.
        /// `n == 1` returns a zero-copy identity view; `n > 1` is a concat
        /// of `self` with itself (one multi-parent node: gradients from all
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

        fn axisView(self: *const Self, ctx: *ExecContext, comptime axes: anytype, comptime target_tags: anytype) !Out(target_tags) {
            var value = try plumbing.axisViewTensorOf(dtype, self.asRawTensor(), axes, target_tags);
            errdefer value.deinit();
            if (comptime !differentiable) return finishTypedNoGrad(Out(target_tags), ctx, value, self.requiresGrad());
            if (!recordsGrad(self.requiresGrad())) return finishNoGrad(target_tags, ctx, value);
            const Record = AxisViewBackward(tags, axes);
            return finishOp(target_tags, ctx, value, Record{ .parents = .{gradStateOf(self)}, .source_shape = rawShapeArray(tags, (&self.value)) });
        }
    };
}
