//! f32-only structural ops: constant fills (`pad`, the 2-D pads, `shiftBy`),
//! the diagonal/band family, and the composed embeds. The dtype-generic
//! views (permutes, split/merge, narrow, slices, concat, ...) live in
//! ../views.zig. A mixin over the ag FloatTensor struct; aliased back onto
//! it in ../../tensor.zig.

const std = @import("std");
const tensor_mod = @import("../../../tensor.zig");
const exec_mod = @import("../../../exec.zig");
const tags_mod = @import("../../../tags.zig");
const backward_shape = @import("../../backward/shape.zig");

const TensorError = tensor_mod.TensorError;
const ExecContext = exec_mod.ExecContext;
const Tag = tags_mod.Tag;
const normalizeTags = tags_mod.normalizeTags;
const tagIndex = tags_mod.tagIndex;
const removeTag = tags_mod.removeTag;
const replaceTag = tags_mod.replaceTag;
const pointwiseResultTags = tags_mod.pointwiseResultTags;
const insertTagAt = tags_mod.insertTagAt;
const PadBackward = backward_shape.PadBackward;
const StridedViewBackward = backward_shape.StridedViewBackward;

pub fn Ops(comptime Self: type) type {
    return struct {
        const tags = Self.axis_tags;
        const tag_rank = Self.tag_count;
        const tensor_rank = Self.tensor_rank;
        const tag_count = Self.tag_count;
        const axis = Self.axis;
        const ag_tensor = Self.ag_root;
        const Tensor = ag_tensor.Tensor;
        const plumbing = @import("../plumbing.zig").Mod(ag_tensor);
        const finishOp = plumbing.finishOp;
        const requireScopeForComposedGrad = plumbing.requireScopeForComposedGrad;
        const padding2dValues = plumbing.padding2dValues;

        /// Non-circular `rollBy`: same per-section offsets and sign
        /// convention, but positions shifted in from outside the axis hold
        /// the constant `fill` instead of wrapping
        /// (`out[..., j, ...] = self[..., j - shift, ...]` when in bounds,
        /// else `fill`). `fill` is a constant and receives no gradient;
        /// source positions shifted out of the axis receive zero gradient
        /// (composed gather + maskedFill: the fill mask zeroes their
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

        /// Main diagonal over the (`tag_a`, `tag_b`) plane (torch.diagonal,
        /// offset 0): a zero-copy strided **view** of length
        /// `min(dim(tag_a), dim(tag_b))`; element i is `self[.., i, .., i,
        /// ..]`. Both tags are removed and the diagonal axis is appended
        /// LAST as `out_tag` (the torch axis order). Works at any rank
        /// carrying both tags; differentiable (strided-view scatter,
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
        /// (torch.trace generalized to named axes): composed diagonal ->
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
        /// the two result axes. Composed zeros -> setRows (flat diagonal
        /// positions) -> reshape, so the gradient is the exact
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
        /// that side. Works at any rank carrying both tags; remaining axes
        /// pass through. Implemented as a constant 0/1 band-plane multiply
        /// (the `bandMask` keep-set cast to f32), so the VJP is the exact
        /// same-band mask on the upstream gradient. Differentiable in
        /// `self`.
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
        /// `out[..., i, j] = self[..., i] * [i == j]`, with `vec_tag`
        /// renamed to `out_tags[0]` (the row axis) in place and
        /// `out_tags[1]` (the column axis) appended LAST. Works at any rank
        /// carrying `vec_tag`; the remaining axes pass through. Composed
        /// rename -> broadcast-multiply against an `eye` constant, so the
        /// gradient is the exact diagonal extraction; scope-required under
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
            var value = try ctx.pad(tag_rank, self.asRawTensor(), pad_axis, before, after, fill);
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
        /// representable). Any rank carrying both tags works; the remaining
        /// axes pass through. The result is always an owned regular tensor:
        /// an all-zero padding is the contiguous identity, and a crop-only
        /// result materializes rather than returning a strided view.
        /// Differentiable in `self`: pad positions are constants and drop
        /// their gradient, cropped source positions receive zero gradient
        /// (composed narrow + pad per axis, so the gradients come from the
        /// existing exact records). When gradients are tracked this
        /// requires an active exec scope (see `maskedSelect`), even for
        /// paddings that degenerate to fewer ops.
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
    };
}
