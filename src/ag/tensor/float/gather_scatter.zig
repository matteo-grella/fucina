//! f32 tensor methods: indexed reads and writes beyond the dtype-generic
//! views in ../views.zig (which own `gather`, `setSlice`, `setRows`). A
//! mixin over the ag FloatTensor struct; aliased back onto it in
//! ../../tensor.zig.

const std = @import("std");
const tensor_mod = @import("../../../tensor.zig");
const dtype_mod = @import("../../../dtype.zig");
const exec_mod = @import("../../../exec.zig");
const tags_mod = @import("../../../tags.zig");
const backward_gather_scatter = @import("../../backward/gather_scatter.zig");

const TensorError = tensor_mod.TensorError;
const ExecContext = exec_mod.ExecContext;
const Tag = tags_mod.Tag;
const replaceTag = tags_mod.replaceTag;
const RelposShiftBackward = backward_gather_scatter.RelposShiftBackward;
const IndexAddBackward = backward_gather_scatter.IndexAddBackward;
const TakeAlongBackward = backward_gather_scatter.TakeAlongBackward;
const ScatterAlongBackward = backward_gather_scatter.ScatterAlongBackward;
const ZeroSliceBackward = backward_gather_scatter.ZeroSliceBackward;
const ZeroRowsBackward = backward_gather_scatter.ZeroRowsBackward;

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
        const TensorObject = plumbing.TensorObject;

        /// No-grad: copy of `self` with `[start, start+length)` along `axis_tag` zeroed.
        pub fn zeroSlice(self: *const Self, ctx: *ExecContext, comptime axis_tag: Tag, start: usize, length: usize) !Self {
            const zero_axis = comptime Self.axis(axis_tag);
            var value = try ctx.zeroSlice(tensor_rank, self.asRawTensor(), zero_axis, start, length);
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad(), ZeroSliceBackward(tags, zero_axis), .{ ctx.allocator, self.grad_state, start, length });
        }

        /// No-grad: copy of `self` with the given `indices` along `axis_tag` zeroed.
        pub fn zeroRows(self: *const Self, ctx: *ExecContext, comptime axis_tag: Tag, indices: []const usize) !Self {
            const zero_axis = comptime Self.axis(axis_tag);
            var value = try ctx.zeroRows(tensor_rank, self.asRawTensor(), zero_axis, indices);
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad(), ZeroRowsBackward(tags, zero_axis), .{ ctx.allocator, self.grad_state, indices });
        }

        /// Transformer-XL relative-shift / "skew": a rank-3
        /// relative-score tensor `[H,Tq,P]` → `[H,Tq,Tk]` with
        /// `out[h,qi,kj] = self[h, qi, kj+(Tq-1)-qi]` (`P >= Tk+Tq-1`). The closed
        /// form of the relpos pad/reshape/view remap; differentiable (scatter VJP).
        /// `out_tags` names the result axes.
        pub fn relposShift(self: *const Self, ctx: *ExecContext, t_k: usize, comptime out_tags: anytype) !Tensor(out_tags) {
            var value = try ctx.relposShift(self.asRawTensor(), t_k);
            errdefer value.deinit();
            return finishOp(out_tags, ctx, value, self.requiresGrad(), RelposShiftBackward, .{ ctx.allocator, self.grad_state, self.value.shape.slice()[2] });
        }

        /// `gather` with a tensor of indices (torch.index_select): `indices`
        /// is a rank-1 **i64** tensor (the argmax/topK/sort index
        /// convention — their outputs feed it directly; any other dtype is
        /// a compile error), read host-side into `gather`'s `[]usize`
        /// buffer; entries outside `[0, dim(tag))` error with
        /// `IndexOutOfBounds` (negatives are not wrapped). The selected
        /// axis is retagged `out_tag` (which may equal `tag`), sized by the
        /// index count. Differentiable in `self` (the exact scatter-add
        /// adjoint — duplicate indices accumulate their gradients); the
        /// index tensor is control data, not part of the graph, and can be
        /// released after the call.
        pub fn indexSelect(
            self: *const Self,
            ctx: *ExecContext,
            comptime tag: Tag,
            indices: anytype,
            comptime out_tag: Tag,
        ) !Tensor(replaceTag(tags, tag, out_tag)) {
            const Indices = TensorObject(@TypeOf(indices));
            comptime {
                if (Indices.dtype != .i64)
                    @compileError("indexSelect expects i64 indices (the argmax/topK/sort output dtype); cast integer indices explicitly");
                if (Indices.axis_tags.len != 1)
                    @compileError("indexSelect expects rank-1 indices (torch.index_select); for per-element index tensors use takeAlongAxis");
            }
            const idx_raw = indices.asRawTensor();
            const idx_buf = try hostIndexBuffer(ctx, try idx_raw.dataConstChecked(), self.asRawTensor().shape.at(comptime axis(tag)));
            defer ctx.allocator.free(idx_buf);
            return self.gather(ctx, tag, idx_buf, out_tag);
        }

        /// Select the elements of `self` where `mask` is nonzero (torch
        /// masked_select): a rank-1 tensor tagged `out_tag` holding the
        /// selected elements in row-major order. The mask must have `self`'s
        /// shape, be contiguous (the nonzero scan reads its storage
        /// host-side), and is a non-grad `!= 0` mask; `self` IS
        /// differentiable — composed flatten + gather, so GatherBackward
        /// scatter-adds the gradient into the source. The gather index
        /// buffer is exact `[]usize` (NOT the f32 index convention of
        /// argmax/topK/sort — no < 2^24 exactness caveat applies here).
        /// Errors with `EmptySelection` when the mask selects nothing
        /// (zero-size tensors are not representable) — a dedicated error,
        /// distinct from the shape errors, so the data-dependent no-match
        /// outcome stays catchable apart from caller bugs; pre-counting via
        /// a mask sum avoids the error path entirely. When gradients are
        /// tracked this requires an active exec scope (see `nllLoss`);
        /// errors with `ActiveExecScopeRequired` otherwise.
        pub fn maskedSelect(self: *const Self, ctx: *ExecContext, mask: anytype, comptime out_tag: Tag) !Tensor(.{out_tag}) {
            const Mask = TensorObject(@TypeOf(mask));
            comptime {
                if (Mask.dtype != .bool and !dtype_mod.supportsForwardFloatMath(Mask.dtype))
                    @compileError("maskedSelect takes a .bool or float mask; cast integer masks explicitly");
            }
            try requireScopeForComposedGrad(ctx, self.requiresGrad());
            const mask_raw = mask.asRawTensor();
            if (!std.mem.eql(usize, self.asRawTensor().shape.slice(), mask_raw.shape.slice())) return TensorError.ShapeMismatch;
            const mask_values = try mask_raw.dataConstChecked();

            var count: usize = 0;
            for (mask_values) |mv| count += @intFromBool(dtype_mod.isTruthy(Mask.dtype, mv));
            if (count == 0) return TensorError.EmptySelection;

            const indices = try ctx.allocator.alloc(usize, count);
            defer ctx.allocator.free(indices);
            var slot: usize = 0;
            for (mask_values, 0..) |mv, i| {
                if (dtype_mod.isTruthy(Mask.dtype, mv)) {
                    indices[slot] = i;
                    slot += 1;
                }
            }

            var flat = try self.flatten(ctx, out_tag);
            defer flat.deinit();
            return flat.gather(ctx, out_tag, indices, out_tag);
        }

        /// Row-major flat indices of the nonzero elements (torch.nonzero
        /// over the flattened tensor; NaN counts as nonzero), returned as
        /// a HOST slice the caller owns and frees with `allocator` — the
        /// design keeps data-dependent cardinality host-side, where
        /// `[]usize` pairs directly with `gather`/`setRows`/`indexAdd`/
        /// `oneHot`, so a no-match result is just an empty slice (no
        /// zero-size tensor needed, unlike `maskedSelect`). Reads `self`
        /// host-side; contiguous only (like `maskedSelect`'s mask).
        pub fn nonzero(self: *const Self, allocator: std.mem.Allocator) ![]usize {
            const values = try self.asRawTensor().dataConstChecked();
            var count: usize = 0;
            for (values) |v| count += @intFromBool(v != 0);
            const indices = try allocator.alloc(usize, count);
            var slot: usize = 0;
            for (values, 0..) |v, i| {
                if (v != 0) {
                    indices[slot] = i;
                    slot += 1;
                }
            }
            return indices;
        }

        /// Scatter a rank-1 `values` tensor into the positions of `self`
        /// where `mask` is nonzero, in row-major order — the inverse of
        /// `maskedSelect` (torch masked_scatter, with an exact-count
        /// contract: `values` must hold exactly `count(mask != 0)`
        /// elements). Unselected positions keep `self`'s values. The mask
        /// follows the `where`/`maskedFill` convention (non-grad `!= 0`
        /// mask, same shape as `self`, contiguous — the nonzero scan reads
        /// its storage host-side). Differentiable in `self` (grad zeroed at
        /// scattered positions) and `values` (grad gathered row-major from
        /// the selected positions) — composed gather + split + where, so
        /// the gradients come from the existing exact records. Errors with
        /// `EmptySelection` when the mask selects nothing (zero-size tensors
        /// are not representable; see `maskedSelect`) and with `InvalidShape`
        /// when `values`' length differs from the selected count. When
        /// gradients are tracked this requires an active exec scope (see
        /// `maskedSelect`); errors with `ActiveExecScopeRequired` otherwise.
        pub fn maskedScatter(
            self: *const Self,
            ctx: *ExecContext,
            mask: anytype,
            comptime values_tag: Tag,
            values: *const Tensor(.{values_tag}),
        ) !Self {
            comptime if (tag_rank == 0) @compileError("maskedScatter requires at least one axis");
            const Mask = TensorObject(@TypeOf(mask));
            comptime {
                if (Mask.dtype != .bool and !dtype_mod.supportsForwardFloatMath(Mask.dtype))
                    @compileError("maskedScatter takes a .bool or float mask; cast integer masks explicitly");
            }
            try requireScopeForComposedGrad(ctx, self.requiresGrad() or values.requiresGrad());
            const mask_raw = mask.asRawTensor();
            if (!std.mem.eql(usize, self.asRawTensor().shape.slice(), mask_raw.shape.slice())) return TensorError.ShapeMismatch;
            const mask_values = try mask_raw.dataConstChecked();

            var count: usize = 0;
            for (mask_values) |mv| count += @intFromBool(dtype_mod.isTruthy(Mask.dtype, mv));
            if (count == 0) return TensorError.EmptySelection;
            if (values.asRawTensor().len() != count) return TensorError.InvalidShape;

            // Selected position i gathers values[k(i)]; unselected positions
            // gather values[0] as a placeholder that `where` discards — its
            // gradient contribution is exactly the zeros `where` routes there.
            const indices = try ctx.allocator.alloc(usize, mask_values.len);
            defer ctx.allocator.free(indices);
            var slot: usize = 0;
            for (mask_values, indices) |mv, *index| {
                if (dtype_mod.isTruthy(Mask.dtype, mv)) {
                    index.* = slot;
                    slot += 1;
                } else {
                    index.* = 0;
                }
            }

            var gathered = try values.gather(ctx, values_tag, indices, values_tag);
            defer gathered.deinit();
            var dense = try gathered.split(ctx, values_tag, tags, self.shape());
            defer dense.deinit();
            return dense.where(ctx, mask, self);
        }

        /// Functional row accumulation (torch.index_add): a copy of `self`
        /// with `update`'s rows ADDED at `indices` along `tag` — unlike
        /// `setRows` this accumulates, and duplicate indices are allowed
        /// (each occurrence adds; torch semantics). `update` must match
        /// `self` except along `tag`, where it has `indices.len` rows.
        /// Differentiable in both: d/dself is the identity, d/dupdate
        /// gathers the addressed rows of the upstream gradient.
        pub fn indexAdd(self: *const Self, ctx: *ExecContext, comptime tag: Tag, indices: []const usize, update: *const Self) !Self {
            const add_axis = comptime axis(tag);
            var scattered = try ctx.scatterAdd(tag_rank, update.asRawTensor(), self.shape(), add_axis, indices);
            defer scattered.deinit();
            var value = try ctx.elementwise(.f32, .add, self.asRawTensor(), &scattered);
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad() or update.requiresGrad(), IndexAddBackward(tags, add_axis), .{ ctx.allocator, self.grad_state, update.grad_state, indices });
        }

        /// Read a same-tagged i64 index tensor into a host `[]usize`
        /// buffer (the argmax/topK/sort index convention; negatives and
        /// out-of-range values error with `IndexOutOfBounds`). Caller
        /// frees.
        fn hostIndexBuffer(ctx: *ExecContext, values: []const i64, limit: usize) ![]usize {
            const out = try ctx.allocator.alloc(usize, values.len);
            errdefer ctx.allocator.free(out);
            for (values, out) |v, *slot| {
                if (v < 0 or v >= limit) return TensorError.IndexOutOfBounds;
                slot.* = @intCast(v);
            }
            return out;
        }

        /// Elementwise gather along `tag` (torch.gather /
        /// np.take_along_axis): `out[.., i, ..] = self[.., indices[.., i,
        /// ..], ..]`. `indices` is a same-tagged i64 tensor (the
        /// argmax/topK/sort index convention — pairing directly with their
        /// outputs), contiguous, matching `self` on every other axis; the
        /// result takes `indices`' shape. Serial deterministic kernel;
        /// differentiable in `self` (the exact scatter-add adjoint;
        /// duplicate reads accumulate their gradients).
        pub fn takeAlongAxis(self: *const Self, ctx: *ExecContext, comptime tag: Tag, indices: anytype) !Self {
            comptime {
                if (TensorObject(@TypeOf(indices)).dtype != .i64)
                    @compileError("takeAlongAxis expects i64 indices (the argmax/topK/sort output dtype)");
            }
            const take_axis = comptime axis(tag);
            const idx_raw = indices.asRawTensor();
            const raw = self.asRawTensor();
            inline for (0..tensor_rank) |i| {
                if (i != take_axis and idx_raw.shape.at(i) != raw.shape.at(i)) return TensorError.ShapeMismatch;
            }
            const idx_buf = try hostIndexBuffer(ctx, try idx_raw.dataConstChecked(), raw.shape.at(take_axis));
            defer ctx.allocator.free(idx_buf);
            var value = try ctx.takeAlong(tag_rank, raw, take_axis, idx_buf, idx_raw.shape.at(take_axis));
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad(), TakeAlongBackward(tags, take_axis), .{ ctx.allocator, self.grad_state, &self.value, idx_buf });
        }

        /// Functional elementwise scatter-add along `tag`
        /// (torch.scatter_add): a copy of `self` with `src[.., i, ..]`
        /// added at row `indices[.., i, ..]` of `tag` — duplicate indices
        /// accumulate. `indices` follows the `takeAlongAxis` convention
        /// and must be shaped exactly like `src`; both match `self` on
        /// every other axis. Serial deterministic kernel; differentiable
        /// in both: d/dself is the identity, d/dsrc gathers the written
        /// slots (`takeAlongAxis` of the upstream gradient).
        pub fn scatterAdd(self: *const Self, ctx: *ExecContext, comptime tag: Tag, indices: anytype, src: *const Self) !Self {
            return scatterAlongImpl(self, ctx, tag, indices, src, true);
        }

        /// Functional elementwise scatter-overwrite along `tag`
        /// (torch.scatter with a tensor source): like `scatterAdd` but
        /// writing — duplicate indices resolve deterministically to the
        /// LAST write in row-major `src` order (torch leaves the order
        /// unspecified; this pins it). Differentiable in both: d/dself is
        /// zeroed at every written slot, d/dsrc gathers the written slots
        /// (the torch formula — on duplicates every writer receives the
        /// winning slot's gradient).
        pub fn scatter(self: *const Self, ctx: *ExecContext, comptime tag: Tag, indices: anytype, src: *const Self) !Self {
            return scatterAlongImpl(self, ctx, tag, indices, src, false);
        }

        fn scatterAlongImpl(self: *const Self, ctx: *ExecContext, comptime tag: Tag, indices: anytype, src: *const Self, comptime accumulate: bool) !Self {
            comptime {
                if (TensorObject(@TypeOf(indices)).dtype != .i64)
                    @compileError("scatter/scatterAdd expect i64 indices (the argmax/topK/sort output dtype)");
            }
            const scatter_axis = comptime axis(tag);
            const idx_raw = indices.asRawTensor();
            const src_raw = src.asRawTensor();
            const raw = self.asRawTensor();
            inline for (0..tensor_rank) |i| {
                if (idx_raw.shape.at(i) != src_raw.shape.at(i)) return TensorError.ShapeMismatch;
            }
            const idx_buf = try hostIndexBuffer(ctx, try idx_raw.dataConstChecked(), raw.shape.at(scatter_axis));
            defer ctx.allocator.free(idx_buf);
            var value = if (comptime accumulate)
                try ctx.scatterAddAlong(tag_rank, raw, src_raw, scatter_axis, idx_buf)
            else
                try ctx.scatterAlong(tag_rank, raw, src_raw, scatter_axis, idx_buf);
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad() or src.requiresGrad(), ScatterAlongBackward(tags, scatter_axis, accumulate), .{ ctx.allocator, self.grad_state, src.grad_state, idx_buf, src_raw.shape.at(scatter_axis) });
        }
    };
}
