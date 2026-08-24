//! f32 tensor methods: top-k, sort, router top-k. A mixin over the ag
//! FloatTensor struct; aliased back onto it in ../../tensor.zig.

const tensor_mod = @import("../../../tensor.zig");
const exec_mod = @import("../../../exec.zig");
const tags_mod = @import("../../../tags.zig");
const backward_topk = @import("../../backward/topk.zig");

const RawTensor = tensor_mod.Tensor;
const ExecContext = exec_mod.ExecContext;
const Tag = tags_mod.Tag;
const replaceTag = tags_mod.replaceTag;
const TopKBackward = backward_topk.TopKBackward;

pub fn Ops(comptime Self: type) type {
    return struct {
        const tags = Self.axis_tags;
        const tag_rank = Self.tag_count;
        const tag_count = Self.tag_count;
        const axis = Self.axis;
        const ag_tensor = Self.ag_root;
        const Tensor = ag_tensor.Tensor;
        const TopKResult = ag_tensor.TopKResult;
        const plumbing = @import("../plumbing.zig").Mod(ag_tensor);
        const finishOp = plumbing.finishOp;

        pub fn topK(self: *const Self, ctx: *ExecContext, comptime tag: Tag, k: usize, comptime out_tag: Tag) !TopKResult(replaceTag(tags, tag, out_tag)) {
            const result_tags = replaceTag(tags, tag, out_tag);
            const raw = try ctx.topK(tag_rank, self.asRawTensor(), axis(tag), k);
            var raw_values: ?RawTensor = raw.values;
            var raw_indices: ?tensor_mod.TensorOf(.i64) = raw.indices;
            errdefer if (raw_values) |*value| value.deinit();
            errdefer if (raw_indices) |*value| value.deinit();
            var values = try finishOp(result_tags, ctx, raw_values.?, self.requiresGrad(), TopKBackward(tags, axis(tag)), .{ ctx.allocator, self.grad_state, &self.value, &raw_indices.? });
            raw_values = null;
            errdefer values.deinit();
            var indices = try Tensor(.{ .dtype = .i64, .tags = result_tags }).fromTensor(ctx, raw_indices.?);
            raw_indices = null;
            errdefer indices.deinit();
            return .{ .values = values, .indices = indices };
        }

        /// Full sort along `tag` (torch.sort): values + the source index of
        /// each output position, both input-shaped. UNSTABLE sort — equal
        /// values keep no particular relative order (torch.sort is also
        /// unstable by default, but the tie ORDER may differ). NaN sorts
        /// LAST regardless of direction (documented divergence from
        /// torch.sort, which puts NaN first when descending — see
        /// `sort`). Values are differentiable (the gradient scatters
        /// back through the saved indices, the topK VJP); indices are a
        /// constant i64 tensor (exact for any axis length, the repo-wide
        /// index convention).
        pub fn sort(self: *const Self, ctx: *ExecContext, comptime tag: Tag, descending: bool) !TopKResult(tags) {
            const raw = try ctx.sort(tag_rank, self.asRawTensor(), axis(tag), descending);
            var raw_values: ?RawTensor = raw.values;
            var raw_indices: ?tensor_mod.TensorOf(.i64) = raw.indices;
            errdefer if (raw_values) |*value| value.deinit();
            errdefer if (raw_indices) |*value| value.deinit();
            var values = try finishOp(tags, ctx, raw_values.?, self.requiresGrad(), TopKBackward(tags, axis(tag)), .{ ctx.allocator, self.grad_state, &self.value, &raw_indices.? });
            raw_values = null;
            errdefer values.deinit();
            var indices = try Tensor(.{ .dtype = .i64, .tags = tags }).fromTensor(ctx, raw_indices.?);
            raw_indices = null;
            errdefer indices.deinit();
            return .{ .values = values, .indices = indices };
        }

        /// The indices arm of `sort` alone (torch.argsort): the source index
        /// of each sorted position as a constant i64 tensor (no grad; exact
        /// for any axis length). Same unstable-sort and NaN-last contract
        /// as `sort`.
        pub fn argsort(self: *const Self, ctx: *ExecContext, comptime tag: Tag, descending: bool) !Tensor(.{ .dtype = .i64, .tags = tags }) {
            var raw = try ctx.sort(tag_rank, self.asRawTensor(), axis(tag), descending);
            raw.values.deinit();
            var raw_indices: ?tensor_mod.TensorOf(.i64) = raw.indices;
            errdefer if (raw_indices) |*value| value.deinit();
            return Tensor(.{ .dtype = .i64, .tags = tags }).fromTensor(ctx, raw_indices.?);
        }

        pub fn routerTopK(
            self: *const Self,
            ctx: *ExecContext,
            comptime expert_tag: Tag,
            k: usize,
            options: exec_mod.RouterTopKOptions,
            selected: []usize,
            weights: []f32,
        ) !void {
            comptime {
                if (tag_rank != 2) @compileError("routerTopK currently requires rank-2 [row, expert] logits");
                if (axis(expert_tag) != 1) @compileError("routerTopK requires the expert tag on the last axis");
            }
            if (self.requiresGrad()) return error.UnsupportedGradient;
            return ctx.routerTopK(self.asRawTensor(), k, options, selected, weights);
        }
    };
}
