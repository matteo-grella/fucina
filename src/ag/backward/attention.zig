//! VJP for the fused grouped causal attention op.

const std = @import("std");
const backend_ops = @import("../../backend.zig").ops;
const backend_quant = @import("../../backend.zig").quantized_matmul;
const tensor_mod = @import("../../tensor.zig");
const dtype_mod = @import("../../dtype.zig");
const exec_mod = @import("../../exec.zig");
const parallel = @import("../../parallel.zig");
const tag_ops = @import("../../tagged.zig");
const core = @import("../core.zig");
const tags_mod = @import("../../tags.zig");
const vector_primitives = @import("../../backend/vector/primitives.zig");

const RawTensor = tensor_mod.Tensor;
const ExecContext = exec_mod.ExecContext;
const GradState = core.GradState;
const Tag = tags_mod.Tag;
const inserted_axis = tags_mod.inserted_axis;
const rawRank = tags_mod.rawRank;
const tagIndex = tags_mod.tagIndex;
const removeTags = tags_mod.removeTags;
const dotResultTags = tags_mod.dotResultTags;
const pointwiseResultTags = tags_mod.pointwiseResultTags;
const intersectTags = tags_mod.intersectTags;
const tagsEqual = tags_mod.tagsEqual;

const common = @import("common.zig");
const PointwiseOp = tag_ops.PointwiseOp;
const rawShapeArray = common.rawShapeArray;
const rawShapeArrayOf = common.rawShapeArrayOf;
const rawStrideArray = common.rawStrideArray;
const taggedShapeArray = common.taggedShapeArray;
const tagsDifference = common.tagsDifference;
const tagsDifferenceLen = common.tagsDifferenceLen;
const reduceGradientToTags = common.reduceGradientToTags;
const contiguousForRead = common.contiguousForRead;
const expandGradientToTags = common.expandGradientToTags;
const contiguousForReadTyped = common.contiguousForReadTyped;
const axisGeometry = common.axisGeometry;
const gateGradientByMask = common.gateGradientByMask;
const cloneInverseRopeTable = common.cloneInverseRopeTable;

pub const GroupedCausalAttentionBackward = struct {
    const Self = @This();

    parents: [3]?*GradState,
    q: RawTensor,
    k: RawTensor,
    v: RawTensor,
    kv_head_for_head: []usize,
    // Forward-saved per-(head, query) softmax {max, sum_exp} pairs (8 bytes
    // per row; empty = none): the backward's tiled route rebuilds this
    // forward's probabilities in ONE pass instead of the max/sum recompute
    // (see `groupedCausalAttentionBackwardTiles`).
    row_stats: []f32,
    // The forward's output (refcounted view, no copy): kept on the record
    // for compatibility; the tiled route derives the softmax-backward row
    // dot from its own panels and does not read it.
    out: RawTensor,
    scale_value: f32,
    window: usize,
    // false = bidirectional attention (block-diffusion canvas); the backward
    // re-masks to the same full key range the forward used.
    causal: bool,
    estimated_work: usize,

    pub fn init(
        self: *GroupedCausalAttentionBackward,
        allocator: std.mem.Allocator,
        q_parent: ?*GradState,
        k_parent: ?*GradState,
        v_parent: ?*GradState,
        q: *const RawTensor,
        k: *const RawTensor,
        v: *const RawTensor,
        kv_head_for_head: []const usize,
        scale_value: f32,
        window: usize,
        causal: bool,
        row_stats: []const f32,
        out: *const RawTensor,
    ) !void {
        self.* = .{
            .parents = .{ q_parent, k_parent, v_parent },
            .q = try q.cloneView(),
            .k = undefined,
            .v = undefined,
            .kv_head_for_head = undefined,
            .row_stats = undefined,
            .out = undefined,
            .scale_value = scale_value,
            .window = window,
            .causal = causal,
            .estimated_work = workEstimate(q_parent, k_parent, v_parent, q, k),
        };
        errdefer self.q.deinit();
        self.k = try k.cloneView();
        errdefer self.k.deinit();
        self.v = try v.cloneView();
        errdefer self.v.deinit();
        self.out = try out.cloneView();
        errdefer self.out.deinit();
        self.kv_head_for_head = try allocator.dupe(usize, kv_head_for_head);
        errdefer allocator.free(self.kv_head_for_head);
        self.row_stats = try allocator.dupe(f32, row_stats);
    }

    pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
        const need_q = needs_grad.len > 0 and needs_grad[0];
        const need_k = needs_grad.len > 1 and needs_grad[1];
        const need_v = needs_grad.len > 2 and needs_grad[2];
        var grads = try ctx.groupedCausalAttentionBackward(
            &self.q,
            &self.k,
            &self.v,
            gy,
            self.kv_head_for_head,
            self.scale_value,
            self.window,
            self.causal,
            if (self.row_stats.len == 0) null else self.row_stats,
            if (self.row_stats.len == 0) null else &self.out,
            need_q,
            need_k,
            need_v,
        );
        defer grads.deinit();
        if (need_q) {
            out[0] = grads.q.?;
            grads.q = null;
        }
        if (need_k) {
            out[1] = grads.k.?;
            grads.k = null;
        }
        if (need_v) {
            out[2] = grads.v.?;
            grads.v = null;
        }
    }

    fn workEstimate(q_parent: ?*GradState, k_parent: ?*GradState, v_parent: ?*GradState, q: *const RawTensor, k: *const RawTensor) usize {
        var branches: usize = 0;
        if (q_parent != null) branches += 1;
        if (k_parent != null) branches += 1;
        if (v_parent != null) branches += 1;
        if (branches == 0) return 0;
        const q_seq = q.shape.at(0);
        const heads = q.shape.at(1);
        const d = q.shape.at(2);
        const kv_seq = k.shape.at(0);
        const base = parallel.saturatedMul3(q_seq, heads, kv_seq);
        return std.math.mul(usize, base, d * branches) catch std.math.maxInt(usize);
    }

    pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
        self.q.deinit();
        self.k.deinit();
        self.v.deinit();
        self.out.deinit();
        allocator.free(self.kv_head_for_head);
        allocator.free(self.row_stats);
    }

    pub const vtable = core.recordVTable(Self);
};
