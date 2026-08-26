//! VJP for the fused grouped causal attention op.

const std = @import("std");
const tensor_mod = @import("../../tensor.zig");
const exec_mod = @import("../../exec.zig");
const parallel = @import("../../parallel.zig");
const core = @import("../core.zig");

const RawTensor = tensor_mod.Tensor;
const ExecContext = exec_mod.ExecContext;
const GradState = core.GradState;

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
    scale_value: f32,
    window: usize,
    // false = bidirectional attention (block-diffusion canvas); the backward
    // re-masks to the same full key range the forward used.
    causal: bool,
    estimated_work: usize,

    pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
        const need_q = core.needs(self, 0);
        const need_k = core.needs(self, 1);
        const need_v = core.needs(self, 2);
        var grads = try ctx.groupedAttentionBackward(.{
            .q = &self.q,
            .k = &self.k,
            .v = &self.v,
            .gy = gy,
            .kv_head_for_head = self.kv_head_for_head,
            .scale = self.scale_value,
            .window = self.window,
            .causal = self.causal,
            .stats = if (self.row_stats.len == 0) null else self.row_stats,
            .need = .{ .q = need_q, .k = need_k, .v = need_v },
        });
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

    pub fn workEstimate(q_parent: ?*GradState, k_parent: ?*GradState, v_parent: ?*GradState, q: *const RawTensor, k: *const RawTensor) usize {
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
        allocator.free(self.kv_head_for_head);
        allocator.free(self.row_stats);
    }

    pub const vtable = core.recordVTable(Self);
};
