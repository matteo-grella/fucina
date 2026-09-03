//! VJPs for loss heads: cross-entropy (+fused linear), mse/huber/bce/kl-div.

const std = @import("std");
const tensor_mod = @import("../../tensor.zig");
const exec_mod = @import("../../exec.zig");
const core = @import("../core.zig");
const AgError = core.AgError;
const tags_mod = @import("../../tags.zig");

const RawTensor = tensor_mod.Tensor;
const ExecContext = exec_mod.ExecContext;
const GradState = core.GradState;
const rawRank = tags_mod.rawRank;

/// VJP record of the fused `linearCrossEntropy` (loss = CE(x·Wᵀ, labels)):
/// saves x, W, the forward's logits, and the per-row {max, sum_exp} stats.
/// SINGLE-USE: the backward overwrites the saved logits in place with the
/// logit gradient (the record owns the buffer exclusively), so the full
/// [rows, classes] gradient never costs a second buffer; a repeat backward
/// over the same record errors loudly instead of computing garbage (the
/// scheduler otherwise permits re-walking a retained graph).
/// Differentiable in BOTH operands.
pub fn LinearCrossEntropyBackward(comptime options: exec_mod.CrossEntropyOptions) type {
    return struct {
        parents: [2]?*GradState,
        x: RawTensor,
        weight: RawTensor,
        logits: RawTensor,
        labels: []usize,
        row_stats: []f32,
        estimated_work: usize,
        consumed: bool = false,

        const Self = @This();

        /// rows x vocab per grad-requiring operand, saturating.
        pub fn workEstimate(x_parent: ?*GradState, weight_parent: ?*GradState, x: *const RawTensor, weight: *const RawTensor) usize {
            var branches: usize = 0;
            if (x_parent != null) branches += 1;
            if (weight_parent != null) branches += 1;
            return std.math.mul(usize, x.len(), weight.shape.at(0) * branches) catch std.math.maxInt(usize);
        }

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
            const need_x = core.needs(self, 0);
            const need_weight = core.needs(self, 1);
            // The record exclusively owns its saved logits and the VJP
            // consumes them in place (single-writer ownership).
            if (self.consumed) return AgError.BackwardAlreadyRun;
            self.consumed = true;
            var grads = try ctx.linearCrossEntropyBackwardUpstream(
                &self.x,
                &self.weight,
                &self.logits,
                self.labels,
                options,
                gy,
                self.row_stats,
                need_x,
                need_weight,
            );
            defer grads.deinit();
            if (need_x) {
                out[0] = grads.dx.?;
                grads.dx = null;
            }
            if (need_weight) {
                out[1] = grads.dweight.?;
                grads.dweight = null;
            }
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            self.x.deinit();
            self.weight.deinit();
            self.logits.deinit();
            allocator.free(self.labels);
            allocator.free(self.row_stats);
        }

        pub const vtable = core.recordVTable(Self);
    };
}

pub fn CrossEntropyBackward(comptime tags: anytype, comptime axis: usize) type {
    return CrossEntropyExtBackward(tags, axis, .{});
}

pub fn CrossEntropyExtBackward(comptime tags: anytype, comptime axis: usize, comptime options: exec_mod.CrossEntropyOptions) type {
    return struct {
        parents: [1]?*GradState,
        logits: RawTensor,
        labels: []usize,
        // Forward-saved per-position {max, sum_exp} (8 bytes per position):
        // the backward emits final gradients in ONE pass over the logits,
        // bitwise identical to recomputing the statistics (see
        // `CrossEntropyOptions.row_stats`).
        row_stats: []f32,

        const Self = @This();

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
            if (!core.needs(self, 0)) return;
            // For `.mean`/`.sum` the upstream gy must be a scalar; for `.none`
            // it's the per-position gradient tensor (class axis removed).
            var stats_options = options;
            stats_options.row_stats = self.row_stats;
            out[0] = try ctx.crossEntropyBackward(rawRank(tags.len), &self.logits, axis, self.labels, stats_options, .{ .tensor = gy });
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            self.logits.deinit();
            allocator.free(self.labels);
            allocator.free(self.row_stats);
        }

        pub const vtable = core.recordVTable(Self);
    };
}

/// Shared two-parent VJP template for the whole-tensor elementwise losses
/// (`mseLoss`/`huberLoss`/`bceLoss`/`klDivLoss`): saves views of input and
/// target and routes the upstream gradient (scalar for `.mean`/`.sum`,
/// per-element for `.none`) through the loss's `*BackwardUpstream` exec arm
/// once per operand that needs a gradient.
fn ElementwiseLossBackward(comptime Options: type, comptime options: Options, comptime upstream_fn: anytype) type {
    return struct {
        parents: [2]?*GradState,
        input: RawTensor,
        target: RawTensor,

        const Self = @This();

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
            if (core.needs(self, 0)) {
                out[0] = try upstream_fn(ctx, &self.input, &self.target, options, gy, .input);
            }
            if (core.needs(self, 1)) {
                out[1] = try upstream_fn(ctx, &self.input, &self.target, options, gy, .target);
            }
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            _ = allocator;
            self.input.deinit();
            self.target.deinit();
        }

        pub const vtable = core.recordVTable(Self);
    };
}

pub fn MseLossBackward(comptime options: exec_mod.MseOptions) type {
    return ElementwiseLossBackward(exec_mod.MseOptions, options, ExecContext.mseBackwardUpstream);
}

pub fn HuberLossBackward(comptime options: exec_mod.HuberOptions) type {
    return ElementwiseLossBackward(exec_mod.HuberOptions, options, ExecContext.huberBackwardUpstream);
}

pub fn BceLossBackward(comptime options: exec_mod.BceOptions) type {
    return ElementwiseLossBackward(exec_mod.BceOptions, options, ExecContext.bceBackwardUpstream);
}

pub fn KlDivLossBackward(comptime options: exec_mod.KlDivOptions) type {
    return ElementwiseLossBackward(exec_mod.KlDivOptions, options, ExecContext.klDivBackwardUpstream);
}
