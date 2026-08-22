//! VJPs for loss heads: cross-entropy (+fused linear), distill,
//! mse/huber/bce/kl-div.

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

        pub fn init(
            self: *Self,
            allocator: std.mem.Allocator,
            x_parent: ?*GradState,
            weight_parent: ?*GradState,
            x: *const RawTensor,
            weight: *const RawTensor,
            logits: *const RawTensor,
            labels: []const usize,
            row_stats: []const f32,
        ) !void {
            var branches: usize = 0;
            if (x_parent != null) branches += 1;
            if (weight_parent != null) branches += 1;
            self.* = .{
                .parents = .{ x_parent, weight_parent },
                .x = try x.cloneView(),
                .weight = undefined,
                .logits = undefined,
                .labels = undefined,
                .row_stats = undefined,
                .estimated_work = std.math.mul(usize, x.len(), weight.shape.at(0) * branches) catch std.math.maxInt(usize),
            };
            errdefer self.x.deinit();
            self.weight = try weight.cloneView();
            errdefer self.weight.deinit();
            self.logits = try logits.cloneView();
            errdefer self.logits.deinit();
            self.labels = try allocator.dupe(usize, labels);
            errdefer allocator.free(self.labels);
            self.row_stats = try allocator.dupe(f32, row_stats);
        }

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const need_x = needs_grad.len > 0 and needs_grad[0];
            const need_weight = needs_grad.len > 1 and needs_grad[1];
            // The record exclusively owns its saved logits and the VJP
            // consumes them in place; the const-cast is the scheduler's
            // `*const` record pointer meeting that single-writer ownership.
            const mut_self: *Self = @constCast(self);
            if (self.consumed) return error.LinearCrossEntropyBackwardConsumed;
            mut_self.consumed = true;
            var grads = try ctx.linearCrossEntropyBackwardUpstream(
                &mut_self.x,
                &mut_self.weight,
                &mut_self.logits,
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

/// VJP record of the fused `linearDistillExt` (sparse-soft-target CE over
/// x·Wᵀ, entries only on the unique supervised rows). Saves the gathered
/// x rows, W, the selected-row logits, and the per-row {max, sum_exp}
/// stats. SINGLE-USE like `LinearCrossEntropyBackward`: the backward
/// consumes the saved logits in place and a repeat walk errors loudly.
/// Differentiable in BOTH operands.
pub const LinearDistillBackward = struct {
    parents: [2]?*GradState,
    x_sel: RawTensor,
    weight: RawTensor,
    logits: RawTensor,
    sel_rows: []usize,
    local_rows: []usize,
    classes: []usize,
    probs: []f32,
    row_stats: []f32,
    row_count: usize,
    options: exec_mod.LinearDistillOptions,
    estimated_work: usize,
    consumed: bool = false,

    const Self = @This();

    pub fn init(
        self: *Self,
        allocator: std.mem.Allocator,
        x_parent: ?*GradState,
        weight_parent: ?*GradState,
        x_sel: *const RawTensor,
        weight: *const RawTensor,
        logits: *const RawTensor,
        sel_rows: []const usize,
        local_rows: []const usize,
        classes: []const usize,
        probs: []const f32,
        row_stats: []const f32,
        row_count: usize,
        options: exec_mod.LinearDistillOptions,
    ) !void {
        var branches: usize = 0;
        if (x_parent != null) branches += 1;
        if (weight_parent != null) branches += 1;
        self.* = .{
            .parents = .{ x_parent, weight_parent },
            .x_sel = try x_sel.cloneView(),
            .weight = undefined,
            .logits = undefined,
            .sel_rows = undefined,
            .local_rows = undefined,
            .classes = undefined,
            .probs = undefined,
            .row_stats = undefined,
            .row_count = row_count,
            .options = options,
            .estimated_work = std.math.mul(usize, x_sel.len(), weight.shape.at(0) * @max(branches, 1)) catch std.math.maxInt(usize),
        };
        errdefer self.x_sel.deinit();
        self.weight = try weight.cloneView();
        errdefer self.weight.deinit();
        self.logits = try logits.cloneView();
        errdefer self.logits.deinit();
        self.sel_rows = try allocator.dupe(usize, sel_rows);
        errdefer allocator.free(self.sel_rows);
        self.local_rows = try allocator.dupe(usize, local_rows);
        errdefer allocator.free(self.local_rows);
        self.classes = try allocator.dupe(usize, classes);
        errdefer allocator.free(self.classes);
        self.probs = try allocator.dupe(f32, probs);
        errdefer allocator.free(self.probs);
        self.row_stats = try allocator.dupe(f32, row_stats);
    }

    pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
        const need_x = needs_grad.len > 0 and needs_grad[0];
        const need_weight = needs_grad.len > 1 and needs_grad[1];
        // Single-writer in-place consumption of the saved logits, as
        // LinearCrossEntropyBackward.
        const mut_self: *Self = @constCast(self);
        if (self.consumed) return error.LinearDistillBackwardConsumed;
        mut_self.consumed = true;
        var grads = try ctx.linearDistillBackwardUpstream(
            &mut_self.x_sel,
            &mut_self.weight,
            &mut_self.logits,
            self.sel_rows,
            self.row_count,
            self.local_rows,
            self.classes,
            self.probs,
            self.options,
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
        self.x_sel.deinit();
        self.weight.deinit();
        self.logits.deinit();
        allocator.free(self.sel_rows);
        allocator.free(self.local_rows);
        allocator.free(self.classes);
        allocator.free(self.probs);
        allocator.free(self.row_stats);
    }

    pub const vtable = core.recordVTable(Self);
};

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
        // `crossEntropyBackwardExStatsAxisRank`).
        row_stats: []f32,

        const Self = @This();

        pub fn init(
            self: *Self,
            allocator: std.mem.Allocator,
            parent: ?*GradState,
            logits: *const RawTensor,
            labels: []const usize,
            row_stats: []const f32,
        ) !void {
            self.* = .{
                .parents = .{parent},
                .logits = try logits.cloneView(),
                .labels = undefined,
                .row_stats = undefined,
            };
            errdefer self.logits.deinit();
            self.labels = try allocator.dupe(usize, labels);
            errdefer allocator.free(self.labels);
            self.row_stats = try allocator.dupe(f32, row_stats);
        }

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            if (needs_grad.len == 0 or !needs_grad[0]) return;
            // For `.mean`/`.sum` the upstream gy must be a scalar; for `.none`
            // it's the per-position gradient tensor (class axis removed).
            out[0] = try ctx.crossEntropyBackwardExUpstreamStatsAxisRank(rawRank(tags.len), &self.logits, axis, self.labels, options, gy, self.row_stats);
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

        pub fn init(
            self: *Self,
            allocator: std.mem.Allocator,
            input_parent: ?*GradState,
            target_parent: ?*GradState,
            input: *const RawTensor,
            target: *const RawTensor,
        ) !void {
            _ = allocator;
            self.* = .{
                .parents = .{ input_parent, target_parent },
                .input = try input.cloneView(),
                .target = undefined,
            };
            errdefer self.input.deinit();
            self.target = try target.cloneView();
        }

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            if (needs_grad.len > 0 and needs_grad[0]) {
                out[0] = try upstream_fn(ctx, &self.input, &self.target, options, gy, .input);
            }
            if (needs_grad.len > 1 and needs_grad[1]) {
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

pub fn MseLossBackward(comptime tags: anytype, comptime options: exec_mod.MseOptions) type {
    _ = tags;
    return ElementwiseLossBackward(exec_mod.MseOptions, options, ExecContext.mseBackwardUpstream);
}

pub fn HuberLossBackward(comptime tags: anytype, comptime options: exec_mod.HuberOptions) type {
    _ = tags;
    return ElementwiseLossBackward(exec_mod.HuberOptions, options, ExecContext.huberBackwardUpstream);
}

pub fn BceLossBackward(comptime tags: anytype, comptime options: exec_mod.BceOptions) type {
    _ = tags;
    return ElementwiseLossBackward(exec_mod.BceOptions, options, ExecContext.bceBackwardUpstream);
}

pub fn KlDivLossBackward(comptime tags: anytype, comptime options: exec_mod.KlDivOptions) type {
    _ = tags;
    return ElementwiseLossBackward(exec_mod.KlDivOptions, options, ExecContext.klDivBackwardUpstream);
}
