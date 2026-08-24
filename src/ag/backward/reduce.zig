//! VJPs for reductions, scans, and linear recurrence.

const std = @import("std");
const tensor_mod = @import("../../tensor.zig");
const exec_mod = @import("../../exec.zig");
const core = @import("../core.zig");
const tags_mod = @import("../../tags.zig");

const RawTensor = tensor_mod.Tensor;
const ExecContext = exec_mod.ExecContext;
const GradState = core.GradState;
const rawRank = tags_mod.rawRank;

const common = @import("common.zig");
const rawShapeArray = common.rawShapeArray;
const reduceGradientToTags = common.reduceGradientToTags;
const contiguousForRead = common.contiguousForRead;
const expandGradientToTags = common.expandGradientToTags;
const axisGeometry = common.axisGeometry;
const gateGradientByMask = common.gateGradientByMask;

pub fn SumBackward(comptime source_tags: anytype, comptime result_tags: anytype) type {
    return struct {
        parents: [1]?*GradState,
        source_shape: [rawRank(source_tags.len)]usize,

        const Self = @This();

        pub fn init(self: *Self, allocator: std.mem.Allocator, parent: ?*GradState, source: *const RawTensor) !void {
            _ = allocator;
            self.* = .{
                .parents = .{parent},
                .source_shape = rawShapeArray(source_tags, source),
            };
        }

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            if (needs_grad.len == 0 or !needs_grad[0]) return;
            out[0] = try expandGradientToTags(result_tags, source_tags, ctx, gy, self.source_shape);
        }

        pub const vtable = core.recordVTable(Self);
    };
}

pub fn MeanBackward(comptime source_tags: anytype, comptime result_tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [1]?*GradState,
        source_shape: [rawRank(source_tags.len)]usize,

        const Self = @This();

        pub fn init(self: *Self, allocator: std.mem.Allocator, parent: ?*GradState, source: *const RawTensor) !void {
            _ = allocator;
            self.* = .{
                .parents = .{parent},
                .source_shape = rawShapeArray(source_tags, source),
            };
        }

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            if (needs_grad.len == 0 or !needs_grad[0]) return;

            var expanded = try expandGradientToTags(result_tags, source_tags, ctx, gy, self.source_shape);
            defer expanded.deinit();
            out[0] = try ctx.scale(.f32, &expanded, 1 / @as(f32, @floatFromInt(self.source_shape[axis])));
        }

        pub const vtable = core.recordVTable(Self);
    };
}

/// VJP for a masked sum: the unmasked scatter, gated by the mask.
///
/// An element the mask excluded contributed nothing to the output, so it
/// receives nothing back. Lanes that selected nothing scatter a gradient no
/// element keeps, which is the same statement.
pub fn MaskedSumBackward(comptime source_tags: anytype, comptime result_tags: anytype, comptime mask_dtype: tensor_mod.DType) type {
    return struct {
        parents: [1]?*GradState,
        source_shape: [rawRank(source_tags.len)]usize,
        mask: tensor_mod.TensorOf(mask_dtype),

        const Self = @This();

        pub fn init(
            self: *Self,
            allocator: std.mem.Allocator,
            parent: ?*GradState,
            source: *const RawTensor,
            mask: *const tensor_mod.TensorOf(mask_dtype),
        ) !void {
            _ = allocator;
            self.* = .{
                .parents = .{parent},
                .source_shape = rawShapeArray(source_tags, source),
                .mask = try mask.cloneView(),
            };
        }

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            if (needs_grad.len == 0 or !needs_grad[0]) return;
            out[0] = try gateGradientByMask(mask_dtype, result_tags, source_tags, ctx, gy, &self.mask, self.source_shape);
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            _ = allocator;
            self.mask.deinit();
        }

        pub const vtable = core.recordVTable(Self);
    };
}

/// VJP for a masked mean: the masked scatter divided by the per-lane count of
/// selected elements, which the forward already computed.
///
/// A lane that selected nothing has count zero and produced the caller's
/// `empty` sentinel rather than a mean of the data, so no gradient flows out
/// of it: the divide would be 0/0, and the honest derivative of a constant is
/// zero. The stored counts carry the per-lane divisor so the backward never
/// recounts.
pub fn MaskedMeanBackward(comptime source_tags: anytype, comptime result_tags: anytype, comptime mask_dtype: tensor_mod.DType) type {
    return struct {
        parents: [1]?*GradState,
        source_shape: [rawRank(source_tags.len)]usize,
        mask: tensor_mod.TensorOf(mask_dtype),
        counts: RawTensor,

        const Self = @This();

        pub fn init(
            self: *Self,
            allocator: std.mem.Allocator,
            parent: ?*GradState,
            source: *const RawTensor,
            mask: *const tensor_mod.TensorOf(mask_dtype),
            counts: *const RawTensor,
        ) !void {
            _ = allocator;
            self.* = .{
                .parents = .{parent},
                .source_shape = rawShapeArray(source_tags, source),
                .mask = try mask.cloneView(),
                .counts = try counts.cloneView(),
            };
        }

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            if (needs_grad.len == 0 or !needs_grad[0]) return;

            // Divide at the RESULT shape (one value per lane) before
            // broadcasting, so the divide runs over the small tensor. A lane
            // that selected nothing produced the caller's `empty` constant,
            // not a mean of the data, so its gradient is zero.
            var gy_ready = try contiguousForRead(ctx, gy);
            defer gy_ready.deinit();
            var counts_ready = try contiguousForRead(ctx, &self.counts);
            defer counts_ready.deinit();

            var scaled = try ctx.empty(.f32, gy_ready.shape.slice());
            defer scaled.deinit();
            for (gy_ready.dataConst(), counts_ready.dataConst(), scaled.data()) |g, count, *dst| {
                dst.* = if (count == 0) 0 else g / count;
            }

            out[0] = try gateGradientByMask(mask_dtype, result_tags, source_tags, ctx, &scaled, &self.mask, self.source_shape);
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            _ = allocator;
            self.mask.deinit();
            self.counts.deinit();
        }

        pub const vtable = core.recordVTable(Self);
    };
}

/// VJP for `cumsum` over an axis: gx[i] = Σ_{j >= i} gy[j] along the axis —
/// the REVERSED cumulative (suffix) sum of the upstream gradient, computed by
/// the dedicated serial `cumsumReverse` exec helper (deterministic:
/// one serial pass per row, same order for any thread count).
pub fn CumsumBackward(comptime source_tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [1]?*GradState,

        const Self = @This();

        pub fn init(self: *Self, allocator: std.mem.Allocator, parent: ?*GradState) !void {
            _ = allocator;
            self.* = .{ .parents = .{parent} };
        }

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            _ = self;
            if (needs_grad.len == 0 or !needs_grad[0]) return;
            out[0] = try ctx.cumsumReverse(.f32, rawRank(source_tags.len), gy, axis);
        }

        pub const vtable = core.recordVTable(Self);
    };
}

/// VJP for `segmentSum`: each input row receives its segment's gradient row
/// (a broadcast along the segmented axis). Owns a copy of the offsets.
pub fn SegmentSumBackward(comptime source_tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [1]?*GradState,
        offsets: []usize,
        n: usize,

        const Self = @This();

        pub fn init(self: *Self, allocator: std.mem.Allocator, parent: ?*GradState, offsets: []const usize, n: usize) !void {
            self.* = .{
                .parents = .{parent},
                .offsets = try allocator.dupe(usize, offsets),
                .n = n,
            };
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.offsets);
        }

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            if (needs_grad.len == 0 or !needs_grad[0]) return;
            out[0] = try ctx.segmentBroadcast(rawRank(source_tags.len), gy, axis, self.offsets, self.n);
        }

        pub const vtable = core.recordVTable(Self);
    };
}

/// VJP for `linearRecurrence` (`h_t = a_t·h_{t-1} + b_t` along an axis):
/// one exec reverse scan produces the input gradient `gb` (the reverse
/// recurrence `gh_t = a_{t+1}·gh_{t+1} + gy_t`), the full-shape decay
/// gradient `da_t = gh_t·h_{t-1}` — reduced back onto the decay's own
/// tags/shape by the pointwise broadcast-backward rule — and the initial
/// state's gradient `a_0·gh_0`. Saves the aligned decay view and the
/// forward OUTPUT (h), not the input.
pub fn LinearRecurrenceBackward(comptime source_tags: anytype, comptime decay_tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [3]?*GradState,
        a_view: RawTensor,
        h_value: RawTensor,
        initial_value: ?RawTensor,
        decay_shape: [rawRank(decay_tags.len)]usize,

        const Self = @This();

        pub fn init(
            self: *Self,
            allocator: std.mem.Allocator,
            b_parent: ?*GradState,
            decay_parent: ?*GradState,
            initial_parent: ?*GradState,
            a_view: *const RawTensor,
            h: *const RawTensor,
            initial: ?*const RawTensor,
            decay: *const RawTensor,
        ) !void {
            _ = allocator;
            self.* = .{
                .parents = .{ b_parent, decay_parent, initial_parent },
                .a_view = try a_view.cloneView(),
                .h_value = undefined,
                .initial_value = null,
                .decay_shape = rawShapeArray(decay_tags, decay),
            };
            errdefer self.a_view.deinit();
            self.h_value = try h.cloneView();
            errdefer self.h_value.deinit();
            if (initial) |ini| self.initial_value = try ini.cloneView();
        }

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const need_b = needs_grad.len > 0 and needs_grad[0];
            const need_a = needs_grad.len > 1 and needs_grad[1];
            const need_init = needs_grad.len > 2 and needs_grad[2];
            if (!need_b and !need_a and !need_init) return;
            const initial_ptr: ?*const RawTensor = if (self.initial_value) |*ini| ini else null;
            var grads = try ctx.linearRecurrenceBackward(rawRank(source_tags.len), gy, &self.a_view, &self.h_value, initial_ptr, axis, need_a, need_init);
            errdefer grads.gb.deinit();
            errdefer if (grads.da) |*t| t.deinit();
            errdefer if (grads.dinitial) |*t| t.deinit();
            if (need_a) {
                var da = grads.da.?;
                grads.da = null;
                defer da.deinit();
                out[1] = try reduceGradientToTags(source_tags, decay_tags, ctx, &da, self.decay_shape);
            }
            if (need_init) {
                out[2] = grads.dinitial.?;
                grads.dinitial = null;
            }
            if (need_b) {
                out[0] = grads.gb;
            } else {
                grads.gb.deinit();
            }
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            _ = allocator;
            self.a_view.deinit();
            self.h_value.deinit();
            if (self.initial_value) |*ini| ini.deinit();
        }

        pub const vtable = core.recordVTable(Self);
    };
}

pub fn ProdBackward(comptime source_tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [1]?*GradState,
        input: RawTensor,

        const Self = @This();
        const rank = rawRank(source_tags.len);

        pub fn init(self: *Self, allocator: std.mem.Allocator, parent: ?*GradState, input: *const RawTensor) !void {
            _ = allocator;
            self.* = .{
                .parents = .{parent},
                .input = try input.cloneView(),
            };
        }

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            if (needs_grad.len == 0 or !needs_grad[0]) return;

            var x_ready = try contiguousForRead(ctx, &self.input);
            defer x_ready.deinit();
            const x = x_ready.dataConst();
            var gy_ready = try contiguousForRead(ctx, gy);
            defer gy_ready.deinit();
            const g = gy_ready.dataConst();

            const source_shape = rawShapeArray(source_tags, &self.input);
            var gx = try ctx.empty(.f32, source_shape);
            errdefer gx.deinit();
            const gxd = gx.data();

            // torch.prod zero-handling: no zeros → g·(prod/x_i); exactly
            // one zero → its slot gets g·Π(nonzero), the rest 0; two or
            // more zeros → all 0. Computed division-free per row.
            const axis_dim = source_shape[axis];
            const geo = axisGeometry(rank, source_shape, axis);
            for (0..geo.outer) |outer_i| {
                const base = outer_i * axis_dim * geo.inner;
                for (0..geo.inner) |inner_i| {
                    var zero_count: usize = 0;
                    var nonzero_prod: f32 = 1;
                    for (0..axis_dim) |i| {
                        const v = x[base + i * geo.inner + inner_i];
                        if (v == 0) zero_count += 1 else nonzero_prod *= v;
                    }
                    const upstream = g[outer_i * geo.inner + inner_i];
                    for (0..axis_dim) |i| {
                        const offset = base + i * geo.inner + inner_i;
                        const v = x[offset];
                        gxd[offset] = switch (zero_count) {
                            0 => upstream * (nonzero_prod / v),
                            1 => if (v == 0) upstream * nonzero_prod else 0,
                            else => 0,
                        };
                    }
                }
            }
            out[0] = gx;
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            _ = allocator;
            self.input.deinit();
        }

        pub const vtable = core.recordVTable(Self);
    };
}

pub fn CumprodBackward(comptime source_tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [1]?*GradState,
        input: RawTensor,
        output: RawTensor,

        const Self = @This();
        const rank = rawRank(source_tags.len);

        pub fn init(self: *Self, allocator: std.mem.Allocator, parent: ?*GradState, input: *const RawTensor, output: *const RawTensor) !void {
            _ = allocator;
            var in_view = try input.cloneView();
            errdefer in_view.deinit();
            self.* = .{
                .parents = .{parent},
                .input = in_view,
                .output = try output.cloneView(),
            };
        }

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            if (needs_grad.len == 0 or !needs_grad[0]) return;

            var x_ready = try contiguousForRead(ctx, &self.input);
            defer x_ready.deinit();
            const x = x_ready.dataConst();
            var y_ready = try contiguousForRead(ctx, &self.output);
            defer y_ready.deinit();
            const y = y_ready.dataConst();
            var gy_ready = try contiguousForRead(ctx, gy);
            defer gy_ready.deinit();
            const g = gy_ready.dataConst();

            const source_shape = rawShapeArray(source_tags, &self.input);
            var gx = try ctx.empty(.f32, source_shape);
            errdefer gx.deinit();
            const gxd = gx.data();

            // Zero-free rows use the O(n) reverse-scan closed form
            // grad_i = (Σ_{j≥i} g_j·y_j)/x_i; rows containing a zero fall
            // back to the exact division-free O(n²) expansion
            // grad_i = Σ_{j≥i} g_j·Π_{k≤j, k≠i} x_k (torch semantics).
            const axis_dim = source_shape[axis];
            const geo = axisGeometry(rank, source_shape, axis);
            for (0..geo.outer) |outer_i| {
                const base = outer_i * axis_dim * geo.inner;
                for (0..geo.inner) |inner_i| {
                    var has_zero = false;
                    for (0..axis_dim) |i| {
                        if (x[base + i * geo.inner + inner_i] == 0) {
                            has_zero = true;
                            break;
                        }
                    }
                    if (!has_zero) {
                        var suffix: f32 = 0;
                        var i: usize = axis_dim;
                        while (i > 0) {
                            i -= 1;
                            const offset = base + i * geo.inner + inner_i;
                            suffix += g[offset] * y[offset];
                            gxd[offset] = suffix / x[offset];
                        }
                    } else {
                        for (0..axis_dim) |i| {
                            var prefix: f32 = 1;
                            for (0..i) |k| prefix *= x[base + k * geo.inner + inner_i];
                            var run = prefix;
                            var acc = g[base + i * geo.inner + inner_i] * run;
                            for (i + 1..axis_dim) |j| {
                                run *= x[base + j * geo.inner + inner_i];
                                acc += g[base + j * geo.inner + inner_i] * run;
                            }
                            gxd[base + i * geo.inner + inner_i] = acc;
                        }
                    }
                }
            }
            out[0] = gx;
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            _ = allocator;
            self.input.deinit();
            self.output.deinit();
        }

        pub const vtable = core.recordVTable(Self);
    };
}
