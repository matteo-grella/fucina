//! VJPs for statistics ops: var/standardize and (masked) min/max.

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
const contiguousForRead = common.contiguousForRead;

/// VJP for a masked extremum: the unmasked first-winner scatter, skipping
/// lanes whose index is the -1 "nothing selected" sentinel.
pub fn MaskedMinMaxBackward(comptime source_tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [1]?*GradState,
        source_shape: [rawRank(source_tags.len)]usize,
        indices: tensor_mod.TensorOf(.i64),

        const Self = @This();

        pub fn init(
            self: *Self,
            allocator: std.mem.Allocator,
            parent: ?*GradState,
            source: *const RawTensor,
            indices: *const tensor_mod.TensorOf(.i64),
        ) !void {
            _ = allocator;
            self.* = .{
                .parents = .{parent},
                .source_shape = rawShapeArray(source_tags, source),
                .indices = try indices.cloneView(),
            };
        }

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            if (needs_grad.len == 0 or !needs_grad[0]) return;

            const rank = comptime rawRank(source_tags.len);
            var gy_ready = try contiguousForRead(ctx, gy);
            defer gy_ready.deinit();
            const idxd = self.indices.dataConst();

            var gx = try ctx.zeros(.f32, self.source_shape[0..]);
            errdefer gx.deinit();
            const gyd = gy_ready.dataConst();
            const gxd = gx.data();

            const axis_dim = self.source_shape[axis];
            var inner: usize = 1;
            inline for (axis + 1..rank) |i| inner *= self.source_shape[i];
            var outer: usize = 1;
            inline for (0..axis) |i| outer *= self.source_shape[i];

            for (0..outer) |outer_i| {
                const gx_base = outer_i * axis_dim * inner;
                for (0..inner) |inner_i| {
                    const flat = outer_i * inner + inner_i;
                    const raw_index = idxd[flat];
                    if (raw_index < 0) continue; // empty lane: nothing participated
                    const index: usize = @intCast(raw_index);
                    gxd[gx_base + index * inner + inner_i] = gyd[flat];
                }
            }
            out[0] = gx;
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            _ = allocator;
            self.indices.deinit();
        }

        pub const vtable = core.recordVTable(Self);
    };
}

/// VJP for variance over an axis: dx = gy·2(x−μ)/(N−ddof) with μ the row
/// mean, recomputed here from the saved input.
pub fn VarBackward(comptime source_tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [1]?*GradState,
        input: RawTensor,
        ddof: u1,

        const Self = @This();

        pub fn init(self: *Self, allocator: std.mem.Allocator, parent: ?*GradState, input: *const RawTensor, ddof: u1) !void {
            _ = allocator;
            self.* = .{
                .parents = .{parent},
                .input = try input.cloneView(),
                .ddof = ddof,
            };
        }

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            if (needs_grad.len == 0 or !needs_grad[0]) return;

            const rank = comptime rawRank(source_tags.len);
            var x_ready = try contiguousForRead(ctx, &self.input);
            defer x_ready.deinit();
            var gy_ready = try contiguousForRead(ctx, gy);
            defer gy_ready.deinit();

            const source_shape = rawShapeArray(source_tags, &self.input);
            var gx = try ctx.empty(.f32, source_shape);
            errdefer gx.deinit();
            const xd = x_ready.dataConst();
            const gyd = gy_ready.dataConst();
            const gxd = gx.data();

            const axis_dim = source_shape[axis];
            var inner: usize = 1;
            inline for (axis + 1..rank) |i| inner *= source_shape[i];
            var outer: usize = 1;
            inline for (0..axis) |i| outer *= source_shape[i];

            const inv_axis_dim = 1 / @as(f32, @floatFromInt(axis_dim));
            const scale_base = 2 / (@as(f32, @floatFromInt(axis_dim)) - @as(f32, @floatFromInt(self.ddof)));
            for (0..outer) |outer_i| {
                const base = outer_i * axis_dim * inner;
                for (0..inner) |inner_i| {
                    var sum: f32 = 0;
                    for (0..axis_dim) |axis_i| {
                        sum += xd[base + axis_i * inner + inner_i];
                    }
                    const mean_value = sum * inv_axis_dim;
                    const upstream = gyd[outer_i * inner + inner_i] * scale_base;
                    for (0..axis_dim) |axis_i| {
                        const offset = base + axis_i * inner + inner_i;
                        gxd[offset] = upstream * (xd[offset] - mean_value);
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

pub fn StandardizeBackward(comptime tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [1]?*GradState,
        input: RawTensor,
        valid_len: ?usize,
        options: exec_mod.StandardizeOptions,

        const Self = @This();

        pub fn init(
            self: *Self,
            allocator: std.mem.Allocator,
            parent: ?*GradState,
            input: *const RawTensor,
            valid_len: ?usize,
            options: exec_mod.StandardizeOptions,
        ) !void {
            _ = allocator;
            self.* = .{
                .parents = .{parent},
                .input = try input.cloneView(),
                .valid_len = valid_len,
                .options = options,
            };
        }

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            if (needs_grad.len == 0 or !needs_grad[0]) return;
            out[0] = try ctx.standardizeBackward(rawRank(tags.len), &self.input, gy, axis, self.valid_len, self.options);
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            _ = allocator;
            self.input.deinit();
        }

        pub const vtable = core.recordVTable(Self);
    };
}

/// VJP for max/min over an axis: the gradient flows ONLY to the first
/// occurrence of the extremum along the axis — the index captured by the
/// forward kernel (strict comparison tie-break), matching PyTorch's
/// single-index routing for torch.max/torch.min over a dim.
pub fn MinMaxBackward(comptime source_tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [1]?*GradState,
        source_shape: [rawRank(source_tags.len)]usize,
        // First-extremum indices from the forward pass (out-shaped, i64 like
        // argmax/topK indices) — stored rather than recomputed.
        indices: tensor_mod.TensorOf(.i64),

        const Self = @This();

        pub fn init(
            self: *Self,
            allocator: std.mem.Allocator,
            parent: ?*GradState,
            source: *const RawTensor,
            indices: *const tensor_mod.TensorOf(.i64),
        ) !void {
            _ = allocator;
            self.* = .{
                .parents = .{parent},
                .source_shape = rawShapeArray(source_tags, source),
                .indices = try indices.cloneView(),
            };
        }

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            if (needs_grad.len == 0 or !needs_grad[0]) return;

            const rank = comptime rawRank(source_tags.len);
            var gy_ready = try contiguousForRead(ctx, gy);
            defer gy_ready.deinit();
            // The saved indices are a view of the forward kernel's freshly
            // allocated (contiguous) i64 tensor.
            const idxd = self.indices.dataConst();

            var gx = try ctx.zeros(.f32, self.source_shape[0..]);
            errdefer gx.deinit();
            const gyd = gy_ready.dataConst();
            const gxd = gx.data();

            const axis_dim = self.source_shape[axis];
            var inner: usize = 1;
            inline for (axis + 1..rank) |i| inner *= self.source_shape[i];
            var outer: usize = 1;
            inline for (0..axis) |i| outer *= self.source_shape[i];

            for (0..outer) |outer_i| {
                const gx_base = outer_i * axis_dim * inner;
                for (0..inner) |inner_i| {
                    const flat = outer_i * inner + inner_i;
                    const index: usize = @intCast(idxd[flat]);
                    gxd[gx_base + index * inner + inner_i] = gyd[flat];
                }
            }
            out[0] = gx;
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            _ = allocator;
            self.indices.deinit();
        }

        pub const vtable = core.recordVTable(Self);
    };
}
