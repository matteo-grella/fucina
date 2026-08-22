//! VJP for top-k selection.

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

pub fn TopKBackward(comptime source_tags: anytype, comptime axis: usize) type {
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
            // The saved indices are a view of the forward kernel's freshly
            // allocated (contiguous) i64 tensor.
            const idxd = self.indices.dataConst();

            var gx = try ctx.zeros(self.source_shape[0..]);
            errdefer gx.deinit();
            const gyd = gy_ready.dataConst();
            const gxd = gx.data();

            const axis_dim = self.source_shape[axis];
            const k = gy_ready.shape.at(axis);
            var inner: usize = 1;
            inline for (axis + 1..rank) |i| inner *= self.source_shape[i];
            var outer: usize = 1;
            inline for (0..axis) |i| outer *= self.source_shape[i];

            for (0..outer) |outer_i| {
                const gy_base = outer_i * k * inner;
                const gx_base = outer_i * axis_dim * inner;
                for (0..k) |slot| {
                    for (0..inner) |inner_i| {
                        const flat = gy_base + slot * inner + inner_i;
                        const index: usize = @intCast(idxd[flat]);
                        gxd[gx_base + index * inner + inner_i] += gyd[flat];
                    }
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
