//! VJPs for softmax-family ops: softmax, log-softmax, logsumexp.

const std = @import("std");
const tensor_mod = @import("../../tensor.zig");
const shape_mod = @import("../../shape.zig");
const exec_mod = @import("../../exec.zig");
const core = @import("../core.zig");
const tags_mod = @import("../../tags.zig");

const RawTensor = tensor_mod.Tensor;
const ExecContext = exec_mod.ExecContext;
const GradState = core.GradState;
const rawRank = tags_mod.rawRank;

const common = @import("common.zig");
const rawShapeArray = common.rawShapeArray;
const axisGeometry = common.axisGeometry;

pub fn LogsumexpBackward(comptime source_tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [1]?*GradState,
        input: RawTensor,

        const Self = @This();
        const rank = rawRank(source_tags.len);

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
            if (!core.needs(self, 0)) return;

            var gy_ready = try ctx.contiguousOwned(.f32, gy);
            defer gy_ready.deinit();
            const g = gy_ready.dataConst();

            // d(logsumexp)/dx = exp(x − lse) IS the softmax of x along
            // `axis`, scaled by the reduced-shape upstream gradient. The
            // softmax row kernels (vexpf lanes, task-parallel over rows)
            // replace the per-element libm @exp loop; the normalizer is
            // recomputed from x instead of read back from the saved lse,
            // so gradients agree in ulps rather than bitwise — the FD
            // suite is the contract.
            var gx = try ctx.softmax(.f32, rank, &self.input, axis);
            errdefer gx.deinit();
            const gxd = gx.data();

            const source_shape = rawShapeArray(source_tags, &self.input);
            const axis_dim = source_shape[axis];
            const geo = shape_mod.AxisGeometry.of(rank, source_shape, axis);
            if (geo.inner == 1) {
                for (0..geo.outer) |outer_i| {
                    const upstream = g[outer_i];
                    for (gxd[outer_i * axis_dim ..][0..axis_dim]) |*value| value.* *= upstream;
                }
            } else {
                for (0..geo.outer) |outer_i| {
                    const g_row = g[outer_i * geo.inner ..][0..geo.inner];
                    const base = outer_i * axis_dim * geo.inner;
                    for (0..axis_dim) |i| {
                        const dst = gxd[base + i * geo.inner ..][0..geo.inner];
                        for (dst, g_row) |*value, upstream| value.* *= upstream;
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

pub fn LogSoftmaxBackward(comptime source_tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [1]?*GradState,
        output: RawTensor,

        const Self = @This();
        const rank = rawRank(source_tags.len);

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
            if (!core.needs(self, 0)) return;

            var y_ready = try ctx.contiguousOwned(.f32, &self.output);
            defer y_ready.deinit();
            const y = y_ready.dataConst();
            var gy_ready = try ctx.contiguousOwned(.f32, gy);
            defer gy_ready.deinit();
            const g = gy_ready.dataConst();

            const source_shape = rawShapeArray(source_tags, &self.output);
            var gx = try ctx.empty(.f32, source_shape);
            errdefer gx.deinit();
            const gxd = gx.data();

            // d(log_softmax)/dx: g - softmax·Σg, with softmax = exp(y)
            // (torch saves the output for exactly this identity).
            const axis_dim = source_shape[axis];
            const geo = shape_mod.AxisGeometry.of(rank, source_shape, axis);
            for (0..geo.outer) |outer_i| {
                const base = outer_i * axis_dim * geo.inner;
                for (0..geo.inner) |inner_i| {
                    var g_sum: f32 = 0;
                    for (0..axis_dim) |i| g_sum += g[base + i * geo.inner + inner_i];
                    for (0..axis_dim) |i| {
                        const offset = base + i * geo.inner + inner_i;
                        gxd[offset] = g[offset] - @exp(y[offset]) * g_sum;
                    }
                }
            }
            out[0] = gx;
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            _ = allocator;
            self.output.deinit();
        }

        pub const vtable = core.recordVTable(Self);
    };
}

pub fn SoftmaxBackward(comptime tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [1]?*GradState,
        output: RawTensor,

        const Self = @This();

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
            if (!core.needs(self, 0)) return;
            out[0] = try ctx.softmaxBackward(rawRank(tags.len), &self.output, gy, axis, 1);
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            _ = allocator;
            self.output.deinit();
        }

        pub const vtable = core.recordVTable(Self);
    };
}

pub fn SoftmaxExtBackward(comptime tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [1]?*GradState,
        output: RawTensor,
        scale: f32,

        const Self = @This();

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
            if (!core.needs(self, 0)) return;
            out[0] = try ctx.softmaxBackward(rawRank(tags.len), &self.output, gy, axis, self.scale);
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            _ = allocator;
            self.output.deinit();
        }

        pub const vtable = core.recordVTable(Self);
    };
}
