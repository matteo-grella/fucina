//! VJPs for indexed reads and writes: gather/scatter,
//! take/index-add, row and slice set/zero, relpos shift.

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
const axisGeometry = common.axisGeometry;

/// VJP of `relposShift` (S.2 skew). Forward is a per-query gather
/// `out[h,qi,kj] = bd[h,qi, kj+(Tq-1)-qi]`; the cotangent scatters back to the
/// gathered `P`-axis index (`kj→r` is a bijection within a query row, so no
/// intra-row accumulation, but unused relpos entries correctly get 0). Saves
/// only `p` (the input relpos-table dim) to size the gradient.
pub const RelposShiftBackward = struct {
    const Self = @This();

    parents: [1]?*GradState,
    p: usize,

    pub fn init(self: *RelposShiftBackward, allocator: std.mem.Allocator, parent: ?*GradState, p: usize) !void {
        _ = allocator;
        self.* = .{ .parents = .{parent}, .p = p };
    }

    pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
        if (needs_grad.len == 0 or !needs_grad[0]) return;

        var gy_ready = try contiguousForRead(ctx, gy);
        defer gy_ready.deinit();
        const gv = try gy_ready.rankView(3); // [H, Tq, Tk]
        const h = gv.shape[0];
        const t_q = gv.shape[1];
        const t_k = gv.shape[2];
        const gyd = gy_ready.dataConst();

        var gbd = try ctx.empty(.f32, .{ h, t_q, self.p });
        errdefer gbd.deinit();
        const gd = gbd.data();
        @memset(gd, 0);
        for (0..h) |hh| {
            for (0..t_q) |qi| {
                const in_row = (hh * t_q + qi) * self.p + ((t_q - 1) - qi);
                const out_row = (hh * t_q + qi) * t_k;
                for (0..t_k) |kj| gd[in_row + kj] += gyd[out_row + kj];
            }
        }
        out[0] = gbd;
    }

    pub const vtable = core.recordVTable(Self);
};

pub fn GatherBackward(comptime source_tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [1]?*GradState,
        source_shape: [rawRank(source_tags.len)]usize,
        // Source element count: the scatter-add VJP touches every element of
        // the dense source-shaped gradient (zero-fill + accumulate), so this
        // is the work the engine weighs against
        // parallel.backward_async_work_threshold.
        estimated_work: usize,
        indices: []usize,

        const Self = @This();

        pub fn init(
            self: *Self,
            allocator: std.mem.Allocator,
            parent: ?*GradState,
            source: *const RawTensor,
            indices: []const usize,
        ) !void {
            self.* = .{
                .parents = .{parent},
                .source_shape = rawShapeArray(source_tags, source),
                .estimated_work = if (parent != null) source.len() else 0,
                .indices = try allocator.dupe(usize, indices),
            };
        }

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            if (needs_grad.len == 0 or !needs_grad[0]) return;
            out[0] = try ctx.scatterAdd(rawRank(source_tags.len), gy, self.source_shape, axis, self.indices);
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.indices);
        }

        pub const vtable = core.recordVTable(Self);
    };
}

pub fn SetSliceBackward(comptime tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [2]?*GradState,
        update_shape: [rawRank(tags.len)]usize,
        start: usize,

        const Self = @This();

        pub fn init(self: *Self, allocator: std.mem.Allocator, base_parent: ?*GradState, update_parent: ?*GradState, update: *const RawTensor, start: usize) !void {
            _ = allocator;
            self.* = .{
                .parents = .{ base_parent, update_parent },
                .update_shape = rawShapeArray(tags, update),
                .start = start,
            };
        }

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            if (needs_grad.len > 0 and needs_grad[0]) {
                out[0] = try ctx.zeroSlice(rawRank(tags.len), gy, axis, self.start, self.update_shape[axis]);
            }
            if (needs_grad.len > 1 and needs_grad[1]) {
                var view = try ctx.narrowAxis(.f32, rawRank(tags.len), gy, axis, self.start, self.update_shape[axis]);
                defer view.deinit();
                out[1] = try ctx.materialize(.f32, &view);
            }
        }

        pub const vtable = core.recordVTable(Self);
    };
}

pub fn SetRowsBackward(comptime tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [2]?*GradState,
        indices: []usize,

        const Self = @This();

        pub fn init(self: *Self, allocator: std.mem.Allocator, base_parent: ?*GradState, update_parent: ?*GradState, indices: []const usize) !void {
            self.* = .{
                .parents = .{ base_parent, update_parent },
                .indices = try allocator.dupe(usize, indices),
            };
        }

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            if (needs_grad.len > 0 and needs_grad[0]) {
                out[0] = try ctx.zeroRows(rawRank(tags.len), gy, axis, self.indices);
            }
            if (needs_grad.len > 1 and needs_grad[1]) {
                out[1] = try ctx.gatherAxis(.f32, rawRank(tags.len), gy, axis, self.indices);
            }
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.indices);
        }

        pub const vtable = core.recordVTable(Self);
    };
}

pub fn TakeAlongBackward(comptime tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [1]?*GradState,
        indices: []usize,
        source_shape: [rank]usize,

        const Self = @This();
        const rank = rawRank(tags.len);

        pub fn init(self: *Self, allocator: std.mem.Allocator, parent: ?*GradState, source: *const RawTensor, indices: []const usize) !void {
            self.* = .{
                .parents = .{parent},
                .indices = try allocator.dupe(usize, indices),
                .source_shape = rawShapeArray(tags, source),
            };
        }

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            if (needs_grad.len == 0 or !needs_grad[0]) return;
            // Adjoint of the elementwise gather: scatter-add gy into zeros.
            var zeros_base = try ctx.zeros(.f32, self.source_shape[0..]);
            defer zeros_base.deinit();
            out[0] = try ctx.scatterAddAlong(rank, &zeros_base, gy, axis, self.indices);
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.indices);
        }

        pub const vtable = core.recordVTable(Self);
    };
}

pub fn ScatterAlongBackward(comptime tags: anytype, comptime axis: usize, comptime accumulate: bool) type {
    return struct {
        parents: [2]?*GradState,
        indices: []usize,
        src_axis_len: usize,

        const Self = @This();
        const rank = rawRank(tags.len);

        pub fn init(self: *Self, allocator: std.mem.Allocator, base_parent: ?*GradState, src_parent: ?*GradState, indices: []const usize, src_axis_len: usize) !void {
            self.* = .{
                .parents = .{ base_parent, src_parent },
                .indices = try allocator.dupe(usize, indices),
                .src_axis_len = src_axis_len,
            };
        }

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            if (needs_grad.len > 0 and needs_grad[0]) {
                if (comptime accumulate) {
                    // scatter-add: d/dbase is the identity.
                    out[0] = try contiguousForRead(ctx, gy);
                } else {
                    // scatter (overwrite): base positions that were written
                    // lose their gradient — zero every addressed slot.
                    var gb = try ctx.materialize(.f32, gy);
                    errdefer gb.deinit();
                    const gbd = gb.data();
                    const out_shape = rawShapeArray(tags, gy);
                    const axis_dim = out_shape[axis];
                    const geo = axisGeometry(rank, out_shape, axis);
                    for (0..geo.outer) |outer_i| {
                        const out_base = outer_i * axis_dim * geo.inner;
                        const src_base = outer_i * self.src_axis_len * geo.inner;
                        for (0..self.src_axis_len) |i| {
                            for (0..geo.inner) |inner_i| {
                                const index = self.indices[src_base + i * geo.inner + inner_i];
                                gbd[out_base + index * geo.inner + inner_i] = 0;
                            }
                        }
                    }
                    out[0] = gb;
                }
            }
            if (needs_grad.len > 1 and needs_grad[1]) {
                // d/dsrc gathers the written slots (the torch formula; on
                // overwrite-duplicates every writer reads the winner's
                // gradient, matching torch.scatter's backward).
                out[1] = try ctx.takeAlong(rank, gy, axis, self.indices, self.src_axis_len);
            }
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.indices);
        }

        pub const vtable = core.recordVTable(Self);
    };
}

pub fn IndexAddBackward(comptime tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [2]?*GradState,
        indices: []usize,

        const Self = @This();

        pub fn init(self: *Self, allocator: std.mem.Allocator, base_parent: ?*GradState, update_parent: ?*GradState, indices: []const usize) !void {
            self.* = .{
                .parents = .{ base_parent, update_parent },
                .indices = try allocator.dupe(usize, indices),
            };
        }

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            // out = self + scatterAdd(update): d/dself is the identity;
            // d/dupdate gathers the addressed rows (duplicate indices each
            // receive their position's gradient — the accumulation adjoint).
            if (needs_grad.len > 0 and needs_grad[0]) {
                out[0] = try contiguousForRead(ctx, gy);
            }
            if (needs_grad.len > 1 and needs_grad[1]) {
                out[1] = try ctx.gatherAxis(.f32, rawRank(tags.len), gy, axis, self.indices);
            }
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.indices);
        }

        pub const vtable = core.recordVTable(Self);
    };
}

pub fn ZeroSliceBackward(comptime tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [1]?*GradState,
        start: usize,
        length: usize,

        const Self = @This();

        pub fn init(self: *Self, allocator: std.mem.Allocator, parent: ?*GradState, start: usize, length: usize) !void {
            _ = allocator;
            self.* = .{
                .parents = .{parent},
                .start = start,
                .length = length,
            };
        }

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            if (needs_grad.len == 0 or !needs_grad[0]) return;
            out[0] = try ctx.zeroSlice(rawRank(tags.len), gy, axis, self.start, self.length);
        }

        pub const vtable = core.recordVTable(Self);
    };
}

pub fn ZeroRowsBackward(comptime tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [1]?*GradState,
        indices: []usize,

        const Self = @This();

        pub fn init(self: *Self, allocator: std.mem.Allocator, parent: ?*GradState, indices: []const usize) !void {
            self.* = .{
                .parents = .{parent},
                .indices = try allocator.dupe(usize, indices),
            };
        }

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            if (needs_grad.len == 0 or !needs_grad[0]) return;
            out[0] = try ctx.zeroRows(rawRank(tags.len), gy, axis, self.indices);
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.indices);
        }

        pub const vtable = core.recordVTable(Self);
    };
}
