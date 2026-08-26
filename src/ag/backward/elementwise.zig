//! VJPs for pointwise, activation, and masking ops.

const std = @import("std");
const backend_ops = @import("../../backend.zig").ops;
const tensor_mod = @import("../../tensor.zig");
const dtype_mod = @import("../../dtype.zig");
const exec_mod = @import("../../exec.zig");
const parallel = @import("../../parallel.zig");
const tag_ops = @import("../../tag_ops.zig");
const core = @import("../core.zig");
const tags_mod = @import("../../tags.zig");
const vector_primitives = @import("../../backend.zig").simd;

const RawTensor = tensor_mod.Tensor;
const ExecContext = exec_mod.ExecContext;
const GradState = core.GradState;
const rawRank = tags_mod.rawRank;
const pointwiseResultTags = tags_mod.pointwiseResultTags;

const common = @import("common.zig");
const PointwiseOp = tag_ops.PointwiseOp;
const rawShapeArray = common.rawShapeArray;
const taggedShapeArray = common.taggedShapeArray;
const reduceGradientToTags = common.reduceGradientToTags;
const contiguousForRead = common.contiguousForRead;
const contiguousForReadTyped = common.contiguousForReadTyped;

pub fn PointwiseBackward(
    comptime op: PointwiseOp,
    comptime left_tags: anytype,
    comptime right_tags: anytype,
    comptime result_tags: anytype,
) type {
    return struct {
        parents: [2]?*GradState,
        left_shape: [rawRank(left_tags.len)]usize,
        right_shape: [rawRank(right_tags.len)]usize,
        left_value: ?RawTensor = null,
        right_value: ?RawTensor = null,

        const Self = @This();

        const Side = enum { left, right };

        /// max/min gradient at the broadcast result shape: gy weighted by
        /// 1 where `favored` wins, 0.5 on exact ties (torch's subgradient,
        /// ±inf ties included), 0 where it loses — NaN positions weigh 0
        /// on both sides (IEEE compares are false; the torch formula).
        fn winnerWeightGrad(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, comptime favored: Side) !RawTensor {
            const result_shape = taggedShapeArray(result_tags, rawShapeArray(result_tags, gy));
            var left_view = try tag_ops.broadcastTensorTo(.f32, left_tags, &self.left_value.?, result_tags, result_shape);
            defer left_view.deinit();
            var right_view = try tag_ops.broadcastTensorTo(.f32, right_tags, &self.right_value.?, result_tags, result_shape);
            defer right_view.deinit();
            const win_op: exec_mod.CompareOp = comptime if ((op == .max) == (favored == .left)) .gt else .lt;
            var wins_mask = try ctx.compare(.f32, win_op, &left_view, &right_view);
            defer wins_mask.deinit();
            var wins = try ctx.cast(.bool, .f32, &wins_mask);
            defer wins.deinit();
            var ties_mask = try ctx.compare(.f32, .eq, &left_view, &right_view);
            defer ties_mask.deinit();
            var ties = try ctx.cast(.bool, .f32, &ties_mask);
            defer ties.deinit();
            var half_ties = try ctx.scale(.f32, &ties, 0.5);
            defer half_ties.deinit();
            var weight = try ctx.elementwise(.f32, .add, &wins, &half_ties);
            defer weight.deinit();
            return tag_ops.pointwise(.f32, .mul, result_tags, gy, ctx, result_tags, &weight);
        }

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
            if (core.needs(self, 0)) {
                var g = switch (op) {
                    .add, .sub => try gy.cloneView(),
                    .mul => try tag_ops.pointwise(.f32, .mul, result_tags, gy, ctx, right_tags, &self.right_value.?),
                    .div => try tag_ops.pointwise(.f32, .div, result_tags, gy, ctx, right_tags, &self.right_value.?),
                    .max, .min => try self.winnerWeightGrad(ctx, gy, .left),
                };
                defer g.deinit();
                out[0] = try reduceGradientToTags(result_tags, left_tags, ctx, &g, self.left_shape);
            }
            if (core.needs(self, 1)) {
                var g = switch (op) {
                    .add => try gy.cloneView(),
                    .sub => try ctx.scale(.f32, gy, -1),
                    .mul => try tag_ops.pointwise(.f32, .mul, result_tags, gy, ctx, left_tags, &self.left_value.?),
                    .div => blk: {
                        const num_tags = comptime pointwiseResultTags(result_tags, left_tags);
                        var numerator = try tag_ops.pointwise(.f32, .mul, result_tags, gy, ctx, left_tags, &self.left_value.?);
                        defer numerator.deinit();
                        var denominator = try tag_ops.pointwise(.f32, .mul, right_tags, &self.right_value.?, ctx, right_tags, &self.right_value.?);
                        defer denominator.deinit();
                        var quotient = try tag_ops.pointwise(.f32, .div, num_tags, &numerator, ctx, right_tags, &denominator);
                        defer quotient.deinit();
                        const neg = try ctx.scale(.f32, &quotient, -1);
                        break :blk neg;
                    },
                    .max, .min => try self.winnerWeightGrad(ctx, gy, .right),
                };
                defer g.deinit();
                out[1] = try reduceGradientToTags(result_tags, right_tags, ctx, &g, self.right_shape);
            }
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            _ = allocator;
            if (self.left_value) |*value| value.deinit();
            if (self.right_value) |*value| value.deinit();
        }

        pub const vtable = core.recordVTable(Self);
    };
}

/// Cast's VJP is the identity: pass the gradient through contiguously.
pub const CastBackward = IdentityBackward;

pub fn IdentityBackward(comptime tags: anytype) type {
    _ = tags;
    return struct {
        parents: [1]?*GradState,

        const Self = @This();

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
            if (!core.needs(self, 0)) return;
            out[0] = try contiguousForRead(ctx, gy);
        }

        pub const vtable = core.recordVTable(Self);
    };
}

pub const ReluBackward = struct {
    const Self = @This();

    parents: [1]?*GradState,
    input: RawTensor,

    pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
        if (!core.needs(self, 0)) return;

        var x = try contiguousForRead(ctx, &self.input);
        defer x.deinit();
        var gy_ready = try contiguousForRead(ctx, gy);
        defer gy_ready.deinit();

        var gx = try ctx.empty(.f32, x.shape.slice());
        errdefer gx.deinit();
        for (x.dataConst(), gy_ready.dataConst(), gx.data()) |value, grad, *dst| {
            dst.* = if (value > 0) grad else 0;
        }
        out[0] = gx;
    }

    pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
        _ = allocator;
        self.input.deinit();
    }

    pub const vtable = core.recordVTable(Self);
};

/// VJP of `softcap(cap)`: `y = cap * tanh(x / cap)`, so `dy/dx = 1 - (y/cap)^2`
/// from the OUTPUT (transcendental-free, and exact for the value the forward
/// produced, the `tanh` convention).
pub const SoftcapBackward = struct {
    const Self = @This();

    parents: [1]?*GradState,
    output: RawTensor,
    cap: f32,

    pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
        if (!core.needs(self, 0)) return;

        var y = try contiguousForRead(ctx, &self.output);
        defer y.deinit();
        var gy_ready = try contiguousForRead(ctx, gy);
        defer gy_ready.deinit();

        const inv = 1.0 / self.cap;
        var gx = try ctx.empty(.f32, y.shape.slice());
        errdefer gx.deinit();
        for (y.dataConst(), gy_ready.dataConst(), gx.data()) |value, grad, *dst| {
            const u = value * inv;
            dst.* = grad * (1 - u * u);
        }
        out[0] = gx;
    }

    pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
        _ = allocator;
        self.output.deinit();
    }

    pub const vtable = core.recordVTable(Self);
};

pub const LeakyReluBackward = struct {
    const Self = @This();

    parents: [1]?*GradState,
    input: RawTensor,
    negative_slope: f32,

    pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
        if (!core.needs(self, 0)) return;

        var x = try contiguousForRead(ctx, &self.input);
        defer x.deinit();
        var gy_ready = try contiguousForRead(ctx, gy);
        defer gy_ready.deinit();

        var gx = try ctx.empty(.f32, x.shape.slice());
        errdefer gx.deinit();
        for (x.dataConst(), gy_ready.dataConst(), gx.data()) |value, grad, *dst| {
            dst.* = if (value > 0) grad else grad * self.negative_slope;
        }
        out[0] = gx;
    }

    pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
        _ = allocator;
        self.input.deinit();
    }

    pub const vtable = core.recordVTable(Self);
};

/// Ops whose derivative is cheaper in terms of the forward OUTPUT t
/// (tanh' = 1 − t²): the node stores the output view instead of the input, so
/// the VJP never re-evaluates the transcendental and differentiates exactly
/// the value the vectorized forward kernel produced.
pub fn unaryUsesOutput(comptime op: exec_mod.UnaryOp) bool {
    // Exhaustive on purpose: a misclassified op makes the backward node save
    // the wrong operand and produce a silently wrong gradient, so a new
    // `UnaryOp` member must be claimed here explicitly (compile error until
    // it is), exactly like the forward dispatch switch.
    return switch (op) {
        .tanh, .reciprocal => true,
        .relu,
        .exp,
        .sqrt,
        .rsqrt,
        .sigmoid,
        .silu,
        .log,
        .log1p,
        .softplus,
        .neg,
        .abs,
        .sin,
        .cos,
        .fast_tanh,
        .gelu,
        .quick_gelu,
        .gelu_quant,
        .elu,
        .gelu_erf,
        .erf,
        .floor,
        .ceil,
        .round,
        .sign,
        => false,
    };
}

pub fn UnaryBackward(comptime op: exec_mod.UnaryOp, comptime tags: anytype) type {
    _ = tags;
    return struct {
        parents: [1]?*GradState,
        /// The forward input — or the forward OUTPUT for unaryUsesOutput ops.
        input: RawTensor,

        const Self = @This();

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
            if (!core.needs(self, 0)) return;

            var x = try contiguousForRead(ctx, &self.input);
            defer x.deinit();
            var gy_ready = try contiguousForRead(ctx, gy);
            defer gy_ready.deinit();

            var gx = try ctx.empty(.f32, x.shape.slice());
            errdefer gx.deinit();
            const xs = x.dataConst();
            const gys = gy_ready.dataConst();
            const dsts = gx.data();
            // Elementwise map: chunking is partition-invariant (bitwise-equal
            // to the serial loop), so parallelize like the forward kernels —
            // transcendental derivatives (tanh/gelu/…) over vocab-sized
            // gradients otherwise serialize the whole backward pass.
            const total = dsts.len;
            if (total >= parallel.vector_elementwise_len_threshold) {
                if (ctx.workPool()) |pool| {
                    const task_count = parallel.cpuThreadCount(parallel.vector_max_threads);
                    var tasks: [parallel.vector_max_threads]ChunkTask = undefined;
                    for (0..task_count) |i| {
                        const s = i * total / task_count;
                        const e = (i + 1) * total / task_count;
                        tasks[i] = .{ .xs = xs[s..e], .gys = gys[s..e], .dsts = dsts[s..e] };
                    }
                    pool.parallelChunks(ChunkTask, tasks[0..task_count], ChunkTask.run);
                    out[0] = gx;
                    return;
                }
            }
            ChunkTask.run(&.{ .xs = xs, .gys = gys, .dsts = dsts });
            out[0] = gx;
        }

        const ChunkTask = struct {
            xs: []const f32,
            gys: []const f32,
            dsts: []f32,

            fn run(t: *const ChunkTask) void {
                if (comptime vector_primitives.unaryVjpVectorizes(op)) {
                    // SIMD derivative body — the scalar loop pays one libm
                    // expf per element for the exp-family ops.
                    vector_primitives.vecUnaryVjp(op, t.dsts, t.xs, t.gys);
                } else if (comptime unaryUsesOutput(op)) {
                    for (t.xs, t.gys, t.dsts) |value, grad, *dst| {
                        dst.* = grad * unaryDerivativeFromOutput(op, value);
                    }
                } else {
                    for (t.xs, t.gys, t.dsts) |value, grad, *dst| {
                        dst.* = grad * unaryDerivative(op, value);
                    }
                }
            }
        };

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            _ = allocator;
            self.input.deinit();
        }

        pub const vtable = core.recordVTable(Self);
    };
}

pub fn ScaleBackward(comptime tags: anytype) type {
    _ = tags;
    return struct {
        parents: [1]?*GradState,
        scalar_value: f32,

        const Self = @This();

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
            if (!core.needs(self, 0)) return;
            out[0] = try ctx.scale(.f32, gy, self.scalar_value);
        }

        pub const vtable = core.recordVTable(Self);
    };
}

/// Dropout VJP: stores only {p, seed} — the mask is NEVER materialized. The
/// exec backward kernel regenerates the identical counter-based (seed, i)
/// mask, so the node is as cheap as ScaleBackward and recompute-safe under
/// activation checkpointing.
pub fn DropoutBackward(comptime tags: anytype) type {
    _ = tags;
    return struct {
        parents: [1]?*GradState,
        p: f32,
        seed: u64,

        const Self = @This();

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
            if (!core.needs(self, 0)) return;
            out[0] = try ctx.dropoutBackward(gy, self.p, self.seed);
        }

        pub const vtable = core.recordVTable(Self);
    };
}

pub const ClampBackward = struct {
    const Self = @This();

    parents: [1]?*GradState,
    input: RawTensor,
    min_value: f32,
    max_value: f32,

    pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
        if (!core.needs(self, 0)) return;

        var x = try contiguousForRead(ctx, &self.input);
        defer x.deinit();
        var gy_ready = try contiguousForRead(ctx, gy);
        defer gy_ready.deinit();

        var gx = try ctx.empty(.f32, x.shape.slice());
        errdefer gx.deinit();
        for (x.dataConst(), gy_ready.dataConst(), gx.data()) |value, grad, *dst| {
            dst.* = if (value >= self.min_value and value <= self.max_value) grad else 0;
        }
        out[0] = gx;
    }

    pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
        _ = allocator;
        self.input.deinit();
    }

    pub const vtable = core.recordVTable(Self);
};

pub fn GatedBackward(
    comptime op: exec_mod.GatedOp,
    comptime left_tags: anytype,
    comptime right_tags: anytype,
    comptime result_tags: anytype,
) type {
    return struct {
        parents: [2]?*GradState,
        left_shape: [rawRank(left_tags.len)]usize,
        right_shape: [rawRank(right_tags.len)]usize,
        result_shape: [rawRank(result_tags.len)]usize,
        left_value: RawTensor,
        right_value: RawTensor,

        const Self = @This();

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
            if (!core.needs(self, 0) and !core.needs(self, 1)) return;

            var gy_ready = try contiguousForRead(ctx, gy);
            defer gy_ready.deinit();
            const gyd = gy_ready.dataConst();

            const result_tagged_shape = taggedShapeArray(result_tags, self.result_shape);
            var left_b = try tag_ops.broadcastTensorTo(.f32, left_tags, &self.left_value, result_tags, result_tagged_shape);
            defer left_b.deinit();
            var right_b = try tag_ops.broadcastTensorTo(.f32, right_tags, &self.right_value, result_tags, result_tagged_shape);
            defer right_b.deinit();
            var left_ready = try contiguousForRead(ctx, &left_b);
            defer left_ready.deinit();
            var right_ready = try contiguousForRead(ctx, &right_b);
            defer right_ready.deinit();
            const left_data = left_ready.dataConst();
            const right_data = right_ready.dataConst();

            if (core.needs(self, 0)) {
                var g = try ctx.empty(.f32, self.result_shape[0..]);
                if (comptime gatedSourceIsIdentity(op)) {
                    for (gyd, right_data, g.data()) |grad, gate, *dst| {
                        dst.* = grad * gatedActivation(op, gate);
                    }
                } else {
                    for (gyd, left_data, right_data, g.data()) |grad, left, gate, *dst| {
                        dst.* = grad * gatedSourceDerivative(op, left) * gatedActivation(op, gate);
                    }
                }
                defer g.deinit();
                out[0] = try reduceGradientToTags(result_tags, left_tags, ctx, &g, self.left_shape);
            }

            if (core.needs(self, 1)) {
                var g = try ctx.empty(.f32, self.result_shape[0..]);
                for (gyd, left_data, right_data, g.data()) |grad, left, gate, *dst| {
                    dst.* = grad * gatedSource(op, left) * gatedActivationDerivative(op, gate);
                }
                defer g.deinit();
                out[1] = try reduceGradientToTags(result_tags, right_tags, ctx, &g, self.right_shape);
            }
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            _ = allocator;
            self.left_value.deinit();
            self.right_value.deinit();
        }

        pub const vtable = core.recordVTable(Self);
    };
}

pub fn SplitSwiGluBackward(comptime tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [1]?*GradState,
        input: RawTensor,

        const Self = @This();

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
            if (!core.needs(self, 0)) return;
            out[0] = try ctx.splitSwiGluBackward(rawRank(tags.len), &self.input, gy, axis);
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            _ = allocator;
            self.input.deinit();
        }

        pub const vtable = core.recordVTable(Self);
    };
}

pub fn SplitGluBackward(comptime tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [1]?*GradState,
        input: RawTensor,

        const Self = @This();

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
            if (!core.needs(self, 0)) return;
            out[0] = try ctx.splitGluBackward(rawRank(tags.len), &self.input, gy, axis);
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            _ = allocator;
            self.input.deinit();
        }

        pub const vtable = core.recordVTable(Self);
    };
}

/// VJP for `addScalar` (and `subScalar`): `d/dx (x + c) = 1`, so grad passes
/// through unchanged.
pub fn AddScalarBackward(comptime tags: anytype) type {
    _ = tags;
    return struct {
        parents: [1]?*GradState,

        const Self = @This();

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
            if (!core.needs(self, 0)) return;
            out[0] = try ctx.scale(.f32, gy, 1); // identity passthrough as a fresh owned tensor
        }

        pub const vtable = core.recordVTable(Self);
    };
}

/// VJP for `powScalar`: `d/dx (x^c) = c·x^(c-1)`, so `grad_x = gy · c · x^(c-1)`.
pub fn PowScalarBackward(comptime tags: anytype) type {
    _ = tags;
    return struct {
        parents: [1]?*GradState,
        input: RawTensor,
        exponent: f32,

        const Self = @This();

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
            if (!core.needs(self, 0)) return;

            var x = try contiguousForRead(ctx, &self.input);
            defer x.deinit();
            var gy_ready = try contiguousForRead(ctx, gy);
            defer gy_ready.deinit();

            var gx = try ctx.empty(.f32, x.shape.slice());
            errdefer gx.deinit();
            const c = self.exponent;
            for (x.dataConst(), gy_ready.dataConst(), gx.data()) |value, grad, *dst| {
                dst.* = grad * c * std.math.pow(f32, value, c - 1);
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

pub fn MaskedFillBackward(comptime tags: anytype, comptime mask_dtype: tensor_mod.DType) type {
    _ = tags;
    return struct {
        parents: [1]?*GradState,
        mask: tensor_mod.TensorOf(mask_dtype),

        const Self = @This();

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
            if (!core.needs(self, 0)) return;
            var m = try contiguousForReadTyped(mask_dtype, ctx, &self.mask);
            defer m.deinit();
            var gy_ready = try contiguousForRead(ctx, gy);
            defer gy_ready.deinit();
            var gx = try ctx.empty(.f32, m.shape.slice());
            errdefer gx.deinit();
            for (m.dataConst(), gy_ready.dataConst(), gx.data()) |mv, grad, *dst| {
                dst.* = if (dtype_mod.isTruthy(mask_dtype, mv)) 0 else grad;
            }
            out[0] = gx;
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            _ = allocator;
            self.mask.deinit();
        }

        pub const vtable = core.recordVTable(Self);
    };
}

/// VJP for `where(x, cond, y)` = `cond ? x : y`: `grad_x = cond ? gy : 0` and
/// `grad_y = cond ? 0 : gy` (`cond` is a non-grad mask).
pub fn WhereBackward(comptime tags: anytype, comptime cond_dtype: tensor_mod.DType) type {
    _ = tags;
    return struct {
        parents: [2]?*GradState,
        cond: tensor_mod.TensorOf(cond_dtype),

        const Self = @This();

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
            var c = try contiguousForReadTyped(cond_dtype, ctx, &self.cond);
            defer c.deinit();
            var gy_ready = try contiguousForRead(ctx, gy);
            defer gy_ready.deinit();
            if (core.needs(self, 0)) {
                var gx = try ctx.empty(.f32, c.shape.slice());
                errdefer gx.deinit();
                for (c.dataConst(), gy_ready.dataConst(), gx.data()) |cv, grad, *dst| {
                    dst.* = if (dtype_mod.isTruthy(cond_dtype, cv)) grad else 0;
                }
                out[0] = gx;
            }
            if (core.needs(self, 1)) {
                var gyy = try ctx.empty(.f32, c.shape.slice());
                errdefer gyy.deinit();
                for (c.dataConst(), gy_ready.dataConst(), gyy.data()) |cv, grad, *dst| {
                    dst.* = if (dtype_mod.isTruthy(cond_dtype, cv)) 0 else grad;
                }
                out[1] = gyy;
            }
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            _ = allocator;
            self.cond.deinit();
        }

        pub const vtable = core.recordVTable(Self);
    };
}

/// Derivative expressed in the forward OUTPUT t (unaryUsesOutput ops only).
fn unaryDerivativeFromOutput(comptime op: exec_mod.UnaryOp, t: f32) f32 {
    return switch (op) {
        .tanh => 1 - t * t,
        // out = 15·tanh(x/15) ⇒ d/dx = 1 − (out/15)².
        // out = 1/x ⇒ d/dx = -1/x² = -out².
        .reciprocal => -t * t,
        else => @compileError("unaryDerivativeFromOutput: op is not output-derivative"),
    };
}

fn unaryDerivative(comptime op: exec_mod.UnaryOp, value: f32) f32 {
    return switch (op) {
        .relu => if (value > 0) 1 else 0,
        .exp => @exp(value),
        .sqrt => 0.5 / @sqrt(value),
        .rsqrt => -0.5 / (value * @sqrt(value)),
        .sigmoid => blk: {
            const s = sigmoid(value);
            break :blk s * (1 - s);
        },
        .softplus => sigmoid(value),
        .silu => blk: {
            const s = sigmoid(value);
            break :blk s * (1 + value * (1 - s));
        },
        .log => 1 / value,
        .log1p => 1 / (1 + value),
        .neg => -1,
        .abs => if (value > 0) 1 else if (value < 0) -1 else 0,
        .sin => @cos(value),
        .cos => -@sin(value),
        .tanh => blk: {
            // Only reached via input-based callers; the autograd node stores
            // the OUTPUT for tanh (unaryUsesOutput) and uses 1 - t².
            const t = std.math.tanh(value);
            break :blk 1 - t * t;
        },
        .fast_tanh => fastTanhDerivative(value),
        .gelu => geluDerivative(value),
        .quick_gelu => quickGeluDerivative(value),
        .gelu_quant => geluDerivative(value), // inference-only; exact-gelu derivative
        .elu => if (value > 0) 1 else @exp(value),
        .gelu_erf => geluErfDerivative(value),
        // d/dx erf(x) = 2/sqrt(pi) * e^(-x^2)
        .erf => 1.1283791670955126 * @exp(-value * value),
        // Piecewise-constant ops: zero gradient almost everywhere (the
        // torch convention — jump points get the a.e. value, 0).
        .floor, .ceil, .round, .sign => 0,
        .reciprocal => -1 / (value * value),
    };
}

fn geluErfDerivative(value: f32) f32 {
    // d/dx [0.5·x·(1 + erf(x/√2))] = 0.5·(1 + erf(x/√2)) + x·φ(x),
    // with the standard-normal pdf φ(x) = exp(-x²/2)/√(2π).
    const inv_sqrt_2pi: f32 = 0.3989422804014327; // 1/√(2π)
    const cdf = 0.5 * (1 + backend_ops.erff(value * 0.70710678118654752440084436210484));
    return cdf + value * @exp(-0.5 * value * value) * inv_sqrt_2pi;
}

fn fastTanhDerivative(value: f32) f32 {
    const a: f32 = 2.45550750702956;
    const b: f32 = 0.893229853513558;
    const c: f32 = 0.821226666969744;
    const d: f32 = 2.44506634652299;
    const e: f32 = 0.814642734961073;

    const ax = @abs(value);
    const dax: f32 = if (value > 0) 1 else if (value < 0) -1 else 0;
    const x2 = value * value;
    const p = a + a * ax + (b + c * ax) * x2;
    const dp = a * dax + c * dax * x2 + (b + c * ax) * 2 * value;
    const numerator = value * p;
    const dnumerator = p + value * dp;

    const q = value + e * value * ax;
    const dq = 1 + e * (ax + value * dax);
    const r = @abs(q);
    const dr: f32 = if (q > 0) dq else if (q < 0) -dq else 0;
    const denominator = d + (d + x2) * r;
    const ddenominator = 2 * value * r + (d + x2) * dr;

    return (dnumerator * denominator - numerator * ddenominator) / (denominator * denominator);
}

fn sigmoid(value: f32) f32 {
    if (value >= 0) {
        const z = @exp(-value);
        return 1 / (1 + z);
    }
    const z = @exp(value);
    return z / (1 + z);
}

fn geluDerivative(value: f32) f32 {
    const sqrt_2_over_pi: f32 = 0.7978845608028654;
    const x2 = value * value;
    const u = sqrt_2_over_pi * (value + 0.044715 * value * x2);
    const t = std.math.tanh(u);
    const du = sqrt_2_over_pi * (1 + 3 * 0.044715 * x2);
    return 0.5 * (1 + t) + 0.5 * value * (1 - t * t) * du;
}

fn quickGeluDerivative(value: f32) f32 {
    const s = sigmoid(1.702 * value);
    return s + value * 1.702 * s * (1 - s);
}

fn gatedActivation(comptime op: exec_mod.GatedOp, value: f32) f32 {
    return switch (op) {
        .glu => sigmoid(value),
        .swiglu => value * sigmoid(value),
        .geglu => 0.5 * value * (1 + std.math.tanh(0.7978845608028654 * (value + 0.044715 * value * value * value))),
        // Inference-only op (DeepSeek V4); no training/backward path.
        .situ => 4.0 * std.math.tanh(value * 0.25) * sigmoid(value),
    };
}

fn gatedActivationDerivative(comptime op: exec_mod.GatedOp, value: f32) f32 {
    return switch (op) {
        .glu => blk: {
            const s = sigmoid(value);
            break :blk s * (1 - s);
        },
        .swiglu => blk: {
            const s = sigmoid(value);
            break :blk s * (1 + value * (1 - s));
        },
        .geglu => geluDerivative(value),
        .situ => blk: {
            // d/dg [4·tanh(g/4)·σ(g)] = (1 − tanh²(g/4))·σ(g)
            //                          + 4·tanh(g/4)·σ(g)·(1 − σ(g))
            const t = std.math.tanh(value * 0.25);
            const s = sigmoid(value);
            break :blk (1 - t * t) * s + 4.0 * t * s * (1 - s);
        },
    };
}

/// The gated pair's `up`-side transform and its derivative — the identity
/// (and 1) for ops that use `up` linearly; `situ`'s 25·tanh(u/25) soft
/// clamp otherwise. Comptime-identity keeps the classic ops' backward
/// loops exactly as before.
fn gatedSource(comptime op: exec_mod.GatedOp, value: f32) f32 {
    return switch (op) {
        .situ => 25.0 * std.math.tanh(value * 0.04),
        else => value,
    };
}

fn gatedSourceDerivative(comptime op: exec_mod.GatedOp, value: f32) f32 {
    return switch (op) {
        .situ => blk: {
            const t = std.math.tanh(value * 0.04);
            break :blk 1 - t * t;
        },
        else => 1,
    };
}

/// Whether the op's `up`-side transform is the identity (its derivative a
/// comptime 1) — selects the untouched two-operand backward loops.
inline fn gatedSourceIsIdentity(comptime op: exec_mod.GatedOp) bool {
    return switch (op) {
        .situ => false,
        else => true,
    };
}

/// VJP of the per-channel PReLU: `gx = x > 0 ? gy : α[c]·gy`;
/// `gα[c] = Σ gy·min(x,0)` over the leading (row) axes.
pub const PreluChannelsBackward = struct {
    parents: [2]?*GradState,
    channels: usize,
    input_value: RawTensor,
    alpha_value: RawTensor,

    const Self = @This();

    pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
        if (core.needs(self, 0)) {
            out[0] = try ctx.preluChannelsBackwardInput(gy, &self.input_value, &self.alpha_value);
        }
        if (core.needs(self, 1)) {
            out[1] = try ctx.preluChannelsBackwardAlpha(gy, &self.input_value, self.channels);
        }
    }

    pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
        _ = allocator;
        self.input_value.deinit();
        self.alpha_value.deinit();
    }

    pub const vtable = core.recordVTable(Self);
};

/// VJP of the per-channel affine `y = x·scale[c] + shift[c]`:
/// `gx = gy·scale[c]` (the same kernel, shift-less), `gscale[c] = Σ gy·x` and
/// `gshift[c] = Σ gy` over the leading axes (suffix reduce).
pub const ChannelAffineBackward = struct {
    parents: [3]?*GradState,
    channels: usize,
    input_value: RawTensor,
    scale_value: RawTensor,

    const Self = @This();

    pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
        if (core.needs(self, 0)) {
            out[0] = try ctx.channelAffine(gy, &self.scale_value, null);
        }
        if (core.needs(self, 1)) {
            var prod = try ctx.elementwise(.f32, .mul, gy, &self.input_value);
            defer prod.deinit();
            out[1] = try ctx.reduceBroadcast(&prod, &.{self.channels});
        }
        if (core.needs(self, 2)) {
            out[2] = try ctx.reduceBroadcast(gy, &.{self.channels});
        }
    }

    pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
        _ = allocator;
        self.input_value.deinit();
        self.scale_value.deinit();
    }

    pub const vtable = core.recordVTable(Self);
};

/// VJP of the per-channel Snake activation. Three operands: input, alpha,
/// inv_b — alpha and inv_b are INDEPENDENT tensor inputs at this level (the
/// caller ties `inv_b = 1/(alpha+1e-9)` at load time; a trainer wanting a
/// single alpha parameter must chain through that relation itself). The two
/// per-channel parameter gradients come from one fused kernel pass; the one
/// that was not requested is dropped.
pub fn SnakeBackward(comptime tags: anytype) type {
    _ = tags;
    return struct {
        parents: [3]?*GradState,
        input_value: RawTensor,
        alpha_value: RawTensor,
        inv_b_value: RawTensor,

        const Self = @This();

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
            if (core.needs(self, 0)) {
                out[0] = try ctx.snakeRowsBackwardInput(&self.input_value, gy, &self.alpha_value, &self.inv_b_value);
            }
            const need_alpha = core.needs(self, 1);
            const need_inv_b = core.needs(self, 2);
            if (need_alpha or need_inv_b) {
                var params = try ctx.snakeRowsBackwardParams(&self.input_value, gy, &self.alpha_value, &self.inv_b_value);
                if (need_alpha) out[1] = params.alpha else params.alpha.deinit();
                if (need_inv_b) out[2] = params.inv_b else params.inv_b.deinit();
            }
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            _ = allocator;
            self.input_value.deinit();
            self.alpha_value.deinit();
            self.inv_b_value.deinit();
        }

        pub const vtable = core.recordVTable(Self);
    };
}
