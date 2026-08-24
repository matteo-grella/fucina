//! f32 tensor methods: pooling and nearest-neighbor upsampling. A mixin over the ag
//! FloatTensor struct; aliased back onto it in ../../tensor.zig.

const exec_mod = @import("../../../exec.zig");
const backward_pool = @import("../../backward/pool.zig");

const ExecContext = exec_mod.ExecContext;
const MaxPool2dBackward = backward_pool.MaxPool2dBackward;
const AvgPool2dBackward = backward_pool.AvgPool2dBackward;
const Upsample2xNearestBackward = backward_pool.Upsample2xNearestBackward;

pub fn Ops(comptime Self: type) type {
    return struct {
        const tags = Self.axis_tags;
        const ag_tensor = Self.ag_root;
        const plumbing = @import("../plumbing.zig").Mod(ag_tensor);
        const finishOp = plumbing.finishOp;

        /// 2-D max pool over a channel-last rank-3 `[H, W, C]` tensor
        /// (`kernel`/`stride`/`padding` in `[h, w]` order; the zero-pad
        /// border reads as −inf). Tags are preserved.
        pub fn maxPool2d(self: *const Self, ctx: *ExecContext, kernel: [2]usize, stride: [2]usize, padding: [2]usize) !Self {
            var value = try ctx.maxPool2d(self.asRawTensor(), kernel, stride, padding);
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad(), MaxPool2dBackward, .{ ctx.allocator, self.grad_state, self.asRawTensor(), kernel, stride, padding });
        }

        /// 2-D average pool over a channel-last rank-3 `[H, W, C]` tensor;
        /// averages the valid taps only (ONNX `count_include_pad=0`).
        pub fn avgPool2d(self: *const Self, ctx: *ExecContext, kernel: [2]usize, stride: [2]usize, padding: [2]usize) !Self {
            var value = try ctx.avgPool2d(self.asRawTensor(), kernel, stride, padding);
            errdefer value.deinit();
            const raw = self.asRawTensor();
            return finishOp(tags, ctx, value, self.requiresGrad(), AvgPool2dBackward, .{ ctx.allocator, self.grad_state, raw.shape.at(0), raw.shape.at(1), kernel, stride, padding });
        }

        /// 2× nearest-neighbour upsample of a channel-last rank-3 tensor:
        /// `[H, W, C]` → `[2H, 2W, C]` (VJP = 2×2 stride-2 sum-pool).
        pub fn upsample2xNearest(self: *const Self, ctx: *ExecContext) !Self {
            var value = try ctx.upsample2xNearest(self.asRawTensor());
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad(), Upsample2xNearestBackward, .{ ctx.allocator, self.grad_state });
        }
    };
}
