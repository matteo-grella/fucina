//! The leaf and gradient plumbing of the gradient-carrying branches: f32,
//! and the 16-bit leaves (f16/bf16, whose VALUE is stored narrow while the
//! accumulated gradient is always f32). Trainable leaves, gradient
//! read-out, and the f32-only backward entry points. A mixin over the
//! tensor struct; aliased back onto it in ../tensor.zig (the typed float
//! branch aliases everything except `backward`/`backwardWithGrad`: a
//! 16-bit tensor is never a loss).

const std = @import("std");
const tensor_mod = @import("../../tensor.zig");
const exec_mod = @import("../../exec.zig");
const tag_ops = @import("../../tag_ops.zig");
const core = @import("../core.zig");
const dtype_mod = @import("../../dtype.zig");

const TensorError = tensor_mod.TensorError;
const ExecContext = exec_mod.ExecContext;
const GradState = core.GradState;
const validateTensorRank = tag_ops.validateTensorRank;

pub fn Ops(comptime Self: type) type {
    return struct {
        const dtype = Self.dtype;
        const tags = Self.axis_tags;
        const tensor_rank = Self.tensor_rank;
        const RawT = tensor_mod.TensorOf(dtype);
        /// The facade element (`dtype_mod.Element`): `Bf16` values on the
        /// bf16 branch, the raw element elsewhere (see `common.zig`).
        const Elem = dtype_mod.Element(dtype);
        const RawElem = RawT.Element;
        const Tensor = Self.ag_root.Tensor;
        /// The gradient's type: f32, same tags (the f32 branch's own type).
        const Grad = Tensor(.{ .dtype = .f32, .tags = tags });

        fn requireGradDtype(comptime what: []const u8) void {
            if (dtype != .f32 and dtype != .f16 and dtype != .bf16) {
                @compileError(what ++ " requires an f32, f16, or bf16 tensor (gradients are always f32; f64 training is unsupported)");
            }
        }

        /// Trainable leaf. Consumes `value` on success; on error, ownership
        /// stays with the caller. On the 16-bit branches the value stays
        /// narrow and the gradient accumulates in f32.
        pub fn variable(ctx: *ExecContext, value: RawT) !Self {
            comptime requireGradDtype("variable");
            var v = value;
            try validateTensorRank(dtype, tags, &v);
            const state = try GradState.leaf(ctx.allocator());
            errdefer state.release();
            return .{ .value = v, .grad_state = state };
        }

        pub fn variableFromSlice(ctx: *ExecContext, raw_shape: [tensor_rank]usize, values: []const Elem) !Self {
            comptime requireGradDtype("variableFromSlice");
            var value = try ctx.fromSlice(dtype, raw_shape, @as([]const RawElem, @ptrCast(values)));
            errdefer value.deinit();
            return Self.variable(ctx, value);
        }

        /// Drop the accumulated gradient (no-op for constants — and for
        /// f64, which carries no gradient slot at all). Training loops
        /// call this between steps so gradients don't accumulate across them.
        pub fn zeroGrad(self: *Self) void {
            if (comptime @FieldType(Self, "grad_state") != void) {
                if (self.grad_state) |state| state.zeroGrad();
            }
        }

        /// The accumulated gradient as an owned f32 constant (null before
        /// backward, or on a constant).
        pub fn grad(self: *const Self, ctx: *ExecContext) !?Grad {
            comptime requireGradDtype("grad");
            const state = self.grad_state orelse return null;
            var value = (try state.gradClone(ctx.allocator())) orelse return null;
            errdefer value.deinit();
            return try Grad.constant(ctx, value);
        }

        /// Aliasing f32 view of the accumulated gradient (see `grad`).
        pub fn gradView(self: *const Self, ctx: *ExecContext) !?Grad {
            comptime requireGradDtype("gradView");
            const state = self.grad_state orelse return null;
            var value = (try state.gradView()) orelse return null;
            errdefer value.deinit();
            return try Grad.constant(ctx, value);
        }

        pub fn backward(self: *Self, ctx: *ExecContext) !void {
            comptime if (dtype != .f32) @compileError("backward runs from an f32 tensor (16-bit tensors are leaves, never losses)");
            const state = self.grad_state orelse return error.NoGradientGraph;
            return core.backwardGradOne(ctx, state, &self.value);
        }

        /// As `backward`, but with an explicit output gradient instead of
        /// the implicit scalar 1: the way to run backward from a non-scalar
        /// output (scalar outputs may take one too). `grad_output` is
        /// same-tagged and must match `self`'s shape
        /// (`error.ShapeMismatch`); it is read as a value, its own gradient
        /// state, if any, is ignored, and it replaces any gradient already
        /// accumulated on `self`.
        pub fn backwardWithGrad(self: *Self, ctx: *ExecContext, grad_output: *const Self) !void {
            comptime if (dtype != .f32) @compileError("backwardWithGrad runs from an f32 tensor (16-bit tensors are leaves, never losses)");
            const state = self.grad_state orelse return error.NoGradientGraph;
            // Checked here too so the error exit leaves `self`'s accumulated
            // gradient untouched (the engine re-checks after setGrad).
            if (state.backward_done) return core.AgError.BackwardAlreadyRun;
            if (!std.mem.eql(usize, self.value.shape.slice(), grad_output.value.shape.slice())) {
                return TensorError.ShapeMismatch;
            }
            state.setGrad(try grad_output.value.cloneView());
            return core.backwardGradOne(ctx, state, &self.value);
        }
    };
}
