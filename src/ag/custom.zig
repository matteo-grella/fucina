//! Public custom VJP adapter over the tagged autograd facade.
//!
//! The public contract stays in terms of f32 `Tensor(..)` facade values. The
//! user-provided `Spec` computes on raw tensors only inside the adapter:
//!
//! ```zig
//! const Spec = struct {
//!     pub const Output = Tensor(.{.d});
//!
//!     pub fn forward(ctx: *ExecContext, extra: anytype, inputs: []const *const RawTensor) !RawTensor { ... }
//!     pub fn backward(
//!         ctx: *ExecContext,
//!         extra: anytype,
//!         inputs: []const *const RawTensor,
//!         output: *const RawTensor,
//!         gy: *const RawTensor,
//!         needs_grad: []const bool,
//!         out: []?RawTensor,
//!     ) !void { ... }
//! };
//! ```
//!
//! `backward` must write owned raw tensors into `out[i]` for every true
//! `needs_grad[i]`; the autograd engine consumes and deinits those tensors.
const std = @import("std");
const exec_mod = @import("../exec.zig");
const tensor_mod = @import("../tensor.zig");
const tag_ops = @import("../tagged.zig");
const control = @import("control.zig");
const core = @import("core.zig");

const Allocator = std.mem.Allocator;
const ExecContext = exec_mod.ExecContext;
const RawTensor = tensor_mod.Tensor;
const GradState = core.GradState;
const BackwardFunction = core.BackwardFunction;

pub fn customVjp(ctx: *ExecContext, comptime Spec: type, extra: anytype, inputs: anytype) !Spec.Output {
    comptime {
        if (!@hasDecl(Spec, "Output")) @compileError("custom VJP Spec must declare pub const Output");
        if (!@hasDecl(Spec, "forward")) @compileError("custom VJP Spec must declare forward");
        if (!@hasDecl(Spec, "backward")) @compileError("custom VJP Spec must declare backward");
        validateFacade(Spec.Output, "custom VJP output");
    }

    const Inputs = @TypeOf(inputs);
    const facade_types = comptime facadeTypes(Inputs);
    const n = facade_types.len;

    var any_grad = false;
    inline for (0..n) |i| {
        if (inputs[i].requiresGrad()) any_grad = true;
    }

    var raw_inputs: [n]*const RawTensor = undefined;
    inline for (0..n) |i| {
        raw_inputs[i] = inputs[i].asRawTensor();
    }

    var value = try Spec.forward(ctx, extra, raw_inputs[0..]);
    errdefer value.deinit();

    if (!any_grad or !control.isGradEnabled()) {
        return finishNoGrad(Spec.Output, ctx, value);
    }

    var views: [n]RawTensor = undefined;
    var states: [n]?*GradState = undefined;
    var captured: usize = 0;
    var node_owns_views = false;
    errdefer if (!node_owns_views) for (views[0..captured]) |*view| view.deinit();

    inline for (0..n) |i| {
        views[i] = try inputs[i].value.cloneView();
        states[i] = inputs[i].grad_state;
        captured = i + 1;
    }

    var output_view = try value.cloneView();
    var node_owns_output = false;
    errdefer if (!node_owns_output) output_view.deinit();

    if (ctx.execScopeActive()) try ctx.reserveScopeSlot();
    const state = try core.createNode(CustomBackward(Spec, @TypeOf(extra), Inputs), .{ ctx.allocator, extra, views, output_view, states });
    node_owns_views = true;
    node_owns_output = true;
    var out = try finishWithBackward(Spec.Output, value, state);
    if (ctx.execScopeActive()) {
        adoptIntoScope(ctx, &out);
        out.scope_owned = true;
    }
    return out;
}

fn CustomBackward(comptime Spec: type, comptime Extra: type, comptime Inputs: type) type {
    const facade_types = facadeTypes(Inputs);
    const n = facade_types.len;

    return struct {
        extra: Extra,
        views: [n]RawTensor,
        output: RawTensor,
        states: [n]?*GradState,

        const Self = @This();

        /// Consumes `views` and `output` on success; on error (the node
        /// allocation in `core.createNode`) they stay with the caller.
        pub fn init(self: *Self, allocator: Allocator, extra: Extra, views: [n]RawTensor, output: RawTensor, states: [n]?*GradState) !void {
            _ = allocator;
            self.* = .{
                .extra = extra,
                .views = views,
                .output = output,
                .states = states,
            };
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.states[0..];
        }

        fn backward(
            ptr: *const anyopaque,
            ctx: *ExecContext,
            gy: *const RawTensor,
            needs_grad: []const bool,
            out: []?RawTensor,
        ) anyerror!void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            std.debug.assert(needs_grad.len == n and out.len == n);

            var raw_inputs: [n]*const RawTensor = undefined;
            inline for (0..n) |i| {
                raw_inputs[i] = &self.views[i];
            }

            try Spec.backward(ctx, self.extra, raw_inputs[0..], &self.output, gy, needs_grad, out);
            inline for (0..n) |i| {
                if (needs_grad[i]) {
                    if (out[i]) |*grad| {
                        if (!std.mem.eql(usize, grad.shape.slice(), self.views[i].shape.slice())) {
                            return tensor_mod.TensorError.ShapeMismatch;
                        }
                    }
                }
            }
        }

        fn deinit(ptr: *anyopaque, allocator: Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            for (&self.views) |*view| view.deinit();
            self.output.deinit();
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

fn finishNoGrad(comptime Output: type, ctx: *ExecContext, value: RawTensor) !Output {
    if (ctx.execScopeActive()) try ctx.reserveScopeSlot();
    var out = try Output.fromTensor(ctx, value);
    if (ctx.execScopeActive()) {
        adoptIntoScope(ctx, &out);
        out.scope_owned = true;
    }
    return out;
}

fn finishWithBackward(comptime Output: type, value: RawTensor, state: *GradState) !Output {
    errdefer state.deinit();
    var owned_value = value;
    try tag_ops.validateTensorRank(Output.axis_tags, &owned_value);
    return .{ .value = owned_value, .grad_state = state };
}

fn adoptIntoScope(ctx: *ExecContext, t: anytype) void {
    ctx.adoptScopeValueAssumeCapacity(
        t.value,
        if (t.grad_state) |state| @ptrCast(state) else null,
        destroyGradStateOpaque,
    );
}

fn destroyGradStateOpaque(ptr: *anyopaque) void {
    const state: *GradState = @ptrCast(@alignCast(ptr));
    state.deinit();
}

fn inputFields(comptime Inputs: type) []const std.builtin.Type.StructField {
    const info = @typeInfo(Inputs);
    if (info != .@"struct" or !info.@"struct".is_tuple) {
        @compileError("custom VJP inputs must be a tuple of facade tensor pointers, got " ++ @typeName(Inputs));
    }
    return info.@"struct".fields;
}

fn facadeTypes(comptime Inputs: type) [inputFields(Inputs).len]type {
    const fields = inputFields(Inputs);
    var types: [fields.len]type = undefined;
    for (fields, 0..) |field, i| types[i] = FacadeOf(field.type);
    return types;
}

fn FacadeOf(comptime P: type) type {
    const info = @typeInfo(P);
    if (info != .pointer) @compileError("custom VJP input must be a pointer to an f32 facade tensor, got " ++ @typeName(P));
    const T = info.pointer.child;
    validateFacade(T, "custom VJP input");
    return T;
}

fn validateFacade(comptime T: type, comptime what: []const u8) void {
    if (@typeInfo(T) != .@"struct" or !@hasDecl(T, "dtype") or T.dtype != .f32 or !@hasField(T, "grad_state")) {
        @compileError(what ++ " must be an f32 autograd facade tensor, got " ++ @typeName(T));
    }
}

test {
    _ = @import("custom_tests.zig");
}
