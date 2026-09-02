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
//!
//! `extra` is stored by value on the backward record. An `extra` type that
//! owns state (a pointer to a forward-computed cache the backward reads)
//! declares `pub fn deinit(self: *Extra, allocator)`: `customVjp` calls it
//! once the record is released, or right after a forward that records no
//! gradient. On an error out of `customVjp` the caller still owns `extra`.
const std = @import("std");
const exec_mod = @import("../exec.zig");
const tensor_mod = @import("../tensor.zig");
const tag_ops = @import("../tag_ops.zig");
const control = @import("control.zig");
const core = @import("core.zig");
const plumbing = @import("tensor/plumbing.zig").Mod(@import("tensor.zig"));

const Allocator = std.mem.Allocator;
const ExecContext = exec_mod.ExecContext;
const RawTensor = tensor_mod.Tensor;
const GradState = core.GradState;

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
        const out = try finishNoGrad(Spec.Output, ctx, value);
        var owned_extra = extra;
        releaseExtra(&owned_extra, ctx.allocator());
        return out;
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

    var owned_value = value;
    try tag_ops.validateTensorRank(.f32, Spec.Output.axis_tags, &owned_value);
    const out = try plumbing.finishWithRecord(Spec.Output, ctx, owned_value, CustomBackward(Spec, @TypeOf(extra), Inputs){
        .extra = extra,
        .views = views,
        .output = output_view,
        .states = states,
    });
    node_owns_views = true;
    node_owns_output = true;
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

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
            std.debug.assert(out.len == n);

            // The public Spec contract keeps its `needs_grad` slice: one
            // flag per input, true where the input carries a state.
            var raw_inputs: [n]*const RawTensor = undefined;
            var needs_grad: [n]bool = undefined;
            inline for (0..n) |i| {
                raw_inputs[i] = &self.views[i];
                needs_grad[i] = core.needs(self, i);
            }

            try Spec.backward(ctx, self.extra, raw_inputs[0..], &self.output, gy, needs_grad[0..], out);
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

        pub fn deinitFields(self: *Self, allocator: Allocator) void {
            for (&self.views) |*view| view.deinit();
            self.output.deinit();
            releaseExtra(&self.extra, allocator);
        }

        pub const vtable = core.recordVTable(Self);
    };
}

/// Runs `Extra.deinit` when the extra type declares one (owned state).
fn releaseExtra(extra: anytype, allocator: Allocator) void {
    const Extra = @TypeOf(extra.*);
    if (comptime @typeInfo(Extra) == .@"struct" and @hasDecl(Extra, "deinit")) extra.deinit(allocator);
}

fn finishNoGrad(comptime Output: type, ctx: *ExecContext, value: RawTensor) !Output {
    return plumbing.finishTyped(Output, ctx, value);
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
