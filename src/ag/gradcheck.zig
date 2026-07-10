//! Finite-difference gradient checks over the tagged f32 autograd facade.
//!
//! `gradcheck` checks a deterministic scalar loss function:
//!
//! ```zig
//! fn loss(ctx: *ExecContext, x: *const Tensor(.{.d})) !Tensor(.{}) { ... }
//! try gradcheck(ctx, loss, .{&x}, .{});
//! ```
//!
//! Inputs are public tagged f32 tensor pointers. Variable inputs are checked;
//! constants may appear in the tuple but are ignored. Variable inputs must be
//! contiguous because the harness perturbs their owned storage directly.
const std = @import("std");
const exec_mod = @import("../exec.zig");
const tensor_mod = @import("../tensor.zig");

const Allocator = std.mem.Allocator;
const ExecContext = exec_mod.ExecContext;
const RawTensor = tensor_mod.Tensor;

pub const Options = struct {
    eps: f64 = 1e-3,
    abs_tol: f64 = 1e-3,
    rel_tol: f64 = 1e-2,
    print_mismatch: bool = true,
};

pub const Result = struct {
    checked: usize = 0,
    max_abs_error: f64 = 0,
    max_rel_error: f64 = 0,
};

pub fn gradcheck(ctx: *ExecContext, comptime loss_fn: anytype, inputs: anytype, options: Options) !Result {
    if (!std.math.isFinite(options.eps) or options.eps <= 0) return error.InvalidGradcheckOptions;
    if (!std.math.isFinite(options.abs_tol) or options.abs_tol < 0) return error.InvalidGradcheckOptions;
    if (!std.math.isFinite(options.rel_tol) or options.rel_tol < 0) return error.InvalidGradcheckOptions;

    const Inputs = @TypeOf(inputs);
    const facade_types = comptime facadeTypes(Inputs);
    const n = facade_types.len;
    comptime validateLoss(loss_fn, n);

    zeroGradInputs(inputs);
    defer zeroGradInputs(inputs);

    try analyticalBackward(ctx, loss_fn, inputs);

    var analytical: [n]?RawTensor = [_]?RawTensor{null} ** n;
    defer for (&analytical) |*g| {
        if (g.*) |*owned| {
            owned.deinit();
            g.* = null;
        }
    };

    var checked_any = false;
    inline for (0..n) |input_i| {
        if (inputs[input_i].requiresGrad()) {
            analytical[input_i] = (try inputs[input_i].grad_state.?.gradClone(ctx.allocator)) orelse
                return error.MissingAnalyticalGradient;
            checked_any = true;
        }
    }
    if (!checked_any) return error.NoVariableInputs;

    var result: Result = .{};
    inline for (0..n) |input_i| {
        if (inputs[input_i].requiresGrad()) {
            const grad_data = try analytical[input_i].?.dataConstChecked();
            const param_data = try inputs[input_i].value.dataChecked();
            if (grad_data.len != param_data.len) return error.GradientShapeMismatch;

            for (param_data, grad_data, 0..) |*param, g_ana_f32, elem_i| {
                const original = param.*;
                {
                    errdefer param.* = original;
                    param.* = original + @as(f32, @floatCast(options.eps));
                    const plus = try lossItem(ctx, loss_fn, inputs);
                    param.* = original - @as(f32, @floatCast(options.eps));
                    const minus = try lossItem(ctx, loss_fn, inputs);
                    param.* = original;

                    const g_ana: f64 = g_ana_f32;
                    const g_num = (plus - minus) / (2 * options.eps);
                    const abs_error = @abs(g_num - g_ana);
                    const denom = @max(@abs(g_num), @abs(g_ana));
                    const rel_error = if (denom == 0) 0 else abs_error / denom;
                    result.max_abs_error = @max(result.max_abs_error, abs_error);
                    result.max_rel_error = @max(result.max_rel_error, rel_error);
                    result.checked += 1;

                    const tol = options.abs_tol + options.rel_tol * @abs(g_ana);
                    if (abs_error > tol) {
                        if (options.print_mismatch) {
                            std.debug.print(
                                "gradcheck mismatch at input {d} element {d}: analytical {e} numerical {e} |dev| {e} > tol {e}\n",
                                .{ input_i, elem_i, g_ana, g_num, abs_error, tol },
                            );
                        }
                        return error.GradientMismatch;
                    }
                }
            }
        }
    }

    return result;
}

fn analyticalBackward(ctx: *ExecContext, comptime loss_fn: anytype, inputs: anytype) !void {
    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    const loss = try callLoss(loss_fn, ctx, inputs);
    try loss.backward(ctx);
}

fn lossItem(ctx: *ExecContext, comptime loss_fn: anytype, inputs: anytype) !f64 {
    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    const loss = try callLoss(loss_fn, ctx, inputs);
    return try loss.item();
}

fn callLoss(comptime loss_fn: anytype, ctx: *ExecContext, inputs: anytype) anyerror!StripError(@typeInfo(@TypeOf(loss_fn)).@"fn".return_type.?) {
    var args: std.meta.ArgsTuple(@TypeOf(loss_fn)) = undefined;
    args[0] = ctx;
    inline for (0..@typeInfo(@TypeOf(inputs)).@"struct".fields.len) |i| {
        args[i + 1] = inputs[i];
    }
    return @call(.auto, loss_fn, args);
}

fn zeroGradInputs(inputs: anytype) void {
    inline for (0..@typeInfo(@TypeOf(inputs)).@"struct".fields.len) |i| {
        inputs[i].zeroGrad();
    }
}

fn validateLoss(comptime loss_fn: anytype, comptime input_count: usize) void {
    const info = @typeInfo(@TypeOf(loss_fn));
    if (info != .@"fn") @compileError("gradcheck loss must be a comptime function");
    const fn_info = info.@"fn";
    if (fn_info.params.len != input_count + 1) {
        @compileError("gradcheck loss parameter count must be 1 (*ExecContext) plus the input tuple length");
    }
    const Ctx = fn_info.params[0].type orelse @compileError("gradcheck loss parameters must be concrete");
    if (Ctx != *ExecContext) @compileError("gradcheck loss first parameter must be *ExecContext");
    const Return = StripError(fn_info.return_type orelse @compileError("gradcheck loss must return a tensor"));
    validateFacade(Return, "gradcheck loss result");
    if (Return.axis_tags.len != 0) @compileError("gradcheck loss must return scalar Tensor(.{})");
}

fn inputFields(comptime Inputs: type) []const std.builtin.Type.StructField {
    const info = @typeInfo(Inputs);
    if (info != .@"struct" or !info.@"struct".is_tuple) {
        @compileError("gradcheck inputs must be a tuple of mutable f32 facade tensor pointers, got " ++ @typeName(Inputs));
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
    if (info != .pointer or info.pointer.is_const) {
        @compileError("gradcheck input must be a mutable pointer to an f32 facade tensor, got " ++ @typeName(P));
    }
    const T = info.pointer.child;
    validateFacade(T, "gradcheck input");
    return T;
}

fn validateFacade(comptime T: type, comptime what: []const u8) void {
    if (@typeInfo(T) != .@"struct" or !@hasDecl(T, "dtype") or T.dtype != .f32 or !@hasField(T, "grad_state")) {
        @compileError(what ++ " must be an f32 autograd facade tensor, got " ++ @typeName(T));
    }
}

fn StripError(comptime T: type) type {
    return switch (@typeInfo(T)) {
        .error_union => |eu| eu.payload,
        else => T,
    };
}

test {
    _ = @import("gradcheck_tests.zig");
}
