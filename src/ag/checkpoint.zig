//! Activation checkpointing (recompute-in-backward) over the autograd facade.
//!
//! `checkpoint` runs a block of facade ops while retaining only refcounted
//! views of the block INPUTS plus one deep copy of the block OUTPUT; every
//! intermediate the block creates is released as soon as the forward pass
//! returns (the block runs inside an inner exec scope that is closed
//! immediately). When gradients reach the checkpoint during backward, the
//! block is re-run on the stored inputs to rebuild the subgraph, the incoming
//! gradient is backpropagated through the recomputed subgraph, and the
//! resulting input gradients are handed to the outer engine. The classic
//! memory-for-compute trade: O(inputs + output) retained per checkpoint
//! instead of O(intermediates).
//!
//! Contract for `block`:
//! - a comptime function over f32 facade tensors whose result is produced by
//!   facade ops on the inputs, in one of two equivalent signature forms:
//!   one pointer parameter per input
//!   (`fn (*ExecContext, *const Tensor(..), ...) !Tensor(..)`), or ONE
//!   trailing tuple parameter carrying all inputs
//!   (`fn (*ExecContext, InputsTuple) !Tensor(..)` where `InputsTuple` is a
//!   tuple of facade-tensor pointer types matching `inputs`) — the tuple
//!   form serves blocks whose input arity is comptime-variable, e.g. a LoRA
//!   layer with N enabled adapters. `checkpoint` always runs the block under
//!   an exec scope, so the defer-deinit forward idiom works unchanged inside
//!   it (deinit on scope-owned results is a no-op, see docs/TRAINING.md);
//! - deterministic and pure in its inputs: the recompute must rebuild the
//!   exact forward values (RNG-using ops such as dropout must derive their
//!   stream from explicit stored seeds, not from ambient RNG state);
//! - nesting is allowed: a `checkpoint` inside a block is recomputed on a
//!   frame of its own, inside the outer recompute. The recompute backward
//!   is single-threaded at the NODE level (`core.backwardGradSerial`;
//!   kernel-level `parallelChunks` parallelism inside VJPs is unaffected),
//!   which is what confines a recompute frame to the thread running it.
//!
//! Threading: each recompute runs its facade ops on a scope stack that
//! lives in the recompute frame (`ExecContext.installScopeStack`), never on
//! the context's own stack, so independent checkpoint nodes driven from
//! pool threads, and recomputes on different contexts, proceed without any
//! shared lock. Checkpoint nodes themselves always execute synchronously on
//! the scheduling thread (`prefer_async_backward` stays false and
//! `estimated_work` stays null, see core.canRunAsync).
//!
//! Blocks that also need frozen state — quantized/f16/bf16 const facade
//! tensors, RoPE tables, config values, layer struct pointers — take it
//! through the `extra` argument of `checkpointWithContext`; only the
//! differentiable f32 inputs travel through `inputs`.
const std = @import("std");
const exec_mod = @import("../exec.zig");
const tensor_mod = @import("../tensor.zig");
const core = @import("core.zig");
const AgError = core.AgError;
const reflect = @import("facade_reflect.zig");
const input_pointer: reflect.Pointer = .{ .single_item = true };
const plumbing = @import("tensor/plumbing.zig").Mod(@import("tensor.zig"));

const Allocator = std.mem.Allocator;
const ExecContext = exec_mod.ExecContext;
const RawTensor = tensor_mod.Tensor;
const GradState = core.GradState;

/// Run `block` under activation checkpointing: forward stores only the inputs
/// (refcounted views); intermediates are freed immediately; backward re-runs
/// the block to rebuild them, then backprops through the recomputed subgraph.
/// `block` must be a deterministic pure function of its inputs (RNG-using ops
/// such as dropout must derive their stream from explicit stored seeds).
///
/// `inputs` is a tuple of pointers to f32 facade tensors matching the block's
/// parameters after the leading `*ExecContext`. The result follows the same
/// ownership contract as any facade op: caller-owned with no scope open,
/// adopted by the innermost scope (a borrow, `scope_owned`) otherwise.
pub fn checkpoint(ctx: *ExecContext, comptime block: anytype, inputs: anytype) !BlockOutput(block, @TypeOf(inputs)) {
    return checkpointImpl(ctx, block, {}, inputs);
}

/// `checkpoint` with a pass-through context argument: `extra` sits between
/// `*ExecContext` and the input pointers in the block signature, is stored BY
/// VALUE in the backward node, and is handed verbatim to both the no-grad
/// forward run and the backward recompute. It is the channel for everything a
/// block needs that is not a differentiable f32 input: frozen weights
/// (quantized/f16/bf16 const facade tensors), RoPE tables, config values,
/// layer struct pointers.
///
/// Contract for `extra` (on top of the `block` contract above):
/// - plain data / pointers that REMAIN VALID until backward completes — the
///   node keeps only the value bits, it does not deep-copy or refcount
///   anything reachable through them;
/// - the block treats everything reachable through `extra` as constants:
///   tensors reachable through `extra` never receive gradients (deliver
///   anything trainable through `inputs` instead);
/// - the block must stay deterministic in (`extra`, `inputs`): the recompute
///   re-runs it with the identical `extra` value and must rebuild the exact
///   forward values.
///
/// A void `extra` (`{}`) degenerates to plain `checkpoint`.
pub fn checkpointWithContext(ctx: *ExecContext, comptime block: anytype, extra: anytype, inputs: anytype) !BlockOutputWithContext(block, @TypeOf(extra), @TypeOf(inputs)) {
    return checkpointImpl(ctx, block, extra, inputs);
}

/// Shared machinery behind `checkpoint` (void `extra`) and
/// `checkpointWithContext`.
fn checkpointImpl(ctx: *ExecContext, comptime block: anytype, extra: anytype, inputs: anytype) !BlockOutputImpl(block, @TypeOf(extra), @TypeOf(inputs)) {
    const Extra = @TypeOf(extra);
    const Inputs = @TypeOf(inputs);
    const Output = BlockOutputImpl(block, Extra, Inputs);
    const facade_types = comptime reflect.facadeTypes(Inputs, "checkpoint", input_pointer);
    const n = facade_types.len;

    var any_grad = false;
    inline for (0..n) |i| {
        if (inputs[i].requiresGrad()) any_grad = true;
    }

    // Snapshot the inputs for the recompute before running the block: the
    // backward node stores refcounted views of the input values plus the
    // input GradState pointers (its operands).
    var views: [n]RawTensor = undefined;
    var states: [n]?*GradState = undefined;
    var captured: usize = 0;
    var node_owns_views = false;
    errdefer if (!node_owns_views) for (views[0..captured]) |*view| view.deinit();
    if (any_grad) {
        inline for (0..n) |i| {
            views[i] = try inputs[i].value.cloneView();
            states[i] = inputs[i].grad_state;
            captured = i + 1;
        }
    }

    // Run the block on grad-free constants inside an inner exec scope: no
    // backward nodes are built, and closing the scope frees every block
    // intermediate immediately — only a deep copy of the output survives.
    // This is the entire memory win.
    var out_value = value: {
        const inner = ctx.openExecScope();
        defer ctx.closeExecScope(inner);
        var quant_gpu_scope = ctx.disableQuantDotGpu();
        defer quant_gpu_scope.close();

        var consts: reflect.FacadeTuple(Inputs, "checkpoint", input_pointer) = undefined;
        var built: usize = 0;
        defer {
            inline for (0..n) |i| {
                if (i < built) consts[i].deinit();
            }
        }
        inline for (0..n) |i| {
            var view = try inputs[i].value.cloneView();
            errdefer view.deinit();
            consts[i] = try facade_types[i].constant(ctx, view);
            built = i + 1;
        }

        // The block result is an op result, so the inner scope owns it (along
        // with everything else the block built); it is not deinited here.
        const out = try callBlock(block, ctx, extra, &consts);
        break :value try out.value.clone(ctx.allocator());
    };
    errdefer out_value.deinit();

    if (!any_grad) {
        // No operand needs gradients: same tail as a no-grad facade op
        // (plumbing.finishNoGrad), including outer-scope adoption.
        return plumbing.finishTyped(Output, ctx, out_value);
    }

    // Same tail as finishOp: adoption is the one fallible step after the
    // record is built, and it happens before the value hand-off.
    const result = try plumbing.finishWithRecord(Output, ctx, out_value, CheckpointBackward(block, Extra, Inputs){ .extra = extra, .views = views, .states = states });
    node_owns_views = true;
    return result;
}

/// Result type of `checkpoint(ctx, block, inputs)`: the block's return type
/// with the error union stripped.
pub fn BlockOutput(comptime block: anytype, comptime Inputs: type) type {
    return BlockOutputImpl(block, void, Inputs);
}

/// Result type of `checkpointWithContext(ctx, block, extra, inputs)`: the
/// block's return type with the error union stripped.
pub fn BlockOutputWithContext(comptime block: anytype, comptime Extra: type, comptime Inputs: type) type {
    return BlockOutputImpl(block, Extra, Inputs);
}

/// Shared result-type computation; also the home of the comptime
/// block/extra/inputs signature validation. `Extra == void` means the block
/// signature has no extra parameter (the plain `checkpoint` shape).
fn BlockOutputImpl(comptime block: anytype, comptime Extra: type, comptime Inputs: type) type {
    const F = @TypeOf(block);
    const fn_info = switch (@typeInfo(F)) {
        .@"fn" => |info| info,
        else => @compileError("checkpoint block must be a comptime function, got " ++ @typeName(F)),
    };
    const lead = leadParamCount(Extra);
    const fields = reflect.inputFields(Inputs, "checkpoint");
    const Ctx = fn_info.params[0].type orelse @compileError("checkpoint block parameters must be concrete (no anytype)");
    if (Ctx != *ExecContext) {
        @compileError("checkpoint block must take *ExecContext as its first parameter, got " ++ @typeName(Ctx));
    }
    if (Extra != void) {
        const Param = fn_info.params[1].type orelse @compileError("checkpoint block parameters must be concrete (no anytype)");
        if (Param != Extra) {
            @compileError("checkpointWithContext extra is " ++ @typeName(Extra) ++ " but the block expects " ++ @typeName(Param));
        }
    }
    if (blockTakesInputsTuple(fn_info, Extra)) {
        // Tuple form: the single trailing parameter carries every input.
        const param_fields = @typeInfo(fn_info.params[lead].type.?).@"struct".fields;
        if (param_fields.len != fields.len) {
            @compileError(std.fmt.comptimePrint(
                "checkpoint block inputs tuple has {d} entries but {d} inputs were supplied",
                .{ param_fields.len, fields.len },
            ));
        }
        for (fields, param_fields, 0..) |field, param_field, i| {
            if (reflect.FacadeOf(param_field.type, "checkpoint", input_pointer) != reflect.FacadeOf(field.type, "checkpoint", input_pointer)) {
                @compileError(std.fmt.comptimePrint(
                    "checkpoint input {d} is {s} but the block expects {s}",
                    .{ i, @typeName(field.type), @typeName(param_field.type) },
                ));
            }
        }
    } else {
        if (fn_info.params.len != fields.len + lead) {
            @compileError(std.fmt.comptimePrint(
                "checkpoint block takes {d} parameters but {s} + {d} inputs were supplied",
                .{ fn_info.params.len, if (Extra == void) "1 (*ExecContext)" else "2 (*ExecContext, extra)", fields.len },
            ));
        }
        for (fields, 0..) |field, i| {
            const Param = fn_info.params[i + lead].type orelse @compileError("checkpoint block parameters must be concrete (no anytype)");
            if (reflect.FacadeOf(Param, "checkpoint", input_pointer) != reflect.FacadeOf(field.type, "checkpoint", input_pointer)) {
                @compileError(std.fmt.comptimePrint(
                    "checkpoint input {d} is {s} but the block expects {s}",
                    .{ i, @typeName(field.type), @typeName(Param) },
                ));
            }
        }
    }
    const ret = fn_info.return_type orelse @compileError("checkpoint block must have a concrete return type");
    const Out = StripError(ret);
    reflect.validateFacade(Out, "checkpoint block result");
    return Out;
}

/// Number of block parameters before the inputs: `*ExecContext` plus, for
/// context blocks, the `extra` value.
fn leadParamCount(comptime Extra: type) usize {
    return if (Extra == void) 1 else 2;
}

/// True when the block declares the tuple form: exactly one trailing
/// parameter that is itself a tuple (of facade-tensor pointers), instead of
/// one pointer parameter per input. Unambiguous — a per-input parameter is
/// always a pointer, never a tuple.
fn blockTakesInputsTuple(comptime fn_info: std.builtin.Type.Fn, comptime Extra: type) bool {
    if (fn_info.params.len != leadParamCount(Extra) + 1) return false;
    const Param = fn_info.params[fn_info.params.len - 1].type orelse return false;
    const info = @typeInfo(Param);
    return info == .@"struct" and info.@"struct".is_tuple;
}

/// Backward node for `checkpoint`/`checkpointWithContext`: owns refcounted
/// views of the block inputs, stores `extra` by value (the caller keeps
/// whatever it points at alive until backward completes), and retains the
/// inputs' GradStates (the operands, one reference each); rebuilds the block
/// subgraph on demand inside backward.
fn CheckpointBackward(comptime block: anytype, comptime Extra: type, comptime Inputs: type) type {
    const facade_types = reflect.facadeTypes(Inputs, "checkpoint", input_pointer);
    const n = facade_types.len;

    return struct {
        extra: Extra,
        views: [n]RawTensor,
        states: [n]?*GradState,

        const Self = @This();

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
            // Wide input tuples (a packed LoRA layer runs 15) push the
            // unrolled per-input loops past the default comptime quota.
            @setEvalBranchQuota(1000 * (n + 1));
            std.debug.assert(out.len == n);

            // The recompute frame: a scope stack owned by this backward
            // call, installed for the calling thread so the re-run facade
            // ops adopt into it rather than into the context's own stack
            // (which another thread may be driving). Nothing is shared, so
            // recomputes never serialize on each other, and a nested
            // checkpoint inside the block simply installs its own frame
            // and hands this one back. The defers unwind in order: close
            // the scope, restore the outer frame, release the stack — so
            // every recomputed intermediate is freed on every exit path.
            var frame = ExecContext.ScopeStack{};
            defer frame.deinit(ctx.allocator());
            const outer_frame = ExecContext.installScopeStack(&frame);
            defer ExecContext.restoreScopeStack(outer_frame);
            const scope = ctx.openExecScope();
            defer ctx.closeExecScope(scope);
            var quant_gpu_scope = ctx.disableQuantDotGpu();
            defer quant_gpu_scope.close();

            // Inputs that need gradients come back as variables (fresh leaf
            // states local to this recompute); the rest stay constants.
            var rewrapped: reflect.FacadeTuple(Inputs, "checkpoint", input_pointer) = undefined;
            var built: usize = 0;
            defer {
                inline for (0..n) |i| {
                    if (i < built) rewrapped[i].deinit();
                }
            }
            inline for (0..n) |i| {
                var view = try self.views[i].cloneView();
                errdefer view.deinit();
                rewrapped[i] = if (core.needs(self, i))
                    try facade_types[i].variable(ctx, view)
                else
                    try facade_types[i].constant(ctx, view);
                built = i + 1;
            }

            const recomputed = try callBlock(block, ctx, self.extra, &rewrapped);
            const out_state = recomputed.grad_state orelse return AgError.NoGradientGraph;

            // Seed the recomputed output with the incoming gradient and run
            // a full backward over the recomputed subgraph. The SERIAL
            // variant keeps every recomputed node on this thread, where the
            // frame is installed: a nested checkpoint node scheduled onto a
            // pool thread would recompute on that thread's view of the
            // context (its own stack), not on this frame.
            out_state.setGrad(try gy.cloneView());
            try core.backwardGradSerial(ctx, &.{out_state}, &.{recomputed.asRawTensor()});

            // Deep-copy the input gradients out of the recompute-local leaf
            // states: they must survive the scope close below. On error the
            // engine deinits any slots already filled (core.executeBackward).
            inline for (0..n) |i| {
                if (core.needs(self, i)) {
                    out[i] = (try rewrapped[i].grad_state.?.gradClone(ctx.allocator())) orelse
                        return AgError.MissingBackwardGradient;
                }
            }
        }

        pub fn deinitFields(self: *Self, allocator: Allocator) void {
            _ = allocator;
            for (&self.views) |*view| view.deinit();
        }

        pub const vtable = core.recordVTable(Self);
    };
}

fn callBlock(
    comptime block: anytype,
    ctx: *ExecContext,
    extra: anytype,
    facades: anytype,
) anyerror!StripError(@typeInfo(@TypeOf(block)).@"fn".return_type.?) {
    const fn_info = @typeInfo(@TypeOf(block)).@"fn";
    const lead = comptime leadParamCount(@TypeOf(extra));
    var args: std.meta.ArgsTuple(@TypeOf(block)) = undefined;
    args[0] = ctx;
    if (comptime @TypeOf(extra) != void) args[1] = extra;
    if (comptime blockTakesInputsTuple(fn_info, @TypeOf(extra))) {
        var tuple: fn_info.params[lead].type.? = undefined;
        inline for (0..@typeInfo(@TypeOf(tuple)).@"struct".fields.len) |i| {
            tuple[i] = &facades.*[i];
        }
        args[lead] = tuple;
    } else {
        inline for (0..args.len - lead) |i| {
            args[i + lead] = &facades.*[i];
        }
    }
    return @call(.auto, block, args);
}

fn StripError(comptime T: type) type {
    return switch (@typeInfo(T)) {
        .error_union => |eu| eu.payload,
        else => T,
    };
}

test {
    _ = @import("checkpoint_tests.zig");
}
