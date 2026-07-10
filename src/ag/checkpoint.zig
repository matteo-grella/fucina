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
//! - a comptime function `fn (*ExecContext, *const Tensor(..), ...) !Tensor(..)`
//!   over f32 facade tensors whose result is produced by facade ops on the
//!   inputs. `checkpoint` always runs the block under an exec scope, so the
//!   defer-deinit forward idiom works unchanged inside it (deinit on
//!   scope-owned results is a no-op, see docs/TRAINING.md);
//! - deterministic and pure in its inputs: the recompute must rebuild the
//!   exact forward values (RNG-using ops such as dropout must derive their
//!   stream from explicit stored seeds, not from ambient RNG state);
//! - no nested `checkpoint` call inside a block: the recompute lock is not
//!   reentrant, so a nested recompute is rejected with
//!   `error.NestedCheckpointRecompute` instead of deadlocking. The recompute
//!   backward is intentionally single-threaded at the NODE level
//!   (`core.backwardGradSerial`; kernel-level `parallelChunks` parallelism
//!   inside VJPs is unaffected) — that is what keeps a nested checkpoint node
//!   on the recompute thread, where the threadlocal guard can see it.
//!
//! Blocks that also need frozen state — quantized/f16/bf16 const facade
//! tensors, RoPE tables, config values, layer struct pointers — take it
//! through the `extra` argument of `checkpointWithContext`; only the
//! differentiable f32 inputs travel through `inputs`.
const std = @import("std");
const exec_mod = @import("../exec.zig");
const tensor_mod = @import("../tensor.zig");
const thread = @import("../thread.zig");
const control = @import("control.zig");
const core = @import("core.zig");

const Allocator = std.mem.Allocator;
const ExecContext = exec_mod.ExecContext;
const RawTensor = tensor_mod.Tensor;
const GradState = core.GradState;
const BackwardFunction = core.BackwardFunction;

/// Recomputes re-run facade ops, which adopt their results into
/// `ctx.rt.scope_entries` — not thread-safe — and the backward engine may invoke
/// independent checkpoint nodes from pool threads. One process-wide lock
/// serializes every recompute. Checkpoint nodes themselves always execute
/// synchronously on the scheduling thread (`prefer_async_backward` stays
/// false and `estimated_work` stays null, see core.canRunAsync).
var recompute_mutex: thread.Mutex = .{};

/// Detects a nested checkpoint recompute on the same thread, which would
/// self-deadlock on the non-reentrant `recompute_mutex`. Sound only because
/// the recompute's inner backward runs node-serial (`core.backwardGradSerial`
/// below): a nested checkpoint node can never be scheduled onto another
/// thread while a recompute is active.
threadlocal var recompute_active: bool = false;

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
    const facade_types = comptime facadeTypes(Inputs);
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
        var quant_gpu_scope = control.disableQuantDotGpu();
        defer quant_gpu_scope.close();

        var consts: FacadeTuple(Inputs) = undefined;
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
        break :value try out.value.clone(ctx.allocator);
    };
    errdefer out_value.deinit();

    if (!any_grad) {
        // No operand needs gradients: same tail as a no-grad facade op
        // (tensor.zig finishNoGrad), including outer-scope adoption.
        if (ctx.execScopeActive()) try ctx.reserveScopeSlot();
        var result = try Output.fromTensor(ctx, out_value);
        if (ctx.execScopeActive()) {
            adoptIntoScope(ctx, &result);
            result.scope_owned = true;
        }
        return result;
    }

    // Reserve the outer-scope slot BEFORE consuming views/value so adoption
    // cannot fail afterwards (same two-phase contract as tensor.zig finishOp).
    if (ctx.execScopeActive()) try ctx.reserveScopeSlot();
    const state = try core.createNode(CheckpointBackward(block, Extra, Inputs), .{ ctx.allocator, extra, views, states });
    node_owns_views = true;
    errdefer state.deinit();

    var result = Output{ .value = out_value, .grad_state = state };
    if (ctx.execScopeActive()) {
        adoptIntoScope(ctx, &result);
        result.scope_owned = true;
    }
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
    const fields = inputFields(Inputs);
    if (fn_info.params.len != fields.len + lead) {
        @compileError(std.fmt.comptimePrint(
            "checkpoint block takes {d} parameters but {s} + {d} inputs were supplied",
            .{ fn_info.params.len, if (Extra == void) "1 (*ExecContext)" else "2 (*ExecContext, extra)", fields.len },
        ));
    }
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
    for (fields, 0..) |field, i| {
        const Param = fn_info.params[i + lead].type orelse @compileError("checkpoint block parameters must be concrete (no anytype)");
        if (FacadeOf(Param) != FacadeOf(field.type)) {
            @compileError(std.fmt.comptimePrint(
                "checkpoint input {d} is {s} but the block expects {s}",
                .{ i, @typeName(field.type), @typeName(Param) },
            ));
        }
    }
    const ret = fn_info.return_type orelse @compileError("checkpoint block must have a concrete return type");
    const Out = StripError(ret);
    validateFacade(Out, "checkpoint block result");
    return Out;
}

/// Number of block parameters before the inputs: `*ExecContext` plus, for
/// context blocks, the `extra` value.
fn leadParamCount(comptime Extra: type) usize {
    return if (Extra == void) 1 else 2;
}

/// Backward node for `checkpoint`/`checkpointWithContext`: owns refcounted
/// views of the block inputs, stores `extra` by value (the caller keeps
/// whatever it points at alive until backward completes), and borrows the
/// inputs' GradStates (the operands); rebuilds the block subgraph on demand
/// inside backward.
fn CheckpointBackward(comptime block: anytype, comptime Extra: type, comptime Inputs: type) type {
    const facade_types = facadeTypes(Inputs);
    const n = facade_types.len;

    return struct {
        extra: Extra,
        views: [n]RawTensor,
        states: [n]?*GradState,

        const Self = @This();

        /// Consumes `views` on success; on error (the node allocation in
        /// `core.createNode`) they stay with the caller.
        pub fn init(self: *Self, allocator: Allocator, extra: Extra, views: [n]RawTensor, states: [n]?*GradState) !void {
            _ = allocator;
            self.* = .{ .extra = extra, .views = views, .states = states };
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

            // A nested recompute on this thread would self-deadlock on the
            // non-reentrant mutex; fail loudly instead.
            if (recompute_active) return error.NestedCheckpointRecompute;
            recompute_mutex.lock();
            recompute_active = true;
            defer {
                recompute_active = false;
                recompute_mutex.unlock();
            }

            // Rebuild the block subgraph inside an inner scope so the
            // recomputed intermediates are released on every exit path.
            const scope = ctx.openExecScope();
            defer ctx.closeExecScope(scope);
            var quant_gpu_scope = control.disableQuantDotGpu();
            defer quant_gpu_scope.close();

            // Inputs that need gradients come back as variables (fresh leaf
            // states local to this recompute); the rest stay constants.
            var rewrapped: FacadeTuple(Inputs) = undefined;
            var built: usize = 0;
            defer {
                inline for (0..n) |i| {
                    if (i < built) rewrapped[i].deinit();
                }
            }
            inline for (0..n) |i| {
                var view = try self.views[i].cloneView();
                errdefer view.deinit();
                rewrapped[i] = if (needs_grad[i])
                    try facade_types[i].variable(ctx, view)
                else
                    try facade_types[i].constant(ctx, view);
                built = i + 1;
            }

            const recomputed = try callBlock(block, ctx, self.extra, &rewrapped);
            const out_state = recomputed.grad_state orelse return error.CheckpointOutputNotDifferentiable;

            // Seed the recomputed output with the incoming gradient and run
            // a full backward over the recomputed subgraph. The SERIAL
            // variant keeps every recomputed node on this thread: a nested
            // checkpoint node scheduled onto a pool thread would pass the
            // threadlocal `recompute_active` check and deadlock on the held
            // `recompute_mutex` instead of erroring.
            out_state.setGrad(try gy.cloneView());
            try core.backwardGradSerial(ctx, &.{out_state}, &.{recomputed.asRawTensor()});

            // Deep-copy the input gradients out of the recompute-local leaf
            // states: they must survive the scope close below. On error the
            // engine deinits any slots already filled (core.executeBackward).
            inline for (0..n) |i| {
                if (needs_grad[i]) {
                    out[i] = (try rewrapped[i].grad_state.?.gradClone(ctx.allocator)) orelse
                        return error.CheckpointMissingInputGradient;
                }
            }
        }

        fn deinit(ptr: *anyopaque, allocator: Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            for (&self.views) |*view| view.deinit();
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

fn callBlock(
    comptime block: anytype,
    ctx: *ExecContext,
    extra: anytype,
    facades: anytype,
) anyerror!StripError(@typeInfo(@TypeOf(block)).@"fn".return_type.?) {
    const lead = comptime leadParamCount(@TypeOf(extra));
    var args: std.meta.ArgsTuple(@TypeOf(block)) = undefined;
    args[0] = ctx;
    if (comptime @TypeOf(extra) != void) args[1] = extra;
    inline for (0..args.len - lead) |i| {
        args[i + lead] = &facades.*[i];
    }
    return @call(.auto, block, args);
}

fn inputFields(comptime Inputs: type) []const std.builtin.Type.StructField {
    const info = @typeInfo(Inputs);
    if (info != .@"struct" or !info.@"struct".is_tuple) {
        @compileError("checkpoint inputs must be a tuple of facade tensor pointers, got " ++ @typeName(Inputs));
    }
    return info.@"struct".fields;
}

fn facadeTypes(comptime Inputs: type) [inputFields(Inputs).len]type {
    const fields = inputFields(Inputs);
    var types: [fields.len]type = undefined;
    for (fields, 0..) |field, i| types[i] = FacadeOf(field.type);
    return types;
}

fn FacadeTuple(comptime Inputs: type) type {
    const types = facadeTypes(Inputs);
    return std.meta.Tuple(&types);
}

fn FacadeOf(comptime Ptr: type) type {
    const info = @typeInfo(Ptr);
    if (info != .pointer or info.pointer.size != .one) {
        @compileError("checkpoint input must be a single-item pointer to a facade tensor, got " ++ @typeName(Ptr));
    }
    validateFacade(info.pointer.child, "checkpoint input");
    return info.pointer.child;
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

// Replicas of the private adoption tail in ag/tensor.zig (finishOp): hand the
// result's value + GradState to the innermost scope via the type-erased entry.
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

test {
    _ = @import("checkpoint_tests.zig");
}
