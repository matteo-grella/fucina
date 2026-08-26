const std = @import("std");
const exec_mod = @import("../exec.zig");
const parallel = @import("../parallel.zig");
const tensor = @import("../tensor.zig");
const thread = @import("../thread.zig");

const Allocator = std.mem.Allocator;
const ExecContext = exec_mod.ExecContext;
const Tensor = tensor.Tensor;

pub const AgError = error{
    MissingOutputGradient,
    MissingBackwardGradient,
    BackwardAlreadyRun,
};

const BackwardState = enum(u8) {
    idle,
    pending,
    ongoing,
};

pub const BackwardFunction = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        operands: *const fn (*const anyopaque) []const ?*GradState,
        backward: *const fn (*anyopaque, *ExecContext, *const Tensor, []?Tensor) anyerror!void,
        deinit: *const fn (*anyopaque, Allocator) void,
        prefer_async_backward: bool = false,
        estimated_work: ?*const fn (*const anyopaque) usize = null,
    };

    pub fn operands(self: BackwardFunction) []const ?*GradState {
        return self.vtable.operands(self.ptr);
    }

    pub fn backward(self: BackwardFunction, ctx: *ExecContext, gy: *const Tensor, out: []?Tensor) !void {
        return self.vtable.backward(self.ptr, ctx, gy, out);
    }

    pub fn deinit(self: BackwardFunction, allocator: Allocator) void {
        self.vtable.deinit(self.ptr, allocator);
    }

    pub fn preferAsyncBackward(self: BackwardFunction) bool {
        return self.vtable.prefer_async_backward;
    }

    pub fn estimatedWork(self: BackwardFunction) ?usize {
        const estimate = self.vtable.estimated_work orelse return null;
        return estimate(self.ptr);
    }
};

/// Header + typed backward record co-allocated as ONE heap node. Every
/// GradState with `grad_fn != null` is the `state` field of a `BackwardNode(Record)`;
/// the record's vtable deinit releases the record's resources and frees the
/// whole node (see `destroyNode`). Leaves stay standalone `GradState`
/// allocations (`GradState.leaf`).
pub fn BackwardNode(comptime Record: type) type {
    return struct {
        state: GradState,
        record: Record,
    };
}

/// Allocate one `BackwardNode(@TypeOf(record))`, move `record` into it,
/// wire the header to the record's `pub const vtable`, and retain every
/// non-null operand: the node holds one reference per parent for as long
/// as it lives (dropped by the record's vtable deinit through
/// `releaseParents`), so a parent handle may be released at any time
/// without dangling the graph. The record is a typed struct literal built
/// by the op: the views and slices it saves were taken by the op and are
/// owned by the node from here on; on allocation failure they stay with
/// the op (its errdefers). This is the last fallible step of an op tail.
/// The returned state carries one reference for the caller;
/// `GradState.release` drops it and, as the last one, frees the entire
/// node through the vtable.
pub fn createNode(allocator: Allocator, record: anytype) !*GradState {
    const Record = @TypeOf(record);
    const node = try allocator.create(BackwardNode(Record));
    node.record = record;
    node.state = .{
        .allocator = allocator,
        .grad_fn = .{ .ptr = &node.record, .vtable = &Record.vtable },
    };
    retainParents(Record.vtable.operands(&node.record));
    return &node.state;
}

/// One reference per non-null operand, taken by `createNode`.
pub fn retainParents(parents: []const ?*GradState) void {
    for (parents) |parent| {
        if (parent) |state| _ = state.retain();
    }
}

/// Drop the references `retainParents` took: the head of every record
/// vtable deinit (`recordVTable` does it; a hand-written vtable calls it
/// before releasing anything the operand slice lives in). Releasing a
/// parent may free it, and with it its own parents, recursively.
pub fn releaseParents(parents: []const ?*GradState) void {
    for (parents) |parent| {
        if (parent) |state| state.release();
    }
}

/// Tail of every record vtable deinit: recover the co-allocated node from the
/// record pointer and free it (header included).
pub fn destroyNode(comptime Record: type, allocator: Allocator, record: *Record) void {
    const node: *BackwardNode(Record) = @fieldParentPtr("record", record);
    allocator.destroy(node);
}

/// The operand slots of a record: its `parents` field (an array or a
/// slice of `?*GradState`) or, for the records that name it so, `states`.
pub fn recordOperands(record: anytype) []const ?*GradState {
    const Record = @TypeOf(record.*);
    if (comptime @hasField(Record, "parents")) return record.parents[0..];
    if (comptime @hasField(Record, "states")) return record.states[0..];
    @compileError(@typeName(Record) ++ " has neither a `parents` nor a `states` field");
}

/// True when operand slot `i` needs a gradient: the slot holds a state.
/// The engine sizes `out` to the operand count, so this is the only test a
/// VJP needs before writing `out[i]`.
pub fn needs(record: anytype, i: usize) bool {
    return recordOperands(record)[i] != null;
}

/// Synthesize a record's `BackwardFunction.VTable` from its typed decls,
/// replacing the hand-written anyopaque plumbing every record used to
/// repeat:
/// - `operands` returns the record's operand slots (`recordOperands`);
/// - `backward` casts and delegates to `Record.vjp(self, ctx, gy, out)`,
///   the record's typed backward body (`core.needs(self, i)` says which
///   `out[i]` to fill);
/// - `deinit` releases the operand references (`releaseParents`), runs
///   `Record.deinitFields(self, allocator)` iff declared (records owning
///   tensors/slices release them there), then frees the co-allocated node;
/// - `.estimated_work` is wired automatically iff the record carries an
///   `estimated_work` field, so a record can never hold the field and
///   silently lose async backward scheduling to a forgotten vtable line;
/// - `.prefer_async_backward` from an optional `pub const` of that name.
pub fn recordVTable(comptime Record: type) BackwardFunction.VTable {
    const Shim = struct {
        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Record = @ptrCast(@alignCast(ptr));
            return recordOperands(self);
        }

        fn backward(ptr: *anyopaque, ctx: *ExecContext, gy: *const Tensor, out: []?Tensor) anyerror!void {
            const self: *Record = @ptrCast(@alignCast(ptr));
            return Record.vjp(self, ctx, gy, out);
        }

        fn deinit(ptr: *anyopaque, allocator: Allocator) void {
            const self: *Record = @ptrCast(@alignCast(ptr));
            releaseParents(recordOperands(self));
            if (comptime @hasDecl(Record, "deinitFields")) self.deinitFields(allocator);
            destroyNode(Record, allocator, self);
        }

        fn estimatedWork(ptr: *const anyopaque) usize {
            const self: *const Record = @ptrCast(@alignCast(ptr));
            return self.estimated_work;
        }
    };
    return .{
        .operands = Shim.operands,
        .backward = Shim.backward,
        .deinit = Shim.deinit,
        .prefer_async_backward = if (@hasDecl(Record, "prefer_async_backward")) Record.prefer_async_backward else false,
        .estimated_work = if (@hasField(Record, "estimated_work")) Shim.estimatedWork else null,
    };
}

pub const GradState = struct {
    allocator: Allocator,
    grad: ?Tensor = null,
    grad_fn: ?BackwardFunction = null,
    state: std.atomic.Value(u8) = std.atomic.Value(u8).init(@intFromEnum(BackwardState.idle)),
    pending_grads: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    grad_mutex: thread.Mutex = .{},
    /// Set once a backward pass with this state as an OUTPUT completes.
    /// The pass leaves its gradient contributions accumulated in every
    /// interior state of the graph, so a second pass over the same graph
    /// would compound them; `backwardGradImpl` rejects a marked output with
    /// `AgError.BackwardAlreadyRun` before installing any scheduling state.
    /// Leaves (`grad_fn == null`) are never marked — they have no graph to
    /// consume. Touched only on the thread driving the pass, never from
    /// pool tasks, so it needs no synchronization.
    backward_done: bool = false,
    /// Set on the outputs a backward pass was asked for. Their gradients
    /// are results and stay readable after the pass; every other interior
    /// gradient is released as soon as its own backward has consumed it
    /// (leaves have no backward and keep theirs for the optimizer).
    pass_output: bool = false,
    /// Reference count. Every owner holds exactly one reference: a facade
    /// handle, a consumer record (one per operand slot, taken by
    /// `createNode`), an exec-scope entry. Starts at one for the creator.
    /// Atomic because a record may be destroyed on the thread that closes
    /// a scope while pool tasks of a finished backward are still unwinding
    /// their own handles.
    refs: std.atomic.Value(u32) = std.atomic.Value(u32).init(1),

    pub fn leaf(allocator: Allocator) !*GradState {
        const self = try allocator.create(GradState);
        self.* = .{ .allocator = allocator };
        return self;
    }

    /// Take one more reference; returns `self` so a retained pointer can be
    /// stored in one expression.
    pub fn retain(self: *GradState) *GradState {
        _ = self.refs.fetchAdd(1, .monotonic);
        return self;
    }

    /// Drop one reference. The last release frees the state: a leaf
    /// directly, an interior node through its record vtable (which also
    /// releases the node's operand references). `self` is dangling after
    /// the last release; a handle that still holds a reference may keep
    /// using it.
    pub fn release(self: *GradState) void {
        if (self.refs.fetchSub(1, .acq_rel) != 1) return;
        self.zeroGrad();
        if (self.grad_fn) |function| {
            // Frees the whole co-allocated node, self included.
            function.deinit(self.allocator);
            return;
        }
        self.allocator.destroy(self);
    }

    pub fn zeroGrad(self: *GradState) void {
        self.grad_mutex.lock();
        defer self.grad_mutex.unlock();

        if (self.grad) |*g| {
            g.deinit();
            self.grad = null;
        }
    }

    pub fn setGrad(self: *GradState, grad: Tensor) void {
        self.zeroGrad();
        self.grad_mutex.lock();
        defer self.grad_mutex.unlock();
        self.grad = grad;
    }

    pub fn gradClone(self: *GradState, allocator: Allocator) !?Tensor {
        self.grad_mutex.lock();
        defer self.grad_mutex.unlock();
        if (self.grad) |*g| {
            return try g.clone(allocator);
        }
        return null;
    }

    pub fn gradView(self: *GradState) !?Tensor {
        self.grad_mutex.lock();
        defer self.grad_mutex.unlock();
        if (self.grad) |*g| {
            return try g.cloneView();
        }
        return null;
    }

    fn hasGradLocked(self: *GradState) bool {
        return self.grad != null;
    }

    fn prepareBackwardPass(self: *GradState) void {
        _ = self.pending_grads.fetchAdd(1, .monotonic);
        if (!self.compareState(.idle, .pending)) {
            return;
        }

        if (self.grad_fn) |function| {
            for (function.operands()) |operand| {
                if (operand) |state| state.prepareBackwardPass();
            }
        }
    }

    /// Seeding phase 1, run BEFORE `prepareBackwardPass` installs pending
    /// counters (see `backwardGradImpl`): validates that this output can be
    /// seeded and returns the implicit scalar seed to contribute in phase 2.
    /// Null means a gradient was installed before the pass started (e.g. the
    /// checkpoint recompute's `setGrad`); that explicit seed is used as-is,
    /// never topped up with the implicit 1.
    fn prepareOutputSeed(self: *GradState, ctx: *ExecContext, output_value: *const Tensor) !?Tensor {
        self.grad_mutex.lock();
        const has_grad = self.hasGradLocked();
        self.grad_mutex.unlock();
        if (has_grad) return null;
        if (output_value.isScalar()) return try ctx.scalar(.f32, 1);
        return AgError.MissingOutputGradient;
    }

    /// Seeding phase 2, run with the counters installed: contribute the seed
    /// from `prepareOutputSeed`. A scalar output whose gradient appeared only
    /// MID-pass (an earlier output's backward already reached it) still
    /// accumulates its own seed on top here.
    fn assignOutputGradient(self: *GradState, engine: *GradEngine, seed: ?Tensor) !void {
        if (seed) |owned| {
            return self.accGradOwned(engine, owned);
        }
        self.finishGradContribution(engine);
    }

    fn accGradOwned(self: *GradState, engine: *GradEngine, gx: Tensor) !void {
        if (try self.accGradOwnedReady(engine, gx)) {
            engine.scheduleReady(self);
        }
    }

    fn prepareInitialAccumulator(engine: *GradEngine, value: *Tensor, will_accumulate_more: bool) !void {
        if (!will_accumulate_more or value.canTakeInPlace()) return;
        const materialized = try engine.ctx.materialize(.f32, value);
        value.deinit();
        value.* = materialized;
    }

    fn prepareMutableAccumulator(engine: *GradEngine, current: *Tensor) !void {
        if (current.canTakeInPlace()) return;
        const materialized = try engine.ctx.materialize(.f32, current);
        current.deinit();
        current.* = materialized;
    }

    fn accGradOwnedReady(self: *GradState, engine: *GradEngine, gx: Tensor) !bool {
        var owned = gx;
        var moved = false;
        var must_finish = self.loadState() != .idle;
        errdefer if (must_finish) {
            if (self.finishGradContributionReady()) engine.scheduleReady(self);
        };
        errdefer if (!moved) owned.deinit();
        const will_accumulate_more = self.pending_grads.load(.acquire) > 1;
        {
            self.grad_mutex.lock();
            defer self.grad_mutex.unlock();

            if (self.grad) |*current| {
                try prepareMutableAccumulator(engine, current);
                try engine.ctx.elementwiseInPlace(.add, current, &owned);
            } else {
                try prepareInitialAccumulator(engine, &owned, will_accumulate_more);
                self.grad = owned;
                moved = true;
            }
        }
        if (!moved) {
            owned.deinit();
        }

        if (self.loadState() != .idle) {
            must_finish = false;
            return self.finishGradContributionReady();
        }

        must_finish = false;
        return false;
    }

    fn finishGradContribution(self: *GradState, engine: *GradEngine) void {
        if (self.finishGradContributionReady()) {
            engine.scheduleReady(self);
        }
    }

    fn finishGradContributionReady(self: *GradState) bool {
        const old = self.pending_grads.fetchSub(1, .acq_rel);
        std.debug.assert(old > 0);
        return old == 1;
    }

    fn executeBackward(self: *GradState, engine: *GradEngine) !void {
        defer self.storeState(.idle);

        const function = self.grad_fn orelse return;
        const operands = function.operands();

        self.grad_mutex.lock();
        const gy = if (self.grad) |*g| g else null;
        self.grad_mutex.unlock();
        const local_gy = gy orelse return;

        const stack_operand_capacity = 8;
        var gxs_stack: [stack_operand_capacity]?Tensor = undefined;
        var gxs_heap: ?[]?Tensor = null;
        defer if (gxs_heap) |buf| engine.allocator.free(buf);
        const gxs = if (operands.len <= stack_operand_capacity)
            gxs_stack[0..operands.len]
        else blk: {
            const buf = try engine.allocator.alloc(?Tensor, operands.len);
            gxs_heap = buf;
            break :blk buf;
        };
        @memset(gxs, null);
        defer {
            for (gxs) |*gx| {
                if (gx.*) |*owned| {
                    owned.deinit();
                    gx.* = null;
                }
            }
        }

        function.backward(engine.ctx, local_gy, gxs) catch |err| {
            for (operands, gxs) |operand, *gx| {
                if (gx.*) |*owned| {
                    owned.deinit();
                    gx.* = null;
                }
                if (operand) |state| state.finishGradContribution(engine);
            }
            return err;
        };

        var ready_stack: [stack_operand_capacity]*GradState = undefined;
        var ready_heap: ?[]*GradState = null;
        defer if (ready_heap) |buf| engine.allocator.free(buf);
        const ready = if (operands.len <= stack_operand_capacity)
            ready_stack[0..operands.len]
        else blk: {
            const buf = try engine.allocator.alloc(*GradState, operands.len);
            ready_heap = buf;
            break :blk buf;
        };
        var ready_len: usize = 0;

        var missing_backward_gradient = false;
        var first_error: ?anyerror = null;
        for (operands, gxs) |operand, *gx| {
            const state = operand orelse continue;
            if (gx.*) |owned| {
                gx.* = null;
                if (state.accGradOwnedReady(engine, owned) catch |err| blk: {
                    if (first_error == null) first_error = err;
                    break :blk false;
                }) {
                    ready[ready_len] = state;
                    ready_len += 1;
                }
            } else {
                state.finishGradContribution(engine);
                missing_backward_gradient = true;
            }
        }

        // An interior gradient has no consumer once this node's backward
        // has run: release it here instead of at scope close, so the
        // backward's memory is a moving window rather than a second copy
        // of the forward. Pass outputs keep theirs (they are results).
        if (!self.pass_output) self.zeroGrad();

        engine.scheduleReadyBatch(ready[0..ready_len]);
        if (first_error) |err| return err;
        if (missing_backward_gradient) return AgError.MissingBackwardGradient;
    }

    fn compareState(self: *GradState, expected: BackwardState, desired: BackwardState) bool {
        return self.state.cmpxchgStrong(
            @intFromEnum(expected),
            @intFromEnum(desired),
            .acq_rel,
            .acquire,
        ) == null;
    }

    fn loadState(self: *const GradState) BackwardState {
        return @enumFromInt(self.state.load(.acquire));
    }

    fn storeState(self: *GradState, state: BackwardState) void {
        self.state.store(@intFromEnum(state), .release);
    }
};

pub const GradEngine = struct {
    allocator: Allocator,
    ctx: *ExecContext,
    pool: ?*thread.Pool,
    wait_group: thread.WaitGroup = .{},
    wait_group_mutex: thread.Mutex = .{},
    active_tasks: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
    done_mutex: thread.Mutex = .{},
    done_cond: thread.Condition = .{},
    error_mutex: thread.Mutex = .{},
    first_error: ?anyerror = null,

    pub fn init(self: *GradEngine, ctx: *ExecContext, jobs: ?u32) !void {
        _ = jobs;
        self.* = .{
            .allocator = ctx.allocator,
            .ctx = ctx,
            .pool = ctx.tryWorkPool() catch null,
        };
    }

    pub fn deinit(self: *GradEngine) void {
        if (self.pool) |pool| pool.waitAndWork(&self.wait_group);
    }

    fn scheduleReady(self: *GradEngine, state: *GradState) void {
        self.scheduleReadyMode(state, false);
    }

    fn scheduleReadyBatch(self: *GradEngine, states: []const *GradState) void {
        var async_candidates: usize = 0;
        for (states) |state| {
            if (self.isAsyncCandidate(state)) async_candidates += 1;
        }

        var async_to_spawn = if (async_candidates > 1) async_candidates - 1 else 0;
        for (states) |state| {
            const spawn = async_to_spawn > 0 and self.isAsyncCandidate(state);
            if (spawn) async_to_spawn -= 1;
            self.scheduleReadyMode(state, spawn);
        }
    }

    fn scheduleReadyMode(self: *GradEngine, state: *GradState, allow_async: bool) void {
        if (!state.compareState(.pending, .ongoing)) {
            return;
        }
        _ = self.active_tasks.fetchAdd(1, .monotonic);
        const function = state.grad_fn orelse {
            runGradBackwardTask(self, state);
            return;
        };
        if (!allow_async or !self.canRunAsync(function)) {
            runGradBackwardTask(self, state);
            return;
        }
        const pool = self.pool orelse {
            runGradBackwardTask(self, state);
            return;
        };
        self.wait_group_mutex.lock();
        const spawned = pool.trySpawnWg(&self.wait_group, runGradBackwardTask, .{ self, state });
        self.wait_group_mutex.unlock();
        if (!spawned) {
            runGradBackwardTask(self, state);
        }
    }

    fn isAsyncCandidate(self: *const GradEngine, state: *const GradState) bool {
        if (self.pool == null) return false;
        const function = state.grad_fn orelse return false;
        return self.canRunAsync(function);
    }

    fn canRunAsync(_: *const GradEngine, function: BackwardFunction) bool {
        if (comptime !exec_mod.parallel_dot_backward_branches) return false;
        if (function.preferAsyncBackward()) return true;
        const work = function.estimatedWork() orelse return false;
        return work >= parallel.backward_async_work_threshold;
    }

    fn recordError(self: *GradEngine, err: anyerror) void {
        self.error_mutex.lock();
        defer self.error_mutex.unlock();
        if (self.first_error == null) {
            self.first_error = err;
        }
    }

    fn takeError(self: *GradEngine) ?anyerror {
        self.error_mutex.lock();
        defer self.error_mutex.unlock();
        return self.first_error;
    }

    fn waitAll(self: *GradEngine) void {
        const pool = self.pool orelse {
            std.debug.assert(self.active_tasks.load(.acquire) == 0);
            return;
        };
        self.done_mutex.lock();
        while (self.active_tasks.load(.acquire) != 0) {
            self.done_cond.wait(pool.io, &self.done_mutex);
        }
        self.done_mutex.unlock();
        pool.waitAndWork(&self.wait_group);
    }

    fn taskDone(self: *GradEngine) void {
        const old = self.active_tasks.fetchSub(1, .acq_rel);
        std.debug.assert(old > 0);
        if (old == 1) {
            const pool = self.pool orelse return;
            self.done_mutex.lock();
            self.done_cond.broadcast(pool.io);
            self.done_mutex.unlock();
        }
    }
};

fn runGradBackwardTask(engine: *GradEngine, state: *GradState) void {
    defer engine.taskDone();
    state.executeBackward(engine) catch |err| {
        state.storeState(.idle);
        engine.recordError(err);
    };
}

pub fn backwardGrad(ctx: *ExecContext, outputs: []const *GradState, output_values: []const *const Tensor) !void {
    return backwardGradImpl(ctx, outputs, output_values, true);
}

/// As `backwardGrad`, but with node-level async spawning disabled (the engine
/// runs with `pool = null`, so every backward node executes inline on the
/// calling thread). Kernel-level `parallelChunks` parallelism inside the
/// individual VJPs is unaffected. Used by the checkpoint recompute
/// (ag/checkpoint.zig), whose threadlocal nested-recompute guard is only
/// sound when the whole recomputed subgraph stays on one thread.
pub fn backwardGradSerial(ctx: *ExecContext, outputs: []const *GradState, output_values: []const *const Tensor) !void {
    return backwardGradImpl(ctx, outputs, output_values, false);
}

fn backwardGradImpl(ctx: *ExecContext, outputs: []const *GradState, output_values: []const *const Tensor, allow_async: bool) !void {
    if (outputs.len == 0) return;
    if (outputs.len != output_values.len) return AgError.MissingOutputGradient;

    var engine: GradEngine = undefined;
    try engine.init(ctx, null);
    if (!allow_async) engine.pool = null;
    defer engine.deinit();

    // Validate every output and pre-allocate the implicit scalar seeds before
    // any pending counter exists: an error exit after `prepareBackwardPass`
    // would strand nonzero counters, and the next backward over the same
    // states would stop at their `.pending` check and report success with
    // missing gradients.
    const stack_output_capacity = 8;
    var seeds_stack: [stack_output_capacity]?Tensor = undefined;
    var seeds_heap: ?[]?Tensor = null;
    defer if (seeds_heap) |buf| ctx.allocator.free(buf);
    const seeds = if (outputs.len <= stack_output_capacity)
        seeds_stack[0..outputs.len]
    else blk: {
        const buf = try ctx.allocator.alloc(?Tensor, outputs.len);
        seeds_heap = buf;
        break :blk buf;
    };
    @memset(seeds, null);
    defer {
        for (seeds) |*seed| {
            if (seed.*) |*owned| {
                owned.deinit();
                seed.* = null;
            }
        }
    }
    for (outputs, output_values, seeds) |output, output_value, *seed| {
        if (output.backward_done) return AgError.BackwardAlreadyRun;
        seed.* = try output.prepareOutputSeed(ctx, output_value);
    }
    for (outputs) |output| output.pass_output = true;

    for (outputs) |output| {
        output.prepareBackwardPass();
    }
    for (outputs, seeds) |output, *seed| {
        const owned = seed.*;
        seed.* = null;
        output.assignOutputGradient(&engine, owned) catch |err| engine.recordError(err);
    }

    engine.waitAll();
    if (engine.takeError()) |err| return err;

    // The completed pass consumed the graph: interior states retain their
    // accumulated gradients, so re-running over the same graph would compound
    // them (one backward per graph; see docs/reference/05-automatic-differentiation.md). Failed passes
    // stay unmarked and re-runnable; leaf outputs have no graph to consume.
    for (outputs) |output| {
        if (output.grad_fn != null) output.backward_done = true;
    }
}

pub fn backwardGradOne(ctx: *ExecContext, output: *GradState, output_value: *const Tensor) !void {
    return backwardGrad(ctx, &.{output}, &.{output_value});
}

test {
    _ = @import("core_tests.zig");
}

test "backward scheduler releases pending operand on missing gradient" {
    const MissingGradientBackward = struct {
        parents: [1]?*GradState,

        const Self = @This();

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const Tensor, out: []?Tensor) !void {
            _ = ctx;
            _ = gy;
            try std.testing.expect(needs(self, 0));
            try std.testing.expectEqual(@as(usize, 1), out.len);
            out[0] = null;
        }

        pub const vtable = recordVTable(Self);
    };

    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const parent = try GradState.leaf(ctx.allocator);
    defer parent.release();

    var output_value = try ctx.scalar(.f32, 0);
    defer output_value.deinit();

    const output = try createNode(ctx.allocator, MissingGradientBackward{ .parents = .{parent} });
    defer output.release();

    try std.testing.expectError(AgError.MissingBackwardGradient, backwardGradOne(&ctx, output, &output_value));
    try std.testing.expectEqual(@as(u32, 0), parent.pending_grads.load(.acquire));
    try std.testing.expectEqual(BackwardState.idle, parent.loadState());
}

test "backward scheduler releases pending operand on backward error" {
    const FailingBackward = struct {
        parents: [1]?*GradState,

        const Self = @This();
        const BackwardError = error{FailedBackward};

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const Tensor, out: []?Tensor) !void {
            _ = ctx;
            _ = gy;
            try std.testing.expect(needs(self, 0));
            try std.testing.expectEqual(@as(usize, 1), out.len);
            return BackwardError.FailedBackward;
        }

        pub const vtable = recordVTable(Self);
    };

    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const parent = try GradState.leaf(ctx.allocator);
    defer parent.release();

    var output_value = try ctx.scalar(.f32, 0);
    defer output_value.deinit();

    const output = try createNode(ctx.allocator, FailingBackward{ .parents = .{parent} });
    defer output.release();

    try std.testing.expectError(FailingBackward.BackwardError.FailedBackward, backwardGradOne(&ctx, output, &output_value));
    try std.testing.expectEqual(@as(u32, 0), parent.pending_grads.load(.acquire));
    try std.testing.expectEqual(BackwardState.idle, parent.loadState());
}
