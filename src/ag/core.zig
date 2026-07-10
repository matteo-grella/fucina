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
        backward: *const fn (*const anyopaque, *ExecContext, *const Tensor, []const bool, []?Tensor) anyerror!void,
        deinit: *const fn (*anyopaque, Allocator) void,
        prefer_async_backward: bool = false,
        estimated_work: ?*const fn (*const anyopaque) usize = null,
    };

    pub fn operands(self: BackwardFunction) []const ?*GradState {
        return self.vtable.operands(self.ptr);
    }

    pub fn backward(
        self: BackwardFunction,
        ctx: *ExecContext,
        gy: *const Tensor,
        needs_grad: []const bool,
        out: []?Tensor,
    ) !void {
        return self.vtable.backward(self.ptr, ctx, gy, needs_grad, out);
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

/// Allocate one `BackwardNode(Record)`, run `Record.init(&node.record, init_args...)`
/// (the tuple must start with the allocator), and wire the header to the
/// record's `pub const vtable`. On init failure the node is freed and any
/// resources `init` did not consume stay with the caller (mirroring the old
/// per-record `create` contracts). The returned state is destroyed through
/// `GradState.deinit`, whose vtable call frees the entire node.
pub fn createNode(comptime Record: type, init_args: anytype) !*GradState {
    const allocator: Allocator = init_args[0];
    const node = try allocator.create(BackwardNode(Record));
    errdefer allocator.destroy(node);
    try @call(.auto, Record.init, .{&node.record} ++ init_args);
    node.state = .{
        .allocator = allocator,
        .grad_fn = .{ .ptr = &node.record, .vtable = &Record.vtable },
    };
    return &node.state;
}

/// Tail of every record vtable deinit: recover the co-allocated node from the
/// record pointer and free it (header included).
pub fn destroyNode(comptime Record: type, allocator: Allocator, record: *Record) void {
    const node: *BackwardNode(Record) = @fieldParentPtr("record", record);
    allocator.destroy(node);
}

pub const GradState = struct {
    allocator: Allocator,
    grad: ?Tensor = null,
    grad_fn: ?BackwardFunction = null,
    state: std.atomic.Value(u8) = std.atomic.Value(u8).init(@intFromEnum(BackwardState.idle)),
    pending_grads: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    grad_mutex: thread.Mutex = .{},

    pub fn leaf(allocator: Allocator) !*GradState {
        const self = try allocator.create(GradState);
        self.* = .{ .allocator = allocator };
        return self;
    }

    pub fn deinit(self: *GradState) void {
        self.zeroGrad();
        if (self.grad_fn) |function| {
            // Frees the whole co-allocated node, self included — self is
            // dangling after this call; return immediately.
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
        if (output_value.isScalar()) return try ctx.scalar(1);
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
        const materialized = try engine.ctx.materialize(value);
        value.deinit();
        value.* = materialized;
    }

    fn prepareMutableAccumulator(engine: *GradEngine, current: *Tensor) !void {
        if (current.canTakeInPlace()) return;
        const materialized = try engine.ctx.materialize(current);
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
                try engine.ctx.addInPlace(current, &owned);
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
        var needs_grad_stack: [stack_operand_capacity]bool = undefined;
        var needs_grad_heap: ?[]bool = null;
        defer if (needs_grad_heap) |buf| engine.allocator.free(buf);
        const needs_grad = if (operands.len <= stack_operand_capacity)
            needs_grad_stack[0..operands.len]
        else blk: {
            const buf = try engine.allocator.alloc(bool, operands.len);
            needs_grad_heap = buf;
            break :blk buf;
        };
        for (operands, needs_grad) |operand, *need| {
            need.* = operand != null;
        }

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

        function.backward(engine.ctx, local_gy, needs_grad, gxs) catch |err| {
            for (operands, gxs, needs_grad) |operand, *gx, need| {
                if (gx.*) |*owned| {
                    owned.deinit();
                    gx.* = null;
                }
                if (need) operand.?.finishGradContribution(engine);
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
        for (operands, gxs, needs_grad) |operand, *gx, need| {
            if (need) {
                if (gx.*) |owned| {
                    gx.* = null;
                    const state = operand.?;
                    if (state.accGradOwnedReady(engine, owned) catch |err| blk: {
                        if (first_error == null) first_error = err;
                        break :blk false;
                    }) {
                        ready[ready_len] = state;
                        ready_len += 1;
                    }
                } else {
                    operand.?.finishGradContribution(engine);
                    missing_backward_gradient = true;
                }
            }
        }

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
        seed.* = try output.prepareOutputSeed(ctx, output_value);
    }

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
}

pub fn backwardGradOne(ctx: *ExecContext, output: *GradState, output_value: *const Tensor) !void {
    return backwardGrad(ctx, &.{output}, &.{output_value});
}

test {
    _ = @import("core_tests.zig");
}

test "backward scheduler releases pending operand on missing gradient" {
    const MissingGradientBackward = struct {
        parent: ?*GradState,

        const Self = @This();

        pub fn init(self: *Self, allocator: Allocator, parent: *GradState) !void {
            _ = allocator;
            self.* = .{ .parent = parent };
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return @as([*]const ?*GradState, @ptrCast(&self.parent))[0..1];
        }

        fn backward(
            ptr: *const anyopaque,
            ctx: *ExecContext,
            gy: *const Tensor,
            needs_grad: []const bool,
            out: []?Tensor,
        ) anyerror!void {
            _ = ptr;
            _ = ctx;
            _ = gy;
            try std.testing.expectEqual(@as(usize, 1), needs_grad.len);
            try std.testing.expect(needs_grad[0]);
            try std.testing.expectEqual(@as(usize, 1), out.len);
            out[0] = null;
        }

        fn deinit(ptr: *anyopaque, allocator: Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };

    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const parent = try GradState.leaf(ctx.allocator);
    defer parent.deinit();

    var output_value = try ctx.scalar(0);
    defer output_value.deinit();

    const output = try createNode(MissingGradientBackward, .{ ctx.allocator, parent });
    defer output.deinit();

    try std.testing.expectError(AgError.MissingBackwardGradient, backwardGradOne(&ctx, output, &output_value));
    try std.testing.expectEqual(@as(u32, 0), parent.pending_grads.load(.acquire));
    try std.testing.expectEqual(BackwardState.idle, parent.loadState());
}

test "backward scheduler releases pending operand on backward error" {
    const FailingBackward = struct {
        parent: ?*GradState,

        const Self = @This();
        const BackwardError = error{FailedBackward};

        pub fn init(self: *Self, allocator: Allocator, parent: *GradState) !void {
            _ = allocator;
            self.* = .{ .parent = parent };
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return @as([*]const ?*GradState, @ptrCast(&self.parent))[0..1];
        }

        fn backward(
            ptr: *const anyopaque,
            ctx: *ExecContext,
            gy: *const Tensor,
            needs_grad: []const bool,
            out: []?Tensor,
        ) anyerror!void {
            _ = ptr;
            _ = ctx;
            _ = gy;
            _ = out;
            try std.testing.expectEqual(@as(usize, 1), needs_grad.len);
            try std.testing.expect(needs_grad[0]);
            return BackwardError.FailedBackward;
        }

        fn deinit(ptr: *anyopaque, allocator: Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };

    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const parent = try GradState.leaf(ctx.allocator);
    defer parent.deinit();

    var output_value = try ctx.scalar(0);
    defer output_value.deinit();

    const output = try createNode(FailingBackward, .{ ctx.allocator, parent });
    defer output.deinit();

    try std.testing.expectError(FailingBackward.BackwardError.FailedBackward, backwardGradOne(&ctx, output, &output_value));
    try std.testing.expectEqual(@as(u32, 0), parent.pending_grads.load(.acquire));
    try std.testing.expectEqual(BackwardState.idle, parent.loadState());
}
