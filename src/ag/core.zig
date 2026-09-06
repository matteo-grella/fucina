const std = @import("std");
const backend_mod = @import("../backend.zig");
const exec_mod = @import("../exec.zig");
const parallel = @import("../parallel.zig");
const tensor = @import("../tensor.zig");
const thread = @import("../thread.zig");

const Allocator = std.mem.Allocator;
const ExecContext = exec_mod.ExecContext;
const Tensor = tensor.Tensor;

/// The autograd band's error domain: the engine's state names and the
/// facade's graph-control names. Band code raises through this set, never
/// as a bare `error.X`, so `ag.Error` (and `fucina.Error`) is derived from
/// it rather than maintained beside it.
pub const AgError = error{
    /// `backwardGrad*` received fewer output gradients than outputs.
    MissingOutputGradient,
    /// A VJP left a required input-gradient slot empty (a custom VJP or a
    /// checkpoint recompute included).
    MissingBackwardGradient,
    /// A second backward over a graph, or over a single-use VJP record,
    /// that already ran.
    BackwardAlreadyRun,
    /// A no-grad-only entry touched a grad-requiring tensor: an in-place
    /// or storage-consuming helper, an inference-only packed kernel, a
    /// prepared-conv entry, a cast off the float seam, a typed-branch
    /// view. The one name for "no VJP here"; the entry's doc says why.
    UnsupportedGradient,
    /// `data()` on a tensor that requires gradients.
    MutableDataRequiresNoGrad,
    /// A backward reached a record whose saved value (an operand view, a
    /// constant included, or the saved output) was mutated through a
    /// mutable host access after the forward: the VJP would differentiate
    /// values other than the ones the forward computed with.
    SavedValueMutated,
    /// `backward` on a tensor with no recorded graph (a recomputed
    /// checkpoint output without one included).
    NoGradientGraph,
    /// An op that must own its result was called on a scope-owned borrow.
    ActiveExecScopeUnsupported,
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
    const node = try allocNode(allocator, @TypeOf(record));
    return initNode(node, allocator, record);
}

/// The two halves of `createNode`, for a tail that must adopt the node's
/// address into an exec scope (fallible) before it moves the record in:
/// `allocNode` is the only allocation, `initNode` cannot fail and takes
/// the record's resources. Between the two the node holds no record and
/// no state; only its address is meaningful.
pub fn allocNode(allocator: Allocator, comptime Record: type) !*BackwardNode(Record) {
    return allocator.create(BackwardNode(Record));
}

pub fn initNode(node: anytype, allocator: Allocator, record: anytype) *GradState {
    const Record = @TypeOf(record);
    node.record = record;
    node.state = .{
        .allocator = allocator,
        .grad_fn = .{ .ptr = &node.record, .vtable = &Record.vtable },
        .saved_generation = savedGenerationSum(&node.record),
    };
    retainParents(Record.vtable.operands(&node.record));
    return &node.state;
}

/// The sum of the storage generations of every raw tensor a record holds:
/// its tensor fields, and the tensors inside its nested structs,
/// optionals, arrays and slices (pointers are not followed — a pointee is
/// not the record's saved value). Generations only grow, so any mutation
/// of any saved storage strictly raises the sum; captured by `initNode`
/// and compared before the VJP runs (`recordVTable`).
fn savedGenerationSum(record: anytype) u64 {
    var sum: u64 = 0;
    addSavedGenerations(&sum, record);
    return sum;
}

fn addSavedGenerations(sum: *u64, ptr: anytype) void {
    const T = @TypeOf(ptr.*);
    if (comptime !holdsRawTensor(T)) return;
    if (comptime isRawTensor(T)) {
        sum.* += ptr.buffer.generation.load(.monotonic);
        return;
    }
    switch (@typeInfo(T)) {
        .@"struct" => |s| inline for (s.fields) |f| addSavedGenerations(sum, &@field(ptr.*, f.name)),
        .optional => if (ptr.*) |*inner| addSavedGenerations(sum, inner),
        .array => for (ptr) |*item| addSavedGenerations(sum, item),
        .pointer => for (ptr.*) |*item| addSavedGenerations(sum, item),
        else => {},
    }
}

/// A raw tensor value (`tensor.TensorOf(dtype)`): the saved-view type of
/// every record.
fn isRawTensor(comptime T: type) bool {
    return @typeInfo(T) == .@"struct" and @hasField(T, "buffer") and @hasField(T, "shape") and
        @hasField(T, "strides") and @hasField(T, "offset") and @hasDecl(T, "dtype");
}

/// Whether a value of `T` can hold a raw tensor by value (through structs,
/// optionals, arrays and slices); the walk skips everything else.
fn holdsRawTensor(comptime T: type) bool {
    if (isRawTensor(T)) return true;
    return switch (@typeInfo(T)) {
        .@"struct" => |s| for (s.fields) |f| {
            if (holdsRawTensor(f.type)) break true;
        } else false,
        .optional => |o| holdsRawTensor(o.child),
        .array => |a| holdsRawTensor(a.child),
        .pointer => |p| p.size == .slice and holdsRawTensor(p.child),
        else => false,
    };
}

/// The exec-scope entry for the reference a handle holds on `state`
/// (`ExecContext.adopt`): the scope takes it over and drops it at close.
pub fn scopeEntry(state: *GradState) ExecContext.ScopeEntry {
    const Shim = struct {
        fn release(ptr: *anyopaque) void {
            const s: *GradState = @ptrCast(@alignCast(ptr));
            s.release();
        }
    };
    return .{ .ptr = state, .release = Shim.release };
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
            // The saved values must be the ones the forward computed with:
            // a mutation of any saved storage since `initNode` (through
            // any handle, a constant or a detached alias included) is
            // refused here rather than differentiated.
            const node: *BackwardNode(Record) = @fieldParentPtr("record", self);
            if (savedGenerationSum(self) != node.state.saved_generation) return AgError.SavedValueMutated;
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

/// Whether the backward pass spends threads on branch parallelism: the
/// engine's node-level spawning (`GradEngine.canRunAsync`) and the
/// contraction records' two-operand split (`backward/common.zig`
/// `runContractionBranches`). Only a native build with BLAS: there the
/// contraction branches run inside BLAS and overlap; without it both would
/// dispatch on the one barrier team, whose second concurrent dispatch runs
/// serially, so the extra thread buys nothing.
pub const parallel_dot_backward_branches = backend_mod.active_kind == .native and backend_mod.native_uses_blas;

pub const GradState = struct {
    allocator: Allocator,
    grad: ?Tensor = null,
    grad_fn: ?BackwardFunction = null,
    state: std.atomic.Value(u8) = std.atomic.Value(u8).init(@intFromEnum(BackwardState.idle)),
    pending_grads: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    grad_mutex: thread.Mutex = .{},
    /// Set once a backward pass with this state as an OUTPUT completes.
    /// The pass leaves the output's gradient in place as a result, so a
    /// later pass reaching this state — as an output again or as an
    /// interior node of a newer graph — would compound it and propagate
    /// the sum; `backwardGradImpl` rejects a marked state anywhere in the
    /// reachable graph with `AgError.BackwardAlreadyRun` before any
    /// gradient moves. Leaves (`grad_fn == null`) are never marked — they
    /// have no graph to consume. Written by the driving thread between
    /// passes and, through `consumeRecord`, by a VJP inside a pass (from a
    /// pool task at most); read only by the next pass's preflight, after
    /// `waitAll` has joined every task, so it needs no synchronization of
    /// its own.
    backward_done: bool = false,
    /// Set on the outputs of the pass in flight. Their gradients are
    /// results and stay readable after the pass; every other interior
    /// gradient is released as soon as its own backward has consumed it
    /// (leaves have no backward and keep theirs for the optimizer). A
    /// failed pass clears it again together with the outputs' gradients.
    pass_output: bool = false,
    /// Reference count. Every owner holds exactly one reference: a facade
    /// handle, a consumer record (one per operand slot, taken by
    /// `createNode`), an exec-scope entry. Starts at one for the creator.
    /// Atomic because a record may be destroyed on the thread that closes
    /// a scope while pool tasks of a finished backward are still unwinding
    /// their own handles.
    refs: std.atomic.Value(u32) = std.atomic.Value(u32).init(1),
    /// The `Worklist` link: a state sits on at most one engine worklist at
    /// a time (a pass's preparation or drain, one executing thread's ready
    /// queue, or the release cascade), never on two.
    next: ?*GradState = null,
    /// The sum of the storage generations of the record's saved tensors
    /// at `initNode` (`savedGenerationSum`); zero on a leaf.
    saved_generation: u64 = 0,

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
    /// using it. The cascade is iterative: a parent whose last reference
    /// drops inside a record's deinit joins this thread's release worklist
    /// instead of freeing recursively, so a chain of any depth tears down
    /// on a bounded stack.
    pub fn release(self: *GradState) void {
        if (self.refs.fetchSub(1, .acq_rel) != 1) return;
        if (release_cascade) |cascade| {
            cascade.push(self);
            return;
        }
        var cascade: Worklist = .{};
        release_cascade = &cascade;
        defer release_cascade = null;
        self.destroy();
        while (cascade.pop()) |state| state.destroy();
    }

    /// Free a state whose last reference dropped.
    fn destroy(self: *GradState) void {
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

    /// Replace the stored gradient with `grad` (taking ownership); one
    /// swap under the mutex, the displaced gradient released outside it.
    pub fn setGrad(self: *GradState, grad: Tensor) void {
        self.grad_mutex.lock();
        var displaced = self.grad;
        self.grad = grad;
        self.grad_mutex.unlock();
        if (displaced) |*g| g.deinit();
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

    /// Install the pass's scheduling state over the graph reachable from
    /// `outputs`: one pending count per consumer (and per requested
    /// output), `.pending` on first reach, the operands through an
    /// explicit worklist — the depth of the graph is not a call-stack
    /// resource. Also the preflight: a reachable state that a completed
    /// pass already consumed (`backward_done`) sets `consumed`, and the
    /// caller unwinds the whole preparation with `drainBackwardPass`
    /// instead of seeding.
    fn prepareBackwardPass(outputs: []const *GradState, consumed: *bool) void {
        var reached: Worklist = .{};
        for (outputs) |output| output.enterPass(&reached, consumed);
        while (reached.pop()) |state| {
            const function = state.grad_fn orelse continue;
            for (function.operands()) |operand| {
                if (operand) |parent| parent.enterPass(&reached, consumed);
            }
        }
    }

    fn enterPass(self: *GradState, reached: *Worklist, consumed: *bool) void {
        _ = self.pending_grads.fetchAdd(1, .monotonic);
        if (!self.compareState(.idle, .pending)) return;
        if (self.backward_done) consumed.* = true;
        reached.push(self);
    }

    /// Undo `prepareBackwardPass` without moving any gradient: drop each
    /// reached state's pending count and, once it reaches zero, return
    /// the state to `.idle` and continue into its operands. Mirrors the
    /// preparation exactly, so a prepared graph unwinds to its pre-pass
    /// scheduling state.
    fn drainBackwardPass(outputs: []const *GradState) void {
        var drained: Worklist = .{};
        for (outputs) |output| output.leavePass(&drained);
        while (drained.pop()) |state| {
            const function = state.grad_fn orelse continue;
            for (function.operands()) |operand| {
                if (operand) |parent| parent.leavePass(&drained);
            }
        }
    }

    fn leavePass(self: *GradState, drained: *Worklist) void {
        if (!self.finishGradContributionReady()) return;
        if (!self.compareState(.pending, .idle)) return;
        drained.push(self);
    }

    /// The pending counts of this node's operands, delivered without a
    /// gradient: the tail of a node that runs with nothing to propagate,
    /// and of one that failed before it could. An operand whose count
    /// reaches zero is scheduled as usual — it propagates whatever other
    /// consumers delivered, or drains its own operands the same way — so
    /// no counter is left installed below a node that did not run.
    fn releaseOperands(self: *GradState, engine: *GradEngine, ready: *Worklist) void {
        const function = self.grad_fn orelse return;
        for (function.operands()) |operand| {
            if (operand) |state| state.finishGradContribution(engine, ready);
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
    fn assignOutputGradient(self: *GradState, engine: *GradEngine, seed: ?Tensor, ready: *Worklist) !void {
        if (seed) |owned| {
            return self.accGradOwned(engine, owned, ready);
        }
        self.finishGradContribution(engine, ready);
    }

    fn accGradOwned(self: *GradState, engine: *GradEngine, gx: Tensor, ready: *Worklist) !void {
        if (try self.accGradOwnedReady(engine, gx, ready)) {
            engine.scheduleReady(self, ready);
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

    /// Accumulate one owned contribution, then count it against the pending
    /// counter iff a pass has counters installed (state not `.idle`); true
    /// means this was the last contribution and the caller must schedule the
    /// node. On error the contribution is released AND still counted (the
    /// entry-time check), so a failing pass never strands a nonzero counter.
    fn accGradOwnedReady(self: *GradState, engine: *GradEngine, gx: Tensor, ready: *Worklist) !bool {
        const counted_at_entry = self.loadState() != .idle;
        self.accumulateOwned(engine, gx) catch |err| {
            if (counted_at_entry and self.finishGradContributionReady()) engine.scheduleReady(self, ready);
            return err;
        };
        if (self.loadState() != .idle) return self.finishGradContributionReady();
        return false;
    }

    /// The accumulation body: add `gx` into the stored gradient (or install
    /// it as the initial accumulator). Consumes `gx` on success and on error
    /// alike; the release stays outside the mutex on the add path.
    fn accumulateOwned(self: *GradState, engine: *GradEngine, gx: Tensor) !void {
        var owned = gx;
        errdefer owned.deinit();
        const will_accumulate_more = self.pending_grads.load(.acquire) > 1;
        var moved = false;
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
        if (!moved) owned.deinit();
    }

    fn finishGradContribution(self: *GradState, engine: *GradEngine, ready: *Worklist) void {
        if (self.finishGradContributionReady()) {
            engine.scheduleReady(self, ready);
        }
    }

    fn finishGradContributionReady(self: *GradState) bool {
        const old = self.pending_grads.fetchSub(1, .acq_rel);
        std.debug.assert(old > 0);
        return old == 1;
    }

    /// Run this node's VJP over its accumulated gradient and deliver the
    /// operand gradients; operands whose last contribution arrived go onto
    /// `ready`, the executing thread's worklist (never run inline here).
    fn executeBackward(self: *GradState, engine: *GradEngine, ready: *Worklist) !void {
        defer self.storeState(.idle);

        const function = self.grad_fn orelse return;
        const operands = function.operands();

        self.grad_mutex.lock();
        const gy = if (self.grad) |*g| g else null;
        self.grad_mutex.unlock();
        // Scheduled with no gradient (every consumer delivered nothing —
        // a missing VJP output, or a failed contribution upstream): there
        // is nothing to propagate, but the operands were counted at
        // preparation and are released here, or their counters would
        // stay installed and the next pass over this graph would stop at
        // them and report success with missing gradients.
        const local_gy = gy orelse {
            self.releaseOperands(engine, ready);
            return;
        };
        // An interior gradient has no consumer once this node's backward
        // has been attempted: release it on every exit (the VJP ran, or
        // failed, or its scratch did), so the backward's memory is a
        // moving window rather than a second copy of the forward, and no
        // failed pass leaves a stale gradient that a retry would compound.
        // Pass outputs keep theirs (they are results). The successors
        // scheduled below run after this returns, so the release precedes
        // the descent.
        defer if (!self.pass_output) self.zeroGrad();

        var gxs_scratch: SmallSlice(?Tensor, 8) = .{};
        defer gxs_scratch.deinit(engine.allocator);
        const gxs = gxs_scratch.init(engine.allocator, operands.len) catch |err| {
            self.releaseOperands(engine, ready);
            return err;
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
                if (operand) |state| state.finishGradContribution(engine, ready);
            }
            return err;
        };

        var ready_scratch: SmallSlice(*GradState, 8) = .{};
        defer ready_scratch.deinit(engine.allocator);
        const ready_states = ready_scratch.init(engine.allocator, operands.len) catch |err| {
            // The produced gradients go with the scratch (the defer above);
            // the operands are still released.
            self.releaseOperands(engine, ready);
            return err;
        };
        var ready_len: usize = 0;

        var missing_backward_gradient = false;
        var first_error: ?anyerror = null;
        for (operands, gxs) |operand, *gx| {
            const state = operand orelse continue;
            if (gx.*) |owned| {
                gx.* = null;
                if (state.accGradOwnedReady(engine, owned, ready) catch |err| blk: {
                    if (first_error == null) first_error = err;
                    break :blk false;
                }) {
                    ready_states[ready_len] = state;
                    ready_len += 1;
                }
            } else {
                state.finishGradContribution(engine, ready);
                missing_backward_gradient = true;
            }
        }

        engine.scheduleReadyBatch(ready_states[0..ready_len], ready);
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
    /// The one completion mechanism: every spawned backward task is a member
    /// of this group (`trySpawnWg`), and `waitAll`/`deinit` await it. Inline
    /// tasks need no membership — they run to completion either on the
    /// driving thread before `waitAll` or inside a member task, so an empty
    /// group means the whole pass is done. A member task spawning a successor
    /// keeps the group nonempty until the spawn returns, the exact proviso
    /// `std.Io.Group.await` requires for concurrent spawns.
    wait_group: thread.WaitGroup = .{},
    error_mutex: thread.Mutex = .{},
    first_error: ?anyerror = null,

    pub const Mode = enum { parallel, serial };

    /// `.parallel` schedules big independent VJPs onto the context's work
    /// pool; `.serial` runs every node inline on the calling thread (the
    /// checkpoint-recompute contract, `backwardGradSerial`).
    pub fn init(ctx: *ExecContext, mode: Mode) GradEngine {
        return .{
            .allocator = ctx.allocator(),
            .ctx = ctx,
            .pool = switch (mode) {
                .parallel => ctx.tryWorkPool() catch null,
                .serial => null,
            },
        };
    }

    pub fn deinit(self: *GradEngine) void {
        self.waitAll();
    }

    /// Hand a state whose last contribution arrived to an executing
    /// thread: the caller's worklist (its loop runs the state next), or a
    /// pool task when the batch scheduler chose to spawn it.
    fn scheduleReady(self: *GradEngine, state: *GradState, ready: *Worklist) void {
        self.scheduleReadyMode(state, false, ready);
    }

    /// The operands one node readied, in operand order. Every async
    /// candidate but the last is spawned; the rest go onto `ready`,
    /// pushed in reverse so the LIFO worklist pops them in operand order —
    /// the depth-first order the recursive scheduler had, kept so the
    /// accumulation order of shared states (and bitwise results) does not
    /// move.
    fn scheduleReadyBatch(self: *GradEngine, states: []const *GradState, ready: *Worklist) void {
        var inline_candidate_seen = false;
        var i = states.len;
        while (i > 0) {
            i -= 1;
            const state = states[i];
            var spawn = false;
            if (self.isAsyncCandidate(state)) {
                spawn = inline_candidate_seen;
                inline_candidate_seen = true;
            }
            self.scheduleReadyMode(state, spawn, ready);
        }
    }

    fn scheduleReadyMode(self: *GradEngine, state: *GradState, allow_async: bool, ready: *Worklist) void {
        if (!state.compareState(.pending, .ongoing)) {
            return;
        }
        if (allow_async and self.isAsyncCandidate(state)) {
            if (self.pool.?.trySpawnWg(&self.wait_group, runGradBackwardTask, .{ self, state })) return;
        }
        ready.push(state);
    }

    fn isAsyncCandidate(self: *const GradEngine, state: *const GradState) bool {
        if (self.pool == null) return false;
        const function = state.grad_fn orelse return false;
        return self.canRunAsync(function);
    }

    fn canRunAsync(_: *const GradEngine, function: BackwardFunction) bool {
        if (comptime !parallel_dot_backward_branches) return false;
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

    /// Await every spawned task (idempotent; serial mode has none). Inline
    /// work needs no wait — see the `wait_group` field doc.
    fn waitAll(self: *GradEngine) void {
        const pool = self.pool orelse return;
        pool.waitAndWork(&self.wait_group);
    }
};

/// One executing thread's loop, entered by a spawned task with the state
/// it was given: runs it and every state its execution readies onto this
/// thread's worklist, to exhaustion. An explicit stack in place of
/// recursion, so the depth of the graph is not a call-stack resource.
fn runGradBackwardTask(engine: *GradEngine, state: *GradState) void {
    var ready: Worklist = .{};
    ready.push(state);
    runReady(engine, &ready);
}

fn runReady(engine: *GradEngine, ready: *Worklist) void {
    while (ready.pop()) |state| {
        state.executeBackward(engine, ready) catch |err| {
            state.storeState(.idle);
            engine.recordError(err);
        };
    }
}

/// An intrusive LIFO of states threaded through `GradState.next`: the
/// engine's worklists (a pass's preparation and drain, one executing
/// thread's ready queue, the release cascade) allocate nothing, and a
/// state sits on at most one of them at a time.
const Worklist = struct {
    head: ?*GradState = null,

    fn push(self: *Worklist, state: *GradState) void {
        std.debug.assert(state.next == null);
        state.next = self.head;
        self.head = state;
    }

    fn pop(self: *Worklist) ?*GradState {
        const state = self.head orelse return null;
        self.head = state.next;
        state.next = null;
        return state;
    }
};

/// The release cascade of the thread inside a `GradState.destroy`: parents
/// whose last reference drops there join it instead of freeing
/// recursively (see `GradState.release`).
threadlocal var release_cascade: ?*Worklist = null;

/// A VJP whose body consumes its saved state irreversibly (an in-place
/// consumption of a saved tensor) calls this BEFORE the first step that
/// can fail: the graph can no longer replay through this record, so its
/// state is marked consumed and a retry after a failure fails at the
/// preflight (`BackwardAlreadyRun`) before any gradient moves, instead of
/// running the record again over destroyed state. `record` is the typed
/// record inside its `BackwardNode`.
pub fn consumeRecord(record: anytype) void {
    const Record = @TypeOf(record.*);
    const node: *BackwardNode(Record) = @fieldParentPtr("record", record);
    node.state.backward_done = true;
}

/// Stack-or-heap scratch: `init` returns a `len`-item slice backed by the
/// inline buffer when it fits, heap-allocated past `capacity`. The caller
/// initializes the items; `deinit` frees the heap case. The value must stay
/// in place while the slice is in use (the slice may point into `buffer`).
fn SmallSlice(comptime T: type, comptime capacity: usize) type {
    return struct {
        buffer: [capacity]T = undefined,
        heap: ?[]T = null,

        const Self = @This();

        fn init(self: *Self, allocator: Allocator, len: usize) ![]T {
            if (len <= capacity) return self.buffer[0..len];
            const buf = try allocator.alloc(T, len);
            self.heap = buf;
            return buf;
        }

        fn deinit(self: *Self, allocator: Allocator) void {
            if (self.heap) |buf| allocator.free(buf);
            self.* = undefined;
        }
    };
}

pub fn backwardGrad(ctx: *ExecContext, outputs: []const *GradState, output_values: []const *const Tensor) !void {
    return backwardGradImpl(ctx, outputs, output_values, .parallel);
}

/// As `backwardGrad`, but with node-level async spawning disabled (the engine
/// runs with `pool = null`, so every backward node executes inline on the
/// calling thread). Kernel-level `parallelChunks` parallelism inside the
/// individual VJPs is unaffected. Used by the checkpoint recompute
/// (ag/checkpoint.zig), whose threadlocal nested-recompute guard is only
/// sound when the whole recomputed subgraph stays on one thread.
pub fn backwardGradSerial(ctx: *ExecContext, outputs: []const *GradState, output_values: []const *const Tensor) !void {
    return backwardGradImpl(ctx, outputs, output_values, .serial);
}

fn backwardGradImpl(ctx: *ExecContext, outputs: []const *GradState, output_values: []const *const Tensor, mode: GradEngine.Mode) !void {
    if (outputs.len == 0) return;
    if (outputs.len != output_values.len) return AgError.MissingOutputGradient;

    var engine = GradEngine.init(ctx, mode);
    defer engine.deinit();

    // Validate every output and pre-allocate the implicit scalar seeds before
    // any pending counter exists: an error exit after `prepareBackwardPass`
    // would strand nonzero counters, and the next backward over the same
    // states would stop at their `.pending` check and report success with
    // missing gradients.
    var seeds_scratch: SmallSlice(?Tensor, 8) = .{};
    defer seeds_scratch.deinit(ctx.allocator());
    const seeds = try seeds_scratch.init(ctx.allocator(), outputs.len);
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

    // Preparation is a transaction: install the counters over the whole
    // reachable graph, and if any reachable state was consumed by an
    // earlier completed pass (a previous output built into a newer graph:
    // its retained gradient would compound with the new contribution and
    // propagate the sum), unwind them all and fail before any gradient
    // moves.
    var consumed = false;
    GradState.prepareBackwardPass(outputs, &consumed);
    if (consumed) {
        GradState.drainBackwardPass(outputs);
        return AgError.BackwardAlreadyRun;
    }
    for (outputs) |output| output.pass_output = true;

    // Each output's subgraph runs to exhaustion on the driving thread's
    // worklist before the next output is seeded (the order a shared
    // state's contributions arrive in, kept stable).
    var ready: Worklist = .{};
    for (outputs, seeds) |output, *seed| {
        const owned = seed.*;
        seed.* = null;
        output.assignOutputGradient(&engine, owned, &ready) catch |err| engine.recordError(err);
        runReady(&engine, &ready);
    }

    engine.waitAll();
    if (engine.takeError()) |err| {
        // The failed pass leaves no gradient on any non-leaf: every interior
        // node that ran released its own, every node that did not run was
        // drained, and the outputs' partial results are dropped here (a
        // retry seeds them afresh; a non-scalar output is re-seeded by its
        // caller). Leaves keep the contributions delivered before the
        // failure; the graph itself stays unconsumed and re-runnable.
        for (outputs) |output| {
            output.pass_output = false;
            if (output.grad_fn != null) output.zeroGrad();
        }
        return err;
    }

    // The completed pass consumed the graph: its outputs keep their
    // gradients as results, so a later pass reaching them would compound
    // (one backward per graph; see docs/reference/05-automatic-differentiation.md).
    // Leaf outputs have no graph to consume.
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

    const parent = try GradState.leaf(ctx.allocator());
    defer parent.release();

    var output_value = try ctx.scalar(.f32, 0);
    defer output_value.deinit();

    const output = try createNode(ctx.allocator(), MissingGradientBackward{ .parents = .{parent} });
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

    const parent = try GradState.leaf(ctx.allocator());
    defer parent.release();

    var output_value = try ctx.scalar(.f32, 0);
    defer output_value.deinit();

    const output = try createNode(ctx.allocator(), FailingBackward{ .parents = .{parent} });
    defer output.release();

    try std.testing.expectError(FailingBackward.BackwardError.FailedBackward, backwardGradOne(&ctx, output, &output_value));
    try std.testing.expectEqual(@as(u32, 0), parent.pending_grads.load(.acquire));
    try std.testing.expectEqual(BackwardState.idle, parent.loadState());
}
