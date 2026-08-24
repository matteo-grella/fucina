//! Substrate of `ExecContext`: lifecycle, exec scopes, the worker team, and
//! the tensor allocation primitives. Every function here takes the context
//! as its first parameter and is aliased into the `ExecContext` struct body
//! in `exec.zig`, so `ctx.empty(.f32, ...)` and `exec_runtime.empty(ctx, ...)` are
//! the same call. The fields these functions operate on are declared on
//! `ExecContext` itself; domain modules under `src/exec/` reach the same
//! substrate through the same `*ExecContext`.

const std = @import("std");
const backend_mod = @import("../backend.zig");
const dtype_mod = @import("../dtype.zig");
const fpenv = @import("../fpenv.zig");
const parallel = @import("../parallel.zig");
const tuning = @import("../tuning.zig");
const tensor = @import("../tensor.zig");
const thread = @import("../thread.zig");

const exec_buffer_pool = @import("buffer_pool.zig");
const ExecContext = @import("../exec.zig").ExecContext;

const Allocator = std.mem.Allocator;
const DType = tensor.DType;
const Tensor = tensor.Tensor;

/// Reusable transient-buffer pool. Defined in the `buffer_pool.zig` leaf.
pub const BufferPool = exec_buffer_pool.BufferPool;

pub const ScopeNodeDestroy = *const fn (*anyopaque) void;

pub const ScopeEntry = struct {
    /// Null when the adopted value lives inside the type-erased `node`
    /// payload (non-f32 facade values; the payload's destructor frees it).
    value: ?Tensor,
    node: ?*anyopaque,
    destroy_node: ScopeNodeDestroy,
};

pub const ExecScope = struct {
    index: usize,
};

pub fn init(self: *ExecContext, allocator: Allocator) void {
    self.thread_safe_allocator = .{ .child_allocator = allocator };
    self.allocator = self.thread_safe_allocator.allocator();
    self.parallel_pool = .init(null);
    self.buffers = BufferPool.init(self.allocator);
    self.work_pool = undefined;
    self.work_pool_ready = false;
    self.work_pool_mutex = .{};
    self.dot_backward_worker = undefined;
    self.dot_backward_worker_ready = false;
    self.dot_backward_worker_mutex = .{};
    self.scope_entries = .empty;
    self.scope_depth = 0;
    self.pin_rowwise_kernels = false;
    self.tuning = .{};
    self.fp_env_at_init = fpenv.get();
    self.moe_scratch = .{};
}

/// `error.FloatEnvironmentChanged` when the calling thread's rounding or
/// underflow mode differs from what it was when this context was created.
///
/// The environment is per-thread state shared with everything else running
/// on the thread — an external CBLAS, a GPU driver, a host application. A
/// change is silent: kernels keep running and keep producing plausible
/// numbers that no longer match the pinned oracles. Call this after
/// crossing into foreign code, or around a run whose results are compared
/// bitwise. Always succeeds where the target does not expose the
/// environment, since there is nothing to observe.
pub fn checkFloatEnvironment(self: *const ExecContext) !void {
    const recorded = self.fp_env_at_init orelse return;
    const current = fpenv.get() orelse return;
    if (!std.meta.eql(recorded, current)) return error.FloatEnvironmentChanged;
}

/// The IEEE floating-point environment recorded at context creation, or
/// null where the target does not expose it.
pub fn floatEnvironmentAtInit(self: *const ExecContext) ?fpenv.Environment {
    return self.fp_env_at_init;
}

/// Speculation-verify kernel pinning: while ON, batched quant-matmul
/// ops reproduce the m == 1 kernel numerics bitwise, so a verify
/// batch's logits equal sequential decode's (the lossless-speculation
/// requirement at any draft depth). Toggle around a speculative
/// VERIFY forward only — batch throughput is sacrificed while pinned.
/// See `ExecContext.pin_rowwise_kernels` for the mechanism.
pub fn pinRowwiseKernels(self: *ExecContext, on: bool) void {
    self.pin_rowwise_kernels = on;
}

/// Per-context tuning overrides: route policy that can differ between
/// two contexts in one process (fields left null follow the process-wide
/// FUCINA_* gates; see `fucina.tuning`).
pub fn setTuning(self: *ExecContext, overrides: tuning.Overrides) void {
    self.tuning = overrides;
}

pub fn deinit(self: *ExecContext) void {
    self.moe_scratch.deinit(self.allocator);
    if (self.dot_backward_worker_ready) {
        self.dot_backward_worker.deinit();
    }
    if (self.work_pool_ready) {
        setWorkPool(self, null);
        self.work_pool.deinit();
    }
    releaseScopeTo(self, 0); // defensive: scopes left open at teardown
    self.scope_entries.deinit(self.allocator);
    self.buffers.deinit();
    self.* = undefined;
}

// ------------------------------------------------------------------
// Exec scopes (see the design note on `ExecContext`).
// ------------------------------------------------------------------

pub fn execScopeActive(self: *const ExecContext) bool {
    return self.scope_depth > 0;
}

/// Open a scope; close it with `closeExecScope(mark)` (typically `defer`).
pub fn openExecScope(self: *ExecContext) ExecScope {
    self.scope_depth += 1;
    return .{ .index = self.scope_entries.items.len };
}

/// Release every tensor adopted since `mark`, newest first. Only safe
/// once no backward() over tensors adopted in the scope is pending.
pub fn closeExecScope(self: *ExecContext, mark: ExecScope) void {
    std.debug.assert(self.scope_depth > 0);
    self.scope_depth -= 1;
    releaseScopeTo(self, mark.index);
}

fn releaseScopeTo(self: *ExecContext, index: usize) void {
    std.debug.assert(index <= self.scope_entries.items.len);
    while (self.scope_entries.items.len > index) {
        var entry = self.scope_entries.pop().?;
        if (entry.value) |*value| value.deinit();
        if (entry.node) |node| entry.destroy_node(node);
    }
}

/// Two-phase adoption so op construction can stay infallible after its
/// "consumes the value on success" point: reserve BEFORE building the
/// result, adopt (cannot fail) after.
pub fn reserveScopeSlot(self: *ExecContext) !void {
    try self.scope_entries.ensureUnusedCapacity(self.allocator, 1);
}

pub fn adoptScopeValueAssumeCapacity(self: *ExecContext, value: Tensor, node: ?*anyopaque, destroy_node: ScopeNodeDestroy) void {
    std.debug.assert(self.scope_depth > 0);
    self.scope_entries.appendAssumeCapacity(.{ .value = value, .node = node, .destroy_node = destroy_node });
}

/// Adopt a node-only entry: the value (any dtype) lives inside the
/// type-erased payload, which the destructor must free along with any
/// attached per-op state.
pub fn adoptScopeNodeAssumeCapacity(self: *ExecContext, node: *anyopaque, destroy_node: ScopeNodeDestroy) void {
    std.debug.assert(self.scope_depth > 0);
    self.scope_entries.appendAssumeCapacity(.{ .value = null, .node = node, .destroy_node = destroy_node });
}

// ------------------------------------------------------------------
// Worker team + one-shot dot-backward worker.
// ------------------------------------------------------------------

pub fn tryWorkPool(self: *ExecContext) !*thread.Pool {
    self.work_pool_mutex.lock();
    defer self.work_pool_mutex.unlock();

    if (!self.work_pool_ready) {
        const worker_threads = parallel.cpuThreadCount(parallel.vector_max_threads) - 1;
        try self.work_pool.init(.{
            .allocator = self.allocator,
            .max_workers = worker_threads,
        });
        self.work_pool_ready = true;
        setWorkPool(self, &self.work_pool);
    }
    return &self.work_pool;
}

pub fn workPool(self: *ExecContext) ?*thread.Pool {
    return self.tryWorkPool() catch null;
}

/// Publish (or retract) the worker team for kernel dispatch. Release
/// ordering pairs with the acquire load in `pc`, so a racing first observer
/// on another thread also sees `Pool.init`'s writes.
fn setWorkPool(self: *ExecContext, pool: ?*thread.Pool) void {
    self.parallel_pool.store(pool, .release);
}

/// The `ParallelConfig` every pool-taking kernel call receives: a snapshot
/// of the published worker team (`null` runs the kernel serially).
pub fn pc(self: *const ExecContext) backend_mod.ParallelConfig {
    return .{ .pool = self.parallel_pool.load(.acquire) };
}

/// One row/lane-range pool dispatch for the domain modules'
/// task-parallel kernels: copy `base_task` per task with
/// `start_field`/`end_field` rewritten to its `[0, count)` sub-range
/// and run the tasks on the work pool. False — the caller runs its
/// serial path — when no pool exists or the split degenerates to a
/// single task. Work GATES stay at the call sites: whether a shape is
/// worth dispatching is per-op knowledge; how a range splits is not.
/// Sub-ranges own disjoint output rows, so the pooled result is
/// bitwise identical to the serial call for any task count.
pub fn dispatchRange(
    self: *ExecContext,
    comptime Task: type,
    comptime start_field: []const u8,
    comptime end_field: []const u8,
    base_task: Task,
    count: usize,
    comptime run: fn (task: *const Task) void,
) bool {
    return self.dispatchRangeCapped(Task, start_field, end_field, base_task, count, count, run);
}

/// `dispatchRange` with a separate task-count cap (the inner-lane
/// dispatches cap tasks by a minimum per-task lane width, not by the
/// range being split).
pub fn dispatchRangeCapped(
    self: *ExecContext,
    comptime Task: type,
    comptime start_field: []const u8,
    comptime end_field: []const u8,
    base_task: Task,
    count: usize,
    max_tasks: usize,
    comptime run: fn (task: *const Task) void,
) bool {
    const pool = self.workPool() orelse return false;
    const task_count = @min(parallel.cpuThreadCount(parallel.vector_max_threads), max_tasks);
    if (task_count <= 1) return false;
    var tasks: [parallel.vector_max_threads]Task = undefined;
    for (0..task_count) |task_i| {
        tasks[task_i] = base_task;
        @field(tasks[task_i], start_field) = task_i * count / task_count;
        @field(tasks[task_i], end_field) = (task_i + 1) * count / task_count;
    }
    pool.parallelChunks(Task, tasks[0..task_count], run);
    return true;
}

/// Chunked parallel map with a serial fallback: split `[0, n)` across the
/// worker team (at most one task per `min_len` items) and run
/// `runRange(context, start, end)` on each chunk; run the whole range
/// serially when the team is absent or the split is not worth it. The
/// chunk grid depends only on `(n, task count)`; whether that makes the
/// results bitwise thread-count-invariant is the kernel's property to
/// state (the optimizer and ES elementwise kernels do).
pub fn parallelMap(
    self: *ExecContext,
    n: usize,
    min_len: usize,
    context: anytype,
    comptime runRange: fn (@TypeOf(context), usize, usize) void,
) void {
    if (n >= min_len) {
        const Task = struct {
            context: @TypeOf(context),
            start: usize = 0,
            end: usize = 0,
            fn run(task: *const @This()) void {
                runRange(task.context, task.start, task.end);
            }
        };
        const base = Task{ .context = context };
        if (self.dispatchRangeCapped(Task, "start", "end", base, n, 1 + n / min_len, Task.run)) return;
    }
    runRange(context, 0, n);
}

fn tryDotBackwardWorker(self: *ExecContext) !*thread.OneShotWorker {
    self.dot_backward_worker_mutex.lock();
    defer self.dot_backward_worker_mutex.unlock();

    if (!self.dot_backward_worker_ready) {
        try self.dot_backward_worker.init();
        self.dot_backward_worker_ready = true;
    }
    return &self.dot_backward_worker;
}

pub fn dotBackwardWorker(self: *ExecContext) ?*thread.OneShotWorker {
    return tryDotBackwardWorker(self) catch null;
}

// ------------------------------------------------------------------
// Tensor allocation primitives. Kernels never allocate; these are the
// only source of transient tensors (all backed by the buffer pool: the
// f32 arm for default-dtype tensors, the byte-slab arm for every other
// storage dtype).
// ------------------------------------------------------------------

/// Comptime rank carried by a shape argument: `[n]usize` arrays, pointers
/// to arrays (`&.{...}`), and integer tuples report their rank through the
/// type; a `[]const usize` slice reports null and takes the runtime-rank
/// path.
pub fn shapeComptimeRank(comptime Shape: type) ?usize {
    const invalid = "shape must be a [n]usize array, a tuple of sizes, or a []const usize slice";
    return switch (@typeInfo(Shape)) {
        .array => |info| info.len,
        .pointer => |info| switch (info.size) {
            .one => shapeComptimeRank(info.child),
            .slice => null,
            else => @compileError(invalid),
        },
        .@"struct" => |info| if (info.is_tuple) info.fields.len else @compileError(invalid),
        else => @compileError(invalid),
    };
}

pub fn shapeArray(comptime rank: usize, shape: anytype) [rank]usize {
    var out: [rank]usize = undefined;
    if (comptime @typeInfo(@TypeOf(shape)) == .pointer) {
        inline for (shape.*, 0..) |dim, i| out[i] = dim;
    } else {
        inline for (shape, 0..) |dim, i| out[i] = dim;
    }
    return out;
}

fn shapeElementCount(shape: anytype) !usize {
    const maybe_rank = comptime shapeComptimeRank(@TypeOf(shape));
    if (comptime maybe_rank != null) {
        const rank = comptime maybe_rank.?;
        return tensor.elementCountArray(rank, shapeArray(rank, shape));
    }
    return tensor.elementCount(shape);
}

fn shapeStorageElementCount(comptime dtype: DType, shape: anytype) !usize {
    const maybe_rank = comptime shapeComptimeRank(@TypeOf(shape));
    if (comptime maybe_rank != null) {
        const rank = comptime maybe_rank.?;
        return tensor.storageElementCountArray(dtype, rank, shapeArray(rank, shape));
    }
    return tensor.storageElementCount(dtype, shape);
}

/// Uninitialized tensor of `dtype` with `shape`: a `[n]usize` array or a
/// tuple of sizes takes the comptime-rank arm, a `[]const usize` slice the
/// runtime-rank arm.
pub fn empty(self: *ExecContext, comptime dtype: DType, shape: anytype) !tensor.TensorOf(dtype) {
    const maybe_rank = comptime shapeComptimeRank(@TypeOf(shape));
    if (comptime maybe_rank != null) {
        const rank = comptime maybe_rank.?;
        const shape_array = shapeArray(rank, shape);
        if (comptime dtype == .f32) {
            const len = try tensor.elementCountArray(rank, shape_array);
            const buffer = try self.buffers.acquire(len);
            errdefer buffer.release();
            return Tensor.fromOwnedBuffer(buffer, shape_array[0..]);
        }
        const len = try tensor.storageElementCountArray(dtype, rank, shape_array);
        const buffer = try self.buffers.acquireTyped(dtype, len);
        errdefer buffer.release();
        return tensor.TensorOf(dtype).fromOwnedBuffer(buffer, shape_array[0..]);
    }
    if (comptime dtype == .f32) {
        const len = try tensor.elementCount(shape);
        const buffer = try self.buffers.acquire(len);
        errdefer buffer.release();
        return Tensor.fromOwnedBuffer(buffer, shape);
    }
    const len = try tensor.storageElementCount(dtype, shape);
    const buffer = try self.buffers.acquireTyped(dtype, len);
    errdefer buffer.release();
    return tensor.TensorOf(dtype).fromOwnedBuffer(buffer, shape);
}

/// Zero-copy broadcast view of `x` with `shape` (array/tuple or slice).
pub fn broadcastTo(self: *ExecContext, x: *const Tensor, shape: anytype) !Tensor {
    _ = self;
    const maybe_rank = comptime shapeComptimeRank(@TypeOf(shape));
    if (comptime maybe_rank != null) {
        const rank = comptime maybe_rank.?;
        return x.broadcastToRank(rank, shapeArray(rank, shape));
    }
    return x.broadcastTo(shape);
}

pub fn zeros(self: *ExecContext, comptime dtype: DType, shape: anytype) !tensor.TensorOf(dtype) {
    var out = try self.empty(dtype, shape);
    @memset(out.data(), dtype_mod.zero(dtype));
    return out;
}

pub fn ones(self: *ExecContext, comptime dtype: DType, shape: anytype) !tensor.TensorOf(dtype) {
    var out = try self.empty(dtype, shape);
    @memset(out.data(), dtype_mod.one(dtype));
    return out;
}

pub fn full(self: *ExecContext, comptime dtype: DType, shape: anytype, value: dtype_mod.Scalar(dtype)) !tensor.TensorOf(dtype) {
    var out = try self.empty(dtype, shape);
    @memset(out.data(), value);
    return out;
}

pub fn scalar(self: *ExecContext, comptime dtype: DType, value: dtype_mod.Scalar(dtype)) !tensor.TensorOf(dtype) {
    var out = try self.empty(dtype, &.{1});
    out.data()[0] = value;
    return out;
}

pub fn fromSlice(self: *ExecContext, comptime dtype: DType, shape: anytype, values: []const dtype_mod.Scalar(dtype)) !tensor.TensorOf(dtype) {
    const len = try shapeElementCount(shape);
    if (len != values.len) return tensor.TensorError.InvalidDataLength;
    var out = try self.empty(dtype, shape);
    @memcpy(out.data(), values);
    return out;
}

pub fn fromBorrowedSlice(self: *ExecContext, comptime dtype: DType, shape: anytype, values: []dtype_mod.Scalar(dtype)) !tensor.TensorOf(dtype) {
    const maybe_rank = comptime shapeComptimeRank(@TypeOf(shape));
    if (comptime maybe_rank != null) {
        const rank = comptime maybe_rank.?;
        const shape_array = shapeArray(rank, shape);
        return tensor.TensorOf(dtype).fromBorrowedSlice(self.allocator, shape_array[0..], values);
    }
    return tensor.TensorOf(dtype).fromBorrowedSlice(self.allocator, shape, values);
}

pub fn fromStorageSlice(self: *ExecContext, comptime dtype: DType, shape: anytype, values: []const dtype_mod.Storage(dtype)) !tensor.TensorOf(dtype) {
    const len = try shapeStorageElementCount(dtype, shape);
    if (len != values.len) return tensor.TensorError.InvalidDataLength;
    var out = try self.empty(dtype, shape);
    @memcpy(out.data(), values);
    return out;
}

pub fn fromBorrowedStorageSlice(self: *ExecContext, comptime dtype: DType, shape: anytype, values: []dtype_mod.Storage(dtype)) !tensor.TensorOf(dtype) {
    const maybe_rank = comptime shapeComptimeRank(@TypeOf(shape));
    if (comptime maybe_rank != null) {
        const rank = comptime maybe_rank.?;
        const shape_array = shapeArray(rank, shape);
        return tensor.TensorOf(dtype).fromBorrowedStorageSlice(self.allocator, shape_array[0..], values);
    }
    return tensor.TensorOf(dtype).fromBorrowedStorageSlice(self.allocator, shape, values);
}

pub fn materialize(self: *ExecContext, comptime dtype: DType, x: *const tensor.TensorOf(dtype)) !tensor.TensorOf(dtype) {
    var out = try self.empty(dtype, x.shape.slice());
    errdefer out.deinit();
    if (comptime dtype_mod.isScalar(dtype)) {
        if (!x.isContiguous()) {
            const dst = out.data();
            if (dst.len >= parallel.materialize_parallel_len_threshold) {
                if (self.workPool()) |pool| {
                    // One host fence up front: the chunk tasks each
                    // waitReady on the same buffer, and N workers
                    // spinning on one pending accelerator Work is
                    // wasted parallelism (waitReady itself is
                    // claimant-safe either way).
                    @constCast(x.buffer).waitReady();
                    materializeChunked(dtype, x, dst, pool);
                    return out;
                }
            }
            x.copyRangeTo(dst, 0, dst.len);
            return out;
        }
    }
    try x.copyTo(out.data());
    return out;
}

fn MaterializeTask(comptime dtype: DType) type {
    return struct {
        src: *const tensor.TensorOf(dtype),
        dst: []dtype_mod.Storage(dtype),
        start: usize,
    };
}

fn runMaterializeTask(comptime dtype: DType) fn (*const MaterializeTask(dtype)) void {
    return struct {
        fn run(task: *const MaterializeTask(dtype)) void {
            task.src.copyRangeTo(task.dst, task.start, task.dst.len);
        }
    }.run;
}

/// Strided-view materialization split across the hot worker team: each
/// task copies a disjoint range of the row-major destination (the source
/// is read-only, so ranges are safe to copy concurrently).
fn materializeChunked(comptime dtype: DType, x: *const tensor.TensorOf(dtype), dst: []dtype_mod.Storage(dtype), pool: *thread.Pool) void {
    const Task = MaterializeTask(dtype);
    var tasks: [parallel.vector_max_threads]Task = undefined;
    const want = (dst.len + parallel.materialize_parallel_min_chunk - 1) / parallel.materialize_parallel_min_chunk;
    const n = @max(1, @min(tasks.len, want));
    const chunk = (dst.len + n - 1) / n;
    var count: usize = 0;
    var start: usize = 0;
    while (start < dst.len) : (start += chunk) {
        const end = @min(start + chunk, dst.len);
        tasks[count] = .{ .src = x, .dst = dst[start..end], .start = start };
        count += 1;
    }
    pool.parallelChunks(Task, tasks[0..count], runMaterializeTask(dtype));
}

pub fn clone(self: *ExecContext, comptime dtype: DType, x: *const tensor.TensorOf(dtype)) !tensor.TensorOf(dtype) {
    return self.materialize(dtype, x);
}

// ------------------------------------------------------------------
// Contiguity preparation: borrow the input when it is already
// contiguous, otherwise materialize a contiguous copy. `PreparedTensor`
// owns the copy (deinit is a no-op for the borrowed arm) so a hot path
// can `defer prepared.deinit()` unconditionally.
// ------------------------------------------------------------------

pub const PreparedTensor = PreparedTensorOf(.f32);

pub fn PreparedTensorOf(comptime dtype: DType) type {
    const TypedTensor = tensor.TensorOf(dtype);
    return union(enum) {
        borrowed: *const TypedTensor,
        owned: TypedTensor,

        pub fn tensor(self: *@This()) *const TypedTensor {
            return switch (self.*) {
                .borrowed => |x| x,
                .owned => |*x| x,
            };
        }

        pub fn deinit(self: *@This()) void {
            switch (self.*) {
                .borrowed => {},
                .owned => |*x| x.deinit(),
            }
        }
    };
}

pub fn prepareContiguous(
    self: *ExecContext,
    comptime dtype: DType,
    x: *const tensor.TensorOf(dtype),
) !PreparedTensorOf(dtype) {
    if (x.isContiguous()) return .{ .borrowed = x };
    return .{ .owned = try self.materialize(dtype, x) };
}

// ------------------------------------------------------------------
// Native-backend pool gates: spin up the worker team only once a job
// crosses the work threshold, and only when the native (non-BLAS)
// vector kernels will actually thread it.
// ------------------------------------------------------------------

pub fn enableNativeVectorPoolForWork(self: *ExecContext, work: usize, threshold: usize) void {
    if (comptime backend_mod.active_kind != .native) return;
    if (work >= threshold) _ = self.tryWorkPool() catch null;
}

/// `dtype` names the storage the matmul kernel walks. The f32 dense route
/// may be BLAS-owned (BLAS threads itself, so the pool stays down); every
/// other storage runs our own kernels and threads through the pool.
pub fn enableNativeMatmulPoolForWork(self: *ExecContext, comptime dtype: DType, m: usize, n: usize, k: usize) void {
    if (comptime backend_mod.active_kind != .native) return;
    if (comptime dtype == .f32 and backend_mod.native_uses_blas) return;
    const work = parallel.saturatedMul3(m, n, k);
    self.enableNativeVectorPoolForWork(work, parallel.vector_matmul_work_threshold);
}

// ------------------------------------------------------------------
// Lifetime helper for the "carried value" pattern: deinitializes `old` and
// returns the freshly computed `new_value`, advancing an accumulator (e.g. a
// transformer residual stream) in one statement instead of the
// create/deinit/reassign dance:
//
//     x = try ctx.replace(x, x.add(ctx, &delta));
//
// `new_value` is evaluated by the caller before `replace` runs; if it is an
// error, `old` is left untouched and the error propagates, so the caller's
// `errdefer old.deinit()` still frees it exactly once. On success `old` is
// released (one ref) and the new value returned. Generic over any owned
// value with a `deinit` method (tagged tensors, projection structs, ...).
// ------------------------------------------------------------------

/// Swap a carried tensor for the result of a block call, e.g.
/// `x = try ctx.replace(x, attentionBlock(ctx, ..., &x, ...));`.
/// `new_value` is an error union on purpose: on error the old tensor is
/// NOT consumed (the caller's binding and defers stay valid) and the
/// error propagates; on success the old tensor is released and the new
/// one returned for rebinding. Inside an exec scope the release is a
/// safe no-op for scope-owned op results (their `deinit` is a no-op —
/// the scope owns them), so the same forward code is also training-safe.
pub fn replace(self: *ExecContext, old: anytype, new_value: anytype) @TypeOf(new_value) {
    _ = self;
    comptime {
        const ret_info = @typeInfo(@TypeOf(new_value));
        if (ret_info != .error_union or ret_info.error_union.payload != @TypeOf(old)) {
            @compileError("ctx.replace expects new_value of type E!" ++ @typeName(@TypeOf(old)));
        }
    }
    const value = try new_value;
    var owned = old;
    owned.deinit();
    return value;
}
