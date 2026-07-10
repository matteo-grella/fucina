//! Generic eager-runtime substrate for `ExecContext`.
//!
//! `Runtime` owns the allocation/thread/scope machinery that every domain op
//! needs but that carries no domain semantics: the buffer pool, the worker
//! team, the exec-scope stack, and the tensor allocation primitives. Domain
//! modules under `src/exec/` receive a `*Runtime` explicitly (never
//! `self: anytype`), so their code is monomorphic and the file-level import
//! graph stays a strict DAG.
//!
//! This is a LEAF module: it imports only base leaves + the buffer-pool leaf,
//! never `exec.zig` or any domain module. `ExecContext` embeds one of these as
//! `rt` and forwards its substrate methods here (see `exec.zig`).

const std = @import("std");
const backend_mod = @import("../backend.zig");
const dtype_mod = @import("../dtype.zig");
const parallel = @import("../parallel.zig");
const tensor = @import("../tensor.zig");
const thread = @import("../thread.zig");

const exec_buffer_pool = @import("buffer_pool.zig");

const Allocator = std.mem.Allocator;
const Backend = backend_mod.Backend;
const DType = tensor.DType;
const Tensor = tensor.Tensor;

/// Reusable transient-buffer pool. Defined in the `buffer_pool.zig` leaf.
pub const BufferPool = exec_buffer_pool.BufferPool;

pub const ScopeNodeDestroy = *const fn (*anyopaque) void;

const ScopeEntry = struct {
    value: Tensor,
    node: ?*anyopaque,
    destroy_node: ScopeNodeDestroy,
};

pub const ExecScope = struct {
    index: usize,
};

pub const Runtime = struct {
    thread_safe_allocator: thread.ThreadSafeAllocator,
    allocator: Allocator,
    backend: Backend,
    buffers: BufferPool,
    work_pool: thread.Pool,
    work_pool_ready: bool = false,
    work_pool_mutex: thread.Mutex = .{},
    dot_backward_worker: thread.OneShotWorker,
    dot_backward_worker_ready: bool = false,
    dot_backward_worker_mutex: thread.Mutex = .{},
    scope_entries: std.ArrayList(ScopeEntry) = .empty,
    scope_depth: usize = 0,

    pub fn init(self: *Runtime, allocator: Allocator) void {
        self.thread_safe_allocator = .{ .child_allocator = allocator };
        self.allocator = self.thread_safe_allocator.allocator();
        self.backend = Backend.init();
        self.buffers = BufferPool.init(self.allocator);
        self.work_pool = undefined;
        self.work_pool_ready = false;
        self.work_pool_mutex = .{};
        self.dot_backward_worker = undefined;
        self.dot_backward_worker_ready = false;
        self.dot_backward_worker_mutex = .{};
        self.scope_entries = .empty;
        self.scope_depth = 0;
    }

    pub fn deinit(self: *Runtime) void {
        if (self.dot_backward_worker_ready) {
            self.dot_backward_worker.deinit();
        }
        if (self.work_pool_ready) {
            self.backend.setWorkPool(null);
            self.work_pool.deinit();
        }
        self.releaseScopeTo(0); // defensive: scopes left open at teardown
        self.scope_entries.deinit(self.allocator);
        self.buffers.deinit();
        self.* = undefined;
    }

    // ------------------------------------------------------------------
    // Exec scopes (see the design note on `ExecContext`).
    // ------------------------------------------------------------------

    pub fn execScopeActive(self: *const Runtime) bool {
        return self.scope_depth > 0;
    }

    pub fn openExecScope(self: *Runtime) ExecScope {
        self.scope_depth += 1;
        return .{ .index = self.scope_entries.items.len };
    }

    pub fn closeExecScope(self: *Runtime, mark: ExecScope) void {
        std.debug.assert(self.scope_depth > 0);
        self.scope_depth -= 1;
        self.releaseScopeTo(mark.index);
    }

    fn releaseScopeTo(self: *Runtime, index: usize) void {
        std.debug.assert(index <= self.scope_entries.items.len);
        while (self.scope_entries.items.len > index) {
            var entry = self.scope_entries.pop().?;
            entry.value.deinit();
            if (entry.node) |node| entry.destroy_node(node);
        }
    }

    pub fn reserveScopeSlot(self: *Runtime) !void {
        try self.scope_entries.ensureUnusedCapacity(self.allocator, 1);
    }

    pub fn adoptScopeValueAssumeCapacity(self: *Runtime, value: Tensor, node: ?*anyopaque, destroy_node: ScopeNodeDestroy) void {
        std.debug.assert(self.scope_depth > 0);
        self.scope_entries.appendAssumeCapacity(.{ .value = value, .node = node, .destroy_node = destroy_node });
    }

    // ------------------------------------------------------------------
    // Worker team + one-shot dot-backward worker.
    // ------------------------------------------------------------------

    pub fn tryWorkPool(self: *Runtime) !*thread.Pool {
        self.work_pool_mutex.lock();
        defer self.work_pool_mutex.unlock();

        if (!self.work_pool_ready) {
            const worker_threads = parallel.cpuThreadCount(parallel.vector_max_threads) - 1;
            try self.work_pool.init(.{
                .allocator = self.allocator,
                .max_workers = worker_threads,
            });
            self.work_pool_ready = true;
            self.backend.setWorkPool(&self.work_pool);
        }
        return &self.work_pool;
    }

    pub fn workPool(self: *Runtime) ?*thread.Pool {
        return self.tryWorkPool() catch null;
    }

    fn tryDotBackwardWorker(self: *Runtime) !*thread.OneShotWorker {
        self.dot_backward_worker_mutex.lock();
        defer self.dot_backward_worker_mutex.unlock();

        if (!self.dot_backward_worker_ready) {
            try self.dot_backward_worker.init();
            self.dot_backward_worker_ready = true;
        }
        return &self.dot_backward_worker;
    }

    pub fn dotBackwardWorker(self: *Runtime) ?*thread.OneShotWorker {
        return self.tryDotBackwardWorker() catch null;
    }

    // ------------------------------------------------------------------
    // Tensor allocation primitives. Kernels never allocate; these are the
    // only source of transient tensors (all backed by the buffer pool: the
    // f32 arm for default-dtype tensors, the byte-slab arm for every other
    // storage dtype).
    // ------------------------------------------------------------------

    pub fn empty(self: *Runtime, shape: []const usize) !Tensor {
        const len = try tensor.elementCount(shape);
        const buffer = try self.buffers.acquire(len);
        errdefer buffer.release();
        return Tensor.fromOwnedBuffer(buffer, shape);
    }

    pub fn emptyRank(self: *Runtime, comptime rank: usize, shape: [rank]usize) !Tensor {
        const len = try tensor.elementCountArray(rank, shape);
        const buffer = try self.buffers.acquire(len);
        errdefer buffer.release();
        return Tensor.fromOwnedBuffer(buffer, shape[0..]);
    }

    pub fn emptyTyped(self: *Runtime, comptime dtype: DType, shape: []const usize) !tensor.TensorOf(dtype) {
        if (comptime dtype == .f32) return self.empty(shape);

        const len = try tensor.storageElementCount(dtype, shape);
        const buffer = try self.buffers.acquireTyped(dtype, len);
        errdefer buffer.release();
        return tensor.TensorOf(dtype).fromOwnedBuffer(buffer, shape);
    }

    pub fn emptyRankTyped(self: *Runtime, comptime dtype: DType, comptime rank: usize, shape: [rank]usize) !tensor.TensorOf(dtype) {
        if (comptime dtype == .f32) return self.emptyRank(rank, shape);

        const len = try tensor.storageElementCountArray(dtype, rank, shape);
        const buffer = try self.buffers.acquireTyped(dtype, len);
        errdefer buffer.release();
        return tensor.TensorOf(dtype).fromOwnedBuffer(buffer, shape[0..]);
    }

    pub fn zeros(self: *Runtime, shape: []const usize) !Tensor {
        var out = try self.empty(shape);
        @memset(out.data(), 0);
        return out;
    }

    pub fn zerosRank(self: *Runtime, comptime rank: usize, shape: [rank]usize) !Tensor {
        var out = try self.emptyRank(rank, shape);
        @memset(out.data(), 0);
        return out;
    }

    pub fn zerosTyped(self: *Runtime, comptime dtype: DType, shape: []const usize) !tensor.TensorOf(dtype) {
        var out = try self.emptyTyped(dtype, shape);
        @memset(out.data(), dtype_mod.zero(dtype));
        return out;
    }

    pub fn zerosRankTyped(self: *Runtime, comptime dtype: DType, comptime rank: usize, shape: [rank]usize) !tensor.TensorOf(dtype) {
        var out = try self.emptyRankTyped(dtype, rank, shape);
        @memset(out.data(), dtype_mod.zero(dtype));
        return out;
    }

    pub fn ones(self: *Runtime, shape: []const usize) !Tensor {
        var out = try self.empty(shape);
        @memset(out.data(), 1);
        return out;
    }

    pub fn onesRank(self: *Runtime, comptime rank: usize, shape: [rank]usize) !Tensor {
        var out = try self.emptyRank(rank, shape);
        @memset(out.data(), 1);
        return out;
    }

    pub fn onesTyped(self: *Runtime, comptime dtype: DType, shape: []const usize) !tensor.TensorOf(dtype) {
        var out = try self.emptyTyped(dtype, shape);
        @memset(out.data(), dtype_mod.one(dtype));
        return out;
    }

    pub fn onesRankTyped(self: *Runtime, comptime dtype: DType, comptime rank: usize, shape: [rank]usize) !tensor.TensorOf(dtype) {
        var out = try self.emptyRankTyped(dtype, rank, shape);
        @memset(out.data(), dtype_mod.one(dtype));
        return out;
    }

    pub fn full(self: *Runtime, shape: []const usize, value: f32) !Tensor {
        var out = try self.empty(shape);
        @memset(out.data(), value);
        return out;
    }

    pub fn fullTyped(self: *Runtime, comptime dtype: DType, shape: []const usize, value: dtype_mod.Scalar(dtype)) !tensor.TensorOf(dtype) {
        var out = try self.emptyTyped(dtype, shape);
        @memset(out.data(), value);
        return out;
    }

    pub fn scalar(self: *Runtime, value: f32) !Tensor {
        var out = try self.empty(&.{1});
        out.data()[0] = value;
        return out;
    }

    pub fn scalarTyped(self: *Runtime, comptime dtype: DType, value: dtype_mod.Scalar(dtype)) !tensor.TensorOf(dtype) {
        var out = try self.emptyTyped(dtype, &.{1});
        out.data()[0] = value;
        return out;
    }

    pub fn fromSlice(self: *Runtime, shape: []const usize, values: []const f32) !Tensor {
        const len = try tensor.elementCount(shape);
        if (len != values.len) return tensor.TensorError.InvalidDataLength;
        var out = try self.empty(shape);
        @memcpy(out.data(), values);
        return out;
    }

    pub fn fromSliceRank(self: *Runtime, comptime rank: usize, shape: [rank]usize, values: []const f32) !Tensor {
        const len = try tensor.elementCountArray(rank, shape);
        if (len != values.len) return tensor.TensorError.InvalidDataLength;
        var out = try self.emptyRank(rank, shape);
        @memcpy(out.data(), values);
        return out;
    }

    pub fn fromBorrowedSliceRank(self: *Runtime, comptime rank: usize, shape: [rank]usize, values: []f32) !Tensor {
        return Tensor.fromBorrowedSlice(self.allocator, shape[0..], values);
    }

    pub fn fromSliceTyped(self: *Runtime, comptime dtype: DType, shape: []const usize, values: []const dtype_mod.Scalar(dtype)) !tensor.TensorOf(dtype) {
        const len = try tensor.elementCount(shape);
        if (len != values.len) return tensor.TensorError.InvalidDataLength;
        var out = try self.emptyTyped(dtype, shape);
        @memcpy(out.data(), values);
        return out;
    }

    pub fn fromSliceRankTyped(
        self: *Runtime,
        comptime dtype: DType,
        comptime rank: usize,
        shape: [rank]usize,
        values: []const dtype_mod.Scalar(dtype),
    ) !tensor.TensorOf(dtype) {
        const len = try tensor.elementCountArray(rank, shape);
        if (len != values.len) return tensor.TensorError.InvalidDataLength;
        var out = try self.emptyRankTyped(dtype, rank, shape);
        @memcpy(out.data(), values);
        return out;
    }

    pub fn fromBorrowedSliceRankTyped(
        self: *Runtime,
        comptime dtype: DType,
        comptime rank: usize,
        shape: [rank]usize,
        values: []dtype_mod.Scalar(dtype),
    ) !tensor.TensorOf(dtype) {
        return tensor.TensorOf(dtype).fromBorrowedSlice(self.allocator, shape[0..], values);
    }

    pub fn fromStorageSliceRankTyped(
        self: *Runtime,
        comptime dtype: DType,
        comptime rank: usize,
        shape: [rank]usize,
        values: []const dtype_mod.Storage(dtype),
    ) !tensor.TensorOf(dtype) {
        const len = try tensor.storageElementCountArray(dtype, rank, shape);
        if (len != values.len) return tensor.TensorError.InvalidDataLength;
        var out = try self.emptyRankTyped(dtype, rank, shape);
        @memcpy(out.data(), values);
        return out;
    }

    pub fn fromBorrowedStorageSliceRankTyped(
        self: *Runtime,
        comptime dtype: DType,
        comptime rank: usize,
        shape: [rank]usize,
        values: []dtype_mod.Storage(dtype),
    ) !tensor.TensorOf(dtype) {
        return tensor.TensorOf(dtype).fromBorrowedStorageSlice(self.allocator, shape[0..], values);
    }

    pub fn materialize(self: *Runtime, x: *const Tensor) !Tensor {
        return self.materializeTyped(.f32, x);
    }

    pub fn materializeTyped(self: *Runtime, comptime dtype: DType, x: *const tensor.TensorOf(dtype)) !tensor.TensorOf(dtype) {
        var out = try self.emptyTyped(dtype, x.shape.slice());
        errdefer out.deinit();
        if (comptime dtype_mod.isScalar(dtype)) {
            if (!x.isContiguous()) {
                const dst = out.data();
                if (dst.len >= parallel.materialize_parallel_len_threshold) {
                    if (self.workPool()) |pool| {
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

    pub fn clone(self: *Runtime, x: *const Tensor) !Tensor {
        return self.materialize(x);
    }

    pub fn cloneTyped(self: *Runtime, comptime dtype: DType, x: *const tensor.TensorOf(dtype)) !tensor.TensorOf(dtype) {
        return self.materializeTyped(dtype, x);
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

    pub fn prepareContiguous(self: *Runtime, x: *const Tensor) !PreparedTensor {
        return self.prepareContiguousTyped(.f32, x);
    }

    pub fn prepareContiguousTyped(
        self: *Runtime,
        comptime dtype: DType,
        x: *const tensor.TensorOf(dtype),
    ) !PreparedTensorOf(dtype) {
        if (x.isContiguous()) return .{ .borrowed = x };
        return .{ .owned = try self.materializeTyped(dtype, x) };
    }

    // ------------------------------------------------------------------
    // Native-backend pool gates: spin up the worker team only once a job
    // crosses the work threshold, and only when the native (non-BLAS)
    // vector kernels will actually thread it.
    // ------------------------------------------------------------------

    pub fn enableNativeVectorPoolForWork(self: *Runtime, work: usize, threshold: usize) void {
        if (comptime Backend.kind != .native) return;
        if (work >= threshold) _ = self.tryWorkPool() catch null;
    }

    pub fn enableNativeMatmulPoolForWork(self: *Runtime, m: usize, n: usize, k: usize) void {
        if (comptime Backend.kind != .native or backend_mod.native_uses_blas) return;
        const work = parallel.saturatedMul3(m, n, k);
        self.enableNativeVectorPoolForWork(work, parallel.vector_matmul_work_threshold);
    }

    pub fn enableNativeTypedMatmulPoolForWork(self: *Runtime, m: usize, n: usize, k: usize) void {
        if (comptime Backend.kind != .native) return;
        const work = parallel.saturatedMul3(m, n, k);
        self.enableNativeVectorPoolForWork(work, parallel.vector_matmul_work_threshold);
    }
};
