//! Substrate of `ExecContext`: the `Runtime` struct the context embeds as
//! `rt`, plus lifecycle, exec scopes, the worker team, and the tensor
//! allocation primitives. Every function here takes the context as its
//! first parameter and is aliased into the `ExecContext` struct body in
//! `exec.zig`, so `ctx.empty(.f32, ...)` and `exec_runtime.empty(ctx, ...)` are
//! the same call; domain modules under `src/exec/` reach the same
//! substrate through the same `*ExecContext` (`ctx.rt.<field>`). The
//! context's trailing fields are model/session execution state, which
//! this file does not manage beyond init/deinit.

const std = @import("std");
const backend_mod = @import("../backend.zig");
const dtype_mod = @import("../dtype.zig");
const fpenv = @import("../fpenv.zig");
const parallel = @import("../parallel.zig");
const shape_mod = @import("../shape.zig");
const storage = @import("../storage.zig");
const tuning = @import("../tuning.zig");
const tensor = @import("../tensor.zig");
const thread = @import("../thread.zig");

const exec_buffer_pool = @import("buffer_pool.zig");
const ExecContext = @import("../exec.zig").ExecContext;

const Allocator = std.mem.Allocator;
const DType = tensor.DType;
const Shape = shape_mod.Shape;
const Tensor = tensor.Tensor;

/// Reusable transient-buffer pool. Defined in the `buffer_pool.zig` leaf.
pub const BufferPool = exec_buffer_pool.BufferPool;

/// The runtime substrate `ExecContext` embeds as `rt`: allocation, the
/// worker team, transient buffers, exec scopes, tuning, and the float
/// environment — model-independent, what every op needs to run at all.
/// PINNED after `ExecContext.init`: `allocator` embeds a pointer to this
/// runtime's own `thread_safe_allocator` field, so an initialized runtime
/// (and the context embedding it) must never be copied or moved — keep
/// the context in a stable stack frame or heap-allocate it, and hand out
/// `*ExecContext` (every op already takes the pointer). Field defaults
/// are the initial state; `ExecContext.init` sets only what a fresh
/// context must compute.
pub const Runtime = struct {
    thread_safe_allocator: thread.ThreadSafeAllocator,
    /// The one allocation seam (a fat pointer into
    /// `thread_safe_allocator`); public code reaches it through
    /// `ctx.allocator()`.
    allocator: Allocator,
    /// The worker team as published to kernel dispatch (`pc` snapshots it).
    /// Atomic: kernels may dispatch on other threads (a backward branch
    /// spawned on the executor) while a lazy `tryWorkPool` retry publishes
    /// the pool;
    /// release/acquire so a racing first observer also sees `Pool.init`'s
    /// writes.
    parallel_pool: std.atomic.Value(?*thread.Pool) = .init(null),
    buffers: BufferPool,
    /// Per-context tuning overrides (`setTuning`); every field
    /// null = follow the process-wide gates (see src/tuning.zig).
    tuning: tuning.Overrides = .{},
    work_pool: thread.Pool,
    work_pool_ready: bool = false,
    /// Latched by `tryWorkPool` when `Pool.init` fails: the context then
    /// runs every kernel serially for its whole life instead of re-paying
    /// the failed init on each dispatch (one warning is logged).
    work_pool_failed: bool = false,
    work_pool_mutex: thread.Mutex = .{},
    /// The context's own exec-scope stack. Scope traffic on a thread that
    /// has installed a recompute frame (`installScopeStack`, the
    /// checkpoint backward) goes to that frame instead.
    scopes: ScopeStack = .{},
    /// The IEEE floating-point environment observed when this context was
    /// created, or null where the target does not expose it. Every numeric
    /// contract the context goes on to honor (backend parity tolerances,
    /// thread-count invariance, checkpoint reproducibility) is stated under
    /// this environment; `checkFloatEnvironment` is how a caller confirms it
    /// still holds after code outside our control has run on the thread.
    fp_env_at_init: ?fpenv.Environment = null,
};

pub const ScopeRelease = *const fn (*anyopaque) void;

/// One reference held by an exec scope: a resource (a value buffer of any
/// dtype, or a type-erased graph node) and the call that drops the
/// reference. The scope knows nothing else about it; every adopted result
/// is one or two of these, whatever its dtype.
pub const ScopeEntry = struct {
    ptr: *anyopaque,
    release: ScopeRelease,
};

/// The entry for one reference on `buffer` that the caller hands over (the
/// adopting handle's own reference; nothing is retained here). Dtype-
/// generic: the release shim is instantiated per buffer type.
pub fn bufferEntry(comptime dtype: DType, buffer: *storage.BufferOf(dtype)) ScopeEntry {
    const Shim = struct {
        fn release(ptr: *anyopaque) void {
            const b: *storage.BufferOf(dtype) = @ptrCast(@alignCast(ptr));
            b.release();
        }
    };
    return .{ .ptr = buffer, .release = Shim.release };
}

pub const ExecScope = struct {
    index: usize,
};

/// The exec-scope stack: the adopted entries plus the open-scope depth.
/// Every context owns one (`ctx.rt.scopes`); a checkpoint recompute
/// installs a second one for its own frame (`installScopeStack`) so the
/// re-run facade ops adopt into the recompute instead of into a stack
/// another thread may be driving.
pub const ScopeStack = struct {
    entries: std.ArrayList(ScopeEntry) = .empty,
    depth: usize = 0,

    /// Releases anything still adopted (scopes left open), then the storage.
    pub fn deinit(self: *ScopeStack, allocator: Allocator) void {
        self.releaseTo(0);
        self.entries.deinit(allocator);
        self.* = undefined;
    }

    fn releaseTo(self: *ScopeStack, index: usize) void {
        std.debug.assert(index <= self.entries.items.len);
        while (self.entries.items.len > index) {
            const entry = self.entries.pop().?;
            entry.release(entry.ptr);
        }
    }
};

/// The calling thread's scope-stack override; null routes scope traffic to
/// the context's own stack. Thread-local by design: a recompute frame is
/// confined to the thread running it (its inner backward is node-serial),
/// so the override never leaks into another thread's view of the same
/// context, and two frames on two threads never share a stack.
threadlocal var installed_scope_stack: ?*ScopeStack = null;

/// Route this thread's scope traffic (open/close/active/adopt) to `stack`
/// until `restoreScopeStack(previous)`. Returns the stack that was
/// installed before, so frames nest (a checkpoint inside a recomputed
/// block installs its own frame and hands the outer one back on exit).
pub fn installScopeStack(stack: *ScopeStack) ?*ScopeStack {
    const previous = installed_scope_stack;
    installed_scope_stack = stack;
    return previous;
}

pub fn restoreScopeStack(previous: ?*ScopeStack) void {
    installed_scope_stack = previous;
}

/// The stack scope traffic goes to on this thread: the installed frame if
/// one is active, else the context's own.
fn activeScopes(self: *ExecContext) *ScopeStack {
    return installed_scope_stack orelse &self.rt.scopes;
}

fn activeScopesConst(self: *const ExecContext) *const ScopeStack {
    return installed_scope_stack orelse &self.rt.scopes;
}

pub fn init(self: *ExecContext, allocator: Allocator) void {
    // Every defaulted field takes its declared default from the struct
    // literal, so the initial state has ONE source (the field decls on
    // `Runtime` and `ExecContext`); only the computed members are set
    // here. `rt.allocator` and `rt.buffers` are assigned after the copy
    // because the allocator interface must capture THIS context's
    // `rt.thread_safe_allocator` field, not the literal's temporary.
    self.* = .{
        .rt = .{
            .thread_safe_allocator = .{ .child_allocator = allocator },
            .allocator = undefined,
            .buffers = undefined,
            .work_pool = undefined,
            .fp_env_at_init = fpenv.get(),
        },
    };
    self.rt.allocator = self.rt.thread_safe_allocator.allocator();
    self.rt.buffers = BufferPool.init(self.rt.allocator);
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
    const recorded = self.rt.fp_env_at_init orelse return;
    const current = fpenv.get() orelse return;
    if (!std.meta.eql(recorded, current)) return ExecError.FloatEnvironmentChanged;
}

/// The IEEE floating-point environment recorded at context creation, or
/// null where the target does not expose it.
pub fn floatEnvironmentAtInit(self: *const ExecContext) ?fpenv.Environment {
    return self.rt.fp_env_at_init;
}

/// Per-context tuning overrides: route policy that can differ between
/// two contexts in one process (fields left null follow the process-wide
/// FUCINA_* gates; see `fucina.tuning`).
pub fn setTuning(self: *ExecContext, overrides: tuning.Overrides) void {
    self.rt.tuning = overrides;
}

pub fn deinit(self: *ExecContext) void {
    self.decode_scratch.deinit(self.rt.allocator);
    if (self.rt.work_pool_ready) {
        setWorkPool(self, null);
        self.rt.work_pool.deinit();
    }
    self.rt.scopes.deinit(self.rt.allocator); // releases scopes left open at teardown
    self.rt.buffers.deinit();
    self.* = undefined;
}

// ------------------------------------------------------------------
// Exec scopes (see the design note on `ExecContext`).
// ------------------------------------------------------------------

pub fn execScopeActive(self: *const ExecContext) bool {
    return activeScopesConst(self).depth > 0;
}

/// Open a scope; close it with `closeExecScope(mark)` (typically `defer`).
pub fn openExecScope(self: *ExecContext) ExecScope {
    const scopes = activeScopes(self);
    scopes.depth += 1;
    return .{ .index = scopes.entries.items.len };
}

/// Release every reference adopted since `mark`, newest first. Every
/// handle adopted in the scope is a borrow and is invalid from here on
/// (its `deinit` stays a no-op). Close after any backward over the
/// scope's results: the graph is reference-counted, but the scope holds
/// the only reference to a result nobody else retained.
pub fn closeExecScope(self: *ExecContext, mark: ExecScope) void {
    const scopes = activeScopes(self);
    std.debug.assert(scopes.depth > 0);
    scopes.depth -= 1;
    scopes.releaseTo(mark.index);
}

/// Hand the innermost scope the references of one result, all or nothing:
/// on failure (the stack could not grow) no entry was taken and the caller
/// still owns everything. Op tails call it BEFORE handing the value to the
/// returned handle, so nothing needs to be un-consumed on the error path.
pub fn adopt(self: *ExecContext, entries: []const ScopeEntry) !void {
    const scopes = activeScopes(self);
    std.debug.assert(scopes.depth > 0);
    try scopes.entries.ensureUnusedCapacity(self.rt.allocator, entries.len);
    scopes.entries.appendSliceAssumeCapacity(entries);
}

// ------------------------------------------------------------------
// Quantized-RHS facade dot: GPU pin.
// ------------------------------------------------------------------

/// Handle returned by `disableQuantDotGpu`; `close` lifts the pin (nests).
pub const QuantDotGpuDisabledScope = struct {
    ctx: *ExecContext,
    active: bool = true,

    pub fn close(self: *QuantDotGpuDisabledScope) void {
        if (!self.active) return;
        const old = self.ctx.quant_dot_gpu_disabled.fetchSub(1, .acq_rel);
        std.debug.assert(old > 0);
        self.active = false;
    }
};

/// Pin the quantized-RHS facade dot to the CPU kernels on this context
/// while the returned scope is open. A depth count, atomic because
/// checkpoint recomputes may hold it from several backward threads at
/// once: the dot takes the GPU only when no scope is open anywhere on the
/// context. Checkpoint blocks pin both the forward run and the recompute
/// so the two passes take the same kernels and stay bitwise.
pub fn disableQuantDotGpu(self: *ExecContext) QuantDotGpuDisabledScope {
    _ = self.quant_dot_gpu_disabled.fetchAdd(1, .acq_rel);
    return .{ .ctx = self };
}

pub fn quantDotGpuEnabled(self: *const ExecContext) bool {
    return self.quant_dot_gpu_disabled.load(.acquire) == 0;
}

// ------------------------------------------------------------------
// Quant-matmul batch numerics: the speculative-verify rowwise pin.
// ------------------------------------------------------------------

/// Handle returned by `pinRowwiseNumerics`; `close` lifts the pin (nests).
pub const RowwiseNumericsScope = struct {
    ctx: *ExecContext,
    active: bool = true,

    pub fn close(self: *RowwiseNumericsScope) void {
        if (!self.active) return;
        std.debug.assert(self.ctx.rowwise_numerics_pinned > 0);
        self.ctx.rowwise_numerics_pinned -= 1;
        self.active = false;
    }
};

/// Pin every quant-matmul entry on this context to the m == 1 kernel
/// numerics while the returned scope is open: batched
/// entries reproduce the m == 1 kernel numerics bitwise, so a
/// speculative-verify batch's logits — and the KV rows it leaves behind —
/// equal sequential decode's (the lossless-speculation requirement at any
/// draft depth). Open around a speculative VERIFY forward only: batch
/// matmul throughput is sacrificed while pinned. A depth count, so scopes
/// nest; like every other context mutation it belongs to one thread.
pub fn pinRowwiseNumerics(self: *ExecContext) RowwiseNumericsScope {
    self.rowwise_numerics_pinned += 1;
    return .{ .ctx = self };
}

/// True while a `pinRowwiseNumerics` scope is open: the quant matmuls
/// then run one row at a time on the m == 1 kernels.
pub fn rowwiseNumericsPinned(self: *const ExecContext) bool {
    return self.rowwise_numerics_pinned > 0;
}

// ------------------------------------------------------------------
// Worker team.
// ------------------------------------------------------------------

/// The runtime's own error names, beyond `TensorError` and the quantized
/// block-length check; `exec.Error` merges the three domains.
pub const ExecError = error{
    /// A latched pool-init failure, reported by every later `tryWorkPool`.
    WorkPoolUnavailable,
    /// `checkFloatEnvironment`: the calling thread's rounding or denormal
    /// mode differs from the one recorded at `init`.
    FloatEnvironmentChanged,
    /// `groupedAttention` has no kernel for this option combination.
    UnsupportedAttentionVariant,
};

/// The worker team, created on the first call. A failed `Pool.init` is
/// latched in `work_pool_failed`: one warning, then every later call
/// returns `WorkPoolUnavailable` without retrying, so a context whose team
/// could not start runs serially (`workPool` = null) instead of paying the
/// failed init on every dispatch.
pub fn tryWorkPool(self: *ExecContext) !*thread.Pool {
    self.rt.work_pool_mutex.lock();
    defer self.rt.work_pool_mutex.unlock();

    if (self.rt.work_pool_failed) return ExecError.WorkPoolUnavailable;
    if (!self.rt.work_pool_ready) {
        const worker_threads = parallel.cpuThreadCount(parallel.vector_max_threads) - 1;
        self.rt.work_pool.init(.{
            .allocator = self.rt.allocator,
            .max_workers = worker_threads,
        }) catch |err| {
            self.rt.work_pool_failed = true;
            std.log.warn("exec: worker pool init failed ({s}); this context runs kernels serially", .{@errorName(err)});
            return err;
        };
        self.rt.work_pool_ready = true;
        setWorkPool(self, &self.rt.work_pool);
    }
    return &self.rt.work_pool;
}

pub fn workPool(self: *ExecContext) ?*thread.Pool {
    return self.tryWorkPool() catch null;
}

/// Publish (or retract) the worker team for kernel dispatch. Release
/// ordering pairs with the acquire load in `pc`, so a racing first observer
/// on another thread also sees `Pool.init`'s writes.
fn setWorkPool(self: *ExecContext, pool: ?*thread.Pool) void {
    self.rt.parallel_pool.store(pool, .release);
}

/// The `ParallelConfig` every pool-taking kernel call receives: a snapshot
/// of the published worker team (`null` runs the kernel serially).
pub fn pc(self: *const ExecContext) backend_mod.ParallelConfig {
    return .{ .pool = self.rt.parallel_pool.load(.acquire) };
}

/// The one range dispatch of the runtime: run `run(ctx, start, end)`
/// over `[0, total)`, split into at most `max_parts` proportional parts
/// across the worker team (`backend.tile.forRange`, the one boundary
/// formula), or as one serial call over the whole range when the team is
/// absent or the cap is 1. The cap is the caller's gate, stated at the
/// call site — `if (worth_it) total else 1`, `innerLaneParts`,
/// `parallel.partsForChunk` — so whether a shape is worth splitting stays
/// per-op knowledge while how a range splits is written once. The actual
/// part count is the cap clamped to the team size. Parts own disjoint
/// sub-ranges, so a pooled result is bitwise the serial call's for any
/// part count.
pub fn forRange(
    self: *ExecContext,
    total: usize,
    max_parts: usize,
    ctx: anytype,
    comptime run: fn (@TypeOf(ctx), usize, usize) void,
) void {
    if (max_parts > 1) {
        if (self.workPool()) |pool| {
            const parts = @min(parallel.cpuThreadCount(parallel.vector_max_threads), max_parts);
            if (parts > 1) return backend_mod.tile.forRange(pool, @TypeOf(ctx), ctx, total, parts, run);
        }
    }
    run(ctx, 0, total);
}

// Minimum lanes per part when splitting an inner-lane kernel across the
// pool: below this the per-dispatch cost outweighs the split.
const min_inner_lanes_per_part = 64;

/// The part cap of the inner-lane family (`backend.rows`' `*Inner`
/// kernels: softmax, the stats/norm non-last-axis arms): `inner` lanes
/// split at `min_inner_lanes_per_part` per part once the whole tensor
/// clears the row-kernel threshold, one part below it. Lanes are
/// independent and scratch columns disjoint, so any part count is bitwise
/// the serial call.
pub fn innerLaneParts(total_len: usize, inner: usize) usize {
    if (total_len < parallel.row_kernel_len_threshold) return 1;
    return inner / min_inner_lanes_per_part;
}

// ------------------------------------------------------------------
// Tensor allocation primitives. Kernels never allocate; these are the
// only source of transient tensors (all backed by the buffer pool: the
// f32 arm for default-dtype tensors, the byte-slab arm for every other
// storage dtype).
// ------------------------------------------------------------------

/// Uninitialized tensor of `dtype` with `shape`: a `[n]usize` array, a
/// tuple of sizes, or a `[]const usize` slice (`shape.Shape.from` is the
/// one normalization).
pub fn empty(self: *ExecContext, comptime dtype: DType, shape: anytype) !tensor.TensorOf(dtype) {
    const dims = try Shape.from(shape);
    const len = try shape_mod.storageElementCount(dtype, dims.slice());
    const buffer = try self.rt.buffers.acquireTyped(dtype, len);
    errdefer buffer.release();
    return tensor.TensorOf(dtype).fromOwnedBuffer(buffer, dims.slice());
}

/// Zero-copy broadcast view of `x` with `shape` (array/tuple or slice).
pub fn broadcastTo(self: *ExecContext, x: *const Tensor, shape: anytype) !Tensor {
    _ = self;
    const dims = try Shape.from(shape);
    return x.broadcastTo(dims.slice());
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
    const dims = try Shape.from(shape);
    if (try shape_mod.elementCount(dims.slice()) != values.len) return tensor.TensorError.InvalidDataLength;
    var out = try self.empty(dtype, dims.slice());
    @memcpy(out.data(), values);
    return out;
}

pub fn fromBorrowedSlice(self: *ExecContext, comptime dtype: DType, shape: anytype, values: []dtype_mod.Scalar(dtype)) !tensor.TensorOf(dtype) {
    const dims = try Shape.from(shape);
    return tensor.TensorOf(dtype).fromBorrowedSlice(self.rt.allocator, dims.slice(), values);
}

pub fn fromStorageSlice(self: *ExecContext, comptime dtype: DType, shape: anytype, values: []const dtype_mod.Storage(dtype)) !tensor.TensorOf(dtype) {
    const dims = try Shape.from(shape);
    if (try shape_mod.storageElementCount(dtype, dims.slice()) != values.len) return tensor.TensorError.InvalidDataLength;
    var out = try self.empty(dtype, dims.slice());
    @memcpy(out.data(), values);
    return out;
}

pub fn fromBorrowedStorageSlice(self: *ExecContext, comptime dtype: DType, shape: anytype, values: []dtype_mod.Storage(dtype)) !tensor.TensorOf(dtype) {
    const dims = try Shape.from(shape);
    return tensor.TensorOf(dtype).fromBorrowedStorageSlice(self.rt.allocator, dims.slice(), values);
}

pub fn materialize(self: *ExecContext, comptime dtype: DType, x: *const tensor.TensorOf(dtype)) !tensor.TensorOf(dtype) {
    var out = try self.empty(dtype, x.shape.slice());
    errdefer out.deinit();
    if (comptime dtype_mod.isScalar(dtype)) {
        if (!x.isContiguous()) {
            const dst = out.data();
            // One host fence up front: the parts each waitReady on the
            // same buffer, and N workers spinning on one pending
            // accelerator Work is wasted parallelism (waitReady itself
            // is claimant-safe either way). The source is read-only, so
            // disjoint ranges of the row-major destination copy
            // concurrently; a copy is order-free, so any part count is
            // the same bytes.
            x.buffer.waitReady();
            const parts = if (dst.len >= parallel.materialize_parallel_len_threshold)
                (dst.len + parallel.materialize_parallel_min_chunk - 1) / parallel.materialize_parallel_min_chunk
            else
                1;
            self.forRange(dst.len, parts, MaterializeRange(dtype){ .src = x, .dst = dst }, MaterializeRange(dtype).run);
            return out;
        }
    }
    try x.copyTo(out.data());
    return out;
}

/// One range of a strided-view materialization: the row-major
/// destination `[start, end)` copied from the source's linearization.
fn MaterializeRange(comptime dtype: DType) type {
    return struct {
        src: *const tensor.TensorOf(dtype),
        dst: []dtype_mod.Storage(dtype),

        fn run(self: @This(), start: usize, end: usize) void {
            self.src.copyRangeTo(self.dst[start..end], start, end - start);
        }
    };
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

/// `x` as the kernel's compute dtype: a contiguous borrow (or copy) when
/// the storage dtype is the compute dtype, else one widening cast. The
/// entry side of the `.widened` dtype policy (`dtype_mod.computeDType`).
pub fn prepareAs(
    self: *ExecContext,
    comptime dtype: DType,
    comptime compute: DType,
    x: *const tensor.TensorOf(dtype),
) !PreparedTensorOf(compute) {
    if (comptime dtype == compute) return self.prepareContiguous(dtype, x);
    return .{ .owned = try self.cast(dtype, compute, x) };
}

/// The compute dtype of a `.widened` op (f32), or a compile error naming
/// the op when no kernel exists for the storage dtype (f64 and the
/// non-float dtypes).
pub fn widenedCompute(comptime dtype: DType, comptime what: []const u8) DType {
    const compute = dtype_mod.computeDType(.widened, dtype);
    if (compute != .f32) @compileError(what ++ ": no " ++ @tagName(dtype) ++ " kernel (f32, f16 and bf16 are supported)");
    return compute;
}

/// A compute-dtype result stored as `out`: the value itself when the
/// dtypes agree, else one narrowing cast (the compute-dtype value is
/// released). The store side of the `.widened` policy: f32 accumulation,
/// one final round.
pub fn storeAs(
    self: *ExecContext,
    comptime compute: DType,
    comptime out: DType,
    value: tensor.TensorOf(compute),
) !tensor.TensorOf(out) {
    if (comptime compute == out) return value;
    var wide = value;
    defer wide.deinit();
    return self.cast(compute, out, &wide);
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
/// safe no-op for scope-owned op results (borrows; the scope releases at
/// close), so the same forward code is also training-safe.
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
