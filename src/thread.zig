const std = @import("std");
const builtin = @import("builtin");
const parallel = @import("parallel.zig");

const Allocator = std.mem.Allocator;
const Alignment = std.mem.Alignment;
const Io = std.Io;

// macOS scheduler hint: pin a thread's quality-of-service so the scheduler
// prefers a performance core. A busy worker that does not set this can be
// classified as background work and migrated to an efficiency (E) core, which
// stalls the fork-join barrier on the slow straggler. This is the one
// platform-specific concession in an otherwise portable team; everywhere else
// `pinToPerformanceCores` compiles to nothing.
const qos_class_user_interactive: c_uint = 0x21;
extern "c" fn pthread_set_qos_class_self_np(qos_class: c_uint, relative_priority: c_int) c_int;

fn pinToPerformanceCores() void {
    if (comptime builtin.os.tag == .macos) {
        _ = pthread_set_qos_class_self_np(qos_class_user_interactive, 0);
    }
}

pub const Mutex = struct {
    inner: Io.Mutex = .init,

    pub fn lock(self: *Mutex) void {
        std.Io.Threaded.mutexLock(&self.inner);
    }

    pub fn unlock(self: *Mutex) void {
        std.Io.Threaded.mutexUnlock(&self.inner);
    }
};

pub const Condition = struct {
    inner: Io.Condition = .init,

    pub fn wait(self: *Condition, io: Io, mutex: *Mutex) void {
        self.inner.waitUncancelable(io, &mutex.inner);
    }

    pub fn broadcast(self: *Condition, io: Io) void {
        self.inner.broadcast(io);
    }
};

pub const ThreadSafeAllocator = struct {
    child_allocator: Allocator,
    mutex: Mutex = .{},

    pub fn allocator(self: *ThreadSafeAllocator) Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .remap = remap,
                .free = free,
            },
        };
    }

    fn alloc(ctx: *anyopaque, len: usize, alignment: Alignment, ret_addr: usize) ?[*]u8 {
        const self: *ThreadSafeAllocator = @ptrCast(@alignCast(ctx));
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.child_allocator.rawAlloc(len, alignment, ret_addr);
    }

    fn resize(ctx: *anyopaque, memory: []u8, alignment: Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *ThreadSafeAllocator = @ptrCast(@alignCast(ctx));
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.child_allocator.rawResize(memory, alignment, new_len, ret_addr);
    }

    fn remap(ctx: *anyopaque, memory: []u8, alignment: Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        const self: *ThreadSafeAllocator = @ptrCast(@alignCast(ctx));
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.child_allocator.rawRemap(memory, alignment, new_len, ret_addr);
    }

    fn free(ctx: *anyopaque, memory: []u8, alignment: Alignment, ret_addr: usize) void {
        const self: *ThreadSafeAllocator = @ptrCast(@alignCast(ctx));
        self.mutex.lock();
        defer self.mutex.unlock();
        self.child_allocator.rawFree(memory, alignment, ret_addr);
    }
};

pub const WaitGroup = struct {
    group: Io.Group = .init,
};

// Continuation handle passed to chained tasks: `enqueue(i)` makes `tasks[i]`
// runnable on the dispatching team. Contract in `Pool.parallelChained`: across
// a dispatch, each index must be enqueued exactly once.
pub const Chain = struct {
    ctx: *anyopaque,
    enqueue_fn: *const fn (*anyopaque, usize) void,

    pub fn enqueue(self: *const Chain, index: usize) void {
        self.enqueue_fn(self.ctx, index);
    }
};

pub const OneShotWorker = struct {
    threaded: Io.Threaded = undefined,
    io: Io = undefined,
    thread: std.Thread = undefined,
    state: std.atomic.Value(State) = .init(.idle),
    job_fn: ?JobFn = null,
    job_arg: ?*anyopaque = null,
    ready: bool = false,

    const State = enum(u32) { idle, reserved, job, done, stop };
    pub const JobFn = *const fn (*anyopaque) void;

    pub fn init(self: *OneShotWorker) !void {
        self.* = .{
            .threaded = Io.Threaded.init(.failing, .{
                .async_limit = .nothing,
                .concurrent_limit = .nothing,
            }),
        };
        self.io = self.threaded.io();
        self.thread = std.Thread.spawn(.{}, workerMain, .{self}) catch |err| {
            self.threaded.deinit();
            self.* = undefined;
            return err;
        };
        self.ready = true;
    }

    pub fn deinit(self: *OneShotWorker) void {
        if (!self.ready) return;

        self.state.store(.stop, .release);
        self.io.futexWake(State, &self.state.raw, 1);

        self.thread.join();
        self.threaded.deinit();
        self.* = undefined;
    }

    pub fn start(self: *OneShotWorker, job_fn: JobFn, job_arg: *anyopaque) bool {
        if (self.state.cmpxchgStrong(.idle, .reserved, .acquire, .monotonic) != null) return false;
        self.job_fn = job_fn;
        self.job_arg = job_arg;
        self.state.store(.job, .release);
        self.io.futexWake(State, &self.state.raw, 1);
        return true;
    }

    pub fn wait(self: *OneShotWorker) void {
        while (true) {
            const state = self.state.load(.acquire);
            switch (state) {
                .done => {
                    self.state.store(.idle, .release);
                    self.io.futexWake(State, &self.state.raw, 1);
                    return;
                },
                .reserved, .job => self.io.futexWaitUncancelable(State, &self.state.raw, state),
                .idle, .stop => unreachable,
            }
        }
    }

    fn workerMain(self: *OneShotWorker) void {
        pinToPerformanceCores();
        while (true) {
            switch (self.state.load(.acquire)) {
                .idle, .reserved, .done => |state| {
                    self.io.futexWaitUncancelable(State, &self.state.raw, state);
                    continue;
                },
                .stop => return,
                .job => {},
            }
            const job_fn = self.job_fn.?;
            const job_arg = self.job_arg.?;

            job_fn(job_arg);

            if (self.state.cmpxchgStrong(.job, .done, .release, .acquire)) |state| {
                std.debug.assert(state == .stop);
                return;
            }
            self.io.futexWake(State, &self.state.raw, 1);
        }
    }
};

pub const Pool = struct {
    threaded: Io.Threaded = undefined,
    io: Io = undefined,
    allocator: Allocator = undefined,
    max_workers: usize = 0,
    barrier: ?*BarrierPool = null,
    barrier_mutex: Mutex = .{},
    parallel_chunks_active: std.atomic.Value(bool) = .init(false),

    pub const InitOptions = struct {
        allocator: Allocator,
        max_workers: usize = std.math.maxInt(usize),
    };

    pub fn init(self: *Pool, options: InitOptions) !void {
        self.* = .{
            .threaded = Io.Threaded.init(options.allocator, .{
                .async_limit = .nothing,
                .concurrent_limit = Io.Limit.limited(options.max_workers),
            }),
            .allocator = options.allocator,
            .max_workers = options.max_workers,
        };
        self.io = self.threaded.io();
    }

    pub fn deinit(self: *Pool) void {
        if (self.barrier) |barrier| {
            barrier.shutdownAndJoin();
            self.allocator.destroy(barrier);
        }
        self.threaded.deinit();
        self.* = undefined;
    }

    // Fork-join parallel-for over a persistent hot team: runs `run(&tasks[i])`
    // for every i in [0, tasks.len), the caller executing chunk 0 and the team
    // the rest, rendezvousing before return. This is the default substrate for
    // splitting a numeric kernel across cores. Unlike `spawnWg`/`waitAndWork`
    // (general async tasks routed through std.Io's executor, which heap-allocs a
    // task node per spawn and parks/wakes each worker via a futex syscall), the
    // team here stays hot between dispatches (spin-then-park), so a dense stream
    // of small ops costs atomics instead of kernel round-trips.
    pub fn parallelChunks(
        self: *Pool,
        comptime Task: type,
        tasks: []const Task,
        comptime run: fn (*const Task) void,
    ) void {
        const n = tasks.len;
        if (n == 0) return;
        if (n == 1) {
            run(&tasks[0]);
            return;
        }
        const barrier = self.ensureBarrier() catch null;
        if (barrier == null or barrier.?.worker_count == 0) {
            for (tasks) |*t| run(t);
            return;
        }
        if (self.parallel_chunks_active.cmpxchgStrong(false, true, .acquire, .monotonic) != null) {
            for (tasks) |*t| run(t);
            return;
        }
        defer self.parallel_chunks_active.store(false, .release);

        const Thunk = struct {
            fn call(ctx: *anyopaque, index: usize) void {
                const base: [*]const Task = @ptrCast(@alignCast(ctx));
                run(&base[index]);
            }
        };
        barrier.?.dispatch(n, @ptrCast(@constCast(tasks.ptr)), Thunk.call);
    }

    // Dependency-chained fork-join over the same hot team: tasks[0..initial_count)
    // start runnable, and a running task makes a successor runnable via
    // `chain.enqueue(i)`. Returns false when the team is unavailable or already
    // dispatching (the caller must then run the graph itself).
    //
    // Enqueue contract (unchecked in ReleaseFast; safety builds panic on
    // violation): every index in [0, tasks.len) must become runnable exactly
    // once — the initial_count seeds plus all chain.enqueue calls together
    // cover each index once and only once. Under-enqueueing never terminates:
    // dispatcher and workers spin on the completion count with no park.
    // Enqueueing an index twice corrupts the intrusive Treiber stack (indices
    // are the nodes; the load-next/CAS pop is ABA-unsafe), silently running a
    // task twice, losing another's entry, or orphaning part of the stack.
    // Enqueueing an index >= tasks.len corrupts the successor table.
    pub fn parallelChained(
        self: *Pool,
        comptime Task: type,
        tasks: []Task,
        initial_count: usize,
        comptime run: fn (*Task, *const Chain) void,
    ) bool {
        const n = tasks.len;
        if (n == 0) return true;
        if (initial_count == 0 or initial_count > n) return false;
        const barrier = self.ensureBarrier() catch return false;
        if (barrier.worker_count == 0) return false;
        if (self.parallel_chunks_active.cmpxchgStrong(false, true, .acquire, .monotonic) != null) return false;
        defer self.parallel_chunks_active.store(false, .release);

        const Thunk = struct {
            fn call(ctx: *anyopaque, index: usize, chain: *const Chain) void {
                const base: [*]Task = @ptrCast(@alignCast(ctx));
                run(&base[index], chain);
            }
        };
        barrier.dispatchChained(n, initial_count, @ptrCast(tasks.ptr), Thunk.call) catch return false;
        return true;
    }

    /// Worker count the barrier team is (or would be) sized to. Routes
    /// through `parallel.cpuThreadCount` so every pool inherits one sizing
    /// policy: physical-core-aware on SMT machines, FUCINA_MAX_THREADS /
    /// setMaxThreads respected, then the pool's own max_workers cap.
    fn sizedWorkerCount(self: *Pool) usize {
        const cpu_workers = parallel.cpuThreadCount(std.math.maxInt(usize)) -| 1;
        return @min(self.max_workers, @max(cpu_workers, 1));
    }

    /// Dispatch participants (workers + the dispatching thread) that
    /// parallelChunks/parallelChained will use, without forcing team
    /// creation. Mirrors ensureBarrier's sizing.
    pub fn teamSize(self: *Pool) usize {
        if (@atomicLoad(?*BarrierPool, &self.barrier, .acquire)) |barrier| return barrier.worker_count + 1;
        return self.sizedWorkerCount() + 1;
    }

    fn ensureBarrier(self: *Pool) !*BarrierPool {
        if (@atomicLoad(?*BarrierPool, &self.barrier, .acquire)) |barrier| return barrier;
        self.barrier_mutex.lock();
        defer self.barrier_mutex.unlock();
        if (self.barrier) |barrier| return barrier;

        const worker_count = self.sizedWorkerCount();

        const barrier = try self.allocator.create(BarrierPool);
        errdefer self.allocator.destroy(barrier);
        try barrier.init(self.allocator, self.io, worker_count);

        @atomicStore(?*BarrierPool, &self.barrier, barrier, .release);
        return barrier;
    }

    pub fn spawnWg(self: *Pool, wait_group: *WaitGroup, comptime function: anytype, args: std.meta.ArgsTuple(@TypeOf(function))) bool {
        wait_group.group.concurrent(self.io, function, args) catch {
            runEager(function, args);
            return false;
        };
        return true;
    }

    pub fn trySpawnWg(self: *Pool, wait_group: *WaitGroup, comptime function: anytype, args: std.meta.ArgsTuple(@TypeOf(function))) bool {
        wait_group.group.concurrent(self.io, function, args) catch return false;
        return true;
    }

    pub fn waitAndWork(self: *Pool, wait_group: *WaitGroup) void {
        wait_group.group.await(self.io) catch |err| switch (err) {
            error.Canceled => {},
        };
    }

    fn runEager(comptime function: anytype, args: std.meta.ArgsTuple(@TypeOf(function))) void {
        _ = @as(Io.Cancelable!void, @call(.auto, function, args)) catch {};
    }
};

// Persistent fork-join team backing `Pool.parallelChunks`. Worker threads stay
// hot by spinning on a generation counter for a short budget after each
// dispatch, then park on a futex so a long-idle team consumes no CPU. The
// budget is sized to bridge the microsecond setup gap between consecutive ops
// in a dense stream (e.g. a transformer forward) without sustaining the all-core
// load that would throttle the clock. A dispatch publishes the job and bumps the
// generation with one broadcast wake; the dispatcher participates as chunk 0 and
// busy-waits the completion counter. Workers pin to performance cores (a no-op
// off macOS) so a brief spin does not get them demoted to efficiency cores.
//
// Memory model: `generation` is the synchronizing variable. The dispatcher
// writes the job fields, then publishes with a release add; each worker observes
// the new generation with an acquire load, which makes the job fields visible.
// Completion is published with a release add on `done` and observed by the
// dispatcher with an acquire load.
const BarrierPool = struct {
    allocator: Allocator,
    io: Io,
    threads: []std.Thread,
    worker_count: usize,

    job_run: JobFn = undefined,
    chain_run: ChainJobFn = undefined,
    job_ctx: *anyopaque = undefined,
    job_count: usize = 0,
    job_mode: JobMode = .chunks,
    job_next: std.atomic.Value(usize) = .init(0),
    job_claim_chunk: usize = 1,
    generation: std.atomic.Value(u32) = .init(0),
    done: std.atomic.Value(u32) = .init(0),
    ready: std.atomic.Value(u32) = .init(0),
    parked: std.atomic.Value(u32) = .init(0),
    shutdown: std.atomic.Value(bool) = .init(false),
    chain_next: []usize = &.{},
    chain_head: std.atomic.Value(usize) = .init(chain_empty),
    chain_completed: std.atomic.Value(usize) = .init(0),
    // Safety-build-only instrumentation of the chained enqueue contract (see
    // Pool.parallelChained); never touched in ReleaseFast/ReleaseSmall.
    chain_seen: []std.atomic.Value(u8) = &.{},
    chain_enqueued: std.atomic.Value(usize) = .init(0),
    chain_inflight: std.atomic.Value(usize) = .init(0),
    // Immutable after init (resolveSpinBudget); workers hoist it to a local.
    spin_budget: u32 = default_spin_budget,

    const JobFn = *const fn (ctx: *anyopaque, index: usize) void;
    const ChainJobFn = *const fn (ctx: *anyopaque, index: usize, chain: *const Chain) void;
    const JobMode = enum(u8) { chunks, chained };
    const chain_empty = std.math.maxInt(usize);

    // Gate for the chained-contract checks: on in Debug/ReleaseSafe, compiled
    // out (zero cost, no allocation) in ReleaseFast/ReleaseSmall.
    const chain_checks = std.debug.runtime_safety;
    // Consecutive nothing-runnable / nothing-in-flight / no-progress samples a
    // safety build tolerates before declaring the dispatch under-enqueued and
    // panicking (~seconds of all-core spin; unreachable when the exact-count
    // contract holds).
    const chain_stall_limit: u64 = 1 << 32;

    // Spin iterations (each a `spinLoopHint`) a worker performs after finishing
    // before it parks. Tunable: larger bridges longer gaps but risks sustained
    // all-core load (DVFS throttle); smaller re-parks more often (re-wake cost).
    // The 32768 default is the measured M1 tuning (long spin throttles the
    // M1 clock — docs/BENCHMARK.md thermal protocol) and it also survived a full
    // cool-machine sweep on x86 (i9-13950HX, 2026-07-03): the response is
    // workload-coupled and U-shaped, so NO static value dominates there
    // either — 512 makes the parakeet m=93 encode ~6% faster (workers park
    // through the encoder's serial/host sections and the power-limited
    // package boosts the active cores) but costs qwen3 q8_0 pp256 ~6% and
    // decode ~2% (parked workers re-wake ~5-30 us late into the denser LLM
    // op stream, and the join waits for every worker); 2048-4096 is the
    // worst of both (burns the whole window, then parks right before the
    // wake); 262144 regresses the encode ~5% (spin power starves the compute
    // cores). Hence a runtime knob instead of a retune: FUCINA_SPIN_BUDGET
    // (read once per team init, exact FUCINA_MAX_THREADS precedent) lets
    // encode-heavy deployments opt into a short window (e.g. 512 for ASR
    // batch/serving) while the default keeps the LLM sentinels intact.
    const default_spin_budget: u32 = 32768;

    fn resolveSpinBudget() u32 {
        const raw = parallel.envSpinBudget() orelse return default_spin_budget;
        return std.math.cast(u32, raw) orelse default_spin_budget;
    }

    fn init(self: *BarrierPool, allocator: Allocator, io: Io, worker_count: usize) !void {
        // The dispatcher thread runs chunk 0 of every parallel op, so it must
        // hold the same QoS as the spawned workers — at default QoS macOS may
        // schedule it on an E-core, and the fork-join barrier then runs every
        // dispatch at the straggler's speed.
        pinToPerformanceCores();
        self.* = .{
            .allocator = allocator,
            .io = io,
            .threads = &.{},
            .worker_count = worker_count,
            .spin_budget = resolveSpinBudget(),
        };
        if (worker_count == 0) return;

        const threads = try allocator.alloc(std.Thread, worker_count);
        errdefer allocator.free(threads);

        var spawned: usize = 0;
        errdefer {
            while (self.ready.load(.acquire) < spawned) std.atomic.spinLoopHint();
            self.shutdown.store(true, .release);
            _ = self.generation.fetchAdd(1, .release);
            self.io.futexWake(u32, &self.generation.raw, @intCast(spawned));
            for (threads[0..spawned]) |t| t.join();
        }
        while (spawned < worker_count) : (spawned += 1) {
            // Worker indices are 1.. so chunk 0 stays with the dispatcher.
            threads[spawned] = try std.Thread.spawn(.{}, workerMain, .{ self, spawned + 1 });
        }
        self.threads = threads;

        // Wait until every worker has latched generation 0 and entered its wait
        // loop. Without this, a dispatch could advance the generation before a
        // slow-starting worker reads its baseline, making that worker miss the
        // dispatch and never reach the completion barrier.
        while (self.ready.load(.acquire) < worker_count) std.atomic.spinLoopHint();
    }

    fn shutdownAndJoin(self: *BarrierPool) void {
        if (self.worker_count == 0) return;
        self.shutdown.store(true, .release);
        _ = self.generation.fetchAdd(1, .release);
        self.io.futexWake(u32, &self.generation.raw, @intCast(self.worker_count));
        for (self.threads) |t| t.join();
        self.allocator.free(self.threads);
        if (self.chain_next.len > 0) self.allocator.free(self.chain_next);
        if (self.chain_seen.len > 0) self.allocator.free(self.chain_seen);
        self.threads = &.{};
        self.chain_next = &.{};
        self.chain_seen = &.{};
        self.chain_head.store(chain_empty, .release);
        self.worker_count = 0;
    }

    fn dynamicClaimChunk(count: usize, participants: usize) usize {
        if (count <= participants * 4) return 1;
        if (count <= participants * 16) return 2;
        return 4;
    }

    fn dispatch(self: *BarrierPool, count: usize, ctx: *anyopaque, run: JobFn) void {
        self.job_ctx = ctx;
        self.job_run = run;
        self.job_count = count;
        const participants = self.worker_count + 1;
        self.job_mode = .chunks;
        self.job_next.store(0, .release);
        self.job_claim_chunk = dynamicClaimChunk(count, participants);
        self.done.store(0, .release);
        // The generation bump and the parked check form a store-load pair with
        // the worker's parked increment + futex re-check of the generation:
        // both sides are seq_cst so either the dispatcher observes the parker
        // (and wakes), or the parker's kernel-side generation compare observes
        // the bump (and refuses to sleep). In a dense op stream every worker is
        // inside its spin window, so this skips one syscall per dispatch.
        _ = self.generation.fetchAdd(1, .seq_cst);
        if (self.parked.load(.seq_cst) != 0) {
            self.io.futexWake(u32, &self.generation.raw, @intCast(self.worker_count));
        }

        self.runChunkClaims();

        // Every worker increments `done` exactly once per generation. The
        // dispatcher owns a core and has no other work until the join, so it
        // pure-spins rather than issuing sched_yield syscalls.
        while (self.done.load(.acquire) < self.worker_count) std.atomic.spinLoopHint();
    }

    fn ensureChainReadyCapacity(self: *BarrierPool, count: usize) !void {
        if (self.chain_next.len >= count) return;
        const grown = try self.allocator.alloc(usize, count);
        errdefer self.allocator.free(grown);
        if (comptime chain_checks) {
            const seen = try self.allocator.alloc(std.atomic.Value(u8), count);
            if (self.chain_seen.len > 0) self.allocator.free(self.chain_seen);
            self.chain_seen = seen;
        }
        if (self.chain_next.len > 0) self.allocator.free(self.chain_next);
        self.chain_next = grown;
    }

    fn dispatchChained(
        self: *BarrierPool,
        count: usize,
        initial_count: usize,
        ctx: *anyopaque,
        run: ChainJobFn,
    ) !void {
        try self.ensureChainReadyCapacity(count);

        self.job_ctx = ctx;
        self.chain_run = run;
        self.job_count = count;
        self.job_mode = .chained;
        self.done.store(0, .release);
        self.chain_completed.store(0, .release);
        if (comptime chain_checks) {
            self.chain_enqueued.store(0, .monotonic);
            self.chain_inflight.store(0, .monotonic);
            for (self.chain_seen[0..count], 0..) |*seen, i| {
                seen.* = .init(if (i < initial_count) 1 else 0);
            }
        }
        for (0..initial_count) |i| {
            self.chain_next[i] = if (i + 1 < initial_count) i + 1 else chain_empty;
        }
        self.chain_head.store(0, .release);

        _ = self.generation.fetchAdd(1, .seq_cst);
        if (self.parked.load(.seq_cst) != 0) {
            self.io.futexWake(u32, &self.generation.raw, @intCast(self.worker_count));
        }

        self.runChainClaims();
        while (self.done.load(.acquire) < self.worker_count) std.atomic.spinLoopHint();

        if (comptime chain_checks) {
            if (initial_count + self.chain_enqueued.load(.monotonic) != count) {
                @panic("BarrierPool.dispatchChained: initial_count + chain.enqueue calls must equal the task count (each index exactly once)");
            }
        }
    }

    fn runChunkClaims(self: *BarrierPool) void {
        while (true) {
            const start = self.job_next.fetchAdd(self.job_claim_chunk, .monotonic);
            if (start >= self.job_count) return;
            const end = @min(self.job_count, start + self.job_claim_chunk);
            var c = start;
            while (c < end) : (c += 1) {
                self.job_run(self.job_ctx, c);
            }
        }
    }

    fn chainEnqueueOpaque(ctx: *anyopaque, index: usize) void {
        const self: *BarrierPool = @ptrCast(@alignCast(ctx));
        std.debug.assert(index < self.job_count);
        if (comptime chain_checks) {
            if (self.chain_seen[index].swap(1, .monotonic) != 0) {
                @panic("BarrierPool.chained: index enqueued twice — the Treiber pop is ABA-unsafe, each index must be enqueued exactly once");
            }
            _ = self.chain_enqueued.fetchAdd(1, .monotonic);
        }
        while (true) {
            const head = self.chain_head.load(.acquire);
            self.chain_next[index] = head;
            if (self.chain_head.cmpxchgStrong(head, index, .release, .monotonic) == null) return;
        }
    }

    fn chainPop(self: *BarrierPool) ?usize {
        while (true) {
            const head = self.chain_head.load(.acquire);
            if (head == chain_empty) return null;
            const next = self.chain_next[head];
            if (self.chain_head.cmpxchgStrong(head, next, .acq_rel, .acquire) == null) return head;
        }
    }

    fn runChainClaims(self: *BarrierPool) void {
        const chain = Chain{
            .ctx = self,
            .enqueue_fn = chainEnqueueOpaque,
        };
        var stall_completed: usize = 0;
        var idle_spins: u64 = 0;
        while (self.chain_completed.load(.acquire) < self.job_count) {
            if (self.chainPop()) |index| {
                if (comptime chain_checks) {
                    _ = self.chain_inflight.fetchAdd(1, .seq_cst);
                    idle_spins = 0;
                }
                self.chain_run(self.job_ctx, index, &chain);
                _ = self.chain_completed.fetchAdd(1, .release);
                if (comptime chain_checks) _ = self.chain_inflight.fetchSub(1, .seq_cst);
            } else {
                if (comptime chain_checks) {
                    // Only count a spin as stalled while no task is in flight
                    // (nothing that could still enqueue) and completion is not
                    // advancing; a genuine under-enqueue keeps both frozen.
                    const completed = self.chain_completed.load(.acquire);
                    if (completed != stall_completed or self.chain_inflight.load(.seq_cst) != 0) {
                        stall_completed = completed;
                        idle_spins = 0;
                    } else {
                        idle_spins += 1;
                        if (idle_spins >= chain_stall_limit) {
                            @panic("BarrierPool.chained: no runnable or in-flight task and no completion progress — a call site under-enqueued (initial_count + chain.enqueue calls must equal the task count)");
                        }
                    }
                }
                std.atomic.spinLoopHint();
            }
        }
    }

    fn workerMain(self: *BarrierPool, index: usize) void {
        pinToPerformanceCores();
        _ = index;
        const spin_budget = self.spin_budget;
        var seen: u32 = self.generation.load(.acquire);
        // Announce readiness only after latching the baseline generation, so
        // `init` cannot let a dispatch advance the generation behind our back.
        _ = self.ready.fetchAdd(1, .release);
        while (true) {
            var spins: u32 = 0;
            var gen = self.generation.load(.acquire);
            while (gen == seen) {
                spins +%= 1;
                if (spins < spin_budget) {
                    std.atomic.spinLoopHint();
                } else {
                    // Park. Announce first (seq_cst, paired with dispatch's
                    // parked check); the futex re-checks the value on entry, so
                    // a wake or wake-skip that raced ahead of the park is not
                    // lost — a stale `gen` makes the kernel refuse to sleep.
                    _ = self.parked.fetchAdd(1, .seq_cst);
                    self.io.futexWaitUncancelable(u32, &self.generation.raw, gen);
                    _ = self.parked.fetchSub(1, .seq_cst);
                    spins = 0;
                }
                gen = self.generation.load(.acquire);
            }
            seen = gen;

            if (self.shutdown.load(.acquire)) return;

            switch (self.job_mode) {
                .chunks => self.runChunkClaims(),
                .chained => self.runChainClaims(),
            }
            _ = self.done.fetchAdd(1, .release);
        }
    }
};

test {
    _ = @import("thread_tests.zig");
}
