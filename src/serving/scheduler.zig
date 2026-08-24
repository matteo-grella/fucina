//! Bounded FIFO queue + the single inference worker. Connection threads
//! accept and validate concurrently, but generation is strictly sequential:
//! one worker owns the backend (and through it the one ExecContext) — the
//! engine's intended shape, since a single forward pass already fork-joins
//! across every performance core and ExecContext is single-threaded by
//! contract (docs/reference/09-backends-cpu-simd-blas-threading-and-gpu-offload.md, Threading).

const std = @import("std");
const types = @import("fucina_models").text.serving;

fn lock(m: *std.Io.Mutex) void {
    std.Io.Threaded.mutexLock(m);
}

fn unlock(m: *std.Io.Mutex) void {
    std.Io.Threaded.mutexUnlock(m);
}

/// One queued generation. The CONNECTION thread owns the Job memory and its
/// request strings (arena) and blocks on `waitTimed`; the WORKER writes
/// reply bytes through `sink` while the job runs. The sink is driven by
/// exactly one thread at a time: the worker until `.finished`, the
/// connection thread afterwards. (The sink's DOWNSTREAM may be a
/// cross-thread pipe the connection thread drains concurrently — that
/// pipe synchronizes itself; see `http.StreamPipe`.)
pub const Job = struct {
    req: types.GenerateRequest,
    /// The OpenAI layer's per-request emitter (SSE frames or an
    /// accumulating body). Written by the worker only.
    sink: *std.Io.Writer,
    /// Optional extra futex word bumped and woken on finish. The connection
    /// thread waits on its stream pipe's sequence word (data arrivals bump
    /// it); finishing must wake that same word or the epilogue would sit
    /// out a full wait tick.
    notify: ?*std.atomic.Value(u32) = null,

    /// Futex word; `res`/`err` are published before the `.finished` store.
    state: std.atomic.Value(u32) = .{ .raw = @intFromEnum(State.queued) },
    /// Set by the connection thread on client disconnect (or by shutdown).
    /// The emitter checks it between tokens and fails the next write, which
    /// aborts generation.
    cancelled: std.atomic.Value(bool) = .{ .raw = false },
    res: types.GenerateResult = undefined,
    err: ?anyerror = null,

    pub const State = enum(u32) { queued, running, finished };

    pub fn cancel(self: *Job) void {
        self.cancelled.store(true, .release);
    }

    pub fn isCancelled(self: *const Job) bool {
        return self.cancelled.load(.acquire);
    }

    pub fn finished(self: *const Job) bool {
        return self.state.load(.acquire) == @intFromEnum(State.finished);
    }

    /// Wait up to `timeout_ns` for the worker to finish this job. Returns
    /// true when finished — callers loop, checking the connection for a
    /// client disconnect between waits (and `cancel` on hang-up).
    pub fn waitTimed(self: *Job, io: std.Io, timeout_ns: u64) bool {
        const current = self.state.load(.acquire);
        if (current == @intFromEnum(State.finished)) return true;
        io.futexWaitTimeout(u32, &self.state.raw, current, .{ .duration = .{
            .raw = .{ .nanoseconds = timeout_ns },
            .clock = .awake,
        } }) catch {};
        return self.finished();
    }

    fn finish(self: *Job, io: std.Io, res: types.GenerateResult, err: ?anyerror) void {
        self.res = res;
        self.err = err;
        self.state.store(@intFromEnum(State.finished), .release);
        io.futexWake(u32, &self.state.raw, std.math.maxInt(u32));
        if (self.notify) |word| {
            _ = word.fetchAdd(1, .release);
            io.futexWake(u32, &word.raw, std.math.maxInt(u32));
        }
    }

    fn setRunning(self: *Job) void {
        self.state.store(@intFromEnum(State.running), .release);
    }
};

pub const SubmitError = error{ QueueFull, ShuttingDown };

pub const Scheduler = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    backend: types.Backend,

    mutex: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    queue: std.ArrayList(*Job) = .empty,
    capacity: usize,
    /// Lockstep width: how many queued jobs one worker pass may decode
    /// together (1 = strictly sequential). Only honored when the backend
    /// has a `generate_batch` entry; batching never WAITS for jobs — the
    /// worker takes whatever is already queued, so an idle server keeps
    /// single-request latency.
    batch: usize,
    shutting_down: bool = false,
    /// The jobs the worker is generating right now (shutdown cancels them).
    running: std.ArrayList(*Job) = .empty,
    thread: ?std.Thread = null,

    pub fn init(allocator: std.mem.Allocator, io: std.Io, backend: types.Backend, queue_capacity: usize, batch: usize) Scheduler {
        return .{
            .allocator = allocator,
            .io = io,
            .backend = backend,
            .capacity = @max(queue_capacity, 1),
            .batch = @max(batch, 1),
        };
    }

    /// Effective lockstep width (1 when the backend cannot batch).
    fn groupCap(self: *const Scheduler) usize {
        return if (self.backend.vtable.generate_batch != null) self.batch else 1;
    }

    pub fn start(self: *Scheduler) !void {
        // Reserve the running list up front: the worker registers a group
        // under the queue lock, where allocation failure has no clean exit.
        try self.running.ensureTotalCapacity(self.allocator, self.groupCap());
        self.thread = try std.Thread.spawn(.{}, workerLoop, .{self});
    }

    /// Stop accepting, cancel the in-flight jobs, finish queued jobs with
    /// `error.ShuttingDown`, and join the worker.
    pub fn stop(self: *Scheduler) void {
        {
            lock(&self.mutex);
            defer unlock(&self.mutex);
            self.shutting_down = true;
            for (self.running.items) |job| job.cancel();
            self.cond.broadcast(self.io);
        }
        if (self.thread) |t| {
            t.join();
            self.thread = null;
        }
        self.queue.deinit(self.allocator);
        self.running.deinit(self.allocator);
    }

    /// Number of requests waiting or running (the /health load signal).
    pub fn depth(self: *Scheduler) usize {
        lock(&self.mutex);
        defer unlock(&self.mutex);
        return self.queue.items.len + self.running.items.len;
    }

    pub fn submit(self: *Scheduler, job: *Job) SubmitError!void {
        lock(&self.mutex);
        defer unlock(&self.mutex);
        if (self.shutting_down) return error.ShuttingDown;
        if (self.queue.items.len >= self.capacity) return error.QueueFull;
        self.queue.append(self.allocator, job) catch return error.QueueFull;
        self.cond.signal(self.io);
    }

    fn workerLoop(self: *Scheduler) void {
        const cap = self.groupCap();
        // Worker-owned scratch for one group; sized once.
        const group = self.allocator.alloc(*Job, cap) catch return;
        defer self.allocator.free(group);
        const reqs = self.allocator.alloc(*const types.GenerateRequest, cap) catch return;
        defer self.allocator.free(reqs);
        const sinks = self.allocator.alloc(*std.Io.Writer, cap) catch return;
        defer self.allocator.free(sinks);
        const results = self.allocator.alloc(types.GenerateResult, cap) catch return;
        defer self.allocator.free(results);
        const errs = self.allocator.alloc(?anyerror, cap) catch return;
        defer self.allocator.free(errs);

        while (true) {
            var g: usize = 0;
            var draining = false;
            {
                lock(&self.mutex);
                defer unlock(&self.mutex);
                while (self.queue.items.len == 0 and !self.shutting_down)
                    self.cond.waitUncancelable(self.io, &self.mutex);
                if (self.queue.items.len == 0) return; // shutting down, drained
                draining = self.shutting_down;
                // Take what is already queued, up to the group cap, and
                // register it while still holding the lock — `stop` must
                // never observe popped-but-unregistered jobs.
                while (g < cap and self.queue.items.len > 0) : (g += 1) {
                    group[g] = self.queue.orderedRemove(0);
                    if (!draining) self.running.appendAssumeCapacity(group[g]);
                }
            }

            if (draining) {
                for (group[0..g]) |job| job.finish(self.io, undefined, error.ShuttingDown);
                continue;
            }

            // Jobs cancelled while queued never run.
            var live: usize = 0;
            for (group[0..g]) |job| {
                if (job.isCancelled()) {
                    job.finish(self.io, undefined, error.Cancelled);
                } else {
                    group[live] = job;
                    live += 1;
                }
            }

            if (live == 1) {
                // The strictly-sequential path — identical to pre-batch
                // serving (and the only path when --batch is 1).
                const job = group[0];
                job.setRunning();
                if (self.backend.generate(&job.req, job.sink)) |res| {
                    job.finish(self.io, res, null);
                } else |err| {
                    job.finish(self.io, undefined, if (job.isCancelled()) error.Cancelled else err);
                }
            } else if (live > 1) {
                for (group[0..live], 0..) |job, k| {
                    job.setRunning();
                    reqs[k] = &job.req;
                    sinks[k] = job.sink;
                    errs[k] = null;
                }
                var batch_err: ?anyerror = null;
                self.backend.generateBatch(reqs[0..live], sinks[0..live], results[0..live], errs[0..live]) catch |err| {
                    batch_err = err;
                };
                for (group[0..live], 0..) |job, k| {
                    const err: ?anyerror = if (errs[k]) |e| e else batch_err;
                    if (err) |e| {
                        job.finish(self.io, undefined, if (job.isCancelled()) error.Cancelled else e);
                    } else {
                        job.finish(self.io, results[k], null);
                    }
                }
            }

            lock(&self.mutex);
            self.running.clearRetainingCapacity();
            unlock(&self.mutex);
        }
    }
};

test "scheduler: submit/run/finish, queue bound, shutdown drain" {
    const FakeBackend = struct {
        fn validate(_: *anyopaque, _: *const types.GenerateRequest) anyerror!void {}
        fn generate(_: *anyopaque, req: *const types.GenerateRequest, sink: *std.Io.Writer) anyerror!types.GenerateResult {
            try sink.writeAll("ok");
            try sink.flush();
            return .{ .prompt_tokens = req.max_tokens, .completion_tokens = 1, .finish = .stop };
        }
    };
    var threaded: std.Io.Threaded = .init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var dummy: u8 = 0;
    const backend = types.Backend{
        .ptr = @ptrCast(&dummy),
        .vtable = &.{ .validate = FakeBackend.validate, .generate = FakeBackend.generate },
        .info = .{ .model_id = "fake", .context_len = 128 },
    };

    var sched = Scheduler.init(std.testing.allocator, io, backend, 2, 1);
    try sched.start();

    var out_buf: [64]u8 = undefined;
    var out = std.Io.Writer.fixed(&out_buf);
    var job = Job{
        .req = .{ .messages = &.{}, .sampling = .{}, .max_tokens = 7 },
        .sink = &out,
    };
    try sched.submit(&job);
    while (!job.waitTimed(io, std.time.ns_per_ms)) {}
    try std.testing.expectEqual(@as(?anyerror, null), job.err);
    try std.testing.expectEqual(@as(usize, 7), job.res.prompt_tokens);
    try std.testing.expectEqualStrings("ok", out.buffered());

    sched.stop();
    var late = Job{ .req = job.req, .sink = &out };
    try std.testing.expectError(error.ShuttingDown, sched.submit(&late));
}

test "scheduler: batch groups queued jobs, isolates per-job errors, honors --batch 1" {
    const Recorder = struct {
        sizes: [8]usize = undefined,
        n: usize = 0,

        fn validate(_: *anyopaque, _: *const types.GenerateRequest) anyerror!void {}
        fn generate(ptr: *anyopaque, req: *const types.GenerateRequest, sink: *std.Io.Writer) anyerror!types.GenerateResult {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            self.sizes[self.n] = 1;
            self.n += 1;
            try sink.writeAll("solo");
            try sink.flush();
            return .{ .prompt_tokens = req.max_tokens, .completion_tokens = 1, .finish = .stop };
        }
        fn generateBatch(
            ptr: *anyopaque,
            reqs: []const *const types.GenerateRequest,
            sinks: []const *std.Io.Writer,
            results: []types.GenerateResult,
            errs: []?anyerror,
        ) anyerror!void {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            self.sizes[self.n] = reqs.len;
            self.n += 1;
            for (reqs, sinks, results, errs) |req, sink, *res, *e| {
                // max_tokens == 99 marks the request that must fail alone.
                if (req.max_tokens == 99) {
                    e.* = error.PromptTooLong;
                    continue;
                }
                try sink.writeAll("batch");
                try sink.flush();
                res.* = .{ .prompt_tokens = req.max_tokens, .completion_tokens = 2, .finish = .stop };
            }
        }
    };
    var threaded: std.Io.Threaded = .init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var rec = Recorder{};
    const backend = types.Backend{
        .ptr = @ptrCast(&rec),
        .vtable = &.{ .validate = Recorder.validate, .generate = Recorder.generate, .generate_batch = Recorder.generateBatch },
        .info = .{ .model_id = "fake", .context_len = 128 },
    };

    // Queue three jobs BEFORE the worker starts: with --batch 2 they must
    // run as a pair (one failing in isolation) plus a solo.
    var sched = Scheduler.init(std.testing.allocator, io, backend, 8, 2);
    var bufs: [3][64]u8 = undefined;
    var outs: [3]std.Io.Writer = .{
        std.Io.Writer.fixed(&bufs[0]),
        std.Io.Writer.fixed(&bufs[1]),
        std.Io.Writer.fixed(&bufs[2]),
    };
    var jobs: [3]Job = .{
        .{ .req = .{ .messages = &.{}, .sampling = .{}, .max_tokens = 5 }, .sink = &outs[0] },
        .{ .req = .{ .messages = &.{}, .sampling = .{}, .max_tokens = 99 }, .sink = &outs[1] },
        .{ .req = .{ .messages = &.{}, .sampling = .{}, .max_tokens = 6 }, .sink = &outs[2] },
    };
    for (&jobs) |*job| try sched.submit(job);
    try sched.start();
    for (&jobs) |*job| {
        while (!job.waitTimed(io, std.time.ns_per_ms)) {}
    }
    sched.stop();

    try std.testing.expectEqual(@as(usize, 2), rec.n);
    try std.testing.expectEqual(@as(usize, 2), rec.sizes[0]);
    try std.testing.expectEqual(@as(usize, 1), rec.sizes[1]);
    try std.testing.expectEqual(@as(?anyerror, null), jobs[0].err);
    try std.testing.expectEqualStrings("batch", outs[0].buffered());
    try std.testing.expectEqual(@as(usize, 2), jobs[0].res.completion_tokens);
    try std.testing.expectEqual(@as(?anyerror, error.PromptTooLong), jobs[1].err);
    try std.testing.expectEqual(@as(?anyerror, null), jobs[2].err);
    try std.testing.expectEqualStrings("solo", outs[2].buffered());
}
