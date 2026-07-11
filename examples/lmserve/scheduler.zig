//! Bounded FIFO queue + the single inference worker. Connection threads
//! accept and validate concurrently, but generation is strictly sequential:
//! one worker owns the backend (and through it the one ExecContext) — the
//! engine's intended shape, since a single forward pass already fork-joins
//! across every performance core and ExecContext is single-threaded by
//! contract (docs/REFERENCE.md, Threading).

const std = @import("std");
const types = @import("types.zig");

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
/// connection thread afterwards.
pub const Job = struct {
    req: types.GenerateRequest,
    /// The OpenAI layer's per-request emitter (SSE frames or an
    /// accumulating body). Written by the worker only.
    sink: *std.Io.Writer,

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
    shutting_down: bool = false,
    /// The job the worker is generating right now (shutdown cancels it).
    current: ?*Job = null,
    thread: ?std.Thread = null,

    pub fn init(allocator: std.mem.Allocator, io: std.Io, backend: types.Backend, queue_capacity: usize) Scheduler {
        return .{ .allocator = allocator, .io = io, .backend = backend, .capacity = @max(queue_capacity, 1) };
    }

    pub fn start(self: *Scheduler) !void {
        self.thread = try std.Thread.spawn(.{}, workerLoop, .{self});
    }

    /// Stop accepting, cancel the in-flight job, finish queued jobs with
    /// `error.ShuttingDown`, and join the worker.
    pub fn stop(self: *Scheduler) void {
        {
            lock(&self.mutex);
            defer unlock(&self.mutex);
            self.shutting_down = true;
            if (self.current) |job| job.cancel();
            self.cond.broadcast(self.io);
        }
        if (self.thread) |t| {
            t.join();
            self.thread = null;
        }
        self.queue.deinit(self.allocator);
    }

    /// Number of requests waiting or running (the /health load signal).
    pub fn depth(self: *Scheduler) usize {
        lock(&self.mutex);
        defer unlock(&self.mutex);
        return self.queue.items.len + @intFromBool(self.current != null);
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
        while (true) {
            lock(&self.mutex);
            while (self.queue.items.len == 0 and !self.shutting_down)
                self.cond.waitUncancelable(self.io, &self.mutex);
            if (self.queue.items.len == 0) {
                // Shutting down with a drained queue.
                unlock(&self.mutex);
                return;
            }
            const job = self.queue.orderedRemove(0);
            const draining = self.shutting_down;
            if (!draining) self.current = job;
            unlock(&self.mutex);

            if (draining) {
                job.finish(self.io, undefined, error.ShuttingDown);
                continue;
            }
            if (job.isCancelled()) {
                job.finish(self.io, undefined, error.Cancelled);
            } else {
                job.setRunning();
                if (self.backend.generate(&job.req, job.sink)) |res| {
                    job.finish(self.io, res, null);
                } else |err| {
                    job.finish(self.io, undefined, if (job.isCancelled()) error.Cancelled else err);
                }
            }

            lock(&self.mutex);
            self.current = null;
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

    var sched = Scheduler.init(std.testing.allocator, io, backend, 2);
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
