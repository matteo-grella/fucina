//! Behavioral tests for the threading module (`thread.zig`): the reusable
//! one-shot worker, the persistent fork-join team behind `parallelChunks`
//! (chunk-once correctness, nested-call serialization, parked-worker wakeups),
//! and the chained-dispatch path (`parallelChained`).
const std = @import("std");
const thread = @import("thread.zig");

const OneShotWorker = thread.OneShotWorker;
const Pool = thread.Pool;
const Chain = thread.Chain;

test "one-shot worker runs reusable jobs" {
    var worker: OneShotWorker = undefined;
    try worker.init();
    defer worker.deinit();

    const Job = struct {
        fn run(ptr: *anyopaque) void {
            const value: *usize = @ptrCast(@alignCast(ptr));
            value.* += 1;
        }
    };

    var value: usize = 0;
    try std.testing.expect(worker.start(Job.run, &value));
    worker.wait();
    try std.testing.expectEqual(@as(usize, 1), value);

    try std.testing.expect(worker.start(Job.run, &value));
    worker.wait();
    try std.testing.expectEqual(@as(usize, 2), value);
}

test "parallelChunks runs every chunk exactly once across many dispatches" {
    var pool: Pool = undefined;
    try pool.init(.{ .allocator = std.testing.allocator, .max_workers = 4 });
    defer pool.deinit();

    const Task = struct { out: []u32, value: u32 };
    const Run = struct {
        fn run(task: *const Task) void {
            for (task.out) |*v| v.* +%= task.value;
        }
    }.run;

    // Repeat to exercise the generation/barrier handshake and the spin/park
    // boundary many times, with varying chunk counts including 1 and > workers.
    var counters = [_]u32{0} ** 8;
    const rounds: u32 = 3000;
    var round: u32 = 0;
    while (round < rounds) : (round += 1) {
        const n = (round % 8) + 1; // 1..8 chunks
        var tasks: [8]Task = undefined;
        for (0..n) |i| tasks[i] = .{ .out = counters[i .. i + 1], .value = 1 };
        pool.parallelChunks(Task, tasks[0..n], Run);
    }

    var expected = [_]u32{0} ** 8;
    round = 0;
    while (round < rounds) : (round += 1) {
        const n = (round % 8) + 1;
        for (0..n) |i| expected[i] += 1;
    }
    try std.testing.expectEqualSlices(u32, &expected, &counters);
}

test "parallelChunks runs nested calls serially instead of reentering barrier" {
    var pool: Pool = undefined;
    try pool.init(.{ .allocator = std.testing.allocator, .max_workers = 4 });
    defer pool.deinit();

    var counters: [12]std.atomic.Value(u32) = undefined;
    for (&counters) |*counter| counter.* = .init(0);

    const InnerTask = struct {
        counter: *std.atomic.Value(u32),
    };
    const RunInner = struct {
        fn run(task: *const InnerTask) void {
            _ = task.counter.fetchAdd(1, .monotonic);
        }
    }.run;

    const OuterTask = struct {
        pool: *Pool,
        counters: []std.atomic.Value(u32),
        start: usize,
    };
    const RunOuter = struct {
        fn run(task: *const OuterTask) void {
            var inner_tasks: [3]InnerTask = undefined;
            for (&inner_tasks, 0..) |*inner, i| {
                inner.* = .{ .counter = &task.counters[task.start + i] };
            }
            task.pool.parallelChunks(InnerTask, &inner_tasks, RunInner);
        }
    }.run;

    var outer_tasks: [4]OuterTask = undefined;
    for (&outer_tasks, 0..) |*task, i| {
        task.* = .{
            .pool = &pool,
            .counters = &counters,
            .start = i * 3,
        };
    }

    pool.parallelChunks(OuterTask, &outer_tasks, RunOuter);

    for (&counters) |*counter| {
        try std.testing.expectEqual(@as(u32, 1), counter.load(.monotonic));
    }
}

test "parallelChunks wakes parked workers across idle gaps" {
    var pool: Pool = undefined;
    try pool.init(.{ .allocator = std.testing.allocator, .max_workers = 4 });
    defer pool.deinit();

    const Task = struct { out: *u32 };
    const Run = struct {
        fn run(task: *const Task) void {
            task.out.* +%= 1;
        }
    }.run;

    // Idle long enough between dispatches that workers exhaust the spin budget
    // and park, exercising the parked-counter wake path (and the skip path on
    // the dispatches that follow immediately after a wake).
    var counters = [_]u32{0} ** 8;
    var round: u32 = 0;
    while (round < 20) : (round += 1) {
        var tasks: [8]Task = undefined;
        for (0..8) |i| tasks[i] = .{ .out = &counters[i] };
        pool.parallelChunks(Task, tasks[0..8], Run);
        pool.parallelChunks(Task, tasks[0..8], Run);
        // Idle past the spin budget so workers park (timekeeping goes through
        // Io in this std; a coarse busy-wait keeps the test self-contained).
        const start_ns = std.Io.Clock.awake.now(pool.io).nanoseconds;
        while (std.Io.Clock.awake.now(pool.io).nanoseconds - start_ns < 20 * std.time.ns_per_ms) {
            std.atomic.spinLoopHint();
        }
    }

    for (counters) |c| try std.testing.expectEqual(@as(u32, 40), c);
}

test "parallelChained runs successors without a phase barrier" {
    var pool: Pool = undefined;
    try pool.init(.{ .allocator = std.testing.allocator, .max_workers = 4 });
    defer pool.deinit();

    const Task = struct {
        value: *std.atomic.Value(u32),
        successor: ?usize,
    };
    const Run = struct {
        fn run(task: *Task, chain: *const Chain) void {
            _ = task.value.fetchAdd(1, .monotonic);
            if (task.successor) |next| chain.enqueue(next);
        }
    }.run;

    var counters: [4]std.atomic.Value(u32) = undefined;
    for (&counters) |*counter| counter.* = .init(0);
    var tasks = [_]Task{
        .{ .value = &counters[0], .successor = 2 },
        .{ .value = &counters[1], .successor = 3 },
        .{ .value = &counters[2], .successor = null },
        .{ .value = &counters[3], .successor = null },
    };

    try std.testing.expect(pool.parallelChained(Task, &tasks, 2, Run));
    for (&counters) |*counter| {
        try std.testing.expectEqual(@as(u32, 1), counter.load(.monotonic));
    }
}

test "parallelChained rejects empty seeds and out-of-range initial counts" {
    var pool: Pool = undefined;
    try pool.init(.{ .allocator = std.testing.allocator, .max_workers = 4 });
    defer pool.deinit();

    const Task = struct { value: *u32 };
    const Run = struct {
        fn run(task: *Task, chain: *const Chain) void {
            _ = chain;
            task.value.* += 1;
        }
    }.run;

    var value: u32 = 0;
    var tasks = [_]Task{ .{ .value = &value }, .{ .value = &value } };

    try std.testing.expect(pool.parallelChained(Task, tasks[0..0], 0, Run));
    try std.testing.expect(!pool.parallelChained(Task, &tasks, 0, Run));
    try std.testing.expect(!pool.parallelChained(Task, &tasks, 3, Run));
    try std.testing.expectEqual(@as(u32, 0), value);
}

test "parallelChained covers every index exactly once across repeated dispatches" {
    var pool: Pool = undefined;
    try pool.init(.{ .allocator = std.testing.allocator, .max_workers = 4 });
    defer pool.deinit();

    // Two seeds, each task enqueues index + 2: the even and odd chains together
    // cover every index exactly once, exercising the enqueue-contract
    // accounting (and its per-dispatch reset) across many rounds. In safety
    // builds a duplicate enqueue or a count mismatch panics inside the run.
    const n = 16;
    const Task = struct {
        counters: *[n]std.atomic.Value(u32),
        index: usize,
    };
    const Run = struct {
        fn run(task: *Task, chain: *const Chain) void {
            _ = task.counters[task.index].fetchAdd(1, .monotonic);
            if (task.index + 2 < n) chain.enqueue(task.index + 2);
        }
    }.run;

    var counters: [n]std.atomic.Value(u32) = undefined;
    for (&counters) |*counter| counter.* = .init(0);
    var tasks: [n]Task = undefined;
    for (&tasks, 0..) |*task, i| task.* = .{ .counters = &counters, .index = i };

    const rounds: u32 = 50;
    var round: u32 = 0;
    while (round < rounds) : (round += 1) {
        try std.testing.expect(pool.parallelChained(Task, &tasks, 2, Run));
    }
    for (&counters) |*counter| {
        try std.testing.expectEqual(rounds, counter.load(.monotonic));
    }
}
