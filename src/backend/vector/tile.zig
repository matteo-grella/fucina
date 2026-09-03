//! The one range splitter behind the vector kernels' parallel dispatch:
//! a payload-generic task (`{ ctx, start, end }`) with the proportional
//! boundary the per-kernel splitters all used — `boundary(i) = i * total /
//! thread_count` — so converting a hand-rolled Task/runParallel pair onto
//! `forRange`/`reduceRange` moves NO split point and reorders NO chunk
//! (bitwise-neutral by construction). Gate decisions (which thread-count
//! table, which work threshold) stay with the callers: this file only
//! splits and spawns. `matmul_quant.zig`'s `QuantizedRhsParallel` keeps
//! its richer grouped/gated policy — this is its plain-range sibling.
const parallel = @import("../../parallel.zig");
const thread = @import("../../thread.zig");

/// The one proportional split boundary: part `i` of `parts` over
/// `[0, total)` starts at `i * total / parts`. Every range split in the
/// tree (this file, `ExecContext.forRange`) draws its boundaries
/// here, so a range moves from one splitter to another with no chunk
/// reordered.
pub fn bound(i: usize, total: usize, parts: usize) usize {
    return i * total / parts;
}

fn RangeTask(comptime Ctx: type) type {
    return struct {
        ctx: Ctx,
        start: usize,
        end: usize,
    };
}

/// Split `[0, total)` into `thread_count` proportional chunks and run
/// `runFn(ctx, start, end)` for each on the pool (the calling thread
/// takes part via `parallelChunks`). The caller has already decided
/// `thread_count > 1` through its own measured gate.
pub fn forRange(
    pool: *thread.Pool,
    comptime Ctx: type,
    ctx: Ctx,
    total: usize,
    thread_count: usize,
    comptime runFn: fn (Ctx, usize, usize) void,
) void {
    forRangeOpts(pool, Ctx, ctx, total, thread_count, runFn, .{});
}

/// `forRange` with per-dispatch options (`Pool.DispatchOptions`).
pub fn forRangeOpts(
    pool: *thread.Pool,
    comptime Ctx: type,
    ctx: Ctx,
    total: usize,
    thread_count: usize,
    comptime runFn: fn (Ctx, usize, usize) void,
    options: thread.Pool.DispatchOptions,
) void {
    const Task = RangeTask(Ctx);
    const runner = struct {
        fn run(task: *const Task) void {
            runFn(task.ctx, task.start, task.end);
        }
    };
    var tasks: [parallel.vector_max_threads]Task = undefined;
    for (0..thread_count) |ti| {
        tasks[ti] = .{
            .ctx = ctx,
            .start = bound(ti, total, thread_count),
            .end = bound(ti + 1, total, thread_count),
        };
    }
    pool.parallelChunksOpts(Task, tasks[0..thread_count], runner.run, options);
}

/// The reduction form: each chunk computes `chunkFn(ctx, start, end)` into
/// its own partial slot, and the partials are folded serially in task
/// order from `identity` — exactly the historical per-kernel
/// partials-array walk, so the reduction tree (and therefore the bits) is
/// unchanged.
pub fn reduceRange(
    pool: *thread.Pool,
    comptime Acc: type,
    comptime Ctx: type,
    ctx: Ctx,
    total: usize,
    thread_count: usize,
    identity: Acc,
    comptime chunkFn: fn (Ctx, usize, usize) Acc,
    comptime combineFn: fn (Acc, Acc) Acc,
) Acc {
    const Slot = struct {
        ctx: Ctx,
        partial: *Acc,
        start: usize,
        end: usize,
    };
    const runner = struct {
        fn run(task: *const Slot) void {
            task.partial.* = chunkFn(task.ctx, task.start, task.end);
        }
    };
    var partials: [parallel.vector_max_threads]Acc = undefined;
    var tasks: [parallel.vector_max_threads]Slot = undefined;
    for (0..thread_count) |ti| {
        partials[ti] = identity;
        tasks[ti] = .{
            .ctx = ctx,
            .partial = &partials[ti],
            .start = bound(ti, total, thread_count),
            .end = bound(ti + 1, total, thread_count),
        };
    }
    pool.parallelChunks(Slot, tasks[0..thread_count], runner.run);

    var acc = identity;
    for (partials[0..thread_count]) |value| acc = combineFn(acc, value);
    return acc;
}
