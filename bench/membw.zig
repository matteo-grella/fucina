// Measured DRAM read-bandwidth probe — the roofline denominator for
// weight-streaming benches. Decode-shaped quantized matmuls are bound by how
// fast weight bytes stream from memory, so an achieved "w GB/s" figure is
// only interpretable against the bandwidth this machine can actually
// sustain; this probe measures that ceiling instead of trusting spec sheets.
//
// Method: one shared region far larger than any cache level is split into
// 8 MiB chunks that participants claim from a shared atomic cursor, each
// chunk streamed once per round as four interleaved quarter streams — four
// cache lines in flight per iteration on independent wrapping @Vector(8,
// u64) accumulators, so line fills overlap instead of serializing on a
// scalar chain. Dynamic claiming is
// load-bearing on heterogeneous parts: with fixed per-thread slices and a
// join-all, the slowest core sets the wall time and the aggregate under-reads
// (~2x on M1 Max, where two E-cores pace eight P-cores); with claiming, fast
// cores simply stream more chunks and the straggler tail is bounded by one
// chunk. A round is wall-clocked from before-spawn to after-join and covers
// exactly one pass over the region (spawn/join overhead biases slightly
// conservative); round 0 is an untimed warmup that also pages the region in,
// and the best timed round is the ceiling. GB/s is decimal (bytes/ns),
// matching the bench tables. Every participant is a spawned thread that
// pins itself to performance-core QoS — the caller only spawns and joins,
// so the probe leaves the calling thread's scheduling state untouched.
//
// The loop shape is the verified pattern of record: line-touch-only reads,
// a single sequential stream, and software prefetch at 4-128-line distances
// all measure equal or worse on M1 Max (prefetch regresses hard past 32
// lines) — do not "optimize" the pass without beating it on measurements.
//
// probe() measures exactly what it is told (threads, rounds, region MiB);
// sizing policy — core count, the 4 GiB footprint cap — lives in main().
// The single-thread ceiling is the reference figure for single-threaded
// kernel benches (bench/ternary.zig); the all-core ceiling is the
// denominator for whole-engine decode rates.
//
//   zig build bench-membw -Doptimize=ReleaseFast -- [--threads N] [--rounds N] [--mb N]

const std = @import("std");
const builtin = @import("builtin");
const Timer = @import("timer.zig").Timer;

// Same macOS performance-core pinning as the engine's worker team
// (src/thread.zig): at default QoS macOS parks streaming threads on
// E-cores/low clocks and the aggregate round under-reads by ~2x.
const qos_class_user_interactive: c_uint = 0x21;
extern "c" fn pthread_set_qos_class_self_np(qos_class: c_uint, relative_priority: c_int) c_int;

fn pinToPerformanceCores() void {
    if (comptime builtin.os.tag == .macos) {
        _ = pthread_set_qos_class_self_np(qos_class_user_interactive, 0);
    }
}

pub const Result = struct {
    gbps: f64, // best-of-rounds aggregate read bandwidth, decimal GB/s
    threads: usize,
    region_mib: usize,
    rounds: usize,
};

pub const default_rounds = 5;
pub const default_single_region_mib = 512;
pub const auto_total_cap_mib = 4096;

const chunk_bytes = 8 * 1024 * 1024;
const chunk_words = chunk_bytes / @sizeOf(u64);
const page_words = 4096 / @sizeOf(u64);

const Job = struct {
    region: []const u64,
    next_chunk: *std.atomic.Value(usize),
    n_chunks: usize,
    sink: u64 = 0,

    fn run(self: *Job) void {
        pinToPerformanceCores();
        var sink: u64 = 0;
        while (true) {
            const c = self.next_chunk.fetchAdd(1, .monotonic);
            if (c >= self.n_chunks) break;
            sink +%= readPass(self.region[c * chunk_words ..][0..chunk_words]);
        }
        self.sink = sink;
    }
};

/// One streaming pass over a chunk, read as four interleaved quarter
/// streams — four cache lines in flight per iteration, each on its own
/// @Vector(8, u64) accumulator. A single sequential stream is paced by one
/// hardware-prefetch stream and under-reads the ceiling; independent streams
/// multiply the line fills in flight. Returns the folded sum so the loads
/// are observable; the probe feeds it to std.mem.doNotOptimizeAway.
fn readPass(buf: []const u64) u64 {
    const V = @Vector(8, u64);
    const n_streams = 4;
    const quarter = (buf.len / n_streams) & ~@as(usize, 7);
    var acc: [n_streams]V = @splat(@splat(0));
    var i: usize = 0;
    while (i < quarter) : (i += 8) {
        inline for (0..n_streams) |s| {
            const chunk: V = buf[s * quarter + i ..][0..8].*;
            acc[s] = acc[s] +% chunk;
        }
    }
    var sum: u64 = 0;
    inline for (0..n_streams) |s| {
        inline for (0..8) |lane| sum +%= acc[s][lane];
    }
    // Tail beyond n_streams * quarter (never taken for 8 MiB chunks; keeps
    // the function honest for arbitrary slices).
    var t = n_streams * quarter;
    while (t < buf.len) : (t += 1) sum +%= buf[t];
    return sum;
}

pub fn probe(
    allocator: std.mem.Allocator,
    io: std.Io,
    threads: usize,
    rounds: usize,
    region_mib: usize,
) !Result {
    // Real runtime checks, not asserts: ReleaseFast strips asserts and a
    // zero round count would loop forever below.
    if (threads < 1 or rounds < 1) return error.InvalidProbeArgs;
    // Threads beyond the chunk count find the cursor exhausted and exit —
    // no region floor, so the caller's footprint budget is never exceeded.
    const n_chunks = @max(1, region_mib * (1024 * 1024) / chunk_bytes);
    const region = try allocator.alloc(u64, n_chunks * chunk_words);
    defer allocator.free(region);
    // Write-touch one word per page: reads of never-written anonymous pages
    // can resolve to a shared zero page and measure nothing.
    var w: usize = 0;
    while (w < region.len) : (w += page_words) region[w] = w | 1;

    var next_chunk = std.atomic.Value(usize).init(0);
    const jobs = try allocator.alloc(Job, threads);
    defer allocator.free(jobs);
    const spawned = try allocator.alloc(std.Thread, threads);
    defer allocator.free(spawned);

    const total_bytes: f64 = @floatFromInt(region.len * @sizeOf(u64));
    var best_gbps: f64 = 0;
    var sink: u64 = 0;
    var round: usize = 0;
    while (round < rounds + 1) : (round += 1) { // round 0 = untimed warmup/page-in
        for (jobs) |*job| job.* = .{ .region = region, .next_chunk = &next_chunk, .n_chunks = n_chunks };
        next_chunk.store(0, .monotonic);
        var t = try Timer.start(io);
        for (spawned, jobs, 0..) |*handle, *job, idx| {
            handle.* = std.Thread.spawn(.{}, Job.run, .{job}) catch |err| {
                // Live workers hold pointers into this frame and the region:
                // exhaust the cursor so they drain, join them, then fail.
                next_chunk.store(n_chunks, .monotonic);
                for (spawned[0..idx]) |h| h.join();
                return err;
            };
        }
        for (spawned) |handle| handle.join();
        const ns = t.read();
        for (jobs) |job| sink +%= job.sink;
        if (round == 0 or ns == 0) continue;
        const gbps = total_bytes / @as(f64, @floatFromInt(ns));
        if (gbps > best_gbps) best_gbps = gbps;
    }
    std.mem.doNotOptimizeAway(sink);

    return .{
        .gbps = best_gbps,
        .threads = threads,
        .region_mib = (n_chunks * chunk_bytes) / (1024 * 1024),
        .rounds = rounds,
    };
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = std.heap.page_allocator;
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    // CLI values are clamped to sane ranges: probe() runs forever on zero
    // rounds and the size math must not overflow on pasted garbage.
    var threads_arg: ?usize = null;
    var mib_arg: ?usize = null;
    var rounds: usize = default_rounds;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--threads") and i + 1 < args.len) {
            i += 1;
            threads_arg = std.math.clamp(try std.fmt.parseInt(usize, args[i], 10), 1, 4096);
        } else if (std.mem.eql(u8, args[i], "--mb") and i + 1 < args.len) {
            i += 1;
            mib_arg = std.math.clamp(try std.fmt.parseInt(usize, args[i], 10), 1, 1024 * 1024);
        } else if (std.mem.eql(u8, args[i], "--rounds") and i + 1 < args.len) {
            i += 1;
            rounds = std.math.clamp(try std.fmt.parseInt(usize, args[i], 10), 1, 1000);
        }
    }

    const all_threads = threads_arg orelse @max(1, std.Thread.getCpuCount() catch 1);
    const single_mib = mib_arg orelse default_single_region_mib;
    // All-core region: enough that every core streams several chunks per
    // round, capped so the probe never allocates more than ~4 GiB.
    const all_mib = mib_arg orelse @min(auto_total_cap_mib, @max(1024, all_threads * 256));

    var buf: [1024]u8 = undefined;
    var sw = std.Io.File.stdout().writer(io, &buf);
    const out = &sw.interface;
    defer out.flush() catch {};

    try out.print("DRAM read-bandwidth probe  rounds={d} best-of, one region pass per round\n", .{rounds});

    const single = try probe(allocator, io, 1, rounds, single_mib);
    try out.print("  single-thread : {d:>7.1} GB/s  ({d} MiB region)\n", .{ single.gbps, single.region_mib });

    if (all_threads > 1) {
        const all = try probe(allocator, io, all_threads, rounds, all_mib);
        try out.print("  {d:>2} threads    : {d:>7.1} GB/s  ({d} MiB region, 8 MiB chunk claiming)\n", .{ all.threads, all.gbps, all.region_mib });
    }
}
