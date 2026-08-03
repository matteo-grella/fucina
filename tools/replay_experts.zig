//! Offline expert-cache replay: run a recorded routing trace (the
//! `--moe-trace=` sidecar, `ExpertStore.saveTrace` format) through three
//! cache policies across a sweep of per-layer capacities, without touching
//! the model or the disk. Answers the questions a hardware ladder cannot
//! answer cheaply: would 2x the cache help? Is the POLICY or the CAPACITY
//! the bottleneck (LRU flat where Belady climbs = policy)? Would a pinned
//! tier beat pure LRU on this router (the auto-pin flatness guard's
//! question, answered exactly)?
//!
//!   zig build replay-experts -- <trace.bin> [caps...]
//!
//! Per capacity C (slots per layer), per layer, over the trace's request
//! sequence:
//!   lru     evict the least-recently-used resident expert
//!   belady  evict the resident expert whose next use is farthest away
//!           (optimal offline upper bound — unreachable, but it separates
//!           "bad policy" from "not enough capacity")
//!   pin+lru pin the C/2 most-used experts (whole-trace histogram — what
//!           auto-pin would learn), LRU over the remaining C/2 slots
//!
//! Belady needs future knowledge; LRU and pin+lru need temporal order.
//! The usage HISTOGRAM the store persists (`<gguf>.experts`) carries
//! neither, which is why the trace exists.

const std = @import("std");

const trace_magic = "FUCTRCE1";

const Request = struct { layer: u32, eid: u32 };

fn fail(comptime fmt: []const u8, args: anytype) noreturn {
    std.debug.print("replay_experts: " ++ fmt ++ "\n", args);
    std.process.exit(1);
}

const Trace = struct {
    n_layers: u32,
    requests: []Request,
    n_expert: u32, // max eid + 1 over the trace

    fn load(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !Trace {
        const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(1 << 32)) catch fail("cannot read {s}", .{path});
        defer allocator.free(bytes);
        if (bytes.len < trace_magic.len + 12 or !std.mem.eql(u8, bytes[0..trace_magic.len], trace_magic))
            fail("{s} is not a FUCTRCE1 trace", .{path});
        var at: usize = trace_magic.len;
        const n_layers = std.mem.readInt(u32, bytes[at..][0..4], .little);
        at += 4;
        const n_pairs: usize = @intCast(std.mem.readInt(u64, bytes[at..][0..8], .little));
        at += 8;
        if (bytes.len != at + n_pairs * 8) fail("torn trace: {d} pairs declared, {d} bytes of payload", .{ n_pairs, bytes.len - at });

        const requests = try allocator.alloc(Request, n_pairs);
        var n_expert: u32 = 0;
        for (requests) |*r| {
            r.layer = std.mem.readInt(u32, bytes[at..][0..4], .little);
            r.eid = std.mem.readInt(u32, bytes[at + 4 ..][0..4], .little);
            at += 8;
            if (r.layer >= n_layers) fail("layer {d} out of range", .{r.layer});
            n_expert = @max(n_expert, r.eid + 1);
        }
        return .{ .n_layers = n_layers, .requests = requests, .n_expert = n_expert };
    }
};

/// One layer's request sequence as indices into the trace, plus per-request
/// next-use links for Belady (next position in THIS layer's sequence where
/// the same eid recurs; maxInt = never again).
const LayerSeq = struct {
    eids: []u32,
    next_use: []usize,
    counts: []u64, // whole-sequence histogram, for the pinned policy

    fn build(allocator: std.mem.Allocator, trace: *const Trace, layer: u32) !LayerSeq {
        var n: usize = 0;
        for (trace.requests) |r| {
            if (r.layer == layer) n += 1;
        }
        const eids = try allocator.alloc(u32, n);
        const next_use = try allocator.alloc(usize, n);
        const counts = try allocator.alloc(u64, trace.n_expert);
        @memset(counts, 0);
        var i: usize = 0;
        for (trace.requests) |r| {
            if (r.layer != layer) continue;
            eids[i] = r.eid;
            counts[r.eid] += 1;
            i += 1;
        }
        // Backward scan: last_seen[eid] = the next occurrence when walking
        // forward.
        const last_seen = try allocator.alloc(usize, trace.n_expert);
        defer allocator.free(last_seen);
        @memset(last_seen, std.math.maxInt(usize));
        i = n;
        while (i > 0) {
            i -= 1;
            next_use[i] = last_seen[eids[i]];
            last_seen[eids[i]] = i;
        }
        return .{ .eids = eids, .next_use = next_use, .counts = counts };
    }

    fn deinit(self: *LayerSeq, allocator: std.mem.Allocator) void {
        allocator.free(self.eids);
        allocator.free(self.next_use);
        allocator.free(self.counts);
    }
};

const Tally = struct { hits: u64 = 0, total: u64 = 0 };

/// LRU over `cap` slots; experts in `pinned` always hit and occupy no slot
/// (the caller charges their budget separately).
fn runLru(allocator: std.mem.Allocator, seq: *const LayerSeq, n_expert: u32, cap: usize, pinned: ?[]const bool, tally: *Tally) !void {
    const last_used = try allocator.alloc(u64, n_expert);
    defer allocator.free(last_used);
    const resident = try allocator.alloc(bool, n_expert);
    defer allocator.free(resident);
    @memset(resident, false);
    var n_resident: usize = 0;
    var clock: u64 = 0;

    for (seq.eids) |eid| {
        clock += 1;
        tally.total += 1;
        if (pinned) |p| {
            if (p[eid]) {
                tally.hits += 1;
                continue;
            }
        }
        if (resident[eid]) {
            tally.hits += 1;
            last_used[eid] = clock;
            continue;
        }
        if (n_resident == cap) {
            var victim: usize = 0;
            var oldest: u64 = std.math.maxInt(u64);
            for (resident, 0..) |res, e| {
                if (res and last_used[e] < oldest) {
                    oldest = last_used[e];
                    victim = e;
                }
            }
            resident[victim] = false;
            n_resident -= 1;
        }
        resident[eid] = true;
        n_resident += 1;
        last_used[eid] = clock;
    }
}

/// Belady's optimal replacement: evict the resident expert whose next use
/// lies farthest in the future (or never).
fn runBelady(allocator: std.mem.Allocator, seq: *const LayerSeq, n_expert: u32, cap: usize, tally: *Tally) !void {
    const next_of = try allocator.alloc(usize, n_expert);
    defer allocator.free(next_of);
    const resident = try allocator.alloc(bool, n_expert);
    defer allocator.free(resident);
    @memset(resident, false);
    var n_resident: usize = 0;

    for (seq.eids, seq.next_use, 0..) |eid, next, i| {
        _ = i;
        tally.total += 1;
        if (resident[eid]) {
            tally.hits += 1;
            next_of[eid] = next;
            continue;
        }
        if (n_resident == cap) {
            var victim: usize = 0;
            var farthest: usize = 0;
            for (resident, 0..) |res, e| {
                if (res and next_of[e] >= farthest) {
                    farthest = next_of[e];
                    victim = e;
                }
            }
            resident[victim] = false;
            n_resident -= 1;
        }
        resident[eid] = true;
        n_resident += 1;
        next_of[eid] = next;
    }
}

/// Top-`n_pin` experts by whole-sequence count (auto-pin's knowledge).
fn pickPins(allocator: std.mem.Allocator, counts: []const u64, n_pin: usize) ![]bool {
    const pinned = try allocator.alloc(bool, counts.len);
    @memset(pinned, false);
    const order = try allocator.alloc(u32, counts.len);
    defer allocator.free(order);
    for (order, 0..) |*o, e| o.* = @intCast(e);
    std.mem.sort(u32, order, counts, struct {
        fn hotter(c: []const u64, a: u32, b: u32) bool {
            if (c[a] != c[b]) return c[a] > c[b];
            return a < b;
        }
    }.hotter);
    var taken: usize = 0;
    for (order) |e| {
        if (taken == n_pin or counts[e] == 0) break;
        pinned[e] = true;
        taken += 1;
    }
    return pinned;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const io = init.io;

    const args = try init.minimal.args.toSlice(allocator);
    if (args.len < 2) fail("usage: replay-experts <trace.bin> [slots-per-layer...]", .{});

    var trace = try Trace.load(allocator, io, args[1]);
    defer allocator.free(trace.requests);

    // Capacity sweep: explicit caps, or powers of two up to the pool size.
    var caps: std.ArrayList(usize) = .empty;
    defer caps.deinit(allocator);
    if (args.len > 2) {
        for (args[2..]) |arg| try caps.append(allocator, try std.fmt.parseInt(usize, arg, 10));
    } else {
        var c: usize = 1;
        while (c < trace.n_expert) : (c *= 2) try caps.append(allocator, c);
        try caps.append(allocator, trace.n_expert);
    }

    const seqs = try allocator.alloc(LayerSeq, trace.n_layers);
    var n_seq: usize = 0;
    defer {
        for (seqs[0..n_seq]) |*s| s.deinit(allocator);
        allocator.free(seqs);
    }
    var distinct_total: u64 = 0;
    for (seqs, 0..) |*s, layer| {
        s.* = try LayerSeq.build(allocator, &trace, @intCast(layer));
        n_seq += 1;
        for (s.counts) |c| {
            if (c != 0) distinct_total += 1;
        }
    }

    std.debug.print("trace: {d} requests, {d} layers, expert pool {d}, distinct (layer,expert) {d} ({d:.1}% of pool)\n", .{
        trace.requests.len,                     trace.n_layers, trace.n_expert, distinct_total,
        100.0 * @as(f64, @floatFromInt(distinct_total)) / @as(f64, @floatFromInt(@as(u64, trace.n_layers) * trace.n_expert)),
    });
    std.debug.print("{s:>6}  {s:>8}  {s:>8}  {s:>8}\n", .{ "slots", "lru", "belady", "pin+lru" });

    for (caps.items) |cap| {
        var lru: Tally = .{};
        var belady: Tally = .{};
        var pin_lru: Tally = .{};
        for (seqs) |*s| {
            if (s.eids.len == 0) continue;
            try runLru(allocator, s, trace.n_expert, cap, null, &lru);
            try runBelady(allocator, s, trace.n_expert, cap, &belady);
            // Half the slots pinned by histogram, half LRU — the store's
            // default budget split.
            const pinned = try pickPins(allocator, s.counts, cap / 2);
            defer allocator.free(pinned);
            try runLru(allocator, s, trace.n_expert, cap - cap / 2, pinned, &pin_lru);
        }
        const pct = struct {
            fn of(t: Tally) f64 {
                return if (t.total == 0) 0 else 100.0 * @as(f64, @floatFromInt(t.hits)) / @as(f64, @floatFromInt(t.total));
            }
        };
        std.debug.print("{d:>6}  {d:>7.2}%  {d:>7.2}%  {d:>7.2}%\n", .{ cap, pct.of(lru), pct.of(belady), pct.of(pin_lru) });
    }
}
