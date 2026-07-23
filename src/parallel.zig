const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");

// Comptime ceiling for the worker team and stack-allocated task arrays, AND the
// runtime default team size. Set at build time via -Dmax-threads=N (1-64); the
// default 8 is the performance-core count on the primary Apple Silicon target
// (M1 Max). Servers with more cores must raise the ceiling at build time (e.g.
// -Dmax-threads=32) — the FUCINA_MAX_THREADS env var only lowers the count at
// runtime, never raises it past this ceiling. NOTE: the best thread count is
// workload- and thermal-dependent. Measured on M1 Max across qwen3 / qwen3.5 /
// qwen3moe / gemma4: prefill is fastest at 8 cores when cool but at ~6 when
// heat-soaked, while decode on small/mid models is ~8-14% faster at 6
// (saturating all P-cores trips DVFS throttle and leaves no OS/spin-wait
// slack). No single value wins everywhere, so the default stays at the
// all-P-core ceiling (best cold prefill — the metric chased vs llama's AMX
// path) and the FUCINA_MAX_THREADS env var (mirrors llama.cpp's -t) drops it
// to e.g. 6 for decode-heavy or sustained workloads.
pub const vector_max_threads: usize = build_options.max_threads;
pub const vector_elementwise_len_threshold: usize = 256 * 1024;
pub const materialize_parallel_len_threshold: usize = 256 * 1024;
pub const materialize_parallel_min_chunk: usize = 64 * 1024;
pub const vector_matmul_work_threshold: usize = 1024 * 1024;
pub const vector_batched_work_threshold: usize = 2 * 1024 * 1024;
pub const vector_column_min_m: usize = 32;
pub const vector_column_min_n: usize = 128;
pub const vector_column_work_multiplier: usize = 1;
pub const vector_column_chunk: usize = 64;

pub const backward_matmul_work_threshold: usize = 262_144;
pub const backward_async_work_threshold: usize = 256 * 1024 * 1024;
pub const bmm_loop_work_threshold: usize = backward_matmul_work_threshold;
pub const bmm_loop_max_chunks: usize = 16;

var cached_cpu_count = std.atomic.Value(usize).init(0);

/// Runtime worker-count override (mirrors llama.cpp `-t` / the cli's
/// `set_num_threads`). Sets the cached CPU count so `cpuThreadCount` returns
/// `min(n, max_threads)` thereafter. Call once at startup before any parallel
/// work. `n == 0` is ignored. Equivalent to `FUCINA_MAX_THREADS` but settable
/// from a CLI flag.
pub fn setMaxThreads(n: usize) void {
    if (n >= 1) cached_cpu_count.store(n, .release);
}

pub fn cpuThreadCount(max_threads: usize) usize {
    var count = cached_cpu_count.load(.acquire);
    if (count == 0) {
        count = std.Thread.getCpuCount() catch 1;
        if (count == 0) count = 1;
        // SMT machines double-book cores in the logical count, and an
        // HT-oversubscribed team collapses throughput (i9-13950HX: a
        // 16-worker team pinned to 8 P-cores' hyperthreads ran 19s of
        // prefill in 43s — the x86 threading finding in docs/BENCHMARK.md).
        // min() never raises, and on no-SMT hosts physical == logical, so
        // this is a structural no-op on all Apple Silicon. A deliberate
        // consequence: FUCINA_MAX_THREADS caps the physical-core base and
        // can no longer reach the logical count on SMT machines;
        // `setMaxThreads` pre-seeds the cache before detection and remains
        // the escape hatch for deliberate oversubscription.
        if (physicalCpuCount()) |physical| count = @min(count, @max(physical, 1));
        // Optional override (mirrors llama.cpp's -t): cap the detected CPU count
        // for per-machine thread tuning. See the note on `vector_max_threads`
        // for when fewer threads help (decode / heat-soaked prefill on M1).
        if (envMaxThreads()) |cap| count = @min(count, cap);
        cached_cpu_count.store(count, .release);
    }
    return @max(@as(usize, 1), @min(count, max_threads));
}

// 0 = not yet probed; maxInt = probed, unknown; anything else = the count.
var cached_physical_cpu_count = std.atomic.Value(usize).init(0);

/// Physical-core count, or null when unknown (callers keep the logical
/// count). Public because the worker team's oversubscription guard
/// (`src/thread.zig` BarrierPool.init) compares its team size against it.
/// Process-cached: the probe runs once (on Linux it costs up to three
/// syscalls per CPU in the affinity mask) and every consumer sees one
/// consistent value — the first caller's affinity mask wins, the same
/// first-call-wins contract as `cpuThreadCount`'s cache.
pub fn physicalCpuCount() ?usize {
    const cached = cached_physical_cpu_count.load(.acquire);
    if (cached != 0) return if (cached == std.math.maxInt(usize)) null else cached;
    const probed = physicalCpuCountUncached();
    cached_physical_cpu_count.store(probed orelse std.math.maxInt(usize), .release);
    return probed;
}

/// The uncached probe behind `physicalCpuCount`. Per-target:
///  - macOS: sysctl hw.physicalcpu. Deliberately NOT hw.perflevel0.physicalcpu
///    (P-cores only): perflevel0 would silently shrink -Dmax-threads=10/16
///    builds, while hw.physicalcpu equals the logical count on all Apple
///    Silicon (no SMT), keeping every macOS config bit-for-bit unchanged;
///  - Linux (libc-free, the readProcSelfEnviron pattern): dedup of
///    /sys/devices/system/cpu/cpuN/topology/thread_siblings_list intersected
///    with the affinity mask — a core counts once, via its lowest schedulable
///    sibling, so `taskset -c 0-15` over 8 hyperthreaded P-cores resolves to
///    8 where a mask-blind dedup would report every core in the machine;
///  - elsewhere (Windows/wasi/freestanding): null.
fn physicalCpuCountUncached() ?usize {
    switch (builtin.os.tag) {
        .macos => {
            var n: c_int = 0;
            var len: usize = @sizeOf(c_int);
            if (std.c.sysctlbyname("hw.physicalcpu", &n, &len, null, 0) != 0) return null;
            if (n < 1) return null;
            return @intCast(n);
        },
        .linux => return linuxPhysicalCpuCount(),
        else => return null,
    }
}

fn cpuSetHas(set: std.os.linux.cpu_set_t, cpu: usize) bool {
    const word_bits = @bitSizeOf(usize);
    const word = cpu / word_bits;
    if (word >= set.len) return false;
    return (set[word] >> @intCast(cpu % word_bits)) & 1 != 0;
}

/// Linux physical-core count over the affinity mask. Any open/read/parse
/// failure (containers or minimal kernels without the sysfs topology files)
/// returns null — detection degrades to the logical count, today's behavior.
fn linuxPhysicalCpuCount() ?usize {
    const posix = std.posix;
    const set = posix.sched_getaffinity(0) catch return null;
    const total_bits = set.len * @bitSizeOf(usize);
    var count: usize = 0;
    for (0..total_bits) |cpu| {
        if (!cpuSetHas(set, cpu)) continue;

        var path_buf: [128]u8 = undefined;
        const path = std.fmt.bufPrintZ(
            &path_buf,
            "/sys/devices/system/cpu/cpu{d}/topology/thread_siblings_list",
            .{cpu},
        ) catch return null;
        const fd = posix.openatZ(posix.AT.FDCWD, path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, 0) catch return null;
        // std.posix has no close in 0.16; the raw syscall is fine on this
        // Linux-only path.
        defer _ = std.os.linux.close(fd);

        var buf: [4096]u8 = undefined;
        var filled: usize = 0;
        while (filled < buf.len) {
            const n = posix.read(fd, buf[filled..]) catch return null;
            if (n == 0) break;
            filled += n;
        }
        if (filled == buf.len) return null; // implausibly long list: treat as unknown

        var lowest_masked: ?usize = null;
        var it = CpuListIterator{ .text = buf[0..filled] };
        while (it.next()) |sibling| {
            if (cpuSetHas(set, sibling)) {
                if (lowest_masked == null or sibling < lowest_masked.?) lowest_masked = sibling;
            }
        }
        // A well-formed list contains `cpu` itself, so a masked sibling
        // always exists; anything else is inconsistent topology data.
        if (it.failed or lowest_masked == null) return null;
        if (lowest_masked.? == cpu) count += 1;
    }
    return if (count >= 1) count else null;
}

/// Iterator over a sysfs cpu-list string ("0,16", "0-1,8-9", "3", optional
/// trailing newline): yields each listed cpu, expanding a-b ranges. Sets
/// `failed` and stops on malformed input. Platform-independent so the
/// parsing logic is unit-tested on every target, not just Linux.
const CpuListIterator = struct {
    text: []const u8,
    i: usize = 0,
    range_next: usize = 0,
    range_end: ?usize = null,
    failed: bool = false,

    fn next(self: *CpuListIterator) ?usize {
        if (self.failed) return null;
        if (self.range_end) |end| {
            if (self.range_next <= end) {
                const cpu = self.range_next;
                self.range_next += 1;
                if (self.range_next > end) self.range_end = null;
                return cpu;
            }
            self.range_end = null;
        }
        // Skip trailing whitespace / separators between entries.
        while (self.i < self.text.len and (self.text[self.i] == '\n' or self.text[self.i] == ',')) self.i += 1;
        if (self.i >= self.text.len) return null;

        const first = self.parseNumber() orelse return null;
        if (self.i < self.text.len and self.text[self.i] == '-') {
            self.i += 1;
            const last = self.parseNumber() orelse return null;
            if (last < first) {
                self.failed = true;
                return null;
            }
            self.range_next = first + 1;
            if (first + 1 <= last) self.range_end = last;
            return first;
        }
        return first;
    }

    fn parseNumber(self: *CpuListIterator) ?usize {
        const start = self.i;
        while (self.i < self.text.len and std.ascii.isDigit(self.text[self.i])) self.i += 1;
        if (self.i == start) {
            self.failed = true;
            return null;
        }
        return std.fmt.parseInt(usize, self.text[start..self.i], 10) catch {
            self.failed = true;
            return null;
        };
    }
};

/// The FUCINA_MAX_THREADS cap, or null when unset/invalid/zero. Consulted only
/// on the first `cpuThreadCount` call (a `setMaxThreads` call before that wins
/// by pre-seeding the cache).
fn envMaxThreads() ?usize {
    return envPositiveUsize("FUCINA_MAX_THREADS");
}

/// The FUCINA_SPIN_BUDGET override for the worker-team spin-then-park window
/// (`src/thread.zig` BarrierPool), or null when unset/invalid. Unlike the
/// positive-usize knobs, `0` is a VALID value here — it means "park
/// immediately, never spin", the manual escape for oversubscribed teams (and
/// the value the guard in BarrierPool.init defaults to when the team exceeds
/// the physical-core count). Read once per BarrierPool init. See the
/// `spin_budget` comment in thread.zig for when (and on which hardware)
/// overriding pays; the default is left alone because sweeps (M1 Max;
/// i9-13950HX, 2026-07-03) found the response U-shaped and workload-coupled —
/// no single value wins every workload.
pub fn envSpinBudget() ?usize {
    if (builtin.link_libc) {
        const value = std.c.getenv("FUCINA_SPIN_BUDGET") orelse return null;
        return parseNonNegativeUsize(std.mem.sliceTo(value, 0));
    } else if (builtin.os.tag == .linux) {
        return readProcSelfEnviron("FUCINA_SPIN_BUDGET", parseNonNegativeUsize);
    } else {
        return null;
    }
}

/// Like `parsePositiveUsize` but `0` parses as a valid value; empty,
/// non-numeric, or sign-prefixed-negative input is still null (no override).
/// The explicit '-' check matters because `parseInt(usize, "-0")` succeeds —
/// without it "-0" would be a live park-immediately override while every
/// sibling knob treats it as unset.
fn parseNonNegativeUsize(s: []const u8) ?usize {
    if (s.len == 0 or s[0] == '-') return null;
    return std.fmt.parseInt(usize, s, 10) catch null;
}

/// Positive-usize environment knob, or null when unset/invalid/zero. Zig 0.16
/// has no libc-free std getenv (the environment block is only handed to `main`
/// via `std.process.Init`), so the read is per-target:
///  - libc builds (macOS always links libSystem): `std.c.getenv`;
///  - static Linux builds (the fully static ReleaseFast server binary, where
///    the libc arm compiles out and FUCINA_MAX_THREADS used to be a silent
///    no-op): scan /proc/self/environ — the kernel's copy of the initial
///    environment, readable without libc and without allocating;
///  - any other libc-free target (Windows/wasi/freestanding): no override.
pub fn envPositiveUsize(comptime name: [:0]const u8) ?usize {
    if (builtin.link_libc) {
        const value = std.c.getenv(name) orelse return null;
        return parsePositiveUsize(std.mem.sliceTo(value, 0));
    } else if (builtin.os.tag == .linux) {
        return readProcSelfEnviron(name, parsePositiveUsize);
    } else {
        return null;
    }
}

/// Boolean environment flag with the getenv-family truthiness contract
/// (set with a first character other than '0'), same per-target arms as
/// `envPositiveUsize` — the ONLY sanctioned way to read a flag from code
/// that must also compile into libc-free Linux binaries (bare
/// `std.c.getenv` is a compile error there).
pub fn envFlag(comptime name: [:0]const u8) bool {
    if (builtin.link_libc) {
        const value = std.c.getenv(name) orelse return false;
        return parseFlag(std.mem.sliceTo(value, 0)) == 1;
    } else if (builtin.os.tag == .linux) {
        return (readProcSelfEnviron(name, parseFlag) orelse 0) == 1;
    } else {
        return false;
    }
}

fn parseFlag(s: []const u8) ?usize {
    return if (s.len > 0 and s[0] != '0') 1 else 0;
}

/// Env-value parse contract (unchanged from the original libc-only arm):
/// base-10 usize; 0 or anything unparsable means "no override".
fn parsePositiveUsize(s: []const u8) ?usize {
    const n = std.fmt.parseInt(usize, s, 10) catch 0;
    return if (n >= 1) n else null;
}

/// Linux-only, libc-free env lookup via /proc/self/environ (NUL-separated
/// `KEY=VALUE` records; a trailing NUL is usual but the final record is
/// handled either way). Fixed stack buffers, no allocation. Any I/O failure
/// degrades to "no override".
fn readProcSelfEnviron(comptime name: [:0]const u8, comptime parse: fn ([]const u8) ?usize) ?usize {
    const posix = std.posix;
    const fd = posix.openatZ(
        posix.AT.FDCWD,
        "/proc/self/environ",
        .{ .ACCMODE = .RDONLY, .CLOEXEC = true },
        0,
    ) catch return null;
    // std.posix has no close in 0.16; the raw syscall is fine on this
    // Linux-only path.
    defer _ = std.os.linux.close(fd);

    var scan: EnvironScan(name, parse) = .init;
    var buf: [4096]u8 = undefined;
    while (true) {
        const n = posix.read(fd, &buf) catch return null;
        if (n == 0) break;
        if (scan.feed(buf[0..n])) |decision| return decision;
    }
    return scan.finish();
}

/// Incremental scanner for a NUL-separated `KEY=VALUE` environment block (the
/// /proc/self/environ format), matching getenv semantics: the FIRST record
/// whose key is `name` decides, and an invalid/zero value decides "no
/// override" (later duplicates are not consulted). Platform-independent so the
/// parsing logic is unit-tested on every target, not just Linux.
fn EnvironScan(comptime name: [:0]const u8, comptime parse: fn ([]const u8) ?usize) type {
    return struct {
        const Self = @This();
        const key = name ++ "=";

        // A valid usize value is at most 20 decimal digits, so 64 bytes covers
        // every record that could ever produce an override. Longer records are
        // tracked as overflowed: a matching-but-overlong record still
        // concludes the scan with "no override" (its value cannot parse; first
        // match wins, like getenv).
        entry: [64]u8,
        entry_len: usize,
        overflowed: bool,

        const init: Self = .{ .entry = undefined, .entry_len = 0, .overflowed = false };

        /// Feed the next chunk. Outer null = keep scanning; otherwise the scan
        /// has concluded and the inner `?usize` is the override decision
        /// (null = no override). Stop feeding once concluded.
        fn feed(self: *Self, chunk: []const u8) ??usize {
            for (chunk) |byte| {
                if (byte != 0) {
                    if (self.entry_len < self.entry.len) {
                        self.entry[self.entry_len] = byte;
                        self.entry_len += 1;
                    } else {
                        self.overflowed = true;
                    }
                    continue;
                }
                if (self.concludeEntry()) |decision| return decision;
            }
            return null;
        }

        /// Handle a final record with no trailing NUL. Returns the override,
        /// if any.
        fn finish(self: *Self) ?usize {
            if (self.entry_len == 0 and !self.overflowed) return null;
            return self.concludeEntry() orelse null;
        }

        fn concludeEntry(self: *Self) ??usize {
            defer {
                self.entry_len = 0;
                self.overflowed = false;
            }
            const e = self.entry[0..self.entry_len];
            if (!std.mem.startsWith(u8, e, key)) return null;
            if (self.overflowed) return @as(?usize, null);
            return parse(e[key.len..]);
        }
    };
}

pub fn saturatedMul3(a: usize, b: usize, c: usize) usize {
    const ab = std.math.mul(usize, a, b) catch return std.math.maxInt(usize);
    return std.math.mul(usize, ab, c) catch std.math.maxInt(usize);
}

test {
    _ = @import("parallel_tests.zig");
}

// The env-scanner tests stay inline: they exercise file-private symbols and
// the repo policy is to never add `pub` just to move a test to the sibling
// file. Real end-to-end FUCINA_MAX_THREADS behavior (process env is fixed
// before main) is covered by the remote static-Linux A/B verification.

test "parsePositiveUsize: usize base-10, 0/invalid mean no override" {
    try std.testing.expectEqual(@as(?usize, 6), parsePositiveUsize("6"));
    try std.testing.expectEqual(@as(?usize, 8), parsePositiveUsize("0008"));
    try std.testing.expectEqual(@as(?usize, null), parsePositiveUsize("0"));
    try std.testing.expectEqual(@as(?usize, null), parsePositiveUsize(""));
    try std.testing.expectEqual(@as(?usize, null), parsePositiveUsize("abc"));
    try std.testing.expectEqual(@as(?usize, null), parsePositiveUsize("6abc"));
    try std.testing.expectEqual(@as(?usize, null), parsePositiveUsize("-4"));
    // 21 digits overflows usize -> parse error -> no override.
    try std.testing.expectEqual(@as(?usize, null), parsePositiveUsize("111111111111111111111"));
}

test "parseNonNegativeUsize: like parsePositiveUsize but 0 is a valid value" {
    try std.testing.expectEqual(@as(?usize, 0), parseNonNegativeUsize("0"));
    try std.testing.expectEqual(@as(?usize, 512), parseNonNegativeUsize("512"));
    try std.testing.expectEqual(@as(?usize, null), parseNonNegativeUsize(""));
    try std.testing.expectEqual(@as(?usize, null), parseNonNegativeUsize("abc"));
    try std.testing.expectEqual(@as(?usize, null), parseNonNegativeUsize("-4"));
    // parseInt(usize, "-0") would succeed; the sign check keeps "-0" unset
    // like every sibling knob instead of a live park-immediately override.
    try std.testing.expectEqual(@as(?usize, null), parseNonNegativeUsize("-0"));
    try std.testing.expectEqual(@as(?usize, null), parseNonNegativeUsize("111111111111111111111"));
}

// The scanner is comptime-keyed (FUCINA_MAX_THREADS / FUCINA_SPIN_BUDGET share
// it); the tests instantiate it with the original key.
const MaxThreadsScan = EnvironScan("FUCINA_MAX_THREADS", parsePositiveUsize);

test "EnvironScan: finds the key among other records" {
    var scan: MaxThreadsScan = .init;
    const block = "PATH=/usr/bin\x00FUCINA_MAX_THREADS=6\x00HOME=/root\x00";
    try std.testing.expectEqual(@as(??usize, @as(?usize, 6)), scan.feed(block));
}

test "EnvironScan: absent key scans to the end with no cap" {
    var scan: MaxThreadsScan = .init;
    try std.testing.expectEqual(@as(??usize, null), scan.feed("PATH=/usr/bin\x00HOME=/root\x00"));
    try std.testing.expectEqual(@as(?usize, null), scan.finish());
}

test "EnvironScan: record split across arbitrary chunk boundaries" {
    const block = "AA=1\x00FUCINA_MAX_THREADS=12\x00BB=2\x00";
    // Byte-at-a-time is the worst-case chunking.
    var scan: MaxThreadsScan = .init;
    var decided: ??usize = null;
    for (block) |byte| {
        if (scan.feed(&[_]u8{byte})) |decision| {
            decided = decision;
            break;
        }
    }
    try std.testing.expectEqual(@as(??usize, @as(?usize, 12)), decided);
}

test "EnvironScan: invalid or zero value concludes with no cap" {
    var scan: MaxThreadsScan = .init;
    try std.testing.expectEqual(
        @as(??usize, @as(?usize, null)),
        scan.feed("FUCINA_MAX_THREADS=abc\x00"),
    );
    scan = .init;
    try std.testing.expectEqual(
        @as(??usize, @as(?usize, null)),
        scan.feed("FUCINA_MAX_THREADS=0\x00FUCINA_MAX_THREADS=5\x00"),
    );
}

test "EnvironScan: final record without trailing NUL" {
    var scan: MaxThreadsScan = .init;
    try std.testing.expectEqual(@as(??usize, null), scan.feed("A=1\x00FUCINA_MAX_THREADS=4"));
    try std.testing.expectEqual(@as(?usize, 4), scan.finish());
}

test "EnvironScan: overlong records" {
    // Overlong record with a matching key concludes with no cap (first match
    // wins; its value cannot be a valid usize).
    var scan: MaxThreadsScan = .init;
    const overlong_match = "FUCINA_MAX_THREADS=" ++ "1" ** 100 ++ "\x00FUCINA_MAX_THREADS=5\x00";
    try std.testing.expectEqual(@as(??usize, @as(?usize, null)), scan.feed(overlong_match));
    // Overlong record with a non-matching key is skipped; the scan continues.
    scan = .init;
    const overlong_other = "JUNK=" ++ "x" ** 200 ++ "\x00FUCINA_MAX_THREADS=6\x00";
    try std.testing.expectEqual(@as(??usize, @as(?usize, 6)), scan.feed(overlong_other));
}

test "EnvironScan: near-miss keys never match" {
    var scan: MaxThreadsScan = .init;
    const block = "FUCINA_MAX_THREADS_X=9\x00XFUCINA_MAX_THREADS=9\x00FUCINA_MAX_THREAD=9\x00\x00";
    try std.testing.expectEqual(@as(??usize, null), scan.feed(block));
    try std.testing.expectEqual(@as(?usize, null), scan.finish());
}

test "readProcSelfEnviron: live smoke on Linux" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    // The test process env is not under our control; assert the read is safe
    // and any produced override is valid.
    if (readProcSelfEnviron("FUCINA_MAX_THREADS", parsePositiveUsize)) |cap| try std.testing.expect(cap >= 1);
}

fn expectCpuList(text: []const u8, expected: []const usize) !void {
    var it = CpuListIterator{ .text = text };
    for (expected) |cpu| try std.testing.expectEqual(@as(?usize, cpu), it.next());
    try std.testing.expectEqual(@as(?usize, null), it.next());
    try std.testing.expect(!it.failed);
}

test "CpuListIterator: singletons, pairs, ranges, trailing newline" {
    try expectCpuList("3", &.{3});
    try expectCpuList("3\n", &.{3});
    try expectCpuList("0,16", &.{ 0, 16 });
    try expectCpuList("0-1", &.{ 0, 1 });
    try expectCpuList("0-1,8-9\n", &.{ 0, 1, 8, 9 });
    try expectCpuList("5-5", &.{5});
    try expectCpuList("", &.{});
}

test "CpuListIterator: malformed input sets failed" {
    const cases = [_][]const u8{ "a", "1-", "-3", "1-a", "3-1" };
    for (cases) |text| {
        var it = CpuListIterator{ .text = text };
        while (it.next()) |_| {}
        try std.testing.expect(it.failed);
    }
}

test "physicalCpuCount: null or within [1, logical]" {
    const logical = std.Thread.getCpuCount() catch 1;
    if (physicalCpuCount()) |physical| {
        try std.testing.expect(physical >= 1);
        try std.testing.expect(physical <= logical);
    }
}
