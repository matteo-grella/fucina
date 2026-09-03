//! CPU topology probes: physical-core count (macOS sysctl; Linux
//! thread-sibling dedup intersected with the affinity mask), the Apple
//! Silicon performance-core count, the cgroup v1/v2 CPU-bandwidth budget,
//! and the tightest schedulable count that folds them together. Every probe
//! runs once and is process-cached (first call wins); `parallel.zig` sizes
//! the worker team from these and `thread.zig`'s barrier pool reads the
//! oversubscription guard. A std-only leaf. Layer stack: docs/ARCHITECTURE.md.
const std = @import("std");
const builtin = @import("builtin");

test "performance-core inquiry is sane on macOS and null-safe elsewhere" {
    if (performanceCoreCount()) |count| {
        try std.testing.expect(count >= 1);
        try std.testing.expect(count <= (std.Thread.getCpuCount() catch return));
    }
}

// 0 = not yet probed; maxInt = probed, unknown; anything else = the count.
var cached_physical_cpu_count = std.atomic.Value(usize).init(0);

var cached_performance_core_count = std.atomic.Value(usize).init(0);

/// Performance-core count on heterogeneous Apple Silicon (sysctl
/// hw.perflevel0.physicalcpu), or null elsewhere/unknown. The compute-team
/// default in `cpuThreadCount` clamps to this; see the rationale there.
pub fn performanceCoreCount() ?usize {
    const cached = cached_performance_core_count.load(.acquire);
    if (cached != 0) return if (cached == std.math.maxInt(usize)) null else cached;
    const probed: ?usize = blk: {
        if (builtin.os.tag != .macos or builtin.cpu.arch != .aarch64) break :blk null;
        var n: c_int = 0;
        var len: usize = @sizeOf(c_int);
        if (std.c.sysctlbyname("hw.perflevel0.physicalcpu", &n, &len, null, 0) != 0) break :blk null;
        if (n < 1) break :blk null;
        break :blk @intCast(n);
    };
    cached_performance_core_count.store(probed orelse std.math.maxInt(usize), .release);
    return probed;
}

/// Physical-core count, or null when unknown (callers keep the logical
/// count). The oversubscription guard reads it through `schedulableCpuCount`,
/// which also folds in the cgroup bandwidth limit.
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
///  - macOS: sysctl hw.physicalcpu — ALL physical cores. This stays the
///    oversubscription guard's reference; the compute-team default takes the
///    tighter `performanceCoreCount` clamp in `cpuThreadCount` (E-core
///    stragglers stall every barrier — the train-step measurement there);
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

// 0 = not yet probed; maxInt = probed, unknown; anything else = the budget.
var cached_cgroup_cpu_budget = std.atomic.Value(usize).init(0);

/// Whole-CPU budget from this process's cgroup CPU-bandwidth limit, or null
/// when unlimited/unknown.
///
/// A bandwidth limit is invisible to `sched_getaffinity`: a container run with
/// `--cpus=2` (or a Kubernetes CPU limit) still sees every CPU in the mask. A
/// team sized to the machine then burns the allowance faster than CFS refills
/// it, and each fork-join barrier stalls until the next period. Folding the
/// quota in keeps one invariant: the same CPU budget performs the same however
/// it was imposed, mask or quota.
///
/// Process-cached, first-call-wins, like `physicalCpuCount`.
pub fn cgroupCpuBudget() ?usize {
    const cached = cached_cgroup_cpu_budget.load(.acquire);
    if (cached != 0) return if (cached == std.math.maxInt(usize)) null else cached;
    const probed = if (builtin.os.tag == .linux) linuxCgroupCpuBudget() else null;
    cached_cgroup_cpu_budget.store(probed orelse std.math.maxInt(usize), .release);
    return probed;
}

/// The tightest CPU budget this process can actually run on: physical cores in
/// the affinity mask, further clamped by any cgroup bandwidth limit. null when
/// neither is known. This is the oversubscription guard's reference
/// (`src/thread.zig` BarrierPool.init) — a team wider than this stalls at every
/// barrier, whichever mechanism imposed the limit.
pub fn schedulableCpuCount() ?usize {
    const physical = physicalCpuCount();
    const cgroup = cgroupCpuBudget();
    if (physical) |p| return if (cgroup) |c| @min(p, c) else p;
    return cgroup;
}

/// cgroup CPU-bandwidth probe (Linux). v2 reads `cpu.max` at this process's own
/// cgroup and at every ancestor up to the mount root — an ancestor's limit
/// binds just as hard, so the tightest wins; v1 falls back to
/// `cpu.cfs_quota_us`/`cpu.cfs_period_us`. Any open/read/parse failure returns
/// null and detection degrades to affinity-only, today's behavior.
fn linuxCgroupCpuBudget() ?usize {
    var proc_buf: [4096]u8 = undefined;
    if (readSmallFile("/proc/self/cgroup", &proc_buf)) |text| {
        if (cgroupV2RelativePath(text)) |relative| {
            var best: ?usize = null;
            var end: usize = relative.len;
            while (true) {
                var path_buf: [640]u8 = undefined;
                const path = std.fmt.bufPrintZ(
                    &path_buf,
                    "/sys/fs/cgroup{s}/cpu.max",
                    .{relative[0..end]},
                ) catch break;
                var file_buf: [64]u8 = undefined;
                if (readSmallFile(path, &file_buf)) |contents| {
                    if (parseCgroupV2CpuMax(contents)) |budget| {
                        best = if (best) |b| @min(b, budget) else budget;
                    }
                }
                if (end == 0) break;
                end = std.mem.lastIndexOfScalar(u8, relative[0..end], '/') orelse 0;
            }
            if (best != null) return best;
        }
    }
    const quota = readCgroupInt("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") orelse return null;
    const period = readCgroupInt("/sys/fs/cgroup/cpu/cpu.cfs_period_us") orelse return null;
    return cpuBudgetFromQuota(quota, period);
}

/// Read a small sysfs/proc file whole. null on any failure, or when the file
/// fills the buffer (implausibly large: treat as unknown rather than truncate).
fn readSmallFile(path: [:0]const u8, buf: []u8) ?[]const u8 {
    const posix = std.posix;
    const fd = posix.openatZ(posix.AT.FDCWD, path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, 0) catch return null;
    // std.posix has no close in 0.16; the raw syscall is fine on this
    // Linux-only path.
    defer _ = std.os.linux.close(fd);
    var filled: usize = 0;
    while (filled < buf.len) {
        const n = posix.read(fd, buf[filled..]) catch return null;
        if (n == 0) break;
        filled += n;
    }
    if (filled == buf.len) return null;
    return buf[0..filled];
}

fn readCgroupInt(path: [:0]const u8) ?i64 {
    var buf: [64]u8 = undefined;
    const text = readSmallFile(path, &buf) orelse return null;
    return std.fmt.parseInt(i64, std.mem.trim(u8, text, " \t\r\n"), 10) catch null;
}

/// The unified-hierarchy (v2) line of /proc/self/cgroup is `0::<path>`; the
/// path is relative to the cgroup mount. Returns null on a v1-only file or
/// malformed input. Pure so the parsing is unit-tested on every target.
fn cgroupV2RelativePath(text: []const u8) ?[]const u8 {
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "0::")) continue;
        const path = std.mem.trim(u8, line[3..], " \t\r");
        if (path.len == 0 or path[0] != '/') return null;
        return path;
    }
    return null;
}

/// Parse cgroup v2 `cpu.max`: "max 100000" (unlimited) -> null,
/// "200000 100000" -> 2. Pure, unit-tested on every target.
fn parseCgroupV2CpuMax(text: []const u8) ?usize {
    var it = std.mem.tokenizeAny(u8, std.mem.trim(u8, text, " \t\r\n"), " \t");
    const quota_text = it.next() orelse return null;
    const period_text = it.next() orelse return null;
    if (it.next() != null) return null;
    if (std.mem.eql(u8, quota_text, "max")) return null;
    const quota = std.fmt.parseInt(i64, quota_text, 10) catch return null;
    const period = std.fmt.parseInt(i64, period_text, 10) catch return null;
    return cpuBudgetFromQuota(quota, period);
}

/// Whole CPUs a quota/period pair allows: floored, but never below 1. Floor
/// because a team wider than the allowance is exactly the barrier stall this
/// probe exists to avoid; the 1 floor because a fractional allowance still
/// schedules one worker. A non-positive period, or v1's negative "unlimited"
/// quota, means no limit.
fn cpuBudgetFromQuota(quota: i64, period: i64) ?usize {
    if (quota <= 0 or period <= 0) return null;
    const budget = @divTrunc(quota, period);
    return if (budget <= 0) 1 else @intCast(budget);
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

test "cpuBudgetFromQuota: floors, never below 1, unlimited stays null" {
    try std.testing.expectEqual(@as(?usize, 2), cpuBudgetFromQuota(200_000, 100_000));
    try std.testing.expectEqual(@as(?usize, 8), cpuBudgetFromQuota(800_000, 100_000));
    // Fractional allowances floor, but still schedule one worker.
    try std.testing.expectEqual(@as(?usize, 1), cpuBudgetFromQuota(150_000, 100_000));
    try std.testing.expectEqual(@as(?usize, 1), cpuBudgetFromQuota(50_000, 100_000));
    // v1 spells "unlimited" as a negative quota; a zero period is malformed.
    try std.testing.expectEqual(@as(?usize, null), cpuBudgetFromQuota(-1, 100_000));
    try std.testing.expectEqual(@as(?usize, null), cpuBudgetFromQuota(0, 100_000));
    try std.testing.expectEqual(@as(?usize, null), cpuBudgetFromQuota(200_000, 0));
}

test "parseCgroupV2CpuMax: quota pair, unlimited, malformed" {
    try std.testing.expectEqual(@as(?usize, 2), parseCgroupV2CpuMax("200000 100000\n"));
    try std.testing.expectEqual(@as(?usize, 4), parseCgroupV2CpuMax("  400000\t100000  "));
    try std.testing.expectEqual(@as(?usize, null), parseCgroupV2CpuMax("max 100000\n"));
    try std.testing.expectEqual(@as(?usize, null), parseCgroupV2CpuMax("200000\n"));
    try std.testing.expectEqual(@as(?usize, null), parseCgroupV2CpuMax("200000 100000 7\n"));
    try std.testing.expectEqual(@as(?usize, null), parseCgroupV2CpuMax("abc 100000\n"));
    try std.testing.expectEqual(@as(?usize, null), parseCgroupV2CpuMax(""));
}

test "cgroupV2RelativePath: the 0:: line, v1-only files, malformed" {
    try std.testing.expectEqualStrings(
        "/user.slice/run-x.scope",
        cgroupV2RelativePath("0::/user.slice/run-x.scope\n").?,
    );
    // The v2 line need not come first.
    try std.testing.expectEqualStrings(
        "/pod",
        cgroupV2RelativePath("4:cpu,cpuacct:/pod\n0::/pod\n").?,
    );
    try std.testing.expectEqualStrings("/", cgroupV2RelativePath("0::/\n").?);
    try std.testing.expectEqual(@as(?[]const u8, null), cgroupV2RelativePath("4:cpu,cpuacct:/pod\n"));
    try std.testing.expectEqual(@as(?[]const u8, null), cgroupV2RelativePath("0::\n"));
    try std.testing.expectEqual(@as(?[]const u8, null), cgroupV2RelativePath("0::relative\n"));
}

test "cgroupCpuBudget / schedulableCpuCount: null or a sane positive budget" {
    if (cgroupCpuBudget()) |budget| try std.testing.expect(budget >= 1);
    const logical = std.Thread.getCpuCount() catch 1;
    if (schedulableCpuCount()) |count| {
        try std.testing.expect(count >= 1);
        // The affinity leg is bounded by the logical count; the cgroup leg can
        // only lower it further.
        if (physicalCpuCount() != null) try std.testing.expect(count <= logical);
        // schedulableCpuCount is the tighter of the two legs it composes.
        if (physicalCpuCount()) |physical| try std.testing.expect(count <= physical);
        if (cgroupCpuBudget()) |budget| try std.testing.expect(count <= budget);
    }
}
