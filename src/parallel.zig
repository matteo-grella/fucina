//! Parallel-dispatch policy and substrate probes. Three things live here,
//! and the actual chunking is none of them (`thread.Pool.parallelChunks`
//! owns that):
//!
//!   1. the COMPTIME CPU crossover constants kernels consult before
//!      engaging the worker pool (`vector_*_threshold`, `row_kernel_*`,
//!      `backward_*`, ...); their runtime counterparts are
//!      `src/tuning.zig`'s table, and the two files split one policy
//!      surface on binding time;
//!   2. CPU-topology detection: physical-core count, Linux affinity masks,
//!      cgroup v1/v2 quotas, and the `FUCINA_MAX_THREADS` cap (the one env
//!      variable read below the tuning table, because it sizes the
//!      substrate the table's loader runs on);
//!   3. the sanctioned environment readers (`envFlag`, `envFlagValue`,
//!      `envPositiveUsize`, `envNonNegativeUsize`, `envStringIs`) with the
//!      libc-free-Linux `/proc/self/environ` fallback, which the tuning
//!      loader and the substrate gates read through.
//!
//! Layer stack: docs/ARCHITECTURE.md.
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
/// Fused-row-kernel parallel gate (softmax/norm/loss rows, quantized row
/// passes): these engage the pool at HALF the plain-elementwise crossover.
/// The ratio is policy in ONE place — retuning the base retunes every row
/// gate with it, deliberately.
pub const row_kernel_len_threshold: usize = vector_elementwise_len_threshold / 2;
/// Fused op-chain walk gate (splitGated forward rows, the fused
/// activation+quantize passes, the fused rmsNorm-mul-rope walk): several
/// ops per element, so the pool pays at an EIGHTH of the
/// plain-elementwise crossover. Same one-place ratio policy as
/// `row_kernel_len_threshold`.
pub const fused_chain_len_threshold: usize = vector_elementwise_len_threshold / 8;
/// Split gated-activation BACKWARD row gate (splitGlu/splitSwiGlu VJPs):
/// the pool pays at a QUARTER of the plain-elementwise crossover (fewer
/// fused ops per element than the forward chains above). Same one-place
/// ratio policy.
pub const split_backward_len_threshold: usize = vector_elementwise_len_threshold / 4;
pub const materialize_parallel_len_threshold: usize = 256 * 1024;
pub const materialize_parallel_min_chunk: usize = 64 * 1024;
pub const vector_matmul_work_threshold: usize = 1024 * 1024;
/// Attention parallel gate: attention kernels engage the pool at HALF the
/// GEMM work crossover. Same one-place ratio policy as
/// `row_kernel_len_threshold`.
pub const attention_work_threshold: usize = vector_matmul_work_threshold / 2;
pub const vector_batched_work_threshold: usize = 2 * 1024 * 1024;
pub const vector_column_min_m: usize = 32;
pub const vector_column_min_n: usize = 128;
pub const vector_column_work_multiplier: usize = 1;
pub const vector_column_chunk: usize = 64;

pub const backward_matmul_work_threshold: usize = 262_144;
pub const backward_async_work_threshold: usize = 256 * 1024 * 1024;
/// Per-batch m·n·k below which the BLAS arm of the batched GEMM splits the
/// batches over the worker team (each range running its batches through
/// BLAS on its own thread); at or above it the batches run sequentially and
/// BLAS threads each one itself. Measured on M1 Max with Accelerate: the
/// split wins up to 33M per batch (8x512x128x512: 370 vs 497 us) and loses
/// at 1G (4x1024^3: 5.96 vs 4.38 ms).
pub const blas_batch_split_max_work: usize = 256 * 1024 * 1024;

// Quantized-RHS dispatch gates of `backend/native.zig`: comptime
// crossovers like the ones above, kept here so the whole policy table is
// one place (native.zig aliases them file-locally). Values and their
// measurement rationale moved verbatim from native.zig.

/// Stack budget (in Q8_0 blocks) for the per-call LHS-quantization
/// scratch of the quantized-RHS dispatch tier: decode-shaped calls stay
/// heap-free, larger LHS rows take the caller-supplied allocator.
pub const q8_0_lhs_stack_blocks: usize = 512;
/// Off-multiple-m row minimum of the q4_k x4 fast path. q4_k pads the
/// final partial row group inside the x4 kernel, so every m >= 4 takes it
/// (one pass over the packed weights).
pub const q4_k_x4_min_rows: usize = 4;
/// q5_k has no padded-group kernel: its bulk+tail split re-reads the
/// packed weights once more for the 1-3 remainder rows, so below 128 rows
/// the one-pass per-row path wins.
pub const q5_k_x4_prefix_min_rows: usize = 128;
/// Prefill row count at/above which the Q2_0 matmul dequantizes weight
/// panels to f32 and rides BLAS (Accelerate AMX / OpenBLAS): the dequant
/// pass costs O(n*k) regardless of m, so its amortization — and the GEMM's
/// O(m) operand reuse, out of the int8 sdot path's reach on AMX-class
/// units — grows with m, while below the threshold (decode, short bursts)
/// the int8 mul-free path wins. Same split llama.cpp's BLAS backend makes
/// for its quantized prefill. The BLAS arm consumes exact f32 activations
/// (no Q8_0 LHS quantization), so its numerics differ from the int path
/// exactly as the dense-f32 BLAS GEMMs already do from the scalar backend.
pub const q2_0_blas_min_m: usize = 192;
/// f32 scratch budget for one dequantized weight panel (48 MiB). Panels
/// slice the CONTRACT dimension, never the output dimension: every GEMM is
/// then full-width with a contiguous C (accumulating across slices via
/// beta=1), where output-dimension panels would give narrow GEMMs writing
/// a strided C — a shape BLAS handles poorly.
pub const q2_0_blas_panel_floats: usize = 12 * 1024 * 1024;
/// Prefill row count at/above which the table-decoded formats (iq*/fp4)
/// dequantize weight panels to f32 and ride BLAS, exactly like the Q2_0
/// arm. Their int path pays a per-block table decode per (weight-row,
/// LHS-row) pair, so the dequant-once panel amortizes even earlier than
/// Q2_0's mul-free path — same accepted-numerics stance: the BLAS arm
/// consumes exact f32 activations, the int kernels keep decode and short
/// bursts (and every bitwise contract).
pub const table_blas_min_m: usize = 64;
/// Prefill row count at/above which the folded tied-K=2 PTQTP path
/// (`tq2_0_fx4`) dequantizes weight panels to f32 and rides BLAS. The
/// mul-free ternary tile owns decode and short bursts — it beats a GEMM
/// there — but its 4-column pack has no AMX-class batch form, so past this
/// width the dequant-once panel plus sgemm wins on operand reuse. Accepted
/// numerics, like every BLAS arm: exact f32 activations instead of the
/// Q8_K-quantized ones the integer kernel consumes.
pub const folded_blas_min_m: usize = 64;

var cached_cpu_count = std.atomic.Value(usize).init(0);

/// Runtime worker-count override (mirrors llama.cpp `-t` / the cli's
/// `set_num_threads`). Sets the cached CPU count so `cpuThreadCount` returns
/// `min(n, max_threads)` thereafter. Call once at startup before any parallel
/// work. `n == 0` is ignored. Equivalent to `FUCINA_MAX_THREADS` but settable
/// from a CLI flag.
pub fn setMaxThreads(n: usize) void {
    if (n >= 1) cached_cpu_count.store(n, .release);
}

/// Performance-core count on heterogeneous Apple Silicon
/// (`hw.perflevel0.physicalcpu`), null elsewhere or on inquiry failure.
/// A fork-join team sized past this count parks chunks on efficiency
/// cores and every barrier waits for them: the 0.6B LoRA training step
/// runs ~9% faster at 8 threads than at the 10-logical-core default on
/// an M1 Max (8P+2E). Pair with `setMaxThreads` where the workload is a
/// dense barrier stream; the library default stays the full count.
pub fn performanceCoreCount() ?usize {
    if (comptime @import("builtin").os.tag != .macos) return null;
    var value: c_int = 0;
    var len: usize = @sizeOf(c_int);
    if (std.c.sysctlbyname("hw.perflevel0.physicalcpu", &value, &len, null, 0) != 0) return null;
    if (value < 1) return null;
    return @intCast(value);
}

test "performance-core inquiry is sane on macOS and null-safe elsewhere" {
    if (performanceCoreCount()) |count| {
        try std.testing.expect(count >= 1);
        try std.testing.expect(count <= (std.Thread.getCpuCount() catch return));
    }
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
        // Apple Silicon: the compute team stays on PERFORMANCE cores. With
        // the two M1 E-cores in the team, every barrier waits for the E-core
        // stragglers — measured on the GPT train-step bench: 8P beats
        // 8P+2E by ~2%, and any spin-budget increase with E-cores enrolled
        // collapses throughput (287.9 -> 401.1 ms/step at 256k spins). The
        // QoS pin biases workers to P-cores but cannot guarantee placement;
        // sizing the team to perflevel0 does. `setMaxThreads` pre-seeds the
        // cache and remains the deliberate-oversubscription escape hatch.
        if (performanceCpuCount()) |performance| count = @min(count, @max(performance, 1));
        // A cgroup CPU-bandwidth limit never shows up in the affinity mask, so
        // size the team to the allowance as well: threads beyond it do not add
        // throughput, they just drain the quota sooner and stall the next
        // barrier until CFS refills it. `setMaxThreads` still pre-seeds the
        // cache as the escape hatch.
        if (cgroupCpuBudget()) |budget| count = @min(count, @max(budget, 1));
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
var cached_performance_cpu_count = std.atomic.Value(usize).init(0);

/// Performance-core count on heterogeneous Apple Silicon (sysctl
/// hw.perflevel0.physicalcpu), or null elsewhere/unknown. The compute-team
/// default in `cpuThreadCount` clamps to this; see the rationale there.
pub fn performanceCpuCount() ?usize {
    const cached = cached_performance_cpu_count.load(.acquire);
    if (cached != 0) return if (cached == std.math.maxInt(usize)) null else cached;
    const probed: ?usize = blk: {
        if (builtin.os.tag != .macos or builtin.cpu.arch != .aarch64) break :blk null;
        var n: c_int = 0;
        var len: usize = @sizeOf(c_int);
        if (std.c.sysctlbyname("hw.perflevel0.physicalcpu", &n, &len, null, 0) != 0) break :blk null;
        if (n < 1) break :blk null;
        break :blk @intCast(n);
    };
    cached_performance_cpu_count.store(probed orelse std.math.maxInt(usize), .release);
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
///    tighter `performanceCpuCount` clamp in `cpuThreadCount` (E-core
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

/// The FUCINA_MAX_THREADS cap, or null when unset/invalid/zero. Consulted only
/// on the first `cpuThreadCount` call (a `setMaxThreads` call before that wins
/// by pre-seeding the cache).
fn envMaxThreads() ?usize {
    return envPositiveUsize("FUCINA_MAX_THREADS");
}

/// Non-negative-usize environment knob, or null when unset/invalid. Unlike
/// `envPositiveUsize`, `0` is a VALID value: the tuning-table leaves that
/// read through this arm (`FUCINA_SPIN_BUDGET`, the GPU work floors and
/// percentages) give zero a meaning of its own (park immediately, always
/// offload, occupancy-blind). Same per-target arms as `envPositiveUsize`.
pub fn envNonNegativeUsize(comptime name: [:0]const u8) ?usize {
    if (builtin.link_libc) {
        const value = std.c.getenv(name) orelse return null;
        return parseNonNegativeUsize(std.mem.sliceTo(value, 0));
    } else if (builtin.os.tag == .linux) {
        return readProcSelfEnviron(name, parseNonNegativeUsize);
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

/// Tri-state boolean environment flag: null when unset or set empty (no
/// override), false when the value's first character is `'0'`, true
/// otherwise. The tuning table's boolean leaves read through this, so
/// setting a gate's variable to `0` forces the route off while an unset
/// variable keeps the measured default. Same per-target arms as `envFlag`.
pub fn envFlagValue(comptime name: [:0]const u8) ?bool {
    if (builtin.link_libc) {
        const value = std.c.getenv(name) orelse return null;
        const parsed = parseFlagValue(std.mem.sliceTo(value, 0)) orelse return null;
        return parsed == 1;
    } else if (builtin.os.tag == .linux) {
        const parsed = readProcSelfEnviron(name, parseFlagValue) orelse return null;
        return parsed == 1;
    } else {
        return null;
    }
}

fn parseFlagValue(s: []const u8) ?usize {
    if (s.len == 0) return null;
    return if (s[0] != '0') 1 else 0;
}

/// True when the variable is set to exactly `expected` (the contract of the
/// string-valued knobs that select a named mode, e.g.
/// `FUCINA_GPU_KERNELS=src`). Same per-target arms as `envFlag`.
pub fn envStringIs(comptime name: [:0]const u8, comptime expected: []const u8) bool {
    if (builtin.link_libc) {
        const value = std.c.getenv(name) orelse return false;
        return std.mem.eql(u8, std.mem.sliceTo(value, 0), expected);
    } else if (builtin.os.tag == .linux) {
        const match = struct {
            fn parse(s: []const u8) ?usize {
                return if (std.mem.eql(u8, s, expected)) 1 else 0;
            }
        };
        return (readProcSelfEnviron(name, match.parse) orelse 0) == 1;
    } else {
        return false;
    }
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

/// Part cap for a map over `n` items in chunks of at least `min_len`: one
/// part below `min_len`, else `1 + n / min_len` (`ExecContext.forRange`
/// clamps it to the team size). The chunk grid then depends only on `n`
/// and the part count; whether that makes a kernel thread-count-invariant
/// is the kernel's property to state (the optimizer and ES maps do).
pub fn partsForChunk(n: usize, min_len: usize) usize {
    return if (n >= min_len) 1 + n / min_len else 1;
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
