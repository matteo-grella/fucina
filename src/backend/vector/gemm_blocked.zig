//! Cache-blocked, packed (BLIS-style) f32 GEMM for the pure-Zig path.
//!
//! The register-tiled row kernels in gemm.zig re-stream the full B operand
//! for every row block of C, so once the operands exceed L2 the large
//! (training-shaped) GEMMs turn memory-bound. This module implements the
//! classic three-level blocked loop nest with packed panels:
//!
//!   jc (nc) -> pc (kc) -> ic (mc), where
//!     B~(kc x nc) is packed once per (jc, pc) into column panels of width nr,
//!     A~(mc x kc) is packed once per (ic, pc) into row panels of height mr,
//!     and an mr x nr register microkernel streams both panels unit-stride.
//!
//! Orientations: TransA/TransB are absorbed by the packing — the pack loops
//! read the transposed source layout but emit the same panel layout, so the
//! microkernel is orientation-agnostic and NN/TN/NT all share it.
//!
//! Scratch memory: backend kernels must stay allocation-free and the dense
//! f32 matmul entry points are infallible (no allocator parameter to thread
//! through, unlike the quantized-RHS paths), so the pack buffers live in a
//! comptime-bounded static workspace (BSS, committed lazily by the OS)
//! guarded by a mutex. The lock is held for the duration of one GEMM;
//! concurrent blocked GEMMs serialize on it — across ExecContexts (the
//! runtime is effectively single-stream) and equally WITHIN one ExecContext
//! (e.g. independent backward branches dispatched async): a known, accepted
//! trade. If training diamonds ever show this lock hot, bench a per-caller
//! workspace before reaching for anything fancier. Pool workers never
//! re-enter this path: dispatch happens only at the top-level 2-D matmul
//! entry points, and the blocked kernel itself calls no other GEMM.
//!
//! Parallelism: across ic blocks via the persistent `parallelChunks` team
//! (capped at `parallel.vector_max_threads`). B~ is shared read-only per
//! (jc, pc); every task packs its own A~ into a per-task workspace slot.
//! On non-aarch64 targets, when there are fewer ic blocks than workers (small
//! m, e.g. LLM prefill with m in the low hundreds), each ic block is further
//! split into nr-aligned column chunks so the (ic block, j chunk) cell grid
//! still feeds every worker; a task re-packs A~ once per ic block it touches
//! (the pack is tiny next to the microkernel work it amortizes). Tasks own
//! disjoint cells, so each C tile is written by exactly one task and the
//! result is deterministic and independent of the thread count.

const std = @import("std");
const builtin = @import("builtin");
const parallel = @import("../../parallel.zig");
const thread = @import("../../thread.zig");
const vm = @import("common.zig");

const ParallelConfig = vm.ParallelConfig;
const vector_len = vm.vector_len;
const Vf32 = vm.Vf32;

// ---------------- Microkernel shape ----------------
//
// Register math:
//   aarch64 NEON (32 x 128-bit v-regs, vector_len = 4): mr = 8, nr = 12 ->
//     24 four-wide accumulators + 3 B vectors + 1 A broadcast = 28 live regs.
//   x86-64 AVX2 (16 ymm, vector_len = 8) and other targets: mr = 6,
//     nr = 2 * vector_len -> 12 accumulators + 2 B vectors + 1 broadcast = 15.
pub const mr: usize = if (builtin.cpu.arch.isAARCH64()) 8 else 6;
pub const nr_vecs: usize = if (builtin.cpu.arch.isAARCH64()) 3 else 2;
pub const nr: usize = nr_vecs * vector_len;

pub const Orientation = enum { nn, tn, nt };

// Cache-blocking factors. aarch64 defaults tuned on M1 Max (L1d 128 KiB per
// P-core, shared L2 ~12 MiB) with `zig build bench-gemm -Dblas=none -- --sweep`
// at 2048^3: kc=128/mc={64,128}/nc=1024 was the consistent winner across runs
// (~620 GFLOP/s vs ~570-610 for most of the plateau). kc=128 x mc=128 keeps
// A~ at 64 KiB (L1d-resident on the M1 P-core) with B~ slivers of
// kc x nr = 6 KiB, and nc=1024 amortizes each A~ pack over more B panels.
//
// x86 default kc=512 tuned on Raptor Lake (i9-13950HX, AVX2, L1d 48 KiB /
// L2 2 MiB per P-core; `--omni-params` interleaved candidates, one thread per
// physical core): the deeper k panel keeps the B~ sliver at kc x nr = 32 KiB
// (L1d-resident), quarters the C accumulate passes, and won or tied every
// shape tested — 253-row NT prefill 850->935 GF/s, 2048^3 NN 767->788 —
// vs kc=128; nc=512 and mc={72,96,192} all measured worse.
pub const x86_default_kc: usize = 512;
pub const BlockParams = struct {
    kc: usize = if (builtin.cpu.arch.isAARCH64()) 128 else x86_default_kc,
    mc: usize = 128,
    nc: usize = 1024,
};

// Comptime workspace bounds; `BlockParams` used at runtime (e.g. by the bench
// sweep) must stay within these.
pub const kc_max: usize = 512;
pub const mc_max: usize = 256;
pub const nc_max: usize = 1024;

fn roundUpComptime(comptime x: usize, comptime m: usize) usize {
    return (x + m - 1) / m * m;
}

const b_panel_capacity: usize = kc_max * roundUpComptime(nc_max, nr);
const a_panel_capacity: usize = roundUpComptime(mc_max, mr) * kc_max;

// Static pack workspace (~2 MiB B~ + max_threads x ~512 KiB A~ slots — ~32
// MiB of BSS at -Dmax-threads=64, lazily committed). Guarded by
// `workspace_lock`; see the module doc comment.
var workspace_lock: thread.Mutex = .{};
var b_panel_storage: [b_panel_capacity]f32 align(64) = undefined;
var a_panel_storage: [parallel.vector_max_threads][a_panel_capacity]f32 align(64) = undefined;

// ---------------- Dispatch gate ----------------
//
// Below the work threshold the register-tiled row kernels already keep their
// operands cache-resident (and avoid the packing traffic); tiny m would waste
// most of each mr-row panel. The row kernels stream B once per 8-row block,
// so for m <= 16 they are within one B pass of optimal and packing can only
// lose; min_m = 32 matches `parallel.vector_column_min_m`.
//
// Threshold tuned on M1 Max via bench-gemm (-Dblas=none, ReleaseFast, 8
// threads): cool-state row kernels peak at ~316 GF/s while operands stay
// L2-resident and beat the blocked kernel up to 512^3 (297 GF/s, 0.94x); by
// 640^3 the blocked kernel wins decisively (374 vs 316, and 2-6x once
// operands spill: 1024^3 ~600 vs ~304, 2048^3 ~610 vs ~102). 192 Mi work
// (= 768 x 512 x 512) sits in that 512^3..640^3 gap.
pub const blocked_min_m: usize = 32;
pub const blocked_min_n: usize = 32;
pub const blocked_min_k: usize = 16;
pub const blocked_work_threshold: usize = 192 * 1024 * 1024;

pub fn shouldUseBlocked(m: usize, n: usize, k: usize) bool {
    if (m < blocked_min_m or n < blocked_min_n or k < blocked_min_k) return false;
    return parallel.saturatedMul3(m, n, k) >= blocked_work_threshold;
}

// ---------------- Entry points ----------------

pub fn gemmBlocked(
    comptime orient: Orientation,
    cd: []f32,
    ad: []const f32,
    bd: []const f32,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) void {
    gemmBlockedWithParams(orient, cd, ad, bd, m, n, k, config, .{});
}

pub fn gemmBlockedWithParams(
    comptime orient: Orientation,
    cd: []f32,
    ad: []const f32,
    bd: []const f32,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
    params: BlockParams,
) void {
    // Unconditional (not std.debug.assert): out-of-bounds BlockParams would
    // overrun the static pack workspace, and the bench sweep feeds runtime
    // params in ReleaseFast where asserts vanish. Once per GEMM — nil cost.
    if (params.kc < 1 or params.kc > kc_max or
        params.mc < 1 or params.mc > mc_max or
        params.nc < 1 or params.nc > nc_max)
    {
        @panic("gemmBlockedWithParams: BlockParams out of the static workspace bounds (1 <= kc <= kc_max, 1 <= mc <= mc_max, 1 <= nc <= nc_max)");
    }
    if (m == 0 or n == 0) return;
    if (k == 0) {
        @memset(cd[0 .. m * n], 0);
        return;
    }

    workspace_lock.lock();
    defer workspace_lock.unlock();

    const threads = @max(@as(usize, 1), parallel.cpuThreadCount(parallel.vector_max_threads));
    const num_ic_blocks = (m + params.mc - 1) / params.mc;

    var jc: usize = 0;
    while (jc < n) : (jc += params.nc) {
        const nc_eff = @min(params.nc, n - jc);
        // (ic block, j chunk) cell grid. aarch64 keeps the historical
        // ic-block-only split (num_j_chunks == 1 reduces to it exactly);
        // elsewhere small-m shapes split each ic block's columns into
        // nr-panel-aligned chunks so every worker gets a cell.
        const num_nr_panels = (nc_eff + nr - 1) / nr;
        const num_j_chunks = if (comptime builtin.cpu.arch.isAARCH64())
            1
        else if (num_ic_blocks >= threads)
            1
        else
            @min((threads + num_ic_blocks - 1) / num_ic_blocks, num_nr_panels);
        const num_cells = num_ic_blocks * num_j_chunks;
        const task_count = @max(@as(usize, 1), @min(threads, num_cells));

        var pc: usize = 0;
        while (pc < k) : (pc += params.kc) {
            const kc_eff = @min(params.kc, k - pc);
            packBParallel(orient, bd, n, k, pc, kc_eff, jc, nc_eff, num_nr_panels, threads, config);

            const b_panel_len = kc_eff * num_nr_panels * nr;
            var tasks: [parallel.vector_max_threads]BlockedTask = undefined;
            for (0..task_count) |ti| {
                tasks[ti] = .{
                    .cd = cd,
                    .ad = ad,
                    .b_panel = b_panel_storage[0..b_panel_len],
                    .a_panel = &a_panel_storage[ti],
                    .m = m,
                    .n = n,
                    .k = k,
                    .jc = jc,
                    .nc_eff = nc_eff,
                    .pc = pc,
                    .kc_eff = kc_eff,
                    .mc = params.mc,
                    .num_j_chunks = num_j_chunks,
                    .num_nr_panels = num_nr_panels,
                    .cell_start = ti * num_cells / task_count,
                    .cell_end = (ti + 1) * num_cells / task_count,
                    .accumulate = pc != 0,
                };
            }
            const run = comptime taskRunner(orient);
            if (config.pool) |pool| {
                pool.parallelChunks(BlockedTask, tasks[0..task_count], run);
            } else {
                for (tasks[0..task_count]) |*task| run(task);
            }
        }
    }
}

// ---------------- Parallel tasks ----------------

const BlockedTask = struct {
    cd: []f32,
    ad: []const f32,
    b_panel: []const f32,
    a_panel: []f32,
    m: usize,
    n: usize,
    k: usize,
    jc: usize,
    nc_eff: usize,
    pc: usize,
    kc_eff: usize,
    mc: usize,
    num_j_chunks: usize,
    num_nr_panels: usize,
    cell_start: usize,
    cell_end: usize,
    accumulate: bool,
};

fn taskRunner(comptime orient: Orientation) fn (*const BlockedTask) void {
    return struct {
        fn run(task: *const BlockedTask) void {
            runBlockedTask(orient, task);
        }
    }.run;
}

fn runBlockedTask(comptime orient: Orientation, task: *const BlockedTask) void {
    const kc_eff = task.kc_eff;
    var packed_block: usize = std.math.maxInt(usize);
    var cell = task.cell_start;
    while (cell < task.cell_end) : (cell += 1) {
        const bi = cell / task.num_j_chunks;
        const ji = cell % task.num_j_chunks;
        const ic = bi * task.mc;
        const mc_eff = @min(task.mc, task.m - ic);
        if (bi != packed_block) {
            packA(orient, task.a_panel, task.ad, task.m, task.k, ic, mc_eff, task.pc, kc_eff);
            packed_block = bi;
        }

        const panel_start = ji * task.num_nr_panels / task.num_j_chunks;
        const panel_end = (ji + 1) * task.num_nr_panels / task.num_j_chunks;
        var jp = panel_start;
        while (jp < panel_end) : (jp += 1) {
            const jr = jp * nr;
            const nr_eff = @min(nr, task.nc_eff - jr);
            const b_sub = task.b_panel[jp * kc_eff * nr ..][0 .. kc_eff * nr];
            var ir: usize = 0;
            while (ir < mc_eff) : (ir += mr) {
                const mr_eff = @min(mr, mc_eff - ir);
                const a_sub = task.a_panel[(ir / mr) * kc_eff * mr ..][0 .. kc_eff * mr];
                const c_off = (ic + ir) * task.n + task.jc + jr;
                if (mr_eff == mr and nr_eff == nr) {
                    if (task.accumulate) {
                        microKernel(true, task.cd[c_off..], task.n, a_sub, b_sub, kc_eff);
                    } else {
                        microKernel(false, task.cd[c_off..], task.n, a_sub, b_sub, kc_eff);
                    }
                } else {
                    microKernelEdge(task.cd[c_off..], task.n, mr_eff, nr_eff, task.accumulate, a_sub, b_sub, kc_eff);
                }
            }
        }
    }
}

// ---------------- Microkernel ----------------

// C[0..mr, 0..nr] (+)= A~ * B~ over kc steps. Both panels are read
// unit-stride: A~ holds mr scalars per k-step, B~ holds nr scalars per
// k-step. `c` is the top-left of the tile inside the full row-major C with
// row stride `ldc`.
fn microKernel(
    comptime accumulate: bool,
    c: []f32,
    ldc: usize,
    a_panel: []const f32,
    b_panel: []const f32,
    kc: usize,
) void {
    var acc: [mr][nr_vecs]Vf32 = undefined;
    inline for (0..mr) |r| {
        inline for (0..nr_vecs) |v| acc[r][v] = @splat(0);
    }

    var p: usize = 0;
    while (p < kc) : (p += 1) {
        var b: [nr_vecs]Vf32 = undefined;
        inline for (0..nr_vecs) |v| {
            b[v] = b_panel[p * nr + v * vector_len ..][0..vector_len].*;
        }
        inline for (0..mr) |r| {
            const a: Vf32 = @splat(a_panel[p * mr + r]);
            inline for (0..nr_vecs) |v| {
                acc[r][v] = @mulAdd(Vf32, a, b[v], acc[r][v]);
            }
        }
    }

    inline for (0..mr) |r| {
        inline for (0..nr_vecs) |v| {
            const dst = c[r * ldc + v * vector_len ..][0..vector_len];
            if (accumulate) {
                dst.* = @as(Vf32, dst.*) + acc[r][v];
            } else {
                dst.* = acc[r][v];
            }
        }
    }
}

// Partial tiles: run the full microkernel into a stack tile (the panels are
// zero-padded to mr/nr, so the extra lanes are exact zeros), then merge only
// the valid mr_eff x nr_eff region into C.
fn microKernelEdge(
    c: []f32,
    ldc: usize,
    mr_eff: usize,
    nr_eff: usize,
    accumulate: bool,
    a_panel: []const f32,
    b_panel: []const f32,
    kc: usize,
) void {
    var tile: [mr * nr]f32 align(@alignOf(Vf32)) = undefined;
    microKernel(false, &tile, nr, a_panel, b_panel, kc);
    for (0..mr_eff) |r| {
        const src = tile[r * nr ..][0..nr];
        const dst = c[r * ldc ..];
        if (accumulate) {
            for (0..nr_eff) |j| dst[j] += src[j];
        } else {
            @memcpy(dst[0..nr_eff], src[0..nr_eff]);
        }
    }
}

// ---------------- Packing ----------------

// Serial packB below a work floor (dispatch overhead) and always on aarch64
// (historical behavior kept bit-and-schedule identical); otherwise the
// nr-panel ranges are packed by the team in parallel — the NT pack in
// particular is a strided scalar transpose that would otherwise serialize a
// meaningful slice of small-m GEMMs while every worker waits.
const pack_b_parallel_min_floats: usize = 64 * 1024;

const PackBTask = struct {
    dst: []f32,
    bd: []const f32,
    n: usize,
    k: usize,
    pc: usize,
    kc_eff: usize,
    jc: usize,
    nc_eff: usize,
};

fn packBTaskRunner(comptime orient: Orientation) fn (*const PackBTask) void {
    return struct {
        fn run(task: *const PackBTask) void {
            packB(orient, task.dst, task.bd, task.n, task.k, task.pc, task.kc_eff, task.jc, task.nc_eff);
        }
    }.run;
}

fn packBParallel(
    comptime orient: Orientation,
    bd: []const f32,
    n: usize,
    k: usize,
    pc: usize,
    kc_eff: usize,
    jc: usize,
    nc_eff: usize,
    num_nr_panels: usize,
    threads: usize,
    config: ParallelConfig,
) void {
    if (comptime !builtin.cpu.arch.isAARCH64()) {
        const pool = config.pool;
        if (pool != null and threads > 1 and kc_eff * nc_eff >= pack_b_parallel_min_floats) {
            const task_count = @max(@as(usize, 1), @min(threads, num_nr_panels));
            if (task_count > 1) {
                var tasks: [parallel.vector_max_threads]PackBTask = undefined;
                for (0..task_count) |ti| {
                    const panel_start = ti * num_nr_panels / task_count;
                    const panel_end = (ti + 1) * num_nr_panels / task_count;
                    const j0 = panel_start * nr;
                    tasks[ti] = .{
                        .dst = b_panel_storage[panel_start * kc_eff * nr ..],
                        .bd = bd,
                        .n = n,
                        .k = k,
                        .pc = pc,
                        .kc_eff = kc_eff,
                        .jc = jc + j0,
                        .nc_eff = @min(nc_eff - j0, (panel_end - panel_start) * nr),
                    };
                }
                pool.?.parallelChunks(PackBTask, tasks[0..task_count], comptime packBTaskRunner(orient));
                return;
            }
        }
    }
    packB(orient, &b_panel_storage, bd, n, k, pc, kc_eff, jc, nc_eff);
}

// B~ layout: ceil(nc_eff / nr) panels, each kc_eff x nr with element (p, c)
// at panel[p * nr + c]. Columns past nc_eff are zero-padded so edge tiles can
// run the full-width microkernel.
fn packB(
    comptime orient: Orientation,
    dst_storage: []f32,
    bd: []const f32,
    n: usize,
    k: usize,
    pc: usize,
    kc_eff: usize,
    jc: usize,
    nc_eff: usize,
) void {
    var j: usize = 0;
    while (j < nc_eff) : (j += nr) {
        const cols = @min(nr, nc_eff - j);
        const dst = dst_storage[(j / nr) * kc_eff * nr ..][0 .. kc_eff * nr];
        switch (orient) {
            // B is [k, n] row-major: panel slice p <- B[pc+p, jc+j .. +cols].
            .nn, .tn => {
                for (0..kc_eff) |p| {
                    const src = bd[(pc + p) * n + jc + j ..][0..cols];
                    const drow = dst[p * nr ..][0..nr];
                    @memcpy(drow[0..cols], src);
                    if (cols < nr) @memset(drow[cols..nr], 0);
                }
            },
            // TransB: B is [n, k] row-major; the pack absorbs the transpose
            // (contiguous reads along k, strided writes into the panel).
            .nt => {
                for (0..cols) |c| {
                    const src = bd[(jc + j + c) * k + pc ..][0..kc_eff];
                    for (0..kc_eff) |p| dst[p * nr + c] = src[p];
                }
                for (cols..nr) |c| {
                    for (0..kc_eff) |p| dst[p * nr + c] = 0;
                }
            },
        }
    }
}

// A~ layout: ceil(mc_eff / mr) panels, each kc_eff x mr with element (p, r)
// at panel[p * mr + r]. Rows past mc_eff are zero-padded.
fn packA(
    comptime orient: Orientation,
    dst_storage: []f32,
    ad: []const f32,
    m: usize,
    k: usize,
    ic: usize,
    mc_eff: usize,
    pc: usize,
    kc_eff: usize,
) void {
    var i: usize = 0;
    while (i < mc_eff) : (i += mr) {
        const rows = @min(mr, mc_eff - i);
        const dst = dst_storage[(i / mr) * kc_eff * mr ..][0 .. kc_eff * mr];
        switch (orient) {
            // A is [m, k] row-major: contiguous reads along k per row,
            // strided writes into the panel.
            .nn, .nt => {
                for (0..rows) |r| {
                    const src = ad[(ic + i + r) * k + pc ..][0..kc_eff];
                    for (0..kc_eff) |p| dst[p * mr + r] = src[p];
                }
                for (rows..mr) |r| {
                    for (0..kc_eff) |p| dst[p * mr + r] = 0;
                }
            },
            // TransA: A is [k, m] row-major; panel slice p <- A[pc+p,
            // ic+i .. +rows] — the transposed layout matches the panel layout
            // directly (contiguous copy per k-step).
            .tn => {
                for (0..kc_eff) |p| {
                    const src = ad[(pc + p) * m + ic + i ..][0..rows];
                    const drow = dst[p * mr ..][0..mr];
                    @memcpy(drow[0..rows], src);
                    if (rows < mr) @memset(drow[rows..mr], 0);
                }
            },
        }
    }
}

// ---------------- Tests ----------------

test {
    _ = @import("gemm_blocked_tests.zig");
}
