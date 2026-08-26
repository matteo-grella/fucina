//! Quantized matmul dispatch: the one parallel 2-D entry `gemm2D` over an
//! `ops.QuantGemm` request, and the QuantizedRhsParallel generic — one
//! Task/run/spawn/maybeParallel body whose SplitPolicy (derived from the
//! request by `splitPolicy`) carries each format's lane grouping and split
//! gates. Two arms keep bespoke splitters for their different task shapes:
//! the i8-blockwise path and the padded packed-Q8_0x4 form (columns only).
//! The tile bodies are the `quant/` children's `gemm` entries (`q8_0`,
//! `q4_k`, `q5_k`, `q6_k`, `ternary`, `cold`) and the W8A8 kernel in
//! `quant.zig`; the shared parallel gates (ParallelConfig,
//! i8ColumnThreadCount / matmulThreadCount) come from `common.zig`.

const std = @import("std");
const parallel = @import("../../parallel.zig");
const ops = @import("../ops.zig");
const quant = @import("../quant.zig");
const types = @import("../quant/types.zig");
const q8_0 = @import("../quant/q8_0.zig");
const thread = @import("../../thread.zig");
const common = @import("common.zig");

const ParallelConfig = common.ParallelConfig;

// ---------------- Quantized MatMul (2-D) ----------------

pub fn matmul2DI8BlockwiseInto(
    pc: ParallelConfig,
    out: []f32,
    qa: []const i8,
    a_scales: []const f32,
    qw: []const i8,
    w_scales: []const f32,
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
    num_groups: usize,
) void {
    if (maybeParallelI8Blockwise(pc, out, qa, a_scales, qw, w_scales, m, n, k, group_size, num_groups)) return;
    quant.matmulI8BlockwiseRange(out, qa, a_scales, qw, w_scales, m, n, k, group_size, num_groups, 0, m);
}

/// The one parallel 2-D quantized GEMM entry over a request `g`
/// (`ops.QuantGemm`): try the row/column task split (`splitPolicy(g)`
/// carries the format's lane grouping and gates), else run the serial full
/// tile through `quant.gemm`. The padded packed-Q8_0x4 form keeps its
/// bespoke columns-only splitter (its task shape has no row arm).
pub fn gemm2D(pc: ParallelConfig, comptime g: ops.QuantGemm, out: []f32, lhs: ops.LhsOf(g), rhs: ops.RhsOf(g), m: usize, n: usize, k: usize) void {
    if (comptime paddedQ8_0x4(g)) {
        if (maybeParallelQ8_0x4PackedPaddedRhs(pc, out, lhs, rhs, m, n, k)) return;
        return q8_0.matmulQ8_0x4PackedPaddedRhsRange(out, lhs, rhs, m, n);
    }
    const Parallel = QuantizedRhsParallel(g);
    if (Parallel.maybeParallel(pc, out, lhs, rhs, m, n, k)) return;
    quant.gemm(g, out, lhs, rhs, ops.Tile.rows(m, n));
}

/// Deprecated spelling of `gemm2D(.{ .weight = .tq2_0, .lhs = .f32 })`
/// kept for the autograd fallback caller; new callers state the request.
pub fn matmul2DTQ2_0F32RhsInto(
    pc: ParallelConfig,
    out: []f32,
    lhs: []const f32,
    rhs: *const types.QuantizedMatmulRhsTQ2_0,
    m: usize,
    n: usize,
    k: usize,
) void {
    gemm2D(pc, .{ .weight = .tq2_0, .lhs = .f32 }, out, lhs, rhs, m, n, k);
}

/// The padded packed-Q8_0x4 form (masked writes, `m % 4` free).
fn paddedQ8_0x4(comptime g: ops.QuantGemm) bool {
    return g.weight == .q8_0 and g.rhs == .x4 and g.lhs == .q8_0x4 and g.order == .col_outer;
}

const I8BlockwiseTask = struct {
    out: []f32,
    qa: []const i8,
    a_scales: []const f32,
    qw: []const i8,
    w_scales: []const f32,
    n: usize,
    k: usize,
    group_size: usize,
    num_groups: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
};

const Q8_0x4PackedPaddedRhsTask = struct {
    out: []f32,
    lhs_blocks: []const types.BlockQ8_0x4,
    rhs: *const types.QuantizedMatmulRhsQ8_0x4,
    m: usize,
    n: usize,
    c0: usize,
    c1: usize,
};

fn runI8BlockwiseTask(task: *const I8BlockwiseTask) void {
    quant.matmulI8BlockwiseTile(task.out, task.qa, task.a_scales, task.qw, task.w_scales, task.n, task.k, task.group_size, task.num_groups, task.r0, task.r1, task.c0, task.c1);
}

fn runQ8_0x4PackedPaddedRhsTask(task: *const Q8_0x4PackedPaddedRhsTask) void {
    quant.gemm(.{ .weight = .q8_0, .rhs = .x4, .lhs = .q8_0x4, .order = .col_outer }, task.out, task.lhs_blocks, task.rhs, .{ .r1 = task.m, .c0 = task.c0, .c1 = task.c1 });
}

fn spawnI8BlockwiseTasks(pool: *thread.Pool, tasks: []I8BlockwiseTask) void {
    pool.parallelChunks(I8BlockwiseTask, tasks, runI8BlockwiseTask);
}

fn spawnQ8_0x4PackedPaddedRhsTasks(pool: *thread.Pool, tasks: []Q8_0x4PackedPaddedRhsTask) void {
    pool.parallelChunks(Q8_0x4PackedPaddedRhsTask, tasks, runQ8_0x4PackedPaddedRhsTask);
}

/// Row/column split policy for `QuantizedRhsParallel`. The default is the
/// plain-block behavior; the interleaved packs round task boundaries to
/// their lane group so a task never bisects a packed column or row group.
const SplitPolicy = struct {
    /// Column-task boundaries are multiples of this (the pack's column
    /// lane group).
    col_group: usize = 1,
    /// When the column arm may fire (the row arm handles the rest).
    col_gate: enum {
        /// Decode-sized m only (the plain-block rule).
        small_m,
        /// Twice the m window: the x8/x2 tiles amortize more per column.
        small_m_x2,
        /// Also at m >= 128 (packed-LHS wide batches favor columns).
        small_m_or_m128,
        /// Columns whenever the count gate allows (lane-packed LHS).
        always,
    } = .small_m,
    /// Row-task boundaries are multiples of this (the packed LHS's row
    /// group).
    row_group: usize = 1,
    /// Row-group rounding: `.floor` never emits a partial group (those
    /// callers guarantee `m % row_group == 0`); `.ceil_clamp` covers a
    /// ragged tail by clamping the last boundary to `m`.
    row_round: enum { floor, ceil_clamp } = .floor,
    /// Cap the row-arm thread count at the group count, bailing to the
    /// serial path when no full group exists.
    row_cap_groups: bool = false,
    /// At m >= 128 the x8 tiles are LHS-bandwidth bound: 3 threads
    /// saturate, more only adds barrier cost.
    row_cap_3_at_m128: bool = false,
};

/// The split policy of a request: lane groups derive from the pack, the
/// gates carry each kernel family's measured splits (transcribed from the
/// per-name instantiations this table replaced).
fn splitPolicy(comptime g: ops.QuantGemm) SplitPolicy {
    return switch (g.weight) {
        .q8_0 => switch (g.rhs) {
            .rows => .{},
            .x4 => switch (g.lhs) {
                .q8_0 => .{ .col_group = 4 },
                .q8_0x4 => .{ .col_group = 4, .col_gate = .small_m_or_m128, .row_group = 4, .row_cap_groups = true },
                else => unreachable,
            },
            else => unreachable,
        },
        .q4_k => switch (g.rhs) {
            .rows => .{},
            .x4 => .{ .col_group = 4 },
            .x8 => if (g.lhs == .q8_kx4)
                .{ .col_group = 8, .col_gate = .always, .row_group = 4, .row_round = .ceil_clamp }
            else
                .{ .col_group = 8, .col_gate = .small_m_x2, .row_cap_3_at_m128 = true },
            .x2mmla => if (g.lhs == .q8_kx2mmla)
                .{ .col_group = 2, .col_gate = .small_m_x2, .row_group = 2 }
            else
                .{ .col_group = 2, .col_gate = .small_m_x2 },
        },
        .q5_k => switch (g.rhs) {
            .rows => .{},
            .x8 => if (g.lhs == .q8_kx4)
                .{ .col_group = 8, .col_gate = .small_m_x2, .row_group = 4 }
            else
                .{ .col_group = 8, .col_gate = .small_m_x2, .row_cap_3_at_m128 = true },
            else => unreachable,
        },
        .q6_k => switch (g.rhs) {
            .rows => .{},
            .x4 => .{ .col_group = 4 },
            else => unreachable,
        },
        else => .{},
    };
}

fn QuantizedRhsParallel(comptime g: ops.QuantGemm) type {
    const policy = splitPolicy(g);
    return struct {
        const Self = @This();

        const Task = struct {
            out: []f32,
            lhs_blocks: ops.LhsOf(g),
            rhs: ops.RhsOf(g),
            n: usize,
            r0: usize,
            r1: usize,
            c0: usize,
            c1: usize,
        };

        fn run(task: *const Task) void {
            quant.gemm(g, task.out, task.lhs_blocks, task.rhs, .{ .r0 = task.r0, .r1 = task.r1, .c0 = task.c0, .c1 = task.c1 });
        }

        fn spawn(pool: *thread.Pool, tasks: []Task) void {
            pool.parallelChunks(Task, tasks, Self.run);
        }

        fn maybeParallel(
            pc: ParallelConfig,
            out: []f32,
            lhs_blocks: ops.LhsOf(g),
            rhs: ops.RhsOf(g),
            m: usize,
            n: usize,
            k: usize,
        ) bool {
            const pool = pc.pool orelse return false;
            const row_groups = switch (policy.row_round) {
                .floor => m / policy.row_group,
                .ceil_clamp => (m + policy.row_group - 1) / policy.row_group,
            };
            if (policy.row_cap_groups and row_groups == 0) return false;

            var tasks: [parallel.vector_max_threads]Task = undefined;
            const base: Task = .{
                .out = out,
                .lhs_blocks = lhs_blocks,
                .rhs = rhs,
                .n = n,
                .r0 = 0,
                .r1 = m,
                .c0 = 0,
                .c1 = n,
            };

            const col_arm = switch (policy.col_gate) {
                .small_m => m < parallel.vector_column_min_m,
                .small_m_x2 => m < parallel.vector_column_min_m * 2,
                .small_m_or_m128 => m < parallel.vector_column_min_m or m >= 128,
                .always => true,
            };
            if (col_arm) {
                const col_threads = common.i8ColumnThreadCount(m, n, k);
                if (col_threads != 1) {
                    const group_count = n / policy.col_group;
                    for (0..col_threads) |ti| {
                        tasks[ti] = base;
                        tasks[ti].c0 = policy.col_group * (ti * group_count / col_threads);
                        tasks[ti].c1 = policy.col_group * ((ti + 1) * group_count / col_threads);
                    }
                    Self.spawn(pool, tasks[0..col_threads]);
                    return true;
                }
            }

            const requested_threads = common.matmulThreadCount(m, n, k, parallel.vector_matmul_work_threshold);
            var thread_count = if (policy.row_cap_3_at_m128 and m >= 128)
                @min(requested_threads, @as(usize, 3))
            else
                requested_threads;
            if (policy.row_cap_groups) thread_count = @min(thread_count, row_groups);
            if (thread_count == 1) return false;

            for (0..thread_count) |ti| {
                tasks[ti] = base;
                tasks[ti].r0 = policy.row_group * (ti * row_groups / thread_count);
                const r1 = policy.row_group * ((ti + 1) * row_groups / thread_count);
                tasks[ti].r1 = if (policy.row_round == .ceil_clamp) @min(m, r1) else r1;
            }
            Self.spawn(pool, tasks[0..thread_count]);
            return true;
        }
    };
}

fn maybeParallelI8Blockwise(
    pc: ParallelConfig,
    out: []f32,
    qa: []const i8,
    a_scales: []const f32,
    qw: []const i8,
    w_scales: []const f32,
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
    num_groups: usize,
) bool {
    const pool = pc.pool orelse return false;
    var tasks: [parallel.vector_max_threads]I8BlockwiseTask = undefined;
    const base: I8BlockwiseTask = .{
        .out = out,
        .qa = qa,
        .a_scales = a_scales,
        .qw = qw,
        .w_scales = w_scales,
        .n = n,
        .k = k,
        .group_size = group_size,
        .num_groups = num_groups,
        .r0 = 0,
        .r1 = m,
        .c0 = 0,
        .c1 = n,
    };

    // Small m (e.g. decode): row splitting would leave one task with all the
    // work, so split output columns instead. Mirrors the float TN/NT paths, but
    // with a lower work gate: the int8 kernel is heavier per element than the
    // float GEMV, so column parallelism pays off at decode-sized work.
    if (m < parallel.vector_column_min_m) {
        const col_threads = common.i8ColumnThreadCount(m, n, k);
        if (col_threads != 1) {
            for (0..col_threads) |ti| {
                tasks[ti] = base;
                tasks[ti].c0 = ti * n / col_threads;
                tasks[ti].c1 = (ti + 1) * n / col_threads;
            }
            spawnI8BlockwiseTasks(pool, tasks[0..col_threads]);
            return true;
        }
    }

    const thread_count = common.matmulThreadCount(m, n, k, parallel.vector_matmul_work_threshold);
    if (thread_count == 1) return false;

    for (0..thread_count) |ti| {
        tasks[ti] = base;
        tasks[ti].r0 = ti * m / thread_count;
        tasks[ti].r1 = (ti + 1) * m / thread_count;
    }
    spawnI8BlockwiseTasks(pool, tasks[0..thread_count]);
    return true;
}

fn maybeParallelQ8_0x4PackedPaddedRhs(
    pc: ParallelConfig,
    out: []f32,
    lhs_blocks: []const types.BlockQ8_0x4,
    rhs: *const types.QuantizedMatmulRhsQ8_0x4,
    m: usize,
    n: usize,
    k: usize,
) bool {
    const pool = pc.pool orelse return false;
    const col_threads = common.i8ColumnThreadCount(m, n, k);
    if (col_threads == 1) return false;

    var tasks: [parallel.vector_max_threads]Q8_0x4PackedPaddedRhsTask = undefined;
    const base: Q8_0x4PackedPaddedRhsTask = .{
        .out = out,
        .lhs_blocks = lhs_blocks,
        .rhs = rhs,
        .m = m,
        .n = n,
        .c0 = 0,
        .c1 = n,
    };
    const group_count = n / 4;
    for (0..col_threads) |ti| {
        tasks[ti] = base;
        tasks[ti].c0 = 4 * (ti * group_count / col_threads);
        tasks[ti].c1 = 4 * ((ti + 1) * group_count / col_threads);
    }
    spawnQ8_0x4PackedPaddedRhsTasks(pool, tasks[0..col_threads]);
    return true;
}

test {
    _ = @import("matmul_quant_tests.zig");
}
