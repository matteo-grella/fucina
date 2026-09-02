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
//! The LHS quantization band (`lhsRows`, `LhsBlocks`, `quantizeLhsUnits`,
//! `quantizedLhsQ8_K`) is the one the native tensor-LHS entries and the
//! reference `scalar` namespace quantize activations through.

const std = @import("std");
const isa = @import("../isa.zig");
const dtype_mod = @import("../../dtype.zig");
const parallel = @import("../../parallel.zig");
const tensor = @import("../../tensor.zig");
const ops = @import("../ops.zig");
const quant = @import("../quant.zig");
const types = @import("../quant/types.zig");
const q8_0 = @import("../quant/q8_0.zig");
const thread = @import("../../thread.zig");
const common = @import("common.zig");
const tile = @import("tile.zig");

const ParallelConfig = common.ParallelConfig;
const Tensor = tensor.Tensor;

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
    if (comptime !isa.reference) {
        if (maybeParallelI8Blockwise(pc, out, qa, a_scales, qw, w_scales, m, n, k, group_size, num_groups)) return;
    }
    quant.matmulI8BlockwiseRange(out, qa, a_scales, qw, w_scales, m, n, k, group_size, num_groups, 0, m);
}

/// The one parallel 2-D quantized GEMM entry over a request `g`
/// (`ops.QuantGemm`): try the row/column task split (`splitPolicy(g)`
/// carries the format's lane grouping and gates), else run the serial full
/// tile through `quant.gemm`. The padded packed-Q8_0x4 form keeps its
/// bespoke columns-only splitter (its task shape has no row arm).
pub fn gemm2D(pc: ParallelConfig, comptime g: ops.QuantGemm, out: []f32, lhs: ops.LhsOf(g), rhs: ops.RhsOf(g), m: usize, n: usize, k: usize) void {
    // The reference arm: one serial full tile (`quant.gemm` over the whole
    // range; the accumulate bodies inside are already the `.scalar` tier).
    if (comptime isa.reference) return quant.gemm(g, out, lhs, rhs, ops.Tile.rows(m, n));
    if (comptime paddedQ8_0x4(g)) {
        if (maybeParallelQ8_0x4PackedPaddedRhs(pc, out, lhs, rhs, m, n, k)) return;
        return q8_0.matmulQ8_0x4PackedPaddedRhsRange(out, lhs, rhs, m, n);
    }
    const Parallel = QuantizedRhsParallel(g);
    if (Parallel.maybeParallel(pc, out, lhs, rhs, m, n, k)) return;
    quant.gemm(g, out, lhs, rhs, ops.Tile.rows(m, n));
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

// ---------------- The LHS quantization band ----------------
// Shared by the native provider's tensor-LHS entries (`backend/native.zig`)
// and the reference `scalar` namespace below.

pub fn checkedTensorProduct(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch tensor.TensorError.InvalidDataLength;
}

/// The `[rows, cols]` f32 activation behind an LHS tensor, validated once
/// at the dispatch tier (rank 2, the declared dims, contiguous) so the
/// allocation-free range quantizers run unchecked below it.
pub fn lhsRows(a: *const Tensor, rows: usize, cols: usize) ![]const f32 {
    const view = try a.rankView(2);
    if (view.dim(0) != rows or view.dim(1) != cols) return tensor.TensorError.ShapeMismatch;
    return try a.dataConstChecked();
}

/// Q8_0-family LHS block scratch: `count` blocks on the stack while they
/// fit below `q8_0_lhs_stack_blocks` (decode-sized activations), on the
/// heap past it. `var scratch: LhsBlocks(Block) = undefined;`, then
/// `acquire`/`release` around the use.
pub fn LhsBlocks(comptime Block: type) type {
    return struct {
        const Self = @This();
        stack: [parallel.q8_0_lhs_stack_blocks]Block,

        pub fn acquire(self: *Self, allocator: std.mem.Allocator, count: usize) ![]Block {
            return if (count <= self.stack.len) self.stack[0..count] else try allocator.alloc(Block, count);
        }

        pub fn release(self: *const Self, allocator: std.mem.Allocator, blocks: []Block) void {
            if (blocks.len > self.stack.len) allocator.free(blocks);
        }
    };
}

/// Row-range LHS quantization over the pool (`tile.forRange`'s
/// proportional split). `quantizeRange(blocks, data, rows, cols,
/// blocks_per_row, unit_start, unit_end)` quantizes units `[unit_start,
/// unit_end)` (rows, or the lane-packed formats' row groups) of the
/// `[rows, cols]` activation into `blocks`; units own disjoint blocks, so
/// any split produces the serial call's bytes. Gate: the fused
/// activation+quantization tasks' (`exec/quant_matmul.zig`) element
/// threshold, `rows * cols >= vector_elementwise_len_threshold / 8`; a
/// decode row never touches the pool, and a call without a team (or from
/// inside one, where `parallelChunks` degrades to the caller) runs the
/// serial walk. The reference build runs the serial walk (`refSerial`).
pub fn quantizeLhsUnits(
    pc: ParallelConfig,
    comptime Block: type,
    comptime quantizeRange: fn ([]Block, []const f32, usize, usize, usize, usize, usize) void,
    blocks: []Block,
    data: []const f32,
    rows: usize,
    cols: usize,
    blocks_per_row: usize,
    unit_count: usize,
) void {
    const Ctx = struct {
        blocks: []Block,
        data: []const f32,
        rows: usize,
        cols: usize,
        blocks_per_row: usize,

        fn run(c: @This(), unit_start: usize, unit_end: usize) void {
            quantizeRange(c.blocks, c.data, c.rows, c.cols, c.blocks_per_row, unit_start, unit_end);
        }
    };
    if (common.refSerial(pc).pool) |pool| {
        if (unit_count > 1 and rows * cols >= parallel.vector_elementwise_len_threshold / 8) {
            const task_count = @min(parallel.cpuThreadCount(parallel.vector_max_threads), unit_count);
            if (task_count > 1) {
                return tile.forRange(pool, Ctx, .{ .blocks = blocks, .data = data, .rows = rows, .cols = cols, .blocks_per_row = blocks_per_row }, unit_count, task_count, Ctx.run);
            }
        }
    }
    quantizeRange(blocks, data, rows, cols, blocks_per_row, 0, unit_count);
}

/// Q8_K row blocks of the `[rows, k]` activation `data`, allocated for the
/// caller (who frees them) and quantized through `quantizeLhsUnits`.
pub fn quantizedLhsQ8_K(pc: ParallelConfig, allocator: std.mem.Allocator, data: []const f32, rows: usize, k: usize) ![]dtype_mod.BlockQ8_K {
    const blocks_per_row = try quant.q8k.qkBlockCount(k);
    const blocks = try allocator.alloc(dtype_mod.BlockQ8_K, try types.checkedProduct(rows, blocks_per_row));
    errdefer allocator.free(blocks);
    quantizeLhsUnits(pc, dtype_mod.BlockQ8_K, quant.q8k.quantizeRowsQ8_KRangeInto, blocks, data, rows, k, blocks_per_row, rows);
    return blocks;
}

test {
    _ = @import("matmul_quant_tests.zig");
}

// ---------------- The scalar reference arms ----------------

/// The reference dispatch of the quantized GEMM entries: the plain row-form
/// requests, serial, no BLAS crossover, no x4-LHS lane packing, plus the two
/// bespoke reference kernels (the Q2_0 reference row twin and the TQ2_0
/// table-decoded Q8_K row). On `-Dbackend=scalar` builds the `native.zig`
/// entries dispatch here, so `-Dbackend=scalar` keeps meaning the serial
/// reference path end to end (the accumulate bodies inside `quant.gemm` are
/// already the `.scalar` tier there). On native builds these arms stay
/// reachable for `backend/parity_test.zig` and `bench/backend.zig`: they
/// then run the native tier's tile bodies under the serial reference
/// dispatch.
pub const scalar = struct {
    /// f32 [m, k] x a quantized RHS container -> f32 [m, n] over the request
    /// `g`: one serial LHS row quantization into `g.lhs`'s block form (Q8_0
    /// rows on the stack below `q8_0_lhs_stack_blocks`), then the one serial
    /// reference tile (`quant.gemm` over the full range).
    pub fn matmulQuantRows(
        comptime g: ops.QuantGemm,
        allocator: std.mem.Allocator,
        out: *Tensor,
        a: *const Tensor,
        rhs: ops.RhsOf(g),
        m: usize,
        n: usize,
        k: usize,
    ) !void {
        if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;
        const cd = (try out.dataChecked())[0 .. m * n];
        switch (comptime g.lhs) {
            .q8_0 => {
                const blocks_per_row = try quant.q8k.q8_0BlockCount(k);
                var scratch: LhsBlocks(dtype_mod.BlockQ8_0) = undefined;
                const qlhs_blocks = try scratch.acquire(allocator, m * blocks_per_row);
                defer scratch.release(allocator, qlhs_blocks);
                try quant.q8k.quantizeRowsQ8_0Into(qlhs_blocks, a);
                quant.gemm(g, cd, qlhs_blocks, rhs, ops.Tile.rows(m, n));
            },
            .q8_1 => {
                var qlhs = try quant.cold.quantizeRowsQ8_1(allocator, a);
                defer qlhs.deinit();
                quant.gemm(g, cd, qlhs.blocks, rhs, ops.Tile.rows(m, n));
            },
            .q8_k => {
                const qlhs = try quant.q8k.quantizeRowsQ8_K(allocator, a);
                defer allocator.free(qlhs);
                quant.gemm(g, cd, qlhs, rhs, ops.Tile.rows(m, n));
            },
            else => @compileError("matmulQuantRows: LHS form ." ++ @tagName(g.lhs) ++ " is quantized by the caller, not this entry"),
        }
    }

    /// The reference dispatch over `AnyQuantizedMatmulRhs`: Q2_0 takes the
    /// reference row twin, TQ2_0 rides the table-decoded Q8_K row, W8A8 the
    /// serial int8 kernel; everything else is the row-form request.
    pub fn matmul2DQuantizedRhs(
        allocator: std.mem.Allocator,
        out: *Tensor,
        a: *const Tensor,
        rhs: quant.AnyQuantizedMatmulRhs,
        m: usize,
        n: usize,
        k: usize,
    ) !void {
        const DType = dtype_mod.DType;
        return switch (rhs) {
            .fucina_w8a8_rhs => |qrhs| matmul2DQuantizedRhsI8(allocator, out, a, qrhs, m, n, k),
            .q2_0 => |qrhs| matmul2DQuantizedRhsQ2_0(allocator, out, a, qrhs, m, n, k),
            .tq2_0 => |qrhs| matmulTQ2_0TableRef(allocator, out, a, qrhs, m, n, k),
            inline else => |qrhs, tag| matmulQuantRows(comptime ops.QuantGemm.rowsFor(@field(DType, @tagName(tag))), allocator, out, a, qrhs, m, n, k),
        };
    }

    /// The serial int8 W8A8 arm: per-row activation quantization, then the
    /// blockwise kernel over the full range.
    pub fn matmul2DQuantizedRhsI8(
        allocator: std.mem.Allocator,
        out: *Tensor,
        a: *const Tensor,
        rhs: *const quant.QuantizedMatmulRhsI8,
        m: usize,
        n: usize,
        k: usize,
    ) !void {
        if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;

        const a_len = try checkedTensorProduct(m, k);
        const out_len = try checkedTensorProduct(m, n);
        const ad = (try a.dataConstChecked())[0..a_len];
        const cd = (try out.dataChecked())[0..out_len];

        const qa = try allocator.alloc(i8, a_len);
        defer allocator.free(qa);
        const a_scales = try allocator.alloc(f32, m);
        defer allocator.free(a_scales);

        quant.quantizeActivationsPerRowI8(qa, a_scales, ad, m, k);
        quant.matmulI8BlockwiseRange(cd, qa, a_scales, rhs.qw.dataConst(), rhs.scales.dataConst(), m, n, k, rhs.group_size, rhs.num_groups, 0, m);
    }

    /// The TQ2_0 reference: the table-decoded Q8_K reference row
    /// (deliberate: one reference decode serves the whole Q8_K table family;
    /// the parity gate pins it against the native direct kernel).
    pub fn matmulTQ2_0TableRef(
        allocator: std.mem.Allocator,
        out: *Tensor,
        a: *const Tensor,
        rhs: *const quant.QuantizedMatmulRhsTQ2_0,
        m: usize,
        n: usize,
        k: usize,
    ) !void {
        if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;
        const cd = (try out.dataChecked())[0 .. m * n];
        const qlhs = try quant.q8k.quantizeRowsQ8_K(allocator, a);
        defer allocator.free(qlhs);
        quant.cold.matmulTableQ8_KRhsRange(.tq2_0, cd, qlhs, rhs, m, n, 0, m);
    }

    /// The Q2_0 reference: the reference row twin
    /// (`matmulQ2_0RhsRefRange`), not the fast ternary kernel.
    pub fn matmul2DQuantizedRhsQ2_0(
        allocator: std.mem.Allocator,
        out: *Tensor,
        a: *const Tensor,
        rhs: *const quant.QuantizedMatmulRhsQ2_0,
        m: usize,
        n: usize,
        k: usize,
    ) !void {
        if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;
        const cd = (try out.dataChecked())[0 .. m * n];
        const blocks_per_row = try quant.q8k.q8_0BlockCount(k);
        var scratch: LhsBlocks(dtype_mod.BlockQ8_0) = undefined;
        const qlhs_blocks = try scratch.acquire(allocator, m * blocks_per_row);
        defer scratch.release(allocator, qlhs_blocks);
        try quant.q8k.quantizeRowsQ8_0Into(qlhs_blocks, a);
        quant.cold.matmulQ2_0RhsRefRange(cd, qlhs_blocks, rhs, m, n, 0, m);
    }

    /// The reference arm of the packed-container GEMM entry for the
    /// quantized containers: every lane pack maps back onto its row-form
    /// request (the dense f32 panel arm lives beside `packDenseRhs` in
    /// `backend/native.zig`).
    pub fn matmulPacked(
        allocator: std.mem.Allocator,
        out: anytype,
        a: anytype,
        rhs: anytype,
        m: usize,
        n: usize,
        k: usize,
    ) !void {
        const Rhs = @TypeOf(rhs.*);
        return matmulQuantRows(comptime .{ .weight = Rhs.dtype, .rhs = Rhs.pack, .lhs = if (Rhs.dtype == .q8_0) .q8_0 else .q8_k }, allocator, out, a, rhs, m, n, k);
    }
};
