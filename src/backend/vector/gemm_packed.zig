//! Register-tiled dense GEMM for load-time f32 RHS panels.
//!
//! Computes `C[m,n] = A[m,k] * W[n,k]^T`. SIMD lanes run along k, so one
//! output-row panel and several A rows stay live in registers. AVX2's sixteen
//! vector registers fit a 3x4 accumulator tile; aarch64's thirty-two fit 6x4.
//! Parallel work is split over the wide output dimension, which keeps skinny-m
//! inference shapes from starving a row-parallel team.

const std = @import("std");
const builtin = @import("builtin");
const parallel = @import("../../parallel.zig");
const thread = @import("../../thread.zig");
const common = @import("common.zig");

const ParallelConfig = common.ParallelConfig;
const Vf32 = common.Vf32;
const vector_len = common.vector_len;
const output_tile = 4;
const max_input_tile = if (builtin.cpu.arch == .aarch64) 6 else 3;
const tasks_per_participant = 3;
const max_tasks = parallel.vector_max_threads * tasks_per_participant;

const Task = struct {
    out: []f32,
    lhs: []const f32,
    rhs: []const f32,
    m: usize,
    n: usize,
    k: usize,
    col_start: usize,
    col_end: usize,
};

pub fn gemmPackedNtIntoWithConfig(
    out: []f32,
    lhs: []const f32,
    rhs: []const f32,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) void {
    if (m == 0 or n == 0) return;
    if (k == 0) {
        @memset(out[0 .. m * n], 0);
        return;
    }

    const pool = config.pool orelse {
        gemmPackedNtCols(out, lhs, rhs, m, n, k, 0, n);
        return;
    };
    const participants = common.columnThreadCount(m, n, k);
    if (participants == 1) {
        gemmPackedNtCols(out, lhs, rhs, m, n, k, 0, n);
        return;
    }

    const panels = std.math.divCeil(usize, n, output_tile) catch unreachable;
    const task_count = @min(panels, participants * tasks_per_participant);
    var tasks: [max_tasks]Task = undefined;
    for (0..task_count) |ti| {
        const p0 = ti * panels / task_count;
        const p1 = (ti + 1) * panels / task_count;
        tasks[ti] = .{
            .out = out,
            .lhs = lhs,
            .rhs = rhs,
            .m = m,
            .n = n,
            .k = k,
            .col_start = p0 * output_tile,
            .col_end = @min(n, p1 * output_tile),
        };
    }
    pool.parallelChunks(Task, tasks[0..task_count], runTask);
}

fn runTask(task: *const Task) void {
    gemmPackedNtCols(task.out, task.lhs, task.rhs, task.m, task.n, task.k, task.col_start, task.col_end);
}

fn gemmPackedNtCols(
    out: []f32,
    lhs: []const f32,
    rhs: []const f32,
    m: usize,
    n: usize,
    k: usize,
    col_start: usize,
    col_end: usize,
) void {
    // Keep one four-weight panel hot while every input-row tile consumes it.
    // The opposite loop order re-streams a multi-megabyte task range once per
    // row tile and collapses on k=9600 after the range exceeds private cache.
    var col = col_start;
    while (col + output_tile <= col_end) : (col += output_tile) {
        gemmPackedNtRowsForCols(out, lhs, rhs, m, n, k, col, col + output_tile);
    }
    while (col < col_end) : (col += 1) {
        gemmPackedNtRowsForCols(out, lhs, rhs, m, n, k, col, col + 1);
    }
}

fn gemmPackedNtRowsForCols(
    out: []f32,
    lhs: []const f32,
    rhs: []const f32,
    m: usize,
    n: usize,
    k: usize,
    col_start: usize,
    col_end: usize,
) void {
    var row: usize = 0;
    if (comptime max_input_tile == 6) {
        // Avoid a 6+1 split: 4+3 uses fewer single-row tail instructions.
        while (row + 6 <= m and m - (row + 6) != 1) : (row += 6) {
            microTile(6, out, lhs, rhs, n, k, row, col_start, col_end);
        }
        if (m - row >= 4) {
            microTile(4, out, lhs, rhs, n, k, row, col_start, col_end);
            row += 4;
        }
        if (m - row >= 3) {
            microTile(3, out, lhs, rhs, n, k, row, col_start, col_end);
            row += 3;
        }
    } else {
        while (row + 3 <= m and m - row != 4) : (row += 3) {
            microTile(3, out, lhs, rhs, n, k, row, col_start, col_end);
        }
        if (m - row == 4) {
            microTile(2, out, lhs, rhs, n, k, row, col_start, col_end);
            microTile(2, out, lhs, rhs, n, k, row + 2, col_start, col_end);
            row += 4;
        }
    }
    if (m - row >= 2) {
        microTile(2, out, lhs, rhs, n, k, row, col_start, col_end);
        row += 2;
    }
    if (row < m) microTile(1, out, lhs, rhs, n, k, row, col_start, col_end);
}

inline fn microTile(
    comptime rows: usize,
    out: []f32,
    lhs: []const f32,
    rhs: []const f32,
    n: usize,
    k: usize,
    row: usize,
    col_start: usize,
    col_end: usize,
) void {
    var col = col_start;
    while (col + output_tile <= col_end) : (col += output_tile) {
        var acc: [rows][output_tile]Vf32 = undefined;
        inline for (0..rows) |r| {
            inline for (0..output_tile) |c| acc[r][c] = @splat(0);
        }

        var p: usize = 0;
        while (p + vector_len <= k) : (p += vector_len) {
            var av: [rows]Vf32 = undefined;
            inline for (0..rows) |r| av[r] = lhs[(row + r) * k + p ..][0..vector_len].*;
            inline for (0..output_tile) |c| {
                const bv: Vf32 = rhs[(col + c) * k + p ..][0..vector_len].*;
                inline for (0..rows) |r| acc[r][c] = @mulAdd(Vf32, av[r], bv, acc[r][c]);
            }
        }

        var sums: [rows][output_tile]f32 = undefined;
        inline for (0..rows) |r| {
            inline for (0..output_tile) |c| sums[r][c] = @reduce(.Add, acc[r][c]);
        }
        while (p < k) : (p += 1) {
            inline for (0..rows) |r| {
                const a = lhs[(row + r) * k + p];
                inline for (0..output_tile) |c| sums[r][c] += a * rhs[(col + c) * k + p];
            }
        }
        inline for (0..rows) |r| {
            inline for (0..output_tile) |c| out[(row + r) * n + col + c] = sums[r][c];
        }
    }

    while (col < col_end) : (col += 1) {
        var acc: [rows]Vf32 = undefined;
        inline for (0..rows) |r| acc[r] = @splat(0);
        var p: usize = 0;
        while (p + vector_len <= k) : (p += vector_len) {
            const bv: Vf32 = rhs[col * k + p ..][0..vector_len].*;
            inline for (0..rows) |r| {
                const av: Vf32 = lhs[(row + r) * k + p ..][0..vector_len].*;
                acc[r] = @mulAdd(Vf32, av, bv, acc[r]);
            }
        }
        var sums: [rows]f32 = undefined;
        inline for (0..rows) |r| sums[r] = @reduce(.Add, acc[r]);
        while (p < k) : (p += 1) {
            const b = rhs[col * k + p];
            inline for (0..rows) |r| sums[r] += lhs[(row + r) * k + p] * b;
        }
        inline for (0..rows) |r| out[(row + r) * n + col] = sums[r];
    }
}

test "packed dense microkernel tails agree with scalar" {
    const testing = std.testing;
    const m = 7;
    const n = 9;
    const k = vector_len + 3;
    var lhs: [m * k]f32 = undefined;
    var rhs: [n * k]f32 = undefined;
    for (&lhs, 0..) |*x, i| x.* = @as(f32, @floatFromInt(@as(isize, @intCast(i % 13)) - 6)) / 7;
    for (&rhs, 0..) |*x, i| x.* = @as(f32, @floatFromInt(@as(isize, @intCast(i % 11)) - 5)) / 9;
    var got: [m * n]f32 = undefined;
    var want: [m * n]f32 = undefined;
    gemmPackedNtIntoWithConfig(&got, &lhs, &rhs, m, n, k, .{});
    for (0..m) |i| for (0..n) |j| {
        var sum: f32 = 0;
        for (0..k) |p| sum += lhs[i * k + p] * rhs[j * k + p];
        want[i * n + j] = sum;
    };
    for (got, want) |a, b| try testing.expectApproxEqAbs(b, a, 2e-6);
}
