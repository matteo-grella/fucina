//! PTQTP — Post-Training Quantization to Trit-Planes (arXiv:2509.16989):
//! data-free decomposition of a weight matrix into K ∈ {1,2,3} ternary
//! planes with one scale per length-G column group,
//!
//!     W ≈ Σ_k diag(α_k)·T_k,   T_k ∈ {-1,0,+1},
//!
//! solved per independent group by alternating (a) a closed-form KxK ridge
//! regression for the scales given the planes — λ escalates x10 from
//! `lambda0` while the Frobenius condition estimate κ exceeds `kappa_max`
//! (essential at init, where all planes start at sign(w) and the
//! unregularized system is singular), clamped at `lambda_max` — and (b) an
//! exhaustive 3^K-way per-element search for the trit tuple given the
//! scales. A group converges when its scale vector moves less than
//! `epsilon` between iterations; groups are fully independent, so the
//! per-group check equals the paper's global max check and the row fan-out
//! below is bitwise identical for any thread count.
//!
//! Implemented from the paper's formulas (Algorithm 1, Eq 3–10; the paper
//! is K = 2 — K = 3 is this port's capacity extension for sensitive
//! tensors); no public reference implementation existed at porting time.
//! Deliberate deltas, recorded in docs/PTQTP.md:
//!
//! - The packed group size is 256 — the TQ2_0 block width — so each plane
//!   is a byte-valid standalone TQ2_0 tensor whose per-block fp16 `d` IS
//!   the group scale: a decorated matmul is K stock ternary matmuls plus
//!   adds, no new kernels. The paper uses G = 128 with unpacked storage;
//!   `reconstructReference` runs any G for that fidelity comparison.
//! - The elementwise search enumerates candidates in a pinned order (zero
//!   tuple first, then sparser before denser with later — finer — planes
//!   first) with strictly-less keeps-first: exact ties prefer sparser
//!   trits, and the fixed order breaks the symmetric sign(w) init
//!   deterministically (mid-magnitude elements land their first ±1 in the
//!   last plane, the planes diverge, and the next ridge step separates the
//!   scales into coarse/fine).
//! - Packing takes |α| (the candidate value set is invariant under scale
//!   sign flips, so trits re-derive losslessly), rounds it to fp16, and
//!   re-runs one search against the rounded scales — the stored trits are
//!   elementwise-optimal for the exact scales inference multiplies by.
//!
//! Non-finite weights: NaN/inf elements are excluded from the scale
//! regression and forced to trit 0 in every plane — one bad element
//! degrades only itself instead of poisoning its group (same stance as the
//! TQ2_0 encoder's NaN clamp).

const std = @import("std");
const backend_mod = @import("backend.zig");
const exec_mod = @import("exec.zig");
const parallel = @import("parallel.zig");

const Allocator = std.mem.Allocator;
const ExecContext = exec_mod.ExecContext;
const quant = backend_mod.quantized_matmul;

pub const BlockTQ2_0 = backend_mod.BlockTQ2_0;
pub const Rhs = backend_mod.QuantizedMatmulRhsTQ2_0;

/// Logical elements per TQ2_0 block — the packed group size.
pub const block_len: usize = 256;

pub const Error = error{ InvalidShape, InvalidOptions };

pub const Options = struct {
    /// Trit-plane count K. 2 = the paper's dual decomposition; 1 = single
    /// plane (ridge scale + 3-way search — a least-squares upgrade over the
    /// blind absmean b1.58 encoder, and the "purged to one plane" mode);
    /// 3 = an extra residual plane (27 representable levels per group —
    /// high-rate distortion bound ~3x below dual — at +2.06 bpw) for
    /// tensors whose sensitivity earns it.
    planes: u8 = 2,
    /// Column-group size: each length-G row segment gets its own scale per
    /// plane. `quantizeMatrix` requires 256 (the packable size); other
    /// values are for `reconstructReference` fidelity studies.
    group_size: usize = block_len,
    /// Iteration cap per group. The paper reports convergence within 50
    /// everywhere it measured, with diminishing returns past ~30.
    max_iterations: usize = 50,
    /// Convergence bound on the per-iteration scale movement ‖Δα‖.
    epsilon: f32 = 1e-4,
    /// Adaptive ridge floor / ceiling and the condition-number trigger.
    lambda0: f32 = 1e-8,
    lambda_max: f32 = 1.0,
    kappa_max: f32 = 1e6,

    /// Scale-tied fit: lock the plane scales to the exact ratio 3
    /// (alpha = [3s, s] at K=2, [9s, 3s, s] at K=3), which makes the K trit
    /// planes one uniform symmetric (3^K)-level quantizer with step s —
    /// codes c = 3*t1 + t2 (+9*...) in {-L..L}, L = (3^K-1)/2 — and
    /// therefore makes the plane-folding identity exact: one combined dot
    /// pass can compute all K planes. Costs fit freedom vs the free ridge
    /// (the free levels are non-uniformly placeable); the quality delta is
    /// the measured experiment this option exists for (docs/PTQTP.md).
    /// Note the stored per-plane f16 scales round independently, so a
    /// future folded kernel must derive the coarser scales from the finest
    /// one in f32 (exact: 3x an f16 value is exact in f32), not re-read
    /// the rounded f16 pair. Meaningless at planes = 1 (rejected).
    tie_scales: bool = false,

    fn validate(self: Options) Error!void {
        if (self.planes < 1 or self.planes > 3) return Error.InvalidOptions;
        if (self.tie_scales and self.planes < 2) return Error.InvalidOptions;
        if (self.group_size == 0 or self.max_iterations == 0) return Error.InvalidOptions;
        if (!(self.epsilon >= 0) or !(self.lambda0 > 0)) return Error.InvalidOptions;
        if (!(self.lambda_max >= self.lambda0) or !(self.kappa_max >= 1)) return Error.InvalidOptions;
    }
};

pub const GroupResult = struct {
    /// Scales as solved (sign included; entries past `planes` are 0).
    /// The returned trits are the argmin against exactly these scales.
    alpha: [3]f32,
    iterations: usize,
    converged: bool,
};

/// Aggregate quantization diagnostics over one matrix.
pub const MatrixStats = struct {
    /// ‖W − Ŵ‖_F / ‖W‖_F over the finite elements (0 when ‖W‖ = 0), with Ŵ
    /// the reconstruction inference will actually use (fp16-rounded scales).
    rel_frob_err: f64,
    /// Fraction of zero trits per plane (entries past `planes` are 0).
    zero_frac: [3]f64,
    mean_iterations: f64,
    /// Groups that hit `max_iterations` without meeting `epsilon`.
    unconverged_groups: usize,
    group_count: usize,
};

/// The packed TQ2_0 planes of one weight matrix, row-major `[rows]` weight
/// rows of `cols/256` blocks each (planes past the built count are empty).
/// Each plane is a standalone valid TQ2_0 tensor; the decorated product is
/// the sum of the per-plane products.
pub const PlanePair = struct {
    plane1: []BlockTQ2_0,
    plane2: []BlockTQ2_0,
    plane3: []BlockTQ2_0,
    /// Weight rows n (output features) and columns k (contract dim).
    rows: usize,
    cols: usize,
    stats: MatrixStats,

    pub fn planeCount(self: *const PlanePair) usize {
        var count: usize = 1;
        if (self.plane2.len != 0) count += 1;
        if (self.plane3.len != 0) count += 1;
        return count;
    }

    /// Borrowed matmul view of one plane (0..2): the pair keeps ownership
    /// and must outlive the view. View row c = weight row c = output col c.
    pub fn rhs(self: *const PlanePair, plane: usize) !Rhs {
        const blocks = switch (plane) {
            0 => self.plane1,
            1 => self.plane2,
            2 => self.plane3,
            else => return Error.InvalidShape,
        };
        if (blocks.len == 0) return Error.InvalidShape;
        return quant.quantizedMatmulRhsTQ2_0FromBorrowedBlocks(self.cols, self.rows, blocks);
    }

    /// Dequantized sum of the planes into `dst` (row-major rows x cols) —
    /// the exact Ŵ the stats above measured.
    pub fn reconstructInto(self: *const PlanePair, dst: []f32) Error!void {
        if (dst.len != self.rows * self.cols) return Error.InvalidShape;
        @memset(dst, 0);
        addPlaneInto(dst, self.plane1, self.cols);
        if (self.plane2.len != 0) addPlaneInto(dst, self.plane2, self.cols);
        if (self.plane3.len != 0) addPlaneInto(dst, self.plane3, self.cols);
    }

    pub fn deinit(self: *PlanePair, allocator: Allocator) void {
        allocator.free(self.plane3);
        allocator.free(self.plane2);
        allocator.free(self.plane1);
        self.* = undefined;
    }
};

/// Candidate trit tuples in the pinned tie-break order (see the header):
/// zero first, then by support size with later (finer) planes first,
/// lexicographic within equal support. The third component is 0 throughout
/// for the K = 2 set, which keeps the K = 2 enumeration identical to the
/// original dual-plane order.
const combos_k1 = [3][3]i8{
    .{ 0, 0, 0 }, .{ -1, 0, 0 }, .{ 1, 0, 0 },
};
const combos_k2 = [9][3]i8{
    .{ 0, 0, 0 },   .{ 0, -1, 0 }, .{ 0, 1, 0 },  .{ -1, 0, 0 }, .{ 1, 0, 0 },
    .{ -1, -1, 0 }, .{ -1, 1, 0 }, .{ 1, -1, 0 }, .{ 1, 1, 0 },
};
const combos_k3 = [27][3]i8{
    .{ 0, 0, 0 },
    .{ 0, 0, -1 },   .{ 0, 0, 1 },
    .{ 0, -1, 0 },   .{ 0, 1, 0 },
    .{ -1, 0, 0 },   .{ 1, 0, 0 },
    .{ 0, -1, -1 },  .{ 0, -1, 1 },  .{ 0, 1, -1 },  .{ 0, 1, 1 },
    .{ -1, 0, -1 },  .{ -1, 0, 1 },  .{ 1, 0, -1 },  .{ 1, 0, 1 },
    .{ -1, -1, 0 },  .{ -1, 1, 0 },  .{ 1, -1, 0 },  .{ 1, 1, 0 },
    .{ -1, -1, -1 }, .{ -1, -1, 1 }, .{ -1, 1, -1 }, .{ -1, 1, 1 },
    .{ 1, -1, -1 },  .{ 1, -1, 1 },  .{ 1, 1, -1 },  .{ 1, 1, 1 },
};

fn combosFor(comptime planes: u2) []const [3]i8 {
    return comptime switch (planes) {
        1 => &combos_k1,
        2 => &combos_k2,
        3 => &combos_k3,
        else => unreachable,
    };
}

/// Solve one group: `w` and the trit slices for planes below
/// `options.planes` must share one length; slices for unused planes may be
/// empty and are never touched. Trit outputs are in {-1, 0, +1}. Pure and
/// deterministic; no allocation.
pub fn solveGroup(w: []const f32, t1: []i8, t2: []i8, t3: []i8, options: Options) GroupResult {
    std.debug.assert(options.planes >= 1 and options.planes <= 3);
    std.debug.assert(t1.len == w.len);
    std.debug.assert(options.planes < 2 or t2.len == w.len);
    std.debug.assert(options.planes < 3 or t3.len == w.len);
    const ts = [3][]i8{ t1, t2, t3 };
    if (options.tie_scales) {
        return switch (options.planes) {
            2 => solveGroupTied(2, w, ts),
            else => solveGroupTied(3, w, ts),
        };
    }
    return switch (options.planes) {
        1 => solveGroupK(1, w, ts, options),
        2 => solveGroupK(2, w, ts, options),
        else => solveGroupK(3, w, ts, options),
    };
}

/// The tie_scales fit (see Options.tie_scales): the optimal uniform
/// symmetric (3^K)-level quantizer per group. Sweep the step s around
/// absmax/L with exact-code MSE, keep the argmin, then decompose each code
/// into balanced base-3 trit digits. Non-finite inputs quantize to 0, like
/// the free fit's finite-element stance.
fn solveGroupTied(comptime planes: u2, w: []const f32, ts: [3][]i8) GroupResult {
    const levels: f32 = if (planes == 2) 4 else 13; // (3^K - 1) / 2
    var absmax: f32 = 0;
    for (w) |x| {
        if (std.math.isFinite(x)) absmax = @max(absmax, @abs(x));
    }
    if (absmax == 0) {
        for (0..planes) |p| @memset(ts[p][0..w.len], 0);
        return .{ .alpha = .{ 0, 0, 0 }, .iterations = 1, .converged = true };
    }

    var best_s: f32 = absmax / levels;
    var best_err: f64 = std.math.inf(f64);
    var i: usize = 0;
    while (i <= 64) : (i += 1) {
        const divisor = (levels - 0.75) + @as(f32, @floatFromInt(i)) * (2.0 / 64.0);
        const s = absmax / divisor;
        var err: f64 = 0;
        for (w) |x| {
            if (!std.math.isFinite(x)) continue;
            const c = std.math.clamp(@round(x / s), -levels, levels);
            const d = x - s * c;
            err += @as(f64, d) * d;
        }
        if (err < best_err) {
            best_err = err;
            best_s = s;
        }
    }

    for (w, 0..) |x, j| {
        var c: i32 = if (std.math.isFinite(x))
            @intFromFloat(std.math.clamp(@round(x / best_s), -levels, levels))
        else
            0;
        if (planes == 3) {
            const t1: i32 = @divFloor(c + 4, 9); // c in {-13..13} -> t1 in {-1,0,1}
            ts[0][j] = @intCast(t1);
            c -= 9 * t1;
        }
        const tm: i32 = @divFloor(c + 1, 3); // c in {-4..4} -> {-1,0,1}
        ts[planes - 2][j] = @intCast(tm);
        ts[planes - 1][j] = @intCast(c - 3 * tm);
    }

    return .{
        .alpha = if (planes == 2)
            .{ 3 * best_s, best_s, 0 }
        else
            .{ 9 * best_s, 3 * best_s, best_s },
        .iterations = 1,
        .converged = true,
    };
}

fn solveGroupK(comptime planes: u2, w: []const f32, ts: [3][]i8, options: Options) GroupResult {
    for (w, 0..) |wj, j| {
        const s: i8 = if (wj > 0) 1 else if (wj < 0) -1 else 0;
        inline for (0..planes) |p| ts[p][j] = s;
    }

    var alpha = [3]f32{ 1, if (planes >= 2) 1 else 0, if (planes == 3) 1 else 0 };
    var iterations: usize = 0;
    var converged = false;
    while (iterations < options.max_iterations) {
        iterations += 1;
        const prev = alpha;

        // Scales step: α = (SᵀS + λI)⁻¹ Sᵀw with integer Gram entries.
        var s_diag = [3]i64{ 0, 0, 0 };
        var s12: i64 = 0;
        var s13: i64 = 0;
        var s23: i64 = 0;
        var b = [3]f64{ 0, 0, 0 };
        for (w, 0..) |wj, j| {
            const finite = std.math.isFinite(wj);
            inline for (0..planes) |p| {
                const c = ts[p][j];
                if (c != 0) {
                    s_diag[p] += 1;
                    if (finite) b[p] += if (c > 0) @as(f64, wj) else -@as(f64, wj);
                }
            }
            if (planes >= 2) s12 += @as(i64, ts[0][j]) * @as(i64, ts[1][j]);
            if (planes == 3) {
                s13 += @as(i64, ts[0][j]) * @as(i64, ts[2][j]);
                s23 += @as(i64, ts[1][j]) * @as(i64, ts[2][j]);
            }
        }

        var lambda: f64 = options.lambda0;
        if (planes == 1) {
            alpha[0] = @floatCast(b[0] / (@as(f64, @floatFromInt(s_diag[0])) + lambda));
        } else if (planes == 2) {
            var a11: f64 = undefined;
            var a22: f64 = undefined;
            var det: f64 = undefined;
            const a12: f64 = @floatFromInt(s12);
            while (true) {
                a11 = @as(f64, @floatFromInt(s_diag[0])) + lambda;
                a22 = @as(f64, @floatFromInt(s_diag[1])) + lambda;
                det = a11 * a22 - a12 * a12; // λ > 0 keeps A positive definite
                const frob2 = a11 * a11 + a22 * a22 + 2 * a12 * a12;
                const kappa = frob2 / det;
                if (kappa <= options.kappa_max or lambda >= options.lambda_max) break;
                lambda = @min(lambda * 10, options.lambda_max);
            }
            alpha[0] = @floatCast((b[0] * a22 - b[1] * a12) / det);
            alpha[1] = @floatCast((b[1] * a11 - b[0] * a12) / det);
        } else {
            // 3x3 symmetric solve via the adjugate; κ_F = ‖A‖_F·‖adj‖_F/det
            // (for the SPD A here, ‖A⁻¹‖_F = ‖adj(A)‖_F / det).
            const a12: f64 = @floatFromInt(s12);
            const a13: f64 = @floatFromInt(s13);
            const a23: f64 = @floatFromInt(s23);
            var adj11: f64 = undefined;
            var adj12: f64 = undefined;
            var adj13: f64 = undefined;
            var adj22: f64 = undefined;
            var adj23: f64 = undefined;
            var adj33: f64 = undefined;
            var det: f64 = undefined;
            while (true) {
                const a11 = @as(f64, @floatFromInt(s_diag[0])) + lambda;
                const a22 = @as(f64, @floatFromInt(s_diag[1])) + lambda;
                const a33 = @as(f64, @floatFromInt(s_diag[2])) + lambda;
                adj11 = a22 * a33 - a23 * a23;
                adj12 = -(a12 * a33 - a23 * a13);
                adj13 = a12 * a23 - a22 * a13;
                adj22 = a11 * a33 - a13 * a13;
                adj23 = -(a11 * a23 - a12 * a13);
                adj33 = a11 * a22 - a12 * a12;
                det = a11 * adj11 + a12 * adj12 + a13 * adj13;
                const frob_a = @sqrt(a11 * a11 + a22 * a22 + a33 * a33 + 2 * (a12 * a12 + a13 * a13 + a23 * a23));
                const frob_adj = @sqrt(adj11 * adj11 + adj22 * adj22 + adj33 * adj33 + 2 * (adj12 * adj12 + adj13 * adj13 + adj23 * adj23));
                const kappa = frob_a * frob_adj / det;
                if (kappa <= options.kappa_max or lambda >= options.lambda_max) break;
                lambda = @min(lambda * 10, options.lambda_max);
            }
            alpha[0] = @floatCast((adj11 * b[0] + adj12 * b[1] + adj13 * b[2]) / det);
            alpha[1] = @floatCast((adj12 * b[0] + adj22 * b[1] + adj23 * b[2]) / det);
            alpha[2] = @floatCast((adj13 * b[0] + adj23 * b[1] + adj33 * b[2]) / det);
        }

        // Topology step against the fresh scales.
        searchTrits(planes, w, ts, alpha);

        var move2: f64 = 0;
        inline for (0..planes) |p| {
            const d = @as(f64, alpha[p]) - @as(f64, prev[p]);
            move2 += d * d;
        }
        if (@sqrt(move2) < options.epsilon) {
            converged = true;
            break;
        }
    }
    return .{ .alpha = alpha, .iterations = iterations, .converged = converged };
}

/// Per-element exhaustive argmin over the candidate values α·c in the
/// pinned combo order (strictly-less keeps the earlier candidate).
fn searchTrits(comptime planes: u2, w: []const f32, ts: [3][]i8, alpha: [3]f32) void {
    const candidates = comptime combosFor(planes);

    var vals: [candidates.len]f32 = undefined;
    inline for (candidates, 0..) |c, i| {
        var v = alpha[0] * @as(f32, @floatFromInt(c[0]));
        if (planes >= 2) v += alpha[1] * @as(f32, @floatFromInt(c[1]));
        if (planes == 3) v += alpha[2] * @as(f32, @floatFromInt(c[2]));
        vals[i] = v;
    }
    for (w, 0..) |wj, j| {
        if (!std.math.isFinite(wj)) {
            inline for (0..planes) |p| ts[p][j] = 0;
            continue;
        }
        var best: usize = 0;
        var best_err = square(wj - vals[0]);
        inline for (1..candidates.len) |i| {
            const e = square(wj - vals[i]);
            if (e < best_err) {
                best = i;
                best_err = e;
            }
        }
        inline for (0..planes) |p| ts[p][j] = candidates[best][p];
    }
}

inline fn square(x: f32) f32 {
    return x * x;
}

/// Quantize a dense row-major [n][k] f32 matrix into packed trit-planes.
/// Requires `options.group_size == 256` (the packable size) and k % 256 == 0;
/// rows fan out over the ExecContext worker team with bitwise-identical
/// results for any thread count (disjoint outputs, per-row stats reduced in
/// row order). The caller owns the returned pair (deinit with the same
/// ExecContext allocator).
pub fn quantizeMatrix(ctx: *ExecContext, weights: []const f32, n: usize, k: usize, options: Options) !PlanePair {
    try options.validate();
    if (options.group_size != block_len) return Error.InvalidOptions;
    if (n == 0 or k == 0 or k % block_len != 0) return Error.InvalidShape;
    const total = std.math.mul(usize, n, k) catch return Error.InvalidShape;
    if (weights.len != total) return Error.InvalidShape;

    const allocator = ctx.allocator;
    const blocks_per_row = k / block_len;
    const plane1 = try allocator.alloc(BlockTQ2_0, n * blocks_per_row);
    errdefer allocator.free(plane1);
    const plane2 = try allocator.alloc(BlockTQ2_0, if (options.planes >= 2) n * blocks_per_row else 0);
    errdefer allocator.free(plane2);
    const plane3 = try allocator.alloc(BlockTQ2_0, if (options.planes == 3) n * blocks_per_row else 0);
    errdefer allocator.free(plane3);
    const row_stats = try allocator.alloc(RowStat, n);
    defer allocator.free(row_stats);

    const range_ctx = RangeCtx{
        .weights = weights,
        .plane1 = plane1,
        .plane2 = plane2,
        .plane3 = plane3,
        .row_stats = row_stats,
        .cols = k,
        .blocks_per_row = blocks_per_row,
        .options = options,
    };
    runRowRanges(ctx, n, blocks_per_row, range_ctx);

    // Serial, row-ordered reduce keeps the diagnostics partition-invariant.
    var err_sum: f64 = 0;
    var w2_sum: f64 = 0;
    var zeros = [3]u64{ 0, 0, 0 };
    var iters_sum: u64 = 0;
    var unconverged: usize = 0;
    for (row_stats) |stat| {
        err_sum += stat.err;
        w2_sum += stat.w2;
        inline for (0..3) |p| zeros[p] += stat.zeros[p];
        iters_sum += stat.iters;
        unconverged += stat.unconverged;
    }
    const group_count = n * blocks_per_row;
    const elems: f64 = @floatFromInt(total);
    return .{
        .plane1 = plane1,
        .plane2 = plane2,
        .plane3 = plane3,
        .rows = n,
        .cols = k,
        .stats = .{
            .rel_frob_err = if (w2_sum > 0) @sqrt(err_sum / w2_sum) else 0,
            .zero_frac = .{
                @as(f64, @floatFromInt(zeros[0])) / elems,
                @as(f64, @floatFromInt(zeros[1])) / elems,
                @as(f64, @floatFromInt(zeros[2])) / elems,
            },
            .mean_iterations = @as(f64, @floatFromInt(iters_sum)) / @as(f64, @floatFromInt(group_count)),
            .unconverged_groups = unconverged,
            .group_count = group_count,
        },
    };
}

/// Reference/fidelity path: solve at an arbitrary group size (e.g. the
/// paper's 128) keeping exact f32 scales, and write the reconstruction Ŵ
/// into `dst`. Serial, unpacked — for ablation against the packed pipeline
/// (comparing at G = 256 also isolates the fp16 scale-rounding cost).
pub fn reconstructReference(allocator: Allocator, weights: []const f32, n: usize, k: usize, options: Options, dst: []f32) !MatrixStats {
    try options.validate();
    if (n == 0 or k == 0 or k % options.group_size != 0) return Error.InvalidShape;
    const total = std.math.mul(usize, n, k) catch return Error.InvalidShape;
    if (weights.len != total or dst.len != total) return Error.InvalidShape;

    const g = options.group_size;
    const t1 = try allocator.alloc(i8, g);
    defer allocator.free(t1);
    const t2 = try allocator.alloc(i8, if (options.planes >= 2) g else 0);
    defer allocator.free(t2);
    const t3 = try allocator.alloc(i8, if (options.planes == 3) g else 0);
    defer allocator.free(t3);

    var err_sum: f64 = 0;
    var w2_sum: f64 = 0;
    var zeros = [3]u64{ 0, 0, 0 };
    var iters_sum: u64 = 0;
    var unconverged: usize = 0;
    const groups_per_row = k / g;
    for (0..n) |r| {
        for (0..groups_per_row) |gi| {
            const seg = weights[r * k + gi * g ..][0..g];
            const out = dst[r * k + gi * g ..][0..g];
            const res = solveGroup(seg, t1, t2, t3, options);
            for (seg, 0..) |wj, j| {
                var rec = res.alpha[0] * @as(f32, @floatFromInt(t1[j]));
                if (options.planes >= 2) rec += res.alpha[1] * @as(f32, @floatFromInt(t2[j]));
                if (options.planes == 3) rec += res.alpha[2] * @as(f32, @floatFromInt(t3[j]));
                out[j] = rec;
                if (t1[j] == 0) zeros[0] += 1;
                if (options.planes >= 2 and t2[j] == 0) zeros[1] += 1;
                if (options.planes == 3 and t3[j] == 0) zeros[2] += 1;
                if (!std.math.isFinite(wj)) continue;
                const e = @as(f64, wj) - @as(f64, rec);
                err_sum += e * e;
                w2_sum += @as(f64, wj) * @as(f64, wj);
            }
            iters_sum += res.iterations;
            if (!res.converged) unconverged += 1;
        }
    }
    const group_count = n * groups_per_row;
    const elems: f64 = @floatFromInt(total);
    return .{
        .rel_frob_err = if (w2_sum > 0) @sqrt(err_sum / w2_sum) else 0,
        .zero_frac = .{
            @as(f64, @floatFromInt(zeros[0])) / elems,
            @as(f64, @floatFromInt(zeros[1])) / elems,
            @as(f64, @floatFromInt(zeros[2])) / elems,
        },
        .mean_iterations = @as(f64, @floatFromInt(iters_sum)) / @as(f64, @floatFromInt(group_count)),
        .unconverged_groups = unconverged,
        .group_count = group_count,
    };
}

// ---------------- packed pipeline internals ----------------

const RowStat = struct {
    err: f64 = 0,
    w2: f64 = 0,
    zeros: [3]u64 = .{ 0, 0, 0 },
    iters: u64 = 0,
    unconverged: u32 = 0,
};

const RangeCtx = struct {
    weights: []const f32,
    plane1: []BlockTQ2_0,
    plane2: []BlockTQ2_0,
    plane3: []BlockTQ2_0,
    row_stats: []RowStat,
    cols: usize,
    blocks_per_row: usize,
    options: Options,
};

fn runRowRanges(ctx: *ExecContext, n: usize, blocks_per_row: usize, range_ctx: RangeCtx) void {
    if (n * blocks_per_row >= 8) {
        if (ctx.workPool()) |pool| {
            const task_count = @min(parallel.cpuThreadCount(parallel.vector_max_threads), n);
            if (task_count > 1) {
                const Task = struct {
                    context: RangeCtx,
                    start: usize,
                    end: usize,
                    fn run(task: *const @This()) void {
                        quantizeRowRange(task.context, task.start, task.end);
                    }
                };
                var tasks: [parallel.vector_max_threads]Task = undefined;
                for (0..task_count) |i| {
                    tasks[i] = .{
                        .context = range_ctx,
                        .start = i * n / task_count,
                        .end = (i + 1) * n / task_count,
                    };
                }
                pool.parallelChunks(Task, tasks[0..task_count], Task.run);
                return;
            }
        }
    }
    quantizeRowRange(range_ctx, 0, n);
}

fn quantizeRowRange(ctx: RangeCtx, r0: usize, r1: usize) void {
    var t1: [block_len]i8 = undefined;
    var t2: [block_len]i8 = undefined;
    var t3: [block_len]i8 = undefined;
    const ts = [3][]i8{ &t1, &t2, &t3 };
    var r = r0;
    while (r < r1) : (r += 1) {
        var stat = RowStat{};
        for (0..ctx.blocks_per_row) |g| {
            const seg = ctx.weights[r * ctx.cols + g * block_len ..][0..block_len];
            const bi = r * ctx.blocks_per_row + g;
            switch (ctx.options.planes) {
                1 => packGroup(1, seg, ts, ctx.options, .{ &ctx.plane1[bi], undefined, undefined }, &stat),
                2 => packGroup(2, seg, ts, ctx.options, .{ &ctx.plane1[bi], &ctx.plane2[bi], undefined }, &stat),
                else => packGroup(3, seg, ts, ctx.options, .{ &ctx.plane1[bi], &ctx.plane2[bi], &ctx.plane3[bi] }, &stat),
            }
        }
        ctx.row_stats[r] = stat;
    }
}

fn packGroup(
    comptime planes: u2,
    seg: *const [block_len]f32,
    ts: [3][]i8,
    options: Options,
    outs: [3]*BlockTQ2_0,
    stat: *RowStat,
) void {
    const res = if (comptime planes >= 2) blk: {
        if (options.tie_scales) break :blk solveGroupTied(planes, seg, ts);
        break :blk solveGroupK(planes, seg, ts, options);
    } else solveGroupK(planes, seg, ts, options);
    // |α| loses nothing (the candidate set is sign-symmetric); fp16-round,
    // then re-derive the trits against the exact scales inference will use.
    var alpha = [3]f32{ 0, 0, 0 };
    inline for (0..planes) |p| alpha[p] = roundScaleF16(@abs(res.alpha[p]));
    searchTrits(planes, seg, ts, alpha);
    inline for (0..planes) |p| packBlock(outs[p], ts[p][0..block_len], alpha[p]);

    for (seg, 0..) |wj, j| {
        inline for (0..planes) |p| {
            if (ts[p][j] == 0) stat.zeros[p] += 1;
        }
        if (!std.math.isFinite(wj)) continue;
        var rec: f32 = 0;
        inline for (0..planes) |p| rec += alpha[p] * @as(f32, @floatFromInt(ts[p][j]));
        const e = @as(f64, wj) - @as(f64, rec);
        stat.err += e * e;
        stat.w2 += @as(f64, wj) * @as(f64, wj);
    }
    stat.iters += res.iterations;
    if (!res.converged) stat.unconverged += 1;
}

/// α as fp16 will store it: non-negative, clamped below f16 inf. NaN cannot
/// reach here (non-finite weights never enter the scale regression).
fn roundScaleF16(a: f32) f32 {
    const clamped = @min(a, 65504.0);
    const h: f16 = @floatCast(clamped);
    return @floatCast(h);
}

/// TQ2_0 crumb layout (matches the ternary encoder/kernels): byte m of
/// half-group hg (qs[hg*32 + m]) holds elements hg*128 + n*32 + m in bits
/// (2n+1):2n as code = trit + 1.
fn packBlock(block: *BlockTQ2_0, trits: *const [block_len]i8, d: f32) void {
    const h: f16 = @floatCast(d);
    block.d = @bitCast(h);
    for (0..2) |hg| {
        for (0..32) |m| {
            var q: u8 = 0;
            inline for (0..4) |n| {
                const code: u8 = @intCast(trits[hg * 128 + n * 32 + m] + 1);
                q |= code << (2 * n);
            }
            block.qs[hg * 32 + m] = q;
        }
    }
}

fn addPlaneInto(dst: []f32, blocks: []const BlockTQ2_0, cols: usize) void {
    const blocks_per_row = cols / block_len;
    for (blocks, 0..) |*block, bi| {
        const row = bi / blocks_per_row;
        const gi = bi % blocks_per_row;
        const out = dst[row * cols + gi * block_len ..][0..block_len];
        const h: f16 = @bitCast(block.d);
        const d: f32 = @floatCast(h);
        for (0..2) |hg| {
            for (0..32) |m| {
                const q = block.qs[hg * 32 + m];
                inline for (0..4) |n| {
                    const code: i32 = (q >> (2 * n)) & 3;
                    out[hg * 128 + n * 32 + m] += d * @as(f32, @floatFromInt(code - 1));
                }
            }
        }
    }
}

test {
    _ = @import("ptqtp_tests.zig");
}
