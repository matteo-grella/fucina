//! f16 row-block attention primitives of the subquadratic evaluator
//! (`subq.zig`): the register-blocked inner loops of online-softmax
//! attention over contiguous or strided f16 rows. Four rows per iteration
//! share the query's vector loads; the accumulator is loaded and stored
//! once per four contributions. Research kernels with one consumer, so
//! they live beside it rather than in the backend's primitives.

const std = @import("std");
const fucina = @import("fucina");

const primitives = fucina.internal.backend_mod.vector_impl.primitives;
const vexpf = fucina.simd.vexpf;
const widenF16x8 = primitives.widenF16x8;
const dotF32F16 = primitives.dotF32F16;

/// Reduce-max over a slice (softmax gauge candidate); -inf for empty input.
pub inline fn vecMaxReduce(x: []const f32) f32 {
    var m: f32 = -std.math.inf(f32);
    for (x) |value| m = @max(m, value);
    return m;
}

/// scores[i] = dot(query, rows[i * stride ..][0..query.len]) for f16 rows.
/// `stride` in elements between row starts (equal to query.len for packed
/// rows). Row-blocked by four.
pub fn scoreRows4F16(scores: []f32, query: []const f32, rows: []const f16, stride: usize) void {
    const V = @Vector(8, f32);
    const d = query.len;
    var i: usize = 0;
    while (i + 4 <= scores.len) : (i += 4) {
        const r0 = rows[i * stride ..];
        const r1 = rows[(i + 1) * stride ..];
        const r2 = rows[(i + 2) * stride ..];
        const r3 = rows[(i + 3) * stride ..];
        var a0: V = @splat(0);
        var a1: V = @splat(0);
        var a2: V = @splat(0);
        var a3: V = @splat(0);
        var b0: V = @splat(0);
        var b1: V = @splat(0);
        var b2: V = @splat(0);
        var b3: V = @splat(0);
        var j: usize = 0;
        while (j + 16 <= d) : (j += 16) {
            const q0: V = query[j..][0..8].*;
            const q1: V = query[j + 8 ..][0..8].*;
            a0 = @mulAdd(V, q0, widenF16x8(r0[j..]), a0);
            a1 = @mulAdd(V, q0, widenF16x8(r1[j..]), a1);
            a2 = @mulAdd(V, q0, widenF16x8(r2[j..]), a2);
            a3 = @mulAdd(V, q0, widenF16x8(r3[j..]), a3);
            b0 = @mulAdd(V, q1, widenF16x8(r0[j + 8 ..]), b0);
            b1 = @mulAdd(V, q1, widenF16x8(r1[j + 8 ..]), b1);
            b2 = @mulAdd(V, q1, widenF16x8(r2[j + 8 ..]), b2);
            b3 = @mulAdd(V, q1, widenF16x8(r3[j + 8 ..]), b3);
        }
        while (j + 8 <= d) : (j += 8) {
            const q0: V = query[j..][0..8].*;
            a0 = @mulAdd(V, q0, widenF16x8(r0[j..]), a0);
            a1 = @mulAdd(V, q0, widenF16x8(r1[j..]), a1);
            a2 = @mulAdd(V, q0, widenF16x8(r2[j..]), a2);
            a3 = @mulAdd(V, q0, widenF16x8(r3[j..]), a3);
        }
        var t0: f32 = @reduce(.Add, a0 + b0);
        var t1: f32 = @reduce(.Add, a1 + b1);
        var t2: f32 = @reduce(.Add, a2 + b2);
        var t3: f32 = @reduce(.Add, a3 + b3);
        while (j < d) : (j += 1) {
            const qj = query[j];
            t0 += qj * @as(f32, @floatCast(r0[j]));
            t1 += qj * @as(f32, @floatCast(r1[j]));
            t2 += qj * @as(f32, @floatCast(r2[j]));
            t3 += qj * @as(f32, @floatCast(r3[j]));
        }
        scores[i] = t0;
        scores[i + 1] = t1;
        scores[i + 2] = t2;
        scores[i + 3] = t3;
    }
    while (i < scores.len) : (i += 1) scores[i] = dotF32F16(query, rows[i * stride ..][0..d]);
}

/// In place: xs[i] = exp(scale * xs[i] + bias); returns the sum. The affine
/// form folds a softmax temperature and gauge shift into the exp pass.
pub fn vecExpAffineSumInPlace(xs: []f32, scale: f32, bias: f32) f64 {
    const V = @Vector(8, f32);
    const sv: V = @splat(scale);
    const bv: V = @splat(bias);
    var vsum: V = @splat(0);
    var i: usize = 0;
    while (i + 8 <= xs.len) : (i += 8) {
        const e = vexpf(8, @mulAdd(V, sv, @as(V, xs[i..][0..8].*), bv));
        xs[i..][0..8].* = e;
        vsum += e;
    }
    var total: f64 = @reduce(.Add, vsum);
    while (i < xs.len) : (i += 1) {
        const e = @exp(@mulAdd(f32, scale, xs[i], bias));
        xs[i] = e;
        total += e;
    }
    return total;
}

/// acc += sum_i w[i] * rows[i * stride ..][0..acc.len] for f16 rows,
/// row-blocked by four so the accumulator round-trips once per four rows.
pub fn weightedAccumRows4F16(acc: []f32, w: []const f32, rows: []const f16, stride: usize) void {
    const V = @Vector(8, f32);
    const d = acc.len;
    var i: usize = 0;
    while (i + 4 <= w.len) : (i += 4) {
        const w0: V = @splat(w[i]);
        const w1: V = @splat(w[i + 1]);
        const w2: V = @splat(w[i + 2]);
        const w3: V = @splat(w[i + 3]);
        const r0 = rows[i * stride ..];
        const r1 = rows[(i + 1) * stride ..];
        const r2 = rows[(i + 2) * stride ..];
        const r3 = rows[(i + 3) * stride ..];
        var j: usize = 0;
        while (j + 8 <= d) : (j += 8) {
            const c0 = @mulAdd(V, w1, widenF16x8(r1[j..]), w0 * widenF16x8(r0[j..]));
            const c1 = @mulAdd(V, w3, widenF16x8(r3[j..]), w2 * widenF16x8(r2[j..]));
            acc[j..][0..8].* = @as(V, acc[j..][0..8].*) + (c0 + c1);
        }
        while (j < d) : (j += 1) {
            acc[j] += w[i] * @as(f32, @floatCast(r0[j])) + w[i + 1] * @as(f32, @floatCast(r1[j])) +
                w[i + 2] * @as(f32, @floatCast(r2[j])) + w[i + 3] * @as(f32, @floatCast(r3[j]));
        }
    }
    while (i < w.len) : (i += 1) {
        const wi: V = @splat(w[i]);
        const row = rows[i * stride ..];
        var j: usize = 0;
        while (j + 8 <= d) : (j += 8) {
            acc[j..][0..8].* = @mulAdd(V, wi, widenF16x8(row[j..]), @as(V, acc[j..][0..8].*));
        }
        while (j < d) : (j += 1) acc[j] += w[i] * @as(f32, @floatCast(row[j]));
    }
}
