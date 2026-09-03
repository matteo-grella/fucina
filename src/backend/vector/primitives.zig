//! Low-level @Vector primitives (dot/add/mul cores, transcendental vector
//! bodies, the f16 row-block attention inner loops). The shared core
//! (V* aliases, vector_len*) comes from `common.zig`.
//!
//! Transcendental pairing: every exp-family lane body evaluates ONE
//! `vexpf` (`.exp`, `.sigmoid`, `.silu`, `.softplus`, `.tanh`, `.gelu`,
//! `.geglu`, `.situ`, the logit softcap), so the forward lanes and the
//! unary-VJP lanes (`unaryDerivativeVec`, which is `vexpf`-based too)
//! agree to `vexpf`'s accuracy; the sub-vector tails use libm and sit
//! within the same 2e-6 of the lanes. The exceptions are the byte-parity
//! ops `gelu_quant` (ggml's f16-LUT table) and `elu` (ggml_vec_elu_f32):
//! their lanes ride `vtanhf`/`vexpm1f`, faithful musl ports whose every
//! lane reproduces the scalar `std.math.tanh`/`std.math.expm1` bytes
//! (`primitives_tests.zig` sweeps them bit-for-bit).

const std = @import("std");
const ops = @import("../ops.zig");
const dtype_mod = @import("../../dtype.zig");
const common = @import("common.zig");

const DType = dtype_mod.DType;

const vector_len = common.vector_len;
const vector_len_f64 = common.vector_len_f64;
const vector_len_f16 = common.vector_len_f16;
const Vf32 = common.Vf32;
const Vf64 = common.Vf64;
const Vf16ForF32 = common.Vf16ForF32;

/// One typed elementwise op over a scalar pair: cast to the compute dtype,
/// apply `op`, cast to the output dtype.
pub fn applyElementwiseTyped(
    comptime dtype: DType,
    comptime op: ops.ElementwiseOp,
    a: dtype_mod.Scalar(dtype),
    b: dtype_mod.Scalar(dtype),
) dtype_mod.Scalar(dtype_mod.outputDType(.pointwise, dtype)) {
    const compute_dtype = comptime dtype_mod.computeDType(.pointwise, dtype);
    const output_dtype = comptime dtype_mod.outputDType(.pointwise, dtype);
    const av = dtype_mod.castFloat(dtype, compute_dtype, a);
    const bv = dtype_mod.castFloat(dtype, compute_dtype, b);
    const out = switch (op) {
        .add => av + bv,
        .sub => av - bv,
        .mul => av * bv,
        .div => av / bv,
        .max => if (av != av or bv != bv) std.math.nan(@TypeOf(av)) else @max(av, bv),
        .min => if (av != av or bv != bv) std.math.nan(@TypeOf(av)) else @min(av, bv),
    };
    return dtype_mod.castFloat(compute_dtype, output_dtype, out);
}

// ---------------- Vector primitives ----------------

/// Lane vector of the typed binary loop: f32/f64/f16 at their own widths
/// (f16 lanes compute in f16, the dtype's pointwise compute dtype), bf16
/// bits at the f32 width (widened to f32 lanes and rounded back).
fn BinaryLanes(comptime dtype: DType) type {
    return switch (dtype) {
        .f32 => Vf32,
        .f64 => Vf64,
        .f16 => common.Vf16,
        .bf16 => common.Vu16ForF32,
        else => @compileError("vecBinary: no vector lanes for " ++ @tagName(dtype)),
    };
}

/// torch.maximum/minimum on a vector: NaN in either lane propagates NaN
/// (bare @max/@min follow IEEE maxNum and would drop it).
pub inline fn applyMaxMinVec(comptime V: type, comptime op: ops.ElementwiseOp, a: V, b: V) V {
    const Elem = @typeInfo(V).vector.child;
    const raw = if (comptime op == .max) @max(a, b) else @min(a, b);
    const nan_v: V = @splat(std.math.nan(Elem));
    return @select(Elem, a != a, nan_v, @select(Elem, b != b, nan_v, raw));
}

pub inline fn applyElementwiseVec(comptime V: type, comptime op: ops.ElementwiseOp, a: V, b: V) V {
    return switch (op) {
        .add => a + b,
        .sub => a - b,
        .mul => a * b,
        .div => a / b,
        .max, .min => applyMaxMinVec(V, op, a, b),
    };
}

/// `op` on one lane vector of `dtype` storage.
inline fn binaryLanes(comptime dtype: DType, comptime op: ops.ElementwiseOp, a: BinaryLanes(dtype), b: BinaryLanes(dtype)) BinaryLanes(dtype) {
    if (comptime dtype == .bf16) return f32VecToBf16(applyElementwiseVec(Vf32, op, bf16VecToF32(a), bf16VecToF32(b)));
    return applyElementwiseVec(BinaryLanes(dtype), op, a, b);
}

/// The one typed binary elementwise loop: four lane vectors per iteration,
/// then single vectors, then the scalar tail through
/// `applyElementwiseTyped`.
pub inline fn vecBinary(
    comptime dtype: DType,
    comptime op: ops.ElementwiseOp,
    z: []dtype_mod.Scalar(dtype),
    x: []const dtype_mod.Scalar(dtype),
    y: []const dtype_mod.Scalar(dtype),
) void {
    const Lanes = BinaryLanes(dtype);
    const L = @typeInfo(Lanes).vector.len;
    var i: usize = 0;
    while (i + 4 * L <= z.len) : (i += 4 * L) {
        var xs: [4]Lanes = undefined;
        var ys: [4]Lanes = undefined;
        inline for (0..4) |u| {
            xs[u] = x[i + u * L ..][0..L].*;
            ys[u] = y[i + u * L ..][0..L].*;
        }
        inline for (0..4) |u| z[i + u * L ..][0..L].* = binaryLanes(dtype, op, xs[u], ys[u]);
    }
    while (i + L <= z.len) : (i += L) {
        z[i..][0..L].* = binaryLanes(dtype, op, x[i..][0..L].*, y[i..][0..L].*);
    }
    while (i < z.len) : (i += 1) z[i] = applyElementwiseTyped(dtype, op, x[i], y[i]);
}

pub inline fn vecScale(z: []f32, x: []const f32, s: f32) void {
    const sv: Vf32 = @splat(s);
    var i: usize = 0;
    while (i + 4 * vector_len <= z.len) : (i += 4 * vector_len) {
        z[i..][0..vector_len].* = @as(Vf32, x[i..][0..vector_len].*) * sv;
        z[i + vector_len ..][0..vector_len].* = @as(Vf32, x[i + vector_len ..][0..vector_len].*) * sv;
        z[i + 2 * vector_len ..][0..vector_len].* = @as(Vf32, x[i + 2 * vector_len ..][0..vector_len].*) * sv;
        z[i + 3 * vector_len ..][0..vector_len].* = @as(Vf32, x[i + 3 * vector_len ..][0..vector_len].*) * sv;
    }
    while (i + vector_len <= z.len) : (i += vector_len) {
        const xv: Vf32 = x[i..][0..vector_len].*;
        z[i..][0..vector_len].* = xv * sv;
    }
    while (i < z.len) : (i += 1) z[i] = x[i] * s;
}

pub inline fn vecAddScaled(z: []f32, x: []const f32, s: f32) void {
    const sv: Vf32 = @splat(s);
    var i: usize = 0;
    while (i + 4 * vector_len <= z.len) : (i += 4 * vector_len) {
        z[i..][0..vector_len].* = @as(Vf32, z[i..][0..vector_len].*) + @as(Vf32, x[i..][0..vector_len].*) * sv;
        z[i + vector_len ..][0..vector_len].* = @as(Vf32, z[i + vector_len ..][0..vector_len].*) + @as(Vf32, x[i + vector_len ..][0..vector_len].*) * sv;
        z[i + 2 * vector_len ..][0..vector_len].* = @as(Vf32, z[i + 2 * vector_len ..][0..vector_len].*) + @as(Vf32, x[i + 2 * vector_len ..][0..vector_len].*) * sv;
        z[i + 3 * vector_len ..][0..vector_len].* = @as(Vf32, z[i + 3 * vector_len ..][0..vector_len].*) + @as(Vf32, x[i + 3 * vector_len ..][0..vector_len].*) * sv;
    }
    while (i + vector_len <= z.len) : (i += vector_len) {
        z[i..][0..vector_len].* = @as(Vf32, z[i..][0..vector_len].*) + @as(Vf32, x[i..][0..vector_len].*) * sv;
    }
    while (i < z.len) : (i += 1) z[i] += x[i] * s;
}

pub inline fn vecUnary(comptime op: ops.UnaryOp, z: []f32, x: []const f32) void {
    var i: usize = 0;
    while (i + 4 * vector_len <= z.len) : (i += 4 * vector_len) {
        z[i..][0..vector_len].* = applyUnaryVec(op, x[i..][0..vector_len].*);
        z[i + vector_len ..][0..vector_len].* = applyUnaryVec(op, x[i + vector_len ..][0..vector_len].*);
        z[i + 2 * vector_len ..][0..vector_len].* = applyUnaryVec(op, x[i + 2 * vector_len ..][0..vector_len].*);
        z[i + 3 * vector_len ..][0..vector_len].* = applyUnaryVec(op, x[i + 3 * vector_len ..][0..vector_len].*);
    }
    while (i + vector_len <= z.len) : (i += vector_len) {
        z[i..][0..vector_len].* = applyUnaryVec(op, x[i..][0..vector_len].*);
    }
    while (i < z.len) : (i += 1) z[i] = ops.unaryScalar(op, x[i]);
}

/// Input-based unary derivatives with a vector body — the unary VJP's hot
/// loop (ag/backward/elementwise.zig) routes these ops here; everything else keeps its
/// scalar derivative loop. The train-path exp-family ops matter most: the
/// scalar loop pays one libm expf PER ELEMENT for silu/sigmoid/softplus.
pub inline fn unaryVjpVectorizes(comptime op: ops.UnaryOp) bool {
    return switch (op) {
        .relu, .exp, .sigmoid, .softplus, .silu, .neg, .abs, .log, .log1p => true,
        else => false,
    };
}

/// Formulas mirror `ag/backward/elementwise.zig unaryDerivative`; the sub-vector tail in
/// `vecUnaryVjp` uses the scalar forms (the same lane/tail split as vecUnary).
pub inline fn unaryDerivativeVec(comptime op: ops.UnaryOp, value: Vf32) Vf32 {
    const zero: Vf32 = @splat(0);
    const one: Vf32 = @splat(1);
    return switch (op) {
        .relu => @select(f32, value > zero, one, zero),
        .exp => vexpf(vector_len, value),
        .sigmoid => blk: {
            const s = sigmoidVec(value);
            break :blk s * (one - s);
        },
        .softplus => sigmoidVec(value),
        .silu => blk: {
            const s = sigmoidVec(value);
            break :blk s * (one + value * (one - s));
        },
        .neg => @as(Vf32, @splat(-1)),
        .abs => @select(f32, value > zero, one, @select(f32, value < zero, @as(Vf32, @splat(-1)), zero)),
        .log => one / value,
        .log1p => one / (one + value),
        else => @compileError("unaryDerivativeVec: op has no vector derivative"),
    };
}

inline fn unaryDerivativeScalar(comptime op: ops.UnaryOp, value: f32) f32 {
    return switch (op) {
        .relu => if (value > 0) 1 else 0,
        .exp => @exp(value),
        .sigmoid => blk: {
            const s = ops.sigmoidScalar(value);
            break :blk s * (1 - s);
        },
        .softplus => ops.sigmoidScalar(value),
        .silu => blk: {
            const s = ops.sigmoidScalar(value);
            break :blk s * (1 + value * (1 - s));
        },
        .neg => -1,
        .abs => if (value > 0) 1 else if (value < 0) -1 else 0,
        .log => 1 / value,
        .log1p => 1 / (1 + value),
        else => @compileError("unaryDerivativeScalar: op has no vector derivative"),
    };
}

/// dst = gy * d(op)/dx elementwise, SIMD lanes + scalar tail.
pub inline fn vecUnaryVjp(comptime op: ops.UnaryOp, dsts: []f32, xs: []const f32, gys: []const f32) void {
    var i: usize = 0;
    while (i + vector_len <= dsts.len) : (i += vector_len) {
        dsts[i..][0..vector_len].* = @as(Vf32, gys[i..][0..vector_len].*) * unaryDerivativeVec(op, xs[i..][0..vector_len].*);
    }
    while (i < dsts.len) : (i += 1) dsts[i] = gys[i] * unaryDerivativeScalar(op, xs[i]);
}

pub inline fn vecAddUnary(comptime op: ops.UnaryOp, z: []f32, x: []const f32, y: []const f32) void {
    var i: usize = 0;
    while (i + 4 * vector_len <= z.len) : (i += 4 * vector_len) {
        z[i..][0..vector_len].* = applyUnaryVec(op, @as(Vf32, x[i..][0..vector_len].*) + @as(Vf32, y[i..][0..vector_len].*));
        z[i + vector_len ..][0..vector_len].* = applyUnaryVec(op, @as(Vf32, x[i + vector_len ..][0..vector_len].*) + @as(Vf32, y[i + vector_len ..][0..vector_len].*));
        z[i + 2 * vector_len ..][0..vector_len].* = applyUnaryVec(op, @as(Vf32, x[i + 2 * vector_len ..][0..vector_len].*) + @as(Vf32, y[i + 2 * vector_len ..][0..vector_len].*));
        z[i + 3 * vector_len ..][0..vector_len].* = applyUnaryVec(op, @as(Vf32, x[i + 3 * vector_len ..][0..vector_len].*) + @as(Vf32, y[i + 3 * vector_len ..][0..vector_len].*));
    }
    while (i + vector_len <= z.len) : (i += vector_len) {
        z[i..][0..vector_len].* = applyUnaryVec(op, @as(Vf32, x[i..][0..vector_len].*) + @as(Vf32, y[i..][0..vector_len].*));
    }
    while (i < z.len) : (i += 1) z[i] = ops.unaryScalar(op, x[i] + y[i]);
}

/// `cap * tanh(x / cap)`: the logit softcap, with the vector tanh on the
/// lanes and libm on the tail.
pub inline fn vecSoftcap(z: []f32, x: []const f32, cap: f32) void {
    const inv = 1.0 / cap;
    const cap_v: Vf32 = @splat(cap);
    const inv_v: Vf32 = @splat(inv);
    var i: usize = 0;
    while (i + vector_len <= z.len) : (i += vector_len) {
        const xv: Vf32 = x[i..][0..vector_len].*;
        z[i..][0..vector_len].* = cap_v * tanhVec(xv * inv_v);
    }
    while (i < z.len) : (i += 1) z[i] = cap * std.math.tanh(x[i] * inv);
}

pub inline fn vecLeakyRelu(z: []f32, x: []const f32, negative_slope: f32) void {
    const zero: Vf32 = @splat(0);
    const slope: Vf32 = @splat(negative_slope);
    var i: usize = 0;
    while (i + 4 * vector_len <= z.len) : (i += 4 * vector_len) {
        const x0: Vf32 = x[i..][0..vector_len].*;
        const x1: Vf32 = x[i + vector_len ..][0..vector_len].*;
        const x2: Vf32 = x[i + 2 * vector_len ..][0..vector_len].*;
        const x3: Vf32 = x[i + 3 * vector_len ..][0..vector_len].*;
        z[i..][0..vector_len].* = @select(f32, x0 >= zero, x0, x0 * slope);
        z[i + vector_len ..][0..vector_len].* = @select(f32, x1 >= zero, x1, x1 * slope);
        z[i + 2 * vector_len ..][0..vector_len].* = @select(f32, x2 >= zero, x2, x2 * slope);
        z[i + 3 * vector_len ..][0..vector_len].* = @select(f32, x3 >= zero, x3, x3 * slope);
    }
    while (i + vector_len <= z.len) : (i += vector_len) {
        const xv: Vf32 = x[i..][0..vector_len].*;
        z[i..][0..vector_len].* = @select(f32, xv >= zero, xv, xv * slope);
    }
    while (i < z.len) : (i += 1) {
        const value = x[i];
        z[i] = if (value >= 0) value else value * negative_slope;
    }
}

pub inline fn vecClamp(z: []f32, x: []const f32, min_value: f32, max_value: f32) void {
    const minv: Vf32 = @splat(min_value);
    const maxv: Vf32 = @splat(max_value);
    var i: usize = 0;
    while (i + 4 * vector_len <= z.len) : (i += 4 * vector_len) {
        z[i..][0..vector_len].* = @min(@max(@as(Vf32, x[i..][0..vector_len].*), minv), maxv);
        z[i + vector_len ..][0..vector_len].* = @min(@max(@as(Vf32, x[i + vector_len ..][0..vector_len].*), minv), maxv);
        z[i + 2 * vector_len ..][0..vector_len].* = @min(@max(@as(Vf32, x[i + 2 * vector_len ..][0..vector_len].*), minv), maxv);
        z[i + 3 * vector_len ..][0..vector_len].* = @min(@max(@as(Vf32, x[i + 3 * vector_len ..][0..vector_len].*), minv), maxv);
    }
    while (i + vector_len <= z.len) : (i += vector_len) {
        z[i..][0..vector_len].* = @min(@max(@as(Vf32, x[i..][0..vector_len].*), minv), maxv);
    }
    while (i < z.len) : (i += 1) z[i] = @min(@max(x[i], min_value), max_value);
}

pub inline fn vecGated(comptime op: ops.GatedOp, z: []f32, x: []const f32, y: []const f32) void {
    var i: usize = 0;
    while (i + 4 * vector_len <= z.len) : (i += 4 * vector_len) {
        z[i..][0..vector_len].* = gatedSourceVec(op, x[i..][0..vector_len].*) * gatedActivationVec(op, y[i..][0..vector_len].*);
        z[i + vector_len ..][0..vector_len].* = gatedSourceVec(op, x[i + vector_len ..][0..vector_len].*) * gatedActivationVec(op, y[i + vector_len ..][0..vector_len].*);
        z[i + 2 * vector_len ..][0..vector_len].* = gatedSourceVec(op, x[i + 2 * vector_len ..][0..vector_len].*) * gatedActivationVec(op, y[i + 2 * vector_len ..][0..vector_len].*);
        z[i + 3 * vector_len ..][0..vector_len].* = gatedSourceVec(op, x[i + 3 * vector_len ..][0..vector_len].*) * gatedActivationVec(op, y[i + 3 * vector_len ..][0..vector_len].*);
    }
    while (i + vector_len <= z.len) : (i += vector_len) {
        z[i..][0..vector_len].* = gatedSourceVec(op, x[i..][0..vector_len].*) * gatedActivationVec(op, y[i..][0..vector_len].*);
    }
    while (i < z.len) : (i += 1) z[i] = ops.gatedPairScalar(op, y[i], x[i]);
}

pub inline fn applyUnaryVec(comptime op: ops.UnaryOp, value: Vf32) Vf32 {
    return switch (op) {
        .relu => @max(value, @as(Vf32, @splat(0))),
        .exp => vexpf(vector_len, value),
        .sqrt => @sqrt(value),
        .rsqrt => @as(Vf32, @splat(1)) / @sqrt(value),
        .sigmoid => sigmoidVec(value),
        .softplus => blk: {
            const zero: Vf32 = @splat(0);
            const one: Vf32 = @splat(1);
            const neg_abs = -@abs(value);
            const soft = @log(one + vexpf(vector_len, neg_abs));
            break :blk @max(value, zero) + soft;
        },
        .silu => value * sigmoidVec(value),
        .log => @log(value),
        .log1p => @log(@as(Vf32, @splat(1)) + value),
        .neg => -value,
        .abs => @abs(value),
        .sin => @sin(value),
        .cos => @cos(value),
        .tanh => tanhVec(value),
        .fast_tanh => fastTanhVec(value),
        .gelu => @as(Vf32, @splat(0.5)) * value * (@as(Vf32, @splat(1)) + tanhVec(geluTanhArgVec(value))),
        .quick_gelu => value * sigmoidVec(@as(Vf32, @splat(1.702)) * value),
        .gelu_quant => geluQuantVec(value),
        // ggml_vec_elu_f32 parity: the negative arm is `vexpm1f`, whose
        // lanes reproduce the scalar `std.math.expm1` bytes exactly.
        .elu => @select(f32, value > @as(Vf32, @splat(0)), value, vexpm1f(vector_len, value)),
        .gelu_erf => perLaneUnary(.gelu_erf, value),
        .erf => perLaneUnary(.erf, value),
        .floor => @floor(value),
        .ceil => @ceil(value),
        .round => rintVec(value),
        .sign => blk: {
            const zero: Vf32 = @splat(0);
            const one: Vf32 = @splat(1);
            const minus_one: Vf32 = @splat(-1);
            // x itself in the else lane preserves ±0 and propagates NaN
            // (ops.unaryScalar .sign, the numpy/torch convention).
            break :blk @select(f32, value > zero, one, @select(f32, value < zero, minus_one, value));
        },
        .reciprocal => @as(Vf32, @splat(1)) / value,
    };
}

/// Vector round-half-to-even: the 2^23 magic-number trick on |x| with a
/// bitwise copysign, lanes at or above 2^23 (and NaN/±inf) passing
/// through. Bit-identical to `ops.rintScalar` by construction.
inline fn rintVec(value: Vf32) Vf32 {
    const Vu32 = @Vector(vector_len, u32);
    const big: Vf32 = @splat(8388608.0); // 2^23
    const ax = @abs(value);
    const shifted = (ax + big) - big;
    const sign_mask: Vu32 = @splat(0x8000_0000);
    const signed: Vf32 = @bitCast((@as(Vu32, @bitCast(shifted)) & ~sign_mask) | (@as(Vu32, @bitCast(value)) & sign_mask));
    return @select(f32, ax < big, signed, value);
}

/// Per-lane scalar fallback for unary ops without a vectorizable form
/// (the erf-based `gelu_erf`/`erf`): evaluates `ops.unaryScalar` on each
/// lane, so the SIMD body and the scalar tail are bit-identical.
inline fn perLaneUnary(comptime op: ops.UnaryOp, value: Vf32) Vf32 {
    var out: Vf32 = undefined;
    inline for (0..vector_len) |lane| {
        out[lane] = ops.unaryScalar(op, value[lane]);
    }
    return out;
}

pub inline fn fastTanhVec(value: Vf32) Vf32 {
    const ax = @abs(value);
    const x2 = value * value;
    const numerator = value * (@as(Vf32, @splat(2.45550750702956)) +
        @as(Vf32, @splat(2.45550750702956)) * ax +
        (@as(Vf32, @splat(0.893229853513558)) + @as(Vf32, @splat(0.821226666969744)) * ax) * x2);
    const denom_abs_arg = value + @as(Vf32, @splat(0.814642734961073)) * value * ax;
    const denominator = @as(Vf32, @splat(2.44506634652299)) +
        (@as(Vf32, @splat(2.44506634652299)) + x2) * @abs(denom_abs_arg);
    return numerator / denominator;
}

/// ggml f16-LUT gelu (see ops.geluQuantScalar): f16-round the input, exact
/// tanh-gelu, f16-round the output, with hard clamps at +/-10. The contract
/// is byte parity with ggml's table, which no approximate lane tanh keeps
/// once the output rounds to f16 — the tanh here is `vtanhf`, the faithful
/// musl port whose lanes reproduce the `std.math.tanh` bytes the scalar
/// form calls, so the lanes match the table exactly at vector speed
/// (`primitives_tests.zig` sweeps every f16 inside the clamps plus the
/// off-grid f16 midpoints).
pub inline fn geluQuantVec(value: Vf32) Vf32 {
    const xr: Vf32 = @floatCast(@as(Vf16ForF32, @floatCast(value)));
    // geluScalar's exact shape: 0.5 * x * (1 + tanh(geluTanhArg(x))).
    const g = @as(Vf32, @splat(0.5)) * xr * (@as(Vf32, @splat(1)) + vtanhf(vector_len, geluTanhArgVec(xr)));
    const gr: Vf32 = @floatCast(@as(Vf16ForF32, @floatCast(g)));
    // The hard clamps test the ORIGINAL input, before the f16 rounding.
    return @select(
        f32,
        value <= @as(Vf32, @splat(-10)),
        @as(Vf32, @splat(0)),
        @select(f32, value >= @as(Vf32, @splat(10)), value, gr),
    );
}

// ---------------------------------------------------------------------------
// vexpm1f / vtanhf: single-precision expm1 and tanh on @Vector lanes,
// translated faithfully from musl libc src/math/expm1f.c and tanhf.c (MIT
// licensed; FDLIBM lineage) through the `std.math.expm1` / `std.math.tanh`
// scalar ports the backend's scalar arms call. Same constants and branch
// structure (branches become masked selects, the float-word tricks become
// integer vector ops), and no fused or reassociated arithmetic — every lane
// produces bytes identical to the scalar call, which the byte-parity ops
// (`gelu_quant`'s ggml f16 LUT, `elu`'s ggml_vec_elu_f32) require.
// `primitives_tests.zig` sweeps both against the scalar forms bit-for-bit.
// ---------------------------------------------------------------------------

/// musl expm1f on lanes, byte-identical to `std.math.expm1(f32)` per lane.
pub inline fn vexpm1f(comptime W: usize, x: @Vector(W, f32)) @Vector(W, f32) {
    const Vec = @Vector(W, f32);
    const VecU = @Vector(W, u32);

    const o_threshold: f32 = 8.8721679688e+01;
    const zero: Vec = @splat(0);
    const one: Vec = @splat(1);

    const ux: VecU = @bitCast(x);
    const hx = ux & @as(VecU, @splat(0x7FFFFFFF));
    const sign = (ux >> @splat(31)) != @as(VecU, @splat(0));

    const is_nan = x != x;
    // |x| >= 27*ln2: negative lanes saturate to -1 (including -inf), lanes
    // above the overflow threshold go to +inf; in-range large lanes fall
    // through to the general path exactly as in the scalar.
    const big = hx >= @as(VecU, @splat(0x4195B844));
    const ret_neg1 = big & sign;
    const ret_inf = big & ~sign & (x > @as(Vec, @splat(o_threshold)));
    const early = is_nan | ret_neg1 | ret_inf;
    // Early-return lanes run the shared pipeline on 0 (k = 0, polynomial at
    // 0) so nothing below traps or overflows; their results are replaced at
    // the end. This also bounds |kf| in the body, keeping @intFromFloat in
    // range.
    const xs = @select(f32, early, zero, x);

    var res = vexpm1fBody(W, false, xs);
    res = @select(f32, ret_inf, x * @as(Vec, @splat(0x1.0p127)), res);
    res = @select(f32, ret_neg1, -one, res);
    return @select(f32, is_nan, @as(Vec, @splat(std.math.nan(f32))), res);
}

/// The argument-reduction + polynomial pipeline shared by `vexpm1f` and
/// `vtanhf`, byte-faithful to the scalar over lanes that are finite and
/// below the overflow threshold (the caller's contract). `bounded` marks
/// the tanhf call site, whose constructed argument is always in
/// [-log(5/3), 20] (musl tanhf never feeds expm1f anything else): the
/// |k| > 56 scale arm and its k = 128 special are unreachable there
/// (k stays in [-1, 29]) and compile out; `vexpm1f` keeps them.
inline fn vexpm1fBody(comptime W: usize, comptime bounded: bool, xs: @Vector(W, f32)) @Vector(W, f32) {
    const Vec = @Vector(W, f32);
    const VecU = @Vector(W, u32);
    const VecI = @Vector(W, i32);

    const ln2_hi: f32 = 6.9313812256e-01;
    const ln2_lo: f32 = 9.0580006145e-06;
    const invln2: f32 = 1.4426950216e+00;
    const q1_coeff: f32 = -3.3333212137e-2;
    const q2_coeff: f32 = 1.5807170421e-3;

    const zero: Vec = @splat(0);
    const one: Vec = @splat(1);

    const uxs: VecU = @bitCast(xs);
    const axs = uxs & @as(VecU, @splat(0x7FFFFFFF));
    const ssign = (uxs >> @splat(31)) != @as(VecU, @splat(0));

    // |x| < 2^-25 (subnormals included): returns x unchanged.
    const tiny = axs < @as(VecU, @splat(0x33000000));

    // Argument reduction x = k*ln2 + reduced, in the scalar's two arms:
    // |x| in (0.5*ln2, 1.5*ln2) takes k = ±1 with the constant hi/lo pair,
    // larger |x| takes k = trunc(x/ln2 ± 0.5) with the computed pair.
    const mid = axs > @as(VecU, @splat(0x3EB17218));
    const near = axs < @as(VecU, @splat(0x3F851592));
    const kf = @as(Vec, @splat(invln2)) * xs + @select(f32, ssign, @as(Vec, @splat(-0.5)), @as(Vec, @splat(0.5)));
    const k_far: VecI = @intFromFloat(kf);
    const t_far: Vec = @floatFromInt(k_far);
    const hi_far = xs - t_far * @as(Vec, @splat(ln2_hi));
    const lo_far = t_far * @as(Vec, @splat(ln2_lo));
    const hi_near = xs - @select(f32, ssign, @as(Vec, @splat(-ln2_hi)), @as(Vec, @splat(ln2_hi)));
    const lo_near = @select(f32, ssign, @as(Vec, @splat(-ln2_lo)), @as(Vec, @splat(ln2_lo)));
    const k_near: VecI = @select(i32, ssign, @as(VecI, @splat(-1)), @as(VecI, @splat(1)));
    const hi = @select(f32, near, hi_near, hi_far);
    const lo = @select(f32, near, lo_near, lo_far);
    const k: VecI = @select(i32, mid, @select(i32, near, k_near, k_far), @as(VecI, @splat(0)));
    const xm = hi - lo;
    const xr = @select(f32, mid, xm, xs);
    const c = @select(f32, mid, (hi - xm) - lo, zero);

    // Degree-2 rational approximation on the reduced interval (the exact
    // scalar operation order — nothing fused, nothing reassociated).
    const hfx = @as(Vec, @splat(0.5)) * xr;
    const hxs = xr * hfx;
    const r1 = one + hxs * (@as(Vec, @splat(q1_coeff)) + hxs * @as(Vec, @splat(q2_coeff)));
    const t = @as(Vec, @splat(3)) - r1 * hfx;
    const e0 = hxs * ((r1 - t) / (@as(Vec, @splat(6)) - xr * t));
    const res_k0 = xr - (xr * e0 - hxs);
    const e1 = (xr * (e0 - c) - c) - hxs;
    const res_km1 = @as(Vec, @splat(0.5)) * (xr - e1) - @as(Vec, @splat(0.5));
    const res_k1 = @select(
        f32,
        xr < @as(Vec, @splat(-0.25)),
        @as(Vec, @splat(-2)) * (e1 - (xr + @as(Vec, @splat(0.5)))),
        one + @as(Vec, @splat(2)) * (xr - e1),
    );
    // 2^k and 2^-k assembled in the exponent field. Wrapping u32 arithmetic
    // produces the scalar's bits for every k that reaches each consumer;
    // out-of-branch lanes hold junk that is never selected.
    const k_u: VecU = @bitCast(k);
    const twopk: Vec = @bitCast((@as(VecU, @splat(0x7F)) +% k_u) << @splat(23));
    const uf: Vec = @bitCast((@as(VecU, @splat(0x7F)) -% k_u) << @splat(23));
    const res_low = ((xr - e1) + (one - uf)) * twopk; // 2 <= k < 23
    const res_high = ((xr - (e1 + uf)) + one) * twopk; // 23 <= k <= 56
    const res_scaled = if (bounded) res_scaled: {
        break :res_scaled @select(f32, k < @as(VecI, @splat(23)), res_low, res_high);
    } else res_scaled: {
        const y0 = (xr - e1) + one;
        const y1 = @select(
            f32,
            k == @as(VecI, @splat(128)),
            (y0 * @as(Vec, @splat(2))) * @as(Vec, @splat(0x1.0p127)),
            y0 * twopk,
        );
        const res_wide = y1 - one;
        const wide = (k < @as(VecI, @splat(0))) | (k > @as(VecI, @splat(56)));
        break :res_scaled @select(f32, wide, res_wide, //
            @select(f32, k < @as(VecI, @splat(23)), res_low, res_high));
    };
    const res_k = @select(f32, k == @as(VecI, @splat(0)), res_k0, //
        @select(f32, k == @as(VecI, @splat(-1)), res_km1, //
            @select(f32, k == @as(VecI, @splat(1)), res_k1, res_scaled)));
    return @select(f32, tiny, xs, res_k);
}

/// musl tanhf on lanes, byte-identical to `std.math.tanh(f32)` per lane:
/// one shared `vexpm1f` for the three expm1 branches (the small-magnitude
/// arm's negation folds into the quotient's numerator, an exact sign move).
pub inline fn vtanhf(comptime W: usize, x: @Vector(W, f32)) @Vector(W, f32) {
    const Vec = @Vector(W, f32);
    const VecU = @Vector(W, u32);

    const ux: VecU = @bitCast(x);
    const uabs = ux & @as(VecU, @splat(0x7FFFFFFF));
    const ax: Vec = @bitCast(uabs);
    const sign = (ux >> @splat(31)) != @as(VecU, @splat(0));

    const cross = uabs > @as(VecU, @splat(0x3F0C9F54)); // |x| > log(3)/2, or nan
    const huge = uabs > @as(VecU, @splat(0x41200000)); // |x| > 10, or nan
    const mid = uabs > @as(VecU, @splat(0x3E82C578)); // |x| > log(5/3)/2
    const norm = uabs >= @as(VecU, @splat(0x00800000));

    // expm1(2|x|) feeds the cross-but-not-huge and mid arms, expm1(-2|x|)
    // the small normal arm; every other lane is fed 0 and overridden.
    const saturating = cross & ~huge;
    const pos_arm = saturating | (~cross & mid);
    const neg_arm = ~cross & ~mid & norm;
    const two_ax = @as(Vec, @splat(2)) * ax;
    const arg = @select(f32, pos_arm, two_ax, @select(f32, neg_arm, -two_ax, @as(Vec, @splat(0))));
    // The constructed argument is finite in [-log(5/3), 20] by the masks
    // above, so the bounded expm1 body applies (its documented contract).
    const em = vexpm1fBody(W, true, arg);
    const num = @select(f32, saturating, @as(Vec, @splat(2)), @select(f32, neg_arm, -em, em));
    const q = num / (em + @as(Vec, @splat(2)));
    const t_expm1 = @select(f32, saturating, @as(Vec, @splat(1)) - q, q);
    // |x| > 10: t = 1 + 0/x (1 on finite lanes, NaN propagation on NaN);
    // this fdiv is independent of the expm1 chain, so it overlaps it.
    const t_huge = @as(Vec, @splat(1)) + @as(Vec, @splat(0)) / x;
    const t1 = @select(f32, pos_arm | neg_arm, t_expm1, ax); // else arm: subnormal t = |x|
    const t2 = @select(f32, huge, t_huge, t1);
    return @select(f32, sign, -t2, t2);
}

pub inline fn gatedActivationVec(comptime op: ops.GatedOp, value: Vf32) Vf32 {
    return switch (op) {
        .glu => sigmoidVec(value),
        .swiglu => value * sigmoidVec(value),
        .geglu => @as(Vf32, @splat(0.5)) * value * (@as(Vf32, @splat(1)) + tanhVec(geluTanhArgVec(value))),
        .situ => @as(Vf32, @splat(4)) * tanhVec(value * @as(Vf32, @splat(0.25))) * sigmoidVec(value),
    };
}

/// Vector twin of `ops.gatedSourceScalar`: the gated pair's `up`-side
/// transform (identity for the classic ops).
pub inline fn gatedSourceVec(comptime op: ops.GatedOp, value: Vf32) Vf32 {
    return switch (op) {
        .situ => @as(Vf32, @splat(25)) * tanhVec(value * @as(Vf32, @splat(0.04))),
        else => value,
    };
}

/// Vector twin of `ops.gatedActivationDerivativeScalar`.
pub inline fn gatedActivationDerivativeVec(comptime op: ops.GatedOp, value: Vf32) Vf32 {
    const one: Vf32 = @splat(1);
    return switch (op) {
        .glu => blk: {
            const s = sigmoidVec(value);
            break :blk s * (one - s);
        },
        .swiglu => blk: {
            const s = sigmoidVec(value);
            break :blk s * (one + value * (one - s));
        },
        .geglu => blk: {
            const half: Vf32 = @splat(0.5);
            const sqrt_2_over_pi: Vf32 = @splat(0.7978845608028654);
            const t = tanhVec(geluTanhArgVec(value));
            const du = sqrt_2_over_pi * (one + @as(Vf32, @splat(3 * 0.044715)) * value * value);
            break :blk half * (one + t) + half * value * (one - t * t) * du;
        },
        .situ => blk: {
            const t = tanhVec(value * @as(Vf32, @splat(0.25)));
            const s = sigmoidVec(value);
            break :blk (one - t * t) * s + @as(Vf32, @splat(4)) * t * s * (one - s);
        },
    };
}

/// Vector twin of `ops.gatedSourceDerivativeScalar`.
pub inline fn gatedSourceDerivativeVec(comptime op: ops.GatedOp, up: Vf32) Vf32 {
    const one: Vf32 = @splat(1);
    return switch (op) {
        .situ => blk: {
            const t = tanhVec(up * @as(Vf32, @splat(0.04)));
            break :blk one - t * t;
        },
        else => one,
    };
}

/// Vector twin of `ops.gatedGradScalar`.
pub inline fn gatedGradVec(comptime op: ops.GatedOp, comptime operand: ops.GatedOperand, up: Vf32, gate: Vf32) Vf32 {
    return switch (operand) {
        .up => if (comptime ops.gatedSourceIsIdentity(op)) gatedActivationVec(op, gate) else gatedSourceDerivativeVec(op, up) * gatedActivationVec(op, gate),
        .gate => gatedSourceVec(op, up) * gatedActivationDerivativeVec(op, gate),
    };
}

/// dst = gy · ∂(gated pair)/∂operand elementwise, SIMD lanes + scalar tail.
pub inline fn vecGatedVjp(comptime op: ops.GatedOp, comptime operand: ops.GatedOperand, dst: []f32, gy: []const f32, up: []const f32, gate: []const f32) void {
    var i: usize = 0;
    while (i + vector_len <= dst.len) : (i += vector_len) {
        dst[i..][0..vector_len].* = @as(Vf32, gy[i..][0..vector_len].*) * gatedGradVec(op, operand, up[i..][0..vector_len].*, gate[i..][0..vector_len].*);
    }
    while (i < dst.len) : (i += 1) dst[i] = gy[i] * ops.gatedGradScalar(op, operand, up[i], gate[i]);
}

pub inline fn sigmoidVec(value: Vf32) Vf32 {
    const one: Vf32 = @splat(1);
    return one / (one + vexpf(vector_len, -value));
}

/// tanh on the lanes from ONE `vexpf`: t = e^(-2|x|) (the argument is
/// never positive, so nothing overflows), tanh(|x|) = (1 - t) / (1 + t),
/// and the sign copied back bitwise. Below |x| = 0.125 the quotient
/// cancels (t -> 1), so those lanes take the degree-7 odd Taylor series
/// instead. Measured against f64 tanh: max relative error 5.1e-7 over
/// [-10, 10] (the vexpf side, at |x| = 0.144), 8.9e-8 below the cut, 6.8e-8
/// below 1e-3; the previous `@exp`-per-lane form was 2.7e-5 overall and
/// lost every digit below |x| ~ 1e-7. One vexpf per vector instead of
/// eight libm calls: 3.6x faster over 1M lanes.
pub inline fn tanhVec(value: Vf32) Vf32 {
    const Vu32 = @Vector(vector_len, u32);
    const one: Vf32 = @splat(1);
    const ax = @abs(value);
    const t = vexpf(vector_len, @as(Vf32, @splat(-2)) * ax);
    const quotient = (one - t) / (one + t);
    // tanh(a) = a - a^3/3 + 2a^5/15 - 17a^7/315 + O(a^9)
    const x2 = ax * ax;
    const series = ax * @mulAdd(Vf32, x2, @mulAdd(Vf32, x2, @mulAdd(Vf32, x2, @as(Vf32, @splat(-17.0 / 315.0)), @as(Vf32, @splat(2.0 / 15.0))), @as(Vf32, @splat(-1.0 / 3.0))), one);
    const magnitude = @select(f32, ax < @as(Vf32, @splat(0.125)), series, quotient);
    const sign_bits = @as(Vu32, @bitCast(value)) & @as(Vu32, @splat(0x8000_0000));
    return @bitCast(@as(Vu32, @bitCast(magnitude)) | sign_bits);
}

pub inline fn geluTanhArgVec(value: Vf32) Vf32 {
    const sqrt_2_over_pi: Vf32 = @splat(0.7978845608028654);
    return sqrt_2_over_pi * (value + @as(Vf32, @splat(0.044715)) * value * value * value);
}

/// SIMD polynomial expf (the ggml_v_expf / ARM optimized-routines scheme):
/// n = round(x*log2(e)) via the 0x1.8p23 shift trick, two-step Cody-Waite
/// reduction r = x - n*ln2, degree-4 polynomial for e^r - 1, then scale by 2^n
/// through the exponent bit field. Lanes with |n| > 126 take the split-scale
/// path (s2*s1 = 2^n) so near-overflow stays finite and near-underflow produces
/// correct subnormals. Inputs are pre-clamped to [-104, 89]: every value below
/// -104 (including -inf) underflows to 0 through the subnormal path, and every
/// value above 89 (including +inf) saturates to +inf. exp(0) == 1 exactly.
/// NaN lanes propagate as NaN (the @min/@max clamp drops NaN, which would
/// silently turn exp(NaN) into exp(-104) = 0; a final select restores it so
/// SIMD softmax/CE rows poison on NaN exactly like the scalar fallbacks).
/// Relative error < 2e-6 over [-87, 88]. No tables, no allocation.
pub inline fn vexpf(comptime W: usize, x: @Vector(W, f32)) @Vector(W, f32) {
    const Vec = @Vector(W, f32);
    const VecU = @Vector(W, u32);
    const xc = @min(@max(x, @as(Vec, @splat(-104.0))), @as(Vec, @splat(89.0)));
    const shift: Vec = @splat(0x1.8p23);
    const z = @mulAdd(Vec, xc, @as(Vec, @splat(0x1.715476p+0)), shift);
    const n = z - shift;
    // Cody-Waite: r = x - n*ln2_hi - n*ln2_lo, kept in [-ln2/2, ln2/2].
    const r = @mulAdd(Vec, n, @as(Vec, @splat(-0x1.7f7d1cp-20)), @mulAdd(Vec, n, @as(Vec, @splat(-0x1.62e4p-1)), xc));
    // z's mantissa holds the integer n; shifting it into the exponent field
    // gives e = n << 23 (the 2^n scale as raw exponent bits).
    const e = @as(VecU, @bitCast(z)) << @splat(23);
    const scale: Vec = @bitCast(e +% @as(VecU, @bitCast(@as(Vec, @splat(1.0)))));
    // Degree-4 polynomial for e^r - 1 on the reduced interval.
    const r2 = r * r;
    const p23 = @mulAdd(Vec, @as(Vec, @splat(0x1.555e66p-3)), r, @as(Vec, @splat(0x1.fffdb6p-2)));
    const p45 = @mulAdd(Vec, @as(Vec, @splat(0x1.0e4020p-7)), r, @as(Vec, @splat(0x1.573e2ep-5)));
    const expm1 = @mulAdd(Vec, @mulAdd(Vec, p45, r2, p23), r2, @as(Vec, @splat(0x1.ffffecp-1)) * r);
    const normal = @mulAdd(Vec, expm1, scale, scale);
    // |n| > 126: split 2^n into s2*s1 so the intermediate stays representable.
    const d = @select(u32, n <= @as(Vec, @splat(0)), @as(VecU, @splat(0x82000000)), @as(VecU, @splat(0)));
    const s1: Vec = @bitCast(d +% @as(VecU, @splat(0x7f000000)));
    const s2: Vec = @bitCast(e -% d);
    const special = @mulAdd(Vec, s2, expm1, s2) * s1;
    const result = @select(f32, @abs(n) > @as(Vec, @splat(126.0)), special, normal);
    // Propagate NaN inputs (NaN != NaN only on NaN lanes; @exp(NaN) == NaN).
    return @select(f32, x != x, x, result);
}

// Fused multiply-add of a contiguous slice with a broadcast scalar:
// out += in * s, one `@mulAdd` per lane (fmla on AArch64, vfmadd on
// x86_64 with FMA), the scalar tail fused the same way.
pub inline fn vecFmaScalar(out: []f32, in: []const f32, s: f32) void {
    const sv: Vf32 = @splat(s);
    var i: usize = 0;
    while (i + vector_len <= out.len) : (i += vector_len) {
        const ov: Vf32 = out[i..][0..vector_len].*;
        const iv: Vf32 = in[i..][0..vector_len].*;
        out[i..][0..vector_len].* = @mulAdd(Vf32, iv, sv, ov);
    }
    while (i < out.len) : (i += 1) out[i] = @mulAdd(f32, in[i], s, out[i]);
}

// ---------------- Reductions ----------------

/// Number of independent accumulator chains in the f32 `vecSum` /
/// `vecDot`: the loops are add-latency bound, and eight chains keep the
/// FMA/FADD pipelines full where four left them half idle (measured, see
/// the commit that widened them). The chain count is part of the
/// summation order and therefore of the result bits.
pub const reduce_chains = 8;

pub const ReduceTerm = enum { sum, prod, dot };

/// Accumulator (and result) of the typed reductions: f32 for f16/bf16/f32,
/// f64 for f64.
pub fn ReduceAcc(comptime dtype: DType) type {
    return dtype_mod.Accumulator(dtype);
}

/// Lane vector of the typed reductions, in the accumulator type: f32 and
/// f64 at their own widths, f16 widened at the f32 width, bf16 widened at
/// the f16 width.
fn ReduceLanes(comptime dtype: DType) type {
    return switch (dtype) {
        .f32, .f16 => Vf32,
        .f64 => Vf64,
        .bf16 => common.Vf32ForF16,
        else => @compileError("vecReduce: no vector lanes for " ++ @tagName(dtype)),
    };
}

const ReduceSpec = struct { ladder: []const usize, fused: bool };

/// Chain ladder and FMA policy per (dtype, term): `ladder` lists the chain
/// counts in order (each level's loop runs while that many lane vectors
/// remain; the widest sets the accumulator count), `fused` folds each
/// product with one `@mulAdd`, tail included. Both are part of the
/// summation order and therefore of the result bits.
fn reduceSpec(comptime dtype: DType, comptime term: ReduceTerm) ReduceSpec {
    return switch (dtype) {
        .f32 => .{ .ladder = if (term == .prod) &.{4} else &.{reduce_chains}, .fused = term == .dot },
        .f16 => .{ .ladder = if (term == .dot) &.{ 8, 4 } else &.{4}, .fused = false },
        .f64, .bf16 => .{ .ladder = &.{4}, .fused = false },
        else => @compileError("vecReduce: unsupported dtype " ++ @tagName(dtype)),
    };
}

/// One lane vector of `dtype` storage, widened to the accumulator lanes.
inline fn widenLanes(comptime dtype: DType, bits: @Vector(@typeInfo(ReduceLanes(dtype)).vector.len, dtype_mod.Scalar(dtype))) ReduceLanes(dtype) {
    return switch (dtype) {
        .f32, .f64 => bits,
        .f16 => @floatCast(bits),
        .bf16 => bf16VecToF32Wide(bits),
        else => unreachable,
    };
}

/// One reduction step on lanes or on the scalar tail (`V` is the lane
/// vector or the accumulator scalar).
inline fn reduceStep(comptime term: ReduceTerm, comptime fused: bool, comptime V: type, acc: V, a: V, b: V) V {
    return switch (term) {
        .sum => acc + a,
        .prod => acc * a,
        .dot => if (fused) @mulAdd(V, a, b, acc) else acc + a * b,
    };
}

inline fn reduceLanes(
    comptime dtype: DType,
    comptime term: ReduceTerm,
    comptime fused: bool,
    acc: ReduceLanes(dtype),
    x: []const dtype_mod.Scalar(dtype),
    y: []const dtype_mod.Scalar(dtype),
    i: usize,
) ReduceLanes(dtype) {
    const L = @typeInfo(ReduceLanes(dtype)).vector.len;
    const a = widenLanes(dtype, x[i..][0..L].*);
    const b = if (comptime term == .dot) widenLanes(dtype, y[i..][0..L].*) else a;
    return reduceStep(term, fused, ReduceLanes(dtype), acc, a, b);
}

inline fn foldPartial(comptime term: ReduceTerm, comptime V: type, acc: V, part: V) V {
    return if (term == .prod) acc * part else acc + part;
}

/// The chained reduction behind every sum/prod/dot primitive: the
/// `reduceSpec` ladder over lane accumulators, one single-chain vector
/// loop, the chains folded in index order, the lanes reduced, then the
/// scalar tail. `y` is read by `.dot` only (pass `x` otherwise).
pub inline fn vecReduce(
    comptime dtype: DType,
    comptime term: ReduceTerm,
    x: []const dtype_mod.Scalar(dtype),
    y: []const dtype_mod.Scalar(dtype),
) ReduceAcc(dtype) {
    const spec = comptime reduceSpec(dtype, term);
    const Acc = ReduceAcc(dtype);
    const V = ReduceLanes(dtype);
    const L = @typeInfo(V).vector.len;
    const chains = spec.ladder[0];
    const identity: Acc = if (term == .prod) 1 else 0;
    var acc: [chains]V = undefined;
    inline for (0..chains) |c| acc[c] = @splat(identity);
    var i: usize = 0;
    inline for (spec.ladder) |level| {
        while (i + level * L <= x.len) : (i += level * L) {
            inline for (0..level) |c| acc[c] = reduceLanes(dtype, term, spec.fused, acc[c], x, y, i + c * L);
        }
    }
    while (i + L <= x.len) : (i += L) acc[0] = reduceLanes(dtype, term, spec.fused, acc[0], x, y, i);
    var total = acc[0];
    inline for (1..chains) |c| total = foldPartial(term, V, total, acc[c]);
    var s: Acc = @reduce(if (term == .prod) .Mul else .Add, total);
    while (i < x.len) : (i += 1) {
        const a = dtype_mod.toAccumulator(dtype, x[i]);
        const b = if (comptime term == .dot) dtype_mod.toAccumulator(dtype, y[i]) else a;
        s = reduceStep(term, spec.fused, Acc, s, a, b);
    }
    return s;
}

pub inline fn vecSum(x: []const f32) f32 {
    return vecReduce(.f32, .sum, x, x);
}

pub inline fn vecProd(x: []const f32) f32 {
    return vecReduce(.f32, .prod, x, x);
}

/// Fused dot product over `reduce_chains` independent chains: every
/// product is folded into its chain with one `@mulAdd` (a single rounding
/// per term, the scalar tail included).
pub inline fn vecDot(x: []const f32, y: []const f32) f32 {
    return vecReduce(.f32, .dot, x, y);
}

pub inline fn vecDotF16ToF32(x: []const f16, y: []const f16) f32 {
    return vecReduce(.f16, .dot, x, y);
}

/// bf16 bits to f32 lanes, exact (bits << 16).
pub inline fn bf16ToF32Lanes(comptime W: usize, bits: @Vector(W, u16)) @Vector(W, f32) {
    const U = @Vector(W, u32);
    const widened: U = @as(U, @intCast(bits)) << @as(U, @splat(16));
    return @bitCast(widened);
}

pub inline fn bf16VecToF32(bits: common.Vu16ForF32) Vf32 {
    return bf16ToF32Lanes(vector_len, bits);
}

pub inline fn bf16VecToF32Wide(bits: common.Vu16ForF16) common.Vf32ForF16 {
    return bf16ToF32Lanes(vector_len_f16, bits);
}

pub inline fn f32VecToBf16(values: Vf32) common.Vu16ForF32 {
    const bits: common.Vu32ForF32 = @bitCast(values);
    const lsb = (bits >> @as(common.Vu32ForF32, @splat(16))) & @as(common.Vu32ForF32, @splat(1));
    const rounded = bits + @as(common.Vu32ForF32, @splat(0x7fff)) + lsb;
    return @truncate(rounded >> @as(common.Vu32ForF32, @splat(16)));
}

/// Eight f16 lanes widened to f32 (the f16 row kernels' load).
pub inline fn widenF16x8(row: []const f16) @Vector(8, f32) {
    const H = @Vector(8, f16);
    return @floatCast(@as(H, row[0..8].*));
}

/// dot(a, b) with an f32 left side and an f16 right side; four independent
/// accumulators keep the widen + FMA chain off the latency path.
pub fn dotF32F16(a: []const f32, b: []const f16) f32 {
    const V = @Vector(8, f32);
    var acc0: V = @splat(0);
    var acc1: V = @splat(0);
    var acc2: V = @splat(0);
    var acc3: V = @splat(0);
    var i: usize = 0;
    while (i + 32 <= a.len) : (i += 32) {
        acc0 = @mulAdd(V, a[i..][0..8].*, widenF16x8(b[i..]), acc0);
        acc1 = @mulAdd(V, a[i + 8 ..][0..8].*, widenF16x8(b[i + 8 ..]), acc1);
        acc2 = @mulAdd(V, a[i + 16 ..][0..8].*, widenF16x8(b[i + 16 ..]), acc2);
        acc3 = @mulAdd(V, a[i + 24 ..][0..8].*, widenF16x8(b[i + 24 ..]), acc3);
    }
    while (i + 8 <= a.len) : (i += 8) {
        acc0 = @mulAdd(V, a[i..][0..8].*, widenF16x8(b[i..]), acc0);
    }
    var total: f32 = @reduce(.Add, (acc0 + acc1) + (acc2 + acc3));
    while (i < a.len) : (i += 1) total += a[i] * @as(f32, @floatCast(b[i]));
    return total;
}

test {
    _ = @import("primitives_tests.zig");
}
