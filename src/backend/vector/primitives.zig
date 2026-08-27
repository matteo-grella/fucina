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

pub inline fn vecAdd(z: []f32, x: []const f32, y: []const f32) void {
    var i: usize = 0;
    while (i + 4 * vector_len <= z.len) : (i += 4 * vector_len) {
        const x0: Vf32 = x[i..][0..vector_len].*;
        const y0: Vf32 = y[i..][0..vector_len].*;
        const x1: Vf32 = x[i + vector_len ..][0..vector_len].*;
        const y1: Vf32 = y[i + vector_len ..][0..vector_len].*;
        const x2: Vf32 = x[i + 2 * vector_len ..][0..vector_len].*;
        const y2: Vf32 = y[i + 2 * vector_len ..][0..vector_len].*;
        const x3: Vf32 = x[i + 3 * vector_len ..][0..vector_len].*;
        const y3: Vf32 = y[i + 3 * vector_len ..][0..vector_len].*;
        z[i..][0..vector_len].* = x0 + y0;
        z[i + vector_len ..][0..vector_len].* = x1 + y1;
        z[i + 2 * vector_len ..][0..vector_len].* = x2 + y2;
        z[i + 3 * vector_len ..][0..vector_len].* = x3 + y3;
    }
    while (i + vector_len <= z.len) : (i += vector_len) {
        const xv: Vf32 = x[i..][0..vector_len].*;
        const yv: Vf32 = y[i..][0..vector_len].*;
        z[i..][0..vector_len].* = xv + yv;
    }
    while (i < z.len) : (i += 1) z[i] = x[i] + y[i];
}

pub inline fn vecSub(z: []f32, x: []const f32, y: []const f32) void {
    var i: usize = 0;
    while (i + 4 * vector_len <= z.len) : (i += 4 * vector_len) {
        const x0: Vf32 = x[i..][0..vector_len].*;
        const y0: Vf32 = y[i..][0..vector_len].*;
        const x1: Vf32 = x[i + vector_len ..][0..vector_len].*;
        const y1: Vf32 = y[i + vector_len ..][0..vector_len].*;
        const x2: Vf32 = x[i + 2 * vector_len ..][0..vector_len].*;
        const y2: Vf32 = y[i + 2 * vector_len ..][0..vector_len].*;
        const x3: Vf32 = x[i + 3 * vector_len ..][0..vector_len].*;
        const y3: Vf32 = y[i + 3 * vector_len ..][0..vector_len].*;
        z[i..][0..vector_len].* = x0 - y0;
        z[i + vector_len ..][0..vector_len].* = x1 - y1;
        z[i + 2 * vector_len ..][0..vector_len].* = x2 - y2;
        z[i + 3 * vector_len ..][0..vector_len].* = x3 - y3;
    }
    while (i + vector_len <= z.len) : (i += vector_len) {
        const xv: Vf32 = x[i..][0..vector_len].*;
        const yv: Vf32 = y[i..][0..vector_len].*;
        z[i..][0..vector_len].* = xv - yv;
    }
    while (i < z.len) : (i += 1) z[i] = x[i] - y[i];
}

pub inline fn vecMul(z: []f32, x: []const f32, y: []const f32) void {
    var i: usize = 0;
    while (i + 4 * vector_len <= z.len) : (i += 4 * vector_len) {
        const x0: Vf32 = x[i..][0..vector_len].*;
        const y0: Vf32 = y[i..][0..vector_len].*;
        const x1: Vf32 = x[i + vector_len ..][0..vector_len].*;
        const y1: Vf32 = y[i + vector_len ..][0..vector_len].*;
        const x2: Vf32 = x[i + 2 * vector_len ..][0..vector_len].*;
        const y2: Vf32 = y[i + 2 * vector_len ..][0..vector_len].*;
        const x3: Vf32 = x[i + 3 * vector_len ..][0..vector_len].*;
        const y3: Vf32 = y[i + 3 * vector_len ..][0..vector_len].*;
        z[i..][0..vector_len].* = x0 * y0;
        z[i + vector_len ..][0..vector_len].* = x1 * y1;
        z[i + 2 * vector_len ..][0..vector_len].* = x2 * y2;
        z[i + 3 * vector_len ..][0..vector_len].* = x3 * y3;
    }
    while (i + vector_len <= z.len) : (i += vector_len) {
        const xv: Vf32 = x[i..][0..vector_len].*;
        const yv: Vf32 = y[i..][0..vector_len].*;
        z[i..][0..vector_len].* = xv * yv;
    }
    while (i < z.len) : (i += 1) z[i] = x[i] * y[i];
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

inline fn maskAnd(comptime W: usize, a: @Vector(W, bool), b: @Vector(W, bool)) @Vector(W, bool) {
    return @select(bool, a, b, @as(@Vector(W, bool), @splat(false)));
}

inline fn maskOr(comptime W: usize, a: @Vector(W, bool), b: @Vector(W, bool)) @Vector(W, bool) {
    return @select(bool, a, @as(@Vector(W, bool), @splat(true)), b);
}

inline fn maskNot(comptime W: usize, a: @Vector(W, bool)) @Vector(W, bool) {
    return @select(bool, a, @as(@Vector(W, bool), @splat(false)), @as(@Vector(W, bool), @splat(true)));
}

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
    const ret_neg1 = maskAnd(W, big, sign);
    const ret_inf = maskAnd(W, maskAnd(W, big, maskNot(W, sign)), x > @as(Vec, @splat(o_threshold)));
    const early = maskOr(W, maskOr(W, is_nan, ret_neg1), ret_inf);
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
        const wide = maskOr(W, k < @as(VecI, @splat(0)), k > @as(VecI, @splat(56)));
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
    const saturating = maskAnd(W, cross, maskNot(W, huge));
    const pos_arm = maskOr(W, saturating, maskAnd(W, maskNot(W, cross), mid));
    const neg_arm = maskAnd(W, maskNot(W, cross), maskAnd(W, maskNot(W, mid), norm));
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
    const t1 = @select(f32, maskOr(W, pos_arm, neg_arm), t_expm1, ax); // else arm: subnormal t = |x|
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

/// Number of independent accumulator chains in `vecSum` / `vecDot`: the
/// loops are add-latency bound, and eight chains keep the FMA/FADD
/// pipelines full where four left them half idle (measured, see the
/// commit that widened them). The chain count is part of the summation
/// order and therefore of the result bits.
pub const reduce_chains = 8;

pub inline fn vecSum(x: []const f32) f32 {
    if (x.len == 0) return 0;
    var acc: [reduce_chains]Vf32 = undefined;
    inline for (0..reduce_chains) |c| acc[c] = @splat(0);
    var i: usize = 0;
    while (i + reduce_chains * vector_len <= x.len) : (i += reduce_chains * vector_len) {
        inline for (0..reduce_chains) |c| acc[c] += @as(Vf32, x[i + c * vector_len ..][0..vector_len].*);
    }
    while (i + vector_len <= x.len) : (i += vector_len) {
        const xv: Vf32 = x[i..][0..vector_len].*;
        acc[0] += xv;
    }
    var total = acc[0];
    inline for (1..reduce_chains) |c| total += acc[c];
    var s = @reduce(.Add, total);
    while (i < x.len) : (i += 1) s += x[i];
    return s;
}

pub inline fn vecDiv(z: []f32, x: []const f32, y: []const f32) void {
    var i: usize = 0;
    while (i + 4 * vector_len <= z.len) : (i += 4 * vector_len) {
        const x0: Vf32 = x[i..][0..vector_len].*;
        const y0: Vf32 = y[i..][0..vector_len].*;
        const x1: Vf32 = x[i + vector_len ..][0..vector_len].*;
        const y1: Vf32 = y[i + vector_len ..][0..vector_len].*;
        const x2: Vf32 = x[i + 2 * vector_len ..][0..vector_len].*;
        const y2: Vf32 = y[i + 2 * vector_len ..][0..vector_len].*;
        const x3: Vf32 = x[i + 3 * vector_len ..][0..vector_len].*;
        const y3: Vf32 = y[i + 3 * vector_len ..][0..vector_len].*;
        z[i..][0..vector_len].* = x0 / y0;
        z[i + vector_len ..][0..vector_len].* = x1 / y1;
        z[i + 2 * vector_len ..][0..vector_len].* = x2 / y2;
        z[i + 3 * vector_len ..][0..vector_len].* = x3 / y3;
    }
    while (i + vector_len <= z.len) : (i += vector_len) {
        const xv: Vf32 = x[i..][0..vector_len].*;
        const yv: Vf32 = y[i..][0..vector_len].*;
        z[i..][0..vector_len].* = xv / yv;
    }
    while (i < z.len) : (i += 1) z[i] = x[i] / y[i];
}

pub inline fn vecMaximum(z: []f32, x: []const f32, y: []const f32) void {
    vecMaxMinBinary(.max, z, x, y);
}

pub inline fn vecMinimum(z: []f32, x: []const f32, y: []const f32) void {
    vecMaxMinBinary(.min, z, x, y);
}

inline fn vecMaxMinBinary(comptime op: ops.ElementwiseOp, z: []f32, x: []const f32, y: []const f32) void {
    var i: usize = 0;
    while (i + 4 * vector_len <= z.len) : (i += 4 * vector_len) {
        z[i..][0..vector_len].* = applyMaxMinVec(Vf32, op, x[i..][0..vector_len].*, y[i..][0..vector_len].*);
        z[i + vector_len ..][0..vector_len].* = applyMaxMinVec(Vf32, op, x[i + vector_len ..][0..vector_len].*, y[i + vector_len ..][0..vector_len].*);
        z[i + 2 * vector_len ..][0..vector_len].* = applyMaxMinVec(Vf32, op, x[i + 2 * vector_len ..][0..vector_len].*, y[i + 2 * vector_len ..][0..vector_len].*);
        z[i + 3 * vector_len ..][0..vector_len].* = applyMaxMinVec(Vf32, op, x[i + 3 * vector_len ..][0..vector_len].*, y[i + 3 * vector_len ..][0..vector_len].*);
    }
    while (i + vector_len <= z.len) : (i += vector_len) {
        z[i..][0..vector_len].* = applyMaxMinVec(Vf32, op, x[i..][0..vector_len].*, y[i..][0..vector_len].*);
    }
    while (i < z.len) : (i += 1) {
        const a = x[i];
        const b = y[i];
        z[i] = if (a != a or b != b) std.math.nan(f32) else if (comptime op == .max) @max(a, b) else @min(a, b);
    }
}

pub inline fn vecProd(x: []const f32) f32 {
    if (x.len == 0) return 1;
    var acc0: Vf32 = @splat(1);
    var acc1: Vf32 = @splat(1);
    var acc2: Vf32 = @splat(1);
    var acc3: Vf32 = @splat(1);
    var i: usize = 0;
    while (i + 4 * vector_len <= x.len) : (i += 4 * vector_len) {
        acc0 *= @as(Vf32, x[i..][0..vector_len].*);
        acc1 *= @as(Vf32, x[i + vector_len ..][0..vector_len].*);
        acc2 *= @as(Vf32, x[i + 2 * vector_len ..][0..vector_len].*);
        acc3 *= @as(Vf32, x[i + 3 * vector_len ..][0..vector_len].*);
    }
    while (i + vector_len <= x.len) : (i += vector_len) {
        const xv: Vf32 = x[i..][0..vector_len].*;
        acc0 *= xv;
    }
    var p = @reduce(.Mul, acc0 * acc1 * acc2 * acc3);
    while (i < x.len) : (i += 1) p *= x[i];
    return p;
}

/// Fused dot product over `reduce_chains` independent chains: every
/// product is folded into its chain with one `@mulAdd` (a single rounding
/// per term, the scalar tail included), the chains are summed in index
/// order and the lanes reduced last.
pub inline fn vecDot(x: []const f32, y: []const f32) f32 {
    if (x.len == 0) return 0;
    var acc: [reduce_chains]Vf32 = undefined;
    inline for (0..reduce_chains) |c| acc[c] = @splat(0);
    var i: usize = 0;
    while (i + reduce_chains * vector_len <= x.len) : (i += reduce_chains * vector_len) {
        inline for (0..reduce_chains) |c| {
            const xv: Vf32 = x[i + c * vector_len ..][0..vector_len].*;
            const yv: Vf32 = y[i + c * vector_len ..][0..vector_len].*;
            acc[c] = @mulAdd(Vf32, xv, yv, acc[c]);
        }
    }
    while (i + vector_len <= x.len) : (i += vector_len) {
        const xv: Vf32 = x[i..][0..vector_len].*;
        const yv: Vf32 = y[i..][0..vector_len].*;
        acc[0] = @mulAdd(Vf32, xv, yv, acc[0]);
    }
    var total = acc[0];
    inline for (1..reduce_chains) |c| total += acc[c];
    var s = @reduce(.Add, total);
    while (i < x.len) : (i += 1) s = @mulAdd(f32, x[i], y[i], s);
    return s;
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

pub inline fn vecElementwiseF64(comptime op: ops.ElementwiseOp, z: []f64, x: []const f64, y: []const f64) void {
    var i: usize = 0;
    while (i + 4 * vector_len_f64 <= z.len) : (i += 4 * vector_len_f64) {
        z[i..][0..vector_len_f64].* = applyElementwiseVecF64(op, x[i..][0..vector_len_f64].*, y[i..][0..vector_len_f64].*);
        z[i + vector_len_f64 ..][0..vector_len_f64].* = applyElementwiseVecF64(op, x[i + vector_len_f64 ..][0..vector_len_f64].*, y[i + vector_len_f64 ..][0..vector_len_f64].*);
        z[i + 2 * vector_len_f64 ..][0..vector_len_f64].* = applyElementwiseVecF64(op, x[i + 2 * vector_len_f64 ..][0..vector_len_f64].*, y[i + 2 * vector_len_f64 ..][0..vector_len_f64].*);
        z[i + 3 * vector_len_f64 ..][0..vector_len_f64].* = applyElementwiseVecF64(op, x[i + 3 * vector_len_f64 ..][0..vector_len_f64].*, y[i + 3 * vector_len_f64 ..][0..vector_len_f64].*);
    }
    while (i + vector_len_f64 <= z.len) : (i += vector_len_f64) {
        z[i..][0..vector_len_f64].* = applyElementwiseVecF64(op, x[i..][0..vector_len_f64].*, y[i..][0..vector_len_f64].*);
    }
    while (i < z.len) : (i += 1) {
        z[i] = applyElementwiseTyped(.f64, op, x[i], y[i]);
    }
}

pub inline fn vecElementwiseF16(comptime op: ops.ElementwiseOp, z: []f16, x: []const f16, y: []const f16) void {
    var i: usize = 0;
    while (i + 4 * vector_len_f16 <= z.len) : (i += 4 * vector_len_f16) {
        z[i..][0..vector_len_f16].* = applyElementwiseVecF16(op, x[i..][0..vector_len_f16].*, y[i..][0..vector_len_f16].*);
        z[i + vector_len_f16 ..][0..vector_len_f16].* = applyElementwiseVecF16(op, x[i + vector_len_f16 ..][0..vector_len_f16].*, y[i + vector_len_f16 ..][0..vector_len_f16].*);
        z[i + 2 * vector_len_f16 ..][0..vector_len_f16].* = applyElementwiseVecF16(op, x[i + 2 * vector_len_f16 ..][0..vector_len_f16].*, y[i + 2 * vector_len_f16 ..][0..vector_len_f16].*);
        z[i + 3 * vector_len_f16 ..][0..vector_len_f16].* = applyElementwiseVecF16(op, x[i + 3 * vector_len_f16 ..][0..vector_len_f16].*, y[i + 3 * vector_len_f16 ..][0..vector_len_f16].*);
    }
    while (i + vector_len_f16 <= z.len) : (i += vector_len_f16) {
        z[i..][0..vector_len_f16].* = applyElementwiseVecF16(op, x[i..][0..vector_len_f16].*, y[i..][0..vector_len_f16].*);
    }
    while (i < z.len) : (i += 1) {
        z[i] = applyElementwiseTyped(.f16, op, x[i], y[i]);
    }
}

pub inline fn vecElementwiseBf16(comptime op: ops.ElementwiseOp, z: []u16, x: []const u16, y: []const u16) void {
    var i: usize = 0;
    while (i + 4 * vector_len <= z.len) : (i += 4 * vector_len) {
        z[i..][0..vector_len].* = f32VecToBf16(applyElementwiseVecF32(op, bf16VecToF32(x[i..][0..vector_len].*), bf16VecToF32(y[i..][0..vector_len].*)));
        z[i + vector_len ..][0..vector_len].* = f32VecToBf16(applyElementwiseVecF32(op, bf16VecToF32(x[i + vector_len ..][0..vector_len].*), bf16VecToF32(y[i + vector_len ..][0..vector_len].*)));
        z[i + 2 * vector_len ..][0..vector_len].* = f32VecToBf16(applyElementwiseVecF32(op, bf16VecToF32(x[i + 2 * vector_len ..][0..vector_len].*), bf16VecToF32(y[i + 2 * vector_len ..][0..vector_len].*)));
        z[i + 3 * vector_len ..][0..vector_len].* = f32VecToBf16(applyElementwiseVecF32(op, bf16VecToF32(x[i + 3 * vector_len ..][0..vector_len].*), bf16VecToF32(y[i + 3 * vector_len ..][0..vector_len].*)));
    }
    while (i + vector_len <= z.len) : (i += vector_len) {
        z[i..][0..vector_len].* = f32VecToBf16(applyElementwiseVecF32(op, bf16VecToF32(x[i..][0..vector_len].*), bf16VecToF32(y[i..][0..vector_len].*)));
    }
    while (i < z.len) : (i += 1) {
        z[i] = applyElementwiseTyped(.bf16, op, x[i], y[i]);
    }
}

/// torch.maximum/minimum on a vector: NaN in either lane propagates NaN
/// (bare @max/@min follow IEEE maxNum and would drop it).
pub inline fn applyMaxMinVec(comptime V: type, comptime op: ops.ElementwiseOp, a: V, b: V) V {
    const Elem = @typeInfo(V).vector.child;
    const raw = if (comptime op == .max) @max(a, b) else @min(a, b);
    const nan_v: V = @splat(std.math.nan(Elem));
    return @select(Elem, a != a, nan_v, @select(Elem, b != b, nan_v, raw));
}

pub inline fn applyElementwiseVecF32(comptime op: ops.ElementwiseOp, a: Vf32, b: Vf32) Vf32 {
    return switch (op) {
        .add => a + b,
        .sub => a - b,
        .mul => a * b,
        .div => a / b,
        .max, .min => applyMaxMinVec(Vf32, op, a, b),
    };
}

pub inline fn applyElementwiseVecF64(comptime op: ops.ElementwiseOp, a: Vf64, b: Vf64) Vf64 {
    return switch (op) {
        .add => a + b,
        .sub => a - b,
        .mul => a * b,
        .div => a / b,
        .max, .min => applyMaxMinVec(Vf64, op, a, b),
    };
}

pub inline fn applyElementwiseVecF16(comptime op: ops.ElementwiseOp, a: common.Vf16, b: common.Vf16) common.Vf16 {
    return switch (op) {
        .add => a + b,
        .sub => a - b,
        .mul => a * b,
        .div => a / b,
        .max, .min => applyMaxMinVec(common.Vf16, op, a, b),
    };
}

pub inline fn vecSumF64(x: []const f64) f64 {
    var i: usize = 0;
    var acc0: Vf64 = @splat(0);
    var acc1: Vf64 = @splat(0);
    var acc2: Vf64 = @splat(0);
    var acc3: Vf64 = @splat(0);

    while (i + 4 * vector_len_f64 <= x.len) : (i += 4 * vector_len_f64) {
        acc0 += x[i..][0..vector_len_f64].*;
        acc1 += x[i + vector_len_f64 ..][0..vector_len_f64].*;
        acc2 += x[i + 2 * vector_len_f64 ..][0..vector_len_f64].*;
        acc3 += x[i + 3 * vector_len_f64 ..][0..vector_len_f64].*;
    }
    while (i + vector_len_f64 <= x.len) : (i += vector_len_f64) {
        acc0 += x[i..][0..vector_len_f64].*;
    }

    var s = @reduce(.Add, acc0 + acc1 + acc2 + acc3);
    while (i < x.len) : (i += 1) s += x[i];
    return s;
}

pub inline fn vecDotF64(x: []const f64, y: []const f64) f64 {
    var i: usize = 0;
    var acc0: Vf64 = @splat(0);
    var acc1: Vf64 = @splat(0);
    var acc2: Vf64 = @splat(0);
    var acc3: Vf64 = @splat(0);

    while (i + 4 * vector_len_f64 <= x.len) : (i += 4 * vector_len_f64) {
        acc0 += @as(Vf64, x[i..][0..vector_len_f64].*) * @as(Vf64, y[i..][0..vector_len_f64].*);
        acc1 += @as(Vf64, x[i + vector_len_f64 ..][0..vector_len_f64].*) * @as(Vf64, y[i + vector_len_f64 ..][0..vector_len_f64].*);
        acc2 += @as(Vf64, x[i + 2 * vector_len_f64 ..][0..vector_len_f64].*) * @as(Vf64, y[i + 2 * vector_len_f64 ..][0..vector_len_f64].*);
        acc3 += @as(Vf64, x[i + 3 * vector_len_f64 ..][0..vector_len_f64].*) * @as(Vf64, y[i + 3 * vector_len_f64 ..][0..vector_len_f64].*);
    }
    while (i + vector_len_f64 <= x.len) : (i += vector_len_f64) {
        acc0 += @as(Vf64, x[i..][0..vector_len_f64].*) * @as(Vf64, y[i..][0..vector_len_f64].*);
    }

    var s = @reduce(.Add, acc0 + acc1 + acc2 + acc3);
    while (i < x.len) : (i += 1) s += x[i] * y[i];
    return s;
}

pub inline fn vecSumF16ToF32(x: []const f16) f32 {
    var i: usize = 0;
    var acc0: Vf32 = @splat(0);
    var acc1: Vf32 = @splat(0);
    var acc2: Vf32 = @splat(0);
    var acc3: Vf32 = @splat(0);

    while (i + 4 * vector_len <= x.len) : (i += 4 * vector_len) {
        acc0 += @as(Vf32, @floatCast(@as(Vf16ForF32, x[i..][0..vector_len].*)));
        acc1 += @as(Vf32, @floatCast(@as(Vf16ForF32, x[i + vector_len ..][0..vector_len].*)));
        acc2 += @as(Vf32, @floatCast(@as(Vf16ForF32, x[i + 2 * vector_len ..][0..vector_len].*)));
        acc3 += @as(Vf32, @floatCast(@as(Vf16ForF32, x[i + 3 * vector_len ..][0..vector_len].*)));
    }
    while (i + vector_len <= x.len) : (i += vector_len) {
        acc0 += @as(Vf32, @floatCast(@as(Vf16ForF32, x[i..][0..vector_len].*)));
    }

    var s = @reduce(.Add, acc0 + acc1 + acc2 + acc3);
    while (i < x.len) : (i += 1) s += @floatCast(x[i]);
    return s;
}

pub inline fn vecDotF16ToF32(x: []const f16, y: []const f16) f32 {
    var i: usize = 0;
    var acc0: Vf32 = @splat(0);
    var acc1: Vf32 = @splat(0);
    var acc2: Vf32 = @splat(0);
    var acc3: Vf32 = @splat(0);
    var acc4: Vf32 = @splat(0);
    var acc5: Vf32 = @splat(0);
    var acc6: Vf32 = @splat(0);
    var acc7: Vf32 = @splat(0);

    while (i + 8 * vector_len <= x.len) : (i += 8 * vector_len) {
        const x0: Vf32 = @floatCast(@as(Vf16ForF32, x[i..][0..vector_len].*));
        const y0: Vf32 = @floatCast(@as(Vf16ForF32, y[i..][0..vector_len].*));
        const x1: Vf32 = @floatCast(@as(Vf16ForF32, x[i + vector_len ..][0..vector_len].*));
        const y1: Vf32 = @floatCast(@as(Vf16ForF32, y[i + vector_len ..][0..vector_len].*));
        const x2: Vf32 = @floatCast(@as(Vf16ForF32, x[i + 2 * vector_len ..][0..vector_len].*));
        const y2: Vf32 = @floatCast(@as(Vf16ForF32, y[i + 2 * vector_len ..][0..vector_len].*));
        const x3: Vf32 = @floatCast(@as(Vf16ForF32, x[i + 3 * vector_len ..][0..vector_len].*));
        const y3: Vf32 = @floatCast(@as(Vf16ForF32, y[i + 3 * vector_len ..][0..vector_len].*));
        const x4: Vf32 = @floatCast(@as(Vf16ForF32, x[i + 4 * vector_len ..][0..vector_len].*));
        const y4: Vf32 = @floatCast(@as(Vf16ForF32, y[i + 4 * vector_len ..][0..vector_len].*));
        const x5: Vf32 = @floatCast(@as(Vf16ForF32, x[i + 5 * vector_len ..][0..vector_len].*));
        const y5: Vf32 = @floatCast(@as(Vf16ForF32, y[i + 5 * vector_len ..][0..vector_len].*));
        const x6: Vf32 = @floatCast(@as(Vf16ForF32, x[i + 6 * vector_len ..][0..vector_len].*));
        const y6: Vf32 = @floatCast(@as(Vf16ForF32, y[i + 6 * vector_len ..][0..vector_len].*));
        const x7: Vf32 = @floatCast(@as(Vf16ForF32, x[i + 7 * vector_len ..][0..vector_len].*));
        const y7: Vf32 = @floatCast(@as(Vf16ForF32, y[i + 7 * vector_len ..][0..vector_len].*));
        acc0 += x0 * y0;
        acc1 += x1 * y1;
        acc2 += x2 * y2;
        acc3 += x3 * y3;
        acc4 += x4 * y4;
        acc5 += x5 * y5;
        acc6 += x6 * y6;
        acc7 += x7 * y7;
    }
    while (i + 4 * vector_len <= x.len) : (i += 4 * vector_len) {
        const x0: Vf32 = @floatCast(@as(Vf16ForF32, x[i..][0..vector_len].*));
        const y0: Vf32 = @floatCast(@as(Vf16ForF32, y[i..][0..vector_len].*));
        const x1: Vf32 = @floatCast(@as(Vf16ForF32, x[i + vector_len ..][0..vector_len].*));
        const y1: Vf32 = @floatCast(@as(Vf16ForF32, y[i + vector_len ..][0..vector_len].*));
        const x2: Vf32 = @floatCast(@as(Vf16ForF32, x[i + 2 * vector_len ..][0..vector_len].*));
        const y2: Vf32 = @floatCast(@as(Vf16ForF32, y[i + 2 * vector_len ..][0..vector_len].*));
        const x3: Vf32 = @floatCast(@as(Vf16ForF32, x[i + 3 * vector_len ..][0..vector_len].*));
        const y3: Vf32 = @floatCast(@as(Vf16ForF32, y[i + 3 * vector_len ..][0..vector_len].*));
        acc0 += x0 * y0;
        acc1 += x1 * y1;
        acc2 += x2 * y2;
        acc3 += x3 * y3;
    }
    while (i + vector_len <= x.len) : (i += vector_len) {
        const xv: Vf32 = @floatCast(@as(Vf16ForF32, x[i..][0..vector_len].*));
        const yv: Vf32 = @floatCast(@as(Vf16ForF32, y[i..][0..vector_len].*));
        acc0 += xv * yv;
    }

    var s = @reduce(.Add, acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7);
    while (i < x.len) : (i += 1) s += @as(f32, @floatCast(x[i])) * @as(f32, @floatCast(y[i]));
    return s;
}

pub inline fn vecSumBf16ToF32(x: []const u16) f32 {
    var i: usize = 0;
    var acc0: common.Vf32ForF16 = @splat(0);
    var acc1: common.Vf32ForF16 = @splat(0);
    var acc2: common.Vf32ForF16 = @splat(0);
    var acc3: common.Vf32ForF16 = @splat(0);

    while (i + 4 * vector_len_f16 <= x.len) : (i += 4 * vector_len_f16) {
        acc0 += bf16VecToF32Wide(x[i..][0..vector_len_f16].*);
        acc1 += bf16VecToF32Wide(x[i + vector_len_f16 ..][0..vector_len_f16].*);
        acc2 += bf16VecToF32Wide(x[i + 2 * vector_len_f16 ..][0..vector_len_f16].*);
        acc3 += bf16VecToF32Wide(x[i + 3 * vector_len_f16 ..][0..vector_len_f16].*);
    }
    while (i + vector_len_f16 <= x.len) : (i += vector_len_f16) {
        acc0 += bf16VecToF32Wide(x[i..][0..vector_len_f16].*);
    }

    var s = @reduce(.Add, acc0 + acc1 + acc2 + acc3);
    while (i < x.len) : (i += 1) s += dtype_mod.bf16ToF32(x[i]);
    return s;
}

pub inline fn vecDotBf16ToF32(x: []const u16, y: []const u16) f32 {
    var i: usize = 0;
    var acc0: common.Vf32ForF16 = @splat(0);
    var acc1: common.Vf32ForF16 = @splat(0);
    var acc2: common.Vf32ForF16 = @splat(0);
    var acc3: common.Vf32ForF16 = @splat(0);

    while (i + 4 * vector_len_f16 <= x.len) : (i += 4 * vector_len_f16) {
        acc0 += bf16VecToF32Wide(x[i..][0..vector_len_f16].*) * bf16VecToF32Wide(y[i..][0..vector_len_f16].*);
        acc1 += bf16VecToF32Wide(x[i + vector_len_f16 ..][0..vector_len_f16].*) * bf16VecToF32Wide(y[i + vector_len_f16 ..][0..vector_len_f16].*);
        acc2 += bf16VecToF32Wide(x[i + 2 * vector_len_f16 ..][0..vector_len_f16].*) * bf16VecToF32Wide(y[i + 2 * vector_len_f16 ..][0..vector_len_f16].*);
        acc3 += bf16VecToF32Wide(x[i + 3 * vector_len_f16 ..][0..vector_len_f16].*) * bf16VecToF32Wide(y[i + 3 * vector_len_f16 ..][0..vector_len_f16].*);
    }
    while (i + vector_len_f16 <= x.len) : (i += vector_len_f16) {
        acc0 += bf16VecToF32Wide(x[i..][0..vector_len_f16].*) * bf16VecToF32Wide(y[i..][0..vector_len_f16].*);
    }

    var s = @reduce(.Add, acc0 + acc1 + acc2 + acc3);
    while (i < x.len) : (i += 1) s += dtype_mod.bf16ToF32(x[i]) * dtype_mod.bf16ToF32(y[i]);
    return s;
}

pub inline fn bf16VecToF32(bits: common.Vu16ForF32) Vf32 {
    const widened: common.Vu32ForF32 = @as(common.Vu32ForF32, @intCast(bits)) << @as(common.Vu32ForF32, @splat(16));
    return @bitCast(widened);
}

pub inline fn bf16VecToF32Wide(bits: common.Vu16ForF16) common.Vf32ForF16 {
    const widened: common.Vu32ForF16 = @as(common.Vu32ForF16, @intCast(bits)) << @as(common.Vu32ForF16, @splat(16));
    return @bitCast(widened);
}

pub inline fn f32VecToBf16(values: Vf32) common.Vu16ForF32 {
    const bits: common.Vu32ForF32 = @bitCast(values);
    const lsb = (bits >> @as(common.Vu32ForF32, @splat(16))) & @as(common.Vu32ForF32, @splat(1));
    const rounded = bits + @as(common.Vu32ForF32, @splat(0x7fff)) + lsb;
    return @truncate(rounded >> @as(common.Vu32ForF32, @splat(16)));
}

test {
    _ = @import("primitives_tests.zig");
}

// ---------------------------------------------------------------------------
// f16 row-block attention primitives: the register-blocked inner loops of
// online-softmax attention over contiguous or strided f16 rows, exposed so
// sparse/selective attention operators (models.research.subq) share one tuned kernel
// shape with the dense attention path. Four rows per iteration share the
// query's vector loads; the accumulator is loaded and stored once per four
// contributions.

/// Reduce-max over a slice (softmax gauge candidate); -inf for empty input.
pub inline fn vecMaxReduce(x: []const f32) f32 {
    var m: f32 = -std.math.inf(f32);
    for (x) |value| m = @max(m, value);
    return m;
}

inline fn widenF16x8(row: []const f16) @Vector(8, f32) {
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
