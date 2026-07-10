//! Low-level @Vector primitives relocated out of vector.zig. See vector.zig for
//! the shared core (V* aliases, vector_len*) which this module aliases via `vm`
//! so the moved bodies compile unchanged.

const std = @import("std");
const ops = @import("../ops.zig");
const dtype_mod = @import("../../dtype.zig");
const vm = @import("common.zig");

const DType = dtype_mod.DType;

// Shared core symbols defined in vector.zig, aliased here so the moved bodies
// compile unchanged.
const vector_len = vm.vector_len;
const vector_len_f64 = vm.vector_len_f64;
const vector_len_f16 = vm.vector_len_f16;
const Vf32 = vm.Vf32;
const Vf64 = vm.Vf64;
const Vf16 = vm.Vf16;
const Vf32ForF16 = vm.Vf32ForF16;
const Vf16ForF32 = vm.Vf16ForF32;
const Vu16ForF16 = vm.Vu16ForF16;
const Vu32ForF16 = vm.Vu32ForF16;
const Vu16ForF32 = vm.Vu16ForF32;
const Vu32ForF32 = vm.Vu32ForF32;
/// Elementwise typed scalar op. Relocated here from elementwise.zig so this leaf
/// module (and the vector children that depend on it) no longer reach back up to
/// the vector.zig barrel — keeps `primitives` a true leaf of the import graph.
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
        z[i..][0..vector_len].* = @as(Vf32, x[i..][0..vector_len].*) * gatedActivationVec(op, y[i..][0..vector_len].*);
        z[i + vector_len ..][0..vector_len].* = @as(Vf32, x[i + vector_len ..][0..vector_len].*) * gatedActivationVec(op, y[i + vector_len ..][0..vector_len].*);
        z[i + 2 * vector_len ..][0..vector_len].* = @as(Vf32, x[i + 2 * vector_len ..][0..vector_len].*) * gatedActivationVec(op, y[i + 2 * vector_len ..][0..vector_len].*);
        z[i + 3 * vector_len ..][0..vector_len].* = @as(Vf32, x[i + 3 * vector_len ..][0..vector_len].*) * gatedActivationVec(op, y[i + 3 * vector_len ..][0..vector_len].*);
    }
    while (i + vector_len <= z.len) : (i += vector_len) {
        z[i..][0..vector_len].* = @as(Vf32, x[i..][0..vector_len].*) * gatedActivationVec(op, y[i..][0..vector_len].*);
    }
    while (i < z.len) : (i += 1) z[i] = x[i] * ops.gatedActivationScalar(op, y[i]);
}

pub inline fn applyUnaryVec(comptime op: ops.UnaryOp, value: Vf32) Vf32 {
    return switch (op) {
        .relu => @max(value, @as(Vf32, @splat(0))),
        .exp => @exp(value),
        .sqrt => @sqrt(value),
        .rsqrt => @as(Vf32, @splat(1)) / @sqrt(value),
        .sigmoid => sigmoidVec(value),
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
        .softcap_30 => @as(Vf32, @splat(30.0)) * tanhVec(value * @as(Vf32, @splat(1.0 / 30.0))),
        .softcap_15 => @as(Vf32, @splat(15.0)) * tanhVec(value * @as(Vf32, @splat(1.0 / 15.0))),
        .gelu_quant => geluQuantVec(value),
        .elu => perLaneUnary(.elu, value),
        .gelu_erf => perLaneUnary(.gelu_erf, value),
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
/// (expm1-based elu, erf-based gelu): evaluates `ops.unaryScalar` on each
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
/// tanh-gelu, f16-round the output, with hard clamps at +/-10.
pub inline fn geluQuantVec(value: Vf32) Vf32 {
    const F16Vec = @Vector(vector_len, f16);
    const xr: Vf32 = @floatCast(@as(F16Vec, @floatCast(value)));
    const g = @as(Vf32, @splat(0.5)) * xr * (@as(Vf32, @splat(1)) + tanhVec(geluTanhArgVec(xr)));
    const gr: Vf32 = @floatCast(@as(F16Vec, @floatCast(g)));
    const ten: Vf32 = @splat(10);
    const lo_clamped = @select(f32, value <= -ten, @as(Vf32, @splat(0)), gr);
    return @select(f32, value >= ten, value, lo_clamped);
}

pub inline fn gatedActivationVec(comptime op: ops.GatedOp, value: Vf32) Vf32 {
    return switch (op) {
        .glu => sigmoidVec(value),
        .swiglu => value * sigmoidVec(value),
        .geglu => @as(Vf32, @splat(0.5)) * value * (@as(Vf32, @splat(1)) + tanhVec(geluTanhArgVec(value))),
    };
}

pub inline fn sigmoidVec(value: Vf32) Vf32 {
    const one: Vf32 = @splat(1);
    return one / (one + vexpf(vector_len, -value));
}

pub inline fn tanhVec(value: Vf32) Vf32 {
    // tanh(v) = 2*sigmoid(2v) - 1, evaluated in the sign-stable form on each
    // branch (the exponent argument stays <= 0 so exp never overflows).
    const zero: Vf32 = @splat(0);
    const one: Vf32 = @splat(1);
    const two: Vf32 = @splat(2);
    const positive = two / (@exp(-two * value) + one) - one;
    const negative = one - two / (@exp(two * value) + one);
    return @select(f32, value >= zero, positive, negative);
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

pub inline fn vecSum(x: []const f32) f32 {
    if (x.len == 0) return 0;
    var acc0: Vf32 = @splat(0);
    var acc1: Vf32 = @splat(0);
    var acc2: Vf32 = @splat(0);
    var acc3: Vf32 = @splat(0);
    var i: usize = 0;
    while (i + 4 * vector_len <= x.len) : (i += 4 * vector_len) {
        acc0 += x[i..][0..vector_len].*;
        acc1 += x[i + vector_len ..][0..vector_len].*;
        acc2 += x[i + 2 * vector_len ..][0..vector_len].*;
        acc3 += x[i + 3 * vector_len ..][0..vector_len].*;
    }
    while (i + vector_len <= x.len) : (i += vector_len) {
        const xv: Vf32 = x[i..][0..vector_len].*;
        acc0 += xv;
    }
    var s = @reduce(.Add, acc0 + acc1 + acc2 + acc3);
    while (i < x.len) : (i += 1) s += x[i];
    return s;
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

pub inline fn vecDot(x: []const f32, y: []const f32) f32 {
    if (x.len == 0) return 0;
    var acc0: Vf32 = @splat(0);
    var acc1: Vf32 = @splat(0);
    var acc2: Vf32 = @splat(0);
    var acc3: Vf32 = @splat(0);
    var i: usize = 0;
    while (i + 4 * vector_len <= x.len) : (i += 4 * vector_len) {
        acc0 += @as(Vf32, x[i..][0..vector_len].*) * @as(Vf32, y[i..][0..vector_len].*);
        acc1 += @as(Vf32, x[i + vector_len ..][0..vector_len].*) * @as(Vf32, y[i + vector_len ..][0..vector_len].*);
        acc2 += @as(Vf32, x[i + 2 * vector_len ..][0..vector_len].*) * @as(Vf32, y[i + 2 * vector_len ..][0..vector_len].*);
        acc3 += @as(Vf32, x[i + 3 * vector_len ..][0..vector_len].*) * @as(Vf32, y[i + 3 * vector_len ..][0..vector_len].*);
    }
    while (i + vector_len <= x.len) : (i += vector_len) {
        const xv: Vf32 = x[i..][0..vector_len].*;
        const yv: Vf32 = y[i..][0..vector_len].*;
        acc0 += xv * yv;
    }
    var s = @reduce(.Add, acc0 + acc1 + acc2 + acc3);
    while (i < x.len) : (i += 1) s += x[i] * y[i];
    return s;
}

// Fused multiply-add of a contiguous slice with a broadcast scalar: out += in * s.
// This is the hot path of matmul / matmulTransA — every output row receives
// one vecFmaScalar per k-step. Compiles to a tight loop of vfma instructions
// on AArch64 and AVX2/AVX-512 on x86_64.
pub inline fn vecFmaScalar(out: []f32, in: []const f32, s: f32) void {
    const sv: Vf32 = @splat(s);
    var i: usize = 0;
    while (i + vector_len <= out.len) : (i += vector_len) {
        const ov: Vf32 = out[i..][0..vector_len].*;
        const iv: Vf32 = in[i..][0..vector_len].*;
        out[i..][0..vector_len].* = ov + iv * sv;
    }
    while (i < out.len) : (i += 1) out[i] += in[i] * s;
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

pub inline fn applyElementwiseVecF16(comptime op: ops.ElementwiseOp, a: Vf16, b: Vf16) Vf16 {
    return switch (op) {
        .add => a + b,
        .sub => a - b,
        .mul => a * b,
        .div => a / b,
        .max, .min => applyMaxMinVec(Vf16, op, a, b),
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
    var acc0: Vf32ForF16 = @splat(0);
    var acc1: Vf32ForF16 = @splat(0);
    var acc2: Vf32ForF16 = @splat(0);
    var acc3: Vf32ForF16 = @splat(0);

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
    var acc0: Vf32ForF16 = @splat(0);
    var acc1: Vf32ForF16 = @splat(0);
    var acc2: Vf32ForF16 = @splat(0);
    var acc3: Vf32ForF16 = @splat(0);

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

pub inline fn bf16VecToF32(bits: Vu16ForF32) Vf32 {
    const widened: Vu32ForF32 = @as(Vu32ForF32, @intCast(bits)) << @as(Vu32ForF32, @splat(16));
    return @bitCast(widened);
}

pub inline fn bf16VecToF32Wide(bits: Vu16ForF16) Vf32ForF16 {
    const widened: Vu32ForF16 = @as(Vu32ForF16, @intCast(bits)) << @as(Vu32ForF16, @splat(16));
    return @bitCast(widened);
}

pub inline fn f32VecToBf16(values: Vf32) Vu16ForF32 {
    const bits: Vu32ForF32 = @bitCast(values);
    const lsb = (bits >> @as(Vu32ForF32, @splat(16))) & @as(Vu32ForF32, @splat(1));
    const rounded = bits + @as(Vu32ForF32, @splat(0x7fff)) + lsb;
    return @truncate(rounded >> @as(Vu32ForF32, @splat(16)));
}

test {
    _ = @import("primitives_tests.zig");
}
