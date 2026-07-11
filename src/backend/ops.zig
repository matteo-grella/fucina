const std = @import("std");

pub const ElementwiseOp = enum {
    add,
    sub,
    mul,
    div,
    // torch.maximum/minimum semantics: NaN in either operand propagates
    // NaN (NOT the IEEE maxNum rule Zig's bare @max/@min follow).
    max,
    min,
};

/// Elementwise comparison predicates (mask-producing; see
/// `exec/elementwise.zig` `compare`/`compareScalar` — there is no backend
/// kernel, the exec loops are memory-bound).
pub const CompareOp = enum {
    eq,
    ne,
    lt,
    le,
    gt,
    ge,
};

/// IEEE-754 comparison semantics: any comparison involving NaN is false,
/// except `ne`, which is true (Zig's float operators have exactly these
/// semantics, made explicit here as the documented contract).
pub inline fn compareScalar(comptime op: CompareOp, a: f32, b: f32) bool {
    return switch (op) {
        .eq => a == b,
        .ne => a != b,
        .lt => a < b,
        .le => a <= b,
        .gt => a > b,
        .ge => a >= b,
    };
}

pub const UnaryOp = enum {
    relu,
    exp,
    sqrt,
    rsqrt,
    sigmoid,
    silu,
    log,
    log1p,
    neg,
    abs,
    sin,
    cos,
    tanh,
    fast_tanh,
    gelu,
    quick_gelu,
    softcap_30,
    // nanochat's logit softcap (gpt.py:515): 15·tanh(x/15) as a single
    // fused elementwise pass.
    softcap_15,
    // ggml's GGML_GELU_FP16 gelu: the input is rounded to f16, the tanh-approx
    // gelu is evaluated, the result is rounded to f16 (a 16-bit LUT in ggml),
    // with hard clamps y=0 for x<=-10 and y=x for x>=10. Used to match llama.cpp
    // numerically (Gemma's GeGLU). `gelu` is the exact (more accurate) form.
    gelu_quant,
    // ELU (alpha = 1): x > 0 ? x : expm1(x). Matches ggml_vec_elu_f32
    // (omnivoice SemanticEncoder).
    elu,
    // Exact-erf GELU: 0.5*x*(1 + erf(x/sqrt(2))). Matches ggml_vec_gelu_erf_f32
    // (which calls libm erff) — NOT the tanh approximation (`gelu`).
    gelu_erf,
    floor,
    ceil,
    // Round-half-to-even (torch.round / IEEE roundTiesToEven), NOT Zig's
    // @round (half away from zero). Implemented with the 2^23
    // magic-number trick so the scalar and vector legs are bit-identical.
    round,
    // sign(x): 1 for x > 0, -1 for x < 0, x itself otherwise — preserves
    // ±0 and propagates NaN (the numpy/torch convention).
    sign,
    reciprocal,
};

pub const GatedOp = enum {
    glu,
    swiglu,
    // GeGLU: the gated activation is GELU (tanh approximation) instead of SiLU.
    // Used by Gemma's GeGLU FFN/MoE; `geglu(up, gate) = up * gelu(gate)`.
    geglu,
    // DeepSeek V4's clamped SwiGLU: gate is clamped to <= +10 before SiLU and
    // up is clamped to [-10, 10] before the multiply (the model's
    // swiglu_clamp_exp metadata; validated == 10 at load).
    swiglu_clamp10,
};

pub inline fn unaryScalar(comptime op: UnaryOp, value: f32) f32 {
    return switch (op) {
        .relu => if (value > 0) value else 0,
        .exp => @exp(value),
        .sqrt => @sqrt(value),
        .rsqrt => 1 / @sqrt(value),
        .sigmoid => sigmoidScalar(value),
        .silu => value * sigmoidScalar(value),
        .log => @log(value),
        .log1p => @log(1 + value),
        .neg => -value,
        .abs => @abs(value),
        .sin => @sin(value),
        .cos => @cos(value),
        .tanh => std.math.tanh(value),
        .fast_tanh => fastTanhScalar(value),
        .gelu => geluScalar(value),
        .quick_gelu => quickGeluScalar(value),
        .softcap_30 => 30.0 * std.math.tanh(value * (1.0 / 30.0)),
        .softcap_15 => 15.0 * std.math.tanh(value * (1.0 / 15.0)),
        .gelu_quant => geluQuantScalar(value),
        .elu => if (value > 0) value else std.math.expm1(value),
        .gelu_erf => 0.5 * value * (1 + erff(value * 0.70710678118654752440084436210484)),
        .floor => @floor(value),
        .ceil => @ceil(value),
        .round => rintScalar(value),
        .sign => if (value > 0) 1 else if (value < 0) -1 else value,
        .reciprocal => 1 / value,
    };
}

/// Round-half-to-even via the 2^23 magic-number trick (the default FP
/// rounding mode does the tie-break): exact for every |x| < 2^23;
/// magnitudes at or above 2^23 are already integral and pass through, as
/// do NaN and ±inf. Bit-identical to the vector leg by construction.
pub inline fn rintScalar(value: f32) f32 {
    const big: f32 = 8388608.0; // 2^23
    const ax = @abs(value);
    if (!(ax < big)) return value;
    const shifted = (ax + big) - big;
    return std.math.copysign(shifted, value);
}

pub inline fn gatedActivationScalar(comptime op: GatedOp, value: f32) f32 {
    return switch (op) {
        .glu => sigmoidScalar(value),
        .swiglu => value * sigmoidScalar(value),
        .geglu => geluScalar(value),
        .swiglu_clamp10 => blk: {
            const g = @min(value, 10.0);
            break :blk g * sigmoidScalar(g);
        },
    };
}

/// The full gated pair for ops whose clamping also touches the `up` input
/// (`swiglu_clamp10`); all other ops reduce to `up * gatedActivationScalar`.
pub inline fn gatedPairScalar(comptime op: GatedOp, gate: f32, up: f32) f32 {
    return switch (op) {
        .swiglu_clamp10 => gatedActivationScalar(op, gate) * @min(@max(up, -10.0), 10.0),
        else => up * gatedActivationScalar(op, gate),
    };
}

/// NeuralAmpModelerCore activations.h fast_tanh rational approximation.
pub inline fn fastTanhScalar(value: f32) f32 {
    const ax = @abs(value);
    const x2 = value * value;
    return value * (2.45550750702956 + 2.45550750702956 * ax + (0.893229853513558 + 0.821226666969744 * ax) * x2) /
        (2.44506634652299 + (2.44506634652299 + x2) * @abs(value + 0.814642734961073 * value * ax));
}

pub inline fn sigmoidScalar(value: f32) f32 {
    return 1 / (1 + @exp(-value));
}

pub inline fn geluScalar(value: f32) f32 {
    return 0.5 * value * (1 + std.math.tanh(geluTanhArg(value)));
}

pub inline fn quickGeluScalar(value: f32) f32 {
    return value * sigmoidScalar(1.702 * value);
}

/// ggml's f16-LUT gelu: round input to f16, exact tanh-gelu, round output to
/// f16, with hard clamps. Reproduces `ggml_vec_gelu_f32` (GGML_GELU_FP16).
pub inline fn geluQuantScalar(value: f32) f32 {
    if (value <= -10) return 0;
    if (value >= 10) return value;
    const xr: f32 = @floatCast(@as(f16, @floatCast(value)));
    const g = geluScalar(xr);
    return @floatCast(@as(f16, @floatCast(g)));
}

pub inline fn geluTanhArg(value: f32) f32 {
    const sqrt_2_over_pi: f32 = 0.7978845608028654;
    return sqrt_2_over_pi * (value + 0.044715 * value * value * value);
}

// ---------------------------------------------------------------------------
// erff: single-precision error function, translated faithfully from musl libc
// src/math/erff.c (MIT licensed; FDLIBM lineage, Copyright (C) 1993 Sun
// Microsystems). Same branch structure, constants, and float word tricks as
// musl so `gelu_erf` matches ggml_vec_gelu_erf_f32 (which calls libm erff).
// ---------------------------------------------------------------------------

const erf_data = struct {
    const erx: f32 = 8.4506291151e-01; // 0x3f58560b
    // Coefficients for approximation to erf on [0, 0.84375]
    const efx8: f32 = 1.0270333290e+00; // 0x3f8375d4
    const pp0: f32 = 1.2837916613e-01; // 0x3e0375d4
    const pp1: f32 = -3.2504209876e-01; // 0xbea66beb
    const pp2: f32 = -2.8481749818e-02; // 0xbce9528f
    const pp3: f32 = -5.7702702470e-03; // 0xbbbd1489
    const pp4: f32 = -2.3763017452e-05; // 0xb7c756b1
    const qq1: f32 = 3.9791721106e-01; // 0x3ecbbbce
    const qq2: f32 = 6.5022252500e-02; // 0x3d852a63
    const qq3: f32 = 5.0813062117e-03; // 0x3ba68116
    const qq4: f32 = 1.3249473704e-04; // 0x390aee49
    const qq5: f32 = -3.9602282413e-06; // 0xb684e21a
    // Coefficients for approximation to erf on [0.84375, 1.25]
    const pa0: f32 = -2.3621185683e-03; // 0xbb1acdc6
    const pa1: f32 = 4.1485610604e-01; // 0x3ed46805
    const pa2: f32 = -3.7220788002e-01; // 0xbebe9208
    const pa3: f32 = 3.1834661961e-01; // 0x3ea2fe54
    const pa4: f32 = -1.1089469492e-01; // 0xbde31cc2
    const pa5: f32 = 3.5478305072e-02; // 0x3d1151b3
    const pa6: f32 = -2.1663755178e-03; // 0xbb0df9c0
    const qa1: f32 = 1.0642088205e-01; // 0x3dd9f331
    const qa2: f32 = 5.4039794207e-01; // 0x3f0a5785
    const qa3: f32 = 7.1828655899e-02; // 0x3d931ae7
    const qa4: f32 = 1.2617121637e-01; // 0x3e013307
    const qa5: f32 = 1.3637083583e-02; // 0x3c5f6e13
    const qa6: f32 = 1.1984500103e-02; // 0x3c445aa3
    // Coefficients for approximation to erfc on [1.25, 1/0.35]
    const ra0: f32 = -9.8649440333e-03; // 0xbc21a093
    const ra1: f32 = -6.9385856390e-01; // 0xbf31a0b7
    const ra2: f32 = -1.0558626175e+01; // 0xc128f022
    const ra3: f32 = -6.2375331879e+01; // 0xc2798057
    const ra4: f32 = -1.6239666748e+02; // 0xc322658c
    const ra5: f32 = -1.8460508728e+02; // 0xc3389ae7
    const ra6: f32 = -8.1287437439e+01; // 0xc2a2932b
    const ra7: f32 = -9.8143291473e+00; // 0xc11d077e
    const sa1: f32 = 1.9651271820e+01; // 0x419d35ce
    const sa2: f32 = 1.3765776062e+02; // 0x4309a863
    const sa3: f32 = 4.3456588745e+02; // 0x43d9486f
    const sa4: f32 = 6.4538726807e+02; // 0x442158c9
    const sa5: f32 = 4.2900814819e+02; // 0x43d6810b
    const sa6: f32 = 1.0863500214e+02; // 0x42d9451f
    const sa7: f32 = 6.5702495575e+00; // 0x40d23f7c
    const sa8: f32 = -6.0424413532e-02; // 0xbd777f97
    // Coefficients for approximation to erfc on [1/0.35, 6]
    const rb0: f32 = -9.8649431020e-03; // 0xbc21a092
    const rb1: f32 = -7.9928326607e-01; // 0xbf4c9dd4
    const rb2: f32 = -1.7757955551e+01; // 0xc18e104b
    const rb3: f32 = -1.6063638306e+02; // 0xc320a2ea
    const rb4: f32 = -6.3756646729e+02; // 0xc41f6441
    const rb5: f32 = -1.0250950928e+03; // 0xc480230b
    const rb6: f32 = -4.8351919556e+02; // 0xc3f1c275
    const sb1: f32 = 3.0338060379e+01; // 0x41f2b459
    const sb2: f32 = 3.2579251099e+02; // 0x43a2e571
    const sb3: f32 = 1.5367296143e+03; // 0x44c01759
    const sb4: f32 = 3.1998581543e+03; // 0x4547fdbb
    const sb5: f32 = 2.5530502930e+03; // 0x451f90ce
    const sb6: f32 = 4.7452853394e+02; // 0x43ed43a7
    const sb7: f32 = -2.2440952301e+01; // 0xc1b38712
};

fn erfc1(x: f32) f32 {
    const d = erf_data;
    const s = @abs(x) - 1;
    const capital_p = d.pa0 + s * (d.pa1 + s * (d.pa2 + s * (d.pa3 + s * (d.pa4 + s * (d.pa5 + s * d.pa6)))));
    const capital_q = 1 + s * (d.qa1 + s * (d.qa2 + s * (d.qa3 + s * (d.qa4 + s * (d.qa5 + s * d.qa6)))));
    return 1 - d.erx - capital_p / capital_q;
}

fn erfc2(ix: u32, x_signed: f32) f32 {
    const d = erf_data;
    if (ix < 0x3fa00000) return erfc1(x_signed); // |x| < 1.25
    const x = @abs(x_signed);
    const s = 1 / (x * x);
    var capital_r: f32 = undefined;
    var capital_s: f32 = undefined;
    if (ix < 0x4036db6d) { // |x| < 1/0.35
        capital_r = d.ra0 + s * (d.ra1 + s * (d.ra2 + s * (d.ra3 + s * (d.ra4 + s * (d.ra5 + s * (d.ra6 + s * d.ra7))))));
        capital_s = 1.0 + s * (d.sa1 + s * (d.sa2 + s * (d.sa3 + s * (d.sa4 + s * (d.sa5 + s * (d.sa6 + s * (d.sa7 + s * d.sa8)))))));
    } else { // |x| >= 1/0.35
        capital_r = d.rb0 + s * (d.rb1 + s * (d.rb2 + s * (d.rb3 + s * (d.rb4 + s * (d.rb5 + s * d.rb6)))));
        capital_s = 1.0 + s * (d.sb1 + s * (d.sb2 + s * (d.sb3 + s * (d.sb4 + s * (d.sb5 + s * (d.sb6 + s * d.sb7))))));
    }
    // SET_FLOAT_WORD(z, ix & 0xffffe000): truncate the mantissa low bits.
    const z: f32 = @bitCast(@as(u32, @bitCast(x)) & 0xffffe000);
    return @exp(-z * z - 0.5625) * @exp((z - x) * (z + x) + capital_r / capital_s) / x;
}

/// Single-precision error function (musl `erff`, translated faithfully).
pub fn erff(x: f32) f32 {
    const d = erf_data;
    var ix: u32 = @bitCast(x);
    const sign = (ix >> 31) != 0;
    ix &= 0x7fffffff;
    if (ix >= 0x7f800000) {
        // erf(nan) = nan, erf(±inf) = ±1
        const signed_one: f32 = if (sign) -1 else 1;
        return signed_one + 1 / x;
    }
    if (ix < 0x3f580000) { // |x| < 0.84375
        if (ix < 0x31800000) { // |x| < 2^-28: avoid underflow
            return 0.125 * (8 * x + d.efx8 * x);
        }
        const z = x * x;
        const r = d.pp0 + z * (d.pp1 + z * (d.pp2 + z * (d.pp3 + z * d.pp4)));
        const s = 1 + z * (d.qq1 + z * (d.qq2 + z * (d.qq3 + z * (d.qq4 + z * d.qq5))));
        const y = r / s;
        return x + x * y;
    }
    const y: f32 = if (ix < 0x40c00000) 1 - erfc2(ix, x) else 1 - 0x1p-120;
    return if (sign) -y else y;
}
