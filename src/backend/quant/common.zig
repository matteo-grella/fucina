//! Shared low-level primitives for the quantized matmul kernels.
//!
//! Leaf module: the generic SIMD vector type aliases, the f16 bit conversions,
//! the AArch64 sdot/smmla helpers, the rounding/quantize primitives, the nibble
//! helpers, the shared row/column blocking consts, and the i8mm feature flag.
//! Imported by quant.zig and the per-type kernel files (q8_0/q4_k/q5_k/q6_k/cold);
//! it imports none of them. `types.zig` supplies the block layouts the shared
//! Q8_Kx4 lane dots read.
const std = @import("std");
const builtin = @import("builtin");
const types = @import("types.zig");

// SMMLA is a separate AArch64 feature from SDOT; Apple M1-class CPUs have
// FEAT_DotProd but not FEAT_I8MM, so the MMLA path must stay gated.
pub const has_aarch64_i8mm = builtin.cpu.arch == .aarch64 and std.Target.aarch64.featureSetHas(builtin.cpu.features, .i8mm);

// x86 int8-GEMM feature gates (mirror has_aarch64_i8mm). AVX-512-VNNI provides the
// vpdpbusd byte dot-product (the analog of NEON sdot); AVX2 falls back to the
// vpmaddwd i16 path.
pub const has_x86_avx2 = builtin.cpu.arch == .x86_64 and std.Target.x86.featureSetHas(builtin.cpu.features, .avx2);
pub const has_x86_avx512vnni = builtin.cpu.arch == .x86_64 and std.Target.x86.featureSetHas(builtin.cpu.features, .avx512vnni);

// AVX-VNNI: the VEX-encoded vpdpbusd (Alder Lake+, Zen4+; CPUID AVX_VNNI) — a
// separate feature bit from AVX512VNNI. No emulation substrate executes
// AVX-VNNI, so it cannot run on a non-x86 dev machine — but it is
// HARDWARE-EXECUTION-VALIDATED 2026-07-03 on an
// i9-13950HX (Raptor Lake, Linux): zig build test + zig build x86dot-check
// pass natively with checksums bit-equal to the M1/Rosetta portable runs
// (coverage table in src/x86dot_check.zig).
// Semantic gap to keep in mind: the AVX2 construction saturates at i16
// (vpmaddubsw) while vpdpbusd accumulates in i32 without saturation, so
// AVX2-validated tests do NOT prove VNNI numerics.
pub const has_x86_avxvnni = builtin.cpu.arch == .x86_64 and std.Target.x86.featureSetHas(builtin.cpu.features, .avxvnni);

pub fn quantizeToI8(x: f32) i8 {
    const clamped = @max(-127.0, @min(127.0, roundHalfAwayFromZero(x)));
    return @intFromFloat(clamped);
}

pub fn roundHalfAwayFromZero(x: f32) f32 {
    return @round(x);
}

pub fn roundNearestEven(x: f32) f32 {
    var rounded = @round(x);
    if (@abs(rounded - x) == 0.5) {
        const rounded_int: i32 = @intFromFloat(rounded);
        if (@mod(rounded_int, 2) != 0) rounded -= if (x < 0) -1.0 else 1.0;
    }
    return rounded;
}

fn roundNearestEvenVec4(x: QKV4f32) QKV4f32 {
    const zero: QKV4f32 = @splat(0);
    var rounded = @round(x);
    const rounded_int: QKV4i32 = @intFromFloat(rounded);
    const odd = (rounded_int & @as(QKV4i32, @splat(1))) != @as(QKV4i32, @splat(0));
    const half = @abs(rounded - x) == @as(QKV4f32, @splat(0.5));
    const adjust = half & odd;
    const direction = @select(f32, x < zero, @as(QKV4f32, @splat(-1.0)), @as(QKV4f32, @splat(1.0)));
    rounded -= @select(f32, adjust, direction, zero);
    return rounded;
}

pub fn roundNearestEvenVec4ToI32(x: QKV4f32) QKV4i32 {
    if (comptime builtin.cpu.arch == .aarch64) {
        var out: QKV4i32 = undefined;
        asm ("fcvtns %[out].4s, %[x].4s"
            : [out] "=w" (out),
            : [x] "w" (x),
        );
        return out;
    }
    return @intFromFloat(roundNearestEvenVec4(x));
}

pub fn roundHalfAwayFromZeroVec4ToI32(x: QKV4f32) QKV4i32 {
    if (comptime builtin.cpu.arch == .aarch64) {
        var out: QKV4i32 = undefined;
        asm ("fcvtas %[out].4s, %[x].4s"
            : [out] "=w" (out),
            : [x] "w" (x),
        );
        return out;
    }
    return @intFromFloat(@round(x));
}

pub fn f32ToF16Bits(x: f32) u16 {
    const h: f16 = @floatCast(x);
    return @bitCast(h);
}

pub fn f16BitsToF32(bits: u16) f32 {
    const h: f16 = @bitCast(bits);
    return @floatCast(h);
}

pub inline fn f16x4BitsToF32(bits: [4]u16) QKV4f32 {
    const h: @Vector(4, f16) = @bitCast(bits);
    return @floatCast(h);
}

// Sign-extending int8 dot product accumulated in i32, for tail columns. Inputs
// are one group, so the running sum cannot overflow i32 (<= group_size*127*127).
// Plain scalar reduction so LLVM can lower it to the target int8 dot instruction.
pub fn i8DotI32(a: []const i8, b: []const i8) i32 {
    var s: i32 = 0;
    for (a, b) |x, y| s += @as(i32, x) * @as(i32, y);
    return s;
}

// Full 16-wide signed int8 dot accumulated in i32 — the building block for the
// non-aarch64 fallback of the K-quant sub-block dots. Two numerically identical
// forms; `dotI8x16Portable` picks one at comptime by target feature:
//   plain : sext(i8)·sext(i8) reduce -> AVX2 lowers to vpmaddwd.
//   bias  : Σa·b = Σ(a+128)·b − 128·Σb, with (a+128) unsigned, so it is a u8·i8
//           dot -> AVX-512-VNNI lowers the Σ(a+128)·b to **vpdpbusd** (the byte
//           dot, ~the throughput of NEON sdot). Exact i32 (no i16 saturation).
// The two forms are asserted bit-equal by the test below (runs on any host), so
// the bias algebra is execution-validated even where vpdpbusd can't be run.
fn dotI8x16PlainForm(a: QKV16i8, b: QKV16i8) i32 {
    const ai: QKV16i32 = @intCast(a);
    const bi: QKV16i32 = @intCast(b);
    return @reduce(.Add, ai * bi);
}

fn dotI8x16BiasForm(a: QKV16i8, b: QKV16i8) i32 {
    const ua: QKV16u8 = @bitCast(a ^ @as(QKV16i8, @splat(-128))); // a + 128 (unsigned)
    const ui: QKV16i32 = @intCast(ua);
    const bi: QKV16i32 = @intCast(b);
    return @reduce(.Add, ui * bi) - 128 * @reduce(.Add, bi);
}

pub fn dotI8x16Portable(a: QKV16i8, b: QKV16i8) i32 {
    // Either VNNI flavor: the bias form lowers to vpdpbusd (EVEX under
    // avx512vnni — execution-unverifiable here but objdump-proven; VEX under
    // avxvnni — COMPILE-ONLY on this dev machine, see has_x86_avxvnni).
    if (comptime has_x86_avx512vnni or has_x86_avxvnni) return dotI8x16BiasForm(a, b);
    // AVX2 without VNNI: ggml-style sign-transfer dot (vpsignb + vpmaddubsw +
    // vpmaddwd). EXACTNESS DOMAIN: requires b[i] != -128 in lanes where
    // a[i] < 0 (see dotI8x16SignTrickForm below). Every call site passes
    // activations ∈ [-127,127] as b, weights as a: Q8_0 activations via
    // quantizeToI8's explicit clamp, Q5_K/Q6_K activations via
    // quantizeRowQ8_KInto's -127/max scale construction (BlockQ8_K qs is
    // never -128).
    if (comptime has_x86_avx2) return dotI8x16SignTrickForm(a, b);
    return dotI8x16PlainForm(a, b);
}

test "dotI8x16 +128-bias form equals the plain signed dot" {
    const patterns = [_][3]i32{ .{ 31, 17, 5 }, .{ 13, 7, 3 }, .{ 101, 59, 1 }, .{ 211, 97, 2 } };
    inline for (patterns) |p| {
        var aa: [16]i8 = undefined;
        var bb: [16]i8 = undefined;
        inline for (0..16) |i| {
            aa[i] = @intCast(@as(i32, @intCast((i * p[0] + p[2]) % 251)) - 125);
            bb[i] = @intCast(@as(i32, @intCast((i * p[1] + p[2]) % 251)) - 125);
        }
        const a: QKV16i8 = aa;
        const b: QKV16i8 = bb;
        try std.testing.expectEqual(dotI8x16PlainForm(a, b), dotI8x16BiasForm(a, b));
        // also matches the naive scalar reference
        var ref: i32 = 0;
        inline for (0..16) |i| ref += @as(i32, aa[i]) * @as(i32, bb[i]);
        try std.testing.expectEqual(ref, dotI8x16BiasForm(a, b));
    }
}

// ---------------------------------------------------------------------------
// x86 AVX2 int8 dot construction (the ggml-style vpmaddubsw + vpmaddwd ladder).
//
// vpmaddubsw multiplies UNSIGNED bytes of the first operand with SIGNED bytes
// of the second and adds adjacent pairs with SIGNED i16 SATURATION:
//     out[i] = sat_i16(a[2i]*b[2i] + a[2i+1]*b[2i+1]),  a: u8, b: i8.
// The saturation is the trap — every caller must prove its pair sums fit in
// i16 (proofs at each form below). vpmaddwd then folds i16 pairs into exact
// i32 (multiply by ones + horizontal pair add; no saturation possible since
// |2 * 32767 * 1| < 2^31). Inline asm is used because LLVM only pattern-matches
// vpmaddubsw from explicitly saturating IR (verified: a clamp-pattern @Vector
// formulation compiles to vpmaddwd, not vpmaddubsw, at -mcpu=x86_64_v3); this
// mirrors how the aarch64 sdot/smmla primitives above are hand-rolled asm.
// All three helpers are only ever called under `comptime has_x86_avx2` (their
// VEX.128 encodings need AVX/AVX2), so they never reach non-x86 codegen.
//
// Execution-validated on x86_64_v3 (hardware and a validated emulator)
// against the scalar reference — see src/x86dot_check.zig and the tests
// below. NOTE: some emulation substrates execute AVX2 SILENTLY WRONG (no
// SIGILL, corrupt lanes) — before trusting any emulator, reproduce the
// recorded checksums in src/x86dot_check.zig's attestation table.

inline fn maddubsI16x8(a: QKV16u8, b: QKV16i8) QKV8i16 {
    // AT&T operand order: vpmaddubsw src2(signed), src1(unsigned), dst.
    return asm ("vpmaddubsw %[b], %[a], %[out]"
        : [out] "=x" (-> QKV8i16),
        : [a] "x" (a),
          [b] "x" (b),
    );
}

inline fn maddwdSumPairsI32x4(v: QKV8i16) QKV4i32 {
    const ones: QKV8i16 = @splat(1);
    return asm ("vpmaddwd %[ones], %[v], %[out]"
        : [out] "=x" (-> QKV4i32),
        : [v] "x" (v),
          [ones] "x" (ones),
    );
}

/// vpsignb: out[i] = y[i] * sign(x[i]) — y negated where x < 0, zeroed where
/// x == 0, passed through where x > 0. Negation WRAPS: -(-128) stays -128.
inline fn psignI8x16(y: QKV16i8, x: QKV16i8) QKV16i8 {
    return asm ("vpsignb %[x], %[y], %[out]"
        : [out] "=x" (-> QKV16i8),
        : [y] "x" (y),
          [x] "x" (x),
    );
}

/// i8 x i8 dot via the ggml `mul_sum_i8_pairs` sign-transfer trick:
///     a·b == |a| · (sign(a)·b),  with |a| on the unsigned side of vpmaddubsw.
///
/// EXACTNESS DOMAIN — exact for ALL a in [-128,127] (|−128| = 128 is a valid
/// u8) provided b[i] != -128 in every lane where a[i] < 0; in that lane psignb
/// wraps −(−128) to −128 and the product is off by 256·|a[i]|. All call sites
/// pass activations ∈ [-127,127] as b — Q8_0 via quantizeToI8's clamp,
/// Q5_K/Q6_K via quantizeRowQ8_KInto's -127/max scale construction.
///
/// SATURATION PROOF (none possible in-domain): per lane, if a[i] >= 0 then
/// |a[i]| <= 127 and |sign·b| <= 128, product magnitude <= 127*128 = 16256;
/// if a[i] < 0 then |a[i]| <= 128 and |sign·b| <= 127 (b != -128 in-domain),
/// product magnitude <= 128*127 = 16256. Pair sums are therefore within
/// ±32512 < 32767 — vpmaddubsw never saturates, the result is exact i32.
fn dotI8x16SignTrickForm(a: QKV16i8, b: QKV16i8) i32 {
    const abs_a: QKV16u8 = @bitCast(psignI8x16(a, a)); // |a| as u8 (128 ok)
    const sb = psignI8x16(b, a); // sign(a) * b
    return @reduce(.Add, maddwdSumPairsI32x4(maddubsI16x8(abs_a, sb)));
}

// u8 x i8 dot, two numerically identical forms picked at comptime:
//   widen  : zext(u8)·sext(i8) reduce — exact for all inputs on every target.
//            LLVM lowers it to vpdpbusd on VNNI targets (objdump-verified on
//            both -mcpu=alderlake VEX and -mcpu=znver4 EVEX) and to a
//            vpmaddwd ladder on plain AVX2.
//   maddubs: vpmaddubsw + vpmaddwd — byte-granularity (2x denser than the
//            widened i16 path), but SATURATING: exact iff every adjacent pair
//            sum a[2i]*b[2i] + a[2i+1]*b[2i+1] fits in i16. The Q4_K call
//            site has a ∈ [0,15] (nibbles), so pair sums are bounded by
//            2*15*128 = 3840 << 32767 — saturation-free. Callers with wider
//            unsigned inputs must prove their own bound (e.g. a <= 127 keeps
//            pair sums within 2*127*128 = 32512).
fn dotU8I8x16WidenForm(a: QKV16u8, b: QKV16i8) i32 {
    const ai: QKV16i32 = @intCast(a); // zext
    const bi: QKV16i32 = @intCast(b); // sext
    return @reduce(.Add, ai * bi);
}

fn dotU8I8x16MaddubsForm(a: QKV16u8, b: QKV16i8) i32 {
    return @reduce(.Add, maddwdSumPairsI32x4(maddubsI16x8(a, b)));
}

pub fn dotU8I8x16Portable(a: QKV16u8, b: QKV16i8) i32 {
    // VNNI (either flavor): the widen form IS the vpdpbusd pattern. The
    // avxvnni arm is COMPILE-ONLY on this dev machine (see has_x86_avxvnni).
    if (comptime has_x86_avx512vnni or has_x86_avxvnni) return dotU8I8x16WidenForm(a, b);
    if (comptime has_x86_avx2) return dotU8I8x16MaddubsForm(a, b);
    return dotU8I8x16WidenForm(a, b);
}

// ---------------------------------------------------------------------------
// 256-bit (ymm) GROUPED int8 dot primitives — the x86 arms of the Q8_0x4
// packed accumulates (and the pattern for the K-quant x4/x8 packs). All of
// them compute the sdot group shape widened to eight groups per 32-byte
// vector: out[g] = acc[g] + Σ_{k<4} a[4g+k]·b[4g+k], exact i32 accumulate.
// Each has a portable @Vector twin so every caller compiles and is testable
// on any target (the M1 dev machine runs the twins; the asm bodies are only
// reachable under their comptime feature gates and are hardware-executed on
// the x86 box — see the coverage table in src/x86dot_check.zig).

pub const has_x86_avx512vl = builtin.cpu.arch == .x86_64 and std.Target.x86.featureSetHas(builtin.cpu.features, .avx512vl);

// The self-hosted x86_64 backend (the Debug-mode default on x86_64-linux) has
// its own assembler that lacks the newer VEX mnemonics (vpdpbusd rejects with
// "invalid mnemonic"); the ymm asm arms below are therefore additionally gated
// on the LLVM backend — Debug builds execute the exact portable twins instead,
// ReleaseSafe/ReleaseFast (LLVM) execute the real instructions.
const has_llvm_asm = builtin.zig_backend == .stage2_llvm;

// vpdpbusd-on-ymm gate: the VEX encoding needs AVX-VNNI; the EVEX.256
// encoding needs AVX512-VNNI *and* AVX512-VL (every shipped VNNI core —
// Ice Lake+, Zen4+ — has VL, but the assembler needs the gate to legalize
// an encoding, so keep it explicit rather than implied).
pub const has_x86_vnni_ymm = has_x86_avxvnni or (has_x86_avx512vnni and has_x86_avx512vl);

fn strideMaskI32(comptime n: comptime_int, comptime start: comptime_int) @Vector(n, i32) {
    var m: [n]i32 = undefined;
    for (&m, 0..) |*x, i| x.* = @intCast(i * 2 + start);
    return m;
}

// Exact grouped reduce of 32 i16 products into 8 i32 group sums (two
// deinterleave/widen/add stages). The i16 inputs are PRODUCTS that each fit
// i16 (callers prove |a·b| ≤ 2^15; both i8·i8 = ±16384 and u8·i8 = ±32640
// do) — the pair sums are widened to i32 BEFORE adding, so no saturation
// anywhere in this helper.
inline fn sumGroupsI16x32(prod: QKV32i16) QKV8i32 {
    const even16 = comptime strideMaskI32(16, 0);
    const odd16 = comptime strideMaskI32(16, 1);
    const pe: QKV16i32 = @intCast(@shuffle(i16, prod, undefined, even16));
    const po: QKV16i32 = @intCast(@shuffle(i16, prod, undefined, odd16));
    const pairs = pe + po;
    const even8 = comptime strideMaskI32(8, 0);
    const odd8 = comptime strideMaskI32(8, 1);
    const ge: QKV8i32 = @shuffle(i32, pairs, undefined, even8);
    const go: QKV8i32 = @shuffle(i32, pairs, undefined, odd8);
    return ge + go;
}

// Universal widening tier: signed grouped dot with NO arch gate and NO input
// domain restriction — i8·i8 products are exact in i16 (|p| ≤ 16384 < 2^15),
// group sums widen to i32. This is the production floor for every ISA that
// is neither aarch64 (sdot) nor gated x86 (vpdpbusd / maddubs); the scalar
// kernels survive only as the bit-exactness reference in tests.
pub fn dotI8GroupsWidenI32x8(acc: QKV8i32, a: QKV32i8, b: QKV32i8) QKV8i32 {
    const ai: QKV32i16 = @intCast(a);
    const bi: QKV32i16 = @intCast(b);
    return acc + sumGroupsI16x32(ai * bi);
}

// Portable twin of dpbusdI32x8 (vpdpbusd semantics): u8·i8 products are exact
// in i16 (|p| ≤ 255·128 = 32640 < 2^15), group sums widen to i32. Exact for
// ALL inputs — vpdpbusd accumulates in i32 without saturation and so does
// this form.
fn dpbusdI32x8Portable(acc: QKV8i32, a: QKV32u8, b: QKV32i8) QKV8i32 {
    const ai: QKV32i16 = @intCast(a);
    const bi: QKV32i16 = @intCast(b);
    return acc + sumGroupsI16x32(ai * bi);
}

/// Grouped u8·i8 dot-accumulate: out[g] = acc[g] + Σ_{k<4} zext(a[4g+k])·sext(b[4g+k]).
/// On VNNI targets this is a single `vpdpbusd` (inline asm — LLVM only
/// pattern-matches the full-reduce shape, not this partial reduction; mirrors
/// how the aarch64 sdot primitives above are hand-rolled). Exact i32 for all
/// inputs, no saturation. AT&T operand order: src2(signed), src1(unsigned), dst.
pub fn dpbusdI32x8(acc: QKV8i32, a: QKV32u8, b: QKV32i8) QKV8i32 {
    if (comptime has_x86_vnni_ymm and has_llvm_asm) {
        // ENCODING IS LOAD-BEARING: LLVM's asm parser does not feature-check
        // inline asm and resolves the bare mnemonic to the EVEX (AVX512-VNNI)
        // form, which SIGILLs on AVX-VNNI-only cores (Alder/Raptor Lake) —
        // their VEX form must be selected with the explicit {vex} prefix
        // (LLVM defines the AVX-VNNI aliases as ExplicitVEXPrefix). Cores
        // gated in via AVX512-VNNI+VL (Ice Lake, no AVX-VNNI) take EVEX.
        const mnemonic = comptime if (has_x86_avxvnni) "{vex} vpdpbusd" else "vpdpbusd";
        var out = acc;
        asm (mnemonic ++ " %[b], %[a], %[out]"
            : [out] "+x" (out),
            : [a] "x" (a),
              [b] "x" (b),
        );
        return out;
    }
    return dpbusdI32x8Portable(acc, a, b);
}

inline fn maddubsI16x16(a: QKV32u8, b: QKV32i8) QKV16i16 {
    // AT&T operand order: vpmaddubsw src2(signed), src1(unsigned), dst.
    return asm ("vpmaddubsw %[b], %[a], %[out]"
        : [out] "=x" (-> QKV16i16),
        : [a] "x" (a),
          [b] "x" (b),
    );
}

inline fn maddwdSumPairsI32x8(v: QKV16i16) QKV8i32 {
    const ones: QKV16i16 = @splat(1);
    return asm ("vpmaddwd %[ones], %[v], %[out]"
        : [out] "=x" (-> QKV8i32),
        : [v] "x" (v),
          [ones] "x" (ones),
    );
}

/// Grouped u8·i8 dot-accumulate via vpmaddubsw + vpmaddwd — the AVX2-only
/// (no-VNNI) arm. SATURATING at i16 in the maddubs pair sums: exact iff every
/// adjacent pair sum a[2i]·b[2i] + a[2i+1]·b[2i+1] fits in i16 — callers must
/// prove their bound (the Q8_0 sign-trick arrangement keeps pairs within
/// ±32512, see accumulateQ8_0x4PackedAvx2). The portable fallback is the
/// exact widening form, identical to the asm INSIDE that proven domain.
pub fn maddubsDotGroupsI32x8(acc: QKV8i32, a: QKV32u8, b: QKV32i8) QKV8i32 {
    if (comptime has_x86_avx2 and has_llvm_asm) {
        return acc + maddwdSumPairsI32x8(maddubsI16x16(a, b));
    }
    return dpbusdI32x8Portable(acc, a, b);
}

/// vpshufb on ymm: per-128-bit-lane 16-entry byte table lookup —
/// out[i] = 0 when idx[i] bit 7 is set, else table[lane_base + (idx[i] & 0x0F)]
/// with lane_base 0 for the low half and 16 for the high half. Callers
/// duplicate a 16-entry table across both halves for a uniform lookup.
pub fn pshufbI8x32(table: QKV32i8, idx: QKV32u8) QKV32i8 {
    if (comptime has_x86_avx2 and has_llvm_asm) {
        return asm ("vpshufb %[idx], %[table], %[out]"
            : [out] "=x" (-> QKV32i8),
            : [table] "x" (table),
              [idx] "x" (idx),
        );
    }
    const t: [32]i8 = table;
    const ix: [32]u8 = idx;
    var out: [32]i8 = undefined;
    inline for (0..32) |i| {
        const lane_base: usize = if (i < 16) 0 else 16;
        out[i] = if (ix[i] & 0x80 != 0) 0 else t[lane_base + (ix[i] & 0x0f)];
    }
    return out;
}

/// vpsignb on ymm: out[i] = y[i]·sign(x[i]) — y negated where x < 0, zeroed
/// where x == 0, passed through where x > 0. Negation WRAPS: -(-128) stays
/// -128 (the portable twin reproduces the wrap via -% exactly).
pub fn psignI8x32(y: QKV32i8, x: QKV32i8) QKV32i8 {
    if (comptime has_x86_avx2 and has_llvm_asm) {
        return asm ("vpsignb %[x], %[y], %[out]"
            : [out] "=x" (-> QKV32i8),
            : [y] "x" (y),
              [x] "x" (x),
        );
    }
    const zero: QKV32i8 = @splat(0);
    const neg = zero -% y;
    return @select(i8, x > zero, y, @select(i8, x < zero, neg, zero));
}

/// Fold the two 128-bit halves of an 8-lane i32 accumulator into 4 lanes
/// (the per-column group totals of the ymm grouped-dot kernels).
pub fn addHalvesI32x8(v: QKV8i32) QKV4i32 {
    const lo: QKV4i32 = @shuffle(i32, v, undefined, [4]i32{ 0, 1, 2, 3 });
    const hi: QKV4i32 = @shuffle(i32, v, undefined, [4]i32{ 4, 5, 6, 7 });
    return lo + hi;
}

pub const q8_0_row_block: usize = 4;
pub const q4_kx8_row_block: usize = 16;
pub const qk_col_block: usize = 2;
pub const Q4V16u8 = @Vector(16, u8);
pub const Q4V16i8 = @Vector(16, i8);
pub const Q4V16i16 = @Vector(16, i16);
pub const QKV8u8 = @Vector(8, u8);
pub const QKV8i8 = @Vector(8, i8);
pub const QKV8i16 = @Vector(8, i16);
pub const QKV8i32 = @Vector(8, i32);
pub const QKV4i32 = @Vector(4, i32);
pub const QKV4f32 = @Vector(4, f32);
pub const QKV16u8 = @Vector(16, u8);
pub const QKV16u16 = @Vector(16, u16);
pub const QKV16i8 = @Vector(16, i8);
pub const QKV16i16 = @Vector(16, i16);
pub const QKV16i32 = @Vector(16, i32);
pub const QKV32u8 = @Vector(32, u8);
pub const QKV32i8 = @Vector(32, i8);
pub const QKV32i16 = @Vector(32, i16);
// Portable scalar/@Vector reference implementations of the AArch64 int8 dot /
// matrix-multiply primitives below. The hardware path lowers each to a single
// sdot/smmla; these reproduce the exact lane semantics for non-aarch64 targets
// (so the whole quant backend compiles and runs correctly on e.g. x86), and are
// the oracle the on-device test checks the asm output against. NOTE: these are
// correctness-first fallbacks, not optimized — the optimized x86 hot paths are
// the VNNI/AVX2 arms in the per-type kernels (q8_0/q4_k/q5_k/q6_k).
fn sdotI8x16Portable(acc: QKV4i32, a: QKV16i8, b: QKV16i8) QKV4i32 {
    const ai: QKV16i32 = @intCast(a);
    const bi: QKV16i32 = @intCast(b);
    const prod = ai * bi;
    var lanes: [4]i32 = undefined;
    inline for (0..4) |l| {
        lanes[l] = prod[l * 4] + prod[l * 4 + 1] + prod[l * 4 + 2] + prod[l * 4 + 3];
    }
    return acc + @as(QKV4i32, lanes);
}

fn sdotI8x16LanePortable(comptime lane: comptime_int, acc: QKV4i32, a: QKV16i8, b: QKV16i8) QKV4i32 {
    const ai: QKV16i32 = @intCast(a);
    const bi: QKV16i32 = @intCast(b);
    var lanes: [4]i32 = undefined;
    inline for (0..4) |l| {
        lanes[l] = ai[l * 4] * bi[lane * 4] +
            ai[l * 4 + 1] * bi[lane * 4 + 1] +
            ai[l * 4 + 2] * bi[lane * 4 + 2] +
            ai[l * 4 + 3] * bi[lane * 4 + 3];
    }
    return acc + @as(QKV4i32, lanes);
}

fn smmlaI8x16Portable(acc: QKV4i32, a: QKV16i8, b: QKV16i8) QKV4i32 {
    // a and b are 2x8 int8 matrices (rows [0..8) and [8..16)); the result is the
    // 2x2 product a · bᵀ accumulated as lanes {a0·b0, a0·b1, a1·b0, a1·b1}.
    const ai: QKV16i32 = @intCast(a);
    const bi: QKV16i32 = @intCast(b);
    var d = [4]i32{ 0, 0, 0, 0 };
    inline for (0..8) |k| {
        d[0] += ai[k] * bi[k];
        d[1] += ai[k] * bi[8 + k];
        d[2] += ai[8 + k] * bi[k];
        d[3] += ai[8 + k] * bi[8 + k];
    }
    return acc + @as(QKV4i32, d);
}

/// 16-entry signed byte table lookup: out[i] = table[idx[i]]. Callers must
/// mask idx to 0..15; lanes outside that range are unspecified on the
/// portable path (NEON `tbl` would zero them, but no caller relies on it).
pub fn tblI8x16(table: QKV16i8, idx: QKV16u8) QKV16i8 {
    if (comptime builtin.cpu.arch == .aarch64) {
        var out: QKV16i8 = undefined;
        asm ("tbl %[out].16b, {%[t].16b}, %[i].16b"
            : [out] "=w" (out),
            : [t] "w" (table),
              [i] "w" (idx),
        );
        return out;
    }
    // Materialize both vectors: a runtime lane index into a @Vector is a
    // compile error, arrays index freely.
    const t: [16]i8 = table;
    const ix: [16]u8 = idx;
    var out: [16]i8 = undefined;
    inline for (0..16) |i| out[i] = t[ix[i] & 0x0f];
    return out;
}

/// e8m0 exponent byte → 2^(e-127) / 2, exact for every non-NaN byte
/// (e < 2 lands in the f32 subnormal range). The MXFP4 nibble table stores
/// doubled e2m1 values so integer dots stay exact; this halved scale folds
/// the /2 back without a rounding step.
pub fn e8m0ToF32Half(x: u8) f32 {
    const bits: u32 = if (x < 2)
        @as(u32, 0x00200000) << @intCast(x)
    else
        @as(u32, x - 1) << 23;
    return @bitCast(bits);
}

pub fn sdotI8x16(acc: QKV4i32, a: QKV16i8, b: QKV16i8) QKV4i32 {
    if (comptime builtin.cpu.arch == .aarch64) {
        var out = acc;
        asm ("sdot %[out].4s, %[a].16b, %[b].16b"
            : [out] "+w" (out),
            : [a] "w" (a),
              [b] "w" (b),
        );
        return out;
    }
    return sdotI8x16Portable(acc, a, b);
}

pub fn sdotI8x16Lane(comptime lane: comptime_int, acc: QKV4i32, a: QKV16i8, b: QKV16i8) QKV4i32 {
    if (comptime builtin.cpu.arch == .aarch64) {
        var out = acc;
        asm ("sdot %[out].4s, %[a].16b, %[b].4b[" ++ std.fmt.comptimePrint("{d}", .{lane}) ++ "]"
            : [out] "+w" (out),
            : [a] "w" (a),
              [b] "w" (b),
        );
        return out;
    }
    return sdotI8x16LanePortable(lane, acc, a, b);
}

pub fn smmlaI8x16(acc: QKV4i32, a: QKV16i8, b: QKV16i8) QKV4i32 {
    // Guarded on i8mm (not just aarch64): Apple M1-class cores are aarch64 but
    // lack FEAT_I8MM, so the smmla instruction would trap there — they take the
    // portable path. Real i8mm hardware and the asm path are exercised by the test.
    if (comptime has_aarch64_i8mm) {
        var out = acc;
        asm ("smmla %[out].4s, %[a].16b, %[b].16b"
            : [out] "+w" (out),
            : [a] "w" (a),
              [b] "w" (b),
        );
        return out;
    }
    return smmlaI8x16Portable(acc, a, b);
}

test "portable int8 dot primitives match the hardware sdot/smmla path" {
    // On aarch64 with FEAT_DotProd this compares the real sdot/smmla output against
    // the scalar reference (so the x86 fallback is validated on-device); elsewhere
    // both sides take the portable path. smmlaI8x16 is portable on non-i8mm cores
    // (e.g. M1), so the comparison never emits an unsupported instruction.
    const patterns = [_][3]i32{ .{ 31, 17, 5 }, .{ 13, 7, 3 }, .{ 101, 59, 1 }, .{ 211, 97, 2 } };
    inline for (patterns) |p| {
        var aa: [16]i8 = undefined;
        var bb: [16]i8 = undefined;
        inline for (0..16) |i| {
            aa[i] = @intCast(@as(i32, @intCast((i * p[0] + p[2]) % 251)) - 125);
            bb[i] = @intCast(@as(i32, @intCast((i * p[1] + p[2]) % 251)) - 125);
        }
        const a: QKV16i8 = aa;
        const b: QKV16i8 = bb;
        const acc = QKV4i32{ 1000, -2000, 3, -4 };
        try std.testing.expectEqual(sdotI8x16Portable(acc, a, b), sdotI8x16(acc, a, b));
        inline for (0..4) |lane| {
            try std.testing.expectEqual(sdotI8x16LanePortable(lane, acc, a, b), sdotI8x16Lane(lane, acc, a, b));
        }
        try std.testing.expectEqual(smmlaI8x16Portable(acc, a, b), smmlaI8x16(acc, a, b));
    }
}

pub fn q4LowNibbleI8(v: QKV16u8) QKV16i8 {
    return @bitCast(v & @as(QKV16u8, @splat(0x0f)));
}

pub fn q4HighNibbleI8(v: QKV16u8) QKV16i8 {
    return @bitCast(v >> @as(QKV16u8, @splat(4)));
}

pub fn dotDense(a: []const f32, b: []const f32) f32 {
    var acc: f32 = 0;
    for (a, b) |x, y| acc += x * y;
    return acc;
}

/// The `matmul<Fmt>RhsRange` trampoline every quant format exposes: the
/// parallel dispatch splits ROWS only, so a range call is the format's Tile
/// kernel over the full column span (`m` is unused by construction — the
/// tile covers rows [row_start, row_end)). Instantiated per format from its
/// Tile kernel; formats whose range semantics differ keep a hand-written
/// body instead.
pub fn RangeFromTile(comptime tile: anytype) blk: {
    const params = @typeInfo(@TypeOf(tile)).@"fn".params;
    break :blk fn (params[0].type.?, params[1].type.?, params[2].type.?, usize, usize, usize, usize) void;
} {
    const params = @typeInfo(@TypeOf(tile)).@"fn".params;
    return struct {
        fn range(out: params[0].type.?, lhs: params[1].type.?, rhs: params[2].type.?, m: usize, n: usize, row_start: usize, row_end: usize) void {
            _ = m;
            tile(out, lhs, rhs, n, row_start, row_end, 0, n);
        }
    }.range;
}

// ---------------------------------------------------------------------------
// Q8_Kx4 lane dots shared by the K-quant column-outer kernels (q4_k, q5_k):
// the per-ISA tier selector, the grouped u8·i8 dot-accumulate step, and the
// four-row sub-block dot over pre-unpacked weights.
// ---------------------------------------------------------------------------

/// x86 and portable arms of the grouped u8·i8 dots: `vnni` = vpdpbusd,
/// `avx2` = vpmaddubsw+vpmaddwd, `widen` = the universal @Vector form.
pub const X86DotTier = enum { vnni, avx2, widen };

/// One grouped weight·activation dot-accumulate step; `w` holds unsigned
/// weight codes, `a` is unrestricted i8. Exactness per tier:
///   vnni : vpdpbusd, exact i32 for all u8·i8 inputs, no saturation.
///   avx2 : vpmaddubsw+vpmaddwd; saturation-free while pair sums stay below
///          2^15 (w <= 15 gives 3840, w <= 31 gives 7936).
///   widen: weight codes below 128 fit i8 (@bitCast is value-preserving) and
///          i8·i8 products are exact in the widening dot on every target.
pub inline fn dotGroupsI32x8(comptime tier: X86DotTier, acc: QKV8i32, w: QKV32u8, a: QKV32i8) QKV8i32 {
    return switch (tier) {
        .vnni => dpbusdI32x8(acc, w, a),
        .avx2 => maddubsDotGroupsI32x8(acc, w, a),
        .widen => dotI8GroupsWidenI32x8(acc, @bitCast(w), a),
    };
}

/// i8 dot of one 32-wide sub-block: two 16-lane halves of unpacked weight
/// codes against two halves of Q8_K activations.
pub fn dotUnpackedI8x32(w0: QKV16i8, w1: QKV16i8, a0: QKV16i8, a1: QKV16i8) i32 {
    if (comptime builtin.cpu.arch == .aarch64) {
        var acc: QKV4i32 = @splat(0);
        acc = sdotI8x16(acc, w0, a0);
        acc = sdotI8x16(acc, w1, a1);
        return @reduce(.Add, acc);
    }
    // non-aarch64: VNNI lowers each to vpdpbusd (via the +128 bias), AVX2 takes
    // the sign-trick vpmaddubsw path (w is a Q4_K or Q5_K code, a comes from
    // BlockQ8_K, i.e. quantizeRowQ8_KInto's -127/max scale construction, so
    // a is in [-127,127]: inside the sign-trick exactness domain; see
    // dotI8x16Portable).
    return dotI8x16Portable(w0, a0) + dotI8x16Portable(w1, a1);
}

/// Per-row activation sum of one K-quant sub-block (= two Q8_K 16-groups,
/// `2*subblock` and `2*subblock+1`) for all four rows of a `BlockQ8_Kx4`,
/// lane = row. Mirrors the `bsums[(g/4)*16 + row*4 + g%4]` interleave that
/// `quantizeRowsQ8_Kx4*Into` writes.
pub inline fn bsumPairQ8_Kx4(a: *const types.BlockQ8_Kx4, comptime subblock: usize) QKV4i32 {
    const g0 = subblock * 2;
    const g1 = subblock * 2 + 1;
    var v: QKV4i32 = undefined;
    inline for (0..4) |row| {
        v[row] = @as(i32, a.bsums[(g0 / 4) * 16 + row * 4 + (g0 % 4)]) +
            @as(i32, a.bsums[(g1 / 4) * 16 + row * 4 + (g1 % 4)]);
    }
    return v;
}

/// Four rows' i8 dot of one 32-wide sub-block against pre-unpacked weights
/// `wv` (`wv[0]` = feature-groups 0..3, `wv[1]` = 4..7), returning the four row
/// dots in the four lanes of one i32x4 (lane = row). On aarch64 each
/// `sdot ...4b[g]` reuses the single unpacked weight register across all four
/// rows, so the whole sub-block is 8 `sdot`s and zero horizontal reductions.
/// Integer dots are order-independent, so this equals the per-row
/// `dotUnpackedI8x32` path exactly.
pub inline fn dot4RowsSubblockQ8_Kx4(a: *const types.BlockQ8_Kx4, comptime subblock: usize, wv: [2]QKV16i8) QKV4i32 {
    if (comptime builtin.cpu.arch == .aarch64) {
        var dot: QKV4i32 = @splat(0);
        inline for (0..4) |g| {
            const ag: QKV16i8 = @bitCast(a.qs[subblock * 128 + g * 16 ..][0..16].*);
            dot = sdotI8x16Lane(g, dot, ag, wv[0]);
        }
        inline for (0..4) |g| {
            const ag: QKV16i8 = @bitCast(a.qs[subblock * 128 + (g + 4) * 16 ..][0..16].*);
            dot = sdotI8x16Lane(g, dot, ag, wv[1]);
        }
        return dot;
    }
    if (comptime has_x86_vnni_ymm) return dot4RowsSubblockQ8_Kx4Simd(.vnni, a, subblock, wv);
    if (comptime has_x86_avx2) return dot4RowsSubblockQ8_Kx4Simd(.avx2, a, subblock, wv);
    return dot4RowsSubblockQ8_Kx4Simd(.widen, a, subblock, wv);
}

// vpermd-class (cross-lane): broadcast dword 2c to the low 128-bit lane and
// dword 2c+1 to the high lane of a 4-dword source, aligning one pre-unpacked
// 16-byte weight half against the [fg | fg+1] activation halves of a 32-byte
// Q8_Kx4 load.
inline fn broadcastPairGroupsI32x4(comptime c: comptime_int, v: QKV4i32) QKV8i32 {
    return @shuffle(i32, v, undefined, [8]i32{ 2 * c, 2 * c, 2 * c, 2 * c, 2 * c + 1, 2 * c + 1, 2 * c + 1, 2 * c + 1 });
}

/// x86/portable ymm arms of `dot4RowsSubblockQ8_Kx4`: each 32-byte Q8_Kx4
/// activation load already holds [fg: 4 rows x 4 features | fg+1: ...]
/// dword-per-row, so broadcasting the matching weight dword pair
/// (`broadcastPairGroupsI32x4`) turns the 32-feature x 4-row sub-block dot
/// into four grouped-dot ops + one half-fold, with no per-row rebuild.
/// OPERAND SHAPE: `wv` holds UNSIGNED weight codes (Q4_K nibbles in [0,15],
/// Q5_K values in [0,31]), natively vpdpbusd's u8 side, dotted directly (no
/// bias, no correction, no sign trick); activations are unrestricted i8.
/// SATURATION (avx2 tier): vpmaddubsw pair sums stay below 2^15 for both
/// code ranges (2*31*128 = 7936). NO OVERFLOW: |sum8 lane| <= 4*4*31*128 <
/// 2^17. Integer sums are order-independent, so the result is bit-identical
/// to `dot4RowsSubblockQ8_Kx4Scalar` (q4_k_tests.zig, q5_k_tests.zig).
pub fn dot4RowsSubblockQ8_Kx4Simd(comptime tier: X86DotTier, a: *const types.BlockQ8_Kx4, comptime subblock: usize, wv: [2]QKV16i8) QKV4i32 {
    var sum8: QKV8i32 = @splat(0);
    inline for (0..4) |c| {
        const act: QKV32i8 = @bitCast(a.qs[subblock * 128 + c * 32 ..][0..32].*);
        const wb: QKV32u8 = @bitCast(broadcastPairGroupsI32x4(c % 2, @as(QKV4i32, @bitCast(wv[c / 2]))));
        sum8 = dotGroupsI32x8(tier, sum8, wb, act);
    }
    return addHalvesI32x8(sum8);
}

/// The bit-exactness reference for `dot4RowsSubblockQ8_Kx4Simd`: the plain
/// per-row rebuild over the interleaved Q8_Kx4 layout (row r's 4 features for
/// feature-group g live at qs[subblock*128 + g*16 + r*4 ..][0..4]).
pub fn dot4RowsSubblockQ8_Kx4Scalar(a: *const types.BlockQ8_Kx4, comptime subblock: usize, wv: [2]QKV16i8) QKV4i32 {
    var dot: QKV4i32 = @splat(0);
    inline for (0..4) |row| {
        var acc: i32 = 0;
        inline for (0..8) |g| {
            inline for (0..4) |t| {
                acc += @as(i32, wv[g / 4][(g % 4) * 4 + t]) * @as(i32, a.qs[subblock * 128 + g * 16 + row * 4 + t]);
            }
        }
        dot[row] = acc;
    }
    return dot;
}

/// Split the eight i32 group sums of a ymm accumulator into the two 4-column
/// halves of an x8 chunk layout (cols 0..3 / cols 4..7).
pub inline fn lowHalfI32x8(v: QKV8i32) QKV4i32 {
    return @shuffle(i32, v, undefined, [4]i32{ 0, 1, 2, 3 });
}

pub inline fn highHalfI32x8(v: QKV8i32) QKV4i32 {
    return @shuffle(i32, v, undefined, [4]i32{ 4, 5, 6, 7 });
}

test {
    _ = @import("common_tests.zig");
}
