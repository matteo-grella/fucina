//! Shared low-level primitives for the quantized matmul kernels.
//!
//! Leaf module: the generic SIMD vector type aliases, the f16 bit conversions,
//! the AArch64 sdot/smmla helpers, the rounding/quantize primitives, the nibble
//! helpers, and the shared row/column blocking consts. Every ISA fact the
//! primitives gate on comes from `backend/isa.zig` (the capability booleans
//! and the int8-dot `tier`); nothing here reads `builtin.cpu`.
//! Imported by quant.zig and the per-type kernel files (q8_0/q4_k/q5_k/q6_k/cold);
//! it imports none of them. `types.zig` and `dtype.zig` supply the block
//! layouts the shared Q8_Kx4 lane dots and tile skeletons read.
const std = @import("std");
const dtype_mod = @import("../../dtype.zig");
const isa = @import("../isa.zig");
const types = @import("types.zig");

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

/// Round-half-to-even on 4 lanes via the 2^23 magic number (the default
/// rounding mode does the tie-break; `ops.rintScalar` is the scalar twin):
/// exact for every |x| < 2^23, larger magnitudes are already integral and
/// pass through. Branch-free and bit-identical to `roundNearestEven`.
fn roundNearestEvenVec4(x: QKV4f32) QKV4f32 {
    const VecU = @Vector(4, u32);
    const big: QKV4f32 = @splat(8388608.0);
    const ax = @abs(x);
    const shifted = @select(f32, ax < big, (ax + big) - big, ax);
    const sign_bits = @as(VecU, @bitCast(x)) & @as(VecU, @splat(0x80000000));
    return @bitCast(@as(VecU, @bitCast(shifted)) | sign_bits);
}

pub fn roundNearestEvenVec4ToI32(x: QKV4f32) QKV4i32 {
    if (comptime isa.has_neon) {
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
    if (comptime isa.has_neon) {
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
    // avxvnni — COMPILE-ONLY on this dev machine, see isa.has_x86_avxvnni).
    if (comptime isa.has_x86_vnni) return dotI8x16BiasForm(a, b);
    // AVX2 without VNNI: ggml-style sign-transfer dot (vpsignb + vpmaddubsw +
    // vpmaddwd). EXACTNESS DOMAIN: requires b[i] != -128 in lanes where
    // a[i] < 0 (see dotI8x16SignTrickForm below). Every call site passes
    // activations ∈ [-127,127] as b, weights as a: Q8_0 activations via
    // quantizeToI8's explicit clamp, Q5_K/Q6_K activations via
    // quantizeRowQ8_KInto's -127/max scale construction (BlockQ8_K qs is
    // never -128).
    if (comptime isa.has_x86_avx2) return dotI8x16SignTrickForm(a, b);
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
// All three helpers are only ever called under `comptime isa.has_x86_avx2` (their
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
    // avxvnni arm is COMPILE-ONLY on this dev machine (see isa.has_x86_avxvnni).
    if (comptime isa.has_x86_vnni) return dotU8I8x16WidenForm(a, b);
    if (comptime isa.has_x86_avx2) return dotU8I8x16MaddubsForm(a, b);
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
    if (comptime isa.has_x86_vnni_ymm and isa.has_llvm_asm) {
        // ENCODING IS LOAD-BEARING: LLVM's asm parser does not feature-check
        // inline asm and resolves the bare mnemonic to the EVEX (AVX512-VNNI)
        // form, which SIGILLs on AVX-VNNI-only cores (Alder/Raptor Lake) —
        // their VEX form must be selected with the explicit {vex} prefix
        // (LLVM defines the AVX-VNNI aliases as ExplicitVEXPrefix). Cores
        // gated in via AVX512-VNNI+VL (Ice Lake, no AVX-VNNI) take EVEX.
        const mnemonic = comptime if (isa.has_x86_avxvnni) "{vex} vpdpbusd" else "vpdpbusd";
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
    if (comptime isa.has_x86_avx2 and isa.has_llvm_asm) {
        return acc + maddwdSumPairsI32x8(maddubsI16x16(a, b));
    }
    return dpbusdI32x8Portable(acc, a, b);
}

/// vpshufb on ymm: per-128-bit-lane 16-entry byte table lookup —
/// out[i] = 0 when idx[i] bit 7 is set, else table[lane_base + (idx[i] & 0x0F)]
/// with lane_base 0 for the low half and 16 for the high half. Callers
/// duplicate a 16-entry table across both halves for a uniform lookup.
pub fn pshufbI8x32(table: QKV32i8, idx: QKV32u8) QKV32i8 {
    if (comptime isa.has_x86_avx2 and isa.has_llvm_asm) {
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
    if (comptime isa.has_x86_avx2 and isa.has_llvm_asm) {
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

/// The accumulate tier ladder every lane-packed format entry spells the
/// same way: `.scalar` takes the scalar reference body, the two NEON tiers
/// take the fused aarch64 body, and the x86/portable tiers take either one
/// comptime-tier body (`arms.x86`, called with `isa.tier` prepended — the
/// K-quant spelling) or three per-tier bodies (`arms.vnni`/`arms.avx2`/
/// `arms.widen` — the q8_0 packed spelling). Comptime-resolved: each build
/// sees exactly the one direct call the spelled-out switch made.
pub inline fn accumulateTier(comptime arms: anytype, args: anytype) TierArmReturn(arms) {
    const Arms = @TypeOf(arms);
    return switch (isa.tier) {
        .scalar => @call(.auto, arms.scalar, args),
        .neon_i8mm, .neon_sdot => @call(.auto, arms.aarch64, args),
        .x86_vnni => if (comptime @hasField(Arms, "x86"))
            @call(.auto, arms.x86, .{isa.tier} ++ args)
        else
            @call(.auto, arms.vnni, args),
        .x86_avx2 => if (comptime @hasField(Arms, "x86"))
            @call(.auto, arms.x86, .{isa.tier} ++ args)
        else
            @call(.auto, arms.avx2, args),
        .portable => if (comptime @hasField(Arms, "x86"))
            @call(.auto, arms.x86, .{isa.tier} ++ args)
        else
            @call(.auto, arms.widen, args),
    };
}

fn TierArmReturn(comptime arms: anytype) type {
    return @typeInfo(@TypeOf(arms.scalar)).@"fn".return_type.?;
}

/// The lane-packed rows shell: accumulate one packed RHS block into
/// `row_block` consecutive LHS rows' accumulators. A format with a fused
/// rows body registers it (`arms.aarch64`, and `arms.x86` for a
/// comptime-tier one); every other tier walks the rows in order with the
/// single-row accumulate `arms.one` (value- or pointer-form, told apart by
/// its return type). The walk (r = 0 upward at one block index) is the
/// bitwise contract; overriding tiers keep their fused bodies' own order.
pub inline fn accumulateLaneRows(
    comptime row_block: usize,
    comptime arms: anytype,
    lhs_blocks: anytype,
    row_start: usize,
    blocks_per_row: usize,
    block_index: usize,
    rhs: anytype,
    acc: anytype,
) void {
    const Arms = @TypeOf(arms);
    switch (isa.tier) {
        .neon_i8mm, .neon_sdot => if (comptime @hasField(Arms, "aarch64")) {
            return arms.aarch64(lhs_blocks, row_start, blocks_per_row, block_index, rhs, acc);
        },
        .x86_vnni, .x86_avx2, .portable => if (comptime @hasField(Arms, "x86")) {
            return arms.x86(isa.tier, lhs_blocks, row_start, blocks_per_row, block_index, rhs, acc);
        },
        .scalar => {},
    }
    inline for (0..row_block) |r| {
        const lhs = &lhs_blocks[(row_start + r) * blocks_per_row + block_index];
        if (comptime @typeInfo(@TypeOf(arms.one)).@"fn".return_type.? == void) {
            arms.one(lhs, rhs, &acc[r]);
        } else {
            acc[r] = arms.one(lhs, rhs, acc[r]);
        }
    }
}

pub const q8_0_row_block: usize = 4;
pub const q4_kx8_row_block: usize = 16;
pub const qk_col_block: usize = 2;
/// LHS rows per column-outer pass of the K-quant compact kernels.
pub const moe_row_tile: usize = 4;
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
    if (comptime isa.has_neon) {
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
    if (comptime isa.has_neon) {
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
    if (comptime isa.has_neon) {
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
    if (comptime isa.has_aarch64_i8mm) {
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

pub const ScaleMinK4 = struct {
    scale: u8,
    min: u8,
};

/// The 6-bit (scale, min) pair of sub-block `index` from a Q4_K/Q5_K
/// 12-byte scales field (ggml `get_scale_min_k4`).
pub fn getScaleMinK4(q: *const [dtype_mod.k_scale_size]u8, index: usize) ScaleMinK4 {
    if (index < 4) {
        return .{
            .scale = q[index] & 63,
            .min = q[index + 4] & 63,
        };
    }
    return .{
        .scale = (q[index + 4] & 0x0f) | ((q[index - 4] >> 6) << 4),
        .min = (q[index + 4] >> 4) | ((q[index] >> 6) << 4),
    };
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
// Tile skeletons. A format keeps its dot/accumulate/unpack bodies and
// instantiates one of these at comptime; the loop nests are the bitwise
// contract every format shares (rows outer, K blocks in order, one f32
// accumulator per output). Every instantiation is a plain
// `matmul<Fmt>RhsTile` (`TileFn`) the format's `gemm` and `RangeFromTile`
// take as-is.
// ---------------------------------------------------------------------------

/// The `matmul<Fmt>RhsTile` signature: `out[r0..r1, c0..c1]` (row stride
/// `n`) from `LhsBlock` rows against an `Rhs` container.
pub fn TileFn(comptime LhsBlock: type, comptime Rhs: type) type {
    return fn ([]f32, []const LhsBlock, *const Rhs, usize, usize, usize, usize, usize) void;
}

/// Every `.rows` container is a `CompactRhs`: `blocks` and
/// `blocks_per_column`.
inline fn rhsBlocksPerColumn(rhs: anytype) usize {
    return rhs.blocks_per_column;
}

fn RhsBlocksSlice(comptime Rhs: type) type {
    return @FieldType(Rhs, "blocks");
}

inline fn rhsBlocks(rhs: anytype) RhsBlocksSlice(@TypeOf(rhs.*)) {
    return rhs.blocks;
}

fn ReturnType(comptime f: anytype) type {
    return @typeInfo(@TypeOf(f)).@"fn".return_type.?;
}

/// The row-outer tile of the direct-dot formats: per LHS row the columns
/// advance `col_block` at a time (one f32 accumulator per column, each LHS
/// unit loaded once per K step) with a single-column tail; every output is
/// the in-order sum of its K blocks' `arms.dot(rhs_block, lhs)` values.
/// `lhs` is the LHS block pointer, or `arms.prepare(block)` when the format
/// hoists a per-block conversion out of the column loop, or the run of
/// `arms.lhs_per_rhs` LHS blocks one RHS block spans (a slice, for the
/// formats whose dot takes one). `col_block == 1` is the plain per-column
/// walk.
pub fn RowOuterTile(comptime LhsBlock: type, comptime Rhs: type, comptime col_block: usize, comptime arms: anytype) TileFn(LhsBlock, Rhs) {
    const Arms = @TypeOf(arms);
    const lhs_run = @hasField(Arms, "lhs_per_rhs");
    const lhs_per_rhs: usize = if (lhs_run) arms.lhs_per_rhs else 1;
    const LhsUnit = if (lhs_run)
        *const [lhs_per_rhs]LhsBlock
    else if (@hasField(Arms, "prepare"))
        ReturnType(arms.prepare)
    else
        *const LhsBlock;
    return struct {
        inline fn lhsUnit(lhs_row: []const LhsBlock, block_index: usize) LhsUnit {
            if (comptime lhs_run) return lhs_row[block_index * lhs_per_rhs ..][0..lhs_per_rhs];
            const block = &lhs_row[block_index];
            return if (comptime @hasField(Arms, "prepare")) arms.prepare(block) else block;
        }

        fn tile(out: []f32, lhs_blocks: []const LhsBlock, rhs: *const Rhs, n: usize, r0: usize, r1: usize, c0: usize, c1: usize) void {
            const blocks_per_row = rhsBlocksPerColumn(rhs);
            const blocks = rhsBlocks(rhs);
            const lhs_blocks_per_row = blocks_per_row * lhs_per_rhs;
            var i = r0;
            while (i < r1) : (i += 1) {
                const lhs_row = lhs_blocks[i * lhs_blocks_per_row ..][0..lhs_blocks_per_row];
                var j = c0;

                if (comptime col_block > 1) {
                    while (j + col_block <= c1) : (j += col_block) {
                        var acc = [_]f32{0} ** col_block;
                        var block_index: usize = 0;
                        while (block_index < blocks_per_row) : (block_index += 1) {
                            const lhs = lhsUnit(lhs_row, block_index);
                            inline for (0..col_block) |c| {
                                const rhs_block = &blocks[(j + c) * blocks_per_row + block_index];
                                acc[c] += arms.dot(rhs_block, lhs);
                            }
                        }
                        inline for (0..col_block) |c| out[i * n + j + c] = acc[c];
                    }
                }

                while (j < c1) : (j += 1) {
                    const rhs_col = rhs.columnBlocks(j);
                    var acc: f32 = 0;
                    var block_index: usize = 0;
                    while (block_index < blocks_per_row) : (block_index += 1) {
                        acc += arms.dot(&rhs_col[block_index], lhsUnit(lhs_row, block_index));
                    }
                    out[i * n + j] = acc;
                }
            }
        }
    }.tile;
}

inline fn storeLanes4(out: []f32, base: usize, acc: QKV4f32) void {
    inline for (0..4) |lane| out[base + lane] = acc[lane];
}

inline fn storeLanes8(out: []f32, base: usize, acc: [2]QKV4f32) void {
    inline for (0..2) |half| {
        inline for (0..4) |lane| out[base + half * 4 + lane] = acc[half][lane];
    }
}

/// The x4 lane-pack tile: `q8_0_row_block` rows per register block through
/// `arms.rows` (one packed RHS block into the block's row accumulators),
/// then single rows through `arms.one` (value-form accumulate) or, when the
/// format splits the acc-independent part out as `arms.contribution`, two
/// blocks per step with the adds landing in block order.
pub fn LaneX4Tile(comptime LhsBlock: type, comptime Rhs: type, comptime arms: anytype) TileFn(LhsBlock, Rhs) {
    const Arms = @TypeOf(arms);
    return struct {
        fn tile(out: []f32, lhs_blocks: []const LhsBlock, rhs: *const Rhs, n: usize, r0: usize, r1: usize, c0: usize, c1: usize) void {
            std.debug.assert(c0 % 4 == 0);
            std.debug.assert(c1 % 4 == 0);

            const blocks_per_row = rhs.blocks_per_group;
            var i = r0;
            while (i + q8_0_row_block <= r1) : (i += q8_0_row_block) {
                var j = c0;
                while (j < c1) : (j += 4) {
                    const rhs_group = rhs.groupBlocks(j / 4);
                    var acc: [q8_0_row_block]QKV4f32 = undefined;
                    inline for (0..q8_0_row_block) |r| acc[r] = @splat(0);

                    var block_index: usize = 0;
                    while (block_index < blocks_per_row) : (block_index += 1) {
                        const rhs_block = &rhs_group[block_index];
                        arms.rows(lhs_blocks, i, blocks_per_row, block_index, rhs_block, &acc);
                    }

                    inline for (0..q8_0_row_block) |r| storeLanes4(out, (i + r) * n + j, acc[r]);
                }
            }

            while (i < r1) : (i += 1) {
                const lhs_row = lhs_blocks[i * blocks_per_row ..][0..blocks_per_row];
                var j = c0;
                while (j < c1) : (j += 4) {
                    const rhs_group = rhs.groupBlocks(j / 4);
                    var acc: QKV4f32 = @splat(0);
                    var block_index: usize = 0;
                    if (comptime @hasField(Arms, "contribution")) {
                        // Block pairs: the two contributions carry no dependency on acc
                        // or each other, so their integer dots pipeline; the adds land
                        // in the original order (bitwise identical to the serial loop).
                        while (block_index + 2 <= blocks_per_row) : (block_index += 2) {
                            const t0 = arms.contribution(&lhs_row[block_index], &rhs_group[block_index]);
                            const t1 = arms.contribution(&lhs_row[block_index + 1], &rhs_group[block_index + 1]);
                            acc += t0;
                            acc += t1;
                        }
                        if (block_index < blocks_per_row) {
                            acc += arms.contribution(&lhs_row[block_index], &rhs_group[block_index]);
                        }
                    } else {
                        while (block_index < blocks_per_row) : (block_index += 1) {
                            acc = arms.one(&lhs_row[block_index], &rhs_group[block_index], acc);
                        }
                    }
                    storeLanes4(out, i * n + j, acc);
                }
            }
        }
    }.tile;
}

/// The x8 lane-pack tile: `q4_kx8_row_block` rows per register block through
/// `arms.rows`, then 8-row and 4-row blocks and single rows through
/// `arms.one` (pointer-form accumulate into one `[2]QKV4f32` per row).
pub fn LaneX8Tile(comptime LhsBlock: type, comptime Rhs: type, comptime arms: anytype) TileFn(LhsBlock, Rhs) {
    return struct {
        fn tile(out: []f32, lhs_blocks: []const LhsBlock, rhs: *const Rhs, n: usize, r0: usize, r1: usize, c0: usize, c1: usize) void {
            std.debug.assert(c0 % 8 == 0);
            std.debug.assert(c1 % 8 == 0);

            const blocks_per_row = rhs.blocks_per_group;
            var i = r0;
            while (i + q4_kx8_row_block <= r1) : (i += q4_kx8_row_block) {
                var j = c0;
                while (j < c1) : (j += 8) {
                    const rhs_group = rhs.groupBlocks(j / 8);
                    var acc: [q4_kx8_row_block][2]QKV4f32 = undefined;
                    inline for (0..q4_kx8_row_block) |r| {
                        acc[r][0] = @splat(0);
                        acc[r][1] = @splat(0);
                    }

                    var block_index: usize = 0;
                    while (block_index < blocks_per_row) : (block_index += 1) {
                        const rhs_block = &rhs_group[block_index];
                        arms.rows(lhs_blocks, i, blocks_per_row, block_index, rhs_block, &acc);
                    }

                    inline for (0..q4_kx8_row_block) |r| storeLanes8(out, (i + r) * n + j, acc[r]);
                }
            }

            while (i + 8 <= r1) : (i += 8) {
                tailRows(8, out, lhs_blocks, rhs, n, i, c0, c1, blocks_per_row);
            }

            while (i + 4 <= r1) : (i += 4) {
                tailRows(4, out, lhs_blocks, rhs, n, i, c0, c1, blocks_per_row);
            }

            while (i < r1) : (i += 1) {
                const lhs_row = lhs_blocks[i * blocks_per_row ..][0..blocks_per_row];
                var j = c0;
                while (j < c1) : (j += 8) {
                    const rhs_group = rhs.groupBlocks(j / 8);
                    var acc: [2]QKV4f32 = .{ @splat(0), @splat(0) };
                    var block_index: usize = 0;
                    while (block_index < blocks_per_row) : (block_index += 1) {
                        arms.one(&lhs_row[block_index], &rhs_group[block_index], &acc);
                    }
                    storeLanes8(out, i * n + j, acc);
                }
            }
        }

        fn tailRows(
            comptime row_block: usize,
            out: []f32,
            lhs_blocks: []const LhsBlock,
            rhs: *const Rhs,
            n: usize,
            row_start: usize,
            c0: usize,
            c1: usize,
            blocks_per_row: usize,
        ) void {
            var j = c0;
            while (j < c1) : (j += 8) {
                const rhs_group = rhs.groupBlocks(j / 8);
                var acc: [row_block][2]QKV4f32 = undefined;
                inline for (0..row_block) |r| {
                    acc[r][0] = @splat(0);
                    acc[r][1] = @splat(0);
                }

                var block_index: usize = 0;
                while (block_index < blocks_per_row) : (block_index += 1) {
                    const rhs_block = &rhs_group[block_index];
                    inline for (0..row_block) |r| {
                        const lhs = &lhs_blocks[(row_start + r) * blocks_per_row + block_index];
                        arms.one(lhs, rhs_block, &acc[r]);
                    }
                }

                inline for (0..row_block) |r| storeLanes8(out, (row_start + r) * n + j, acc[r]);
            }
        }
    }.tile;
}

/// Whether an x8-over-Q8_Kx4 tile accepts a final row group shorter than
/// four rows (stores only the `r1 - i` real rows) or requires `r1 % 4 == 0`.
pub const RowGroupTail = enum { full, partial };

/// The x8 lane-pack tile over 4-row-interleaved Q8_Kx4 activations: each
/// step feeds one Q8_Kx4 group and one packed RHS block to `accumulate`
/// (four rows x eight columns of accumulators).
pub fn LaneX8Q8_Kx4Tile(comptime Rhs: type, comptime accumulate: anytype, comptime tail: RowGroupTail) TileFn(types.BlockQ8_Kx4, Rhs) {
    return struct {
        fn tile(out: []f32, lhs_blocks: []const types.BlockQ8_Kx4, rhs: *const Rhs, n: usize, r0: usize, r1: usize, c0: usize, c1: usize) void {
            std.debug.assert(r0 % 4 == 0);
            if (comptime tail == .full) std.debug.assert(r1 % 4 == 0);
            std.debug.assert(c0 % 8 == 0);
            std.debug.assert(c1 % 8 == 0);

            const blocks_per_row = rhs.blocks_per_group;
            var i = r0;
            while (i < r1) : (i += 4) {
                const full_row_group = tail == .full or i + 4 <= r1;
                var j = c0;
                while (j < c1) : (j += 8) {
                    const rhs_group = rhs.groupBlocks(j / 8);
                    const lhs_row_group = (i / 4) * blocks_per_row;
                    var acc: [4][2]QKV4f32 = undefined;
                    inline for (0..4) |row| {
                        acc[row][0] = @splat(0);
                        acc[row][1] = @splat(0);
                    }

                    var block_index: usize = 0;
                    while (block_index < blocks_per_row) : (block_index += 1) {
                        accumulate(&lhs_blocks[lhs_row_group + block_index], &rhs_group[block_index], &acc);
                    }

                    if (full_row_group) {
                        inline for (0..4) |row| storeLanes8(out, (i + row) * n + j, acc[row]);
                    } else {
                        const valid_rows = r1 - i;
                        inline for (0..4) |row| {
                            if (row < valid_rows) storeLanes8(out, (i + row) * n + j, acc[row]);
                        }
                    }
                }
            }
        }
    }.tile;
}

/// Column-outer K-quant (Q4_K/Q5_K) matmul for the m>1 (batched MoE
/// prefill) case: each weight sub-block is unpacked ONCE (`unpack`, 32 codes
/// as two i8x16) and dotted against a tile of `moe_row_tile` LHS rows,
/// amortizing the unpack over the batch instead of re-unpacking per row
/// like the row-outer tile. Numerically identical to it (same per-block
/// deferred-f32 reduction, same cross-block accumulation order).
pub fn KQuantColOuterTile(comptime Rhs: type, comptime unpack: anytype) TileFn(dtype_mod.BlockQ8_K, Rhs) {
    return struct {
        fn tile(out: []f32, lhs_blocks: []const dtype_mod.BlockQ8_K, rhs: *const Rhs, n: usize, r0: usize, r1: usize, c0: usize, c1: usize) void {
            const bpc = rhs.blocks_per_column;
            var j = c0;
            while (j < c1) : (j += 1) {
                const col = rhs.columnBlocks(j);
                var row0 = r0;
                while (row0 < r1) : (row0 += moe_row_tile) {
                    const tn = @min(moe_row_tile, r1 - row0);
                    var acc_f32 = [_]f32{0} ** moe_row_tile;
                    var bi: usize = 0;
                    while (bi < bpc) : (bi += 1) {
                        const w = &col[bi];
                        var iscale = [_]i32{0} ** moe_row_tile;
                        var imin = [_]i32{0} ** moe_row_tile;
                        inline for (0..8) |subblock| {
                            const wv = unpack(w, subblock);
                            const sm = getScaleMinK4(&w.scales, subblock);
                            var r: usize = 0;
                            while (r < tn) : (r += 1) {
                                const a = &lhs_blocks[(row0 + r) * bpc + bi];
                                const a0: QKV16i8 = @bitCast(a.qs[subblock * 32 ..][0..16].*);
                                const a1: QKV16i8 = @bitCast(a.qs[subblock * 32 + 16 ..][0..16].*);
                                const acc = dotUnpackedI8x32(wv[0], wv[1], a0, a1);
                                const bsum = @as(i32, a.bsums[subblock * 2]) + @as(i32, a.bsums[subblock * 2 + 1]);
                                iscale[r] += @as(i32, sm.scale) * acc;
                                imin[r] += @as(i32, sm.min) * bsum;
                            }
                        }
                        const d = f16BitsToF32(w.dm[0]);
                        const dmin = f16BitsToF32(w.dm[1]);
                        var r: usize = 0;
                        while (r < tn) : (r += 1) {
                            const ad = lhs_blocks[(row0 + r) * bpc + bi].d;
                            acc_f32[r] += (d * ad) * @as(f32, @floatFromInt(iscale[r])) - (dmin * ad) * @as(f32, @floatFromInt(imin[r]));
                        }
                    }
                    var r: usize = 0;
                    while (r < tn) : (r += 1) out[(row0 + r) * n + j] = acc_f32[r];
                }
            }
        }
    }.tile;
}

/// Column-outer K-quant (Q4_K/Q5_K) matmul over **4-row-interleaved Q8_Kx4**
/// activations. Like `KQuantColOuterTile` it unpacks each weight sub-block
/// once and reuses it across the row tile, but it packs the four rows into
/// the `sdot` lanes (`dot4RowsSubblockQ8_Kx4`) so the four rows share one
/// i32x4 accumulator with no per-row horizontal reduction, and the
/// deferred-f32 epilogue runs vector-wide over the four rows. `lhs_blocks`
/// holds `ceil(m/4)` Q8_Kx4 groups per K-block (tail rows zero-padded, e.g.
/// via `quantizeRowsQ8_Kx4PaddedInto`); `m` is the real row count so padded
/// lanes are never stored. Bit-identical to the per-row column-outer /
/// row-outer tile (same integer reduction, same cross-block f32
/// accumulation order).
pub fn KQuantColOuterQ8_Kx4Tile(comptime Rhs: type, comptime unpack: anytype) fn ([]f32, []const types.BlockQ8_Kx4, *const Rhs, usize, usize, usize, usize) void {
    return struct {
        fn tile(out: []f32, lhs_blocks: []const types.BlockQ8_Kx4, rhs: *const Rhs, n: usize, m: usize, c0: usize, c1: usize) void {
            const bpc = rhs.blocks_per_column;
            const row_groups = (m + 3) / 4;
            var j = c0;
            while (j < c1) : (j += 1) {
                const col = rhs.columnBlocks(j);
                var rg: usize = 0;
                while (rg < row_groups) : (rg += 1) {
                    const row0 = rg * 4;
                    const tn = @min(@as(usize, 4), m - row0);
                    var acc_f32: QKV4f32 = @splat(0);
                    var bi: usize = 0;
                    while (bi < bpc) : (bi += 1) {
                        const w = &col[bi];
                        const a = &lhs_blocks[rg * bpc + bi];
                        var iscale: QKV4i32 = @splat(0);
                        var imin: QKV4i32 = @splat(0);
                        inline for (0..8) |subblock| {
                            const wv = unpack(w, subblock);
                            const sm = getScaleMinK4(&w.scales, subblock);
                            const dot = dot4RowsSubblockQ8_Kx4(a, subblock, wv);
                            iscale += @as(QKV4i32, @splat(@as(i32, sm.scale))) * dot;
                            imin += @as(QKV4i32, @splat(@as(i32, sm.min))) * bsumPairQ8_Kx4(a, subblock);
                        }
                        const d = f16BitsToF32(w.dm[0]);
                        const dmin = f16BitsToF32(w.dm[1]);
                        const ad: QKV4f32 = a.d;
                        acc_f32 += (@as(QKV4f32, @splat(d)) * ad) * @as(QKV4f32, @floatFromInt(iscale)) -
                            (@as(QKV4f32, @splat(dmin)) * ad) * @as(QKV4f32, @floatFromInt(imin));
                    }
                    const acc_arr: [4]f32 = acc_f32;
                    var r: usize = 0;
                    while (r < tn) : (r += 1) out[(row0 + r) * n + j] = acc_arr[r];
                }
            }
        }
    }.tile;
}

// ---------------------------------------------------------------------------
// Q8_Kx4 lane dots shared by the K-quant column-outer kernels (q4_k, q5_k):
// the grouped u8·i8 dot-accumulate step over the x86/portable tiers of
// `isa.Tier`, and the four-row sub-block dot over pre-unpacked weights.
// ---------------------------------------------------------------------------

/// One grouped weight·activation dot-accumulate step; `w` holds unsigned
/// weight codes, `a` is unrestricted i8. Exactness per tier:
///   x86_vnni: vpdpbusd, exact i32 for all u8·i8 inputs, no saturation.
///   x86_avx2: vpmaddubsw+vpmaddwd; saturation-free while pair sums stay
///             below 2^15 (w <= 15 gives 3840, w <= 31 gives 7936).
///   portable: weight codes below 128 fit i8 (@bitCast is value-preserving)
///             and i8·i8 products are exact in the widening dot on every
///             target.
/// The NEON tiers never reach this step: they dot with `sdot` lanes.
pub inline fn dotGroupsI32x8(comptime tier: isa.Tier, acc: QKV8i32, w: QKV32u8, a: QKV32i8) QKV8i32 {
    return switch (tier) {
        .x86_vnni => dpbusdI32x8(acc, w, a),
        .x86_avx2 => maddubsDotGroupsI32x8(acc, w, a),
        .portable => dotI8GroupsWidenI32x8(acc, @bitCast(w), a),
        .neon_i8mm, .neon_sdot => @compileError("dotGroupsI32x8: the NEON tiers dot with sdot lanes, not the grouped ymm step"),
        .scalar => @compileError("dotGroupsI32x8: the scalar tier takes the formats' *Scalar twins"),
    };
}

/// i8 dot of one 32-wide sub-block: two 16-lane halves of unpacked weight
/// codes against two halves of Q8_K activations.
pub fn dotUnpackedI8x32(w0: QKV16i8, w1: QKV16i8, a0: QKV16i8, a1: QKV16i8) i32 {
    switch (isa.tier) {
        .neon_i8mm, .neon_sdot => {
            var acc: QKV4i32 = @splat(0);
            acc = sdotI8x16(acc, w0, a0);
            acc = sdotI8x16(acc, w1, a1);
            return @reduce(.Add, acc);
        },
        // VNNI lowers each to vpdpbusd (via the +128 bias), AVX2 takes the
        // sign-trick vpmaddubsw path (w is a Q4_K or Q5_K code, a comes from
        // BlockQ8_K, i.e. quantizeRowQ8_KInto's -127/max scale construction,
        // so a is in [-127,127]: inside the sign-trick exactness domain; see
        // dotI8x16Portable).
        .x86_vnni, .x86_avx2, .portable => return dotI8x16Portable(w0, a0) + dotI8x16Portable(w1, a1),
        .scalar => {
            var acc: i32 = 0;
            inline for (0..16) |i| acc += @as(i32, w0[i]) * @as(i32, a0[i]) + @as(i32, w1[i]) * @as(i32, a1[i]);
            return acc;
        },
    }
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
    switch (isa.tier) {
        .neon_i8mm, .neon_sdot => {
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
        },
        .x86_vnni, .x86_avx2, .portable => return dot4RowsSubblockQ8_Kx4Simd(isa.tier, a, subblock, wv),
        .scalar => return dot4RowsSubblockQ8_Kx4Scalar(a, subblock, wv),
    }
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
pub fn dot4RowsSubblockQ8_Kx4Simd(comptime tier: isa.Tier, a: *const types.BlockQ8_Kx4, comptime subblock: usize, wv: [2]QKV16i8) QKV4i32 {
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
