//! Standalone cross-ISA parity checker for the int8 dot primitives and the
//! Q4_K / Q8_0 / TQ2_0 dot kernels (the x86 AVX2/AVX-VNNI bring-up validation
//! vehicle).
//!
//! Self-contained main(): runs semantic asserts (kernel vs scalar reference)
//! on deterministic randomized + extreme inputs and exits nonzero on any
//! mismatch. It also prints FNV-1a checksums of the raw result bit patterns,
//! so runs from DIFFERENT environments can be diffed for bit-exactness.
//!
//! Wired build step: `zig build x86dot-check` builds AND runs this checker
//! (always ReleaseSafe, the run-book config; the run leg follows -Dtarget) and
//! additionally COMPILES — never runs — one leg per feature gate no local
//! substrate can execute: x86_64_v3 (AVX2), alderlake (AVX-VNNI), znver4
//! (AVX512-VNNI), neoverse_v1 (aarch64 smmla/FEAT_I8MM). Those legs only stop
//! bit-rot at build time; what has actually EXECUTED is the table below.
//!
//! Per-arm execution coverage. The emulated/hardware rows are dated
//! attestations the build cannot enforce — re-run the matrix legs below after
//! touching the kernels.
//!
//!   arm                              | executed on                        | attestation
//!   aarch64 sdot asm                 | natively, Apple M1 Max             | ongoing: zig build test + zig build x86dot-check
//!   aarch64 smmla asm (FEAT_I8MM)    | NEVER — M1 lacks I8MM              | compile+objdump-verified 2026-06-11; execution needs Graviton3+/Grace hardware
//!   x86 portable (< AVX2)            | Rosetta 2 (real x86 semantics)     | 2026-06-11, leg (b) below
//!   x86 AVX2 sign-trick asm          | valid. emulator + i9-13950HX hw    | 2026-06-11 leg (c); 2026-07-03 native run on the x86 box
//!   x86 AVX-VNNI (VEX vpdpbusd)      | i9-13950HX (Raptor Lake), Linux    | EXECUTED 2026-07-03 (ReleaseFast; see note below): bias/widen forms AND the {vex} ymm asm, checksum bit-equal to the M1/Rosetta runs
//!   x86 AVX512-VNNI (EVEX vpdpbusd)  | NEVER                              | compile-verified only; execution needs Ice Lake/Zen4 hardware
//!   x86 ymm dpbusdI32x8 asm          | i9-13950HX                         | EXECUTED 2026-07-03, native run (the Q8_0x4 packed-kernel VNNI arm)
//!   x86 ymm maddubs/psignb grouped   | i9-13950HX                         | EXECUTED 2026-07-03, native run of the -Dcpu=x86_64_v3 build (sign-trick arm)
//!   tq2_0 aarch64 sdot arm           | natively, Apple M1 Max             | EXECUTED 2026-07-07 (this checker's tq2_0 section + the ternary parity suites); ongoing via zig build x86dot-check
//!   tq2_0 x86 AVX2 maddubs arm       | i9-13950HX (Raptor Lake), Linux    | EXECUTED 2026-07-07, -Dcpu=x86_64_v3 native run (avxvnni=false); checksum bit-equal to the VNNI run (b1f84dde82d0c0a4)
//!   tq2_0 x86 AVX-VNNI vpdpbusd arm  | i9-13950HX (Raptor Lake), Linux    | EXECUTED 2026-07-07, native run (avx2+avxvnni); full zig build test also green on the box
//!   tq2_0x4 aarch64 by-elem sdot arm | natively, Apple M1 Max             | EXECUTED 2026-07-24 (this checker's tq2_0x4 section + ternary parity suites + the bench-ternary bitwise gate)
//!   tq2_0x4 x86 portable tier        | Rosetta 2 (real x86 semantics)     | EXECUTED 2026-07-24, leg (b); x86-chain checksum f79a3f29e3be6cab
//!   tq2_0x4 x86 AVX2 maddubs arm     | validated x86-64 emulator          | EXECUTED 2026-07-24, leg (c), x86_64_v3 static musl; checksum bit-equal to the Rosetta run (f79a3f29e3be6cab)
//!   tq2_0x4 x86 VNNI ymm arm         | NEVER                              | compile-verified 2026-07-24 (alderlake leg); execution needs AVX-VNNI hardware — no emulator implements AVX-VNNI
//!   tq2_0 folded aarch64 arm         | natively, Apple M1 Max             | EXECUTED 2026-07-24 (this checker's folded section + ternary parity suites)
//!   tq2_0 folded x86 portable tier   | Rosetta 2 (real x86 semantics)     | EXECUTED 2026-07-24, leg (b); x86-chain checksum 3ad62348d3bdb3f6
//!   tq2_0 folded x86 AVX2 arm        | validated x86-64 emulator          | EXECUTED 2026-07-24, leg (c), x86_64_v3 static musl; checksum bit-equal to the Rosetta run (3ad62348d3bdb3f6)
//!   tq2_0 folded x86 VNNI ymm arm    | NEVER                              | compile-verified 2026-07-24 (alderlake leg); execution needs AVX-VNNI hardware
//!   portable widening tier (256-bit) | every host (no gate)               | ongoing: zig build test everywhere + this checker
//!
//! The 2026-07-03 hardware attestations ran `zig build test -Doptimize=ReleaseFast`
//! + `zig build x86dot-check` natively on the box; the K-quant/q8_0 packed-arm
//! parity suites ("ggml_q8_0x4/q4_kx4/q4_kx8/q5_kx8/q6_kx4 ... SIMD arms match
//! the scalar arm bit-exactly") are the execution vehicle for the kernel arms.
//! ReleaseFast matters: Debug builds execute the portable twins instead (the
//! stage2-assembler gate — see common.has_llvm_asm).
//!
//! ENCODING TRAP (2026-07-03, hardware-found): LLVM's asm parser does not
//! feature-check inline asm — a bare `vpdpbusd` assembles to the EVEX
//! (AVX512-VNNI) form even when only AVX-VNNI is enabled, and SIGILLs on
//! Alder/Raptor Lake. The VEX form must be selected with an explicit {vex}
//! prefix (see common.dpbusdI32x8).
//!
//! Build & run matrix (from the aarch64 dev machine):
//!
//!   # (a) native aarch64 (sdot paths) — the wired step runs exactly this:
//!   zig build-exe src/x86dot_check.zig -O ReleaseSafe -femit-bin=zig-out/x86dot_native
//!   ./zig-out/x86dot_native
//!
//!   # (b) x86-64 BASELINE on real x86 semantics via Rosetta 2 (portable path;
//!   #     Rosetta executes <= SSE4.2 only — do NOT build v3 for it):
//!   zig build-exe src/x86dot_check.zig -target x86_64-macos -mcpu=baseline \
//!       -O ReleaseSafe -femit-bin=zig-out/x86dot_rosetta
//!   arch -x86_64 ./zig-out/x86dot_rosetta
//!
//!   # (c) x86-64-v3 (AVX2 vpmaddubsw/vpsignb path), static musl so it runs on
//!   #     any x86-64 Linux substrate — real hardware, or a user-mode emulator
//!   #     that passes the validation gate below:
//!   zig build-exe src/x86dot_check.zig -target x86_64-linux-musl -mcpu=x86_64_v3 \
//!       -O ReleaseSafe -femit-bin=zig-out/x86dot_v3
//!   # then execute zig-out/x86dot_v3 on the substrate and diff the checksums.
//!
//! WARNING — emulators are guilty until proven: at least one widely deployed
//! x86-64 user-mode emulator version executes AVX2 SILENTLY WRONG (no SIGILL,
//! corrupt lanes). Before trusting any emulation substrate, reproduce this
//! checker's recorded x86 checksums (the attestation rows above) on it first;
//! a substrate that cannot reproduce them attests nothing.
//! No emulator implements AVX-VNNI or executes AVX512-VNNI here; the AVX-VNNI
//! arms are hardware-attested instead (the 2026-07-03 rows above), and the
//! EVEX AVX512-VNNI arm remains compile-verified only (needs Ice Lake/Zen4).

const std = @import("std");
const builtin = @import("builtin");
const quant = @import("backend/quant.zig");
const common = @import("backend/quant/common.zig");
const tensor_mod = @import("tensor.zig");

const BlockQ4_K = quant.BlockQ4_K;
const BlockQ8_K = quant.BlockQ8_K;
const BlockQ8_0 = quant.BlockQ8_0;
const BlockTQ2_0 = quant.BlockTQ2_0;
const qk_k_block_size = quant.qk_k_block_size;

var failures: usize = 0;
var fnv: u64 = 0xcbf29ce484222325;

fn fnvAdd(bits: anytype) void {
    const bytes = std.mem.asBytes(&bits);
    for (bytes) |b| {
        fnv ^= b;
        fnv *%= 0x100000001b3;
    }
}

fn checkI32(label: []const u8, expected: i32, got: i32) void {
    fnvAdd(got);
    if (expected != got) {
        failures += 1;
        std.debug.print("FAIL {s}: expected {d}, got {d}\n", .{ label, expected, got });
    }
}

fn checkF32Exact(label: []const u8, expected: f32, got: f32) void {
    fnvAdd(@as(u32, @bitCast(got)));
    if (@as(u32, @bitCast(expected)) != @as(u32, @bitCast(got))) {
        failures += 1;
        std.debug.print("FAIL {s}: expected {x:0>8} ({d}), got {x:0>8} ({d})\n", .{
            label, @as(u32, @bitCast(expected)), expected, @as(u32, @bitCast(got)), got,
        });
    }
}

fn checkI32x4(label: []const u8, expected: [4]i32, got: common.QKV4i32) void {
    const got_arr: [4]i32 = got;
    fnvAdd(got_arr);
    if (!std.mem.eql(i32, &expected, &got_arr)) {
        failures += 1;
        std.debug.print("FAIL {s}: expected {any}, got {any}\n", .{ label, expected, got_arr });
    }
}

fn checkI32x8(label: []const u8, expected: [8]i32, got: common.QKV8i32) void {
    const got_arr: [8]i32 = got;
    fnvAdd(got_arr);
    if (!std.mem.eql(i32, &expected, &got_arr)) {
        failures += 1;
        std.debug.print("FAIL {s}: expected {any}, got {any}\n", .{ label, expected, got_arr });
    }
}

fn checkI8x32(label: []const u8, expected: [32]i8, got: common.QKV32i8) void {
    const got_arr: [32]i8 = got;
    fnvAdd(got_arr);
    if (!std.mem.eql(i8, &expected, &got_arr)) {
        failures += 1;
        std.debug.print("FAIL {s}: expected {any}, got {any}\n", .{ label, expected, got_arr });
    }
}

fn checkF32Rel(label: []const u8, expected: f32, got: f32, tol: f32) void {
    fnvAdd(@as(u32, @bitCast(got)));
    const denom = @max(@abs(expected), @abs(got));
    if (denom != 0 and @abs(expected - got) / denom > tol) {
        failures += 1;
        std.debug.print("FAIL {s}: expected {d}, got {d}\n", .{ label, expected, got });
    }
}

// ---- scalar references (mirror the in-kernel test replicas) ----------------

fn refDotU8I8(a: *const [16]u8, b: *const [16]i8) i32 {
    var s: i32 = 0;
    for (a, b) |x, y| s += @as(i32, x) * @as(i32, y);
    return s;
}

fn refDotI8I8(a: *const [16]i8, b: *const [16]i8) i32 {
    var s: i32 = 0;
    for (a, b) |x, y| s += @as(i32, x) * @as(i32, y);
    return s;
}

// The 4-lane aarch64 matrix/dot primitives (single sdot/smmla on hardware,
// portable forms elsewhere). Exact i32 accumulate, no saturation anywhere.
fn refSdot(acc: [4]i32, a: *const [16]i8, b: *const [16]i8) [4]i32 {
    var out = acc;
    for (0..4) |lane| {
        for (0..4) |k| out[lane] += @as(i32, a[lane * 4 + k]) * @as(i32, b[lane * 4 + k]);
    }
    return out;
}

fn refSdotLane(lane: usize, acc: [4]i32, a: *const [16]i8, b: *const [16]i8) [4]i32 {
    var out = acc;
    for (0..4) |i| {
        for (0..4) |k| out[i] += @as(i32, a[i * 4 + k]) * @as(i32, b[lane * 4 + k]);
    }
    return out;
}

fn refSmmla(acc: [4]i32, a: *const [16]i8, b: *const [16]i8) [4]i32 {
    // a, b = 2x8 i8 matrices (rows [0..8), [8..16)); acc += a · bᵀ as
    // lanes {a0·b0, a0·b1, a1·b0, a1·b1}.
    var out = acc;
    for (0..2) |i| {
        for (0..2) |j| {
            for (0..8) |k| out[i * 2 + j] += @as(i32, a[i * 8 + k]) * @as(i32, b[j * 8 + k]);
        }
    }
    return out;
}

// The 256-bit grouped primitives behind the Q8_0x4 packed-accumulate arms:
// out[g] = acc[g] + Σ_{k<4} a[4g+k]·b[4g+k], eight groups per 32-byte vector.
fn refGroupDotsU8I8x32(acc: [8]i32, a: *const [32]u8, b: *const [32]i8) [8]i32 {
    var out = acc;
    for (0..8) |g| {
        for (0..4) |k| out[g] += @as(i32, a[g * 4 + k]) * @as(i32, b[g * 4 + k]);
    }
    return out;
}

fn refGroupDotsI8I8x32(acc: [8]i32, a: *const [32]i8, b: *const [32]i8) [8]i32 {
    var out = acc;
    for (0..8) |g| {
        for (0..4) |k| out[g] += @as(i32, a[g * 4 + k]) * @as(i32, b[g * 4 + k]);
    }
    return out;
}

fn refPsignI8x32(y: *const [32]i8, x: *const [32]i8) [32]i8 {
    var out: [32]i8 = undefined;
    for (&out, y, x) |*o, yv, xv| {
        o.* = if (xv > 0) yv else if (xv < 0) -%yv else 0; // wraps: -(-128) = -128
    }
    return out;
}

fn refDotQ4_KQ8_K(w: *const BlockQ4_K, a: *const BlockQ8_K) f32 {
    // Mirrors quant/q4_k.zig refDotQ4_KQ8_K: deferred integer scale*acc /
    // min*bsum reduction, two f32 ops at the end (the dotQ4_KQ8_K structure).
    const d = common.f16BitsToF32(w.dm[0]) * a.d;
    const dmin = common.f16BitsToF32(w.dm[1]) * a.d;
    var iscale: i32 = 0;
    var imin: i32 = 0;
    var subblock: usize = 0;
    while (subblock < 8) : (subblock += 1) {
        const scale_min = quant.getScaleMinK4(&w.scales, subblock);
        var acc: i32 = 0;
        var i: usize = 0;
        while (i < 32) : (i += 1) {
            const byte = w.qs[(subblock / 2) * 32 + i];
            const q: i32 = if (subblock % 2 == 0) (byte & 0x0f) else (byte >> 4);
            acc += q * @as(i32, a.qs[subblock * 32 + i]);
        }
        const bsum = @as(i32, a.bsums[subblock * 2]) + @as(i32, a.bsums[subblock * 2 + 1]);
        iscale += @as(i32, scale_min.scale) * acc;
        imin += @as(i32, scale_min.min) * bsum;
    }
    return d * @as(f32, @floatFromInt(iscale)) - dmin * @as(f32, @floatFromInt(imin));
}

fn refDotQ8_0Q8_0(a: *const BlockQ8_0, b: *const BlockQ8_0) f32 {
    var acc: i32 = 0;
    for (a.qs, b.qs) |x, y| acc += @as(i32, x) * @as(i32, y);
    return @as(f32, @floatFromInt(acc)) * (common.f16BitsToF32(a.d) * common.f16BitsToF32(b.d));
}

fn refDotTQ2_0F32(wblocks: []const BlockTQ2_0, x: []const f32) f32 {
    // Order-matched scalar replica of dotTQ2_0F32 (mirrors quant/
    // ternary_tests.zig refDotF32): the same 4-lane accumulator and the same
    // lane-fold order, so the mul-free hot path must match it bit-exactly.
    var total: f32 = 0;
    for (wblocks, 0..) |*w, bi| {
        const xb = x[bi * qk_k_block_size ..][0..qk_k_block_size];
        var acc = [4]f32{ 0, 0, 0, 0 };
        for ([_]usize{ 0, 32 }) |j| {
            for (0..4) |lane| {
                var m: usize = 0;
                while (m < 32) : (m += 1) {
                    const code: u8 = (w.qs[j + m] >> @intCast(2 * lane)) & 3;
                    const xv = xb[j * 4 + lane * 32 + m];
                    const term: f32 = switch (code) {
                        0 => -xv,
                        1 => 0.0,
                        else => xv,
                    };
                    acc[m % 4] += term;
                }
            }
        }
        const lane_sum = (acc[0] + acc[1]) + (acc[2] + acc[3]);
        total += common.f16BitsToF32(w.d) * lane_sum;
    }
    return total;
}

// ---- deterministic fills ----------------------------------------------------

fn fillRandomBlockQ4_K(block: *BlockQ4_K, random: std.Random) void {
    block.dm = .{ common.f32ToF16Bits(0.25 + random.float(f32)), common.f32ToF16Bits(random.float(f32) * 0.5) };
    for (&block.scales) |*s| s.* = random.int(u8);
    for (&block.qs) |*q| q.* = random.int(u8);
}

fn fillRandomBlockQ8_K(block: *BlockQ8_K, random: std.Random, extreme: bool) void {
    block.d = 0.25 + random.float(f32);
    for (&block.qs, 0..) |*q, i| {
        if (extreme) {
            q.* = if (i % 2 == 0) 127 else -127;
        } else {
            q.* = @intCast(@as(i32, random.uintLessThan(u8, 255)) - 127); // quantizeToI8 domain
        }
    }
    for (&block.bsums, 0..) |*sum, group| {
        var acc: i32 = 0;
        for (block.qs[group * 16 ..][0..16]) |q| acc += q;
        sum.* = @intCast(acc);
    }
}

fn fillUniformF32(random: std.Random, values: []f32, scale: f32) void {
    for (values) |*v| v.* = (random.float(f32) * 2.0 - 1.0) * scale;
}

fn fillRandomBlockQ8_0(block: *BlockQ8_0, random: std.Random, allow_m128: bool) void {
    block.d = common.f32ToF16Bits(0.25 + random.float(f32));
    for (&block.qs) |*q| {
        if (allow_m128) {
            q.* = @bitCast(random.int(u8)); // full i8 range incl. -128
        } else {
            q.* = @intCast(@as(i32, random.uintLessThan(u8, 255)) - 127);
        }
    }
}

// ---- check phases -----------------------------------------------------------

fn checkPrimitives() void {
    var prng = std.Random.DefaultPrng.init(0x9e3779b97f4a7c15);
    const random = prng.random();
    var iter: usize = 0;
    while (iter < 500) : (iter += 1) {
        var au: [16]u8 = undefined;
        var ai: [16]i8 = undefined;
        var bb: [16]i8 = undefined;
        // u8·i8: nibble domain (Q4_K call site) against full i8 incl. -128.
        for (&au) |*x| x.* = random.uintLessThan(u8, 16);
        for (&bb) |*y| y.* = @bitCast(random.int(u8));
        checkI32("dotU8I8x16 nibble", refDotU8I8(&au, &bb), common.dotU8I8x16Portable(au, bb));
        // u8·i8: a <= 127 (pair sums <= 32512, saturation-free) vs full i8.
        for (&au) |*x| x.* = random.uintLessThan(u8, 128);
        checkI32("dotU8I8x16 wide", refDotU8I8(&au, &bb), common.dotU8I8x16Portable(au, bb));
        // i8·i8 sign-trick domain: a unrestricted incl. -128, b in [-127,127].
        for (&ai) |*x| x.* = @bitCast(random.int(u8));
        for (&bb) |*y| y.* = @intCast(@as(i32, random.uintLessThan(u8, 255)) - 127);
        checkI32("dotI8x16 domain", refDotI8I8(&ai, &bb), common.dotI8x16Portable(ai, bb));
        // aarch64 sdot/smmla dispatchers vs scalar reference: full i8 range on
        // both operands (exact i32, no saturation); acc bounded to i16 range so
        // the accumulate cannot overflow. On non-i8mm aarch64 (M1) smmlaI8x16
        // is its portable form — this still pins the numerics AND keeps the
        // smmla asm referenced (the neoverse_v1 compile-only build leg
        // instruction-selects it); on FEAT_I8MM hardware (Graviton3+/Grace)
        // this same check is the smmla asm's execution gate.
        var av: [16]i8 = undefined;
        var bv: [16]i8 = undefined;
        for (&av) |*x| x.* = @bitCast(random.int(u8));
        for (&bv) |*y| y.* = @bitCast(random.int(u8));
        var acc: [4]i32 = undefined;
        for (&acc) |*l| l.* = random.int(i16);
        const accv: common.QKV4i32 = acc;
        checkI32x4("sdotI8x16", refSdot(acc, &av, &bv), common.sdotI8x16(accv, av, bv));
        inline for (0..4) |lane| {
            checkI32x4("sdotI8x16Lane", refSdotLane(lane, acc, &av, &bv), common.sdotI8x16Lane(lane, accv, av, bv));
        }
        checkI32x4("smmlaI8x16", refSmmla(acc, &av, &bv), common.smmlaI8x16(accv, av, bv));
        // 256-bit grouped primitives (the Q8_0x4 packed-accumulate arms).
        var acc8: [8]i32 = undefined;
        for (&acc8) |*l| l.* = random.int(i16);
        const acc8v: common.QKV8i32 = acc8;
        // vpdpbusd: unrestricted domains — full u8 against full i8 incl. -128.
        var a32u: [32]u8 = undefined;
        var b32: [32]i8 = undefined;
        for (&a32u) |*x| x.* = random.int(u8);
        for (&b32) |*y| y.* = @bitCast(random.int(u8));
        checkI32x8("dpbusdI32x8", refGroupDotsU8I8x32(acc8, &a32u, &b32), common.dpbusdI32x8(acc8v, a32u, b32));
        // widening tier: unrestricted signed domains on both sides.
        var a32i: [32]i8 = undefined;
        for (&a32i) |*x| x.* = @bitCast(random.int(u8));
        checkI32x8("dotI8GroupsWidenI32x8", refGroupDotsI8I8x32(acc8, &a32i, &b32), common.dotI8GroupsWidenI32x8(acc8v, a32i, b32));
        // maddubs grouped dot: its saturation-free domain (a <= 128 against
        // b in [-127,127] — pair sums <= 32512; the Q8_0 sign-trick shape).
        var am32: [32]u8 = undefined;
        var bm32: [32]i8 = undefined;
        for (&am32) |*x| x.* = random.uintLessThan(u8, 129);
        for (&bm32) |*y| y.* = @intCast(@as(i32, random.uintLessThan(u8, 255)) - 127);
        checkI32x8("maddubsDotGroupsI32x8", refGroupDotsU8I8x32(acc8, &am32, &bm32), common.maddubsDotGroupsI32x8(acc8v, am32, bm32));
        // vpsignb: full range on both operands (x == 0 lanes pinned).
        var py: [32]i8 = undefined;
        var px: [32]i8 = undefined;
        for (&py) |*v| v.* = @bitCast(random.int(u8));
        for (&px) |*v| v.* = @bitCast(random.int(u8));
        px[0] = 0;
        px[17] = 0;
        checkI8x32("psignI8x32", refPsignI8x32(&py, &px), common.psignI8x32(py, px));
        // composed sign-trick recipe exactly as the AVX2 kernel arms arrange
        // it: weights (full i8 incl. -128) as sign source, activations in
        // [-127,127] as value operand — must equal the signed grouped dot.
        const w32v: common.QKV32i8 = a32i;
        const act32v: common.QKV32i8 = bm32;
        const abs_w: common.QKV32u8 = @bitCast(common.psignI8x32(w32v, w32v));
        const signed_act = common.psignI8x32(act32v, w32v);
        checkI32x8("signTrickGroupsI32x8", refGroupDotsI8I8x32(acc8, &a32i, &bm32), common.maddubsDotGroupsI32x8(acc8v, abs_w, signed_act));
    }
    // Saturation-edge stress (pair sums of exactly ±32512 / ±3840 extremes).
    const a_edge: [16]u8 = @splat(128);
    const b_hi: [16]i8 = @splat(127);
    const b_lo: [16]i8 = @splat(-127);
    checkI32("dotU8I8x16 +edge", 16 * 128 * 127, common.dotU8I8x16Portable(a_edge, b_hi));
    checkI32("dotU8I8x16 -edge", 16 * 128 * -127, common.dotU8I8x16Portable(a_edge, b_lo));
    const a_nib: [16]u8 = @splat(15);
    const b_min: [16]i8 = @splat(-128);
    checkI32("dotU8I8x16 b=-128", 16 * 15 * -128, common.dotU8I8x16Portable(a_nib, b_min));
    const a_min: [16]i8 = @splat(-128);
    checkI32("dotI8x16 a=-128 b=+127", 16 * -128 * 127, common.dotI8x16Portable(a_min, b_hi));
    checkI32("dotI8x16 a=-128 b=-127", 16 * -128 * -127, common.dotI8x16Portable(a_min, b_lo));
    var am: [16]i8 = undefined;
    var bm: [16]i8 = undefined;
    for (&am, 0..) |*x, i| x.* = if (i % 2 == 0) -128 else 127;
    for (&bm, 0..) |*y, i| y.* = if (i % 3 == 0) -127 else 127;
    checkI32("dotI8x16 mixed", refDotI8I8(&am, &bm), common.dotI8x16Portable(am, bm));
    // sdot/smmla at the i8 extremes (4·128² and 8·128² per lane, exact i32).
    const all_min: common.QKV16i8 = @splat(-128);
    const all_max: common.QKV16i8 = @splat(127);
    const acc0: common.QKV4i32 = @splat(0);
    checkI32x4("sdotI8x16 -128*-128", @splat(4 * 128 * 128), common.sdotI8x16(acc0, all_min, all_min));
    checkI32x4("sdotI8x16 -128*+127", @splat(4 * -128 * 127), common.sdotI8x16(acc0, all_min, all_max));
    checkI32x4("smmlaI8x16 -128*-128", @splat(8 * 128 * 128), common.smmlaI8x16(acc0, all_min, all_min));
    checkI32x4("smmlaI8x16 -128*+127", @splat(8 * -128 * 127), common.smmlaI8x16(acc0, all_min, all_max));
    // 256-bit grouped extremes. dpbusd at the u8·i8 magnitude ceiling, the
    // widening tier at (-128)², maddubs at its exact saturation-free maxima
    // (pair sums ±32512), psign at the -(-128) wrap.
    const acc8z: common.QKV8i32 = @splat(0);
    const a255: common.QKV32u8 = @splat(255);
    const bmin32: common.QKV32i8 = @splat(-128);
    const bmax32: common.QKV32i8 = @splat(127);
    checkI32x8("dpbusdI32x8 255*-128", @splat(4 * 255 * -128), common.dpbusdI32x8(acc8z, a255, bmin32));
    checkI32x8("dpbusdI32x8 255*+127", @splat(4 * 255 * 127), common.dpbusdI32x8(acc8z, a255, bmax32));
    checkI32x8("dotI8GroupsWiden -128*-128", @splat(4 * 128 * 128), common.dotI8GroupsWidenI32x8(acc8z, bmin32, bmin32));
    checkI32x8("dotI8GroupsWiden -128*+127", @splat(4 * -128 * 127), common.dotI8GroupsWidenI32x8(acc8z, bmin32, bmax32));
    const a128: common.QKV32u8 = @splat(128);
    const b127n: common.QKV32i8 = @splat(-127);
    checkI32x8("maddubsDotGroups 128*+127", @splat(4 * 128 * 127), common.maddubsDotGroupsI32x8(acc8z, a128, bmax32));
    checkI32x8("maddubsDotGroups 128*-127", @splat(4 * 128 * -127), common.maddubsDotGroupsI32x8(acc8z, a128, b127n));
    const wrap_expect: [32]i8 = @splat(-128);
    checkI8x32("psignI8x32 wrap", wrap_expect, common.psignI8x32(bmin32, bmin32));
    // composed sign-trick at the weight = -128 corner (|w| = 128 as u8).
    const act127: common.QKV32i8 = @splat(127);
    const abs_wmin: common.QKV32u8 = @bitCast(common.psignI8x32(bmin32, bmin32));
    const s_act = common.psignI8x32(act127, bmin32);
    checkI32x8("signTrickGroups -128*+127", @splat(4 * -128 * 127), common.maddubsDotGroupsI32x8(acc8z, abs_wmin, s_act));
    std.debug.print("primitives: done (checksum so far {x:0>16})\n", .{fnv});
}

fn checkQ4K(allocator: std.mem.Allocator) !void {
    var prng = std.Random.DefaultPrng.init(0x2545f4914f6cdd1d);
    const random = prng.random();

    const m = 3;
    const n = 5;
    const blocks_per_row = 2;
    const k = blocks_per_row * 256;

    var lhs_blocks: [m * blocks_per_row]BlockQ8_K = undefined;
    for (&lhs_blocks, 0..) |*b, idx| fillRandomBlockQ8_K(b, random, idx == 1);
    var rhs_blocks: [n * blocks_per_row]BlockQ4_K = undefined;
    for (&rhs_blocks) |*b| fillRandomBlockQ4_K(b, random);
    // stress column: all-0xFF nibbles + max scales against the ±127 lhs block
    for (&rhs_blocks[0].scales) |*s| s.* = 0xff;
    for (&rhs_blocks[0].qs) |*q| q.* = 0xff;

    var rhs = try quant.quantizedMatmulRhsQ4_KFromBlocks(allocator, k, n, &rhs_blocks);
    defer rhs.deinit();

    var out: [m * n]f32 = undefined;
    quant.matmulQ4_KRhsRange(&out, &lhs_blocks, &rhs, m, n, 0, m);

    // matmulQ4_KRhsTile has no aarch64 specialization: the per-(i,j) f32
    // accumulation order matches the replica on EVERY target → bit-exact.
    for (0..m) |i| {
        for (0..n) |j| {
            var expected: f32 = 0;
            for (0..blocks_per_row) |bi| {
                expected += refDotQ4_KQ8_K(&rhs_blocks[j * blocks_per_row + bi], &lhs_blocks[i * blocks_per_row + bi]);
            }
            checkF32Exact("q4_k matmul", expected, out[i * n + j]);
        }
    }
    std.debug.print("q4_k matmul: done (checksum so far {x:0>16})\n", .{fnv});
}

fn checkQ8_0(allocator: std.mem.Allocator) !void {
    var prng = std.Random.DefaultPrng.init(0xd1b54a32d192ed03);
    const random = prng.random();

    const m = 3;
    const n = 5;
    const blocks_per_row = 2;
    const k = blocks_per_row * 32;

    var lhs_blocks: [m * blocks_per_row]BlockQ8_0 = undefined;
    for (&lhs_blocks) |*blk| fillRandomBlockQ8_0(blk, random, false);
    const rhs_blocks = try allocator.alloc(BlockQ8_0, n * blocks_per_row);
    for (rhs_blocks) |*blk| fillRandomBlockQ8_0(blk, random, true);
    // stress column: every weight byte -128 (the sign-trick wrap corner is
    // excluded by construction — activations are the clamped side).
    for (&rhs_blocks[0].qs) |*q| q.* = -128;

    var rhs = quant.QuantizedMatmulRhsQ8_0{
        .rows = .{
            .allocator = allocator,
            .blocks = rhs_blocks,
            .rows = n,
            .cols = k,
            .blocks_per_row = blocks_per_row,
        },
        .k = k,
        .n = n,
    };
    defer rhs.deinit();

    var out: [m * n]f32 = undefined;
    quant.matmulQ8_0RhsTile(&out, &lhs_blocks, &rhs, n, 0, m, 0, n);

    for (0..m) |i| {
        for (0..n) |j| {
            var expected: f32 = 0;
            for (0..blocks_per_row) |bi| {
                expected += refDotQ8_0Q8_0(&lhs_blocks[i * blocks_per_row + bi], &rhs_blocks[j * blocks_per_row + bi]);
            }
            if (comptime builtin.cpu.arch == .aarch64) {
                // aarch64 routes to the sdot tile kernel: 4-lane f32 accumulate
                // + one final reduce — a different f32 association → tolerance.
                checkF32Rel("q8_0 matmul", expected, out[i * n + j], 1e-5);
            } else {
                // generic path (incl. the new AVX2 branch): exact integer dot,
                // identical f32 op order → bit-exact.
                checkF32Exact("q8_0 matmul", expected, out[i * n + j]);
            }
        }
    }
    std.debug.print("q8_0 matmul: done (checksum so far {x:0>16})\n", .{fnv});
}

fn checkTQ2_0(allocator: std.mem.Allocator) !void {
    var prng = std.Random.DefaultPrng.init(0x94d049bb133111eb);
    const random = prng.random();

    const m = 3;
    const n = 5; // exercises the 4-column fused block AND a tail column
    const blocks_per_row = 2;
    const k = blocks_per_row * qk_k_block_size;

    const weights = try allocator.alloc(f32, n * k);
    defer allocator.free(weights);
    fillUniformF32(random, weights, 1.5);
    const acts = try allocator.alloc(f32, m * k);
    defer allocator.free(acts);
    fillUniformF32(random, acts, 3.0);

    // Both encoders: ggml per-block-absmax rows and the BitNet b1.58
    // per-tensor absmean round-clip (uniform d across every block).
    var rhs = try quant.quantizedMatmulRhsTQ2_0FromF32(allocator, k, n, weights);
    defer rhs.deinit();
    var rhs_absmean = try quant.quantizedMatmulRhsTQ2_0FromF32Absmean(allocator, k, n, weights);
    defer rhs_absmean.deinit();

    var a = try tensor_mod.Tensor.fromSlice(allocator, &.{ m, k }, acts);
    defer a.deinit();
    const qlhs = try quant.quantizeRowsQ8_K(allocator, &a);
    defer allocator.free(qlhs);

    var got: [m * n]f32 = undefined;
    var want: [m * n]f32 = undefined;

    // Hot int8 kernel (sdot / vpdpbusd / maddubs / portable twins) vs the
    // cold ggml-parity table path: every arm accumulates the exact block
    // integer and folds f32 in the same block-major order → bit-exact.
    quant.matmulTQ2_0RhsRange(&got, qlhs, &rhs, m, n, 0, m);
    quant.matmulTableQ8_KRhsRange(.tq2_0, &want, qlhs, &rhs, m, n, 0, m);
    for (0..m * n) |i| checkF32Exact("tq2_0 matmul", want[i], got[i]);

    quant.matmulTQ2_0RhsRange(&got, qlhs, &rhs_absmean, m, n, 0, m);
    quant.matmulTableQ8_KRhsRange(.tq2_0, &want, qlhs, &rhs_absmean, m, n, 0, m);
    for (0..m * n) |i| checkF32Exact("tq2_0 absmean matmul", want[i], got[i]);

    // Mul-free f32 path ((x XOR s) AND m, fixed 4-lane fold) vs the
    // order-matched scalar replica: bitwise reproducible on every ISA.
    for (0..m) |r| {
        const xrow = acts[r * k ..][0..k];
        for (0..n) |c| {
            checkF32Exact("tq2_0 f32 dot", refDotTQ2_0F32(rhs.columnBlocks(c), xrow), quant.dotTQ2_0F32(rhs.columnBlocks(c), xrow));
            checkF32Exact("tq2_0 f32 dot absmean", refDotTQ2_0F32(rhs_absmean.columnBlocks(c), xrow), quant.dotTQ2_0F32(rhs_absmean.columnBlocks(c), xrow));
        }
    }
    std.debug.print("tq2_0 matmul: done (checksum so far {x:0>16})\n", .{fnv});
}

fn checkTQ2_0X4(allocator: std.mem.Allocator) !void {
    var prng = std.Random.DefaultPrng.init(0x2545f4914f6cdd1d);
    const random = prng.random();

    const m = 3;
    const n = 8; // two 4-column groups; the pack refuses n % 4 != 0
    const blocks_per_row = 3; // odd block count exercises group indexing
    const k = blocks_per_row * qk_k_block_size;

    const weights = try allocator.alloc(f32, n * k);
    defer allocator.free(weights);
    fillUniformF32(random, weights, 1.5);
    const acts = try allocator.alloc(f32, m * k);
    defer allocator.free(acts);
    fillUniformF32(random, acts, 3.0);

    var rhs = try quant.quantizedMatmulRhsTQ2_0FromF32(allocator, k, n, weights);
    defer rhs.deinit();
    const packed_groups = try quant.packMatmulRhsTQ2_0x4(allocator, &rhs);
    defer allocator.free(packed_groups);

    var a = try tensor_mod.Tensor.fromSlice(allocator, &.{ m, k }, acts);
    defer a.deinit();
    const qlhs = try quant.quantizeRowsQ8_K(allocator, &a);
    defer allocator.free(qlhs);

    var got: [m * n]f32 = undefined;
    var want: [m * n]f32 = undefined;

    // x4 column-interleaved kernel (by-element sdot / ymm-granule
    // vpdpbusd-maddubs / portable twins) vs the row kernel: identical exact
    // block integers in every lane arrangement, identical per-column f32
    // sequence → bit-exact on every ISA, and bit-exact to the cold table
    // path transitively (checked directly too).
    quant.matmulTQ2_0X4RhsRange(&got, qlhs, packed_groups, blocks_per_row, n, 0, m);
    quant.matmulTQ2_0RhsRange(&want, qlhs, &rhs, m, n, 0, m);
    for (0..m * n) |i| checkF32Exact("tq2_0x4 vs row", want[i], got[i]);
    quant.matmulTableQ8_KRhsRange(.tq2_0, &want, qlhs, &rhs, m, n, 0, m);
    for (0..m * n) |i| checkF32Exact("tq2_0x4 vs cold", want[i], got[i]);

    // The accumulating twin equals materialize-then-add exactly.
    var acc: [m * n]f32 = undefined;
    quant.matmulTQ2_0X4RhsRange(&acc, qlhs, packed_groups, blocks_per_row, n, 0, m);
    quant.matmulTQ2_0X4RhsTileAcc(&acc, qlhs, packed_groups, blocks_per_row, n, 0, m, 0, n);
    for (0..m * n) |i| checkF32Exact("tq2_0x4 acc", want[i] + want[i], acc[i]);

    std.debug.print("tq2_0x4 matmul: done (checksum so far {x:0>16})\n", .{fnv});
}

fn checkTQ2_0Folded(allocator: std.mem.Allocator) !void {
    var prng = std.Random.DefaultPrng.init(0x853c49e6748fea9b);
    const random = prng.random();

    const m = 2;
    const n = 8;
    const blocks_per_row = 2;
    const k = blocks_per_row * qk_k_block_size;

    const w1 = try allocator.alloc(f32, n * k);
    defer allocator.free(w1);
    fillUniformF32(random, w1, 1.5);
    const w2 = try allocator.alloc(f32, n * k);
    defer allocator.free(w2);
    fillUniformF32(random, w2, 0.5);
    const acts = try allocator.alloc(f32, m * k);
    defer allocator.free(acts);
    fillUniformF32(random, acts, 3.0);

    var rhs1 = try quant.quantizedMatmulRhsTQ2_0FromF32(allocator, k, n, w1);
    defer rhs1.deinit();
    var rhs2 = try quant.quantizedMatmulRhsTQ2_0FromF32(allocator, k, n, w2);
    defer rhs2.deinit();
    const folded = try quant.packMatmulRhsTQ2_0Foldedx4(allocator, &rhs1, &rhs2);
    defer allocator.free(folded);

    var a = try tensor_mod.Tensor.fromSlice(allocator, &.{ m, k }, acts);
    defer a.deinit();
    const qlhs = try quant.quantizeRowsQ8_K(allocator, &a);
    defer allocator.free(qlhs);

    var got: [m * n]f32 = undefined;
    quant.matmulTQ2_0FoldedX4RhsRange(&got, qlhs, folded, blocks_per_row, n, 0, m);

    // Order-matched scalar reference of the folded semantics: per column,
    // per block, sums += s * a.d * (sum((3*u1+u2)*a) - 4*sum(a)). Bit-exact
    // on every arm (exact block integers, one f32 sequence).
    for (0..m) |r| {
        for (0..n) |c| {
            var sum: f32 = 0;
            for (0..blocks_per_row) |bi| {
                const b1 = &rhs1.columnBlocks(c)[bi];
                const b2 = &rhs2.columnBlocks(c)[bi];
                const ab = &qlhs[r * blocks_per_row + bi];
                var dot: i32 = 0;
                var asum: i32 = 0;
                for (0..qk_k_block_size) |e| {
                    const byte = (e / 128) * 32 + (e % 32);
                    const shift: u3 = @intCast(2 * ((e % 128) / 32));
                    const u1c: i32 = (b1.qs[byte] >> shift) & 3;
                    const u2c: i32 = (b2.qs[byte] >> shift) & 3;
                    const av: i32 = ab.qs[e];
                    dot += (3 * u1c + u2c) * av;
                    asum += av;
                }
                const s = common.f16BitsToF32(b2.d);
                sum += s * ab.d * @as(f32, @floatFromInt(dot - 4 * asum));
            }
            checkF32Exact("tq2_0 folded matmul", sum, got[r * n + c]);
        }
    }
    std.debug.print("tq2_0 folded matmul: done (checksum so far {x:0>16})\n", .{fnv});
}

pub fn main(init: std.process.Init) !void {
    _ = init;
    const allocator = std.heap.page_allocator;

    std.debug.print("x86dot-check arch={s} avx2={} avxvnni={} avx512vnni={}\n", .{
        @tagName(builtin.cpu.arch), common.has_x86_avx2, common.has_x86_avxvnni, common.has_x86_avx512vnni,
    });

    checkPrimitives();
    try checkQ4K(allocator);
    try checkQ8_0(allocator);
    try checkTQ2_0(allocator);
    try checkTQ2_0X4(allocator);
    try checkTQ2_0Folded(allocator);

    if (failures != 0) {
        std.debug.print("x86dot-check FAIL ({d} mismatches)\n", .{failures});
        std.process.exit(1);
    }
    std.debug.print("x86dot-check PASS checksum={x:0>16}\n", .{fnv});
}
