//! The one ISA leaf: every comptime CPU-feature fact the kernels select on
//! is resolved here, once. `quant/*` and `vector/*` never read
//! `builtin.cpu` themselves; the quantized kernels switch on `tier` (the
//! int8-dot ladder a build is compiled against) and the primitives and the
//! dense GEMM shapes read the capability booleans. Every `switch (tier)` is
//! exhaustive, so adding a tier is a compile error at each arm site.
//!
//! Resolution is comptime and target-driven: with no `-Dtarget` the build
//! targets the compiling machine's exact CPU, a bare `-Dtarget` drops to the
//! architecture baseline (portable on x86; sdot on aarch64, see below), and
//! `-Dcpu` pins a model. Nothing here is a runtime dispatch.
const std = @import("std");
const builtin = @import("builtin");

pub const is_aarch64 = builtin.cpu.arch == .aarch64;
pub const is_x86_64 = builtin.cpu.arch == .x86_64;

/// NEON with FEAT_DotProd (`sdot`): the aarch64 baseline this repo assumes.
/// Every aarch64 arm emits `sdot` ungated (LLVM's asm parser does not
/// feature-check inline asm), so pre-v8.2 cores without DotProd are outside
/// the supported set; there is no portable aarch64 fallback.
pub const has_neon = is_aarch64;

/// SMMLA is a separate AArch64 feature from SDOT; Apple M1-class CPUs have
/// FEAT_DotProd but not FEAT_I8MM, so the MMLA arms stay gated on it.
pub const has_aarch64_i8mm = is_aarch64 and std.Target.aarch64.featureSetHas(builtin.cpu.features, .i8mm);

// x86 int8-GEMM feature gates (mirror has_aarch64_i8mm). AVX-512-VNNI provides the
// vpdpbusd byte dot-product (the analog of NEON sdot); AVX2 falls back to the
// vpmaddwd i16 path.
pub const has_x86_avx2 = is_x86_64 and std.Target.x86.featureSetHas(builtin.cpu.features, .avx2);
pub const has_x86_avx512vnni = is_x86_64 and std.Target.x86.featureSetHas(builtin.cpu.features, .avx512vnni);
pub const has_x86_avx512vl = is_x86_64 and std.Target.x86.featureSetHas(builtin.cpu.features, .avx512vl);

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
pub const has_x86_avxvnni = is_x86_64 and std.Target.x86.featureSetHas(builtin.cpu.features, .avxvnni);

/// Either VNNI flavor at xmm width: the 16-lane dot forms lower to vpdpbusd
/// (EVEX under avx512vnni, VEX under avxvnni).
pub const has_x86_vnni = has_x86_avx512vnni or has_x86_avxvnni;

/// vpdpbusd-on-ymm gate: the VEX encoding needs AVX-VNNI; the EVEX.256
/// encoding needs AVX512-VNNI *and* AVX512-VL (every shipped VNNI core —
/// Ice Lake+, Zen4+ — has VL, but the assembler needs the gate to legalize
/// an encoding, so keep it explicit rather than implied).
pub const has_x86_vnni_ymm = has_x86_avxvnni or (has_x86_avx512vnni and has_x86_avx512vl);

/// The self-hosted x86_64 backend (the Debug-mode default on x86_64-linux) has
/// its own assembler that lacks the newer VEX mnemonics (vpdpbusd rejects with
/// "invalid mnemonic"); the x86 asm arms are therefore additionally gated on
/// the LLVM backend — Debug builds execute the exact portable twins instead,
/// ReleaseSafe/ReleaseFast (LLVM) execute the real instructions.
pub const has_llvm_asm = builtin.zig_backend == .stage2_llvm;

/// The int8-dot ladder the quantized kernels are built on, one arm per
/// build. The NEON tiers dot with `sdot` lanes (`neon_i8mm` additionally
/// selects the `smmla` packs); the x86 tiers are the grouped u8·i8 ymm dots
/// (`x86_vnni` = vpdpbusd, `x86_avx2` = vpmaddubsw+vpmaddwd); `portable` is
/// the universal exact widening `@Vector` form every other target takes.
pub const Tier = enum {
    neon_i8mm,
    neon_sdot,
    x86_vnni,
    x86_avx2,
    portable,
};

pub const tier: Tier = if (has_aarch64_i8mm)
    .neon_i8mm
else if (has_neon)
    .neon_sdot
else if (has_x86_vnni_ymm)
    .x86_vnni
else if (has_x86_avx2)
    .x86_avx2
else
    .portable;

test "tier agrees with the capability booleans" {
    switch (tier) {
        .neon_i8mm => try std.testing.expect(has_neon and has_aarch64_i8mm),
        .neon_sdot => try std.testing.expect(has_neon and !has_aarch64_i8mm),
        .x86_vnni => try std.testing.expect(!has_neon and has_x86_vnni_ymm),
        .x86_avx2 => try std.testing.expect(!has_neon and !has_x86_vnni_ymm and has_x86_avx2),
        .portable => try std.testing.expect(!has_neon and !has_x86_avx2),
    }
}
