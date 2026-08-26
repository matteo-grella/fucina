//! Runtime tuning policy: one typed table (`Table`) of the env-tunable
//! route gates and numeric crossovers, the reflective env loader that fills
//! it, and the per-context overrides an `ExecContext` can carry.
//!
//! Scope, stated exactly. Every RUNTIME `FUCINA_*` policy value lives in
//! the table. The exceptions are substrate and diagnostics that cannot or
//! should not ride it: `FUCINA_MAX_THREADS` (read by `parallel.zig` before
//! any table exists; it sizes the substrate the loader itself runs on),
//! `FUCINA_GPU_KERNELS` (string-valued; the CUDA provider documents its
//! direct read), `FUCINA_GPU_DEBUG` (read inside the Objective-C Metal
//! shim, which has no Zig), and the test-run switches
//! (`FUCINA_TEST_REQUIRE_MODELS`, per-suite bench flags) plus the
//! model-band `FUCINA_MM_PROFILE` diagnostic, documented at their read
//! sites. The COMPTIME CPU crossovers (work thresholds burned into kernel
//! dispatch at build time) are `parallel.zig`'s constants: the two files
//! split one policy surface on binding time, runtime here, comptime there.
//!
//! Each leaf field derives its environment variable from its path: `FUCINA_`
//! plus the path segments upper-cased and joined with `_` (`decode_compact`
//! reads `FUCINA_DECODE_COMPACT`, `gpu.min_work.attn` reads
//! `FUCINA_GPU_MIN_WORK_ATTN`). A leaf named `base` or `enabled` names its
//! group itself (`gpu.min_work.base` reads `FUCINA_GPU_MIN_WORK`,
//! `gpu.enabled` reads `FUCINA_GPU`). Booleans follow the getenv-family
//! truthiness contract with an explicit off spelling: a set, non-empty value
//! whose first character is `0` forces the route off, any other set,
//! non-empty value forces it on, unset (or empty) keeps the measured
//! default. Integers parse base-10; an unparsable value keeps the default.
//! Top-level crossovers treat `0` as unset; leaves under `gpu` accept `0`
//! (a zero work floor always offloads, a zero fill gate is occupancy-blind).
//! All reads go through `parallel.envFlagValue` / `envPositiveUsize` /
//! `envNonNegativeUsize`, the only sanctioned env readers (libc-free Linux
//! builds have no `std.c.getenv`).
//!
//! The table is read once and cached at the first `get()`; changing the
//! process environment afterwards has no effect. `set`/`setField` are the
//! programmatic override path (test hooks and CLI flags): they pin fields
//! over the env-loaded base, and pinning `null` re-arms the env/default
//! value. Setters are for startup and tests; calling them while another
//! thread dispatches kernels is outside the contract.
//!
//! Process-wide values answer "which kernel implements this op here", a
//! per-machine fact. Policy that can differ per workload (two
//! `ExecContext`s in one process wanting different route choices) goes
//! through `Overrides`, carried by the context and consulted via `resolve`
//! by the routes that support per-context policy; process-wide gates ignore
//! it by design.

const std = @import("std");
const build_options = @import("build_options");
const parallel = @import("parallel.zig");

/// Comptime default selector for GPU crossovers the two providers measured
/// differently. The active provider is a build-time fact (`-Dgpu`), so the
/// table carries the active provider's numbers; `-Dgpu=none` builds never
/// read them.
fn perGpu(comptime cuda_value: u64, comptime metal_value: u64) u64 {
    return if (build_options.gpu_kind == .cuda) cuda_value else metal_value;
}

/// How a group's integer leaves parse their env values.
pub const EnvIntParse = enum {
    /// Base-10 positive integer; `0`, empty, or unparsable keeps the default.
    positive,
    /// Base-10 non-negative integer; `0` is a meaningful value, empty or
    /// unparsable (including a sign prefix) keeps the default.
    non_negative,
};

/// The entire runtime tuning surface as one struct. Field defaults are the
/// measured route defaults; provenance for the GPU crossovers lives on the
/// field docs below and in docs/reference/02-toolchain-build-and-project-wiring.md section 2.6.
pub const Table = struct {
    /// Winograd conv2d route. Default follows the GEMM provider: on for
    /// `-Dblas=none` builds (the pure-Zig GEMM gains ~1.7-1.9x from the
    /// 2.25x MAC cut and the dropped col-matrix traffic; measured
    /// i9-13950HX), off when a platform BLAS backs the matmul (Accelerate's
    /// AMX makes one big im2col GEMM faster than 16 tile-element GEMMs;
    /// measured M1 Max).
    winograd: bool = !build_options.use_blas,
    /// F(4x4,3x3) tier for Winograd-routed large maps; off pins them to F2.
    winograd_f4: bool = true,
    /// Fused normalize+quantize+packed-GEMM route of `linearSeqNormed`
    /// (prefill shapes on the packed CPU arms only; the fused route matches
    /// the unfused `rmsNormMul` + linear pair to f32 roundoff, not bitwise).
    norm_quant_fused: bool = true,
    /// Decode-route gate shared by the K-quant packed weights: at decode
    /// shapes (m < 4) the dense quantized matmul is a DRAM-bound GEMV, and
    /// the byte-expanded packed layouts stream more weight bytes than the
    /// GGUF-native compact blocks already resident in the weight tensor
    /// (per 256-weight block per column: Q4_K 276 B packed vs 144 B
    /// compact, 1.92x; Q5_K 276 B vs 176 B, 1.57x; Q6_K 274 B vs 210 B,
    /// 1.30x). Routing decode through the compact tensor-RHS path is
    /// bitwise-equal to the packed family (same Q8_K LHS quantization, same
    /// order-independent i32 integer stage, same f32 epilogue association;
    /// proven by the cross-layout tests in q4_k/q5_k/q6_k_tests.zig) and
    /// wins where bandwidth is the limit.
    decode_compact: bool = true,
    /// conv2d backward GEMM route: the groups == 1 backward entries
    /// decompose into matmul dispatch + im2col/col2im (the forward's
    /// adjoint), which is both GEMM-fast and pool-parallel; off pins both
    /// entries to the direct gather kernels.
    conv_bwd_gemm: bool = true,
    /// BLAS-strip attention backward on BLAS builds (the same
    /// kernel-selection contract as `shouldUseBlas` on the plain matmuls);
    /// off reverts to the register-tiled route, the escape hatch for
    /// non-BLAS parity work. Only consulted on BLAS-backed native builds.
    attn_bwd_blas: bool = true,
    /// Forward-saved stats route for the backward softmax reconstruction;
    /// off pins the 3-pass recompute. Only consulted when the caller has
    /// stats at all (the autograd record); the stats-less exec path always
    /// recomputes.
    attn_bwd_stats: bool = true,
    /// Fused distill route in cartridge training; off forces the composed
    /// logits + `cartridge.distillLoss` tail (the fused route matches it to
    /// f32 roundoff, not bitwise).
    fused_distill: bool = true,
    /// CPU f32 weight-shadow route (CPU builds only): attaches a widen-once
    /// f32 shadow to a 16-bit weight's storage and routes
    /// m >= `cpu_f32_shadow_min_m` GEMMs through the BLAS f32 path. Opt-in:
    /// +4 bytes/weight resident, and 16-bit training should leave it off
    /// (the streaming kernels read the live bytes).
    cpu_f32_shadow: bool = false,
    /// Folded one-pass serving of tie-fitted PTQTP MoE plane sets; off
    /// serves them through the per-plane path (A/B on one binary). The
    /// expert-store L2 tier stripes primary-file plane bytes and skips
    /// folded slab sections entirely, so a striped tier only covers
    /// unfolded layers; disabling the fold trades the halved cache-hit dot
    /// for L2 coverage. Applied to both the resident and streamed MoE
    /// loaders so the two tiers keep serving the same numbers.
    ptqtp_fold: bool = true,
    /// Pure-LRU victim scan in the MoE expert store (A/B on one binary).
    moe_lru: bool = false,
    /// Keeps the expert-store L2 tier page-cached instead of uncached I/O.
    moe_l2_cached: bool = false,

    /// CPU f32 weight-shadow crossover: rows m >= this take the BLAS route
    /// over the widened-once shadow (measured bench-f16gemm, M1 Max +
    /// Accelerate).
    cpu_f32_shadow_min_m: u64 = 32,
    /// Minimum output spatial size for the Winograd F4 tier (bench-tuned,
    /// i9-13950HX sweep: F4 pays where the tile count keeps its 36 GEMMs
    /// well-shaped and the transform cost is amortized).
    winograd_f4_min: u64 = 14,
    /// Maximum input channels for the F4 tier: 56 keeps SCRFD-class
    /// shallow-channel maps on F4 and pushes ArcFace-class deep-channel
    /// stacks (cin >= 64) to F2, the measured crossover (F4 on the
    /// 112 x 112 x 64 stage costs ~30% on the recognizer; F4 on the
    /// 28/56-channel detector maps saves ~22% on detect).
    winograd_f4_maxcin: u64 = 56,
    /// Worker-team spin-then-park window override (`src/thread.zig`
    /// BarrierPool). Unlike the positive-integer knobs, `0` is a valid
    /// value here: park immediately, never spin, the manual escape for
    /// oversubscribed teams (and what the guard in `BarrierPool.init`
    /// defaults to when the team exceeds the physical-core count). The
    /// default is left alone because sweeps (M1 Max; i9-13950HX,
    /// 2026-07-03) found the response U-shaped and workload-coupled; no
    /// single value wins every workload. See the `spin_budget` comment in
    /// thread.zig for when (and on which hardware) overriding pays.
    spin_budget: ?u64 = null,
    /// Worker-team profiling: per-worker dispatch/park counters, read at
    /// pool init (`src/thread.zig`). Diagnostic, not a route gate.
    pool_profile: bool = false,

    gpu: Gpu = .{},

    /// GPU offload policy, read by the active `-Dgpu` provider at its
    /// one-time configuration. Fields only one provider consults are
    /// ignored by the other; docs/reference/02-toolchain-build-and-project-wiring.md section 2.6 carries the
    /// per-provider annotations.
    pub const Gpu = struct {
        /// Integer leaves under `gpu` accept 0: a zero work floor always
        /// offloads, a zero fill gate is occupancy-blind, a zero VRAM
        /// budget disables the bound.
        pub const env_int_parse: EnvIntParse = .non_negative;

        /// Kill switch: `FUCINA_GPU=0` disables the GPU provider entirely.
        enabled: bool = true,
        /// CUDA only: opt-in quantized decode for m <= 8 and resident
        /// weights (GEMV generally; Q5_K uses tiled MMA at m = 4..8).
        /// Default off because CPU/GPU quant arithmetic is
        /// tolerance-equivalent, not bit-identical.
        decode: bool = false,
        /// CUDA only: opts f32 GEMMs into TF32 tensor cores (default is
        /// strict FP32).
        tf32: bool = false,
        /// Dispatch tracing; dump via the provider's `traceDump` (no-op
        /// when off).
        trace: bool = false,
        /// CUDA only: tensor-core quantized prefill kernels; off selects
        /// the scalar-FFMA fallback for parity/performance diagnosis.
        quant_mma: bool = true,
        /// CUDA only: split substantially underfilled quantized prefill
        /// along K, then reduce on-stream; off keeps one block/output tile.
        quant_split_k: bool = true,
        /// CUDA only: weight-residency budget in bytes; `0` disables the
        /// bound, unset selects ~80% of free VRAM at init.
        vram_budget: ?u64 = null,
        /// Minimum grouped-MoE tile occupancy (percent of the 32-row
        /// token-tile slots that carry real rows) before the GPU arm
        /// engages. Per-tile GPU cost is fill-independent (weight dequant
        /// dominates; ~45-53 us/tile at 12% and at 100% fill, measured
        /// 2026-07-03), so below ~50% occupancy the raw CPU path wins.
        /// `0` is occupancy-blind, above 100 the grouped GPU path never
        /// engages.
        qmoe_min_fill: u64 = 50,
        /// CUDA only: row floor applied alongside the transient work floor
        /// (`min_work.transient`) — the measured break-even vs the CPU
        /// blocked kernel is m of roughly 35-40 at LLM widths, and the
        /// default keeps a ~3x safety margin.
        transient_min_m: u64 = 128,
        min_work: MinWork = .{},

        /// Offload work floors in m*n*k units (attention: q*kv*heads*d).
        /// Defaults are the active provider's measured crossovers; the
        /// measurement notes live in docs/reference/02-toolchain-build-and-project-wiring.md section 2.6 and the
        /// provider modules.
        pub const MinWork = struct {
            /// Base f32 GEMM gate. Metal 2^32 (cold single-op crossover vs
            /// AMX, M1 Max); CUDA 2^30 (the transient floor below dominates
            /// ordinary host RHS).
            base: u64 = perGpu(1 << 30, 1 << 32),
            /// CUDA: dense f32 GEMM with an already device-resident RHS.
            /// Against OpenBLAS-32 on the reference RTX 5000 Ada host,
            /// 256^3 loses, 512^3 is a narrow GPU win, and 640^3 up is
            /// decisive; 2^27 is the first competitive tier.
            resident: u64 = 1 << 27,
            /// f16 GEMM gate: the CPU f16 competitor has no AMX-class arm
            /// and f16 operands halve the transfer bytes, so offload pays
            /// off early.
            f16: u64 = 1 << 27,
            /// CUDA: f16 gate when the RHS already has a device address;
            /// permits small-m decode without admitting a streamed weight
            /// (1x4096x1024 runs 4.2x faster than the 32-thread CPU kernel
            /// on the reference Ada GPU; 2^20 keeps smaller launch-bound
            /// dots on CPU).
            f16_resident: u64 = 1 << 20,
            /// Metal: f16/bf16 GEMM gate when the RHS is already
            /// Metal-mapped; admits batched decode at m >= 16 and the
            /// lm-head row while narrow decode stays on the CPU streaming
            /// kernels (measured Qwen3-1.7B-BF16, M1 Max).
            @"16bit_resident": u64 = 1 << 27,
            /// Resident dense-f32 GEMV/small-m GEMM gate (m <= 8): no RHS
            /// transfer, only a small activation/output crossing. The
            /// reference Ada beats OpenBLAS-32 by 12x at the 4096-wide
            /// GEMV; M1 Max crosses Accelerate around 8 Mi work; 16 Mi
            /// keeps launch/copy overhead amortized.
            gemv: u64 = 1 << 24,
            /// Attention-forward offload floor. Metal 2^29; CUDA 2^28
            /// (the ~0.5-1 ms round trip amortizes once the score work is a
            /// few hundred MFLOP; decode never reaches this seam).
            attn: u64 = perGpu(1 << 28, 1 << 29),
            /// Grouped quantized MoE GEMM gate (total m*n*k across both
            /// projections of a layer). The CPU competitor is the
            /// gather/quantize/small-m packed-kernel path (~10 GF/s
            /// effective, per-call overhead dominated), so the GPU pays
            /// off well below the f32 crossover; 2^30 keeps the two
            /// ~0.5 ms round trips plus the gather/geglu/scatter CPU
            /// phases amortized.
            qmoe: u64 = 1 << 30,
            /// CUDA: work floor for non-resident operands. Without
            /// residency every RHS streams over PCIe (measured 10.6 GB/s
            /// pageable on the reference rig, ~9.5 ms per ffn-sized f32
            /// matrix); the measured break-even vs the CPU blocked kernel
            /// is m of roughly 35-40 at LLM widths and the default keeps a
            /// ~3x safety margin. An m >= 128 row floor applies alongside
            /// it.
            transient: u64 = 1 << 33,
            /// CUDA: Q5_K-only decode gate after `gpu.decode`, against the
            /// compact (not x8-packed) CPU route: 4096x4096 loses
            /// narrowly, while 6144x4096 and m >= 2 at 4096 square win.
            decode_q5: u64 = 3 << 23,
            /// Dense Q4_K gate against the load-time-packed CPU fallback.
            dense_q4: u64 = perGpu(1 << 27, 1 << 30),
            /// CUDA: dense Q5_K gate against the load-time-packed CPU
            /// fallback (Q5_K wins at the smallest admitted 32x1024x512
            /// shape on the reference Ada host).
            dense_q5: u64 = 1 << 24,
            /// Dense Q6_K gate for the compact/raw tier (dense Q6_K linears
            /// win after warmup far below the MoE gate). An explicit value
            /// also re-seeds the packed tier (`dense_q6_packed`) unless
            /// that leaf is itself explicit; the rule is `gpuQ6Seeding`.
            dense_q6: u64 = 1 << 22,
            /// Packed-tier Q6_K crossover against the load-time-packed CPU
            /// fallback. Metal: paired eager measurements on M1 Max put
            /// the conservative packed-CPU crossovers at 2^30 (Q4_K), 2^31
            /// (Q6_K), 2^29 (Q8_0). CUDA: Q5_K wins at the smallest
            /// admitted 32x1024x512 shape on the reference Ada host
            /// (47.7 vs 63.3 us), so 2^24 keeps the measured boundary
            /// while rejecting lower-work calls.
            dense_q6_packed: u64 = perGpu(1 << 24, 1 << 31),
            /// Dense Q8_0 gate against the load-time-packed CPU fallback.
            dense_q8: u64 = perGpu(1 << 24, 1 << 29),
            /// Metal: dense/PTQTP ternary TQ2_0 gate against the x4
            /// interleaved CPU kernels.
            dense_tq2: u64 = 1 << 25,
        };
    };
};

/// Per-context tuning overrides (`ExecContext.setTuning`): the same field
/// tree as `Table` with every leaf optional. A leaf left null follows the
/// process-wide value (env, default, or programmatic pin). Consulted via
/// `resolve` by the routes that support per-context policy (first consumer:
/// the CPU f32 weight-shadow route); process-wide gates ignore it by design.
pub const Overrides = OptionalShadow(Table);

/// The `Table` field tree with every leaf `?T = null`; also the shape of
/// the process-wide programmatic pins.
pub fn OptionalShadow(comptime T: type) type {
    const src_fields = @typeInfo(T).@"struct".fields;
    var names: [src_fields.len][]const u8 = undefined;
    var types: [src_fields.len]type = undefined;
    var attrs: [src_fields.len]std.builtin.Type.StructField.Attributes = undefined;
    for (src_fields, &names, &types, &attrs) |f, *name, *FieldType, *attr| {
        name.* = f.name;
        switch (@typeInfo(f.type)) {
            .@"struct" => {
                const Shadow = OptionalShadow(f.type);
                const default: Shadow = .{};
                FieldType.* = Shadow;
                attr.* = .{ .default_value_ptr = &default };
            },
            else => {
                const default: ?f.type = null;
                FieldType.* = ?f.type;
                attr.* = .{ .default_value_ptr = &default };
            },
        }
    }
    return @Struct(.auto, null, &names, &types, &attrs);
}

/// The env variable a table field reads, derived from its dotted path
/// (`envNameOf("gpu.min_work.base")` is "FUCINA_GPU_MIN_WORK").
pub fn envNameOf(comptime path: []const u8) [:0]const u8 {
    return comptime blk: {
        var name: [:0]const u8 = "FUCINA";
        var it = std.mem.tokenizeScalar(u8, path, '.');
        while (it.next()) |seg| name = appendSegment(name, seg);
        break :blk name;
    };
}

fn appendSegment(comptime prefix: [:0]const u8, comptime segment: []const u8) [:0]const u8 {
    return comptime blk: {
        if (std.mem.eql(u8, segment, "base") or std.mem.eql(u8, segment, "enabled")) break :blk prefix;
        var upper: [segment.len]u8 = undefined;
        for (segment, 0..) |c, i| upper[i] = std.ascii.toUpper(c);
        const frozen = upper;
        break :blk prefix ++ "_" ++ frozen;
    };
}

/// One uncached read of the whole tuning surface from the environment.
/// `get` is the cached accessor; call this directly only where a fresh
/// read is the point (tests).
pub fn load() Table {
    var t: Table = .{};
    overlay(Table, &t, loadEnv());
    return t;
}

/// The env-supplied leaves alone, as an optional shadow: non-null exactly
/// where an environment variable provided a value. Keeping this separate
/// from the merged `Table` is what lets `wasSet` distinguish "explicitly
/// configured" from "measured default" after the values merge.
fn loadEnv() Overrides {
    var e: Overrides = .{};
    loadGroup(Table, &e, "FUCINA", .positive);
    return e;
}

fn loadGroup(
    comptime T: type,
    out: *OptionalShadow(T),
    comptime prefix: [:0]const u8,
    comptime mode: EnvIntParse,
) void {
    inline for (@typeInfo(T).@"struct".fields) |f| {
        const name = comptime appendSegment(prefix, f.name);
        switch (@typeInfo(f.type)) {
            .@"struct" => {
                const child_mode = comptime if (@hasDecl(f.type, "env_int_parse")) f.type.env_int_parse else mode;
                loadGroup(f.type, &@field(out, f.name), name, child_mode);
            },
            .bool => if (parallel.envFlagValue(name)) |v| {
                @field(out, f.name) = v;
            },
            .int => if (readInt(name, mode)) |v| {
                @field(out, f.name) = v;
            },
            .optional => if (parallel.envNonNegativeUsize(name)) |v| {
                @field(out, f.name) = v;
            },
            else => @compileError("unsupported tuning leaf type: " ++ @typeName(f.type)),
        }
    }
}

fn readInt(comptime name: [:0]const u8, comptime mode: EnvIntParse) ?u64 {
    return switch (mode) {
        .positive => parallel.envPositiveUsize(name),
        .non_negative => parallel.envNonNegativeUsize(name),
    };
}

var base: Table = undefined;
var env_values: Overrides = .{};
var pins: Overrides = .{};
var current: Table = undefined;
var loaded = std.atomic.Value(bool).init(false);
// std.Io.Threaded's mutex, the same primitive `thread.Mutex` wraps
// (tuning sits below thread.zig, which imports this module for the
// spin-budget leaf).
var mutex: std.Io.Mutex = .init;

fn lockTable() void {
    std.Io.Threaded.mutexLock(&mutex);
}

fn unlockTable() void {
    std.Io.Threaded.mutexUnlock(&mutex);
}

fn ensureLoaded() void {
    if (loaded.load(.acquire)) return;
    lockTable();
    defer unlockTable();
    if (!loaded.load(.monotonic)) {
        env_values = loadEnv();
        base = .{};
        overlay(Table, &base, env_values);
        current = base;
        loaded.store(true, .release);
    }
}

/// The process-wide tuning table: env values over the measured defaults,
/// with any programmatic pins applied. Read once and cached; the returned
/// pointer stays valid for the process lifetime.
pub fn get() *const Table {
    ensureLoaded();
    return &current;
}

/// Replaces the entire pin set: non-null leaves of `patch` override the
/// env-loaded base, null leaves follow it. `set(.{})` clears every pin.
pub fn set(patch: Overrides) void {
    ensureLoaded();
    lockTable();
    defer unlockTable();
    pins = patch;
    recomputeLocked();
}

/// Pins one field over the env-loaded base, or re-arms the env/default
/// value with null: `setField("decode_compact", false)`,
/// `setField("decode_compact", null)`. The per-route `set*` hooks forward
/// here.
pub fn setField(comptime path: []const u8, value: anytype) void {
    ensureLoaded();
    lockTable();
    defer unlockTable();
    leafPtr(&pins, path).* = value;
    recomputeLocked();
}

fn recomputeLocked() void {
    current = base;
    overlay(Table, &current, pins);
}

fn overlay(comptime T: type, out: *T, patch: OptionalShadow(T)) void {
    inline for (@typeInfo(T).@"struct".fields) |f| {
        switch (@typeInfo(f.type)) {
            .@"struct" => overlay(f.type, &@field(out, f.name), @field(patch, f.name)),
            else => if (@field(patch, f.name)) |v| {
                @field(out, f.name) = v;
            },
        }
    }
}

/// The type of the `Table` leaf at a dotted path.
pub fn Leaf(comptime path: []const u8) type {
    return LeafIn(Table, path);
}

/// Per-context read: the context override when set, else the process-wide
/// value. Routes that support per-context policy consult this instead of
/// `get()`.
pub fn resolve(overrides: *const Overrides, comptime path: []const u8) Leaf(path) {
    if (leafGet(Overrides, overrides.*, path)) |v| return v;
    return leafGet(Table, get().*, path);
}

/// True when the field at `path` carries an EXPLICIT value right now: an
/// environment variable supplied it at load, or a programmatic pin is
/// active. The measured default does not count. This is the query behind
/// seeding rules that depend on "did the user say anything", so consumers
/// never re-read an env variable by hard-coded name.
pub fn wasSet(comptime path: []const u8) bool {
    ensureLoaded();
    lockTable();
    defer unlockTable();
    if (leafGet(Overrides, pins, path) != null) return true;
    return leafGet(Overrides, env_values, path) != null;
}

/// The derived Q6 floors a GPU provider latches at its one-time
/// configuration read (the seeding rule consults `wasSet`, so a later test
/// pin must not re-derive them).
pub const GpuQ6Floors = struct { dense_q6: u64, packed_q6: u64 };

/// The Q6 seeding rule both GPU providers apply at their one-time
/// configuration, stated once: an explicit `gpu.min_work.dense_q6` (env or
/// pin) also re-seeds the packed-Q6 tier unless `gpu.min_work.dense_q6_packed`
/// is itself explicit; otherwise an explicit `gpu.min_work.qmoe` seeds the
/// dense-Q6 floor; with none explicit both floors keep their measured
/// defaults.
pub fn gpuQ6Seeding() GpuQ6Floors {
    const t = get();
    const packed_value = t.gpu.min_work.dense_q6_packed;
    if (wasSet("gpu.min_work.dense_q6")) {
        return .{
            .dense_q6 = t.gpu.min_work.dense_q6,
            .packed_q6 = if (wasSet("gpu.min_work.dense_q6_packed")) packed_value else t.gpu.min_work.dense_q6,
        };
    }
    if (wasSet("gpu.min_work.qmoe")) {
        return .{ .dense_q6 = t.gpu.min_work.qmoe, .packed_q6 = packed_value };
    }
    return .{ .dense_q6 = t.gpu.min_work.dense_q6, .packed_q6 = packed_value };
}

fn LeafIn(comptime T: type, comptime path: []const u8) type {
    return comptime blk: {
        var t = T;
        var it = std.mem.tokenizeScalar(u8, path, '.');
        while (it.next()) |seg| t = @FieldType(t, seg);
        break :blk t;
    };
}

fn leafGet(comptime T: type, value: T, comptime path: []const u8) LeafIn(T, path) {
    if (comptime std.mem.indexOfScalar(u8, path, '.')) |dot| {
        return leafGet(@FieldType(T, path[0..dot]), @field(value, path[0..dot]), path[dot + 1 ..]);
    }
    return @field(value, path);
}

fn leafPtr(ptr: anytype, comptime path: []const u8) *LeafIn(@typeInfo(@TypeOf(ptr)).pointer.child, path) {
    if (comptime std.mem.indexOfScalar(u8, path, '.')) |dot| {
        return leafPtr(&@field(ptr.*, path[0..dot]), path[dot + 1 ..]);
    }
    return &@field(ptr.*, path);
}

test {
    _ = @import("tuning_tests.zig");
}
