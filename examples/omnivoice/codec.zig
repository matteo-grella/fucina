//! omnivoice-tokenizer GGUF loader (codec weights). `Codec.load` loads the
//! DECODE side (RVQ codebooks + project_out, fc2, DAC decoder); the ENCODE
//! side (HuBERT / SemanticEncoder / DAC encoder / RVQ project_in / fc) is a
//! separate `Encoder.load` so decode-only CLI runs stay light. RVQ encode
//! also reads the decode-side quantizers (codebooks + project_out +
//! embed_sq), so the encode path needs BOTH loaded.
//!
//! Tensor names, shapes and load transforms follow the reference's GGUF
//! export (refs/omnivoice.cpp convert.py). GGUF `ne` order lists the
//! fastest axis first; INTERNAL layout is Fucina's `[T, C]` rows (channel
//! fast), so conv weights are repacked at load:
//! - Conv1d GGUF ne=(K, IC, OC), buffer `src[((oc*IC)+ic)*K + k]` → our
//!   `[tap, in, out]` layout `dst[(k*IC + ic)*OC + oc]`.
//! - ConvTranspose1d GGUF ne=(K, OC, IC), buffer `src[((ic*OC)+oc)*K + k]` →
//!   the facade convTranspose1d `weight2` layout `[K*OC, IC]`
//!   `dst[(oc*K + k)*IC + ic]` (k fastest inside each oc block).
//! Non-f32 tensors are widened/dequantized via `gguf.decodeF32` (exact bf16
//! bit-shift widening, ggml-parity block decoders), matching the reference's
//! type-trait widening. `project_out`/`fc2` matrices stay at their native
//! GGUF dtype behind `llm.weights.LinearWeight`.
//!
//! Ownership: `Codec.load` copies/repacks every tensor it materializes, but
//! quantized `LinearWeight` arms may borrow the mmapped GGUF bytes — keep the
//! `gguf.File` open for the lifetime of the `Codec`.

const std = @import("std");
const builtin = @import("builtin");
const fucina = @import("fucina");
const llm = @import("fucina_llm");

const gguf = fucina.gguf;
const ExecContext = fucina.ExecContext;
const Allocator = std.mem.Allocator;

pub const Error = error{
    NotOmniVoiceTokenizer,
    UnexpectedMetadata,
    InvalidTensorShape,
};

/// Conv1d weight in the facade layout `[tap, in, out]`.
pub const ConvWeight = fucina.Tensor(.{ .tap, .in, .out });
/// Conv1d weight repacked for the DAC decoder's im2col+GEMM path:
/// `[OC, K*IC]` rows with column index `k*IC + ic` — one row per output
/// channel, matching the im2col matrix whose row `t` is the K (dilated)
/// input rows at `t + k·dilation − pad` concatenated. k=1 convs use it
/// directly as a plain `[OC, IC]` linear (no im2col copy).
pub const GemmConvWeight = fucina.Tensor(.{ .out, .kin });
/// ConvTranspose1d repacked weight `[K*OC, IC]` (facade `weight2`).
pub const ConvTWeight = fucina.Tensor(.{ .kout, .in });
/// Per-channel vector on the CURRENT activation channel axis (`.in`).
pub const ChannelVec = fucina.Tensor(.{.in});
/// Per-out-channel vector (ConvTranspose1d bias).
pub const OutVec = fucina.Tensor(.{.out});
/// RVQ codebook `[V, d]` rows (code fast over `.cdim`).
pub const Codebook = fucina.Tensor(.{ .code, .cdim });

pub const n_codebooks = 8;

/// KV config (asserted against the reference's hardcoded constants).
pub const Config = struct {
    sample_rate: usize,
    hop_length: usize,
    codebook_size: usize,
    codebook_dim: usize,
};

/// `1/(alpha + 1e-9)` in f32 arithmetic exactly (dac-decoder.h:101).
pub fn snakeInvB(alpha: f32) f32 {
    return @as(f32, 1.0) / (alpha + @as(f32, 1.0e-9));
}

/// Repack a GGUF Conv1d weight buffer ne=(K, IC, OC) —
/// `src[((oc*IC)+ic)*K + k]` — into our conv1d `[tap, in, out]` layout:
/// `dst[(k*IC + ic)*OC + oc]`.
pub fn repackConv1dWeight(dst: []f32, src: []const f32, taps: usize, in_ch: usize, out_ch: usize) void {
    std.debug.assert(dst.len == taps * in_ch * out_ch);
    std.debug.assert(src.len == dst.len);
    for (0..out_ch) |oc| {
        for (0..in_ch) |ic| {
            for (0..taps) |k| {
                dst[(k * in_ch + ic) * out_ch + oc] = src[(oc * in_ch + ic) * taps + k];
            }
        }
    }
}

/// Repack a GGUF Conv1d weight buffer ne=(K, IC, OC) —
/// `src[((oc*IC)+ic)*K + k]` — into the im2col GEMM layout `[OC, K*IC]`:
/// `dst[oc*(K*IC) + k*IC + ic]` (ic fastest inside each k slot, matching the
/// im2col row layout).
pub fn repackConv1dGemmWeight(dst: []f32, src: []const f32, taps: usize, in_ch: usize, out_ch: usize) void {
    std.debug.assert(dst.len == taps * in_ch * out_ch);
    std.debug.assert(src.len == dst.len);
    for (0..out_ch) |oc| {
        for (0..in_ch) |ic| {
            for (0..taps) |k| {
                dst[oc * (taps * in_ch) + k * in_ch + ic] = src[(oc * in_ch + ic) * taps + k];
            }
        }
    }
}

/// Repack a GGUF ConvTranspose1d weight buffer ne=(K, OC, IC) —
/// `src[((ic*OC)+oc)*K + k]` — into the facade convTranspose1d `weight2`
/// `[K*OC, IC]` layout: `dst[(oc*K + k)*IC + ic]` (k fastest inside each oc
/// block; identical to the reference's load-time repack, dac-decoder.h).
pub fn repackConvT1dWeight(dst: []f32, src: []const f32, taps: usize, out_ch: usize, in_ch: usize) void {
    std.debug.assert(dst.len == taps * out_ch * in_ch);
    std.debug.assert(src.len == dst.len);
    for (0..in_ch) |ic| {
        for (0..out_ch) |oc| {
            for (0..taps) |k| {
                dst[(oc * taps + k) * in_ch + ic] = src[(ic * out_ch + oc) * taps + k];
            }
        }
    }
}

// --- ggml-parity f16 conv (the encode-path conv arithmetic) -----------------
//
// The reference computes every codec conv as ggml_conv_1d = im2col with an
// F16 destination + mul_mat against the F16 weight, and the CPU f16 dot
// (ggml_vec_dot_f16, NEON fp16 path) accumulates in FOUR f16x8 lanes with
// fused f16 FMA, reduces pairwise in f16, converts to f32 and finishes the
// (n % 32) tail in f64 from f32 products. Computing the convs in f32 is
// *more* precise but drifts ~5e-4 relative per conv — enough compounded
// noise to flip ~5% of the RVQ codes vs the reference. The encode path
// therefore mirrors the reference arithmetic exactly on aarch64: weights
// cast to f16 at load (gf_load_conv_f16), activations cast to f16 rows
// (im2col), and the dot below reproduces ggml_vec_dot_f16
// operation-for-operation. That aarch64 replica IS the bit-exact-encode
// parity contract (macOS/Accelerate goldens).
//
// On other arches the dot instead accumulates in f32 (F16C widen + f32
// FMA), mirroring ggml's OWN x86 vec_dot_f16 (the GGML_F32Cx8 path): the
// shipped x86 reference binary does the same there, so it does not produce
// NEON-identical codes either — bit-parity with the macOS goldens is a
// macOS-only contract, and f16-accumulate emulation on x86 (no AVX512-FP16)
// is a per-op promote/round disaster (~8x on the whole codec encode).
//
// Structurally, non-aarch64 also computes the conv as a chunked f16 im2col
// MATRIX + one TransB GEMM per group (`ggmlConv1dGemmInto`) instead of one
// short dot per (frame, out-channel): the per-call overhead of the replica
// loop dominates on x86 (k is as small as 10 floats in the first HuBERT
// feature-extractor layer), and the reference computes conv the same way
// (im2col + one f16 mul_mat). The im2col values are bit-identical to the
// replica's col rows; only the f32 accumulation ORDER differs (the backend
// GEMM's register blocking vs vecDotF16Ggml's 4-lane shape), which the
// tolerance-based default tests absorb. aarch64 keeps the per-frame replica
// path bit-unchanged. (A pure-f32 GEMM over pre-widened operands was tried
// and measured SLOWER — the f32 weight matrix spills L2 on the big
// feature-extractor shapes; the f16 kernel's in-register F16C widen wins.)

/// ggml-parity conv weight: the flat GGUF buffer order `w[oc][ic_pg][k]`
/// (row length `in_per_group*taps` per output channel), cast to f16.
pub const GgmlConvWeight = struct {
    /// `[out_ch][in_per_group*taps]` rows.
    data: []f16,
    taps: usize,
    in_per_group: usize,
    out_ch: usize,
    groups: usize,

    pub fn deinit(self: *GgmlConvWeight, allocator: Allocator) void {
        allocator.free(self.data);
        self.* = undefined;
    }
};

/// `[t, c]` rows result of `ggmlConv1d` (caller frees `data`).
pub const ConvRows = struct {
    t: usize,
    c: usize,
    data: []f32,
};

/// ggml_vec_dot_f16. aarch64: the NEON
/// __ARM_FEATURE_FP16_VECTOR_ARITHMETIC path, operation-for-operation — 4 ×
/// f16x8 fused-FMA lane accumulators over 32-element steps, pairwise f16
/// reduction, f32 convert + vaddvq-style pairwise horizontal sum. This is
/// the bit-exact-encode parity contract vs the macOS reference goldens.
/// Other arches: ggml's own x86 GGML_F32Cx8 shape — widen each f16x8 lane
/// to f32 (vcvtph2ps under F16C) and accumulate in 4 × f32x8 fused-FMA
/// lanes; the shipped x86 reference does the same, so NEON-identical
/// results were never on offer there, and f16-accumulate emulation without
/// AVX512-FP16 is catastrophically slow. Both arms share the exact f64 tail
/// of f32 products for n % 32.
pub fn vecDotF16Ggml(x: []const f16, y: []const f16) f32 {
    std.debug.assert(x.len == y.len);
    const n = x.len;
    const np = n & ~@as(usize, 31);
    var sumf: f64 = 0.0;
    if (np > 0) {
        if (comptime builtin.cpu.arch.isAARCH64()) {
            const V = @Vector(8, f16);
            var sum: [4]V = .{ @splat(0), @splat(0), @splat(0), @splat(0) };
            var i: usize = 0;
            while (i < np) : (i += 32) {
                inline for (0..4) |j| {
                    const ax: V = x[i + j * 8 ..][0..8].*;
                    const ay: V = y[i + j * 8 ..][0..8].*;
                    sum[j] = @mulAdd(V, ax, ay, sum[j]);
                }
            }
            // GGML_F16x8_REDUCE: sum0+=sum2, sum1+=sum3, sum0+=sum1 (f16 adds),
            // then f32-convert halves, vaddq, vaddvq (pairwise) → f32.
            sum[0] += sum[2];
            sum[1] += sum[3];
            sum[0] += sum[1];
            const lanes: [8]f16 = sum[0];
            var wide: [8]f32 = undefined;
            for (&wide, lanes) |*dst, v| dst.* = v;
            const v0 = wide[0] + wide[4];
            const v1 = wide[1] + wide[5];
            const v2 = wide[2] + wide[6];
            const v3 = wide[3] + wide[7];
            sumf = (v0 + v1) + (v2 + v3);
        } else {
            const H = @Vector(8, f16);
            const F = @Vector(8, f32);
            var sum: [4]F = .{ @splat(0), @splat(0), @splat(0), @splat(0) };
            var i: usize = 0;
            while (i < np) : (i += 32) {
                inline for (0..4) |j| {
                    const ax: F = @floatCast(@as(H, x[i + j * 8 ..][0..8].*));
                    const ay: F = @floatCast(@as(H, y[i + j * 8 ..][0..8].*));
                    sum[j] = @mulAdd(F, ax, ay, sum[j]);
                }
            }
            const acc = (sum[0] + sum[1]) + (sum[2] + sum[3]);
            sumf = @reduce(.Add, acc);
        }
    }
    var i = np;
    while (i < n) : (i += 1) {
        sumf += @as(f32, x[i]) * @as(f32, y[i]);
    }
    return @floatCast(sumf);
}

// System libm (what ggml's CPU kernels call): the encode path matches the
// reference's elementwise transcendentals exactly — Fucina's kernels use a
// musl erff port and Zig's own expm1, which differ from Apple libm in the
// last ulp on a fair fraction of inputs; those 1-ulp diffs flip f16 im2col
// roundings downstream and cascade through the conv stacks.
extern "c" fn erff(x: f32) f32;
extern "c" fn expm1f(x: f32) f32;
extern "c" fn sinf(x: f32) f32;
extern "c" fn expf(x: f32) f32;

/// Reference-parity builds link Accelerate (the reference ggml build has
/// GGML_ACCELERATE + the Apple BLAS backend). Without it (-Dblas=none /
/// non-macOS) the fallbacks below keep the pipeline correct but give up
/// bit-parity with the shipped macOS reference binary.
pub const use_accelerate = fucina.native_blas_kind == .accelerate;

// Accelerate vDSP — ggml_norm's GGML_USE_ACCELERATE path: the sum /
// mean-of-squares reductions have library-internal accumulation orders, so
// parity means calling the same routines.
const vdsp = if (use_accelerate) struct {
    extern "c" fn vDSP_sve(a: [*]const f32, ia: isize, c: *f32, n: usize) void;
    extern "c" fn vDSP_vsadd(a: [*]const f32, ia: isize, b: *const f32, c: [*]f32, ic: isize, n: usize) void;
    extern "c" fn vDSP_measqv(a: [*]const f32, ia: isize, c: *f32, n: usize) void;
} else struct {};

/// Run `runFn` over `tasks` on raw threads (task 0 on the caller); tasks
/// whose spawn fails run serially after the join. Shared by the parallel
/// elementwise/load helpers below (+ hubert.zig's per-head attention split).
pub fn runTaskThreads(comptime Task: type, comptime runFn: fn (task: *const Task) void, tasks: []const Task) void {
    const max_threads = fucina.parallel.vector_max_threads;
    std.debug.assert(tasks.len <= max_threads);
    if (tasks.len == 1) return runFn(&tasks[0]);
    var threads: [max_threads]std.Thread = undefined;
    var spawned: usize = 0;
    for (1..tasks.len) |i| {
        threads[i] = std.Thread.spawn(.{}, runFn, .{&tasks[i]}) catch break;
        spawned = i;
    }
    runFn(&tasks[0]);
    for (1..spawned + 1) |i| threads[i].join();
    // Any tasks that failed to spawn run serially here.
    for (spawned + 1..tasks.len) |i| runFn(&tasks[i]);
}

/// Split `values` into per-thread ranges aligned to `row` elements and run
/// `rangeFn(slice, args)` on raw threads. For the ELEMENTWISE ggml-parity
/// kernels only: no cross-element accumulation, so any split is
/// bit-identical to the serial loop. Returns false when the work is too
/// small or only one thread is available (caller runs the serial range).
fn parallelElemSplitRows(comptime rangeFn: anytype, values: []f32, row: usize, args: anytype) bool {
    const max_threads = fucina.parallel.vector_max_threads;
    if (values.len < 1 << 17) return false;
    const rows = values.len / row;
    const want = @min(fucina.parallel.cpuThreadCount(max_threads), rows);
    if (want <= 1) return false;
    const Task = struct {
        slice: []f32,
        args: @TypeOf(args),
        fn run(task: *const @This()) void {
            rangeFn(task.slice, task.args);
        }
    };
    var tasks: [max_threads]Task = undefined;
    for (0..want) |i| {
        tasks[i] = .{ .slice = values[(i * rows / want) * row .. ((i + 1) * rows / want) * row], .args = args };
    }
    runTaskThreads(Task, Task.run, tasks[0..want]);
    return true;
}

fn parallelElemSplit(comptime rangeFn: anytype, values: []f32, args: anytype) bool {
    return parallelElemSplitRows(rangeFn, values, 1, args);
}

/// ggml_vec_gelu_erf_f32 parity, in place: `0.5*x*(1+erff(x*SQRT_2_INV))`
/// with the system erff.
pub fn geluErfGgml(values: []f32) void {
    if (comptime !builtin.cpu.arch.isAARCH64()) {
        // Elementwise → any split is bit-identical; erff dominates the
        // encode's serial wall on x86 (feature-extractor activations).
        if (parallelElemSplit(geluErfRange, values, {})) return;
    }
    geluErfRange(values, {});
}

fn geluErfRange(values: []f32, args: void) void {
    _ = args;
    for (values) |*v| {
        const x = v.*;
        v.* = 0.5 * x * (1.0 + erff(x * 0.70710678118654752440084436210484));
    }
}

/// ggml_vec_elu_f32 parity, in place: `x > 0 ? x : expm1f(x)` with the
/// system expm1f.
pub fn eluGgml(values: []f32) void {
    for (values) |*v| {
        if (!(v.* > 0)) v.* = expm1f(v.*);
    }
}

/// Snake exactly as ggml's fused kernel executes it
/// (ggml_compute_forward_snake_fused: `yc = xi + si*si*bc`, `si =
/// sinf(ac*xi)`) — clang compiles that expression with the default
/// fp-contract=on, so the trailing multiply-add is a single fused
/// `fma(si*si, bc, xi)` (verified 1-ulp divergence otherwise).
/// In place over `[t, c]` rows.
pub fn snakeGgml(values: []f32, t: usize, c: usize, alpha: []const f32, inv_b: []const f32) void {
    std.debug.assert(values.len == t * c and alpha.len == c and inv_b.len == c);
    const args: SnakeArgs = .{ .alpha = alpha, .inv_b = inv_b };
    if (comptime !builtin.cpu.arch.isAARCH64()) {
        // Elementwise → row splits are bit-identical; sinf dominates the
        // DAC-encoder snakes' serial wall on x86.
        if (parallelElemSplitRows(snakeRange, values, c, args)) return;
    }
    snakeRange(values, args);
}

const SnakeArgs = struct { alpha: []const f32, inv_b: []const f32 };

fn snakeRange(values: []f32, args: SnakeArgs) void {
    const c = args.alpha.len;
    var i: usize = 0;
    while (i < values.len) : (i += c) {
        const row = values[i..][0..c];
        for (row, args.alpha, args.inv_b) |*v, a, ib| {
            const s = sinf(a * v.*);
            v.* = @mulAdd(f32, s * s, ib, v.*);
        }
    }
}

/// ggml_group_norm parity over `[t, c]` rows + the reference's separate
/// affine mul/add, in place: per group of channels, f64 sums nested per
/// channel (inner over time), mean/variance cast to f32 like the reference,
/// `v = x − mean` stored f32 with `v*v` accumulated from the f32 product,
/// `scale = 1/sqrtf(variance + eps)`, then `y = (v*scale)*w + b`.
pub fn groupNormGgml(values: []f32, t: usize, c: usize, groups: usize, eps: f32, weight: ?[]const f32, bias: ?[]const f32) void {
    std.debug.assert(values.len == t * c and c % groups == 0);
    const step = c / groups;
    for (0..groups) |g| {
        const start = g * step;

        var sum: f64 = 0.0;
        for (start..start + step) |ch| {
            var sumr: f64 = 0.0;
            for (0..t) |ti| sumr += values[ti * c + ch];
            sum += sumr;
        }
        const mean: f32 = @floatCast(sum / @as(f64, @floatFromInt(t * step)));

        var sum2: f64 = 0.0;
        for (start..start + step) |ch| {
            var sumr: f64 = 0.0;
            for (0..t) |ti| {
                const v = values[ti * c + ch] - mean;
                values[ti * c + ch] = v;
                sumr += @as(f32, v * v);
            }
            sum2 += sumr;
        }
        const variance: f32 = @floatCast(sum2 / @as(f64, @floatFromInt(t * step)));
        const scale = 1.0 / @sqrt(variance + eps);

        for (start..start + step) |ch| {
            for (0..t) |ti| {
                var v = values[ti * c + ch] * scale;
                if (weight) |w| v *= w[ch];
                if (bias) |b| v += b[ch];
                values[ti * c + ch] = v;
            }
        }
    }
}

/// ggml_norm parity over `[t, c]` rows (the reference's Accelerate path:
/// vDSP_sve sum → f32 mean → vDSP_vsadd subtract → vDSP_measqv variance →
/// `scale = 1/sqrtf(variance + eps)`), plus the reference's separate affine
/// mul/add ops. In place.
pub fn layerNormGgml(values: []f32, t: usize, c: usize, eps: f32, weight: ?[]const f32, bias: ?[]const f32) void {
    std.debug.assert(values.len == t * c);
    const cf: f32 = @floatFromInt(c);
    for (0..t) |ti| {
        const row = values[ti * c ..][0..c];
        var mean: f32 = undefined;
        var variance: f32 = undefined;
        if (comptime use_accelerate) {
            var sum: f32 = 0.0;
            vdsp.vDSP_sve(row.ptr, 1, &sum, c);
            var neg_mean: f32 = -(sum / cf);
            vdsp.vDSP_vsadd(row.ptr, 1, &neg_mean, row.ptr, 1, c);
            vdsp.vDSP_measqv(row.ptr, 1, &variance, c);
            mean = 0.0;
        } else {
            // ggml_norm's non-Accelerate branch shape (f64 accumulation).
            var sum: f64 = 0.0;
            for (row) |v| sum += v;
            mean = @floatCast(sum / @as(f64, cf));
            var sum2: f64 = 0.0;
            for (row) |*v| {
                const d = v.* - mean;
                v.* = d;
                sum2 += @as(f32, d * d);
            }
            variance = @floatCast(sum2 / @as(f64, cf));
        }
        const scale = 1.0 / @sqrt(variance + eps);
        for (row, 0..) |*v, ch| {
            var y = v.* * scale;
            if (weight) |w| y *= w[ch];
            if (bias) |b| y += b[ch];
            v.* = y;
        }
    }
}

/// ggml_v_expf (NEON f32x4 path), instruction-for-instruction: the vector
/// exp approximation ggml's softmax uses.
fn vExpf(x: @Vector(4, f32)) @Vector(4, f32) {
    const V = @Vector(4, f32);
    const U = @Vector(4, u32);
    const r: V = @splat(0x1.8p23);
    const z = @mulAdd(V, x, @splat(0x1.715476p+0), r);
    const n = z - r;
    // vfmsq_f32(a, b, c) = a − b*c (fused).
    const b = @mulAdd(V, -n, @splat(0x1.7f7d1cp-20), @mulAdd(V, -n, @splat(0x1.62e4p-1), x));
    const e: U = @as(U, @bitCast(z)) << @splat(23);
    const k: V = @bitCast(e +% @as(U, @bitCast(@as(V, @splat(1.0)))));
    const c_mask = @abs(n) > @as(V, @splat(126.0));
    const u = b * b;
    const j = @mulAdd(
        V,
        @mulAdd(V, @mulAdd(V, @as(V, @splat(0x1.0e4020p-7)), b, @splat(0x1.573e2ep-5)), u, @mulAdd(V, @as(V, @splat(0x1.555e66p-3)), b, @splat(0x1.fffdb6p-2))),
        u,
        @as(V, @splat(0x1.ffffecp-1)) * b,
    );
    const fast = @mulAdd(V, k, j, k);
    const d = @select(u32, n <= @as(V, @splat(0.0)), @as(U, @splat(0x82000000)), @as(U, @splat(0)));
    const s1: V = @bitCast(d +% @as(U, @splat(0x7f000000)));
    const s2: V = @bitCast(e -% d);
    const big = @select(f32, @abs(n) > @as(V, @splat(192.0)), s1 * s1, @select(f32, c_mask, @mulAdd(V, s2, j, s2) * s1, fast));
    return @select(f32, c_mask, big, fast);
}

/// ggml_compute_forward_soft_max_f32 parity for one plain row (scale 1, no
/// mask/sinks): running max, ggml_vec_soft_max_f32 (v_expf 4-wide chunks
/// with pairwise-horizontal f64 chunk sums, scalar expf tail), then
/// `y *= (float)(1.0/sum)`. In place.
pub fn softMaxRowGgml(row: []f32) void {
    var max = -std.math.inf(f32);
    for (row) |v| max = @max(max, v);

    var sum: f64 = 0.0;
    var i: usize = 0;
    while (i + 4 <= row.len) : (i += 4) {
        const x: @Vector(4, f32) = row[i..][0..4].*;
        const val = vExpf(x - @as(@Vector(4, f32), @splat(max)));
        row[i..][0..4].* = val;
        // vaddvq_f32 pairwise horizontal sum.
        sum += (val[0] + val[1]) + (val[2] + val[3]);
    }
    while (i < row.len) : (i += 1) {
        const val = expf(row[i] - max);
        sum += val;
        row[i] = val;
    }
    const scale: f32 = @floatCast(1.0 / sum);
    for (row) |*v| v.* *= scale;
}

const GgmlConvTask = struct {
    input: []const f32,
    t_in: usize,
    in_ch: usize,
    w: *const GgmlConvWeight,
    bias: ?[]const f32,
    stride: usize,
    pad: usize,
    dilation: usize,
    t_out: usize,
    out: []f32,
    col: []f16, // per-task scratch, one im2col row (in_per_group*taps)
    ol_start: usize,
    ol_end: usize,
};

fn ggmlConvRange(task: *const GgmlConvTask) void {
    const w = task.w;
    const n = w.in_per_group * w.taps;
    const oc_per_group = w.out_ch / w.groups;
    for (task.ol_start..task.ol_end) |ol| {
        const out_row = task.out[ol * w.out_ch ..][0..w.out_ch];
        for (0..w.groups) |g| {
            // im2col row for this (frame, group): col[ic*K + k] =
            // f16(x[ol*s + k*d − p][g*icpg + ic]), 0 when out of bounds.
            for (0..w.in_per_group) |ic| {
                const chan = g * w.in_per_group + ic;
                for (0..w.taps) |k| {
                    const pos = ol * task.stride + k * task.dilation;
                    task.col[ic * w.taps + k] = if (pos < task.pad or pos - task.pad >= task.t_in)
                        0.0
                    else
                        @floatCast(task.input[(pos - task.pad) * task.in_ch + chan]);
                }
            }
            for (0..oc_per_group) |oc_local| {
                const oc = g * oc_per_group + oc_local;
                const w_row = w.data[oc * n ..][0..n];
                var v = vecDotF16Ggml(task.col[0..n], w_row);
                if (task.bias) |b| v += b[oc];
                out_row[oc] = v;
            }
        }
    }
}

fn runGgmlConvTask(task: *const GgmlConvTask) void {
    ggmlConvRange(task);
}

/// ggml_conv_1d parity forward over `[t_in, in_ch]` f32 rows → `[t_out,
/// out_ch]` f32 rows (+ optional f32 bias, added after the f32 store like
/// the reference's separate ggml_add). Threaded over output frames; each
/// (frame, channel) dot is independent, so the result is bit-identical to
/// serial. Caller frees `.data`.
pub fn ggmlConv1d(
    allocator: Allocator,
    input: []const f32,
    t_in: usize,
    in_ch: usize,
    w: *const GgmlConvWeight,
    bias: ?[]const f32,
    stride: usize,
    pad: usize,
    dilation: usize,
) !ConvRows {
    std.debug.assert(input.len == t_in * in_ch);
    std.debug.assert(in_ch == w.in_per_group * w.groups);
    std.debug.assert(w.out_ch % w.groups == 0);
    if (bias) |b| std.debug.assert(b.len == w.out_ch);
    const span = dilation * (w.taps - 1) + 1;
    if (t_in + 2 * pad < span) return Error.InvalidTensorShape;
    const t_out = (t_in + 2 * pad - span) / stride + 1;

    const out = try allocator.alloc(f32, t_out * w.out_ch);
    errdefer allocator.free(out);

    if (comptime !builtin.cpu.arch.isAARCH64()) {
        // Chunked im2col + f16 GEMM: same col values, batched accumulation
        // (see the arch note above). aarch64 stays on the replica loop below.
        try ggmlConv1dGemmInto(allocator, out, input, t_in, in_ch, w, bias, stride, pad, dilation, t_out);
        return .{ .t = t_out, .c = w.out_ch, .data = out };
    }

    const n = w.in_per_group * w.taps;

    const max_threads = fucina.parallel.vector_max_threads;
    const work = t_out * w.out_ch * n;
    const want_threads: usize = if (work < 1 << 20) 1 else @min(fucina.parallel.cpuThreadCount(max_threads), t_out);
    const col = try allocator.alloc(f16, n * want_threads);
    defer allocator.free(col);

    var tasks: [max_threads]GgmlConvTask = undefined;
    for (0..want_threads) |i| {
        tasks[i] = .{
            .input = input,
            .t_in = t_in,
            .in_ch = in_ch,
            .w = w,
            .bias = bias,
            .stride = stride,
            .pad = pad,
            .dilation = dilation,
            .t_out = t_out,
            .out = out,
            .col = col[i * n ..][0..n],
            .ol_start = i * t_out / want_threads,
            .ol_end = (i + 1) * t_out / want_threads,
        };
    }
    if (want_threads == 1) {
        ggmlConvRange(&tasks[0]);
    } else {
        var threads: [max_threads]std.Thread = undefined;
        var spawned: usize = 0;
        for (1..want_threads) |i| {
            threads[i] = std.Thread.spawn(.{}, runGgmlConvTask, .{&tasks[i]}) catch break;
            spawned = i;
        }
        ggmlConvRange(&tasks[0]);
        for (1..spawned + 1) |i| threads[i].join();
        // Any tasks that failed to spawn run serially here.
        for (spawned + 1..want_threads) |i| ggmlConvRange(&tasks[i]);
    }

    return .{ .t = t_out, .c = w.out_ch, .data = out };
}

// --- non-aarch64 conv path: chunked f16 im2col + per-group TransB GEMM ------

/// Raw borrowed-view types for the backend GEMM call below (example-local
/// allowed-raw zone: this file already speaks the raw ggml layouts).
const F16View = fucina.internal.tensor_mod.TensorOf(.f16);
const F32View = fucina.internal.RawTensor;

/// Whole-call im2col scratch ceiling, split across threads (the decoder's
/// measured 32 MiB sweet spot, dac.zig).
const conv_gemm_chunk_bytes: usize = 32 << 20;

const ConvGemmTask = struct {
    input: []const f32,
    t_in: usize,
    in_ch: usize,
    w: *const GgmlConvWeight,
    bias: ?[]const f32,
    stride: usize,
    pad: usize,
    dilation: usize,
    out: []f32,
    chunk_rows: usize,
    backend: *const fucina.Backend,
    /// This thread's `[chunk_rows, n_col]` f16 im2col scratch (+ its view).
    col: []f16,
    col_view: F16View,
    /// Shared `[out_ch, n_col]` weight-row view (value copy of a borrow; the
    /// caller owns/deinits the original).
    w_view: F16View,
    /// Shared `[t_out, out_ch]` output view (value copy of a borrow).
    out_view: F32View,
    /// This thread's `[chunk_rows, oc_per_group]` f32 GEMM staging for
    /// grouped convs (empty/undefined when groups == 1).
    stage: []f32,
    stage_view: F32View,
    ol_start: usize,
    ol_end: usize,
};

fn runConvGemmTask(task: *const ConvGemmTask) void {
    const w = task.w;
    const n_col = w.in_per_group * w.taps;
    const oc_per_group = w.out_ch / w.groups;
    var row0 = task.ol_start;
    while (row0 < task.ol_end) : (row0 += task.chunk_rows) {
        const rows = @min(task.chunk_rows, task.ol_end - row0);
        for (0..w.groups) |g| {
            // im2col rows for this (chunk, group), bit-identical to the
            // replica path's col rows: col[r][ic*K + k] =
            // f16(x[(row0+r)*s + k*d − p][g*icpg + ic]), 0 out of bounds.
            for (0..rows) |r| {
                const ol = row0 + r;
                const dst = task.col[r * n_col ..][0..n_col];
                for (0..w.in_per_group) |ic| {
                    const chan = g * w.in_per_group + ic;
                    for (0..w.taps) |k| {
                        const pos = ol * task.stride + k * task.dilation;
                        dst[ic * w.taps + k] = if (pos < task.pad or pos - task.pad >= task.t_in)
                            0.0
                        else
                            @floatCast(task.input[(pos - task.pad) * task.in_ch + chan]);
                    }
                }
            }
            if (w.groups == 1) {
                // GEMM straight into the output rows. The unchecked GEMM only
                // reads buffer/offset plus the explicit m·n·k extents, so a
                // value copy of the shared view shifted to this chunk's first
                // row addresses exactly the [rows, out_ch] destination.
                var out_view = task.out_view;
                out_view.offset += row0 * w.out_ch;
                task.backend.matmulTransB2DIntoUncheckedF16Operands(&out_view, &task.col_view, &task.w_view, rows, w.out_ch, n_col);
                if (task.bias) |b| {
                    for (0..rows) |r| {
                        const out_row = task.out[(row0 + r) * w.out_ch ..][0..w.out_ch];
                        for (out_row, b) |*v, bv| v.* += bv;
                    }
                }
            } else {
                // Grouped conv: the group's output columns are strided by
                // out_ch in `out`, so GEMM into dense staging and scatter.
                var w_view = task.w_view;
                w_view.offset += g * oc_per_group * n_col;
                var stage_view = task.stage_view;
                task.backend.matmulTransB2DIntoUncheckedF16Operands(&stage_view, &task.col_view, &w_view, rows, oc_per_group, n_col);
                for (0..rows) |r| {
                    const src = task.stage[r * oc_per_group ..][0..oc_per_group];
                    const out_row = task.out[(row0 + r) * w.out_ch + g * oc_per_group ..][0..oc_per_group];
                    if (task.bias) |b| {
                        for (out_row, src, b[g * oc_per_group ..][0..oc_per_group]) |*v, s, bv| v.* = s + bv;
                    } else {
                        @memcpy(out_row, src);
                    }
                }
            }
        }
    }
}

/// Non-aarch64 `ggmlConv1d` body: per thread, loop ≤`chunk_rows` frame
/// chunks; per (chunk, group), build the f16 im2col matrix and run ONE
/// f16-operands TransB GEMM (f32 accumulation — the same arithmetic class as
/// the wide `vecDotF16Ggml` arm). Threads split the output frames, so each
/// GEMM call runs its single-threaded range (the pool-less `Backend`).
fn ggmlConv1dGemmInto(
    allocator: Allocator,
    out: []f32,
    input: []const f32,
    t_in: usize,
    in_ch: usize,
    w: *const GgmlConvWeight,
    bias: ?[]const f32,
    stride: usize,
    pad: usize,
    dilation: usize,
    t_out: usize,
) !void {
    const n_col = w.in_per_group * w.taps;
    const oc_per_group = w.out_ch / w.groups;

    const max_threads = fucina.parallel.vector_max_threads;
    const work = t_out * w.out_ch * n_col;
    const want_threads: usize = if (work < 1 << 20) 1 else @min(fucina.parallel.cpuThreadCount(max_threads), t_out);

    const rows_per_thread = (t_out + want_threads - 1) / want_threads;
    const budget_rows = conv_gemm_chunk_bytes / (@sizeOf(f16) * n_col * want_threads);
    const chunk_rows = @max(1, @min(budget_rows, rows_per_thread));

    const col = try allocator.alloc(f16, want_threads * chunk_rows * n_col);
    defer allocator.free(col);
    const stage = try allocator.alloc(f32, if (w.groups > 1) want_threads * chunk_rows * oc_per_group else 0);
    defer allocator.free(stage);

    // Pool-less backend: each spawned thread below runs its own
    // single-threaded GEMM range.
    var conv_backend = fucina.Backend.init();

    var w_view = try F16View.fromBorrowedSlice(allocator, &.{ w.out_ch, n_col }, w.data);
    defer w_view.deinit();
    var out_view = try F32View.fromBorrowedSlice(allocator, &.{ t_out, w.out_ch }, out);
    defer out_view.deinit();

    var tasks: [max_threads]ConvGemmTask = undefined;
    var made: usize = 0;
    defer for (tasks[0..made]) |*task| {
        task.col_view.deinit();
        if (w.groups > 1) task.stage_view.deinit();
    };
    for (0..want_threads) |i| {
        const col_slice = col[i * chunk_rows * n_col ..][0 .. chunk_rows * n_col];
        var col_view = try F16View.fromBorrowedSlice(allocator, &.{ chunk_rows, n_col }, col_slice);
        errdefer col_view.deinit();
        var stage_slice: []f32 = stage[0..0];
        var stage_view: F32View = undefined;
        if (w.groups > 1) {
            stage_slice = stage[i * chunk_rows * oc_per_group ..][0 .. chunk_rows * oc_per_group];
            stage_view = try F32View.fromBorrowedSlice(allocator, &.{ chunk_rows, oc_per_group }, stage_slice);
        }
        tasks[i] = .{
            .input = input,
            .t_in = t_in,
            .in_ch = in_ch,
            .w = w,
            .bias = bias,
            .stride = stride,
            .pad = pad,
            .dilation = dilation,
            .out = out,
            .chunk_rows = chunk_rows,
            .backend = &conv_backend,
            .col = col_slice,
            .col_view = col_view,
            .w_view = w_view,
            .out_view = out_view,
            .stage = stage_slice,
            .stage_view = stage_view,
            .ol_start = i * t_out / want_threads,
            .ol_end = (i + 1) * t_out / want_threads,
        };
        made += 1;
    }

    if (want_threads == 1) {
        runConvGemmTask(&tasks[0]);
    } else {
        var threads: [max_threads]std.Thread = undefined;
        var spawned: usize = 0;
        for (1..want_threads) |i| {
            threads[i] = std.Thread.spawn(.{}, runConvGemmTask, .{&tasks[i]}) catch break;
            spawned = i;
        }
        runConvGemmTask(&tasks[0]);
        for (1..spawned + 1) |i| threads[i].join();
        // Any tasks that failed to spawn run serially here.
        for (spawned + 1..want_threads) |i| runConvGemmTask(&tasks[i]);
    }
}

/// DAC decoder up-block geometry: K = 2S, pad = (S+1)/2, output_pad = S%2.
pub const BlockSpec = struct {
    in_ch: usize,
    out_ch: usize,
    stride: usize,
    taps: usize,
    pad: usize,
    output_pad: usize,

    pub fn init(in_ch: usize, out_ch: usize, stride: usize) BlockSpec {
        return .{
            .in_ch = in_ch,
            .out_ch = out_ch,
            .stride = stride,
            .taps = 2 * stride,
            .pad = (stride + 1) / 2,
            .output_pad = stride % 2,
        };
    }
};

/// The 5 decoder up-blocks (upsampling ratios 8·5·4·2·3 = 960).
pub const dac_block_specs = [5]BlockSpec{
    BlockSpec.init(1024, 512, 8),
    BlockSpec.init(512, 256, 5),
    BlockSpec.init(256, 128, 4),
    BlockSpec.init(128, 64, 2),
    BlockSpec.init(64, 32, 3),
};

/// Res-unit dilations inside every up-block (conv1 pad = 3·dilation).
pub const res_unit_dilations = [3]usize{ 1, 3, 9 };

/// DAC encoder down-block geometry (dac-encoder.h): res units on `in_ch`
/// first, then block-level snake + the strided downsampling conv.
pub const EncBlockSpec = struct {
    in_ch: usize,
    out_ch: usize,
    taps: usize,
    stride: usize,
    pad: usize,
};

/// The 5 encoder down-blocks (downsampling ratios 8·5·4·2·3 = 960).
pub const dac_enc_block_specs = [5]EncBlockSpec{
    .{ .in_ch = 64, .out_ch = 128, .taps = 16, .stride = 8, .pad = 4 },
    .{ .in_ch = 128, .out_ch = 256, .taps = 10, .stride = 5, .pad = 3 },
    .{ .in_ch = 256, .out_ch = 512, .taps = 8, .stride = 4, .pad = 2 },
    .{ .in_ch = 512, .out_ch = 1024, .taps = 4, .stride = 2, .pad = 1 },
    .{ .in_ch = 1024, .out_ch = 2048, .taps = 6, .stride = 3, .pad = 2 },
};

// --- HuBERT architecture constants (hubert-enc.h; hardcoded upstream) ------

pub const hubert_feat_num_layers = 7;
pub const hubert_feat_kernels = [hubert_feat_num_layers]usize{ 10, 3, 3, 3, 3, 2, 2 };
pub const hubert_feat_strides = [hubert_feat_num_layers]usize{ 5, 2, 2, 2, 2, 2, 2 };
pub const hubert_feat_dim = 512;
pub const hubert_hidden = 768;
pub const hubert_num_heads = 12;
pub const hubert_head_dim = 64;
pub const hubert_ffn_inner = 3072;
pub const hubert_num_layers = 12;
pub const hubert_pos_k = 128;
pub const hubert_pos_groups = 16;
pub const hubert_pos_ic_pg = hubert_hidden / hubert_pos_groups; // 48
pub const hubert_pos_pad = hubert_pos_k / 2; // 64
pub const hubert_ln_eps: f32 = 1.0e-5;
pub const semantic_hidden = 768;

/// One RVQ codebook stage (decode side; `project_in` lands in stage C).
pub const Quantizer = struct {
    /// `[V=1024, d=64]` f32 codebook rows.
    embed: Codebook,
    /// `embed_sq[j] = Σ_d embed[j,d]²` in f32 from the DECODED values —
    /// the encode-side NN-search constant (stage C), precomputed at load
    /// like the reference (rvq-codec.h:83-96).
    embed_sq: []f32,
    /// Linear(64→1024) at native GGUF dtype.
    project_out: llm.weights.LinearWeight,
    /// `[1024]` f32.
    project_out_bias: []f32,

    pub fn deinit(self: *Quantizer, allocator: Allocator) void {
        self.embed.deinit();
        allocator.free(self.embed_sq);
        self.project_out.deinit();
        allocator.free(self.project_out_bias);
        self.* = undefined;
    }
};

/// RVQ decode weights + the joint fc2 projection (RVQ latent → DAC input).
pub const RvqDecoder = struct {
    quantizers: [n_codebooks]Quantizer,
    /// Linear(1024→256) at native GGUF dtype.
    fc2: llm.weights.LinearWeight,
    /// `[256]` f32.
    fc2_bias: []f32,

    pub fn deinit(self: *RvqDecoder, allocator: Allocator) void {
        for (&self.quantizers) |*q| q.deinit(allocator);
        self.fc2.deinit();
        allocator.free(self.fc2_bias);
        self.* = undefined;
    }
};

/// Dilated residual unit (shared decoder shape): snake→conv(k=7, d)→snake→
/// conv(k=1) with an outer skip. Channels constant.
pub const ResUnit = struct {
    snake1_a: ChannelVec,
    snake1_inv_b: ChannelVec,
    conv1_w: ConvWeight, // k=7, pad=3·dilation
    conv1_gw: GemmConvWeight, // same weights, [OC, K*IC] (im2col GEMM path)
    conv1_b: []f32,
    snake2_a: ChannelVec,
    snake2_inv_b: ChannelVec,
    conv2_w: ConvWeight, // k=1, pad=0
    conv2_gw: GemmConvWeight, // [OC, IC] (pure GEMM)
    conv2_b: []f32,
    dilation: usize,

    pub fn deinit(self: *ResUnit, allocator: Allocator) void {
        self.snake1_a.deinit();
        self.snake1_inv_b.deinit();
        self.conv1_w.deinit();
        self.conv1_gw.deinit();
        allocator.free(self.conv1_b);
        self.snake2_a.deinit();
        self.snake2_inv_b.deinit();
        self.conv2_w.deinit();
        self.conv2_gw.deinit();
        allocator.free(self.conv2_b);
        self.* = undefined;
    }
};

/// One decoder up-block: snake → ConvTranspose1d → 3 res units.
pub const UpBlock = struct {
    spec: BlockSpec,
    snake1_a: ChannelVec, // on in_ch
    snake1_inv_b: ChannelVec,
    conv_t_w2: ConvTWeight, // [K*OC, IC]
    conv_t_b: OutVec, // [OC]
    res: [3]ResUnit, // dilations 1, 3, 9 on out_ch

    pub fn deinit(self: *UpBlock, allocator: Allocator) void {
        self.snake1_a.deinit();
        self.snake1_inv_b.deinit();
        self.conv_t_w2.deinit();
        self.conv_t_b.deinit();
        for (&self.res) |*r| r.deinit(allocator);
        self.* = undefined;
    }
};

/// DAC decoder weights (dac-decoder.h). NO final tanh, NO clamp.
pub const DacDecoder = struct {
    conv1_w: ConvWeight, // [7, 256, 1024], pad 3
    conv1_gw: GemmConvWeight, // [1024, 7*256]
    conv1_b: []f32,
    blocks: [5]UpBlock,
    final_snake_a: ChannelVec, // 32 ch
    final_snake_inv_b: ChannelVec,
    conv2_w: ConvWeight, // [7, 32, 1], pad 3
    conv2_gw: GemmConvWeight, // [1, 7*32]
    conv2_b: []f32,

    pub fn deinit(self: *DacDecoder, allocator: Allocator) void {
        self.conv1_w.deinit();
        self.conv1_gw.deinit();
        allocator.free(self.conv1_b);
        for (&self.blocks) |*b| b.deinit(allocator);
        self.final_snake_a.deinit();
        self.final_snake_inv_b.deinit();
        self.conv2_w.deinit();
        self.conv2_gw.deinit();
        allocator.free(self.conv2_b);
        self.* = undefined;
    }
};

// --- encode-side weight structs --------------------------------------------

/// One HuBERT feature_extractor conv layer (pad 0, no bias); layer 0 carries
/// the affine GroupNorm(G == C == 512) whose GGUF name is `layer_norm`.
pub const HubertFeatLayer = struct {
    conv_w: GgmlConvWeight, // rows [512][IC*K]
    stride: usize,
    gn_w: ?[]f32, // layer 0 only
    gn_b: ?[]f32,

    pub fn deinit(self: *HubertFeatLayer, allocator: Allocator) void {
        self.conv_w.deinit(allocator);
        if (self.gn_w) |w| allocator.free(w);
        if (self.gn_b) |b| allocator.free(b);
        self.* = undefined;
    }
};

/// One HuBERT Post-LN transformer layer: MHA (all 4 projections biased) +
/// FFN (both denses biased) + the two affine LayerNorms.
pub const HubertLayer = struct {
    q_proj: llm.weights.LinearWeight,
    q_bias: []f32,
    k_proj: llm.weights.LinearWeight,
    k_bias: []f32,
    v_proj: llm.weights.LinearWeight,
    v_bias: []f32,
    out_proj: llm.weights.LinearWeight,
    out_bias: []f32,
    fc1: llm.weights.LinearWeight, // intermediate_dense 768→3072
    fc1_bias: []f32,
    fc2: llm.weights.LinearWeight, // output_dense 3072→768
    fc2_bias: []f32,
    ln_attn_w: []f32, // layer_norm (post attn add)
    ln_attn_b: []f32,
    ln_final_w: []f32, // final_layer_norm (post ffn add)
    ln_final_b: []f32,

    pub fn deinit(self: *HubertLayer, allocator: Allocator) void {
        self.q_proj.deinit();
        allocator.free(self.q_bias);
        self.k_proj.deinit();
        allocator.free(self.k_bias);
        self.v_proj.deinit();
        allocator.free(self.v_bias);
        self.out_proj.deinit();
        allocator.free(self.out_bias);
        self.fc1.deinit();
        allocator.free(self.fc1_bias);
        self.fc2.deinit();
        allocator.free(self.fc2_bias);
        allocator.free(self.ln_attn_w);
        allocator.free(self.ln_attn_b);
        allocator.free(self.ln_final_w);
        allocator.free(self.ln_final_b);
        self.* = undefined;
    }
};

/// HuBERT base semantic encoder weights (hubert-enc.h).
pub const Hubert = struct {
    feat: [hubert_feat_num_layers]HubertFeatLayer,
    fp_ln_w: []f32, // feature_projection.layer_norm (512)
    fp_ln_b: []f32,
    fp_proj: llm.weights.LinearWeight, // 512→768
    fp_proj_bias: []f32,
    /// pos_conv grouped weight, flat rows `[768][48*128]` (groups=16;
    /// weight_norm already folded in the GGUF).
    pos_conv_w: GgmlConvWeight,
    pos_conv_bias: []f32, // [768]
    enc_ln_w: []f32, // encoder.layer_norm (768)
    enc_ln_b: []f32,
    layers: [hubert_num_layers]HubertLayer,

    pub fn deinit(self: *Hubert, allocator: Allocator) void {
        for (&self.feat) |*l| l.deinit(allocator);
        allocator.free(self.fp_ln_w);
        allocator.free(self.fp_ln_b);
        self.fp_proj.deinit();
        allocator.free(self.fp_proj_bias);
        self.pos_conv_w.deinit(allocator);
        allocator.free(self.pos_conv_bias);
        allocator.free(self.enc_ln_w);
        allocator.free(self.enc_ln_b);
        for (&self.layers) |*l| l.deinit(allocator);
        self.* = undefined;
    }
};

/// SemanticEncoder res unit (semantic-enc.h): ELU→conv1(k=3, p=d, dil=d)→
/// ELU→conv2(k=1) with an outer skip; NO bias on either conv.
pub const SemanticResUnit = struct {
    conv1_w: GgmlConvWeight, // rows [768][768*3]
    conv2_w: GgmlConvWeight, // rows [768][768*1]
    dilation: usize,

    pub fn deinit(self: *SemanticResUnit, allocator: Allocator) void {
        self.conv1_w.deinit(allocator);
        self.conv2_w.deinit(allocator);
        self.* = undefined;
    }
};

/// One SemanticEncoder block: 2 res units then a k=3 p=1 conv WITH bias.
pub const SemanticBlock = struct {
    res: [2]SemanticResUnit,
    conv_w: GgmlConvWeight, // rows [768][768*3]
    conv_b: []f32,

    pub fn deinit(self: *SemanticBlock, allocator: Allocator) void {
        for (&self.res) |*r| r.deinit(allocator);
        self.conv_w.deinit(allocator);
        allocator.free(self.conv_b);
        self.* = undefined;
    }
};

/// SemanticEncoder weights: initial k=3 p=1 conv (no bias) + 2 blocks.
pub const SemanticEncoder = struct {
    conv_w: GgmlConvWeight, // rows [768][768*3], no bias
    blocks: [2]SemanticBlock,

    pub fn deinit(self: *SemanticEncoder, allocator: Allocator) void {
        self.conv_w.deinit(allocator);
        for (&self.blocks) |*b| b.deinit(allocator);
        self.* = undefined;
    }
};

/// DAC ENCODER res unit: snake→conv1(k=7, p=3d, dil=d)→snake→conv2(k=1),
/// outer skip; channels constant. Same math as the decoder's `ResUnit`, but
/// the convs use the ggml-parity f16 arithmetic.
pub const EncResUnit = struct {
    snake1_a: []f32,
    snake1_inv_b: []f32,
    conv1_w: GgmlConvWeight, // k=7, pad=3·dilation
    conv1_b: []f32,
    snake2_a: []f32,
    snake2_inv_b: []f32,
    conv2_w: GgmlConvWeight, // k=1
    conv2_b: []f32,
    dilation: usize,

    pub fn deinit(self: *EncResUnit, allocator: Allocator) void {
        allocator.free(self.snake1_a);
        allocator.free(self.snake1_inv_b);
        self.conv1_w.deinit(allocator);
        allocator.free(self.conv1_b);
        allocator.free(self.snake2_a);
        allocator.free(self.snake2_inv_b);
        self.conv2_w.deinit(allocator);
        allocator.free(self.conv2_b);
        self.* = undefined;
    }
};

/// One DAC encoder down-block: 3 res units (on `in_ch`) → block-level snake
/// → strided downsampling conv (in→out) + bias.
pub const DacEncBlock = struct {
    spec: EncBlockSpec,
    res: [3]EncResUnit, // dilations 1, 3, 9 on in_ch
    snake_a: []f32, // block.{i}.snake1.alpha, on in_ch
    snake_inv_b: []f32,
    conv_w: GgmlConvWeight, // rows [out_ch][in_ch*K]
    conv_b: []f32,

    pub fn deinit(self: *DacEncBlock, allocator: Allocator) void {
        for (&self.res) |*r| r.deinit(allocator);
        allocator.free(self.snake_a);
        allocator.free(self.snake_inv_b);
        self.conv_w.deinit(allocator);
        allocator.free(self.conv_b);
        self.* = undefined;
    }
};

/// DAC acoustic encoder weights (dac-encoder.h).
pub const DacEncoder = struct {
    conv1_w: GgmlConvWeight, // rows [64][1*7], pad 3
    conv1_b: []f32,
    blocks: [5]DacEncBlock,
    final_snake_a: []f32, // 2048 ch
    final_snake_inv_b: []f32,
    conv2_w: GgmlConvWeight, // rows [256][2048*3], pad 1
    conv2_b: []f32,

    pub fn deinit(self: *DacEncoder, allocator: Allocator) void {
        self.conv1_w.deinit(allocator);
        allocator.free(self.conv1_b);
        for (&self.blocks) |*b| b.deinit(allocator);
        allocator.free(self.final_snake_a);
        allocator.free(self.final_snake_inv_b);
        self.conv2_w.deinit(allocator);
        allocator.free(self.conv2_b);
        self.* = undefined;
    }
};

/// RVQ encode-side per-codebook projection: Linear(1024→64) + bias.
pub const ProjectIn = struct {
    weight: llm.weights.LinearWeight,
    bias: []f32, // [64]

    pub fn deinit(self: *ProjectIn, allocator: Allocator) void {
        self.weight.deinit();
        allocator.free(self.bias);
        self.* = undefined;
    }
};

/// Encode-side weights: HuBERT + SemanticEncoder + DAC encoder + RVQ
/// project_in + the joint fc (1024→1024). Loaded separately from `Codec`
/// (whose decode-side quantizers the RVQ encode chain also reads).
pub const Encoder = struct {
    allocator: Allocator,
    hubert: Hubert,
    semantic: SemanticEncoder,
    dac: DacEncoder,
    project_in: [n_codebooks]ProjectIn,
    fc: llm.weights.LinearWeight, // Linear(1024→1024)
    fc_bias: []f32,

    /// Loads the encode-side weights. Same ownership contract as
    /// `Codec.load`: keep `file` open while the `Encoder` lives.
    pub fn load(ctx: *ExecContext, file: *const gguf.File) !Encoder {
        var hubert = try loadHubert(ctx, file);
        errdefer hubert.deinit(ctx.allocator);
        var semantic = try loadSemanticEncoder(ctx, file);
        errdefer semantic.deinit(ctx.allocator);
        var dac_enc = try loadDacEncoder(ctx, file);
        errdefer dac_enc.deinit(ctx.allocator);

        var name_buf: [128]u8 = undefined;
        var project_in: [n_codebooks]ProjectIn = undefined;
        var loaded: usize = 0;
        errdefer for (project_in[0..loaded]) |*p| p.deinit(ctx.allocator);
        for (0..n_codebooks) |k| {
            project_in[k] = try loadProjectIn(ctx, file, &name_buf, k);
            loaded = k + 1;
        }

        const fc_info = try file.get("fc.weight");
        var fc = try llm.weights.LinearWeight.load(ctx, fc_info, 1024, 1024);
        errdefer fc.deinit();
        const fc_bias = try loadVectorF32(ctx.allocator, file, "fc.bias", 1024);
        errdefer ctx.allocator.free(fc_bias);

        return .{
            .allocator = ctx.allocator,
            .hubert = hubert,
            .semantic = semantic,
            .dac = dac_enc,
            .project_in = project_in,
            .fc = fc,
            .fc_bias = fc_bias,
        };
    }

    pub fn deinit(self: *Encoder) void {
        self.hubert.deinit(self.allocator);
        self.semantic.deinit(self.allocator);
        self.dac.deinit(self.allocator);
        for (&self.project_in) |*p| p.deinit(self.allocator);
        self.fc.deinit();
        self.allocator.free(self.fc_bias);
        self.* = undefined;
    }
};

/// The omnivoice-tokenizer codec model. Stage B: decode side only.
pub const Codec = struct {
    allocator: Allocator,
    config: Config,
    rvq: RvqDecoder,
    dac: DacDecoder,

    /// Parses + asserts the KV metadata and loads the decode-side weights.
    /// All slices/tensors are owned by the returned `Codec` (allocated from
    /// `ctx.allocator`); keep `file` open while the `Codec` lives (quantized
    /// linears may borrow mmapped bytes).
    pub fn load(ctx: *ExecContext, file: *const gguf.File) !Codec {
        const config = try parseConfig(file);
        var rvq = try loadRvqDecoder(ctx, file, config);
        errdefer rvq.deinit(ctx.allocator);
        var dac = try loadDacDecoder(ctx, file);
        errdefer dac.deinit(ctx.allocator);
        return .{ .allocator = ctx.allocator, .config = config, .rvq = rvq, .dac = dac };
    }

    pub fn deinit(self: *Codec) void {
        self.rvq.deinit(self.allocator);
        self.dac.deinit(self.allocator);
        self.* = undefined;
    }
};

/// Reads + asserts the consumed KVs — the reference reads only these four
/// from metadata and hardcodes every other architecture constant.
pub fn parseConfig(file: *const gguf.File) !Config {
    const arch = file.getString("general.architecture") orelse return Error.NotOmniVoiceTokenizer;
    if (!std.mem.eql(u8, arch, "omnivoice-tokenizer")) return Error.NotOmniVoiceTokenizer;

    const config = Config{
        .sample_rate = try requireKvInt(file, "omnivoice.sample_rate"),
        .hop_length = try requireKvInt(file, "omnivoice.acoustic.hop_length"),
        .codebook_size = try requireKvInt(file, "omnivoice.codebook_size"),
        .codebook_dim = try requireKvInt(file, "omnivoice.codebook_dim"),
    };
    if (config.sample_rate != 24000 or config.hop_length != 960 or
        config.codebook_size != 1024 or config.codebook_dim != 64)
    {
        return Error.UnexpectedMetadata;
    }
    return config;
}

fn requireKvInt(file: *const gguf.File, key: []const u8) !usize {
    const v = file.getInt(key) orelse return Error.UnexpectedMetadata;
    if (v <= 0) return Error.UnexpectedMetadata;
    return @intCast(v);
}

// --- decode helpers -------------------------------------------------------

fn tensorNumel(info: *const gguf.TensorInfo) usize {
    var numel: usize = 1;
    for (info.dims[0..info.n_dims]) |d| numel *= d;
    return numel;
}

/// Decode (widen/dequantize) a whole tensor to f32; caller frees.
fn decodeTensorF32(allocator: Allocator, info: *const gguf.TensorInfo, expected_numel: usize) ![]f32 {
    if (tensorNumel(info) != expected_numel) return Error.InvalidTensorShape;
    const values = try allocator.alloc(f32, expected_numel);
    errdefer allocator.free(values);
    if (comptime !builtin.cpu.arch.isAARCH64()) {
        // Scalar widen/copy is elementwise → per-thread element ranges are
        // bit-identical, and the split also parallelizes the mmap page-in
        // (the serial decode is a large share of the x86 load wall).
        if (decodeTensorF32Parallel(info, values)) return values;
    }
    try gguf.decodeF32(info.ggml_type, info.data, values);
    return values;
}

/// Split a big scalar-dtype (f32/f16/bf16) tensor decode into per-thread
/// element ranges. Returns false (caller decodes serially) for
/// block-quantized dtypes, small tensors, or a single thread.
fn decodeTensorF32Parallel(info: *const gguf.TensorInfo, dst: []f32) bool {
    const elem_bytes: usize = switch (info.ggml_type) {
        .f32 => 4,
        .f16, .bf16 => 2,
        else => return false,
    };
    if (dst.len < 1 << 19) return false;
    const max_threads = fucina.parallel.vector_max_threads;
    const want = fucina.parallel.cpuThreadCount(max_threads);
    if (want <= 1) return false;
    const Task = struct {
        ggml_type: gguf.GgmlType,
        src: []const u8,
        dst: []f32,
        fn run(task: *const @This()) void {
            // Sub-range src/dst lengths agree by construction and the scalar
            // decoders have no other failure mode.
            gguf.decodeF32(task.ggml_type, task.src, task.dst) catch unreachable;
        }
    };
    var tasks: [max_threads]Task = undefined;
    for (0..want) |i| {
        const start = i * dst.len / want;
        const end = (i + 1) * dst.len / want;
        tasks[i] = .{
            .ggml_type = info.ggml_type,
            .src = info.data[start * elem_bytes .. end * elem_bytes],
            .dst = dst[start..end],
        };
    }
    runTaskThreads(Task, Task.run, tasks[0..want]);
    return true;
}

fn loadVectorF32(allocator: Allocator, file: *const gguf.File, name: []const u8, len: usize) ![]f32 {
    const info = try file.get(name);
    return decodeTensorF32(allocator, info, len);
}

const SnakeParams = struct {
    a: ChannelVec,
    inv_b: ChannelVec,
};

const SnakeRaw = struct {
    a: []f32,
    inv_b: []f32,
};

/// As `loadSnake`, but as raw owned slices (the encode-side host snake).
fn loadSnakeRaw(allocator: Allocator, file: *const gguf.File, name: []const u8, channels: usize) !SnakeRaw {
    const alpha = try loadVectorF32(allocator, file, name, channels);
    errdefer allocator.free(alpha);
    const inv = try allocator.alloc(f32, channels);
    errdefer allocator.free(inv);
    for (inv, alpha) |*dst, a| dst.* = snakeInvB(a);
    return .{ .a = alpha, .inv_b = inv };
}

/// Loads `*.snake*.alpha` ne=(1, C, 1) as the two per-channel f32 vectors
/// `a[c] = alpha[c]` and `inv_b[c] = 1/(alpha[c] + 1e-9)`.
fn loadSnake(ctx: *ExecContext, file: *const gguf.File, name: []const u8, channels: usize) !SnakeParams {
    const alpha = try loadVectorF32(ctx.allocator, file, name, channels);
    defer ctx.allocator.free(alpha);
    const inv = try ctx.allocator.alloc(f32, channels);
    defer ctx.allocator.free(inv);
    for (inv, alpha) |*dst, a| dst.* = snakeInvB(a);

    var a_t = try ChannelVec.fromSlice(ctx, .{channels}, alpha);
    errdefer a_t.deinit();
    var inv_t = try ChannelVec.fromSlice(ctx, .{channels}, inv);
    errdefer inv_t.deinit();
    return .{ .a = a_t, .inv_b = inv_t };
}

/// Direct-kernel + im2col-GEMM representations of one decoder Conv1d weight
/// (one GGUF decode, two repacks).
const ConvWeightPair = struct {
    direct: ConvWeight,
    gemm: GemmConvWeight,
};

/// Loads a Conv1d weight (GGUF ne=(K, IC, OC)) as BOTH the `[tap, in, out]`
/// direct-kernel layout and the `[OC, K*IC]` im2col GEMM layout.
fn loadConv1dWeightPair(ctx: *ExecContext, file: *const gguf.File, name: []const u8, taps: usize, in_ch: usize, out_ch: usize) !ConvWeightPair {
    const info = try file.get(name);
    if (info.n_dims < 1 or info.dims[0] != taps) return Error.InvalidTensorShape;
    const numel = taps * in_ch * out_ch;
    const src = try decodeTensorF32(ctx.allocator, info, numel);
    defer ctx.allocator.free(src);
    const dst = try ctx.allocator.alloc(f32, numel);
    defer ctx.allocator.free(dst);

    repackConv1dWeight(dst, src, taps, in_ch, out_ch);
    var direct = try ConvWeight.fromSlice(ctx, .{ taps, in_ch, out_ch }, dst);
    errdefer direct.deinit();

    repackConv1dGemmWeight(dst, src, taps, in_ch, out_ch);
    var gemm = try GemmConvWeight.fromSlice(ctx, .{ out_ch, taps * in_ch }, dst);
    errdefer gemm.deinit();

    return .{ .direct = direct, .gemm = gemm };
}

/// Loads a ConvTranspose1d weight (GGUF ne=(K, OC, IC)) repacked to the
/// facade `weight2` `[K*OC, IC]` layout.
fn loadConvT1dWeight(ctx: *ExecContext, file: *const gguf.File, name: []const u8, taps: usize, out_ch: usize, in_ch: usize) !ConvTWeight {
    const info = try file.get(name);
    if (info.n_dims < 1 or info.dims[0] != taps) return Error.InvalidTensorShape;
    const numel = taps * out_ch * in_ch;
    const src = try decodeTensorF32(ctx.allocator, info, numel);
    defer ctx.allocator.free(src);
    const dst = try ctx.allocator.alloc(f32, numel);
    defer ctx.allocator.free(dst);
    repackConvT1dWeight(dst, src, taps, out_ch, in_ch);
    return ConvTWeight.fromSlice(ctx, .{ taps * out_ch, in_ch }, dst);
}

/// Loads ONLY the RVQ decode-side weights (quantizers + fc2). Public so the
/// encode-only CLI path can skip the (much larger) DAC decoder load — RVQ
/// encode reads the decode-side codebooks/project_out but never `Codec.dac`.
pub fn loadRvqDecoder(ctx: *ExecContext, file: *const gguf.File, config: Config) !RvqDecoder {
    const allocator = ctx.allocator;
    var name_buf: [128]u8 = undefined;

    var quantizers: [n_codebooks]Quantizer = undefined;
    var loaded: usize = 0;
    errdefer for (quantizers[0..loaded]) |*q| q.deinit(allocator);
    for (0..n_codebooks) |k| {
        quantizers[k] = try loadQuantizer(ctx, file, &name_buf, k, config);
        loaded = k + 1;
    }

    const fc2_info = try file.get("fc2.weight");
    var fc2 = try llm.weights.LinearWeight.load(ctx, fc2_info, 256, 1024);
    errdefer fc2.deinit();
    const fc2_bias = try loadVectorF32(allocator, file, "fc2.bias", 256);
    errdefer allocator.free(fc2_bias);

    return .{ .quantizers = quantizers, .fc2 = fc2, .fc2_bias = fc2_bias };
}

fn loadQuantizer(ctx: *ExecContext, file: *const gguf.File, name_buf: []u8, k: usize, config: Config) !Quantizer {
    const allocator = ctx.allocator;
    const v = config.codebook_size;
    const d = config.codebook_dim;

    // codebook.embed: GGUF ne=(64, 1024), buffer j*64+d row-major — already
    // our [V, d] layout.
    const embed_name = try std.fmt.bufPrint(name_buf, "quantizer.quantizers.{d}.codebook.embed", .{k});
    const embed_info = try file.get(embed_name);
    const embed_values = try decodeTensorF32(allocator, embed_info, v * d);
    defer allocator.free(embed_values);

    var embed = try Codebook.fromSlice(ctx, .{ v, d }, embed_values);
    errdefer embed.deinit();

    // embed_sq from the DECODED values in f32 (rvq-codec.h:83-96).
    const embed_sq = try allocator.alloc(f32, v);
    errdefer allocator.free(embed_sq);
    for (embed_sq, 0..) |*dst, j| {
        var sum: f32 = 0.0;
        for (embed_values[j * d ..][0..d]) |x| sum += x * x;
        dst.* = sum;
    }

    const w_name = try std.fmt.bufPrint(name_buf, "quantizer.quantizers.{d}.project_out.weight", .{k});
    const w_info = try file.get(w_name);
    var project_out = try llm.weights.LinearWeight.load(ctx, w_info, 1024, 64);
    errdefer project_out.deinit();

    const b_name = try std.fmt.bufPrint(name_buf, "quantizer.quantizers.{d}.project_out.bias", .{k});
    const project_out_bias = try loadVectorF32(allocator, file, b_name, 1024);
    errdefer allocator.free(project_out_bias);

    return .{
        .embed = embed,
        .embed_sq = embed_sq,
        .project_out = project_out,
        .project_out_bias = project_out_bias,
    };
}

fn loadDacDecoder(ctx: *ExecContext, file: *const gguf.File) !DacDecoder {
    const allocator = ctx.allocator;
    var name_buf: [128]u8 = undefined;

    var conv1 = try loadConv1dWeightPair(ctx, file, "acoustic_decoder.conv1.weight", 7, 256, 1024);
    errdefer {
        conv1.direct.deinit();
        conv1.gemm.deinit();
    }
    const conv1_b = try loadVectorF32(allocator, file, "acoustic_decoder.conv1.bias", 1024);
    errdefer allocator.free(conv1_b);

    var blocks: [5]UpBlock = undefined;
    var loaded: usize = 0;
    errdefer for (blocks[0..loaded]) |*b| b.deinit(allocator);
    for (0..5) |i| {
        blocks[i] = try loadUpBlock(ctx, file, &name_buf, i);
        loaded = i + 1;
    }

    var final_snake = try loadSnake(ctx, file, "acoustic_decoder.snake1.alpha", 32);
    errdefer {
        final_snake.a.deinit();
        final_snake.inv_b.deinit();
    }
    var conv2 = try loadConv1dWeightPair(ctx, file, "acoustic_decoder.conv2.weight", 7, 32, 1);
    errdefer {
        conv2.direct.deinit();
        conv2.gemm.deinit();
    }
    const conv2_b = try loadVectorF32(allocator, file, "acoustic_decoder.conv2.bias", 1);
    errdefer allocator.free(conv2_b);

    return .{
        .conv1_w = conv1.direct,
        .conv1_gw = conv1.gemm,
        .conv1_b = conv1_b,
        .blocks = blocks,
        .final_snake_a = final_snake.a,
        .final_snake_inv_b = final_snake.inv_b,
        .conv2_w = conv2.direct,
        .conv2_gw = conv2.gemm,
        .conv2_b = conv2_b,
    };
}

fn loadUpBlock(ctx: *ExecContext, file: *const gguf.File, name_buf: []u8, i: usize) !UpBlock {
    const allocator = ctx.allocator;
    const spec = dac_block_specs[i];

    const snake_name = try std.fmt.bufPrint(name_buf, "acoustic_decoder.block.{d}.snake1.alpha", .{i});
    var snake1 = try loadSnake(ctx, file, snake_name, spec.in_ch);
    errdefer {
        snake1.a.deinit();
        snake1.inv_b.deinit();
    }

    const w_name = try std.fmt.bufPrint(name_buf, "acoustic_decoder.block.{d}.conv_t1.weight", .{i});
    var conv_t_w2 = try loadConvT1dWeight(ctx, file, w_name, spec.taps, spec.out_ch, spec.in_ch);
    errdefer conv_t_w2.deinit();

    const b_name = try std.fmt.bufPrint(name_buf, "acoustic_decoder.block.{d}.conv_t1.bias", .{i});
    const bias_values = try loadVectorF32(allocator, file, b_name, spec.out_ch);
    defer allocator.free(bias_values);
    var conv_t_b = try OutVec.fromSlice(ctx, .{spec.out_ch}, bias_values);
    errdefer conv_t_b.deinit();

    var res: [3]ResUnit = undefined;
    var loaded: usize = 0;
    errdefer for (res[0..loaded]) |*r| r.deinit(allocator);
    for (0..3) |r| {
        res[r] = try loadResUnit(ctx, file, name_buf, i, r, spec.out_ch);
        loaded = r + 1;
    }

    return .{
        .spec = spec,
        .snake1_a = snake1.a,
        .snake1_inv_b = snake1.inv_b,
        .conv_t_w2 = conv_t_w2,
        .conv_t_b = conv_t_b,
        .res = res,
    };
}

fn loadResUnit(ctx: *ExecContext, file: *const gguf.File, name_buf: []u8, block_i: usize, r: usize, channels: usize) !ResUnit {
    const allocator = ctx.allocator;
    // Res units are ONE-indexed with no dot: res_unit1..res_unit3.
    const unit = r + 1;

    const s1_name = try std.fmt.bufPrint(name_buf, "acoustic_decoder.block.{d}.res_unit{d}.snake1.alpha", .{ block_i, unit });
    var snake1 = try loadSnake(ctx, file, s1_name, channels);
    errdefer {
        snake1.a.deinit();
        snake1.inv_b.deinit();
    }

    const w1_name = try std.fmt.bufPrint(name_buf, "acoustic_decoder.block.{d}.res_unit{d}.conv1.weight", .{ block_i, unit });
    var conv1 = try loadConv1dWeightPair(ctx, file, w1_name, 7, channels, channels);
    errdefer {
        conv1.direct.deinit();
        conv1.gemm.deinit();
    }
    const b1_name = try std.fmt.bufPrint(name_buf, "acoustic_decoder.block.{d}.res_unit{d}.conv1.bias", .{ block_i, unit });
    const conv1_b = try loadVectorF32(allocator, file, b1_name, channels);
    errdefer allocator.free(conv1_b);

    const s2_name = try std.fmt.bufPrint(name_buf, "acoustic_decoder.block.{d}.res_unit{d}.snake2.alpha", .{ block_i, unit });
    var snake2 = try loadSnake(ctx, file, s2_name, channels);
    errdefer {
        snake2.a.deinit();
        snake2.inv_b.deinit();
    }

    const w2_name = try std.fmt.bufPrint(name_buf, "acoustic_decoder.block.{d}.res_unit{d}.conv2.weight", .{ block_i, unit });
    var conv2 = try loadConv1dWeightPair(ctx, file, w2_name, 1, channels, channels);
    errdefer {
        conv2.direct.deinit();
        conv2.gemm.deinit();
    }
    const b2_name = try std.fmt.bufPrint(name_buf, "acoustic_decoder.block.{d}.res_unit{d}.conv2.bias", .{ block_i, unit });
    const conv2_b = try loadVectorF32(allocator, file, b2_name, channels);
    errdefer allocator.free(conv2_b);

    return .{
        .snake1_a = snake1.a,
        .snake1_inv_b = snake1.inv_b,
        .conv1_w = conv1.direct,
        .conv1_gw = conv1.gemm,
        .conv1_b = conv1_b,
        .snake2_a = snake2.a,
        .snake2_inv_b = snake2.inv_b,
        .conv2_w = conv2.direct,
        .conv2_gw = conv2.gemm,
        .conv2_b = conv2_b,
        .dilation = res_unit_dilations[r],
    };
}

// --- encode-side loaders ----------------------------------------------------

/// Loads a conv weight as ggml-parity f16 rows in the ORIGINAL GGUF buffer
/// order `w[oc][ic_pg][k]` (no repack; gf_load_conv_f16's decode→f16 cast).
fn loadGgmlConvWeight(
    ctx: *ExecContext,
    file: *const gguf.File,
    name: []const u8,
    taps: usize,
    in_per_group: usize,
    out_ch: usize,
    groups: usize,
) !GgmlConvWeight {
    const allocator = ctx.allocator;
    const info = try file.get(name);
    if (info.n_dims < 1 or info.dims[0] != taps) return Error.InvalidTensorShape;
    const numel = taps * in_per_group * out_ch;
    const src = try decodeTensorF32(allocator, info, numel);
    defer allocator.free(src);
    const data = try allocator.alloc(f16, numel);
    errdefer allocator.free(data);
    castF16Rows(data, src);
    return .{
        .data = data,
        .taps = taps,
        .in_per_group = in_per_group,
        .out_ch = out_ch,
        .groups = groups,
    };
}

/// f32 → f16 weight cast, split across raw threads for big tensors
/// (elementwise → bit-identical to the serial loop).
fn castF16Rows(data: []f16, src: []const f32) void {
    std.debug.assert(data.len == src.len);
    if (comptime !builtin.cpu.arch.isAARCH64()) {
        const max_threads = fucina.parallel.vector_max_threads;
        const want = fucina.parallel.cpuThreadCount(max_threads);
        if (data.len >= 1 << 19 and want > 1) {
            const Task = struct {
                dst: []f16,
                src: []const f32,
                fn run(task: *const @This()) void {
                    for (task.dst, task.src) |*dst, v| dst.* = @floatCast(v);
                }
            };
            var tasks: [max_threads]Task = undefined;
            for (0..want) |i| {
                const start = i * data.len / want;
                const end = (i + 1) * data.len / want;
                tasks[i] = .{ .dst = data[start..end], .src = src[start..end] };
            }
            runTaskThreads(Task, Task.run, tasks[0..want]);
            return;
        }
    }
    for (data, src) |*dst, v| dst.* = @floatCast(v);
}

fn loadHubert(ctx: *ExecContext, file: *const gguf.File) !Hubert {
    const allocator = ctx.allocator;
    var name_buf: [128]u8 = undefined;

    var feat: [hubert_feat_num_layers]HubertFeatLayer = undefined;
    var feat_loaded: usize = 0;
    errdefer for (feat[0..feat_loaded]) |*l| l.deinit(allocator);
    var prev_dim: usize = 1;
    for (0..hubert_feat_num_layers) |i| {
        const w_name = try std.fmt.bufPrint(&name_buf, "semantic_model.feature_extractor.conv_layers.{d}.conv.weight", .{i});
        var conv_w = try loadGgmlConvWeight(ctx, file, w_name, hubert_feat_kernels[i], prev_dim, hubert_feat_dim, 1);
        errdefer conv_w.deinit(allocator);

        // Layer 0 only: affine GroupNorm(G == C) — HF names it layer_norm.
        var gn_w: ?[]f32 = null;
        errdefer if (gn_w) |w| allocator.free(w);
        var gn_b: ?[]f32 = null;
        errdefer if (gn_b) |b| allocator.free(b);
        if (i == 0) {
            gn_w = try loadVectorF32(allocator, file, "semantic_model.feature_extractor.conv_layers.0.layer_norm.weight", hubert_feat_dim);
            gn_b = try loadVectorF32(allocator, file, "semantic_model.feature_extractor.conv_layers.0.layer_norm.bias", hubert_feat_dim);
        }

        feat[i] = .{ .conv_w = conv_w, .stride = hubert_feat_strides[i], .gn_w = gn_w, .gn_b = gn_b };
        feat_loaded = i + 1;
        prev_dim = hubert_feat_dim;
    }

    const fp_ln_w = try loadVectorF32(allocator, file, "semantic_model.feature_projection.layer_norm.weight", hubert_feat_dim);
    errdefer allocator.free(fp_ln_w);
    const fp_ln_b = try loadVectorF32(allocator, file, "semantic_model.feature_projection.layer_norm.bias", hubert_feat_dim);
    errdefer allocator.free(fp_ln_b);
    const fp_proj_info = try file.get("semantic_model.feature_projection.projection.weight");
    var fp_proj = try llm.weights.LinearWeight.load(ctx, fp_proj_info, hubert_hidden, hubert_feat_dim);
    errdefer fp_proj.deinit();
    const fp_proj_bias = try loadVectorF32(allocator, file, "semantic_model.feature_projection.projection.bias", hubert_hidden);
    errdefer allocator.free(fp_proj_bias);

    // pos_conv grouped weight: GGUF ne=(128, 48, 768) — rows [oc][ic_pg][k]
    // with oc global over 768; group g of output channels reads input
    // channels [48g, 48(g+1)), exactly the reference's per-group slices.
    var pos_conv_w = try loadGgmlConvWeight(ctx, file, "semantic_model.encoder.pos_conv_embed.conv.weight", hubert_pos_k, hubert_pos_ic_pg, hubert_hidden, hubert_pos_groups);
    errdefer pos_conv_w.deinit(allocator);
    const pos_conv_bias = try loadVectorF32(allocator, file, "semantic_model.encoder.pos_conv_embed.conv.bias", hubert_hidden);
    errdefer allocator.free(pos_conv_bias);

    const enc_ln_w = try loadVectorF32(allocator, file, "semantic_model.encoder.layer_norm.weight", hubert_hidden);
    errdefer allocator.free(enc_ln_w);
    const enc_ln_b = try loadVectorF32(allocator, file, "semantic_model.encoder.layer_norm.bias", hubert_hidden);
    errdefer allocator.free(enc_ln_b);

    var layers: [hubert_num_layers]HubertLayer = undefined;
    var loaded: usize = 0;
    errdefer for (layers[0..loaded]) |*l| l.deinit(allocator);
    for (0..hubert_num_layers) |i| {
        layers[i] = try loadHubertLayer(ctx, file, &name_buf, i);
        loaded = i + 1;
    }

    return .{
        .feat = feat,
        .fp_ln_w = fp_ln_w,
        .fp_ln_b = fp_ln_b,
        .fp_proj = fp_proj,
        .fp_proj_bias = fp_proj_bias,
        .pos_conv_w = pos_conv_w,
        .pos_conv_bias = pos_conv_bias,
        .enc_ln_w = enc_ln_w,
        .enc_ln_b = enc_ln_b,
        .layers = layers,
    };
}

fn loadHubertLinear(ctx: *ExecContext, file: *const gguf.File, name_buf: []u8, layer_i: usize, suffix: []const u8, out_dim: usize, in_dim: usize) !llm.weights.LinearWeight {
    const name = try std.fmt.bufPrint(name_buf, "semantic_model.encoder.layers.{d}.{s}.weight", .{ layer_i, suffix });
    const info = try file.get(name);
    return llm.weights.LinearWeight.load(ctx, info, out_dim, in_dim);
}

fn loadHubertBias(ctx: *ExecContext, file: *const gguf.File, name_buf: []u8, layer_i: usize, suffix: []const u8, len: usize) ![]f32 {
    const name = try std.fmt.bufPrint(name_buf, "semantic_model.encoder.layers.{d}.{s}.bias", .{ layer_i, suffix });
    return loadVectorF32(ctx.allocator, file, name, len);
}

fn loadHubertLn(ctx: *ExecContext, file: *const gguf.File, name_buf: []u8, layer_i: usize, suffix: []const u8) ![]f32 {
    const name = try std.fmt.bufPrint(name_buf, "semantic_model.encoder.layers.{d}.{s}", .{ layer_i, suffix });
    return loadVectorF32(ctx.allocator, file, name, hubert_hidden);
}

fn loadHubertLayer(ctx: *ExecContext, file: *const gguf.File, name_buf: []u8, i: usize) !HubertLayer {
    const allocator = ctx.allocator;
    const h = hubert_hidden;

    var q_proj = try loadHubertLinear(ctx, file, name_buf, i, "attention.q_proj", h, h);
    errdefer q_proj.deinit();
    const q_bias = try loadHubertBias(ctx, file, name_buf, i, "attention.q_proj", h);
    errdefer allocator.free(q_bias);
    var k_proj = try loadHubertLinear(ctx, file, name_buf, i, "attention.k_proj", h, h);
    errdefer k_proj.deinit();
    const k_bias = try loadHubertBias(ctx, file, name_buf, i, "attention.k_proj", h);
    errdefer allocator.free(k_bias);
    var v_proj = try loadHubertLinear(ctx, file, name_buf, i, "attention.v_proj", h, h);
    errdefer v_proj.deinit();
    const v_bias = try loadHubertBias(ctx, file, name_buf, i, "attention.v_proj", h);
    errdefer allocator.free(v_bias);
    var out_proj = try loadHubertLinear(ctx, file, name_buf, i, "attention.out_proj", h, h);
    errdefer out_proj.deinit();
    const out_bias = try loadHubertBias(ctx, file, name_buf, i, "attention.out_proj", h);
    errdefer allocator.free(out_bias);

    var fc1 = try loadHubertLinear(ctx, file, name_buf, i, "feed_forward.intermediate_dense", hubert_ffn_inner, h);
    errdefer fc1.deinit();
    const fc1_bias = try loadHubertBias(ctx, file, name_buf, i, "feed_forward.intermediate_dense", hubert_ffn_inner);
    errdefer allocator.free(fc1_bias);
    var fc2 = try loadHubertLinear(ctx, file, name_buf, i, "feed_forward.output_dense", h, hubert_ffn_inner);
    errdefer fc2.deinit();
    const fc2_bias = try loadHubertBias(ctx, file, name_buf, i, "feed_forward.output_dense", h);
    errdefer allocator.free(fc2_bias);

    const ln_attn_w = try loadHubertLn(ctx, file, name_buf, i, "layer_norm.weight");
    errdefer allocator.free(ln_attn_w);
    const ln_attn_b = try loadHubertLn(ctx, file, name_buf, i, "layer_norm.bias");
    errdefer allocator.free(ln_attn_b);
    const ln_final_w = try loadHubertLn(ctx, file, name_buf, i, "final_layer_norm.weight");
    errdefer allocator.free(ln_final_w);
    const ln_final_b = try loadHubertLn(ctx, file, name_buf, i, "final_layer_norm.bias");
    errdefer allocator.free(ln_final_b);

    return .{
        .q_proj = q_proj,
        .q_bias = q_bias,
        .k_proj = k_proj,
        .k_bias = k_bias,
        .v_proj = v_proj,
        .v_bias = v_bias,
        .out_proj = out_proj,
        .out_bias = out_bias,
        .fc1 = fc1,
        .fc1_bias = fc1_bias,
        .fc2 = fc2,
        .fc2_bias = fc2_bias,
        .ln_attn_w = ln_attn_w,
        .ln_attn_b = ln_attn_b,
        .ln_final_w = ln_final_w,
        .ln_final_b = ln_final_b,
    };
}

fn loadSemanticEncoder(ctx: *ExecContext, file: *const gguf.File) !SemanticEncoder {
    const allocator = ctx.allocator;
    const c = semantic_hidden;
    var name_buf: [128]u8 = undefined;

    var conv_w = try loadGgmlConvWeight(ctx, file, "encoder_semantic.conv.weight", 3, c, c, 1);
    errdefer conv_w.deinit(allocator);

    var blocks: [2]SemanticBlock = undefined;
    var loaded: usize = 0;
    errdefer for (blocks[0..loaded]) |*b| b.deinit(allocator);
    for (0..2) |i| {
        // Res units are ZERO-indexed WITH dot: res_units.0 / res_units.1
        // (unlike the DAC family's res_unit1..3), dilations {1, 1}.
        var res: [2]SemanticResUnit = undefined;
        var res_loaded: usize = 0;
        errdefer for (res[0..res_loaded]) |*r| r.deinit(allocator);
        for (0..2) |r| {
            const w1_name = try std.fmt.bufPrint(&name_buf, "encoder_semantic.conv_blocks.{d}.res_units.{d}.conv1.weight", .{ i, r });
            var c1 = try loadGgmlConvWeight(ctx, file, w1_name, 3, c, c, 1);
            errdefer c1.deinit(allocator);
            const w2_name = try std.fmt.bufPrint(&name_buf, "encoder_semantic.conv_blocks.{d}.res_units.{d}.conv2.weight", .{ i, r });
            var c2 = try loadGgmlConvWeight(ctx, file, w2_name, 1, c, c, 1);
            errdefer c2.deinit(allocator);
            res[r] = .{ .conv1_w = c1, .conv2_w = c2, .dilation = 1 };
            res_loaded = r + 1;
        }

        const bw_name = try std.fmt.bufPrint(&name_buf, "encoder_semantic.conv_blocks.{d}.conv.weight", .{i});
        var block_w = try loadGgmlConvWeight(ctx, file, bw_name, 3, c, c, 1);
        errdefer block_w.deinit(allocator);
        const bb_name = try std.fmt.bufPrint(&name_buf, "encoder_semantic.conv_blocks.{d}.conv.bias", .{i});
        const block_b = try loadVectorF32(allocator, file, bb_name, c);
        errdefer allocator.free(block_b);

        blocks[i] = .{ .res = res, .conv_w = block_w, .conv_b = block_b };
        loaded = i + 1;
    }

    return .{ .conv_w = conv_w, .blocks = blocks };
}

fn loadDacEncoder(ctx: *ExecContext, file: *const gguf.File) !DacEncoder {
    const allocator = ctx.allocator;
    var name_buf: [128]u8 = undefined;

    var conv1_w = try loadGgmlConvWeight(ctx, file, "acoustic_encoder.conv1.weight", 7, 1, 64, 1);
    errdefer conv1_w.deinit(allocator);
    const conv1_b = try loadVectorF32(allocator, file, "acoustic_encoder.conv1.bias", 64);
    errdefer allocator.free(conv1_b);

    var blocks: [5]DacEncBlock = undefined;
    var loaded: usize = 0;
    errdefer for (blocks[0..loaded]) |*b| b.deinit(allocator);
    for (0..5) |i| {
        blocks[i] = try loadDacEncBlock(ctx, file, &name_buf, i);
        loaded = i + 1;
    }

    const final_snake = try loadSnakeRaw(allocator, file, "acoustic_encoder.snake1.alpha", 2048);
    errdefer {
        allocator.free(final_snake.a);
        allocator.free(final_snake.inv_b);
    }
    var conv2_w = try loadGgmlConvWeight(ctx, file, "acoustic_encoder.conv2.weight", 3, 2048, 256, 1);
    errdefer conv2_w.deinit(allocator);
    const conv2_b = try loadVectorF32(allocator, file, "acoustic_encoder.conv2.bias", 256);
    errdefer allocator.free(conv2_b);

    return .{
        .conv1_w = conv1_w,
        .conv1_b = conv1_b,
        .blocks = blocks,
        .final_snake_a = final_snake.a,
        .final_snake_inv_b = final_snake.inv_b,
        .conv2_w = conv2_w,
        .conv2_b = conv2_b,
    };
}

fn loadDacEncBlock(ctx: *ExecContext, file: *const gguf.File, name_buf: []u8, i: usize) !DacEncBlock {
    const allocator = ctx.allocator;
    const spec = dac_enc_block_specs[i];

    // Res units operate on in_ch; naming trap: `block.{i}.snake1.alpha` and
    // `block.{i}.conv1.weight` (no res_unit segment) are the BLOCK-level
    // post-residual snake and the strided downsampling conv.
    var res: [3]EncResUnit = undefined;
    var loaded: usize = 0;
    errdefer for (res[0..loaded]) |*r| r.deinit(allocator);
    for (0..3) |r| {
        res[r] = try loadEncResUnit(ctx, file, name_buf, i, r, spec.in_ch);
        loaded = r + 1;
    }

    const snake_name = try std.fmt.bufPrint(name_buf, "acoustic_encoder.block.{d}.snake1.alpha", .{i});
    const snake_post = try loadSnakeRaw(allocator, file, snake_name, spec.in_ch);
    errdefer {
        allocator.free(snake_post.a);
        allocator.free(snake_post.inv_b);
    }

    const w_name = try std.fmt.bufPrint(name_buf, "acoustic_encoder.block.{d}.conv1.weight", .{i});
    var conv_w = try loadGgmlConvWeight(ctx, file, w_name, spec.taps, spec.in_ch, spec.out_ch, 1);
    errdefer conv_w.deinit(allocator);
    const b_name = try std.fmt.bufPrint(name_buf, "acoustic_encoder.block.{d}.conv1.bias", .{i});
    const conv_b = try loadVectorF32(allocator, file, b_name, spec.out_ch);
    errdefer allocator.free(conv_b);

    return .{
        .spec = spec,
        .res = res,
        .snake_a = snake_post.a,
        .snake_inv_b = snake_post.inv_b,
        .conv_w = conv_w,
        .conv_b = conv_b,
    };
}

/// Loads one DAC ENCODER res unit (`acoustic_encoder.block.{i}.res_unit{r+1}`,
/// one-indexed no dot) with ggml-parity conv weights.
fn loadEncResUnit(ctx: *ExecContext, file: *const gguf.File, name_buf: []u8, block_i: usize, r: usize, channels: usize) !EncResUnit {
    const allocator = ctx.allocator;
    const unit = r + 1;

    const s1_name = try std.fmt.bufPrint(name_buf, "acoustic_encoder.block.{d}.res_unit{d}.snake1.alpha", .{ block_i, unit });
    const snake1 = try loadSnakeRaw(allocator, file, s1_name, channels);
    errdefer {
        allocator.free(snake1.a);
        allocator.free(snake1.inv_b);
    }

    const w1_name = try std.fmt.bufPrint(name_buf, "acoustic_encoder.block.{d}.res_unit{d}.conv1.weight", .{ block_i, unit });
    var conv1_w = try loadGgmlConvWeight(ctx, file, w1_name, 7, channels, channels, 1);
    errdefer conv1_w.deinit(allocator);
    const b1_name = try std.fmt.bufPrint(name_buf, "acoustic_encoder.block.{d}.res_unit{d}.conv1.bias", .{ block_i, unit });
    const conv1_b = try loadVectorF32(allocator, file, b1_name, channels);
    errdefer allocator.free(conv1_b);

    const s2_name = try std.fmt.bufPrint(name_buf, "acoustic_encoder.block.{d}.res_unit{d}.snake2.alpha", .{ block_i, unit });
    const snake2 = try loadSnakeRaw(allocator, file, s2_name, channels);
    errdefer {
        allocator.free(snake2.a);
        allocator.free(snake2.inv_b);
    }

    const w2_name = try std.fmt.bufPrint(name_buf, "acoustic_encoder.block.{d}.res_unit{d}.conv2.weight", .{ block_i, unit });
    var conv2_w = try loadGgmlConvWeight(ctx, file, w2_name, 1, channels, channels, 1);
    errdefer conv2_w.deinit(allocator);
    const b2_name = try std.fmt.bufPrint(name_buf, "acoustic_encoder.block.{d}.res_unit{d}.conv2.bias", .{ block_i, unit });
    const conv2_b = try loadVectorF32(allocator, file, b2_name, channels);
    errdefer allocator.free(conv2_b);

    return .{
        .snake1_a = snake1.a,
        .snake1_inv_b = snake1.inv_b,
        .conv1_w = conv1_w,
        .conv1_b = conv1_b,
        .snake2_a = snake2.a,
        .snake2_inv_b = snake2.inv_b,
        .conv2_w = conv2_w,
        .conv2_b = conv2_b,
        .dilation = res_unit_dilations[r],
    };
}

fn loadProjectIn(ctx: *ExecContext, file: *const gguf.File, name_buf: []u8, k: usize) !ProjectIn {
    const w_name = try std.fmt.bufPrint(name_buf, "quantizer.quantizers.{d}.project_in.weight", .{k});
    const w_info = try file.get(w_name);
    var weight = try llm.weights.LinearWeight.load(ctx, w_info, 64, 1024);
    errdefer weight.deinit();

    const b_name = try std.fmt.bufPrint(name_buf, "quantizer.quantizers.{d}.project_in.bias", .{k});
    const bias = try loadVectorF32(ctx.allocator, file, b_name, 64);
    errdefer ctx.allocator.free(bias);

    return .{ .weight = weight, .bias = bias };
}

test {
    _ = @import("codec_tests.zig");
}
