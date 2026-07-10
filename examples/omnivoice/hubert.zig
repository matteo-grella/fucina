//! HuBERT base semantic feature extractor forward (hubert-enc.h +
//! pipeline-codec.cpp `pipeline_codec_hubert_features_test`): 16 kHz mono f32
//! (pre-resampled; this module zero-pads 160 samples each side) →
//! feature_extractor (7 valid convs, GroupNorm+GELU-erf on layer 0, GELU-erf
//! on the rest) → feature_projection (LN + Linear 512→768) → pos_conv_embed
//! (grouped k=128 conv, drop the trailing frame, GELU-erf) + residual +
//! encoder LayerNorm → 12 Post-LN transformer layers → mean of the 13 hidden
//! states → decimate-by-2 (keep frames 0, 2, 4, …) → `[T_s, 768]` @ 25 Hz.
//!
//! PARITY: every stage reproduces the reference's exact arithmetic — convs
//! run the ggml f16 im2col + vec_dot_f16 path (`codec.ggmlConv1d`),
//! GroupNorm/LayerNorm/GELU/softmax are the ggml CPU kernels ported
//! operation-for-operation (incl. the Accelerate vDSP reductions ggml_norm
//! uses and ggml's v_expf softmax), the Linears go through the same
//! Accelerate cblas_sgemm call shapes the reference's ggml-BLAS backend
//! emits, and attention runs per-head sgemm pairs exactly like the
//! reference's mul_mat decomposition. Internal layout is `[T, C]` rows,
//! matching the reference dumps' (T, C) tap layout directly.

const std = @import("std");
const builtin = @import("builtin");
const fucina = @import("fucina");
const llm = @import("fucina_llm");

const codec = @import("codec.zig");

const ExecContext = fucina.ExecContext;
const Allocator = std.mem.Allocator;

pub const Error = error{InputTooShort};

/// Conv-stage activation rows `[T, C]` (channel axis tagged `.in`).
pub const Act = fucina.Tensor(.{ .seq, .in });

/// Zero padding applied to the 16 kHz input on EACH side
/// (pipeline-codec.cpp:362-377).
pub const pad_each_side = 160;
/// Time decimation factor after the 13-state mean.
pub const downsample_factor = 2;
const n_states = codec.hubert_num_layers + 1; // 13
const hidden = codec.hubert_hidden;

// CBLAS (Accelerate): the reference's ggml-BLAS backend computes every f32
// mul_mat as cblas_sgemm(RowMajor, NoTrans, Trans, ...); per-head attention
// GEMMs below reproduce those calls exactly.
const cblas_row_major: c_int = 101;
const cblas_no_trans: c_int = 111;
const cblas_trans: c_int = 112;
const cblas = if (codec.use_accelerate) struct {
    extern "c" fn cblas_sgemm(
        order: c_int,
        trans_a: c_int,
        trans_b: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: f32,
        a: [*]const f32,
        lda: c_int,
        b: [*]const f32,
        ldb: c_int,
        beta: f32,
        c: [*]f32,
        ldc: c_int,
    ) void;
} else struct {};

/// `C[m,n] = A[m,k]·B[n,k]ᵀ` — the exact cblas_sgemm(RowMajor, NoTrans,
/// Trans, …) call the reference's ggml-BLAS backend emits for every f32
/// mul_mat (Accelerate); plain host loop fallback without Accelerate.
/// Non-aarch64 uses the hand-vectorized fallback below instead — LLVM keeps
/// the naive reduction loop SCALAR (f32 reductions can't reassociate), and
/// that serial loop dominated the x86 encode wall via per-head attention.
/// The non-Accelerate arm was never bit-parity with the macOS reference
/// (see `codec.use_accelerate`); tolerance tests absorb the order change.
fn sgemmTransB(m: usize, n: usize, k: usize, a: []const f32, b: []const f32, c: []f32) void {
    if (comptime codec.use_accelerate) {
        cblas.cblas_sgemm(cblas_row_major, cblas_no_trans, cblas_trans, @intCast(m), @intCast(n), @intCast(k), 1.0, a.ptr, @intCast(k), b.ptr, @intCast(k), 0.0, c.ptr, @intCast(n));
        return;
    }
    if (comptime !builtin.cpu.arch.isAARCH64()) {
        return sgemmTransBWide(m, n, k, a, b, c);
    }
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f32 = 0.0;
            for (0..k) |kk| sum += a[i * k + kk] * b[j * k + kk];
            c[i * n + j] = sum;
        }
    }
}

/// 8-wide f32 FMA TransB GEMM fallback, 4 output columns per strip sharing
/// each A-row load (the attention shapes are small — m,n ≤ a few hundred,
/// k ∈ {64, t} — so a simple register-blocked dot ladder suffices).
fn sgemmTransBWide(m: usize, n: usize, k: usize, a: []const f32, b: []const f32, c: []f32) void {
    const V = @Vector(8, f32);
    const kp = k & ~@as(usize, 7);
    for (0..m) |i| {
        const a_row = a[i * k ..][0..k];
        var j: usize = 0;
        while (j + 4 <= n) : (j += 4) {
            var acc: [4]V = .{ @splat(0), @splat(0), @splat(0), @splat(0) };
            var p: usize = 0;
            while (p < kp) : (p += 8) {
                const av: V = a_row[p..][0..8].*;
                inline for (0..4) |jj| {
                    acc[jj] = @mulAdd(V, av, b[(j + jj) * k + p ..][0..8].*, acc[jj]);
                }
            }
            var sums: [4]f32 = undefined;
            inline for (0..4) |jj| sums[jj] = @reduce(.Add, acc[jj]);
            while (p < k) : (p += 1) {
                inline for (0..4) |jj| {
                    sums[jj] += a_row[p] * b[(j + jj) * k + p];
                }
            }
            c[i * n + j ..][0..4].* = sums;
        }
        while (j < n) : (j += 1) {
            var acc: V = @splat(0);
            var p: usize = 0;
            while (p < kp) : (p += 8) {
                const av: V = a_row[p..][0..8].*;
                acc = @mulAdd(V, av, b[j * k + p ..][0..8].*, acc);
            }
            var sum = @reduce(.Add, acc);
            while (p < k) : (p += 1) sum += a_row[p] * b[j * k + p];
            c[i * n + j] = sum;
        }
    }
}

/// One captured stage: `data` is `[t, c]` row-major.
pub const Tap = struct {
    t: usize,
    c: usize,
    data: []f32,
};

/// Per-stage taps for parity dumps (filled when passed to `forward`).
/// Matches the reference's hubert-feat-extract / -proj-ln / -proj /
/// -enc-init / -l{i} dump points (each dumped as (T, C)).
pub const Taps = struct {
    allocator: Allocator,
    feat_extract: ?Tap = null,
    proj_ln: ?Tap = null,
    proj: ?Tap = null,
    enc_init: ?Tap = null,
    /// `layer[i]` = states[i+1] (the output of transformer layer i).
    layer: [codec.hubert_num_layers]?Tap = @splat(null),

    pub fn init(allocator: Allocator) Taps {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Taps) void {
        if (self.feat_extract) |tap| self.allocator.free(tap.data);
        if (self.proj_ln) |tap| self.allocator.free(tap.data);
        if (self.proj) |tap| self.allocator.free(tap.data);
        if (self.enc_init) |tap| self.allocator.free(tap.data);
        for (self.layer) |maybe_tap| {
            if (maybe_tap) |tap| self.allocator.free(tap.data);
        }
        self.* = undefined;
    }
};

/// Output frame count of the 7-layer feature extractor for `n_samples` of
/// (already padded) 16 kHz input: per layer `T = (T - K)/S + 1` (valid
/// padding), cumulative stride 320.
pub fn featOutputLength(n_samples: usize) ?usize {
    var t = n_samples;
    for (codec.hubert_feat_kernels, codec.hubert_feat_strides) |k, s| {
        if (t < k) return null;
        t = (t - k) / s + 1;
    }
    return t;
}

/// Runs the full HuBERT features pipeline on UNPADDED 16 kHz mono samples
/// (the ±160 zero pad is applied here) and returns the mean+decimated
/// features `[T_s, 768]` tagged `[.seq, .in]`, ready for the SemanticEncoder
/// conv stack.
pub fn forward(ctx: *ExecContext, hub: *const codec.Hubert, audio_16k: []const f32, taps: ?*Taps) !Act {
    const allocator = ctx.allocator;
    const eps = codec.hubert_ln_eps;

    // --- input prep: zero-pad 160 samples on each side --------------------
    const n_padded = audio_16k.len + 2 * pad_each_side;
    const padded = try allocator.alloc(f32, n_padded);
    defer allocator.free(padded);
    @memset(padded, 0);
    @memcpy(padded[pad_each_side..][0..audio_16k.len], audio_16k);
    if ((featOutputLength(n_padded) orelse 0) < downsample_factor) return Error.InputTooShort;

    // --- feature_extractor: 7 valid convs, no bias; layer 0 gets
    // GroupNorm(G == C == 512, affine); GELU-erf everywhere.
    var feat = codec.ConvRows{ .t = n_padded, .c = 1, .data = padded };
    var feat_owned = false; // `padded` is freed by its own defer
    defer if (feat_owned) allocator.free(feat.data);
    for (&hub.feat) |*layer| {
        const conved = try codec.ggmlConv1d(allocator, feat.data, feat.t, feat.c, &layer.conv_w, null, layer.stride, 0, 1);
        if (feat_owned) allocator.free(feat.data);
        feat = conved;
        feat_owned = true;
        if (layer.gn_w != null) {
            codec.groupNormGgml(feat.data, feat.t, feat.c, feat.c, eps, layer.gn_w.?, layer.gn_b.?);
        }
        codec.geluErfGgml(feat.data);
    }
    if (taps) |tp| tp.feat_extract = try captureTap(tp.allocator, feat.data, feat.t, feat.c);

    // --- feature_projection: LN(512) + Linear(512→768) + bias -------------
    codec.layerNormGgml(feat.data, feat.t, feat.c, eps, hub.fp_ln_w, hub.fp_ln_b);
    if (taps) |tp| tp.proj_ln = try captureTap(tp.allocator, feat.data, feat.t, feat.c);

    const t = feat.t;
    const proj = try linearHost(ctx, &hub.fp_proj, feat.data, t, feat.c, .in, .embed, hub.fp_proj_bias);
    defer allocator.free(proj);
    if (taps) |tp| tp.proj = try captureTap(tp.allocator, proj, t, hidden);

    // --- enc init + 12 transformer layers; collect the 13 states ----------
    var states: [n_states]?[]f32 = @splat(null);
    defer for (&states) |maybe_s| {
        if (maybe_s) |s| allocator.free(s);
    };

    states[0] = try encInit(ctx, hub, proj, t);
    if (taps) |tp| tp.enc_init = try captureTap(tp.allocator, states[0].?, t, hidden);
    for (&hub.layers, 0..) |*layer, i| {
        states[i + 1] = try layerForward(ctx, layer, states[i].?, t);
        if (taps) |tp| tp.layer[i] = try captureTap(tp.allocator, states[i + 1].?, t, hidden);
    }

    // --- mean of the 13 states (sum in ascending order, then * 1/13) ------
    const mean = try allocator.dupe(f32, states[0].?);
    defer allocator.free(mean);
    for (states[1..]) |maybe_s| {
        for (mean, maybe_s.?) |*v, s| v.* = v.* + s;
    }
    const inv_n_states: f32 = 1.0 / @as(f32, n_states);
    for (mean) |*v| v.* *= inv_n_states;

    // --- decimate by 2: keep even frames (pure decimation, no averaging) --
    const t_out = t / downsample_factor;
    const decimated = try allocator.alloc(f32, t_out * hidden);
    defer allocator.free(decimated);
    for (0..t_out) |ti| {
        @memcpy(decimated[ti * hidden ..][0..hidden], mean[ti * downsample_factor * hidden ..][0..hidden]);
    }
    return Act.fromSlice(ctx, .{ t_out, hidden }, decimated);
}

/// pos_conv_embed(x) → residual add → encoder LayerNorm (states[0]).
/// Returns a new owned `[t, 768]` buffer.
fn encInit(ctx: *ExecContext, hub: *const codec.Hubert, x: []const f32, t: usize) ![]f32 {
    const allocator = ctx.allocator;

    // Grouped pos conv (k=128, groups=16, pad=64, WITH bias): the even
    // kernel yields T+1 frames; SamePad drops the LAST frame; GELU-erf.
    const rows = try codec.ggmlConv1d(allocator, x, t, hidden, &hub.pos_conv_w, hub.pos_conv_bias, 1, codec.hubert_pos_pad, 1);
    errdefer allocator.free(rows.data);
    const pos = rows.data[0 .. t * hidden];
    codec.geluErfGgml(pos);

    // x = x + pos, then encoder LayerNorm.
    for (pos, x) |*v, xv| v.* = xv + v.*;
    codec.layerNormGgml(pos, t, hidden, codec.hubert_ln_eps, hub.enc_ln_w, hub.enc_ln_b);

    if (pos.len != rows.data.len) {
        const shrunk = try allocator.realloc(rows.data, pos.len);
        return shrunk;
    }
    return rows.data;
}

/// One Post-LN layer: x = LN(r + MHA(x)); x = LN(x + FFN(x)). Returns a new
/// owned `[t, 768]` buffer.
fn layerForward(ctx: *ExecContext, layer: *const codec.HubertLayer, x: []const f32, t: usize) ![]f32 {
    const allocator = ctx.allocator;
    const eps = codec.hubert_ln_eps;

    const cur = try attention(ctx, layer, x, t);
    errdefer allocator.free(cur);
    for (cur, x) |*v, r| v.* = r + v.*; // residual + attn (ggml_add order)
    codec.layerNormGgml(cur, t, hidden, eps, layer.ln_attn_w, layer.ln_attn_b);

    const f = try ffn(ctx, layer, cur, t);
    defer allocator.free(f);
    for (cur, f) |*v, fv| v.* = v.* + fv; // x + ffn_out
    codec.layerNormGgml(cur, t, hidden, eps, layer.ln_final_w, layer.ln_final_b);
    return cur;
}

/// Bidirectional MHA, 12 heads × 64, scale 1/√64 = 0.125, all projections
/// WITH bias, no mask. Per head: scores = q·kᵀ (sgemm) → ×0.125 →
/// ggml-parity softmax → out = scores·v (sgemm), exactly the reference's
/// per-head mul_mat decomposition on the BLAS backend.
fn attention(ctx: *ExecContext, layer: *const codec.HubertLayer, x: []const f32, t: usize) ![]f32 {
    const allocator = ctx.allocator;
    const heads = codec.hubert_num_heads;
    const d = codec.hubert_head_dim;

    const q = try linearHost(ctx, &layer.q_proj, x, t, hidden, .embed, .q, layer.q_bias);
    defer allocator.free(q);
    const k = try linearHost(ctx, &layer.k_proj, x, t, hidden, .embed, .k, layer.k_bias);
    defer allocator.free(k);
    const v = try linearHost(ctx, &layer.v_proj, x, t, hidden, .embed, .v, layer.v_bias);
    defer allocator.free(v);

    const merged = try allocator.alloc(f32, t * hidden);
    defer allocator.free(merged);

    // Heads are fully independent (disjoint merged columns, per-task
    // scratch), so a per-head thread split is bit-identical to the serial
    // loop. Threaded on x86 only, where the serial per-head walk dominated
    // the encode wall; aarch64 keeps the one-task serial execution.
    const max_threads = fucina.parallel.vector_max_threads;
    var want: usize = 1;
    if (comptime !builtin.cpu.arch.isAARCH64()) {
        want = @min(fucina.parallel.cpuThreadCount(max_threads), heads);
    }
    const per_task = 4 * t * d + t * t; // qh ++ kh ++ vpl ++ outh ++ scores
    const scratch = try allocator.alloc(f32, want * per_task);
    defer allocator.free(scratch);

    var tasks: [max_threads]AttnHeadTask = undefined;
    for (0..want) |i| {
        tasks[i] = .{
            .t = t,
            .q = q,
            .k = k,
            .v = v,
            .merged = merged,
            .scratch = scratch[i * per_task ..][0..per_task],
            .h_start = i * heads / want,
            .h_end = (i + 1) * heads / want,
        };
    }
    codec.runTaskThreads(AttnHeadTask, AttnHeadTask.run, tasks[0..want]);

    return linearHost(ctx, &layer.out_proj, merged, t, hidden, .embed, .attn, layer.out_bias);
}

/// One thread's contiguous head range of the per-head attention walk (the
/// reference's mul_mat decomposition, unchanged op-for-op per head).
const AttnHeadTask = struct {
    t: usize,
    q: []const f32,
    k: []const f32,
    v: []const f32,
    merged: []f32,
    /// `qh ++ kh ++ vpl ++ outh ++ scores` (4·t·d + t·t floats).
    scratch: []f32,
    h_start: usize,
    h_end: usize,

    fn run(task: *const @This()) void {
        const d = codec.hubert_head_dim;
        const t = task.t;
        const qh = task.scratch[0 .. t * d];
        const kh = task.scratch[t * d ..][0 .. t * d];
        const vpl = task.scratch[2 * t * d ..][0 .. t * d]; // v planar [d][t] (ggml cont)
        const outh = task.scratch[3 * t * d ..][0 .. t * d];
        const scores = task.scratch[4 * t * d ..][0 .. t * t];
        for (task.h_start..task.h_end) |h| {
            for (0..t) |ti| {
                for (0..d) |di| {
                    qh[ti * d + di] = task.q[ti * hidden + h * d + di];
                    kh[ti * d + di] = task.k[ti * hidden + h * d + di];
                    vpl[di * t + ti] = task.v[ti * hidden + h * d + di];
                }
            }
            // scores[q, t] = Σ_d qh[q, d]·kh[t, d]  (mul_mat(k, q) on BLAS).
            sgemmTransB(t, t, d, qh, kh, scores);
            for (scores) |*s| s.* *= 0.125; // ggml_scale, 1/sqrtf(64)
            for (0..t) |qi| codec.softMaxRowGgml(scores[qi * t ..][0..t]);
            // outh[q, d] = Σ_t scores[q, t]·vpl[d, t]  (mul_mat(v, scores)).
            sgemmTransB(t, d, t, scores, vpl, outh);
            for (0..t) |ti| {
                @memcpy(task.merged[ti * hidden + h * d ..][0..d], outh[ti * d ..][0..d]);
            }
        }
    }
};

/// FFN: Linear(768→3072) + bias → GELU-erf → Linear(3072→768) + bias.
/// Returns a new owned `[t, 768]` buffer.
fn ffn(ctx: *ExecContext, layer: *const codec.HubertLayer, x: []const f32, t: usize) ![]f32 {
    const f = try linearHost(ctx, &layer.fc1, x, t, hidden, .embed, .ffn, layer.fc1_bias);
    defer ctx.allocator.free(f);
    codec.geluErfGgml(f);
    return linearHost(ctx, &layer.fc2, f, t, codec.hubert_ffn_inner, .ffn, .embed, layer.fc2_bias);
}

/// Linear + bias over raw `[t, in]` rows via the facade `linearSeq` (the
/// same Accelerate cblas_sgemm(RowMajor, NoTrans, Trans, T, OUT, IN, …)
/// call the reference's ggml-BLAS backend makes). Returns owned rows.
fn linearHost(
    ctx: *ExecContext,
    weight: *const llm.weights.LinearWeight,
    input: []const f32,
    t: usize,
    in_dim: usize,
    comptime in_tag: anytype,
    comptime out_tag: anytype,
    bias: []const f32,
) ![]f32 {
    var xt = try fucina.Tensor(.{ .seq, in_tag }).fromSlice(ctx, .{ t, in_dim }, input);
    defer xt.deinit();
    var y = try weight.linearSeq(ctx, &xt, in_tag, out_tag);
    defer y.deinit();
    try y.addAxisVectorInPlace(ctx, bias, out_tag);
    return ctx.allocator.dupe(f32, try y.dataConst());
}

fn captureTap(allocator: Allocator, data: []const f32, t: usize, c: usize) !Tap {
    return .{ .t = t, .c = c, .data = try allocator.dupe(f32, data) };
}

test {
    _ = @import("hubert_tests.zig");
}
