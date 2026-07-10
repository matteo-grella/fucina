//! DAC decoder forward (dac-decoder.h): fc2 output `[T, 256]` → waveform
//! `[T*960]`. Structure: conv1 (k=7, pad 3) → 5 up-blocks (snake →
//! ConvTranspose1d → 3 dilated res units) → final snake → conv2 (32→1, k=7,
//! pad 3). NO final tanh, NO clamp — the raw conv2 output is the waveform.
//!
//! Decoder convs run as an im2col + `matmulTransB` composition (Accelerate
//! sgemm; k=1 convs skip the im2col and go straight to the GEMM), ~4x faster
//! than the direct per-tap kernel on Apple Silicon. The im2col matrix is
//! built threaded into ONE caller-owned scratch buffer reused across every
//! conv of the decode, chunked to ≤32 MiB of rows per GEMM call (a whole
//! late-block im2col would be ~100 MB of fresh pages per conv = fault churn;
//! much smaller chunks starve the sgemm — 32 MiB is the measured sweet spot).
//!
//! DAC ENCODER forward (dac-encoder.h, `encodeForward`): 24 kHz waveform
//! `[T_in]` → latent `[T_in/960, 256]`. Inverse block order vs the decoder:
//! res units FIRST, then block-level snake + strided downsampling conv. The
//! encoder deliberately stays on the ggml-parity f16 conv arithmetic
//! (`codec.ggmlConv1d`) — bit-parity with the reference's RVQ codes is a
//! feature there; do NOT route it through this GEMM path.
//!
//! Internal layout is Fucina's `[T, C]` rows (channel fast) everywhere; the
//! per-stage taps captured for parity dumps keep that layout (the CLI
//! transposes to the reference's channel-planar buffers when writing).

const std = @import("std");
const fucina = @import("fucina");

const codec = @import("codec.zig");

const ExecContext = fucina.ExecContext;
const Allocator = std.mem.Allocator;

pub const Error = error{LengthMismatch};

/// Activation rows `[T, C]` (the running channel axis is `.in`).
pub const Act = fucina.Tensor(.{ .seq, .in });

/// im2col matrix `[T_out, K*IC]` (row t = the K dilated input rows at
/// `t + k·dilation − pad` concatenated, zeros where out of range).
const Im2col = fucina.Tensor(.{ .seq, .kin });

/// Debug/test escape hatch: route the decoder convs through the original
/// direct per-tap `conv1d` kernel instead of the default im2col+GEMM
/// composition (same math, f32 GEMM reassociation-level differences only).
/// The GEMM path is ~4x faster on Apple Silicon (Accelerate sgemm).
pub var use_direct_conv: bool = false;

/// One captured stage: `data` is `[t, c]` row-major (our layout).
pub const StageTap = struct {
    t: usize,
    c: usize,
    data: []f32,
};

/// Per-stage taps for parity dumps (all optional; filled by `decodeForward`
/// when passed in). Matches the reference's cpp_dac_after_conv1 /
/// cpp_dac_after_blk{0..4} dump points.
pub const Taps = struct {
    allocator: Allocator,
    after_conv1: ?StageTap = null,
    after_blk: [5]?StageTap = .{ null, null, null, null, null },

    pub fn init(allocator: Allocator) Taps {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Taps) void {
        if (self.after_conv1) |tap| self.allocator.free(tap.data);
        for (self.after_blk) |maybe_tap| {
            if (maybe_tap) |tap| self.allocator.free(tap.data);
        }
        self.* = undefined;
    }
};

/// Grow-only host scratch for the im2col matrix, owned by `decodeForward`
/// and reused across every conv of the decode (contents are rebuilt per
/// conv, so growth never copies).
const Im2colScratch = struct {
    allocator: Allocator,
    buf: []f32 = &.{},

    fn ensure(self: *Im2colScratch, len: usize) ![]f32 {
        if (self.buf.len < len) {
            self.allocator.free(self.buf);
            self.buf = &.{};
            self.buf = try self.allocator.alloc(f32, len);
        }
        return self.buf[0..len];
    }

    fn deinit(self: *Im2colScratch) void {
        self.allocator.free(self.buf);
        self.* = undefined;
    }
};

/// Runs the DAC decoder on the fc2 output (`[T, 256]`, tagged `[.seq, .in]`)
/// and returns the waveform samples `[T*960]` (allocated from `allocator`;
/// caller frees). When `taps` is non-null the per-stage activations are
/// copied into it (after conv1 and after each block's res units).
pub fn decodeForward(
    ctx: *ExecContext,
    allocator: Allocator,
    dec: *const codec.DacDecoder,
    fc2_out: *const Act,
    taps: ?*Taps,
) ![]f32 {
    const t_in = fc2_out.dim(.seq);

    var scratch = Im2colScratch{ .allocator = allocator };
    defer scratch.deinit();

    // conv1: 256→1024, k=7, s=1, p=3 (+ bias).
    var x = try convStep(ctx, &scratch, fc2_out, &dec.conv1_w, &dec.conv1_gw, dec.conv1_b, 3, 1);
    errdefer x.deinit();
    if (taps) |tp| tp.after_conv1 = try captureTap(tp.allocator, &x);

    var expected_t = t_in;
    for (&dec.blocks, 0..) |*blk, i| {
        const next = try blockForward(ctx, &scratch, blk, &x);
        x.deinit();
        x = next;
        // Length algebra: (T-1)*S + K - 2*pad + output_pad = S*T per block.
        expected_t *= blk.spec.stride;
        if (x.dim(.seq) != expected_t) return Error.LengthMismatch;
        if (taps) |tp| tp.after_blk[i] = try captureTap(tp.allocator, &x);
    }

    // Final snake (32 ch) + conv2 (32→1, k=7, p=3) + bias. No tanh, no clamp.
    {
        const snaked = try x.snake(ctx, .in, &dec.final_snake_a, &dec.final_snake_inv_b);
        x.deinit();
        x = snaked;
    }
    var out = try convForward(ctx, &scratch, &x, &dec.conv2_w, &dec.conv2_gw, dec.conv2_b, 3, 1);
    errdefer out.deinit();

    if (out.dim(.seq) != expected_t or out.dim(.out) != 1) return Error.LengthMismatch;
    const audio = try allocator.dupe(f32, try out.dataConst());
    out.deinit();
    x.deinit();
    return audio;
}

/// Analytic DAC ENCODER output frame count for `n_samples` of 24 kHz audio
/// (`compute_dac_output_length`): conv1/conv2 keep T; per down-block
/// `T = (T + 2P − K)/S + 1` with C-style truncating division. Used by the
/// encode pipeline to decide the ±480 zero-pad branch BEFORE running the
/// encoder.
pub fn encodeOutputLength(n_samples: usize) isize {
    var t: isize = @intCast(n_samples);
    for (codec.dac_enc_block_specs) |spec| {
        const k: isize = @intCast(spec.taps);
        const s: isize = @intCast(spec.stride);
        const p: isize = @intCast(spec.pad);
        t = @divTrunc(t + 2 * p - k, s) + 1;
    }
    return t;
}

/// Runs the DAC ENCODER on 24 kHz mono samples (already hop-aligned and, if
/// needed, ±480 zero-padded by the caller) and returns the acoustic latent
/// `[T_a, 256]` rows. The whole stack runs host-side with the reference's
/// exact arithmetic: ggml-parity f16 convs + libm-parity snake.
pub fn encodeForward(ctx: *ExecContext, enc: *const codec.DacEncoder, audio: []const f32) !Act {
    const allocator = ctx.allocator;

    // conv1: 1→64, k=7, s=1, p=3 (+ bias).
    var x = try codec.ggmlConv1d(allocator, audio, audio.len, 1, &enc.conv1_w, enc.conv1_b, 1, 3, 1);
    errdefer allocator.free(x.data);

    // 5 down blocks: res_unit1..3 → block snake → strided conv (+ bias).
    for (&enc.blocks) |*blk| {
        for (&blk.res) |*ru| {
            // skip = x; snake1 → conv1(k=7, p=3·d, dil=d) → snake2 →
            // conv2(k=1); x = skip + x. `x` stays untouched as the skip.
            const s1 = try allocator.dupe(f32, x.data);
            defer allocator.free(s1);
            codec.snakeGgml(s1, x.t, x.c, ru.snake1_a, ru.snake1_inv_b);
            const c1 = try codec.ggmlConv1d(allocator, s1, x.t, x.c, &ru.conv1_w, ru.conv1_b, 1, 3 * ru.dilation, ru.dilation);
            defer allocator.free(c1.data);
            codec.snakeGgml(c1.data, c1.t, c1.c, ru.snake2_a, ru.snake2_inv_b);
            const c2 = try codec.ggmlConv1d(allocator, c1.data, c1.t, c1.c, &ru.conv2_w, ru.conv2_b, 1, 0, 1);
            std.debug.assert(c2.t == x.t and c2.c == x.c);
            for (c2.data, x.data) |*v, s| v.* = s + v.*; // ggml_add(skip, x)
            allocator.free(x.data);
            x = c2;
        }
        codec.snakeGgml(x.data, x.t, x.c, blk.snake_a, blk.snake_inv_b);
        const down = try codec.ggmlConv1d(allocator, x.data, x.t, x.c, &blk.conv_w, blk.conv_b, blk.spec.stride, blk.spec.pad, 1);
        allocator.free(x.data);
        x = down;
    }

    // Final snake (2048 ch) + conv2 (2048→256, k=3, p=1) + bias.
    codec.snakeGgml(x.data, x.t, x.c, enc.final_snake_a, enc.final_snake_inv_b);
    const latent = try codec.ggmlConv1d(allocator, x.data, x.t, x.c, &enc.conv2_w, enc.conv2_b, 1, 1, 1);
    allocator.free(x.data);
    x = latent; // the top-level errdefer now owns the latent buffer
    const result = try Act.fromSlice(ctx, .{ x.t, x.c }, x.data);
    allocator.free(x.data);
    return result;
}

/// snake → ConvTranspose1d(+bias, output_pad) → res_unit1..3.
fn blockForward(ctx: *ExecContext, scratch: *Im2colScratch, blk: *const codec.UpBlock, x: *const Act) !Act {
    var snaked = try x.snake(ctx, .in, &blk.snake1_a, &blk.snake1_inv_b);
    defer snaked.deinit();

    var h = up: {
        var up = try snaked.convTranspose1d(
            ctx,
            .seq,
            .in,
            .kout,
            .out,
            &blk.conv_t_w2,
            &blk.conv_t_b,
            blk.spec.out_ch,
            blk.spec.taps,
            blk.spec.stride,
            blk.spec.pad,
            blk.spec.output_pad,
        );
        errdefer up.deinit();
        const renamed = try renameOutToIn(ctx, &up);
        up.deinit();
        break :up renamed;
    };
    errdefer h.deinit();

    for (&blk.res) |*ru| {
        const next = try resUnitForward(ctx, scratch, ru, &h);
        h.deinit();
        h = next;
    }
    return h;
}

/// skip = x; x = snake1 → conv1(k=7, pad=3·d, dil=d) → snake2 → conv2(k=1);
/// return skip + x.
fn resUnitForward(ctx: *ExecContext, scratch: *Im2colScratch, ru: *const codec.ResUnit, x: *const Act) !Act {
    var s1 = try x.snake(ctx, .in, &ru.snake1_a, &ru.snake1_inv_b);
    defer s1.deinit();
    var c1 = try convStep(ctx, scratch, &s1, &ru.conv1_w, &ru.conv1_gw, ru.conv1_b, 3 * ru.dilation, ru.dilation);
    defer c1.deinit();
    var s2 = try c1.snake(ctx, .in, &ru.snake2_a, &ru.snake2_inv_b);
    defer s2.deinit();
    var out = try convStep(ctx, scratch, &s2, &ru.conv2_w, &ru.conv2_gw, ru.conv2_b, 0, 1);
    errdefer out.deinit();
    try out.addScaledInPlace(ctx, x.*, 1.0); // + skip
    return out;
}

/// Decoder conv1d (+ bias), `[.seq, .out]` result. Stride is always 1 in the
/// DAC decoder. Default path is the im2col+GEMM composition (Accelerate
/// sgemm); `use_direct_conv` falls back to the direct per-tap kernel.
fn convForward(
    ctx: *ExecContext,
    scratch: *Im2colScratch,
    x: *const Act,
    weight: *const codec.ConvWeight,
    gemm_weight: *const codec.GemmConvWeight,
    bias: []const f32,
    pad: usize,
    dilation: usize,
) !fucina.Tensor(.{ .seq, .out }) {
    const taps = weight.dim(.tap);
    var conv = conv: {
        if (use_direct_conv) {
            break :conv try x.conv1d(ctx, .seq, .in, .tap, .out, weight, 1, pad, dilation, 1);
        }
        if (taps == 1 and pad == 0) {
            // k=1 convs are pure GEMMs: [T, IC]·[OC, IC]ᵀ.
            break :conv try x.matmul(ctx, gemm_weight, .trans_b, .{ .seq, .out });
        }

        // im2col + GEMM, chunked over output rows: the im2col matrix of a
        // whole late-block activation would be ~100 MB (fresh pages =
        // fault-churn every conv); a ≤32 MiB rolling chunk reuses ONE
        // scratch for the entire decode while keeping each sgemm call big
        // enough to parallelize (4-16 MiB chunks measured slower).
        const t_in = x.dim(.seq);
        const ic = x.dim(.in);
        const span = dilation * (taps - 1) + 1;
        std.debug.assert(t_in + 2 * pad >= span);
        const t_out = t_in + 2 * pad + 1 - span;
        const row_len = taps * ic;
        const oc = gemm_weight.dim(.out);
        const chunk_rows = @max(@as(usize, 256), (32 << 20) / (row_len * @sizeOf(f32)));
        const src = try x.dataConst();
        const dst = try scratch.ensure(@min(t_out, chunk_rows) * row_len);

        if (t_out <= chunk_rows) {
            // Single chunk: GEMM straight to the result tensor.
            buildIm2col(ctx, dst, src, t_in, ic, taps, pad, dilation, 0, t_out);
            var col = try Im2col.fromBorrowedSlice(ctx, .{ t_out, row_len }, dst[0 .. t_out * row_len]);
            defer col.deinit();
            break :conv try col.matmul(ctx, gemm_weight, .trans_b, .{ .seq, .out });
        }

        var out = try fucina.Tensor(.{ .seq, .out }).empty(ctx, .{ t_out, oc });
        errdefer out.deinit();
        const out_data = try out.data();
        var t0: usize = 0;
        while (t0 < t_out) : (t0 += chunk_rows) {
            const rows = @min(chunk_rows, t_out - t0);
            buildIm2col(ctx, dst, src, t_in, ic, taps, pad, dilation, t0, rows);
            var col = try Im2col.fromBorrowedSlice(ctx, .{ rows, row_len }, dst[0 .. rows * row_len]);
            defer col.deinit();
            var part = try col.matmul(ctx, gemm_weight, .trans_b, .{ .seq, .out });
            defer part.deinit();
            @memcpy(out_data[t0 * oc ..][0 .. rows * oc], try part.dataConst());
        }
        break :conv out;
    };
    errdefer conv.deinit();
    try conv.addAxisVectorInPlace(ctx, bias, .out);
    return conv;
}

/// `convForward`, result renamed back to the `[.seq, .in]` activation tags.
fn convStep(
    ctx: *ExecContext,
    scratch: *Im2colScratch,
    x: *const Act,
    weight: *const codec.ConvWeight,
    gemm_weight: *const codec.GemmConvWeight,
    bias: []const f32,
    pad: usize,
    dilation: usize,
) !Act {
    var conv = try convForward(ctx, scratch, x, weight, gemm_weight, bias, pad, dilation);
    errdefer conv.deinit();
    const renamed = try renameOutToIn(ctx, &conv);
    conv.deinit();
    return renamed;
}

const Im2colTask = struct {
    dst: []f32,
    src: []const f32,
    t_in: usize,
    ic: usize,
    taps: usize,
    pad: usize,
    dilation: usize,
    t_base: usize,
    t_start: usize,
    t_end: usize,
};

fn runIm2colTask(task: *const Im2colTask) void {
    im2colRange(task.dst, task.src, task.t_in, task.ic, task.taps, task.pad, task.dilation, task.t_base, task.t_start, task.t_end);
}

/// Builds im2col rows for the ABSOLUTE output rows `[t_start, t_end)` into
/// `dst` (indexed relative to `t_base`): row t is the K input rows at
/// `t + k·dilation − pad` concatenated (column block k at `k*IC`), zero
/// where out of range. Straight memcpy loops — memory-bound.
fn im2colRange(
    dst: []f32,
    src: []const f32,
    t_in: usize,
    ic: usize,
    taps: usize,
    pad: usize,
    dilation: usize,
    t_base: usize,
    t_start: usize,
    t_end: usize,
) void {
    const row_len = taps * ic;
    for (t_start..t_end) |t| {
        const row = dst[(t - t_base) * row_len ..][0..row_len];
        for (0..taps) |k| {
            const pos = t + k * dilation;
            const seg = row[k * ic ..][0..ic];
            if (pos < pad or pos - pad >= t_in) {
                @memset(seg, 0.0);
            } else {
                @memcpy(seg, src[(pos - pad) * ic ..][0..ic]);
            }
        }
    }
}

/// Builds `rows` im2col rows starting at absolute output row `t0` into the
/// chunk scratch `dst`, threaded over disjoint row ranges via the runtime's
/// persistent hot team (⇒ bit-identical to serial).
fn buildIm2col(
    ctx: *ExecContext,
    dst: []f32,
    src: []const f32,
    t_in: usize,
    ic: usize,
    taps: usize,
    pad: usize,
    dilation: usize,
    t0: usize,
    rows: usize,
) void {
    const max_threads = fucina.parallel.vector_max_threads;
    const pool = if (rows * taps * ic >= 1 << 18) ctx.workPool() else null;
    const want_threads: usize = if (pool == null) 1 else @min(fucina.parallel.cpuThreadCount(max_threads), rows);
    if (want_threads <= 1) {
        im2colRange(dst, src, t_in, ic, taps, pad, dilation, t0, t0, t0 + rows);
        return;
    }
    var tasks: [max_threads]Im2colTask = undefined;
    for (0..want_threads) |i| {
        tasks[i] = .{
            .dst = dst,
            .src = src,
            .t_in = t_in,
            .ic = ic,
            .taps = taps,
            .pad = pad,
            .dilation = dilation,
            .t_base = t0,
            .t_start = t0 + i * rows / want_threads,
            .t_end = t0 + (i + 1) * rows / want_threads,
        };
    }
    pool.?.parallelChunks(Im2colTask, tasks[0..want_threads], runIm2colTask);
}

fn renameOutToIn(ctx: *ExecContext, x: *const fucina.Tensor(.{ .seq, .out })) !Act {
    return x.withTags(ctx, .{ .seq, .in });
}

fn captureTap(allocator: Allocator, x: *const Act) !StageTap {
    const data = try allocator.dupe(f32, try x.dataConst());
    return .{ .t = x.dim(.seq), .c = x.dim(.in), .data = data };
}

test {
    _ = @import("dac_tests.zig");
}
