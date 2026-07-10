//! MaskGIT iterative decode loop for OmniVoice TTS: port of maskgit_generate
//! (refs/omnivoice.cpp/src/maskgit-tts.h:135-366).
//!
//! Every float op preserves the reference's width and order (f32 scalar
//! sequential host loops via maskgit.zig); the Philox `ctr_lo` accounting is
//! exact: with the default config (class_temperature 0, position_temperature
//! 5) each EXECUTED step consumes one uniform kernel (K*T draws, ctr_lo += 1);
//! class_temperature > 0 adds one kernel per (k, t) slot per step, including
//! already-decoded slots. Greedy (both temperatures 0) performs zero Philox
//! calls and leaves ctr_lo untouched.

const std = @import("std");
const fucina = @import("fucina");

const dump = @import("dump.zig");
const lm = @import("lm.zig");
const maskgit = @import("maskgit.zig");
const prompt_mod = @import("prompt.zig");

const Allocator = std.mem.Allocator;
const ExecContext = fucina.ExecContext;

pub const Error = error{InvalidTarget};

/// Optional dump sink mirroring the reference --dump contract for this stage:
/// step-0 `mg-log-probs-step0` [K,T,V] (post CFG, mask_id slot = -inf),
/// `mg-pred-tokens-step0` [K,T], `mg-scores-step0` [K,T] (PRE layer penalty,
/// like the reference maskgit-tts.h:295-305 dump point), and the final
/// `mg-tokens` [K,T]. Files land as `<dir>/<name>.bin` in the reference dump
/// byte format ([ndims i32][shape i32 x n][f32 data]).
pub const DumpSink = struct {
    io: std.Io,
    dir: []const u8,

    fn writeF32(self: *const DumpSink, name: []const u8, shape: []const i32, data: []const f32) !void {
        var path_buf: [1024]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buf, "{s}/{s}.bin", .{ self.dir, name });
        try dump.writeFile(self.io, path, shape, data);
    }

    fn writeI32(self: *const DumpSink, name: []const u8, shape: []const i32, values: []const i32) !void {
        var path_buf: [1024]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buf, "{s}/{s}.bin", .{ self.dir, name });
        try dump.writeI32AsF32File(self.io, path, shape, values);
    }
};

/// Per-step progress observer for `generate`: type-erased context + function
/// pointer, called once per EXECUTED step (zero-demask steps skip the
/// forward and never fire), after the step's demask commit. `step` is
/// 1-based over the full schedule (`num_steps` = cfg.num_step), `demasked` /
/// `total` count committed (k, t) slots of the [K, T] canvas — the final
/// executed step always reports `demasked == total`. Observers must not
/// fail; anything driving a UI writes to stderr (stdout may be the WAV
/// stream).
pub const Progress = struct {
    ctx: ?*anyopaque = null,
    func: *const fn (ctx: ?*anyopaque, step: usize, num_steps: usize, demasked: usize, total: usize) void,

    pub fn call(self: Progress, step: usize, num_steps: usize, demasked: usize, total: usize) void {
        self.func(self.ctx, step, num_steps, demasked, total);
    }
};

/// Repack one forward's audio-logit rows `[T, K*V]` (the reference source
/// index (t*K + k)*V + v) into `[K, T, V]` (dst index (k*T + t)*V + v),
/// mirroring the maskgit-tts.h:208-219 extraction.
pub fn extractToKTV(src: []const f32, num_codebooks: usize, target_len: usize, vocab: usize, dst: []f32) void {
    std.debug.assert(src.len == num_codebooks * target_len * vocab);
    std.debug.assert(dst.len == src.len);
    for (0..num_codebooks) |k| {
        for (0..target_len) |t| {
            const src_off = (t * num_codebooks + k) * vocab;
            const dst_off = (k * target_len + t) * vocab;
            @memcpy(dst[dst_off..][0..vocab], src[src_off..][0..vocab]);
        }
    }
}

/// Run the iterative decoder. Returns flat audio tokens [K, T] (k slow,
/// t fast), owned by `allocator`. The prompt's input_ids are mutated in
/// place; `ctr_lo` threads the Philox counter across successive calls
/// (chunked synthesis keeps one global RNG state, like the reference).
pub fn generate(
    allocator: Allocator,
    ctx: *ExecContext,
    model: *const lm.Model,
    prompt: *prompt_mod.Prompt,
    cfg: maskgit.Config,
    target_len: usize,
    ctr_lo: *u32,
    dumps: ?DumpSink,
    progress: ?Progress,
) ![]i32 {
    const num_k = prompt.num_codebooks;
    const seq_len = prompt.s_max;
    const vocab = model.config.audio_vocab_size;
    const mask_id: i32 = @intCast(model.config.audio_mask_id);
    if (num_k != model.config.num_audio_codebook) return Error.InvalidTarget;
    if (target_len == 0 or target_len > seq_len or target_len != prompt.u_len) return Error.InvalidTarget;

    // Cond row: the audio target window is the row tail.
    const audio_start_cond = seq_len - target_len;

    const tokens = try allocator.alloc(i32, num_k * target_len);
    errdefer allocator.free(tokens);
    @memset(tokens, mask_id);

    const ts = try allocator.alloc(f32, cfg.num_step + 1);
    defer allocator.free(ts);
    maskgit.timesteps(cfg.t_shift, cfg.num_step, ts);
    const sched = try allocator.alloc(i32, cfg.num_step);
    defer allocator.free(sched);
    maskgit.schedule(target_len * num_k, ts, sched);

    // Constant across the whole decode call (the reference builds its
    // batched masks once per chunk too).
    var bias = try lm.buildUncondBias(ctx, seq_len, prompt.u_len);
    defer bias.deinit();

    const c_log = try allocator.alloc(f32, num_k * target_len * vocab);
    defer allocator.free(c_log);
    const u_log = try allocator.alloc(f32, num_k * target_len * vocab);
    defer allocator.free(u_log);
    const log_probs = try allocator.alloc(f32, num_k * target_len * vocab);
    defer allocator.free(log_probs);
    const work = try allocator.alloc(f32, vocab);
    defer allocator.free(work);
    const pred_tokens = try allocator.alloc(i32, num_k * target_len);
    defer allocator.free(pred_tokens);
    const confidence = try allocator.alloc(f32, num_k * target_len);
    defer allocator.free(confidence);
    const select_idx = try allocator.alloc(usize, num_k * target_len);
    defer allocator.free(select_idx);

    const cond_ids = prompt.condIds();
    const uncond_ids = prompt.uncondIds();
    const cond_mask = prompt.condAudioMask();
    const uncond_mask = prompt.uncondAudioMask();
    const neg_inf = -std.math.inf(f32);
    var n_demasked_total: usize = 0;

    for (0..cfg.num_step) |step| {
        const k_demask = sched[step];
        // Zero-demask steps skip the forward entirely (maskgit-tts.h:179-181).
        if (k_demask <= 0) continue;

        // Forward the CFG pair as two independent single-row forwards: cond
        // full row (audio window = the [S-T, S) tail), uncond padded row with
        // the additive bias (window [0, T)).
        {
            var cond_logits = try model.forward(ctx, cond_ids, cond_mask, audio_start_cond, target_len, null);
            defer cond_logits.deinit();
            extractToKTV(try cond_logits.dataConst(), num_k, target_len, vocab, c_log);
        }
        {
            var uncond_logits = try model.forwardUncondPadded(ctx, uncond_ids, uncond_mask, &bias, target_len);
            defer uncond_logits.deinit();
            extractToKTV(try uncond_logits.dataConst(), num_k, target_len, vocab, u_log);
        }

        // CFG + log_softmax per (k, t) row, then mask-id exclusion ALWAYS.
        for (0..num_k) |k| {
            for (0..target_len) |t| {
                const off = (k * target_len + t) * vocab;
                maskgit.cfgCombine(c_log[off..][0..vocab], u_log[off..][0..vocab], cfg.guidance_scale, log_probs[off..][0..vocab]);
                log_probs[off + model.config.audio_mask_id] = neg_inf;
            }
        }

        // Predict + confidence per (k, t), k outer / t inner. The class-
        // temperature path consumes one Philox block PER slot EVERY step
        // (including already-decoded slots); confidence always comes from lp.
        for (0..num_k) |k| {
            for (0..target_len) |t| {
                const off = (k * target_len + t) * vocab;
                const lp = log_probs[off..][0..vocab];
                var sample_src: []const f32 = lp;
                if (cfg.class_temperature > 0.0) {
                    @memcpy(work, lp);
                    maskgit.topKFilterInplace(work, 0.1);
                    maskgit.gumbelInplace(work, cfg.class_temperature, cfg.seed, ctr_lo);
                    sample_src = work;
                }
                pred_tokens[k * target_len + t] = @intCast(maskgit.argmaxStrict(sample_src));
                confidence[k * target_len + t] = maskgit.maxValue(lp);
            }
        }

        // Step-0 dumps happen BEFORE the layer penalty (the reference dumps
        // pre-penalty scores to match Python _predict_tokens_with_scoring).
        if (step == 0) {
            if (dumps) |*sink| {
                const kt_shape = [_]i32{ @intCast(num_k), @intCast(target_len) };
                const ktv_shape = [_]i32{ @intCast(num_k), @intCast(target_len), @intCast(vocab) };
                try sink.writeI32("mg-pred-tokens-step0", &kt_shape, pred_tokens);
                try sink.writeF32("mg-scores-step0", &kt_shape, confidence);
                try sink.writeF32("mg-log-probs-step0", &ktv_shape, log_probs);
            }
        }

        // Layer penalty: scores -= k * layer_penalty_factor.
        for (0..num_k) |k| {
            for (0..target_len) |t| {
                confidence[k * target_len + t] -= @as(f32, @floatFromInt(k)) * cfg.layer_penalty_factor;
            }
        }

        // Position noise: ONE gumbel kernel over the whole flat [K*T] score
        // array (one more Philox block), BEFORE masking decoded slots.
        if (cfg.position_temperature > 0.0) {
            maskgit.gumbelInplace(confidence, cfg.position_temperature, cfg.seed, ctr_lo);
        }

        // Mask already-decoded slots.
        for (tokens, confidence) |token, *score| {
            if (token != mask_id) score.* = neg_inf;
        }

        // Top-k demask on the flat scores (k slow, t fast), then apply the
        // predictions to tokens AND both prompt rows in place.
        const n_demask: usize = @intCast(k_demask);
        maskgit.topKSelect(confidence, n_demask, select_idx);
        for (select_idx[0..n_demask]) |i| {
            const k = i / target_len;
            const t = i % target_len;
            const v = pred_tokens[k * target_len + t];
            tokens[k * target_len + t] = v;
            // cond row: input_ids[0, k, audio_start_cond + t]
            prompt.input_ids[k * seq_len + audio_start_cond + t] = v;
            // uncond row: input_ids[1, k, t]
            prompt.input_ids[(num_k + k) * seq_len + t] = v;
        }

        n_demasked_total += n_demask;
        if (progress) |p| p.call(step + 1, cfg.num_step, n_demasked_total, num_k * target_len);
    }

    if (dumps) |*sink| {
        const kt_shape = [_]i32{ @intCast(num_k), @intCast(target_len) };
        try sink.writeI32("mg-tokens", &kt_shape, tokens);
    }

    return tokens;
}

test {
    _ = @import("mg_decode_tests.zig");
}
