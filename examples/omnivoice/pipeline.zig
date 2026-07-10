//! OmniVoice TTS pipeline orchestration:
//! ports `pipeline_tts_synthesize` + `tts_encode_ref` +
//! `tts_synthesize_one_chunk` + `tts_synthesize_long_internal` +
//! `tts_synthesize_long_stream_internal` + `pipeline_tts_generate` from
//! refs/omnivoice.cpp/src/pipeline-tts.cpp. SRT dubbing is out of scope.
//!
//! Key semantics preserved verbatim:
//! - raw-WAV and pre-encoded references are mutually exclusive; ref_text gets
//!   `add_punctuation` when preprocess_prompt, BEFORE the raw/tokens routing;
//! - the raw reference keeps its ORIGINAL (pre-gain) RMS for post-proc,
//!   `--ref-rvq` and no-ref run with ref_rms = -1 (peak/0.5 branch);
//! - one shared Philox `ctr_lo` starting at 0 threads through ALL MaskGIT
//!   calls of a synthesize (PyTorch's continuously advancing global RNG);
//! - chunk 0 of auto-voice runs ref-free and is then promoted (tokens+text)
//!   to the voice prompt of chunks 1..N; per-chunk durations are re-estimated
//!   against the CURRENT reference; only chunk 0 dumps;
//! - buffered post order: remove_silence → ref_rms volume branch (the quiet-
//!   ref rescale runs EVEN with postproc=false) → fade_and_pad, after the
//!   0.3 s cross-fade.

const std = @import("std");
const fucina = @import("fucina");
const llm = @import("fucina_llm");

const chunker = @import("chunker.zig");
const codec = @import("codec.zig");
const dac = @import("dac.zig");
const dump = @import("dump.zig");
const duration = @import("duration.zig");
const lm = @import("lm.zig");
const maskgit = @import("maskgit.zig");
const mg_decode = @import("mg_decode.zig");
const postproc = @import("postproc.zig");
const postproc_stream = @import("postproc_stream.zig");
const prompt_mod = @import("prompt.zig");
const rvq = @import("rvq.zig");
const voicedesign = @import("voicedesign.zig");

const Allocator = std.mem.Allocator;
const ExecContext = fucina.ExecContext;
const Tokenizer = llm.tokenizer.Tokenizer;

pub const Error = error{
    MutuallyExclusiveRefs,
    InvalidInstruct,
    MissingEncoder,
    RefEncodeFailed,
    TokenShapeMismatch,
    ResidualMaskTokens,
    AudioLengthMismatch,
    NoChunks,
};

/// The loaded pieces one synthesize call composes. All pointers borrow; the
/// caller owns loading/unloading (keep the GGUF files open behind `model`,
/// `cdc` and `enc` — quantized weights may borrow mmapped bytes).
pub const Tts = struct {
    allocator: Allocator,
    io: std.Io,
    ctx: *ExecContext,
    model: *const lm.Model,
    tok: *const Tokenizer,
    cdc: *const codec.Codec,
    /// Encode-side codec weights; required only for raw-WAV references.
    enc: ?*const codec.Encoder = null,
};

/// Mirror of `ov_tts_params` (defaults = `ov_tts_default_params`).
pub const Params = struct {
    text: []const u8,
    lang: []const u8 = "",
    /// RAW instruct; `synthesize` resolves it against the voice-design
    /// vocabulary with use_zh = hasCjk(SYNTHESIS text).
    instruct: []const u8 = "",
    /// > 0 forces the single-shot path with exactly this frame count.
    t_override: i32 = 0,
    chunk_duration_sec: f32 = 15.0,
    chunk_threshold_sec: f32 = 30.0,
    denoise: bool = true,
    preprocess_prompt: bool = true,
    postproc: bool = true,
    /// MaskGIT sampler config (reference defaults; set `.seed`).
    mg: maskgit.Config = .{},
    /// Raw 24 kHz mono reference waveform (voice cloning); mutually
    /// exclusive with `ref_audio_tokens`. The CLI does the readMono(24k).
    ref_audio_24k: ?[]const f32 = null,
    /// Pre-encoded reference codes `[K, ref_len]` (k slow, `--ref-rvq`).
    ref_audio_tokens: ?[]const i32 = null,
    ref_len: usize = 0,
    ref_text: []const u8 = "",
    /// Reference-named dump directory (chunk 0 only on the chunked path).
    dump_dir: ?[]const u8 = null,
    /// Optional per-step MaskGIT progress observer, threaded into every
    /// `mg_decode.generate` of this synthesize (all chunks). Purely
    /// additive: null = no signals.
    progress: ?mg_decode.Progress = null,
};

/// The resolved per-synthesize state shared by the buffered and streaming
/// entries: normalized instruct, punctuated ref_text, freshly encoded raw
/// reference and the final (tokens, len, text, rms) triple. `instruct`,
/// `ref_text_owned` and `re` are owned; the triple borrows from them (or
/// from the caller's Params).
const Prepared = struct {
    instruct: []u8,
    ref_text_owned: ?[]u8,
    re: RefEncoded,
    ref_tokens: ?[]const i32,
    ref_len: usize,
    ref_text: []const u8,
    ref_rms: f32,

    fn deinit(self: *Prepared, allocator: Allocator) void {
        allocator.free(self.instruct);
        if (self.ref_text_owned) |s| allocator.free(s);
        self.re.deinit(allocator);
        self.* = undefined;
    }
};

/// The shared front half of `pipeline_tts_synthesize`: reference-exclusivity
/// validation, ref_text punctuation, instruct resolution, raw-reference
/// encode and the reference-triple routing.
fn prepare(tts: *const Tts, params: *const Params) !Prepared {
    const allocator = tts.allocator;

    const has_raw = params.ref_audio_24k != null and params.ref_audio_24k.?.len > 0;
    const has_tokens = params.ref_audio_tokens != null and params.ref_len > 0;
    if (has_raw and has_tokens) {
        std.debug.print("[TTS] ref_audio_24k and ref_audio_tokens are mutually exclusive\n", .{});
        return Error.MutuallyExclusiveRefs;
    }

    // ref_text punctuation, applied BEFORE the raw/tokens routing so both
    // reference formats see the same transcript.
    var ref_text_owned: ?[]u8 = null;
    errdefer if (ref_text_owned) |s| allocator.free(s);
    var ref_text: []const u8 = params.ref_text;
    if (params.preprocess_prompt and ref_text.len != 0) {
        ref_text_owned = try chunker.addPunctuation(allocator, ref_text);
        ref_text = ref_text_owned.?;
    }

    // Instruct resolution (`pipeline_tts_resolve_instruct`): the target
    // language comes from the SYNTHESIS text (any CJK ideograph → Chinese).
    const normalized = try voicedesign.normalize(allocator, params.instruct, voicedesign.hasCjk(params.text));
    const instruct: []u8 = switch (normalized) {
        .ok => |s| s,
        .invalid => |msg| {
            defer allocator.free(msg);
            std.debug.print("[TTS] {s}\n", .{msg});
            return Error.InvalidInstruct;
        },
    };
    errdefer allocator.free(instruct);

    // Encode the optional raw reference once, before any synthesis.
    var re: RefEncoded = if (has_raw)
        try encodeRef(tts, params.ref_audio_24k.?, params.preprocess_prompt, params.dump_dir)
    else
        .{};
    errdefer re.deinit(allocator);

    // Reference triple: raw → (fresh codes, original RMS); pre-encoded →
    // (tokens, rms = -1: never the loudness-rescale branch); none → null/-1.
    var out = Prepared{
        .instruct = instruct,
        .ref_text_owned = ref_text_owned,
        .re = re,
        .ref_tokens = null,
        .ref_len = 0,
        .ref_text = "",
        .ref_rms = -1.0,
    };
    if (has_raw) {
        out.ref_tokens = out.re.codes.?;
        out.ref_len = out.re.ref_len;
        out.ref_text = ref_text;
        out.ref_rms = out.re.ref_rms;
    } else if (has_tokens) {
        out.ref_tokens = params.ref_audio_tokens.?;
        out.ref_len = params.ref_len;
        out.ref_text = ref_text;
        out.ref_rms = -1.0;
    }
    return out;
}

/// Public buffered entry (`pipeline_tts_synthesize`, buffered mode).
/// Returns 24 kHz mono samples owned by `tts.allocator`.
pub fn synthesize(tts: *const Tts, params: Params) ![]f32 {
    var prep = try prepare(tts, &params);
    defer prep.deinit(tts.allocator);
    return synthesizeLong(tts, &params, prep.instruct, prep.ref_text, prep.ref_tokens, prep.ref_len, prep.ref_rms);
}

/// Public streaming entry (`pipeline_tts_synthesize`, streaming mode /
/// `tts_synthesize_long_stream_internal`): every fully synthesized chunk is
/// pushed through the streaming post-proc chain (cross fade → silence
/// remove → volume scale → fade+pad) and emitted via `sink` as it becomes
/// available; nothing is returned. `params.postproc` is ignored — the
/// streaming path always post filters, like the reference. Voice design
/// (ref_rms < 0) skips the peak/0.5 normalisation entirely (the global peak
/// is unknowable while streaming), so its output runs ~6-12 dB below the
/// buffered path.
pub fn synthesizeStream(tts: *const Tts, params: Params, sink: postproc_stream.Emit) !void {
    var prep = try prepare(tts, &params);
    defer prep.deinit(tts.allocator);
    return synthesizeLongStream(tts, &params, prep.instruct, prep.ref_text, prep.ref_tokens, prep.ref_len, prep.ref_rms, sink);
}

// ---------------------------------------------------------------------------
// Reference encode (`tts_encode_ref`)
// ---------------------------------------------------------------------------

pub const RefEncoded = struct {
    /// `[K, ref_len]` codes (k slow), owned; null when no raw reference.
    codes: ?[]i32 = null,
    ref_len: usize = 0,
    /// ORIGINAL pre-gain RMS of the 24 kHz reference; -1 when absent.
    ref_rms: f32 = -1.0,

    pub fn deinit(self: *RefEncoded, allocator: Allocator) void {
        if (self.codes) |c| allocator.free(c);
        self.* = undefined;
    }
};

/// Encodes a raw 24 kHz mono reference into RVQ codes: RMS + unconditional
/// auto-gain (+ silence trim when `preprocess_prompt`) via
/// `refPreprocessAudio`, hop alignment `(n/960)*960`, codec encode. Dumps
/// `ref-audio-codes` `[K, ref_T]` when `dump_dir` is set.
pub fn encodeRef(tts: *const Tts, ref_audio_24k: []const f32, preprocess_prompt: bool, dump_dir: ?[]const u8) !RefEncoded {
    if (ref_audio_24k.len == 0) return .{};
    const enc = tts.enc orelse return Error.MissingEncoder;
    const allocator = tts.allocator;
    const sr: i32 = @intCast(tts.cdc.config.sample_rate);
    const hop = tts.cdc.config.hop_length;

    var audio = try allocator.dupe(f32, ref_audio_24k);
    defer allocator.free(audio);
    const ref_rms = try postproc.refPreprocessAudio(allocator, &audio, sr, preprocess_prompt);

    const n_aligned = (audio.len / hop) * hop;
    std.debug.print("[TTS] Reference: {d} samples @ 24 kHz mono ({d:.2} s), aligned to {d} (clip {d})\n", .{
        audio.len,
        @as(f64, @floatFromInt(audio.len)) / 24000.0,
        n_aligned,
        audio.len - n_aligned,
    });
    if (n_aligned == 0) {
        std.debug.print("[TTS] reference too short after preprocessing (hop {d})\n", .{hop});
        return Error.RefEncodeFailed;
    }

    const codes = try rvq.encode(tts.ctx, allocator, tts.cdc.config, &tts.cdc.rvq, enc, audio[0..n_aligned], null);
    errdefer allocator.free(codes);

    const num_k = tts.model.config.num_audio_codebook;
    if (codes.len == 0 or codes.len % num_k != 0) return Error.RefEncodeFailed;
    const ref_len = codes.len / num_k;
    std.debug.print("[TTS] Reference: encoded to [K={d}, T={d}] codes\n", .{ num_k, ref_len });

    if (dump_dir) |dir| {
        var path_buf: [1024]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buf, "{s}/ref-audio-codes.bin", .{dir});
        const shape = [_]i32{ @intCast(num_k), @intCast(ref_len) };
        try dump.writeI32AsF32File(tts.io, path, &shape, codes);
    }

    return .{ .codes = codes, .ref_len = ref_len, .ref_rms = ref_rms };
}

// ---------------------------------------------------------------------------
// Generate (`pipeline_tts_generate`)
// ---------------------------------------------------------------------------

/// Builds the prompt, dumps `prompt-{cond,uncond}-ids` (row k=0, shape
/// `[c_len]`, PRE-decode) and runs the MaskGIT decoder. Returns flat audio
/// tokens `[K, target_tokens]` (k slow) owned by `allocator`. `ctr_lo`
/// threads the shared Philox counter across chunked calls. Standalone (no
/// codec) so `--maskgit-test` can drive it directly.
pub fn generateTokens(
    allocator: Allocator,
    io: std.Io,
    ctx: *ExecContext,
    model: *const lm.Model,
    tok: *const Tokenizer,
    text: []const u8,
    lang: []const u8,
    instruct: []const u8,
    target_tokens: usize,
    denoise: bool,
    mg_cfg: maskgit.Config,
    ref_text: []const u8,
    ref_tokens: ?[]const i32,
    ref_len: usize,
    dump_dir: ?[]const u8,
    ctr_lo: *u32,
    progress: ?mg_decode.Progress,
) ![]i32 {
    var built = try prompt_mod.build(allocator, tok, &model.config, .{
        .text = text,
        .lang = lang,
        .instruct = instruct,
        .num_target_tokens = target_tokens,
        .denoise = denoise,
        .ref_text = ref_text,
        .ref_audio_tokens = ref_tokens,
        .ref_len = ref_len,
    });
    defer built.deinit();

    // Prompt dumps BEFORE the decode mutates input_ids. Style/text ids are
    // duplicated across K, so row k=0 of each batch item is enough; the
    // uncond row is dumped at the padded cond length (c_len == S_max).
    if (dump_dir) |dir| {
        var path_buf: [1024]u8 = undefined;
        const shape = [_]i32{@intCast(built.c_len)};
        const cond_row = built.input_ids[0..built.c_len];
        const uncond_row = built.input_ids[built.num_codebooks * built.s_max ..][0..built.c_len];
        const cond_path = try std.fmt.bufPrint(&path_buf, "{s}/prompt-cond-ids.bin", .{dir});
        try dump.writeI32AsF32File(io, cond_path, &shape, cond_row);
        const uncond_path = try std.fmt.bufPrint(&path_buf, "{s}/prompt-uncond-ids.bin", .{dir});
        try dump.writeI32AsF32File(io, uncond_path, &shape, uncond_row);
    }

    std.debug.print("[TTS] Prompt: B'=2 K={d} S={d} c_len={d} u_len={d}\n", .{
        built.num_codebooks, built.s_max, built.c_len, built.u_len,
    });

    const dumps: ?mg_decode.DumpSink = if (dump_dir) |dir| .{ .io = io, .dir = dir } else null;
    return mg_decode.generate(allocator, ctx, model, &built, mg_cfg, target_tokens, ctr_lo, dumps, progress);
}

// ---------------------------------------------------------------------------
// One chunk (`tts_synthesize_one_chunk`)
// ---------------------------------------------------------------------------

/// Token shape + residual-mask validation (a residual `audio_mask_id` would
/// corrupt the RVQ lookup — hard error, like the reference).
fn validateTokens(config: *const lm.Config, tokens: []const i32, target_tokens: usize) !void {
    if (tokens.len != config.num_audio_codebook * target_tokens) {
        std.debug.print("[TTS] token vector size {d} does not match K*T={d}*{d}\n", .{
            tokens.len, config.num_audio_codebook, target_tokens,
        });
        return Error.TokenShapeMismatch;
    }
    const mask_id: i32 = @intCast(config.audio_mask_id);
    var n_residual: usize = 0;
    for (tokens) |v| {
        if (v == mask_id) n_residual += 1;
    }
    if (n_residual != 0) {
        std.debug.print("[TTS] {d} residual mask tokens left after MaskGIT, refusing to decode\n", .{n_residual});
        return Error.ResidualMaskTokens;
    }
}

/// RVQ decode + fc2 + DAC decode; returns exactly `t * hop_length` samples.
fn decodeAudio(tts: *const Tts, tokens: []const i32, t: usize) ![]f32 {
    var decoded = try rvq.decode(tts.ctx, &tts.cdc.rvq, tokens, t);
    defer decoded.deinit();
    var dac_in = try decoded.fc2_out.withTags(tts.ctx, .{ .seq, .in });
    defer dac_in.deinit();
    const audio = try dac.decodeForward(tts.ctx, tts.allocator, &tts.cdc.dac, &dac_in, null);
    errdefer tts.allocator.free(audio);
    if (audio.len != t * tts.cdc.config.hop_length) return Error.AudioLengthMismatch;
    return audio;
}

fn dumpOutputAudio(io: std.Io, dir: []const u8, audio: []const f32) !void {
    var path_buf: [1024]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "{s}/output-audio.bin", .{dir});
    const shape = [_]i32{@intCast(audio.len)};
    try dump.writeFile(io, path, &shape, audio);
}

fn nowNs(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).nanoseconds;
}

fn msOf(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

/// Generate → validate → codec decode, with the reference's `mg-tokens` /
/// `output-audio` dumps and `[Perf]` lines.
fn synthesizeOneChunk(
    tts: *const Tts,
    text: []const u8,
    lang: []const u8,
    instruct: []const u8,
    target_tokens: usize,
    denoise: bool,
    mg_cfg: maskgit.Config,
    ref_text: []const u8,
    ref_tokens: ?[]const i32,
    ref_len: usize,
    dump_dir: ?[]const u8,
    ctr_lo: *u32,
    progress: ?mg_decode.Progress,
) ![]f32 {
    const allocator = tts.allocator;
    const t_total0 = nowNs(tts.io);

    const tokens = try generateTokens(
        allocator,
        tts.io,
        tts.ctx,
        tts.model,
        tts.tok,
        text,
        lang,
        instruct,
        target_tokens,
        denoise,
        mg_cfg,
        ref_text,
        ref_tokens,
        ref_len,
        dump_dir,
        ctr_lo,
        progress,
    );
    defer allocator.free(tokens);
    const gen_ns = nowNs(tts.io) - t_total0;

    // mg-tokens is dumped by mg_decode's DumpSink when dump_dir is set.
    try validateTokens(&tts.model.config, tokens, target_tokens);

    const hop = tts.cdc.config.hop_length;
    std.debug.print("[TTS] Decode: K={d} T={d} expected_samples={d}\n", .{
        tts.model.config.num_audio_codebook, target_tokens, target_tokens * hop,
    });

    const t_codec0 = nowNs(tts.io);
    const audio = try decodeAudio(tts, tokens, target_tokens);
    errdefer allocator.free(audio);
    const codec_ns = nowNs(tts.io) - t_codec0;

    if (dump_dir) |dir| try dumpOutputAudio(tts.io, dir, audio);

    const total_ms = msOf(nowNs(tts.io) - t_total0);
    const audio_sec = @as(f64, @floatFromInt(target_tokens * hop)) / @as(f64, @floatFromInt(tts.cdc.config.sample_rate));
    const rtf = if (audio_sec > 0.0) (total_ms / 1000.0) / audio_sec else 0.0;
    std.debug.print("[Perf] Generate {d:.1} ms (MaskGIT, {d} steps)\n", .{ msOf(gen_ns), mg_cfg.num_step });
    std.debug.print("[Perf] CodecDecode {d:.1} ms\n", .{msOf(codec_ns)});
    std.debug.print("[Perf] Total {d:.1} ms (T={d}, audio {d:.2} s, RTF {d:.3})\n", .{
        total_ms, target_tokens, audio_sec, rtf,
    });
    return audio;
}

// ---------------------------------------------------------------------------
// Long-form (`tts_synthesize_long_internal`, buffered)
// ---------------------------------------------------------------------------

/// `threshold_frames = (int)(chunk_threshold_sec * (float)frame_rate)` —
/// f32 multiply, C-cast truncation.
pub fn thresholdFrames(chunk_threshold_sec: f32, frame_rate: usize) i32 {
    return @intFromFloat(chunk_threshold_sec * @as(f32, @floatFromInt(frame_rate)));
}

/// `chunk_len = (int)((double)chunk_duration_sec * frame_rate / avg)` with
/// `avg = T_total / n_chars` in f64; floor 1. Codepoints per chunk.
pub fn chunkLenCodepoints(chunk_duration_sec: f32, frame_rate: usize, t_total: i32, n_chars: usize) i32 {
    const avg = @as(f64, @floatFromInt(t_total)) / @as(f64, @floatFromInt(n_chars));
    const raw: i32 = @intFromFloat(@as(f64, chunk_duration_sec) * @as(f64, @floatFromInt(frame_rate)) / avg);
    return if (raw < 1) 1 else raw;
}

/// Buffered post filtering, exact reference order: remove_silence (postproc
/// only) → ref_rms volume branch (`< 0` → peak/0.5 when postproc; `< 0.1` →
/// scale by ref_rms/0.1 EVEN when postproc=false; else nothing) →
/// fade_and_pad (postproc only). `audio.*` must be owned by `allocator`.
pub fn postFilter(allocator: Allocator, audio: *[]f32, sr: i32, ref_rms: f32, do_postproc: bool) !void {
    if (do_postproc) {
        try postproc.removeSilence(allocator, audio, sr, 500, 100, 100, -50.0);
    }

    if (ref_rms < 0.0) {
        if (do_postproc) postproc.peakNormalizeHalf(audio.*);
    } else if (ref_rms < 0.1) {
        const k: f32 = ref_rms / 0.1;
        for (audio.*) |*s| s.* *= k;
    }

    if (do_postproc) {
        try postproc.fadeAndPad(allocator, audio, sr, 0.1, 0.1);
    }
}

fn synthesizeLong(
    tts: *const Tts,
    params: *const Params,
    instruct: []const u8,
    ref_text: []const u8,
    ext_ref_tokens: ?[]const i32,
    ext_ref_len: usize,
    ref_rms: f32,
) ![]f32 {
    const allocator = tts.allocator;
    const sr: i32 = @intCast(tts.cdc.config.sample_rate);
    const frame_rate = tts.cdc.config.sample_rate / tts.cdc.config.hop_length; // 25 (integer division)

    // Chunking trigger uses the same estimator as the single-shot path.
    const t_total: i32 = if (params.t_override > 0)
        params.t_override
    else
        duration.estimateTokens(params.text, ref_text, @intCast(ext_ref_len));

    const threshold = thresholdFrames(params.chunk_threshold_sec, frame_rate);
    const no_chunk = params.t_override > 0 or params.chunk_duration_sec <= 0.0 or t_total <= threshold;

    // Shared Philox counter across ALL MaskGIT calls of this synthesize.
    // 0 == PyTorch's freshly seeded generator right after fix_random_seed().
    var shared_ctr_lo: u32 = 0;

    var audio: []f32 = undefined;
    if (no_chunk) {
        std.debug.print("[TTS-Long] Single-shot path: T={d} frames ({d:.2}s), threshold={d} frames\n", .{
            t_total, @as(f64, @floatFromInt(t_total)) / @as(f64, @floatFromInt(frame_rate)), threshold,
        });
        audio = try synthesizeOneChunk(
            tts,
            params.text,
            params.lang,
            instruct,
            @intCast(t_total),
            params.denoise,
            params.mg,
            ref_text,
            ext_ref_tokens,
            ext_ref_len,
            params.dump_dir,
            &shared_ctr_lo,
            params.progress,
        );
    } else {
        audio = try synthesizeChunked(tts, params, instruct, ref_text, ext_ref_tokens, ext_ref_len, t_total, frame_rate, &shared_ctr_lo);
    }
    errdefer allocator.free(audio);

    const before = audio.len;
    try postFilter(allocator, &audio, sr, ref_rms, params.postproc);
    std.debug.print("[TTS-Long] Post-proc: {d} -> {d} samples ({d:.2}s at {d} Hz, ref_rms={d:.4})\n", .{
        before, audio.len, @as(f64, @floatFromInt(audio.len)) / @as(f64, @floatFromInt(sr)), sr, ref_rms,
    });

    return audio;
}

fn synthesizeChunked(
    tts: *const Tts,
    params: *const Params,
    instruct: []const u8,
    ref_text: []const u8,
    ext_ref_tokens: ?[]const i32,
    ext_ref_len: usize,
    t_total: i32,
    frame_rate: usize,
    shared_ctr_lo: *u32,
) ![]f32 {
    const allocator = tts.allocator;

    var n_chars = chunker.utf8Count(params.text);
    if (n_chars < 1) n_chars = 1;
    const chunk_len = chunkLenCodepoints(params.chunk_duration_sec, frame_rate, t_total, n_chars);

    const chunks = try chunker.chunkTextPunctuation(allocator, params.text, chunk_len, chunker.min_chunk_len_default);
    defer chunker.freeChunks(allocator, chunks);
    if (chunks.len == 0) {
        std.debug.print("[TTS-Long] chunker produced no chunks for input of {d} chars\n", .{n_chars});
        return Error.NoChunks;
    }

    std.debug.print("[TTS-Long] Chunked: {d} chunks, T_total={d} frames, chunk_len={d} codepoints\n", .{
        chunks.len, t_total, chunk_len,
    });

    var chunk_audios: std.ArrayList([]const f32) = .empty;
    defer {
        for (chunk_audios.items) |a| allocator.free(a);
        chunk_audios.deinit(allocator);
    }

    // Chunk-0 tokens outlive their iteration: they become the voice prompt
    // for chunks 1..N in auto-voice mode.
    var chunk0_tokens: ?[]i32 = null;
    defer if (chunk0_tokens) |t| allocator.free(t);

    // Active voice prompt: the external reference if provided, otherwise
    // promoted from chunk 0.
    var prompt_tokens: ?[]const i32 = ext_ref_tokens;
    var prompt_len: usize = ext_ref_len;
    var prompt_text: []const u8 = ref_text;

    for (chunks, 0..) |ct, i| {
        // Chunk 0 in pure auto-voice runs without any reference.
        const first_no_ref = (i == 0 and ext_ref_tokens == null);
        const this_ref: ?[]const i32 = if (first_no_ref) null else prompt_tokens;
        const this_len: usize = if (first_no_ref) 0 else prompt_len;
        const this_ref_text: []const u8 = if (first_no_ref) "" else prompt_text;

        // Per-chunk duration re-estimated against the CURRENT reference.
        const ti: usize = @intCast(duration.estimateTokens(ct, this_ref_text, @intCast(this_len)));

        // Only chunk 0 dumps (cossim tests compare matching chunks).
        const chunk_dump = if (i == 0) params.dump_dir else null;

        std.debug.print("[TTS-Long] Chunk {d}/{d}: chars={d} T={d} ref_T={d}\n", .{
            i + 1, chunks.len, chunker.utf8Count(ct), ti, this_len,
        });

        if (first_no_ref) {
            // Capture the tokens before decoding so they can become the
            // voice prompt for chunks 1..N.
            const toks = try generateTokens(
                allocator,
                tts.io,
                tts.ctx,
                tts.model,
                tts.tok,
                ct,
                params.lang,
                instruct,
                ti,
                params.denoise,
                params.mg,
                this_ref_text,
                this_ref,
                this_len,
                chunk_dump,
                shared_ctr_lo,
                params.progress,
            );
            chunk0_tokens = toks;
            try validateTokens(&tts.model.config, toks, ti);

            const a = try decodeAudio(tts, toks, ti);
            errdefer allocator.free(a);
            // Mirror the single-shot dumps for chunk 0 (mg-tokens was
            // already written by generateTokens' DumpSink).
            if (chunk_dump) |dir| try dumpOutputAudio(tts.io, dir, a);
            try chunk_audios.append(allocator, a);

            prompt_tokens = toks;
            prompt_len = ti;
            prompt_text = ct;
        } else {
            const a = try synthesizeOneChunk(
                tts,
                ct,
                params.lang,
                instruct,
                ti,
                params.denoise,
                params.mg,
                this_ref_text,
                this_ref,
                this_len,
                chunk_dump,
                shared_ctr_lo,
                params.progress,
            );
            errdefer allocator.free(a);
            try chunk_audios.append(allocator, a);
        }
    }

    const merged = try postproc.crossFadeChunks(allocator, chunk_audios.items, @intCast(tts.cdc.config.sample_rate), 0.3);
    std.debug.print("[TTS-Long] Cross-faded {d} chunks -> {d} samples\n", .{ chunk_audios.items.len, merged.len });
    return merged;
}

// ---------------------------------------------------------------------------
// Long-form (`tts_synthesize_long_stream_internal`, streaming)
// ---------------------------------------------------------------------------

/// Same orchestration as `synthesizeLong` up to chunk decoding, then drives
/// the audio through the streaming post-proc pipeline and emits via `sink`.
/// The volume scale is resolved up front: voice cloning applies ref_rms/0.1
/// when the reference is quiet, no-op when it is loud; voice design
/// (ref_rms < 0) skips peak/0.5 and runs at native level.
fn synthesizeLongStream(
    tts: *const Tts,
    params: *const Params,
    instruct: []const u8,
    ref_text: []const u8,
    ext_ref_tokens: ?[]const i32,
    ext_ref_len: usize,
    ref_rms: f32,
    sink: postproc_stream.Emit,
) !void {
    const allocator = tts.allocator;
    const sr: i32 = @intCast(tts.cdc.config.sample_rate);
    const frame_rate = tts.cdc.config.sample_rate / tts.cdc.config.hop_length; // 25 (integer division)

    var volume_scale: f32 = 1.0;
    if (ref_rms < 0.0) {
        std.debug.print("[TTS-Stream] voice design + streaming : peak normalisation disabled, output level runs ~6 to 12 dB below buffered path\n", .{});
    } else if (ref_rms < 0.1) {
        volume_scale = ref_rms / 0.1;
        std.debug.print("[TTS-Stream] voice clone scale {d:.4} (ref_rms {d:.4})\n", .{ volume_scale, ref_rms });
    }

    // Pipeline stages: cross fade (0.3 s), silence remove (mid=500,
    // lead=100, trail=100, -50 dBFS), volume scale, fade+pad (0.1/0.1 s).
    var pipe = postproc_stream.Pipeline.init(allocator, sr, volume_scale, sink);
    defer pipe.deinit();

    // Same chunking decision as the buffered path: single shot below the
    // threshold, otherwise split on punctuation and chain chunks.
    const t_total: i32 = if (params.t_override > 0)
        params.t_override
    else
        duration.estimateTokens(params.text, ref_text, @intCast(ext_ref_len));

    const threshold = thresholdFrames(params.chunk_threshold_sec, frame_rate);
    const no_chunk = params.t_override > 0 or params.chunk_duration_sec <= 0.0 or t_total <= threshold;

    // Shared Philox counter across ALL MaskGIT calls of this synthesize;
    // fresh (0) per synthesize call, exactly like the buffered path.
    var shared_ctr_lo: u32 = 0;

    if (no_chunk) {
        std.debug.print("[TTS-Stream] Single-shot path: T={d} frames ({d:.2}s), threshold={d} frames\n", .{
            t_total, @as(f64, @floatFromInt(t_total)) / @as(f64, @floatFromInt(frame_rate)), threshold,
        });
        const a = try synthesizeOneChunk(
            tts,
            params.text,
            params.lang,
            instruct,
            @intCast(t_total),
            params.denoise,
            params.mg,
            ref_text,
            ext_ref_tokens,
            ext_ref_len,
            params.dump_dir,
            &shared_ctr_lo,
            params.progress,
        );
        defer allocator.free(a);
        try pipe.pushChunk(a);
    } else {
        try synthesizeChunkedStream(tts, params, instruct, ref_text, ext_ref_tokens, ext_ref_len, t_total, frame_rate, &shared_ctr_lo, &pipe);
    }

    // Drain stages in pipeline order (cf → silence remove → fade+pad).
    try pipe.finish();
    std.debug.print("[TTS-Stream] Done\n", .{});
}

/// The chunked branch of the streaming path: identical chunking, voice
/// promotion and dump behaviour to `synthesizeChunked`, but each decoded
/// chunk is pushed straight through the streaming pipeline instead of being
/// collected for a batch cross-fade.
fn synthesizeChunkedStream(
    tts: *const Tts,
    params: *const Params,
    instruct: []const u8,
    ref_text: []const u8,
    ext_ref_tokens: ?[]const i32,
    ext_ref_len: usize,
    t_total: i32,
    frame_rate: usize,
    shared_ctr_lo: *u32,
    pipe: *postproc_stream.Pipeline,
) !void {
    const allocator = tts.allocator;

    var n_chars = chunker.utf8Count(params.text);
    if (n_chars < 1) n_chars = 1;
    const chunk_len = chunkLenCodepoints(params.chunk_duration_sec, frame_rate, t_total, n_chars);

    const chunks = try chunker.chunkTextPunctuation(allocator, params.text, chunk_len, chunker.min_chunk_len_default);
    defer chunker.freeChunks(allocator, chunks);
    if (chunks.len == 0) {
        std.debug.print("[TTS-Stream] chunker produced no chunks for input of {d} chars\n", .{n_chars});
        return Error.NoChunks;
    }

    std.debug.print("[TTS-Stream] Chunked: {d} chunks, T_total={d} frames, chunk_len={d} codepoints\n", .{
        chunks.len, t_total, chunk_len,
    });

    // Chunk-0 tokens outlive their iteration: they become the voice prompt
    // for chunks 1..N in auto-voice mode.
    var chunk0_tokens: ?[]i32 = null;
    defer if (chunk0_tokens) |t| allocator.free(t);

    var prompt_tokens: ?[]const i32 = ext_ref_tokens;
    var prompt_len: usize = ext_ref_len;
    var prompt_text: []const u8 = ref_text;

    for (chunks, 0..) |ct, i| {
        const first_no_ref = (i == 0 and ext_ref_tokens == null);
        const this_ref: ?[]const i32 = if (first_no_ref) null else prompt_tokens;
        const this_len: usize = if (first_no_ref) 0 else prompt_len;
        const this_ref_text: []const u8 = if (first_no_ref) "" else prompt_text;

        const ti: usize = @intCast(duration.estimateTokens(ct, this_ref_text, @intCast(this_len)));
        const chunk_dump = if (i == 0) params.dump_dir else null;

        std.debug.print("[TTS-Stream] Chunk {d}/{d}: chars={d} T={d} ref_T={d}\n", .{
            i + 1, chunks.len, chunker.utf8Count(ct), ti, this_len,
        });

        if (first_no_ref) {
            const toks = try generateTokens(
                allocator,
                tts.io,
                tts.ctx,
                tts.model,
                tts.tok,
                ct,
                params.lang,
                instruct,
                ti,
                params.denoise,
                params.mg,
                this_ref_text,
                this_ref,
                this_len,
                chunk_dump,
                shared_ctr_lo,
                params.progress,
            );
            chunk0_tokens = toks;
            try validateTokens(&tts.model.config, toks, ti);

            const a = try decodeAudio(tts, toks, ti);
            defer allocator.free(a);
            if (chunk_dump) |dir| try dumpOutputAudio(tts.io, dir, a);
            try pipe.pushChunk(a);

            prompt_tokens = toks;
            prompt_len = ti;
            prompt_text = ct;
        } else {
            const a = try synthesizeOneChunk(
                tts,
                ct,
                params.lang,
                instruct,
                ti,
                params.denoise,
                params.mg,
                this_ref_text,
                this_ref,
                this_len,
                chunk_dump,
                shared_ctr_lo,
                params.progress,
            );
            defer allocator.free(a);
            try pipe.pushChunk(a);
        }
    }
}

test {
    _ = @import("pipeline_tests.zig");
}
