//! Parakeet ASR (NVIDIA NeMo FastConformer) CPU inference harness — Fucina port.
//!
//! Loads a parakeet GGUF (mmap) and prints its architecture, config, full
//! metadata, and tensor map; transcribes audio (offline `--transcribe`, streaming
//! `--stream`, live `--mic`); and runs the per-stage parity gates
//! (`--compare <stage> <dump.pkd>`) against PKD1 dumps from the parakeet.cpp
//! reference. The hard parity target is an exact decoded token-id sequence;
//! intermediate stages gate on cosine (op-order makes bit-exact unrealistic).
//!
//!   zig build parakeet -- --model <model.gguf>
//!   zig build parakeet -- --model <model.gguf> --audio <speech.wav>
//!   zig build parakeet -- --model <model.gguf> --compare encoder <encoder_out.pkd> --tol 1e-4
const std = @import("std");
const fucina = @import("fucina");
const llm = @import("fucina_llm");
const build_options = @import("build_options");

const Value = fucina.gguf.Value;
const parakeet_loader = llm.parakeet.loader;
const parakeet_frontend = llm.parakeet.frontend;
const parakeet_subsampling = llm.parakeet.subsampling;
const parakeet_encoder = llm.parakeet.encoder;
const parakeet_decoder = llm.parakeet.decoder;
const parakeet_weights = llm.parakeet.weights;
const parakeet_streaming = llm.parakeet.streaming;
const parakeet_tokenizer = llm.parakeet.tokenizer;
const parakeet_transcription = llm.parakeet.transcription;

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    var stdout_buffer: [8192]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    var model_path: ?[]const u8 = null;
    var audio_path: ?[]const u8 = null;
    var compare_stage: ?[]const u8 = null;
    var compare_dump: ?[]const u8 = null;
    var tol: f32 = 1e-4;
    var transcribe = false;
    var stream = false;
    var mic_sim = false;
    var mic = false;
    var stream_bench = false;
    var lang: []const u8 = "auto"; // target locale for multilingual prompt-conditioned models
    var decoder: []const u8 = "tdt"; // model default
    var f32_cache = false;
    var fast_mel = true;
    var bench_reps: usize = 1;
    var json_out = false; // --json (offline only)
    var timestamps = false; // --timestamps (per-word start-end-conf)
    var threads: usize = 0; // --threads N (0 = default)
    var manifest_path: ?[]const u8 = null; // --manifest <file>

    var arg_i: usize = 1;
    while (arg_i < args.len) : (arg_i += 1) {
        const arg = args[arg_i];
        if (std.mem.eql(u8, arg, "--model")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingModelPath;
            model_path = args[arg_i];
        } else if (std.mem.eql(u8, arg, "--audio")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingAudioPath;
            audio_path = args[arg_i];
        } else if (std.mem.eql(u8, arg, "--transcribe")) {
            transcribe = true;
        } else if (std.mem.eql(u8, arg, "--stream")) {
            stream = true;
        } else if (std.mem.eql(u8, arg, "--mic-sim")) {
            mic_sim = true;
        } else if (std.mem.eql(u8, arg, "--mic")) {
            mic = true;
        } else if (std.mem.eql(u8, arg, "--stream-bench")) {
            stream_bench = true;
        } else if (std.mem.eql(u8, arg, "--f32-cache")) {
            f32_cache = true;
        } else if (std.mem.eql(u8, arg, "--fast-mel")) {
            fast_mel = true;
        } else if (std.mem.eql(u8, arg, "--no-fast-mel")) {
            fast_mel = false;
        } else if (std.mem.eql(u8, arg, "--json")) {
            json_out = true;
        } else if (std.mem.eql(u8, arg, "--timestamps")) {
            timestamps = true;
        } else if (std.mem.eql(u8, arg, "--threads")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingThreads;
            threads = try std.fmt.parseInt(usize, args[arg_i], 10);
        } else if (std.mem.eql(u8, arg, "--manifest")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingManifest;
            manifest_path = args[arg_i];
        } else if (std.mem.eql(u8, arg, "--bench-reps")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingBenchReps;
            bench_reps = try std.fmt.parseInt(usize, args[arg_i], 10);
            if (bench_reps == 0) return error.InvalidBenchReps;
        } else if (std.mem.eql(u8, arg, "--decoder")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingDecoder;
            decoder = args[arg_i];
        } else if (std.mem.eql(u8, arg, "--lang")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingLang;
            lang = args[arg_i];
        } else if (std.mem.eql(u8, arg, "--compare")) {
            // --compare <stage> <dumpfile>  (e.g. mel|subsampling|encoder|... <dump.pkd>)
            if (arg_i + 2 >= args.len) return error.MissingCompareArgs;
            compare_stage = args[arg_i + 1];
            compare_dump = args[arg_i + 2];
            arg_i += 2;
        } else if (std.mem.eql(u8, arg, "--tol")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingTolValue;
            tol = try std.fmt.parseFloat(f32, args[arg_i]);
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            try printUsage(stdout);
            return;
        } else {
            try stdout.print("unknown argument: {s}\n\n", .{arg});
            try printUsage(stdout);
            return;
        }
    }

    if (model_path == null) {
        try printUsage(stdout);
        return;
    }

    if (threads > 0) fucina.parallel.setMaxThreads(threads); // --threads N

    const allocator = std.heap.smp_allocator;
    var file = try fucina.gguf.File.loadMmap(allocator, init.io, model_path.?);
    defer file.deinit();

    // --manifest: transcribe a list of audio files (per-file output, respecting --json/--timestamps).
    if (manifest_path) |mp| {
        try runManifest(stdout, &file, init.io, init.arena.allocator(), mp, decoder, f32_cache, fast_mel, json_out, timestamps);
        return;
    }
    // --transcribe: clean transcript only (no config dump), for diffing vs parakeet-cli.
    if (transcribe) {
        try runTranscribe(stdout, &file, init.io, init.arena.allocator(), audio_path, decoder, f32_cache, fast_mel, bench_reps, json_out, timestamps);
        return;
    }
    // --stream: full cache-aware streaming pipeline (audio -> mel -> windowed chunks
    // -> StreamingSession -> tokens -> text), for diffing vs parakeet-cli --stream.
    if (stream) {
        if (json_out) { // match the cli: --json is not supported with --stream
            std.debug.print("parakeet: --json is not supported with --stream\n", .{});
            return;
        }
        try runStreamTranscribe(stdout, &file, init.io, init.arena.allocator(), audio_path, lang, timestamps);
        return;
    }
    // --mic-sim: feed --audio in batches through the incremental driver (a simulated
    // mic) — automated gate for the live-mic path (transcript == --stream).
    if (mic_sim) {
        try runMicSim(stdout, &file, init.io, init.arena.allocator(), audio_path, lang);
        return;
    }
    // --mic: live microphone capture (NAM miniaudio) → incremental streaming.
    if (mic) {
        try runMic(stdout, &file, init.io, init.arena.allocator(), lang);
        return;
    }
    // --stream-bench: streaming RTF + first-token + per-chunk latency (best of N).
    if (stream_bench) {
        try runStreamBench(stdout, &file, init.io, init.arena.allocator(), audio_path, bench_reps, lang);
        return;
    }

    try printConfig(stdout, &file, model_path.?, init.arena.allocator());
    try printLoader(stdout, &file, init.arena.allocator());

    if (compare_stage) |stage| {
        // `gate` accumulates the PASS/FAIL verdict of every gated comparison in
        // this run; a FAIL makes `--compare` mechanically enforcing via a nonzero
        // exit, so a scripted parity gate can be a plain exit-code check.
        var gate = true;
        try runCompare(stdout, &file, init.io, init.arena.allocator(), stage, compare_dump.?, audio_path, tol, lang, &gate);
        if (!gate) {
            try stdout.print("\nFAIL: a gated --compare stage did not pass\n", .{});
            stdout.flush() catch {};
            std.process.exit(1);
        }
    } else if (audio_path) |ap| {
        try runAudio(stdout, &file, init.io, init.arena.allocator(), ap);
    }
}

// Extract `audio_filepath` from a NeMo-style JSONL manifest line (minimal — no
// escape handling; manifest paths don't contain escaped quotes in practice).
fn extractAudioFilepath(line: []const u8) ?[]const u8 {
    const key = "\"audio_filepath\"";
    const ki = std.mem.indexOf(u8, line, key) orelse return null;
    var i = ki + key.len;
    while (i < line.len and (line[i] == ' ' or line[i] == ':' or line[i] == '\t')) i += 1;
    if (i >= line.len or line[i] != '"') return null;
    i += 1;
    const start = i;
    while (i < line.len and line[i] != '"') i += 1;
    if (i >= line.len) return null;
    return line[start..i];
}

// --manifest: transcribe each audio path listed in the manifest file (one path
// per line, or a NeMo JSONL line with "audio_filepath"; blank/`#` lines skipped).
// Each file's output (text/json/timestamps) is preceded by a `# <path>` header,
// and is identical to running that file singly with the same flags.
fn runManifest(stdout: *std.Io.Writer, file: *const fucina.gguf.File, io: std.Io, arena: std.mem.Allocator, manifest_path: []const u8, decoder: []const u8, f32_cache: bool, fast_mel: bool, json: bool, timestamps: bool) !void {
    var mf = try std.Io.Dir.cwd().openFile(io, manifest_path, .{});
    defer mf.close(io);
    const stat = try mf.stat(io);
    const bytes = try arena.alloc(u8, @intCast(stat.size));
    var rd: usize = 0;
    while (rd < bytes.len) {
        const n = try mf.readStreaming(io, &.{bytes[rd..]});
        if (n == 0) break;
        rd += n;
    }
    var lines = std.mem.splitScalar(u8, bytes[0..rd], '\n');
    while (lines.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \t\r");
        if (line.len == 0 or line[0] == '#') continue;
        const path = if (line[0] == '{') (extractAudioFilepath(line) orelse continue) else line;
        try stdout.print("# {s}\n", .{path});
        try runTranscribe(stdout, file, io, arena, path, decoder, f32_cache, fast_mel, 1, json, timestamps);
    }
}

// Full pipeline → transcript text (just the text + newline, for parity diff).
// `json` → emit the C-API JSON document; `timestamps` → per-word lines; else text.
fn runTranscribe(stdout: *std.Io.Writer, file: *const fucina.gguf.File, io: std.Io, arena: std.mem.Allocator, audio_path: ?[]const u8, decoder: []const u8, f32_cache: bool, fast_mel: bool, bench_reps: usize, json: bool, timestamps: bool) !void {
    const ap = audio_path orelse {
        try stdout.print("--transcribe requires --audio <wav>\n", .{});
        return;
    };
    // --decoder ctc|tdt head selection (default tdt). Validate + require the
    // CTC head for --decoder ctc (matches the cli's clean error, no crash).
    if (!std.mem.eql(u8, decoder, "ctc") and !std.mem.eql(u8, decoder, "tdt")) {
        try stdout.print("parakeet: unknown --decoder '{s}' (want ctc|tdt)\n", .{decoder});
        return;
    }
    if (std.mem.eql(u8, decoder, "ctc") and file.maybeGet("ctc_decoder.decoder_layers.0.weight") == null) {
        try stdout.print("parakeet: CTC head tensor not found: this model has no CTC head (--decoder ctc unavailable)\n", .{});
        return;
    }
    const cfg = try parakeet_loader.Config.fromGguf(file);
    const feat = try parakeet_loader.loadFeaturizer(file, cfg);
    const audio = try parakeet_frontend.loadWav16kMonoFile(arena, io, ap);
    var dft_basis = if (fast_mel) try parakeet_frontend.DftBasis.init(arena, cfg.n_fft) else null;
    defer if (dft_basis) |*basis| basis.deinit();
    const t_mel0 = nowNs(io);
    const mel_params: parakeet_frontend.MelParams = .{
        .stft = .{ .n_fft = cfg.n_fft, .hop = cfg.hop_length, .win_length = cfg.win_length, .mag_power = cfg.mag_power, .preemph = cfg.preemph },
        .n_mels = cfg.n_mels,
        .log_guard = cfg.log_zero_guard,
        .normalize_per_feature = cfg.normalize == .per_feature,
    };
    var mel = if (fast_mel)
        try parakeet_frontend.melSpectrogramFastWithBasis(arena, audio.samples, mel_params, feat.fb, feat.window, &dft_basis.?)
    else
        try parakeet_frontend.melSpectrogram(arena, audio.samples, mel_params, feat.fb, feat.window);
    defer mel.deinit(arena);
    const mel_ms = @as(f64, @floatFromInt(nowNs(io) - t_mel0)) / 1e6;

    var ctx: fucina.ExecContext = undefined;
    ctx.init(arena);
    defer ctx.deinit();

    var weights = parakeet_weights.ParakeetWeights.init(&ctx, file);
    defer weights.deinit();
    if (f32_cache) weights.enableF32Blas();

    var best_enc_ns: i96 = std.math.maxInt(i96);
    var best_dec_ns: i96 = std.math.maxInt(i96);
    var best_total_ns: i96 = std.math.maxInt(i96);
    var ids: []i32 = &.{};
    const want_meta = json or timestamps; // collect per-token TokenInfo only when needed
    var token_meta: std.ArrayList(parakeet_decoder.TokenInfo) = .empty;
    defer token_meta.deinit(arena);
    fucina.internal.gpu.traceReset(); // FUCINA_GPU_TRACE=1: count the warm window (no-op otherwise)
    for (0..bench_reps) |_| {
        const t0 = nowNs(io);
        var enc = try parakeet_encoder.encodeWithWeights(&ctx, file, cfg, mel.feats, cfg.n_mels, mel.n_frames, &weights);
        const t1 = nowNs(io);
        if (want_meta) token_meta.clearRetainingCapacity();
        const mp: parakeet_decoder.TokenMeta = if (want_meta) &token_meta else null;
        ids = if (std.mem.eql(u8, decoder, "ctc"))
            try parakeet_decoder.ctcDecode(&ctx, file, cfg, &enc, arena, mp)
        else
            try parakeet_decoder.tdtDecodeWithWeights(&ctx, cfg, &enc, arena, &weights, mp);
        const t2 = nowNs(io);
        enc.deinit();
        const total_ns = t2 - t0;
        if (total_ns < best_total_ns) {
            best_total_ns = total_ns;
            best_enc_ns = t1 - t0;
            best_dec_ns = t2 - t1;
        }
    }

    fucina.internal.gpu.traceDump(); // FUCINA_GPU_TRACE=1: print the warm-window breakdown to stderr
    const pieces = try parakeet_loader.loadPieces(file, arena);
    const text = try parakeet_tokenizer.detokenize(arena, pieces, ids);
    if (want_meta) {
        const frame_sec: f32 = @as(f32, @floatFromInt(cfg.hop_length * cfg.subsampling_factor)) / @as(f32, @floatFromInt(cfg.sample_rate));
        const words = try parakeet_transcription.groupWords(arena, token_meta.items, pieces, frame_sec);
        if (json) {
            const j = try parakeet_transcription.toJson(arena, text, frame_sec, token_meta.items, words);
            try stdout.print("{s}\n", .{j});
        } else { // timestamps: per-word "%.2f-%.2f  %s  (%.2f)"
            for (words) |w| try stdout.print("{d:.2}-{d:.2}  {s}  ({d:.2})\n", .{ w.start, w.end, w.text, w.conf });
        }
    } else {
        try stdout.print("{s}\n", .{text});
    }

    // Timing → stderr (keeps stdout = transcript for parity diffs).
    const enc_ms = @as(f64, @floatFromInt(best_enc_ns)) / 1e6;
    const dec_ms = @as(f64, @floatFromInt(best_dec_ns)) / 1e6;
    const audio_s = @as(f64, @floatFromInt(audio.samples.len)) / @as(f64, @floatFromInt(cfg.sample_rate));
    var ebuf: [256]u8 = undefined;
    var ew = std.Io.File.stderr().writer(io, &ebuf);
    ew.interface.print("[timing] mel={d:.1}ms encode={d:.1}ms decode={d:.1}ms total={d:.1}ms audio={d:.3}s RTF={d:.4} reps={d} f32_cache={} fast_mel={}\n", .{ mel_ms, enc_ms, dec_ms, mel_ms + enc_ms + dec_ms, audio_s, (mel_ms + enc_ms + dec_ms) / 1000.0 / audio_s, bench_reps, f32_cache, fast_mel }) catch {};
    ew.interface.flush() catch {};
}

// Full cache-aware streaming pipeline → transcript text (for diffing vs
// parakeet-cli --stream). audio → mel (normalize per cfg; the realtime model is
// NA = no per-feature norm) → chunk-schedule windowing → StreamingSession.
fn runStreamTranscribe(stdout: *std.Io.Writer, file: *const fucina.gguf.File, io: std.Io, arena: std.mem.Allocator, audio_path: ?[]const u8, lang: []const u8, timestamps: bool) !void {
    const ap = audio_path orelse {
        try stdout.print("--stream requires --audio <wav>\n", .{});
        return;
    };
    const cfg = try parakeet_loader.Config.fromGguf(file);
    const sc = (parakeet_loader.StreamingConfig.fromGguf(file) catch null) orelse {
        try stdout.print("--stream requires a streaming model (no streaming.* metadata)\n", .{});
        return;
    };
    // Resolve the language prompt index for multilingual (prompt-conditioned)
    // models. -1 for non-prompt models (the prompt path is a no-op). The prompt
    // projection itself is applied per chunk in the session.
    const prompt_index: i32 = if (parakeet_loader.PromptConfig.fromGguf(file)) |pc|
        (try pc.resolveLang(file, arena, lang)) orelse {
            try stdout.print("--lang '{s}' is not a known locale for this model\n", .{lang});
            return;
        }
    else
        -1;
    const feat = try parakeet_loader.loadFeaturizer(file, cfg);
    const audio = try parakeet_frontend.loadWav16kMonoFile(arena, io, ap);
    var dft_basis = try parakeet_frontend.DftBasis.init(arena, cfg.n_fft);
    defer dft_basis.deinit();
    const t_mel0 = nowNs(io);
    const mel_params: parakeet_frontend.MelParams = .{
        .stft = .{ .n_fft = cfg.n_fft, .hop = cfg.hop_length, .win_length = cfg.win_length, .mag_power = cfg.mag_power, .preemph = cfg.preemph },
        .n_mels = cfg.n_mels,
        .log_guard = cfg.log_zero_guard,
        .normalize_per_feature = cfg.normalize == .per_feature,
    };
    var mel = try parakeet_frontend.melSpectrogramFastWithBasis(arena, audio.samples, mel_params, feat.fb, feat.window, &dft_basis);
    defer mel.deinit(arena);
    const mel_ms = @as(f64, @floatFromInt(nowNs(io) - t_mel0)) / 1e6;

    var ctx: fucina.ExecContext = undefined;
    ctx.init(arena);
    defer ctx.deinit();
    var weights = parakeet_weights.ParakeetWeights.init(&ctx, file);
    defer weights.deinit();
    var sess = try parakeet_streaming.StreamingSession.init(arena, file, cfg, sc, &weights, lang);
    defer sess.deinit();
    sess.collect_meta = timestamps; // gather per-token TokenInfo for --timestamps

    const t0 = nowNs(io);
    try sess.feedMel(&ctx, file, &weights, mel.feats, cfg.n_mels, mel.n_frames);
    const dec_ms = @as(f64, @floatFromInt(nowNs(io) - t0)) / 1e6;
    const pieces = try parakeet_loader.loadPieces(file, arena);

    const text = try parakeet_tokenizer.detokenize(arena, pieces, sess.tokens.items);
    try stdout.print("{s}\n", .{text});
    if (timestamps) { // per-word "%.2f-%.2f  %s  (%.2f)" lines (cli --stream --timestamps recap)
        const frame_sec: f32 = @as(f32, @floatFromInt(cfg.hop_length * cfg.subsampling_factor)) / @as(f32, @floatFromInt(cfg.sample_rate));
        const words = try parakeet_transcription.groupWords(arena, sess.token_meta.items, pieces, frame_sec);
        for (words) |w| try stdout.print("{d:.2}-{d:.2}  {s}  ({d:.2})\n", .{ w.start, w.end, w.text, w.conf });
    }

    const audio_s = @as(f64, @floatFromInt(audio.samples.len)) / @as(f64, @floatFromInt(cfg.sample_rate));
    var ebuf: [256]u8 = undefined;
    var ew = std.Io.File.stderr().writer(io, &ebuf);
    ew.interface.print("[stream] lang={s} prompt_index={d} mel={d:.1}ms encode+decode={d:.1}ms total={d:.1}ms audio={d:.3}s RTF={d:.4} tokens={d} eou_events={d}\n", .{ lang, prompt_index, mel_ms, dec_ms, mel_ms + dec_ms, audio_s, (mel_ms + dec_ms) / 1000.0 / audio_s, sess.tokens.items.len, sess.eou_events }) catch {};
    ew.interface.flush() catch {};
}

// Incremental streaming driver: accumulates audio samples, recomputes the
// (frame-local, normalize=NA) mel on the buffer, and feeds COMPLETE chunks to the
// StreamingSession as stable frames arrive — holding back a `tail_margin` of
// end-of-buffer frames mid-stream (those depend on the mel's trailing pad and
// would change as more audio arrives) and flushing everything on `is_final`. The
// frames it feeds are byte-identical to the whole-clip mel, so a sample stream fed
// through `feed()` then finalized produces the SAME transcript as `--stream`. The
// re-mel-per-flush is O(T²) (fine for a live demo / short clips; a true streaming
// STFT is the optimization). Drives both `--mic` (real capture) and `--mic-sim`.
const IncrementalStreamer = struct {
    sess: *parakeet_streaming.StreamingSession,
    ctx: *fucina.ExecContext,
    file: *const fucina.gguf.File,
    weights: *parakeet_weights.ParakeetWeights,
    arena: std.mem.Allocator,
    feat: parakeet_loader.Featurizer,
    dft_basis: *parakeet_frontend.DftBasis,
    mel_params: parakeet_frontend.MelParams,
    n_mels: usize,
    chunk0: usize,
    chunk_main: usize,
    pre_cache: usize,
    tail_margin: usize,
    samples: std.ArrayList(f32) = .empty,
    buffer_idx: usize = 0,
    first: bool = true,

    fn feed(self: *IncrementalStreamer, new_samples: []const f32, is_final: bool) !void {
        if (new_samples.len > 0) try self.samples.appendSlice(self.arena, new_samples);
        if (self.samples.items.len < self.mel_params.stft.n_fft) {
            if (!is_final) return;
        }
        var mel = try parakeet_frontend.melSpectrogramFastWithBasis(self.arena, self.samples.items, self.mel_params, self.feat.fb, self.feat.window, self.dft_basis);
        defer mel.deinit(self.arena);
        const t = mel.n_frames;
        const stable_t = if (is_final) t else (if (t > self.tail_margin) t - self.tail_margin else 0);

        while (self.buffer_idx < t) {
            const chunk_size = if (self.first) self.chunk0 else self.chunk_main;
            const chunk_hi = @min(self.buffer_idx + chunk_size, t);
            if (chunk_hi <= self.buffer_idx) break;
            if (!is_final and chunk_hi > stable_t) break; // hold pad-affected tail
            const lo = if (self.first) self.buffer_idx else (if (self.buffer_idx > self.pre_cache) self.buffer_idx - self.pre_cache else 0);
            const win_frames = chunk_hi - lo;
            const is_last = is_final and (chunk_hi >= t);
            const win = try self.arena.alloc(f32, self.n_mels * win_frames);
            defer self.arena.free(win);
            for (0..self.n_mels) |m| {
                for (0..win_frames) |tt| win[m * win_frames + tt] = mel.feats[m * t + (lo + tt)];
            }
            try self.sess.feedMelChunk(self.ctx, self.file, self.weights, win, self.n_mels, win_frames, is_last);
            self.buffer_idx += chunk_size;
            self.first = false;
        }
    }
};

// Lock-free SPSC ring for mic samples: the realtime audio callback writes (head),
// the decode loop reads (tail). Fixed capacity; on overrun (decode too slow) the
// oldest unread samples are overwritten (acceptable for a live demo). Power-of-two
// capacity so the modulo is a mask.
const MicRing = struct {
    const cap = 1 << 16; // 65536 samples ≈ 4.1 s @ 16 kHz
    buf: [cap]f32 = undefined,
    head: std.atomic.Value(usize) = .init(0), // write cursor (callback)
    tail: std.atomic.Value(usize) = .init(0), // read cursor (decode loop)

    fn write(self: *MicRing, samples: []const f32) void {
        var h = self.head.load(.monotonic);
        for (samples) |s| {
            self.buf[h & (cap - 1)] = s;
            h += 1;
        }
        self.head.store(h, .release);
    }
    fn read(self: *MicRing, out: []f32) usize {
        const h = self.head.load(.acquire);
        var t = self.tail.load(.monotonic);
        var n: usize = 0;
        while (t < h and n < out.len) : (n += 1) {
            out[n] = self.buf[t & (cap - 1)];
            t += 1;
        }
        self.tail.store(t, .release);
        return n;
    }
};

fn micCallback(user: ?*anyopaque, output: ?[*]f32, input: ?[*]const f32, frame_count: c_uint) callconv(.c) void {
    _ = output;
    const ring: *MicRing = @ptrCast(@alignCast(user.?));
    if (input) |inp| ring.write(inp[0..frame_count]);
}

// --mic: live microphone streaming (NAM miniaudio capture → ring → incremental
// driver → live transcript). Live I/O — manually verified, no automated parity
// (the loop environment has no audio device). Gated behind -Dparakeet-mic so the
// default parakeet build does not link the audio stack.
fn runMic(stdout: *std.Io.Writer, file: *const fucina.gguf.File, io: std.Io, arena: std.mem.Allocator, lang: []const u8) !void {
    if (comptime !build_options.parakeet_mic) {
        try stdout.print("--mic requires building with -Dparakeet-mic=true (links the vendored miniaudio capture stack).\n", .{});
        return;
    } else {
        const audio = @import("nam_audio");
        const cfg = try parakeet_loader.Config.fromGguf(file);
        const sc = (parakeet_loader.StreamingConfig.fromGguf(file) catch null) orelse {
            try stdout.print("--mic requires a streaming model\n", .{});
            return;
        };
        const feat = try parakeet_loader.loadFeaturizer(file, cfg);
        var dft_basis = try parakeet_frontend.DftBasis.init(arena, cfg.n_fft);
        defer dft_basis.deinit();

        var ctx: fucina.ExecContext = undefined;
        ctx.init(arena);
        defer ctx.deinit();
        var weights = parakeet_weights.ParakeetWeights.init(&ctx, file);
        defer weights.deinit();
        var sess = try parakeet_streaming.StreamingSession.init(arena, file, cfg, sc, &weights, lang);
        defer sess.deinit();
        const pieces = try parakeet_loader.loadPieces(file, arena);

        var streamer = IncrementalStreamer{
            .sess = &sess,
            .ctx = &ctx,
            .file = file,
            .weights = &weights,
            .arena = arena,
            .feat = feat,
            .dft_basis = &dft_basis,
            .mel_params = .{
                .stft = .{ .n_fft = cfg.n_fft, .hop = cfg.hop_length, .win_length = cfg.win_length, .mag_power = cfg.mag_power, .preemph = cfg.preemph },
                .n_mels = cfg.n_mels,
                .log_guard = cfg.log_zero_guard,
                .normalize_per_feature = cfg.normalize == .per_feature,
            },
            .n_mels = cfg.n_mels,
            .chunk0 = @intCast(@max(1, sc.chunk_size[0])),
            .chunk_main = @intCast(@max(1, sc.chunk_size[1])),
            .pre_cache = @intCast(@max(0, sc.pre_encode_cache_size[1])),
            .tail_margin = 8,
        };
        defer streamer.samples.deinit(arena);

        var dev = try audio.Audio.init();
        defer dev.deinit();
        var caps: [audio.max_devices]audio.DeviceInfo = undefined;
        const list = try dev.listDevices(.capture, &caps);
        if (list.len == 0) {
            try stdout.print("no capture (microphone) device found\n", .{});
            return;
        }
        var ring = MicRing{};
        try dev.startCapture(0, 16000, 256, micCallback, &ring);
        defer dev.stop();

        const run_seconds: i96 = 20;
        try stdout.print("[mic] listening on '{s}' for {d}s @ 16kHz...\n", .{ list[0].nameSlice(), run_seconds });
        try stdout.flush();

        const t_start = nowNs(io);
        const deadline = t_start + run_seconds * std.time.ns_per_s;
        var drain: [4096]f32 = undefined;
        var last_tokens: usize = 0;
        while (nowNs(io) < deadline) {
            const n = ring.read(&drain);
            if (n == 0) {
                std.Io.sleep(io, .{ .nanoseconds = 10 * std.time.ns_per_ms }, .awake) catch {};
                continue;
            }
            try streamer.feed(drain[0..n], false);
            if (sess.tokens.items.len != last_tokens) {
                last_tokens = sess.tokens.items.len;
                const text = try parakeet_tokenizer.detokenize(arena, pieces, sess.tokens.items);
                try stdout.print("\r{s}", .{text});
                try stdout.flush();
            }
        }
        // Drain any tail samples, then finalize.
        while (true) {
            const n = ring.read(&drain);
            if (n == 0) break;
            try streamer.feed(drain[0..n], false);
        }
        try streamer.feed(&.{}, true);
        const text = try parakeet_tokenizer.detokenize(arena, pieces, sess.tokens.items);
        try stdout.print("\r{s}\n", .{text});
    }
}

// --mic-sim: validate the incremental driver by feeding --audio in small batches
// (a simulated mic) — the resulting transcript must be byte-identical to --stream
// (and so to parakeet-cli --stream). This is the automated gate for the live-mic
// path (the actual capture device is the only manually-verified piece).
fn runMicSim(stdout: *std.Io.Writer, file: *const fucina.gguf.File, io: std.Io, arena: std.mem.Allocator, audio_path: ?[]const u8, lang: []const u8) !void {
    const ap = audio_path orelse {
        try stdout.print("--mic-sim requires --audio <wav>\n", .{});
        return;
    };
    const cfg = try parakeet_loader.Config.fromGguf(file);
    const sc = (parakeet_loader.StreamingConfig.fromGguf(file) catch null) orelse {
        try stdout.print("--mic-sim requires a streaming model\n", .{});
        return;
    };
    const feat = try parakeet_loader.loadFeaturizer(file, cfg);
    const audio = try parakeet_frontend.loadWav16kMonoFile(arena, io, ap);
    var dft_basis = try parakeet_frontend.DftBasis.init(arena, cfg.n_fft);
    defer dft_basis.deinit();

    var ctx: fucina.ExecContext = undefined;
    ctx.init(arena);
    defer ctx.deinit();
    var weights = parakeet_weights.ParakeetWeights.init(&ctx, file);
    defer weights.deinit();
    var sess = try parakeet_streaming.StreamingSession.init(arena, file, cfg, sc, &weights, lang);
    defer sess.deinit();

    var streamer = IncrementalStreamer{
        .sess = &sess,
        .ctx = &ctx,
        .file = file,
        .weights = &weights,
        .arena = arena,
        .feat = feat,
        .dft_basis = &dft_basis,
        .mel_params = .{
            .stft = .{ .n_fft = cfg.n_fft, .hop = cfg.hop_length, .win_length = cfg.win_length, .mag_power = cfg.mag_power, .preemph = cfg.preemph },
            .n_mels = cfg.n_mels,
            .log_guard = cfg.log_zero_guard,
            .normalize_per_feature = cfg.normalize == .per_feature,
        },
        .n_mels = cfg.n_mels,
        .chunk0 = @intCast(@max(1, sc.chunk_size[0])),
        .chunk_main = @intCast(@max(1, sc.chunk_size[1])),
        .pre_cache = @intCast(@max(0, sc.pre_encode_cache_size[1])),
        .tail_margin = 8, // > the mel trailing-pad reach (n_fft/2/hop ≈ 2 frames)
    };
    defer streamer.samples.deinit(arena);

    const batch: usize = 1600; // 100 ms @ 16 kHz — a realistic mic period
    var off: usize = 0;
    while (off < audio.samples.len) {
        const n = @min(batch, audio.samples.len - off);
        try streamer.feed(audio.samples[off .. off + n], false);
        off += n;
    }
    try streamer.feed(&.{}, true); // finalize: flush the tail

    const pieces = try parakeet_loader.loadPieces(file, arena);
    const text = try parakeet_tokenizer.detokenize(arena, pieces, sess.tokens.items);
    try stdout.print("{s}\n", .{text});
}

// --stream-bench: streaming RTF + first-token + per-chunk latency (best of N reps,
// warmed). Replicates the chunk schedule in-line so each chunk's encode+decode can
// be timed. Byte-parity is the --stream / --mic-sim gate; this only measures speed.
fn runStreamBench(stdout: *std.Io.Writer, file: *const fucina.gguf.File, io: std.Io, arena: std.mem.Allocator, audio_path: ?[]const u8, reps: usize, lang: []const u8) !void {
    const ap = audio_path orelse {
        try stdout.print("--stream-bench requires --audio <wav>\n", .{});
        return;
    };
    const cfg = try parakeet_loader.Config.fromGguf(file);
    const sc = (parakeet_loader.StreamingConfig.fromGguf(file) catch null) orelse {
        try stdout.print("--stream-bench requires a streaming model\n", .{});
        return;
    };
    const feat = try parakeet_loader.loadFeaturizer(file, cfg);
    const audio = try parakeet_frontend.loadWav16kMonoFile(arena, io, ap);
    var dft_basis = try parakeet_frontend.DftBasis.init(arena, cfg.n_fft);
    defer dft_basis.deinit();
    const mel_params: parakeet_frontend.MelParams = .{
        .stft = .{ .n_fft = cfg.n_fft, .hop = cfg.hop_length, .win_length = cfg.win_length, .mag_power = cfg.mag_power, .preemph = cfg.preemph },
        .n_mels = cfg.n_mels,
        .log_guard = cfg.log_zero_guard,
        .normalize_per_feature = cfg.normalize == .per_feature,
    };
    const chunk0: usize = @intCast(@max(1, sc.chunk_size[0]));
    const chunk_main: usize = @intCast(@max(1, sc.chunk_size[1]));
    const pre_cache: usize = @intCast(@max(0, sc.pre_encode_cache_size[1]));
    const n_mels = cfg.n_mels;
    const audio_s = @as(f64, @floatFromInt(audio.samples.len)) / @as(f64, @floatFromInt(cfg.sample_rate));

    var best_total_ms: f64 = std.math.inf(f64);
    var best_mel_ms: f64 = 0;
    var best_first_ms: f64 = 0;
    var best_chunk_avg_ms: f64 = 0;
    var best_chunk_max_ms: f64 = 0;
    var n_chunks: usize = 0;
    var n_tokens: usize = 0;

    for (0..reps) |_| {
        var ctx: fucina.ExecContext = undefined;
        ctx.init(arena);
        defer ctx.deinit();
        var weights = parakeet_weights.ParakeetWeights.init(&ctx, file);
        defer weights.deinit();
        var sess = try parakeet_streaming.StreamingSession.init(arena, file, cfg, sc, &weights, lang);
        defer sess.deinit();

        const t0 = nowNs(io);
        var mel = try parakeet_frontend.melSpectrogramFastWithBasis(arena, audio.samples, mel_params, feat.fb, feat.window, &dft_basis);
        defer mel.deinit(arena);
        const t_mel = nowNs(io);
        const tt = mel.n_frames;

        var buffer_idx: usize = 0;
        var first = true;
        var first_token_ns: i96 = 0;
        var sum_chunk_ns: i96 = 0;
        var max_chunk_ns: i96 = 0;
        var chunks: usize = 0;
        while (buffer_idx < tt) {
            const chunk_size = if (first) chunk0 else chunk_main;
            const chunk_hi = @min(buffer_idx + chunk_size, tt);
            if (chunk_hi <= buffer_idx) break;
            const lo = if (first) buffer_idx else (if (buffer_idx > pre_cache) buffer_idx - pre_cache else 0);
            const win_frames = chunk_hi - lo;
            const is_last = chunk_hi >= tt;
            const win = try arena.alloc(f32, n_mels * win_frames);
            defer arena.free(win);
            for (0..n_mels) |m| {
                for (0..win_frames) |t| win[m * win_frames + t] = mel.feats[m * tt + (lo + t)];
            }
            const c0 = nowNs(io);
            try sess.feedMelChunk(&ctx, file, &weights, win, n_mels, win_frames, is_last);
            const c1 = nowNs(io);
            const cn = c1 - c0;
            sum_chunk_ns += cn;
            if (cn > max_chunk_ns) max_chunk_ns = cn;
            if (first_token_ns == 0 and sess.tokens.items.len > 0) first_token_ns = c1 - t0;
            buffer_idx += chunk_size;
            first = false;
            chunks += 1;
        }
        const t_end = nowNs(io);
        const total_ms = @as(f64, @floatFromInt(t_end - t0)) / 1e6;
        if (total_ms < best_total_ms) {
            best_total_ms = total_ms;
            best_mel_ms = @as(f64, @floatFromInt(t_mel - t0)) / 1e6;
            best_first_ms = @as(f64, @floatFromInt(first_token_ns)) / 1e6;
            best_chunk_avg_ms = (@as(f64, @floatFromInt(sum_chunk_ns)) / 1e6) / @as(f64, @floatFromInt(@max(1, chunks)));
            best_chunk_max_ms = @as(f64, @floatFromInt(max_chunk_ns)) / 1e6;
            n_chunks = chunks;
            n_tokens = sess.tokens.items.len;
        }
    }

    const budget_ms = @as(f64, @floatFromInt(chunk_main * cfg.hop_length)) / @as(f64, @floatFromInt(cfg.sample_rate)) * 1000.0;
    try stdout.print("[stream-bench] reps={d} audio={d:.3}s chunks={d} tokens={d}\n", .{ reps, audio_s, n_chunks, n_tokens });
    try stdout.print("  mel={d:.1}ms  total={d:.1}ms  RTF={d:.4} ({d:.1}x realtime)\n", .{ best_mel_ms, best_total_ms, best_total_ms / 1000.0 / audio_s, audio_s / (best_total_ms / 1000.0) });
    try stdout.print("  first-token latency={d:.1}ms\n", .{best_first_ms});
    try stdout.print("  per-chunk encode+decode: avg={d:.2}ms max={d:.2}ms  (real-time budget={d:.0}ms/chunk)\n", .{ best_chunk_avg_ms, best_chunk_max_ms, budget_ms });
}

fn nowNs(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).nanoseconds;
}

// --- parity harness: --compare <stage> <dump.pkd> ---

const Pkd = struct { dtype: i32, dims: [4]usize, ndim: usize, data: []f32 };

fn readPkd(arena: std.mem.Allocator, io: std.Io, path: []const u8) !Pkd {
    var f = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer f.close(io);
    const stat = try f.stat(io);
    const len: usize = @intCast(stat.size);
    const bytes = try arena.alloc(u8, len);
    var read_len: usize = 0;
    while (read_len < bytes.len) {
        const n = try f.readStreaming(io, &.{bytes[read_len..]});
        if (n == 0) break;
        read_len += n;
    }
    if (len < 12 or !std.mem.eql(u8, bytes[0..4], "PKD1")) return error.BadPkd;
    const dtype = std.mem.readInt(i32, bytes[4..8], .little);
    const ndim: usize = @intCast(std.mem.readInt(i32, bytes[8..12], .little));
    if (ndim > 4) return error.BadPkd;
    var dims = [_]usize{ 0, 0, 0, 0 };
    var off: usize = 12;
    for (0..ndim) |i| {
        dims[i] = @intCast(std.mem.readInt(i32, bytes[off..][0..4], .little));
        off += 4;
    }
    const payload = bytes[off..];
    const count = payload.len / 4;
    const data = try arena.alloc(f32, count);
    for (0..count) |i| data[i] = @bitCast(std.mem.readInt(u32, payload[i * 4 ..][0..4], .little));
    return .{ .dtype = dtype, .dims = dims, .ndim = ndim, .data = data };
}

fn runCompare(stdout: *std.Io.Writer, file: *const fucina.gguf.File, io: std.Io, arena: std.mem.Allocator, stage: []const u8, dump_path: []const u8, audio_path: ?[]const u8, tol: f32, lang: []const u8, gate: *bool) !void {
    try stdout.print("\ncompare {s} vs {s} (tol {d}):\n", .{ stage, dump_path, tol });
    const is_stft = std.mem.eql(u8, stage, "stft");
    const is_mel = std.mem.eql(u8, stage, "mel");
    const is_sub = std.mem.eql(u8, stage, "subsampling");
    const is_enc = std.mem.eql(u8, stage, "encoder");
    const is_ctc = std.mem.eql(u8, stage, "ctc");
    const is_tdt = std.mem.eql(u8, stage, "tdt");
    const is_joint0 = std.mem.eql(u8, stage, "joint0");
    const is_stream_mel = std.mem.eql(u8, stage, "stream-mel");
    if (is_stream_mel) {
        // Streaming mel front-end parity. Compute the full mel from --audio
        // (normalize per cfg; realtime model = NA / no per-feature norm), window
        // chunk-0 (frames [0, chunk0)), compare to stream_chunk0_mel.pkd.
        const ap = audio_path orelse {
            try stdout.print("  --compare stream-mel requires --audio <wav>\n", .{});
            return;
        };
        const cfg = parakeet_loader.Config.fromGguf(file) catch |err| {
            try stdout.print("  Config.fromGguf failed: {s}\n", .{@errorName(err)});
            return;
        };
        const mel_dump = try readPkd(arena, io, dump_path); // [n_mels, chunk0]
        const n_mels = mel_dump.dims[0];
        const win = mel_dump.dims[1];
        const feat = try parakeet_loader.loadFeaturizer(file, cfg);
        const audio = try parakeet_frontend.loadWav16kMonoFile(arena, io, ap);
        var dft_basis = try parakeet_frontend.DftBasis.init(arena, cfg.n_fft);
        defer dft_basis.deinit();
        const mel_params: parakeet_frontend.MelParams = .{
            .stft = .{ .n_fft = cfg.n_fft, .hop = cfg.hop_length, .win_length = cfg.win_length, .mag_power = cfg.mag_power, .preemph = cfg.preemph },
            .n_mels = cfg.n_mels,
            .log_guard = cfg.log_zero_guard,
            .normalize_per_feature = cfg.normalize == .per_feature,
        };
        var mel = try parakeet_frontend.melSpectrogramFastWithBasis(arena, audio.samples, mel_params, feat.fb, feat.window, &dft_basis);
        defer mel.deinit(arena);
        // window [0, win): w[m*win + t] = mel[m*T + t].
        const w0 = try arena.alloc(f32, n_mels * win);
        for (0..n_mels) |m| {
            for (0..win) |t| w0[m * win + t] = mel.feats[m * mel.n_frames + t];
        }
        try stdout.print("  fucina mel[:, 0:{d}] (T={d})  dump stream_chunk0_mel [{d},{d}]\n", .{ win, mel.n_frames, mel_dump.dims[0], mel_dump.dims[1] });
        gate.* = compareArrays(stdout, w0, mel_dump.data, null, 0.99999) and gate.*;
        return;
    }
    const is_stream_session = std.mem.eql(u8, stage, "stream-session");
    if (is_stream_session) {
        // Full StreamingSession over the reference mel chunks — encoder.step
        // + RNN-T decode chunk by chunk (caches + decoder state carried) — token
        // sequence vs stream_token_ids. Only the mel FRONT-END is replaced by the
        // dumped chunks (stream-mel covers it). dump_path = stream_token_ids.pkd.
        const cfg = parakeet_loader.Config.fromGguf(file) catch |err| {
            try stdout.print("  Config.fromGguf failed: {s}\n", .{@errorName(err)});
            return;
        };
        const sc = (parakeet_loader.StreamingConfig.fromGguf(file) catch null) orelse {
            try stdout.print("  not a streaming model\n", .{});
            return;
        };
        const slash = std.mem.lastIndexOfScalar(u8, dump_path, '/');
        const dir = if (slash) |s| dump_path[0 .. s + 1] else "";
        const nchunks = (try readPkdI32(arena, io, try std.fmt.allocPrint(arena, "{s}stream_nchunks.pkd", .{dir})))[0];
        const ids = try readPkdI32(arena, io, dump_path);

        var ctx: fucina.ExecContext = undefined;
        ctx.init(arena);
        defer ctx.deinit();
        var pw = parakeet_weights.ParakeetWeights.init(&ctx, file);
        defer pw.deinit();
        var sess = try parakeet_streaming.StreamingSession.init(arena, file, cfg, sc, &pw, lang);
        defer sess.deinit();

        for (0..@intCast(nchunks)) |n| {
            const mel = try readPkd(arena, io, try std.fmt.allocPrint(arena, "{s}stream_mel_chunk{d}.pkd", .{ dir, n }));
            try sess.feedMelChunk(&ctx, file, &pw, mel.data, mel.dims[0], mel.dims[1], n == @as(usize, @intCast(nchunks)) - 1);
        }

        const got = sess.tokens.items;
        var mism: usize = 0;
        for (0..@min(got.len, ids.len)) |i| {
            if (got[i] != ids[i]) {
                mism += 1;
                if (mism == 1) try stdout.print("  first mismatch @ {d}: fucina={d} dump={d}\n", .{ i, got[i], ids[i] });
            }
        }
        const ok = (got.len == ids.len) and (mism == 0);
        try stdout.print("  fucina session tokens n={d} over {d} chunks (eou_events={d})  dump n={d}\n", .{ got.len, nchunks, sess.eou_events, ids.len });
        try stdout.print("  {s} (token sequence {s}byte-identical)\n", .{ if (ok) "PASS" else "FAIL", if (ok) "" else "NOT " });
        gate.* = ok and gate.*;
        return;
    }
    const is_stream_decode = std.mem.eql(u8, stage, "stream-decode");
    if (is_stream_decode) {
        // Streaming RNN-T decoder parity. Feed the reference encoder output
        // (stream_encoder_out.pkd) per-chunk (valid_out_len frames at a time,
        // carrying the RnntDecodeState) through the StreamingSession and compare
        // the non-special token sequence to stream_token_ids.pkd byte-for-byte.
        // dump_path = stream_token_ids.pkd; stream_encoder_out.pkd is its sibling.
        const cfg = parakeet_loader.Config.fromGguf(file) catch |err| {
            try stdout.print("  Config.fromGguf failed: {s}\n", .{@errorName(err)});
            return;
        };
        const sc = (parakeet_loader.StreamingConfig.fromGguf(file) catch null) orelse {
            try stdout.print("  not a streaming model\n", .{});
            return;
        };
        const slash = std.mem.lastIndexOfScalar(u8, dump_path, '/');
        const dir = if (slash) |s| dump_path[0 .. s + 1] else "";
        const enc_dump = try readPkd(arena, io, try std.fmt.allocPrint(arena, "{s}stream_encoder_out.pkd", .{dir}));
        const ids = try readPkdI32(arena, io, dump_path);
        const d = cfg.d_model;
        const t_total = enc_dump.dims[0];
        const vol: usize = @intCast(@max(1, sc.valid_out_len));

        var ctx: fucina.ExecContext = undefined;
        ctx.init(arena);
        defer ctx.deinit();
        var pw = parakeet_weights.ParakeetWeights.init(&ctx, file);
        defer pw.deinit();
        var sess = try parakeet_streaming.StreamingSession.init(arena, file, cfg, sc, &pw, lang);
        defer sess.deinit();

        var off: usize = 0;
        while (off < t_total) {
            const n = @min(vol, t_total - off);
            var ef = try fucina.Tensor(2).fromSlice(&ctx, .{ n, d }, enc_dump.data[off * d ..][0 .. n * d]);
            defer ef.deinit();
            try sess.feedEncoderFrames(&ctx, &ef);
            off += n;
        }

        const got = sess.tokens.items;
        var mism: usize = 0;
        const ncmp = @min(got.len, ids.len);
        for (0..ncmp) |i| {
            if (got[i] != ids[i]) {
                mism += 1;
                if (mism == 1) try stdout.print("  first mismatch @ {d}: fucina={d} dump={d}\n", .{ i, got[i], ids[i] });
            }
        }
        const ok = (got.len == ids.len) and (mism == 0);
        try stdout.print("  fucina tokens n={d} (eou_events={d})  dump stream_token_ids n={d}\n", .{ got.len, sess.eou_events, ids.len });
        try stdout.print("  {s} (token sequence {s}byte-identical)\n", .{ if (ok) "PASS" else "FAIL", if (ok) "" else "NOT " });
        gate.* = ok and gate.*;
        return;
    }
    const is_stream_prompt = std.mem.eql(u8, stage, "stream-prompt");
    if (is_stream_prompt) {
        // Full multi-chunk encoder + prompt_kernel conditioning. Feed all
        // reference mel chunks through StreamingSession.encodeChunkPrompted (enc.step
        // + prompt) and compare to the POST-prompt stream_encoder_out.pkd. Uses
        // --lang to resolve the prompt index. dump_path = stream_encoder_out.pkd.
        const cfg = parakeet_loader.Config.fromGguf(file) catch |err| {
            try stdout.print("  Config.fromGguf failed: {s}\n", .{@errorName(err)});
            return;
        };
        const sc = (parakeet_loader.StreamingConfig.fromGguf(file) catch null) orelse {
            try stdout.print("  not a streaming model\n", .{});
            return;
        };
        const slash = std.mem.lastIndexOfScalar(u8, dump_path, '/');
        const dir = if (slash) |s| dump_path[0 .. s + 1] else "";
        const nchunks = (try readPkdI32(arena, io, try std.fmt.allocPrint(arena, "{s}stream_nchunks.pkd", .{dir})))[0];
        const enc_dump = try readPkd(arena, io, dump_path); // [T_total, d_model] post-prompt
        const d = cfg.d_model;

        var ctx: fucina.ExecContext = undefined;
        ctx.init(arena);
        defer ctx.deinit();
        var pw = parakeet_weights.ParakeetWeights.init(&ctx, file);
        defer pw.deinit();
        var sess = try parakeet_streaming.StreamingSession.init(arena, file, cfg, sc, &pw, lang);
        defer sess.deinit();

        var acc: std.ArrayList(f32) = .empty;
        for (0..@intCast(nchunks)) |n| {
            const mel = try readPkd(arena, io, try std.fmt.allocPrint(arena, "{s}stream_mel_chunk{d}.pkd", .{ dir, n }));
            var y = try sess.encodeChunkPrompted(&ctx, file, &pw, mel.data, mel.dims[0], mel.dims[1], n == @as(usize, @intCast(nchunks)) - 1);
            defer y.deinit();
            try acc.appendSlice(arena, try y.dataConst());
        }
        const got_frames = acc.items.len / d;
        try stdout.print("  fucina stream-prompt [{d},{d}] over {d} chunks (lang={s} pi={d})  dump stream_encoder_out [{d},{d}]\n", .{ got_frames, d, nchunks, lang, sess.prompt_index, enc_dump.dims[0], enc_dump.dims[1] });
        const n_cmp = @min(acc.items.len, enc_dump.data.len);
        gate.* = compareArrays(stdout, acc.items[0..n_cmp], enc_dump.data[0..n_cmp], null, 0.99999) and gate.*;
        return;
    }
    const is_stream_full = std.mem.eql(u8, stage, "stream-full");
    if (is_stream_full) {
        // Full multi-chunk streaming encoder + cache carry-over. Feed all
        // reference mel chunks (stream_mel_chunk{N}.pkd, count = stream_nchunks.pkd)
        // through StreamingEncoder.step in sequence (caches carried), concat the
        // valid outputs, and compare to stream_encoder_out.pkd. dump_path =
        // stream_encoder_out.pkd; the chunk dumps are its siblings.
        const cfg = parakeet_loader.Config.fromGguf(file) catch |err| {
            try stdout.print("  Config.fromGguf failed: {s}\n", .{@errorName(err)});
            return;
        };
        const sc = (parakeet_loader.StreamingConfig.fromGguf(file) catch null) orelse {
            try stdout.print("  not a streaming model\n", .{});
            return;
        };
        const slash = std.mem.lastIndexOfScalar(u8, dump_path, '/');
        const dir = if (slash) |s| dump_path[0 .. s + 1] else "";
        const nchunks = (try readPkdI32(arena, io, try std.fmt.allocPrint(arena, "{s}stream_nchunks.pkd", .{dir})))[0];
        const enc_dump = try readPkd(arena, io, dump_path); // [T_total, d_model]
        const d = cfg.d_model;

        var ctx: fucina.ExecContext = undefined;
        ctx.init(arena);
        defer ctx.deinit();
        var pw = parakeet_weights.ParakeetWeights.init(&ctx, file);
        defer pw.deinit();
        var senc = try parakeet_streaming.StreamingEncoder.init(arena, cfg, sc);
        defer senc.deinit();

        var acc: std.ArrayList(f32) = .empty;
        for (0..@intCast(nchunks)) |n| {
            const mel = try readPkd(arena, io, try std.fmt.allocPrint(arena, "{s}stream_mel_chunk{d}.pkd", .{ dir, n }));
            var y = try senc.step(&ctx, file, cfg, &pw, mel.data, mel.dims[0], mel.dims[1], n == @as(usize, @intCast(nchunks)) - 1);
            defer y.deinit();
            try acc.appendSlice(arena, try y.dataConst());
        }
        const got_frames = acc.items.len / d;
        try stdout.print("  fucina stream-full [{d},{d}] over {d} chunks  dump stream_encoder_out [{d},{d}]\n", .{ got_frames, d, nchunks, enc_dump.dims[0], enc_dump.dims[1] });
        const n_cmp = @min(acc.items.len, enc_dump.data.len);
        gate.* = compareArrays(stdout, acc.items[0..n_cmp], enc_dump.data[0..n_cmp], null, 0.99999) and gate.*;
        return;
    }
    const is_stream_sub = std.mem.eql(u8, stage, "stream-sub");
    if (is_stream_sub) {
        // Streaming causal subsampling parity. Feed the reference chunk-0
        // mel window (stream_chunk0_mel.pkd) through streamingSubsample and compare
        // to stream_chunk0_sub.pkd. dump_path = stream_chunk0_sub.pkd (sibling mel).
        const cfg = parakeet_loader.Config.fromGguf(file) catch |err| {
            try stdout.print("  Config.fromGguf failed: {s}\n", .{@errorName(err)});
            return;
        };
        const slash = std.mem.lastIndexOfScalar(u8, dump_path, '/');
        const dir = if (slash) |s| dump_path[0 .. s + 1] else "";
        const mel_path = try std.fmt.allocPrint(arena, "{s}stream_chunk0_mel.pkd", .{dir});
        const mel = try readPkd(arena, io, mel_path); // [n_mels, T]
        const sub_dump = try readPkd(arena, io, dump_path); // [T', d_model]
        const n_mels = mel.dims[0];
        const t_in = mel.dims[1];

        var ctx: fucina.ExecContext = undefined;
        ctx.init(arena);
        defer ctx.deinit();
        var pw = parakeet_weights.ParakeetWeights.init(&ctx, file);
        defer pw.deinit();
        var sub = try parakeet_subsampling.streamingSubsample(&ctx, file, cfg, &pw, mel.data, n_mels, t_in, t_in);
        defer sub.deinit();
        const sv = sub.shape();
        try stdout.print("  fucina stream-sub [{d},{d}]  dump stream_chunk0_sub [{d},{d}]\n", .{ sv[0], sv[1], sub_dump.dims[0], sub_dump.dims[1] });
        gate.* = compareArrays(stdout, try sub.dataConst(), sub_dump.data, null, 0.99999) and gate.*;
        return;
    }
    const is_stream_enc = std.mem.eql(u8, stage, "stream-encoder");
    if (is_stream_enc) {
        // Streaming conformer layer-stack parity. Feed the reference
        // post-subsampling chunk-0 input (stream_chunk0_sub.pkd) through the 17
        // streaming layers (fresh caches) and compare to stream_chunk0_out.pkd.
        // dump_path = stream_chunk0_out.pkd; stream_chunk0_sub.pkd is its sibling.
        const cfg = parakeet_loader.Config.fromGguf(file) catch |err| {
            try stdout.print("  Config.fromGguf failed: {s}\n", .{@errorName(err)});
            return;
        };
        const sc = (parakeet_loader.StreamingConfig.fromGguf(file) catch null) orelse {
            try stdout.print("  not a streaming model (no streaming.* metadata)\n", .{});
            return;
        };
        const slash = std.mem.lastIndexOfScalar(u8, dump_path, '/');
        const dir = if (slash) |s| dump_path[0 .. s + 1] else "";
        const sub_path = try std.fmt.allocPrint(arena, "{s}stream_chunk0_sub.pkd", .{dir});
        const sub = try readPkd(arena, io, sub_path); // [Tc, d_model]
        const out_dump = try readPkd(arena, io, dump_path); // [Tc, d_model]
        const d = cfg.d_model;
        const tc = sub.dims[0];

        var ctx: fucina.ExecContext = undefined;
        ctx.init(arena);
        defer ctx.deinit();
        var pw = parakeet_weights.ParakeetWeights.init(&ctx, file);
        defer pw.deinit();
        var senc = try parakeet_streaming.StreamingEncoder.init(arena, cfg, sc);
        defer senc.deinit();

        var x = try fucina.Tensor(2).fromSlice(&ctx, .{ tc, d }, sub.data[0 .. tc * d]);
        defer x.deinit();
        var y = try senc.layerStack(&ctx, &pw, cfg, &x);
        defer y.deinit();

        try stdout.print("  fucina stream layer-stack [{d},{d}]  dump stream_chunk0_out [{d},{d}]\n", .{ tc, d, out_dump.dims[0], out_dump.dims[1] });
        gate.* = compareArrays(stdout, try y.dataConst(), out_dump.data[0 .. tc * d], null, 0.99999) and gate.*;
        return;
    }
    if (!is_stft and !is_mel and !is_sub and !is_enc and !is_ctc and !is_tdt and !is_joint0) {
        try stdout.print("  unknown --compare stage '{s}'\n", .{stage});
        return;
    }
    if (is_joint0) {
        // Isolated predictor+joint step-0 parity: feed parakeet's exact encoder
        // output (encoder_out.pkd, sibling of the tdt_logits dump), run the SOS
        // predictor + joint, compare to tdt_logits row 0. No audio/encode needed.
        const cfg = parakeet_loader.Config.fromGguf(file) catch |err| {
            try stdout.print("  Config.fromGguf failed: {s}\n", .{@errorName(err)});
            return;
        };
        const slash = std.mem.lastIndexOfScalar(u8, dump_path, '/');
        const dir = if (slash) |s| dump_path[0 .. s + 1] else "";
        const enc_path = try std.fmt.allocPrint(arena, "{s}encoder_out.pkd", .{dir});
        const enc_dump = try readPkd(arena, io, enc_path); // [d_model, T']
        const tdt_dump = try readPkd(arena, io, dump_path); // [n_steps, V_plus]
        const dm = cfg.d_model;
        const tp = enc_dump.dims[1];
        const vp = cfg.vPlus();

        // enc frame 0: enc_t[c] = encoder_out[c*T' + 0].
        const enc_t = try arena.alloc(f32, dm);
        for (0..dm) |c| enc_t[c] = enc_dump.data[c * tp + 0];

        var ctx: fucina.ExecContext = undefined;
        ctx.init(arena);
        defer ctx.deinit();
        var pw = parakeet_weights.ParakeetWeights.init(&ctx, file);
        defer pw.deinit();
        var pred = try parakeet_decoder.Predictor.init(arena, &pw, cfg);
        defer pred.deinit();
        var joint = try parakeet_decoder.Joint.init(arena, &pw, cfg);
        defer joint.deinit();

        const h = cfg.pred_hidden;
        const zeros = try arena.alloc(f32, h);
        @memset(zeros, 0);
        const g = try arena.alloc(f32, h);
        const ho = try arena.alloc(f32, h);
        const co = try arena.alloc(f32, h);
        try pred.step(&ctx, -1, true, zeros, zeros, g, ho, co); // SOS

        // enc frame 0 as [1, d_model] → encProjAll → [1, jh].
        var enc_t_tensor = try fucina.Tensor(2).fromSlice(&ctx, .{ 1, dm }, enc_t);
        defer enc_t_tensor.deinit();
        var enc_proj0_t = try joint.encProjAll(&enc_t_tensor);
        defer enc_proj0_t.deinit();
        const logits0 = try arena.alloc(f32, vp);
        try joint.step(&ctx, (try enc_proj0_t.dataConst())[0..joint.jh], g, logits0);

        try stdout.print("  fucina: joint logits[0] [{d}]  dump tdt_logits: [{d}, {d}]\n", .{ vp, tdt_dump.dims[0], tdt_dump.dims[1] });
        if (tdt_dump.dims[1] != vp) {
            try stdout.print("  SHAPE MISMATCH\n", .{});
            return;
        }
        gate.* = compareArrays(stdout, logits0, tdt_dump.data[0..vp], null, 0.9999) and gate.*; // cosine-gated (§4)
        return;
    }
    const ap = audio_path orelse {
        try stdout.print("  --compare {s} requires --audio <wav>\n", .{stage});
        return;
    };
    const cfg = parakeet_loader.Config.fromGguf(file) catch |err| {
        try stdout.print("  Config.fromGguf failed: {s}\n", .{@errorName(err)});
        return;
    };
    const feat = parakeet_loader.loadFeaturizer(file, cfg) catch |err| {
        try stdout.print("  loadFeaturizer failed: {s}\n", .{@errorName(err)});
        return;
    };
    const audio = parakeet_frontend.loadWav16kMonoFile(arena, io, ap) catch |err| {
        try stdout.print("  loadWav failed: {s}\n", .{@errorName(err)});
        return;
    };
    const stft_params = parakeet_frontend.StftParams{
        .n_fft = cfg.n_fft,
        .hop = cfg.hop_length,
        .win_length = cfg.win_length,
        .mag_power = cfg.mag_power,
        .preemph = cfg.preemph,
    };
    const dump = try readPkd(arena, io, dump_path);

    if (is_stft) {
        const spec = try parakeet_frontend.stftPower(arena, audio.samples, stft_params, feat.window);
        try stdout.print("  fucina: [T={d}, bins={d}]  dump: [{d}, {d}]\n", .{ spec.n_frames, spec.n_bins, dump.dims[0], dump.dims[1] });
        if (dump.ndim != 2 or dump.dims[0] != spec.n_frames or dump.dims[1] != spec.n_bins) {
            try stdout.print("  SHAPE MISMATCH\n", .{});
            return;
        }
        gate.* = compareArrays(stdout, spec.power, dump.data, tol, null) and gate.*;
    } else if (is_mel) {
        const mel = try parakeet_frontend.melSpectrogram(arena, audio.samples, .{
            .stft = stft_params,
            .n_mels = cfg.n_mels,
            .log_guard = cfg.log_zero_guard,
            .normalize_per_feature = cfg.normalize == .per_feature,
        }, feat.fb, feat.window);
        // mel.pkd is feat-major [n_mels, T]; melSpectrogram is feats[m*T+t] too.
        try stdout.print("  fucina: [n_mels={d}, T={d}]  dump: [{d}, {d}]\n", .{ mel.n_mels, mel.n_frames, dump.dims[0], dump.dims[1] });
        if (dump.ndim != 2 or dump.dims[0] != mel.n_mels or dump.dims[1] != mel.n_frames) {
            try stdout.print("  SHAPE MISMATCH\n", .{});
            return;
        }
        gate.* = compareArrays(stdout, mel.feats, dump.data, tol, 0.9999) and gate.*;
    } else if (is_sub) {
        var mel = try parakeet_frontend.melSpectrogram(arena, audio.samples, .{
            .stft = stft_params,
            .n_mels = cfg.n_mels,
            .log_guard = cfg.log_zero_guard,
            .normalize_per_feature = cfg.normalize == .per_feature,
        }, feat.fb, feat.window);
        defer mel.deinit(arena);

        var ctx: fucina.ExecContext = undefined;
        ctx.init(arena);
        defer ctx.deinit();
        var out = try parakeet_subsampling.subsample(&ctx, file, cfg, mel.feats, cfg.n_mels, mel.n_frames);
        defer out.deinit();
        const view = out.shape(); // [T', d_model], row-major out[t*d_model+c]
        try stdout.print("  fucina: [T'={d}, d_model={d}]  dump: [{d}, {d}]\n", .{ view[0], view[1], dump.dims[0], dump.dims[1] });
        if (dump.ndim != 2 or dump.dims[0] != view[0] or dump.dims[1] != view[1]) {
            try stdout.print("  SHAPE MISMATCH\n", .{});
            return;
        }
        gate.* = compareArrays(stdout, try out.dataConst(), dump.data, null, 0.9999) and gate.*; // cosine-gated (§4)
    } else if (is_enc) {
        var mel = try parakeet_frontend.melSpectrogram(arena, audio.samples, .{
            .stft = stft_params,
            .n_mels = cfg.n_mels,
            .log_guard = cfg.log_zero_guard,
            .normalize_per_feature = cfg.normalize == .per_feature,
        }, feat.fb, feat.window);
        defer mel.deinit(arena);

        var ctx: fucina.ExecContext = undefined;
        ctx.init(arena);
        defer ctx.deinit();
        var out = try parakeet_encoder.encode(&ctx, file, cfg, mel.feats, cfg.n_mels, mel.n_frames);
        defer out.deinit();
        const view = out.shape(); // [T', d_model] row-major out[t*d_model+c]
        const tp = view[0];
        const dm = view[1];
        // encoder_out.pkd is [d_model, T'] (channels-first enc[c*T'+t]); transpose ours.
        try stdout.print("  fucina: [T'={d}, d_model={d}] -> [{d}, {d}]  dump: [{d}, {d}]\n", .{ tp, dm, dm, tp, dump.dims[0], dump.dims[1] });
        if (dump.ndim != 2 or dump.dims[0] != dm or dump.dims[1] != tp) {
            try stdout.print("  SHAPE MISMATCH\n", .{});
            return;
        }
        const od = try out.dataConst();
        const tr = try arena.alloc(f32, tp * dm);
        for (0..dm) |c| {
            for (0..tp) |t| tr[c * tp + t] = od[t * dm + c];
        }
        gate.* = compareArrays(stdout, tr, dump.data, null, 0.9999) and gate.*; // cosine-gated (§4)
    } else { // ctc | tdt — token-id EXACT parity (the hard target)
        var mel = try parakeet_frontend.melSpectrogram(arena, audio.samples, .{
            .stft = stft_params,
            .n_mels = cfg.n_mels,
            .log_guard = cfg.log_zero_guard,
            .normalize_per_feature = cfg.normalize == .per_feature,
        }, feat.fb, feat.window);
        defer mel.deinit(arena);

        var ctx: fucina.ExecContext = undefined;
        ctx.init(arena);
        defer ctx.deinit();
        var enc = try parakeet_encoder.encode(&ctx, file, cfg, mel.feats, cfg.n_mels, mel.n_frames);
        defer enc.deinit();
        const ids = if (is_ctc)
            try parakeet_decoder.ctcDecode(&ctx, file, cfg, &enc, arena, null)
        else
            try parakeet_decoder.tdtDecode(&ctx, file, cfg, &enc, arena, null);
        const ref = try readPkdI32(arena, io, dump_path);

        try stdout.print("  fucina ids: {d}, dump ids: {d}\n", .{ ids.len, ref.len });
        try printIds(stdout, "  fucina", ids);
        try printIds(stdout, "  dump  ", ref);
        var exact = ids.len == ref.len;
        if (exact) {
            for (ids, ref) |a, b| {
                if (a != b) {
                    exact = false;
                    break;
                }
            }
        }
        try stdout.print("  {s} (token-id exact match)\n", .{if (exact) "PASS" else "FAIL"});
        gate.* = exact and gate.*;
    }
}

fn printIds(stdout: *std.Io.Writer, label: []const u8, ids: []const i32) !void {
    try stdout.print("{s} [", .{label});
    const n = @min(ids.len, 12);
    for (0..n) |i| {
        if (i != 0) try stdout.print(",", .{});
        try stdout.print("{d}", .{ids[i]});
    }
    if (ids.len > n) try stdout.print(",…", .{});
    try stdout.print("]\n", .{});
}

fn readPkdI32(arena: std.mem.Allocator, io: std.Io, path: []const u8) ![]i32 {
    var f = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer f.close(io);
    const stat = try f.stat(io);
    const len: usize = @intCast(stat.size);
    const bytes = try arena.alloc(u8, len);
    var read_len: usize = 0;
    while (read_len < bytes.len) {
        const n = try f.readStreaming(io, &.{bytes[read_len..]});
        if (n == 0) break;
        read_len += n;
    }
    if (len < 12 or !std.mem.eql(u8, bytes[0..4], "PKD1")) return error.BadPkd;
    const ndim: usize = @intCast(std.mem.readInt(i32, bytes[8..12], .little));
    const payload = bytes[12 + 4 * ndim ..];
    const count = payload.len / 4;
    const out = try arena.alloc(i32, count);
    for (0..count) |i| out[i] = std.mem.readInt(i32, payload[i * 4 ..][0..4], .little);
    return out;
}

/// Pure pass/fail gate for a numeric comparison: max-abs within `tol` (when set)
/// AND cosine >= `cos_min` (when set); both null is vacuously true. Extracted so
/// the gate logic is unit-testable without a writer, and is the SINGLE source of
/// truth that `compareArrays` reports and `runCompare`/`main` enforce as an exit
/// code (making `--compare` mechanically enforcing).
pub fn compareGate(a: []const f32, b: []const f32, tol: ?f32, cos_min: ?f64) bool {
    var max_abs: f64 = 0;
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (a, b) |x, y| {
        const d = @abs(@as(f64, x) - @as(f64, y));
        if (d > max_abs) max_abs = d;
        dot += @as(f64, x) * @as(f64, y);
        na += @as(f64, x) * @as(f64, x);
        nb += @as(f64, y) * @as(f64, y);
    }
    const cosine = if (na > 0 and nb > 0) dot / (@sqrt(na) * @sqrt(nb)) else 1.0;
    return (tol == null or max_abs <= tol.?) and (cos_min == null or cosine >= cos_min.?);
}

/// Print comparison stats and the PASS/FAIL verdict; RETURNS the verdict so the
/// caller can enforce it as an exit code.
fn compareArrays(stdout: *std.Io.Writer, a: []const f32, b: []const f32, tol: ?f32, cos_min: ?f64) bool {
    var max_abs: f64 = 0;
    var max_idx: usize = 0;
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (a, b, 0..) |x, y, i| {
        const d = @abs(@as(f64, x) - @as(f64, y));
        if (d > max_abs) {
            max_abs = d;
            max_idx = i;
        }
        dot += @as(f64, x) * @as(f64, y);
        na += @as(f64, x) * @as(f64, x);
        nb += @as(f64, y) * @as(f64, y);
    }
    const cosine = if (na > 0 and nb > 0) dot / (@sqrt(na) * @sqrt(nb)) else 1.0;
    // Relative-error stats over elements with |dump|>1 (distinguishes broad
    // precision noise from outliers/permutation bugs).
    var max_rel: f64 = 0;
    var n_rel_1em3: usize = 0;
    var n_big: usize = 0;
    for (a, b) |x, y| {
        if (@abs(@as(f64, y)) <= 1.0) continue;
        n_big += 1;
        const rel = @abs(@as(f64, x) - @as(f64, y)) / @abs(@as(f64, y));
        if (rel > max_rel) max_rel = rel;
        if (rel > 1e-3) n_rel_1em3 += 1;
    }
    stdout.print("  n={d} max_abs={e:.3} at idx {d} (fucina={d:.6} dump={d:.6}) cosine={d:.8}\n", .{
        a.len, max_abs, max_idx, a[max_idx], b[max_idx], cosine,
    }) catch {};
    stdout.print("  rel(|dump|>1): max={e:.3} mean-of-big n={d}, with rel>1e-3: {d}/{d}\n", .{
        max_rel, n_big, n_rel_1em3, n_big,
    }) catch {};
    stdout.print("  first6 fucina: ", .{}) catch {};
    for (0..@min(a.len, 6)) |i| stdout.print("{d:.5} ", .{a[i]}) catch {};
    stdout.print("\n  first6 dump  : ", .{}) catch {};
    for (0..@min(b.len, 6)) |i| stdout.print("{d:.5} ", .{b[i]}) catch {};
    stdout.print("\n", .{}) catch {};
    const ok = compareGate(a, b, tol, cos_min);
    if (tol) |t| {
        stdout.print("  {s} (max_abs {e:.3} vs tol {e:.3}{s})\n", .{ if (ok) "PASS" else "FAIL", max_abs, t, if (cos_min != null) " + cosine gate" else "" }) catch {};
    } else {
        // Intermediate stage: max_abs informational (f16 weights + op-order make
        // bit-exact unrealistic); gate on cosine.
        stdout.print("  {s} (cosine {d:.8} >= 0.9999; max_abs {e:.3} informational)\n", .{ if (ok) "PASS" else "FAIL", cosine, max_abs }) catch {};
    }
    return ok;
}

fn printUsage(stdout: *std.Io.Writer) !void {
    try stdout.print(
        \\usage: zig build parakeet -- --model <model.gguf> [options]
        \\  --model <path>            parakeet GGUF (required); prints arch/config/tensors
        \\  --audio <path>            16 kHz mono WAV to transcribe
        \\  --transcribe              offline transcription of --audio
        \\  --stream                  streaming pipeline (cache-aware chunked encode) over --audio
        \\  --manifest <file>         batch transcription: one audio path per line
        \\  --mic                     live microphone capture (needs -Dparakeet-mic)
        \\  --mic-sim                 feed --audio through the incremental mic driver
        \\  --stream-bench            streaming RTF + first-token latency benchmark
        \\  --json                    JSON output (offline decode only)
        \\  --timestamps              per-word start/end/confidence (offline decode only)
        \\  --threads <n>             worker thread count (0 = default)
        \\  --decoder tdt|ctc         decoder head for hybrid models (default tdt)
        \\  --lang <XX>               target locale for multilingual prompt-conditioned models (default auto)
        \\  --compare <stage> <dump>  parity-check a stage vs a parakeet.cpp PKD1 dump
        \\                            (stages: mel, subsampling, encoder, ctc_logits, tdt_logits, token_ids)
        \\  --f32-cache               cache sequence linear weights as f32 and route them through BLAS
        \\  --fast-mel                BLAS mel filterbank projection (on by default; --no-fast-mel disables)
        \\  --bench-reps <n>          run transcribe n times in one loaded session and report best timing
        \\  --tol <f>                 max-abs tolerance for --compare (default 1e-4)
        \\  --help                    this message
        \\
    , .{});
}

fn printConfig(
    stdout: *std.Io.Writer,
    file: *const fucina.gguf.File,
    path: []const u8,
    arena: std.mem.Allocator,
) !void {
    try stdout.print("parakeet model: {s}\n", .{path});

    // Curated summary (mirrors parakeet-cli `info` for easy cross-checking).
    try stdout.print("  general.architecture : {?s}\n", .{file.getString("general.architecture")});
    try stdout.print("  parakeet.arch        : {?s}\n", .{file.getString("parakeet.arch")});
    try stdout.print("  encoder              : d_model={?d} layers={?d} heads={?d} ff_dim={?d} feat_in={?d}\n", .{
        file.getInt("parakeet.encoder.d_model"),
        file.getInt("parakeet.encoder.n_layers"),
        file.getInt("parakeet.encoder.n_heads"),
        file.getInt("parakeet.encoder.ff_dim"),
        file.getInt("parakeet.encoder.feat_in"),
    });
    try stdout.print("  conv/subsampling     : kernel={?d} norm={?s} factor=x{?d} conv_channels={?d}\n", .{
        file.getInt("parakeet.encoder.conv_kernel"),
        file.getString("parakeet.encoder.conv_norm_type"),
        file.getInt("parakeet.encoder.subsampling_factor"),
        file.getInt("parakeet.encoder.subsampling_conv_channels"),
    });
    try stdout.print("  preprocessor         : sr={?d} n_mels={?d} n_fft={?d} win={?d} hop={?d} preemph={?d} mag_power={?d}\n", .{
        file.getInt("parakeet.preprocessor.sample_rate"),
        file.getInt("parakeet.preprocessor.n_mels"),
        file.getInt("parakeet.preprocessor.n_fft"),
        file.getInt("parakeet.preprocessor.win_length"),
        file.getInt("parakeet.preprocessor.hop_length"),
        file.getFloat("parakeet.preprocessor.preemph"),
        file.getFloat("parakeet.preprocessor.mag_power"),
    });
    try stdout.print("  preprocessor.normalize : {?s}\n", .{file.getString("parakeet.preprocessor.normalize")});
    try stdout.print("  vocab/blank          : {?d} / {?d}\n", .{
        file.getInt("parakeet.vocab_size"),
        file.getInt("parakeet.blank_id"),
    });
    try stdout.print("  decoder              : pred_hidden={?d} pred_rnn_layers={?d}  max_symbols={?d}\n", .{
        file.getInt("parakeet.decoder.pred_hidden"),
        file.getInt("parakeet.decoder.pred_rnn_layers"),
        file.getInt("parakeet.decoding.max_symbols"),
    });
    try stdout.print("  joint                : joint_hidden={?d} activation={?s}\n", .{
        file.getInt("parakeet.joint.joint_hidden"),
        file.getString("parakeet.joint.activation"),
    });
    // TDT durations (a small int array): decode and print inline.
    if (file.getArray("parakeet.tdt.durations")) |arr| {
        try stdout.print("  tdt.durations        : ", .{});
        try printIntArray(stdout, arr);
        try stdout.print("\n", .{});
    }

    // Streaming config: present only on streaming-variant models.
    if (parakeet_loader.StreamingConfig.fromGguf(file) catch null) |sc| {
        try stdout.print("  streaming            : att_context=[{d},{d}] {s}  chunk={d}/{d} shift={d}/{d} pre_enc_cache={d}/{d} last_ch_cache={d} valid_out_len={d} drop_extra={d}\n", .{
            sc.att_context_left,        sc.att_context_right,        @tagName(sc.att_context_style),
            sc.chunk_size[0],           sc.chunk_size[1],            sc.shift_size[0],
            sc.shift_size[1],           sc.pre_encode_cache_size[0], sc.pre_encode_cache_size[1],
            sc.last_channel_cache_size, sc.valid_out_len,            sc.drop_extra_pre_encoded,
        });
    }

    // Full metadata dump (sorted; big arrays summarized, never expanded).
    try stdout.print("\nmetadata ({d} keys):\n", .{file.metadata.count()});
    const keys = try arena.alloc([]const u8, file.metadata.count());
    var it = file.metadata.iterator();
    var n: usize = 0;
    while (it.next()) |entry| : (n += 1) keys[n] = entry.key_ptr.*;
    std.mem.sort([]const u8, keys, {}, lessThanStr);
    for (keys) |k| {
        try stdout.print("  {s} = ", .{k});
        try fmtValue(stdout, file.metadata.get(k).?);
        try stdout.print("\n", .{});
    }

    // Tensor map: name, dtype, shape.
    try stdout.print("\ntensors ({d}):\n", .{file.tensors.len});
    for (file.tensors) |t| {
        try stdout.print("  {s:<52} {s:<7} [", .{ t.name, @tagName(t.ggml_type) });
        for (0..t.n_dims) |i| {
            if (i != 0) try stdout.print(", ", .{});
            try stdout.print("{d}", .{t.dims[i]});
        }
        try stdout.print("]\n", .{});
    }
}

// Front-end inspection (--audio without a decode mode): decode the WAV to mono
// 16 kHz f32 and apply NeMo preemphasis, printing the leading samples of each.
fn runAudio(stdout: *std.Io.Writer, file: *const fucina.gguf.File, io: std.Io, arena: std.mem.Allocator, path: []const u8) !void {
    try stdout.print("\naudio: {s}\n", .{path});
    const cfg = parakeet_loader.Config.fromGguf(file) catch |err| {
        try stdout.print("  Config.fromGguf failed: {s}\n", .{@errorName(err)});
        return;
    };
    const audio = parakeet_frontend.loadWav16kMonoFile(arena, io, path) catch |err| {
        try stdout.print("  loadWav failed: {s}\n", .{@errorName(err)});
        return;
    };
    try stdout.print("  samples={d} sample_rate={d} duration={d:.3}s\n", .{
        audio.samples.len,
        audio.sample_rate,
        @as(f64, @floatFromInt(audio.samples.len)) / @as(f64, @floatFromInt(audio.sample_rate)),
    });
    try printHead(stdout, "  raw[0..6]   ", audio.samples);

    const pre = try arena.alloc(f32, audio.samples.len);
    try parakeet_frontend.preemphasis(audio.samples, pre, cfg.preemph);
    try stdout.print("  preemph coeff={d}\n", .{cfg.preemph});
    try printHead(stdout, "  preemph[0..6]", pre);
    try stdout.print("  (inspection stops at preemphasis; use --transcribe for the full pipeline)\n", .{});
}

fn printHead(stdout: *std.Io.Writer, label: []const u8, xs: []const f32) !void {
    try stdout.print("{s} = [", .{label});
    const n = @min(xs.len, 6);
    for (0..n) |i| {
        if (i != 0) try stdout.print(", ", .{});
        try stdout.print("{d:.6}", .{xs[i]});
    }
    try stdout.print("]\n", .{});
}

// Drive the loader on the real model: parse Config, resolve+validate every
// expected tensor, borrow the featurizer + pieces. This is the "loads the
// GGUF; every expected tensor resolves" gate.
fn printLoader(stdout: *std.Io.Writer, file: *const fucina.gguf.File, arena: std.mem.Allocator) !void {
    try stdout.print("\nloader:\n", .{});
    const cfg = parakeet_loader.Config.fromGguf(file) catch |err| {
        try stdout.print("  Config.fromGguf failed: {s}\n", .{@errorName(err)});
        return;
    };
    try stdout.print("  config: arch={s} d_model={d} layers={d} heads={d} head_dim={d} v_plus={d} durations={d}\n", .{
        @tagName(cfg.arch), cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.head_dim, cfg.vPlus(), cfg.num_durations,
    });
    const count = parakeet_loader.validateTensors(file, cfg) catch |err| {
        try stdout.print("  validateTensors failed: {s}\n", .{@errorName(err)});
        return;
    };
    try stdout.print("  tensors: {d} expected tensors resolved + shape/dtype-validated OK\n", .{count});

    const feat = parakeet_loader.loadFeaturizer(file, cfg) catch |err| {
        try stdout.print("  loadFeaturizer failed: {s}\n", .{@errorName(err)});
        return;
    };
    try stdout.print("  featurizer: fb [{d} mels x {d} bins] window [{d}]\n", .{ feat.fb_mels, feat.fb_bins, feat.window.len });

    const pieces = parakeet_loader.loadPieces(file, arena) catch |err| {
        try stdout.print("  loadPieces failed: {s}\n", .{@errorName(err)});
        return;
    };
    try stdout.print("  tokenizer: {d} pieces (e.g. [0]={s})\n", .{ pieces.len, if (pieces.len > 0) pieces[0] else "" });
}

fn fmtValue(stdout: *std.Io.Writer, v: Value) !void {
    switch (v) {
        .int => |x| try stdout.print("{d}", .{x}),
        .float => |x| try stdout.print("{d}", .{x}),
        .boolean => |x| try stdout.print("{}", .{x}),
        .string => |s| try stdout.print("\"{s}\"", .{s}),
        .array => |a| try stdout.print("array(item_type={d}, len={d})", .{ a.item_type, a.len }),
    }
}

// Decode a GGUF int array (item types 4=i32, 5=u32, 10=i64, 11=u64, etc. all
// fit i64 for the small config arrays we print here) and emit "[a, b, c]".
fn printIntArray(stdout: *std.Io.Writer, arr: fucina.gguf.Array) !void {
    const elem_size: usize = switch (arr.item_type) {
        4, 5 => 4, // i32 / u32
        10, 11 => 8, // i64 / u64
        else => {
            try stdout.print("array(item_type={d}, len={d})", .{ arr.item_type, arr.len });
            return;
        },
    };
    try stdout.print("[", .{});
    var i: usize = 0;
    while (i < arr.len) : (i += 1) {
        if (i != 0) try stdout.print(", ", .{});
        const off = i * elem_size;
        const val: i64 = switch (arr.item_type) {
            4 => std.mem.readInt(i32, arr.data[off..][0..4], .little),
            5 => @intCast(std.mem.readInt(u32, arr.data[off..][0..4], .little)),
            10 => std.mem.readInt(i64, arr.data[off..][0..8], .little),
            11 => @intCast(std.mem.readInt(u64, arr.data[off..][0..8], .little)),
            else => unreachable,
        };
        try stdout.print("{d}", .{val});
    }
    try stdout.print("]", .{});
}

fn lessThanStr(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.lessThan(u8, a, b);
}

test {
    _ = @import("parakeet_tests.zig");
}
