//! OmniVoice example — see examples/omnivoice/README.md for models and usage.
//!
//! Fucina port of omnivoice.cpp (k2-fsa/OmniVoice): MaskGIT non-autoregressive
//! multilingual TTS on a Qwen3-0.6B backbone with the Higgs Audio v2 codec
//! (HuBERT semantic + DAC acoustic + 8-codebook RVQ @ 25 fps, 24 kHz mono).
//! CPU-only; voice cloning and voice design are the primary modes.
//!
//! Reference: omnivoice.cpp (github.com/ServeurpersoCom/omnivoice.cpp) — the
//! parity baseline: token and RVQ code streams match it byte-exact (F32,
//! fixed seed).

const std = @import("std");
const fucina = @import("fucina");
const llm = @import("fucina_llm");

const philox = @import("omnivoice/philox.zig");
const resample = @import("omnivoice/resample.zig");
const wav = @import("omnivoice/wav.zig");
const dump = @import("omnivoice/dump.zig");
const rvq_file = @import("omnivoice/rvq_file.zig");
const postproc = @import("omnivoice/postproc.zig");
const postproc_stream = @import("omnivoice/postproc_stream.zig");
const chunker = @import("omnivoice/chunker.zig");
const chunker_stream = @import("omnivoice/chunker_stream.zig");
const duration = @import("omnivoice/duration.zig");
const voicedesign = @import("omnivoice/voicedesign.zig");
const langmap = @import("omnivoice/langmap.zig");
const maskgit = @import("omnivoice/maskgit.zig");
const mg_decode = @import("omnivoice/mg_decode.zig");
const play = @import("omnivoice/play.zig");
const codec = @import("omnivoice/codec.zig");
const rvq = @import("omnivoice/rvq.zig");
const dac = @import("omnivoice/dac.zig");
const lm = @import("omnivoice/lm.zig");
const pipeline = @import("omnivoice/pipeline.zig");

const usage =
    \\fucina OmniVoice — MaskGIT TTS from GGUF (guide: examples/omnivoice/README.md)
    \\
    \\usage: zig build omnivoice [-Doptimize=ReleaseFast] -- <command> [args]
    \\
    \\commands:
    \\  tts --model <base.gguf> --codec <tokenizer.gguf> [options] [-o <out.wav>] [--play] < text.txt
    \\      synthesize speech from the text on stdin (24 kHz mono WAV);
    \\      progress/liveness signals go to stderr ([Load]/[MaskGIT]/[TTS-Long]/[Perf])
    \\      required:
    \\        --model <gguf>          LLM GGUF (F32 / BF16 / Q8_0 / Q4_K_M)
    \\        --codec <gguf>          codec GGUF (omnivoice-tokenizer-*.gguf; synthesis only)
    \\      output (at least one of -o / --play):
    \\        -o <path>               output WAV path; '-' streams to stdout (pipe friendly):
    \\                                stdin is read incrementally and synthesis starts as soon
    \\                                as the first sentence boundary is reached (logs -> stderr)
    \\        --play                  play through the speakers; with -o <file> plays the final
    \\                                waveform after writing it (gapless), alone it streams each
    \\                                chunk as synthesis completes (expect gaps between chunks:
    \\                                CPU generation runs slower than realtime); invalid with -o -
    \\        --playback <idx>        playback device index (see `devices`; default: system device)
    \\      options:
    \\        --format <fmt>          wav16 | wav24 | wav32 (default: wav16)
    \\        --lang <str>            language label (default 'None')
    \\        --instruct <str>        style instruction (voice-design vocabulary)
    \\        --duration <sec>        output duration; forces single-shot at (int)(sec*25) frames
    \\        --no-denoise            omit the <|denoise|> prefix
    \\        --ref-wav <path>        reference WAV for voice cloning
    \\        --ref-text <path>       transcript FILE (required with --ref-wav / --ref-rvq)
    \\        --ref-rvq <path>        pre-encoded reference codes (replaces --ref-wav)
    \\        --seed <int>            sampling seed (default -1 = nondeterministic random)
    \\        --no-preprocess-prompt  skip ref silence trim + ref-text punctuation
    \\        --chunk-duration <sec>  long-form chunk duration (default 15.0; <= 0 disables)
    \\        --chunk-threshold <sec> chunk above this estimated duration (default 30.0)
    \\        --stream-by-line        flush synthesis at each newline, one WAV header per line (-o '-')
    \\        --no-postproc           skip remove-silence / peak-0.5 / fade+pad (buffered only;
    \\                                the streaming path always post-filters)
    \\      debug:
    \\        --no-fa | --clamp-fp16  accepted for reference CLI compatibility (no-ops on CPU)
    \\        --dump <dir>            dump reference-named tensors (chunk 0 only)
    \\        --llm-test <input.bin>  full LLM forward -> [V][K][S] logits to -o (no codec)
    \\        --maskgit-test          greedy MaskGIT -> raw [K,T] i32 tokens to -o (no codec)
    \\  devices
    \\      list playback devices with indices (for --playback)
    \\  info --model <base.gguf> --codec <tokenizer.gguf>
    \\      print parsed configs of both GGUFs
    \\  codec --model <tokenizer.gguf> -i <input.{wav|rvq}> [-o <out>]
    \\        [--format wav16|wav24|wav32] [--dump-stages <dir>]
    \\      .wav in -> encode to RVQ codes (reference preprocessing: RMS
    \\      auto-gain, silence trim, hop truncation); .rvq in -> decode to a
    \\      WAV. Output lands next to the input (extension swapped) unless
    \\      -o; --dump-stages writes headerless raw f32 parity taps
    \\  compare <a.bin> <b.bin>
    \\      cosine similarity + max abs diff of two dump files
    \\  compare --raw <a.raw> <b.raw>
    \\      same stats over two headerless raw f32 files
    \\
;

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    if (args.len < 2) {
        try stdout.writeAll(usage);
        return;
    }

    const cmd = args[1];
    if (std.mem.eql(u8, cmd, "tts")) {
        try runTts(allocator, init.io, stdout, args[2..]);
    } else if (std.mem.eql(u8, cmd, "devices")) {
        try runDevices(stdout, args[2..]);
    } else if (std.mem.eql(u8, cmd, "info")) {
        try runInfo(allocator, init.io, stdout, args[2..]);
    } else if (std.mem.eql(u8, cmd, "codec")) {
        try runCodec(allocator, init.io, stdout, args[2..]);
    } else if (std.mem.eql(u8, cmd, "compare")) {
        try runCompare(allocator, init.io, stdout, args[2..]);
    } else {
        try stdout.print("unknown command: {s}\n\n", .{cmd});
        try stdout.writeAll(usage);
        return error.UnknownCommand;
    }
}

/// `tts` subcommand — mirrors the reference `omnivoice-tts` CLI surface,
/// including streaming (`-o -`) and `--stream-by-line`; only `--srt`
/// dubbing is out of scope for the port.
fn runTts(
    allocator: std.mem.Allocator,
    io: std.Io,
    stdout: *std.Io.Writer,
    args: []const []const u8,
) !void {
    var model_path: ?[]const u8 = null;
    var codec_path: ?[]const u8 = null;
    var out_path: ?[]const u8 = null;
    var format: wav.Format = .s16;
    var lang: []const u8 = "None";
    var instruct: []const u8 = "";
    var duration_sec: f32 = 0.0;
    var denoise = true;
    var preprocess_prompt = true;
    var postproc_on = true;
    var chunk_duration: f32 = 15.0;
    var chunk_threshold: f32 = 30.0;
    var stream_by_line = false;
    var play_out = false;
    var playback_index: ?usize = null;
    var seed_arg: i64 = -1;
    var ref_wav_path: ?[]const u8 = null;
    var ref_rvq_path: ?[]const u8 = null;
    var ref_text_path: ?[]const u8 = null;
    var dump_dir: ?[]const u8 = null;
    var llm_test_in: ?[]const u8 = null;
    var maskgit_test = false;

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--model")) {
            model_path = try nextArg(args, &i);
        } else if (std.mem.eql(u8, arg, "--codec")) {
            codec_path = try nextArg(args, &i);
        } else if (std.mem.eql(u8, arg, "-o")) {
            out_path = try nextArg(args, &i);
        } else if (std.mem.eql(u8, arg, "--format")) {
            format = wav.parseFormat(try nextArg(args, &i)) orelse {
                try stdout.print("tts: unknown format: {s}\n", .{args[i]});
                return error.UnknownFormat;
            };
        } else if (std.mem.eql(u8, arg, "--lang")) {
            lang = try nextArg(args, &i);
        } else if (std.mem.eql(u8, arg, "--instruct")) {
            instruct = try nextArg(args, &i);
        } else if (std.mem.eql(u8, arg, "--duration")) {
            duration_sec = try std.fmt.parseFloat(f32, try nextArg(args, &i));
        } else if (std.mem.eql(u8, arg, "--no-denoise")) {
            denoise = false;
        } else if (std.mem.eql(u8, arg, "--no-preprocess-prompt")) {
            preprocess_prompt = false;
        } else if (std.mem.eql(u8, arg, "--no-postproc")) {
            postproc_on = false;
        } else if (std.mem.eql(u8, arg, "--chunk-duration")) {
            chunk_duration = try std.fmt.parseFloat(f32, try nextArg(args, &i));
        } else if (std.mem.eql(u8, arg, "--chunk-threshold")) {
            chunk_threshold = try std.fmt.parseFloat(f32, try nextArg(args, &i));
        } else if (std.mem.eql(u8, arg, "--stream-by-line")) {
            stream_by_line = true;
        } else if (std.mem.eql(u8, arg, "--play")) {
            play_out = true;
        } else if (std.mem.eql(u8, arg, "--playback")) {
            playback_index = try std.fmt.parseInt(usize, try nextArg(args, &i), 10);
        } else if (std.mem.eql(u8, arg, "--ref-wav")) {
            ref_wav_path = try nextArg(args, &i);
        } else if (std.mem.eql(u8, arg, "--ref-rvq")) {
            ref_rvq_path = try nextArg(args, &i);
        } else if (std.mem.eql(u8, arg, "--ref-text")) {
            ref_text_path = try nextArg(args, &i);
        } else if (std.mem.eql(u8, arg, "--seed")) {
            seed_arg = try std.fmt.parseInt(i64, try nextArg(args, &i), 10);
        } else if (std.mem.eql(u8, arg, "--dump")) {
            dump_dir = try nextArg(args, &i);
        } else if (std.mem.eql(u8, arg, "--llm-test")) {
            llm_test_in = try nextArg(args, &i);
        } else if (std.mem.eql(u8, arg, "--maskgit-test")) {
            maskgit_test = true;
        } else if (std.mem.eql(u8, arg, "--no-fa") or std.mem.eql(u8, arg, "--clamp-fp16")) {
            // Reference debug flags: flash attention never activates on CPU
            // and the FP16 clamp only matters on FP16 GPU paths. Accepted so
            // reference harness command lines run verbatim.
        } else {
            try stdout.print("tts: unknown arg: {s}\n\n", .{arg});
            try stdout.writeAll(usage);
            return error.UnknownArg;
        }
    }

    // Mode resolution mirrors the reference: the debug modes are mutually
    // exclusive with each other; everything else is full TTS synthesis.
    if (llm_test_in != null and maskgit_test) {
        try stdout.writeAll("tts: --llm-test and --maskgit-test are mutually exclusive\n");
        return error.InvalidArgs;
    }
    const tts_mode = llm_test_in == null and !maskgit_test;

    const model = model_path orelse {
        try stdout.writeAll("tts: --model is required\n");
        return error.InvalidArgs;
    };
    if (playback_index != null and !play_out) {
        try stdout.writeAll("tts: --playback requires --play\n");
        return error.InvalidArgs;
    }
    if (play_out and !tts_mode) {
        try stdout.writeAll("tts: --play is only supported in synthesis mode\n");
        return error.InvalidArgs;
    }
    // Synthesis needs at least one output: a WAV path and/or the speakers.
    // The debug modes always need -o (they write token/logit files).
    if (out_path == null and !play_out) {
        try stdout.writeAll(if (tts_mode)
            "tts: need -o <path> and/or --play\n"
        else
            "tts: -o <path> is required\n");
        return error.InvalidArgs;
    }
    // Streaming detection mirrors the reference: -o '-' writes a wide RIFF
    // header to stdout up front and pipes encoded samples as the synthesis
    // emits them (synthesis mode only; the debug modes treat '-' as a file
    // name, exactly like the reference's fopen("-")).
    const stream_to_stdout = tts_mode and out_path != null and std.mem.eql(u8, out_path.?, "-");
    if (play_out and stream_to_stdout) {
        try stdout.writeAll("tts: --play cannot be combined with -o - (stdout is the WAV stream)\n");
        return error.InvalidArgs;
    }
    if (tts_mode and codec_path == null) {
        try stdout.writeAll("tts: synthesis requires --codec\n");
        return error.InvalidArgs;
    }
    if (ref_wav_path != null and ref_rvq_path != null) {
        try stdout.writeAll("tts: --ref-wav and --ref-rvq are mutually exclusive\n");
        return error.InvalidArgs;
    }
    if ((ref_wav_path != null or ref_rvq_path != null) and ref_text_path == null) {
        try stdout.writeAll("tts: --ref-wav / --ref-rvq requires --ref-text <path>\n");
        return error.InvalidArgs;
    }
    if ((ref_wav_path != null or ref_rvq_path != null) and !tts_mode) {
        try stdout.writeAll("tts: --ref-wav / --ref-rvq is only supported in synthesis mode\n");
        return error.InvalidArgs;
    }

    // Progress/liveness signals go to stderr in EVERY mode (the reference
    // logs to stderr unconditionally, and in streaming mode stdout is the
    // WAV pipe); stdout keeps only results. Streaming writer: stderr must
    // append like std.debug.print does — a positional writer overwrites the
    // debug/log interleave when stderr is redirected to a regular file.
    var log_buffer: [4096]u8 = undefined;
    var stderr_writer = std.Io.File.stderr().writerStreaming(io, &log_buffer);
    const log: *std.Io.Writer = &stderr_writer.interface;

    // MaskGIT per-step progress printer: \r-updating single line on a TTY,
    // plain lines every 8 steps otherwise. Debug/parity modes stay silent.
    const stderr_tty = std.Io.File.stderr().isTty(io) catch false;
    var progress_printer = ProgressPrinter{ .io = io, .tty = stderr_tty };
    const progress: ?mg_decode.Progress = if (tts_mode)
        .{ .ctx = &progress_printer, .func = ProgressPrinter.onStep }
    else
        null;

    // Seed resolution: -1 draws a fresh 32-bit value from the OS entropy
    // source (reference: std::random_device) — such runs are
    // NONDETERMINISTIC; any other value is used verbatim for reproducible
    // runs.
    const seed: u64 = if (seed_arg < 0) blk: {
        var seed_bytes: [4]u8 = undefined;
        io.random(&seed_bytes);
        break :blk std.mem.readInt(u32, &seed_bytes, .little);
    } else @intCast(seed_arg);
    try log.print("[CLI] Seed: {d}{s}\n", .{ seed, if (seed_arg < 0) " (random)" else "" });
    try log.flush();

    if (dump_dir) |dir| {
        std.Io.Dir.cwd().createDirPath(io, dir) catch |err| switch (err) {
            error.PathAlreadyExists => {},
            else => return err,
        };
    }

    // The LM loads in every mode.
    const lm_load0 = nowNs(io);
    var lm_file = try fucina.gguf.File.loadMmap(allocator, io, model);
    defer lm_file.deinit();
    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();
    var lm_model = try lm.loadModel(&ctx, &lm_file);
    defer lm_model.deinit();
    try logLoad(log, "base LM", &lm_file, nowNs(io) - lm_load0);

    if (llm_test_in) |in_path| {
        return runLlmTest(allocator, io, stdout, &ctx, &lm_model, in_path, out_path.?);
    }

    var tok = try llm.tokenizer.Tokenizer.initFromGguf(allocator, &lm_file, .{});
    defer tok.deinit();

    if (maskgit_test) {
        const text = try readStdinText(allocator, io);
        defer allocator.free(text);
        // Greedy decoder: both temperatures forced to zero for bytewise
        // reproducibility; the resolved seed is wired in for consistency
        // but has no effect (greedy performs no Philox draws).
        const normalized = try voicedesign.normalize(allocator, instruct, voicedesign.hasCjk(text));
        const resolved: []u8 = switch (normalized) {
            .ok => |s| s,
            .invalid => |msg| {
                defer allocator.free(msg);
                try stdout.print("[TTS] {s}\n", .{msg});
                return error.InvalidInstruct;
            },
        };
        defer allocator.free(resolved);

        // 25 fps hardcoded (24000 / 960): the codec is not loaded here.
        const target_tokens: usize = if (duration_sec > 0.0) blk: {
            const t: i32 = @intFromFloat(duration_sec * 25.0);
            break :blk @intCast(@max(t, 1));
        } else @intCast(duration.estimateTokens(text, "", 0));

        const mg_cfg = maskgit.Config{ .class_temperature = 0.0, .position_temperature = 0.0, .seed = seed };
        var ctr_lo: u32 = 0;
        const tokens = try pipeline.generateTokens(
            allocator,
            io,
            &ctx,
            &lm_model,
            &tok,
            text,
            lang,
            resolved,
            target_tokens,
            denoise,
            mg_cfg,
            "",
            null,
            0,
            dump_dir,
            &ctr_lo,
            progress,
        );
        defer allocator.free(tokens);

        try writeI32File(io, out_path.?, tokens);
        try stdout.print("[OmniVoice-TTS] MaskGIT test: wrote {s} (K={d} T={d} i32)\n", .{
            out_path.?, lm_model.config.num_audio_codebook, target_tokens,
        });
        return;
    }

    // Full synthesis: codec (+ encoder only when a raw reference needs
    // encoding), optional reference triple, one buffered synthesize.
    const codec_load0 = nowNs(io);
    var codec_file = try fucina.gguf.File.loadMmap(allocator, io, codec_path.?);
    defer codec_file.deinit();
    var cdc = try codec.Codec.load(&ctx, &codec_file);
    defer cdc.deinit();
    var enc: ?codec.Encoder = null;
    defer if (enc) |*e| e.deinit();
    if (ref_wav_path != null) enc = try codec.Encoder.load(&ctx, &codec_file);
    try logLoad(log, "codec", &codec_file, nowNs(io) - codec_load0);

    var ref_text: []u8 = try allocator.alloc(u8, 0);
    defer allocator.free(ref_text);
    if (ref_text_path) |path| {
        allocator.free(ref_text);
        ref_text = try readTextFileTrimmed(allocator, io, path);
    }

    var ref_audio: ?[]f32 = null;
    defer if (ref_audio) |a| allocator.free(a);
    if (ref_wav_path) |path| {
        try log.print("[CLI] Reference WAV: {s}\n", .{path});
        try log.flush();
        ref_audio = try wav.readMono(io, allocator, path, @intCast(cdc.config.sample_rate));
    }

    var ref_tokens: ?[]i32 = null;
    defer if (ref_tokens) |t| allocator.free(t);
    var ref_len: usize = 0;
    if (ref_rvq_path) |path| {
        const codes = try rvq_file.readFile(allocator, io, path, codec.n_codebooks);
        ref_tokens = codes;
        ref_len = codes.len / codec.n_codebooks;
        try log.print("[CLI] Reference RVQ: {s}, K={d} T={d}\n", .{ path, codec.n_codebooks, ref_len });
        try log.flush();
    }

    // --duration → frame-count override; forces the single-shot path.
    var t_override: i32 = 0;
    if (duration_sec > 0.0) {
        const frame_rate = @as(f32, @floatFromInt(cdc.config.sample_rate)) / @as(f32, @floatFromInt(cdc.config.hop_length));
        t_override = @max(@as(i32, @intFromFloat(duration_sec * frame_rate)), 1);
    }

    const tts = pipeline.Tts{
        .allocator = allocator,
        .io = io,
        .ctx = &ctx,
        .model = &lm_model,
        .tok = &tok,
        .cdc = &cdc,
        .enc = if (enc) |*e| e else null,
    };

    // Streaming stdin -> streaming stdout: bytes arrive as the upstream
    // produces them; the incremental text chunker drives one full
    // synthesize per ready chunk, with the audio piped through the wav
    // stream sink.
    if (stream_to_stdout) {
        return runTtsStream(allocator, io, stdout, log, &tts, .{
            .lang = lang,
            .instruct = instruct,
            .t_override = t_override,
            .chunk_duration = chunk_duration,
            .chunk_threshold = chunk_threshold,
            .denoise = denoise,
            .preprocess_prompt = preprocess_prompt,
            .seed = seed,
            .ref_audio = ref_audio,
            .ref_tokens = ref_tokens,
            .ref_len = ref_len,
            .ref_text = ref_text,
            .dump_dir = dump_dir,
            .format = format,
            .stream_by_line = stream_by_line,
            .progress = progress,
        });
    }

    // Buffered / speaker path: read full stdin, one synthesize.
    const text = try readStdinText(allocator, io);
    defer allocator.free(text);

    const params = pipeline.Params{
        .text = text,
        .lang = lang,
        .instruct = instruct,
        .t_override = t_override,
        .chunk_duration_sec = chunk_duration,
        .chunk_threshold_sec = chunk_threshold,
        .denoise = denoise,
        .preprocess_prompt = preprocess_prompt,
        .postproc = postproc_on,
        .mg = .{ .seed = seed },
        .ref_audio_24k = ref_audio,
        .ref_audio_tokens = ref_tokens,
        .ref_len = ref_len,
        .ref_text = ref_text,
        .dump_dir = dump_dir,
        .progress = progress,
    };

    // --play without -o: stream chunk-by-chunk into the playback ring so
    // audio starts at the first chunk (long-form chunks arrive as they are
    // synthesized; the device plays silence between them — CPU generation
    // runs slower than realtime). With -o the buffered synthesize below
    // keeps the WAV byte-identical and plays the final waveform gaplessly.
    if (play_out and out_path == null) {
        var player = try play.Player.init(allocator, io, .{
            .device_index = playback_index,
            .sample_rate = @intCast(cdc.config.sample_rate),
        });
        defer player.deinit();
        try log.print("[Play] playback device open ({d} Hz stream, device native {d} Hz)\n", .{
            cdc.config.sample_rate, player.deviceSampleRate(),
        });
        try log.flush();
        try pipeline.synthesizeStream(&tts, params, .{ .ctx = player, .func = playSinkEmit });
        const stats = try player.drainAndStop();
        try printPlayStats(stdout, stats, @intCast(cdc.config.sample_rate));
        return;
    }

    const audio = try pipeline.synthesize(&tts, params);
    defer allocator.free(audio);

    if (out_path) |out| {
        try wav.writeMono(io, allocator, out, audio, @intCast(cdc.config.sample_rate), format);
        try stdout.print("[OmniVoice-TTS] TTS: wrote {s} ({d} samples @ {d} Hz, {d:.2} s)\n", .{
            out,
            audio.len,
            cdc.config.sample_rate,
            @as(f64, @floatFromInt(audio.len)) / @as(f64, @floatFromInt(cdc.config.sample_rate)),
        });
        try stdout.flush();
    }

    if (play_out) {
        var player = try play.Player.init(allocator, io, .{
            .device_index = playback_index,
            .sample_rate = @intCast(cdc.config.sample_rate),
        });
        defer player.deinit();
        try player.pushSamples(audio);
        const stats = try player.drainAndStop();
        try printPlayStats(stdout, stats, @intCast(cdc.config.sample_rate));
    }
}

/// `pipeline.synthesizeStream` sink that queues each post-processed chunk
/// for speaker playback (blocking while the ring is full).
fn playSinkEmit(ctx: *anyopaque, samples: []const f32) anyerror!void {
    const player: *play.Player = @ptrCast(@alignCast(ctx));
    try player.pushSamples(samples);
}

fn printPlayStats(stdout: *std.Io.Writer, stats: play.Stats, sample_rate: u32) !void {
    try stdout.print("[OmniVoice-TTS] played {d} frames ({d:.2} s @ {d} Hz, {d} gap(s), {d:.2} s inserted silence)\n", .{
        stats.frames_played,
        @as(f64, @floatFromInt(stats.frames_played)) / @as(f64, @floatFromInt(sample_rate)),
        sample_rate,
        stats.underrun_episodes,
        @as(f64, @floatFromInt(stats.silence_frames)) / @as(f64, @floatFromInt(sample_rate)),
    });
    try stdout.flush();
}

/// `devices` subcommand: playback device enumeration (indices feed
/// `tts --play --playback <idx>`), mirroring the NAM example's format.
fn runDevices(stdout: *std.Io.Writer, args: []const []const u8) !void {
    _ = args;
    var storage: [play.max_devices]play.DeviceInfo = undefined;
    try stdout.writeAll("playback devices:\n");
    for (try play.listPlaybackDevices(&storage), 0..) |*info, i| {
        try stdout.print("  [{d}] {s}{s}\n", .{ i, info.nameSlice(), if (info.is_default) " (default)" else "" });
    }
}

/// `[Load] base LM: 626 MB (Q8_0) in 1.2s`: whole-GGUF size plus the
/// byte-dominant tensor dtype (the file's effective quantization).
fn logLoad(log: *std.Io.Writer, label: []const u8, file: *const fucina.gguf.File, elapsed_ns: i96) !void {
    var by_type = [_]u64{0} ** 64;
    var best: fucina.gguf.GgmlType = .f32;
    var best_bytes: u64 = 0;
    for (file.tensors) |t| {
        const idx: usize = @intFromEnum(t.ggml_type);
        by_type[idx] += t.data.len;
        if (by_type[idx] > best_bytes) {
            best_bytes = by_type[idx];
            best = t.ggml_type;
        }
    }
    var name_buf: [16]u8 = undefined;
    const type_name = std.ascii.upperString(name_buf[0..@tagName(best).len], @tagName(best));
    try log.print("[Load] {s}: {d} MB ({s}) in {d:.1}s\n", .{
        label,
        file.bytes.len / (1024 * 1024),
        type_name,
        @as(f64, @floatFromInt(elapsed_ns)) / 1e9,
    });
    try log.flush();
}

/// Stderr progress printer for `mg_decode.Progress`. TTY: one \r-updating
/// line per generation, finalized with a newline at completion. Non-TTY:
/// a plain line every 8 executed steps plus the final one. Writes via
/// `std.debug.print` (locked stderr, like the pipeline's phase logs);
/// stdout is never touched — it may be the WAV stream.
const ProgressPrinter = struct {
    io: std.Io,
    tty: bool,
    start_ns: i96 = 0,
    last_step: usize = 0,

    fn onStep(ctx: ?*anyopaque, step: usize, num_steps: usize, demasked: usize, total: usize) void {
        const self: *ProgressPrinter = @ptrCast(@alignCast(ctx.?));
        // A new generation begins whenever the step counter restarts.
        if (self.last_step == 0 or step <= self.last_step) self.start_ns = nowNs(self.io);
        self.last_step = step;
        const done = demasked >= total;
        const elapsed = @as(f64, @floatFromInt(nowNs(self.io) - self.start_ns)) / 1e9;
        if (self.tty) {
            std.debug.print("\r[MaskGIT] step {d}/{d} · demasked {d}/{d} · {d:.1}s{s}", .{
                step, num_steps, demasked, total, elapsed, if (done) "\n" else @as([]const u8, ""),
            });
        } else if (done or step % 8 == 0) {
            std.debug.print("[MaskGIT] step {d}/{d} · demasked {d}/{d} · {d:.1}s\n", .{
                step, num_steps, demasked, total, elapsed,
            });
        }
        if (done) self.last_step = 0;
    }
};

/// Options for the streaming stdin -> stdout loop. Every ready text chunk
/// gets one full `pipeline.synthesizeStream` call with these synthesize
/// params (fresh Philox ctr_lo per call). NOTE the reference-mirrored voice
/// behaviour: the CLI-level reference (if any) rides on EVERY call; in
/// auto-voice mode each call is independent — chunk-0 promotion happens only
/// WITHIN a synthesize (its own long-form chunks), never across the outer
/// text chunks. `t_override` (--duration) and `dump_dir` (--dump) also pass
/// through to every call verbatim, exactly like the reference.
const StreamOpts = struct {
    lang: []const u8,
    instruct: []const u8,
    t_override: i32,
    chunk_duration: f32,
    chunk_threshold: f32,
    denoise: bool,
    preprocess_prompt: bool,
    seed: u64,
    ref_audio: ?[]const f32,
    ref_tokens: ?[]const i32,
    ref_len: usize,
    ref_text: []const u8,
    dump_dir: ?[]const u8,
    format: wav.Format,
    stream_by_line: bool,
    progress: ?mg_decode.Progress,
};

fn wavSinkEmit(ctx: *anyopaque, samples: []const f32) anyerror!void {
    const sink: *wav.StreamSink = @ptrCast(@alignCast(ctx));
    try sink.writeSamples(samples);
}

/// One streamed utterance: lazily re-arms the RIFF header (line-oriented
/// mode), then a full synthesize with the audio forwarded to the wav sink.
const StreamSynth = struct {
    tts: *const pipeline.Tts,
    opts: *const StreamOpts,
    sink: *wav.StreamSink,
    need_header: *bool,
    n_emitted: *usize,

    fn one(self: *const StreamSynth, chunk_text: []const u8) !void {
        if (self.need_header.*) {
            try self.sink.writeHeader();
            self.need_header.* = false;
        }
        try pipeline.synthesizeStream(self.tts, .{
            .text = chunk_text,
            .lang = self.opts.lang,
            .instruct = self.opts.instruct,
            .t_override = self.opts.t_override,
            .chunk_duration_sec = self.opts.chunk_duration,
            .chunk_threshold_sec = self.opts.chunk_threshold,
            .denoise = self.opts.denoise,
            .preprocess_prompt = self.opts.preprocess_prompt,
            .mg = .{ .seed = self.opts.seed },
            .ref_audio_24k = self.opts.ref_audio,
            .ref_audio_tokens = self.opts.ref_tokens,
            .ref_len = self.opts.ref_len,
            .ref_text = self.opts.ref_text,
            .dump_dir = self.opts.dump_dir,
            .progress = self.opts.progress,
        }, .{ .ctx = self.sink, .func = wavSinkEmit });
        self.n_emitted.* += 1;
    }
};

/// Streaming stdin -> streaming stdout (reference omnivoice-tts.cpp
/// stream_to_stdout loop). Bytes arrive as the upstream produces them; the
/// incremental text chunker fires as soon as a chunk of text is ready.
///
/// chunk_len is computed from chunk_duration assuming a typical 1 frame per
/// codepoint ratio: `(int)((24000/480) * chunk_duration)` codepoints — the
/// 480 is the reference's deliberate text-budget heuristic (50 cps/s),
/// DISTINCT from the codec hop (960); kept verbatim. CJK produces shorter
/// audio per chunk, which is the safe direction for prosody.
///
/// With --stream-by-line the read is line oriented: every newline drains
/// the chunker so the line synthesises now, and the next line opens with a
/// fresh RIFF header (armed at line end, consumed lazily at the next audio,
/// so a trailing or empty line never emits an orphan header).
fn runTtsStream(
    allocator: std.mem.Allocator,
    io: std.Io,
    wav_out: *std.Io.Writer,
    log: *std.Io.Writer,
    tts: *const pipeline.Tts,
    opts: StreamOpts,
) !void {
    var sink = wav.StreamSink{
        .writer = wav_out,
        .sample_rate = @intCast(tts.cdc.config.sample_rate),
        .format = opts.format,
    };
    try sink.writeHeader();
    try log.print("[WAV-Stream] stdout: {d} Hz, mono {s}\n", .{
        sink.sample_rate,
        switch (opts.format) {
            .s16 => @as([]const u8, "S16"),
            .s24 => "S24",
            .f32 => "F32",
        },
    });
    try log.flush();

    const frame_rate_text: i32 = 24000 / 480; // deliberate text-budget heuristic, reference-verbatim
    const chunk_len_text: i32 = @intFromFloat(@as(f32, @floatFromInt(frame_rate_text)) * opts.chunk_duration);

    var chk = chunker_stream.Stream.init(allocator, chunk_len_text, chunker.min_chunk_len_default);
    defer chk.deinit();

    var n_emitted: usize = 0;
    var bytes_in: usize = 0;
    var need_header = false;

    const synth = StreamSynth{
        .tts = tts,
        .opts = &opts,
        .sink = &sink,
        .need_header = &need_header,
        .n_emitted = &n_emitted,
    };

    var stdin_buf: [4096]u8 = undefined;
    var stdin_reader = std.Io.File.stdin().reader(io, &stdin_buf);
    const in = &stdin_reader.interface;

    if (opts.stream_by_line) {
        // Line-oriented: every complete line drains the chunker and arms a
        // fresh header for the next one. Lines are text: an embedded NUL
        // truncates the read (the reference's fgets/strlen quirk).
        var line: std.ArrayList(u8) = .empty;
        defer line.deinit(allocator);
        var eof = false;
        while (!eof) {
            line.clearRetainingCapacity();
            while (true) {
                const b = in.takeByte() catch |err| switch (err) {
                    error.EndOfStream => {
                        eof = true;
                        break;
                    },
                    else => |e| return e,
                };
                try line.append(allocator, b);
                if (b == '\n') break;
            }

            var data: []const u8 = line.items;
            if (std.mem.indexOfScalar(u8, data, 0)) |z| data = data[0..z];
            if (data.len == 0) continue;
            const flush_line = data[data.len - 1] == '\n';

            bytes_in += data.len;
            const ready = try chk.pushBytes(data);
            defer chunker.freeChunks(allocator, ready);
            for (ready) |ct| try synth.one(ct);

            if (flush_line) {
                const line_tail = try chk.flushEof();
                defer chunker.freeChunks(allocator, line_tail);
                for (line_tail) |ct| try synth.one(ct);
                need_header = true;
            }
        }
    } else {
        // 4 KiB reads, blocking between reads like the reference's fread:
        // suitable for piped LLM output producing bytes at its own pace.
        var buf: [4096]u8 = undefined;
        while (true) {
            const n = try in.readSliceShort(&buf);
            if (n == 0) break;
            bytes_in += n;
            const ready = try chk.pushBytes(buf[0..n]);
            defer chunker.freeChunks(allocator, ready);
            for (ready) |ct| try synth.one(ct);
        }
    }

    const tail = try chk.flushEof();
    defer chunker.freeChunks(allocator, tail);
    for (tail) |ct| try synth.one(ct);

    try wav_out.flush();
    try log.print("[OmniVoice-TTS] streamed {d} chunks ({d} bytes input) to stdout\n", .{ n_emitted, bytes_in });
    try log.flush();
}

/// `--llm-test`: input `[i32 K][i32 S][K*S i32 ids (k slow)][S i32 mask]`,
/// full-S bidirectional forward (NULL attention mask), output
/// `[i32 V][i32 K][i32 S][V*K*S f32]` (v fast / k mid / s slow — identical
/// flat order to the forward's `[S, K*V]` rows).
fn runLlmTest(
    allocator: std.mem.Allocator,
    io: std.Io,
    stdout: *std.Io.Writer,
    ctx: *fucina.ExecContext,
    lm_model: *const lm.Model,
    in_path: []const u8,
    out_path: []const u8,
) !void {
    const bytes = try wav.readFileBytes(io, allocator, in_path);
    defer allocator.free(bytes);
    if (bytes.len < 8) return error.CorruptInput;
    const k_raw = std.mem.readInt(i32, bytes[0..4], .little);
    const s_raw = std.mem.readInt(i32, bytes[4..8], .little);
    if (k_raw <= 0 or s_raw <= 0) return error.CorruptInput;
    const num_k: usize = @intCast(k_raw);
    const seq_len: usize = @intCast(s_raw);
    if (num_k != lm_model.config.num_audio_codebook) return error.CorruptInput;
    if (bytes.len != 8 + (num_k * seq_len + seq_len) * 4) return error.CorruptInput;

    const ids = try allocator.alloc(i32, num_k * seq_len);
    defer allocator.free(ids);
    for (ids, 0..) |*dst, j| {
        dst.* = std.mem.readInt(i32, bytes[8 + j * 4 ..][0..4], .little);
    }
    const audio_mask = try allocator.alloc(i32, seq_len);
    defer allocator.free(audio_mask);
    const mask_base = 8 + ids.len * 4;
    for (audio_mask, 0..) |*dst, j| {
        dst.* = std.mem.readInt(i32, bytes[mask_base + j * 4 ..][0..4], .little);
    }

    try stdout.print("[OmniVoice-TTS] LM forward: K={d} S={d}\n", .{ num_k, seq_len });
    try stdout.flush();

    var logits = try lm_model.forward(ctx, ids, audio_mask, 0, seq_len, null);
    defer logits.deinit();
    const data = try logits.dataConst();
    const vocab = lm_model.config.audio_vocab_size;

    var file = try std.Io.Dir.cwd().createFile(io, out_path, .{});
    defer file.close(io);
    var write_buffer: [64 * 1024]u8 = undefined;
    var writer = file.writer(io, &write_buffer);
    const w = &writer.interface;
    try w.writeInt(i32, @intCast(vocab), .little);
    try w.writeInt(i32, @intCast(num_k), .little);
    try w.writeInt(i32, @intCast(seq_len), .little);
    for (data) |v| {
        try w.writeInt(u32, @bitCast(v), .little);
    }
    try w.flush();

    try stdout.print("[OmniVoice-TTS] LM forward: wrote {s} (V={d} K={d} S={d} f32)\n", .{
        out_path, vocab, num_k, seq_len,
    });
}

fn nextArg(args: []const []const u8, i: *usize) ![]const u8 {
    i.* += 1;
    if (i.* >= args.len) return error.MissingArgValue;
    return args[i.*];
}

/// Reads all of stdin; trims ALL trailing '\n'/'\r' (reference
/// `read_stdin_text`). Caller frees.
fn readStdinText(allocator: std.mem.Allocator, io: std.Io) ![]u8 {
    var stdin = std.Io.File.stdin();
    var list: std.ArrayList(u8) = .empty;
    errdefer list.deinit(allocator);
    var tmp: [4096]u8 = undefined;
    while (true) {
        var bufs = [_][]u8{tmp[0..]};
        const n = stdin.readStreaming(io, &bufs) catch |err| switch (err) {
            error.EndOfStream => break,
            else => |e| return e,
        };
        try list.appendSlice(allocator, tmp[0..n]);
    }
    trimTrailingNewlines(&list);
    return list.toOwnedSlice(allocator);
}

/// Reads a small text file (transcript); trims ALL trailing '\n'/'\r'
/// (reference `read_text_file`). Caller frees.
fn readTextFileTrimmed(allocator: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    const bytes = try wav.readFileBytes(io, allocator, path);
    errdefer allocator.free(bytes);
    var len = bytes.len;
    while (len > 0 and (bytes[len - 1] == '\n' or bytes[len - 1] == '\r')) len -= 1;
    if (len == bytes.len) return bytes;
    return allocator.realloc(bytes, len);
}

fn trimTrailingNewlines(list: *std.ArrayList(u8)) void {
    while (list.items.len > 0) {
        const c = list.items[list.items.len - 1];
        if (c != '\n' and c != '\r') break;
        list.items.len -= 1;
    }
}

/// Writes raw little-endian i32 values, no header (`--maskgit-test` output).
fn writeI32File(io: std.Io, path: []const u8, values: []const i32) !void {
    var file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var write_buffer: [64 * 1024]u8 = undefined;
    var writer = file.writer(io, &write_buffer);
    const w = &writer.interface;
    for (values) |v| {
        try w.writeInt(i32, v, .little);
    }
    try w.flush();
}

fn runInfo(
    allocator: std.mem.Allocator,
    io: std.Io,
    stdout: *std.Io.Writer,
    args: []const []const u8,
) !void {
    var model_path: ?[]const u8 = null;
    var codec_path: ?[]const u8 = null;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--model")) {
            i += 1;
            if (i >= args.len) return error.MissingArgValue;
            model_path = args[i];
        } else if (std.mem.eql(u8, args[i], "--codec")) {
            i += 1;
            if (i >= args.len) return error.MissingArgValue;
            codec_path = args[i];
        } else {
            return error.UnknownArg;
        }
    }

    if (model_path) |path| {
        var file = try fucina.gguf.File.loadMmap(allocator, io, path);
        defer file.deinit();
        try stdout.print("base LM: {s}\n", .{path});
        try printKv(&file, stdout);
    }
    if (codec_path) |path| {
        var file = try fucina.gguf.File.loadMmap(allocator, io, path);
        defer file.deinit();
        try stdout.print("codec: {s}\n", .{path});
        try printKv(&file, stdout);
    }
}

fn printKv(file: *fucina.gguf.File, stdout: *std.Io.Writer) !void {
    const arch = file.getString("general.architecture") orelse "(missing)";
    try stdout.print("  architecture: {s}\n  tensors: {d}\n", .{ arch, file.tensors.len });
}

fn runCompare(
    allocator: std.mem.Allocator,
    io: std.Io,
    stdout: *std.Io.Writer,
    args: []const []const u8,
) !void {
    if (args.len == 3 and std.mem.eql(u8, args[0], "--raw")) {
        const a = try readRawF32(allocator, io, args[1]);
        defer allocator.free(a);
        const b = try readRawF32(allocator, io, args[2]);
        defer allocator.free(b);
        const stats = dump.compare(a, b);
        try stdout.print(
            "n a={d} b={d} cos={d:.9} max_abs={e:.3} mean_abs={e:.3}\n",
            .{ a.len, b.len, stats.cosine, stats.max_abs_diff, stats.mean_abs_diff },
        );
        return;
    }
    if (args.len != 2) return error.UnknownArg;
    const a = try dump.readFile(allocator, io, args[0]);
    defer allocator.free(a.shape);
    defer allocator.free(a.data);
    const b = try dump.readFile(allocator, io, args[1]);
    defer allocator.free(b.shape);
    defer allocator.free(b.data);
    const stats = dump.compare(a.data, b.data);
    try stdout.print(
        "n a={d} b={d} cos={d:.9} max_abs={e:.3} mean_abs={e:.3}\n",
        .{ a.data.len, b.data.len, stats.cosine, stats.max_abs_diff, stats.mean_abs_diff },
    );
}

fn runCodec(
    allocator: std.mem.Allocator,
    io: std.Io,
    stdout: *std.Io.Writer,
    args: []const []const u8,
) !void {
    var model_path: ?[]const u8 = null;
    var input_path: ?[]const u8 = null;
    var out_path: ?[]const u8 = null;
    var format: wav.Format = .s16;
    var dump_dir: ?[]const u8 = null;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--model")) {
            i += 1;
            if (i >= args.len) return error.MissingArgValue;
            model_path = args[i];
        } else if (std.mem.eql(u8, args[i], "-i")) {
            i += 1;
            if (i >= args.len) return error.MissingArgValue;
            input_path = args[i];
        } else if (std.mem.eql(u8, args[i], "-o")) {
            i += 1;
            if (i >= args.len) return error.MissingArgValue;
            out_path = args[i];
        } else if (std.mem.eql(u8, args[i], "--format")) {
            i += 1;
            if (i >= args.len) return error.MissingArgValue;
            format = wav.parseFormat(args[i]) orelse return error.UnknownFormat;
        } else if (std.mem.eql(u8, args[i], "--dump-stages")) {
            i += 1;
            if (i >= args.len) return error.MissingArgValue;
            dump_dir = args[i];
        } else {
            return error.UnknownArg;
        }
    }
    const model = model_path orelse return error.MissingArgValue;
    const input = input_path orelse return error.MissingArgValue;

    const is_decode = std.mem.endsWith(u8, input, ".rvq");
    if (!is_decode and !std.mem.endsWith(u8, input, ".wav")) {
        try stdout.print("codec: unsupported input extension (expect .wav or .rvq)\n", .{});
        return error.UnsupportedInput;
    }

    var file = try fucina.gguf.File.loadMmap(allocator, io, model);
    defer file.deinit();

    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    if (!is_decode) {
        return runCodecEncode(allocator, io, stdout, &ctx, &file, input, out_path, dump_dir);
    }

    const load_start = nowNs(io);
    var cdc = try codec.Codec.load(&ctx, &file);
    defer cdc.deinit();
    const load_ns: u64 = @intCast(nowNs(io) - load_start);

    const codes = try rvq_file.readFile(allocator, io, input, codec.n_codebooks);
    defer allocator.free(codes);
    const t = codes.len / codec.n_codebooks;

    const decode_start = nowNs(io);
    var decoded = try rvq.decode(&ctx, &cdc.rvq, codes, t);
    defer decoded.deinit();

    var dac_in = try decoded.fc2_out.withTags(&ctx, .{ .seq, .in });
    defer dac_in.deinit();

    var taps = dac.Taps.init(allocator);
    defer taps.deinit();
    const taps_ptr: ?*dac.Taps = if (dump_dir != null) &taps else null;

    const audio = try dac.decodeForward(&ctx, allocator, &cdc.dac, &dac_in, taps_ptr);
    defer allocator.free(audio);
    const decode_ns: u64 = @intCast(nowNs(io) - decode_start);

    const wav_path = out_path orelse try swapExtension(allocator, input, ".wav");
    defer if (out_path == null) allocator.free(wav_path);
    try wav.writeMono(io, allocator, wav_path, audio, @intCast(cdc.config.sample_rate), format);

    try stdout.print(
        "codec decode: T={d} frames -> {d} samples ({d:.2} s), load {d:.1} ms, decode {d:.1} ms -> {s}\n",
        .{
            t,
            audio.len,
            @as(f64, @floatFromInt(audio.len)) / @as(f64, @floatFromInt(cdc.config.sample_rate)),
            @as(f64, @floatFromInt(load_ns)) / 1e6,
            @as(f64, @floatFromInt(decode_ns)) / 1e6,
            wav_path,
        },
    );

    if (dump_dir) |dir| {
        try writeRawF32(io, allocator, dir, "fuc_rvq_out.raw", try decoded.latent.dataConst());
        try writeRawF32(io, allocator, dir, "fuc_fc2_out.raw", try decoded.fc2_out.dataConst());
        try writePlanarTap(io, allocator, dir, "fuc_dac_after_conv1.raw", taps.after_conv1.?);
        var name_buf: [64]u8 = undefined;
        for (taps.after_blk, 0..) |maybe_tap, blk_i| {
            const name = try std.fmt.bufPrint(&name_buf, "fuc_dac_after_blk{d}.raw", .{blk_i});
            try writePlanarTap(io, allocator, dir, name, maybe_tap.?);
        }
        try stdout.print("stage dumps written to {s}\n", .{dir});
    }
}

/// Codec encode direction (reference omnivoice-codec encode mode): WAV →
/// 24 kHz mono read → reference preprocessing (RMS auto-gain + silence trim)
/// → hop truncation → encode → .rvq next to the input (or -o).
fn runCodecEncode(
    allocator: std.mem.Allocator,
    io: std.Io,
    stdout: *std.Io.Writer,
    ctx: *fucina.ExecContext,
    file: *const fucina.gguf.File,
    input: []const u8,
    out_path: ?[]const u8,
    dump_dir: ?[]const u8,
) !void {
    // Encode never touches the DAC decoder: load only the RVQ decode side
    // (codebooks + project_out + fc2) next to the encoder weights.
    const load_start = nowNs(io);
    const config = try codec.parseConfig(file);
    var rvq_dec = try codec.loadRvqDecoder(ctx, file, config);
    defer rvq_dec.deinit(allocator);
    var enc = try codec.Encoder.load(ctx, file);
    defer enc.deinit();
    const load_ns: u64 = @intCast(nowNs(io) - load_start);

    var samples = try wav.readMono(io, allocator, input, @intCast(config.sample_rate));
    defer allocator.free(samples);
    _ = try postproc.refPreprocessAudio(allocator, &samples, @intCast(config.sample_rate), true);
    const n_aligned = (samples.len / config.hop_length) * config.hop_length;
    if (n_aligned == 0) {
        try stdout.print("codec: input too short after preprocessing ({d} samples, hop {d})\n", .{ samples.len, config.hop_length });
        return error.InputTooShort;
    }

    var taps = rvq.EncodeTaps.init(allocator);
    defer taps.deinit();
    const taps_ptr: ?*rvq.EncodeTaps = if (dump_dir != null) &taps else null;

    const encode_start = nowNs(io);
    const codes = try rvq.encode(ctx, allocator, config, &rvq_dec, &enc, samples[0..n_aligned], taps_ptr);
    defer allocator.free(codes);
    const encode_ns: u64 = @intCast(nowNs(io) - encode_start);

    const rvq_path = out_path orelse try swapExtension(allocator, input, ".rvq");
    defer if (out_path == null) allocator.free(rvq_path);
    try rvq_file.writeFile(allocator, io, rvq_path, codes);

    const t = codes.len / codec.n_codebooks;
    try stdout.print(
        "codec encode: {d} samples ({d:.2} s) -> K={d} T={d}, load {d:.1} ms, encode {d:.1} ms -> {s}\n",
        .{
            n_aligned,
            @as(f64, @floatFromInt(n_aligned)) / @as(f64, @floatFromInt(config.sample_rate)),
            codec.n_codebooks,
            t,
            @as(f64, @floatFromInt(load_ns)) / 1e6,
            @as(f64, @floatFromInt(encode_ns)) / 1e6,
            rvq_path,
        },
    );

    if (dump_dir) |dir| {
        try writeRawF32(io, allocator, dir, "fuc-ref-audio-16k.raw", taps.audio_16k.?);
        // HuBERT taps are [T, C] row-major — the same layout the reference's
        // .bin goldens use for their data payload.
        try writeRawF32(io, allocator, dir, "fuc-hubert-feat-extract.raw", taps.hubert.feat_extract.?.data);
        try writeRawF32(io, allocator, dir, "fuc-hubert-feat-proj-ln.raw", taps.hubert.proj_ln.?.data);
        try writeRawF32(io, allocator, dir, "fuc-hubert-feat-proj.raw", taps.hubert.proj.?.data);
        try writeRawF32(io, allocator, dir, "fuc-hubert-enc-init.raw", taps.hubert.enc_init.?.data);
        var name_buf: [64]u8 = undefined;
        for ([_]usize{ 0, 5, 7, 9, 11 }) |layer_i| {
            const name = try std.fmt.bufPrint(&name_buf, "fuc-hubert-l{d}.raw", .{layer_i});
            try writeRawF32(io, allocator, dir, name, taps.hubert.layer[layer_i].?.data);
        }
        try writeRawF32(io, allocator, dir, "fuc-ref-hubert-features.raw", taps.features.?.data);
        try writeRawF32(io, allocator, dir, "fuc-e-semantic.raw", taps.e_semantic.?.data);
        try writeRawF32(io, allocator, dir, "fuc-e-acoustic.raw", taps.e_acoustic.?.data);
        // pre_fc/embed: [T, 1024] row-major — identical flat layout to the
        // reference's cpp_encode_pre_fc/cpp_encode_embed ne=(1024, T) dumps.
        try writeRawF32(io, allocator, dir, "fuc_encode_pre_fc.raw", taps.pre_fc.?.data);
        try writeRawF32(io, allocator, dir, "fuc_encode_embed.raw", taps.embed.?.data);
        try stdout.print("stage dumps written to {s}\n", .{dir});
    }
}

fn nowNs(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).nanoseconds;
}

/// `foo.rvq` → `foo<new_ext>` (like the reference: swap the extension,
/// keeping the directory). Caller frees.
fn swapExtension(allocator: std.mem.Allocator, path: []const u8, new_ext: []const u8) ![]u8 {
    const ext = std.fs.path.extension(path);
    const stem = path[0 .. path.len - ext.len];
    return std.fmt.allocPrint(allocator, "{s}{s}", .{ stem, new_ext });
}

/// Writes a headerless little-endian raw f32 file `<dir>/<name>`.
fn writeRawF32(io: std.Io, allocator: std.mem.Allocator, dir: []const u8, name: []const u8, values: []const f32) !void {
    const path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ dir, name });
    defer allocator.free(path);

    var file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var write_buffer: [64 * 1024]u8 = undefined;
    var writer = file.writer(io, &write_buffer);
    const w = &writer.interface;
    for (values) |v| {
        try w.writeInt(u32, @bitCast(v), .little);
    }
    try w.flush();
}

/// Writes a `[t, c]` row-major stage tap as the reference's CHANNEL-PLANAR
/// raw buffer (ggml ne=(T,C): index c*T + t).
fn writePlanarTap(io: std.Io, allocator: std.mem.Allocator, dir: []const u8, name: []const u8, tap: dac.StageTap) !void {
    const planar = try allocator.alloc(f32, tap.data.len);
    defer allocator.free(planar);
    for (0..tap.t) |ti| {
        for (0..tap.c) |ci| {
            planar[ci * tap.t + ti] = tap.data[ti * tap.c + ci];
        }
    }
    try writeRawF32(io, allocator, dir, name, planar);
}

/// Reads a headerless little-endian raw f32 file. Caller frees.
fn readRawF32(allocator: std.mem.Allocator, io: std.Io, path: []const u8) ![]f32 {
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);

    const stat = try file.stat(io);
    if (stat.kind != .file) return error.NotAFile;
    const byte_len: usize = @intCast(stat.size);
    if (byte_len % 4 != 0) return error.CorruptRawFile;
    const bytes = try allocator.alloc(u8, byte_len);
    defer allocator.free(bytes);

    var read_len: usize = 0;
    while (read_len < bytes.len) {
        const n = try file.readStreaming(io, &.{bytes[read_len..]});
        if (n == 0) return error.EndOfStream;
        read_len += n;
    }

    const values = try allocator.alloc(f32, byte_len / 4);
    for (values, 0..) |*dst, vi| {
        dst.* = @bitCast(std.mem.readInt(u32, bytes[vi * 4 ..][0..4], .little));
    }
    return values;
}

test {
    _ = philox;
    _ = resample;
    _ = wav;
    _ = dump;
    _ = rvq_file;
    _ = postproc;
    _ = postproc_stream;
    _ = chunker;
    _ = chunker_stream;
    _ = duration;
    _ = voicedesign;
    _ = langmap;
    _ = maskgit;
    _ = mg_decode;
    _ = play;
    _ = codec;
    _ = rvq;
    _ = dac;
    _ = lm;
    _ = pipeline;
    _ = @import("omnivoice/root_tests.zig");
}
