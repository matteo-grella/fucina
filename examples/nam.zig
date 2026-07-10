//! Neural Amp Modeler example — see examples/nam/README.md.
//!
//! End-to-end NAM in Fucina: import/run/export canonical `.nam` profiles
//! (interoperable with NeuralAmpModelerCore / neural-amp-modeler in both
//! directions), CPU profiling/training, and live guitar -> model -> output
//! processing on normal audio devices.

const std = @import("std");
const fucina = @import("fucina");

const nam_file = @import("nam/nam_file.zig");
const wav = @import("nam/wav.zig");
const engine_mod = @import("nam/engine.zig");
const ir_cab = @import("nam/ir_cab.zig");
const chain_mod = @import("nam/chain.zig");
const data = @import("nam/data.zig");
const train_mod = @import("nam/train.zig");
const nam_export = @import("nam/nam_export.zig");
const audio_mod = @import("nam/audio.zig");
const midi_mod = @import("nam/midi.zig");
const live_mod = @import("nam/live.zig");
const gguf_compat = @import("nam/gguf_compat.zig");
const ui = @import("nam/ui.zig");

const usage =
    \\fucina NAM — Neural Amp Modeler (see examples/nam/README.md)
    \\
    \\usage: zig build nam [-Doptimize=ReleaseFast] -- <command> [args]
    \\       (run with no command for the interactive amp menu)
    \\
    \\play:
    \\  live [<profile>...] [--ir cab.wav] [--chain rig.chain] [--capture N] [--playback N]
    \\       [--rate 48000] [--period 128] [--gain dB] [--input-gain dB] [--no-normalize]
    \\       [--gate dB] [--auto-input] [--midi N | --no-midi] [--midi-channel C] [--midi-map ...]
    \\      play a guitar through profiles and/or chains in realtime
    \\  render <model> <input.wav> <output.wav> [--blocksize N] [--no-prewarm] [--ir cab.wav]
    \\      offline file processing (matches upstream tools/render); --ir appends a cab
    \\  bench <model> [--blocksize N] [--seconds S]
    \\      per-block cost vs the realtime budget (use -Doptimize=ReleaseFast)
    \\  devices
    \\      list capture/playback devices and MIDI sources with indices
    \\
    \\profile / train:
    \\  profile --signal s.wav --reamp-out r.wav --out m.nam [--capture N] [--playback N] [...]
    \\      one-step reamp capture + train + export
    \\  train --input in.wav --output reamp.wav --out m.nam [--spec standard|tiny|a2|a2-nano|packed | --init model.nam] [...]
    \\      train a WaveNet from an existing pair; default is standard, --init fine-tunes a supported WaveNet
    \\  validate <model> --input in.wav --output reamp.wav [--write-wavs dir]
    \\      report ESR and write time-aligned A/B WAVs
    \\
    \\manage / interchange:
    \\  inspect <model.nam|.gguf>                 print structure + metadata
    \\  list [--profiles-dir d]                   list profiles in ./nam-profiles ($FUCINA_NAM_PROFILES)
    \\  export-gguf / import-gguf                 lossless .nam <-> GGUF interchange
    \\  loopback-test --capture N --playback N    measure the true round-trip latency
    \\
;

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);
    const io = init.io;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    if (args.len < 2) {
        return interactive(io, allocator, stdout) catch |err| switch (err) {
            error.NoProfilesFound => {},
            else => return err,
        };
    }

    const command = args[1];
    if (std.mem.eql(u8, command, "help") or std.mem.eql(u8, command, "--help") or std.mem.eql(u8, command, "-h")) {
        try stdout.writeAll(usage);
        return;
    }
    const run = if (std.mem.eql(u8, command, "inspect"))
        inspect(io, allocator, stdout, args[2..])
    else if (std.mem.eql(u8, command, "render"))
        render(io, allocator, stdout, args[2..])
    else if (std.mem.eql(u8, command, "bench"))
        bench(io, allocator, stdout, args[2..])
    else if (std.mem.eql(u8, command, "train"))
        train(io, allocator, stdout, args[2..])
    else if (std.mem.eql(u8, command, "validate"))
        validate(io, allocator, stdout, args[2..])
    else if (std.mem.eql(u8, command, "list"))
        listProfiles(io, allocator, stdout, args[2..])
    else if (std.mem.eql(u8, command, "export-gguf"))
        exportGguf(io, allocator, stdout, args[2..])
    else if (std.mem.eql(u8, command, "import-gguf"))
        importGguf(io, allocator, stdout, args[2..])
    else if (std.mem.eql(u8, command, "profile"))
        profileCapture(io, allocator, stdout, args[2..])
    else if (std.mem.eql(u8, command, "loopback-test"))
        loopbackTest(io, allocator, stdout, args[2..])
    else if (std.mem.eql(u8, command, "devices"))
        devices(stdout, args[2..])
    else if (std.mem.eql(u8, command, "live"))
        live(io, allocator, stdout, args[2..])
    else {
        try stdout.print("unknown command: {s}\n\n", .{command});
        try stdout.writeAll(usage);
        return;
    };

    run catch |err| switch (err) {
        nam_file.Error.UnsupportedFeature => {
            try stdout.writeAll("error: this model uses a NAM feature this runtime does not support " ++
                "(for example slimmable WaveNet submodels or non-WaveNet condition_dsp)\n");
            return err;
        },
        nam_file.Error.UnsupportedVersion => {
            try stdout.writeAll("error: .nam file version outside the supported 0.5.0..0.7.x window\n");
            return err;
        },
        nam_file.Error.WeightCountMismatch => {
            try stdout.writeAll("error: weights array length does not match the architecture config (corrupt file?)\n");
            return err;
        },
        else => return err,
    };
}

fn loadModel(io: std.Io, allocator: std.mem.Allocator, stdout: *std.Io.Writer, path: []const u8) !nam_file.NamModel {
    const model = try gguf_compat.loadAny(io, allocator, path);
    if (model.partial_support) {
        try stdout.print("note: {s} has a newer patch version than 0.7.0; loading with partial support (same as upstream)\n", .{path});
    }
    return model;
}

/// Returns the value following the current flag, advancing `i`. Errors instead
/// of panicking when the flag was the last argument (a trailing-flag typo).
fn nextArg(args: []const []const u8, i: *usize) error{MissingArgumentValue}![]const u8 {
    i.* += 1;
    if (i.* >= args.len) return error.MissingArgumentValue;
    return args[i.*];
}

test "nextArg returns the next value, or errors on a trailing flag" {
    const args = [_][]const u8{ "--input", "x.wav", "--latency" };
    var i: usize = 0;
    try std.testing.expectEqualStrings("x.wav", try nextArg(&args, &i));
    try std.testing.expectEqual(@as(usize, 1), i);
    i = 2; // points at the last arg ("--latency") with no value after it
    try std.testing.expectError(error.MissingArgumentValue, nextArg(&args, &i));
}

fn inspect(io: std.Io, allocator: std.mem.Allocator, stdout: *std.Io.Writer, args: []const []const u8) !void {
    if (args.len < 1) {
        try stdout.writeAll("usage: zig build nam -- inspect <model.nam>\n");
        return;
    }
    var model = try loadModel(io, allocator, stdout, args[0]);
    defer model.deinit();

    try stdout.print("file:          {s}\n", .{args[0]});
    try stdout.print("version:       {d}.{d}.{d}\n", .{ model.version.major, model.version.minor, model.version.patch });
    try stdout.print("architecture:  {s}\n", .{@tagName(model.architecture)});
    if (model.sample_rate < 0) {
        try stdout.writeAll("sample_rate:   unknown (old file)\n");
    } else {
        try stdout.print("sample_rate:   {d}\n", .{model.sample_rate});
    }
    if (model.submodel_info) |info| {
        try stdout.print("container:     SlimmableContainer — loaded submodel {d}/{d} (max_value {d}, highest quality)\n", .{ info.index + 1, info.count, info.max_value });
    }
    try stdout.print("weights:       {d}\n", .{model.weights.len});
    if (model.metadata.loudness) |v| try stdout.print("loudness:      {d:.2} dB\n", .{v});
    if (model.metadata.input_level_dbu) |v| try stdout.print("input level:   {d:.2} dBu\n", .{v});
    if (model.metadata.output_level_dbu) |v| try stdout.print("output level:  {d:.2} dBu\n", .{v});

    switch (model.config) {
        .wavenet => |c| {
            try stdout.print("receptive:     {d} samples\n", .{c.receptiveField()});
            try stdout.print("layer arrays:  {d}\n", .{c.layers.len});
            for (c.layers, 0..) |*array, i| {
                try stdout.print("  [{d}] channels {d} bottleneck {d} layers {d} head {d}x{d}{s}\n", .{
                    i,              array.channels,    array.bottleneck,                      array.layerCount(),
                    array.head_out, array.head_kernel, if (array.head_bias) " +bias" else "",
                });
            }
        },
        .lstm => |c| try stdout.print("lstm:          {d} layer(s), hidden {d}\n", .{ c.num_layers, c.hidden_size }),
        .convnet => |c| try stdout.print("convnet:       {d} blocks, channels {d}, batchnorm {}\n", .{ c.dilations.len, c.channels, c.batchnorm }),
        .linear => |c| try stdout.print("linear:        {d} taps, bias {}\n", .{ c.receptive_field, c.bias }),
    }
}

fn render(io: std.Io, allocator: std.mem.Allocator, stdout: *std.Io.Writer, args: []const []const u8) !void {
    if (args.len < 3) {
        try stdout.writeAll("usage: zig build nam -- render <model.nam> <input.wav> <output.wav> [--blocksize N] [--no-prewarm] [--ir cab.wav]\n");
        return;
    }
    var blocksize: usize = 64; // upstream tools/render uses fixed 64-frame blocks
    var prewarm = true;
    var ir_path: ?[]const u8 = null;
    var i: usize = 3;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--blocksize")) {
            i += 1;
            if (i >= args.len) return error.MissingBlocksize;
            blocksize = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.startsWith(u8, args[i], "--blocksize=")) {
            blocksize = try std.fmt.parseInt(usize, args[i]["--blocksize=".len..], 10);
        } else if (std.mem.eql(u8, args[i], "--no-prewarm")) {
            prewarm = false;
        } else if (std.mem.eql(u8, args[i], "--ir")) {
            i += 1;
            if (i >= args.len) return error.MissingArgumentValue;
            ir_path = args[i];
        } else return error.UnknownArgument;
    }
    if (blocksize == 0) return error.InvalidBlocksize;

    var model = try loadModel(io, allocator, stdout, args[0]);
    defer model.deinit();

    var input = try wav.readFile(io, allocator, args[1]);
    defer input.deinit();
    const samples = try input.requireMono();

    // Upstream render errors when the input WAV rate differs from the
    // model's expected rate; unknown-rate models accept anything.
    if (model.sample_rate > 0 and @as(f64, @floatFromInt(input.sample_rate)) != model.sample_rate) {
        try stdout.print("error: input is {d} Hz but the model expects {d} Hz (no resampling, same as upstream render)\n", .{ input.sample_rate, model.sample_rate });
        return error.SampleRateMismatch;
    }

    var engine = try engine_mod.Engine.init(allocator, &model);
    defer engine.deinit();
    try engine.reset(blocksize, prewarm);

    // Optional cab IR (resampled to the render rate at load); applied per
    // block after the model, exactly like the live path.
    var ir_storage: ?ir_cab.IrCab = null;
    defer if (ir_storage) |*c| c.deinit();
    if (ir_path) |p| {
        ir_storage = try ir_cab.IrCab.loadFile(io, allocator, p, input.sample_rate, blocksize);
        try stdout.print("cab IR: {s} ({d} taps @ {d} Hz)\n", .{ p, ir_storage.?.taps, input.sample_rate });
    }

    const output = try allocator.alloc(f32, samples.len);
    defer allocator.free(output);

    var offset: usize = 0;
    while (offset < samples.len) {
        const n = @min(blocksize, samples.len - offset);
        try engine.process(samples[offset..], output[offset..], n);
        if (ir_storage) |*c| c.process(output[offset..], output[offset..], n);
        offset += n;
    }

    try wav.writeMono(io, allocator, args[2], output, input.sample_rate, .float32);
    try stdout.print("rendered {d} samples ({d:.2}s @ {d} Hz) -> {s}\n", .{
        samples.len, @as(f64, @floatFromInt(samples.len)) / @as(f64, @floatFromInt(input.sample_rate)), input.sample_rate, args[2],
    });
}

fn nowNs(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).nanoseconds;
}

fn bench(io: std.Io, allocator: std.mem.Allocator, stdout: *std.Io.Writer, args: []const []const u8) !void {
    if (args.len < 1) {
        try stdout.writeAll("usage: zig build nam -- bench <model.nam> [--blocksize N] [--seconds S]\n");
        return;
    }
    var blocksize: usize = 64;
    var seconds: f64 = 5.0;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--blocksize")) {
            i += 1;
            if (i >= args.len) return error.MissingBlocksize;
            blocksize = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.startsWith(u8, args[i], "--blocksize=")) {
            blocksize = try std.fmt.parseInt(usize, args[i]["--blocksize=".len..], 10);
        } else if (std.mem.eql(u8, args[i], "--seconds")) {
            i += 1;
            if (i >= args.len) return error.MissingSeconds;
            seconds = try std.fmt.parseFloat(f64, args[i]);
        } else if (std.mem.startsWith(u8, args[i], "--seconds=")) {
            seconds = try std.fmt.parseFloat(f64, args[i]["--seconds=".len..]);
        } else return error.UnknownArgument;
    }
    if (blocksize == 0) {
        try stdout.writeAll("error: --blocksize must be > 0\n");
        return error.InvalidBlocksize;
    }
    // Finite + bounded: a non-finite or huge --seconds would panic the
    // float->int cast computing total_blocks (and nan ran one block silently).
    if (!std.math.isFinite(seconds) or seconds <= 0 or seconds > 3600) {
        try stdout.writeAll("error: --seconds must be finite and in (0, 3600]\n");
        return error.InvalidSeconds;
    }

    var model = try loadModel(io, allocator, stdout, args[0]);
    defer model.deinit();
    const rate: f64 = if (model.sample_rate > 0) model.sample_rate else 48000.0;

    var engine = try engine_mod.Engine.init(allocator, &model);
    defer engine.deinit();

    const reset_start = nowNs(io);
    try engine.reset(blocksize, true);
    const reset_ns: u64 = @intCast(nowNs(io) - reset_start);

    const input = try allocator.alloc(f32, blocksize);
    defer allocator.free(input);
    const output = try allocator.alloc(f32, blocksize);
    defer allocator.free(output);
    for (input, 0..) |*v, idx| v.* = 0.3 * @sin(@as(f32, @floatFromInt(idx)) * 0.02);

    const total_blocks: usize = @intFromFloat(@max(1.0, seconds * rate / @as(f64, @floatFromInt(blocksize))));
    // Warm up caches/branch predictors before timing.
    for (0..@min(total_blocks, 64)) |_| try engine.process(input, output, blocksize);

    const bench_start = nowNs(io);
    for (0..total_blocks) |_| {
        try engine.process(input, output, blocksize);
    }
    const elapsed_ns: u64 = @intCast(nowNs(io) - bench_start);

    const ns_per_block = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(total_blocks));
    const budget_ns = @as(f64, @floatFromInt(blocksize)) / rate * 1e9;
    try stdout.print("model:        {s} ({s}, {d} weights)\n", .{ args[0], @tagName(model.architecture), model.weights.len });
    try stdout.print("prewarm:      {d} samples, reset took {d:.1} ms\n", .{ engine.prewarmSamples(), @as(f64, @floatFromInt(reset_ns)) / 1e6 });
    try stdout.print("blocksize:    {d} frames @ {d} Hz (budget {d:.0} us/block)\n", .{ blocksize, rate, budget_ns / 1e3 });
    try stdout.print("measured:     {d:.1} us/block over {d} blocks ({d:.1} ns/sample)\n", .{ ns_per_block / 1e3, total_blocks, ns_per_block / @as(f64, @floatFromInt(blocksize)) });
    try stdout.print("realtime:     {d:.1}x headroom\n", .{budget_ns / ns_per_block});
}

const TrainSplits = struct {
    version: data.InputVersion,
    latency: i64,
    calibration: ?data.LatencyCalibration,
    checks_passed: bool,
    train_x: []const f32,
    train_y: []const f32,
    val_x: []const f32,
    val_y: []const f32,
};

const NormalizedOutputs = struct {
    train_y: []f32,
    val_y: []f32,
    train_scale: f32,

    const target_dbfs: f64 = -18.0;

    fn deinit(self: *NormalizedOutputs, allocator: std.mem.Allocator) void {
        allocator.free(self.train_y);
        allocator.free(self.val_y);
        self.* = undefined;
    }

    fn exportCompensation(self: *const NormalizedOutputs) f32 {
        return 1.0 / self.train_scale;
    }
};

fn normalizeJointOutput(allocator: std.mem.Allocator, train_y: []const f32, val_y: []const f32) !NormalizedOutputs {
    var sum_sq: f64 = 0;
    for (train_y) |v| sum_sq += @as(f64, v) * v;
    if (train_y.len == 0) return error.EmptyTrainingData;
    if (sum_sq == 0) return error.ZeroTrainingOutput;
    const train_rms = @sqrt(sum_sq / @as(f64, @floatFromInt(train_y.len)));
    const target_rms = std.math.pow(f64, 10.0, NormalizedOutputs.target_dbfs / 20.0);
    const scale: f32 = @floatCast(target_rms / train_rms);
    if (!std.math.isFinite(scale) or scale == 0) return error.InvalidOutputScale;

    const train_scaled = try allocator.alloc(f32, train_y.len);
    errdefer allocator.free(train_scaled);
    const val_scaled = try allocator.alloc(f32, val_y.len);
    errdefer allocator.free(val_scaled);
    for (train_scaled, train_y) |*dst, v| dst.* = scale * v;
    for (val_scaled, val_y) |*dst, v| dst.* = scale * v;
    return .{ .train_y = train_scaled, .val_y = val_scaled, .train_scale = scale };
}

test "normalizeJointOutput scales train and validation from training RMS" {
    const allocator = std.testing.allocator;
    const train_y = [_]f32{ 0.25, -0.25, 0.5, -0.5 };
    const val_y = [_]f32{ 0.125, -0.125 };
    var normalized = try normalizeJointOutput(allocator, &train_y, &val_y);
    defer normalized.deinit(allocator);

    const train_rms = @sqrt((4.0 * 0.25 * 0.25 + 4.0 * 0.5 * 0.5) / 8.0);
    const target_rms = std.math.pow(f64, 10.0, NormalizedOutputs.target_dbfs / 20.0);
    const expected_scale: f32 = @floatCast(target_rms / train_rms);
    try std.testing.expectApproxEqAbs(expected_scale, normalized.train_scale, 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25 * expected_scale), normalized.train_y[0], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, -0.5 * expected_scale), normalized.train_y[3], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.125 * expected_scale), normalized.val_y[0], 1e-7);
    try std.testing.expectApproxEqAbs(1.0 / expected_scale, normalized.exportCompensation(), 1e-7);
}

test "normalizeJointOutput rejects empty or zero training output" {
    const allocator = std.testing.allocator;
    const val_y = [_]f32{0.125};
    try std.testing.expectError(error.EmptyTrainingData, normalizeJointOutput(allocator, &.{}, &val_y));
    const zero_train = [_]f32{ 0, 0, 0 };
    try std.testing.expectError(error.ZeroTrainingOutput, normalizeJointOutput(allocator, &zero_train, &val_y));
}

fn checkV3InputPreSilence(x: []const f32) data.DataError!void {
    try data.checkInputPreSilence(x, data.v3.train_start, data.standard_sample_rate);
    try data.checkInputPreSilence(x, x.len - data.v3.t_validate, data.standard_sample_rate);
}

test "v3 input pre-silence is required before train and validation splits" {
    const allocator = std.testing.allocator;
    const n = data.v3.train_start + data.v3.t_validate + @as(usize, 48_000);
    const x = try allocator.alloc(f32, n);
    defer allocator.free(x);

    @memset(x, 0);
    try checkV3InputPreSilence(x);

    x[data.v3.train_start - 1] = 0.125;
    try std.testing.expectError(data.DataError.InputPreSilenceMissing, checkV3InputPreSilence(x));

    @memset(x, 0);
    const validation_start = x.len - data.v3.t_validate;
    x[validation_start - 1] = 0.125;
    try std.testing.expectError(data.DataError.InputPreSilenceMissing, checkV3InputPreSilence(x));
}

/// Resolves capture version, latency, checks, and the train/validation
/// splits for an (input, reamp) pair, matching the upstream
/// neural-amp-modeler trainer: latency (delay) calibration, the v3 data
/// checks, and the per-input-version train/validation split points.
fn resolveSplits(
    stdout: *std.Io.Writer,
    input_bytes: []const u8,
    x: []const f32,
    y: []const f32,
    manual_latency: ?i64,
    ignore_checks: bool,
) !TrainSplits {
    const version = data.detectInputVersion(input_bytes);
    switch (version) {
        .v1_0_0, .v1_1_1, .v2_0_0, .v4_0_0 => {
            try stdout.writeAll("error: v1/v2/v4 capture files are deprecated upstream; re-record with the v3 input file\n");
            return error.DeprecatedInputVersion;
        },
        else => {},
    }

    var calibration: ?data.LatencyCalibration = null;
    var latency: i64 = manual_latency orelse 0;
    var checks_passed = true;

    if (version == .v3_0_0) {
        const cal = data.calibrateLatencyV3(y);
        calibration = cal;
        if (manual_latency == null) {
            latency = cal.recommended orelse {
                try stdout.writeAll("error: latency blips not detected; pass --latency or re-record\n");
                return error.LatencyNotDetected;
            };
        }
        if (cal.warn_matches_lookahead) try stdout.writeAll("warning: latency trigger fired at the scan start (noisy capture?)\n");

        const check = data.checkV3(x.len, y);
        checks_passed = check.passed;
        try stdout.print("v3 capture detected; latency {d} samples; replicate self-ESR {d:.6} ({s})\n", .{ latency, check.replicate_esr, if (check.passed) "ok" else "FAILED" });
        if (!check.passed and !ignore_checks) {
            try stdout.writeAll("error: validation replicates disagree (> 0.01 self-ESR): noise/gate/time-based FX or drift. Use --ignore-checks to proceed anyway.\n");
            return error.DataChecksFailed;
        }
        // The v3 split slices x/y as [train_start .. len - t_validate] and
        // [len - t_validate ..]; a reamp shorter than train_start + t_validate
        // would make the train end precede its start (and underflow the usize
        // subtraction). checkV3 only sizes the validation windows, so guard here.
        const v3_min = data.v3.train_start + data.v3.t_validate;
        if (x.len < v3_min or y.len < v3_min) {
            try stdout.writeAll("error: capture too short — the v3 input/reamp is missing the training/validation tail; re-record the full file\n");
            return error.CaptureTooShort;
        }
        try checkV3InputPreSilence(x);

        const train_pair = try data.applyDelay(x[data.v3.train_start .. x.len - data.v3.t_validate], y[data.v3.train_start .. y.len - data.v3.t_validate], latency);
        const val_pair = try data.applyDelay(x[x.len - data.v3.t_validate ..], y[y.len - data.v3.t_validate ..], latency);
        try data.checkOutputNotClipped(train_pair.y);
        try data.checkOutputNotClipped(val_pair.y);
        return .{ .version = version, .latency = latency, .calibration = calibration, .checks_passed = checks_passed, .train_x = train_pair.x, .train_y = train_pair.y, .val_x = val_pair.x, .val_y = val_pair.y };
    }

    // Generic pair: -9 s validation tail (single_pair.json), manual latency.
    if (manual_latency == null) {
        try stdout.writeAll("note: unrecognized capture signal; assuming --latency 0 (pass it explicitly if your interface has loopback delay)\n");
    }
    const n = @min(x.len, y.len);
    var val_len: usize = @intFromFloat(9.0 * data.standard_sample_rate);
    if (val_len * 2 > n) {
        val_len = n / 4; // short clips: hold out the last quarter
        try stdout.writeAll("note: short capture; holding out the last 25% for validation instead of 9 s\n");
    }
    const train_pair = try data.applyDelay(x[0 .. n - val_len], y[0 .. n - val_len], latency);
    const val_pair = try data.applyDelay(x[n - val_len .. n], y[n - val_len .. n], latency);
    try data.checkOutputNotClipped(train_pair.y);
    try data.checkOutputNotClipped(val_pair.y);
    return .{ .version = version, .latency = latency, .calibration = calibration, .checks_passed = checks_passed, .train_x = train_pair.x, .train_y = train_pair.y, .val_x = val_pair.x, .val_y = val_pair.y };
}

fn validationEsrConfig(allocator: std.mem.Allocator, config: *const nam_file.WaveNetConfig, weights: []const f32, val_x: []const f32, val_y: []const f32, nx: usize) !f64 {
    const pred = try allocator.alloc(f32, val_x.len);
    defer allocator.free(pred);
    try train_mod.renderWaveNetConfig(allocator, config, weights, val_x, pred);
    return data.esr(pred[nx - 1 ..], val_y[nx - 1 ..]);
}

fn validationEsrSnapshot(allocator: std.mem.Allocator, snapshot: *const train_mod.TrainingSnapshot, val_x: []const f32, val_y: []const f32, nx: usize) !f64 {
    return switch (snapshot.*) {
        .wavenet => |*s| try validationEsrConfig(allocator, &s.config, s.weights, val_x, val_y, nx),
        .packed_wavenet => |*packed_snapshot| blk: {
            var total: f64 = 0;
            for (packed_snapshot.submodels) |*submodel| {
                total += try validationEsrConfig(allocator, &submodel.config, submodel.weights, val_x, val_y, nx);
            }
            break :blk total / @as(f64, @floatFromInt(packed_snapshot.submodels.len));
        },
    };
}

fn train(io: std.Io, allocator: std.mem.Allocator, stdout: *std.Io.Writer, args: []const []const u8) !void {
    var input_path: ?[]const u8 = null;
    var output_path: ?[]const u8 = null;
    var out_path: ?[]const u8 = null;
    var init_path: ?[]const u8 = null;
    var spec = train_mod.TrainingSpec{ .classic = train_mod.ModelSpec.classic };
    var epochs_override: ?usize = null;
    var batch_size: usize = 16;
    var ny: usize = 8192;
    var lr_override: ?f32 = null;
    var weight_decay_override: ?f32 = null;
    var gamma_override: ?f32 = null;
    var mrstft_weight_override: ?f32 = null;
    var seed: u64 = 0;
    var manual_latency: ?i64 = null;
    var ignore_checks = false;
    var user = nam_export.UserMetadata{};

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--input")) {
            input_path = try nextArg(args, &i);
        } else if (std.mem.eql(u8, arg, "--output")) {
            output_path = try nextArg(args, &i);
        } else if (std.mem.eql(u8, arg, "--out")) {
            out_path = try nextArg(args, &i);
        } else if (std.mem.eql(u8, arg, "--init")) {
            init_path = try nextArg(args, &i);
        } else if (std.mem.eql(u8, arg, "--spec")) {
            const v = try nextArg(args, &i);
            spec = try train_mod.TrainingSpec.parse(v);
        } else if (std.mem.eql(u8, arg, "--epochs")) {
            epochs_override = try std.fmt.parseInt(usize, try nextArg(args, &i), 10);
        } else if (std.mem.eql(u8, arg, "--batch")) {
            batch_size = try std.fmt.parseInt(usize, try nextArg(args, &i), 10);
        } else if (std.mem.eql(u8, arg, "--ny")) {
            ny = try std.fmt.parseInt(usize, try nextArg(args, &i), 10);
        } else if (std.mem.eql(u8, arg, "--lr")) {
            lr_override = try std.fmt.parseFloat(f32, try nextArg(args, &i));
        } else if (std.mem.eql(u8, arg, "--weight-decay")) {
            weight_decay_override = try std.fmt.parseFloat(f32, try nextArg(args, &i));
        } else if (std.mem.eql(u8, arg, "--gamma")) {
            gamma_override = try std.fmt.parseFloat(f32, try nextArg(args, &i));
        } else if (std.mem.eql(u8, arg, "--mrstft-weight")) {
            mrstft_weight_override = try std.fmt.parseFloat(f32, try nextArg(args, &i));
        } else if (std.mem.eql(u8, arg, "--seed")) {
            seed = try std.fmt.parseInt(u64, try nextArg(args, &i), 10);
        } else if (std.mem.eql(u8, arg, "--latency")) {
            manual_latency = try std.fmt.parseInt(i64, try nextArg(args, &i), 10);
        } else if (std.mem.eql(u8, arg, "--ignore-checks")) {
            ignore_checks = true;
        } else if (std.mem.eql(u8, arg, "--name")) {
            user.name = try nextArg(args, &i);
        } else if (std.mem.eql(u8, arg, "--modeled-by")) {
            user.modeled_by = try nextArg(args, &i);
        } else if (std.mem.eql(u8, arg, "--gear-type")) {
            user.gear_type = try nextArg(args, &i);
        } else if (std.mem.eql(u8, arg, "--gear-make")) {
            user.gear_make = try nextArg(args, &i);
        } else if (std.mem.eql(u8, arg, "--gear-model")) {
            user.gear_model = try nextArg(args, &i);
        } else if (std.mem.eql(u8, arg, "--tone-type")) {
            user.tone_type = try nextArg(args, &i);
        } else return error.UnknownArgument;
    }
    if (input_path == null or output_path == null or out_path == null) {
        try stdout.writeAll("usage: zig build nam -Doptimize=ReleaseFast -- train --input in.wav --output reamp.wav --out model.nam [--spec standard|tiny|a2|a2-nano|packed | --init model.nam] [options]\n");
        return;
    }
    if (batch_size == 0) {
        try stdout.writeAll("error: --batch must be > 0\n");
        return error.InvalidBatchSize;
    }
    if (ny == 0) {
        try stdout.writeAll("error: --ny must be > 0\n");
        return error.InvalidNy;
    }
    const epochs = epochs_override orelse spec.defaultEpochs();
    const lr0 = lr_override orelse spec.defaultLr();
    const weight_decay = weight_decay_override orelse spec.defaultWeightDecay();
    const gamma = gamma_override orelse spec.defaultGamma();
    const mrstft_weight = mrstft_weight_override orelse spec.defaultMrstftWeight();
    const loss_options = train_mod.LossOptions{ .mrstft_weight = mrstft_weight };

    const input_bytes = try wav.readFileBytes(io, allocator, input_path.?);
    defer allocator.free(input_bytes);
    var input_wav = try wav.parse(allocator, input_bytes);
    defer input_wav.deinit();
    var output_wav = try wav.readFile(io, allocator, output_path.?);
    defer output_wav.deinit();
    const x = try input_wav.requireMono();
    const y = try output_wav.requireMono();
    if (input_wav.sample_rate != output_wav.sample_rate) return error.SampleRateMismatch;
    if (input_wav.sample_rate != 48000) {
        try stdout.print("error: training expects 48 kHz captures (got {d} Hz)\n", .{input_wav.sample_rate});
        return error.SampleRateMismatch;
    }

    var init_model: ?nam_file.NamModel = null;
    defer if (init_model) |*model| model.deinit();
    var owned_template_config: ?nam_file.WaveNetConfig = null;
    defer if (owned_template_config) |*config| train_mod.freeEngineConfig(allocator, config);
    var template_config: ?*const nam_file.WaveNetConfig = null;
    var train_name: []const u8 = undefined;
    if (init_path) |path| {
        init_model = try nam_file.loadFile(io, allocator, path);
        const loaded = &init_model.?;
        switch (loaded.config) {
            .wavenet => |*config| {
                template_config = config;
                train_name = "loaded-wavenet";
            },
            else => {
                try stdout.writeAll("error: --init currently trains WaveNet .nam files only\n");
                return error.UnsupportedArchitecture;
            },
        }
    } else {
        switch (spec) {
            .packed_wavenet => {
                train_name = spec.name();
            },
            else => {
                owned_template_config = try spec.makeEngineConfig(allocator);
                template_config = &owned_template_config.?;
                train_name = spec.name();
            },
        }
    }

    const splits = try resolveSplits(stdout, input_bytes, x, y, manual_latency, ignore_checks);
    var normalized = try normalizeJointOutput(allocator, splits.train_y, splits.val_y);
    defer normalized.deinit(allocator);
    const nx = if (template_config) |config| config.receptiveField() else spec.receptiveField();
    const dataset = data.Dataset{ .x = splits.train_x, .y = normalized.train_y, .nx = nx, .ny = ny };
    const example_count = dataset.len();
    const steps_per_epoch = example_count / batch_size;
    if (steps_per_epoch == 0) return error.NotEnoughTrainingData;
    if (splits.val_x.len <= nx) return error.NotEnoughValidationData;

    try stdout.print("training {s} spec: {d} examples (ny {d}), {d} steps/epoch x {d} epochs, batch {d}, lr {d}, gamma {d}, wd {d}, mrstft {d}, output scale {d}\n", .{
        train_name, example_count, ny, steps_per_epoch, epochs, batch_size, lr0, gamma, weight_decay, mrstft_weight, normalized.train_scale,
    });
    try stdout.flush();

    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model: train_mod.ActiveTrainable = undefined;
    if (init_model) |*loaded| {
        model = .{ .a2 = try train_mod.A2Trainable.initFromWaveNet(allocator, &ctx, template_config.?, loaded.weights) };
    } else {
        model = try spec.initTrainable(allocator, &ctx, seed);
    }
    defer model.deinit();
    var opt = fucina.optim.Adam.init(allocator, .{ .lr = lr0, .weight_decay = weight_decay });
    defer opt.deinit();
    try model.registerParams(&opt);

    const order = try allocator.alloc(usize, example_count);
    defer allocator.free(order);
    for (order, 0..) |*v, idx| v.* = idx;

    var best_snapshot: ?train_mod.TrainingSnapshot = null;
    defer if (best_snapshot) |*snapshot| snapshot.deinit(allocator);
    var best_esr = std.math.inf(f64);

    for (0..epochs) |epoch| {
        opt.config.lr = lr0 * std.math.pow(f32, gamma, @floatFromInt(epoch));
        // Deterministic shuffle (Fisher-Yates over rng.at counters).
        for (0..example_count) |idx| {
            const j = idx + rng.at(seed +% 0x5851f42d4c957f2d, epoch * example_count + idx) % (example_count - idx);
            std.mem.swap(usize, &order[idx], &order[j]);
        }

        const epoch_start = nowNs(io);
        var loss_sum: f64 = 0;
        for (0..steps_per_epoch) |step_index| {
            for (order[step_index * batch_size ..][0..batch_size]) |example_index| {
                const example = dataset.get(example_index);
                const scope = ctx.openExecScope();
                defer ctx.closeExecScope(scope);
                const loss = try model.segmentLossWithOptions(&ctx, example.input, example.target, loss_options);
                const scaled = try loss.scale(&ctx, 1.0 / @as(f32, @floatFromInt(batch_size)));
                loss_sum += try loss.item();
                try scaled.backward(&ctx);
            }
            try opt.step(&ctx);
            opt.zeroGrad();
        }

        var snapshot = try model.extractTrainingSnapshot(allocator, template_config);
        const val_esr = try validationEsrSnapshot(allocator, &snapshot, splits.val_x, normalized.val_y, nx);
        const epoch_seconds = @as(f64, @floatFromInt(@as(u64, @intCast(nowNs(io) - epoch_start)))) / 1e9;
        const improved = val_esr < best_esr;
        if (improved) {
            best_esr = val_esr;
            if (best_snapshot) |*old| old.deinit(allocator);
            best_snapshot = snapshot;
        } else {
            snapshot.deinit(allocator);
        }
        try stdout.print("epoch {d:>3}/{d}: train loss {d:.6}  val ESR {d:.6}{s}  ({d:.1}s)\n", .{
            epoch + 1, epochs, loss_sum / @as(f64, @floatFromInt(steps_per_epoch * batch_size)), val_esr, if (improved) " *" else "", epoch_seconds,
        });
        try stdout.flush();
    }

    const final_snapshot = if (best_snapshot) |*snapshot| snapshot else return error.NoEpochsRun;
    try stdout.print("validation ESR {d:.6} — {s}\n", .{ best_esr, data.esrComment(best_esr) });

    const unix_seconds: u64 = @intCast(@divTrunc(std.Io.Clock.real.now(io).nanoseconds, std.time.ns_per_s));
    const export_info = nam_export.ExportInfo{
        .user = user,
        .training = .{
            .ignore_checks = ignore_checks,
            .latency_manual = manual_latency,
            .calibration = splits.calibration,
            .checks_version = 3,
            .checks_passed = splits.checks_passed,
            .validation_esr = best_esr,
        },
        .unix_seconds = unix_seconds,
        .sample_rate = 48000.0,
        .output_scale_compensation = normalized.exportCompensation(),
    };
    switch (final_snapshot.*) {
        .wavenet => |*snapshot| try nam_export.exportWaveNetConfig(io, allocator, out_path.?, &snapshot.config, snapshot.weights, export_info),
        .packed_wavenet => |*snapshot| try nam_export.exportSlimmableContainer(io, allocator, out_path.?, snapshot.submodels, export_info),
    }
    try stdout.print("exported {s}\n", .{out_path.?});
}

fn validate(io: std.Io, allocator: std.mem.Allocator, stdout: *std.Io.Writer, args: []const []const u8) !void {
    if (args.len < 1) {
        try stdout.writeAll("usage: zig build nam -- validate <model.nam> --input in.wav --output reamp.wav [--latency N] [--write-wavs dir]\n");
        return;
    }
    const model_path = args[0];
    var input_path: ?[]const u8 = null;
    var output_path: ?[]const u8 = null;
    var manual_latency: ?i64 = null;
    var wav_dir: ?[]const u8 = null;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--input")) {
            input_path = try nextArg(args, &i);
        } else if (std.mem.eql(u8, args[i], "--output")) {
            output_path = try nextArg(args, &i);
        } else if (std.mem.eql(u8, args[i], "--latency")) {
            manual_latency = try std.fmt.parseInt(i64, try nextArg(args, &i), 10);
        } else if (std.mem.eql(u8, args[i], "--write-wavs")) {
            wav_dir = try nextArg(args, &i);
        } else return error.UnknownArgument;
    }
    if (input_path == null or output_path == null) return error.MissingArgumentValue;

    var model = try loadModel(io, allocator, stdout, model_path);
    defer model.deinit();

    const input_bytes = try wav.readFileBytes(io, allocator, input_path.?);
    defer allocator.free(input_bytes);
    var input_wav = try wav.parse(allocator, input_bytes);
    defer input_wav.deinit();
    var output_wav = try wav.readFile(io, allocator, output_path.?);
    defer output_wav.deinit();
    const x = try input_wav.requireMono();
    const y = try output_wav.requireMono();

    const splits = try resolveSplits(stdout, input_bytes, x, y, manual_latency, true);

    var engine = try engine_mod.Engine.init(allocator, &model);
    defer engine.deinit();
    try engine.reset(4096, true);
    const pred = try allocator.alloc(f32, splits.val_x.len);
    defer allocator.free(pred);
    var offset: usize = 0;
    while (offset < splits.val_x.len) {
        const n = @min(@as(usize, 4096), splits.val_x.len - offset);
        try engine.process(splits.val_x[offset..], pred[offset..], n);
        offset += n;
    }
    const value = data.esr(pred, splits.val_y);
    try stdout.print("validation ESR {d:.6} — {s}\n", .{ value, data.esrComment(value) });

    if (wav_dir) |dir| {
        var path_buf: [1024]u8 = undefined;
        const target_path = try std.fmt.bufPrint(&path_buf, "{s}/validation_target.wav", .{dir});
        try wav.writeMono(io, allocator, target_path, splits.val_y, input_wav.sample_rate, .float32);
        var path_buf2: [1024]u8 = undefined;
        const pred_path = try std.fmt.bufPrint(&path_buf2, "{s}/validation_model.wav", .{dir});
        try wav.writeMono(io, allocator, pred_path, pred, input_wav.sample_rate, .float32);
        try stdout.print("wrote A/B wavs to {s}/\n", .{dir});
    }
}

fn profilesDirPath(allocator: std.mem.Allocator, args: []const []const u8) error{MissingArgumentValue}![]const u8 {
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--profiles-dir")) {
            i += 1;
            if (i >= args.len) return error.MissingArgumentValue;
            return args[i];
        }
    }
    _ = allocator;
    if (std.c.getenv("FUCINA_NAM_PROFILES")) |env| {
        return std.mem.span(env);
    }
    return "nam-profiles";
}

fn listProfiles(io: std.Io, allocator: std.mem.Allocator, stdout: *std.Io.Writer, args: []const []const u8) !void {
    const dir_path = try profilesDirPath(allocator, args);
    var dir = std.Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true }) catch {
        try stdout.print("no profile directory at {s} (create it, or pass --profiles-dir / set $FUCINA_NAM_PROFILES)\n", .{dir_path});
        return;
    };
    defer dir.close(io);

    var found: usize = 0;
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .file) continue;
        const is_nam = std.ascii.endsWithIgnoreCase(entry.name, ".nam");
        const is_gguf = std.ascii.endsWithIgnoreCase(entry.name, ".gguf");
        if (!is_nam and !is_gguf) continue;
        found += 1;

        var path_buf: [1024]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buf, "{s}/{s}", .{ dir_path, entry.name });
        var model = gguf_compat.loadAny(io, allocator, path) catch |err| {
            try stdout.print("  {s:<40} (unreadable: {s})\n", .{ entry.name, @errorName(err) });
            continue;
        };
        defer model.deinit();
        try stdout.print("  {s:<40} {s:<8} v{d}.{d}.{d}  {d} weights", .{
            entry.name, @tagName(model.architecture), model.version.major, model.version.minor, model.version.patch, model.weights.len,
        });
        if (model.metadata.loudness) |v| try stdout.print("  loudness {d:.1} dB", .{v});
        try stdout.writeAll("\n");
    }
    if (found == 0) try stdout.print("no .nam/.gguf profiles in {s}\n", .{dir_path});
}

fn exportGguf(io: std.Io, allocator: std.mem.Allocator, stdout: *std.Io.Writer, args: []const []const u8) !void {
    if (args.len < 2) {
        try stdout.writeAll("usage: zig build nam -- export-gguf <model.nam> <out.gguf>\n");
        return;
    }
    var model = try loadModel(io, allocator, stdout, args[0]);
    defer model.deinit();
    try gguf_compat.exportGguf(io, allocator, &model, args[1]);
    try stdout.print("wrote {s} (nam.weights f32[{d}] + byte-verbatim nam.file_json)\n", .{ args[1], model.weights.len });
}

fn importGguf(io: std.Io, allocator: std.mem.Allocator, stdout: *std.Io.Writer, args: []const []const u8) !void {
    if (args.len < 2) {
        try stdout.writeAll("usage: zig build nam -- import-gguf <model.gguf> <out.nam>\n");
        return;
    }
    try gguf_compat.exportNamFromGguf(io, allocator, args[0], args[1]);
    try stdout.print("recovered {s} (byte-identical to the .nam embedded at conversion)\n", .{args[1]});
}

const CaptureState = struct {
    signal: []const f32,
    recorded: []f32,
    cursor: std.atomic.Value(usize) = .init(0),
    in_peak_bits: std.atomic.Value(u32) = .init(0),
};

fn captureCallback(user: ?*anyopaque, output: ?[*]f32, input: ?[*]const f32, frame_count: c_uint) callconv(.c) void {
    const state: *CaptureState = @ptrCast(@alignCast(user.?));
    const frames: usize = frame_count;
    const out = output orelse return;
    const in = input orelse return;
    var pos = state.cursor.load(.monotonic);
    var in_peak: f32 = @bitCast(state.in_peak_bits.load(.monotonic));
    for (0..frames) |i| {
        out[i] = if (pos < state.signal.len) state.signal[pos] else 0.0;
        if (pos < state.recorded.len) state.recorded[pos] = in[i];
        in_peak = @max(in_peak, @abs(in[i]));
        pos += 1;
    }
    state.in_peak_bits.store(@bitCast(in_peak), .monotonic);
    state.cursor.store(pos, .release);
}

fn profileCapture(io: std.Io, allocator: std.mem.Allocator, stdout: *std.Io.Writer, args: []const []const u8) !void {
    var signal_path: ?[]const u8 = null;
    var reamp_path: ?[]const u8 = null;
    var capture_index: ?usize = null;
    var playback_index: ?usize = null;
    var period: u32 = 256;
    var train_args: std.ArrayList([]const u8) = .empty;
    defer train_args.deinit(allocator);

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--signal")) {
            signal_path = try nextArg(args, &i);
        } else if (std.mem.eql(u8, arg, "--reamp-out")) {
            reamp_path = try nextArg(args, &i);
        } else if (std.mem.eql(u8, arg, "--capture")) {
            capture_index = try std.fmt.parseInt(usize, try nextArg(args, &i), 10);
        } else if (std.mem.eql(u8, arg, "--playback")) {
            playback_index = try std.fmt.parseInt(usize, try nextArg(args, &i), 10);
        } else if (std.mem.eql(u8, arg, "--period")) {
            period = try std.fmt.parseInt(u32, try nextArg(args, &i), 10);
        } else {
            // Everything else (incl. --out) passes through to train.
            try train_args.append(allocator, arg);
        }
    }
    if (signal_path == null or reamp_path == null) {
        try stdout.writeAll("usage: zig build nam -Doptimize=ReleaseFast -- profile --signal capture.wav --reamp-out reamp.wav --out model.nam [--capture N] [--playback N] [--period 256] [train flags]\n");
        return;
    }

    var signal_wav = try wav.readFile(io, allocator, signal_path.?);
    defer signal_wav.deinit();
    const signal = try signal_wav.requireMono();
    if (signal_wav.sample_rate != 48000) {
        try stdout.print("error: capture signal must be 48 kHz (got {d})\n", .{signal_wav.sample_rate});
        return error.SampleRateMismatch;
    }

    // Record the signal length plus a 1 s tail (latency + reverb decay).
    const total = signal.len + 48000;
    const recorded = try allocator.alloc(f32, total);
    defer allocator.free(recorded);
    @memset(recorded, 0);

    var state = CaptureState{ .signal = signal, .recorded = recorded };
    var audio = try audio_mod.Audio.init();
    defer audio.deinit();

    try stdout.print("reamping {d:.1}s through the device chain — plug the interface output into the amp/pedal input and the return into the capture channel\n", .{@as(f64, @floatFromInt(signal.len)) / 48000.0});
    try stdout.flush();

    try audio.start(capture_index, playback_index, 48000, period, captureCallback, &state);
    while (state.cursor.load(.acquire) < total) {
        const pos = state.cursor.load(.monotonic);
        const in_peak: f32 = @bitCast(state.in_peak_bits.load(.monotonic));
        ui.statusLine(io, "capturing {d:>5.1}s / {d:.1}s   input peak {d:>6.1} dB", .{
            @as(f64, @floatFromInt(pos)) / 48000.0, @as(f64, @floatFromInt(total)) / 48000.0, ui.dbfs(in_peak),
        });
        std.Io.sleep(io, .{ .nanoseconds = 200 * std.time.ns_per_ms }, .awake) catch {};
    }
    audio.stop();
    ui.plainLine(io, "", .{});

    const final_peak: f32 = @bitCast(state.in_peak_bits.load(.monotonic));
    if (final_peak < 1e-4) {
        try stdout.writeAll("error: the capture channel recorded silence — check cabling and macOS microphone permission for this terminal\n");
        return error.SilentCapture;
    }

    try wav.writeMono(io, allocator, reamp_path.?, recorded, 48000, .float32);
    try stdout.print("saved reamp to {s}; training...\n", .{reamp_path.?});
    try stdout.flush();

    var full_args: std.ArrayList([]const u8) = .empty;
    defer full_args.deinit(allocator);
    try full_args.appendSlice(allocator, &.{ "--input", signal_path.?, "--output", reamp_path.? });
    try full_args.appendSlice(allocator, train_args.items);
    try train(io, allocator, stdout, full_args.items);
}

const LoopbackState = struct {
    recorded: []f32,
    clock: std.atomic.Value(usize) = .init(0),
    impulse_every: usize,
};

fn loopbackCallback(user: ?*anyopaque, output: ?[*]f32, input: ?[*]const f32, frame_count: c_uint) callconv(.c) void {
    const state: *LoopbackState = @ptrCast(@alignCast(user.?));
    const frames: usize = frame_count;
    const out = output orelse return;
    const in = input orelse return;
    const clock = state.clock.load(.monotonic);
    for (0..frames) |i| {
        const t = clock + i;
        out[i] = if (t % state.impulse_every == 0) 0.9 else 0.0;
        if (t < state.recorded.len) state.recorded[t] = in[i];
    }
    state.clock.store(clock + frames, .release);
}

/// Measures the software round-trip latency of the duplex stack by sending
/// an impulse every second and finding it again in the capture stream.
/// Run it on a loopback device (e.g. VB-Cable) so the physical path is ~0:
/// what remains is OUR stack (playback unit -> loopback -> capture unit ->
/// duplex ring -> callback) — the number to minimize.
fn loopbackTest(io: std.Io, allocator: std.mem.Allocator, stdout: *std.Io.Writer, args: []const []const u8) !void {
    var capture: ?usize = null;
    var playback: ?usize = null;
    var period: u32 = 64;
    var seconds: usize = 4;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--capture")) {
            capture = try std.fmt.parseInt(usize, try nextArg(args, &i), 10);
        } else if (std.mem.eql(u8, args[i], "--playback")) {
            playback = try std.fmt.parseInt(usize, try nextArg(args, &i), 10);
        } else if (std.mem.eql(u8, args[i], "--period")) {
            period = try std.fmt.parseInt(u32, try nextArg(args, &i), 10);
        } else if (std.mem.eql(u8, args[i], "--seconds")) {
            seconds = try std.fmt.parseInt(usize, try nextArg(args, &i), 10);
        } else return error.UnknownArgument;
    }
    // Bounded so `rate * seconds` can't overflow / over-allocate (a loopback
    // round-trip test is short).
    if (seconds == 0 or seconds > 600) {
        try stdout.writeAll("error: --seconds must be in 1..600\n");
        return error.InvalidSeconds;
    }

    const rate: usize = 48000;
    const recorded = try allocator.alloc(f32, rate * seconds);
    defer allocator.free(recorded);
    @memset(recorded, 0);
    var state = LoopbackState{ .recorded = recorded, .impulse_every = rate };

    var audio = try audio_mod.Audio.init();
    defer audio.deinit();
    try audio.start(capture, playback, @intCast(rate), period, loopbackCallback, &state);
    var waited: usize = 0;
    while (state.clock.load(.acquire) < recorded.len and waited < (seconds + 3) * 1000) : (waited += 100) {
        std.Io.sleep(io, .{ .nanoseconds = 100 * std.time.ns_per_ms }, .awake) catch {};
    }
    audio.stop();

    // Find each impulse arrival relative to its emission tick.
    var deltas: [64]usize = undefined;
    var count: usize = 0;
    var k: usize = 0;
    while ((k + 1) * rate <= recorded.len and count < deltas.len) : (k += 1) {
        const window = recorded[k * rate ..][0..rate];
        var best_idx: usize = 0;
        var best_mag: f32 = 0;
        for (window, 0..) |v, idx| {
            const mag = @abs(v);
            if (mag > best_mag) {
                best_mag = mag;
                best_idx = idx;
            }
        }
        if (best_mag > 0.2) {
            deltas[count] = best_idx;
            count += 1;
        }
    }
    if (count == 0) {
        try stdout.writeAll("no impulse came back — use a loopback device (e.g. VB-Cable) for --capture AND --playback\n");
        return;
    }
    std.mem.sort(usize, deltas[0..count], {}, std.sort.asc(usize));
    const median = deltas[count / 2];
    try stdout.print("round-trip software latency: {d} samples = {d:.2} ms @48 kHz (period {d} = {d:.2} ms; {d} impulses, min {d} max {d})\n", .{
        median,
        @as(f64, @floatFromInt(median)) / 48.0,
        period,
        @as(f64, @floatFromInt(period)) / 48.0,
        count,
        deltas[0],
        deltas[count - 1],
    });
}

/// Zero-argument guided mode for non-experts: discover profiles, pick by
/// number, and start playing with every decision pre-made (auto input
/// detection, same-device output, loudness normalization, noise gate).
fn interactive(io: std.Io, allocator: std.mem.Allocator, stdout: *std.Io.Writer) !void {
    try stdout.writeAll(
        \\
        \\  NAM — play your guitar through neural amp profiles
        \\
    );

    var paths: std.ArrayList([]const u8) = .empty;
    defer paths.deinit(allocator);
    const search_dirs = [_][]const u8{ "nam-profiles", "models", "." };
    for (search_dirs) |dir| try discoverProfiles(io, allocator, dir, 2, &paths);

    if (paths.items.len == 0) {
        try stdout.writeAll(
            \\
            \\  No amp profiles found yet.
            \\
            \\  Put .nam files in a folder named "nam-profiles" (or "models") next to
            \\  this program. Thousands of free profiles: https://www.tone3000.com
            \\  Then run this again.
            \\
        );
        return error.NoProfilesFound;
    }

    std.mem.sort([]const u8, paths.items, {}, struct {
        fn lessThan(_: void, a: []const u8, b: []const u8) bool {
            return std.ascii.lessThanIgnoreCase(std.fs.path.basename(a), std.fs.path.basename(b));
        }
    }.lessThan);

    // Same profile in both formats: keep one entry (prefer the .nam).
    var deduped: usize = 0;
    for (paths.items) |path| {
        const name = stripProfileExt(std.fs.path.basename(path));
        if (deduped > 0) {
            const previous = stripProfileExt(std.fs.path.basename(paths.items[deduped - 1]));
            if (std.ascii.eqlIgnoreCase(name, previous)) {
                if (std.ascii.endsWithIgnoreCase(path, ".nam")) paths.items[deduped - 1] = path;
                continue;
            }
        }
        paths.items[deduped] = path;
        deduped += 1;
    }
    paths.items.len = deduped;

    const shown = @min(paths.items.len, 30);
    try stdout.writeAll("\n  Which amp do you want to play?\n\n");
    for (paths.items[0..shown], 0..) |path, i| {
        try stdout.print("  [{d:>2}] {s}\n", .{ i + 1, stripProfileExt(std.fs.path.basename(path)) });
    }
    if (paths.items.len > shown) try stdout.print("  (+{d} more — use the `live` command for those)\n", .{paths.items.len - shown});
    try stdout.print("\n  Number [1-{d}], then Enter (just Enter = 1): ", .{shown});
    try stdout.flush();

    var stdin_buf: [4096]u8 = undefined;
    var stdin_reader = std.Io.File.stdin().reader(io, &stdin_buf);
    const line = (try stdin_reader.interface.takeDelimiter('\n')) orelse "";
    const trimmed = std.mem.trim(u8, line, " \t\r");
    var choice: usize = 1;
    if (trimmed.len > 0) {
        choice = std.fmt.parseInt(usize, trimmed, 10) catch 1;
        if (choice < 1 or choice > shown) choice = 1;
    }

    // Chosen profile first, then up to 8 more for instant switching.
    var live_args: std.ArrayList([]const u8) = .empty;
    defer live_args.deinit(allocator);
    try live_args.append(allocator, paths.items[choice - 1]);
    for (paths.items[0..shown], 0..) |path, i| {
        if (i + 1 == choice or live_args.items.len >= 9) continue;
        try live_args.append(allocator, path);
    }
    try live_args.appendSlice(allocator, &.{ "--auto-input", "--gate", "-65" });

    try stdout.writeAll(
        \\
        \\  Plug your guitar into the audio interface (instrument input) and KEEP
        \\  PLAYING — the right input is detected from your signal automatically.
        \\
        \\
    );
    try stdout.flush();
    try live(io, allocator, stdout, live_args.items);
}

fn stripProfileExt(name: []const u8) []const u8 {
    if (std.ascii.endsWithIgnoreCase(name, ".nam")) return name[0 .. name.len - 4];
    if (std.ascii.endsWithIgnoreCase(name, ".gguf")) return name[0 .. name.len - 5];
    return name;
}

fn isNamGguf(io: std.Io, allocator: std.mem.Allocator, path: []const u8) bool {
    var file = fucina.gguf.File.loadMmap(allocator, io, path) catch return false;
    defer file.deinit();
    return file.getString(gguf_compat.file_json_key) != null;
}

/// Collects .nam/.gguf files under `dir` up to `depth` levels deep.
fn discoverProfiles(io: std.Io, allocator: std.mem.Allocator, dir_path: []const u8, depth: usize, out: *std.ArrayList([]const u8)) !void {
    var dir = std.Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true }) catch return;
    defer dir.close(io);
    var it = dir.iterate();
    while (it.next(io) catch null) |entry| {
        if (entry.kind == .directory and depth > 1) {
            if (entry.name.len > 0 and entry.name[0] == '.') continue;
            if (std.mem.eql(u8, dir_path, ".") and (std.mem.eql(u8, entry.name, "nam-profiles") or std.mem.eql(u8, entry.name, "models"))) continue;
            if (std.mem.eql(u8, dir_path, ".") and !std.mem.eql(u8, entry.name, "models")) continue;
            const sub = try std.fs.path.join(allocator, &.{ dir_path, entry.name });
            try discoverProfiles(io, allocator, sub, depth - 1, out);
            continue;
        }
        if (entry.kind != .file) continue;
        const is_nam = std.ascii.endsWithIgnoreCase(entry.name, ".nam");
        const is_gguf = std.ascii.endsWithIgnoreCase(entry.name, ".gguf");
        if (!is_nam and !is_gguf) continue;
        const path = try std.fs.path.join(allocator, &.{ dir_path, entry.name });
        // .gguf is also the LLM weights format — only list NAM containers
        // (metadata-only mmap peek; tensor data is never touched).
        if (is_gguf and !isNamGguf(io, allocator, path)) continue;
        try out.append(allocator, path);
    }
}

fn devices(stdout: *std.Io.Writer, args: []const []const u8) !void {
    _ = args;
    var audio = try audio_mod.Audio.init();
    defer audio.deinit();
    var storage: [audio_mod.max_devices]audio_mod.DeviceInfo = undefined;

    try stdout.writeAll("capture devices:\n");
    for (try audio.listDevices(.capture, &storage), 0..) |*info, i| {
        try stdout.print("  [{d}] {s}{s}\n", .{ i, info.nameSlice(), if (info.is_default) " (default)" else "" });
    }
    try stdout.writeAll("playback devices:\n");
    for (try audio.listDevices(.playback, &storage), 0..) |*info, i| {
        try stdout.print("  [{d}] {s}{s}\n", .{ i, info.nameSlice(), if (info.is_default) " (default)" else "" });
    }

    try stdout.writeAll("midi sources:\n");
    var midi = midi_mod.Midi.init() catch {
        try stdout.writeAll("  (no MIDI backend on this platform)\n");
        return;
    };
    defer midi.deinit();
    var midi_storage: [midi_mod.max_sources]midi_mod.SourceInfo = undefined;
    const sources = midi.listSources(&midi_storage);
    for (sources, 0..) |*info, i| {
        try stdout.print("  [{d}] {s}\n", .{ i, info.nameSlice() });
    }
    if (sources.len == 0) try stdout.writeAll("  (none connected)\n");
}

fn live(io: std.Io, allocator: std.mem.Allocator, stdout: *std.Io.Writer, args: []const []const u8) !void {
    var options = live_mod.Options{};
    var profile_paths: [16][]const u8 = undefined;
    var profile_count: usize = 0;
    var chain_paths: [16][]const u8 = undefined;
    var chain_count: usize = 0;
    var ir_path: ?[]const u8 = null;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--capture")) {
            i += 1;
            if (i >= args.len) return error.MissingArgumentValue;
            options.capture = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--playback")) {
            i += 1;
            if (i >= args.len) return error.MissingArgumentValue;
            options.playback = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--rate")) {
            i += 1;
            if (i >= args.len) return error.MissingArgumentValue;
            options.sample_rate = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--period")) {
            i += 1;
            if (i >= args.len) return error.MissingArgumentValue;
            options.period = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--gain")) {
            i += 1;
            if (i >= args.len) return error.MissingArgumentValue;
            options.gain_db = try std.fmt.parseFloat(f32, args[i]);
        } else if (std.mem.eql(u8, arg, "--input-gain")) {
            i += 1;
            if (i >= args.len) return error.MissingArgumentValue;
            options.input_gain_db = try std.fmt.parseFloat(f32, args[i]);
        } else if (std.mem.eql(u8, arg, "--no-normalize")) {
            options.normalize = false;
        } else if (std.mem.eql(u8, arg, "--gate")) {
            i += 1;
            if (i >= args.len) return error.MissingArgumentValue;
            options.gate_db = try std.fmt.parseFloat(f32, args[i]);
        } else if (std.mem.eql(u8, arg, "--ir")) {
            i += 1;
            if (i >= args.len) return error.MissingArgumentValue;
            ir_path = args[i];
        } else if (std.mem.eql(u8, arg, "--chain")) {
            i += 1;
            if (i >= args.len) return error.MissingArgumentValue;
            if (chain_count >= chain_paths.len) return error.TooManyChains;
            chain_paths[chain_count] = args[i];
            chain_count += 1;
        } else if (std.mem.eql(u8, arg, "--auto-input")) {
            options.auto_input = true;
        } else if (std.mem.eql(u8, arg, "--tuner")) {
            options.tuner = true;
        } else if (std.mem.eql(u8, arg, "--a4")) {
            i += 1;
            if (i >= args.len) return error.MissingArgumentValue;
            options.a4 = try std.fmt.parseFloat(f64, args[i]);
            if (options.a4 < 400.0 or options.a4 > 480.0) return error.InvalidA4Reference;
        } else if (std.mem.eql(u8, arg, "--midi")) {
            i += 1;
            if (i >= args.len) return error.MissingArgumentValue;
            options.midi_source = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--no-midi")) {
            options.midi = false;
        } else if (std.mem.eql(u8, arg, "--midi-channel")) {
            i += 1;
            if (i >= args.len) return error.MissingArgumentValue;
            const channel = try std.fmt.parseInt(u8, args[i], 10);
            if (channel < 1 or channel > 16) return error.InvalidMidiChannel;
            options.midi_channel = channel;
        } else if (std.mem.eql(u8, arg, "--midi-map")) {
            i += 1;
            if (i >= args.len) return error.MissingArgumentValue;
            options.cc_map = try midi_mod.CcMap.parse(args[i]);
        } else if (std.mem.startsWith(u8, arg, "--")) {
            return error.UnknownArgument;
        } else {
            if (profile_count >= profile_paths.len) return error.TooManyProfiles;
            profile_paths[profile_count] = arg;
            profile_count += 1;
        }
    }
    if (profile_count == 0 and chain_count == 0) {
        try stdout.writeAll("usage: zig build nam -Doptimize=ReleaseFast -- live <profile.nam> [more.nam ...] [--ir cab.wav] [--chain rig.chain ...] [--capture N] [--playback N] [--rate 48000] [--period 128] [--tuner] [--a4 440] [--midi N | --no-midi] [--midi-channel 1-16] [--midi-map out-gain=7,in-gain=11,gate-threshold=1,bypass=64,gate=80,normalize=81,mute=85]\n");
        return;
    }

    // Preallocate the audio-thread scratch (the engines/cabs are sized to
    // frame_cap too): input trim buffer + gate gains + two inter-stage ping
    // buffers for chains with >1 stage. With noFixedSizedCallback the device
    // delivers its own block sizes, which may exceed the requested period.
    const frame_cap = @max(@as(usize, 2048), @as(usize, options.period) * 4);
    const scratch_in = try allocator.alloc(f32, frame_cap);
    defer allocator.free(scratch_in);
    const gate_gains = try allocator.alloc(f32, frame_cap);
    defer allocator.free(gate_gains);
    const ping0 = try allocator.alloc(f32, frame_cap);
    defer allocator.free(ping0);
    const ping1 = try allocator.alloc(f32, frame_cap);
    defer allocator.free(ping1);

    // --ir only appends a cab to bare profiles; with manifests-only it does
    // nothing (put the cab in the manifest instead).
    if (ir_path != null and profile_count == 0) {
        try stdout.writeAll("note: --ir applies to bare profiles; with only --chain manifests it is ignored — add the cab as a stage in the manifest\n");
    }

    // Build every chain (bare profiles become 1-stage chains, +cab when --ir;
    // then explicit --chain manifests). Preloaded + prewarmed up front so
    // switching is one atomic index store.
    var set = try buildChains(io, allocator, stdout, profile_paths[0..profile_count], ir_path, chain_paths[0..chain_count], options.sample_rate, frame_cap);
    defer set.deinit();
    try stdout.flush();

    var shared = live_mod.Shared{
        .chains = set.chains,
        .period = frame_cap,
        .scratch_in = scratch_in,
        .gate_gains = gate_gains,
        .ping = .{ ping0, ping1 },
    };
    shared.setGain(std.math.pow(f32, 10.0, options.gain_db / 20.0));
    shared.setInputGain(std.math.pow(f32, 10.0, options.input_gain_db / 20.0));
    shared.normalize.store(options.normalize, .monotonic);

    var audio = try audio_mod.Audio.init();
    defer audio.deinit();
    try live_mod.run(io, allocator, &shared, &audio, options);
}

const BuiltStage = struct { cs: live_mod.ChainStage, norm_gain: f32 };

/// Loads one stage instance (NAM model or cab IR) for `spec`, sized to
/// `frame_cap` and prewarmed. The instance is heap-allocated for a stable
/// pointer in the chain; on error it is fully cleaned up. `norm_gain` is the
/// NAM stage's loudness comp to -18 dBFS (1.0 for a cab).
fn buildStage(io: std.Io, allocator: std.mem.Allocator, stdout: *std.Io.Writer, spec: chain_mod.StageSpec, sample_rate: u32, frame_cap: usize) !BuiltStage {
    const trim = std.math.pow(f32, 10.0, spec.trim_db / 20.0);
    switch (spec.kind) {
        .nam => {
            var model = try loadModel(io, allocator, stdout, spec.path);
            defer model.deinit();
            if (model.sample_rate > 0 and model.sample_rate != @as(f64, @floatFromInt(sample_rate))) {
                try stdout.print("error: {s} expects {d} Hz but the stream is {d} Hz (no resampling; pass --rate)\n", .{ spec.path, model.sample_rate, sample_rate });
                return error.SampleRateMismatch;
            }
            const engine = try allocator.create(engine_mod.Engine);
            errdefer allocator.destroy(engine);
            engine.* = try engine_mod.Engine.init(allocator, &model);
            errdefer engine.deinit();
            try engine.reset(frame_cap, true);
            // Player-style loudness normalization to the -18 dBFS target (capped
            // so bogus metadata can't blast the output). The engine itself never
            // applies loudness, same as the upstream core.
            var norm_gain: f32 = 1.0;
            if (model.metadata.loudness) |loudness| {
                const boost_db = std.math.clamp(-18.0 - loudness, -40.0, 20.0);
                norm_gain = std.math.pow(f32, 10.0, @as(f32, @floatCast(boost_db)) / 20.0);
            }
            const gear = classifyGearModel(&model);
            return .{ .cs = .{ .stage = .{ .nam = .{ .engine = engine, .gear = gear } }, .in_trim = trim }, .norm_gain = norm_gain };
        },
        .cab => {
            const cab = try allocator.create(ir_cab.IrCab);
            errdefer allocator.destroy(cab);
            cab.* = ir_cab.IrCab.loadFile(io, allocator, spec.path, sample_rate, frame_cap) catch |err| {
                try stdout.print("error: could not load cab IR {s}: {s}\n", .{ spec.path, @errorName(err) });
                return err;
            };
            errdefer cab.deinit();
            try stdout.print("cab IR: {s} ({d} taps @ {d} Hz)\n", .{ spec.path, cab.taps, sample_rate });
            return .{ .cs = .{ .stage = .{ .cab = cab }, .in_trim = trim }, .norm_gain = 1.0 };
        },
    }
}

/// Assembles all chains: bare profiles first (each a 1-stage chain, plus a cab
/// stage when --ir is given), then explicit --chain manifests. Every stage is
/// duplicate-loaded into its own instance (single-owner — see live.zig). On any
/// failure everything built so far is freed.
fn buildChains(io: std.Io, allocator: std.mem.Allocator, stdout: *std.Io.Writer, profile_paths: []const []const u8, ir_path: ?[]const u8, chain_paths: []const []const u8, sample_rate: u32, frame_cap: usize) !live_mod.ChainSet {
    var arena_inst = std.heap.ArenaAllocator.init(allocator);
    errdefer arena_inst.deinit();
    const arena = arena_inst.allocator();

    const total = profile_paths.len + chain_paths.len;
    const chains = try allocator.alloc(live_mod.Chain, total);
    errdefer allocator.free(chains);

    var built: usize = 0;
    var pending: std.ArrayList(live_mod.ChainStage) = .empty;
    // Single-owner cleanup of the temporary buffer: frees exactly once on every
    // exit (including if the success-path print below fails). Registered before
    // the errdefer so on error the destroyStage loops (which read pending.items)
    // run first, then this frees the buffer.
    defer pending.deinit(allocator);
    errdefer {
        for (pending.items) |*cs| live_mod.destroyStage(allocator, cs);
        for (chains[0..built]) |*c| for (c.stages) |*cs| live_mod.destroyStage(allocator, cs);
    }

    // (1) bare profiles -> 1-stage chains (+ optional --ir cab stage).
    for (profile_paths) |ppath| {
        const r = try buildStage(io, allocator, stdout, .{ .path = ppath, .kind = .nam }, sample_rate, frame_cap);
        try pending.append(allocator, r.cs);
        if (ir_path) |irp| {
            const c = try buildStage(io, allocator, stdout, .{ .path = irp, .kind = .cab }, sample_rate, frame_cap);
            try pending.append(allocator, c.cs);
        }
        const cname = std.fs.path.basename(ppath);
        live_mod.adviseChain(stdout, cname, pending.items) catch {};
        chains[built] = .{
            .name = try arena.dupe(u8, cname),
            .stages = try arena.dupe(live_mod.ChainStage, pending.items),
            .norm_gain = r.norm_gain,
        };
        pending.clearRetainingCapacity(); // ownership moved into chains[built]
        built += 1;
    }

    // (2) explicit --chain manifests.
    for (chain_paths) |cpath| {
        const text = try wav.readFileBytes(io, allocator, cpath);
        defer allocator.free(text);
        var spec = chain_mod.parse(allocator, text) catch |err| {
            try stdout.print("error: bad chain manifest {s}: {s}\n", .{ cpath, @errorName(err) });
            return err;
        };
        defer spec.deinit();
        var norm: f32 = 1.0;
        for (spec.stages) |sspec| {
            const r = try buildStage(io, allocator, stdout, sspec, sample_rate, frame_cap);
            if (sspec.kind == .nam) norm = r.norm_gain; // last NAM stage wins
            try pending.append(allocator, r.cs);
        }
        const cname = spec.name orelse std.fs.path.stem(std.fs.path.basename(cpath));
        live_mod.adviseChain(stdout, cname, pending.items) catch {};
        chains[built] = .{
            .name = try arena.dupe(u8, cname),
            .stages = try arena.dupe(live_mod.ChainStage, pending.items),
            .norm_gain = norm,
        };
        pending.clearRetainingCapacity();
        built += 1;
    }

    try stdout.print("loaded {d} chain(s)\n", .{total});
    return .{ .allocator = allocator, .arena = arena_inst, .chains = chains };
}

/// gear_type-based cab classification + a Tone3000 "full rig" name/model hint.
fn classifyGearModel(model: *const nam_file.NamModel) live_mod.GearClass {
    var hint = false;
    const doc = model.document();
    if (doc == .object) {
        if (doc.object.get("metadata")) |m| {
            if (m == .object) {
                for ([_][]const u8{ "name", "gear_model", "tone_type" }) |k| {
                    if (m.object.get(k)) |v| {
                        if (v == .string and hasFullRig(v.string)) hint = true;
                    }
                }
            }
        }
    }
    return live_mod.classifyGear(model.metadata.gear_type, hint);
}

fn hasFullRig(s: []const u8) bool {
    if (std.ascii.indexOfIgnoreCase(s, "full-rig") != null) return true;
    if (std.ascii.indexOfIgnoreCase(s, "full rig") != null) return true;
    return std.ascii.indexOfIgnoreCase(s, "full") != null and std.ascii.indexOfIgnoreCase(s, "rig") != null;
}

const rng = fucina.rng;

test {
    _ = @import("nam/wav.zig");
    _ = @import("nam/nam_file.zig");
    _ = @import("nam/activations.zig");
    _ = @import("nam/stream_conv.zig");
    _ = @import("nam/wavenet.zig");
    _ = @import("nam/models.zig");
    _ = @import("nam/engine.zig");
    _ = @import("nam/data.zig");
    _ = @import("nam/train.zig");
    _ = @import("nam/nam_export.zig");
    _ = @import("nam/gguf_compat.zig");
    _ = @import("nam/ir_cab.zig");
    _ = @import("nam/chain.zig");
    _ = @import("nam/live.zig");
    _ = @import("nam/midi.zig");
    _ = @import("nam/tuner.zig");
    _ = @import("nam/ui.zig");
}
