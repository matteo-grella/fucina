//! Canonical `.nam` writer for Fucina-trained WaveNets. Emits the modern
//! upstream exporter shape key-for-key (neural-amp-modeler@a11ed88
//! nam/models/exportable.py:187-194 + wavenet/_layer_array.py:1003-1025),
//! which both NeuralAmpModelerCore and the current Python `init_from_nam`
//! re-importer accept. Top-level key order: version, metadata, architecture,
//! config, weights, sample_rate. Loudness/gain metadata are measured exactly
//! like upstream (base.py:121-156) on the vendored standardized 1 s signal.

const std = @import("std");
const nam_file = @import("nam_file.zig");
const wav = @import("wav.zig");
const train = @import("train.zig");
const data = @import("data.zig");

pub const UserMetadata = struct {
    name: ?[]const u8 = null,
    modeled_by: ?[]const u8 = null,
    gear_type: ?[]const u8 = null, // amp|pedal|pedal_amp|amp_cab|amp_pedal_cab|preamp|studio
    gear_make: ?[]const u8 = null,
    gear_model: ?[]const u8 = null,
    tone_type: ?[]const u8 = null, // clean|overdrive|crunch|hi_gain|fuzz
};

pub const TrainingMetadata = struct {
    ignore_checks: bool = false,
    latency_manual: ?i64 = null,
    calibration: ?data.LatencyCalibration = null,
    checks_version: u32 = 3,
    checks_passed: bool = true,
    validation_esr: ?f64 = null,
};

pub const ExportInfo = struct {
    user: UserMetadata = .{},
    training: ?TrainingMetadata = null,
    /// Unix seconds for the date stamp (UTC).
    unix_seconds: u64,
    sample_rate: f64 = 48000.0,
    /// Inverse of the output scale used during training. Upstream's
    /// Dataset._ScaleOutputHook applies this to WaveNet head_scale and the
    /// duplicated final weight so exported models predict the original target
    /// level, not the normalized training target.
    output_scale_compensation: f32 = 1.0,
};

pub const LoudnessGain = struct {
    loudness: f64,
    gain: f64,
};

/// Measures upstream's loudness + gain metadata (base.py:121-156) by
/// streaming the trained weights over the vendored standardized input.
pub fn measureLoudnessAndGain(allocator: std.mem.Allocator, spec: *const train.ModelSpec, weights: []const f32) !LoudnessGain {
    var config = try train.toEngineConfig(allocator, spec);
    defer train.freeEngineConfig(allocator, &config);
    return measureLoudnessAndGainConfig(allocator, &config, weights);
}

pub fn measureLoudnessAndGainConfig(
    allocator: std.mem.Allocator,
    config: *const nam_file.WaveNetConfig,
    weights: []const f32,
) !LoudnessGain {
    var signal = try wav.parse(allocator, @embedFile("resources/loudness_input.wav"));
    defer signal.deinit();
    const x = try signal.requireMono();

    const scaled = try allocator.alloc(f32, x.len);
    defer allocator.free(scaled);
    const out = try allocator.alloc(f32, x.len);
    defer allocator.free(out);

    // gain sweep: rms(model(g*x)) for g = 0, 0.1, ..., 1.0.
    var levels: [11]f64 = undefined;
    for (&levels, 0..) |*level, i| {
        const g = @as(f32, @floatFromInt(i)) / 10.0;
        for (scaled, x) |*dst, v| dst.* = g * v;
        try train.renderWaveNetConfig(allocator, config, weights, scaled, out);
        var sum_sq: f64 = 0;
        for (out) |v| sum_sq += @as(f64, v) * v;
        level.* = @sqrt(sum_sq / @as(f64, @floatFromInt(out.len)));
    }
    const loudness = 20.0 * std.math.log10(@max(levels[10], 1e-20));
    var total: f64 = 0;
    for (levels) |v| total += v;
    const max_gain = levels[10] * 11.0;
    const min_gain = 0.5 * max_gain;
    const gain = if (max_gain > min_gain)
        std.math.clamp((total - min_gain) / (max_gain - min_gain), 0.0, 1.0)
    else
        0.0;
    return .{ .loudness = loudness, .gain = gain };
}

pub fn exportWaveNet(
    io: std.Io,
    allocator: std.mem.Allocator,
    path: []const u8,
    spec: *const train.ModelSpec,
    weights: []const f32,
    info: ExportInfo,
) !void {
    var config = try train.toEngineConfig(allocator, spec);
    defer train.freeEngineConfig(allocator, &config);
    try exportWaveNetConfig(io, allocator, path, &config, weights, info);
}

pub fn exportWaveNetConfig(
    io: std.Io,
    allocator: std.mem.Allocator,
    path: []const u8,
    config: *const nam_file.WaveNetConfig,
    weights: []const f32,
    info: ExportInfo,
) !void {
    var export_view = try compensatedWaveNet(allocator, config, weights, info.output_scale_compensation);
    defer export_view.deinit(allocator);
    const measured = try measureLoudnessAndGainConfig(allocator, &export_view.config, export_view.weights);

    var file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buffer: [64 * 1024]u8 = undefined;
    var writer = file.writer(io, &buffer);
    const w = &writer.interface;

    try writeWaveNetDocument(w, &export_view.config, export_view.weights, info, measured);
    try w.flush();
}

pub fn exportSlimmableContainer(
    io: std.Io,
    allocator: std.mem.Allocator,
    path: []const u8,
    submodels: []const train.WaveNetSnapshot,
    info: ExportInfo,
) !void {
    if (submodels.len == 0) return error.InvalidPackedSnapshot;
    const export_views = try allocator.alloc(CompensatedWaveNet, submodels.len);
    defer allocator.free(export_views);
    var views_built: usize = 0;
    defer {
        for (export_views[0..views_built]) |*view| view.deinit(allocator);
    }
    const measured = try allocator.alloc(LoudnessGain, submodels.len);
    defer allocator.free(measured);
    for (submodels, export_views, measured) |*submodel, *view, *dst| {
        view.* = try compensatedWaveNet(allocator, &submodel.config, submodel.weights, info.output_scale_compensation);
        views_built += 1;
        dst.* = try measureLoudnessAndGainConfig(allocator, &view.config, view.weights);
    }

    var file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buffer: [64 * 1024]u8 = undefined;
    var writer = file.writer(io, &buffer);
    const w = &writer.interface;

    try w.writeAll("{\"version\": \"0.7.0\", \"metadata\": ");
    try writeMetadata(w, info, measured[submodels.len - 1]);
    try w.writeAll(", \"architecture\": \"SlimmableContainer\", \"config\": {\"submodels\": [");
    const nested_info = ExportInfo{ .unix_seconds = info.unix_seconds, .sample_rate = info.sample_rate };
    for (export_views, measured, 0..) |*submodel, sub_measured, i| {
        if (i != 0) try w.writeAll(", ");
        const max_value = if (i + 1 == export_views.len)
            1.0
        else
            @as(f64, @floatFromInt(i + 1)) / @as(f64, @floatFromInt(export_views.len));
        try w.print("{{\"max_value\": {d}, \"model\": ", .{max_value});
        try writeWaveNetDocument(w, &submodel.config, submodel.weights, nested_info, sub_measured);
        try w.writeAll("}");
    }
    try w.print("]}}, \"weights\": [], \"sample_rate\": {d}}}", .{info.sample_rate});
    try w.flush();
}

const CompensatedWaveNet = struct {
    config: nam_file.WaveNetConfig,
    weights: []f32,

    fn deinit(self: *CompensatedWaveNet, allocator: std.mem.Allocator) void {
        allocator.free(self.weights);
        self.* = undefined;
    }
};

fn compensatedWaveNet(
    allocator: std.mem.Allocator,
    config: *const nam_file.WaveNetConfig,
    weights: []const f32,
    scale: f32,
) !CompensatedWaveNet {
    if (weights.len == 0) return error.WeightCountMismatch;
    const copied = try allocator.dupe(f32, weights);
    errdefer allocator.free(copied);
    var export_config = config.*;
    export_config.head_scale *= scale;
    copied[copied.len - 1] *= scale;
    return .{ .config = export_config, .weights = copied };
}

fn writeWaveNetDocument(
    w: *std.Io.Writer,
    config: *const nam_file.WaveNetConfig,
    weights: []const f32,
    info: ExportInfo,
    measured: LoudnessGain,
) !void {
    try w.writeAll("{\"version\": \"0.7.0\", \"metadata\": ");
    try writeMetadata(w, info, measured);
    try w.writeAll(", \"architecture\": \"WaveNet\", \"config\": ");
    try writeWaveNetConfig(w, config);
    try w.writeAll(", \"weights\": ");
    try writeWeights(w, weights);
    try w.print(", \"sample_rate\": {d}}}", .{info.sample_rate});
}

fn writeMetadata(w: *std.Io.Writer, info: ExportInfo, measured: LoudnessGain) !void {
    try w.writeAll("{");
    try writeDate(w, info.unix_seconds);
    try w.print(", \"loudness\": {d}, \"gain\": {d}", .{ measured.loudness, measured.gain });
    try writeOptionalString(w, "name", info.user.name);
    try writeOptionalString(w, "modeled_by", info.user.modeled_by);
    try writeOptionalString(w, "gear_type", info.user.gear_type);
    try writeOptionalString(w, "gear_make", info.user.gear_make);
    try writeOptionalString(w, "gear_model", info.user.gear_model);
    try writeOptionalString(w, "tone_type", info.user.tone_type);
    try w.writeAll(", \"input_level_dbu\": null, \"output_level_dbu\": null");
    if (info.training) |t| {
        try w.print(", \"training\": {{\"settings\": {{\"ignore_checks\": {}}}, \"data\": {{\"latency\": {{\"manual\": ", .{t.ignore_checks});
        if (t.latency_manual) |v| try w.print("{d}", .{v}) else try w.writeAll("null");
        try w.writeAll(", \"calibration\": ");
        if (t.calibration) |c| {
            try w.writeAll("{\"algorithm_version\": 1, \"delays\": [");
            if (c.delay) |d| try w.print("{d}", .{d});
            try w.writeAll("], \"safety_factor\": 1, \"recommended\": ");
            if (c.recommended) |r| try w.print("{d}", .{r}) else try w.writeAll("null");
            try w.print(", \"warnings\": {{\"matches_lookahead\": {}, \"disagreement_too_high\": {}, \"not_detected\": {}}}}}", .{ c.warn_matches_lookahead, c.warn_disagreement_too_high, c.warn_not_detected });
        } else {
            try w.writeAll("null");
        }
        try w.print("}}, \"checks\": {{\"version\": {d}, \"passed\": {}}}}}, \"validation_esr\": ", .{ t.checks_version, t.checks_passed });
        if (t.validation_esr) |v| try w.print("{d}", .{v}) else try w.writeAll("null");
        try w.writeAll("}");
    }
    try w.writeAll("}");
}

fn writeWaveNetConfig(w: *std.Io.Writer, config: *const nam_file.WaveNetConfig) anyerror!void {
    try w.writeAll("{");
    if (config.condition_dsp) |dsp| {
        try w.writeAll("\"condition_dsp\": ");
        try writeConditionDsp(w, dsp);
        try w.writeAll(", ");
    }
    try w.writeAll("\"layers\": [");
    for (config.layers, 0..) |*layer, layer_index| {
        if (layer_index != 0) try w.writeAll(", ");
        try w.print("{{\"input_size\": {d}, \"condition_size\": {d}, \"head\": {{\"out_channels\": {d}, \"kernel_size\": {d}, \"bias\": {}}}, \"channels\": {d}, \"kernel_sizes\": [", .{
            layer.input_size,
            layer.condition_size,
            layer.head_out,
            layer.head_kernel,
            layer.head_bias,
            layer.channels,
        });
        for (layer.kernel_sizes, 0..) |k, i| {
            if (i != 0) try w.writeAll(", ");
            try w.print("{d}", .{k});
        }
        try w.writeAll("], \"dilations\": [");
        for (layer.dilations, 0..) |d, i| {
            if (i != 0) try w.writeAll(", ");
            try w.print("{d}", .{d});
        }
        try w.writeAll("], \"activation\": [");
        for (layer.activations, 0..) |*activation, i| {
            if (i != 0) try w.writeAll(", ");
            try writeActivation(w, activation);
        }
        try w.print("], \"bottleneck\": {d}, \"head1x1\": {{\"active\": {}, \"out_channels\": {d}, \"groups\": {d}}}, \"layer1x1\": {{\"active\": {}, \"groups\": {d}}}, \"groups_input\": {d}, \"groups_input_mixin\": {d}", .{
            layer.bottleneck,
            layer.head1x1_active,
            layer.head1x1_out,
            layer.head1x1_groups,
            layer.layer1x1_active,
            layer.layer1x1_groups,
            layer.groups_input,
            layer.groups_input_mixin,
        });
        try writeFilm(w, "conv_pre_film", layer.conv_pre_film);
        try writeFilm(w, "conv_post_film", layer.conv_post_film);
        try writeFilm(w, "input_mixin_pre_film", layer.input_mixin_pre_film);
        try writeFilm(w, "input_mixin_post_film", layer.input_mixin_post_film);
        try writeFilm(w, "activation_pre_film", layer.activation_pre_film);
        try writeFilm(w, "activation_post_film", layer.activation_post_film);
        try writeFilm(w, "layer1x1_post_film", layer.layer1x1_post_film);
        try writeFilm(w, "head1x1_post_film", layer.head1x1_post_film);
        try w.writeAll(", \"gating_mode\": [");
        for (layer.gating_modes, 0..) |mode, i| {
            if (i != 0) try w.writeAll(", ");
            try w.print("\"{s}\"", .{@tagName(mode)});
        }
        try w.writeAll("], \"secondary_activation\": [");
        for (layer.secondary_activations, layer.gating_modes, 0..) |*activation, mode, i| {
            if (i != 0) try w.writeAll(", ");
            if (mode == .none) try w.writeAll("null") else try writeActivation(w, activation);
        }
        try w.writeAll("], \"slimmable\": null}");
    }
    try w.writeAll("], \"head\": ");
    if (config.head) |*head| {
        try w.writeAll("{\"channels\": ");
        try w.print("{d}, \"out_channels\": {d}, \"kernel_sizes\": [", .{ head.channels, head.out_channels });
        for (head.kernel_sizes, 0..) |k, i| {
            if (i != 0) try w.writeAll(", ");
            try w.print("{d}", .{k});
        }
        try w.writeAll("], \"activation\": ");
        try writeActivation(w, &head.activation);
        try w.writeAll("}");
    } else {
        try w.writeAll("null");
    }
    try w.print(", \"head_scale\": {d}", .{config.head_scale});
    if (config.in_channels != 1) try w.print(", \"in_channels\": {d}", .{config.in_channels});
    try w.writeAll("}");
}

fn writeConditionDsp(w: *std.Io.Writer, dsp: *const nam_file.ConditionDsp) anyerror!void {
    switch (dsp.config) {
        .wavenet => |*config| {
            try w.writeAll("{\"version\": \"0.7.0\", \"architecture\": \"WaveNet\", \"config\": ");
            try writeWaveNetConfig(w, config);
            try w.writeAll(", \"weights\": ");
            try writeWeights(w, dsp.weights);
            if (dsp.sample_rate >= 0) try w.print(", \"sample_rate\": {d}", .{dsp.sample_rate});
            try w.writeAll("}");
        },
        .lstm, .convnet, .linear => return error.UnsupportedFeature,
    }
}

fn writeWeights(w: *std.Io.Writer, weights: []const f32) !void {
    try w.writeAll("[");
    for (weights, 0..) |value, i| {
        if (i != 0) try w.writeAll(", ");
        try w.print("{d}", .{@as(f64, value)});
    }
    try w.writeAll("]");
}

fn writeActivation(w: *std.Io.Writer, activation: *const nam_file.Activation) !void {
    try w.writeAll("{\"type\": \"");
    try w.writeAll(switch (activation.kind) {
        .tanh => "Tanh",
        .hardtanh => "Hardtanh",
        .fasttanh => "Fasttanh",
        .relu => "ReLU",
        .leaky_relu => "LeakyReLU",
        .prelu => "PReLU",
        .sigmoid => "Sigmoid",
        .silu => "SiLU",
        .hardswish => "Hardswish",
        .leaky_hardtanh => "LeakyHardtanh",
        .softsign => "Softsign",
    });
    try w.writeAll("\"");
    switch (activation.kind) {
        .leaky_relu => try w.print(", \"negative_slope\": {d}", .{activation.negative_slope}),
        .prelu => {
            if (activation.negative_slopes.len > 0) {
                try w.writeAll(", \"negative_slopes\": [");
                for (activation.negative_slopes, 0..) |slope, i| {
                    if (i != 0) try w.writeAll(", ");
                    try w.print("{d}", .{slope});
                }
                try w.writeAll("]");
            } else {
                try w.print(", \"negative_slope\": {d}", .{activation.negative_slope});
            }
        },
        .leaky_hardtanh => try w.print(", \"min_val\": {d}, \"max_val\": {d}, \"min_slope\": {d}, \"max_slope\": {d}", .{
            activation.min_val,
            activation.max_val,
            activation.min_slope,
            activation.max_slope,
        }),
        else => {},
    }
    try w.writeAll("}");
}

fn writeFilm(w: *std.Io.Writer, key: []const u8, film: nam_file.FiLMParams) !void {
    try w.print(", \"{s}\": {{\"active\": {}, \"shift\": {}, \"groups\": {d}}}", .{ key, film.active, film.shift, film.groups });
}

fn writeDate(w: *std.Io.Writer, unix_seconds: u64) !void {
    const epoch = std.time.epoch.EpochSeconds{ .secs = unix_seconds };
    const day = epoch.getEpochDay();
    const year_day = day.calculateYearDay();
    const month_day = year_day.calculateMonthDay();
    const day_seconds = epoch.getDaySeconds();
    try w.print("\"date\": {{\"year\": {d}, \"month\": {d}, \"day\": {d}, \"hour\": {d}, \"minute\": {d}, \"second\": {d}}}", .{
        year_day.year,
        month_day.month.numeric(),
        month_day.day_index + 1,
        day_seconds.getHoursIntoDay(),
        day_seconds.getMinutesIntoHour(),
        day_seconds.getSecondsIntoMinute(),
    });
}

fn writeOptionalString(w: *std.Io.Writer, key: []const u8, value: ?[]const u8) !void {
    try w.print(", \"{s}\": ", .{key});
    if (value) |v| {
        try w.writeAll("\"");
        for (v) |ch| {
            switch (ch) {
                '"' => try w.writeAll("\\\""),
                '\\' => try w.writeAll("\\\\"),
                else => if (ch >= 0x20) try w.writeByte(ch) else {},
            }
        }
        try w.writeAll("\"");
    } else {
        try w.writeAll("null");
    }
}

test "WaveNet export compensation scales head scale and duplicated tail weight" {
    const allocator = std.testing.allocator;
    var config = try train.toEngineConfig(allocator, &train.ModelSpec.tiny);
    defer train.freeEngineConfig(allocator, &config);
    const expected = nam_file.expectedWeightCount(&.{ .wavenet = config });
    const weights = try allocator.alloc(f32, expected);
    defer allocator.free(weights);
    for (weights, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i + 1)) * 0.01;
    weights[weights.len - 1] = config.head_scale;

    const scale: f32 = 2.5;
    var view = try compensatedWaveNet(allocator, &config, weights, scale);
    defer view.deinit(allocator);

    try std.testing.expectApproxEqAbs(@as(f32, 0.02), config.head_scale, 1e-9);
    try std.testing.expectApproxEqAbs(config.head_scale * scale, view.config.head_scale, 1e-9);
    try std.testing.expectApproxEqAbs(config.head_scale, weights[weights.len - 1], 1e-9);
    try std.testing.expectApproxEqAbs(config.head_scale * scale, view.weights[view.weights.len - 1], 1e-9);
    try std.testing.expectApproxEqAbs(weights[weights.len - 2], view.weights[view.weights.len - 2], 1e-9);
}

test {
    _ = @import("nam_export_tests.zig");
}
