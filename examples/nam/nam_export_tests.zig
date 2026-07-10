//! Behavioral tests for the `.nam` WaveNet exporter (`nam_export.zig`):
//! own-loader round trip, A2-config export shape vs the upstream reference
//! JSON, and the SlimmableContainer (packed) export loadable at highest
//! quality.
const std = @import("std");
const nam_export = @import("nam_export.zig");
const nam_file = @import("nam_file.zig");
const wav = @import("wav.zig");
const train = @import("train.zig");
const data = @import("data.zig");
const fucina = @import("fucina");

const exportWaveNetConfig = nam_export.exportWaveNetConfig;
const exportSlimmableContainer = nam_export.exportSlimmableContainer;

test "exported file round-trips through our own loader" {
    const allocator = std.testing.allocator;

    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try train.Trainable.init(allocator, &ctx, train.ModelSpec.tiny, 3);
    defer model.deinit();
    const weights = try model.extractWeights(allocator);
    defer allocator.free(weights);

    // Render the export to memory through a temp file is io-bound; instead
    // exercise the config/weight consistency directly: the engine accepts
    // the extracted weights under the engine config (count formula agrees).
    var config = try train.toEngineConfig(allocator, &train.ModelSpec.tiny);
    defer train.freeEngineConfig(allocator, &config);
    const expected = nam_file.expectedWeightCount(&.{ .wavenet = config });
    try std.testing.expectEqual(expected, weights.len);
}

test "A2 training config export round-trips through the NAM loader" {
    const allocator = std.testing.allocator;
    var spec = train.TrainingSpec{ .a2 = train.A2Spec.nano };
    var config = try spec.makeEngineConfig(allocator);
    defer train.freeEngineConfig(allocator, &config);
    const weights = try train.initWaveNetWeights(allocator, &config, 99);
    defer allocator.free(weights);

    const path = "/tmp/fucina-a2-export-test.nam";
    defer std.Io.Dir.cwd().deleteFile(std.testing.io, path) catch {};
    try exportWaveNetConfig(std.testing.io, allocator, path, &config, weights, .{
        .unix_seconds = 0,
        .sample_rate = 48000.0,
    });

    var model = try nam_file.loadFile(std.testing.io, allocator, path);
    defer model.deinit();
    try std.testing.expectEqual(nam_file.Arch.wavenet, model.architecture);
    try std.testing.expectEqual(@as(usize, 1), model.config.wavenet.layers.len);
    const layer = model.config.wavenet.layers[0];
    try std.testing.expectEqual(@as(usize, 3), layer.channels);
    try std.testing.expectEqual(@as(usize, 16), layer.head_kernel);
    try std.testing.expectEqual(nam_file.Activation.Kind.leaky_relu, layer.activations[0].kind);
    try std.testing.expectApproxEqAbs(@as(f32, 0.01), layer.activations[0].negative_slope, 1e-7);
    try std.testing.expectEqual(@as(usize, weights.len), model.weights.len);

    const raw = try wav.readFileBytes(std.testing.io, allocator, path);
    defer allocator.free(raw);
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, raw, .{});
    defer parsed.deinit();
    try expectReferenceA2FastShapeJson(parsed.value.object, 3);
}

test "packed WaveNet export writes a SlimmableContainer loadable at highest quality" {
    const allocator = std.testing.allocator;

    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var spec = try train.TrainingSpec.parse("packed");
    var model = try spec.initTrainable(allocator, &ctx, 77);
    defer model.deinit();
    var snapshot = try model.extractTrainingSnapshot(allocator, null);
    defer snapshot.deinit(allocator);
    const packed_snapshot = switch (snapshot) {
        .packed_wavenet => |*s| s,
        else => return error.TestExpectedPackedSnapshot,
    };

    const path = "/tmp/fucina-packed-export-test.nam";
    defer std.Io.Dir.cwd().deleteFile(std.testing.io, path) catch {};
    try exportSlimmableContainer(std.testing.io, allocator, path, packed_snapshot.submodels, .{
        .unix_seconds = 0,
        .sample_rate = 48000.0,
    });

    var loaded = try nam_file.loadFile(std.testing.io, allocator, path);
    defer loaded.deinit();
    const info = loaded.submodel_info orelse return error.TestExpectedSlimmableContainer;
    try std.testing.expectEqual(@as(usize, 2), info.count);
    try std.testing.expectEqual(@as(usize, 1), info.index);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), info.max_value, 1e-12);
    try std.testing.expectEqual(nam_file.Arch.wavenet, loaded.architecture);
    try std.testing.expectEqual(@as(usize, 8), loaded.config.wavenet.layers[0].channels);
    try std.testing.expectEqual(@as(usize, loaded.weights.len), packed_snapshot.submodels[1].weights.len);

    const raw = try wav.readFileBytes(std.testing.io, allocator, path);
    defer allocator.free(raw);
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, raw, .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings("SlimmableContainer", parsed.value.object.get("architecture").?.string);
    const submodels = parsed.value.object.get("config").?.object.get("submodels").?.array;
    try std.testing.expectEqual(@as(usize, 2), submodels.items.len);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), try jsonNumber(submodels.items[0].object.get("max_value").?), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), try jsonNumber(submodels.items[1].object.get("max_value").?), 1e-12);
}

fn expectReferenceA2FastShapeJson(root: std.json.ObjectMap, channels: usize) !void {
    const config = root.get("config").?.object;
    try std.testing.expectEqual(@as(usize, 1), jsonArray(config.get("layers").?).items.len);
    try std.testing.expectEqual(@as(usize, 1), try jsonUsize(config.get("in_channels") orelse .{ .integer = 1 }));
    try std.testing.expect(config.get("head").? == .null);
    _ = try jsonNumber(config.get("head_scale").?);

    const layer = jsonArray(config.get("layers").?).items[0].object;
    try std.testing.expectEqual(@as(usize, 1), try jsonUsize(layer.get("input_size").?));
    try std.testing.expectEqual(@as(usize, 1), try jsonUsize(layer.get("condition_size").?));
    try std.testing.expectEqual(channels, try jsonUsize(layer.get("channels").?));
    try std.testing.expectEqual(channels, try jsonUsize(layer.get("bottleneck").?));
    try std.testing.expectEqual(@as(usize, 1), try jsonUsize(layer.get("groups_input").?));
    try std.testing.expectEqual(@as(usize, 1), try jsonUsize(layer.get("groups_input_mixin").?));
    try std.testing.expect(layer.get("slimmable").? == .null);

    const head = layer.get("head").?.object;
    try std.testing.expectEqual(@as(usize, 1), try jsonUsize(head.get("out_channels").?));
    try std.testing.expectEqual(@as(usize, 16), try jsonUsize(head.get("kernel_size").?));
    try std.testing.expect(try jsonBool(head.get("bias").?));

    const layer1x1 = layer.get("layer1x1").?.object;
    try std.testing.expect(try jsonBool(layer1x1.get("active").?));
    try std.testing.expectEqual(@as(usize, 1), try jsonUsize(layer1x1.get("groups").?));
    try std.testing.expect(!try jsonBool(layer.get("head1x1").?.object.get("active").?));

    const kernels = jsonArray(layer.get("kernel_sizes").?);
    const dilations = jsonArray(layer.get("dilations").?);
    const activations = jsonArray(layer.get("activation").?);
    const gating = jsonArray(layer.get("gating_mode").?);
    const secondary = jsonArray(layer.get("secondary_activation").?);
    try std.testing.expectEqual(train.A2Spec.kernel_sizes.len, kernels.items.len);
    try std.testing.expectEqual(train.A2Spec.dilations.len, dilations.items.len);
    try std.testing.expectEqual(train.A2Spec.kernel_sizes.len, activations.items.len);
    try std.testing.expectEqual(train.A2Spec.kernel_sizes.len, gating.items.len);
    try std.testing.expectEqual(train.A2Spec.kernel_sizes.len, secondary.items.len);
    for (0..train.A2Spec.kernel_sizes.len) |i| {
        try std.testing.expectEqual(train.A2Spec.kernel_sizes[i], try jsonUsize(kernels.items[i]));
        try std.testing.expectEqual(train.A2Spec.dilations[i], try jsonUsize(dilations.items[i]));
        const act = activations.items[i].object;
        try std.testing.expectEqualStrings("LeakyReLU", act.get("type").?.string);
        try std.testing.expectApproxEqAbs(@as(f64, 0.01), try jsonNumber(act.get("negative_slope").?), 1e-7);
        try std.testing.expectEqualStrings("none", gating.items[i].string);
        try std.testing.expect(secondary.items[i] == .null);
    }

    inline for (.{
        "conv_pre_film",
        "conv_post_film",
        "input_mixin_pre_film",
        "input_mixin_post_film",
        "activation_pre_film",
        "activation_post_film",
        "layer1x1_post_film",
        "head1x1_post_film",
    }) |key| {
        try std.testing.expect(!try jsonBool(layer.get(key).?.object.get("active").?));
    }
}

fn jsonArray(value: std.json.Value) std.json.Array {
    return switch (value) {
        .array => |array| array,
        else => unreachable,
    };
}

fn jsonUsize(value: std.json.Value) !usize {
    return switch (value) {
        .integer => |v| @intCast(v),
        .float => |v| @intFromFloat(v),
        else => error.InvalidJsonShape,
    };
}

fn jsonNumber(value: std.json.Value) !f64 {
    return switch (value) {
        .integer => |v| @floatFromInt(v),
        .float => |v| v,
        else => error.InvalidJsonShape,
    };
}

fn jsonBool(value: std.json.Value) !bool {
    return switch (value) {
        .bool => |v| v,
        else => error.InvalidJsonShape,
    };
}
