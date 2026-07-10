//! Behavioral tests for the `.nam` model-file reader (`nam_file.zig`): loads the
//! upstream WaveNet/LSTM/SlimmableContainer examples, the modern A2 fields
//! (grouped FiLMs + condition DSP), the version gate, the legacy spellings, and
//! the per-architecture flat-weight-count formulas.
const std = @import("std");
const nam_file = @import("nam_file.zig");

const Error = nam_file.Error;
const Arch = nam_file.Arch;
const Activation = nam_file.Activation;
const GatingMode = nam_file.GatingMode;
const Config = nam_file.Config;
const expectedWeightCount = nam_file.expectedWeightCount;
const loadFromSlice = nam_file.loadFromSlice;

test "loads the upstream tiny WaveNet example exactly" {
    const allocator = std.testing.allocator;
    var model = try loadFromSlice(allocator, @embedFile("testdata/wavenet.nam"));
    defer model.deinit();

    try std.testing.expectEqual(Arch.wavenet, model.architecture);
    try std.testing.expectEqual(@as(usize, 131), model.weights.len);
    try std.testing.expectEqual(@as(f64, 48000.0), model.sample_rate);
    try std.testing.expect(!model.partial_support);

    const c = model.config.wavenet;
    try std.testing.expectEqual(@as(usize, 2), c.layers.len);
    try std.testing.expectEqual(@as(usize, 3), c.layers[0].channels);
    try std.testing.expectEqual(@as(usize, 1), c.layers[0].input_size);
    try std.testing.expectEqualSlices(usize, &.{ 1, 2 }, c.layers[0].dilations);
    try std.testing.expectEqualSlices(usize, &.{ 3, 3 }, c.layers[0].kernel_sizes);
    try std.testing.expectEqual(@as(usize, 2), c.layers[0].head_out);
    try std.testing.expect(!c.layers[0].head_bias);
    try std.testing.expectEqual(@as(usize, 1), c.layers[1].head_out);
    try std.testing.expect(c.layers[1].head_bias);
    try std.testing.expectEqual(Activation.Kind.tanh, c.layers[0].activations[0].kind);
    try std.testing.expect(model.metadata.loudness != null);
    // head_scale duplicated as the final weight (gotcha 1).
    try std.testing.expectApproxEqAbs(c.head_scale, model.weights[model.weights.len - 1], 1e-12);
}

test "loads the upstream LSTM example exactly" {
    const allocator = std.testing.allocator;
    var model = try loadFromSlice(allocator, @embedFile("testdata/lstm.nam"));
    defer model.deinit();

    try std.testing.expectEqual(Arch.lstm, model.architecture);
    try std.testing.expectEqual(@as(usize, 70), model.weights.len);
    const c = model.config.lstm;
    try std.testing.expectEqual(@as(usize, 1), c.input_size);
    try std.testing.expectEqual(@as(usize, 3), c.hidden_size);
    try std.testing.expectEqual(@as(usize, 1), c.num_layers);
}

test "loads the upstream SlimmableContainer at its highest-quality submodel" {
    const allocator = std.testing.allocator;
    var model = try loadFromSlice(allocator, @embedFile("testdata/slimmable_container.nam"));
    defer model.deinit();
    const info = model.submodel_info.?;
    try std.testing.expectEqual(info.count - 1, info.index);
    try std.testing.expect(info.max_value >= 1.0);
    // Weight count was validated against the chosen submodel's config by
    // the loader (exact-count check); the engine must accept it too.
    try std.testing.expect(model.weights.len > 0);
}

test "rejects slimmable WaveNet with a named unsupported-feature error" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(Error.UnsupportedFeature, loadFromSlice(allocator, @embedFile("testdata/slimmable_wavenet.nam")));
}

test "loads modern A2 WaveNet fields including grouped FiLMs and condition DSP" {
    const allocator = std.testing.allocator;
    var weights_text: std.ArrayList(u8) = .empty;
    defer weights_text.deinit(allocator);
    for (0..101) |i| {
        if (i != 0) try weights_text.appendSlice(allocator, ", ");
        try weights_text.appendSlice(allocator, "0.0");
    }

    const json = try std.fmt.allocPrint(allocator,
        \\{{"version":"0.7.0","architecture":"WaveNet","sample_rate":48000,"config":{{"condition_dsp":{{"version":"0.7.0","architecture":"Linear","config":{{"receptive_field":3,"bias":true}},"weights":[0.0,0.0,0.0,0.0]}},"layers":[{{"input_size":1,"condition_size":1,"channels":2,"bottleneck":2,"groups_input":2,"groups_input_mixin":1,"dilations":[1,2],"kernel_sizes":[1,1],"activation":["ReLU","ReLU"],"secondary_activation":["Hardtanh","Sigmoid"],"gating_mode":["gated","blended"],"head":{{"out_channels":1,"kernel_size":1,"bias":false}},"layer1x1":{{"active":true,"groups":2}},"head1x1":{{"active":true,"out_channels":2,"groups":2}},"conv_post_film":{{"active":true,"shift":false,"groups":1}},"activation_post_film":{{"active":true,"shift":true,"groups":1}},"layer1x1_post_film":{{"active":true,"shift":true,"groups":1}},"head1x1_post_film":{{"active":true,"shift":false,"groups":1}}}}],"head":null,"head_scale":1.0}},"weights":[{s}]}}
    , .{weights_text.items});
    defer allocator.free(json);

    var model = try loadFromSlice(allocator, json);
    defer model.deinit();

    try std.testing.expectEqual(Arch.wavenet, model.architecture);
    try std.testing.expectEqual(@as(usize, 101), model.weights.len);
    try std.testing.expectEqual(@as(usize, 101), expectedWeightCount(&model.config));
    try std.testing.expectEqual(@as(f64, 48000.0), model.sample_rate);

    const c = model.config.wavenet;
    try std.testing.expect(c.condition_dsp != null);
    const dsp = c.condition_dsp.?;
    try std.testing.expectEqual(Arch.linear, dsp.architecture);
    try std.testing.expectEqual(@as(usize, 4), dsp.weights.len);
    try std.testing.expectEqual(@as(usize, 3), c.receptiveField());

    const layer = c.layers[0];
    try std.testing.expectEqual(@as(usize, 2), layer.groups_input);
    try std.testing.expectEqual(@as(usize, 1), layer.groups_input_mixin);
    try std.testing.expectEqual(GatingMode.gated, layer.gating_modes[0]);
    try std.testing.expectEqual(GatingMode.blended, layer.gating_modes[1]);
    try std.testing.expectEqual(Activation.Kind.hardtanh, layer.secondary_activations[0].kind);
    try std.testing.expect(layer.conv_post_film.active);
    try std.testing.expect(!layer.conv_post_film.shift);
    try std.testing.expect(layer.activation_post_film.shift);
    try std.testing.expect(layer.layer1x1_post_film.active);
    try std.testing.expect(layer.head1x1_post_film.active);
}

test "weight-count formula reproduces the classic standard WaveNet (13802)" {
    // The classic full-mode config (nam_full_configs/models/wavenet.json):
    // two arrays 16->8 channels, k=3, dilations 1..512, Tanh, heads
    // {8,k1,no-bias} and {1,k1,bias}.
    const dilations = [_]usize{ 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 };
    const kernels = [_]usize{3} ** 10;
    const tanhs = [_]Activation{Activation.tanh_default} ** 10;
    const none = [_]GatingMode{.none} ** 10;
    const sigmoids = [_]Activation{Activation.sigmoid_default} ** 10;
    const config = Config{ .wavenet = .{
        .layers = &.{
            .{
                .input_size = 1,
                .condition_size = 1,
                .channels = 16,
                .bottleneck = 16,
                .head_out = 8,
                .head_kernel = 1,
                .head_bias = false,
                .dilations = &dilations,
                .kernel_sizes = &kernels,
                .activations = &tanhs,
                .gating_modes = &none,
                .secondary_activations = &sigmoids,
                .layer1x1_active = true,
                .layer1x1_groups = 1,
                .head1x1_active = false,
                .head1x1_out = 16,
                .head1x1_groups = 1,
                .groups_input = 1,
                .groups_input_mixin = 1,
            },
            .{
                .input_size = 16,
                .condition_size = 1,
                .channels = 8,
                .bottleneck = 8,
                .head_out = 1,
                .head_kernel = 1,
                .head_bias = true,
                .dilations = &dilations,
                .kernel_sizes = &kernels,
                .activations = &tanhs,
                .gating_modes = &none,
                .secondary_activations = &sigmoids,
                .layer1x1_active = true,
                .layer1x1_groups = 1,
                .head1x1_active = false,
                .head1x1_out = 8,
                .head1x1_groups = 1,
                .groups_input = 1,
                .groups_input_mixin = 1,
            },
        },
        .head = null,
        .head_scale = 0.02,
        .in_channels = 1,
        .condition_dsp = null,
    } };
    try std.testing.expectEqual(@as(usize, 13802), expectedWeightCount(&config));
    // Receptive field of the classic standard: 1 + 2*(2*1023) = 4093.
    try std.testing.expectEqual(@as(usize, 4093), config.wavenet.receptiveField());
}

test "weight-count formulas for ConvNet and Linear" {
    // ConvNet with batchnorm: per block cout*cin*2 + 4*cout + 1; head w+b.
    const convnet = Config{ .convnet = .{
        .channels = 4,
        .dilations = &.{ 1, 2 },
        .batchnorm = true,
        .activation = Activation.tanh_default,
        .in_channels = 1,
        .out_channels = 1,
    } };
    // block0: 4*1*2 + 17 = 25; block1: 4*4*2 + 17 = 49; head: 4+1=5 -> 79
    try std.testing.expectEqual(@as(usize, 79), expectedWeightCount(&convnet));

    const linear = Config{ .linear = .{ .receptive_field = 8192, .bias = true, .in_channels = 1, .out_channels = 1 } };
    try std.testing.expectEqual(@as(usize, 8193), expectedWeightCount(&linear));
}

test "pathological NAM dimensions saturate instead of overflowing" {
    const huge = std.math.maxInt(usize);
    const convnet = Config{ .convnet = .{
        .channels = huge,
        .dilations = &.{huge},
        .batchnorm = false,
        .activation = Activation.tanh_default,
        .in_channels = 2,
        .out_channels = 1,
    } };
    try std.testing.expectEqual(huge, expectedWeightCount(&convnet));
    try std.testing.expectEqual(huge, convnet.convnet.receptiveField());

    const linear = Config{ .linear = .{ .receptive_field = huge, .bias = true, .in_channels = 1, .out_channels = 1 } };
    try std.testing.expectEqual(huge, expectedWeightCount(&linear));
}

test "version gate matches upstream semantics" {
    const allocator = std.testing.allocator;
    // minor > 7 rejected
    try std.testing.expectError(Error.UnsupportedVersion, loadFromSlice(allocator,
        \\{"version": "0.8.0", "architecture": "Linear", "config": {"receptive_field": 2, "bias": false}, "weights": [1.0, 2.0]}
    ));
    // below 0.5.0 rejected
    try std.testing.expectError(Error.UnsupportedVersion, loadFromSlice(allocator,
        \\{"version": "0.4.9", "architecture": "Linear", "config": {"receptive_field": 2, "bias": false}, "weights": [1.0, 2.0]}
    ));
    // patch-newer loads with partial_support
    var partial = try loadFromSlice(allocator,
        \\{"version": "0.7.3", "architecture": "Linear", "config": {"receptive_field": 2, "bias": false}, "weights": [1.0, 2.0]}
    );
    defer partial.deinit();
    try std.testing.expect(partial.partial_support);
    try std.testing.expectEqual(@as(f64, -1.0), partial.sample_rate);

    // wrong weight count is loud
    try std.testing.expectError(Error.WeightCountMismatch, loadFromSlice(allocator,
        \\{"version": "0.7.0", "architecture": "Linear", "config": {"receptive_field": 2, "bias": false}, "weights": [1.0]}
    ));
}

test "legacy spellings: head_size/head_bias, scalar kernel_size, boolean gated" {
    const allocator = std.testing.allocator;
    // A minimal legacy-style gated WaveNet: 1 array, 2 channels, K=2,
    // dilations [1,2], gated => Bg = 2*B = 4.
    // rechannel: 2*1 = 2
    // per layer: conv 4*2*2+4 = 20; mixin 4*1 = 4; layer1x1 2*2+2 = 6 -> 30
    // head (legacy size 1, bias default false, k=1): 1*2 = 2
    // total = 2 + 60 + 2 + 1 = 65
    var json_buf: [4096]u8 = undefined;
    var weights_text: std.ArrayList(u8) = .empty;
    defer weights_text.deinit(allocator);
    for (0..65) |i| {
        if (i != 0) try weights_text.appendSlice(allocator, ", ");
        try weights_text.appendSlice(allocator, "0.5");
    }
    const json = try std.fmt.bufPrint(&json_buf,
        \\{{"version": "0.5.0", "architecture": "WaveNet", "config": {{"layers": [{{"input_size": 1, "condition_size": 1, "channels": 2, "head_size": 1, "kernel_size": 2, "dilations": [1, 2], "activation": "Tanh", "gated": true}}], "head": null, "head_scale": 0.02}}, "weights": [{s}]}}
    , .{weights_text.items});

    var model = try loadFromSlice(allocator, json);
    defer model.deinit();
    const c = model.config.wavenet;
    try std.testing.expectEqual(@as(usize, 1), c.layers.len);
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, c.layers[0].kernel_sizes);
    try std.testing.expectEqual(GatingMode.gated, c.layers[0].gating_modes[0]);
    try std.testing.expectEqual(Activation.Kind.sigmoid, c.layers[0].secondary_activations[0].kind);
    try std.testing.expectEqual(@as(usize, 1), c.layers[0].head_out);
    try std.testing.expectEqual(@as(usize, 1), c.layers[0].head_kernel);
    try std.testing.expect(!c.layers[0].head_bias);
    try std.testing.expectEqual(@as(usize, 65), model.weights.len);
}
