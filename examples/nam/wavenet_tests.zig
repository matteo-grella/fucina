//! Behavioral tests for the streaming WaveNet engine (`wavenet.zig`): the A2
//! reference paths (gated/blended residual FiLM handling, head-accumulator
//! seeding across arrays) and chunked-vs-one-shot streaming parity on the
//! upstream tiny model.
const std = @import("std");
const wavenet = @import("wavenet.zig");
const nam_file = @import("nam_file.zig");

const WaveNetEngine = wavenet.WaveNetEngine;
const Activation = nam_file.Activation;

test "wavenet A2: gated residual ignores layer1x1 post-FiLM and survives head scratch reuse" {
    const allocator = std.testing.allocator;
    const dilations = [_]usize{1};
    const kernels = [_]usize{1};
    const relu = [_]Activation{.{ .kind = .relu }};
    const hardtanh = [_]Activation{.{ .kind = .hardtanh }};
    const gated = [_]nam_file.GatingMode{.gated};
    const none = [_]nam_file.GatingMode{.none};
    const layers = [_]nam_file.WaveNetLayerArray{
        .{
            .input_size = 1,
            .condition_size = 1,
            .channels = 1,
            .bottleneck = 1,
            .head_out = 1,
            .head_kernel = 1,
            .head_bias = false,
            .dilations = &dilations,
            .kernel_sizes = &kernels,
            .activations = &relu,
            .gating_modes = &gated,
            .secondary_activations = &hardtanh,
            .layer1x1_active = true,
            .layer1x1_groups = 1,
            .head1x1_active = true,
            .head1x1_out = 1,
            .head1x1_groups = 1,
            .groups_input = 1,
            .groups_input_mixin = 1,
            .activation_post_film = .{ .active = true, .shift = true },
            .layer1x1_post_film = .{ .active = true, .shift = true },
            .head1x1_post_film = .{ .active = true, .shift = true },
        },
        .{
            .input_size = 1,
            .condition_size = 1,
            .channels = 1,
            .bottleneck = 1,
            .head_out = 1,
            .head_kernel = 1,
            .head_bias = false,
            .dilations = &dilations,
            .kernel_sizes = &kernels,
            .activations = &relu,
            .gating_modes = &none,
            .secondary_activations = &hardtanh,
            .layer1x1_active = false,
            .layer1x1_groups = 1,
            .head1x1_active = false,
            .head1x1_out = 1,
            .head1x1_groups = 1,
            .groups_input = 1,
            .groups_input_mixin = 1,
        },
    };
    const config = nam_file.WaveNetConfig{
        .layers = &layers,
        .head = null,
        .head_scale = 1.0,
        .in_channels = 1,
        .condition_dsp = null,
    };
    const weights = [_]f32{
        // Array 0: x starts at 1; gated activation is forced to 2 by
        // activation_post_film. head1x1_post_film writes 100 into scratch,
        // but residual must already own/copy the activated value.
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        2.0,
        0.0,
        0.0,
        0.0,
        50.0,
        0.0,
        0.0,
        0.0,
        100.0,
        0.0,
        // Array 1 observes array 0's residual stream through its head.
        1.0,
        1.0,
        0.0,
        0.0,
        1.0,
        1.0,
    };
    const file_config = nam_file.Config{ .wavenet = config };
    try std.testing.expectEqual(@as(usize, weights.len), nam_file.expectedWeightCount(&file_config));

    var engine = try WaveNetEngine.init(allocator, &config, &weights);
    defer engine.deinit();
    try engine.reset(1);

    const input = [_]f32{1.0};
    var output: [1]f32 = undefined;
    engine.process(&input, &output, 1);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), output[0], 1e-7);
}

test "wavenet A2: blended residual applies layer1x1 post-FiLM" {
    const allocator = std.testing.allocator;
    const dilations = [_]usize{1};
    const kernels = [_]usize{1};
    const relu = [_]Activation{.{ .kind = .relu }};
    const hardtanh = [_]Activation{.{ .kind = .hardtanh }};
    const blended = [_]nam_file.GatingMode{.blended};
    const none = [_]nam_file.GatingMode{.none};
    const layers = [_]nam_file.WaveNetLayerArray{
        .{
            .input_size = 1,
            .condition_size = 1,
            .channels = 1,
            .bottleneck = 1,
            .head_out = 1,
            .head_kernel = 1,
            .head_bias = false,
            .dilations = &dilations,
            .kernel_sizes = &kernels,
            .activations = &relu,
            .gating_modes = &blended,
            .secondary_activations = &hardtanh,
            .layer1x1_active = true,
            .layer1x1_groups = 1,
            .head1x1_active = false,
            .head1x1_out = 1,
            .head1x1_groups = 1,
            .groups_input = 1,
            .groups_input_mixin = 1,
            .layer1x1_post_film = .{ .active = true, .shift = true },
        },
        .{
            .input_size = 1,
            .condition_size = 1,
            .channels = 1,
            .bottleneck = 1,
            .head_out = 1,
            .head_kernel = 1,
            .head_bias = false,
            .dilations = &dilations,
            .kernel_sizes = &kernels,
            .activations = &relu,
            .gating_modes = &none,
            .secondary_activations = &hardtanh,
            .layer1x1_active = false,
            .layer1x1_groups = 1,
            .head1x1_active = false,
            .head1x1_out = 1,
            .head1x1_groups = 1,
            .groups_input = 1,
            .groups_input_mixin = 1,
        },
    };
    const config = nam_file.WaveNetConfig{
        .layers = &layers,
        .head = null,
        .head_scale = 1.0,
        .in_channels = 1,
        .condition_dsp = null,
    };
    const weights = [_]f32{
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        7.0,
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
        1.0,
        1.0,
    };
    const file_config = nam_file.Config{ .wavenet = config };
    try std.testing.expectEqual(@as(usize, weights.len), nam_file.expectedWeightCount(&file_config));

    var engine = try WaveNetEngine.init(allocator, &config, &weights);
    defer engine.deinit();
    try engine.reset(1);

    const input = [_]f32{1.0};
    var output: [1]f32 = undefined;
    engine.process(&input, &output, 1);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), output[0], 1e-7);
}

test "wavenet A2: later arrays seed their head accumulator from prior head output" {
    const allocator = std.testing.allocator;
    const dilations = [_]usize{1};
    const kernels = [_]usize{1};
    const relu = [_]Activation{.{ .kind = .relu }};
    const sigmoid = [_]Activation{Activation.sigmoid_default};
    const none = [_]nam_file.GatingMode{.none};
    const layers = [_]nam_file.WaveNetLayerArray{
        .{
            .input_size = 1,
            .condition_size = 1,
            .channels = 1,
            .bottleneck = 1,
            .head_out = 1,
            .head_kernel = 1,
            .head_bias = false,
            .dilations = &dilations,
            .kernel_sizes = &kernels,
            .activations = &relu,
            .gating_modes = &none,
            .secondary_activations = &sigmoid,
            .layer1x1_active = false,
            .layer1x1_groups = 1,
            .head1x1_active = false,
            .head1x1_out = 1,
            .head1x1_groups = 1,
            .groups_input = 1,
            .groups_input_mixin = 1,
        },
        .{
            .input_size = 1,
            .condition_size = 1,
            .channels = 1,
            .bottleneck = 1,
            .head_out = 1,
            .head_kernel = 1,
            .head_bias = false,
            .dilations = &dilations,
            .kernel_sizes = &kernels,
            .activations = &relu,
            .gating_modes = &none,
            .secondary_activations = &sigmoid,
            .layer1x1_active = false,
            .layer1x1_groups = 1,
            .head1x1_active = false,
            .head1x1_out = 1,
            .head1x1_groups = 1,
            .groups_input = 1,
            .groups_input_mixin = 1,
        },
    };
    const config = nam_file.WaveNetConfig{
        .layers = &layers,
        .head = null,
        .head_scale = 1.0,
        .in_channels = 1,
        .condition_dsp = null,
    };
    const weights = [_]f32{
        0.0,
        0.0,
        0.0,
        1.0,
        5.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
    };
    const file_config = nam_file.Config{ .wavenet = config };
    try std.testing.expectEqual(@as(usize, weights.len), nam_file.expectedWeightCount(&file_config));

    var engine = try WaveNetEngine.init(allocator, &config, &weights);
    defer engine.deinit();
    try engine.reset(1);

    const input = [_]f32{1.0};
    var output: [1]f32 = undefined;
    engine.process(&input, &output, 1);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), output[0], 1e-7);
}

test "wavenet engine: chunked streaming equals one-shot on the upstream tiny model" {
    const allocator = std.testing.allocator;
    var model = try nam_file.loadFromSlice(allocator, @embedFile("testdata/wavenet.nam"));
    defer model.deinit();

    var engine = try WaveNetEngine.init(allocator, &model.config.wavenet, model.weights);
    defer engine.deinit();

    const total = 333;
    var input: [total]f32 = undefined;
    for (&input, 0..) |*v, i| v.* = 0.5 * @sin(@as(f32, @floatFromInt(i)) * 0.05);

    try engine.reset(total);
    var oneshot: [total]f32 = undefined;
    engine.process(&input, &oneshot, total);

    try engine.reset(64);
    var chunked: [total]f32 = undefined;
    var offset: usize = 0;
    while (offset < total) {
        const n = @min(@as(usize, 64), total - offset);
        engine.process(input[offset..], chunked[offset..], n);
        offset += n;
    }
    // Per-sample math is chunk-independent => exact equality.
    try std.testing.expectEqualSlices(f32, &oneshot, &chunked);

    // The output must be non-trivial (the model actually transforms audio).
    var energy: f64 = 0;
    for (oneshot) |v| energy += @as(f64, v) * v;
    try std.testing.expect(energy > 1e-12);
}
