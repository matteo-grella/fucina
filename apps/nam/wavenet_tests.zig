//! Behavioral tests for the tensor WaveNet (`wavenet.zig`): the A2
//! reference paths (gated/blended residual FiLM handling, head-accumulator
//! seeding across arrays), chunked-vs-one-shot streaming parity on the
//! upstream tiny model, the two regimes of the one code path (the recorded
//! window forward against the streamed chunks, bitwise), gradients against
//! finite differences through grouped convs and FiLM branches, the
//! steady-state allocation count, and the upstream max fixture when present.

const std = @import("std");
const fucina = @import("fucina");
const wavenet = @import("wavenet.zig");
const nam_file = @import("nam_file.zig");

const ExecContext = fucina.ExecContext;
const WaveNet = wavenet.WaveNet;
const Activation = nam_file.Activation;

fn processOne(config: *const nam_file.WaveNetConfig, weights: []const f32, input: f32) !f32 {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();
    var model = try WaveNet.init(allocator, &ctx, config, weights, .{});
    defer model.deinit();
    const block = [_]f32{input};
    var output: [1]f32 = undefined;
    try model.process(&ctx, &block, &output, 1);
    return output[0];
}

test "wavenet A2: gated residual ignores layer1x1 post-FiLM and survives head scratch reuse" {
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
        // activation_post_film. head1x1_post_film writes 100 into its
        // output, but the residual must already own the activated value.
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 50.0, 0.0, 0.0, 0.0, 100.0, 0.0,
        // Array 1 observes array 0's residual stream through its head.
        1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
    };
    const file_config = nam_file.Config{ .wavenet = config };
    try std.testing.expectEqual(@as(usize, weights.len), nam_file.expectedWeightCount(&file_config));
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), try processOne(&config, &weights, 1.0), 1e-7);
}

test "wavenet A2: blended residual applies layer1x1 post-FiLM" {
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
    const weights = [_]f32{ 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0 };
    const file_config = nam_file.Config{ .wavenet = config };
    try std.testing.expectEqual(@as(usize, weights.len), nam_file.expectedWeightCount(&file_config));
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), try processOne(&config, &weights, 1.0), 1e-7);
}

test "wavenet A2: later arrays seed their head accumulator from prior head output" {
    const dilations = [_]usize{1};
    const kernels = [_]usize{1};
    const relu = [_]Activation{.{ .kind = .relu }};
    const sigmoid = [_]Activation{Activation.sigmoid_default};
    const none = [_]nam_file.GatingMode{.none};
    const array = nam_file.WaveNetLayerArray{
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
    };
    const layers = [_]nam_file.WaveNetLayerArray{ array, array };
    const config = nam_file.WaveNetConfig{
        .layers = &layers,
        .head = null,
        .head_scale = 1.0,
        .in_channels = 1,
        .condition_dsp = null,
    };
    const weights = [_]f32{ 0.0, 0.0, 0.0, 1.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0 };
    const file_config = nam_file.Config{ .wavenet = config };
    try std.testing.expectEqual(@as(usize, weights.len), nam_file.expectedWeightCount(&file_config));
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), try processOne(&config, &weights, 1.0), 1e-7);
}

test "wavenet: chunked streaming equals one-shot on the upstream tiny model" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();
    var model = try nam_file.loadFromSlice(allocator, @embedFile("testdata/wavenet.nam"));
    defer model.deinit();
    var engine = try WaveNet.init(allocator, &ctx, &model.config.wavenet, model.weights, .{});
    defer engine.deinit();

    const total = 333;
    var input: [total]f32 = undefined;
    for (&input, 0..) |*v, i| v.* = 0.5 * @sin(@as(f32, @floatFromInt(i)) * 0.05);

    var oneshot: [total]f32 = undefined;
    try engine.process(&ctx, &input, &oneshot, total);

    engine.reset();
    var chunked: [total]f32 = undefined;
    var offset: usize = 0;
    while (offset < total) {
        const n = @min(@as(usize, 64), total - offset);
        try engine.process(&ctx, input[offset..], chunked[offset..], n);
        offset += n;
    }
    // Per-sample math is chunk-independent => exact equality.
    try std.testing.expectEqualSlices(f32, &oneshot, &chunked);

    // The output must be non-trivial (the model actually transforms audio).
    var energy: f64 = 0;
    for (oneshot) |v| energy += @as(f64, v) * v;
    try std.testing.expect(energy > 1e-12);
}

// ---------------------------------------------------------------------------
// The two regimes of the one code path: the trainable model's recorded
// window forward against the constant model's streamed chunks.
// ---------------------------------------------------------------------------

fn runStream(allocator: std.mem.Allocator, config: *const nam_file.WaveNetConfig, weights: []const f32, input: []const f32, out: []f32, chunk: usize) !void {
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();
    var model = try WaveNet.init(allocator, &ctx, config, weights, .{ .chunk_hint = chunk });
    defer model.deinit();
    var offset: usize = 0;
    while (offset < input.len) {
        const n = @min(chunk, input.len - offset);
        try model.process(&ctx, input[offset..], out[offset..], n);
        offset += n;
    }
}

fn runRecorded(allocator: std.mem.Allocator, config: *const nam_file.WaveNetConfig, weights: []const f32, input: []const f32, out: []f32) !void {
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();
    var model = try WaveNet.init(allocator, &ctx, config, weights, .{ .trainable = true });
    defer model.deinit();
    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    const pred = try model.forward(&ctx, input);
    try std.testing.expect(pred.requiresGrad());
    try pred.copyTo(out);
    // The same model streams its validation audio without recording, from
    // a fresh stream.
    const streamed = try allocator.alloc(f32, out.len);
    defer allocator.free(streamed);
    model.reset();
    try model.process(&ctx, input, streamed, input.len);
    try std.testing.expectEqualSlices(f32, out, streamed);
}

fn expectTwoRegimes(config: *const nam_file.WaveNetConfig, weights: []const f32, total: usize) !void {
    const allocator = std.testing.allocator;
    const input = try allocator.alloc(f32, total);
    defer allocator.free(input);
    wavenet.fillSignal(input, 7);
    const recorded = try allocator.alloc(f32, total);
    defer allocator.free(recorded);
    const streamed = try allocator.alloc(f32, total);
    defer allocator.free(streamed);
    const streamed_odd = try allocator.alloc(f32, total);
    defer allocator.free(streamed_odd);
    try runRecorded(allocator, config, weights, input, recorded);
    try runStream(allocator, config, weights, input, streamed, 64);
    try runStream(allocator, config, weights, input, streamed_odd, 37);
    try std.testing.expectEqualSlices(f32, recorded, streamed);
    try std.testing.expectEqualSlices(f32, recorded, streamed_odd);
    var energy: f64 = 0;
    for (recorded) |v| energy += @as(f64, v) * v;
    try std.testing.expect(energy > 1e-9);
}

// A2-style layer array: gated or blended, grouped convs, every FiLM slot on.
fn a2Layers(comptime gating: nam_file.GatingMode) [1]nam_file.WaveNetLayerArray {
    const dilations = [_]usize{ 1, 2 };
    const kernels = [_]usize{ 3, 2 };
    const acts = [_]Activation{ .{ .kind = .tanh }, .{ .kind = .fasttanh } };
    const secondary = [_]Activation{ .{ .kind = .sigmoid }, .{ .kind = .silu } };
    const gates = [_]nam_file.GatingMode{ gating, gating };
    return .{.{
        .input_size = 1,
        .condition_size = 1,
        .channels = 4,
        .bottleneck = 4,
        .head_out = 1,
        .head_kernel = 2,
        .head_bias = true,
        .dilations = &dilations,
        .kernel_sizes = &kernels,
        .activations = &acts,
        .gating_modes = &gates,
        .secondary_activations = &secondary,
        .layer1x1_active = true,
        .layer1x1_groups = 2,
        .head1x1_active = true,
        .head1x1_out = 2,
        .head1x1_groups = 1,
        .groups_input = 2,
        .groups_input_mixin = 1,
        .conv_pre_film = .{ .active = true, .shift = false },
        .conv_post_film = .{ .active = true, .shift = true },
        .input_mixin_pre_film = .{ .active = true, .shift = true },
        .input_mixin_post_film = .{ .active = true, .shift = false },
        .activation_pre_film = .{ .active = true, .shift = true },
        .activation_post_film = .{ .active = true, .shift = true },
        .layer1x1_post_film = .{ .active = true, .shift = true },
        .head1x1_post_film = .{ .active = true, .shift = true },
    }};
}

const a2_gated_layers = a2Layers(.gated);
const a2_blended_layers = a2Layers(.blended);
const a2_post_head_kernels = [_]usize{ 2, 1 };

fn a2Config(layers: []const nam_file.WaveNetLayerArray) nam_file.WaveNetConfig {
    return .{
        .layers = layers,
        .head = .{ .channels = 3, .out_channels = 1, .kernel_sizes = &a2_post_head_kernels, .activation = .{ .kind = .relu } },
        .head_scale = 0.5,
        .in_channels = 1,
        .condition_dsp = null,
    };
}

test "wavenet: the recorded window forward is bitwise the streamed chunks (standard, tiny, A2 gated and blended)" {
    const allocator = std.testing.allocator;
    {
        const weights = try wavenet.syntheticWeights(allocator, &wavenet.standard_config, 11, 0.25);
        defer allocator.free(weights);
        try expectTwoRegimes(&wavenet.standard_config, weights, 1500);
    }
    {
        var model = try nam_file.loadFromSlice(allocator, @embedFile("testdata/wavenet.nam"));
        defer model.deinit();
        try expectTwoRegimes(&model.config.wavenet, model.weights, 999);
    }
    inline for (.{ &a2_gated_layers, &a2_blended_layers }) |layers| {
        const config = a2Config(layers);
        const weights = try wavenet.syntheticWeights(allocator, &config, 23, 0.4);
        defer allocator.free(weights);
        try expectTwoRegimes(&config, weights, 777);
    }
}

test "wavenet: the block path stops allocating once the buffer pool is warm" {
    var counting = wavenet.CountingAllocator{ .backing = std.testing.allocator };
    const allocator = counting.allocator();
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();
    const weights = try wavenet.syntheticWeights(std.testing.allocator, &wavenet.standard_config, 11, 0.25);
    defer std.testing.allocator.free(weights);
    var engine = try WaveNet.init(std.testing.allocator, &ctx, &wavenet.standard_config, weights, .{ .chunk_hint = 64 });
    defer engine.deinit();

    const block = 64;
    var input: [block]f32 = undefined;
    wavenet.fillSignal(&input, 3);
    var output: [block]f32 = undefined;
    for (0..32) |_| try engine.process(&ctx, &input, &output, block);
    const warm_allocs = counting.allocs.load(.monotonic);
    for (0..32) |_| try engine.process(&ctx, &input, &output, block);
    const steady_allocs = counting.allocs.load(.monotonic) - warm_allocs;
    try std.testing.expectEqual(@as(usize, 0), steady_allocs);
}

// ---------------------------------------------------------------------------
// Gradients through grouped convs and FiLM branches, against finite
// differences on the NAM weight stream.
// ---------------------------------------------------------------------------

fn sumLossForWeights(allocator: std.mem.Allocator, config: *const nam_file.WaveNetConfig, weights: []const f32, input: []const f32) !f32 {
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();
    var model = try WaveNet.init(allocator, &ctx, config, weights, .{ .trainable = true });
    defer model.deinit();
    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    const pred = try model.forward(&ctx, input);
    const loss = try pred.sumAll(&ctx);
    return loss.item();
}

fn sumAbs(ctx: *ExecContext, g: anytype) !f32 {
    var magnitude = try g.abs(ctx);
    defer magnitude.deinit();
    var total = try magnitude.sumAll(ctx);
    defer total.deinit();
    return total.item();
}

fn addFilmGradAbs(sum: *f32, film: *?wavenet.Film, ctx: *ExecContext) !void {
    if (film.*) |*f| {
        var weight_grad = try f.conv.weight.grad(ctx);
        if (weight_grad) |*g| {
            defer g.deinit();
            sum.* += try sumAbs(ctx, g);
        }
        if (f.conv.bias) |*bias| {
            var bias_grad = try bias.grad(ctx);
            if (bias_grad) |*g| {
                defer g.deinit();
                sum.* += try sumAbs(ctx, g);
            }
        }
    }
}

test "wavenet backward matches finite difference through grouped conv and FiLM branches" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    const dilations = [_]usize{1};
    const kernels = [_]usize{2};
    const activations = [_]Activation{.{ .kind = .fasttanh }};
    const secondary = [_]Activation{.{ .kind = .sigmoid }};
    const gating = [_]nam_file.GatingMode{.blended};
    const layers = [_]nam_file.WaveNetLayerArray{.{
        .input_size = 1,
        .condition_size = 1,
        .channels = 2,
        .bottleneck = 2,
        .head_out = 1,
        .head_kernel = 1,
        .head_bias = true,
        .dilations = &dilations,
        .kernel_sizes = &kernels,
        .activations = &activations,
        .gating_modes = &gating,
        .secondary_activations = &secondary,
        .layer1x1_active = true,
        .layer1x1_groups = 2,
        .head1x1_active = true,
        .head1x1_out = 1,
        .head1x1_groups = 1,
        .groups_input = 2,
        .groups_input_mixin = 1,
        .conv_post_film = .{ .active = true, .shift = true },
        .activation_post_film = .{ .active = true, .shift = true },
        .layer1x1_post_film = .{ .active = true, .shift = true },
        .head1x1_post_film = .{ .active = true, .shift = true },
    }};
    const config = nam_file.WaveNetConfig{
        .layers = &layers,
        .head = null,
        .head_scale = 1.0,
        .in_channels = 1,
        .condition_dsp = null,
    };
    const file_config = nam_file.Config{ .wavenet = config };
    const weight_count = nam_file.expectedWeightCount(&file_config);
    var weights = try allocator.alloc(f32, weight_count);
    defer allocator.free(weights);
    for (weights, 0..) |*v, i| {
        v.* = 0.17 * @sin(@as(f32, @floatFromInt(i)) * 0.37) + 0.03 * @cos(@as(f32, @floatFromInt(i)) * 0.11);
    }
    weights[weights.len - 1] = 1.0;

    const input = [_]f32{ -0.4, 0.2, 0.7, -0.1, 0.5 };

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try WaveNet.init(allocator, &ctx, &config, weights, .{ .trainable = true });
    defer model.deinit();
    const roundtrip = try model.extractWeights(&ctx, allocator);
    defer allocator.free(roundtrip);
    try std.testing.expectEqualSlices(f32, weights, roundtrip);

    var analytic: f32 = undefined;
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const pred = try model.forward(&ctx, &input);
        var loss = try pred.sumAll(&ctx);
        try loss.backward(&ctx);

        var conv_grad = (try model.arrays[0].layers[0].conv.weight.grad(&ctx)).?;
        defer conv_grad.deinit();
        analytic = (try conv_grad.dataConst())[0];
        try std.testing.expect(std.math.isFinite(analytic));

        var film_grad_sum: f32 = 0;
        try addFilmGradAbs(&film_grad_sum, &model.arrays[0].layers[0].conv_post_film, &ctx);
        try addFilmGradAbs(&film_grad_sum, &model.arrays[0].layers[0].activation_post_film, &ctx);
        try addFilmGradAbs(&film_grad_sum, &model.arrays[0].layers[0].layer1x1_post_film, &ctx);
        try addFilmGradAbs(&film_grad_sum, &model.arrays[0].layers[0].head1x1_post_film, &ctx);
        try std.testing.expect(film_grad_sum > 1e-6);
    }

    // Flat NAM index 2 is the first grouped dilated-conv weight after the
    // two rechannel weights; internally that is conv.weight[0].
    const selected_flat_index: usize = 2;
    const eps: f32 = 1e-3;
    var plus = try allocator.dupe(f32, weights);
    defer allocator.free(plus);
    var minus = try allocator.dupe(f32, weights);
    defer allocator.free(minus);
    plus[selected_flat_index] += eps;
    minus[selected_flat_index] -= eps;
    const plus_loss = try sumLossForWeights(allocator, &config, plus, &input);
    const minus_loss = try sumLossForWeights(allocator, &config, minus, &input);
    const numeric = (plus_loss - minus_loss) / (2.0 * eps);
    try std.testing.expect(std.math.isFinite(numeric));
    try std.testing.expectApproxEqAbs(numeric, analytic, 2e-2);
}

test "wavenet: the upstream max fixture round-trips and streams as it records, when present" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var file_model = nam_file.loadFile(
        std.testing.io,
        allocator,
        "refs/NeuralAmpModelerCore/example_models/wavenet_a2_max.nam",
    ) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer file_model.deinit();
    if (file_model.config != .wavenet) return error.TestExpectedWaveNet;

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try WaveNet.init(allocator, &ctx, &file_model.config.wavenet, file_model.weights, .{ .trainable = true });
    defer model.deinit();
    const roundtrip = try model.extractWeights(&ctx, allocator);
    defer allocator.free(roundtrip);
    try std.testing.expectEqualSlices(f32, file_model.weights, roundtrip);

    const frames = 257;
    var input: [frames]f32 = undefined;
    for (&input, 0..) |*v, i| {
        v.* = 0.35 * @sin(@as(f32, @floatFromInt(i)) * 0.071) + 0.12 * @cos(@as(f32, @floatFromInt(i)) * 0.019);
    }
    var recorded: [frames]f32 = undefined;
    var streamed: [frames]f32 = undefined;
    try runRecorded(allocator, &file_model.config.wavenet, file_model.weights, &input, &recorded);
    try runStream(allocator, &file_model.config.wavenet, file_model.weights, &input, &streamed, 64);
    try std.testing.expectEqualSlices(f32, &recorded, &streamed);
}

// ---------------------------------------------------------------------------
// Per-channel PReLU: the slope tensor built once at load, the activation
// against a hand computation, the refusals, and both regimes through a
// model that carries it in a layer and in the post head.
// ---------------------------------------------------------------------------

test "wavenet: per-channel PReLU slopes apply per channel and refuse a missing or mismatched tensor" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();
    const slopes = [_]f32{ 0.1, 0.5, -0.25 };
    const act = Activation{ .kind = .prelu, .negative_slopes = &slopes };
    var slope_tensor = (try wavenet.preluSlopes(&ctx, &act, 6)).?;
    defer slope_tensor.deinit();
    // The upstream list cycles over the channels.
    try std.testing.expectEqualSlices(f32, &.{ 0.1, 0.5, -0.25, 0.1, 0.5, -0.25 }, try slope_tensor.dataConst());

    const values = [_]f32{ 1.0, -1.0, -2.0, 0.5, -0.5, -4.0, -1.0, 2.0, 3.0, -3.0, 0.0, -0.1 };
    var x = try wavenet.TimeOut.fromSlice(&ctx, .{ 2, 6 }, &values);
    defer x.deinit();
    var y = try wavenet.activate(&ctx, &act, &x, false, &slope_tensor);
    defer y.deinit();
    var expected: [12]f32 = undefined;
    for (values, 0..) |v, i| expected[i] = if (v >= 0) v else v * slopes[(i % 6) % 3];
    try std.testing.expectEqualSlices(f32, &expected, try y.dataConst());

    // A single slope needs no tensor.
    const scalar_act = Activation{ .kind = .prelu, .negative_slope = 0.2 };
    try std.testing.expect((try wavenet.preluSlopes(&ctx, &scalar_act, 6)) == null);
    var y_scalar = try wavenet.activate(&ctx, &scalar_act, &x, false, null);
    defer y_scalar.deinit();
    for (values, try y_scalar.dataConst()) |v, got| try std.testing.expectEqual(if (v >= 0) v else v * 0.2, got);

    // Per-channel slopes without their tensor, or with one of another width, are refused.
    try std.testing.expectError(error.PreluSlopesMissing, wavenet.activate(&ctx, &act, &x, false, null));
    var narrow_slopes = (try wavenet.preluSlopes(&ctx, &act, 3)).?;
    defer narrow_slopes.deinit();
    try std.testing.expectError(error.PreluWidthMismatch, wavenet.activate(&ctx, &act, &x, false, &narrow_slopes));
}

test "wavenet: a per-channel PReLU model records and streams alike" {
    const allocator = std.testing.allocator;
    const slopes = [_]f32{ 0.05, 0.3, -0.1 };
    const dilations = [_]usize{ 1, 2 };
    const kernels = [_]usize{ 3, 2 };
    const acts = [_]Activation{ .{ .kind = .prelu, .negative_slopes = &slopes }, .{ .kind = .prelu, .negative_slopes = &slopes } };
    const secondary = [_]Activation{ .{ .kind = .prelu, .negative_slopes = &slopes }, .{ .kind = .sigmoid } };
    const gates = [_]nam_file.GatingMode{ .gated, .none };
    const layers = [_]nam_file.WaveNetLayerArray{.{
        .input_size = 1,
        .condition_size = 1,
        .channels = 4,
        .bottleneck = 4,
        .head_out = 2,
        .head_kernel = 1,
        .head_bias = true,
        .dilations = &dilations,
        .kernel_sizes = &kernels,
        .activations = &acts,
        .gating_modes = &gates,
        .secondary_activations = &secondary,
        .layer1x1_active = true,
        .layer1x1_groups = 1,
        .head1x1_active = false,
        .head1x1_out = 4,
        .head1x1_groups = 1,
        .groups_input = 1,
        .groups_input_mixin = 1,
    }};
    const post_kernels = [_]usize{ 2, 1 };
    const config = nam_file.WaveNetConfig{
        .layers = &layers,
        .head = .{ .channels = 3, .out_channels = 1, .kernel_sizes = &post_kernels, .activation = .{ .kind = .prelu, .negative_slopes = &slopes } },
        .head_scale = 0.5,
        .in_channels = 1,
        .condition_dsp = null,
    };
    const weights = try wavenet.syntheticWeights(allocator, &config, 29, 0.6);
    defer allocator.free(weights);
    try expectTwoRegimes(&config, weights, 500);
}
