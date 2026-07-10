//! Tests for the NAM WaveNet trainer (examples/nam/train.zig): forward/export
//! parity vs the streaming engine, optimizer-step loss reduction, MRSTFT torch
//! goldens, and packed / A2 training-spec round-trips through the public API.

const std = @import("std");
const fucina = @import("fucina");
const nam_file = @import("nam_file.zig");
const wavenet = @import("wavenet.zig");
const train = @import("train.zig");

const Tensor = fucina.Tensor;
const ExecContext = fucina.ExecContext;

const Trainable = train.Trainable;
const A2Trainable = train.A2Trainable;
const ModelSpec = train.ModelSpec;
const TrainingSpec = train.TrainingSpec;
const PackedSpec = train.PackedSpec;
const MrstftResolution = train.MrstftResolution;
const mrstftLoss = train.mrstftLoss;
const renderWeights = train.renderWeights;
const renderTrainingSpec = train.renderTrainingSpec;
const freeEngineConfig = train.freeEngineConfig;
const initWaveNetWeights = train.initWaveNetWeights;

test "trainable forward matches the streaming engine through export" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try Trainable.init(allocator, &ctx, ModelSpec.tiny, 42);
    defer model.deinit();

    var window: [200]f32 = undefined;
    for (&window, 0..) |*v, i| v.* = 0.4 * @sin(@as(f32, @floatFromInt(i)) * 0.21);

    // Autograd-graph forward.
    const scope = ctx.openExecScope();
    var graph_pred: [200]f32 = undefined;
    {
        const pred = try model.forward(&ctx, &window);
        const values = try pred.dataConst();
        @memcpy(&graph_pred, values);
    }
    ctx.closeExecScope(scope);

    // Streaming-engine forward through the exported flat weights.
    const weights = try model.extractWeights(allocator);
    defer allocator.free(weights);
    var engine_pred: [200]f32 = undefined;
    try renderWeights(allocator, &model.spec, weights, &window, &engine_pred);

    for (graph_pred, engine_pred) |a, b| {
        try std.testing.expect(@abs(a - b) < 2e-5);
    }
}

test "one optimizer step reduces the segment loss" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try Trainable.init(allocator, &ctx, ModelSpec.tiny, 7);
    defer model.deinit();
    var opt = fucina.optim.Adam.init(allocator, .{ .lr = 0.004, .weight_decay = 0 });
    defer opt.deinit();
    try model.registerParams(&opt);

    var window: [400]f32 = undefined;
    for (&window, 0..) |*v, i| v.* = 0.5 * @sin(@as(f32, @floatFromInt(i)) * 0.13);
    var target: [300]f32 = undefined;
    // A nonlinear "amp": soft clip of the input.
    for (&target, window[100..]) |*t, v| t.* = std.math.tanh(2.0 * v) * 0.7;

    var first_loss: f32 = undefined;
    var last_loss: f32 = undefined;
    for (0..12) |step_index| {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const loss = try model.segmentLoss(&ctx, &window, &target);
        const value = try loss.item();
        if (step_index == 0) first_loss = value;
        last_loss = value;
        try loss.backward(&ctx);
        try opt.step(&ctx);
        opt.zeroGrad();
    }
    // Adam at lr 0.004 walks steadily on this toy problem; assert a clear
    // monotone improvement rather than a fragile percentage.
    try std.testing.expect(last_loss < first_loss * 0.99);
    try std.testing.expect(std.math.isFinite(last_loss));
}

test "A2 WaveNet snapshot owns updated recursive condition DSP weights" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    const child_dilations = [_]usize{1};
    const child_kernels = [_]usize{1};
    const child_acts = [_]nam_file.Activation{.{ .kind = .leaky_relu, .negative_slope = 0.1 }};
    const child_secondary = [_]nam_file.Activation{nam_file.Activation.sigmoid_default};
    const child_gating = [_]nam_file.GatingMode{.none};
    const child_layers = [_]nam_file.WaveNetLayerArray{.{
        .input_size = 1,
        .condition_size = 1,
        .channels = 1,
        .bottleneck = 1,
        .head_out = 1,
        .head_kernel = 1,
        .head_bias = true,
        .dilations = &child_dilations,
        .kernel_sizes = &child_kernels,
        .activations = &child_acts,
        .gating_modes = &child_gating,
        .secondary_activations = &child_secondary,
        .layer1x1_active = true,
        .layer1x1_groups = 1,
        .head1x1_active = false,
        .head1x1_out = 1,
        .head1x1_groups = 1,
        .groups_input = 1,
        .groups_input_mixin = 1,
    }};
    const child_config = nam_file.WaveNetConfig{
        .layers = &child_layers,
        .head = null,
        .head_scale = 1.0,
        .in_channels = 1,
        .condition_dsp = null,
    };
    const child_weights = try initWaveNetWeights(allocator, &child_config, 123);
    defer allocator.free(child_weights);

    const top_dilations = [_]usize{1};
    const top_kernels = [_]usize{1};
    const top_acts = [_]nam_file.Activation{.{ .kind = .leaky_relu, .negative_slope = 0.1 }};
    const top_secondary = [_]nam_file.Activation{nam_file.Activation.sigmoid_default};
    const top_gating = [_]nam_file.GatingMode{.none};
    const top_layers = [_]nam_file.WaveNetLayerArray{.{
        .input_size = 1,
        .condition_size = 1,
        .channels = 1,
        .bottleneck = 1,
        .head_out = 1,
        .head_kernel = 1,
        .head_bias = true,
        .dilations = &top_dilations,
        .kernel_sizes = &top_kernels,
        .activations = &top_acts,
        .gating_modes = &top_gating,
        .secondary_activations = &top_secondary,
        .layer1x1_active = true,
        .layer1x1_groups = 1,
        .head1x1_active = false,
        .head1x1_out = 1,
        .head1x1_groups = 1,
        .groups_input = 1,
        .groups_input_mixin = 1,
    }};
    const child_dsp = nam_file.ConditionDsp{
        .architecture = .wavenet,
        .config = .{ .wavenet = child_config },
        .weights = child_weights,
        .sample_rate = 48000.0,
    };
    const top_config = nam_file.WaveNetConfig{
        .layers = &top_layers,
        .head = null,
        .head_scale = 1.0,
        .in_channels = 1,
        .condition_dsp = &child_dsp,
    };
    const top_weights = try initWaveNetWeights(allocator, &top_config, 456);
    defer allocator.free(top_weights);

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try A2Trainable.initFromWaveNet(allocator, &ctx, &top_config, top_weights);
    defer model.deinit();

    var opt = fucina.optim.Adam.init(allocator, .{ .lr = 0.01, .weight_decay = 0 });
    defer opt.deinit();
    try model.registerParams(&opt);
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const input = [_]f32{ 0.2, -0.1, 0.4, 0.7 };
        const target = [_]f32{ -0.3, 0.25 };
        const loss = try model.segmentLoss(&ctx, &input, &target);
        try loss.backward(&ctx);
    }
    try opt.step(&ctx);

    var snapshot = try model.extractWaveNetSnapshot(allocator, &top_config);
    defer snapshot.deinit(allocator);
    const snapshot_dsp = snapshot.config.condition_dsp orelse return error.TestExpectedConditionDsp;
    try std.testing.expectEqual(nam_file.Arch.wavenet, snapshot_dsp.architecture);
    try std.testing.expectEqual(@as(usize, child_weights.len), snapshot_dsp.weights.len);
    var child_delta: f32 = 0;
    for (child_weights, snapshot_dsp.weights) |before, after| child_delta += @abs(after - before);
    try std.testing.expect(child_delta > 1e-7);
    try std.testing.expectEqual(@as(usize, top_weights.len), snapshot.weights.len);
}

test "A2 training spec initializes, steps, extracts, and renders through shared path" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var spec = try TrainingSpec.parse("a2-nano");
    try std.testing.expectEqualStrings("a2-nano", spec.name());
    try std.testing.expect(spec.receptiveField() > 0);

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try spec.initTrainable(allocator, &ctx, 123);
    defer model.deinit();
    var opt = fucina.optim.Adam.init(allocator, .{ .lr = 0.001, .weight_decay = 0 });
    defer opt.deinit();
    try model.registerParams(&opt);

    var window: [64]f32 = undefined;
    for (&window, 0..) |*v, i| v.* = 0.25 * @sin(@as(f32, @floatFromInt(i)) * 0.17);
    var target: [16]f32 = undefined;
    for (&target, window[window.len - target.len ..]) |*dst, v| dst.* = 0.5 * std.math.tanh(2.0 * v);

    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const loss = try model.segmentLoss(&ctx, &window, &target);
        try std.testing.expect(std.math.isFinite(try loss.item()));
        try loss.backward(&ctx);
        try opt.step(&ctx);
        opt.zeroGrad();
    }

    const weights = try model.extractWeights(allocator);
    defer allocator.free(weights);
    var config = try spec.makeEngineConfig(allocator);
    defer freeEngineConfig(allocator, &config);
    try std.testing.expectEqual(nam_file.expectedWeightCount(&.{ .wavenet = config }), weights.len);

    var rendered: [64]f32 = undefined;
    try renderTrainingSpec(allocator, &spec, weights, &window, &rendered);
    for (rendered) |v| try std.testing.expect(std.math.isFinite(v));
}

test "MRSTFT loss is finite and differentiable through public Tensor ops" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var pred_values: [32]f32 = undefined;
    var target_values: [32]f32 = undefined;
    for (&pred_values, &target_values, 0..) |*p, *t, i| {
        const x = @as(f32, @floatFromInt(i));
        p.* = 0.4 * @sin(0.21 * x) + 0.05 * @cos(0.07 * x);
        t.* = 0.35 * @sin(0.19 * x + 0.2);
    }
    var pred = try Tensor(.{.time}).variableFromSlice(&ctx, .{pred_values.len}, &pred_values);
    defer pred.deinit();
    var target = try Tensor(.{.time}).fromSlice(&ctx, .{target_values.len}, &target_values);
    defer target.deinit();

    const resolutions = [_]MrstftResolution{.{ .fft_size = 8, .hop_size = 4, .win_length = 8 }};
    const expected_loss: f32 = 0.557691574097;
    const expected_grad = [_]f32{
        -1.161966800690e+00,
        -1.946405768394e-01,
        4.001387119293e+00,
        -7.443846702576e+00,
        1.017409324646e+01,
        -1.110456275940e+01,
        1.081744766235e+01,
        -7.012096405029e+00,
        5.188246726990e+00,
        -6.130017280579e+00,
        1.096848869324e+01,
        -1.349970912933e+01,
        1.354017639160e+01,
        -5.058225154877e+00,
        -1.264877700806e+01,
        3.041127395630e+01,
        -3.885288619995e+01,
        3.476785278320e+01,
        -2.379175376892e+01,
        1.189271545410e+01,
        -8.480752944946e+00,
        9.822794914246e+00,
        -1.085803413391e+01,
        8.347366333008e+00,
        -6.377999305725e+00,
        1.814589023590e+00,
        6.442652225494e+00,
        -1.529901981354e+01,
        2.089739608765e+01,
        -1.747794723511e+01,
        1.007735061646e+01,
        -1.948114752769e+00,
    };
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const loss = try mrstftLoss(&ctx, &pred, &target, .{ .resolutions = &resolutions });
        const value = try loss.item();
        try std.testing.expect(std.math.isFinite(value));
        try std.testing.expectApproxEqAbs(expected_loss, value, 5e-6);
        try loss.backward(&ctx);
    }

    var grad = (try pred.grad(&ctx)).?;
    defer grad.deinit();
    var grad_l1: f32 = 0;
    const grad_values = try grad.dataConst();
    var max_abs: f32 = 0;
    var max_rel: f32 = 0;
    var max_index: usize = 0;
    for (grad_values, expected_grad, 0..) |actual, expected, i| {
        grad_l1 += @abs(actual);
        const abs_err = @abs(actual - expected);
        if (abs_err > max_abs) {
            max_abs = abs_err;
            max_index = i;
        }
        max_rel = @max(max_rel, abs_err / @max(@abs(expected), 1e-6));
    }
    if (!(max_abs < 5e-3 and max_rel < 1e-3)) {
        std.debug.print("MRSTFT torch golden mismatch: max_abs={d} max_rel={d} index={d} actual={d} expected={d}\n", .{
            max_abs,
            max_rel,
            max_index,
            grad_values[max_index],
            expected_grad[max_index],
        });
    }
    try std.testing.expect(max_abs < 5e-3);
    try std.testing.expect(max_rel < 1e-3);
    try std.testing.expect(grad_l1 > 1e-5);
}

test "MRSTFT loss matches torch centered short-window STFT" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const pred_values = [_]f32{
        7.000000000000e-02,
        1.311922334453e-01,
        1.897855726044e-01,
        2.438074885155e-01,
        2.909175947512e-01,
        3.284040709691e-01,
        3.533524913359e-01,
        3.629728170195e-01,
        3.550300728557e-01,
        3.282949339964e-01,
        2.829188094259e-01,
        2.206469345323e-01,
        1.448109304133e-01,
        6.008357067035e-02,
        -2.797549168879e-02,
        -1.135132300547e-01,
        -1.910187565471e-01,
        -2.558956315111e-01,
        -3.048788855311e-01,
    };
    const target_values = [_]f32{
        1.215684949534e-01,
        1.682013922916e-01,
        2.100784987650e-01,
        2.456744058166e-01,
        2.736865759143e-01,
        2.930822763235e-01,
        3.031358100558e-01,
        3.034547038386e-01,
        2.939939103914e-01,
        2.750575136531e-01,
        2.472878753780e-01,
        2.116426135038e-01,
        1.693602406295e-01,
        1.219156990667e-01,
        7.096739254764e-02,
        1.829762070663e-02,
        -3.425144011709e-02,
        -8.484390592860e-02,
        -1.317161227868e-01,
    };
    const expected_grad = [_]f32{
        -4.282745361328e+00,
        7.019609212875e-01,
        4.225007057190e+00,
        -4.557493686676e+00,
        1.268936038017e+00,
        5.728219985962e+00,
        -7.470562458038e+00,
        -5.236407279968e+00,
        2.639431381226e+01,
        -3.814723968506e+01,
        3.659790039062e+01,
        -1.807651710510e+01,
        -9.920687079430e-01,
        6.217460632324e+00,
        -2.566514730453e+00,
        -4.004501342773e+00,
        5.666151523590e+00,
        -5.134201049805e+00,
        7.500512003899e-01,
    };

    var pred = try Tensor(.{.time}).variableFromSlice(&ctx, .{pred_values.len}, &pred_values);
    defer pred.deinit();
    var target = try Tensor(.{.time}).fromSlice(&ctx, .{target_values.len}, &target_values);
    defer target.deinit();

    const resolutions = [_]MrstftResolution{.{ .fft_size = 16, .hop_size = 5, .win_length = 10 }};
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const loss = try mrstftLoss(&ctx, &pred, &target, .{ .resolutions = &resolutions });
        try std.testing.expectApproxEqAbs(@as(f32, 0.594567775726), try loss.item(), 5e-6);
        try loss.backward(&ctx);
    }

    var grad = (try pred.grad(&ctx)).?;
    defer grad.deinit();
    var max_abs: f32 = 0;
    var max_rel: f32 = 0;
    var max_index: usize = 0;
    const grad_values = try grad.dataConst();
    for (grad_values, expected_grad, 0..) |actual, expected, i| {
        const abs_err = @abs(actual - expected);
        if (abs_err > max_abs) {
            max_abs = abs_err;
            max_index = i;
        }
        max_rel = @max(max_rel, abs_err / @max(@abs(expected), 1e-6));
    }
    if (!(max_abs < 5e-3 and max_rel < 2e-3)) {
        std.debug.print("MRSTFT short-window torch golden mismatch: max_abs={d} max_rel={d} index={d} actual={d} expected={d}\n", .{
            max_abs,
            max_rel,
            max_index,
            grad_values[max_index],
            expected_grad[max_index],
        });
    }
    try std.testing.expect(max_abs < 5e-3);
    try std.testing.expect(max_rel < 2e-3);
}

test "packed WaveNet spec sums submodel losses and extracts slimmable snapshots" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var spec = try TrainingSpec.parse("packed");
    try std.testing.expectEqualStrings("packed", spec.name());
    try std.testing.expectEqual(PackedSpec.default_epochs, spec.defaultEpochs());
    try std.testing.expectApproxEqAbs(PackedSpec.default_mrstft_weight, spec.defaultMrstftWeight(), 1e-9);

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try spec.initTrainable(allocator, &ctx, 321);
    defer model.deinit();
    var opt = fucina.optim.Adam.init(allocator, .{ .lr = 0.001, .weight_decay = PackedSpec.default_weight_decay });
    defer opt.deinit();
    try model.registerParams(&opt);

    var window: [64]f32 = undefined;
    for (&window, 0..) |*v, i| v.* = 0.25 * @sin(@as(f32, @floatFromInt(i)) * 0.17);
    var target: [16]f32 = undefined;
    for (&target, window[window.len - target.len ..]) |*dst, v| dst.* = 0.6 * std.math.tanh(1.7 * v);

    const resolutions = [_]MrstftResolution{.{ .fft_size = 8, .hop_size = 4, .win_length = 8 }};
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const loss = try model.segmentLossWithOptions(&ctx, &window, &target, .{
            .mrstft_weight = 0.001,
            .mrstft = .{ .resolutions = &resolutions },
        });
        try std.testing.expect(std.math.isFinite(try loss.item()));
        try loss.backward(&ctx);
        try opt.step(&ctx);
        opt.zeroGrad();
    }

    var snapshot = try model.extractTrainingSnapshot(allocator, null);
    defer snapshot.deinit(allocator);
    const packed_snapshot = switch (snapshot) {
        .packed_wavenet => |*s| s,
        else => return error.TestExpectedPackedSnapshot,
    };
    try std.testing.expectEqual(@as(usize, 2), packed_snapshot.submodels.len);
    try std.testing.expectEqual(@as(usize, 3), packed_snapshot.submodels[0].config.layers[0].channels);
    try std.testing.expectEqual(@as(usize, 8), packed_snapshot.submodels[1].config.layers[0].channels);
    try std.testing.expectEqual(nam_file.expectedWeightCount(&.{ .wavenet = packed_snapshot.submodels[0].config }), packed_snapshot.submodels[0].weights.len);
    try std.testing.expectEqual(nam_file.expectedWeightCount(&.{ .wavenet = packed_snapshot.submodels[1].config }), packed_snapshot.submodels[1].weights.len);
}

test "A2 trainable forward matches the streaming engine on the upstream max fixture when present" {
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

    var model = try A2Trainable.initFromWaveNet(allocator, &ctx, &file_model.config.wavenet, file_model.weights);
    defer model.deinit();
    const roundtrip = try model.extractWeights(allocator);
    defer allocator.free(roundtrip);
    try std.testing.expectEqualSlices(f32, file_model.weights, roundtrip);

    const frames = 257;
    const out_channels = model.outputChannels();
    var input: [frames]f32 = undefined;
    for (&input, 0..) |*v, i| {
        v.* = 0.35 * @sin(@as(f32, @floatFromInt(i)) * 0.071) + 0.12 * @cos(@as(f32, @floatFromInt(i)) * 0.019);
    }

    const graph_pred = try allocator.alloc(f32, frames * out_channels);
    defer allocator.free(graph_pred);
    const scope = ctx.openExecScope();
    {
        const pred = try model.forward(&ctx, &input);
        try pred.copyTo(graph_pred);
    }
    ctx.closeExecScope(scope);

    var const_model = try A2Trainable.initConstFromWaveNet(allocator, &ctx, &file_model.config.wavenet, file_model.weights);
    defer const_model.deinit();
    const const_pred = try allocator.alloc(f32, frames * out_channels);
    defer allocator.free(const_pred);
    try const_model.renderBorrowed(&ctx, &input, const_pred);

    var engine = try wavenet.WaveNetEngine.init(allocator, &file_model.config.wavenet, file_model.weights);
    defer engine.deinit();
    try engine.reset(frames);
    const engine_pred = try allocator.alloc(f32, frames * out_channels);
    defer allocator.free(engine_pred);
    engine.process(&input, engine_pred, frames);

    var max_abs: f32 = 0;
    for (graph_pred, engine_pred) |a, b| max_abs = @max(max_abs, @abs(a - b));
    try std.testing.expect(max_abs < 2e-5);
    max_abs = 0;
    for (const_pred, engine_pred) |a, b| max_abs = @max(max_abs, @abs(a - b));
    try std.testing.expect(max_abs < 2e-5);
}
