//! Behavioral tests for the streaming NAM engines (`models.zig`): the
//! Linear engine against upstream's known-values test, and the LSTM engine's
//! deterministic settled output (including chunked-after-reset reproduction).

const std = @import("std");
const models = @import("models.zig");
const nam_file = @import("nam_file.zig");
const activations = @import("activations.zig");
const stream_conv = @import("stream_conv.zig");

const StreamConv = stream_conv.StreamConv;
const LstmEngine = models.LstmEngine;
const LinearEngine = models.LinearEngine;

test "linear engine matches the upstream known-values test" {
    // Core/tools/test/test_linear.cpp:101-113: weights {0.5,-0.25,0.125},
    // input {1,2,3,4} => output {0.5, 0.75, 1.125, 1.5}.
    const allocator = std.testing.allocator;
    const config = nam_file.LinearConfig{ .receptive_field = 3, .bias = false, .in_channels = 1, .out_channels = 1 };
    var engine = try LinearEngine.init(allocator, &config, &.{ 0.5, -0.25, 0.125 });
    defer engine.deinit();

    const input = [_]f32{ 1, 2, 3, 4 };
    var output: [4]f32 = undefined;
    engine.process(&input, &output, 4);
    const expected = [_]f32{ 0.5, 0.75, 1.125, 1.5 };
    for (expected, output) |e, o| try std.testing.expectApproxEqAbs(e, o, 1e-6);
}

test "lstm engine on the upstream example produces deterministic settled output" {
    const allocator = std.testing.allocator;
    var model = try nam_file.loadFromSlice(allocator, @embedFile("testdata/lstm.nam"));
    defer model.deinit();

    var engine = try LstmEngine.init(allocator, &model.config.lstm, model.weights, model.sample_rate);
    defer engine.deinit();
    try std.testing.expectEqual(@as(usize, 24000), engine.prewarmSamples());

    var input: [256]f32 = undefined;
    for (&input, 0..) |*v, i| v.* = 0.25 * @sin(@as(f32, @floatFromInt(i)) * 0.1);
    var out_a: [256]f32 = undefined;
    var out_b: [256]f32 = undefined;

    engine.process(&input, &out_a, 256);
    engine.reset();
    // Chunked processing after reset reproduces the same stream.
    engine.process(input[0..100], out_b[0..100], 100);
    engine.process(input[100..], out_b[100..], 156);
    try std.testing.expectEqualSlices(f32, &out_a, &out_b);
}
