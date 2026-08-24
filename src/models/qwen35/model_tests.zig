//! Behavioral tests for the Qwen3.5 (`qwen35`) module (`qwen35.zig`): the
//! causal depthwise conv1d used by the DeltaNet linear blocks — hand-computed
//! prefill reference and streaming-state continuation across a split sequence.
const std = @import("std");
const qwen35 = @import("model.zig");
const fucina = @import("fucina");

const ExecContext = fucina.ExecContext;

test "causalDepthwiseConv1d matches a hand-computed reference (prefill, zero state)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const d_conv = 3;
    const ch = 2;
    const seq = 4;
    // input[token][channel]: t*10 + c
    const in_data = [_]f32{ 0, 1, 10, 11, 20, 21, 30, 31 };
    // kernel[channel][tap]: ch0 = [1,2,3], ch1 = [10,20,30]
    const kernel = [_]f32{ 1, 2, 3, 10, 20, 30 };
    var input = try fucina.Tensor(.{ .seq, .conv }).fromSlice(&ctx, .{ seq, ch }, &in_data);
    defer input.deinit();
    var kernel_t = try fucina.Tensor(.{ .conv, .tap }).fromSlice(&ctx, .{ ch, d_conv }, &kernel);
    defer kernel_t.deinit();

    var out = try input.causalDepthwiseConv1d(&ctx, .seq, .conv, .tap, &kernel_t, 1, null);
    defer out.deinit();
    const od = try out.dataConst();
    // Hand-computed: out[t][c] = Σ token[t-2+k][c]·kernel[c][k], negatives = 0.
    const expect = [_]f32{ 0, 30, 30, 350, 80, 860, 140, 1460 };
    for (expect, 0..) |e, i| try std.testing.expectApproxEqAbs(e, od[i], 1e-4);
}

test "causalDepthwiseConv1d streaming state continues a split sequence" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const d_conv = 3;
    const ch = 2;
    const kernel = [_]f32{ 1, 2, 3, 10, 20, 30 };

    // Second chunk = tokens [2,3], seeded with the previous d_conv-1=2 tokens
    // (from the full sequence {0,1, 10,11, 20,21, 30,31} used in the prefill test).
    const chunk2 = [_]f32{ 20, 21, 30, 31 };
    const state = [_]f32{ 0, 1, 10, 11 }; // tokens [0,1] as state rows
    var input = try fucina.Tensor(.{ .seq, .conv }).fromSlice(&ctx, .{ 2, ch }, &chunk2);
    defer input.deinit();
    var kernel_t = try fucina.Tensor(.{ .conv, .tap }).fromSlice(&ctx, .{ ch, d_conv }, &kernel);
    defer kernel_t.deinit();

    var out = try input.causalDepthwiseConv1d(&ctx, .seq, .conv, .tap, &kernel_t, 1, &state);
    defer out.deinit();
    const od = try out.dataConst();
    // Must equal the prefill output's tokens [2,3]: {80,860, 140,1460}.
    const expect = [_]f32{ 80, 860, 140, 1460 };
    for (expect, 0..) |e, i| try std.testing.expectApproxEqAbs(e, od[i], 1e-4);
}
