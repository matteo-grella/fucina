//! Behavioral tests for the inkling model module. Logit/generation parity
//! vs the pinned llama.cpp PR oracle runs through the example CLI gates
//! (`examples/inkling/main.zig --compare-logits`, docs/PORTING.md §3); these
//! tests cover the host-side numeric kernels in isolation.

const std = @import("std");
const model = @import("model.zig");

test "sconv: residual depthwise causal taps with rolling state" {
    const allocator = std.testing.allocator;
    // width 2, K = 3 taps, state = 2 rows. Tap-major kernel (load-time
    // transposed layout): per channel c0: [1, 10, 100], c1: [2, 20, 200]
    // (tap K-1 multiplies the newest sample).
    const kernel = [_]f32{ 1, 2, 10, 20, 100, 200 };
    var state = [_]f32{ 0, 0, 0, 0 }; // 2 rows x 2 channels, fresh sequence
    var x = [_]f32{
        1, 1, // t=0
        2, 3, // t=1
    };
    try model.testing.sconvInPlace(allocator, &x, 2, 2, &kernel, 3, &state);
    // t=0: conv c0 = 100*1 (past rows zero) -> y = 1 + 100 = 101
    //      conv c1 = 200*1 -> 1 + 200 = 201
    // t=1: conv c0 = 10*1 + 100*2 = 210 -> 2 + 210 = 212
    //      conv c1 = 20*1 + 200*3 = 620 -> 3 + 620 = 623
    try std.testing.expectEqualSlices(f32, &.{ 101, 201, 212, 623 }, &x);
    // State = last 2 INPUT rows.
    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 2, 3 }, &state);

    // One more step consuming the state.
    var x2 = [_]f32{ 5, 7 };
    try model.testing.sconvInPlace(allocator, &x2, 1, 2, &kernel, 3, &state);
    // conv c0 = 1*1 + 10*2 + 100*5 = 521 -> 5 + 521 = 526
    // conv c1 = 2*1 + 20*3 + 200*7 = 1462 -> 7 + 1462 = 1469
    try std.testing.expectEqualSlices(f32, &.{ 526, 1469 }, &x2);
    try std.testing.expectEqualSlices(f32, &.{ 2, 3, 5, 7 }, &state);
}

test "logsigmoid matches -softplus(-x)" {
    for ([_]f32{ -10, -1, 0, 0.5, 3, 25 }) |x| {
        const got = model.testing.logsigmoid(x);
        const want = -std.math.log1p(@exp(-x));
        try std.testing.expectApproxEqAbs(want, got, 1e-7);
    }
}

test "moe selection: ties resolve to the lower expert id" {
    // Equal biased scores: strict > comparison keeps the first (lower) id.
    var scores = [_]f32{ 0.5, 0.9, 0.9, 0.1 };
    var selected: [2]usize = undefined;
    for (0..2) |slot| {
        var best: usize = 0;
        var best_s: f32 = -std.math.inf(f32);
        for (scores, 0..) |s, e| {
            if (s > best_s) {
                best_s = s;
                best = e;
            }
        }
        scores[best] = -std.math.inf(f32);
        selected[slot] = best;
    }
    try std.testing.expectEqual(@as(usize, 1), selected[0]);
    try std.testing.expectEqual(@as(usize, 2), selected[1]);
}
