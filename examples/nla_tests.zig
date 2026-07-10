//! Unit tests for the NLA example's pure helpers (`nla.zig`): corpus split,
//! placeholder location, SFT sample construction/masking, and the
//! normalization / round-trip metric math. Model-dependent behavior (the
//! forwardHidden / lossInjected seam) is covered by the trainer tests in
//! src/llm/qwen3/train_tests.zig; the end-to-end loop needs a real GGUF and
//! runs via `zig build nla -- --demo`.
const std = @import("std");
const nla = @import("nla.zig");
const llm = @import("fucina_llm");

const ignore_index = llm.qwen3.train.ignore_index;

test "splitCounts holds out the tail fifth, at least one, never everything" {
    try std.testing.expectEqual(nla.Split{ .train = 0, .eval = 0 }, nla.splitCounts(0));
    try std.testing.expectEqual(nla.Split{ .train = 1, .eval = 0 }, nla.splitCounts(1));
    try std.testing.expectEqual(nla.Split{ .train = 1, .eval = 1 }, nla.splitCounts(2));
    try std.testing.expectEqual(nla.Split{ .train = 4, .eval = 1 }, nla.splitCounts(5));
    try std.testing.expectEqual(nla.Split{ .train = 8, .eval = 2 }, nla.splitCounts(10));
    try std.testing.expectEqual(nla.Split{ .train = 80, .eval = 20 }, nla.splitCounts(100));
    // Consistency: the two parts always recompose the corpus.
    for ([_]usize{ 0, 1, 2, 3, 7, 10, 99 }) |n| {
        const split = nla.splitCounts(n);
        try std.testing.expectEqual(n, split.train + split.eval);
    }
}

test "findOnce: exactly-once hits, misses, and duplicates" {
    const ids = [_]usize{ 5, 9, 151662, 13, 2 };
    try std.testing.expectEqual(@as(?usize, 2), nla.findOnce(&ids, 151662));
    try std.testing.expectEqual(@as(?usize, 0), nla.findOnce(&ids, 5));
    try std.testing.expectEqual(@as(?usize, null), nla.findOnce(&ids, 42));
    const dup = [_]usize{ 1, 7, 1 };
    try std.testing.expectEqual(@as(?usize, null), nla.findOnce(&dup, 1));
}

test "buildSample: shift-by-one with the prompt masked and the response supervised" {
    const allocator = std.testing.allocator;
    const prompt = [_]usize{ 10, 11, 12 };
    const response = [_]usize{ 20, 21 };
    var sample = try nla.buildSample(allocator, &prompt, &response);
    defer sample.deinit(allocator);

    // inputs = full sequence minus the last token.
    try std.testing.expectEqualSlices(usize, &.{ 10, 11, 12, 20 }, sample.inputs);
    // labels: position i-1 predicts token i; prompt positions masked. The
    // LAST prompt position predicts the first response token, so it IS
    // supervised — the boundary the AV loss trains hardest.
    try std.testing.expectEqualSlices(usize, &.{ ignore_index, ignore_index, 20, 21 }, sample.labels);

    // Degenerate inputs are rejected.
    try std.testing.expectError(error.EmptySample, nla.buildSample(allocator, &.{}, &response));
    try std.testing.expectError(error.EmptySample, nla.buildSample(allocator, &prompt, &.{}));
}

test "l2NormalizeInPlace: unit norm, reported magnitude, zero-vector guard" {
    var v = [_]f32{ 3, 4 };
    const norm = nla.l2NormalizeInPlace(&v);
    try std.testing.expectApproxEqAbs(@as(f64, 5), norm, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), v[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), v[1], 1e-6);

    var zero = [_]f32{ 0, 0, 0 };
    try std.testing.expectEqual(@as(f64, 0), nla.l2NormalizeInPlace(&zero));
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 0 }, &zero);
}

test "cosine and the round-trip metric 2(1-cos)" {
    const a = [_]f32{ 1, 0, 0 };
    const b = [_]f32{ 0, 1, 0 };
    const c = [_]f32{ -1, 0, 0 };
    try std.testing.expectApproxEqAbs(@as(f64, 1), nla.cosine(&a, &a), 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 0), nla.cosine(&a, &b), 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, -1), nla.cosine(&a, &c), 1e-9);
    // Scale invariance.
    const a2 = [_]f32{ 2, 0, 0 };
    try std.testing.expectApproxEqAbs(@as(f64, 1), nla.cosine(&a, &a2), 1e-9);
    // Zero vectors report 0, not NaN.
    const zero = [_]f32{ 0, 0, 0 };
    try std.testing.expectEqual(@as(f64, 0), nla.cosine(&a, &zero));

    try std.testing.expectApproxEqAbs(@as(f64, 0), nla.roundTripMse(1), 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 2), nla.roundTripMse(0), 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 4), nla.roundTripMse(-1), 1e-9);
}
