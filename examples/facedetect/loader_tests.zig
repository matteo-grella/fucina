const std = @import("std");
const fucina = @import("fucina");
const gguf = fucina.gguf;
const loader = @import("loader.zig");

test "Submodel.ofTensor classifies verbatim GGUF prefixes" {
    const S = loader.Submodel;
    try std.testing.expectEqual(S.det, S.ofTensor("det.stage1.conv.weight"));
    try std.testing.expectEqual(S.rec, S.ofTensor("rec.fc.weight"));
    try std.testing.expectEqual(S.ga, S.ofTensor("ga.conv_1_conv2d_weight"));
    try std.testing.expectEqual(S.antispoof, S.ofTensor("as0.conv1.weight"));
    try std.testing.expectEqual(S.antispoof, S.ofTensor("as1.se.fc.weight"));
    try std.testing.expectEqual(S.antispoof, S.ofTensor("as10.x")); // multi-digit index
    try std.testing.expectEqual(S.landmark2d, S.ofTensor("l2d.fc1.weight"));
    try std.testing.expectEqual(S.landmark3d, S.ofTensor("l3d.bn1.weight"));
    // Non-members must NOT masquerade as a sub-model.
    try std.testing.expectEqual(S.other, S.ofTensor("as.foo")); // no digit
    try std.testing.expectEqual(S.other, S.ofTensor("asx.foo"));
    try std.testing.expectEqual(S.other, S.ofTensor("detector.bias")); // "det" but not "det."
    try std.testing.expectEqual(S.other, S.ofTensor("general.name"));
}

// Integration gates against the pinned reference GGUFs (refs @ e22260d5). They
// SKIP without the weights so `zig build test` stays green in CI; run locally
// with `models/buffalo_l.gguf` and `models/landmarks-*.gguf` present.

test "buffalo_l loader: total tensor coverage + dtype class + graph-KV ignored (skips without models/)" {
    const allocator = std.testing.allocator;
    var file = gguf.File.loadMmap(allocator, std.testing.io, "models/buffalo_l.gguf") catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer file.deinit();

    // Expected per-sub-model counts, pinned to refs/face-detect.cpp @ e22260d5
    // (verified by parsing the GGUF header): det 125, rec 237, ga 161,
    // anti-spoof 286 (as0 137 + as1 149) → 809 total. Note det.392 is a legitimate
    // 0-element tensor (rank-1 [0]); it still counts and must load (see src/gguf.zig).
    const cov = loader.coverage(&file);
    try std.testing.expectEqual(@as(usize, 809), cov.total);
    try std.testing.expectEqual(@as(usize, 125), cov.det);
    try std.testing.expectEqual(@as(usize, 237), cov.rec);
    try std.testing.expectEqual(@as(usize, 161), cov.ga);
    try std.testing.expectEqual(@as(usize, 286), cov.antispoof); // as0 137 + as1 149
    try std.testing.expectEqual(@as(usize, 0), cov.other);
    try loader.assertFullCoverage(&file); // resolved == total
    try loader.assertKnownDtypes(&file);

    // The anti-spoof graph KVs EXIST — antispoof.zig replays them through the
    // app-level graph.zig interpreter at runtime. The coverage above is
    // computed purely from tensors, never from these KVs.
    try std.testing.expect(file.getArray("facedetect.antispoof.0.graph") != null);
    try std.testing.expect(file.getArray("facedetect.antispoof.1.graph") != null);
    // The hand-mapped heads carry NO graph KV at all.
    try std.testing.expect(file.getArray("facedetect.detector.graph") == null);
    try std.testing.expect(file.getArray("facedetect.recognizer.graph") == null);
}

test "landmarks loader: total tensor coverage (skips without models/)" {
    const allocator = std.testing.allocator;
    var file = gguf.File.loadMmap(allocator, std.testing.io, "models/landmarks-2d106-1k3d68.gguf") catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer file.deinit();

    // Pinned counts (e22260d5): l2d 172, l3d 264.
    const cov = loader.coverage(&file);
    try std.testing.expectEqual(@as(usize, 436), cov.total);
    try std.testing.expectEqual(@as(usize, 172), cov.landmark2d);
    try std.testing.expectEqual(@as(usize, 264), cov.landmark3d);
    try std.testing.expectEqual(@as(usize, 0), cov.other);
    try loader.assertFullCoverage(&file);
    try loader.assertKnownDtypes(&file);
}
