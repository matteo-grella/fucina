//! Recorded-logits anchor for the deepseek2 port: a real DeepSeek-V2-Lite
//! forward pinned against recorded values — greedy argmax chain everywhere,
//! exact FNV-1a logit bits on the recording machine class (aarch64 native).
//! Skips without models/. Native-only (model wiring, not kernels).

const std = @import("std");
const test_support = @import("../test_support.zig");
const builtin = @import("builtin");
const fucina = @import("fucina");
const deepseek2 = @import("model.zig");

const ExecContext = fucina.ExecContext;

const strict_bits = builtin.cpu.arch == .aarch64;

fn argmaxRow(row: []const f32) usize {
    var best: usize = 0;
    for (row, 0..) |x, i| {
        if (x > row[best]) best = i;
    }
    return best;
}

fn fnvHash(values: []const f32) u64 {
    var h: u64 = 0xcbf29ce484222325;
    for (std.mem.sliceAsBytes(values)) |b| {
        h ^= b;
        h *%= 0x100000001b3;
    }
    return h;
}

test "deepseek2 matches the recorded DeepSeek-V2-Lite forward (Q8_0; skips without models/)" {
    try test_support.requireNative();
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var file = try test_support.openGgufOrSkip(allocator, std.testing.io, "models/DeepSeek-V2-Lite-Chat.Q8_0.gguf");
    defer file.deinit();

    var model = try deepseek2.Model.loadGgufFromFile(&ctx, &file);
    defer model.deinit();

    var cache = try model.initCache(16);
    defer cache.deinit();

    // BOS + a short prompt; recorded 2026-08-23 (aarch64 native, Accelerate).
    const prompt = [_]usize{ 100000, 5726, 207, 12 };
    const golden_chain = [_]usize{ 59948, 220, 11598, 11 };
    const golden_hash: u64 = 0xb446357d85ea25f0;

    var logits = try model.stepBatch(&ctx, &cache, &prompt);
    if (strict_bits) try std.testing.expectEqual(golden_hash, fnvHash(try logits.dataConst()));
    for (golden_chain[0 .. golden_chain.len - 1]) |want| {
        const next = argmaxRow(try logits.dataConst());
        try std.testing.expectEqual(want, next);
        logits.deinit();
        logits = try model.step(&ctx, &cache, next);
    }
    try std.testing.expectEqual(golden_chain[golden_chain.len - 1], argmaxRow(try logits.dataConst()));
    logits.deinit();
}
