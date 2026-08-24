//! Decoder-contract checks: the conforming families instantiate
//! `assertDecoder` at comptime, and their declared `caps` match what the
//! generic layers rely on.

const std = @import("std");
const decoder = @import("decoder.zig");
const runner = @import("qwen3/runner.zig");
const gemma = @import("gemma/model.zig");
const shine = @import("qwen3/shine.zig");
const qwen35 = @import("qwen35/model.zig");
const deepseek2 = @import("deepseek2/model.zig");
const glm4moe = @import("glm4moe/model.zig");
const inkling = @import("inkling/model.zig");
const deepseek4 = @import("deepseek4/model.zig");
const kv_cache = @import("text/kv_cache.zig");

test "assertDecoder: runner (qwen3/qwen3moe), gemma4, shine AdaptedModel conform" {
    comptime decoder.assertDecoder(runner.Model);
    comptime decoder.assertDecoder(gemma.Model);
    comptime decoder.assertDecoder(shine.AdaptedModel);
}

test "assertDecoder: qwen35 conforms (recurrent cache, no rewind, no batch)" {
    comptime decoder.assertDecoder(qwen35.Model);
    try std.testing.expectEqual(decoder.Caps{ .rewind = false, .batch = false }, qwen35.Model.caps);
}

test "assertDecoder: deepseek2 conforms (MLA cache, no rewind, no batch)" {
    comptime decoder.assertDecoder(deepseek2.Model);
    try std.testing.expectEqual(decoder.Caps{ .rewind = false, .batch = false }, deepseek2.Model.caps);
}

test "assertDecoder: inkling conforms (shortconv states, no rewind, no batch)" {
    comptime decoder.assertDecoder(inkling.Model);
    try std.testing.expectEqual(decoder.Caps{ .rewind = false, .batch = false }, inkling.Model.caps);
}

test "assertDecoder: glm4moe conforms (host cache rewinds, mutable-self forwards)" {
    comptime decoder.assertDecoder(glm4moe.Model);
    try std.testing.expectEqual(decoder.Caps{ .rewind = true, .batch = false }, glm4moe.Model.caps);
    comptime std.debug.assert(decoder.ModelPtr(glm4moe.Model) == *glm4moe.Model);
    // The MTP draft adapter and the decoder instantiate over the family.
    const mtp = @import("text/speculative/mtp.zig");
    const core = @import("text/speculative/core.zig");
    comptime {
        _ = mtp.MtpDraftSource(glm4moe.Model);
        _ = core.SpeculativeDecoder(glm4moe.Model);
    }
}

test "assertDecoder: deepseek4 conforms (Session decode state, no rewind, no batch)" {
    comptime decoder.assertDecoder(deepseek4.Model);
    try std.testing.expectEqual(decoder.Caps{ .rewind = false, .batch = false }, deepseek4.Model.caps);
    comptime std.debug.assert(deepseek4.Model.Cache == deepseek4.Session);
}

test "caps: rewind and batch match each family's surface" {
    try std.testing.expectEqual(decoder.Caps{ .rewind = true, .batch = true }, runner.Model.caps);
    try std.testing.expectEqual(decoder.Caps{ .rewind = true, .batch = true }, gemma.Model.caps);
    // The AdaptedModel box holds one adapter at a time; batch decode is
    // per-slot-heterogeneous, so the family declares batch = false.
    try std.testing.expectEqual(decoder.Caps{ .rewind = true, .batch = false }, shine.AdaptedModel.caps);
}

test "shared KvCache satisfies the contract's Cache surface" {
    comptime {
        std.debug.assert(runner.Model.Cache == kv_cache.KvCache);
        std.debug.assert(gemma.Model.Cache == kv_cache.KvCache);
        std.debug.assert(shine.AdaptedModel.Cache == kv_cache.KvCache);
    }
    // len() is the method; count is the state it reads.
    var cache: kv_cache.KvCache = undefined;
    cache.count = 3;
    try std.testing.expectEqual(@as(usize, 3), cache.len());
}

test "ModelPtr: const-self families resolve to *const Model" {
    comptime {
        std.debug.assert(decoder.ModelPtr(runner.Model) == *const runner.Model);
        std.debug.assert(decoder.ModelPtr(gemma.Model) == *const gemma.Model);
    }
}
