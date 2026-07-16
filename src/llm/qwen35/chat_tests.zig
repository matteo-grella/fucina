//! Rendering tests for the Qwen3.5/Bonsai ChatML prompt: the shared ChatML
//! renderer plus the Qwen3.6 generation-prompt think prefill, pinned against
//! the forms the GGUF's own chat template produces (non-tool text path).

const std = @import("std");
const chat = @import("../chat.zig");
const qwen35_chat = @import("chat.zig");

const Message = chat.Message;
const template = chat.Template{ .format = .chatml };

fn expectRender(messages: []const Message, think_off: bool, expected: []const u8) !void {
    const a = std.testing.allocator;
    const got = try qwen35_chat.renderPrompt(a, template, messages, .{ .think_off = think_off });
    defer a.free(got);
    try std.testing.expectEqualStrings(expected, got);
}

test "thinking on prefills the think opener in the generation prompt" {
    try expectRender(&.{
        .{ .role = .system, .content = "Be terse." },
        .{ .role = .user, .content = "Hi" },
    }, false,
        "<|im_start|>system\nBe terse.<|im_end|>\n" ++
            "<|im_start|>user\nHi<|im_end|>\n" ++
            "<|im_start|>assistant\n<think>\n");
}

test "thinking off prefills the empty think block" {
    try expectRender(&.{
        .{ .role = .user, .content = "Hi" },
    }, true,
        "<|im_start|>user\nHi<|im_end|>\n" ++
            "<|im_start|>assistant\n<think>\n\n</think>\n\n");
}

test "historical assistant turns render stripped of reasoning" {
    try expectRender(&.{
        .{ .role = .user, .content = "2+2?" },
        .{ .role = .assistant, .content = "<think>\nsum\n</think>\n\n4" },
        .{ .role = .user, .content = "And +1?" },
    }, true,
        "<|im_start|>user\n2+2?<|im_end|>\n" ++
            "<|im_start|>assistant\n4<|im_end|>\n" ++
            "<|im_start|>user\nAnd +1?<|im_end|>\n" ++
            "<|im_start|>assistant\n<think>\n\n</think>\n\n");
}
