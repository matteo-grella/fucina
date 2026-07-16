//! Golden tests for the Inkling wire-format renderer. Expected strings are
//! byte-exact captures of llama.cpp's own minja rendering of the pinned
//! `models/templates/Inkling.jinja` (via common_chat_templates_apply; the
//! render oracle is documented in the port log). Regenerate the goldens by
//! re-running that oracle if the pinned template changes.

const std = @import("std");
const chat = @import("chat.zig");
const msg_mod = @import("../chat.zig");

const Message = msg_mod.Message;

test "render: single user (effort line auto-injected before it)" {
    const a = std.testing.allocator;
    const messages = [_]Message{.{ .role = .user, .content = "Hi!" }};
    const out = try chat.renderPrompt(a, &messages, .{});
    defer a.free(out);
    try std.testing.expectEqualStrings(
        "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>" ++
            "<|message_user|><|content_text|>Hi!<|end_message|>" ++
            "<|message_model|>",
        out,
    );
}

test "render: system then user (effort line after the real system)" {
    const a = std.testing.allocator;
    const messages = [_]Message{
        .{ .role = .system, .content = "You are helpful." },
        .{ .role = .user, .content = "Hi!" },
    };
    const out = try chat.renderPrompt(a, &messages, .{});
    defer a.free(out);
    try std.testing.expectEqualStrings(
        "<|message_system|><|content_text|>You are helpful.<|end_message|>" ++
            "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>" ++
            "<|message_user|><|content_text|>Hi!<|end_message|>" ++
            "<|message_model|>",
        out,
    );
}

test "render: multi-turn with assistant history (end_sampling after assistant)" {
    const a = std.testing.allocator;
    const messages = [_]Message{
        .{ .role = .user, .content = "2+2?" },
        .{ .role = .assistant, .content = "4" },
        .{ .role = .user, .content = "x3?" },
    };
    const out = try chat.renderPrompt(a, &messages, .{});
    defer a.free(out);
    try std.testing.expectEqualStrings(
        "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>" ++
            "<|message_user|><|content_text|>2+2?<|end_message|>" ++
            "<|message_model|><|content_text|>4<|end_message|><|content_model_end_sampling|>" ++
            "<|message_user|><|content_text|>x3?<|end_message|>" ++
            "<|message_model|>",
        out,
    );
}

test "render: system + multi-turn (oracle-captured, all rules at once)" {
    const a = std.testing.allocator;
    const messages = [_]Message{
        .{ .role = .system, .content = "Be brief." },
        .{ .role = .user, .content = "2+2?" },
        .{ .role = .assistant, .content = "4" },
        .{ .role = .user, .content = "and x3?" },
    };
    const out = try chat.renderPrompt(a, &messages, .{});
    defer a.free(out);
    try std.testing.expectEqualStrings(
        "<|message_system|><|content_text|>Be brief.<|end_message|>" ++
            "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>" ++
            "<|message_user|><|content_text|>2+2?<|end_message|>" ++
            "<|message_model|><|content_text|>4<|end_message|><|content_model_end_sampling|>" ++
            "<|message_user|><|content_text|>and x3?<|end_message|>" ++
            "<|message_model|>",
        out,
    );
}

test "render: think_off primes a content_text block after the generation prompt" {
    const a = std.testing.allocator;
    const messages = [_]Message{.{ .role = .user, .content = "Hi!" }};
    const out = try chat.renderPrompt(a, &messages, .{ .think_off = true });
    defer a.free(out);
    try std.testing.expect(std.mem.endsWith(u8, out, "<|message_model|><|content_text|>"));
}

test "render: validation rejects empty and trailing-assistant" {
    const a = std.testing.allocator;
    try std.testing.expectError(chat.Error.EmptyMessages, chat.renderPrompt(a, &.{}, .{}));
    const trailing = [_]Message{
        .{ .role = .user, .content = "hi" },
        .{ .role = .assistant, .content = "hello" },
    };
    try std.testing.expectError(chat.Error.TrailingAssistantMessage, chat.renderPrompt(a, &trailing, .{}));
}
