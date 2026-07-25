//! Anthropic Messages API wire layer (`POST /v1/messages`): a translation
//! layer, not a second engine path — requests normalize into the same
//! `openai.Parsed` the OpenAI dialects produce, so prompt rendering,
//! scheduling, and generation stay single-sourced; only request parsing
//! (here) and the response/SSE framing (`emitter.zig`, `.anthropic` wire)
//! are Anthropic-shaped. Claude Code is the reference client.
//!
//! Deliberate deviations from the hosted API, all visible on the wire:
//! * `tools` declarations are accepted and dropped (never rendered into the
//!   prompt): the models served here are not tool-trained, and under
//!   `tool_choice: auto` a reply with no `tool_use` blocks is valid
//!   behavior. `tool_choice` `any`/`tool` (a guaranteed tool call) is
//!   rejected, as are `tool_use`/`tool_result` blocks in the history — this
//!   server never emits `tool_use`, so no conversation held with it
//!   contains them.
//! * `thinking: {"type": "enabled"|"adaptive"}` switches the model's
//!   reasoning channel on when the backend has one and is otherwise a no-op
//!   (the reply simply carries no `thinking` blocks) — a hard error would
//!   make thinking-by-default clients unusable against non-reasoning
//!   models.
//! * `stop_sequences` are rejected: the engine could stop on them, but it
//!   does not report which sequence fired, and misreporting
//!   `stop_reason`/`stop_sequence` would be worse than refusing.
//! * Mid-conversation `{"role":"system"}` messages (Claude Code appends
//!   them) fold into a user turn as a `<system-reminder>` block — the
//!   hosted API's degradation path for models without the feature; the
//!   templates here have a single leading system slot.
//! * `output_config.format` with `{"type": "json_schema", "schema": ...}`
//!   maps to the same llguidance constraint as the OpenAI dialects
//!   (requires a `-Dllguidance=true` build).
//!
//! Errors use the Anthropic envelope `{"type":"error","error":{type,
//! message}}` with the type derived from the HTTP status; the envelope has
//! no `param` field, so offending fields are named in the message.

const std = @import("std");
const llm = @import("fucina_llm");
const types = @import("types.zig");
const openai = @import("openai.zig");

const Allocator = std.mem.Allocator;
const Value = std.json.Value;
const ErrorInfo = openai.ErrorInfo;

/// Anthropic error `type` for an HTTP status.
pub fn errorTypeName(status: std.http.Status) []const u8 {
    return switch (status) {
        // 501 (constrained output on a build without llguidance) has no
        // Anthropic type of its own: the request asked for something this
        // server cannot do.
        .bad_request, .not_implemented => "invalid_request_error",
        .unauthorized => "authentication_error",
        .forbidden => "permission_error",
        .not_found => "not_found_error",
        .payload_too_large => "request_too_large",
        .too_many_requests => "rate_limit_error",
        .service_unavailable => "overloaded_error",
        else => "api_error",
    };
}

/// Serialize the Anthropic error body for `info`.
pub fn writeErrorBody(info: ErrorInfo, w: *std.Io.Writer) !void {
    var s: std.json.Stringify = .{ .writer = w };
    try s.beginObject();
    try s.objectField("type");
    try s.write("error");
    try s.objectField("error");
    try s.beginObject();
    try s.objectField("type");
    try s.write(errorTypeName(info.status));
    try s.objectField("message");
    try s.write(info.message);
    try s.endObject();
    try s.endObject();
}

/// Parse and validate one `/v1/messages` body against the backend's
/// capabilities. All allocation goes to `arena` (freed with the request).
pub fn parse(arena: Allocator, body: []const u8, info: types.Info) openai.ParseOutcome {
    const root = std.json.parseFromSliceLeaky(Value, arena, body, .{}) catch {
        return .{ .err = ErrorInfo.invalid("request body is not valid JSON", null) };
    };
    if (root != .object) return .{ .err = ErrorInfo.invalid("request body must be a JSON object", null) };

    var p = Parser{ .arena = arena, .obj = root.object, .info = info };
    const parsed = p.parseMessages() catch |e| switch (e) {
        error.Invalid => return .{ .err = p.err.? },
        error.OutOfMemory => return .{ .err = .{ .status = .internal_server_error, .kind = "api_error", .message = "out of memory" } },
    };
    return .{ .ok = parsed };
}

const Parser = struct {
    arena: Allocator,
    obj: std.json.ObjectMap,
    info: types.Info,
    err: ?ErrorInfo = null,

    const Error = error{ Invalid, OutOfMemory };

    fn fail(self: *Parser, info: ErrorInfo) Error {
        if (self.err == null) self.err = info;
        return error.Invalid;
    }

    fn failInvalid(self: *Parser, message: []const u8, param: ?[]const u8) Error {
        return self.fail(ErrorInfo.invalid(message, param));
    }

    // ---- typed field access over the dynamic Value ----

    fn optField(self: *Parser, obj: std.json.ObjectMap, name: []const u8) ?Value {
        _ = self;
        const v = obj.get(name) orelse return null;
        if (v == .null) return null;
        return v;
    }

    fn optString(self: *Parser, obj: std.json.ObjectMap, name: []const u8) Error!?[]const u8 {
        const v = self.optField(obj, name) orelse return null;
        if (v != .string) return self.failInvalid("expected a string", name);
        return v.string;
    }

    fn optBool(self: *Parser, obj: std.json.ObjectMap, name: []const u8) Error!?bool {
        const v = self.optField(obj, name) orelse return null;
        if (v != .bool) return self.failInvalid("expected a boolean", name);
        return v.bool;
    }

    fn optF32(self: *Parser, obj: std.json.ObjectMap, name: []const u8, min: f32, max: f32) Error!?f32 {
        const v = self.optField(obj, name) orelse return null;
        const x: f64 = switch (v) {
            .integer => |i| @floatFromInt(i),
            .float => |f| f,
            else => return self.failInvalid("expected a number", name),
        };
        if (!std.math.isFinite(x) or x < min or x > max) return self.failInvalid("value out of range", name);
        return @floatCast(x);
    }

    fn optInt(self: *Parser, obj: std.json.ObjectMap, name: []const u8, min: i64) Error!?i64 {
        const v = self.optField(obj, name) orelse return null;
        if (v != .integer) return self.failInvalid("expected an integer", name);
        if (v.integer < min) return self.failInvalid("value out of range", name);
        return v.integer;
    }

    // ---- content blocks ----

    /// Flatten `system` (string or array of text blocks) into one string.
    /// System blocks are text-only in the Messages API; `cache_control` and
    /// other block bookkeeping is ignored.
    fn systemText(self: *Parser, content: Value) Error![]const u8 {
        switch (content) {
            .string => |s| return s,
            .array => |parts| {
                var out: std.ArrayList(u8) = .empty;
                for (parts.items) |part| {
                    if (part != .object) return self.failInvalid("system blocks must be objects", "system");
                    const ptype = (try self.optString(part.object, "type")) orelse
                        return self.failInvalid("system block missing \"type\"", "system");
                    if (!std.mem.eql(u8, ptype, "text"))
                        return self.failInvalid("system blocks must be text blocks", "system");
                    const text = (try self.optString(part.object, "text")) orelse
                        return self.failInvalid("text block missing \"text\"", "system");
                    try out.appendSlice(self.arena, text);
                }
                return out.items;
            },
            else => return self.failInvalid("\"system\" must be a string or an array of text blocks", "system"),
        }
    }

    /// Flatten one message's content (string or block array) into text.
    /// `thinking`/`redacted_thinking` blocks are prior-turn reasoning the
    /// templates re-render without (same policy as the Responses dialect's
    /// reasoning items); tool and media blocks are rejected.
    fn messageText(self: *Parser, content: Value) Error![]const u8 {
        switch (content) {
            .string => |s| return s,
            .array => |parts| {
                var out: std.ArrayList(u8) = .empty;
                for (parts.items) |part| {
                    if (part != .object) return self.failInvalid("content blocks must be objects", "messages");
                    const ptype = (try self.optString(part.object, "type")) orelse
                        return self.failInvalid("content block missing \"type\"", "messages");
                    if (std.mem.eql(u8, ptype, "text")) {
                        const text = (try self.optString(part.object, "text")) orelse
                            return self.failInvalid("text block missing \"text\"", "messages");
                        try out.appendSlice(self.arena, text);
                    } else if (std.mem.eql(u8, ptype, "thinking") or std.mem.eql(u8, ptype, "redacted_thinking")) {
                        // Dropped: prior-turn reasoning.
                    } else if (std.mem.eql(u8, ptype, "tool_use") or std.mem.eql(u8, ptype, "tool_result")) {
                        return self.fail(ErrorInfo.unsupported("tool use is not supported by this server; it never emits tool_use blocks, so a conversation held with it contains none", "messages"));
                    } else if (std.mem.eql(u8, ptype, "image") or std.mem.eql(u8, ptype, "document")) {
                        return self.fail(ErrorInfo.unsupported("only text content is supported (no images or documents)", "messages"));
                    } else {
                        return self.fail(ErrorInfo.unsupported("unsupported content block type", "messages"));
                    }
                }
                return out.items;
            },
            else => return self.failInvalid("message content must be a string or an array of blocks", "messages"),
        }
    }

    // ---- the request ----

    fn parseMessages(self: *Parser) Error!openai.Parsed {
        const obj = self.obj;
        const inf = self.info;
        var parsed = openai.Parsed{ .gen = .{ .messages = &.{}, .sampling = .{}, .max_tokens = 0 } };

        // Required by the Messages API; the value is not matched — this
        // server hosts exactly one model and echoes its own id back.
        if ((try self.optString(obj, "model")) == null)
            return self.failInvalid("\"model\" is required", "model");

        // Hard rejections first: fields whose silent loss would corrupt the
        // reply's meaning.
        if (self.optField(obj, "stop_sequences")) |v| {
            if (v != .array) return self.failInvalid("expected an array", "stop_sequences");
            if (v.array.items.len > 0)
                return self.fail(ErrorInfo.unsupported("stop_sequences are not supported by this server: the engine does not report which sequence fired, so stop_reason could not be attributed", "stop_sequences"));
        }
        if (self.optField(obj, "tools")) |v| {
            // Accepted and dropped (see the module doc); only the shape is
            // checked.
            if (v != .array) return self.failInvalid("expected an array", "tools");
        }
        if (self.optField(obj, "tool_choice")) |v| {
            if (v != .object) return self.failInvalid("expected an object", "tool_choice");
            const kind = (try self.optString(v.object, "type")) orelse
                return self.failInvalid("tool_choice missing \"type\"", "tool_choice");
            if (!std.mem.eql(u8, kind, "auto") and !std.mem.eql(u8, kind, "none"))
                return self.fail(ErrorInfo.unsupported("this server never emits tool_use blocks, so a forced tool call cannot be honored; use tool_choice \"auto\" or \"none\"", "tool_choice"));
        }

        // system -> leading system message.
        var messages: std.ArrayList(llm.chat.Message) = .empty;
        if (self.optField(obj, "system")) |sys| {
            const text = try self.systemText(sys);
            if (text.len > 0) try messages.append(self.arena, .{ .role = .system, .content = text });
        }

        // messages: user/assistant only; a system prompt goes in the
        // top-level "system" field.
        const messages_v = self.optField(obj, "messages") orelse
            return self.failInvalid("\"messages\" is required", "messages");
        if (messages_v != .array) return self.failInvalid("expected an array", "messages");
        if (messages_v.array.items.len == 0)
            return self.failInvalid("\"messages\" must be a non-empty array", "messages");
        for (messages_v.array.items, 0..) |mv, i| {
            if (mv != .object) return self.failInvalid("messages must be objects", "messages");
            const mobj = mv.object;
            const role_s = (try self.optString(mobj, "role")) orelse
                return self.failInvalid("message missing \"role\"", "messages");
            const content_v = self.optField(mobj, "content") orelse
                return self.failInvalid("message missing \"content\"", "messages");
            if (std.mem.eql(u8, role_s, "system")) {
                // Mid-conversation operator instructions (Claude Code
                // appends {"role":"system"} entries to the history). The
                // chat templates here have a single leading system slot, so
                // these fold into a user turn as a <system-reminder> block
                // — the hosted API's own degradation path for models
                // without the feature. messages[0] stays invalid there too.
                if (i == 0)
                    return self.failInvalid("the first message must use the \"user\" role; a system prompt goes in the top-level \"system\" field", "messages");
                const text = try self.messageText(content_v);
                try messages.append(self.arena, .{
                    .role = .user,
                    .content = try std.fmt.allocPrint(self.arena, "<system-reminder>\n{s}\n</system-reminder>", .{text}),
                });
                continue;
            }
            const role: llm.chat.Message.Role = if (std.mem.eql(u8, role_s, "user"))
                .user
            else if (std.mem.eql(u8, role_s, "assistant"))
                .assistant
            else
                return self.failInvalid("unknown message role", "messages");
            if (i == 0 and role != .user)
                return self.failInvalid("the first message must use the \"user\" role", "messages");
            try messages.append(self.arena, .{
                .role = role,
                .content = try self.messageText(content_v),
            });
        }
        parsed.gen.messages = messages.items;

        // Sampling: absent fields keep the model's recommended defaults
        // (same documented deviation as the OpenAI dialects). Anthropic
        // ranges: temperature 0..1, top_p 0..1, top_k >= 0.
        var cfg = inf.default_sampling;
        if (try self.optF32(obj, "temperature", 0, 1)) |v| cfg.temperature = v;
        if (try self.optF32(obj, "top_p", 0, 1)) |v| cfg.top_p = v;
        if (try self.optInt(obj, "top_k", 0)) |v| cfg.top_k = @intCast(v);
        parsed.gen.sampling = cfg;

        if (try self.optBool(obj, "stream")) |v| parsed.stream = v;

        // thinking: enabled/adaptive turn the reasoning channel on when the
        // backend has one (no-op otherwise — see the module doc);
        // budget_tokens is a hosted-API knob with no local equivalent.
        if (self.optField(obj, "thinking")) |tv| {
            if (tv != .object) return self.failInvalid("expected an object", "thinking");
            const kind = (try self.optString(tv.object, "type")) orelse
                return self.failInvalid("thinking missing \"type\"", "thinking");
            if (std.mem.eql(u8, kind, "enabled") or std.mem.eql(u8, kind, "adaptive")) {
                parsed.gen.think = inf.caps.think;
            } else if (std.mem.eql(u8, kind, "disabled")) {
                parsed.gen.think = false;
            } else {
                return self.failInvalid("thinking type must be \"enabled\", \"adaptive\", or \"disabled\"", "thinking");
            }
        }

        // output_config.format: structured outputs, mapped onto the same
        // llguidance constraint the OpenAI dialects use. effort is ignored.
        if (self.optField(obj, "output_config")) |ov| {
            if (ov != .object) return self.failInvalid("expected an object", "output_config");
            if (self.optField(ov.object, "format")) |fv| {
                if (fv != .object) return self.failInvalid("expected an object", "output_config.format");
                const kind = (try self.optString(fv.object, "type")) orelse
                    return self.failInvalid("format missing \"type\"", "output_config.format");
                if (std.mem.eql(u8, kind, "json_schema")) {
                    if (!inf.caps.grammar) {
                        return self.fail(.{
                            .status = .not_implemented,
                            .kind = "not_supported_error",
                            .message = "structured output is not available: this server (or this model's tokenizer) was built without llguidance support",
                            .param = "output_config.format",
                        });
                    }
                    const schema = self.optField(fv.object, "schema") orelse
                        return self.failInvalid("format missing \"schema\"", "output_config.format");
                    // llguidance takes the schema as text.
                    const schema_text = std.json.Stringify.valueAlloc(self.arena, schema, .{}) catch return error.OutOfMemory;
                    parsed.format_kind = .json_schema;
                    parsed.gen.constraint = .{ .json_schema = schema_text };
                } else if (!std.mem.eql(u8, kind, "text")) {
                    return self.failInvalid("format type must be \"text\" or \"json_schema\"", "output_config.format");
                }
            }
        }

        // A grammar constraint governs the reply from token 0: reasoning off.
        if (parsed.gen.think and parsed.gen.constraint != null) parsed.gen.think = false;

        // max_tokens is required by the Messages API, and the generation
        // budget stays bounded by the context window like the OpenAI
        // dialects.
        const requested = (try self.optInt(obj, "max_tokens", 1)) orelse
            return self.failInvalid("\"max_tokens\" is required", "max_tokens");
        parsed.max_tokens_requested = requested;
        parsed.gen.max_tokens = @intCast(@min(requested, @as(i64, @intCast(inf.context_len -| 1))));

        // metadata, service_tier, anthropic-version bookkeeping: ignored.
        return parsed;
    }
};

test "anthropic parse: happy path — system blocks, content blocks, thinking, sampling" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const info = types.Info{
        .model_id = "test",
        .context_len = 4096,
        .caps = .{ .grammar = true, .think = true },
        .default_sampling = .{ .temperature = 0.7, .top_k = 20 },
    };
    const body =
        \\{"model":"claude-x","max_tokens":64,"stream":true,
        \\ "system":[{"type":"text","text":"be terse","cache_control":{"type":"ephemeral"}}],
        \\ "messages":[
        \\  {"role":"user","content":[{"type":"text","text":"hi "},{"type":"text","text":"there"}]},
        \\  {"role":"assistant","content":[{"type":"thinking","thinking":"prior","signature":""},{"type":"text","text":"hello"}]},
        \\  {"role":"user","content":"follow-up"}
        \\ ],
        \\ "temperature":0.5,"top_k":40,
        \\ "thinking":{"type":"enabled","budget_tokens":2048},
        \\ "metadata":{"user_id":"u1"},
        \\ "tools":[{"name":"bash","input_schema":{"type":"object"}}],
        \\ "tool_choice":{"type":"auto"}}
    ;
    const outcome = parse(arena, body, info);
    const p = outcome.ok;
    try std.testing.expectEqual(@as(usize, 4), p.gen.messages.len);
    try std.testing.expectEqual(llm.chat.Message.Role.system, p.gen.messages[0].role);
    try std.testing.expectEqualStrings("be terse", p.gen.messages[0].content);
    try std.testing.expectEqualStrings("hi there", p.gen.messages[1].content);
    // The prior-turn thinking block is dropped; the text remains.
    try std.testing.expectEqualStrings("hello", p.gen.messages[2].content);
    try std.testing.expectEqual(@as(f32, 0.5), p.gen.sampling.temperature);
    try std.testing.expectEqual(@as(usize, 40), p.gen.sampling.top_k);
    try std.testing.expect(p.stream);
    try std.testing.expect(p.gen.think);
    try std.testing.expectEqual(@as(usize, 64), p.gen.max_tokens);
}

test "anthropic parse: rejections carry the offending param" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const info = types.Info{ .model_id = "t", .context_len = 128 };

    const user_msg = "\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]";
    const cases = [_]struct { body: []const u8, param: []const u8 }{
        // max_tokens is required.
        .{ .body = std.fmt.comptimePrint("{{\"model\":\"m\",{s}}}", .{user_msg}), .param = "max_tokens" },
        // model is required.
        .{ .body = std.fmt.comptimePrint("{{\"max_tokens\":8,{s}}}", .{user_msg}), .param = "model" },
        // system role inside messages.
        .{ .body = "{\"model\":\"m\",\"max_tokens\":8,\"messages\":[{\"role\":\"system\",\"content\":\"x\"}]}", .param = "messages" },
        // first message must be user.
        .{ .body = "{\"model\":\"m\",\"max_tokens\":8,\"messages\":[{\"role\":\"assistant\",\"content\":\"x\"}]}", .param = "messages" },
        // tool blocks in history.
        .{ .body = "{\"model\":\"m\",\"max_tokens\":8,\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"t1\",\"content\":\"ok\"}]}]}", .param = "messages" },
        // forced tool call.
        .{ .body = std.fmt.comptimePrint("{{\"model\":\"m\",\"max_tokens\":8,{s},\"tool_choice\":{{\"type\":\"any\"}}}}", .{user_msg}), .param = "tool_choice" },
        // stop_sequences.
        .{ .body = std.fmt.comptimePrint("{{\"model\":\"m\",\"max_tokens\":8,{s},\"stop_sequences\":[\"\\n\"]}}", .{user_msg}), .param = "stop_sequences" },
        // images.
        .{ .body = "{\"model\":\"m\",\"max_tokens\":8,\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image\",\"source\":{}}]}]}", .param = "messages" },
        // Anthropic temperature range is 0..1.
        .{ .body = std.fmt.comptimePrint("{{\"model\":\"m\",\"max_tokens\":8,{s},\"temperature\":1.5}}", .{user_msg}), .param = "temperature" },
        // empty messages.
        .{ .body = "{\"model\":\"m\",\"max_tokens\":8,\"messages\":[]}", .param = "messages" },
    };
    for (cases) |case| {
        const outcome = parse(arena, case.body, info);
        try std.testing.expectEqualStrings(case.param, outcome.err.param.?);
        try std.testing.expectEqual(std.http.Status.bad_request, outcome.err.status);
    }

    // An empty stop_sequences array is absent, not a rejection.
    const ok = parse(arena, std.fmt.comptimePrint("{{\"model\":\"m\",\"max_tokens\":8,{s},\"stop_sequences\":[]}}", .{user_msg}), info);
    try std.testing.expectEqual(@as(usize, 0), ok.ok.gen.stop.len);
}

test "anthropic parse: mid-conversation system message folds into a user turn" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const info = types.Info{ .model_id = "t", .context_len = 128 };

    const body =
        \\{"model":"m","max_tokens":8,"messages":[
        \\  {"role":"user","content":"question"},
        \\  {"role":"system","content":[{"type":"text","text":"terse mode"}]}
        \\]}
    ;
    const p = parse(arena, body, info).ok;
    try std.testing.expectEqual(@as(usize, 2), p.gen.messages.len);
    try std.testing.expectEqual(llm.chat.Message.Role.user, p.gen.messages[1].role);
    try std.testing.expectEqualStrings("<system-reminder>\nterse mode\n</system-reminder>", p.gen.messages[1].content);
}

test "anthropic parse: thinking is a no-op without a reasoning channel" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const info = types.Info{ .model_id = "t", .context_len = 128 }; // caps.think = false

    const body = "{\"model\":\"m\",\"max_tokens\":8,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"thinking\":{\"type\":\"adaptive\"}}";
    const outcome = parse(arena, body, info);
    try std.testing.expect(!outcome.ok.gen.think);
}

test "anthropic parse: output_config.format maps to the grammar constraint" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const body =
        \\{"model":"m","max_tokens":8,
        \\ "messages":[{"role":"user","content":"x"}],
        \\ "thinking":{"type":"enabled"},
        \\ "output_config":{"format":{"type":"json_schema","schema":{"type":"object"}}}}
    ;
    // With grammar support: constraint set, reasoning forced off.
    {
        const info = types.Info{ .model_id = "t", .context_len = 128, .caps = .{ .grammar = true, .think = true } };
        const p = parse(arena, body, info).ok;
        try std.testing.expectEqualStrings("{\"type\":\"object\"}", p.gen.constraint.?.json_schema);
        try std.testing.expect(!p.gen.think);
    }
    // Without: 501, pointing at the field.
    {
        const info = types.Info{ .model_id = "t", .context_len = 128 };
        const outcome = parse(arena, body, info);
        try std.testing.expectEqual(std.http.Status.not_implemented, outcome.err.status);
        try std.testing.expectEqualStrings("output_config.format", outcome.err.param.?);
    }
}

test "anthropic error envelope: type derived from status" {
    var buf: [256]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try writeErrorBody(.{ .status = .bad_request, .message = "\"max_tokens\" is required" }, &w);
    try std.testing.expectEqualStrings(
        "{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"\\\"max_tokens\\\" is required\"}}",
        w.buffered(),
    );
    try std.testing.expectEqualStrings("rate_limit_error", errorTypeName(.too_many_requests));
    try std.testing.expectEqualStrings("overloaded_error", errorTypeName(.service_unavailable));
    try std.testing.expectEqualStrings("api_error", errorTypeName(.internal_server_error));
}
