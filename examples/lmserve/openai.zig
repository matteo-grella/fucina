//! OpenAI wire-format parsing/validation: Chat Completions
//! (`POST /v1/chat/completions`) and the stateless subset of the Responses
//! API (`POST /v1/responses`), both normalized into one internal
//! `types.GenerateRequest`. Every accepted, rejected, and ignored field is
//! deliberate — see SERVER-EXPLORATION.md for the mapping tables. Rejections
//! use OpenAI's error shape (`{"error":{message,type,param,code}}`) with the
//! offending field in `param`; unsupported-but-harmless bookkeeping fields
//! (`metadata`, `user`, `store:false`, …) are ignored like llama.cpp does.

const std = @import("std");
const llm = @import("fucina_llm");
const types = @import("types.zig");

const Allocator = std.mem.Allocator;
const Value = std.json.Value;

pub const Dialect = enum { chat, responses };

pub const ErrorInfo = struct {
    status: std.http.Status,
    kind: []const u8 = "invalid_request_error",
    message: []const u8,
    param: ?[]const u8 = null,
    code: ?[]const u8 = null,

    pub fn invalid(message: []const u8, param: ?[]const u8) ErrorInfo {
        return .{ .status = .bad_request, .message = message, .param = param };
    }

    pub fn unsupported(message: []const u8, param: ?[]const u8) ErrorInfo {
        return .{ .status = .bad_request, .message = message, .param = param, .code = "unsupported_parameter" };
    }
};

/// Serialize the OpenAI error body for `info`.
pub fn writeErrorBody(info: ErrorInfo, w: *std.Io.Writer) !void {
    var s: std.json.Stringify = .{ .writer = w };
    try s.beginObject();
    try s.objectField("error");
    try s.beginObject();
    try s.objectField("message");
    try s.write(info.message);
    try s.objectField("type");
    try s.write(info.kind);
    try s.objectField("param");
    try s.write(info.param);
    try s.objectField("code");
    try s.write(info.code);
    try s.endObject();
    try s.endObject();
}

pub const FormatKind = enum { text, json_object, json_schema };

/// One request, normalized. Slices point into the request arena (the parsed
/// JSON is arena-leaky), alive for the whole request.
pub const Parsed = struct {
    gen: types.GenerateRequest,
    stream: bool = false,
    /// Chat streaming: emit the trailing usage-only chunk.
    include_usage: bool = false,
    /// Echoes for the Responses response object.
    max_tokens_requested: ?i64 = null,
    format_kind: FormatKind = .text,
    format_name: ?[]const u8 = null,
};

pub const ParseOutcome = union(enum) {
    ok: Parsed,
    err: ErrorInfo,
};

/// Parse and validate one request body against the backend's capabilities.
/// All allocation goes to `arena` (freed with the request).
pub fn parse(arena: Allocator, dialect: Dialect, body: []const u8, info: types.Info) ParseOutcome {
    const root = std.json.parseFromSliceLeaky(Value, arena, body, .{}) catch {
        return .{ .err = ErrorInfo.invalid("request body is not valid JSON", null) };
    };
    if (root != .object) return .{ .err = ErrorInfo.invalid("request body must be a JSON object", null) };

    var p = Parser{ .arena = arena, .obj = root.object, .info = info };
    const parsed = switch (dialect) {
        .chat => p.parseChat(),
        .responses => p.parseResponses(),
    } catch |e| switch (e) {
        error.Invalid => return .{ .err = p.err.? },
        error.OutOfMemory => return .{ .err = .{ .status = .internal_server_error, .kind = "server_error", .message = "out of memory" } },
    };
    return .{ .ok = parsed };
}

/// Map a backend/scheduler error to the OpenAI error shape. `Cancelled`
/// never reaches serialization (the client is gone).
pub fn mapError(err: anyerror) ErrorInfo {
    return switch (err) {
        error.PromptTooLong, error.ContextFull => .{
            .status = .bad_request,
            .message = "this request exceeds the model's context window; reduce the message history or max output tokens",
            .code = "context_length_exceeded",
        },
        error.EmptyMessages => ErrorInfo.invalid("at least one message is required", "messages"),
        error.TrailingAssistantMessage => ErrorInfo.invalid("the final message must not be an assistant message", "messages"),
        error.SystemMidConversation => ErrorInfo.invalid("this model's template accepts system messages only at the start of the conversation", "messages"),
        error.NoSystemRole => ErrorInfo.invalid("this model's chat protocol has no system role; fold instructions into the user message", "messages"),
        error.InvalidGrammar => ErrorInfo.invalid("the requested output constraint could not be compiled (unsupported JSON-schema keyword, or invalid regex/grammar)", "response_format"),
        error.LlguidanceNotEnabled => .{
            .status = .not_implemented,
            .kind = "not_supported_error",
            .message = "constrained output requires a server built with -Dllguidance=true",
        },
        error.ShuttingDown => .{ .status = .service_unavailable, .kind = "unavailable_error", .message = "the server is shutting down" },
        else => .{ .status = .internal_server_error, .kind = "server_error", .message = "internal generation failure" },
    };
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

    /// Reject `name` with a 400 when present and not `null`.
    fn rejectField(self: *Parser, obj: std.json.ObjectMap, name: []const u8, message: []const u8) Error!void {
        if (self.optField(obj, name) != null) return self.fail(ErrorInfo.unsupported(message, name));
    }

    // ---- shared post-normalization ----

    fn finishCommon(self: *Parser, parsed: *Parsed) Error!void {
        const obj = self.obj;
        const inf = self.info;

        // Sampling: absent fields keep the model's recommended defaults —
        // better output than OpenAI's nominal temperature=1 on small local
        // models, and what llama.cpp does too (documented deviation).
        var cfg = inf.default_sampling;
        if (try self.optF32(obj, "temperature", 0, 2)) |v| cfg.temperature = v;
        if (try self.optF32(obj, "top_p", 0, 1)) |v| cfg.top_p = v;
        if (try self.optF32(obj, "presence_penalty", -2, 2)) |v| cfg.presence_penalty = v;
        if (try self.optF32(obj, "frequency_penalty", -2, 2)) |v| cfg.freq_penalty = v;
        if (try self.optInt(obj, "seed", 0)) |v| cfg.seed = @intCast(v);
        // llama.cpp-style extensions on the same endpoint.
        if (try self.optInt(obj, "top_k", 0)) |v| cfg.top_k = @intCast(v);
        if (try self.optF32(obj, "min_p", 0, 1)) |v| cfg.min_p = v;
        if (try self.optF32(obj, "repeat_penalty", 0, 4)) |v| cfg.repeat_penalty = v;
        parsed.gen.sampling = cfg;

        if (try self.optBool(obj, "stream")) |v| parsed.stream = v;
        if (self.optField(obj, "stream_options")) |so| {
            if (so != .object) return self.failInvalid("expected an object", "stream_options");
            if (!parsed.stream) return self.failInvalid("stream_options requires stream=true", "stream_options");
            if (try self.optBool(so.object, "include_usage")) |v| parsed.include_usage = v;
        }

        // Stop strings (max 4, like OpenAI).
        if (self.optField(obj, "stop")) |v| {
            const list: []const Value = switch (v) {
                .string => &.{v},
                .array => |a| a.items,
                else => return self.failInvalid("expected a string or array of strings", "stop"),
            };
            if (list.len > 4) return self.failInvalid("at most 4 stop sequences are supported", "stop");
            if (list.len > 0 and !inf.caps.stop_sequences)
                return self.fail(ErrorInfo.unsupported("stop sequences are not supported by this model backend", "stop"));
            const stops = try self.arena.alloc([]const u8, list.len);
            for (list, stops) |item, *dst| {
                if (item != .string or item.string.len == 0)
                    return self.failInvalid("stop sequences must be non-empty strings", "stop");
                dst.* = item.string;
            }
            parsed.gen.stop = stops;
        }

        // Constraint extensions (regex / lark), shared by both dialects.
        var constraint: ?types.ConstraintSpec = parsed.gen.constraint;
        if (try self.optString(obj, "regex")) |v| {
            if (constraint != null) return self.failInvalid("only one of response_format/text.format, regex, and lark may be set", "regex");
            constraint = .{ .regex = v };
        }
        if (try self.optString(obj, "lark")) |v| {
            if (constraint != null) return self.failInvalid("only one of response_format/text.format, regex, and lark may be set", "lark");
            constraint = .{ .lark = v };
        }
        if (constraint != null and !inf.caps.grammar) {
            return self.fail(.{
                .status = .not_implemented,
                .kind = "not_supported_error",
                .message = "constrained output is not available: this server (or this model's tokenizer) was built without llguidance support",
                .param = "response_format",
            });
        }
        parsed.gen.constraint = constraint;

        // Reasoning: off by default (predictable JSON-first serving; matches
        // constrained decoding, which governs the reply from token 0).
        // A grammar constraint forces it off.
        if (parsed.gen.think and parsed.gen.constraint != null) parsed.gen.think = false;

        // Generation budget: always bounded (an unbounded budget plus an
        // open-ended grammar field can loop; docs/CONSTRAINED-DECODING.md §7).
        const default_max: i64 = 1024;
        const requested = parsed.max_tokens_requested orelse default_max;
        parsed.gen.max_tokens = @intCast(@min(requested, @as(i64, @intCast(inf.context_len -| 1))));
    }

    fn parseReasoningEffort(self: *Parser, effort: []const u8, param: []const u8) Error!bool {
        if (std.mem.eql(u8, effort, "none") or std.mem.eql(u8, effort, "minimal")) return false;
        const known = [_][]const u8{ "low", "medium", "high", "xhigh", "default" };
        for (known) |k| {
            if (std.mem.eql(u8, effort, k)) {
                if (!self.info.caps.think) return self.fail(ErrorInfo.unsupported("this model has no toggleable reasoning channel; use \"none\"", param));
                return true;
            }
        }
        return self.failInvalid("unknown reasoning effort", param);
    }

    /// response_format (chat) / text.format (responses). The chat flavor
    /// nests the schema under "json_schema"; the responses flavor is flat.
    fn parseFormat(self: *Parser, parsed: *Parsed, format: Value, param: []const u8, comptime flat: bool) Error!void {
        if (format != .object) return self.failInvalid("expected an object", param);
        const fobj = format.object;
        const kind = (try self.optString(fobj, "type")) orelse return self.failInvalid("missing \"type\"", param);
        if (std.mem.eql(u8, kind, "text")) return;
        if (std.mem.eql(u8, kind, "json_object")) {
            parsed.format_kind = .json_object;
            parsed.gen.constraint = .{ .json_schema = "{\"type\":\"object\"}" };
            return;
        }
        if (!std.mem.eql(u8, kind, "json_schema"))
            return self.failInvalid("type must be one of \"text\", \"json_object\", \"json_schema\"", param);

        const holder = if (flat) fobj else blk: {
            const nested = self.optField(fobj, "json_schema") orelse
                return self.failInvalid("missing \"json_schema\"", param);
            if (nested != .object) return self.failInvalid("expected an object", param);
            break :blk nested.object;
        };
        parsed.format_kind = .json_schema;
        parsed.format_name = try self.optString(holder, "name");
        const schema = self.optField(holder, "schema") orelse
            return self.failInvalid("missing \"schema\"", param);
        // llguidance takes the schema as text: re-serialize the parsed value.
        const schema_text = std.json.Stringify.valueAlloc(self.arena, schema, .{}) catch return error.OutOfMemory;
        parsed.gen.constraint = .{ .json_schema = schema_text };
    }

    /// Flatten a content string / array of text parts into one string.
    fn contentText(self: *Parser, content: Value, param: []const u8) Error![]const u8 {
        switch (content) {
            .string => |s| return s,
            .array => |parts| {
                var out: std.ArrayList(u8) = .empty;
                for (parts.items) |part| {
                    if (part != .object) return self.failInvalid("content parts must be objects", param);
                    const ptype = (try self.optString(part.object, "type")) orelse
                        return self.failInvalid("content part missing \"type\"", param);
                    if (std.mem.eql(u8, ptype, "text") or
                        std.mem.eql(u8, ptype, "input_text") or
                        std.mem.eql(u8, ptype, "output_text"))
                    {
                        const text = (try self.optString(part.object, "text")) orelse
                            return self.failInvalid("text part missing \"text\"", param);
                        try out.appendSlice(self.arena, text);
                    } else if (std.mem.eql(u8, ptype, "refusal")) {
                        // Historical refusal parts carry no renderable text.
                    } else {
                        return self.fail(ErrorInfo.unsupported("only text content is supported (no images, audio, or files)", param));
                    }
                }
                return out.items;
            },
            else => return self.failInvalid("content must be a string or an array of parts", param),
        }
    }

    // ---- Chat Completions ----

    fn parseChat(self: *Parser) Error!Parsed {
        const obj = self.obj;
        var parsed = Parsed{ .gen = .{ .messages = &.{}, .sampling = .{}, .max_tokens = 0 } };

        // Hard rejections first: fields whose silent loss would corrupt the
        // conversation semantics.
        try self.rejectField(obj, "tools", "function calling is not supported by this server");
        try self.rejectField(obj, "functions", "function calling is not supported by this server");
        try self.rejectField(obj, "function_call", "function calling is not supported by this server");
        try self.rejectField(obj, "logprobs", "logprobs are not supported by this server");
        try self.rejectField(obj, "top_logprobs", "logprobs are not supported by this server");
        try self.rejectField(obj, "logit_bias", "logit_bias is not supported by this server");
        try self.rejectField(obj, "audio", "audio output is not supported by this server");
        try self.rejectField(obj, "prediction", "predicted outputs are not supported by this server");
        try self.rejectField(obj, "web_search_options", "web search is not supported by this server");
        if (try self.optString(obj, "tool_choice")) |tc| {
            if (!std.mem.eql(u8, tc, "none") and !std.mem.eql(u8, tc, "auto"))
                return self.fail(ErrorInfo.unsupported("function calling is not supported by this server", "tool_choice"));
        }
        if (try self.optInt(obj, "n", 1)) |n| {
            if (n != 1) return self.fail(ErrorInfo.unsupported("only n=1 is supported by this server", "n"));
        }

        // messages
        const messages_v = self.optField(obj, "messages") orelse
            return self.failInvalid("missing \"messages\"", "messages");
        if (messages_v != .array) return self.failInvalid("expected an array", "messages");
        var messages: std.ArrayList(llm.chat.Message) = .empty;
        for (messages_v.array.items) |mv| {
            if (mv != .object) return self.failInvalid("messages must be objects", "messages");
            const mobj = mv.object;
            const role_s = (try self.optString(mobj, "role")) orelse
                return self.failInvalid("message missing \"role\"", "messages");
            const role: llm.chat.Message.Role = if (std.mem.eql(u8, role_s, "system") or std.mem.eql(u8, role_s, "developer"))
                .system
            else if (std.mem.eql(u8, role_s, "user"))
                .user
            else if (std.mem.eql(u8, role_s, "assistant"))
                .assistant
            else if (std.mem.eql(u8, role_s, "tool") or std.mem.eql(u8, role_s, "function"))
                return self.fail(ErrorInfo.unsupported("tool messages are not supported by this server", "messages"))
            else
                return self.failInvalid("unknown message role", "messages");
            if (self.optField(mobj, "tool_calls") != null)
                return self.fail(ErrorInfo.unsupported("tool messages are not supported by this server", "messages"));
            const content_v = self.optField(mobj, "content") orelse
                return self.failInvalid("message missing \"content\"", "messages");
            try messages.append(self.arena, .{
                .role = role,
                .content = try self.contentText(content_v, "messages"),
            });
        }
        parsed.gen.messages = messages.items;

        if (self.optField(obj, "response_format")) |rf|
            try self.parseFormat(&parsed, rf, "response_format", false);

        if (try self.optString(obj, "reasoning_effort")) |effort|
            parsed.gen.think = try self.parseReasoningEffort(effort, "reasoning_effort");

        // max_completion_tokens preferred; the deprecated max_tokens accepted.
        parsed.max_tokens_requested = (try self.optInt(obj, "max_completion_tokens", 1)) orelse
            (try self.optInt(obj, "max_tokens", 1));

        try self.finishCommon(&parsed);
        return parsed;
    }

    // ---- Responses (stateless subset) ----

    fn parseResponses(self: *Parser) Error!Parsed {
        const obj = self.obj;
        var parsed = Parsed{ .gen = .{ .messages = &.{}, .sampling = .{}, .max_tokens = 0 } };

        // Stateful / hosted features: rejected loudly, per the stateless
        // Responses profile (what Codex CLI and the SDKs' basic paths use).
        try self.rejectField(obj, "previous_response_id", "this server is stateless: resend the full conversation via \"input\" instead of chaining previous_response_id");
        try self.rejectField(obj, "conversation", "this server is stateless and has no conversation store");
        try self.rejectField(obj, "prompt", "stored prompt templates are not supported by this server");
        try self.rejectField(obj, "background", "background responses are not supported by this server");
        if (self.optField(obj, "tools")) |tools_v| {
            if (tools_v != .array) return self.failInvalid("expected an array", "tools");
            if (tools_v.array.items.len > 0)
                return self.fail(ErrorInfo.unsupported("tools are not supported by this server", "tools"));
        }
        if (try self.optString(obj, "truncation")) |tr| {
            if (!std.mem.eql(u8, tr, "disabled"))
                return self.fail(ErrorInfo.unsupported("only truncation=\"disabled\" is supported", "truncation"));
        }

        var messages: std.ArrayList(llm.chat.Message) = .empty;

        // instructions -> leading system message.
        if (try self.optString(obj, "instructions")) |instructions|
            try messages.append(self.arena, .{ .role = .system, .content = instructions });

        const input_v = self.optField(obj, "input") orelse
            return self.failInvalid("missing \"input\"", "input");
        switch (input_v) {
            .string => |s| try messages.append(self.arena, .{ .role = .user, .content = s }),
            .array => |items| for (items.items) |item| {
                if (item != .object) return self.failInvalid("input items must be objects", "input");
                const iobj = item.object;
                const itype = (try self.optString(iobj, "type")) orelse "message";
                if (std.mem.eql(u8, itype, "message")) {
                    const role_s = (try self.optString(iobj, "role")) orelse
                        return self.failInvalid("message item missing \"role\"", "input");
                    const role: llm.chat.Message.Role = if (std.mem.eql(u8, role_s, "system") or std.mem.eql(u8, role_s, "developer"))
                        .system
                    else if (std.mem.eql(u8, role_s, "user"))
                        .user
                    else if (std.mem.eql(u8, role_s, "assistant"))
                        .assistant
                    else
                        return self.failInvalid("unknown message role", "input");
                    const content_v = self.optField(iobj, "content") orelse
                        return self.failInvalid("message item missing \"content\"", "input");
                    try messages.append(self.arena, .{
                        .role = role,
                        .content = try self.contentText(content_v, "input"),
                    });
                } else if (std.mem.eql(u8, itype, "reasoning")) {
                    // Prior-turn reasoning items (Codex replays them): the
                    // reference templates drop prior reasoning, so do we.
                } else if (std.mem.eql(u8, itype, "function_call") or
                    std.mem.eql(u8, itype, "function_call_output") or
                    std.mem.eql(u8, itype, "item_reference"))
                {
                    return self.fail(ErrorInfo.unsupported("tool items are not supported by this server", "input"));
                } else {
                    return self.fail(ErrorInfo.unsupported("unsupported input item type", "input"));
                }
            },
            else => return self.failInvalid("input must be a string or an array of items", "input"),
        }
        parsed.gen.messages = messages.items;

        if (self.optField(obj, "text")) |text_v| {
            if (text_v != .object) return self.failInvalid("expected an object", "text");
            if (self.optField(text_v.object, "format")) |format|
                try self.parseFormat(&parsed, format, "text.format", true);
        }

        if (self.optField(obj, "reasoning")) |rv| {
            if (rv != .object) return self.failInvalid("expected an object", "reasoning");
            if (try self.optString(rv.object, "effort")) |effort|
                parsed.gen.think = try self.parseReasoningEffort(effort, "reasoning.effort");
        }

        parsed.max_tokens_requested = try self.optInt(obj, "max_output_tokens", 1);

        try self.finishCommon(&parsed);
        return parsed;
    }
};

test "chat parse: happy path with schema, stop, sampling overrides" {
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
        \\{"model":"x","messages":[
        \\  {"role":"system","content":"be terse"},
        \\  {"role":"user","content":[{"type":"text","text":"hi "},{"type":"text","text":"there"}]}
        \\ ],
        \\ "temperature":0.5,"seed":7,"stop":["\n\n"],"stream":true,
        \\ "stream_options":{"include_usage":true},
        \\ "max_completion_tokens":64,
        \\ "response_format":{"type":"json_schema","json_schema":{"name":"out","schema":{"type":"object"},"strict":true}}}
    ;
    const outcome = parse(arena, .chat, body, info);
    const p = outcome.ok;
    try std.testing.expectEqual(@as(usize, 2), p.gen.messages.len);
    try std.testing.expectEqual(llm.chat.Message.Role.system, p.gen.messages[0].role);
    try std.testing.expectEqualStrings("hi there", p.gen.messages[1].content);
    try std.testing.expectEqual(@as(f32, 0.5), p.gen.sampling.temperature);
    try std.testing.expectEqual(@as(u64, 7), p.gen.sampling.seed);
    try std.testing.expectEqual(@as(usize, 20), p.gen.sampling.top_k); // default kept
    try std.testing.expect(p.stream and p.include_usage);
    try std.testing.expectEqual(@as(usize, 64), p.gen.max_tokens);
    try std.testing.expectEqual(@as(usize, 1), p.gen.stop.len);
    try std.testing.expectEqualStrings("{\"type\":\"object\"}", p.gen.constraint.?.json_schema);
    try std.testing.expect(!p.gen.think); // constraint forces reasoning off
    try std.testing.expectEqualStrings("out", p.format_name.?);
}

test "chat parse: rejections carry the offending param" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const info = types.Info{ .model_id = "t", .context_len = 128 };

    const cases = [_]struct { body: []const u8, param: []const u8 }{
        .{ .body = "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"tools\":[{}]}", .param = "tools" },
        .{ .body = "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"n\":2}", .param = "n" },
        .{ .body = "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"logprobs\":true}", .param = "logprobs" },
        .{ .body = "{\"messages\":[{\"role\":\"tool\",\"content\":\"x\"}]}", .param = "messages" },
        .{ .body = "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"temperature\":9}", .param = "temperature" },
        .{ .body = "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream_options\":{}}", .param = "stream_options" },
        // grammar unavailable on this backend (caps.grammar = false)
        .{ .body = "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"response_format\":{\"type\":\"json_object\"}}", .param = "response_format" },
    };
    for (cases) |case| {
        const outcome = parse(arena, .chat, case.body, info);
        try std.testing.expectEqualStrings(case.param, outcome.err.param.?);
    }

    const bad_json = parse(arena, .chat, "{nope", info);
    try std.testing.expectEqual(std.http.Status.bad_request, bad_json.err.status);
}

test "responses parse: input forms, instructions, statelessness" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const info = types.Info{ .model_id = "t", .context_len = 4096, .caps = .{ .grammar = true } };

    // String input + instructions.
    {
        const outcome = parse(arena, .responses, "{\"input\":\"hi\",\"instructions\":\"be terse\",\"max_output_tokens\":32}", info);
        const p = outcome.ok;
        try std.testing.expectEqual(@as(usize, 2), p.gen.messages.len);
        try std.testing.expectEqual(llm.chat.Message.Role.system, p.gen.messages[0].role);
        try std.testing.expectEqual(llm.chat.Message.Role.user, p.gen.messages[1].role);
        try std.testing.expectEqual(@as(usize, 32), p.gen.max_tokens);
    }
    // Item-list input: typed message items + output_text parts + skipped reasoning item.
    {
        const body =
            \\{"input":[
            \\  {"role":"user","content":[{"type":"input_text","text":"question"}]},
            \\  {"type":"reasoning","id":"rs_1","summary":[]},
            \\  {"type":"message","role":"assistant","content":[{"type":"output_text","text":"answer"}]},
            \\  {"role":"user","content":"follow-up"}
            \\],"text":{"format":{"type":"json_schema","name":"o","schema":{"type":"object"}}}}
        ;
        const outcome = parse(arena, .responses, body, info);
        const p = outcome.ok;
        try std.testing.expectEqual(@as(usize, 3), p.gen.messages.len);
        try std.testing.expectEqualStrings("answer", p.gen.messages[1].content);
        try std.testing.expect(p.gen.constraint != null);
        try std.testing.expectEqual(FormatKind.json_schema, p.format_kind);
    }
    // Stateful chaining is rejected with a pointer at the field.
    {
        const outcome = parse(arena, .responses, "{\"input\":\"x\",\"previous_response_id\":\"resp_123\"}", info);
        try std.testing.expectEqualStrings("previous_response_id", outcome.err.param.?);
    }
}
