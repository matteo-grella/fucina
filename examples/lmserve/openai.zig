//! OpenAI wire-format parsing/validation: Chat Completions
//! (`POST /v1/chat/completions`) and the stateless subset of the Responses
//! API (`POST /v1/responses`), both normalized into one internal
//! `types.GenerateRequest`. Every accepted, rejected, and ignored field is
//! deliberate — see SERVER-EXPLORATION.md for the mapping tables. Rejections
//! use OpenAI's error shape (`{"error":{message,type,param,code}}`) with the
//! offending field in `param`; unsupported-but-harmless bookkeeping fields
//! (`metadata`, `user`, `store:false`, …) are ignored like llama.cpp does.
//! Function calling works on hermes-style backends (`types.ToolStyle`):
//! declarations and tool history fold into the prompt via `toolcall.zig`,
//! and the emitter scans the reply for calls. A "required" or named
//! `tool_choice` compiles to a forced-call grammar
//! (`toolcall.forcedCallGrammar`; schema-enforced arguments under
//! `strict: true`) — needs a `-Dllguidance=true` build, rejected
//! otherwise.

const std = @import("std");
const llm = @import("fucina_llm");
const types = @import("fucina_llm").serving;
const toolcall = @import("toolcall.zig");

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
    /// Tool declarations were rendered into the prompt: the emitter scans
    /// the reply for `<tool_call>` regions (`toolcall.zig`).
    tools_active: bool = false,
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
        else => blk: {
            // The client gets a generic 500; the operator gets the name.
            std.log.err("generation failed: {t}", .{err});
            break :blk .{ .status = .internal_server_error, .kind = "server_error", .message = "internal generation failure" };
        },
    };
}

const Parser = struct {
    arena: Allocator,
    obj: std.json.ObjectMap,
    info: types.Info,
    err: ?ErrorInfo = null,
    /// Set by a "required"/named tool_choice: `finishCommon` compiles these
    /// into the forced-call grammar.
    forced_decls: ?[]const toolcall.Decl = null,

    const Error = error{ Invalid, OutOfMemory };

    /// Declarations in both forms the pipeline needs: the serialized
    /// objects the hermes system block embeds, and the (name, schema)
    /// pairs the forced-call grammar is built from.
    const ToolSet = struct {
        json: []const []const u8 = &.{},
        decls: []const toolcall.Decl = &.{},
    };

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

        // A forced tool_choice compiles to a grammar over the hermes call
        // shape; it owns the reply from token 0 and excludes the other
        // constraint forms.
        if (self.forced_decls) |decls| {
            if (constraint != null)
                return self.failInvalid("a forced tool_choice already constrains the reply; response_format, regex, and lark cannot combine with it", "tool_choice");
            if (!inf.caps.grammar) {
                return self.fail(.{
                    .status = .not_implemented,
                    .kind = "not_supported_error",
                    .message = "a guaranteed tool call needs constrained decoding: this server (or this model's tokenizer) was built without llguidance support",
                    .param = "tool_choice",
                });
            }
            constraint = .{ .lark = toolcall.forcedCallGrammar(self.arena, decls) catch return error.OutOfMemory };
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

    // ---- tool calling (hermes backends; `toolcall.zig` renders) ----

    /// Flush accumulated tool-result sections as one user turn (Qwen3's
    /// template shape: consecutive results share the turn).
    fn flushToolResponses(self: *Parser, messages: *std.ArrayList(llm.chat.Message), fold: *std.ArrayList(u8)) Error!void {
        if (fold.items.len == 0) return;
        try messages.append(self.arena, .{ .role = .user, .content = try fold.toOwnedSlice(self.arena) });
    }

    /// Render declarations into the leading system slot and arm the reply
    /// scanner.
    fn applyTools(self: *Parser, parsed: *Parsed, messages: *std.ArrayList(llm.chat.Message), tools_json: []const []const u8) Error!void {
        if (tools_json.len == 0) return;
        if (messages.items.len > 0 and messages.items[0].role == .system) {
            messages.items[0].content = toolcall.renderSystemWithTools(self.arena, messages.items[0].content, tools_json) catch return error.OutOfMemory;
        } else {
            const content = toolcall.renderSystemWithTools(self.arena, "", tools_json) catch return error.OutOfMemory;
            messages.insert(self.arena, 0, .{ .role = .system, .content = content }) catch return error.OutOfMemory;
        }
        parsed.tools_active = true;
    }

    /// A Responses `function_call` item joins the assistant turn it
    /// follows (message + calls are sibling items of one turn).
    fn appendCallToAssistant(self: *Parser, messages: *std.ArrayList(llm.chat.Message), name: []const u8, args_json: []const u8) Error!void {
        var turn: std.ArrayList(u8) = .empty;
        if (messages.items.len > 0 and messages.items[messages.items.len - 1].role == .assistant) {
            turn.appendSlice(self.arena, messages.items[messages.items.len - 1].content) catch return error.OutOfMemory;
            toolcall.appendCallSection(self.arena, &turn, name, args_json) catch return error.OutOfMemory;
            messages.items[messages.items.len - 1].content = turn.items;
        } else {
            toolcall.appendCallSection(self.arena, &turn, name, args_json) catch return error.OutOfMemory;
            messages.append(self.arena, .{ .role = .assistant, .content = turn.items }) catch return error.OutOfMemory;
        }
    }

    /// The declaration's grammar schema: `parameters` when `strict` asked
    /// for exact enforcement, the permissive object schema otherwise (a
    /// non-strict schema may use JSON-schema keywords llguidance rejects,
    /// and non-strict semantics promise none of that enforcement anyway).
    fn declSchema(self: *Parser, holder: std.json.ObjectMap, strict: bool) Error![]const u8 {
        if (!strict) return "{\"type\":\"object\"}";
        const pv = self.optField(holder, "parameters") orelse return "{\"type\":\"object\"}";
        return std.json.Stringify.valueAlloc(self.arena, pv, .{}) catch return error.OutOfMemory;
    }

    /// Chat `tools`: entries validated, then re-serialized verbatim — the
    /// `{"type":"function","function":{…}}` object is exactly what the
    /// hermes system block embeds.
    fn parseChatTools(self: *Parser) Error!ToolSet {
        const tools_v = self.optField(self.obj, "tools") orelse return .{};
        if (tools_v != .array) return self.failInvalid("expected an array", "tools");
        const items = tools_v.array.items;
        if (items.len == 0) return .{};
        if (self.info.tool_style == .none)
            return self.fail(ErrorInfo.unsupported("function calling is not supported by this model backend", "tools"));
        const json = try self.arena.alloc([]const u8, items.len);
        const decls = try self.arena.alloc(toolcall.Decl, items.len);
        for (items, json, decls) |tv, *dst, *decl| {
            if (tv != .object) return self.failInvalid("tools must be objects", "tools");
            if (try self.optString(tv.object, "type")) |t| {
                if (!std.mem.eql(u8, t, "function"))
                    return self.fail(ErrorInfo.unsupported("only function tools are supported", "tools"));
            }
            const fv = self.optField(tv.object, "function") orelse
                return self.failInvalid("tool missing \"function\"", "tools");
            if (fv != .object) return self.failInvalid("expected an object", "tools");
            const name = (try self.optString(fv.object, "name")) orelse
                return self.failInvalid("function missing \"name\"", "tools");
            const strict = (try self.optBool(fv.object, "strict")) orelse false;
            decl.* = .{ .name = name, .params_json = try self.declSchema(fv.object, strict) };
            dst.* = std.json.Stringify.valueAlloc(self.arena, tv, .{}) catch return error.OutOfMemory;
        }
        return .{ .json = json, .decls = decls };
    }

    /// Responses `tools`: the flat function shape, rebuilt into the nested
    /// object the hermes system block embeds.
    fn parseResponsesTools(self: *Parser) Error!ToolSet {
        const tools_v = self.optField(self.obj, "tools") orelse return .{};
        if (tools_v != .array) return self.failInvalid("expected an array", "tools");
        const items = tools_v.array.items;
        if (items.len == 0) return .{};
        if (self.info.tool_style == .none)
            return self.fail(ErrorInfo.unsupported("function calling is not supported by this model backend", "tools"));
        const json = try self.arena.alloc([]const u8, items.len);
        const decls = try self.arena.alloc(toolcall.Decl, items.len);
        for (items, json, decls) |tv, *dst, *decl| {
            if (tv != .object) return self.failInvalid("tools must be objects", "tools");
            const t = (try self.optString(tv.object, "type")) orelse
                return self.failInvalid("tool missing \"type\"", "tools");
            if (!std.mem.eql(u8, t, "function"))
                return self.fail(ErrorInfo.unsupported("hosted tool types are not supported by this server", "tools"));
            const name = (try self.optString(tv.object, "name")) orelse
                return self.failInvalid("tool missing \"name\"", "tools");
            const strict = (try self.optBool(tv.object, "strict")) orelse false;
            decl.* = .{ .name = name, .params_json = try self.declSchema(tv.object, strict) };
            dst.* = self.nestedFunctionJson(tv.object, name) catch return error.OutOfMemory;
        }
        return .{ .json = json, .decls = decls };
    }

    fn nestedFunctionJson(self: *Parser, tool: std.json.ObjectMap, name: []const u8) ![]const u8 {
        var aw = std.Io.Writer.Allocating.init(self.arena);
        var s: std.json.Stringify = .{ .writer = &aw.writer };
        try s.beginObject();
        try s.objectField("type");
        try s.write("function");
        try s.objectField("function");
        try s.beginObject();
        try s.objectField("name");
        try s.write(name);
        if (tool.get("description")) |d| {
            if (d == .string) {
                try s.objectField("description");
                try s.write(d.string);
            }
        }
        if (tool.get("parameters")) |pv| {
            if (pv != .null) {
                const ptext = try std.json.Stringify.valueAlloc(self.arena, pv, .{});
                try s.objectField("parameters");
                try s.print("{s}", .{ptext});
            }
        }
        try s.endObject();
        try s.endObject();
        return aw.written();
    }

    /// Shared `tool_choice` handling: "none" drops the declarations (the
    /// model must not call tools, so they are not offered), "auto" keeps
    /// them, "required" and a named function stash the forced set —
    /// `finishCommon` compiles it into the forced-call grammar. `flat_name`
    /// selects the Responses object shape ({"type":"function","name"})
    /// over chat's nested one.
    fn applyToolChoice(self: *Parser, set: *ToolSet, comptime flat_name: bool) Error!void {
        const tc = self.optField(self.obj, "tool_choice") orelse return;
        switch (tc) {
            .string => |s| {
                if (std.mem.eql(u8, s, "none")) {
                    set.* = .{};
                    return;
                }
                if (std.mem.eql(u8, s, "auto")) return;
                if (std.mem.eql(u8, s, "required")) return self.forceDecls(set.decls);
                return self.failInvalid("tool_choice must be \"none\", \"auto\", \"required\", or a named function", "tool_choice");
            },
            .object => |o| {
                const t = (try self.optString(o, "type")) orelse
                    return self.failInvalid("tool_choice missing \"type\"", "tool_choice");
                if (!std.mem.eql(u8, t, "function"))
                    return self.failInvalid("tool_choice type must be \"function\"", "tool_choice");
                const name = if (flat_name)
                    (try self.optString(o, "name")) orelse
                        return self.failInvalid("tool_choice missing \"name\"", "tool_choice")
                else blk: {
                    const fv = self.optField(o, "function") orelse
                        return self.failInvalid("tool_choice missing \"function\"", "tool_choice");
                    if (fv != .object) return self.failInvalid("expected an object", "tool_choice");
                    break :blk (try self.optString(fv.object, "name")) orelse
                        return self.failInvalid("tool_choice function missing \"name\"", "tool_choice");
                };
                return self.forceNamed(set.decls, name);
            },
            else => return self.failInvalid("tool_choice must be a string or an object", "tool_choice"),
        }
    }

    fn forceNamed(self: *Parser, decls: []const toolcall.Decl, name: []const u8) Error!void {
        for (decls) |d| {
            if (std.mem.eql(u8, d.name, name)) {
                const one = try self.arena.alloc(toolcall.Decl, 1);
                one[0] = d;
                return self.forceDecls(one);
            }
        }
        return self.failInvalid("tool_choice names an undeclared function", "tool_choice");
    }

    fn forceDecls(self: *Parser, decls: []const toolcall.Decl) Error!void {
        if (decls.len == 0)
            return self.failInvalid("tool_choice requires tools to be declared", "tool_choice");
        for (decls) |d| {
            if (!toolcall.plainName(d.name))
                return self.failInvalid("tool names must use [A-Za-z0-9_.:-] characters for a forced tool_choice", "tools");
        }
        self.forced_decls = decls;
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
        try self.rejectField(obj, "functions", "the legacy functions API is not supported; use tools");
        try self.rejectField(obj, "function_call", "the legacy functions API is not supported; use tool_choice");
        try self.rejectField(obj, "logprobs", "logprobs are not supported by this server");
        try self.rejectField(obj, "top_logprobs", "logprobs are not supported by this server");
        try self.rejectField(obj, "logit_bias", "logit_bias is not supported by this server");
        try self.rejectField(obj, "audio", "audio output is not supported by this server");
        try self.rejectField(obj, "prediction", "predicted outputs are not supported by this server");
        try self.rejectField(obj, "web_search_options", "web search is not supported by this server");
        if (try self.optInt(obj, "n", 1)) |n| {
            if (n != 1) return self.fail(ErrorInfo.unsupported("only n=1 is supported by this server", "n"));
        }

        var tools = try self.parseChatTools();
        try self.applyToolChoice(&tools, false);

        // messages
        const messages_v = self.optField(obj, "messages") orelse
            return self.failInvalid("missing \"messages\"", "messages");
        if (messages_v != .array) return self.failInvalid("expected an array", "messages");
        var messages: std.ArrayList(llm.chat.Message) = .empty;
        var tool_fold: std.ArrayList(u8) = .empty;
        for (messages_v.array.items) |mv| {
            if (mv != .object) return self.failInvalid("messages must be objects", "messages");
            const mobj = mv.object;
            const role_s = (try self.optString(mobj, "role")) orelse
                return self.failInvalid("message missing \"role\"", "messages");

            if (std.mem.eql(u8, role_s, "tool")) {
                if (self.info.tool_style == .none)
                    return self.fail(ErrorInfo.unsupported("tool messages are not supported by this model backend", "messages"));
                const content_v = self.optField(mobj, "content") orelse
                    return self.failInvalid("tool message missing \"content\"", "messages");
                toolcall.appendResponseSection(self.arena, &tool_fold, try self.contentText(content_v, "messages")) catch return error.OutOfMemory;
                continue;
            }
            if (std.mem.eql(u8, role_s, "function"))
                return self.fail(ErrorInfo.unsupported("the legacy functions API is not supported; use tool messages", "messages"));

            try self.flushToolResponses(&messages, &tool_fold);

            const role: llm.chat.Message.Role = if (std.mem.eql(u8, role_s, "system") or std.mem.eql(u8, role_s, "developer"))
                .system
            else if (std.mem.eql(u8, role_s, "user"))
                .user
            else if (std.mem.eql(u8, role_s, "assistant"))
                .assistant
            else
                return self.failInvalid("unknown message role", "messages");

            const calls_v = self.optField(mobj, "tool_calls");
            if (calls_v != null and self.info.tool_style == .none)
                return self.fail(ErrorInfo.unsupported("tool messages are not supported by this model backend", "messages"));
            if (calls_v != null and role != .assistant)
                return self.failInvalid("tool_calls belong on assistant messages", "messages");

            var turn: std.ArrayList(u8) = .empty;
            if (self.optField(mobj, "content")) |content_v| {
                turn.appendSlice(self.arena, try self.contentText(content_v, "messages")) catch return error.OutOfMemory;
            } else if (calls_v == null) {
                // Content is optional only on a call-carrying assistant turn.
                return self.failInvalid("message missing \"content\"", "messages");
            }
            if (calls_v) |cv| {
                if (cv != .array) return self.failInvalid("expected an array", "messages");
                for (cv.array.items) |tcv| {
                    if (tcv != .object) return self.failInvalid("tool_calls must be objects", "messages");
                    const fv = self.optField(tcv.object, "function") orelse
                        return self.failInvalid("tool call missing \"function\"", "messages");
                    if (fv != .object) return self.failInvalid("expected an object", "messages");
                    const name = (try self.optString(fv.object, "name")) orelse
                        return self.failInvalid("tool call missing function \"name\"", "messages");
                    const args = (try self.optString(fv.object, "arguments")) orelse "{}";
                    toolcall.appendCallSection(self.arena, &turn, name, args) catch return error.OutOfMemory;
                }
            }
            try messages.append(self.arena, .{ .role = role, .content = turn.items });
        }
        try self.flushToolResponses(&messages, &tool_fold);
        try self.applyTools(&parsed, &messages, tools.json);
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
        var tools = try self.parseResponsesTools();
        try self.applyToolChoice(&tools, true);
        if (try self.optString(obj, "truncation")) |tr| {
            if (!std.mem.eql(u8, tr, "disabled"))
                return self.fail(ErrorInfo.unsupported("only truncation=\"disabled\" is supported", "truncation"));
        }

        var messages: std.ArrayList(llm.chat.Message) = .empty;
        var tool_fold: std.ArrayList(u8) = .empty;

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
                    try self.flushToolResponses(&messages, &tool_fold);
                    try messages.append(self.arena, .{
                        .role = role,
                        .content = try self.contentText(content_v, "input"),
                    });
                } else if (std.mem.eql(u8, itype, "reasoning")) {
                    // Prior-turn reasoning items (Codex replays them): the
                    // reference templates drop prior reasoning, so do we.
                } else if (std.mem.eql(u8, itype, "function_call")) {
                    if (self.info.tool_style == .none)
                        return self.fail(ErrorInfo.unsupported("tool items are not supported by this model backend", "input"));
                    const name = (try self.optString(iobj, "name")) orelse
                        return self.failInvalid("function_call item missing \"name\"", "input");
                    const args = (try self.optString(iobj, "arguments")) orelse "{}";
                    try self.flushToolResponses(&messages, &tool_fold);
                    try self.appendCallToAssistant(&messages, name, args);
                } else if (std.mem.eql(u8, itype, "function_call_output")) {
                    if (self.info.tool_style == .none)
                        return self.fail(ErrorInfo.unsupported("tool items are not supported by this model backend", "input"));
                    const output = (try self.optString(iobj, "output")) orelse
                        return self.failInvalid("function_call_output item missing \"output\"", "input");
                    toolcall.appendResponseSection(self.arena, &tool_fold, output) catch return error.OutOfMemory;
                } else if (std.mem.eql(u8, itype, "item_reference")) {
                    return self.fail(ErrorInfo.unsupported("item references need a conversation store; this server is stateless", "input"));
                } else {
                    return self.fail(ErrorInfo.unsupported("unsupported input item type", "input"));
                }
            },
            else => return self.failInvalid("input must be a string or an array of items", "input"),
        }
        try self.flushToolResponses(&messages, &tool_fold);
        try self.applyTools(&parsed, &messages, tools.json);
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

test "chat parse: tools render into the system slot, history folds" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const info = types.Info{ .model_id = "t", .context_len = 8192, .tool_style = .hermes };

    const body =
        \\{"messages":[
        \\  {"role":"system","content":"Be terse."},
        \\  {"role":"user","content":"weather?"},
        \\  {"role":"assistant","tool_calls":[{"id":"call_1","type":"function",
        \\    "function":{"name":"get_weather","arguments":"{\"city\": \"Paris\"}"}}]},
        \\  {"role":"tool","tool_call_id":"call_1","content":"22C"},
        \\  {"role":"tool","tool_call_id":"call_1","content":"sunny"},
        \\  {"role":"user","content":"and tomorrow?"}
        \\ ],
        \\ "tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object"}}}]}
    ;
    const p = parse(arena, .chat, body, info).ok;
    try std.testing.expect(p.tools_active);
    // system + user + assistant(with call) + folded tool turn + user
    try std.testing.expectEqual(@as(usize, 5), p.gen.messages.len);
    try std.testing.expectEqual(llm.chat.Message.Role.system, p.gen.messages[0].role);
    try std.testing.expect(std.mem.startsWith(u8, p.gen.messages[0].content, "Be terse.\n\n# Tools\n\n"));
    try std.testing.expect(std.mem.indexOf(u8, p.gen.messages[0].content, "\"name\":\"get_weather\"") != null);
    try std.testing.expectEqualStrings(
        "\n<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\": \"Paris\"}}\n</tool_call>",
        p.gen.messages[2].content,
    );
    try std.testing.expectEqual(llm.chat.Message.Role.user, p.gen.messages[3].role);
    try std.testing.expectEqualStrings(
        "<tool_response>\n22C\n</tool_response>\n<tool_response>\nsunny\n</tool_response>",
        p.gen.messages[3].content,
    );

    // tool_choice "none" drops the declarations.
    {
        const none_body =
            \\{"messages":[{"role":"user","content":"x"}],"tool_choice":"none",
            \\ "tools":[{"type":"function","function":{"name":"f"}}]}
        ;
        const q = parse(arena, .chat, none_body, info).ok;
        try std.testing.expect(!q.tools_active);
        try std.testing.expectEqual(llm.chat.Message.Role.user, q.gen.messages[0].role);
    }
    // "required" and named forcing cannot be guaranteed.
    {
        const req_body =
            \\{"messages":[{"role":"user","content":"x"}],"tool_choice":"required",
            \\ "tools":[{"type":"function","function":{"name":"f"}}]}
        ;
        try std.testing.expectEqualStrings("tool_choice", parse(arena, .chat, req_body, info).err.param.?);
    }
    // Backends without a tool convention keep rejecting.
    {
        const no_style = types.Info{ .model_id = "t", .context_len = 8192 };
        const tool_body = "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"f\"}}]}";
        try std.testing.expectEqualStrings("tools", parse(arena, .chat, tool_body, no_style).err.param.?);
    }
}

test "chat parse: forced tool_choice compiles to the call grammar" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const info = types.Info{ .model_id = "t", .context_len = 8192, .tool_style = .hermes, .caps = .{ .grammar = true } };

    const tools_frag =
        \\"tools":[
        \\ {"type":"function","function":{"name":"get_weather","strict":true,
        \\   "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}}},
        \\ {"type":"function","function":{"name":"ping"}}]
    ;
    // "required": one grammar alternative per declared tool; strict brings
    // its schema, non-strict gets the permissive object.
    {
        const body = std.fmt.comptimePrint("{{\"messages\":[{{\"role\":\"user\",\"content\":\"x\"}}],\"tool_choice\":\"required\",{s}}}", .{tools_frag});
        const p = parse(arena, .chat, body, info).ok;
        const grammar = p.gen.constraint.?.lark;
        try std.testing.expect(std.mem.startsWith(u8, grammar, "start: c0 | c1\n"));
        try std.testing.expect(std.mem.indexOf(u8, grammar, "\\\"get_weather\\\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, grammar, "a0: %json {\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"],\"additionalProperties\":false}") != null);
        try std.testing.expect(std.mem.indexOf(u8, grammar, "a1: %json {\"type\":\"object\"}") != null);
        try std.testing.expect(p.tools_active);
    }
    // Named function: only that alternative.
    {
        const body = std.fmt.comptimePrint("{{\"messages\":[{{\"role\":\"user\",\"content\":\"x\"}}],\"tool_choice\":{{\"type\":\"function\",\"function\":{{\"name\":\"ping\"}}}},{s}}}", .{tools_frag});
        const p = parse(arena, .chat, body, info).ok;
        const grammar = p.gen.constraint.?.lark;
        try std.testing.expect(std.mem.startsWith(u8, grammar, "start: c0\n"));
        try std.testing.expect(std.mem.indexOf(u8, grammar, "\\\"ping\\\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, grammar, "get_weather") == null);
    }
    // The grammar owns the reply: response_format cannot combine with it.
    {
        const body = std.fmt.comptimePrint("{{\"messages\":[{{\"role\":\"user\",\"content\":\"x\"}}],\"tool_choice\":\"required\",\"response_format\":{{\"type\":\"json_object\"}},{s}}}", .{tools_frag});
        try std.testing.expectEqualStrings("tool_choice", parse(arena, .chat, body, info).err.param.?);
    }
    // Unknown named function.
    {
        const body = std.fmt.comptimePrint("{{\"messages\":[{{\"role\":\"user\",\"content\":\"x\"}}],\"tool_choice\":{{\"type\":\"function\",\"function\":{{\"name\":\"nope\"}}}},{s}}}", .{tools_frag});
        try std.testing.expectEqualStrings("tool_choice", parse(arena, .chat, body, info).err.param.?);
    }
    // Without llguidance the guarantee cannot be honored: 501.
    {
        const no_grammar = types.Info{ .model_id = "t", .context_len = 8192, .tool_style = .hermes };
        const body = std.fmt.comptimePrint("{{\"messages\":[{{\"role\":\"user\",\"content\":\"x\"}}],\"tool_choice\":\"required\",{s}}}", .{tools_frag});
        const outcome = parse(arena, .chat, body, no_grammar);
        try std.testing.expectEqual(std.http.Status.not_implemented, outcome.err.status);
        try std.testing.expectEqualStrings("tool_choice", outcome.err.param.?);
    }
    // Responses named form is flat.
    {
        const body =
            \\{"input":"x","tool_choice":{"type":"function","name":"f"},
            \\ "tools":[{"type":"function","name":"f","parameters":{"type":"object"}}]}
        ;
        const p = parse(arena, .responses, body, info).ok;
        try std.testing.expect(std.mem.startsWith(u8, p.gen.constraint.?.lark, "start: c0\n"));
    }
}

test "responses parse: flat tools, function_call items join the turn" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const info = types.Info{ .model_id = "t", .context_len = 8192, .tool_style = .hermes };

    const body =
        \\{"input":[
        \\  {"role":"user","content":"weather?"},
        \\  {"type":"message","role":"assistant","content":"Checking."},
        \\  {"type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{\"city\": \"Paris\"}"},
        \\  {"type":"function_call_output","call_id":"call_1","output":"22C"},
        \\  {"role":"user","content":"thanks"}
        \\ ],
        \\ "tools":[{"type":"function","name":"get_weather","description":"d","parameters":{"type":"object"}}]}
    ;
    const p = parse(arena, .responses, body, info).ok;
    try std.testing.expect(p.tools_active);
    // tools-system + user + assistant(text+call) + folded output + user
    try std.testing.expectEqual(@as(usize, 5), p.gen.messages.len);
    try std.testing.expectEqual(llm.chat.Message.Role.system, p.gen.messages[0].role);
    try std.testing.expect(std.mem.indexOf(u8, p.gen.messages[0].content, "{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"d\",\"parameters\":{\"type\":\"object\"}}}") != null);
    try std.testing.expectEqualStrings(
        "Checking.\n<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\": \"Paris\"}}\n</tool_call>",
        p.gen.messages[2].content,
    );
    try std.testing.expectEqualStrings("<tool_response>\n22C\n</tool_response>", p.gen.messages[3].content);

    // Hosted tool types stay rejected.
    {
        const hosted = "{\"input\":\"x\",\"tools\":[{\"type\":\"web_search\"}]}";
        try std.testing.expectEqualStrings("tools", parse(arena, .responses, hosted, info).err.param.?);
    }
}
