//! Anthropic Messages API wire layer (`POST /v1/messages`): a translation
//! layer, not a second engine path — requests normalize into the same
//! `openai.Parsed` the OpenAI dialects produce, so prompt rendering,
//! scheduling, and generation stay single-sourced; only request parsing
//! (here) and the response/SSE framing (`emitter.zig`, `.anthropic` wire)
//! are Anthropic-shaped. Claude Code is the reference client.
//!
//! Deliberate deviations from the hosted API, all visible on the wire:
//! * Tool calling works on hermes-style backends (qwen3;
//!   `types.ToolStyle`): declarations render into the system block via
//!   `toolcall.zig`, `tool_use`/`tool_result` history folds into the
//!   template text, and the emitter scans replies for calls. On backends
//!   without a tool convention, declarations are accepted and dropped —
//!   under `tool_choice: auto` a reply with no `tool_use` blocks is valid
//!   behavior, and it keeps tool-sending clients usable as plain chat —
//!   while tool blocks in the history are rejected. `tool_choice`
//!   `any`/`tool` (a guaranteed tool call) compiles to a forced-call
//!   grammar (`toolcall.forcedCallGrammar`, input_schema-conformant
//!   arguments; needs a hermes backend and a `-Dllguidance=true` build) —
//!   rejected where that machinery is absent.
//! * `thinking: {"type": "enabled"|"adaptive"}` switches the model's
//!   reasoning channel on when the backend has one and is otherwise a no-op
//!   (the reply simply carries no `thinking` blocks) — a hard error would
//!   make thinking-by-default clients unusable against non-reasoning
//!   models.
//! * `stop_sequences` stop generation before the matching text streams and
//!   are attributed: the reply's `stop_reason` is `stop_sequence` with the
//!   fired sequence echoed in `stop_sequence`.
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
const types = @import("fucina_llm").serving;
const openai = @import("openai.zig");
const toolcall = @import("toolcall.zig");

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
    /// Set by tool_choice `any`/`tool`: compiled into the forced-call
    /// grammar once the other constraints are resolved.
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

    /// Flatten a text-only content value (mid-conversation system messages,
    /// tool_result payloads) into one string.
    fn messageText(self: *Parser, content: Value) Error![]const u8 {
        switch (content) {
            .string => |s| return s,
            .array => |parts| {
                var out: std.ArrayList(u8) = .empty;
                for (parts.items) |part| {
                    if (part != .object) return self.failInvalid("content blocks must be objects", "messages");
                    const ptype = (try self.optString(part.object, "type")) orelse
                        return self.failInvalid("content block missing \"type\"", "messages");
                    if (!std.mem.eql(u8, ptype, "text"))
                        return self.fail(ErrorInfo.unsupported("only text blocks are supported here", "messages"));
                    const text = (try self.optString(part.object, "text")) orelse
                        return self.failInvalid("text block missing \"text\"", "messages");
                    try out.appendSlice(self.arena, text);
                }
                return out.items;
            },
            else => return self.failInvalid("message content must be a string or an array of blocks", "messages"),
        }
    }

    /// Flush accumulated tool-result sections as one user turn (Qwen3's
    /// template shape: consecutive results share the turn).
    fn flushToolResponses(self: *Parser, messages: *std.ArrayList(llm.chat.Message), fold: *std.ArrayList(u8)) Error!void {
        if (fold.items.len == 0) return;
        try messages.append(self.arena, .{ .role = .user, .content = try fold.toOwnedSlice(self.arena) });
    }

    /// One user/assistant message: text blocks concatenate,
    /// `thinking`/`redacted_thinking` blocks drop (prior-turn reasoning the
    /// templates re-render without, like the Responses dialect's reasoning
    /// items), `tool_use` becomes a call section of this turn, and
    /// `tool_result` accumulates into `fold` — results answer the previous
    /// assistant turn, so they flush as their own user turn before this
    /// message's text.
    fn appendTurn(self: *Parser, messages: *std.ArrayList(llm.chat.Message), fold: *std.ArrayList(u8), role: llm.chat.Message.Role, content: Value) Error!void {
        const hermes = self.info.tool_style == .hermes;
        const parts = switch (content) {
            .string => |s| {
                try self.flushToolResponses(messages, fold);
                try messages.append(self.arena, .{ .role = role, .content = s });
                return;
            },
            .array => |parts| parts.items,
            else => return self.failInvalid("message content must be a string or an array of blocks", "messages"),
        };
        var turn: std.ArrayList(u8) = .empty;
        var had_call = false;
        var had_result = false;
        for (parts) |part| {
            if (part != .object) return self.failInvalid("content blocks must be objects", "messages");
            const pobj = part.object;
            const ptype = (try self.optString(pobj, "type")) orelse
                return self.failInvalid("content block missing \"type\"", "messages");
            if (std.mem.eql(u8, ptype, "text")) {
                const text = (try self.optString(pobj, "text")) orelse
                    return self.failInvalid("text block missing \"text\"", "messages");
                turn.appendSlice(self.arena, text) catch return error.OutOfMemory;
            } else if (std.mem.eql(u8, ptype, "thinking") or std.mem.eql(u8, ptype, "redacted_thinking")) {
                // Dropped: prior-turn reasoning.
            } else if (std.mem.eql(u8, ptype, "tool_use")) {
                if (!hermes)
                    return self.fail(ErrorInfo.unsupported("tool use is not supported by this model backend", "messages"));
                const name = (try self.optString(pobj, "name")) orelse
                    return self.failInvalid("tool_use block missing \"name\"", "messages");
                const args: []const u8 = if (self.optField(pobj, "input")) |input| blk: {
                    if (input != .object) return self.failInvalid("tool_use \"input\" must be an object", "messages");
                    break :blk std.json.Stringify.valueAlloc(self.arena, input, .{}) catch return error.OutOfMemory;
                } else "{}";
                toolcall.appendCallSection(self.arena, &turn, name, args) catch return error.OutOfMemory;
                had_call = true;
            } else if (std.mem.eql(u8, ptype, "tool_result")) {
                if (!hermes)
                    return self.fail(ErrorInfo.unsupported("tool use is not supported by this model backend", "messages"));
                const result = if (self.optField(pobj, "content")) |rc| try self.messageText(rc) else "";
                toolcall.appendResponseSection(self.arena, fold, result) catch return error.OutOfMemory;
                had_result = true;
            } else if (std.mem.eql(u8, ptype, "image") or std.mem.eql(u8, ptype, "document")) {
                return self.fail(ErrorInfo.unsupported("only text content is supported (no images or documents)", "messages"));
            } else {
                return self.fail(ErrorInfo.unsupported("unsupported content block type", "messages"));
            }
        }
        // A results-only message contributes no turn of its own (the fold
        // keeps accumulating across consecutive result messages).
        const has_turn = turn.items.len > 0 or had_call or (role == .user and !had_result);
        if (has_turn) {
            try self.flushToolResponses(messages, fold);
            try messages.append(self.arena, .{ .role = role, .content = turn.items });
        }
    }

    /// Declared custom tools ({name, description?, input_schema?}) rebuilt
    /// into the nested object the hermes system block embeds. Server-tool
    /// types have no local execution and are refused. `input_schema` is the
    /// wire contract for the call's input, so a forced choice enforces it
    /// in the grammar.
    fn parseTools(self: *Parser, items: []const Value) Error!ToolSet {
        if (items.len == 0) return .{};
        const json = try self.arena.alloc([]const u8, items.len);
        const decls = try self.arena.alloc(toolcall.Decl, items.len);
        for (items, json, decls) |tv, *dst, *decl| {
            if (tv != .object) return self.failInvalid("tools must be objects", "tools");
            if (try self.optString(tv.object, "type")) |t| {
                if (!std.mem.eql(u8, t, "custom"))
                    return self.fail(ErrorInfo.unsupported("server-side tools are not supported by this server", "tools"));
            }
            const name = (try self.optString(tv.object, "name")) orelse
                return self.failInvalid("tool missing \"name\"", "tools");
            decl.* = .{ .name = name };
            if (self.optField(tv.object, "input_schema")) |sv|
                decl.params_json = std.json.Stringify.valueAlloc(self.arena, sv, .{}) catch return error.OutOfMemory;
            dst.* = self.nestedFunctionJson(tv.object, name) catch return error.OutOfMemory;
        }
        return .{ .json = json, .decls = decls };
    }

    fn forceNamed(self: *Parser, decls: []const toolcall.Decl, name: []const u8) Error!void {
        for (decls) |d| {
            if (std.mem.eql(u8, d.name, name)) {
                const one = try self.arena.alloc(toolcall.Decl, 1);
                one[0] = d;
                return self.forceDecls(one);
            }
        }
        return self.failInvalid("tool_choice names an undeclared tool", "tool_choice");
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
        if (tool.get("input_schema")) |sv| {
            if (sv != .null) {
                const stext = try std.json.Stringify.valueAlloc(self.arena, sv, .{});
                try s.objectField("parameters");
                try s.print("{s}", .{stext});
            }
        }
        try s.endObject();
        try s.endObject();
        return aw.written();
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

        var stop_list: std.ArrayList([]const u8) = .empty;
        if (self.optField(obj, "stop_sequences")) |v| {
            if (v != .array) return self.failInvalid("expected an array", "stop_sequences");
            for (v.array.items) |item| {
                if (item != .string or item.string.len == 0)
                    return self.failInvalid("expected non-empty strings", "stop_sequences");
                stop_list.append(self.arena, item.string) catch return error.OutOfMemory;
            }
        }
        // Declarations render on hermes backends; elsewhere they are
        // accepted and dropped (see the module doc).
        var tools = ToolSet{};
        if (self.optField(obj, "tools")) |v| {
            if (v != .array) return self.failInvalid("expected an array", "tools");
            if (inf.tool_style == .hermes) tools = try self.parseTools(v.array.items);
        }
        if (self.optField(obj, "tool_choice")) |v| {
            if (v != .object) return self.failInvalid("expected an object", "tool_choice");
            const kind = (try self.optString(v.object, "type")) orelse
                return self.failInvalid("tool_choice missing \"type\"", "tool_choice");
            // disable_parallel_tool_use is moot here: a forced reply is one
            // call, and auto-mode call counts follow the model.
            if (std.mem.eql(u8, kind, "none")) {
                tools = .{}; // the model must not call tools: don't offer them
            } else if (std.mem.eql(u8, kind, "any") or std.mem.eql(u8, kind, "tool")) {
                if (inf.tool_style == .none)
                    return self.fail(ErrorInfo.unsupported("this model backend has no tool convention; a forced tool call cannot be honored", "tool_choice"));
                if (std.mem.eql(u8, kind, "any")) {
                    try self.forceDecls(tools.decls);
                } else {
                    const name = (try self.optString(v.object, "name")) orelse
                        return self.failInvalid("tool_choice missing \"name\"", "tool_choice");
                    try self.forceNamed(tools.decls, name);
                }
            } else if (!std.mem.eql(u8, kind, "auto")) {
                return self.failInvalid("tool_choice type must be \"auto\", \"none\", \"any\", or \"tool\"", "tool_choice");
            }
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
        var tool_fold: std.ArrayList(u8) = .empty;
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
                try self.flushToolResponses(&messages, &tool_fold);
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
            try self.appendTurn(&messages, &tool_fold, role, content_v);
        }
        try self.flushToolResponses(&messages, &tool_fold);
        if (tools.json.len > 0) {
            // Declarations render into the leading system slot (Qwen3's
            // template shape) and the reply scanner arms.
            if (messages.items.len > 0 and messages.items[0].role == .system) {
                messages.items[0].content = toolcall.renderSystemWithTools(self.arena, messages.items[0].content, tools.json) catch return error.OutOfMemory;
            } else {
                const content = toolcall.renderSystemWithTools(self.arena, "", tools.json) catch return error.OutOfMemory;
                messages.insert(self.arena, 0, .{ .role = .system, .content = content }) catch return error.OutOfMemory;
            }
            parsed.tools_active = true;
        }
        parsed.gen.messages = messages.items;
        parsed.gen.stop = stop_list.items;

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

        // A forced tool_choice compiles to a grammar over the hermes call
        // shape; it owns the reply from token 0 and excludes
        // output_config.format.
        if (self.forced_decls) |decls| {
            if (parsed.gen.constraint != null)
                return self.failInvalid("a forced tool_choice already constrains the reply; output_config.format cannot combine with it", "tool_choice");
            if (!inf.caps.grammar) {
                return self.fail(.{
                    .status = .not_implemented,
                    .kind = "not_supported_error",
                    .message = "a guaranteed tool call needs constrained decoding: this server (or this model's tokenizer) was built without llguidance support",
                    .param = "tool_choice",
                });
            }
            parsed.gen.constraint = .{ .lark = toolcall.forcedCallGrammar(self.arena, decls) catch return error.OutOfMemory };
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
        // stop_sequences must be non-empty strings.
        .{ .body = std.fmt.comptimePrint("{{\"model\":\"m\",\"max_tokens\":8,{s},\"stop_sequences\":[\"\"]}}", .{user_msg}), .param = "stop_sequences" },
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

    // Populated stop_sequences land on the request in list order.
    const with = parse(arena, std.fmt.comptimePrint("{{\"model\":\"m\",\"max_tokens\":8,{s},\"stop_sequences\":[\"END\",\"\\n\\n\"]}}", .{user_msg}), info);
    try std.testing.expectEqual(@as(usize, 2), with.ok.gen.stop.len);
    try std.testing.expectEqualStrings("END", with.ok.gen.stop[0]);
    try std.testing.expectEqualStrings("\n\n", with.ok.gen.stop[1]);
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

test "anthropic parse: hermes tools render, tool history folds" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const info = types.Info{ .model_id = "t", .context_len = 8192, .tool_style = .hermes };

    const body =
        \\{"model":"m","max_tokens":32,
        \\ "system":"Be terse.",
        \\ "messages":[
        \\  {"role":"user","content":"weather?"},
        \\  {"role":"assistant","content":[
        \\    {"type":"text","text":"Checking."},
        \\    {"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"Paris"}}]},
        \\  {"role":"user","content":[
        \\    {"type":"tool_result","tool_use_id":"toolu_1","content":"22C"},
        \\    {"type":"text","text":"and tomorrow?"}]}
        \\ ],
        \\ "tools":[{"name":"get_weather","description":"d","input_schema":{"type":"object"}}]}
    ;
    const p = parse(arena, body, info).ok;
    try std.testing.expect(p.tools_active);
    // system(+tools) + user + assistant(text+call) + folded result turn + user text
    try std.testing.expectEqual(@as(usize, 5), p.gen.messages.len);
    try std.testing.expect(std.mem.startsWith(u8, p.gen.messages[0].content, "Be terse.\n\n# Tools\n\n"));
    try std.testing.expect(std.mem.indexOf(u8, p.gen.messages[0].content, "{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"d\",\"parameters\":{\"type\":\"object\"}}}") != null);
    try std.testing.expectEqualStrings(
        "Checking.\n<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}\n</tool_call>",
        p.gen.messages[2].content,
    );
    try std.testing.expectEqual(llm.chat.Message.Role.user, p.gen.messages[3].role);
    try std.testing.expectEqualStrings("<tool_response>\n22C\n</tool_response>", p.gen.messages[3].content);
    try std.testing.expectEqualStrings("and tomorrow?", p.gen.messages[4].content);

    // tool_choice none drops the declarations (and the scanner stays off).
    {
        const none_body =
            \\{"model":"m","max_tokens":8,
            \\ "messages":[{"role":"user","content":"x"}],
            \\ "tool_choice":{"type":"none"},
            \\ "tools":[{"name":"f","input_schema":{"type":"object"}}]}
        ;
        const q = parse(arena, none_body, info).ok;
        try std.testing.expect(!q.tools_active);
    }
    // Server-side tool types are refused.
    {
        const hosted =
            \\{"model":"m","max_tokens":8,
            \\ "messages":[{"role":"user","content":"x"}],
            \\ "tools":[{"type":"web_search_20260209","name":"web_search"}]}
        ;
        try std.testing.expectEqualStrings("tools", parse(arena, hosted, info).err.param.?);
    }
}

test "anthropic parse: tool_choice any/tool compile to the call grammar" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const info = types.Info{ .model_id = "t", .context_len = 8192, .tool_style = .hermes, .caps = .{ .grammar = true, .think = true } };

    const base =
        \\"model":"m","max_tokens":32,
        \\"messages":[{"role":"user","content":"x"}],
        \\"tools":[{"name":"get_weather","input_schema":{"type":"object","properties":{"city":{"type":"string"}}}}]
    ;
    // any: forced over every declared tool, input_schema enforced;
    // thinking yields to the grammar.
    {
        const body = std.fmt.comptimePrint("{{{s},\"thinking\":{{\"type\":\"enabled\"}},\"tool_choice\":{{\"type\":\"any\"}}}}", .{base});
        const p = parse(arena, body, info).ok;
        const grammar = p.gen.constraint.?.lark;
        try std.testing.expect(std.mem.startsWith(u8, grammar, "start: c0\n"));
        try std.testing.expect(std.mem.indexOf(u8, grammar, "a0: %json {\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}}}") != null);
        try std.testing.expect(!p.gen.think);
    }
    // tool: the named one.
    {
        const body = std.fmt.comptimePrint("{{{s},\"tool_choice\":{{\"type\":\"tool\",\"name\":\"get_weather\"}}}}", .{base});
        const p = parse(arena, body, info).ok;
        try std.testing.expect(std.mem.indexOf(u8, p.gen.constraint.?.lark, "\\\"get_weather\\\"") != null);
    }
    // Unknown name.
    {
        const body = std.fmt.comptimePrint("{{{s},\"tool_choice\":{{\"type\":\"tool\",\"name\":\"nope\"}}}}", .{base});
        try std.testing.expectEqual(std.http.Status.bad_request, parse(arena, body, info).err.status);
    }
    // Without llguidance: 501.
    {
        const no_grammar = types.Info{ .model_id = "t", .context_len = 8192, .tool_style = .hermes };
        const body = std.fmt.comptimePrint("{{{s},\"tool_choice\":{{\"type\":\"any\"}}}}", .{base});
        try std.testing.expectEqual(std.http.Status.not_implemented, parse(arena, body, no_grammar).err.status);
    }
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
