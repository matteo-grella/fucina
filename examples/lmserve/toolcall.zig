//! Hermes-style tool calling — the Qwen3 chat-template convention, shared
//! by all three wire dialects (`types.ToolStyle.hermes`). Three pieces:
//!
//! * prompt rendering: declarations as JSON inside `<tools></tools>` in the
//!   system block, prior calls as `<tool_call>` sections of assistant
//!   turns, results as `<tool_response>` sections of user turns — the text
//!   Qwen3's own chat template produces, built here so the dialect parsers
//!   fold everything into plain role+content messages and the backends
//!   stay tool-agnostic;
//! * the reply `Scanner`: splits streamed content bytes into plain text
//!   and captured `<tool_call>…</tool_call>` regions (markers may sit
//!   anywhere in the reply and split across chunks);
//! * `parseCall`: validates a captured region as `{"name", "arguments"}` —
//!   anything else passes through as plain content, marker included, so
//!   malformed model output is never dropped.

const std = @import("std");

const Allocator = std.mem.Allocator;

pub const open_marker = "<tool_call>";
pub const close_marker = "</tool_call>";

/// The system-block "# Tools" section, appended to the request's own
/// system text (`base`, may be empty). `tools_json` holds one serialized
/// `{"type":"function","function":{…}}` object per declared tool.
pub fn renderSystemWithTools(arena: Allocator, base: []const u8, tools_json: []const []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    if (base.len > 0) {
        try out.appendSlice(arena, base);
        try out.appendSlice(arena, "\n\n");
    }
    try out.appendSlice(arena, "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>");
    for (tools_json) |tool| {
        try out.append(arena, '\n');
        try out.appendSlice(arena, tool);
    }
    try out.appendSlice(arena, "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>");
    return out.items;
}

/// The call object a `<tool_call>` section carries: name JSON-escaped,
/// `args_json` embedded verbatim (already-validated JSON).
pub fn callJson(arena: Allocator, name: []const u8, args_json: []const u8) ![]const u8 {
    var aw = std.Io.Writer.Allocating.init(arena);
    var s: std.json.Stringify = .{ .writer = &aw.writer };
    try s.beginObject();
    try s.objectField("name");
    try s.write(name);
    try s.objectField("arguments");
    try s.print("{s}", .{args_json});
    try s.endObject();
    return aw.written();
}

/// Append one prior-call section to an assistant turn's content.
pub fn appendCallSection(arena: Allocator, out: *std.ArrayList(u8), name: []const u8, args_json: []const u8) !void {
    try out.appendSlice(arena, "\n" ++ open_marker ++ "\n");
    try out.appendSlice(arena, try callJson(arena, name, args_json));
    try out.appendSlice(arena, "\n" ++ close_marker);
}

/// Append one result section to a tool-response user turn (consecutive
/// results share the turn, each in its own section).
pub fn appendResponseSection(arena: Allocator, out: *std.ArrayList(u8), content: []const u8) !void {
    if (out.items.len > 0) try out.append(arena, '\n');
    try out.appendSlice(arena, "<tool_response>\n");
    try out.appendSlice(arena, content);
    try out.appendSlice(arena, "\n</tool_response>");
}

pub const Call = struct {
    name: []const u8,
    /// Canonically re-serialized JSON object.
    args_json: []const u8,
};

/// Validate a captured region as a call. `arguments` must be an object (or
/// absent — an empty one); any other shape returns null and the caller
/// passes the capture through as content.
pub fn parseCall(arena: Allocator, raw: []const u8) ?Call {
    const root = std.json.parseFromSliceLeaky(std.json.Value, arena, raw, .{}) catch return null;
    if (root != .object) return null;
    const name_v = root.object.get("name") orelse return null;
    if (name_v != .string or name_v.string.len == 0) return null;
    const args_json: []const u8 = if (root.object.get("arguments")) |args| switch (args) {
        .object => std.json.Stringify.valueAlloc(arena, args, .{}) catch return null,
        .null => "{}",
        else => return null,
    } else "{}";
    return .{ .name = name_v.string, .args_json = args_json };
}

/// Streaming splitter over reply content. `ctx` (duck-typed) receives
/// `toolContent(bytes)` for plain text and `toolCall(raw)` for each
/// complete capture (markers stripped, whitespace-trimmed). Bytes are
/// buffered only as far as marker ambiguity requires: text older than a
/// possible `<tool_call>` prefix flushes immediately.
pub const Scanner = struct {
    arena: Allocator,
    state: enum { text, capture } = .text,
    /// Text mode: a trailing chunk suffix that could open a marker.
    /// Capture mode: everything since the open marker.
    buf: std.ArrayList(u8) = .empty,

    pub fn init(arena: Allocator) Scanner {
        return .{ .arena = arena };
    }

    pub fn feed(self: *Scanner, ctx: anytype, bytes: []const u8) !void {
        try self.buf.appendSlice(self.arena, bytes);
        while (true) {
            switch (self.state) {
                .text => {
                    if (std.mem.indexOf(u8, self.buf.items, open_marker)) |at| {
                        if (at > 0) try ctx.toolContent(self.buf.items[0..at]);
                        const rest_len = self.buf.items.len - (at + open_marker.len);
                        std.mem.copyForwards(u8, self.buf.items[0..rest_len], self.buf.items[at + open_marker.len ..]);
                        self.buf.shrinkRetainingCapacity(rest_len);
                        self.state = .capture;
                        continue;
                    }
                    const keep = markerPrefixSuffix(self.buf.items, open_marker);
                    const flush = self.buf.items.len - keep;
                    if (flush > 0) {
                        try ctx.toolContent(self.buf.items[0..flush]);
                        std.mem.copyForwards(u8, self.buf.items[0..keep], self.buf.items[flush..]);
                        self.buf.shrinkRetainingCapacity(keep);
                    }
                    return;
                },
                .capture => {
                    const at = std.mem.indexOf(u8, self.buf.items, close_marker) orelse return;
                    const raw = std.mem.trim(u8, self.buf.items[0..at], " \t\r\n");
                    try ctx.toolCall(try self.arena.dupe(u8, raw));
                    const rest_len = self.buf.items.len - (at + close_marker.len);
                    std.mem.copyForwards(u8, self.buf.items[0..rest_len], self.buf.items[at + close_marker.len ..]);
                    self.buf.shrinkRetainingCapacity(rest_len);
                    self.state = .text;
                },
            }
        }
    }

    /// End of reply. Held text flushes as content; an unterminated capture
    /// (budget cut the reply mid-call) is not a call — it flows back as
    /// content with its open marker restored.
    pub fn finish(self: *Scanner, ctx: anytype) !void {
        switch (self.state) {
            .text => if (self.buf.items.len > 0) try ctx.toolContent(self.buf.items),
            .capture => {
                try ctx.toolContent(open_marker);
                if (self.buf.items.len > 0) try ctx.toolContent(self.buf.items);
            },
        }
        self.buf.clearRetainingCapacity();
        self.state = .text;
    }
};

/// Length of the longest `bytes` suffix that is a proper prefix of
/// `marker` (the only text that may not flush yet).
fn markerPrefixSuffix(bytes: []const u8, marker: []const u8) usize {
    var len = @min(bytes.len, marker.len - 1);
    while (len > 0) : (len -= 1) {
        if (std.mem.endsWith(u8, bytes, marker[0..len])) return len;
    }
    return 0;
}

// ---- tests ----

const Collect = struct {
    arena: Allocator,
    content: std.ArrayList(u8) = .empty,
    calls: std.ArrayList([]const u8) = .empty,

    fn toolContent(self: *Collect, bytes: []const u8) !void {
        try self.content.appendSlice(self.arena, bytes);
    }

    fn toolCall(self: *Collect, raw: []const u8) !void {
        try self.calls.append(self.arena, raw);
    }
};

fn scan(arena: Allocator, chunks: []const []const u8) !Collect {
    var c = Collect{ .arena = arena };
    var s = Scanner.init(arena);
    for (chunks) |chunk| try s.feed(&c, chunk);
    try s.finish(&c);
    return c;
}

test "tool scanner: text, captures, split markers, trailing text" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    // Plain text passes through.
    {
        const r = try scan(arena, &.{ "hello ", "world" });
        try std.testing.expectEqualStrings("hello world", r.content.items);
        try std.testing.expectEqual(@as(usize, 0), r.calls.items.len);
    }
    // Text, then two calls with markers split across chunks.
    {
        const r = try scan(arena, &.{
            "Sure.\n<tool_",
            "call>\n{\"name\": \"a\"}\n</tool_call>",
            "\n<tool_call>{\"name\": \"b\"}</to",
            "ol_call>",
        });
        try std.testing.expectEqualStrings("Sure.\n\n", r.content.items);
        try std.testing.expectEqual(@as(usize, 2), r.calls.items.len);
        try std.testing.expectEqualStrings("{\"name\": \"a\"}", r.calls.items[0]);
        try std.testing.expectEqualStrings("{\"name\": \"b\"}", r.calls.items[1]);
    }
    // A lone '<' that never becomes a marker flushes as text.
    {
        const r = try scan(arena, &.{ "a < b, a <tool", " even" });
        try std.testing.expectEqualStrings("a < b, a <tool even", r.content.items);
    }
    // Budget-truncated capture flows back as content, marker restored.
    {
        const r = try scan(arena, &.{"answer <tool_call>{\"name\": \"cut"});
        try std.testing.expectEqualStrings("answer <tool_call>{\"name\": \"cut", r.content.items);
        try std.testing.expectEqual(@as(usize, 0), r.calls.items.len);
    }
    // Text after the last call keeps flowing as content.
    {
        const r = try scan(arena, &.{"<tool_call>{\"name\":\"x\"}</tool_call>done"});
        try std.testing.expectEqualStrings("done", r.content.items);
        try std.testing.expectEqual(@as(usize, 1), r.calls.items.len);
    }
}

test "parseCall: object args canonicalized, bad shapes rejected" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const ok = parseCall(arena, "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}").?;
    try std.testing.expectEqualStrings("get_weather", ok.name);
    try std.testing.expectEqualStrings("{\"city\":\"Paris\"}", ok.args_json);

    const no_args = parseCall(arena, "{\"name\": \"ping\"}").?;
    try std.testing.expectEqualStrings("{}", no_args.args_json);

    try std.testing.expect(parseCall(arena, "not json") == null);
    try std.testing.expect(parseCall(arena, "{\"arguments\": {}}") == null);
    try std.testing.expect(parseCall(arena, "{\"name\": \"x\", \"arguments\": \"str\"}") == null);
}

test "prompt rendering: system tools block, call and response sections" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const sys = try renderSystemWithTools(arena, "Be terse.", &.{"{\"type\":\"function\",\"function\":{\"name\":\"f\"}}"});
    try std.testing.expect(std.mem.startsWith(u8, sys, "Be terse.\n\n# Tools\n\n"));
    try std.testing.expect(std.mem.indexOf(u8, sys, "<tools>\n{\"type\":\"function\",\"function\":{\"name\":\"f\"}}\n</tools>") != null);
    try std.testing.expect(std.mem.indexOf(u8, sys, "<tool_call></tool_call> XML tags") != null);

    var turn: std.ArrayList(u8) = .empty;
    try turn.appendSlice(arena, "Let me check.");
    try appendCallSection(arena, &turn, "f", "{\"x\":1}");
    try std.testing.expectEqualStrings("Let me check.\n<tool_call>\n{\"name\":\"f\",\"arguments\":{\"x\":1}}\n</tool_call>", turn.items);

    var responses: std.ArrayList(u8) = .empty;
    try appendResponseSection(arena, &responses, "42");
    try appendResponseSection(arena, &responses, "ok");
    try std.testing.expectEqualStrings("<tool_response>\n42\n</tool_response>\n<tool_response>\nok\n</tool_response>", responses.items);
}
