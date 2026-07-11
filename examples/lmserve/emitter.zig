//! Per-request response emission. The backend streams raw reply bytes into
//! `Emitter.sink()`; the emitter routes reasoning-block text (qwen3
//! `<think>…</think>`) away from the content channel, frames deltas in the
//! requested dialect (Chat Completions chunks / Responses semantic events),
//! and builds the final bodies. Threading contract: the WORKER drives the
//! sink between job states running->finished; the CONNECTION thread calls
//! `finish` afterwards — never concurrently.
//!
//! Streaming responses start lazily on the first delta: a request that fails
//! before producing anything (bad grammar, context overflow) still gets a
//! plain JSON error with a proper status code — the llama.cpp "first error
//! is a normal response" rule.

const std = @import("std");
const types = @import("types.zig");
const openai = @import("openai.zig");
const scheduler = @import("scheduler.zig");

const Allocator = std.mem.Allocator;
const Stringify = std.json.Stringify;

/// Deferred SSE start: implemented by the HTTP layer; writes the
/// `text/event-stream` response head and returns the body writer.
pub const StreamStarter = struct {
    ptr: *anyopaque,
    startFn: *const fn (ptr: *anyopaque) anyerror!*std.Io.Writer,

    fn start(self: StreamStarter) anyerror!*std.Io.Writer {
        return self.startFn(self.ptr);
    }
};

pub const Config = struct {
    dialect: openai.Dialect,
    model_id: []const u8,
    /// Unix seconds, stamped at request arrival.
    created: i64,
    think_markers: ?types.ThinkMarkers = null,
    starter: StreamStarter,
};

pub const Outcome = union(enum) {
    /// Nothing streamed; respond with this error and a plain JSON body.
    plain_error: openai.ErrorInfo,
    /// Nothing streamed; respond 200 with this JSON body (non-stream mode).
    body: []const u8,
    /// SSE frames were written; the caller ends the chunked body.
    streamed: void,
};

pub const Emitter = struct {
    arena: Allocator,
    parsed: *const openai.Parsed,
    cfg: Config,
    job: *scheduler.Job = undefined,

    /// Hex id shared by the request's object ids (resp_/msg_/rs_/chatcmpl-).
    id_hex: [24]u8,
    /// Responses SSE event ordinal.
    seq: u64 = 0,

    /// Full reply accumulation (final bodies + Responses done-events).
    content: std.ArrayList(u8) = .empty,
    reasoning: std.ArrayList(u8) = .empty,

    /// Set once the SSE head is written; frames may only follow.
    out: ?*std.Io.Writer = null,
    sent_role: bool = false,
    reasoning_item_open: bool = false,
    message_item_open: bool = false,

    scanner: ThinkScanner,
    /// Per-feed delta staging (one SSE frame per token flush).
    pending_content: std.ArrayList(u8) = .empty,
    pending_reasoning: std.ArrayList(u8) = .empty,

    /// The real failure behind a sink error.WriteFailed, when it was ours.
    sink_err: ?anyerror = null,

    interface: std.Io.Writer,
    sink_buffer: [512]u8 = undefined,

    /// Monotonic per-process ordinal; combined with the arrival timestamp it
    /// keeps request ids unique without needing a CSPRNG.
    var id_counter = std.atomic.Value(u64).init(0);

    pub fn init(arena: Allocator, parsed: *const openai.Parsed, cfg: Config) Emitter {
        var id_hex: [24]u8 = undefined;
        const ordinal = id_counter.fetchAdd(1, .monotonic);
        _ = std.fmt.bufPrint(&id_hex, "{x:0>8}{x:0>16}", .{
            @as(u32, @truncate(@as(u64, @bitCast(cfg.created)))),
            ordinal,
        }) catch unreachable;
        return .{
            .arena = arena,
            .parsed = parsed,
            .cfg = cfg,
            .id_hex = id_hex,
            .scanner = ThinkScanner.init(cfg.think_markers),
            .interface = .{ .vtable = &.{ .drain = drain }, .buffer = &.{} },
        };
    }

    /// The backend-facing sink. Must be called after the Emitter reached its
    /// final address (self-referential buffer).
    pub fn sink(self: *Emitter) *std.Io.Writer {
        self.interface.buffer = &self.sink_buffer;
        return &self.interface;
    }

    // ---- sink side (worker thread) ----

    fn drain(w: *std.Io.Writer, data: []const []const u8, splat: usize) std.Io.Writer.Error!usize {
        const self: *Emitter = @alignCast(@fieldParentPtr("interface", w));
        self.feedAll(w.buffered()) catch |err| return self.sinkFail(err);
        w.end = 0;
        var consumed: usize = 0;
        for (data[0 .. data.len - 1]) |slice| {
            self.feedAll(slice) catch |err| return self.sinkFail(err);
            consumed += slice.len;
        }
        const last = data[data.len - 1];
        for (0..splat) |_| self.feedAll(last) catch |err| return self.sinkFail(err);
        consumed += last.len * splat;
        self.dispatchPending() catch |err| return self.sinkFail(err);
        return consumed;
    }

    fn sinkFail(self: *Emitter, err: anyerror) std.Io.Writer.Error {
        if (self.sink_err == null) self.sink_err = err;
        return error.WriteFailed;
    }

    fn feedAll(self: *Emitter, bytes: []const u8) !void {
        if (bytes.len == 0) return;
        if (self.job.isCancelled()) return error.Cancelled;
        try self.scanner.feed(self, bytes);
    }

    fn stageContent(self: *Emitter, bytes: []const u8) !void {
        try self.pending_content.appendSlice(self.arena, bytes);
    }

    fn stageReasoning(self: *Emitter, bytes: []const u8) !void {
        try self.pending_reasoning.appendSlice(self.arena, bytes);
    }

    /// Turn staged bytes into accumulated text + (when streaming) one frame
    /// per channel. Only whole UTF-8 sequences are dispatched — a token can
    /// end mid-code-point (byte-level BPE, SPM byte fallback), and a split
    /// code point would corrupt the JSON delta; the tail carries over to the
    /// next dispatch.
    fn dispatchPending(self: *Emitter) !void {
        try self.dispatchChannel(.reasoning, &self.pending_reasoning);
        try self.dispatchChannel(.content, &self.pending_content);
    }

    fn dispatchChannel(self: *Emitter, channel: Channel, pending: *std.ArrayList(u8)) !void {
        const complete = utf8CompletePrefix(pending.items);
        if (complete == 0) return;
        const delta = pending.items[0..complete];
        switch (channel) {
            .reasoning => try self.reasoning.appendSlice(self.arena, delta),
            .content => try self.content.appendSlice(self.arena, delta),
        }
        if (self.parsed.stream) try self.emitDelta(channel, delta);
        const tail = pending.items.len - complete;
        std.mem.copyForwards(u8, pending.items[0..tail], pending.items[complete..]);
        pending.shrinkRetainingCapacity(tail);
    }

    const Channel = enum { content, reasoning };

    fn emitDelta(self: *Emitter, channel: Channel, delta: []const u8) !void {
        try self.ensureStarted();
        switch (self.cfg.dialect) {
            .chat => try self.chatChunk(channel, delta),
            .responses => switch (channel) {
                .reasoning => {
                    try self.ensureReasoningItem();
                    try self.beginEvent("response.reasoning_text.delta");
                    var s = self.eventJson();
                    try s.beginObject();
                    try self.eventCommon(&s, "response.reasoning_text.delta");
                    try s.objectField("item_id");
                    try s.print("\"rs_{s}\"", .{&self.id_hex});
                    try s.objectField("output_index");
                    try s.write(self.reasoningIndex());
                    try s.objectField("content_index");
                    try s.write(0);
                    try s.objectField("delta");
                    try s.write(delta);
                    try s.endObject();
                    try self.endEvent();
                },
                .content => {
                    try self.ensureMessageItem();
                    try self.beginEvent("response.output_text.delta");
                    var s = self.eventJson();
                    try s.beginObject();
                    try self.eventCommon(&s, "response.output_text.delta");
                    try s.objectField("item_id");
                    try s.print("\"msg_{s}\"", .{&self.id_hex});
                    try s.objectField("output_index");
                    try s.write(self.messageIndex());
                    try s.objectField("content_index");
                    try s.write(0);
                    try s.objectField("delta");
                    try s.write(delta);
                    try s.objectField("logprobs");
                    try s.beginArray();
                    try s.endArray();
                    try s.endObject();
                    try self.endEvent();
                },
            },
        }
        try self.out.?.flush();
    }

    // ---- finish side (connection thread) ----

    /// Complete the response after the job finished. Never throws client
    /// errors: a dead client surfaces as an ordinary write failure the
    /// caller treats like any closed connection.
    pub fn finish(self: *Emitter, job_err: ?anyerror) !Outcome {
        // Anything the scanner still holds is reply text; a trailing
        // incomplete UTF-8 sequence is truncated (llama.cpp does the same).
        self.scanner.finishFlush(self) catch {};
        self.dispatchPending() catch {};
        self.pending_content.clearRetainingCapacity();
        self.pending_reasoning.clearRetainingCapacity();

        if (job_err) |err| {
            const real = if (err == error.WriteFailed) (self.sink_err orelse err) else err;
            if (real == error.Cancelled or real == error.WriteFailed) {
                // Client gone: nothing to say, nowhere to say it.
                return .streamed;
            }
            const info = openai.mapError(real);
            if (self.out == null) return .{ .plain_error = info };
            try self.streamError(info);
            return .streamed;
        }

        const res = self.job.res;
        if (!self.parsed.stream) {
            return .{ .body = switch (self.cfg.dialect) {
                .chat => try self.buildChatBody(res),
                .responses => try self.buildResponsesBody(res),
            } };
        }

        try self.ensureStarted();
        switch (self.cfg.dialect) {
            .chat => {
                // Final chunk carries the finish_reason on an empty delta.
                try self.beginData();
                var s = self.eventJson();
                try self.chatChunkOpen(&s);
                try s.beginObject();
                try s.objectField("index");
                try s.write(0);
                try s.objectField("delta");
                try s.beginObject();
                try s.endObject();
                try s.objectField("logprobs");
                try s.write(null);
                try s.objectField("finish_reason");
                try s.write(finishReasonName(res.finish));
                try s.endObject();
                try self.chatChunkClose(&s, false);
                try self.endEvent();
                if (self.parsed.include_usage) {
                    try self.beginData();
                    var su = self.eventJson();
                    try self.chatChunkOpen(&su);
                    try self.chatChunkClose(&su, true);
                    try self.endEvent();
                }
                try self.out.?.writeAll("data: [DONE]\n\n");
                try self.out.?.flush();
            },
            .responses => {
                try self.closeReasoningItem();
                // An empty reply still gets the full item skeleton — the
                // SDKs' stream state machines require added-before-done.
                try self.ensureMessageItem();

                try self.beginEvent("response.output_text.done");
                var s0 = self.eventJson();
                try s0.beginObject();
                try self.eventCommon(&s0, "response.output_text.done");
                try s0.objectField("item_id");
                try s0.print("\"msg_{s}\"", .{&self.id_hex});
                try s0.objectField("output_index");
                try s0.write(self.messageIndex());
                try s0.objectField("content_index");
                try s0.write(0);
                try s0.objectField("text");
                try s0.write(self.content.items);
                try s0.objectField("logprobs");
                try s0.beginArray();
                try s0.endArray();
                try s0.endObject();
                try self.endEvent();

                try self.beginEvent("response.content_part.done");
                var s1 = self.eventJson();
                try s1.beginObject();
                try self.eventCommon(&s1, "response.content_part.done");
                try s1.objectField("item_id");
                try s1.print("\"msg_{s}\"", .{&self.id_hex});
                try s1.objectField("output_index");
                try s1.write(self.messageIndex());
                try s1.objectField("content_index");
                try s1.write(0);
                try s1.objectField("part");
                try self.writeTextPart(&s1);
                try s1.endObject();
                try self.endEvent();

                try self.beginEvent("response.output_item.done");
                var s2 = self.eventJson();
                try s2.beginObject();
                try self.eventCommon(&s2, "response.output_item.done");
                try s2.objectField("output_index");
                try s2.write(self.messageIndex());
                try s2.objectField("item");
                try self.writeMessageItem(&s2, "completed");
                try s2.endObject();
                try self.endEvent();

                const terminal: []const u8 = if (res.finish == .length) "response.incomplete" else "response.completed";
                const status: []const u8 = if (res.finish == .length) "incomplete" else "completed";
                try self.beginEvent(terminal);
                var s3 = self.eventJson();
                try s3.beginObject();
                try self.eventCommon(&s3, terminal);
                try s3.objectField("response");
                try self.writeResponseObject(&s3, status, res, true, null);
                try s3.endObject();
                try self.endEvent();
                try self.out.?.flush();
            },
        }
        return .streamed;
    }

    /// Mid-stream failure: the dialect's in-band error, then terminate.
    fn streamError(self: *Emitter, info: openai.ErrorInfo) !void {
        switch (self.cfg.dialect) {
            .chat => {
                try self.beginData();
                var s = self.eventJson();
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
                try self.endEvent();
                try self.out.?.writeAll("data: [DONE]\n\n");
            },
            .responses => {
                try self.beginEvent("error");
                var s = self.eventJson();
                try s.beginObject();
                try self.eventCommon(&s, "error");
                try s.objectField("code");
                try s.write(info.code orelse "server_error");
                try s.objectField("message");
                try s.write(info.message);
                try s.objectField("param");
                try s.write(info.param);
                try s.endObject();
                try self.endEvent();

                try self.beginEvent("response.failed");
                var sf = self.eventJson();
                try sf.beginObject();
                try self.eventCommon(&sf, "response.failed");
                try sf.objectField("response");
                try self.writeResponseObject(&sf, "failed", null, false, info.message);
                try sf.endObject();
                try self.endEvent();
            },
        }
        try self.out.?.flush();
    }

    // ---- shared plumbing ----

    fn ensureStarted(self: *Emitter) !void {
        if (self.out != null) return;
        self.out = try self.cfg.starter.start();
        if (self.cfg.dialect == .responses) {
            inline for (.{ "response.created", "response.in_progress" }) |event| {
                try self.beginEvent(event);
                var s = self.eventJson();
                try s.beginObject();
                try self.eventCommon(&s, event);
                try s.objectField("response");
                try self.writeResponseObject(&s, "in_progress", null, false, null);
                try s.endObject();
                try self.endEvent();
            }
        }
    }

    fn eventJson(self: *Emitter) Stringify {
        return .{ .writer = self.out.? };
    }

    fn eventCommon(self: *Emitter, s: *Stringify, event_type: []const u8) !void {
        try s.objectField("type");
        try s.write(event_type);
        try s.objectField("sequence_number");
        try s.write(self.seq);
        self.seq += 1;
    }

    fn beginEvent(self: *Emitter, name: []const u8) !void {
        const w = self.out.?;
        try w.writeAll("event: ");
        try w.writeAll(name);
        try w.writeAll("\ndata: ");
    }

    /// Chat frames carry no `event:` line.
    fn beginData(self: *Emitter) !void {
        try self.out.?.writeAll("data: ");
    }

    fn endEvent(self: *Emitter) !void {
        try self.out.?.writeAll("\n\n");
    }

    fn reasoningIndex(self: *const Emitter) u32 {
        _ = self;
        return 0;
    }

    fn messageIndex(self: *const Emitter) u32 {
        return if (self.reasoning_item_open or self.reasoning.items.len > 0) 1 else 0;
    }

    // ---- chat framing ----

    fn chatChunkOpen(self: *Emitter, s: *Stringify) !void {
        try s.beginObject();
        try s.objectField("id");
        try s.print("\"chatcmpl-{s}\"", .{&self.id_hex});
        try s.objectField("object");
        try s.write("chat.completion.chunk");
        try s.objectField("created");
        try s.write(self.cfg.created);
        try s.objectField("model");
        try s.write(self.cfg.model_id);
        try s.objectField("choices");
        try s.beginArray();
    }

    /// Close the choices array and the chunk. `usage_chunk` emits the
    /// trailing usage-only chunk shape (empty choices + populated usage).
    fn chatChunkClose(self: *Emitter, s: *Stringify, usage_chunk: bool) !void {
        try s.endArray();
        if (self.parsed.include_usage) {
            try s.objectField("usage");
            if (usage_chunk) {
                try self.writeChatUsage(s, self.job.res);
            } else {
                try s.write(null);
            }
        }
        try s.endObject();
    }

    fn chatChunk(self: *Emitter, channel: Channel, delta: []const u8) !void {
        try self.beginData();
        var s = self.eventJson();
        try self.chatChunkOpen(&s);
        try s.beginObject();
        try s.objectField("index");
        try s.write(0);
        try s.objectField("delta");
        try s.beginObject();
        if (!self.sent_role) {
            try s.objectField("role");
            try s.write("assistant");
            self.sent_role = true;
        }
        try s.objectField(switch (channel) {
            .content => "content",
            .reasoning => "reasoning_content",
        });
        try s.write(delta);
        try s.endObject();
        try s.objectField("logprobs");
        try s.write(null);
        try s.objectField("finish_reason");
        try s.write(null);
        try s.endObject();
        try self.chatChunkClose(&s, false);
        try self.endEvent();
    }

    fn writeChatUsage(self: *Emitter, s: *Stringify, res: types.GenerateResult) !void {
        _ = self;
        try s.beginObject();
        try s.objectField("prompt_tokens");
        try s.write(res.prompt_tokens);
        try s.objectField("completion_tokens");
        try s.write(res.completion_tokens);
        try s.objectField("total_tokens");
        try s.write(res.prompt_tokens + res.completion_tokens);
        try s.endObject();
    }

    fn buildChatBody(self: *Emitter, res: types.GenerateResult) ![]const u8 {
        var aw = std.Io.Writer.Allocating.init(self.arena);
        var s: Stringify = .{ .writer = &aw.writer };
        try s.beginObject();
        try s.objectField("id");
        try s.print("\"chatcmpl-{s}\"", .{&self.id_hex});
        try s.objectField("object");
        try s.write("chat.completion");
        try s.objectField("created");
        try s.write(self.cfg.created);
        try s.objectField("model");
        try s.write(self.cfg.model_id);
        try s.objectField("choices");
        try s.beginArray();
        try s.beginObject();
        try s.objectField("index");
        try s.write(0);
        try s.objectField("message");
        try s.beginObject();
        try s.objectField("role");
        try s.write("assistant");
        try s.objectField("content");
        try s.write(self.content.items);
        if (self.reasoning.items.len > 0) {
            try s.objectField("reasoning_content");
            try s.write(self.reasoning.items);
        }
        try s.objectField("refusal");
        try s.write(null);
        try s.endObject();
        try s.objectField("logprobs");
        try s.write(null);
        try s.objectField("finish_reason");
        try s.write(finishReasonName(res.finish));
        try s.endObject();
        try s.endArray();
        try s.objectField("usage");
        try self.writeChatUsage(&s, res);
        try s.endObject();
        return aw.written();
    }

    // ---- responses framing ----

    fn ensureReasoningItem(self: *Emitter) !void {
        if (self.reasoning_item_open) return;
        self.reasoning_item_open = true;
        try self.beginEvent("response.output_item.added");
        var s = self.eventJson();
        try s.beginObject();
        try self.eventCommon(&s, "response.output_item.added");
        try s.objectField("output_index");
        try s.write(self.reasoningIndex());
        try s.objectField("item");
        try s.beginObject();
        try s.objectField("id");
        try s.print("\"rs_{s}\"", .{&self.id_hex});
        try s.objectField("type");
        try s.write("reasoning");
        try s.objectField("summary");
        try s.beginArray();
        try s.endArray();
        try s.objectField("content");
        try s.beginArray();
        try s.endArray();
        try s.objectField("status");
        try s.write("in_progress");
        try s.endObject();
        try s.endObject();
        try self.endEvent();
    }

    fn closeReasoningItem(self: *Emitter) !void {
        if (!self.reasoning_item_open) return;
        self.reasoning_item_open = false;

        try self.beginEvent("response.reasoning_text.done");
        var s0 = self.eventJson();
        try s0.beginObject();
        try self.eventCommon(&s0, "response.reasoning_text.done");
        try s0.objectField("item_id");
        try s0.print("\"rs_{s}\"", .{&self.id_hex});
        try s0.objectField("output_index");
        try s0.write(self.reasoningIndex());
        try s0.objectField("content_index");
        try s0.write(0);
        try s0.objectField("text");
        try s0.write(self.reasoning.items);
        try s0.endObject();
        try self.endEvent();

        try self.beginEvent("response.output_item.done");
        var s1 = self.eventJson();
        try s1.beginObject();
        try self.eventCommon(&s1, "response.output_item.done");
        try s1.objectField("output_index");
        try s1.write(self.reasoningIndex());
        try s1.objectField("item");
        try self.writeReasoningItem(&s1, "completed");
        try s1.endObject();
        try self.endEvent();
    }

    fn ensureMessageItem(self: *Emitter) !void {
        if (self.message_item_open) return;
        try self.closeReasoningItem();
        self.message_item_open = true;

        try self.beginEvent("response.output_item.added");
        var s = self.eventJson();
        try s.beginObject();
        try self.eventCommon(&s, "response.output_item.added");
        try s.objectField("output_index");
        try s.write(self.messageIndex());
        try s.objectField("item");
        try s.beginObject();
        try s.objectField("id");
        try s.print("\"msg_{s}\"", .{&self.id_hex});
        try s.objectField("type");
        try s.write("message");
        try s.objectField("status");
        try s.write("in_progress");
        try s.objectField("role");
        try s.write("assistant");
        try s.objectField("content");
        try s.beginArray();
        try s.endArray();
        try s.endObject();
        try s.endObject();
        try self.endEvent();

        try self.beginEvent("response.content_part.added");
        var sp = self.eventJson();
        try sp.beginObject();
        try self.eventCommon(&sp, "response.content_part.added");
        try sp.objectField("item_id");
        try sp.print("\"msg_{s}\"", .{&self.id_hex});
        try sp.objectField("output_index");
        try sp.write(self.messageIndex());
        try sp.objectField("content_index");
        try sp.write(0);
        try sp.objectField("part");
        try sp.beginObject();
        try sp.objectField("type");
        try sp.write("output_text");
        try sp.objectField("text");
        try sp.write("");
        try sp.objectField("annotations");
        try sp.beginArray();
        try sp.endArray();
        try sp.endObject();
        try sp.endObject();
        try self.endEvent();
    }

    fn writeTextPart(self: *Emitter, s: *Stringify) !void {
        try s.beginObject();
        try s.objectField("type");
        try s.write("output_text");
        try s.objectField("text");
        try s.write(self.content.items);
        try s.objectField("annotations");
        try s.beginArray();
        try s.endArray();
        try s.objectField("logprobs");
        try s.beginArray();
        try s.endArray();
        try s.endObject();
    }

    fn writeMessageItem(self: *Emitter, s: *Stringify, status: []const u8) !void {
        try s.beginObject();
        try s.objectField("id");
        try s.print("\"msg_{s}\"", .{&self.id_hex});
        try s.objectField("type");
        try s.write("message");
        try s.objectField("status");
        try s.write(status);
        try s.objectField("role");
        try s.write("assistant");
        try s.objectField("content");
        try s.beginArray();
        try self.writeTextPart(s);
        try s.endArray();
        try s.endObject();
    }

    fn writeReasoningItem(self: *Emitter, s: *Stringify, status: []const u8) !void {
        try s.beginObject();
        try s.objectField("id");
        try s.print("\"rs_{s}\"", .{&self.id_hex});
        try s.objectField("type");
        try s.write("reasoning");
        try s.objectField("summary");
        try s.beginArray();
        try s.endArray();
        try s.objectField("content");
        try s.beginArray();
        try s.beginObject();
        try s.objectField("type");
        try s.write("reasoning_text");
        try s.objectField("text");
        try s.write(self.reasoning.items);
        try s.endObject();
        try s.endArray();
        try s.objectField("status");
        try s.write(status);
        try s.endObject();
    }

    /// The Response object: snapshots inside stream events and the
    /// non-stream body.
    fn writeResponseObject(
        self: *Emitter,
        s: *Stringify,
        status: []const u8,
        res: ?types.GenerateResult,
        include_output: bool,
        error_message: ?[]const u8,
    ) Stringify.Error!void {
        const p = self.parsed;
        try s.beginObject();
        try s.objectField("id");
        try s.print("\"resp_{s}\"", .{&self.id_hex});
        try s.objectField("object");
        try s.write("response");
        try s.objectField("created_at");
        try s.write(self.cfg.created);
        try s.objectField("status");
        try s.write(status);
        try s.objectField("background");
        try s.write(false);
        try s.objectField("error");
        if (error_message) |msg| {
            try s.beginObject();
            try s.objectField("code");
            try s.write("server_error");
            try s.objectField("message");
            try s.write(msg);
            try s.endObject();
        } else {
            try s.write(null);
        }
        try s.objectField("incomplete_details");
        if (res != null and res.?.finish == .length) {
            try s.beginObject();
            try s.objectField("reason");
            try s.write("max_output_tokens");
            try s.endObject();
        } else {
            try s.write(null);
        }
        try s.objectField("instructions");
        try s.write(null);
        try s.objectField("max_output_tokens");
        try s.write(p.max_tokens_requested);
        try s.objectField("max_tool_calls");
        try s.write(null);
        try s.objectField("model");
        try s.write(self.cfg.model_id);
        try s.objectField("output");
        try s.beginArray();
        if (include_output) {
            if (self.reasoning.items.len > 0) try self.writeReasoningItem(s, "completed");
            try self.writeMessageItem(s, "completed");
        }
        try s.endArray();
        try s.objectField("parallel_tool_calls");
        try s.write(true);
        try s.objectField("previous_response_id");
        try s.write(null);
        try s.objectField("reasoning");
        try s.beginObject();
        try s.objectField("effort");
        try s.write(if (p.gen.think) "medium" else null);
        try s.objectField("summary");
        try s.write(null);
        try s.endObject();
        try s.objectField("store");
        try s.write(false);
        try s.objectField("temperature");
        try s.write(p.gen.sampling.temperature);
        try s.objectField("text");
        try s.beginObject();
        try s.objectField("format");
        try s.beginObject();
        try s.objectField("type");
        try s.write(switch (p.format_kind) {
            .text => "text",
            .json_object => "json_object",
            .json_schema => "json_schema",
        });
        if (p.format_kind == .json_schema) {
            try s.objectField("name");
            try s.write(p.format_name orelse "response");
        }
        try s.endObject();
        try s.endObject();
        try s.objectField("tool_choice");
        try s.write("auto");
        try s.objectField("tools");
        try s.beginArray();
        try s.endArray();
        try s.objectField("top_p");
        try s.write(p.gen.sampling.top_p);
        try s.objectField("truncation");
        try s.write("disabled");
        try s.objectField("usage");
        if (res) |r| {
            try s.beginObject();
            try s.objectField("input_tokens");
            try s.write(r.prompt_tokens);
            try s.objectField("input_tokens_details");
            try s.beginObject();
            try s.objectField("cached_tokens");
            try s.write(0);
            try s.endObject();
            try s.objectField("output_tokens");
            try s.write(r.completion_tokens);
            try s.objectField("output_tokens_details");
            try s.beginObject();
            try s.objectField("reasoning_tokens");
            try s.write(0);
            try s.endObject();
            try s.objectField("total_tokens");
            try s.write(r.prompt_tokens + r.completion_tokens);
            try s.endObject();
        } else {
            try s.write(null);
        }
        try s.objectField("user");
        try s.write(null);
        try s.objectField("metadata");
        try s.beginObject();
        try s.endObject();
        try s.endObject();
    }

    fn buildResponsesBody(self: *Emitter, res: types.GenerateResult) ![]const u8 {
        var aw = std.Io.Writer.Allocating.init(self.arena);
        var s: Stringify = .{ .writer = &aw.writer };
        const status: []const u8 = if (res.finish == .length) "incomplete" else "completed";
        try self.writeResponseObject(&s, status, res, true, null);
        return aw.written();
    }
};

fn finishReasonName(finish: types.FinishReason) []const u8 {
    return switch (finish) {
        .stop => "stop",
        .length => "length",
    };
}

/// Length of the longest prefix that ends on a UTF-8 sequence boundary. A
/// trailing INCOMPLETE multi-byte sequence is excluded; invalid bytes pass
/// through (best effort — only split-across-chunks sequences are repaired).
fn utf8CompletePrefix(bytes: []const u8) usize {
    if (bytes.len == 0) return 0;
    // Find the start byte of the final sequence (at most 3 continuation
    // bytes precede it).
    var i: usize = bytes.len;
    var back: usize = 0;
    while (i > 0 and back < 4) : (back += 1) {
        const b = bytes[i - 1];
        if (b & 0xC0 != 0x80) break; // not a continuation byte
        i -= 1;
    }
    if (i == 0) return bytes.len; // continuation bytes only: pass through
    const start = bytes[i - 1];
    const seq_len = std.unicode.utf8ByteSequenceLength(start) catch return bytes.len;
    const have = bytes.len - (i - 1);
    return if (have < seq_len) i - 1 else bytes.len;
}

test utf8CompletePrefix {
    try std.testing.expectEqual(@as(usize, 5), utf8CompletePrefix("hello"));
    const emoji = "a😊"; // 1 + 4 bytes
    try std.testing.expectEqual(@as(usize, 5), utf8CompletePrefix(emoji));
    try std.testing.expectEqual(@as(usize, 1), utf8CompletePrefix(emoji[0..3])); // split emoji
    try std.testing.expectEqual(@as(usize, 1), utf8CompletePrefix(emoji[0..4]));
    // An incomplete sequence spanning the whole chunk is held entirely.
    try std.testing.expectEqual(@as(usize, 0), utf8CompletePrefix(emoji[1..3]));
    // Headless continuation bytes are invalid, not incomplete: pass through.
    try std.testing.expectEqual(@as(usize, 2), utf8CompletePrefix(emoji[2..4]));
}

/// Byte-wise scanner over the reply head, active whenever the family has
/// reasoning markers (qwen3 `<think>…</think>`):
///
/// * a LEADING `open…close` block routes to the reasoning channel (never
///   into OpenAI `content`, whichever way the request set reasoning);
/// * a LEADING stray `close` — what qwen3 emits under the primed-empty
///   think block of think-off prompts — is dropped as a template artifact;
/// * everything else passes through as content.
///
/// Partial matches across chunk boundaries sit in a small hold buffer.
const ThinkScanner = struct {
    markers: ?types.ThinkMarkers,
    state: enum { detect, think, content_skip_ws, content } = .detect,
    /// Partial-match progress: `.detect` holds candidate marker bytes;
    /// `.think` counts matched bytes of `close`.
    hold: [max_marker_len]u8 = undefined,
    held: usize = 0,

    const max_marker_len = 32;

    fn init(markers: ?types.ThinkMarkers) ThinkScanner {
        if (markers) |m| std.debug.assert(m.open.len <= max_marker_len and m.close.len <= max_marker_len);
        return .{
            .markers = markers,
            .state = if (markers == null) .content else .detect,
        };
    }

    fn feed(self: *ThinkScanner, em: *Emitter, bytes: []const u8) !void {
        if (self.state == .content) return em.stageContent(bytes);
        const m = self.markers.?;
        var rest = bytes;
        while (rest.len > 0) {
            switch (self.state) {
                .detect => {
                    const c = rest[0];
                    if (self.held == 0 and (c == ' ' or c == '\t' or c == '\r' or c == '\n')) {
                        rest = rest[1..];
                        continue;
                    }
                    self.hold[self.held] = c;
                    self.held += 1;
                    rest = rest[1..];
                    const held = self.hold[0..self.held];
                    if (std.mem.eql(u8, held, m.open)) {
                        self.state = .think;
                        self.held = 0;
                    } else if (std.mem.eql(u8, held, m.close)) {
                        // Stray close marker with no open: a think-off
                        // template artifact, not reply content.
                        self.state = .content_skip_ws;
                        self.held = 0;
                    } else if (!std.mem.startsWith(u8, m.open, held) and
                        !std.mem.startsWith(u8, m.close, held))
                    {
                        try em.stageContent(held);
                        self.held = 0;
                        self.state = .content;
                        return em.stageContent(rest);
                    }
                },
                .think => {
                    const c = rest[0];
                    if (c == m.close[self.held]) {
                        self.held += 1;
                        rest = rest[1..];
                        if (self.held == m.close.len) {
                            self.state = .content_skip_ws;
                            self.held = 0;
                        }
                    } else if (self.held > 0) {
                        // False partial close: it was reasoning text.
                        try em.stageReasoning(m.close[0..self.held]);
                        self.held = 0;
                    } else {
                        try em.stageReasoning(rest[0..1]);
                        rest = rest[1..];
                    }
                },
                .content_skip_ws => {
                    const c = rest[0];
                    if (c == ' ' or c == '\t' or c == '\r' or c == '\n') {
                        rest = rest[1..];
                    } else {
                        self.state = .content;
                        return em.stageContent(rest);
                    }
                },
                .content => return em.stageContent(rest),
            }
        }
    }

    /// End of reply: whatever a partial match held belongs to its channel.
    fn finishFlush(self: *ThinkScanner, em: *Emitter) !void {
        const m = self.markers orelse return;
        switch (self.state) {
            .detect => if (self.held > 0) try em.stageContent(self.hold[0..self.held]),
            .think => if (self.held > 0) try em.stageReasoning(m.close[0..self.held]),
            else => {},
        }
        self.held = 0;
    }
};

/// Feed `chunks` through a scanner into a throwaway non-streaming Emitter;
/// returns what landed on each channel.
fn scanChunks(arena: Allocator, chunks: []const []const u8) !struct { content: []const u8, reasoning: []const u8 } {
    const Fail = struct {
        fn start(_: *anyopaque) anyerror!*std.Io.Writer {
            return error.WriteFailed;
        }
    };
    const parsed = try arena.create(openai.Parsed);
    parsed.* = .{ .gen = .{ .messages = &.{}, .sampling = .{}, .max_tokens = 8 } };
    var dummy: u8 = 0;
    const em = try arena.create(Emitter);
    em.* = Emitter.init(arena, parsed, .{
        .dialect = .chat,
        .model_id = "t",
        .created = 0,
        .think_markers = .{ .open = "<think>", .close = "</think>" },
        .starter = .{ .ptr = &dummy, .startFn = Fail.start },
    });
    for (chunks) |c| try em.scanner.feed(em, c);
    try em.scanner.finishFlush(em);
    try em.dispatchPending();
    return .{ .content = em.content.items, .reasoning = em.reasoning.items };
}

test "think scanner: routes blocks, drops stray close, holds partial markers" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    // Plain content passes through untouched.
    {
        const r = try scanChunks(arena, &.{ "hello ", "world" });
        try std.testing.expectEqualStrings("hello world", r.content);
        try std.testing.expectEqualStrings("", r.reasoning);
    }
    // Full think block, split across awkward chunk boundaries.
    {
        const r = try scanChunks(arena, &.{ "<thi", "nk>step one</th", "ink>\n\nanswer" });
        try std.testing.expectEqualStrings("answer", r.content);
        try std.testing.expectEqualStrings("step one", r.reasoning);
    }
    // The qwen3 think-off artifact: stray close marker is dropped.
    {
        const r = try scanChunks(arena, &.{"</think>\n\nhello world"});
        try std.testing.expectEqualStrings("hello world", r.content);
        try std.testing.expectEqualStrings("", r.reasoning);
    }
    // Content that merely STARTS like a marker is flushed as content,
    // including a trailing partial match at end-of-reply.
    {
        const r = try scanChunks(arena, &.{"<div>x</div"});
        try std.testing.expectEqualStrings("<div>x</div", r.content);
    }
    // False partial close inside the think block stays reasoning.
    {
        const r = try scanChunks(arena, &.{"<think>a </thing b</think>c"});
        try std.testing.expectEqualStrings("c", r.content);
        try std.testing.expectEqualStrings("a </thing b", r.reasoning);
    }
}
