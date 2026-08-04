//! OpenAI-compatible chat backend for the voice agent.
//!
//! The in-process path binds one architecture at compile time (`llm.qwen3`),
//! which is why a Gemma or Qwen3.5 GGUF cannot be handed to `--chat` even
//! though fucina can run both. Rather than grow a second architecture dispatch
//! here, this speaks `/v1/chat/completions` to something that already has one:
//! `fucina-lmserve` picks its backend from the GGUF's `general.architecture`,
//! and any other OpenAI-compatible server (llama.cpp, vLLM, a hosted provider)
//! works the same way.
//!
//! The seam is deliberately the same as the local path: deltas are written to
//! a `std.Io.Writer` as they arrive, so the sentence splitter, the synced
//! reveal and barge-in behave identically whoever generated the tokens.

const std = @import("std");
const Serve = @import("lmserve");

pub const Role = enum { system, user, assistant };

pub const Message = struct {
    role: Role,
    content: []u8, // owned

    fn roleName(r: Role) []const u8 {
        return switch (r) {
            .system => "system",
            .user => "user",
            .assistant => "assistant",
        };
    }
};

/// A conversation held against a remote endpoint. The server is stateless per
/// request, so the whole history rides every call — the same shape parlor uses
/// against llama-server, and what makes a prefix cache on the server side
/// (lmserve's KV slots, llama.cpp's `cache_prompt`) do the work our in-process
/// KV does locally.
pub const Client = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    /// Origin only, no path: "http://127.0.0.1:8080".
    base: []const u8,
    /// Path prefix under the origin, usually "/v1".
    prefix: []const u8,
    model: []const u8,
    api_key: ?[]const u8,
    max_tokens: usize,
    temperature: f32 = 0.7,
    history: std.ArrayList(Message) = .empty,
    http: std.http.Client,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        url: []const u8,
        model: []const u8,
        system: []const u8,
        max_tokens: usize,
        api_key: ?[]const u8,
    ) !Client {
        var self = Client{
            .allocator = allocator,
            .io = io,
            .base = undefined,
            .prefix = undefined,
            .model = model,
            .api_key = api_key,
            .max_tokens = max_tokens,
            .http = .{ .allocator = allocator, .io = io },
        };
        try self.splitUrl(url);
        if (system.len > 0) try self.push(.system, system);
        return self;
    }

    /// Split "http://host:port/v1" into origin and path prefix. A URL given
    /// without a prefix gets "/v1", the near-universal default; one given with
    /// a trailing slash keeps working.
    fn splitUrl(self: *Client, url: []const u8) !void {
        const trimmed = std.mem.trimEnd(u8, url, "/");
        const scheme_end = if (std.mem.indexOf(u8, trimmed, "://")) |i| i + 3 else 0;
        const rest = trimmed[scheme_end..];
        const slash = std.mem.indexOfScalar(u8, rest, '/');
        const origin_len = scheme_end + (slash orelse rest.len);
        self.base = try self.allocator.dupe(u8, trimmed[0..origin_len]);
        self.prefix = try self.allocator.dupe(u8, if (slash != null) trimmed[origin_len..] else "/v1");
    }

    pub fn deinit(self: *Client) void {
        for (self.history.items) |m| self.allocator.free(m.content);
        self.history.deinit(self.allocator);
        self.allocator.free(self.base);
        self.allocator.free(self.prefix);
        self.http.deinit();
    }

    fn push(self: *Client, role: Role, content: []const u8) !void {
        const owned = try self.allocator.dupe(u8, content);
        errdefer self.allocator.free(owned);
        try self.history.append(self.allocator, .{ .role = role, .content = owned });
    }

    /// Drop the oldest exchange, never the system message, so a long session
    /// cannot grow the request past the server's context budget.
    pub fn trim(self: *Client, keep: usize) void {
        const first = @intFromBool(self.history.items.len > 0 and self.history.items[0].role == .system);
        while (self.history.items.len > first + keep) {
            const m = self.history.orderedRemove(first);
            self.allocator.free(m.content);
        }
    }

    fn writeJsonString(w: *std.Io.Writer, s: []const u8) !void {
        try w.writeByte('"');
        for (s) |c| switch (c) {
            '"' => try w.writeAll("\\\""),
            '\\' => try w.writeAll("\\\\"),
            '\n' => try w.writeAll("\\n"),
            '\r' => try w.writeAll("\\r"),
            '\t' => try w.writeAll("\\t"),
            else => if (c < 0x20) {
                try w.print("\\u{x:0>4}", .{c});
            } else try w.writeByte(c),
        };
        try w.writeByte('"');
    }

    fn buildBody(self: *Client, buf: *std.ArrayList(u8), user: []const u8) ![]u8 {
        var w: std.Io.Writer.Allocating = .fromArrayList(self.allocator, buf);
        // Sync back BEFORE returning: a `defer` here would run after
        // `buf.items` was already evaluated and ship an empty body.
        errdefer buf.* = w.toArrayList();
        const out = &w.writer;
        try out.writeAll("{\"model\":");
        try writeJsonString(out, self.model);
        try out.writeAll(",\"stream\":true,\"messages\":[");
        for (self.history.items, 0..) |m, i| {
            if (i > 0) try out.writeByte(',');
            try out.writeAll("{\"role\":\"");
            try out.writeAll(Message.roleName(m.role));
            try out.writeAll("\",\"content\":");
            try writeJsonString(out, m.content);
            try out.writeByte('}');
        }
        if (self.history.items.len > 0) try out.writeByte(',');
        try out.writeAll("{\"role\":\"user\",\"content\":");
        try writeJsonString(out, user);
        try out.writeAll("}],\"max_tokens\":");
        try out.print("{d}", .{self.max_tokens});
        try out.print(",\"temperature\":{d:.2}}}", .{self.temperature});
        try out.flush();
        buf.* = w.toArrayList();
        return buf.items;
    }

    const StreamChunk = struct {
        const Delta = struct { content: ?[]const u8 = null };
        const Choice = struct { delta: Delta = .{} };
        choices: []const Choice = &.{},
    };

    /// Send one user turn and stream the assistant reply into `writer`,
    /// flushing per delta so the caller's sentence splitter sees a sentence the
    /// moment it is complete. Returns the reply text (owned by the history).
    pub fn send(self: *Client, user: []const u8, writer: *std.Io.Writer) ![]const u8 {
        return self.sendCancellable(user, writer, null, true);
    }

    /// Record an exchange that actually happened. Separate from `send` because
    /// a SPECULATIVE generation must not touch the history: it answers the
    /// words heard so far, and if the user keeps talking those words were only
    /// half a question. Recording it anyway leaves the model reading a
    /// conversation where it answered something nobody finished asking — which
    /// is heard as the assistant losing the thread, not as a cache problem.
    pub fn commit(self: *Client, user: []const u8, assistant: []const u8) !void {
        try self.push(.user, user);
        try self.push(.assistant, assistant);
    }

    /// `send` that abandons the generation when `cancel` goes true. Returning
    /// early drops the request, and the server sees the hang-up: lmserve's
    /// connection thread flips the job's `cancelled` flag, so a speculative
    /// turn the user talked past stops costing GPU almost immediately.
    pub fn sendCancellable(
        self: *Client,
        user: []const u8,
        writer: *std.Io.Writer,
        cancel: ?*const std.atomic.Value(bool),
        record: bool,
    ) ![]const u8 {
        var body_buf: std.ArrayList(u8) = .empty;
        defer body_buf.deinit(self.allocator);
        const body = try self.buildBody(&body_buf, user);

        const url = try std.fmt.allocPrint(self.allocator, "{s}{s}/chat/completions", .{ self.base, self.prefix });
        defer self.allocator.free(url);
        const uri = try std.Uri.parse(url);

        var auth_buf: [512]u8 = undefined;
        var headers: [1]std.http.Header = undefined;
        var header_n: usize = 0;
        if (self.api_key) |k| {
            headers[0] = .{
                .name = "Authorization",
                .value = std.fmt.bufPrint(&auth_buf, "Bearer {s}", .{k}) catch return error.ApiKeyTooLong,
            };
            header_n = 1;
        }

        var req = try self.http.request(.POST, uri, .{
            .headers = .{ .content_type = .{ .override = "application/json" } },
            .extra_headers = headers[0..header_n],
        });
        defer req.deinit();
        try req.sendBodyComplete(body);

        var redirect_buf: [2048]u8 = undefined;
        var response = try req.receiveHead(&redirect_buf);
        if (response.head.status != .ok) {
            var err_buf: [4096]u8 = undefined;
            const er = response.reader(&err_buf);
            var seen: [512]u8 = undefined;
            const n = er.readSliceShort(&seen) catch 0;
            std.debug.print("[chat] HTTP {d}: {s}\n  request: {s}\n", .{
                @intFromEnum(response.head.status), seen[0..n], body[0..@min(body.len, 400)],
            });
            return error.HttpStatus;
        }

        var transfer_buf: [16 * 1024]u8 = undefined;
        const body_reader = response.reader(&transfer_buf);

        var reply: std.ArrayList(u8) = .empty;
        errdefer reply.deinit(self.allocator);

        while (true) {
            if (cancel) |c| if (c.load(.acquire)) return error.Cancelled;
            // Inclusive, not exclusive: the exclusive variant tosses only the
            // content and leaves the '\n' in the buffer, so the next call
            // returns an empty line forever.
            const line = body_reader.takeDelimiterInclusive('\n') catch |e| switch (e) {
                error.EndOfStream => break,
                else => return e,
            };
            const trimmed = std.mem.trim(u8, line, " \r\n");
            if (trimmed.len == 0) continue;
            if (!std.mem.startsWith(u8, trimmed, "data:")) continue;
            const payload = std.mem.trim(u8, trimmed["data:".len..], " ");
            if (std.mem.eql(u8, payload, "[DONE]")) break;

            const parsed = std.json.parseFromSlice(StreamChunk, self.allocator, payload, .{
                .ignore_unknown_fields = true,
            }) catch continue; // a malformed chunk costs a token, not the turn
            defer parsed.deinit();
            if (parsed.value.choices.len == 0) continue;
            const delta = parsed.value.choices[0].delta.content orelse continue;
            if (delta.len == 0) continue;
            try reply.appendSlice(self.allocator, delta);
            try writer.writeAll(delta);
            try writer.flush(); // the splitter must see each delta as it lands
        }

        const text = try reply.toOwnedSlice(self.allocator);
        if (!record) {
            // The caller commits this only if the turn is adopted.
            self.allocator.free(text);
            return "";
        }
        errdefer self.allocator.free(text);
        try self.push(.user, user);
        try self.history.append(self.allocator, .{ .role = .assistant, .content = text });
        return text;
    }
};

/// The lmserve chat server, hosted INSIDE this process on a thread.
///
/// Not a child process: one binary, one process, no external executable to
/// find, and nothing to leave orphaned if the agent dies. What it buys over an
/// in-process `llm.chat.Conversation` is the architecture dispatch —
/// `serveBlocking` picks its backend from the GGUF's `general.architecture`,
/// so Gemma, Qwen3.5 and the rest work — and a generation that can be
/// abandoned mid-flight, which is what speculative turns are built on.
///
/// The loopback socket is the seam. It costs microseconds and buys a hard
/// boundary: the server owns its own ExecContext, KV slots and worker team,
/// and the agent talks to it the same way it would talk to llama.cpp or a
/// hosted provider.
pub const Hosted = struct {
    url: []u8,
    allocator: std.mem.Allocator,

    /// Everything the server thread needs, heap-owned: the thread outlives
    /// this call and the process exits out from under it.
    const Job = struct {
        io: std.Io,
        allocator: std.mem.Allocator,
        args: Serve.Args,
    };

    /// The server's own logs would fight the TUI for the terminal.
    const Discard = struct {
        var buf: [256]u8 = undefined;
        fn drain(w: *std.Io.Writer, data: []const []const u8, splat: usize) std.Io.Writer.Error!usize {
            _ = splat;
            w.end = 0;
            var n: usize = 0;
            for (data) |c| n += c.len;
            return n;
        }
    };

    fn threadMain(job: *Job) void {
        var sink: std.Io.Writer = .{ .vtable = &.{ .drain = Discard.drain }, .buffer = &Discard.buf };
        Serve.serveBlocking(job.io, job.allocator, &sink, job.args) catch {};
    }

    pub fn start(
        allocator: std.mem.Allocator,
        io: std.Io,
        model_path: []const u8,
        port: u16,
        ctx_len: usize,
    ) !Hosted {
        const job = try allocator.create(Job);
        errdefer allocator.destroy(job);
        job.* = .{
            .io = io,
            .allocator = allocator,
            .args = .{
                .model_path = model_path,
                .host = "127.0.0.1",
                .port = port,
                .ctx_len = ctx_len,
                // One resident KV slot is what makes re-sending the whole
                // history every turn cheap — the agent's conversation stays
                // warm across requests instead of re-prefilling.
                .kv_slots = 2,
                // Ctrl-C belongs to the agent. lmserve's handler only sets a
                // shutdown flag and closes the listener — installing it here
                // would clobber the TUI's, so Ctrl-C would stop the server and
                // leave the agent running with the terminal in raw mode.
                .own_signals = false,
            },
        };
        var thread = try std.Thread.spawn(.{}, threadMain, .{job});
        thread.detach(); // it runs until the process does

        const url = try std.fmt.allocPrint(allocator, "http://127.0.0.1:{d}/v1", .{port});
        errdefer allocator.free(url);
        var self = Hosted{ .url = url, .allocator = allocator };
        try self.waitReady(io);
        return self;
    }

    /// Poll /health until the model is loaded. A big GGUF takes a while, so the
    /// deadline is generous.
    ///
    /// A NON-200 answer is not "not ready yet" — the server only starts
    /// listening once its model is loaded, so anything replying on this port is
    /// somebody else's.
    fn waitReady(self: *Hosted, io: std.Io) !void {
        var http: std.http.Client = .{ .allocator = self.allocator, .io = io };
        defer http.deinit();
        const health = try std.fmt.allocPrint(self.allocator, "{s}/health", .{
            self.url[0 .. self.url.len - "/v1".len],
        });
        defer self.allocator.free(health);
        const uri = try std.Uri.parse(health);

        var waited_ms: usize = 0;
        while (waited_ms < 600_000) : (waited_ms += 200) {
            if (http.request(.GET, uri, .{})) |*maybe_req| {
                var req = maybe_req.*;
                defer req.deinit();
                if (req.sendBodiless()) |_| {
                    var redirect_buf: [1024]u8 = undefined;
                    if (req.receiveHead(&redirect_buf)) |resp| {
                        if (resp.head.status == .ok) return;
                        return error.PortNotOurs;
                    } else |_| {}
                } else |_| {}
            } else |_| {}
            std.Io.sleep(io, .{ .nanoseconds = 200 * std.time.ns_per_ms }, .awake) catch {};
        }
        return error.ServerNotReady;
    }

    /// First port in `[base, base + span)` that we can actually bind.
    ///
    /// A fixed default is a coin flip — 8080 is squatted on by half the world's
    /// dev tooling, and a stranger answering /health looks exactly like our own
    /// server being slow to load. The range sits in the registered band,
    /// deliberately NOT the ephemeral one (49152+), which the OS hands out to
    /// outbound connections — a listener parked there gets collided with.
    pub fn pickPort(io: std.Io, base: u16, span: u16) !u16 {
        var p: u16 = base;
        while (p < base +| span) : (p += 1) {
            const addr = std.Io.net.IpAddress.parse("127.0.0.1", p) catch continue;
            var probe = addr.listen(io, .{ .reuse_address = true }) catch continue;
            probe.deinit(io);
            return p;
        }
        return error.NoFreePort;
    }

    pub fn deinit(self: *Hosted) void {
        self.allocator.free(self.url);
    }
};
