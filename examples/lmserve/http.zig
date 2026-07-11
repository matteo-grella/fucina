//! The HTTP front end: accept loop, per-connection threads (capped),
//! socket deadlines, routing, SSE plumbing, and graceful shutdown. Built on
//! Zig 0.16 `std.http.Server` (one blocking server per connection) — the
//! pieces std deliberately leaves out (timeouts, connection cap, lifecycle)
//! are handled here.

const std = @import("std");
const posix = std.posix;
const types = @import("types.zig");
const openai = @import("openai.zig");
const emitter_mod = @import("emitter.zig");
const scheduler_mod = @import("scheduler.zig");

const Allocator = std.mem.Allocator;

pub const Options = struct {
    host: []const u8 = "127.0.0.1",
    port: u16 = 8080,
    /// When set, POST endpoints and /v1/models require
    /// `Authorization: Bearer <key>`.
    api_key: ?[]const u8 = null,
    max_body_bytes: usize = 8 * 1024 * 1024,
    max_connections: usize = 32,
    /// Socket receive deadline: bounds slow request reads and reaps idle
    /// keep-alive connections.
    read_timeout_s: u31 = 60,
    /// Socket send deadline: bounds stalled clients during streaming.
    write_timeout_s: u31 = 30,
};

pub const Server = struct {
    allocator: Allocator,
    io: std.Io,
    opts: Options,
    backend: types.Backend,
    sched: *scheduler_mod.Scheduler,
    shutdown: *std.atomic.Value(bool),

    listener: std.Io.net.Server = undefined,
    active: std.atomic.Value(usize) = .{ .raw = 0 },

    /// Bind the listener. Separate from `run` so the caller can publish the
    /// socket handle to its signal handler before serving.
    pub fn bind(self: *Server) !void {
        const addr = std.Io.net.IpAddress.parse(self.opts.host, self.opts.port) catch
            return error.InvalidBindAddress;
        self.listener = try addr.listen(self.io, .{ .reuse_address = true });
    }

    /// Serve until `shutdown` flips (the signal handler also shuts the
    /// listener socket down to unblock `accept`). Requires `bind`.
    pub fn run(self: *Server) !void {
        defer self.listener.deinit(self.io);

        std.log.info("listening on http://{s}:{d} (model {s})", .{
            self.opts.host, self.opts.port, self.backend.info.model_id,
        });

        while (!self.shutdown.load(.acquire)) {
            const stream = self.listener.accept(self.io) catch |err| switch (err) {
                error.SocketNotListening, error.Canceled => break,
                error.ConnectionAborted, error.ProtocolFailure, error.BlockedByFirewall => continue,
                else => {
                    if (self.shutdown.load(.acquire)) break;
                    std.log.warn("accept failed: {t}", .{err});
                    continue;
                },
            };

            if (self.active.load(.acquire) >= self.opts.max_connections) {
                self.rejectOverloaded(stream);
                continue;
            }

            _ = self.active.fetchAdd(1, .acq_rel);
            const thread = std.Thread.spawn(.{}, connectionThread, .{ self, stream }) catch {
                _ = self.active.fetchSub(1, .acq_rel);
                stream.close(self.io);
                continue;
            };
            thread.detach();
        }

        // Drain: connection threads observe shutdown (their jobs get
        // cancelled) within one wait tick; give them a bounded window.
        var waited_ns: u64 = 0;
        while (self.active.load(.acquire) > 0 and waited_ns < 10 * std.time.ns_per_s) {
            std.Io.sleep(self.io, .{ .nanoseconds = 50 * std.time.ns_per_ms }, .awake) catch break;
            waited_ns += 50 * std.time.ns_per_ms;
        }
    }

    /// The listener's socket handle, for the signal handler's shutdown kick.
    pub fn listenerHandle(self: *Server) posix.socket_t {
        return self.listener.socket.handle;
    }

    fn rejectOverloaded(self: *Server, stream: std.Io.net.Stream) void {
        defer stream.close(self.io);
        var out_buf: [512]u8 = undefined;
        var out = stream.writer(self.io, &out_buf);
        const body = "{\"error\":{\"message\":\"too many connections\",\"type\":\"unavailable_error\",\"param\":null,\"code\":null}}";
        out.interface.print("HTTP/1.1 503 Service Unavailable\r\ncontent-type: application/json\r\nretry-after: 2\r\ncontent-length: {d}\r\nconnection: close\r\n\r\n{s}", .{ body.len, body }) catch return;
        out.interface.flush() catch return;
    }

    fn connectionThread(self: *Server, stream: std.Io.net.Stream) void {
        defer _ = self.active.fetchSub(1, .acq_rel);
        defer stream.close(self.io);
        self.setSocketDeadlines(stream.socket.handle);

        // The head must fit the read buffer (error.HttpHeadersOversize otherwise).
        var read_buf: [16 * 1024]u8 = undefined;
        var write_buf: [8 * 1024]u8 = undefined;
        var reader = stream.reader(self.io, &read_buf);
        var writer = stream.writer(self.io, &write_buf);
        var http = std.http.Server.init(&reader.interface, &writer.interface);

        while (!self.shutdown.load(.acquire)) {
            var request = http.receiveHead() catch return;
            self.handleRequest(&request, stream) catch return;
            if (http.reader.state != .ready) return; // connection not reusable
        }
    }

    fn setSocketDeadlines(self: *Server, fd: posix.socket_t) void {
        const rcv = posix.timeval{ .sec = self.opts.read_timeout_s, .usec = 0 };
        const snd = posix.timeval{ .sec = self.opts.write_timeout_s, .usec = 0 };
        posix.setsockopt(fd, posix.SOL.SOCKET, posix.SO.RCVTIMEO, std.mem.asBytes(&rcv)) catch {};
        posix.setsockopt(fd, posix.SOL.SOCKET, posix.SO.SNDTIMEO, std.mem.asBytes(&snd)) catch {};
    }

    const cors_headers = [_]std.http.Header{
        .{ .name = "access-control-allow-origin", .value = "*" },
    };

    fn handleRequest(self: *Server, request: *std.http.Server.Request, stream: std.Io.net.Stream) !void {
        const target = request.head.target;
        const path = target[0 .. std.mem.indexOfScalar(u8, target, '?') orelse target.len];
        const method = request.head.method;

        if (method == .OPTIONS) {
            return request.respond("", .{ .status = .no_content, .extra_headers = &[_]std.http.Header{
                .{ .name = "access-control-allow-origin", .value = "*" },
                .{ .name = "access-control-allow-methods", .value = "GET, POST, OPTIONS" },
                .{ .name = "access-control-allow-headers", .value = "Content-Type, Authorization" },
            } });
        }

        if (method == .GET and std.mem.eql(u8, path, "/health"))
            return self.handleHealth(request);

        if (method == .GET and (std.mem.eql(u8, path, "/v1/models") or std.mem.eql(u8, path, "/models"))) {
            if (!self.authorized(request)) return self.respondError(request, .{
                .status = .unauthorized,
                .kind = "invalid_request_error",
                .message = "missing or invalid API key",
                .code = "invalid_api_key",
            });
            return self.handleModels(request);
        }

        const dialect: ?openai.Dialect = if (method == .POST and
            (std.mem.eql(u8, path, "/v1/chat/completions") or std.mem.eql(u8, path, "/chat/completions")))
            .chat
        else if (method == .POST and
            (std.mem.eql(u8, path, "/v1/responses") or std.mem.eql(u8, path, "/responses")))
            .responses
        else
            null;

        if (dialect) |d| {
            if (!self.authorized(request)) return self.respondError(request, .{
                .status = .unauthorized,
                .kind = "invalid_request_error",
                .message = "missing or invalid API key",
                .code = "invalid_api_key",
            });
            return self.handleGenerate(request, stream, d);
        }

        return self.respondError(request, .{
            .status = .not_found,
            .kind = "invalid_request_error",
            .message = "unknown endpoint",
        });
    }

    fn authorized(self: *Server, request: *std.http.Server.Request) bool {
        const key = self.opts.api_key orelse return true;
        var it = request.iterateHeaders();
        while (it.next()) |h| {
            if (std.ascii.eqlIgnoreCase(h.name, "authorization")) {
                const prefix = "Bearer ";
                if (h.value.len == prefix.len + key.len and
                    std.ascii.startsWithIgnoreCase(h.value, prefix) and
                    std.mem.eql(u8, h.value[prefix.len..], key)) return true;
            }
        }
        return false;
    }

    fn respondJson(self: *Server, request: *std.http.Server.Request, status: std.http.Status, body: []const u8) !void {
        _ = self;
        try request.respond(body, .{
            .status = status,
            .extra_headers = &(cors_headers ++ [_]std.http.Header{
                .{ .name = "content-type", .value = "application/json" },
            }),
        });
    }

    fn respondError(self: *Server, request: *std.http.Server.Request, info: openai.ErrorInfo) !void {
        var buf: [2048]u8 = undefined;
        var fixed = std.Io.Writer.fixed(&buf);
        openai.writeErrorBody(info, &fixed) catch return error.WriteFailed;
        try self.respondJson(request, info.status, fixed.buffered());
    }

    fn handleHealth(self: *Server, request: *std.http.Server.Request) !void {
        var buf: [512]u8 = undefined;
        var fixed = std.Io.Writer.fixed(&buf);
        var s: std.json.Stringify = .{ .writer = &fixed };
        s.beginObject() catch return error.WriteFailed;
        s.objectField("status") catch return error.WriteFailed;
        s.write("ok") catch return error.WriteFailed;
        s.objectField("model") catch return error.WriteFailed;
        s.write(self.backend.info.model_id) catch return error.WriteFailed;
        s.objectField("queue_depth") catch return error.WriteFailed;
        s.write(self.sched.depth()) catch return error.WriteFailed;
        s.endObject() catch return error.WriteFailed;
        try self.respondJson(request, .ok, fixed.buffered());
    }

    fn handleModels(self: *Server, request: *std.http.Server.Request) !void {
        var buf: [1024]u8 = undefined;
        var fixed = std.Io.Writer.fixed(&buf);
        var s: std.json.Stringify = .{ .writer = &fixed };
        blk: {
            s.beginObject() catch break :blk;
            s.objectField("object") catch break :blk;
            s.write("list") catch break :blk;
            s.objectField("data") catch break :blk;
            s.beginArray() catch break :blk;
            s.beginObject() catch break :blk;
            s.objectField("id") catch break :blk;
            s.write(self.backend.info.model_id) catch break :blk;
            s.objectField("object") catch break :blk;
            s.write("model") catch break :blk;
            s.objectField("created") catch break :blk;
            s.write(self.nowSeconds()) catch break :blk;
            s.objectField("owned_by") catch break :blk;
            s.write("fucina") catch break :blk;
            s.endObject() catch break :blk;
            s.endArray() catch break :blk;
            s.endObject() catch break :blk;
        }
        try self.respondJson(request, .ok, fixed.buffered());
    }

    fn nowSeconds(self: *Server) i64 {
        const ns = std.Io.Clock.real.now(self.io).nanoseconds;
        return @intCast(@divTrunc(ns, std.time.ns_per_s));
    }

    /// SSE start callback for the emitter: writes the response head on
    /// first use and exposes a through-flushing writer (frame flushes reach
    /// the socket, not just the chunk buffer).
    const SseState = struct {
        request: *std.http.Server.Request,
        conn_out: *std.Io.Writer,
        body: std.http.BodyWriter = undefined,
        body_buf: [4096]u8 = undefined,
        started: bool = false,
        interface: std.Io.Writer,

        fn starter(self: *SseState) emitter_mod.StreamStarter {
            return .{ .ptr = self, .startFn = start };
        }

        fn start(ptr: *anyopaque) anyerror!*std.Io.Writer {
            const self: *SseState = @ptrCast(@alignCast(ptr));
            self.body = try self.request.respondStreaming(&self.body_buf, .{
                .respond_options = .{
                    .extra_headers = &(cors_headers ++ [_]std.http.Header{
                        .{ .name = "content-type", .value = "text/event-stream" },
                        .{ .name = "cache-control", .value = "no-cache" },
                    }),
                },
            });
            self.started = true;
            self.interface = .{
                .vtable = &.{ .drain = drain, .flush = flushThrough },
                .buffer = &.{},
            };
            return &self.interface;
        }

        fn drain(w: *std.Io.Writer, data: []const []const u8, splat: usize) std.Io.Writer.Error!usize {
            const self: *SseState = @alignCast(@fieldParentPtr("interface", w));
            var consumed: usize = 0;
            for (data[0 .. data.len - 1]) |slice| {
                try self.body.writer.writeAll(slice);
                consumed += slice.len;
            }
            const last = data[data.len - 1];
            for (0..splat) |_| try self.body.writer.writeAll(last);
            consumed += last.len * splat;
            return consumed;
        }

        fn flushThrough(w: *std.Io.Writer) std.Io.Writer.Error!void {
            const self: *SseState = @alignCast(@fieldParentPtr("interface", w));
            try self.body.writer.flush();
            try self.conn_out.flush();
        }
    };

    fn handleGenerate(self: *Server, request: *std.http.Server.Request, stream: std.Io.net.Stream, dialect: openai.Dialect) !void {
        var arena_state = std.heap.ArenaAllocator.init(self.allocator);
        defer arena_state.deinit();
        const arena = arena_state.allocator();

        // Read the whole body (bounded) before anything else.
        var transfer_buf: [4096]u8 = undefined;
        const body_reader = request.readerExpectContinue(&transfer_buf) catch
            return self.respondError(request, .{ .status = .bad_request, .message = "bad expect header" });
        const body = body_reader.allocRemaining(arena, .limited(self.opts.max_body_bytes)) catch |err| switch (err) {
            error.StreamTooLong => return self.respondError(request, .{
                .status = .payload_too_large,
                .message = "request body too large",
            }),
            else => return err,
        };

        const parsed: openai.Parsed = switch (openai.parse(arena, dialect, body, self.backend.info)) {
            .ok => |p| p,
            .err => |info| return self.respondError(request, info),
        };

        // Cheap backend validation (message shape, prompt length) — still a
        // plain 400, before anything streams.
        self.backend.validate(&parsed.gen) catch |err|
            return self.respondError(request, openai.mapError(err));

        var sse = SseState{
            .request = request,
            .conn_out = request.server.out,
            .interface = undefined,
        };
        var em = emitter_mod.Emitter.init(arena, &parsed, .{
            .dialect = dialect,
            .model_id = self.backend.info.model_id,
            .created = self.nowSeconds(),
            .think_markers = self.backend.info.think_markers,
            .starter = sse.starter(),
        });
        var job = scheduler_mod.Job{ .req = parsed.gen, .sink = em.sink() };
        em.job = &job;

        self.sched.submit(&job) catch |err| switch (err) {
            error.QueueFull => return self.respondError(request, .{
                .status = .too_many_requests,
                .kind = "rate_limit_error",
                .message = "the request queue is full; retry shortly",
                .code = "rate_limit_exceeded",
            }),
            error.ShuttingDown => return self.respondError(request, openai.mapError(error.ShuttingDown)),
        };

        // Wait for the worker, watching the socket for client hang-up.
        while (!job.waitTimed(self.io, std.time.ns_per_s)) {
            if (clientGone(stream.socket.handle) or self.shutdown.load(.acquire)) job.cancel();
        }

        // A failure here means the SSE tail could not be written (client
        // vanished mid-epilogue): drop the connection.
        const outcome = em.finish(job.err) catch return error.WriteFailed;
        switch (outcome) {
            .plain_error => |info| try self.respondError(request, info),
            .body => |bytes| try self.respondJson(request, .ok, bytes),
            .streamed => if (sse.started) {
                sse.body.end() catch return error.WriteFailed;
            },
        }
    }
};

/// True when the peer closed its end (half-close or reset): a zero-byte
/// MSG_PEEK read. Pending pipelined bytes mean the client is alive.
fn clientGone(fd: posix.socket_t) bool {
    var probe: [1]u8 = undefined;
    const rc = std.c.recv(fd, &probe, 1, std.c.MSG.PEEK | std.c.MSG.DONTWAIT);
    if (rc == 0) return true;
    if (rc > 0) return false;
    return switch (std.posix.errno(rc)) {
        .AGAIN, .INTR => false,
        else => true,
    };
}
