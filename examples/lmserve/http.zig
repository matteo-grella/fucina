//! The HTTP front end: accept loop, per-connection threads (capped),
//! socket deadlines, routing, SSE plumbing, and graceful shutdown. Built on
//! Zig 0.16 `std.http.Server` (one blocking server per connection) — the
//! pieces std deliberately leaves out (timeouts, connection cap, lifecycle)
//! are handled here.

const std = @import("std");
const posix = std.posix;
const types = @import("types.zig");
const openai = @import("openai.zig");
const anthropic = @import("anthropic.zig");
const emitter_mod = @import("emitter.zig");
const scheduler_mod = @import("scheduler.zig");

/// One generation endpoint: the request parser, response framing, and error
/// envelope all follow from it.
const Route = enum {
    chat,
    responses,
    anthropic,

    fn wire(self: Route) emitter_mod.Wire {
        return switch (self) {
            .chat => .chat,
            .responses => .responses,
            .anthropic => .anthropic,
        };
    }
};

fn lock(m: *std.Io.Mutex) void {
    std.Io.Threaded.mutexLock(m);
}

fn unlock(m: *std.Io.Mutex) void {
    std.Io.Threaded.mutexUnlock(m);
}

/// Cross-thread byte pipe between the inference worker (producer: the
/// emitter's SSE frames) and the connection thread (consumer: HTTP head +
/// chunked body writes to the socket). The worker never touches the
/// socket, so a stalled client stalls only its own connection thread —
/// never generation, neither the sequential queue behind it nor the other
/// streams of a `--batch` lockstep. Capacity is intrinsically bounded by
/// the reply itself (`max_tokens` frames); a consumer whose socket died
/// marks the pipe failed and the producer's next write aborts that
/// stream's generation, exactly as a direct socket write used to.
pub const StreamPipe = struct {
    allocator: Allocator,
    io: std.Io,
    mutex: std.Io.Mutex = .init,
    /// Frame bytes; `consumed` is the socket-written prefix (compacted
    /// whenever the consumer fully catches up — the common case).
    data: std.ArrayList(u8) = .empty,
    consumed: usize = 0,
    /// SSE streaming began: the consumer must write the HTTP head before
    /// any body bytes.
    started: bool = false,
    /// The consumer's socket died: producer writes fail from here on.
    failed: bool = false,
    /// Futex word for the consumer's wait: bumped on every append and on
    /// stream start (the scheduler bumps it on job finish via
    /// `Job.notify`), so a load-then-wait consumer can never miss an
    /// event.
    seq: std.atomic.Value(u32) = .{ .raw = 0 },
    interface: std.Io.Writer = .{ .vtable = &.{ .drain = drain }, .buffer = &.{} },

    pub fn deinit(self: *StreamPipe) void {
        self.data.deinit(self.allocator);
    }

    /// `emitter.StreamStarter`: called by the WORKER on the first delta.
    /// Only flags the stream — the response head is written by the
    /// consumer, keeping every socket byte on the connection thread.
    pub fn starter(self: *StreamPipe) emitter_mod.StreamStarter {
        return .{ .ptr = self, .startFn = start };
    }

    fn start(ptr: *anyopaque) anyerror!*std.Io.Writer {
        const self: *StreamPipe = @ptrCast(@alignCast(ptr));
        lock(&self.mutex);
        defer unlock(&self.mutex);
        if (self.failed) return error.WriteFailed;
        self.started = true;
        self.bumpLocked();
        return &self.interface;
    }

    fn bumpLocked(self: *StreamPipe) void {
        _ = self.seq.fetchAdd(1, .release);
        self.io.futexWake(u32, &self.seq.raw, std.math.maxInt(u32));
    }

    fn drain(w: *std.Io.Writer, data: []const []const u8, splat: usize) std.Io.Writer.Error!usize {
        const self: *StreamPipe = @alignCast(@fieldParentPtr("interface", w));
        lock(&self.mutex);
        defer unlock(&self.mutex);
        if (self.failed) return error.WriteFailed;
        var n: usize = 0;
        for (data[0 .. data.len - 1]) |slice| {
            self.data.appendSlice(self.allocator, slice) catch return error.WriteFailed;
            n += slice.len;
        }
        const last = data[data.len - 1];
        for (0..splat) |_| self.data.appendSlice(self.allocator, last) catch return error.WriteFailed;
        n += last.len * splat;
        self.bumpLocked();
        return n;
    }

    pub const Taken = struct { n: usize, started: bool };

    /// Consumer: copy the next pending chunk into `buf` (n == 0 when
    /// drained). The copy keeps the mutex held only for a memcpy — never
    /// across a socket write.
    pub fn take(self: *StreamPipe, buf: []u8) Taken {
        lock(&self.mutex);
        defer unlock(&self.mutex);
        const pending = self.data.items[self.consumed..];
        const n = @min(pending.len, buf.len);
        @memcpy(buf[0..n], pending[0..n]);
        self.consumed += n;
        if (self.consumed == self.data.items.len) {
            self.data.clearRetainingCapacity();
            self.consumed = 0;
        }
        return .{ .n = n, .started = self.started };
    }

    /// Consumer: the socket died — producer writes fail from now on (the
    /// worker aborts that stream at its next write).
    pub fn markFailed(self: *StreamPipe) void {
        lock(&self.mutex);
        defer unlock(&self.mutex);
        self.failed = true;
    }

    /// Consumer: wait until `seq` moves past `expected` (data, stream
    /// start, or job finish), up to `timeout_ns` — the `Job.waitTimed`
    /// futex pattern. Load `seq` BEFORE draining, so an event that lands
    /// after the drain returns immediately instead of sleeping.
    pub fn waitSeq(self: *StreamPipe, io: std.Io, expected: u32, timeout_ns: u64) void {
        if (self.seq.load(.acquire) != expected) return;
        io.futexWaitTimeout(u32, &self.seq.raw, expected, .{ .duration = .{
            .raw = .{ .nanoseconds = timeout_ns },
            .clock = .awake,
        } }) catch {};
    }
};

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
    /// Extra Host-header names accepted beyond the loopback set and the
    /// bind host (`--allow-host`). Setting any also ARMS the check on
    /// non-loopback binds (see `hostAllowed`).
    extra_hosts: []const []const u8 = &.{},
    /// When set (`--cors-origin`), responses carry
    /// `access-control-allow-origin: <value>` and OPTIONS preflights are
    /// answered permissively, letting browser pages from that origin (`*`
    /// for any) call the server. Default null emits NO CORS headers: any
    /// web page can still SEND a request to a localhost bind (and, with no
    /// `api_key`, have it served), but the browser withholds the response
    /// from the page's script. Non-browser clients are unaffected.
    cors_origin: ?[]const u8 = null,
};

/// The hostname of a Host header value: strips an optional port and IPv6
/// brackets (`[::1]:8080` -> `::1`, `example.com:80` -> `example.com`).
fn hostHeaderName(value: []const u8) []const u8 {
    if (value.len == 0) return value;
    if (value[0] == '[') {
        const close = std.mem.indexOfScalar(u8, value, ']') orelse return value;
        return value[1..close];
    }
    if (std.mem.lastIndexOfScalar(u8, value, ':')) |colon| {
        // Only a digits-only tail is a port (a bare IPv6 literal is not
        // valid in Host, but never mistake its groups for ports).
        const tail = value[colon + 1 ..];
        if (tail.len > 0 and std.mem.indexOfNone(u8, tail, "0123456789") == null and
            std.mem.indexOfScalar(u8, value[0..colon], ':') == null)
            return value[0..colon];
    }
    return value;
}

fn isLoopbackName(name: []const u8) bool {
    return std.ascii.eqlIgnoreCase(name, "localhost") or
        std.mem.eql(u8, name, "127.0.0.1") or
        std.mem.eql(u8, name, "::1");
}

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

    /// `base` plus the configured CORS origin header, if any. `buf` must
    /// hold `base.len + 1` entries; the returned slice aliases it.
    fn withCors(cors_origin: ?[]const u8, buf: []std.http.Header, base: []const std.http.Header) []const std.http.Header {
        @memcpy(buf[0..base.len], base);
        var n = base.len;
        if (cors_origin) |origin| {
            buf[n] = .{ .name = "access-control-allow-origin", .value = origin };
            n += 1;
        }
        return buf[0..n];
    }

    /// DNS-rebinding guard: a browser lured to an attacker page resolves the
    /// attacker's domain to 127.0.0.1 and reaches this server with the
    /// attacker's Host header. Enforced whenever the bind is loopback (the
    /// legitimate names are then known: the loopback set + the bind host) or
    /// `--allow-host` was given; a non-loopback bind without `--allow-host`
    /// skips the check — the legitimate external names are unknowable here.
    /// A missing Host header passes: every browser sends one, and non-browser
    /// clients are not the rebinding audience.
    fn hostAllowed(self: *Server, request: *std.http.Server.Request) bool {
        const bind_is_loopback = isLoopbackName(self.opts.host) or std.mem.eql(u8, self.opts.host, "::1");
        if (!bind_is_loopback and self.opts.extra_hosts.len == 0) return true;

        var it = request.iterateHeaders();
        const host = blk: {
            while (it.next()) |h| {
                if (std.ascii.eqlIgnoreCase(h.name, "host")) break :blk h.value;
            }
            return true;
        };
        const name = hostHeaderName(host);
        if (isLoopbackName(name)) return true;
        if (std.ascii.eqlIgnoreCase(name, self.opts.host)) return true;
        for (self.opts.extra_hosts) |allowed| {
            if (std.ascii.eqlIgnoreCase(name, allowed)) return true;
        }
        return false;
    }

    fn handleRequest(self: *Server, request: *std.http.Server.Request, stream: std.Io.net.Stream) !void {
        if (!self.hostAllowed(request)) {
            return self.respondError(request, .{
                .status = .forbidden,
                .kind = "invalid_request_error",
                .message = "Host header not allowed (DNS-rebinding guard); add it with --allow-host",
            });
        }
        const target = request.head.target;
        const path = target[0 .. std.mem.indexOfScalar(u8, target, '?') orelse target.len];
        const method = request.head.method;

        if (method == .OPTIONS) {
            // Without --cors-origin the 204 carries no CORS headers, so the
            // browser fails the preflight and never reads a response.
            if (self.opts.cors_origin) |origin| {
                return request.respond("", .{ .status = .no_content, .extra_headers = &[_]std.http.Header{
                    .{ .name = "access-control-allow-origin", .value = origin },
                    .{ .name = "access-control-allow-methods", .value = "GET, POST, OPTIONS" },
                    .{ .name = "access-control-allow-headers", .value = "Content-Type, Authorization, x-api-key, anthropic-version, anthropic-beta" },
                } });
            }
            return request.respond("", .{ .status = .no_content });
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

        const route: ?Route = if (method == .POST and
            (std.mem.eql(u8, path, "/v1/chat/completions") or std.mem.eql(u8, path, "/chat/completions")))
            .chat
        else if (method == .POST and
            (std.mem.eql(u8, path, "/v1/responses") or std.mem.eql(u8, path, "/responses")))
            .responses
        else if (method == .POST and
            (std.mem.eql(u8, path, "/v1/messages") or std.mem.eql(u8, path, "/messages")))
            .anthropic
        else
            null;

        if (route) |r| {
            if (!self.authorized(request)) return self.respondRouteError(request, r, .{
                .status = .unauthorized,
                .kind = "invalid_request_error",
                .message = "missing or invalid API key",
                .code = "invalid_api_key",
            });
            return self.handleGenerate(request, stream, r);
        }

        // Anthropic clients under /v1/messages/* (e.g. count_tokens) parse
        // their own error envelope.
        const not_found = openai.ErrorInfo{
            .status = .not_found,
            .kind = "invalid_request_error",
            .message = "unknown endpoint",
        };
        if (std.mem.startsWith(u8, path, "/v1/messages"))
            return self.respondRouteError(request, .anthropic, not_found);
        return self.respondError(request, not_found);
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
            } else if (std.ascii.eqlIgnoreCase(h.name, "x-api-key")) {
                // Anthropic clients (Claude Code, the SDKs) send the key in
                // x-api-key; accepted wherever Bearer is.
                if (std.mem.eql(u8, h.value, key)) return true;
            }
        }
        return false;
    }

    fn respondJson(self: *Server, request: *std.http.Server.Request, status: std.http.Status, body: []const u8) !void {
        var hdrs: [2]std.http.Header = undefined;
        try request.respond(body, .{
            .status = status,
            .extra_headers = withCors(self.opts.cors_origin, &hdrs, &.{
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

    /// Like `respondError`, in the route's own error envelope.
    fn respondRouteError(self: *Server, request: *std.http.Server.Request, route: Route, info: openai.ErrorInfo) !void {
        if (route != .anthropic) return self.respondError(request, info);
        var buf: [2048]u8 = undefined;
        var fixed = std.Io.Writer.fixed(&buf);
        anthropic.writeErrorBody(info, &fixed) catch return error.WriteFailed;
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

    /// The connection thread's half of SSE: writes the response head on the
    /// first drained sign of streaming, then relays pipe bytes through the
    /// chunked body writer. Driven by the CONNECTION thread only — the
    /// worker's frames arrive through the `StreamPipe`.
    const SseState = struct {
        request: *std.http.Server.Request,
        conn_out: *std.Io.Writer,
        cors_origin: ?[]const u8 = null,
        body: std.http.BodyWriter = undefined,
        body_buf: [4096]u8 = undefined,
        started: bool = false,

        fn begin(self: *SseState) !void {
            var hdrs: [3]std.http.Header = undefined;
            self.body = try self.request.respondStreaming(&self.body_buf, .{
                .respond_options = .{
                    .extra_headers = withCors(self.cors_origin, &hdrs, &.{
                        .{ .name = "content-type", .value = "text/event-stream" },
                        .{ .name = "cache-control", .value = "no-cache" },
                    }),
                },
            });
            self.started = true;
        }
    };

    fn handleGenerate(self: *Server, request: *std.http.Server.Request, stream: std.Io.net.Stream, route: Route) !void {
        var arena_state = std.heap.ArenaAllocator.init(self.allocator);
        defer arena_state.deinit();
        const arena = arena_state.allocator();

        // Read the whole body (bounded) before anything else.
        var transfer_buf: [4096]u8 = undefined;
        const body_reader = request.readerExpectContinue(&transfer_buf) catch
            return self.respondRouteError(request, route, .{ .status = .bad_request, .message = "bad expect header" });
        const body = body_reader.allocRemaining(arena, .limited(self.opts.max_body_bytes)) catch |err| switch (err) {
            error.StreamTooLong => return self.respondRouteError(request, route, .{
                .status = .payload_too_large,
                .message = "request body too large",
            }),
            else => return err,
        };

        const parsed: openai.Parsed = switch (switch (route) {
            .chat => openai.parse(arena, .chat, body, self.backend.info),
            .responses => openai.parse(arena, .responses, body, self.backend.info),
            .anthropic => anthropic.parse(arena, body, self.backend.info),
        }) {
            .ok => |p| p,
            .err => |info| return self.respondRouteError(request, route, info),
        };

        // Cheap backend validation (message shape, prompt length) — still a
        // plain 400, before anything streams.
        self.backend.validate(&parsed.gen) catch |err|
            return self.respondRouteError(request, route, openai.mapError(err));

        var sse = SseState{
            .request = request,
            .conn_out = request.server.out,
            .cors_origin = self.opts.cors_origin,
        };
        var pipe = StreamPipe{ .allocator = self.allocator, .io = self.io };
        defer pipe.deinit();
        var em = emitter_mod.Emitter.init(arena, &parsed, .{
            .dialect = route.wire(),
            .model_id = self.backend.info.model_id,
            .created = self.nowSeconds(),
            .think_markers = self.backend.info.think_markers,
            .starter = pipe.starter(),
        });
        var job = scheduler_mod.Job{ .req = parsed.gen, .sink = em.sink(), .notify = &pipe.seq };
        em.job = &job;

        self.sched.submit(&job) catch |err| switch (err) {
            error.QueueFull => return self.respondRouteError(request, route, .{
                .status = .too_many_requests,
                .kind = "rate_limit_error",
                .message = "the request queue is full; retry shortly",
                .code = "rate_limit_exceeded",
            }),
            error.ShuttingDown => return self.respondRouteError(request, route, openai.mapError(error.ShuttingDown)),
        };

        // Relay pipe bytes to the socket and watch for client hang-up until
        // the worker finishes. A socket failure cancels the job and fails
        // the pipe (aborting that stream's generation at its next write)
        // but keeps looping: the job's memory lives on this frame, so the
        // thread may not leave before the worker is done with it.
        var sock_ok = true;
        while (true) {
            const seen = pipe.seq.load(.acquire);
            if (sock_ok) self.drainToSocket(&pipe, &sse) catch {
                sock_ok = false;
                pipe.markFailed();
                job.cancel();
            };
            if (job.finished()) break;
            if (clientGone(stream.socket.handle) or self.shutdown.load(.acquire)) job.cancel();
            pipe.waitSeq(self.io, seen, std.time.ns_per_s);
        }

        // A failure here means the SSE tail could not be written (client
        // vanished mid-epilogue): drop the connection. The emitter's tail
        // frames land in the pipe (and fail fast when the pipe is marked),
        // so the final drain below pushes them out before the terminator.
        const outcome = em.finish(job.err) catch return error.WriteFailed;
        switch (outcome) {
            .plain_error => |info| try self.respondRouteError(request, route, info),
            .body => |bytes| try self.respondJson(request, .ok, bytes),
            .streamed => {
                if (!sock_ok) return error.WriteFailed;
                self.drainToSocket(&pipe, &sse) catch return error.WriteFailed;
                if (sse.started) sse.body.end() catch return error.WriteFailed;
            },
        }
    }

    /// Push everything pending in the pipe to the socket: the SSE head on
    /// the first sign of streaming, then body bytes, flushed through to the
    /// wire once per call. Connection thread only.
    fn drainToSocket(self: *Server, pipe: *StreamPipe, sse: *SseState) !void {
        _ = self;
        var buf: [4096]u8 = undefined;
        var wrote = false;
        while (true) {
            const taken = pipe.take(&buf);
            if (taken.started and !sse.started) try sse.begin();
            if (taken.n == 0) break;
            try sse.body.writer.writeAll(buf[0..taken.n]);
            wrote = true;
        }
        if (wrote) {
            try sse.body.writer.flush();
            try sse.conn_out.flush();
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

test "stream pipe: frames in order, head-before-body flag, failure propagation" {
    var threaded: std.Io.Threaded = .init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var pipe = StreamPipe{ .allocator = std.testing.allocator, .io = io };
    defer pipe.deinit();

    // Nothing pending: empty take, not started.
    var buf: [7]u8 = undefined;
    var taken = pipe.take(&buf);
    try std.testing.expectEqual(@as(usize, 0), taken.n);
    try std.testing.expect(!taken.started);

    // Producer side: stream start + two frames through the emitter-facing
    // writer.
    const w = try StreamPipe.start(&pipe);
    try w.writeAll("data: a\n\n");
    try w.writeAll("data: b\n\n");

    // Consumer sees started with the bytes, in order, across takes smaller
    // than the pending span.
    var got: std.ArrayList(u8) = .empty;
    defer got.deinit(std.testing.allocator);
    while (true) {
        taken = pipe.take(&buf);
        try std.testing.expect(taken.started);
        if (taken.n == 0) break;
        try got.appendSlice(std.testing.allocator, buf[0..taken.n]);
    }
    try std.testing.expectEqualStrings("data: a\n\ndata: b\n\n", got.items);

    // Every event bumped seq, so a stale expected value returns instantly.
    try std.testing.expect(pipe.seq.load(.acquire) != 0);
    pipe.waitSeq(io, 0, std.time.ns_per_s);

    // Dead consumer: producer writes fail from here on.
    pipe.markFailed();
    try std.testing.expectError(error.WriteFailed, w.writeAll("data: c\n\n"));
}

test "host header name: ports and IPv6 brackets strip, loopback set matches" {
    try std.testing.expectEqualStrings("localhost", hostHeaderName("localhost:8080"));
    try std.testing.expectEqualStrings("localhost", hostHeaderName("localhost"));
    try std.testing.expectEqualStrings("127.0.0.1", hostHeaderName("127.0.0.1:80"));
    try std.testing.expectEqualStrings("::1", hostHeaderName("[::1]:8080"));
    try std.testing.expectEqualStrings("::1", hostHeaderName("[::1]"));
    try std.testing.expectEqualStrings("example.com", hostHeaderName("example.com:443"));
    try std.testing.expectEqualStrings("example.com", hostHeaderName("example.com"));
    // A non-numeric tail is not a port.
    try std.testing.expectEqualStrings("weird:name", hostHeaderName("weird:name"));

    try std.testing.expect(isLoopbackName("LOCALHOST"));
    try std.testing.expect(isLoopbackName("127.0.0.1"));
    try std.testing.expect(isLoopbackName("::1"));
    try std.testing.expect(!isLoopbackName("evil.example"));
    try std.testing.expect(!isLoopbackName("127.0.0.2"));
}
