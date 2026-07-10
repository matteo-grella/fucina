//! Direct speaker playback for OmniVoice (`--play`): a playback-only
//! miniaudio device (play_shim.c, reusing NAM's vendored miniaudio TU) fed
//! by a lock-free SPSC ring buffer. The device callback runs on the OS
//! realtime audio thread — no allocation, no locks, no logging in there;
//! underrun accounting happens via atomics and is reported (throttled) from
//! the producer thread.
//!
//! Library-style seams for embedders: `Player.init` / `pushSamples` /
//! `drainAndStop` / `deinit`, plus `listPlaybackDevices` for enumeration.
//! Synthesis slower than realtime (the CPU MaskGIT streaming path) is an
//! EXPECTED producer regime: gaps between chunks play as silence and are
//! counted, not treated as errors.

const std = @import("std");

pub const max_devices = 64;
pub const name_cap = 256;

const OvPlay = opaque {};

/// `output` is always non-null for playback-only streams.
pub const RawCallback = *const fn (user: ?*anyopaque, output: [*]f32, frame_count: c_uint) callconv(.c) void;

extern fn ov_play_create() ?*OvPlay;
extern fn ov_play_destroy(play: ?*OvPlay) void;
extern fn ov_play_list_devices(play: ?*OvPlay, name_buf: [*]u8, name_cap_arg: c_int, default_flags: [*]u8, cap: c_int) c_int;
extern fn ov_play_start(play: ?*OvPlay, playback_index: c_int, sample_rate: c_uint, period_frames: c_uint, callback: RawCallback, user: ?*anyopaque) c_int;
extern fn ov_play_stop(play: ?*OvPlay) void;
extern fn ov_play_internal_sample_rate(play: ?*OvPlay) c_uint;

pub const DeviceInfo = struct {
    name: [name_cap]u8,
    is_default: bool,

    pub fn nameSlice(self: *const DeviceInfo) []const u8 {
        const end = std.mem.indexOfScalar(u8, &self.name, 0) orelse self.name.len;
        return self.name[0..end];
    }
};

/// Enumerates playback devices into `out` (indices are the `--playback`
/// argument); creates and destroys a temporary device context.
pub fn listPlaybackDevices(out: *[max_devices]DeviceInfo) ![]DeviceInfo {
    const handle = ov_play_create() orelse return error.AudioContextInit;
    defer ov_play_destroy(handle);
    var names: [max_devices][name_cap]u8 = undefined;
    var defaults: [max_devices]u8 = undefined;
    const count = ov_play_list_devices(handle, @ptrCast(&names), name_cap, &defaults, max_devices);
    if (count < 0) return error.DeviceEnumeration;
    const n = @min(@as(usize, @intCast(count)), max_devices);
    for (0..n) |i| {
        out[i] = .{ .name = names[i], .is_default = defaults[i] != 0 };
    }
    return out[0..n];
}

/// Lock-free single-producer/single-consumer f32 ring over a power-of-two
/// buffer. Positions are monotonically increasing frame counters (masked on
/// access), so `write - read` is always the exact fill level. The producer
/// side is `pushSlice`, the consumer side (`popSlice`) is the realtime audio
/// callback: both are allocation- and lock-free.
pub const Ring = struct {
    buf: []f32,
    write_pos: std.atomic.Value(usize),
    read_pos: std.atomic.Value(usize),

    pub fn init(allocator: std.mem.Allocator, capacity: usize) !Ring {
        std.debug.assert(capacity >= 2 and std.math.isPowerOfTwo(capacity));
        return .{
            .buf = try allocator.alloc(f32, capacity),
            .write_pos = .init(0),
            .read_pos = .init(0),
        };
    }

    pub fn deinit(self: *Ring, allocator: std.mem.Allocator) void {
        allocator.free(self.buf);
        self.* = undefined;
    }

    /// Frames currently readable.
    pub fn len(self: *const Ring) usize {
        return self.write_pos.load(.acquire) - self.read_pos.load(.acquire);
    }

    /// Frames currently writable without overwriting unread data.
    pub fn space(self: *const Ring) usize {
        return self.buf.len - self.len();
    }

    /// Copies as much of `samples` as fits right now; returns the count
    /// copied (producer side).
    pub fn pushSlice(self: *Ring, samples: []const f32) usize {
        const write = self.write_pos.load(.monotonic);
        const read = self.read_pos.load(.acquire);
        const n = @min(samples.len, self.buf.len - (write - read));
        if (n == 0) return 0;
        const mask = self.buf.len - 1;
        const start = write & mask;
        const first = @min(n, self.buf.len - start);
        @memcpy(self.buf[start..][0..first], samples[0..first]);
        @memcpy(self.buf[0 .. n - first], samples[first..n]);
        self.write_pos.store(write + n, .release);
        return n;
    }

    /// Fills the head of `out` with available frames; returns the count of
    /// real frames written — the caller supplies silence for the remainder
    /// (consumer side; realtime-safe).
    pub fn popSlice(self: *Ring, out: []f32) usize {
        const read = self.read_pos.load(.monotonic);
        const write = self.write_pos.load(.acquire);
        const n = @min(out.len, write - read);
        if (n == 0) return 0;
        const mask = self.buf.len - 1;
        const start = read & mask;
        const first = @min(n, self.buf.len - start);
        @memcpy(out[0..first], self.buf[start..][0..first]);
        @memcpy(out[first..n], self.buf[0 .. n - first]);
        self.read_pos.store(read + n, .release);
        return n;
    }
};

pub const Options = struct {
    /// Playback device enumeration index (see `listPlaybackDevices`);
    /// null = system default.
    device_index: ?usize = null,
    /// Stream rate handed to miniaudio; the device converts internally.
    sample_rate: u32 = 24000,
    /// Requested device period. Playback tolerates large periods; 512
    /// frames ≈ 21 ms @ 24 kHz keeps gap boundaries reasonably tight.
    period_frames: u32 = 512,
    /// Ring capacity in frames (power of two). 2^19 ≈ 21.8 s @ 24 kHz —
    /// a full long-form chunk fits without blocking the producer.
    ring_frames: usize = 1 << 19,
    /// Print a throttled stderr note when the device starves mid-stream
    /// (expected while streamed synthesis runs slower than realtime).
    log_underruns: bool = true,
};

pub const Stats = struct {
    /// Real synthesis frames delivered to the device.
    frames_played: usize,
    /// INTERIOR starvation episodes: gaps with real audio on both sides
    /// (chunk gaps). Lead-in and drain-tail silence never counts.
    underrun_episodes: usize,
    /// Silence frames inserted during those episodes.
    silence_frames: usize,
};

/// A running playback device + ring. Heap-allocated by `init` (the device
/// callback holds its address). One producer thread pushes with
/// `pushSamples`; `drainAndStop` blocks until the device has consumed
/// everything, then stops it. `deinit` releases everything (stopping the
/// device first if needed).
pub const Player = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    handle: *OvPlay,
    ring: Ring,
    opts: Options,

    // Shared with the realtime callback (atomics only).
    played_any: std.atomic.Value(bool),
    finished: std.atomic.Value(bool),
    underrun_episodes: std.atomic.Value(usize),
    silence_frames: std.atomic.Value(usize),
    /// Callback-thread-local: whether the device is currently starved, and
    /// how much silence the open gap has accumulated. A gap is committed to
    /// the atomics only when real data RESUMES — a gap still open at drain
    /// time is the normal end-of-stream tail, not an underrun.
    in_gap: bool,
    gap_frames: usize,

    // Producer-thread-local underrun log throttle.
    logged_episodes: usize,
    last_log_ns: i96,

    pub fn init(allocator: std.mem.Allocator, io: std.Io, opts: Options) !*Player {
        const self = try allocator.create(Player);
        errdefer allocator.destroy(self);
        self.* = .{
            .allocator = allocator,
            .io = io,
            .handle = undefined,
            .ring = try Ring.init(allocator, opts.ring_frames),
            .opts = opts,
            .played_any = .init(false),
            .finished = .init(false),
            .underrun_episodes = .init(0),
            .silence_frames = .init(0),
            .in_gap = false,
            .gap_frames = 0,
            .logged_episodes = 0,
            .last_log_ns = 0,
        };
        errdefer self.ring.deinit(allocator);

        self.handle = ov_play_create() orelse return error.AudioContextInit;
        errdefer ov_play_destroy(self.handle);
        const rc = ov_play_start(
            self.handle,
            if (opts.device_index) |d| @intCast(d) else -1,
            opts.sample_rate,
            opts.period_frames,
            deviceCallback,
            self,
        );
        if (rc != 0) return switch (rc) {
            -3 => error.DeviceIndexOutOfRange,
            -4 => error.DeviceInit,
            -5 => error.DeviceStart,
            else => error.AudioStart,
        };
        return self;
    }

    pub fn deinit(self: *Player) void {
        ov_play_stop(self.handle);
        ov_play_destroy(self.handle);
        self.ring.deinit(self.allocator);
        const allocator = self.allocator;
        self.* = undefined;
        allocator.destroy(self);
    }

    /// The playback device's native rate (0 before start); differing from
    /// `opts.sample_rate` means miniaudio resamples internally — expected.
    pub fn deviceSampleRate(self: *Player) u32 {
        return ov_play_internal_sample_rate(self.handle);
    }

    /// Queues samples for playback, blocking while the ring is full (the
    /// device drains at realtime speed). Producer thread only.
    pub fn pushSamples(self: *Player, samples: []const f32) !void {
        var rest = samples;
        while (rest.len > 0) {
            const n = self.ring.pushSlice(rest);
            rest = rest[n..];
            self.reportUnderruns();
            if (rest.len > 0) try std.Io.sleep(self.io, .{ .nanoseconds = 2 * std.time.ns_per_ms }, .awake);
        }
    }

    /// Blocks until the ring is empty and the device tail has played out,
    /// then stops the device and returns the playback stats.
    pub fn drainAndStop(self: *Player) !Stats {
        while (self.ring.len() > 0) {
            self.reportUnderruns();
            try std.Io.sleep(self.io, .{ .nanoseconds = 5 * std.time.ns_per_ms }, .awake);
        }
        // The last real frames are still in the device's own buffer: give it
        // two periods plus a safety margin before tearing the stream down.
        self.finished.store(true, .release);
        const tail_ms: u64 = 100 + (2 * @as(u64, self.opts.period_frames) * 1000) / self.opts.sample_rate;
        try std.Io.sleep(self.io, .{ .nanoseconds = tail_ms * std.time.ns_per_ms }, .awake);
        ov_play_stop(self.handle);
        self.reportUnderruns();
        return .{
            .frames_played = self.ring.read_pos.load(.acquire),
            .underrun_episodes = self.underrun_episodes.load(.acquire),
            .silence_frames = self.silence_frames.load(.acquire),
        };
    }

    /// Throttled producer-side note for new starvation episodes: at most one
    /// line per second, never from the realtime callback.
    fn reportUnderruns(self: *Player) void {
        if (!self.opts.log_underruns) return;
        const episodes = self.underrun_episodes.load(.acquire);
        if (episodes <= self.logged_episodes) return;
        const now_ns = std.Io.Clock.awake.now(self.io).nanoseconds;
        if (self.last_log_ns != 0 and now_ns - self.last_log_ns < std.time.ns_per_s) return;
        self.logged_episodes = episodes;
        self.last_log_ns = now_ns;
        std.debug.print("[Play] playback gap {d} (device starved between chunks) — expected when generation runs slower than realtime\n", .{episodes});
    }

    /// Realtime audio thread: drain the ring, pad with silence, account for
    /// starvation. No allocation, no locks, no I/O.
    fn deviceCallback(user: ?*anyopaque, output: [*]f32, frame_count: c_uint) callconv(.c) void {
        const self: *Player = @ptrCast(@alignCast(user.?));
        const out = output[0..frame_count];
        const got = self.ring.popSlice(out);
        if (got > 0) {
            self.played_any.store(true, .release);
            if (self.in_gap) {
                // Real data resumed: the gap was interior — commit it.
                _ = self.underrun_episodes.fetchAdd(1, .monotonic);
                _ = self.silence_frames.fetchAdd(self.gap_frames, .monotonic);
                self.in_gap = false;
                self.gap_frames = 0;
            }
        }
        if (got == out.len) return;
        @memset(out[got..], 0);
        // Silence before the first chunk or after the final drain is normal;
        // only mid-stream starvation opens a gap.
        if (self.played_any.load(.acquire) and !self.finished.load(.acquire)) {
            self.in_gap = true;
            self.gap_frames += out.len - got;
        }
    }
};

test {
    _ = @import("play_tests.zig");
}
