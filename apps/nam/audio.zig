//! Zig wrapper over the miniaudio C shim (audio_shim.c): device
//! enumeration and a full-duplex mono f32 stream with a fixed period size.
//! The data callback runs on the OS realtime audio thread — no allocation,
//! no locks, no Fucina thread pool in there.

const std = @import("std");

pub const max_devices = 64;
pub const name_cap = 256;

const NamAudio = opaque {};

/// `output` is null for capture-only streams; `input` may be null in
/// pathological device states — callbacks must handle both.
pub const RawCallback = *const fn (user: ?*anyopaque, output: ?[*]f32, input: ?[*]const f32, frame_count: c_uint) callconv(.c) void;

extern fn nam_audio_create() ?*NamAudio;
extern fn nam_audio_destroy(audio: ?*NamAudio) void;
extern fn nam_audio_list_devices(audio: ?*NamAudio, kind: c_int, name_buf: [*]u8, name_cap_arg: c_int, default_flags: [*]u8, cap: c_int) c_int;
extern fn nam_audio_start(audio: ?*NamAudio, capture_index: c_int, playback_index: c_int, sample_rate: c_uint, period_frames: c_uint, callback: RawCallback, user: ?*anyopaque) c_int;
extern fn nam_audio_start_capture(audio: ?*NamAudio, capture_index: c_int, sample_rate: c_uint, period_frames: c_uint, callback: RawCallback, user: ?*anyopaque) c_int;
extern fn nam_audio_stop(audio: ?*NamAudio) void;
extern fn nam_audio_actual_sample_rate(audio: ?*NamAudio) c_uint;
extern fn nam_audio_internal_sample_rate(audio: ?*NamAudio, kind: c_int) c_uint;
extern fn nam_audio_device_latency_frames(audio: ?*NamAudio, kind: c_int) c_uint;
extern fn nam_audio_internal_period_frames(audio: ?*NamAudio, kind: c_int) c_uint;
extern fn nam_audio_running_names(audio: ?*NamAudio, capture_name: [*]u8, playback_name: [*]u8, cap: c_int) void;

pub const DeviceKind = enum(c_int) { playback = 0, capture = 1 };

pub const DeviceInfo = struct {
    name: [name_cap]u8,
    is_default: bool,

    pub fn nameSlice(self: *const DeviceInfo) []const u8 {
        const end = std.mem.indexOfScalar(u8, &self.name, 0) orelse self.name.len;
        return self.name[0..end];
    }
};

pub const Audio = struct {
    handle: *NamAudio,

    pub fn init() !Audio {
        const handle = nam_audio_create() orelse return error.AudioContextInit;
        return .{ .handle = handle };
    }

    pub fn deinit(self: *Audio) void {
        nam_audio_destroy(self.handle);
        self.* = undefined;
    }

    /// Fills `out` and returns the slice of discovered devices.
    pub fn listDevices(self: *Audio, kind: DeviceKind, out: *[max_devices]DeviceInfo) ![]DeviceInfo {
        var names: [max_devices][name_cap]u8 = undefined;
        var defaults: [max_devices]u8 = undefined;
        const count = nam_audio_list_devices(self.handle, @intFromEnum(kind), @ptrCast(&names), name_cap, &defaults, max_devices);
        if (count < 0) return error.DeviceEnumeration;
        const n = @min(@as(usize, @intCast(count)), max_devices);
        for (0..n) |i| {
            out[i] = .{ .name = names[i], .is_default = defaults[i] != 0 };
        }
        return out[0..n];
    }

    /// Starts the duplex stream; `capture`/`playback` are enumeration
    /// indices or null for the system default.
    pub fn start(self: *Audio, capture: ?usize, playback: ?usize, sample_rate: u32, period_frames: u32, callback: RawCallback, user: ?*anyopaque) !void {
        const rc = nam_audio_start(
            self.handle,
            if (capture) |c| @intCast(c) else -1,
            if (playback) |p| @intCast(p) else -1,
            sample_rate,
            period_frames,
            callback,
            user,
        );
        if (rc != 0) return switch (rc) {
            -3 => error.DeviceIndexOutOfRange,
            -4 => error.DeviceInit,
            -5 => error.DeviceStart,
            else => error.AudioStart,
        };
    }

    /// Capture-only stream for input probing; the callback sees output == null.
    pub fn startCapture(self: *Audio, capture: usize, sample_rate: u32, period_frames: u32, callback: RawCallback, user: ?*anyopaque) !void {
        const rc = nam_audio_start_capture(self.handle, @intCast(capture), sample_rate, period_frames, callback, user);
        if (rc != 0) return error.CaptureStart;
    }

    pub fn stop(self: *Audio) void {
        nam_audio_stop(self.handle);
    }

    pub fn actualSampleRate(self: *Audio) u32 {
        return nam_audio_actual_sample_rate(self.handle);
    }

    /// The device's native rate on one side; differing from the stream
    /// rate means miniaudio is resampling (extra latency).
    pub fn internalSampleRate(self: *Audio, kind: DeviceKind) u32 {
        return nam_audio_internal_sample_rate(self.handle, @intFromEnum(kind));
    }

    /// Real per-device latency in frames (CoreAudio device latency +
    /// safety offset + device buffer); 0 when unknown.
    pub fn deviceLatencyFrames(self: *Audio, kind: DeviceKind) u32 {
        return nam_audio_device_latency_frames(self.handle, @intFromEnum(kind));
    }

    /// The device-side period miniaudio actually negotiated.
    pub fn internalPeriodFrames(self: *Audio, kind: DeviceKind) u32 {
        return nam_audio_internal_period_frames(self.handle, @intFromEnum(kind));
    }

    pub fn runningNames(self: *Audio, capture_buf: *[name_cap]u8, playback_buf: *[name_cap]u8) void {
        nam_audio_running_names(self.handle, capture_buf, playback_buf, name_cap);
    }
};
