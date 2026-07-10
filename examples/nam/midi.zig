//! Zig wrapper over the CoreMIDI C shim (midi_shim.c) plus everything
//! parseable in pure Zig: a MIDI 1.0 byte-stream parser, a lock-free SPSC
//! message queue (producer = CoreMIDI's read thread, consumer = the live
//! UI loop at its 33 ms tick), and the CC-number map for `--midi-map`.
//! MIDI never touches the audio callback: control changes land in the
//! same atomics the keyboard already writes.

const std = @import("std");

pub const max_sources = 16;
pub const name_cap = 256;

const NamMidi = opaque {};

const RawCallback = *const fn (user: ?*anyopaque, source: c_uint, bytes: ?[*]const u8, len: c_uint) callconv(.c) void;

extern fn nam_midi_create() ?*NamMidi;
extern fn nam_midi_destroy(midi: ?*NamMidi) void;
extern fn nam_midi_source_count(midi: ?*NamMidi) c_int;
extern fn nam_midi_sources_signature(midi: ?*NamMidi) i64;
extern fn nam_midi_list_sources(midi: ?*NamMidi, name_buf: [*]u8, name_cap_arg: c_int, cap: c_int) c_int;
extern fn nam_midi_start(midi: ?*NamMidi, source_index: c_int, callback: RawCallback, user: ?*anyopaque) c_int;
extern fn nam_midi_stop(midi: ?*NamMidi) void;

/// One complete MIDI channel message (status + the 1-2 data bytes).
pub const Message = struct {
    status: u8,
    data1: u8,
    data2: u8,

    /// High nibble: 0xB0 = control change, 0xC0 = program change, ...
    pub fn kind(self: Message) u8 {
        return self.status & 0xF0;
    }

    /// 0-based channel (wire format; "channel 1" on a pedal = 0 here).
    pub fn channel(self: Message) u8 {
        return self.status & 0x0F;
    }
};

pub const control_change: u8 = 0xB0;
pub const program_change: u8 = 0xC0;

/// Stateful MIDI 1.0 byte-stream parser: emits complete channel voice
/// messages, honors running status, skips SysEx and system common, and
/// tolerates realtime bytes interleaved mid-message (the wire allows
/// them anywhere). One instance per SOURCE (packets from different
/// devices interleave, and a multi-packet SysEx from one must not turn
/// another's bytes into garbage); CoreMIDI thread only.
pub const Parser = struct {
    status: u8 = 0,
    data: [2]u8 = .{ 0, 0 },
    have: u8 = 0,
    in_sysex: bool = false,

    fn dataLen(status: u8) u8 {
        // Program change and channel pressure carry one data byte;
        // every other channel voice message carries two.
        return switch (status & 0xF0) {
            0xC0, 0xD0 => 1,
            else => 2,
        };
    }

    /// Feeds one byte; returns a message when one completes on this byte.
    pub fn feed(self: *Parser, byte: u8) ?Message {
        if (byte >= 0xF8) return null; // realtime: legal mid-message, ignore
        if (byte >= 0xF0) { // SysEx boundaries / system common
            self.in_sysex = byte == 0xF0;
            self.status = 0; // system common cancels running status
            self.have = 0;
            return null;
        }
        if (byte >= 0x80) { // channel status: a new message begins
            self.in_sysex = false;
            self.status = byte;
            self.have = 0;
            return null;
        }
        // Data byte. Inside SysEx (or with no status to attach to) it's
        // payload we don't care about.
        if (self.in_sysex or self.status == 0) return null;
        self.data[self.have] = byte;
        self.have += 1;
        if (self.have < dataLen(self.status)) return null;
        self.have = 0; // running status: the next data byte starts a new message
        return .{
            .status = self.status,
            .data1 = self.data[0],
            .data2 = if (dataLen(self.status) == 2) self.data[1] else 0,
        };
    }
};

/// SPSC ring: the CoreMIDI thread pushes, the UI loop pops. When full the
/// newest message is dropped (and counted) — it's control data, and the
/// UI drains 33 ms later anyway.
pub const Queue = struct {
    const capacity = 256; // power of two; index arithmetic relies on it

    slots: [capacity]std.atomic.Value(u32) = [_]std.atomic.Value(u32){.init(0)} ** capacity,
    head: std.atomic.Value(usize) = .init(0), // consumer cursor
    tail: std.atomic.Value(usize) = .init(0), // producer cursor
    dropped: std.atomic.Value(usize) = .init(0),

    pub fn push(self: *Queue, msg: Message) void {
        const tail = self.tail.load(.monotonic);
        const head = self.head.load(.acquire);
        if (tail -% head == capacity) {
            _ = self.dropped.fetchAdd(1, .monotonic);
            return;
        }
        const word = @as(u32, msg.status) << 16 | @as(u32, msg.data1) << 8 | @as(u32, msg.data2);
        self.slots[tail % capacity].store(word, .monotonic);
        self.tail.store(tail +% 1, .release);
    }

    pub fn pop(self: *Queue) ?Message {
        const head = self.head.load(.monotonic);
        const tail = self.tail.load(.acquire);
        if (head == tail) return null;
        const word = self.slots[head % capacity].load(.monotonic);
        self.head.store(head +% 1, .release);
        return .{
            .status = @truncate(word >> 16),
            .data1 = @truncate(word >> 8),
            .data2 = @truncate(word),
        };
    }
};

/// Per-source parsers + one queue for an open input; the address is handed
/// to the C shim, so it must outlive start()..stop(). Sources beyond
/// max_sources share the last parser slot — a collision there degrades to
/// single-parser behavior, never to memory unsafety.
pub const Stream = struct {
    parsers: [max_sources]Parser = [_]Parser{.{}} ** max_sources,
    queue: Queue = .{},

    fn rawCallback(user: ?*anyopaque, source: c_uint, bytes: ?[*]const u8, len: c_uint) callconv(.c) void {
        const stream: *Stream = @ptrCast(@alignCast(user.?));
        const parser = &stream.parsers[@min(source, max_sources - 1)];
        const data = (bytes orelse return)[0..len];
        for (data) |byte| {
            if (parser.feed(byte)) |msg| stream.queue.push(msg);
        }
    }
};

pub const SourceInfo = struct {
    name: [name_cap]u8,

    pub fn nameSlice(self: *const SourceInfo) []const u8 {
        const end = std.mem.indexOfScalar(u8, &self.name, 0) orelse self.name.len;
        return self.name[0..end];
    }
};

pub const Midi = struct {
    handle: *NamMidi,

    /// Fails on platforms without a MIDI backend (everything but macOS)
    /// or when the MIDI server is unreachable — callers degrade to
    /// keyboard-only control.
    pub fn init() !Midi {
        const handle = nam_midi_create() orelse return error.MidiUnavailable;
        return .{ .handle = handle };
    }

    pub fn deinit(self: *Midi) void {
        nam_midi_destroy(self.handle);
        self.* = undefined;
    }

    pub fn sourceCount(self: *Midi) usize {
        const count = nam_midi_source_count(self.handle);
        return if (count > 0) @intCast(count) else 0;
    }

    /// Identity hash of the connected sources (count + unique IDs): the
    /// hot-plug rescan keys on this so an unplug+replug that lands on the
    /// same COUNT still reads as a change.
    pub fn sourcesSignature(self: *Midi) i64 {
        return nam_midi_sources_signature(self.handle);
    }

    /// Fills `out` and returns the slice of discovered sources.
    pub fn listSources(self: *Midi, out: *[max_sources]SourceInfo) []SourceInfo {
        var names: [max_sources][name_cap]u8 = undefined;
        const count = nam_midi_list_sources(self.handle, @ptrCast(&names), name_cap, max_sources);
        if (count <= 0) return out[0..0];
        const n = @min(@as(usize, @intCast(count)), max_sources);
        for (0..n) |i| out[i] = .{ .name = names[i] };
        return out[0..n];
    }

    /// Connects `source` (an enumeration index, or null for every source);
    /// returns how many sources are connected. Safe to call again for a
    /// hot-plug rescan — the previous port is torn down first.
    pub fn start(self: *Midi, source: ?usize, stream: *Stream) !usize {
        // Connection indices are positional and get renumbered on every
        // reconnect: parser slot 0 may now be a different device, so a
        // parser left mid-SysEx (or mid-message) by an unplugged device
        // must not swallow — or worse, complete — the new occupant's
        // bytes. Stop first so the reset doesn't race normal delivery.
        nam_midi_stop(self.handle);
        stream.parsers = [_]Parser{.{}} ** max_sources;
        const rc = nam_midi_start(
            self.handle,
            if (source) |s| @intCast(s) else -1,
            Stream.rawCallback,
            stream,
        );
        if (rc < 0) return switch (rc) {
            -3 => error.SourceIndexOutOfRange,
            else => error.MidiStart,
        };
        return @intCast(rc);
    }

    pub fn stop(self: *Midi) void {
        nam_midi_stop(self.handle);
    }
};

/// CC assignments for the live controls (override with --midi-map). The
/// defaults follow GM conventions where one exists: CC 7 volume = output
/// gain, CC 11 expression = input trim (drive), CC 1 mod wheel = gate
/// threshold, CC 64 sustain = bypass; gate/normalize/mute take free
/// general-purpose numbers. Program change always selects the profile
/// slot (PC 0 = slot 1).
pub const CcMap = struct {
    out_gain: u8 = 7,
    in_gain: u8 = 11,
    gate_threshold: u8 = 1,
    bypass: u8 = 64,
    gate: u8 = 80,
    normalize: u8 = 81,
    // 85, not 82: the parser rejects default collisions, and 82 is this
    // module's own documented bypass-remap example.
    mute: u8 = 85,

    /// Parses "name=cc,name=cc,..." over the field names above with '-'
    /// for '_' (e.g. "out-gain=20,bypass=82"); unmentioned controls keep
    /// their defaults. Two controls landing on one CC number — including
    /// a remap colliding with the DEFAULT of an unmentioned control — are
    /// rejected: the dispatch is first-match-wins, so the shadowed control
    /// would silently stop responding.
    pub fn parse(spec: []const u8) !CcMap {
        var map = CcMap{};
        var it = std.mem.splitScalar(u8, spec, ',');
        while (it.next()) |entry| {
            if (entry.len == 0) continue;
            const eq = std.mem.indexOfScalar(u8, entry, '=') orelse return error.InvalidMidiMap;
            const name = entry[0..eq];
            const cc = std.fmt.parseInt(u8, entry[eq + 1 ..], 10) catch return error.InvalidMidiMap;
            if (cc > 127) return error.InvalidMidiMap;
            var matched = false;
            inline for (@typeInfo(CcMap).@"struct".fields) |field| {
                var field_name: [32]u8 = undefined;
                const dashed = field_name[0..field.name.len];
                _ = std.mem.replace(u8, field.name, "_", "-", dashed);
                if (std.mem.eql(u8, name, dashed)) {
                    @field(map, field.name) = cc;
                    matched = true;
                }
            }
            if (!matched) return error.InvalidMidiMap;
        }
        const fields = @typeInfo(CcMap).@"struct".fields;
        inline for (fields, 0..) |a, i| {
            inline for (fields[0..i]) |b| {
                if (@field(map, a.name) == @field(map, b.name)) return error.MidiMapCollision;
            }
        }
        return map;
    }
};

/// CC value 0..127 -> dB, linear in dB (matches how the keyboard keys
/// step the same controls).
pub fn ccToDb(value: u8, min_db: f32, max_db: f32) f32 {
    const t = @as(f32, @floatFromInt(@min(value, 127))) / 127.0;
    return min_db + (max_db - min_db) * t;
}

/// Switch-style CC semantics (sustain-pedal convention): >= 64 is on.
pub fn ccOn(value: u8) bool {
    return value >= 64;
}

test "queue: fifo order and overflow drops the newest" {
    var queue = Queue{};
    for (0..Queue.capacity) |i| {
        queue.push(.{ .status = 0xB0, .data1 = @truncate(i), .data2 = @truncate(i >> 7) });
    }
    queue.push(.{ .status = 0xB0, .data1 = 1, .data2 = 1 }); // full: dropped
    try std.testing.expectEqual(@as(usize, 1), queue.dropped.load(.monotonic));
    for (0..Queue.capacity) |i| {
        const msg = queue.pop().?;
        try std.testing.expectEqual(@as(u8, @truncate(i)), msg.data1);
    }
    try std.testing.expectEqual(@as(?Message, null), queue.pop());
}

test "stream: multi-packet sysex from one source can't corrupt another's parse" {
    var stream = Stream{};
    // Source 0 starts a SysEx whose payload continues in later packets.
    const sysex_head = [_]u8{ 0xF0, 0x41, 0x10 };
    Stream.rawCallback(&stream, 0, &sysex_head, sysex_head.len);
    // Source 1 interleaves a complete CC message.
    const cc = [_]u8{ 0xB0, 7, 99 };
    Stream.rawCallback(&stream, 1, &cc, cc.len);
    // Source 0's SysEx continuation + terminator: data bytes that a shared
    // parser would have attached to source 1's running status, fabricating
    // CC messages from SysEx payload.
    const sysex_tail = [_]u8{ 0x20, 0x30, 0x40, 0x50, 0xF7 };
    Stream.rawCallback(&stream, 0, &sysex_tail, sysex_tail.len);

    const msg = stream.queue.pop().?;
    try std.testing.expectEqual(@as(u8, 0xB0), msg.status);
    try std.testing.expectEqual(@as(u8, 7), msg.data1);
    try std.testing.expectEqual(@as(u8, 99), msg.data2);
    try std.testing.expectEqual(@as(?Message, null), stream.queue.pop());
}

test {
    _ = @import("midi_tests.zig");
}
