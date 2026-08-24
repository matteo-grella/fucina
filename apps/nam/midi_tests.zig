//! Behavioral tests for the MIDI module (`midi.zig`): the MIDI 1.0
//! byte-stream parser (running status, realtime interleave, SysEx skip),
//! the per-source stream isolation, the CC-map parser, and the CC->dB /
//! switch-CC value helpers.
const std = @import("std");
const midi = @import("midi.zig");

const Parser = midi.Parser;
const Message = midi.Message;
const CcMap = midi.CcMap;
const program_change = midi.program_change;
const ccToDb = midi.ccToDb;
const ccOn = midi.ccOn;

test "parser: complete CC and PC messages" {
    var parser = Parser{};
    try std.testing.expectEqual(@as(?Message, null), parser.feed(0xB0));
    try std.testing.expectEqual(@as(?Message, null), parser.feed(7));
    const cc = parser.feed(100).?;
    try std.testing.expectEqual(@as(u8, 0xB0), cc.status);
    try std.testing.expectEqual(@as(u8, 7), cc.data1);
    try std.testing.expectEqual(@as(u8, 100), cc.data2);

    try std.testing.expectEqual(@as(?Message, null), parser.feed(0xC2));
    const pc = parser.feed(5).?;
    try std.testing.expectEqual(@as(u8, 0xC2), pc.status);
    try std.testing.expectEqual(@as(u8, 5), pc.data1);
    try std.testing.expectEqual(@as(u8, 0), pc.data2);
    try std.testing.expectEqual(@as(u8, 2), pc.channel());
    try std.testing.expectEqual(program_change, pc.kind());
}

test "parser: running status emits successive messages" {
    var parser = Parser{};
    _ = parser.feed(0xB0);
    _ = parser.feed(7);
    try std.testing.expect(parser.feed(10) != null);
    // No new status byte: same controller, next value pair.
    _ = parser.feed(7);
    const again = parser.feed(20).?;
    try std.testing.expectEqual(@as(u8, 0xB0), again.status);
    try std.testing.expectEqual(@as(u8, 20), again.data2);
}

test "parser: realtime bytes interleave mid-message without corruption" {
    var parser = Parser{};
    _ = parser.feed(0xB0);
    try std.testing.expectEqual(@as(?Message, null), parser.feed(0xF8)); // clock between bytes
    _ = parser.feed(7);
    try std.testing.expectEqual(@as(?Message, null), parser.feed(0xFE)); // active sensing
    const msg = parser.feed(99).?;
    try std.testing.expectEqual(@as(u8, 99), msg.data2);
}

test "parser: sysex payload is skipped, next status resumes normally" {
    var parser = Parser{};
    _ = parser.feed(0xF0);
    for ([_]u8{ 0x10, 0x20, 0x30 }) |b| try std.testing.expectEqual(@as(?Message, null), parser.feed(b));
    try std.testing.expectEqual(@as(?Message, null), parser.feed(0xF7));
    // Data bytes with no status (e.g. a truncated stream) are ignored too.
    try std.testing.expectEqual(@as(?Message, null), parser.feed(0x40));
    _ = parser.feed(0x91);
    _ = parser.feed(60);
    const note = parser.feed(100).?;
    try std.testing.expectEqual(@as(u8, 0x91), note.status);
}

test "cc map: parse overrides named controls, rejects junk and collisions" {
    const map = try CcMap.parse("out-gain=20,gate-threshold=2,bypass=82");
    try std.testing.expectEqual(@as(u8, 20), map.out_gain);
    try std.testing.expectEqual(@as(u8, 2), map.gate_threshold);
    try std.testing.expectEqual(@as(u8, 82), map.bypass);
    try std.testing.expectEqual(@as(u8, 11), map.in_gain); // default kept
    try std.testing.expectError(error.InvalidMidiMap, CcMap.parse("volume=7"));
    try std.testing.expectError(error.InvalidMidiMap, CcMap.parse("out-gain=200"));
    try std.testing.expectError(error.InvalidMidiMap, CcMap.parse("out-gain"));
    // Collisions are first-match-wins at dispatch, so reject them up
    // front — including against the default of an UNMENTIONED control
    // (bypass=1 collides with gate-threshold's default CC 1).
    try std.testing.expectError(error.MidiMapCollision, CcMap.parse("bypass=1"));
    try std.testing.expectError(error.MidiMapCollision, CcMap.parse("out-gain=5,in-gain=5"));
    _ = try CcMap.parse("bypass=1,gate-threshold=64"); // explicit swap is fine
}

test "ccToDb: endpoints and midpoint" {
    try std.testing.expectEqual(@as(f32, -40.0), ccToDb(0, -40.0, 24.0));
    try std.testing.expectEqual(@as(f32, 24.0), ccToDb(127, -40.0, 24.0));
    const mid = ccToDb(64, -90.0, -30.0);
    try std.testing.expect(mid > -61.0 and mid < -59.0);
    try std.testing.expect(ccOn(64) and ccOn(127) and !ccOn(63) and !ccOn(0));
}
