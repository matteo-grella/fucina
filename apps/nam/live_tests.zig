//! Behavioral tests for the live processing module (`live.zig`): the cab
//! advisory classification (`classifyGear` gear-type mapping) and the
//! bi-directional redundant/missing-cab advisory (`cabAdvice`).
const std = @import("std");
const live = @import("live.zig");

const GearClass = live.GearClass;
const ChainStage = live.ChainStage;
const classifyGear = live.classifyGear;
const cabAdvice = live.cabAdvice;

test "classifyGear: gear_type mapping + full-rig hint" {
    try std.testing.expectEqual(GearClass.has_cab, classifyGear("amp_cab", false));
    try std.testing.expectEqual(GearClass.has_cab, classifyGear("amp_pedal_cab", false));
    try std.testing.expectEqual(GearClass.has_cab, classifyGear("studio", false));
    try std.testing.expectEqual(GearClass.has_cab, classifyGear("full-rig", false));
    try std.testing.expectEqual(GearClass.has_cab, classifyGear("AMP_CAB", false)); // case-insensitive
    try std.testing.expectEqual(GearClass.needs_cab, classifyGear("amp", false));
    try std.testing.expectEqual(GearClass.needs_cab, classifyGear("preamp", false));
    try std.testing.expectEqual(GearClass.needs_cab, classifyGear("pedal_amp", false));
    try std.testing.expectEqual(GearClass.neutral, classifyGear("pedal", false));
    try std.testing.expectEqual(GearClass.neutral, classifyGear(null, false));
    try std.testing.expectEqual(GearClass.neutral, classifyGear("banana", false));
    try std.testing.expectEqual(GearClass.has_cab, classifyGear("amp", true)); // full-rig hint overrides
    try std.testing.expectEqual(GearClass.has_cab, classifyGear(null, true));
}

test "cabAdvice: redundant + missing cab, both directions" {
    const M = struct {
        fn nam(g: GearClass) ChainStage {
            return .{ .stage = .{ .nam = .{ .engine = undefined, .gear = g } } };
        }
        fn cab() ChainStage {
            return .{ .stage = .{ .cab = undefined } }; // tag-only; cabAdvice never derefs
        }
    };
    {
        var s = [_]ChainStage{ M.nam(.has_cab), M.cab() }; // cab after baked-in cab
        const a = cabAdvice(&s);
        try std.testing.expect(a.redundant_cab and !a.missing_cab);
    }
    {
        var s = [_]ChainStage{M.nam(.needs_cab)}; // amp, no cab
        const a = cabAdvice(&s);
        try std.testing.expect(a.missing_cab and !a.redundant_cab);
    }
    {
        var s = [_]ChainStage{ M.nam(.needs_cab), M.cab() }; // correct amp -> cab
        const a = cabAdvice(&s);
        try std.testing.expect(!a.missing_cab and !a.redundant_cab);
    }
    {
        var s = [_]ChainStage{M.nam(.has_cab)}; // baked-in cab, no separate cab
        const a = cabAdvice(&s);
        try std.testing.expect(!a.missing_cab and !a.redundant_cab);
    }
    {
        var s = [_]ChainStage{ M.cab(), M.nam(.needs_cab) }; // cab before amp -> still "missing"
        const a = cabAdvice(&s);
        try std.testing.expect(a.missing_cab and !a.redundant_cab);
    }
    {
        // Multi-amp: the cab serves the trailing needs_cab amp, not the earlier
        // has_cab one -> neither redundant nor missing.
        var s = [_]ChainStage{ M.nam(.has_cab), M.nam(.needs_cab), M.cab() };
        const a = cabAdvice(&s);
        try std.testing.expect(!a.redundant_cab and !a.missing_cab);
    }
    {
        var s = [_]ChainStage{M.nam(.neutral)}; // pedal-only
        const a = cabAdvice(&s);
        try std.testing.expect(!a.missing_cab and !a.redundant_cab);
    }
}
