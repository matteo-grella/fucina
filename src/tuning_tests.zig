//! Contract tests for the tuning gates. Env vars are process-global state,
//! so these only exercise the default and `set` arms (the env arm is the
//! same read path, covered operationally by every FUCINA_* A/B switch).

const std = @import("std");
const tuning = @import("tuning.zig");

test "Switch: default answer, pin, and re-arm" {
    const gate = tuning.Switch(.{
        .on = "FUCINA_TEST_TUNING_ON",
        .off = "FUCINA_TEST_TUNING_OFF",
        .default = true,
    });
    try std.testing.expect(gate.enabled()); // default (env unset)
    gate.set(false);
    try std.testing.expect(!gate.enabled());
    gate.set(true);
    try std.testing.expect(gate.enabled());
    gate.set(null); // re-arm: env unset, back to default
    try std.testing.expect(gate.enabled());
}

test "Switch: distinct instantiations own distinct state" {
    const a = tuning.Switch(.{ .on = "FUCINA_TEST_TUNING_A", .default = false });
    const b = tuning.Switch(.{ .on = "FUCINA_TEST_TUNING_B", .default = true });
    a.set(true);
    b.set(false);
    try std.testing.expect(a.enabled());
    try std.testing.expect(!b.enabled());
    a.set(null);
    b.set(null);
    try std.testing.expect(!a.enabled());
    try std.testing.expect(b.enabled());
}

test "Threshold: default, override, and reset" {
    const t = tuning.Threshold("FUCINA_TEST_TUNING_MIN", 32);
    try std.testing.expectEqual(@as(u64, 32), t.get()); // default (env unset)
    t.set(4);
    try std.testing.expectEqual(@as(u64, 4), t.get());
    t.set(null);
    try std.testing.expectEqual(@as(u64, 32), t.get());
}
