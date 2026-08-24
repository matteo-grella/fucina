//! Opt-in stderr diagnostics for passing tests. The zig build runner renders
//! ANY stderr from a passing test step as a warning block ending in
//! "failed command: ...", so per-case replay margins print only when
//! FUCINA_TEST_VERBOSE is set (any value). Failure-path prints stay on
//! std.debug.print directly: when the step fails the block is real.

const std = @import("std");

pub fn print(comptime fmt: []const u8, args: anytype) void {
    if (std.testing.environ.getPosix("FUCINA_TEST_VERBOSE") == null) return;
    std.debug.print(fmt, args);
}
