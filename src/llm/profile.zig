//! Wall-clock pair for the per-forward profile structs each family
//! defines: `start` reads the clock only when profiling is armed (non-null
//! profile pointer), so unprofiled steps never touch it; `elapsed` is the
//! matching delta. `io` may be null exactly when profiling is off (the
//! profiled entries thread a real `std.Io`).
const std = @import("std");

pub fn start(profile: anytype, io: ?std.Io) i128 {
    return if (profile != null) std.Io.Clock.awake.now(io.?).nanoseconds else 0;
}

pub fn elapsed(start_ns: i128, io: ?std.Io) i128 {
    return std.Io.Clock.awake.now(io.?).nanoseconds - start_ns;
}
