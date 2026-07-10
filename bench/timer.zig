const std = @import("std");

pub const Timer = struct {
    io: std.Io,
    start_ns: i96,

    pub fn start(io: std.Io) !Timer {
        return .{
            .io = io,
            .start_ns = nowNs(io),
        };
    }

    pub fn reset(self: *Timer) void {
        self.start_ns = nowNs(self.io);
    }

    pub fn read(self: *const Timer) u64 {
        return @intCast(nowNs(self.io) - self.start_ns);
    }
};

fn nowNs(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).nanoseconds;
}
