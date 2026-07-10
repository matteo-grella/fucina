const std = @import("std");

pub const AllocatorMode = enum {
    debug,
    smp,
};

pub fn parseAllocatorModeArg(arg: []const u8) !?AllocatorMode {
    if (std.mem.eql(u8, arg, "--prod-allocator")) return .smp;
    if (!std.mem.startsWith(u8, arg, "--allocator=")) return null;

    const value = arg["--allocator=".len..];
    if (std.mem.eql(u8, value, "debug")) return .debug;
    if (std.mem.eql(u8, value, "smp")) return .smp;
    return error.UnknownAllocatorMode;
}

pub fn parseAllocatorMode(args: []const []const u8) !AllocatorMode {
    var mode: AllocatorMode = .debug;
    for (args[1..]) |arg| {
        if (try parseAllocatorModeArg(arg)) |parsed| {
            mode = parsed;
        } else {
            return error.UnknownArgument;
        }
    }
    return mode;
}

pub const BenchmarkAllocator = struct {
    mode: AllocatorMode,
    debug: std.heap.DebugAllocator(.{}) = .{},

    pub fn init(mode: AllocatorMode) BenchmarkAllocator {
        return .{ .mode = mode };
    }

    pub fn allocator(self: *BenchmarkAllocator) std.mem.Allocator {
        return switch (self.mode) {
            .debug => self.debug.allocator(),
            .smp => std.heap.smp_allocator,
        };
    }

    pub fn deinit(self: *BenchmarkAllocator) void {
        if (self.mode == .debug and self.debug.deinit() != .ok) {
            @panic("benchmark leaked memory");
        }
        self.* = undefined;
    }
};

pub const CountingAllocator = struct {
    child: std.mem.Allocator,
    alloc_count: usize = 0,
    free_count: usize = 0,
    bytes_allocated: usize = 0,
    live_bytes: usize = 0,
    peak_live: usize = 0,

    pub fn init(child: std.mem.Allocator) CountingAllocator {
        return .{ .child = child };
    }

    pub fn allocator(self: *CountingAllocator) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .remap = remap,
                .free = free,
            },
        };
    }

    pub fn resetWindow(self: *CountingAllocator) void {
        self.alloc_count = 0;
        self.free_count = 0;
        self.bytes_allocated = 0;
        self.peak_live = self.live_bytes;
    }

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        const ptr = self.child.rawAlloc(len, alignment, ret_addr) orelse return null;
        self.alloc_count += 1;
        self.bytes_allocated += len;
        self.live_bytes += len;
        self.peak_live = @max(self.peak_live, self.live_bytes);
        return ptr;
    }

    fn resize(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        if (!self.child.rawResize(buf, alignment, new_len, ret_addr)) return false;

        if (new_len > buf.len) {
            const delta = new_len - buf.len;
            self.bytes_allocated += delta;
            self.live_bytes += delta;
        } else {
            self.live_bytes -= buf.len - new_len;
        }
        self.peak_live = @max(self.peak_live, self.live_bytes);
        return true;
    }

    fn remap(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        const ptr = self.child.rawRemap(buf, alignment, new_len, ret_addr) orelse return null;
        if (new_len > buf.len) {
            const delta = new_len - buf.len;
            self.bytes_allocated += delta;
            self.live_bytes += delta;
        } else {
            self.live_bytes -= buf.len - new_len;
        }
        self.peak_live = @max(self.peak_live, self.live_bytes);
        return ptr;
    }

    fn free(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        self.free_count += 1;
        self.live_bytes -= buf.len;
        self.child.rawFree(buf, alignment, ret_addr);
    }
};
