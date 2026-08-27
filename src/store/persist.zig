//! The streamed tier's persistence formats: the FUCEXPT1 usage-histogram
//! sidecar (`saveUsage`/`loadUsage` -- the learning cache's memory) and
//! the FUCTRCE1 routing trace (`saveTrace`, the input of
//! `tools/replay_experts.zig`). Both write atomically (tmp + rename).
//! Store-parameterized functions take the owning `ExpertStore` as
//! `anytype` (see tiers.zig for the cycle rationale).
const std = @import("std");
const io = @import("io.zig");

const Allocator = std.mem.Allocator;
const Error = io.Error;

pub const usage_magic = "FUCEXPT1";
pub const trace_magic = "FUCTRCE1";

/// Write the routing trace (`Options.trace_path`): magic, layer count,
/// then little-endian u32 (layer, eid) pairs in request order. The
/// input format of `tools/replay_experts.zig`. Called from `destroy`;
/// also callable early to snapshot mid-session.
pub fn saveTrace(self: anytype) Error!void {
    const path = self.options.trace_path orelse return;
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(self.allocator);
    try bytes.appendSlice(self.allocator, trace_magic);
    try appendInt(u32, self.allocator, &bytes, @intCast(self.layers.len));
    try appendInt(u64, self.allocator, &bytes, @intCast(self.trace.items.len / 2));
    for (self.trace.items) |v| try appendInt(u32, self.allocator, &bytes, v);

    const tmp_path = try std.fmt.allocPrint(self.allocator, "{s}.tmp", .{path});
    defer self.allocator.free(tmp_path);
    {
        const fd = try io.openWriteTrunc(self.allocator, tmp_path);
        defer io.closeFd(fd);
        try io.writeFull(fd, bytes.items);
    }
    try io.renamePath(self.allocator, tmp_path, path);
}

/// Persist the usage histogram to the sidecar (`<gguf>.experts`),
/// atomically (tmp + rename): counts accumulate across sessions, and at
/// the next startup auto-pin turns them into a pinned hot tier. Call at
/// end of generation / turn boundaries; a failure loses only learning.
pub fn saveUsage(self: anytype) Error!void {
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(self.allocator);
    try bytes.appendSlice(self.allocator, usage_magic);
    try appendInt(u32, self.allocator, &bytes, @intCast(self.layers.len));
    for (self.layers, self.registered, 0..) |*ls, reg, layer_i| {
        if (!reg) continue;
        try appendInt(u32, self.allocator, &bytes, @intCast(layer_i));
        try appendInt(u32, self.allocator, &bytes, @intCast(ls.n_expert));
        for (ls.usage) |count| try appendInt(u64, self.allocator, &bytes, count);
    }

    const tmp_path = try std.fmt.allocPrint(self.allocator, "{s}.tmp", .{self.usage_path});
    defer self.allocator.free(tmp_path);
    {
        const fd = try io.openWriteTrunc(self.allocator, tmp_path);
        defer io.closeFd(fd);
        try io.writeFull(fd, bytes.items);
    }
    try io.renamePath(self.allocator, tmp_path, self.usage_path);
}

/// Merge the sidecar histogram into the in-memory counts (add, so a
/// session's own routing keeps accumulating on top). Any mismatch —
/// missing file, other model's geometry, torn write — ignores the file
/// wholesale. Returns the total merged pair count.
pub fn loadUsage(self: anytype) u64 {
    const bytes = io.readWholeFile(self.allocator, self.usage_path) orelse return 0;
    defer self.allocator.free(bytes);
    var r = UsageReader{ .bytes = bytes };
    if (!std.mem.eql(u8, r.take(usage_magic.len) orelse return 0, usage_magic)) return 0;
    const n_layers = r.int(u32) orelse return 0;
    if (n_layers != self.layers.len) return 0;

    // Validate the whole record set against the registered geometry
    // before merging anything: an invalid tail must not half-apply.
    var check = r;
    for (self.layers, self.registered, 0..) |*ls, reg, layer_i| {
        if (!reg) continue;
        if ((check.int(u32) orelse return 0) != layer_i) return 0;
        if ((check.int(u32) orelse return 0) != ls.n_expert) return 0;
        if (check.take(ls.n_expert * @sizeOf(u64)) == null) return 0;
    }
    if (check.bytes.len != check.at) return 0;

    var total: u64 = 0;
    for (self.layers, self.registered) |*ls, reg| {
        if (!reg) continue;
        _ = r.int(u32);
        _ = r.int(u32);
        for (ls.usage) |*count| {
            const stored = r.int(u64) orelse unreachable;
            count.* +|= stored;
            total +|= stored;
        }
    }
    return total;
}

const UsageReader = struct {
    bytes: []const u8,
    at: usize = 0,

    fn take(self: *UsageReader, n: usize) ?[]const u8 {
        if (self.at + n > self.bytes.len) return null;
        defer self.at += n;
        return self.bytes[self.at..][0..n];
    }

    fn int(self: *UsageReader, comptime T: type) ?T {
        const raw = self.take(@sizeOf(T)) orelse return null;
        return std.mem.readInt(T, raw[0..@sizeOf(T)], .little);
    }
};

fn appendInt(comptime T: type, allocator: Allocator, bytes: *std.ArrayList(u8), value: T) Error!void {
    var raw: [@sizeOf(T)]u8 = undefined;
    std.mem.writeInt(T, &raw, value, .little);
    try bytes.appendSlice(allocator, &raw);
}
