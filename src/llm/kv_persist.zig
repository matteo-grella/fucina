//! Crash-safe KV-cache persistence: conversations reopen WARM across
//! process restarts, with zero re-prefill — the cost that dominates when a
//! big model decodes below 1 tok/s. The sidecar is append-only: a fixed
//! header whose record count (`nrec`) is rewritten LAST after every append,
//! so a crash mid-append leaves the old count and the file stays a
//! consistent prefix of the conversation.
//!
//! Layout (little endian):
//!   "FUXKV001"                     magic, 8 bytes
//!   nrec: u64                      valid record count (rewritten last)
//!   n_layers: u32, dtype: u32      geometry guard — any mismatch with the
//!   per layer: kv_heads: u32,      opening cache ignores the file wholesale
//!              head_dim: u32
//!   records[nrec], each:
//!     token: u32
//!     per layer: K row bytes, V row bytes for one position
//!       (f16: kv_heads*head_dim*2; q8_0: kv_heads*head_dim/32 * 34)
const std = @import("std");
const fucina = @import("fucina");
const kv_cache = @import("kv_cache.zig");

const Allocator = std.mem.Allocator;
const KvCache = kv_cache.KvCache;

const magic = "FUXKV001";
const nrec_offset: u64 = magic.len;

pub const Error = error{KvPersistTokenMismatch};

fn headerLen(n_layers: usize) u64 {
    return magic.len + 8 + 4 + 4 + @as(u64, n_layers) * 8;
}

fn layerRowBytes(kv: *const KvCache, layer_i: usize) usize {
    const elems = kv.kv_heads[layer_i] * kv.head_dim[layer_i];
    return switch (kv.dtype) {
        .f16 => elems * @sizeOf(f16),
        .q8_0 => (elems / fucina.q8_0_block_size) * @sizeOf(fucina.BlockQ8_0),
    };
}

fn recordLen(kv: *const KvCache) usize {
    var total: usize = 4;
    for (0..kv.kv_heads.len) |i| total += 2 * layerRowBytes(kv, i);
    return total;
}

fn dtypeTag(dtype: kv_cache.KvDtype) u32 {
    return switch (dtype) {
        .f16 => 0,
        .q8_0 => 1,
    };
}

fn buildHeader(allocator: Allocator, kv: *const KvCache, nrec: u64) Allocator.Error![]u8 {
    const n_layers = kv.kv_heads.len;
    const bytes = try allocator.alloc(u8, headerLen(n_layers));
    var at: usize = 0;
    @memcpy(bytes[at..][0..magic.len], magic);
    at += magic.len;
    std.mem.writeInt(u64, bytes[at..][0..8], nrec, .little);
    at += 8;
    std.mem.writeInt(u32, bytes[at..][0..4], @intCast(n_layers), .little);
    at += 4;
    std.mem.writeInt(u32, bytes[at..][0..4], dtypeTag(kv.dtype), .little);
    at += 4;
    for (0..n_layers) |i| {
        std.mem.writeInt(u32, bytes[at..][0..4], @intCast(kv.kv_heads[i]), .little);
        at += 4;
        std.mem.writeInt(u32, bytes[at..][0..4], @intCast(kv.head_dim[i]), .little);
        at += 4;
    }
    std.debug.assert(at == bytes.len);
    return bytes;
}

/// Serialize position `pos` into `rec` (caller-sized to `recordLen`).
fn buildRecord(kv: *const KvCache, token: usize, pos: usize, rec: []u8) !void {
    var at: usize = 0;
    std.mem.writeInt(u32, rec[at..][0..4], @intCast(token), .little);
    at += 4;
    for (0..kv.kv_heads.len) |layer_i| {
        const row = layerRowBytes(kv, layer_i);
        switch (kv.dtype) {
            .f16 => {
                const elems = kv.kv_heads[layer_i] * kv.head_dim[layer_i];
                const k = std.mem.sliceAsBytes((try kv.k[layer_i].dataConst())[pos * elems ..][0..elems]);
                const v = std.mem.sliceAsBytes((try kv.v[layer_i].dataConst())[pos * elems ..][0..elems]);
                @memcpy(rec[at..][0..row], k);
                at += row;
                @memcpy(rec[at..][0..row], v);
                at += row;
            },
            .q8_0 => {
                const blocks = (kv.kv_heads[layer_i] * kv.head_dim[layer_i]) / fucina.q8_0_block_size;
                const k = std.mem.sliceAsBytes(kv.k_q8[layer_i][pos * blocks ..][0..blocks]);
                const v = std.mem.sliceAsBytes(kv.v_q8[layer_i][pos * blocks ..][0..blocks]);
                @memcpy(rec[at..][0..row], k);
                at += row;
                @memcpy(rec[at..][0..row], v);
                at += row;
            },
        }
    }
    std.debug.assert(at == rec.len);
}

/// Scatter one serialized record into cache position `pos`; returns its token.
fn applyRecord(kv: *KvCache, pos: usize, rec: []const u8) !usize {
    var at: usize = 0;
    const token = std.mem.readInt(u32, rec[at..][0..4], .little);
    at += 4;
    for (0..kv.kv_heads.len) |layer_i| {
        const row = layerRowBytes(kv, layer_i);
        switch (kv.dtype) {
            .f16 => {
                const elems = kv.kv_heads[layer_i] * kv.head_dim[layer_i];
                @memcpy(std.mem.sliceAsBytes((try kv.k[layer_i].data())[pos * elems ..][0..elems]), rec[at..][0..row]);
                at += row;
                @memcpy(std.mem.sliceAsBytes((try kv.v[layer_i].data())[pos * elems ..][0..elems]), rec[at..][0..row]);
                at += row;
            },
            .q8_0 => {
                const blocks = (kv.kv_heads[layer_i] * kv.head_dim[layer_i]) / fucina.q8_0_block_size;
                @memcpy(std.mem.sliceAsBytes(kv.k_q8[layer_i][pos * blocks ..][0..blocks]), rec[at..][0..row]);
                at += row;
                @memcpy(std.mem.sliceAsBytes(kv.v_q8[layer_i][pos * blocks ..][0..blocks]), rec[at..][0..row]);
                at += row;
            },
        }
    }
    return token;
}

/// Read and validate an existing sidecar's header against `kv`'s geometry.
/// Returns the stored `nrec`, or null when the file is absent or belongs to
/// another model/dtype (ignored wholesale).
fn readHeader(io: std.Io, allocator: Allocator, path: []const u8, kv: *const KvCache) ?u64 {
    var file = std.Io.Dir.cwd().openFile(io, path, .{}) catch return null;
    defer file.close(io);
    const n_layers = kv.kv_heads.len;
    const hdr = allocator.alloc(u8, headerLen(n_layers)) catch return null;
    defer allocator.free(hdr);
    const got = file.readPositionalAll(io, hdr, 0) catch return null;
    if (got != hdr.len) return null;

    var expect = buildHeader(allocator, kv, 0) catch return null;
    defer allocator.free(expect);
    // Compare everything except the nrec field.
    if (!std.mem.eql(u8, hdr[0..magic.len], expect[0..magic.len])) return null;
    if (!std.mem.eql(u8, hdr[magic.len + 8 ..], expect[magic.len + 8 ..])) return null;
    return std.mem.readInt(u64, hdr[nrec_offset..][0..8], .little);
}

/// Reset the sidecar to an empty conversation for this cache's geometry:
/// a fresh header with `nrec = 0` caps whatever records follow. Use when
/// arming persistence over a file that could not be resumed — appending
/// onto a foreign prefix must be impossible.
pub fn reset(io: std.Io, allocator: Allocator, path: []const u8, kv: *const KvCache) !void {
    const hdr = try buildHeader(allocator, kv, 0);
    defer allocator.free(hdr);
    var file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    try file.writePositionalAll(io, hdr, 0);
}

/// Append the cache positions the sidecar does not hold yet (from its own
/// record count up to `kv.len`, with `tokens` indexed by absolute
/// position), creating the file if needed, then publish the new count by
/// rewriting `nrec` last — data first, counter after, so a torn append is
/// invisible. Call at turn/generation boundaries; the caller must have
/// resumed from (or `reset`) this file, so the existing prefix is this
/// conversation's own.
pub fn appendRange(io: std.Io, allocator: Allocator, path: []const u8, kv: *const KvCache, tokens: []const usize) !void {
    if (tokens.len != kv.len) return Error.KvPersistTokenMismatch;

    const disk_nrec: u64 = readHeader(io, allocator, path, kv) orelse blk: {
        try reset(io, allocator, path, kv);
        break :blk 0;
    };
    // A file ahead of the session (a rolled-back turn) is truncated to the
    // session's truth by the final nrec rewrite.
    const start = @min(@as(usize, @intCast(disk_nrec)), kv.len);
    if (kv.len == start and kv.len == disk_nrec) return;

    var file = try std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_write });
    defer file.close(io);

    const rec_len = recordLen(kv);
    const rec = try allocator.alloc(u8, rec_len);
    defer allocator.free(rec);
    const header_len = headerLen(kv.kv_heads.len);
    for (start..kv.len) |pos| {
        try buildRecord(kv, tokens[pos], pos, rec);
        try file.writePositionalAll(io, rec, header_len + @as(u64, pos) * rec_len);
    }
    var nrec_bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &nrec_bytes, kv.len, .little);
    try file.writePositionalAll(io, &nrec_bytes, nrec_offset);
}

/// Resume a persisted conversation into the (empty) cache: validates the
/// header, loads up to `nrec` records (stopping early at a torn tail — the
/// prefix stays usable), sets `kv.len`, and returns the caller-owned token
/// history. Null when there is nothing usable to resume (absent file,
/// foreign geometry, or a history larger than the cache capacity).
pub fn load(io: std.Io, allocator: Allocator, path: []const u8, kv: *KvCache) !?[]usize {
    std.debug.assert(kv.len == 0);
    const nrec_stored = readHeader(io, allocator, path, kv) orelse return null;
    const nrec: usize = std.math.cast(usize, nrec_stored) orelse return null;
    if (nrec == 0 or nrec > kv.capacity) return null;

    var file = std.Io.Dir.cwd().openFile(io, path, .{}) catch return null;
    defer file.close(io);

    const rec_len = recordLen(kv);
    const rec = try allocator.alloc(u8, rec_len);
    defer allocator.free(rec);
    const tokens = try allocator.alloc(usize, nrec);
    errdefer allocator.free(tokens);

    const header_len = headerLen(kv.kv_heads.len);
    var loaded: usize = 0;
    while (loaded < nrec) : (loaded += 1) {
        const got = file.readPositionalAll(io, rec, header_len + @as(u64, loaded) * rec_len) catch break;
        if (got != rec_len) break;
        tokens[loaded] = try applyRecord(kv, loaded, rec);
    }
    if (loaded == 0) {
        allocator.free(tokens);
        return null;
    }
    kv.len = loaded;
    if (loaded == nrec) return tokens;
    const trimmed = try allocator.realloc(tokens, loaded);
    return trimmed;
}

test {
    _ = @import("kv_persist_tests.zig");
}
