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
//!
//! Conversations served behind a preloaded KV prefix (a cartridge —
//! docs/CARTRIDGES.md — whose rows no tokens produced) write "FUXKV002":
//! identical except one extra header field, `prefix_rows: u64`, between
//! `nrec` and `n_layers`. Records still cover EVERY cache position (the
//! file is self-describing: a restore carries its own prefix even if the
//! server later runs a different cartridge); the first `prefix_rows`
//! records store the token sentinel 0, ignored on load. Prefix-free
//! conversations keep writing byte-identical V1 files.
const std = @import("std");
const fucina = @import("fucina");
const kv_cache = @import("kv_cache.zig");

const Allocator = std.mem.Allocator;
const KvCache = kv_cache.KvCache;

const magic_v1 = "FUXKV001";
const magic_v2 = "FUXKV002";
const nrec_offset: u64 = magic_v1.len;

pub const Error = error{KvPersistTokenMismatch};

fn headerLen(prefix_rows: usize, n_layers: usize) u64 {
    const prefix_field: u64 = if (prefix_rows > 0) 8 else 0;
    return magic_v1.len + 8 + prefix_field + 4 + 4 + @as(u64, n_layers) * 8;
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

fn buildHeader(allocator: Allocator, kv: *const KvCache, nrec: u64, prefix_rows: usize) Allocator.Error![]u8 {
    const n_layers = kv.kv_heads.len;
    const bytes = try allocator.alloc(u8, headerLen(prefix_rows, n_layers));
    var at: usize = 0;
    @memcpy(bytes[at..][0..magic_v1.len], if (prefix_rows > 0) magic_v2 else magic_v1);
    at += magic_v1.len;
    std.mem.writeInt(u64, bytes[at..][0..8], nrec, .little);
    at += 8;
    if (prefix_rows > 0) {
        std.mem.writeInt(u64, bytes[at..][0..8], prefix_rows, .little);
        at += 8;
    }
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

const Header = struct {
    nrec: u64,
    prefix_rows: usize,
    header_len: u64,
};

/// Read and validate an existing sidecar's header against `kv`'s geometry
/// (either format version). Returns the stored counts, or null when the
/// file is absent or belongs to another model/dtype (ignored wholesale).
fn readHeader(io: std.Io, allocator: Allocator, path: []const u8, kv: *const KvCache) ?Header {
    var file = std.Io.Dir.cwd().openFile(io, path, .{}) catch return null;
    defer file.close(io);

    var magic_buf: [magic_v1.len]u8 = undefined;
    if ((file.readPositionalAll(io, &magic_buf, 0) catch return null) != magic_buf.len) return null;
    const v2 = std.mem.eql(u8, &magic_buf, magic_v2);
    if (!v2 and !std.mem.eql(u8, &magic_buf, magic_v1)) return null;

    const n_layers = kv.kv_heads.len;
    const hdr_len = headerLen(@intFromBool(v2), n_layers);
    const hdr = allocator.alloc(u8, hdr_len) catch return null;
    defer allocator.free(hdr);
    const got = file.readPositionalAll(io, hdr, 0) catch return null;
    if (got != hdr.len) return null;

    const nrec = std.mem.readInt(u64, hdr[nrec_offset..][0..8], .little);
    const prefix_rows: u64 = if (v2) std.mem.readInt(u64, hdr[nrec_offset + 8 ..][0..8], .little) else 0;
    if (v2 and prefix_rows == 0) return null; // V2 demands a prefix; a zero is a foreign write
    const geometry_at: usize = nrec_offset + 8 + @as(usize, if (v2) 8 else 0);

    var expect = buildHeader(allocator, kv, 0, @intCast(prefix_rows)) catch return null;
    defer allocator.free(expect);
    // Compare the geometry section (everything past nrec/prefix_rows).
    if (!std.mem.eql(u8, hdr[geometry_at..], expect[geometry_at..])) return null;
    return .{
        .nrec = nrec,
        .prefix_rows = std.math.cast(usize, prefix_rows) orelse return null,
        .header_len = hdr_len,
    };
}

/// Reset the sidecar to an empty conversation for this cache's geometry:
/// a fresh header with `nrec = 0` caps whatever records follow. Use when
/// arming persistence over a file that could not be resumed — appending
/// onto a foreign prefix must be impossible.
pub fn reset(io: std.Io, allocator: Allocator, path: []const u8, kv: *const KvCache, prefix_rows: usize) !void {
    const hdr = try buildHeader(allocator, kv, 0, prefix_rows);
    defer allocator.free(hdr);
    var file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    try file.writePositionalAll(io, hdr, 0);
}

/// Append the cache positions the sidecar does not hold yet (from its own
/// record count up to `kv.len`), creating the file if needed, then publish
/// the new count by rewriting `nrec` last — data first, counter after, so
/// a torn append is invisible. `tokens` describes positions
/// `[prefix_rows, kv.len)`; the leading `prefix_rows` positions are a
/// token-less preloaded prefix (0 for classic conversations). Call at
/// turn/generation boundaries; the caller must have resumed from (or
/// `reset`) this file, so the existing content is this conversation's own —
/// a stored prefix shape that disagrees is treated as foreign and reset.
pub fn appendRange(io: std.Io, allocator: Allocator, path: []const u8, kv: *const KvCache, tokens: []const usize, prefix_rows: usize) !void {
    if (prefix_rows + tokens.len != kv.len) return Error.KvPersistTokenMismatch;

    const disk_nrec: u64 = blk: {
        if (readHeader(io, allocator, path, kv)) |hdr| {
            if (hdr.prefix_rows == prefix_rows) break :blk hdr.nrec;
        }
        try reset(io, allocator, path, kv, prefix_rows);
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
    const header_len = headerLen(prefix_rows, kv.kv_heads.len);
    for (start..kv.len) |pos| {
        const token: usize = if (pos < prefix_rows) 0 else tokens[pos - prefix_rows];
        try buildRecord(kv, token, pos, rec);
        try file.writePositionalAll(io, rec, header_len + @as(u64, pos) * rec_len);
    }
    var nrec_bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &nrec_bytes, kv.len, .little);
    try file.writePositionalAll(io, &nrec_bytes, nrec_offset);
}

/// A resumed conversation: the token history for cache rows
/// `[prefix_rows, len)` plus the token-less preloaded-prefix row count
/// (0 for classic conversations; `Conversation`'s `kv_prefix_rows` /
/// `WarmState.prefix_rows` on the adopt side).
pub const Loaded = struct {
    tokens: []usize,
    prefix_rows: usize,
};

/// Resume a persisted conversation into the (empty) cache: validates the
/// header, loads up to `nrec` records (stopping early at a torn tail — the
/// prefix stays usable), sets `kv.len`, and returns the caller-owned token
/// history. Null when there is nothing usable to resume (absent file,
/// foreign geometry, a history larger than the cache capacity, or a tear
/// inside the token-less prefix — a prefix without any conversation row is
/// not a resumable state).
pub fn load(io: std.Io, allocator: Allocator, path: []const u8, kv: *KvCache) !?Loaded {
    std.debug.assert(kv.len == 0);
    const hdr = readHeader(io, allocator, path, kv) orelse return null;
    const nrec: usize = std.math.cast(usize, hdr.nrec) orelse return null;
    if (nrec <= hdr.prefix_rows or nrec > kv.capacity) return null;

    var file = std.Io.Dir.cwd().openFile(io, path, .{}) catch return null;
    defer file.close(io);

    const rec_len = recordLen(kv);
    const rec = try allocator.alloc(u8, rec_len);
    defer allocator.free(rec);
    const tokens = try allocator.alloc(usize, nrec - hdr.prefix_rows);
    errdefer allocator.free(tokens);

    var loaded: usize = 0;
    while (loaded < nrec) : (loaded += 1) {
        const got = file.readPositionalAll(io, rec, hdr.header_len + @as(u64, loaded) * rec_len) catch break;
        if (got != rec_len) break;
        const token = try applyRecord(kv, loaded, rec);
        if (loaded >= hdr.prefix_rows) tokens[loaded - hdr.prefix_rows] = token;
    }
    if (loaded <= hdr.prefix_rows) {
        allocator.free(tokens);
        return null;
    }
    kv.len = loaded;
    const kept = loaded - hdr.prefix_rows;
    if (kept == tokens.len) return .{ .tokens = tokens, .prefix_rows = hdr.prefix_rows };
    const trimmed = try allocator.realloc(tokens, kept);
    return .{ .tokens = trimmed, .prefix_rows = hdr.prefix_rows };
}

test {
    _ = @import("kv_persist_tests.zig");
}
