//! Minimal PNG reader/writer for the LocateAnything example.
//!
//! Decodes non-interlaced 8/16-bit gray / gray+alpha / RGB / RGBA / palette
//! PNGs to tightly-packed RGB8, matching what the reference gets from
//! `stbi_load(path, .., 3)` on the same file: PNG decoding is lossless and
//! stb's channel reduction drops alpha without blending (gray replicates,
//! 16-bit takes the high byte), so the pixel bytes are identical by
//! construction. Interlaced (Adam7) PNGs are rejected. JPEG input is out of
//! scope for the parity port (the fixture and the benchmark scenes are PNG).

const std = @import("std");

const Allocator = std.mem.Allocator;
const flate = std.compress.flate;

pub const Error = error{
    NotPng,
    UnsupportedPng,
    CorruptPng,
} || Allocator.Error || std.Io.Reader.Error;

pub const Image = struct {
    allocator: Allocator,
    w: usize,
    h: usize,
    /// Tightly packed RGB8, row-major.
    rgb: []u8,

    pub fn deinit(self: *Image) void {
        self.allocator.free(self.rgb);
        self.* = undefined;
    }
};

const png_signature = [_]u8{ 0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n' };

pub fn decodePng(allocator: Allocator, bytes: []const u8) Error!Image {
    if (bytes.len < png_signature.len or !std.mem.eql(u8, bytes[0..8], &png_signature))
        return Error.NotPng;

    var w: usize = 0;
    var h: usize = 0;
    var bit_depth: u8 = 0;
    var color_type: u8 = 0;
    var palette: []const u8 = &.{};
    var idat: std.ArrayList(u8) = .empty;
    defer idat.deinit(allocator);

    var pos: usize = 8;
    var seen_ihdr = false;
    var seen_iend = false;
    while (pos + 8 <= bytes.len and !seen_iend) {
        const chunk_len = std.mem.readInt(u32, bytes[pos..][0..4], .big);
        const chunk_type = bytes[pos + 4 ..][0..4];
        pos += 8;
        if (pos + chunk_len + 4 > bytes.len) return Error.CorruptPng;
        const data = bytes[pos..][0..chunk_len];

        if (std.mem.eql(u8, chunk_type, "IHDR")) {
            if (chunk_len != 13) return Error.CorruptPng;
            w = std.mem.readInt(u32, data[0..4], .big);
            h = std.mem.readInt(u32, data[4..8], .big);
            bit_depth = data[8];
            color_type = data[9];
            const interlace = data[12];
            if (w == 0 or h == 0) return Error.CorruptPng;
            // Cap dimensions before any size arithmetic: unchecked u32 dims
            // overflow `h * (1 + stride)` below (2^67 for 4G x 4G RGBA16),
            // which is a panic in safe builds and wrapped-allocation OOB
            // writes in ReleaseFast. 2^26 per axis / 2^28 pixels is far above
            // anything the 512-patch model limit can consume.
            if (w > (1 << 26) or h > (1 << 26) or w * h > (1 << 28)) return Error.UnsupportedPng;
            if (interlace != 0) return Error.UnsupportedPng; // Adam7
            if (bit_depth != 8 and bit_depth != 16) return Error.UnsupportedPng;
            if (color_type == 3 and bit_depth != 8) return Error.UnsupportedPng;
            seen_ihdr = true;
        } else if (std.mem.eql(u8, chunk_type, "PLTE")) {
            if (chunk_len % 3 != 0) return Error.CorruptPng;
            palette = data;
        } else if (std.mem.eql(u8, chunk_type, "IDAT")) {
            try idat.appendSlice(allocator, data);
        } else if (std.mem.eql(u8, chunk_type, "IEND")) {
            seen_iend = true;
        }
        // tRNS is intentionally ignored: output is RGB (alpha dropped).
        pos += chunk_len + 4; // skip CRC
    }
    if (!seen_ihdr) return Error.CorruptPng;

    const channels: usize = switch (color_type) {
        0 => 1, // gray
        2 => 3, // rgb
        3 => 1, // palette index
        4 => 2, // gray + alpha
        6 => 4, // rgba
        else => return Error.UnsupportedPng,
    };
    if (color_type == 3 and palette.len == 0) return Error.CorruptPng;

    const bytes_per_sample: usize = bit_depth / 8;
    const bpp = channels * bytes_per_sample; // filter unit, bytes
    const stride = w * bpp;
    const raw_len = h * (1 + stride);

    const raw = try allocator.alloc(u8, raw_len);
    defer allocator.free(raw);
    {
        const window = try allocator.alloc(u8, flate.max_window_len);
        defer allocator.free(window);
        var input: std.Io.Reader = .fixed(idat.items);
        var decompress = flate.Decompress.init(&input, .zlib, window);
        decompress.reader.readSliceAll(raw) catch return Error.CorruptPng;
    }

    // Un-filter in place, row by row.
    var prev_row: ?[]u8 = null;
    var row_i: usize = 0;
    while (row_i < h) : (row_i += 1) {
        const filter = raw[row_i * (1 + stride)];
        const row = raw[row_i * (1 + stride) + 1 ..][0..stride];
        switch (filter) {
            0 => {},
            1 => { // Sub
                var i: usize = bpp;
                while (i < stride) : (i += 1) row[i] +%= row[i - bpp];
            },
            2 => { // Up
                if (prev_row) |up| {
                    for (row, up) |*x, b| x.* +%= b;
                }
            },
            3 => { // Average
                if (prev_row) |up| {
                    var i: usize = 0;
                    while (i < bpp) : (i += 1) row[i] +%= up[i] / 2;
                    while (i < stride) : (i += 1)
                        row[i] +%= @intCast((@as(u16, row[i - bpp]) + up[i]) / 2);
                } else {
                    var i: usize = bpp;
                    while (i < stride) : (i += 1) row[i] +%= row[i - bpp] / 2;
                }
            },
            4 => { // Paeth
                var i: usize = 0;
                while (i < stride) : (i += 1) {
                    const a: i32 = if (i >= bpp) row[i - bpp] else 0;
                    const b: i32 = if (prev_row) |up| up[i] else 0;
                    const c: i32 = if (i >= bpp and prev_row != null) prev_row.?[i - bpp] else 0;
                    const p = a + b - c;
                    const pa = @abs(p - a);
                    const pb = @abs(p - b);
                    const pc = @abs(p - c);
                    const pred: i32 = if (pa <= pb and pa <= pc) a else if (pb <= pc) b else c;
                    row[i] +%= @intCast(pred);
                }
            },
            else => return Error.CorruptPng,
        }
        prev_row = row;
    }

    // Convert to RGB8.
    const rgb = try allocator.alloc(u8, w * h * 3);
    errdefer allocator.free(rgb);
    row_i = 0;
    while (row_i < h) : (row_i += 1) {
        const row = raw[row_i * (1 + stride) + 1 ..][0..stride];
        const out_row = rgb[row_i * w * 3 ..][0 .. w * 3];
        var x: usize = 0;
        while (x < w) : (x += 1) {
            const src = row[x * bpp ..];
            const dst = out_row[x * 3 ..][0..3];
            // 16-bit samples take the high byte (stb's convert_16_to_8).
            switch (color_type) {
                0 => {
                    const g = src[0];
                    dst.* = .{ g, g, g };
                },
                2 => {
                    dst.* = .{ src[0], src[bytes_per_sample], src[2 * bytes_per_sample] };
                },
                3 => {
                    const idx: usize = src[0];
                    if (idx * 3 + 2 >= palette.len) return Error.CorruptPng;
                    dst.* = .{ palette[idx * 3], palette[idx * 3 + 1], palette[idx * 3 + 2] };
                },
                4 => {
                    const g = src[0];
                    dst.* = .{ g, g, g };
                },
                6 => {
                    dst.* = .{ src[0], src[bytes_per_sample], src[2 * bytes_per_sample] };
                },
                else => unreachable,
            }
        }
    }

    return .{ .allocator = allocator, .w = w, .h = h, .rgb = rgb };
}

pub fn loadPng(allocator: Allocator, io: std.Io, path: []const u8) !Image {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(512 * 1024 * 1024));
    defer allocator.free(bytes);
    return decodePng(allocator, bytes);
}

/// Write a tightly-packed RGB8 buffer as a PNG (filter 0, zlib). For the
/// `--annotated` output; not a parity surface.
pub fn writePng(allocator: Allocator, io: std.Io, path: []const u8, w: usize, h: usize, rgb: []const u8) !void {
    std.debug.assert(rgb.len == w * h * 3);

    var raw: std.ArrayList(u8) = .empty;
    defer raw.deinit(allocator);
    try raw.ensureTotalCapacity(allocator, h * (1 + w * 3));
    for (0..h) |y| {
        raw.appendAssumeCapacity(0);
        raw.appendSliceAssumeCapacity(rgb[y * w * 3 ..][0 .. w * 3]);
    }

    var compressed: std.Io.Writer.Allocating = try .initCapacity(allocator, 4096);
    defer compressed.deinit();
    {
        const compress_buf = try allocator.alloc(u8, flate.max_window_len);
        defer allocator.free(compress_buf);
        var compress = try flate.Compress.init(&compressed.writer, compress_buf, .zlib, .level_6);
        try compress.writer.writeAll(raw.items);
        try compress.finish();
    }

    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(allocator);
    try out.appendSlice(allocator, &png_signature);

    var ihdr: [13]u8 = undefined;
    std.mem.writeInt(u32, ihdr[0..4], @intCast(w), .big);
    std.mem.writeInt(u32, ihdr[4..8], @intCast(h), .big);
    ihdr[8] = 8; // bit depth
    ihdr[9] = 2; // color type: RGB
    ihdr[10] = 0; // compression
    ihdr[11] = 0; // filter
    ihdr[12] = 0; // interlace
    try appendChunk(allocator, &out, "IHDR", &ihdr);
    try appendChunk(allocator, &out, "IDAT", compressed.written());
    try appendChunk(allocator, &out, "IEND", &.{});

    var file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    try file.writeStreamingAll(io, out.items);
}

fn appendChunk(allocator: Allocator, out: *std.ArrayList(u8), chunk_type: *const [4]u8, data: []const u8) !void {
    var len_bytes: [4]u8 = undefined;
    std.mem.writeInt(u32, &len_bytes, @intCast(data.len), .big);
    try out.appendSlice(allocator, &len_bytes);
    try out.appendSlice(allocator, chunk_type);
    try out.appendSlice(allocator, data);
    var crc = std.hash.Crc32.init();
    crc.update(chunk_type);
    crc.update(data);
    var crc_bytes: [4]u8 = undefined;
    std.mem.writeInt(u32, &crc_bytes, crc.final(), .big);
    try out.appendSlice(allocator, &crc_bytes);
}

test {
    _ = @import("image_tests.zig");
}
