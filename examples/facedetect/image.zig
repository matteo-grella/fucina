//! Minimal image I/O for the face-detect port: a canonical 8-bit RGB image, a
//! self-describing raw-pixel format (the "reference-pixel ingest" the strict
//! parity gates feed), and a pure-Zig PNG codec (decode: 8-bit
//! grayscale/RGB/RGBA non-interlaced via std zlib inflate; encode: RGB via
//! stored zlib blocks). JPEG is not supported.

const std = @import("std");
const flate = std.compress.flate;
const Crc32 = std.hash.crc.Crc32;

pub const Image = struct {
    allocator: std.mem.Allocator,
    width: usize,
    height: usize,
    channels: usize = 3, // canonical RGB
    pixels: []u8, // row-major, height * width * channels, 8-bit

    pub fn initRgb(allocator: std.mem.Allocator, width: usize, height: usize) !Image {
        const pixels = try allocator.alloc(u8, width * height * 3);
        return .{ .allocator = allocator, .width = width, .height = height, .pixels = pixels };
    }

    pub fn deinit(self: *Image) void {
        self.allocator.free(self.pixels);
        self.* = undefined;
    }
};

// ---------------------------------------------------------------------------
// Reference-pixel ingest — a self-describing raw RGB blob ("FDR1" + w,h,c + px).
// The reference dumps its exact preprocessed pixels here; parity stages feed
// them so a JPEG/PNG decoder never sits upstream of a strict gate.
// ---------------------------------------------------------------------------

const raw_magic = "FDR1";

pub fn toRaw(img: Image, allocator: std.mem.Allocator) ![]u8 {
    var out = try allocator.alloc(u8, 4 + 4 + 4 + 1 + img.pixels.len);
    errdefer allocator.free(out);
    @memcpy(out[0..4], raw_magic);
    std.mem.writeInt(u32, out[4..8], @intCast(img.width), .little);
    std.mem.writeInt(u32, out[8..12], @intCast(img.height), .little);
    out[12] = @intCast(img.channels);
    @memcpy(out[13..], img.pixels);
    return out;
}

pub fn fromRaw(allocator: std.mem.Allocator, bytes: []const u8) !Image {
    if (bytes.len < 13 or !std.mem.eql(u8, bytes[0..4], raw_magic)) return error.BadRawImage;
    const w = std.mem.readInt(u32, bytes[4..8], .little);
    const h = std.mem.readInt(u32, bytes[8..12], .little);
    const c = bytes[12];
    if (c != 3) return error.UnsupportedRawChannels;
    const need = @as(usize, w) * @as(usize, h) * c;
    if (bytes.len - 13 != need) return error.BadRawImage;
    const pixels = try allocator.dupe(u8, bytes[13 .. 13 + need]);
    return .{ .allocator = allocator, .width = w, .height = h, .channels = c, .pixels = pixels };
}

// ---------------------------------------------------------------------------
// PNG decode (8-bit, color types 0/2/6, non-interlaced).
// ---------------------------------------------------------------------------

const png_sig = [_]u8{ 137, 80, 78, 71, 13, 10, 26, 10 };

fn paeth(a: i32, b: i32, c: i32) u8 {
    const p = a + b - c;
    const pa = @abs(p - a);
    const pb = @abs(p - b);
    const pc = @abs(p - c);
    if (pa <= pb and pa <= pc) return @intCast(a);
    if (pb <= pc) return @intCast(b);
    return @intCast(c);
}

pub fn decodePng(allocator: std.mem.Allocator, bytes: []const u8) !Image {
    if (bytes.len < 8 or !std.mem.eql(u8, bytes[0..8], &png_sig)) return error.NotPng;
    var o: usize = 8;

    var width: usize = 0;
    var height: usize = 0;
    var src_channels: usize = 0;

    var idat = std.ArrayList(u8).empty;
    defer idat.deinit(allocator);

    while (o + 8 <= bytes.len) {
        const len = std.mem.readInt(u32, bytes[o..][0..4], .big);
        const ctype = bytes[o + 4 .. o + 8];
        const data_start = o + 8;
        if (data_start + len + 4 > bytes.len) return error.TruncatedPng;
        const data = bytes[data_start .. data_start + len];

        if (std.mem.eql(u8, ctype, "IHDR")) {
            width = std.mem.readInt(u32, data[0..4], .big);
            height = std.mem.readInt(u32, data[4..8], .big);
            const bit_depth = data[8];
            const color_type = data[9];
            const interlace = data[12];
            if (bit_depth != 8) return error.UnsupportedBitDepth;
            if (interlace != 0) return error.InterlaceUnsupported;
            src_channels = switch (color_type) {
                0 => 1, // grayscale
                2 => 3, // RGB
                6 => 4, // RGBA
                else => return error.UnsupportedColorType,
            };
        } else if (std.mem.eql(u8, ctype, "IDAT")) {
            try idat.appendSlice(allocator, data);
        } else if (std.mem.eql(u8, ctype, "IEND")) {
            break;
        }
        o = data_start + len + 4; // skip data + CRC
    }
    if (width == 0 or height == 0 or src_channels == 0) return error.MissingIhdr;

    // zlib-inflate the concatenated IDAT into filtered scanlines.
    var in_reader = std.Io.Reader.fixed(idat.items);
    var window: [flate.max_window_len]u8 = undefined;
    var dc = flate.Decompress.init(&in_reader, .zlib, &window);
    const stride = width * src_channels;
    const raw = try dc.reader.allocRemaining(allocator, .limited(height * (1 + stride) + 16));
    defer allocator.free(raw);
    if (raw.len != height * (1 + stride)) return error.BadPngData;

    // Unfilter in place into a contiguous src-channel buffer.
    const buf = try allocator.alloc(u8, height * stride);
    defer allocator.free(buf);
    const bpp = src_channels;
    var y: usize = 0;
    while (y < height) : (y += 1) {
        const filter = raw[y * (1 + stride)];
        const in_row = raw[y * (1 + stride) + 1 ..][0..stride];
        const out_row = buf[y * stride ..][0..stride];
        const prev_row: ?[]const u8 = if (y == 0) null else buf[(y - 1) * stride ..][0..stride];
        var x: usize = 0;
        while (x < stride) : (x += 1) {
            const a: i32 = if (x >= bpp) out_row[x - bpp] else 0;
            const b: i32 = if (prev_row) |p| p[x] else 0;
            const c: i32 = if (prev_row != null and x >= bpp) prev_row.?[x - bpp] else 0;
            const v: i32 = in_row[x];
            out_row[x] = switch (filter) {
                0 => @intCast(v),
                1 => @truncate(@as(u32, @bitCast(v + a))),
                2 => @truncate(@as(u32, @bitCast(v + b))),
                3 => @truncate(@as(u32, @bitCast(v + @divFloor(a + b, 2)))),
                4 => @truncate(@as(u32, @bitCast(v + paeth(a, b, c)))),
                else => return error.BadPngFilter,
            };
        }
    }

    // Canonicalize to RGB.
    var img = try Image.initRgb(allocator, width, height);
    errdefer img.deinit();
    var i: usize = 0;
    while (i < width * height) : (i += 1) {
        const s = i * src_channels;
        const d = i * 3;
        switch (src_channels) {
            1 => {
                img.pixels[d] = buf[s];
                img.pixels[d + 1] = buf[s];
                img.pixels[d + 2] = buf[s];
            },
            3, 4 => {
                img.pixels[d] = buf[s];
                img.pixels[d + 1] = buf[s + 1];
                img.pixels[d + 2] = buf[s + 2];
            },
            else => unreachable,
        }
    }
    return img;
}

// ---------------------------------------------------------------------------
// PNG encode (RGB, filter None, stored zlib blocks — no compression needed).
// ---------------------------------------------------------------------------

fn adler32(data: []const u8) u32 {
    var a: u32 = 1;
    var b: u32 = 0;
    for (data) |byte| {
        a = (a + byte) % 65521;
        b = (b + a) % 65521;
    }
    return (b << 16) | a;
}

fn appendChunk(out: *std.ArrayList(u8), gpa: std.mem.Allocator, ctype: []const u8, data: []const u8) !void {
    var lenbe: [4]u8 = undefined;
    std.mem.writeInt(u32, &lenbe, @intCast(data.len), .big);
    try out.appendSlice(gpa, &lenbe);
    try out.appendSlice(gpa, ctype);
    try out.appendSlice(gpa, data);
    var crc = Crc32.init();
    crc.update(ctype);
    crc.update(data);
    var crcbe: [4]u8 = undefined;
    std.mem.writeInt(u32, &crcbe, crc.final(), .big);
    try out.appendSlice(gpa, &crcbe);
}

pub fn encodePng(allocator: std.mem.Allocator, img: Image) ![]u8 {
    std.debug.assert(img.channels == 3);
    // Filtered scanlines (filter 0 = None), one filter byte per row.
    const stride = img.width * 3;
    const filtered = try allocator.alloc(u8, img.height * (1 + stride));
    defer allocator.free(filtered);
    var y: usize = 0;
    while (y < img.height) : (y += 1) {
        filtered[y * (1 + stride)] = 0;
        @memcpy(filtered[y * (1 + stride) + 1 ..][0..stride], img.pixels[y * stride ..][0..stride]);
    }

    // zlib stream: header + stored deflate blocks + adler32.
    var zlib = std.ArrayList(u8).empty;
    defer zlib.deinit(allocator);
    try zlib.appendSlice(allocator, &.{ 0x78, 0x01 }); // CMF, FLG (no dict, fastest)
    var pos: usize = 0;
    while (pos < filtered.len) {
        const block = @min(filtered.len - pos, 0xffff);
        const final: u8 = if (pos + block >= filtered.len) 1 else 0;
        try zlib.append(allocator, final); // BFINAL, BTYPE=00, byte-aligned
        var lenle: [2]u8 = undefined;
        std.mem.writeInt(u16, &lenle, @intCast(block), .little);
        try zlib.appendSlice(allocator, &lenle);
        std.mem.writeInt(u16, &lenle, @intCast(~@as(u16, @intCast(block))), .little);
        try zlib.appendSlice(allocator, &lenle);
        try zlib.appendSlice(allocator, filtered[pos .. pos + block]);
        pos += block;
    }
    var adlerbe: [4]u8 = undefined;
    std.mem.writeInt(u32, &adlerbe, adler32(filtered), .big);
    try zlib.appendSlice(allocator, &adlerbe);

    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, &png_sig);
    var ihdr: [13]u8 = undefined;
    std.mem.writeInt(u32, ihdr[0..4], @intCast(img.width), .big);
    std.mem.writeInt(u32, ihdr[4..8], @intCast(img.height), .big);
    ihdr[8] = 8; // bit depth
    ihdr[9] = 2; // color type RGB
    ihdr[10] = 0; // compression
    ihdr[11] = 0; // filter
    ihdr[12] = 0; // interlace
    try appendChunk(&out, allocator, "IHDR", &ihdr);
    try appendChunk(&out, allocator, "IDAT", zlib.items);
    try appendChunk(&out, allocator, "IEND", &.{});
    return out.toOwnedSlice(allocator);
}
