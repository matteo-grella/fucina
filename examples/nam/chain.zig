//! `.chain` manifest parser for the NAM serial-chain feature: an ordered list
//! of processing stages (NAM models and/or cab IRs) that `examples/nam.zig`
//! turns into a `live.Chain`. Parse-only and example-local — the returned
//! slices BORROW the caller's text buffer (which must outlive the spec).
//!
//! Grammar (line-based, UTF-8, LF or CRLF; paths may contain spaces):
//!   - blank lines and lines whose first non-space char is `#` are ignored
//!   - `name: <text>` (case-insensitive prefix, first wins) sets the chain name
//!   - any other line is a stage: `<path> [ :: trim=<dB> ]...`
//!     the ENTIRE text before the first ` :: ` (space-colon-colon-space) is the
//!     path, so spaces/commas/`-`/`#`-mid-name are preserved verbatim; options
//!     after each ` :: ` are `key=value` (v1 defines only `trim=<dB>`).
//!   - stage kind is by extension: `.wav` -> cab, `.nam`/`.gguf` -> NAM model.

const std = @import("std");

pub const StageKind = enum { nam, cab };

pub const StageSpec = struct {
    /// Borrows the manifest text buffer.
    path: []const u8,
    kind: StageKind,
    /// Per-stage input trim in dB (0 = unity); converted to linear at build.
    trim_db: f32 = 0.0,
};

pub const ManifestSpec = struct {
    /// Borrows the text buffer; null => caller defaults to the manifest's
    /// basename without its extension.
    name: ?[]const u8,
    stages: []StageSpec,
    list: std.ArrayList(StageSpec),
    allocator: std.mem.Allocator,

    pub fn deinit(self: *ManifestSpec) void {
        self.list.deinit(self.allocator);
    }
};

pub const ParseError = error{ EmptyChain, UnknownStageType, UnknownOption, BadTrim };

const ws = " \t\r\n";
const delim = " :: ";

fn classify(path: []const u8) ParseError!StageKind {
    if (std.ascii.endsWithIgnoreCase(path, ".wav")) return .cab;
    if (std.ascii.endsWithIgnoreCase(path, ".nam")) return .nam;
    if (std.ascii.endsWithIgnoreCase(path, ".gguf")) return .nam;
    return ParseError.UnknownStageType;
}

/// Parse a `.chain` manifest. `text` must outlive the returned spec (its
/// `path`/`name` slices borrow it).
pub fn parse(allocator: std.mem.Allocator, text: []const u8) !ManifestSpec {
    var name: ?[]const u8 = null;
    var list: std.ArrayList(StageSpec) = .empty;
    errdefer list.deinit(allocator);

    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |raw| {
        const line = std.mem.trim(u8, raw, ws);
        if (line.len == 0 or line[0] == '#') continue;
        if (line.len >= 5 and std.ascii.eqlIgnoreCase(line[0..5], "name:")) {
            const v = std.mem.trim(u8, line[5..], ws);
            if (name == null and v.len > 0) name = v;
            continue;
        }

        var path: []const u8 = line;
        var opts: []const u8 = "";
        if (std.mem.indexOf(u8, line, delim)) |k| {
            // The line is already whitespace-trimmed, so line[0] is non-space:
            // the text before the first delimiter is always a non-empty path.
            path = std.mem.trim(u8, line[0..k], ws);
            opts = line[k + delim.len ..];
        }

        var spec: StageSpec = .{ .path = path, .kind = try classify(path), .trim_db = 0.0 };
        var oit = std.mem.splitSequence(u8, opts, delim);
        while (oit.next()) |orow| {
            const o = std.mem.trim(u8, orow, ws);
            if (o.len == 0) continue;
            if (std.mem.startsWith(u8, o, "trim=")) {
                spec.trim_db = std.fmt.parseFloat(f32, o["trim=".len..]) catch return ParseError.BadTrim;
            } else return ParseError.UnknownOption;
        }
        try list.append(allocator, spec);
    }

    if (list.items.len == 0) return ParseError.EmptyChain;
    return .{ .name = name, .stages = list.items, .list = list, .allocator = allocator };
}

test {
    _ = @import("chain_tests.zig");
}
