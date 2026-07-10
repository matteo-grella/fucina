//! Doc-index link checker for Fucina (`zig build doc-check`).
//!
//! AGENTS.md's "## Doc index" section is the routing table injected into every
//! agent session; dead rows there have repeatedly outlived the docs they
//! pointed at. This tool parses that section for backtick-quoted `*.md` names
//! and fails when a referenced doc does not exist on disk. The index routes
//! to root-level docs (`README.md`) and to `docs/<name>.md`; tokens with any
//! other path shape are skipped. Deliberately minimal — the arch-check
//! counterpart for doc rot.

const std = @import("std");

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const io = init.io;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    var stderr_buffer: [4096]u8 = undefined;
    var stderr_writer = std.Io.File.stderr().writer(io, &stderr_buffer);
    const stderr = &stderr_writer.interface;
    defer stderr.flush() catch {};

    const contents = try std.Io.Dir.cwd().readFileAlloc(io, "AGENTS.md", allocator, .limited(4 * 1024 * 1024));

    const section = docIndexSection(contents) orelse {
        try stderr.print("doc-check: AGENTS.md has no '## Doc index' section\n", .{});
        return error.DocIndexMissing;
    };

    var seen: std.StringHashMapUnmanaged(void) = .empty;
    defer seen.deinit(allocator);

    var checked: usize = 0;
    var missing: usize = 0;
    // splitScalar keeps empty pieces, so with balanced backticks the
    // odd-indexed pieces are exactly the backtick-quoted spans.
    var it = std.mem.splitScalar(u8, section, '`');
    var idx: usize = 0;
    while (it.next()) |token| : (idx += 1) {
        if (idx % 2 == 0) continue; // outside backticks
        if (!std.mem.endsWith(u8, token, ".md")) continue;
        // Route only root docs and docs/<name>.md; other paths are prose.
        if (std.mem.indexOfScalar(u8, token, '/')) |slash| {
            if (!std.mem.eql(u8, token[0..slash], "docs")) continue;
            if (std.mem.indexOfScalar(u8, token[slash + 1 ..], '/') != null) continue;
        }
        if ((try seen.getOrPut(allocator, token)).found_existing) continue;

        checked += 1;
        var handle = std.Io.Dir.cwd().openFile(io, token, .{}) catch {
            missing += 1;
            try stderr.print("doc-check: AGENTS.md doc index references missing doc: {s}\n", .{token});
            continue;
        };
        handle.close(io);
    }

    if (missing != 0) {
        try stderr.print("doc-check: {d} of {d} referenced root docs missing\n", .{ missing, checked });
        return error.DeadDocReference;
    }
    try stdout.print("doc index: {d} docs referenced, all present\n", .{checked});
}

/// The byte range of AGENTS.md's "## Doc index" section (up to the next `## `
/// heading or EOF).
fn docIndexSection(contents: []const u8) ?[]const u8 {
    const heading = "\n## Doc index";
    const start = std.mem.indexOf(u8, contents, heading) orelse return null;
    const body_start = start + heading.len;
    const rel_end = std.mem.indexOf(u8, contents[body_start..], "\n## ") orelse
        return contents[body_start..];
    return contents[body_start..][0..rel_end];
}
