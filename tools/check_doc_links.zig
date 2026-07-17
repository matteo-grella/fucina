//! Doc-index link checker for Fucina (`zig build doc-check`).
//!
//! AGENTS.md's "## Doc index" section is the routing table injected into every
//! agent session; dead rows there have repeatedly outlived the docs they
//! pointed at. This tool parses that section for backtick-quoted `*.md` names
//! and fails when a referenced doc does not exist on disk. The index routes
//! to root-level docs (`README.md`), to `docs/<name>.md`, and to per-example
//! `examples/<name>/README.md` files; tokens with any other path shape are
//! skipped. docs/RUNNING-MODELS.md is additionally scanned for
//! `examples/<name>/README.md` references (backtick-quoted, markdown-link
//! targets, or bare paths), which are existence-checked the same way.
//! Deliberately minimal — the arch-check counterpart for doc rot.

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

    var checker: Checker = .{ .allocator = allocator, .io = io, .stderr = stderr };
    defer checker.seen.deinit(allocator);

    // splitScalar keeps empty pieces, so with balanced backticks the
    // odd-indexed pieces are exactly the backtick-quoted spans.
    var it = std.mem.splitScalar(u8, section, '`');
    var idx: usize = 0;
    while (it.next()) |token| : (idx += 1) {
        if (idx % 2 == 0) continue; // outside backticks
        if (!std.mem.endsWith(u8, token, ".md")) continue;
        // Route root docs, docs/<name>.md, and examples/<name>/README.md;
        // tokens with any other path shape are prose.
        if (std.mem.indexOfScalar(u8, token, '/')) |slash| {
            const is_docs = std.mem.eql(u8, token[0..slash], "docs") and
                std.mem.indexOfScalar(u8, token[slash + 1 ..], '/') == null;
            if (!is_docs and !isExampleReadme(token)) continue;
        }
        try checker.check("AGENTS.md doc index", token);
    }

    // docs/RUNNING-MODELS.md routes to the per-example READMEs; scan the whole
    // file for `examples/<name>/README.md` references and check those too.
    const running = try std.Io.Dir.cwd().readFileAlloc(io, "docs/RUNNING-MODELS.md", allocator, .limited(4 * 1024 * 1024));
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, running, pos, "examples/")) |start| {
        var end = start;
        while (end < running.len and isPathChar(running[end])) end += 1;
        pos = end;
        if (start > 0 and isNameChar(running[start - 1])) continue; // tail of a longer word
        var token = running[start..end];
        // Trim sentence punctuation glued to a bare-prose path.
        while (token.len > 0 and token[token.len - 1] == '.') token = token[0 .. token.len - 1];
        if (!isExampleReadme(token)) continue;
        try checker.check("docs/RUNNING-MODELS.md", token);
    }

    if (checker.missing != 0) {
        try stderr.print("doc-check: {d} of {d} referenced docs missing\n", .{ checker.missing, checker.checked });
        return error.DeadDocReference;
    }
    try stdout.print("doc links: {d} docs referenced, all present\n", .{checker.checked});
}

/// Dedupe + existence-check state shared across the scanned sources.
const Checker = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    stderr: *std.Io.Writer,
    seen: std.StringHashMapUnmanaged(void) = .empty,
    checked: usize = 0,
    missing: usize = 0,

    fn check(self: *Checker, source: []const u8, token: []const u8) !void {
        if ((try self.seen.getOrPut(self.allocator, token)).found_existing) return;
        self.checked += 1;
        var handle = std.Io.Dir.cwd().openFile(self.io, token, .{}) catch {
            self.missing += 1;
            try self.stderr.print("doc-check: {s} references missing doc: {s}\n", .{ source, token });
            return;
        };
        handle.close(self.io);
    }
};

/// True when `token` names a per-example README: `examples/<name>/README.md`.
fn isExampleReadme(token: []const u8) bool {
    const prefix = "examples/";
    const suffix = "/README.md";
    return token.len > prefix.len + suffix.len and
        std.mem.startsWith(u8, token, prefix) and
        std.mem.endsWith(u8, token, suffix);
}

fn isPathChar(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or switch (c) {
        '_', '-', '.', '/' => true,
        else => false,
    };
}

fn isNameChar(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_' or c == '-';
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
