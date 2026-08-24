//! Doc link checker for Fucina (`zig build doc-check`) — the arch-check
//! counterpart for doc rot. Four checks:
//!
//! 1. `docs/README.md` is THE doc index. Every doc it names (backtick tokens
//!    and markdown-link targets shaped like root `*.md`, `docs/<name>.md`,
//!    `docs/reference/<name>.md`, or `examples/<name>/README.md`) must exist,
//!    and the set of top-level `docs/*.md` it lists must be exactly the docs
//!    carrying a `<!-- docs-nav: ... -->` header (the sidebar set of
//!    tools/gen_docs_site.zig), so the index and the site nav cannot drift
//!    apart. `AGENTS.md` must point at `docs/README.md`.
//! 2. Every intra-doc markdown link in `README.md`, `docs/*.md`, and
//!    `docs/reference/*.md` (docs/course is generated-content, pinned, and
//!    covered by the site build instead) that targets a `.md` file resolves
//!    to an existing file, and its `#anchor` (when present) to an existing
//!    heading (GitHub slug rules, duplicate slugs numbered `-1`, `-2`, …).
//!    Same-file `#anchor` links are checked too.
//! 3. Every `src/<path>.zig:<line>` (or `:<start>-<end>`) citation in those
//!    files points at existing, non-blank lines.
//! 4. `README.md`'s `zig fetch ...#vX.Y.Z` pin must match `build.zig.zon`'s
//!    `.version` — the one line every consumer copies is otherwise the first
//!    to rot on release.
//!
//! `docs/RUNNING-MODELS.md` is additionally scanned for
//! `examples/<name>/README.md` references (backtick-quoted, markdown-link
//! targets, or bare paths), which are existence-checked the same way.

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

    var checker: Checker = .{ .allocator = allocator, .io = io, .stderr = stderr };

    try checkDocIndex(&checker);
    try checkAgentsPointer(&checker);

    // The link/citation surface: README.md + every top-level doc + the
    // reference chapters.
    var files: std.ArrayList([]const u8) = .empty;
    try files.append(allocator, "README.md");
    try files.append(allocator, "docs/README.md");
    try listMarkdown(allocator, io, "docs", "docs", &files);
    try listMarkdown(allocator, io, "docs/reference", "docs/reference", &files);
    for (files.items) |file| {
        try checkFileLinks(&checker, file);
        try checkSrcCitations(&checker, file);
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
        try checker.checkExists("docs/RUNNING-MODELS.md", token);
    }

    if (checker.problems != 0) {
        try stderr.print("doc-check: {d} problems ({d} files/links/citations checked)\n", .{ checker.problems, checker.checked });
        return error.DocCheckFailed;
    }
    try stdout.print("doc links: {d} files/links/citations checked, all resolve\n", .{checker.checked});

    try checkFetchPin(allocator, io, stdout, stderr);
}

/// Shared state: problem counter plus caches for file reads and per-file
/// anchor sets.
const Checker = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    stderr: *std.Io.Writer,
    seen_exists: std.StringHashMapUnmanaged(bool) = .empty, // path -> exists
    anchors: std.StringHashMapUnmanaged(*std.StringHashMapUnmanaged(void)) = .empty,
    checked: usize = 0,
    problems: usize = 0,

    fn fail(self: *Checker, comptime fmt: []const u8, args: anytype) !void {
        self.problems += 1;
        try self.stderr.print("doc-check: " ++ fmt ++ "\n", args);
    }

    fn exists(self: *Checker, path: []const u8) !bool {
        const gop = try self.seen_exists.getOrPut(self.allocator, path);
        if (!gop.found_existing) {
            gop.key_ptr.* = try self.allocator.dupe(u8, path);
            gop.value_ptr.* = blk: {
                var handle = std.Io.Dir.cwd().openFile(self.io, path, .{}) catch break :blk false;
                handle.close(self.io);
                break :blk true;
            };
        }
        return gop.value_ptr.*;
    }

    fn checkExists(self: *Checker, source: []const u8, path: []const u8) !void {
        self.checked += 1;
        if (!try self.exists(path))
            try self.fail("{s} references missing doc: {s}", .{ source, path });
    }

    /// The heading-anchor set of a markdown file (GitHub slugs, duplicate
    /// slugs numbered), cached per path. Null when the file cannot be read.
    fn anchorSet(self: *Checker, path: []const u8) !?*std.StringHashMapUnmanaged(void) {
        const a = self.allocator;
        const gop = try self.anchors.getOrPut(a, path);
        if (gop.found_existing) return gop.value_ptr.*;
        gop.key_ptr.* = try a.dupe(u8, path);
        const set = try a.create(std.StringHashMapUnmanaged(void));
        set.* = .empty;
        gop.value_ptr.* = set;
        const content = std.Io.Dir.cwd().readFileAlloc(self.io, path, a, .limited(16 * 1024 * 1024)) catch return null;
        var counts: std.StringHashMapUnmanaged(usize) = .empty;
        var in_fence = false;
        var lines = std.mem.splitScalar(u8, content, '\n');
        while (lines.next()) |line| {
            if (std.mem.startsWith(u8, line, "```")) {
                in_fence = !in_fence;
                continue;
            }
            if (in_fence) continue;
            const h = parseHeading(line) orelse continue;
            var slug = try slugifyHeading(a, h.text);
            const c = try counts.getOrPut(a, slug);
            if (c.found_existing) {
                c.value_ptr.* += 1;
                slug = try std.fmt.allocPrint(a, "{s}-{d}", .{ slug, c.value_ptr.* });
            } else {
                c.value_ptr.* = 0;
            }
            try set.put(a, slug, {});
        }
        return set;
    }
};

// ---------------------------------------------------------------- index

/// `docs/README.md` existence + nav-set equality (check 1).
fn checkDocIndex(checker: *Checker) !void {
    const a = checker.allocator;
    const io = checker.io;
    const index = std.Io.Dir.cwd().readFileAlloc(io, "docs/README.md", a, .limited(4 * 1024 * 1024)) catch {
        try checker.fail("docs/README.md (the doc index) is missing", .{});
        return;
    };

    // Every doc-shaped token in the index: backtick spans and link targets.
    var listed: std.StringHashMapUnmanaged(void) = .empty;
    var it = std.mem.splitScalar(u8, index, '`');
    var idx: usize = 0;
    while (it.next()) |token| : (idx += 1) {
        if (idx % 2 == 1) try collectDocToken(a, &listed, token);
    }
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, index, pos, "](")) |start| {
        const close = std.mem.indexOfScalarPos(u8, index, start + 2, ')') orelse break;
        pos = close + 1;
        var target = index[start + 2 .. close];
        if (std.mem.indexOfScalar(u8, target, '#')) |h| target = target[0..h];
        // Index links are written repo-relative from docs/ (e.g.
        // `RUNNING-MODELS.md`, `reference/00-index.md`, `../README.md`).
        const repo = resolveFrom(a, "docs/README.md", target) catch continue;
        try collectDocToken(a, &listed, repo);
    }

    var lit = listed.iterator();
    while (lit.next()) |entry| try checker.checkExists("docs/README.md", entry.key_ptr.*);

    // Nav-set equality over top-level docs/*.md (README.md itself excluded).
    var nav_docs: std.ArrayList([]const u8) = .empty;
    try listMarkdown(a, io, "docs", "docs", &nav_docs);
    for (nav_docs.items) |path| {
        const content = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 * 1024 * 1024)) catch continue;
        const has_nav = std.mem.indexOf(u8, content, "<!-- docs-nav:") != null;
        const is_listed = listed.get(path) != null;
        checker.checked += 1;
        if (has_nav and !is_listed)
            try checker.fail("docs/README.md does not list {s} (it carries a docs-nav header)", .{path});
        if (!has_nav and is_listed)
            try checker.fail("docs/README.md lists {s}, which has no docs-nav header (add one or drop the row)", .{path});
    }
}

/// Record `token` when it is doc-shaped: root `*.md`, `docs/<name>.md`,
/// `docs/reference/<name>.md`, or `examples/<name>/README.md`.
fn collectDocToken(a: std.mem.Allocator, listed: *std.StringHashMapUnmanaged(void), token: []const u8) !void {
    if (!std.mem.endsWith(u8, token, ".md")) return;
    if (std.mem.indexOfScalar(u8, token, ' ') != null) return;
    if (std.mem.indexOfScalar(u8, token, '/')) |slash| {
        const head = token[0..slash];
        const rest = token[slash + 1 ..];
        const is_docs = std.mem.eql(u8, head, "docs") and std.mem.indexOfScalar(u8, rest, '/') == null;
        const is_reference = std.mem.startsWith(u8, token, "docs/reference/") and
            std.mem.indexOfScalar(u8, token["docs/reference/".len..], '/') == null;
        if (!is_docs and !is_reference and !isExampleReadme(token)) return;
    }
    try listed.put(a, try a.dupe(u8, token), {});
}

/// AGENTS.md routes agents to the index: it must name `docs/README.md`.
fn checkAgentsPointer(checker: *Checker) !void {
    const contents = std.Io.Dir.cwd().readFileAlloc(checker.io, "AGENTS.md", checker.allocator, .limited(4 * 1024 * 1024)) catch {
        try checker.fail("AGENTS.md is missing", .{});
        return;
    };
    checker.checked += 1;
    if (std.mem.indexOf(u8, contents, "docs/README.md") == null)
        try checker.fail("AGENTS.md does not point at docs/README.md (the doc index)", .{});
}

// ---------------------------------------------------------------- links

/// Intra-doc markdown links (check 2): `.md` targets must exist and their
/// anchors must be real headings; same-file `#anchor` links likewise.
/// Links inside code fences and inline code spans are ignored.
fn checkFileLinks(checker: *Checker, file: []const u8) !void {
    const a = checker.allocator;
    const content = std.Io.Dir.cwd().readFileAlloc(checker.io, file, a, .limited(16 * 1024 * 1024)) catch return;
    var in_fence = false;
    var code_state: usize = 0;
    var lines = std.mem.splitScalar(u8, content, '\n');
    while (lines.next()) |line| {
        if (std.mem.startsWith(u8, line, "```")) {
            in_fence = !in_fence;
            code_state = 0;
            continue;
        }
        if (in_fence) continue;
        const trimmed = std.mem.trim(u8, line, " \t");
        if (parseHeading(line) != null or trimmed.len == 0 or trimmed[0] == '|') code_state = 0;
        var i: usize = 0;
        while (i < line.len) {
            const c = line[i];
            if (c == '`') {
                var n: usize = 0;
                while (i + n < line.len and line[i + n] == '`') n += 1;
                if (code_state == 0) {
                    code_state = n;
                } else if (n >= code_state) {
                    code_state = 0;
                }
                i += n;
                continue;
            }
            if (code_state == 0 and c == ']' and i + 2 < line.len and line[i + 1] == '(') {
                if (std.mem.indexOfScalarPos(u8, line, i + 2, ')')) |close| {
                    try checkLinkTarget(checker, file, line[i + 2 .. close]);
                    i = close + 1;
                    continue;
                }
            }
            i += 1;
        }
    }
}

fn checkLinkTarget(checker: *Checker, file: []const u8, target: []const u8) !void {
    const a = checker.allocator;
    if (target.len == 0 or isExternal(target)) return;
    if (target[0] == '#') {
        checker.checked += 1;
        if (try checker.anchorSet(file)) |set| {
            if (set.get(target[1..]) == null)
                try checker.fail("{s}: link to missing anchor #{s}", .{ file, target[1..] });
        }
        return;
    }
    const hash = std.mem.indexOfScalar(u8, target, '#');
    const path_part = if (hash) |h| target[0..h] else target;
    const anchor = if (hash) |h| target[h + 1 ..] else "";
    if (!std.mem.endsWith(u8, path_part, ".md")) return;
    const repo_path = resolveFrom(a, file, path_part) catch {
        try checker.fail("{s}: link '{s}' escapes the repo", .{ file, target });
        return;
    };
    checker.checked += 1;
    if (!try checker.exists(repo_path)) {
        try checker.fail("{s}: link to missing file {s}", .{ file, repo_path });
        return;
    }
    if (anchor.len != 0) {
        if (try checker.anchorSet(repo_path)) |set| {
            if (set.get(anchor) == null)
                try checker.fail("{s}: link to missing anchor {s}#{s}", .{ file, repo_path, anchor });
        }
    }
}

// ---------------------------------------------------------------- citations

/// `src/<path>.zig:<line>` citations (check 3): the file exists and every
/// cited endpoint is an existing, non-blank line.
fn checkSrcCitations(checker: *Checker, file: []const u8) !void {
    const a = checker.allocator;
    const content = std.Io.Dir.cwd().readFileAlloc(checker.io, file, a, .limited(16 * 1024 * 1024)) catch return;
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, content, pos, "src/")) |start| {
        pos = start + "src/".len;
        if (start > 0 and (isNameChar(content[start - 1]) or content[start - 1] == '/')) continue;
        var end = start;
        while (end < content.len and isPathChar(content[end])) end += 1;
        if (!std.mem.endsWith(u8, content[start..end], ".zig")) continue;
        if (end >= content.len or content[end] != ':') continue;
        var num_end = end + 1;
        while (num_end < content.len and std.ascii.isDigit(content[num_end])) num_end += 1;
        if (num_end == end + 1) continue;
        const line_a = std.fmt.parseInt(usize, content[end + 1 .. num_end], 10) catch continue;
        var line_b = line_a;
        var cite_end = num_end;
        if (cite_end < content.len and content[cite_end] == '-' and cite_end + 1 < content.len and std.ascii.isDigit(content[cite_end + 1])) {
            var b_end = cite_end + 1;
            while (b_end < content.len and std.ascii.isDigit(content[b_end])) b_end += 1;
            line_b = std.fmt.parseInt(usize, content[cite_end + 1 .. b_end], 10) catch line_a;
            cite_end = b_end;
        }
        const src_path = content[start..end];
        const cite = content[start..cite_end];
        pos = cite_end;
        checker.checked += 1;
        const src = std.Io.Dir.cwd().readFileAlloc(checker.io, src_path, a, .limited(64 * 1024 * 1024)) catch {
            try checker.fail("{s}: citation {s} names a missing file", .{ file, cite });
            continue;
        };
        var line_list: std.ArrayList([]const u8) = .empty;
        var lit = std.mem.splitScalar(u8, src, '\n');
        while (lit.next()) |l| try line_list.append(a, l);
        for ([2]usize{ line_a, line_b }) |n| {
            if (n == 0 or n > line_list.items.len) {
                try checker.fail("{s}: citation {s} points past the end ({d} lines)", .{ file, cite, line_list.items.len });
                break;
            }
            if (std.mem.trim(u8, line_list.items[n - 1], " \t\r").len == 0) {
                try checker.fail("{s}: citation {s} points at a blank line ({d})", .{ file, cite, n });
                break;
            }
        }
    }
}

// ---------------------------------------------------------------- fetch pin

/// README.md's `zig fetch --save git+...#vX.Y.Z` line must pin the version
/// declared in build.zig.zon. Both are parsed leniently (first occurrence);
/// a README without a fetch pin or a zon without `.version` is an error too,
/// so silent format drift cannot disable the check.
fn checkFetchPin(allocator: std.mem.Allocator, io: std.Io, stdout: *std.Io.Writer, stderr: *std.Io.Writer) !void {
    const zon = try std.Io.Dir.cwd().readFileAlloc(io, "build.zig.zon", allocator, .limited(1024 * 1024));
    const version = valueAfter(zon, ".version = \"", '"') orelse {
        try stderr.print("doc-check: build.zig.zon has no .version field\n", .{});
        return error.VersionPinMissing;
    };

    const readme = try std.Io.Dir.cwd().readFileAlloc(io, "README.md", allocator, .limited(4 * 1024 * 1024));
    const fetch_marker = "zig fetch --save git+";
    const fetch_start = std.mem.indexOf(u8, readme, fetch_marker) orelse {
        try stderr.print("doc-check: README.md has no '{s}' line\n", .{fetch_marker});
        return error.VersionPinMissing;
    };
    const tag = valueAfter(readme[fetch_start..], "#v", '\n') orelse {
        try stderr.print("doc-check: README.md fetch line has no '#v' tag pin\n", .{});
        return error.VersionPinMissing;
    };

    if (!std.mem.eql(u8, std.mem.trimEnd(u8, tag, " `"), version)) {
        try stderr.print(
            "doc-check: README.md fetch pin v{s} != build.zig.zon version {s}\n",
            .{ std.mem.trimEnd(u8, tag, " `"), version },
        );
        return error.VersionPinMismatch;
    }
    try stdout.print("fetch pin: README.md pins v{s}, matches build.zig.zon\n", .{version});
}

/// The span between the first `prefix` occurrence and the next `terminator`.
fn valueAfter(haystack: []const u8, prefix: []const u8, terminator: u8) ?[]const u8 {
    const start = (std.mem.indexOf(u8, haystack, prefix) orelse return null) + prefix.len;
    const len = std.mem.indexOfScalar(u8, haystack[start..], terminator) orelse return null;
    return haystack[start..][0..len];
}

// ---------------------------------------------------------------- helpers

/// Append every top-level `*.md` under `dir_path` (as `prefix/<name>`),
/// sorted; `docs/README.md` is the index itself and is excluded.
fn listMarkdown(allocator: std.mem.Allocator, io: std.Io, dir_path: []const u8, prefix: []const u8, files: *std.ArrayList([]const u8)) !void {
    var names: std.ArrayList([]const u8) = .empty;
    var dir = try std.Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true });
    defer dir.close(io);
    var walker = try dir.walk(allocator);
    defer walker.deinit();
    while (try walker.next(io)) |entry| {
        if (entry.kind != .file) continue;
        if (std.mem.indexOfScalar(u8, entry.path, '/') != null) continue; // top level only
        if (!std.mem.endsWith(u8, entry.path, ".md")) continue;
        if (std.mem.eql(u8, dir_path, "docs") and std.mem.eql(u8, entry.path, "README.md")) continue;
        try names.append(allocator, try allocator.dupe(u8, entry.path));
    }
    std.mem.sort([]const u8, names.items, {}, struct {
        fn lt(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.lt);
    for (names.items) |name|
        try files.append(allocator, try std.fmt.allocPrint(allocator, "{s}/{s}", .{ prefix, name }));
}

/// Resolve `target` relative to the directory of `file`, normalizing "."
/// and ".." (error when the path escapes the repo root).
fn resolveFrom(allocator: std.mem.Allocator, file: []const u8, target: []const u8) ![]const u8 {
    const base_dir = std.fs.path.dirname(file) orelse "";
    var parts: std.ArrayList([]const u8) = .empty;
    var base_it = std.mem.splitScalar(u8, base_dir, '/');
    while (base_it.next()) |seg| {
        if (seg.len != 0) try parts.append(allocator, seg);
    }
    var it = std.mem.splitScalar(u8, target, '/');
    while (it.next()) |seg| {
        if (seg.len == 0 or std.mem.eql(u8, seg, ".")) continue;
        if (std.mem.eql(u8, seg, "..")) {
            if (parts.items.len == 0) return error.LinkEscapesRepo;
            _ = parts.pop();
            continue;
        }
        try parts.append(allocator, seg);
    }
    return std.mem.join(allocator, "/", parts.items);
}

const Heading = struct { level: usize, text: []const u8 };

/// A column-0 ATX heading outside a fence, or null.
fn parseHeading(line: []const u8) ?Heading {
    var n: usize = 0;
    while (n < line.len and line[n] == '#') n += 1;
    if (n == 0 or n > 6 or n >= line.len or line[n] != ' ') return null;
    return .{ .level = n, .text = std.mem.trim(u8, line[n + 1 ..], " \t") };
}

/// Heading slug for a heading that may contain markdown links: GitHub slugs
/// the RENDERED text, so `[text](url)` contributes only `text`.
fn slugifyHeading(allocator: std.mem.Allocator, text: []const u8) ![]const u8 {
    var stripped: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < text.len) {
        if (text[i] == '[') {
            if (std.mem.indexOfScalarPos(u8, text, i + 1, ']')) |rb| {
                if (rb + 1 < text.len and text[rb + 1] == '(') {
                    if (std.mem.indexOfScalarPos(u8, text, rb + 2, ')')) |close| {
                        try stripped.appendSlice(allocator, text[i + 1 .. rb]);
                        i = close + 1;
                        continue;
                    }
                }
            }
        }
        try stripped.append(allocator, text[i]);
        i += 1;
    }
    var out: std.ArrayList(u8) = .empty;
    for (std.mem.trim(u8, stripped.items, " \t")) |c| {
        switch (c) {
            'A'...'Z' => try out.append(allocator, c - 'A' + 'a'),
            'a'...'z', '0'...'9', '_', '-' => try out.append(allocator, c),
            ' ' => try out.append(allocator, '-'),
            else => {},
        }
    }
    return out.items;
}

fn isExternal(target: []const u8) bool {
    return std.mem.startsWith(u8, target, "http://") or
        std.mem.startsWith(u8, target, "https://") or
        std.mem.startsWith(u8, target, "mailto:");
}

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
