//! Docs-site staging generator (`zig run tools/gen_docs_site.zig -- <repo root> <out dir>`).
//!
//! Feeds the MkDocs build behind `.github/workflows/pages.yml`. The repo
//! markdown (README.md, docs/, docs/reference/, docs/course/,
//! examples/*/README.md) is the single source of truth; this tool writes a
//! transformed copy into the gitignored staging directory that `mkdocs.yml`
//! uses as `docs_dir`:
//!
//! - `docs/reference/` is the chapter-per-file reference: `00-index.md` is
//!   staged as `reference/index.md`, and every `NN-<title-slug>.md` chapter
//!   page is staged under its own name, so a chapter's repo filename IS its
//!   site URL. The tool errors when a filename does not match its
//!   `# N. Title` heading (numbering must stay sequential from 1), so a
//!   retitled chapter cannot silently detach from its URL. Every runnable
//!   snippet (the blocks `zig build snippet-check` extracts, see
//!   tools/gen_snippet_tests.zig) gets a "compiled & run in CI" badge, and
//!   a bare `§N` reference outside code fails the build — reference
//!   cross-references are ordinary markdown links to the chapter files.
//! - The mkdocs nav is generated too: the tool appends it to the committed
//!   `mkdocs.yml` (which must not define `nav:`) and writes the result to
//!   `<repo root>/.mkdocs-gen.yml`, which the Pages workflow builds with
//!   `mkdocs build -f .mkdocs-gen.yml`. Course chapters and examples are
//!   enumerated from disk; guides carry their own sidebar placement in an
//!   invisible HTML comment (same convention as the snippet markers):
//!
//!       <!-- docs-nav: group="Run & serve" title="Running models" weight=10 -->
//!
//!   All keys are optional. `title` defaults to the file's H1; `group`
//!   names the sidebar group under Guides (`group=""` means a top-level
//!   entry, no group; no group key at all files the page under "More");
//!   `weight` orders entries and, via first appearance, the groups
//!   (default 1000, ties broken by filename). A brand-new docs/*.md file
//!   therefore ships with NO changes anywhere else, under "More" until it
//!   gets a directive. `docs/README.md` (the repo's doc index; the site nav
//!   is its counterpart) is not staged.
//! - Every staged heading gets an explicit GitHub-style anchor id via
//!   `{: #slug }` (attr_list), so anchors behave identically on GitHub and
//!   on the site.
//! - Relative links are re-resolved: targets that map to a staged page are
//!   rewritten to it; everything else (source files, vendor trees) becomes
//!   a GitHub URL.
//! - A final pass verifies every rewritten link and anchor against the
//!   staged tree, so a broken cross-reference fails the Pages build.

const std = @import("std");

const github_blob = "https://github.com/matteo-grella/fucina/blob/main/";

/// One reference chapter; index 0 is the index page (`00-index.md`).
/// Titles come from each file's `# N. Title` heading.
const Chapter = struct { title: []const u8, file: []const u8 };

/// A guide page's resolved sidebar placement (from its `docs-nav`
/// directive, defaults where absent). `group == null` files the page under
/// "More"; `group == ""` puts it directly under Guides.
const GuidePlacement = struct {
    file: []const u8,
    title: []const u8,
    group: ?[]const u8,
    weight: usize,
};

const snippet_badge =
    "<p class=\"snippet-ci\" title=\"Extracted from the documentation source and compiled + run against the current build by CI (zig build snippet-check).\">compiled &amp; run in CI \u{2713}</p>";

const Marker = enum { none, helper, skip };

const LinkRec = struct {
    from: []const u8, // staged page the link lives on
    to: []const u8, // staged page it points at
    anchor: []const u8, // "" when none
};

const NavEntry = struct { file: []const u8, title: []const u8 };

const Ctx = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    root: []const u8,
    out: []const u8,

    chapters: std.ArrayList(Chapter) = .empty, // [0] = the reference index page

    // nav inputs collected while staging
    guides: std.ArrayList(GuidePlacement) = .empty,
    course_entries: std.ArrayList(NavEntry) = .empty,
    example_names: std.ArrayList([]const u8) = .empty,

    // staged-tree records for the verify pass
    staged_pages: std.StringHashMapUnmanaged(void) = .empty,
    slug_keys: std.StringHashMapUnmanaged(void) = .empty, // "page#slug"
    links: std.ArrayList(LinkRec) = .empty,

    n_pages: usize = 0,
    n_badges: usize = 0,
    n_rewritten: usize = 0,

    fn repoPath(self: *Ctx, rel: []const u8) ![]const u8 {
        return std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ self.root, rel });
    }

    fn readRepoFile(self: *Ctx, rel: []const u8) ![]const u8 {
        const path = try self.repoPath(rel);
        return std.Io.Dir.cwd().readFileAlloc(self.io, path, self.allocator, .limited(16 * 1024 * 1024));
    }

    fn writeStaged(self: *Ctx, rel: []const u8, data: []const u8) !void {
        const path = try std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ self.out, rel });
        if (std.fs.path.dirname(path)) |dir| try std.Io.Dir.cwd().createDirPath(self.io, dir);
        var file = try std.Io.Dir.cwd().createFile(self.io, path, .{});
        defer file.close(self.io);
        var buffer: [64 * 1024]u8 = undefined;
        var writer = file.writer(self.io, &buffer);
        try writer.interface.writeAll(data);
        try writer.interface.flush();
        if (std.mem.endsWith(u8, rel, ".md")) {
            try self.staged_pages.put(self.allocator, try self.allocator.dupe(u8, rel), {});
            self.n_pages += 1;
        }
    }

    fn recordSlug(self: *Ctx, page: []const u8, slug: []const u8) !void {
        const key = try std.fmt.allocPrint(self.allocator, "{s}#{s}", .{ page, slug });
        try self.slug_keys.put(self.allocator, key, {});
    }

    fn recordLink(self: *Ctx, from: []const u8, to: []const u8, anchor: []const u8) !void {
        try self.links.append(self.allocator, .{
            .from = try self.allocator.dupe(u8, from),
            .to = try self.allocator.dupe(u8, to),
            .anchor = try self.allocator.dupe(u8, anchor),
        });
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const io = init.io;

    const args = try init.minimal.args.toSlice(allocator);
    if (args.len != 3) {
        std.debug.print("usage: gen_docs_site <repo root> <output dir>\n", .{});
        return error.BadUsage;
    }

    var ctx = Ctx{ .allocator = allocator, .io = io, .root = args[1], .out = args[2] };

    std.Io.Dir.cwd().deleteTree(io, ctx.out) catch {};
    try std.Io.Dir.cwd().createDirPath(io, ctx.out);

    try processReference(&ctx);
    try stageGuides(&ctx);
    try stagePage(&ctx, "README.md", "index.md");
    try stageCourse(&ctx);
    try stageExamples(&ctx);
    try copyAsset(&ctx, "site/docs-extra.css", "assets/extra.css");
    try verify(&ctx);
    try writeGeneratedConfig(&ctx);

    var stdout_buffer: [512]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};
    try stdout.print(
        "docs-site: {d} pages staged ({d} reference chapters), {d} CI badges, {d} links rewritten, {d} cross-links verified; nav in .mkdocs-gen.yml\n",
        .{ ctx.n_pages, ctx.chapters.items.len - 1, ctx.n_badges, ctx.n_rewritten, ctx.links.items.len },
    );
}

// ---------------------------------------------------------------- slugs

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
    return slugify(allocator, stripped.items);
}

/// GitHub-style heading slug: lowercase; keep [a-z0-9_-]; each space
/// becomes a hyphen (consecutive spaces stay consecutive hyphens, as on
/// GitHub); everything else (punctuation, backticks, non-ASCII) drops.
fn slugify(allocator: std.mem.Allocator, text: []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    const trimmed = std.mem.trim(u8, text, " \t");
    for (trimmed) |c| {
        switch (c) {
            'A'...'Z' => try out.append(allocator, c - 'A' + 'a'),
            'a'...'z', '0'...'9', '_', '-' => try out.append(allocator, c),
            ' ' => try out.append(allocator, '-'),
            else => {},
        }
    }
    return out.items;
}

const Heading = struct { level: usize, text: []const u8 };

/// A column-0 ATX heading outside a fence, or null.
fn parseHeading(line: []const u8) ?Heading {
    var n: usize = 0;
    while (n < line.len and line[n] == '#') n += 1;
    if (n == 0 or n > 6 or n >= line.len or line[n] != ' ') return null;
    return .{ .level = n, .text = std.mem.trim(u8, line[n + 1 ..], " \t") };
}

fn isExternal(target: []const u8) bool {
    return std.mem.startsWith(u8, target, "http://") or
        std.mem.startsWith(u8, target, "https://") or
        std.mem.startsWith(u8, target, "mailto:");
}

// ---------------------------------------------------------------- paths

/// Normalize a repo-relative path: resolve "." and "..".
fn normalizePath(allocator: std.mem.Allocator, path: []const u8) ![]const u8 {
    var parts: std.ArrayList([]const u8) = .empty;
    var it = std.mem.splitScalar(u8, path, '/');
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

/// Relative path from the directory of staged page `from` to staged path `to`.
fn relativize(allocator: std.mem.Allocator, from_page: []const u8, to: []const u8) ![]const u8 {
    const from_dir = std.fs.path.dirname(from_page) orelse "";
    var f = std.mem.splitScalar(u8, from_dir, '/');
    var t = std.mem.splitScalar(u8, to, '/');
    // Drop the common prefix.
    var f_rest: []const u8 = from_dir;
    var t_rest: []const u8 = to;
    while (true) {
        const fi = f.peek() orelse break;
        const ti = t.peek() orelse break;
        if (fi.len == 0) {
            _ = f.next();
            continue;
        }
        if (!std.mem.eql(u8, fi, ti)) break;
        _ = f.next();
        _ = t.next();
        f_rest = f.rest();
        t_rest = t.rest();
    }
    if (from_dir.len == 0) return to;
    var ups: usize = 0;
    var fr = std.mem.splitScalar(u8, f_rest, '/');
    while (fr.next()) |seg| {
        if (seg.len != 0) ups += 1;
    }
    var out: std.ArrayList(u8) = .empty;
    for (0..ups) |_| try out.appendSlice(allocator, "../");
    try out.appendSlice(allocator, t_rest);
    return out.items;
}

/// Map a normalized repo path to its staged page, or null (-> GitHub URL).
fn mapRepoToStaged(ctx: *Ctx, repo_path: []const u8) !?[]const u8 {
    const a = ctx.allocator;
    if (std.mem.eql(u8, repo_path, "README.md")) return "index.md";
    // The repo's doc index: not staged (the generated nav is the site's
    // index), so links to it lead to the file on GitHub.
    if (std.mem.eql(u8, repo_path, "docs/README.md")) return null;
    // Pre-split spelling, still used by the pinned docs/course tree: the
    // reference lives in docs/reference/ and its index is the successor.
    if (std.mem.eql(u8, repo_path, "docs/REFERENCE.md")) return "reference/index.md";
    if (std.mem.eql(u8, repo_path, "docs/reference/00-index.md")) return "reference/index.md";
    if (std.mem.startsWith(u8, repo_path, "docs/reference/") and std.mem.endsWith(u8, repo_path, ".md")) {
        const base = repo_path["docs/reference/".len..];
        if (std.mem.indexOfScalar(u8, base, '/') == null)
            return try std.fmt.allocPrint(a, "reference/{s}", .{base});
    }
    if (std.mem.eql(u8, repo_path, "docs/course/README.md")) return "course/index.md";
    if (std.mem.startsWith(u8, repo_path, "docs/course/") and std.mem.endsWith(u8, repo_path, ".md")) {
        const base = repo_path["docs/course/".len..];
        if (std.mem.indexOfScalar(u8, base, '/') == null)
            return try std.fmt.allocPrint(a, "course/{s}", .{base});
    }
    if (std.mem.startsWith(u8, repo_path, "docs/") and std.mem.endsWith(u8, repo_path, ".md")) {
        const base = repo_path["docs/".len..];
        if (std.mem.indexOfScalar(u8, base, '/') == null) {
            const lower = try std.ascii.allocLowerString(a, base);
            return try std.fmt.allocPrint(a, "guides/{s}", .{lower});
        }
    }
    if (std.mem.startsWith(u8, repo_path, "examples/") and std.mem.endsWith(u8, repo_path, "/README.md")) {
        const name = repo_path["examples/".len .. repo_path.len - "/README.md".len];
        if (std.mem.indexOfScalar(u8, name, '/') == null)
            return try std.fmt.allocPrint(a, "examples/{s}.md", .{name});
    }
    return null;
}

/// Resolve a relative markdown link found in `src_repo_path` and return the
/// rewritten target (staged-relative or GitHub URL), recording staged links
/// for the verify pass.
fn resolveLink(ctx: *Ctx, src_repo_path: []const u8, staged_page: []const u8, target: []const u8) ![]const u8 {
    const a = ctx.allocator;
    const hash = std.mem.indexOfScalar(u8, target, '#');
    const path_part = if (hash) |h| target[0..h] else target;
    const anchor = if (hash) |h| target[h + 1 ..] else "";

    const base_dir = std.fs.path.dirname(src_repo_path) orelse "";
    const joined = try std.fmt.allocPrint(a, "{s}/{s}", .{ base_dir, path_part });
    const repo_path = normalizePath(a, joined) catch {
        std.debug.print("docs-site: {s}: link '{s}' escapes the repo\n", .{ src_repo_path, target });
        return error.BadLink;
    };

    ctx.n_rewritten += 1;
    if (try mapRepoToStaged(ctx, repo_path)) |staged_target| {
        try ctx.recordLink(staged_page, staged_target, anchor);
        const rel = try relativize(a, staged_page, staged_target);
        if (anchor.len != 0) return std.fmt.allocPrint(a, "{s}#{s}", .{ rel, anchor });
        return rel;
    }
    // Not staged: point at the file on GitHub.
    if (anchor.len != 0) return std.fmt.allocPrint(a, "{s}{s}#{s}", .{ github_blob, repo_path, anchor });
    return std.fmt.allocPrint(a, "{s}{s}", .{ github_blob, repo_path });
}

// ---------------------------------------------------------------- REFERENCE

/// Stage the chapter-per-file reference under `docs/reference/`. Chapter
/// order, titles, and staged names all come from the directory listing and
/// each file's `# N. Title` heading; the filename must be
/// `NN-<title slug>.md` and numbering must run 1..N with no gaps.
fn processReference(ctx: *Ctx) !void {
    const a = ctx.allocator;
    var names: std.ArrayList([]const u8) = .empty;
    {
        const ref_path = try ctx.repoPath("docs/reference");
        var dir = try std.Io.Dir.cwd().openDir(ctx.io, ref_path, .{ .iterate = true });
        defer dir.close(ctx.io);
        var walker = try dir.walk(a);
        defer walker.deinit();
        while (try walker.next(ctx.io)) |entry| {
            if (entry.kind != .file) continue;
            if (std.mem.indexOfScalar(u8, entry.path, '/') != null) continue; // top level only
            if (!std.mem.endsWith(u8, entry.path, ".md")) continue;
            try names.append(a, try a.dupe(u8, entry.path));
        }
    }
    sortStrings(names.items);

    try ctx.chapters.append(a, .{ .title = "", .file = "index.md" });
    var found_index = false;
    for (names.items) |name| {
        if (std.mem.eql(u8, name, "00-index.md")) {
            found_index = true;
            continue;
        }
        const src = try std.fmt.allocPrint(a, "docs/reference/{s}", .{name});
        const title = try referenceChapterTitle(ctx, src, ctx.chapters.items.len);
        const expect = try std.fmt.allocPrint(a, "{d:0>2}-{s}.md", .{ ctx.chapters.items.len, try slugify(a, title) });
        if (!std.mem.eql(u8, name, expect)) {
            std.debug.print("docs-site: {s}: filename does not match its chapter heading '{s}' (expected {s}); the filename is the page URL\n", .{ src, title, expect });
            return error.ChapterFileMismatch;
        }
        try ctx.chapters.append(a, .{ .title = title, .file = name });
    }
    if (!found_index) {
        std.debug.print("docs-site: docs/reference/00-index.md (the reference index page) is missing\n", .{});
        return error.NoReferenceIndex;
    }
    if (ctx.chapters.items.len < 2) {
        std.debug.print("docs-site: docs/reference/ has no numbered chapter files\n", .{});
        return error.NoChapters;
    }

    try stageReferencePage(ctx, "docs/reference/00-index.md", "reference/index.md");
    for (ctx.chapters.items[1..]) |ch| {
        const src = try std.fmt.allocPrint(a, "docs/reference/{s}", .{ch.file});
        const staged = try std.fmt.allocPrint(a, "reference/{s}", .{ch.file});
        try stageReferencePage(ctx, src, staged);
    }
}

/// The chapter title from a reference file's `# N. Title` heading, which
/// must carry the expected sequential number (the chapter list and the
/// numbered anchors depend on the order).
fn referenceChapterTitle(ctx: *Ctx, repo_rel: []const u8, expected_num: usize) ![]const u8 {
    const content = try ctx.readRepoFile(repo_rel);
    var lines = std.mem.splitScalar(u8, content, '\n');
    const h1 = while (lines.next()) |line| {
        if (std.mem.startsWith(u8, line, "# ")) break std.mem.trim(u8, line[2..], " \t");
    } else {
        std.debug.print("docs-site: {s} has no `# N. Title` heading\n", .{repo_rel});
        return error.UnnumberedChapter;
    };
    const dot = std.mem.indexOfScalar(u8, h1, '.') orelse {
        std.debug.print("docs-site: {s}: unnumbered chapter heading '{s}'\n", .{ repo_rel, h1 });
        return error.UnnumberedChapter;
    };
    const num = std.fmt.parseInt(usize, h1[0..dot], 10) catch {
        std.debug.print("docs-site: {s}: unnumbered chapter heading '{s}'\n", .{ repo_rel, h1 });
        return error.UnnumberedChapter;
    };
    if (num != expected_num) {
        std.debug.print("docs-site: {s} breaks the chapter sequence (expected {d}); chapters must be numbered 1..N in order\n", .{ repo_rel, expected_num });
        return error.ChapterSequenceBroken;
    }
    return std.mem.trim(u8, h1[dot + 1 ..], " ");
}

/// Stage one reference page: the generic staging (anchors, link rewriting)
/// plus the snippet-CI badge after every runnable ```zig block, and a guard
/// that no bare `§N` survives outside code — reference cross-references are
/// markdown links to the chapter files, so a bare one is rot.
fn stageReferencePage(ctx: *Ctx, src_repo_path: []const u8, staged_rel: []const u8) !void {
    const a = ctx.allocator;
    const src = try ctx.readRepoFile(src_repo_path);

    var out: std.ArrayList(u8) = .empty;
    var seen_slugs: std.StringHashMapUnmanaged(usize) = .empty;
    var in_fence = false;
    var fence_zig = false;
    var fence_has_test = false;
    var marker: Marker = .none;
    var fence_marker: Marker = .none;
    // Inline-code parity carries ACROSS lines inside a paragraph (a code
    // span may wrap); paragraph breaks, headings, fences, and table rows
    // reset it.
    var code_state: usize = 0;

    var it = std.mem.splitScalar(u8, src, '\n');
    while (it.next()) |line| {
        if (in_fence) {
            try out.appendSlice(a, line);
            try out.append(a, '\n');
            if (std.mem.eql(u8, std.mem.trimEnd(u8, line, " \t"), "```")) {
                in_fence = false;
                if (fence_zig and fence_marker == .none and fence_has_test) {
                    try out.append(a, '\n');
                    try out.appendSlice(a, snippet_badge);
                    try out.append(a, '\n');
                    ctx.n_badges += 1;
                }
                continue;
            }
            if (fence_zig and std.mem.startsWith(u8, line, "test \"")) fence_has_test = true;
            continue;
        }
        if (std.mem.startsWith(u8, line, "```")) {
            in_fence = true;
            fence_zig = std.mem.startsWith(u8, line, "```zig");
            fence_has_test = false;
            fence_marker = marker;
            marker = .none;
            code_state = 0;
            try out.appendSlice(a, line);
            try out.append(a, '\n');
            continue;
        }
        if (std.mem.startsWith(u8, line, "<!-- snippet: helper -->")) {
            marker = .helper;
            try out.appendSlice(a, line);
            try out.append(a, '\n');
            continue;
        }
        if (std.mem.startsWith(u8, line, "<!-- snippet: skip -->")) {
            marker = .skip;
            try out.appendSlice(a, line);
            try out.append(a, '\n');
            continue;
        }
        if (parseHeading(line)) |h| {
            marker = .none;
            code_state = 0;
            var slug = try slugifyHeading(a, h.text);
            const gop = try seen_slugs.getOrPut(a, slug);
            if (gop.found_existing) {
                gop.value_ptr.* += 1;
                slug = try std.fmt.allocPrint(a, "{s}-{d}", .{ slug, gop.value_ptr.* });
            } else {
                gop.value_ptr.* = 0;
            }
            try checkNoBareSectionRef(src_repo_path, h.text, 0);
            try out.appendSlice(a, line[0 .. h.level + 1]);
            var head_state: usize = 0;
            try transformPageInline(ctx, &out, h.text, src_repo_path, staged_rel, &head_state);
            const attr = try std.fmt.allocPrint(a, " {{: #{s} }}\n", .{slug});
            try out.appendSlice(a, attr);
            try ctx.recordSlug(staged_rel, slug);
            continue;
        }
        const trimmed = std.mem.trim(u8, line, " \t");
        if (trimmed.len != 0) marker = .none;
        if (trimmed.len == 0 or trimmed[0] == '|') code_state = 0;
        try checkNoBareSectionRef(src_repo_path, line, code_state);
        try transformPageInline(ctx, &out, line, src_repo_path, staged_rel, &code_state);
        try out.append(a, '\n');
    }
    try ctx.writeStaged(staged_rel, out.items);
}

/// Error on a bare `§<digit>` outside inline code and outside markdown
/// link text (`[§4.7](...)` and `[DEVELOPMENT.md §7](...)` are fine).
/// '§' is U+00A7: bytes C2 A7. `start_ticks` is the inline-code parity at
/// line start.
fn checkNoBareSectionRef(src_repo_path: []const u8, line: []const u8, start_ticks: usize) !void {
    var ticks = start_ticks;
    var i: usize = 0;
    while (i < line.len) {
        const c = line[i];
        if (c == '`') {
            var n: usize = 0;
            while (i + n < line.len and line[i + n] == '`') n += 1;
            if (ticks == 0) {
                ticks = n;
            } else if (n >= ticks) {
                ticks = 0;
            }
            i += n;
            continue;
        }
        if (ticks == 0 and c == '[') {
            // A `[label](target)` span: the label is link text; skip it.
            if (std.mem.indexOfScalarPos(u8, line, i + 1, ']')) |rb| {
                if (rb + 1 < line.len and line[rb + 1] == '(') {
                    i = rb + 1;
                    continue;
                }
            }
        }
        if (ticks == 0 and c == 0xC2 and i + 2 < line.len and line[i + 1] == 0xA7 and
            std.ascii.isDigit(line[i + 2]))
        {
            std.debug.print("docs-site: {s}: bare section reference '{s}' — write it as a markdown link to the chapter file\n", .{ src_repo_path, line[i..@min(line.len, i + 8)] });
            return error.BareSectionReference;
        }
        i += 1;
    }
}

// ---------------------------------------------------------------- other pages

/// Stage a non-reference markdown page: inject GitHub-style anchors on every
/// heading, rewrite relative links against the staged tree.
fn stagePage(ctx: *Ctx, src_repo_path: []const u8, staged_rel: []const u8) !void {
    const a = ctx.allocator;
    const src = try ctx.readRepoFile(src_repo_path);

    var out: std.ArrayList(u8) = .empty;
    var seen_slugs: std.StringHashMapUnmanaged(usize) = .empty;
    var in_fence = false;
    var code_state: usize = 0;

    var it = std.mem.splitScalar(u8, src, '\n');
    while (it.next()) |line| {
        if (std.mem.startsWith(u8, line, "```")) {
            in_fence = !in_fence;
            code_state = 0;
            try out.appendSlice(a, line);
            try out.append(a, '\n');
            continue;
        }
        if (in_fence) {
            try out.appendSlice(a, line);
            try out.append(a, '\n');
            continue;
        }
        if (parseHeading(line)) |h| {
            code_state = 0;
            var slug = try slugifyHeading(a, h.text);
            // GitHub numbers duplicate slugs -1, -2, ... per file.
            const gop = try seen_slugs.getOrPut(a, slug);
            if (gop.found_existing) {
                gop.value_ptr.* += 1;
                slug = try std.fmt.allocPrint(a, "{s}-{d}", .{ slug, gop.value_ptr.* });
            } else {
                gop.value_ptr.* = 0;
            }
            try out.appendSlice(a, line[0 .. h.level + 1]);
            var head_state: usize = 0;
            try transformPageInline(ctx, &out, h.text, src_repo_path, staged_rel, &head_state);
            const attr = try std.fmt.allocPrint(a, " {{: #{s} }}\n", .{slug});
            try out.appendSlice(a, attr);
            try ctx.recordSlug(staged_rel, slug);
            continue;
        }
        const trimmed = std.mem.trim(u8, line, " \t");
        if (trimmed.len == 0 or (trimmed.len != 0 and trimmed[0] == '|')) code_state = 0;
        try transformPageInline(ctx, &out, line, src_repo_path, staged_rel, &code_state);
        try out.append(a, '\n');
    }
    try ctx.writeStaged(staged_rel, out.items);
}

fn transformPageInline(ctx: *Ctx, out: *std.ArrayList(u8), line: []const u8, src_repo_path: []const u8, staged_rel: []const u8, code_ticks: *usize) !void {
    const a = ctx.allocator;
    var i: usize = 0;
    while (i < line.len) {
        const c = line[i];
        if (c == '`') {
            var n: usize = 0;
            while (i + n < line.len and line[i + n] == '`') n += 1;
            if (code_ticks.* == 0) {
                code_ticks.* = n;
            } else if (n >= code_ticks.*) {
                code_ticks.* = 0;
            }
            try out.appendSlice(a, line[i .. i + n]);
            i += n;
            continue;
        }
        if (code_ticks.* == 0 and c == ']' and i + 2 < line.len and line[i + 1] == '(') {
            if (std.mem.indexOfScalarPos(u8, line, i + 2, ')')) |close| {
                const target = line[i + 2 .. close];
                if (target.len > 1 and target[0] == '#') {
                    // Same-page anchor: keep, but verify it.
                    try ctx.recordLink(staged_rel, staged_rel, target[1..]);
                } else if (target.len > 0 and target[0] != '#' and !isExternal(target)) {
                    const rewritten = try resolveLink(ctx, src_repo_path, staged_rel, target);
                    try out.appendSlice(a, "](");
                    try out.appendSlice(a, rewritten);
                    try out.append(a, ')');
                    i = close + 1;
                    continue;
                }
            }
        }
        try out.append(a, c);
        i += 1;
    }
}

fn stageGuides(ctx: *Ctx) !void {
    const a = ctx.allocator;
    var names: std.ArrayList([]const u8) = .empty;
    {
        const docs_path = try ctx.repoPath("docs");
        var dir = try std.Io.Dir.cwd().openDir(ctx.io, docs_path, .{ .iterate = true });
        defer dir.close(ctx.io);
        var walker = try dir.walk(a);
        defer walker.deinit();
        while (try walker.next(ctx.io)) |entry| {
            if (entry.kind != .file) continue;
            if (std.mem.indexOfScalar(u8, entry.path, '/') != null) continue; // top level only
            if (!std.mem.endsWith(u8, entry.path, ".md")) continue;
            if (std.mem.eql(u8, entry.path, "README.md")) continue; // the repo doc index; not staged
            try names.append(a, try a.dupe(u8, entry.path));
        }
    }
    sortStrings(names.items);
    for (names.items) |name| {
        const src = try std.fmt.allocPrint(a, "docs/{s}", .{name});
        const lower = try std.ascii.allocLowerString(a, name);
        const staged = try std.fmt.allocPrint(a, "guides/{s}", .{lower});
        try stagePage(ctx, src, staged);
        const directive = try parseNavDirective(ctx, src);
        try ctx.guides.append(a, .{
            .file = lower,
            .title = directive.title orelse try firstHeadingText(ctx, src),
            .group = directive.group,
            .weight = directive.weight orelse 1000,
        });
    }
}

/// The first heading's text in a repo markdown file (nav-title fallback).
fn firstHeadingText(ctx: *Ctx, repo_rel: []const u8) ![]const u8 {
    const content = try ctx.readRepoFile(repo_rel);
    var lines = std.mem.splitScalar(u8, content, '\n');
    while (lines.next()) |line| {
        if (std.mem.startsWith(u8, line, "# ")) return std.mem.trim(u8, line[2..], " \t");
    }
    return repo_rel;
}

const NavDirective = struct { title: ?[]const u8 = null, group: ?[]const u8 = null, weight: ?usize = null };

/// The page's `<!-- docs-nav: ... -->` directive (see the doc comment at
/// the top), or all-defaults when the file has none. Malformed directives
/// and unknown keys fail the build rather than silently mis-filing a page.
fn parseNavDirective(ctx: *Ctx, repo_rel: []const u8) !NavDirective {
    const a = ctx.allocator;
    const content = try ctx.readRepoFile(repo_rel);
    const prefix = "<!-- docs-nav:";
    var lines = std.mem.splitScalar(u8, content, '\n');
    const body = while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t");
        if (!std.mem.startsWith(u8, trimmed, prefix)) continue;
        if (!std.mem.endsWith(u8, trimmed, "-->")) {
            std.debug.print("docs-site: {s}: docs-nav directive is not closed with -->\n", .{repo_rel});
            return error.BadNavDirective;
        }
        break trimmed[prefix.len .. trimmed.len - "-->".len];
    } else return .{};

    var d = NavDirective{};
    var i: usize = 0;
    while (i < body.len) {
        while (i < body.len and (body[i] == ' ' or body[i] == ',')) i += 1;
        if (i >= body.len) break;
        const key_start = i;
        while (i < body.len and std.ascii.isAlphabetic(body[i])) i += 1;
        const key = body[key_start..i];
        if (i >= body.len or body[i] != '=') {
            std.debug.print("docs-site: {s}: expected key=value in docs-nav directive at '{s}'\n", .{ repo_rel, body[key_start..] });
            return error.BadNavDirective;
        }
        i += 1;
        var value: []const u8 = undefined;
        if (i < body.len and body[i] == '"') {
            const close = std.mem.indexOfScalarPos(u8, body, i + 1, '"') orelse {
                std.debug.print("docs-site: {s}: unterminated string in docs-nav directive\n", .{repo_rel});
                return error.BadNavDirective;
            };
            value = body[i + 1 .. close];
            i = close + 1;
        } else {
            const value_start = i;
            while (i < body.len and body[i] != ' ' and body[i] != ',') i += 1;
            value = body[value_start..i];
        }
        if (std.mem.eql(u8, key, "title")) {
            d.title = try a.dupe(u8, value);
        } else if (std.mem.eql(u8, key, "group")) {
            d.group = try a.dupe(u8, value);
        } else if (std.mem.eql(u8, key, "weight")) {
            d.weight = std.fmt.parseInt(usize, value, 10) catch {
                std.debug.print("docs-site: {s}: docs-nav weight '{s}' is not a number\n", .{ repo_rel, value });
                return error.BadNavDirective;
            };
        } else {
            std.debug.print("docs-site: {s}: unknown docs-nav key '{s}'\n", .{ repo_rel, key });
            return error.BadNavDirective;
        }
    }
    return d;
}

fn stageCourse(ctx: *Ctx) !void {
    const a = ctx.allocator;
    var names: std.ArrayList([]const u8) = .empty;
    {
        const course_path = try ctx.repoPath("docs/course");
        var dir = try std.Io.Dir.cwd().openDir(ctx.io, course_path, .{ .iterate = true });
        defer dir.close(ctx.io);
        var walker = try dir.walk(a);
        defer walker.deinit();
        while (try walker.next(ctx.io)) |entry| {
            if (entry.kind != .file) continue;
            if (std.mem.indexOfScalar(u8, entry.path, '/') != null) continue; // skip video-scripts/
            if (!std.mem.endsWith(u8, entry.path, ".md")) continue;
            try names.append(a, try a.dupe(u8, entry.path));
        }
    }
    sortStrings(names.items);
    for (names.items) |name| {
        const src = try std.fmt.allocPrint(a, "docs/course/{s}", .{name});
        const staged = if (std.mem.eql(u8, name, "README.md"))
            try a.dupe(u8, "course/index.md")
        else
            try std.fmt.allocPrint(a, "course/{s}", .{name});
        try stagePage(ctx, src, staged);
        if (!std.mem.eql(u8, name, "README.md")) {
            const title = try courseNavTitle(ctx, name, try firstHeadingText(ctx, src));
            try ctx.course_entries.append(a, .{ .file = name, .title = title });
        }
    }
}

/// Short sidebar title for a course chapter: "NN · Short", with the short
/// part taken from the H1 pattern "Chapter NN — Title[: subtitle]" (the
/// subtitle is dropped); falls back to the full H1.
fn courseNavTitle(ctx: *Ctx, file_name: []const u8, h1: []const u8) ![]const u8 {
    var title = h1;
    if (std.mem.indexOf(u8, h1, " \u{2014} ")) |p| title = h1[p + " \u{2014} ".len ..];
    if (std.mem.indexOfScalar(u8, title, ':')) |p| title = title[0..p];
    title = std.mem.trim(u8, title, " \t");
    if (file_name.len >= 2 and std.ascii.isDigit(file_name[0]) and std.ascii.isDigit(file_name[1])) {
        return std.fmt.allocPrint(ctx.allocator, "{s} \u{00B7} {s}", .{ file_name[0..2], title });
    }
    return title;
}

fn stageExamples(ctx: *Ctx) !void {
    const a = ctx.allocator;
    var names: std.ArrayList([]const u8) = .empty;
    {
        const examples_path = try ctx.repoPath("examples");
        var dir = try std.Io.Dir.cwd().openDir(ctx.io, examples_path, .{ .iterate = true });
        defer dir.close(ctx.io);
        var walker = try dir.walk(a);
        defer walker.deinit();
        while (try walker.next(ctx.io)) |entry| {
            if (entry.kind != .file) continue;
            // Exactly examples/<name>/README.md.
            const slash = std.mem.indexOfScalar(u8, entry.path, '/') orelse continue;
            if (!std.mem.eql(u8, entry.path[slash + 1 ..], "README.md")) continue;
            try names.append(a, try a.dupe(u8, entry.path[0..slash]));
        }
    }
    sortStrings(names.items);

    var index_body: std.ArrayList(u8) = .empty;
    try index_body.appendSlice(a,
        \\# Examples {: #examples }
        \\
        \\Every example is a standalone program under
        \\[`examples/`](https://github.com/matteo-grella/fucina/tree/main/examples)
        \\in the repository, built by `zig build`; each page below is that
        \\example's README, verbatim.
        \\
        \\
    );
    for (names.items) |name| {
        const src = try std.fmt.allocPrint(a, "examples/{s}/README.md", .{name});
        const staged = try std.fmt.allocPrint(a, "examples/{s}.md", .{name});
        try stagePage(ctx, src, staged);
        try ctx.example_names.append(a, name);

        const entry = try std.fmt.allocPrint(a, "- [{s}]({s}.md)\n", .{ try firstHeadingText(ctx, src), name });
        try index_body.appendSlice(a, entry);
        try ctx.recordLink("examples/index.md", staged, "");
    }
    try ctx.writeStaged("examples/index.md", index_body.items);
    try ctx.recordSlug("examples/index.md", "examples");
}

fn copyAsset(ctx: *Ctx, src_rel: []const u8, staged_rel: []const u8) !void {
    const data = try ctx.readRepoFile(src_rel);
    try ctx.writeStaged(staged_rel, data);
}

fn sortStrings(items: [][]const u8) void {
    std.mem.sort([]const u8, items, {}, struct {
        fn lt(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.lt);
}

// ---------------------------------------------------------------- config

/// Concatenate the committed mkdocs.yml (which must not define nav:) with
/// the generated navigation and write `<repo root>/.mkdocs-gen.yml`, the
/// config the site is actually built from (`mkdocs build -f .mkdocs-gen.yml`).
fn writeGeneratedConfig(ctx: *Ctx) !void {
    const a = ctx.allocator;
    const base = try ctx.readRepoFile("mkdocs.yml");
    var check = std.mem.splitScalar(u8, base, '\n');
    while (check.next()) |line| {
        if (std.mem.startsWith(u8, line, "nav:")) {
            std.debug.print("docs-site: mkdocs.yml must not define nav: (the nav is generated; a guide's placement lives in its own docs-nav comment)\n", .{});
            return error.NavInBaseConfig;
        }
    }

    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(a, "# GENERATED by tools/gen_docs_site.zig from mkdocs.yml plus the staged tree — do not edit.\n");
    try out.appendSlice(a, base);
    if (base.len != 0 and base[base.len - 1] != '\n') try out.append(a, '\n');
    try out.appendSlice(a, "\nnav:\n  - Home: index.md\n  - Reference:\n      - reference/index.md\n");
    for (ctx.chapters.items[1..]) |ch| {
        const entry = try std.fmt.allocPrint(a, "      - reference/{s}\n", .{ch.file});
        try out.appendSlice(a, entry);
    }

    try out.appendSlice(a, "  - Guides:\n");
    // Placement comes from each guide's docs-nav directive: sort by
    // (weight, file); a group is emitted whole where its first member
    // lands; directive-less pages default to weight 1000 in "More".
    std.mem.sort(GuidePlacement, ctx.guides.items, {}, struct {
        fn lt(_: void, x: GuidePlacement, y: GuidePlacement) bool {
            if (x.weight != y.weight) return x.weight < y.weight;
            return std.mem.lessThan(u8, x.file, y.file);
        }
    }.lt);
    var emitted_groups: std.StringHashMapUnmanaged(void) = .empty;
    for (ctx.guides.items, 0..) |g, gi| {
        if (g.group != null and g.group.?.len == 0) {
            // group="": a top-level entry directly under Guides.
            const entry = try std.fmt.allocPrint(a, "      - {s}: guides/{s}\n", .{ try yamlString(a, g.title), g.file });
            try out.appendSlice(a, entry);
            continue;
        }
        const group = g.group orelse "More";
        if (emitted_groups.get(group) != null) continue;
        try emitted_groups.put(a, group, {});
        const header = try std.fmt.allocPrint(a, "      - {s}:\n", .{try yamlString(a, group)});
        try out.appendSlice(a, header);
        for (ctx.guides.items[gi..]) |m| {
            if (m.group != null and m.group.?.len == 0) continue;
            if (!std.mem.eql(u8, m.group orelse "More", group)) continue;
            const entry = try std.fmt.allocPrint(a, "          - {s}: guides/{s}\n", .{ try yamlString(a, m.title), m.file });
            try out.appendSlice(a, entry);
        }
    }

    try out.appendSlice(a, "  - Learn:\n      - course/index.md\n");
    for (ctx.course_entries.items) |e| {
        const entry = try std.fmt.allocPrint(a, "      - {s}: course/{s}\n", .{ try yamlString(a, e.title), e.file });
        try out.appendSlice(a, entry);
    }

    try out.appendSlice(a, "  - Examples:\n      - examples/index.md\n");
    for (ctx.example_names.items) |name| {
        const entry = try std.fmt.allocPrint(a, "      - {s}: examples/{s}.md\n", .{ try yamlString(a, name), name });
        try out.appendSlice(a, entry);
    }

    const path = try ctx.repoPath(".mkdocs-gen.yml");
    var file = try std.Io.Dir.cwd().createFile(ctx.io, path, .{});
    defer file.close(ctx.io);
    var buffer: [64 * 1024]u8 = undefined;
    var writer = file.writer(ctx.io, &buffer);
    try writer.interface.writeAll(out.items);
    try writer.interface.flush();
}

fn yamlString(allocator: std.mem.Allocator, s: []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.append(allocator, '"');
    for (s) |c| switch (c) {
        '"' => try out.appendSlice(allocator, "\\\""),
        '\\' => try out.appendSlice(allocator, "\\\\"),
        else => try out.append(allocator, c),
    };
    try out.append(allocator, '"');
    return out.items;
}

// ---------------------------------------------------------------- verify

fn verify(ctx: *Ctx) !void {
    var bad: usize = 0;
    for (ctx.links.items) |link| {
        if (ctx.staged_pages.get(link.to) == null) {
            std.debug.print("docs-site: {s} links to unstaged page {s}\n", .{ link.from, link.to });
            bad += 1;
            continue;
        }
        if (link.anchor.len != 0) {
            const key = try std.fmt.allocPrint(ctx.allocator, "{s}#{s}", .{ link.to, link.anchor });
            if (ctx.slug_keys.get(key) == null) {
                std.debug.print("docs-site: {s} links to missing anchor {s}#{s}\n", .{ link.from, link.to, link.anchor });
                bad += 1;
            }
        }
    }
    if (bad != 0) return error.BrokenCrossLinks;
}
