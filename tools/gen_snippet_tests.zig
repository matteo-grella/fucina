//! REFERENCE.md snippet-test generator (`zig build snippet-check`).
//!
//! The doc's runnable examples are ```zig fenced blocks containing a
//! column-0 `test` declaration. This tool extracts every such block into a
//! standalone test file under an output directory plus a `root.zig` that
//! forwards them all; the build compiles that root against the real
//! `fucina`/`fucina_llm` modules and runs it, so a doc snippet that stops
//! compiling or asserting fails the build — the doc-check counterpart for
//! snippet rot.
//!
//! Conventions (rendered-invisible HTML comment markers in the doc):
//!
//! - Every emitted file gets the implicit prelude the doc assumes:
//!   `const std`, `const fucina = @import("fucina")`, and
//!   `const llm = @import("fucina_llm")`.
//! - `<!-- snippet: helper -->` on the line before a ```zig fence marks a
//!   non-test block (an Op/Spec/fn definition the prose introduces) that
//!   later snippets reference. Helpers accumulate and are prepended to
//!   every following test file until the next `## ` chapter heading.
//! - `<!-- snippet: skip -->` on the line before a fence excludes a
//!   test-shaped block that cannot run hermetically (model assets, env).
//!
//! Usage: gen_snippet_tests <REFERENCE.md path> <output dir>

const std = @import("std");

/// Implicit prelude the doc's snippets assume, one entry per name; an entry
/// is emitted only when the snippet (helpers included) does not declare
/// that name itself (e.g. the §1.4 first program shows its own imports).
const prelude_decls = [_]struct { name: []const u8, decl: []const u8 }{
    .{ .name = "std", .decl = "const std = @import(\"std\");" },
    .{ .name = "fucina", .decl = "const fucina = @import(\"fucina\");" },
    .{ .name = "llm", .decl = "const llm = @import(\"fucina_llm\");" },
    .{ .name = "optim", .decl = "const optim = fucina.optim;" },
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const io = init.io;

    const args = try init.minimal.args.toSlice(allocator);
    if (args.len != 3) {
        var stderr_buffer: [256]u8 = undefined;
        var stderr_writer = std.Io.File.stderr().writer(io, &stderr_buffer);
        const stderr = &stderr_writer.interface;
        defer stderr.flush() catch {};
        try stderr.print("usage: gen_snippet_tests <REFERENCE.md> <output dir>\n", .{});
        return error.BadUsage;
    }
    const doc_path = args[1];
    const out_path = args[2];

    const contents = try std.Io.Dir.cwd().readFileAlloc(io, doc_path, allocator, .limited(16 * 1024 * 1024));
    try std.Io.Dir.cwd().createDirPath(io, out_path);

    var helpers: std.ArrayList(Block) = .empty;
    var emitted: std.ArrayList([]const u8) = .empty;

    var pending_marker: Marker = .none;
    var in_fence = false;
    var fence_marker: Marker = .none;
    var block_start_line: usize = 0;
    var block: std.ArrayList(u8) = .empty;

    var line_no: usize = 0;
    var lines = std.mem.splitScalar(u8, contents, '\n');
    while (lines.next()) |line| {
        line_no += 1;
        if (!in_fence) {
            if (std.mem.startsWith(u8, line, "## ")) {
                // Chapter boundary: helper scope ends.
                helpers.clearRetainingCapacity();
                pending_marker = .none;
            } else if (std.mem.startsWith(u8, line, "<!-- snippet: helper -->")) {
                pending_marker = .helper;
            } else if (std.mem.startsWith(u8, line, "<!-- snippet: skip -->")) {
                pending_marker = .skip;
            } else if (std.mem.startsWith(u8, line, "```zig")) {
                in_fence = true;
                fence_marker = pending_marker;
                pending_marker = .none;
                block_start_line = line_no;
                block.clearRetainingCapacity();
            } else if (std.mem.trim(u8, line, " \t").len != 0) {
                // Any other non-blank prose detaches a dangling marker.
                pending_marker = .none;
            }
            continue;
        }
        if (std.mem.eql(u8, std.mem.trimEnd(u8, line, " \t"), "```")) {
            in_fence = false;
            const text = try allocator.dupe(u8, block.items);
            const is_test = hasTopLevelTest(text);
            switch (fence_marker) {
                .skip => {},
                .helper => try helpers.append(allocator, .{ .line = block_start_line, .text = text }),
                .none => if (is_test) {
                    const name = try std.fmt.allocPrint(allocator, "s_{d:0>5}.zig", .{block_start_line});
                    var file_body: std.ArrayList(u8) = .empty;
                    const head = try std.fmt.allocPrint(allocator, "//! {s}:{d}\n", .{ doc_path, block_start_line });
                    try file_body.appendSlice(allocator, head);
                    for (prelude_decls) |entry| {
                        if (declaresName(text, entry.name)) continue;
                        var declared_by_helper = false;
                        for (helpers.items) |helper| declared_by_helper = declared_by_helper or declaresName(helper.text, entry.name);
                        if (declared_by_helper) continue;
                        try file_body.appendSlice(allocator, entry.decl);
                        try file_body.append(allocator, '\n');
                    }
                    try file_body.append(allocator, '\n');
                    for (helpers.items) |helper| {
                        const helper_head = try std.fmt.allocPrint(allocator, "// helper from {s}:{d}\n", .{ doc_path, helper.line });
                        try file_body.appendSlice(allocator, helper_head);
                        try file_body.appendSlice(allocator, helper.text);
                        try file_body.append(allocator, '\n');
                    }
                    try file_body.appendSlice(allocator, text);
                    try writeOut(io, allocator, out_path, name, file_body.items);
                    try emitted.append(allocator, name);
                },
            }
            continue;
        }
        try block.appendSlice(allocator, line);
        try block.append(allocator, '\n');
    }

    var root: std.ArrayList(u8) = .empty;
    try root.appendSlice(allocator, "//! Generated snippet-test root; one import per runnable doc snippet.\ntest {\n");
    for (emitted.items) |name| {
        const import_line = try std.fmt.allocPrint(allocator, "    _ = @import(\"{s}\");\n", .{name});
        try root.appendSlice(allocator, import_line);
    }
    try root.appendSlice(allocator, "}\n");
    try writeOut(io, allocator, out_path, "root.zig", root.items);

    var stdout_buffer: [256]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};
    try stdout.print("snippet-gen: {d} runnable snippets extracted from {s}\n", .{ emitted.items.len, doc_path });
}

const Marker = enum { none, helper, skip };

const Block = struct {
    line: usize,
    text: []const u8,
};

fn writeOut(io: std.Io, allocator: std.mem.Allocator, dir: []const u8, name: []const u8, data: []const u8) !void {
    const path = try std.fs.path.join(allocator, &.{ dir, name });
    defer allocator.free(path);
    var file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buffer: [64 * 1024]u8 = undefined;
    var writer = file.writer(io, &buffer);
    try writer.interface.writeAll(data);
    try writer.interface.flush();
}

/// A block is runnable iff it declares a column-0 NAMED `test "..."` (a
/// bare `test {` stanza is the §2.7 forwarding-pattern illustration, not a
/// runnable example).
fn hasTopLevelTest(text: []const u8) bool {
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        if (std.mem.startsWith(u8, line, "test \"")) return true;
    }
    return false;
}

/// True when the block itself declares column-0 `const <name> =` (with or
/// without `pub`), so the prelude must not re-declare it.
fn declaresName(text: []const u8, name: []const u8) bool {
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        var rest = line;
        if (std.mem.startsWith(u8, rest, "pub ")) rest = rest[4..];
        if (!std.mem.startsWith(u8, rest, "const ")) continue;
        rest = rest[6..];
        if (std.mem.startsWith(u8, rest, name) and
            std.mem.startsWith(u8, std.mem.trimStart(u8, rest[name.len..], " "), "=")) return true;
    }
    return false;
}
