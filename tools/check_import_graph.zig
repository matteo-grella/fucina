//! Production import-graph checker for Fucina.
//!
//! Sentrux scans the whole repository, including sibling test files. This tool
//! enforces the stricter production invariant: non-test `src/**/*.zig` local
//! imports must have no nontrivial strongly-connected components.
//!
//! Test awareness inside production files: an `@import` is counted only when
//! it is reachable from production code. Skipped are (a) imports inside `test`
//! declarations, and (b) imports inside non-pub file-scope decls that no
//! production decl references (e.g. a private test-only helper fn).
//! Reachability is approximated per file by matching identifier tokens against
//! file-scope decl names, seeded from pub decls and unnamed non-test roots
//! (comptime blocks, container fields). Shadowed names can only over-count
//! edges; references made through strings (`@field`) are not seen. Files that
//! fail to parse conservatively count every `@import`.

const std = @import("std");

const Allocator = std.mem.Allocator;

const Error = error{
    ImportCycleDetected,
};

const FileInfo = struct {
    path: []const u8,
    edges: std.ArrayListUnmanaged(usize) = .empty,

    fn deinit(self: *FileInfo, allocator: Allocator) void {
        allocator.free(self.path);
        self.edges.deinit(allocator);
        self.* = undefined;
    }
};

const Graph = struct {
    files: std.ArrayListUnmanaged(FileInfo) = .empty,
    index_by_path: std.StringHashMapUnmanaged(usize) = .empty,

    fn deinit(self: *Graph, allocator: Allocator) void {
        for (self.files.items) |*file| file.deinit(allocator);
        self.files.deinit(allocator);
        self.index_by_path.deinit(allocator);
        self.* = undefined;
    }
};

const Tarjan = struct {
    graph: *const Graph,
    index: usize = 0,
    node_index: []?usize,
    lowlink: []usize,
    on_stack: []bool,
    stack: std.ArrayListUnmanaged(usize) = .empty,
    cycles: std.ArrayListUnmanaged([]usize) = .empty,
    allocator: Allocator,

    fn init(allocator: Allocator, graph: *const Graph) !Tarjan {
        const n = graph.files.items.len;
        const node_index = try allocator.alloc(?usize, n);
        errdefer allocator.free(node_index);
        @memset(node_index, null);
        const lowlink = try allocator.alloc(usize, n);
        errdefer allocator.free(lowlink);
        const on_stack = try allocator.alloc(bool, n);
        errdefer allocator.free(on_stack);
        @memset(on_stack, false);
        return .{
            .graph = graph,
            .node_index = node_index,
            .lowlink = lowlink,
            .on_stack = on_stack,
            .allocator = allocator,
        };
    }

    fn deinit(self: *Tarjan) void {
        for (self.cycles.items) |cycle| self.allocator.free(cycle);
        self.cycles.deinit(self.allocator);
        self.stack.deinit(self.allocator);
        self.allocator.free(self.node_index);
        self.allocator.free(self.lowlink);
        self.allocator.free(self.on_stack);
        self.* = undefined;
    }

    fn run(self: *Tarjan) !void {
        for (0..self.graph.files.items.len) |node| {
            if (self.node_index[node] == null) try self.visit(node);
        }
    }

    fn visit(self: *Tarjan, node: usize) !void {
        self.node_index[node] = self.index;
        self.lowlink[node] = self.index;
        self.index += 1;
        try self.stack.append(self.allocator, node);
        self.on_stack[node] = true;

        for (self.graph.files.items[node].edges.items) |next| {
            if (self.node_index[next] == null) {
                try self.visit(next);
                self.lowlink[node] = @min(self.lowlink[node], self.lowlink[next]);
            } else if (self.on_stack[next]) {
                self.lowlink[node] = @min(self.lowlink[node], self.node_index[next].?);
            }
        }

        if (self.lowlink[node] != self.node_index[node].?) return;

        var component: std.ArrayListUnmanaged(usize) = .empty;
        errdefer component.deinit(self.allocator);
        while (true) {
            const member = self.stack.pop().?;
            self.on_stack[member] = false;
            try component.append(self.allocator, member);
            if (member == node) break;
        }

        if (component.items.len > 1 or hasSelfEdge(self.graph, node)) {
            const owned = try component.toOwnedSlice(self.allocator);
            try self.cycles.append(self.allocator, owned);
        } else {
            component.deinit(self.allocator);
        }
    }
};

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

    var graph: Graph = .{};
    defer graph.deinit(allocator);

    try collectFiles(allocator, io, &graph);
    try collectEdges(allocator, io, &graph);

    var tarjan = try Tarjan.init(allocator, &graph);
    defer tarjan.deinit();
    try tarjan.run();

    const edge_count = countEdges(&graph);
    if (tarjan.cycles.items.len == 0) {
        try stdout.print(
            "production import graph: {d} files, {d} edges, 0 SCCs\n",
            .{ graph.files.items.len, edge_count },
        );
        return;
    }

    try stderr.print(
        "production import graph: {d} files, {d} edges, {d} SCC(s)\n",
        .{ graph.files.items.len, edge_count, tarjan.cycles.items.len },
    );
    for (tarjan.cycles.items, 0..) |cycle, i| {
        try stderr.print("SCC {d}:\n", .{i + 1});
        for (cycle) |node| {
            try stderr.print("  {s}\n", .{graph.files.items[node].path});
        }
    }
    return Error.ImportCycleDetected;
}

fn collectFiles(allocator: Allocator, io: std.Io, graph: *Graph) !void {
    var src_dir = try std.Io.Dir.cwd().openDir(io, "src", .{ .iterate = true });
    defer src_dir.close(io);

    var walker = try src_dir.walk(allocator);
    defer walker.deinit();

    while (try walker.next(io)) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.path, ".zig")) continue;
        if (isTestPath(entry.path)) continue;

        const path = try std.fmt.allocPrint(allocator, "src/{s}", .{entry.path});
        errdefer allocator.free(path);
        try graph.index_by_path.put(allocator, path, graph.files.items.len);
        try graph.files.append(allocator, .{ .path = path });
    }
}

fn collectEdges(allocator: Allocator, io: std.Io, graph: *Graph) !void {
    for (graph.files.items, 0..) |*file, file_index| {
        const contents = try readFileSentinel(allocator, io, file.path);
        defer allocator.free(contents);
        var ast = try std.zig.Ast.parse(allocator, contents, .zig);
        defer ast.deinit(allocator);

        var spans: std.ArrayListUnmanaged(TokenSpan) = .empty;
        defer spans.deinit(allocator);
        try collectProductionSpans(allocator, &ast, &spans);

        for (spans.items) |span| {
            var tok = span.start;
            while (tok + 2 <= span.end) : (tok += 1) {
                if (ast.tokenTag(tok) != .builtin) continue;
                if (!std.mem.eql(u8, ast.tokenSlice(tok), "@import")) continue;
                if (ast.tokenTag(tok + 1) != .l_paren) continue;
                if (ast.tokenTag(tok + 2) != .string_literal) continue;

                const raw = ast.tokenSlice(tok + 2);
                const imported = std.zig.string_literal.parseAlloc(allocator, raw) catch continue;
                defer allocator.free(imported);
                const resolved = try resolveLocalImport(allocator, file.path, imported) orelse continue;
                defer allocator.free(resolved);
                const target_index = graph.index_by_path.get(resolved) orelse continue;
                try addEdge(allocator, &graph.files.items[file_index], target_index);
            }
        }
    }
}

const TokenSpan = struct {
    start: std.zig.Ast.TokenIndex,
    end: std.zig.Ast.TokenIndex, // inclusive
};

const DeclInfo = struct {
    name: ?[]const u8, // borrows the Ast source
    span: TokenSpan,
    production: bool,
};

/// Collect the token spans of the file-scope decls that belong to the
/// production build (see the header for the skip rules and approximation).
fn collectProductionSpans(
    allocator: Allocator,
    ast: *const std.zig.Ast,
    spans: *std.ArrayListUnmanaged(TokenSpan),
) !void {
    if (ast.errors.len != 0) {
        try spans.append(allocator, .{ .start = 0, .end = @intCast(ast.tokens.len - 1) });
        return;
    }

    var decls: std.ArrayListUnmanaged(DeclInfo) = .empty;
    defer decls.deinit(allocator);
    var decl_by_name: std.StringHashMapUnmanaged(usize) = .empty;
    defer decl_by_name.deinit(allocator);

    for (ast.rootDecls()) |node| {
        if (ast.nodeTag(node) == .test_decl) continue;
        const span: TokenSpan = .{ .start = ast.firstToken(node), .end = ast.lastToken(node) };
        var buf: [1]std.zig.Ast.Node.Index = undefined;
        const decl: DeclInfo = if (ast.fullVarDecl(node)) |var_decl| .{
            .name = ast.tokenSlice(var_decl.ast.mut_token + 1),
            .span = span,
            .production = var_decl.visib_token != null,
        } else if (ast.fullFnProto(&buf, node)) |fn_proto| .{
            .name = if (fn_proto.name_token) |name_token| ast.tokenSlice(name_token) else null,
            .span = span,
            .production = fn_proto.visib_token != null,
        } else .{
            // comptime blocks, container fields, ...: unconditional roots.
            .name = null,
            .span = span,
            .production = true,
        };
        if (decl.name) |name| try decl_by_name.put(allocator, name, decls.items.len);
        try decls.append(allocator, decl);
    }

    // Fixpoint over identifier references: a non-pub decl referenced from a
    // production decl is production itself. Each decl is scanned at most once.
    var worklist: std.ArrayListUnmanaged(usize) = .empty;
    defer worklist.deinit(allocator);
    for (decls.items, 0..) |decl, i| {
        if (decl.production) try worklist.append(allocator, i);
    }
    while (worklist.pop()) |i| {
        const span = decls.items[i].span;
        var tok = span.start;
        while (tok <= span.end) : (tok += 1) {
            if (ast.tokenTag(tok) != .identifier) continue;
            const target = decl_by_name.get(ast.tokenSlice(tok)) orelse continue;
            if (decls.items[target].production) continue;
            decls.items[target].production = true;
            try worklist.append(allocator, target);
        }
    }

    for (decls.items) |decl| {
        if (decl.production) try spans.append(allocator, decl.span);
    }
}

fn readFileSentinel(allocator: Allocator, io: std.Io, path: []const u8) ![:0]u8 {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(16 * 1024 * 1024));
    defer allocator.free(bytes);
    const out = try allocator.allocSentinel(u8, bytes.len, 0);
    @memcpy(out[0..bytes.len], bytes);
    return out;
}

fn resolveLocalImport(allocator: Allocator, file_path: []const u8, imported: []const u8) !?[]const u8 {
    if (!std.mem.endsWith(u8, imported, ".zig")) return null;
    if (imported.len == 0 or imported[0] == '/') return null;

    const dirname = std.fs.path.dirname(file_path) orelse "";
    const joined = if (dirname.len == 0)
        try allocator.dupe(u8, imported)
    else
        try std.fmt.allocPrint(allocator, "{s}/{s}", .{ dirname, imported });
    defer allocator.free(joined);

    return try normalizeRelativePath(allocator, joined);
}

fn normalizeRelativePath(allocator: Allocator, path: []const u8) !?[]const u8 {
    var parts: std.ArrayListUnmanaged([]const u8) = .empty;
    defer parts.deinit(allocator);

    var it = std.mem.splitScalar(u8, path, '/');
    while (it.next()) |part| {
        if (part.len == 0 or std.mem.eql(u8, part, ".")) continue;
        if (std.mem.eql(u8, part, "..")) {
            if (parts.items.len == 0) return null;
            _ = parts.pop();
        } else {
            try parts.append(allocator, part);
        }
    }

    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(allocator);
    for (parts.items, 0..) |part, i| {
        if (i != 0) try out.append(allocator, '/');
        try out.appendSlice(allocator, part);
    }
    return try out.toOwnedSlice(allocator);
}

fn addEdge(allocator: Allocator, file: *FileInfo, target_index: usize) !void {
    for (file.edges.items) |existing| {
        if (existing == target_index) return;
    }
    try file.edges.append(allocator, target_index);
}

fn isTestPath(path: []const u8) bool {
    return std.mem.endsWith(u8, path, "_tests.zig") or
        std.mem.endsWith(u8, path, "_test.zig");
}

fn hasSelfEdge(graph: *const Graph, node: usize) bool {
    for (graph.files.items[node].edges.items) |target| {
        if (target == node) return true;
    }
    return false;
}

fn countEdges(graph: *const Graph) usize {
    var total: usize = 0;
    for (graph.files.items) |file| total += file.edges.items.len;
    return total;
}
