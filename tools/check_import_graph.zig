//! Production import-graph checker for Fucina.
//!
//! Sentrux scans the whole repository, including sibling test files. This tool
//! enforces the stricter PRODUCTION invariants over the non-test import
//! graph of `src/`, `examples/`, `apps/`, `bench/`, and `tools/`, in-tree so
//! `zig build arch-check` is the gate: no forbidden strongly-connected
//! components, no band inversions, and no unforwarded sibling test files.
//!
//! First invariant, cycles. A nontrivial SCC is permitted only when (a)
//! every member is in the same band and (b) at least one member is a
//! directory root of the others: a file `P.zig` with another member under
//! `P/` (the "struct body in the root, methods in the children" shape;
//! `src/exec.zig` with `src/exec/*.zig`, as `std.zig` with
//! `std/array_list.zig` in Zig's own std). Every other SCC is an error:
//! children cycling among themselves without their root, or any cycle
//! that crosses a band.
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
//!
//! Second invariant — direction. Every production file belongs to exactly one
//! band of docs/ARCHITECTURE.md's Layer Stack (`band_table` below), and every
//! production import must point at a band at or below the importer's. A file
//! in no band is an error too: a new `src/` root cannot silently inherit
//! whatever band its neighbours happen to have.
//!
//! Third invariant — test-file forwarding: every test file (`*_tests.zig` /
//! `*_test.zig`, and every file under a `<name>_tests/` directory suite)
//! must be reachable from a production forwarding stanza
//! (`test { _ = @import("x_tests.zig"); }`), directly or through other test
//! files (a directory suite's shared helper is imported by its siblings,
//! not by production code). Zig's lazy analysis means an unforwarded test
//! file is silently absent from `zig build test` — it neither runs nor even
//! compiles. The scan covers ALL tokens of every production file (test
//! decls included, since that is exactly where the stanzas live), then
//! propagates through test-to-test imports.

const std = @import("std");

const Allocator = std.mem.Allocator;

const Error = error{
    ImportCycleDetected,
    TestFileNotForwarded,
    BandInversionDetected,
    FileOutsideBandTable,
};

/// The Layer Stack of docs/ARCHITECTURE.md, top-down. A band may depend only
/// on bands at or below it, so a production import is legal exactly when the
/// target's ordinal is >= the importer's. Declaration order IS the contract:
/// do not reorder without changing the documented stack.
const Band = enum {
    apps,
    serving,
    models,
    facade,
    ag, // ag + training/serialization + model I/O
    tagged,
    moe,
    exec,
    store,
    backend,
    tags,
    tensor,
    primitives,
    core,

    fn label(self: Band) []const u8 {
        return switch (self) {
            .apps => "apps",
            .serving => "serving",
            .models => "models",
            .facade => "facade",
            .ag => "ag + training/serialization",
            .tagged => "tagged",
            .moe => "moe",
            .exec => "exec",
            .store => "store",
            .backend => "backend",
            .tags => "tags",
            .tensor => "tensor",
            .primitives => "primitives",
            .core => "core",
        };
    }
};

/// Band membership. An entry ending in `/` claims a directory subtree;
/// anything else is an exact path. Longest match wins, so `src/ag.zig` and
/// `src/ag/` can both appear. Every production file must match exactly one
/// band — a new `src/` root that nobody added here fails the check rather
/// than silently acquiring whatever band its neighbours have.
const band_table = [_]struct { path: []const u8, band: Band }{
    .{ .path = "src/bench_raw.zig", .band = .apps },
    .{ .path = "src/x86dot_check.zig", .band = .apps },
    .{ .path = "examples/", .band = .apps },
    .{ .path = "apps/", .band = .apps },
    .{ .path = "bench/", .band = .apps },
    .{ .path = "tools/", .band = .apps },

    .{ .path = "src/serving.zig", .band = .serving },
    .{ .path = "src/serving/", .band = .serving },

    .{ .path = "src/models.zig", .band = .models },
    .{ .path = "src/models/", .band = .models },

    .{ .path = "src/fucina.zig", .band = .facade },

    .{ .path = "src/ag.zig", .band = .ag },
    .{ .path = "src/ag/", .band = .ag },
    .{ .path = "src/optim.zig", .band = .ag },
    .{ .path = "src/optim/", .band = .ag },
    .{ .path = "src/es.zig", .band = .ag },
    .{ .path = "src/es/", .band = .ag },
    .{ .path = "src/ptqtp.zig", .band = .ag },
    .{ .path = "src/gguf.zig", .band = .ag },
    .{ .path = "src/gguf/", .band = .ag },
    .{ .path = "src/gguf_meta.zig", .band = .ag },
    .{ .path = "src/ptqtp_gguf.zig", .band = .ag },
    .{ .path = "src/lora.zig", .band = .ag },
    .{ .path = "src/safetensors.zig", .band = .ag },
    .{ .path = "src/state_dict.zig", .band = .ag },
    .{ .path = "src/training_checkpoint.zig", .band = .ag },
    .{ .path = "src/param_registry.zig", .band = .ag },
    .{ .path = "src/weights.zig", .band = .ag },
    .{ .path = "src/weights/", .band = .ag },

    .{ .path = "src/tag_ops.zig", .band = .tagged },

    .{ .path = "src/moe.zig", .band = .moe },
    .{ .path = "src/moe/", .band = .moe },

    .{ .path = "src/exec.zig", .band = .exec },
    .{ .path = "src/exec/", .band = .exec },

    .{ .path = "src/store/", .band = .store },

    .{ .path = "src/backend.zig", .band = .backend },
    .{ .path = "src/backend/", .band = .backend },

    .{ .path = "src/tags.zig", .band = .tags },
    .{ .path = "src/tensor.zig", .band = .tensor },

    .{ .path = "src/thread.zig", .band = .primitives },
    .{ .path = "src/parallel.zig", .band = .primitives },
    .{ .path = "src/tuning.zig", .band = .primitives },

    .{ .path = "src/dtype.zig", .band = .core },
    .{ .path = "src/shape.zig", .band = .core },
    .{ .path = "src/storage.zig", .band = .core },
    .{ .path = "src/accelerator.zig", .band = .core },
    .{ .path = "src/rng.zig", .band = .core },
    .{ .path = "src/fpenv.zig", .band = .core },
    .{ .path = "src/caching_allocator.zig", .band = .core },
    .{ .path = "src/streamconv.zig", .band = .core },
};

fn bandOf(path: []const u8) ?Band {
    var best: ?Band = null;
    var best_len: usize = 0;
    for (band_table) |entry| {
        const matches = if (entry.path[entry.path.len - 1] == '/')
            std.mem.startsWith(u8, path, entry.path)
        else
            std.mem.eql(u8, path, entry.path);
        if (matches and entry.path.len > best_len) {
            best = entry.band;
            best_len = entry.path.len;
        }
    }
    return best;
}

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

const TestFiles = struct {
    /// test-file path -> has an incoming `@import` from a non-test src file.
    forwarded_by_path: std.StringHashMapUnmanaged(bool) = .empty,

    fn deinit(self: *TestFiles, allocator: Allocator) void {
        var it = self.forwarded_by_path.keyIterator();
        while (it.next()) |key| allocator.free(key.*);
        self.forwarded_by_path.deinit(allocator);
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

    // Failure diagnostics go through std.debug.print (unbuffered fd 2): the
    // build runner relays that channel on a failing run step, where the
    // buffered writer's late flush is not shown.

    var graph: Graph = .{};
    defer graph.deinit(allocator);
    var test_files: TestFiles = .{};
    defer test_files.deinit(allocator);

    try collectFiles(allocator, io, &graph, &test_files);
    try collectEdges(allocator, io, &graph, &test_files);
    try propagateTestForwarding(allocator, io, &test_files);

    var tarjan = try Tarjan.init(allocator, &graph);
    defer tarjan.deinit();
    try tarjan.run();

    var root_anchored: usize = 0;
    var forbidden: std.ArrayListUnmanaged([]usize) = .empty;
    defer forbidden.deinit(allocator);
    for (tarjan.cycles.items) |cycle| {
        if (isRootAnchoredSameBand(&graph, cycle)) root_anchored += 1 else try forbidden.append(allocator, cycle);
    }

    var unforwarded: std.ArrayListUnmanaged([]const u8) = .empty;
    defer unforwarded.deinit(allocator);
    var it = test_files.forwarded_by_path.iterator();
    while (it.next()) |entry| {
        if (!entry.value_ptr.*) try unforwarded.append(allocator, entry.key_ptr.*);
    }
    std.mem.sort([]const u8, unforwarded.items, {}, stringLessThan);

    var unbanded: std.ArrayListUnmanaged([]const u8) = .empty;
    defer unbanded.deinit(allocator);
    var inversions: std.ArrayListUnmanaged(Inversion) = .empty;
    defer inversions.deinit(allocator);
    try checkBands(allocator, &graph, &unbanded, &inversions);

    const edge_count = countEdges(&graph);
    if (forbidden.items.len == 0 and unforwarded.items.len == 0 and
        unbanded.items.len == 0 and inversions.items.len == 0)
    {
        try stdout.print(
            "production import graph: {d} files, {d} edges, {d} root-anchored SCC(s), 0 forbidden SCCs, 0 band inversions; {d} test files, all forwarded\n",
            .{ graph.files.items.len, edge_count, root_anchored, test_files.forwarded_by_path.count() },
        );
        return;
    }

    if (unbanded.items.len != 0) {
        std.debug.print(
            "{d} production file(s) in no band of the Layer Stack (docs/ARCHITECTURE.md):\n",
            .{unbanded.items.len},
        );
        for (unbanded.items) |path| {
            std.debug.print("  {s} (add it to band_table in this tool and to the Layer Stack table)\n", .{path});
        }
    }
    if (inversions.items.len != 0) {
        std.debug.print("{d} band inversion(s) — a band may depend only on bands at or below it:\n", .{inversions.items.len});
        for (inversions.items) |inv| {
            std.debug.print("  {s} [{s}] imports {s} [{s}]\n", .{
                inv.from, inv.from_band.label(), inv.to, inv.to_band.label(),
            });
        }
    }

    if (forbidden.items.len != 0) {
        std.debug.print(
            "production import graph: {d} files, {d} edges, {d} root-anchored SCC(s), {d} forbidden SCC(s)\n",
            .{ graph.files.items.len, edge_count, root_anchored, forbidden.items.len },
        );
        for (forbidden.items, 0..) |cycle, i| {
            std.debug.print("forbidden SCC {d} (same band + a directory root among the members would permit it):\n", .{i + 1});
            for (cycle) |node| {
                std.debug.print("  {s}\n", .{graph.files.items[node].path});
            }
        }
    }
    if (unforwarded.items.len != 0) {
        std.debug.print(
            "{d} test file(s) with no incoming @import from any production src file — silently absent from `zig build test`:\n",
            .{unforwarded.items.len},
        );
        for (unforwarded.items) |path| {
            std.debug.print("  {s} (add `test {{ _ = @import(\"...\"); }}` to its production sibling)\n", .{path});
        }
    }
    if (forbidden.items.len != 0) return Error.ImportCycleDetected;
    if (unbanded.items.len != 0) return Error.FileOutsideBandTable;
    if (inversions.items.len != 0) return Error.BandInversionDetected;
    return Error.TestFileNotForwarded;
}

/// The one permitted SCC shape: every member in one band, and some member
/// `P.zig` is the directory root of another member under `P/`. This is the
/// "struct body in the root, methods in the children" layout, where the
/// children name the root's type and the root aliases their functions.
fn isRootAnchoredSameBand(graph: *const Graph, cycle: []const usize) bool {
    const first_band = bandOf(graph.files.items[cycle[0]].path) orelse return false;
    for (cycle) |node| {
        const band = bandOf(graph.files.items[node].path) orelse return false;
        if (band != first_band) return false;
    }
    for (cycle) |root| {
        const root_path = graph.files.items[root].path;
        if (!std.mem.endsWith(u8, root_path, ".zig")) continue;
        const dir_prefix = root_path[0 .. root_path.len - ".zig".len];
        for (cycle) |child| {
            if (child == root) continue;
            const child_path = graph.files.items[child].path;
            if (child_path.len > dir_prefix.len + 1 and
                std.mem.startsWith(u8, child_path, dir_prefix) and
                child_path[dir_prefix.len] == '/') return true;
        }
    }
    return false;
}

const Inversion = struct {
    from: []const u8,
    from_band: Band,
    to: []const u8,
    to_band: Band,
};

/// Second production invariant: direction. Every file belongs to exactly one
/// band of the Layer Stack, and every production import points at a band at
/// or below the importer's. This is the contract docs/ARCHITECTURE.md states;
/// before it lived here it was checked only by an out-of-tree lint.
fn checkBands(
    allocator: Allocator,
    graph: *const Graph,
    unbanded: *std.ArrayListUnmanaged([]const u8),
    inversions: *std.ArrayListUnmanaged(Inversion),
) !void {
    for (graph.files.items) |file| {
        if (bandOf(file.path) == null) try unbanded.append(allocator, file.path);
    }
    if (unbanded.items.len != 0) {
        std.mem.sort([]const u8, unbanded.items, {}, stringLessThan);
        return; // band ranks are meaningless until every file has one
    }
    for (graph.files.items) |file| {
        const from = bandOf(file.path).?;
        for (file.edges.items) |target| {
            const to_path = graph.files.items[target].path;
            const to = bandOf(to_path).?;
            if (@intFromEnum(to) < @intFromEnum(from)) try inversions.append(allocator, .{
                .from = file.path,
                .from_band = from,
                .to = to_path,
                .to_band = to,
            });
        }
    }
}

/// Roots the checker walks. `src` is the library; the apps-band roots are
/// scanned too, because an example is not exempt from having no import
/// cycles and no silently-dead test file — several of them are complete
/// model ports, not snippets.
const scan_roots = [_][]const u8{ "src", "examples", "apps", "bench", "tools" };

fn collectFiles(allocator: Allocator, io: std.Io, graph: *Graph, test_files: *TestFiles) !void {
    for (scan_roots) |root| {
        var dir = std.Io.Dir.cwd().openDir(io, root, .{ .iterate = true }) catch continue;
        defer dir.close(io);

        var walker = try dir.walk(allocator);
        defer walker.deinit();

        while (try walker.next(io)) |entry| {
            if (entry.kind != .file) continue;
            if (!std.mem.endsWith(u8, entry.path, ".zig")) continue;
            const path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ root, entry.path });
            errdefer allocator.free(path);
            if (isTestPath(entry.path) and !isProgramRoot(allocator, io, path)) {
                try test_files.forwarded_by_path.put(allocator, path, false);
                continue;
            }
            try graph.index_by_path.put(allocator, path, graph.files.items.len);
            try graph.files.append(allocator, .{ .path = path });
        }
    }
}

fn collectEdges(allocator: Allocator, io: std.Io, graph: *Graph, test_files: *TestFiles) !void {
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

        // Test-file forwarding: scan ALL tokens (test decls included — that is
        // where the forwarding stanzas live) for imports that resolve to a
        // test file, and mark those files as wired into a test root.
        if (ast.tokens.len < 3) continue;
        var tok: std.zig.Ast.TokenIndex = 0;
        const last_start: std.zig.Ast.TokenIndex = @intCast(ast.tokens.len - 2);
        while (tok < last_start) : (tok += 1) {
            if (ast.tokenTag(tok) != .builtin) continue;
            if (!std.mem.eql(u8, ast.tokenSlice(tok), "@import")) continue;
            if (ast.tokenTag(tok + 1) != .l_paren) continue;
            if (ast.tokenTag(tok + 2) != .string_literal) continue;

            const raw = ast.tokenSlice(tok + 2);
            const imported = std.zig.string_literal.parseAlloc(allocator, raw) catch continue;
            defer allocator.free(imported);
            if (!isTestPath(imported)) continue;
            const resolved = try resolveLocalImport(allocator, file.path, imported) orelse continue;
            defer allocator.free(resolved);
            if (test_files.forwarded_by_path.getPtr(resolved)) |forwarded| forwarded.* = true;
        }
    }
}

/// Test-to-test forwarding propagation: a test file counts as forwarded when
/// it is reachable from a production forwarding stanza THROUGH other test
/// files. This is the directory-suite shape: `src/ag/tensor.zig` forwards
/// `tensor_tests/<domain>.zig`, and those files file-scope-import the shared
/// `tensor_tests/util.zig` helper, which no production file names directly.
fn propagateTestForwarding(allocator: Allocator, io: std.Io, test_files: *TestFiles) !void {
    const TestEdge = struct { from: []const u8, to: []const u8 };
    var edges: std.ArrayListUnmanaged(TestEdge) = .empty;
    defer {
        for (edges.items) |edge| allocator.free(edge.to);
        edges.deinit(allocator);
    }

    var key_it = test_files.forwarded_by_path.keyIterator();
    while (key_it.next()) |key| {
        const contents = readFileSentinel(allocator, io, key.*) catch continue;
        defer allocator.free(contents);
        var ast = try std.zig.Ast.parse(allocator, contents, .zig);
        defer ast.deinit(allocator);
        if (ast.tokens.len < 3) continue;
        var tok: std.zig.Ast.TokenIndex = 0;
        const last_start: std.zig.Ast.TokenIndex = @intCast(ast.tokens.len - 2);
        while (tok < last_start) : (tok += 1) {
            if (ast.tokenTag(tok) != .builtin) continue;
            if (!std.mem.eql(u8, ast.tokenSlice(tok), "@import")) continue;
            if (ast.tokenTag(tok + 1) != .l_paren) continue;
            if (ast.tokenTag(tok + 2) != .string_literal) continue;

            const raw = ast.tokenSlice(tok + 2);
            const imported = std.zig.string_literal.parseAlloc(allocator, raw) catch continue;
            defer allocator.free(imported);
            const resolved = try resolveLocalImport(allocator, key.*, imported) orelse continue;
            if (test_files.forwarded_by_path.contains(resolved)) {
                try edges.append(allocator, .{ .from = key.*, .to = resolved });
            } else {
                allocator.free(resolved);
            }
        }
    }

    var changed = true;
    while (changed) {
        changed = false;
        for (edges.items) |edge| {
            if (!(test_files.forwarded_by_path.get(edge.from) orelse false)) continue;
            const to_forwarded = test_files.forwarded_by_path.getPtr(edge.to) orelse continue;
            if (!to_forwarded.*) {
                to_forwarded.* = true;
                changed = true;
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
    // Sibling suites (`x_tests.zig` / `x_test.zig`) and directory suites:
    // every file under a `<name>_tests/` directory is a test file, shared
    // helpers included (src/ag/tensor_tests/). Without the directory rule
    // those files would be classified as production, so the forwarding
    // invariant would not protect them and their imports would count as
    // production edges.
    return std.mem.endsWith(u8, path, "_tests.zig") or
        std.mem.endsWith(u8, path, "_test.zig") or
        std.mem.indexOf(u8, path, "_tests/") != null;
}

/// A file whose NAME looks like a sibling test suite but which is actually
/// an executable root (`tools/gen_snippet_tests.zig` GENERATES tests; it is
/// not one). Nothing forwards a program, so the forwarding invariant must
/// not claim it. Keyed on `pub fn main` rather than a path allowlist, so a
/// new generator needs no edit here.
fn isProgramRoot(allocator: Allocator, io: std.Io, path: []const u8) bool {
    const contents = readFileSentinel(allocator, io, path) catch return false;
    defer allocator.free(contents);
    return std.mem.indexOf(u8, contents, "pub fn main(") != null;
}

fn stringLessThan(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.lessThan(u8, a, b);
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
