//! Cartridge fleets: per-document cartridges at scale (arXiv 2606.04557,
//! "Cartridges at Scale"). Instead of compressing a whole collection into
//! one monolithic KV prefix, every document gets its OWN cartridge; the
//! pieces this module provides make a fleet of them trainable and servable
//! on one machine:
//!
//! - `Manifest` — the fleet's on-disk record (`fleet.json`): per-document
//!   cartridge/optimizer file names, token counts, and optimizer-step
//!   counters (the budget manager's eviction currency).
//! - `Fleet` — the RAM/disk budget manager: at most `policy.budget`
//!   cartridges are RESIDENT (rows + Adam moments live, each with its own
//!   optimizer); the rest persist on disk. Every `policy.every` rounds the
//!   most-trained residents rotate out (rows AND moments saved, so an
//!   evict/reload cycle is bit-identical to never leaving memory) and the
//!   least-trained absentees rotate in — the paper's uniform-coverage
//!   policy. A per-cartridge lr warm-up ramps each (re)entrant.
//! - `EmbedIndex` — cartridge-RAG selection (`index.safetensors`):
//!   L2-normalized chunk embeddings with a chunk→document map; `topDocs`
//!   is a hand-rolled cosine top-k over the chunks, deduplicated to
//!   document ids in best-chunk order. No external retrieval stack — the
//!   embeddings come from the serving model itself (the CLI computes them),
//!   and similarity is a dot product over normalized rows.
//!
//! Composition (the serving side of a fleet: selected cartridges
//! concatenated ahead of the query) and the joint-training forward live in
//! `cartridge.zig` (`writeComposedToCache`) and the qwen3 trainer
//! (`ForwardOptions.cartridges`); the self-study loop lives in
//! `examples/cartridge_fleet/main.zig`. Design record: `docs/CARTRIDGES.md`.

const std = @import("std");
const fucina = @import("fucina");
const cartridge = @import("cartridge.zig");

const Allocator = std.mem.Allocator;
const ExecContext = fucina.ExecContext;
const optim = fucina.optim;

pub const Error = error{
    InvalidFleet,
    InvalidManifest,
    InvalidIndex,
    InvalidDoc,
    DocNotResident,
};

pub const manifest_file = "fleet.json";
pub const index_file = "index.safetensors";
const manifest_version = 1;
const read_limit: std.Io.Limit = .limited(1024 * 1024 * 1024);

/// A read-only private mapping of a whole file. Artifact retrieval goes
/// through this instead of a heap read: the budget manager reloads
/// cartridges (and their optimizer snapshots) every rotation, and the
/// loaders only stream the mapped pages once while copying rows into fresh
/// tensors — no transient whole-file allocation.
pub const MappedFile = struct {
    bytes: []align(std.heap.page_size_min) const u8,

    pub fn deinit(self: *MappedFile) void {
        std.posix.munmap(self.bytes);
        self.* = undefined;
    }
};

/// mmap `path` read-only (the safetensors loader's mapping recipe).
pub fn mmapFile(io: std.Io, path: []const u8) !MappedFile {
    var handle = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer handle.close(io);
    const stat = try handle.stat(io);
    if (stat.kind != .file) return Error.InvalidFleet;
    const len: usize = @intCast(stat.size);
    if (len == 0) return Error.InvalidFleet;
    const mapped = try std.posix.mmap(null, len, .{ .READ = true }, .{ .TYPE = .PRIVATE }, handle.handle, 0);
    return .{ .bytes = mapped };
}

// ---------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------

/// One document's fleet record. File names are leaves inside the fleet
/// directory; `steps` counts the optimizer steps this document's cartridge
/// has received (the rotation policy's ordering key).
pub const DocState = struct {
    name: []u8,
    cart_file: []u8,
    opt_file: []u8,
    tokens: usize,
    steps: u64,
};

/// The fleet's persistent record (`fleet.json`): cartridge geometry, the
/// retrieval-chunk contract, and the per-document table. Everything the
/// budget manager and the serving side need to reopen a fleet — except the
/// tensors themselves, which live in the per-document safetensors files.
pub const Manifest = struct {
    allocator: Allocator,
    p: usize,
    frozen_prefix: usize = 1,
    /// Retrieval-chunk length in tokens (the indexing/serving contract:
    /// queries embed against chunks of this size).
    embed_chunk: usize = 256,
    /// Embedding dimensionality; 0 until an index has been built.
    embed_dim: usize = 0,
    /// Global optimizer rounds taken so far (resume + rotation cadence).
    rounds: u64 = 0,
    docs: std.ArrayList(DocState) = .empty,

    pub fn init(allocator: Allocator, p: usize) Manifest {
        return .{ .allocator = allocator, .p = p };
    }

    pub fn deinit(self: *Manifest) void {
        for (self.docs.items) |*doc| {
            self.allocator.free(doc.name);
            self.allocator.free(doc.cart_file);
            self.allocator.free(doc.opt_file);
        }
        self.docs.deinit(self.allocator);
        self.* = undefined;
    }

    /// Register a document; file leaves derive from the index
    /// ("doc-<i>.safetensors" / "doc-<i>.fza"). Returns the doc id.
    pub fn addDoc(self: *Manifest, name: []const u8, tokens: usize) !usize {
        const id = self.docs.items.len;
        const owned_name = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(owned_name);
        const cart_file = try std.fmt.allocPrint(self.allocator, "doc-{d:0>3}.safetensors", .{id});
        errdefer self.allocator.free(cart_file);
        const opt_file = try std.fmt.allocPrint(self.allocator, "doc-{d:0>3}.fza", .{id});
        errdefer self.allocator.free(opt_file);
        try self.docs.append(self.allocator, .{
            .name = owned_name,
            .cart_file = cart_file,
            .opt_file = opt_file,
            .tokens = tokens,
            .steps = 0,
        });
        return id;
    }

    /// The doc id carrying `name`, or null.
    pub fn findDoc(self: *const Manifest, name: []const u8) ?usize {
        for (self.docs.items, 0..) |*doc, i| {
            if (std.mem.eql(u8, doc.name, name)) return i;
        }
        return null;
    }

    pub fn write(self: *const Manifest, writer: *std.Io.Writer) !void {
        try writer.print(
            "{{\n  \"format\": \"fucina.cartridge_fleet\",\n  \"version\": {d},\n" ++
                "  \"p\": {d},\n  \"frozen_prefix\": {d},\n  \"embed_chunk\": {d},\n" ++
                "  \"embed_dim\": {d},\n  \"rounds\": {d},\n  \"docs\": [",
            .{ manifest_version, self.p, self.frozen_prefix, self.embed_chunk, self.embed_dim, self.rounds },
        );
        for (self.docs.items, 0..) |*doc, i| {
            try writer.writeAll(if (i == 0) "\n" else ",\n");
            try writer.writeAll("    {\"name\": ");
            try writeJsonString(writer, doc.name);
            try writer.writeAll(", \"cart_file\": ");
            try writeJsonString(writer, doc.cart_file);
            try writer.writeAll(", \"opt_file\": ");
            try writeJsonString(writer, doc.opt_file);
            try writer.print(", \"tokens\": {d}, \"steps\": {d}}}", .{ doc.tokens, doc.steps });
        }
        try writer.writeAll("\n  ]\n}\n");
    }

    pub fn parse(allocator: Allocator, bytes: []const u8) !Manifest {
        var parsed = std.json.parseFromSlice(std.json.Value, allocator, bytes, .{}) catch return Error.InvalidManifest;
        defer parsed.deinit();
        if (parsed.value != .object) return Error.InvalidManifest;
        const root = parsed.value.object;
        const format = root.get("format") orelse return Error.InvalidManifest;
        if (format != .string or !std.mem.eql(u8, format.string, "fucina.cartridge_fleet")) return Error.InvalidManifest;
        if ((jsonUsize(root.get("version")) orelse 0) != manifest_version) return Error.InvalidManifest;

        var manifest = Manifest{
            .allocator = allocator,
            .p = jsonUsize(root.get("p")) orelse return Error.InvalidManifest,
            .frozen_prefix = jsonUsize(root.get("frozen_prefix")) orelse return Error.InvalidManifest,
            .embed_chunk = jsonUsize(root.get("embed_chunk")) orelse return Error.InvalidManifest,
            .embed_dim = jsonUsize(root.get("embed_dim")) orelse return Error.InvalidManifest,
            .rounds = jsonUsize(root.get("rounds")) orelse return Error.InvalidManifest,
        };
        errdefer manifest.deinit();

        const docs = root.get("docs") orelse return Error.InvalidManifest;
        if (docs != .array) return Error.InvalidManifest;
        for (docs.array.items) |entry| {
            if (entry != .object) return Error.InvalidManifest;
            const object = entry.object;
            const name = jsonString(object.get("name")) orelse return Error.InvalidManifest;
            const cart_file = jsonString(object.get("cart_file")) orelse return Error.InvalidManifest;
            const opt_file = jsonString(object.get("opt_file")) orelse return Error.InvalidManifest;
            const tokens = jsonUsize(object.get("tokens")) orelse return Error.InvalidManifest;
            const steps = jsonUsize(object.get("steps")) orelse return Error.InvalidManifest;
            const owned_name = try allocator.dupe(u8, name);
            errdefer allocator.free(owned_name);
            const owned_cart = try allocator.dupe(u8, cart_file);
            errdefer allocator.free(owned_cart);
            const owned_opt = try allocator.dupe(u8, opt_file);
            errdefer allocator.free(owned_opt);
            try manifest.docs.append(allocator, .{
                .name = owned_name,
                .cart_file = owned_cart,
                .opt_file = owned_opt,
                .tokens = tokens,
                .steps = steps,
            });
        }
        return manifest;
    }
};

fn jsonUsize(value: ?std.json.Value) ?usize {
    const v = value orelse return null;
    if (v != .integer or v.integer < 0) return null;
    return @intCast(v.integer);
}

fn jsonString(value: ?std.json.Value) ?[]const u8 {
    const v = value orelse return null;
    if (v != .string) return null;
    return v.string;
}

fn writeJsonString(writer: *std.Io.Writer, s: []const u8) !void {
    try writer.writeByte('"');
    for (s) |c| switch (c) {
        '"' => try writer.writeAll("\\\""),
        '\\' => try writer.writeAll("\\\\"),
        '\n' => try writer.writeAll("\\n"),
        '\r' => try writer.writeAll("\\r"),
        '\t' => try writer.writeAll("\\t"),
        else => if (c < 0x20) {
            try writer.print("\\u{x:0>4}", .{c});
        } else {
            try writer.writeByte(c);
        },
    };
    try writer.writeByte('"');
}

// ---------------------------------------------------------------------------
// Cartridge-RAG selection: the cosine chunk index
// ---------------------------------------------------------------------------

/// The retrieval-embedding recipe's instruction suffix. THE CONTRACT (index
/// build and query embed must match exactly, or cosine scores are
/// meaningless): tokenize the text and this suffix SEPARATELY, concatenate
/// the ids, take the trainer's `embedLastHidden` (final-norm last hidden
/// state), and let `EmbedIndex` L2-normalize + center. Measured against
/// mean pooling and bare last-token, both of which mis-rank documents
/// (docs/CARTRIDGES.md).
pub const embed_suffix = "\nIn one word, the main topic of the text above is:";

/// L2-normalize in place; an all-zero vector stays zero (its cosine against
/// anything is 0, which is the honest score).
pub fn l2Normalize(v: []f32) void {
    var sum_sq: f64 = 0;
    for (v) |x| sum_sq += @as(f64, x) * x;
    if (sum_sq == 0) return;
    const inv: f32 = @floatCast(1.0 / @sqrt(sum_sq));
    for (v) |*x| x.* *= inv;
}

/// Plain dot product — cosine similarity once both sides are L2-normalized.
pub fn dot(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    var acc: f32 = 0;
    for (a, b) |x, y| acc += x * y;
    return acc;
}

/// Chunk-embedding index for cartridge selection: `n` rows of dimension
/// `dim`, each mapped to the document it came from. Rows are L2-normalized
/// on append, then CENTERED at `finalize` — the chunk centroid is
/// subtracted from every row (and from queries) before re-normalizing.
/// Raw causal-LM embeddings are anisotropic: one dominant shared direction
/// (corpus style) crowds the cosine and drowns topical differences;
/// removing it is the standard all-but-the-top correction, measured here
/// to fix cross-document ranking (docs/CARTRIDGES.md). Persisted as a
/// three-tensor safetensors file ("embeddings" f32 [n, dim], "centroid"
/// f32 [dim], "chunk_doc" i64 [n]). Selection = cosine top-k over chunks
/// (a dot product per row — no external vector store), then
/// chunk→document resolution in score order.
pub const EmbedIndex = struct {
    allocator: Allocator,
    dim: usize,
    vecs: std.ArrayList(f32) = .empty,
    chunk_doc: std.ArrayList(u32) = .empty,
    /// The pre-centering mean of the unit rows; empty until `finalize`.
    centroid: []f32 = &.{},

    pub fn init(allocator: Allocator, dim: usize) EmbedIndex {
        return .{ .allocator = allocator, .dim = dim };
    }

    pub fn deinit(self: *EmbedIndex) void {
        self.vecs.deinit(self.allocator);
        self.chunk_doc.deinit(self.allocator);
        if (self.centroid.len > 0) self.allocator.free(self.centroid);
        self.* = undefined;
    }

    pub fn len(self: *const EmbedIndex) usize {
        return self.chunk_doc.items.len;
    }

    pub fn finalized(self: *const EmbedIndex) bool {
        return self.centroid.len > 0;
    }

    /// Append one chunk vector (copied, then L2-normalized in storage).
    /// Rejected after `finalize`.
    pub fn append(self: *EmbedIndex, doc: u32, vec: []const f32) !void {
        if (vec.len != self.dim or self.dim == 0) return Error.InvalidIndex;
        if (self.finalized()) return Error.InvalidIndex;
        try self.vecs.appendSlice(self.allocator, vec);
        errdefer self.vecs.shrinkRetainingCapacity(self.vecs.items.len - self.dim);
        try self.chunk_doc.append(self.allocator, doc);
        l2Normalize(self.vecs.items[self.vecs.items.len - self.dim ..]);
    }

    /// Center the index: subtract the unit-row centroid from every row and
    /// re-normalize. Call once, after the last `append`; queries centered
    /// with the same vector in `topDocs`.
    pub fn finalize(self: *EmbedIndex) !void {
        if (self.finalized()) return Error.InvalidIndex;
        const centroid = try self.allocator.alloc(f32, self.dim);
        errdefer self.allocator.free(centroid);
        @memset(centroid, 0);
        const n = self.len();
        if (n > 0) {
            for (0..n) |chunk| {
                const row = self.vecs.items[chunk * self.dim ..][0..self.dim];
                for (centroid, row) |*c, x| c.* += x;
            }
            const inv_n = 1.0 / @as(f32, @floatFromInt(n));
            for (centroid) |*c| c.* *= inv_n;
            for (0..n) |chunk| {
                const row = self.vecs.items[chunk * self.dim ..][0..self.dim];
                for (row, centroid) |*x, c| x.* -= c;
                l2Normalize(row);
            }
        }
        self.centroid = centroid;
    }

    pub const Hit = struct {
        doc: usize,
        /// The document's best chunk cosine.
        score: f32,
    };

    /// Center + normalize a query copy exactly like the stored rows
    /// (caller frees).
    fn centeredQuery(self: *const EmbedIndex, allocator: Allocator, query: []const f32) ![]f32 {
        const q = try allocator.dupe(f32, query);
        l2Normalize(q);
        for (q, self.centroid) |*x, c| x.* -= c;
        l2Normalize(q);
        return q;
    }

    /// Best-chunk cosine of `query` against EACH of `docs` (parallel to
    /// `out`; -1 for a doc with no chunks) — the adaptive-serving
    /// hysteresis probe: how does a conversation's CURRENT selection score
    /// under a new query? Requires a finalized index.
    pub fn docScores(self: *const EmbedIndex, allocator: Allocator, query: []const f32, docs: []const usize, out: []f32) !void {
        if (query.len != self.dim or docs.len != out.len) return Error.InvalidIndex;
        if (!self.finalized()) return Error.InvalidIndex;
        const q = try self.centeredQuery(allocator, query);
        defer allocator.free(q);
        @memset(out, -1);
        for (0..self.len()) |chunk| {
            const doc = self.chunk_doc.items[chunk];
            for (docs, out) |want, *best| {
                if (want != doc) continue;
                const score = dot(q, self.vecs.items[chunk * self.dim ..][0..self.dim]);
                best.* = @max(best.*, score);
            }
        }
    }

    /// Cosine top-`k_chunks` over every chunk (ties keep the lower chunk
    /// index), resolved to at most `max_docs` distinct documents in
    /// best-chunk order. The query is centered and normalized internally
    /// from a copy. Requires a finalized index. Caller frees the hits.
    pub fn topDocs(self: *const EmbedIndex, allocator: Allocator, query: []const f32, k_chunks: usize, max_docs: usize) ![]Hit {
        if (query.len != self.dim) return Error.InvalidIndex;
        if (!self.finalized()) return Error.InvalidIndex;
        const n = self.len();
        if (n == 0 or k_chunks == 0 or max_docs == 0) return allocator.alloc(Hit, 0);

        const q = try self.centeredQuery(allocator, query);
        defer allocator.free(q);

        // Insertion top-k over chunk scores (k is tiny; ties → lower index
        // because strictly-greater is required to displace).
        const k = @min(k_chunks, n);
        const best = try allocator.alloc(usize, k);
        defer allocator.free(best);
        const best_scores = try allocator.alloc(f32, k);
        defer allocator.free(best_scores);
        var filled: usize = 0;
        for (0..n) |chunk| {
            const score = dot(q, self.vecs.items[chunk * self.dim ..][0..self.dim]);
            var at = filled;
            while (at > 0 and score > best_scores[at - 1]) at -= 1;
            if (at >= k) continue;
            if (filled < k) filled += 1;
            var j = filled - 1;
            while (j > at) : (j -= 1) {
                best[j] = best[j - 1];
                best_scores[j] = best_scores[j - 1];
            }
            best[at] = chunk;
            best_scores[at] = score;
        }

        // Chunk hits → distinct documents, best chunk first.
        var hits: std.ArrayList(Hit) = .empty;
        errdefer hits.deinit(allocator);
        for (best[0..filled], best_scores[0..filled]) |chunk, score| {
            const doc: usize = self.chunk_doc.items[chunk];
            var seen = false;
            for (hits.items) |hit| seen = seen or (hit.doc == doc);
            if (seen) continue;
            try hits.append(allocator, .{ .doc = doc, .score = score });
            if (hits.items.len == max_docs) break;
        }
        return hits.toOwnedSlice(allocator);
    }

    /// Serialize as safetensors ("embeddings" f32 [n, dim], "centroid" f32
    /// [dim], "chunk_doc" i64 [n]). Requires a finalized index.
    pub fn serialize(self: *const EmbedIndex, allocator: Allocator, writer: *std.Io.Writer) !void {
        if (!self.finalized()) return Error.InvalidIndex;
        const n = self.len();
        const ids = try allocator.alloc(i64, n);
        defer allocator.free(ids);
        for (ids, self.chunk_doc.items) |*id, doc| id.* = doc;
        const tensors = [_]fucina.safetensors.Tensor{
            .{
                .name = "embeddings",
                .dtype = .F32,
                .shape = &.{ n, self.dim },
                .data = std.mem.sliceAsBytes(self.vecs.items),
            },
            .{
                .name = "centroid",
                .dtype = .F32,
                .shape = &.{self.dim},
                .data = std.mem.sliceAsBytes(self.centroid),
            },
            .{
                .name = "chunk_doc",
                .dtype = .I64,
                .shape = &.{n},
                .data = std.mem.sliceAsBytes(ids),
            },
        };
        try fucina.safetensors.serialize(allocator, writer, &tensors, null);
    }

    /// Rebuild from `serialize` bytes (finalized by construction).
    pub fn initFromBytes(allocator: Allocator, bytes: []const u8) !EmbedIndex {
        var file = try fucina.safetensors.File.parse(allocator, bytes);
        defer file.deinit();
        const emb = file.maybeTensor("embeddings") orelse return Error.InvalidIndex;
        const cen = file.maybeTensor("centroid") orelse return Error.InvalidIndex;
        const ids = file.maybeTensor("chunk_doc") orelse return Error.InvalidIndex;
        if (emb.dtype != .F32 or emb.shape.len != 2) return Error.InvalidIndex;
        if (cen.dtype != .F32 or cen.shape.len != 1 or cen.shape[0] != emb.shape[1]) return Error.InvalidIndex;
        if (ids.dtype != .I64 or ids.shape.len != 1 or ids.shape[0] != emb.shape[0]) return Error.InvalidIndex;
        const n = emb.shape[0];
        const dim = emb.shape[1];
        if (emb.data.len != n * dim * 4 or cen.data.len != dim * 4 or ids.data.len != n * 8) return Error.InvalidIndex;

        var index = EmbedIndex.init(allocator, dim);
        errdefer index.deinit();
        try index.vecs.resize(allocator, n * dim);
        @memcpy(std.mem.sliceAsBytes(index.vecs.items), emb.data);
        try index.chunk_doc.resize(allocator, n);
        for (index.chunk_doc.items, 0..) |*doc, i| {
            var value: i64 = undefined;
            @memcpy(std.mem.asBytes(&value), ids.data[i * 8 ..][0..8]);
            if (value < 0) return Error.InvalidIndex;
            doc.* = @intCast(value);
        }
        const centroid = try allocator.alloc(f32, dim);
        @memcpy(std.mem.sliceAsBytes(centroid), cen.data);
        index.centroid = centroid;
        return index;
    }
};

// ---------------------------------------------------------------------------
// Budget manager
// ---------------------------------------------------------------------------

/// The rotation knobs (paper Sec on the budget manager; defaults follow it
/// scaled to CLI budgets: R = 10, φ = 0.5, per-cartridge warm-up).
pub const RotationPolicy = struct {
    /// Max resident cartridges (rows + Adam moments live).
    budget: usize,
    /// Rotate every `every` optimizer rounds (0 = never).
    every: u64 = 10,
    /// Fraction of residents evicted per rotation.
    evict_fraction: f32 = 0.5,
    /// Per-cartridge lr warm-up length, in optimizer steps after (re)entry.
    warmup: u64 = 8,
};

/// A resident document: its cartridge and a dedicated optimizer whose
/// moments travel with it through evict/reload cycles. The optimizer
/// borrows the cartridge's registry entries — deinit order is opt first.
pub const Resident = struct {
    doc: usize,
    cart: cartridge.Cartridge,
    opt: optim.AdamW,
    /// `docs[doc].steps` at (re)entry — drives the warm-up ramp.
    entered_step: u64,

    fn deinit(self: *Resident) void {
        self.opt.deinit();
        self.cart.deinit();
        self.* = undefined;
    }
};

/// Pick up to `n` eviction victims: RESIDENT docs with the MOST steps first
/// (ties → lower doc id). Returns doc ids, most-trained first; caller frees.
pub fn pickEvictions(allocator: Allocator, steps: []const u64, resident: []const bool, n: usize) ![]usize {
    return pickBySteps(allocator, steps, resident, n, true, true);
}

/// Pick up to `n` docs to load: ABSENT docs with the FEWEST steps first
/// (ties → lower doc id). Returns doc ids, least-trained first; caller frees.
pub fn pickLoads(allocator: Allocator, steps: []const u64, resident: []const bool, n: usize) ![]usize {
    return pickBySteps(allocator, steps, resident, n, false, false);
}

fn pickBySteps(allocator: Allocator, steps: []const u64, resident: []const bool, n: usize, want_resident: bool, descending: bool) ![]usize {
    std.debug.assert(steps.len == resident.len);
    var picked: std.ArrayList(usize) = .empty;
    errdefer picked.deinit(allocator);
    if (n == 0) return picked.toOwnedSlice(allocator);
    for (steps, resident, 0..) |s, r, doc| {
        if (r != want_resident) continue;
        // Insertion sort by steps; strict comparison keeps ties in doc order.
        var at = picked.items.len;
        while (at > 0) : (at -= 1) {
            const other = steps[picked.items[at - 1]];
            const displaces = if (descending) s > other else s < other;
            if (!displaces) break;
        }
        if (at >= n) continue;
        try picked.insert(allocator, at, doc);
        if (picked.items.len > n) picked.shrinkRetainingCapacity(n);
    }
    return picked.toOwnedSlice(allocator);
}

/// The fleet: manifest + resident pool + disk I/O. All state mutations that
/// touch disk are atomic (temp file + rename), so a crash mid-rotation
/// leaves every artifact either old or new, never torn.
pub const Fleet = struct {
    allocator: Allocator,
    dir: []u8,
    manifest: Manifest,
    residents: std.ArrayList(Resident) = .empty,
    base_lr: f32,
    policy: RotationPolicy,

    /// Create a fleet directory around an existing `manifest` and persist
    /// it. Ownership of `manifest` transfers ON SUCCESS only — on error the
    /// caller still owns (and deinits) it.
    pub fn create(allocator: Allocator, io: std.Io, dir: []const u8, manifest: Manifest, base_lr: f32, policy: RotationPolicy) !Fleet {
        try std.Io.Dir.cwd().createDirPath(io, dir);
        var fleet = Fleet{
            .allocator = allocator,
            .dir = try allocator.dupe(u8, dir),
            .manifest = manifest,
            .base_lr = base_lr,
            .policy = policy,
        };
        errdefer allocator.free(fleet.dir);
        try fleet.writeManifest(io);
        return fleet;
    }

    /// Reopen a fleet from its directory.
    pub fn open(allocator: Allocator, io: std.Io, dir: []const u8, base_lr: f32, policy: RotationPolicy) !Fleet {
        const manifest_path = try std.fs.path.join(allocator, &.{ dir, manifest_file });
        defer allocator.free(manifest_path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, manifest_path, allocator, read_limit);
        defer allocator.free(bytes);
        var manifest = try Manifest.parse(allocator, bytes);
        errdefer manifest.deinit();
        return .{
            .allocator = allocator,
            .dir = try allocator.dupe(u8, dir),
            .manifest = manifest,
            .base_lr = base_lr,
            .policy = policy,
        };
    }

    /// Tear down WITHOUT saving (pair with `saveAll` for a clean shutdown).
    pub fn deinit(self: *Fleet) void {
        for (self.residents.items) |*resident| resident.deinit();
        self.residents.deinit(self.allocator);
        self.manifest.deinit();
        self.allocator.free(self.dir);
        self.* = undefined;
    }

    pub fn residentIndex(self: *const Fleet, doc: usize) ?usize {
        for (self.residents.items, 0..) |*resident, i| {
            if (resident.doc == doc) return i;
        }
        return null;
    }

    fn path(self: *const Fleet, leaf: []const u8) ![]u8 {
        return std.fs.path.join(self.allocator, &.{ self.dir, leaf });
    }

    pub fn writeManifest(self: *const Fleet, io: std.Io) !void {
        const manifest_path = try self.path(manifest_file);
        defer self.allocator.free(manifest_path);
        try fucina.training_checkpoint.writeFileAtomic(io, manifest_path, &self.manifest, writeManifestTo);
    }

    fn writeManifestTo(manifest: *const Manifest, writer: *std.Io.Writer) anyerror!void {
        try manifest.write(writer);
    }

    /// Adopt a freshly built cartridge as `doc`'s resident (first
    /// residency: fresh Adam moments). Takes ownership of `cart` even on
    /// error. Returns the resident index.
    pub fn adoptResident(self: *Fleet, doc: usize, cart: cartridge.Cartridge) !usize {
        var owned = cart;
        errdefer owned.deinit();
        if (doc >= self.manifest.docs.items.len) return Error.InvalidDoc;
        if (self.residentIndex(doc) != null) return Error.InvalidDoc;
        if (self.residents.items.len >= self.policy.budget) return Error.InvalidFleet;

        var resident = Resident{
            .doc = doc,
            .cart = owned,
            .opt = optim.AdamW.init(self.allocator, .{ .lr = self.base_lr, .weight_decay = 0 }),
            .entered_step = self.manifest.docs.items[doc].steps,
        };
        errdefer resident.opt.deinit();
        try resident.cart.registerParams(&resident.opt);
        try self.residents.append(self.allocator, resident);
        return self.residents.items.len - 1;
    }

    /// Load `doc` from disk into the resident pool: cartridge rows from its
    /// safetensors, Adam moments from its FZT1 snapshot when one exists
    /// (a rows-only artifact gets fresh moments). Returns the resident index.
    pub fn loadResident(self: *Fleet, ctx: *ExecContext, io: std.Io, doc: usize) !usize {
        if (doc >= self.manifest.docs.items.len) return Error.InvalidDoc;
        const state = &self.manifest.docs.items[doc];

        const cart_path = try self.path(state.cart_file);
        defer self.allocator.free(cart_path);
        var mapped = try mmapFile(io, cart_path);
        defer mapped.deinit();
        const cart = try cartridge.Cartridge.initFromStateDict(ctx, self.allocator, mapped.bytes);

        const idx = try self.adoptResident(doc, cart);
        const resident = &self.residents.items[idx];

        const opt_path = try self.path(state.opt_file);
        defer self.allocator.free(opt_path);
        if (mmapFile(io, opt_path)) |mapped_opt_const| {
            var mapped_opt = mapped_opt_const;
            defer mapped_opt.deinit();
            var reader = std.Io.Reader.fixed(mapped_opt.bytes);
            try resident.opt.loadState(&reader);
        } else |err| switch (err) {
            error.FileNotFound => {},
            else => |e| return e,
        }
        return idx;
    }

    /// Persist resident `idx` (rows AND Adam moments, both atomic) and
    /// remove it from the pool. The evict/reload cycle is bit-identical to
    /// staying resident — pinned by a fleet test.
    pub fn evictResident(self: *Fleet, io: std.Io, idx: usize) !void {
        const resident = &self.residents.items[idx];
        const state = &self.manifest.docs.items[resident.doc];

        const cart_path = try self.path(state.cart_file);
        defer self.allocator.free(cart_path);
        const cart_ptr: *const cartridge.Cartridge = &resident.cart;
        try fucina.training_checkpoint.writeFileAtomic(io, cart_path, cart_ptr, writeCartridgeTo);

        const opt_path = try self.path(state.opt_file);
        defer self.allocator.free(opt_path);
        const opt_ptr: *const optim.AdamW = &resident.opt;
        try fucina.training_checkpoint.writeFileAtomic(io, opt_path, opt_ptr, writeOptimizerTo);

        var removed = self.residents.swapRemove(idx);
        removed.deinit();
    }

    fn writeCartridgeTo(cart: *const cartridge.Cartridge, writer: *std.Io.Writer) anyerror!void {
        try cart.saveState(writer);
    }

    fn writeOptimizerTo(opt: *const optim.AdamW, writer: *std.Io.Writer) anyerror!void {
        try opt.saveState(writer);
    }

    /// The learning rate for resident `idx` this round: base lr scaled by
    /// the per-cartridge warm-up ramp (paper: applied whenever a cartridge
    /// enters the pool). Install it on the resident's optimizer before
    /// `step`.
    pub fn residentLr(self: *const Fleet, idx: usize) f32 {
        const resident = &self.residents.items[idx];
        const taken = self.manifest.docs.items[resident.doc].steps - resident.entered_step;
        if (self.policy.warmup == 0 or taken >= self.policy.warmup) return self.base_lr;
        const ramp = @as(f32, @floatFromInt(taken + 1)) / @as(f32, @floatFromInt(self.policy.warmup));
        return self.base_lr * ramp;
    }

    /// Record one optimizer step on `doc`'s cartridge.
    pub fn noteStep(self: *Fleet, doc: usize) void {
        self.manifest.docs.items[doc].steps += 1;
    }

    /// Rotation (call after bumping `manifest.rounds`): when the round hits
    /// the cadence and docs are waiting on disk, evict
    /// `ceil(evict_fraction * residents)` MOST-trained residents and load
    /// as many LEAST-trained absentees as budget allows. Returns how many
    /// docs rotated in.
    pub fn maybeRotate(self: *Fleet, ctx: *ExecContext, io: std.Io, log: ?*std.Io.Writer) !usize {
        const n_docs = self.manifest.docs.items.len;
        if (n_docs <= self.policy.budget) return 0;
        if (self.policy.every == 0 or self.manifest.rounds % self.policy.every != 0) return 0;

        const steps = try self.allocator.alloc(u64, n_docs);
        defer self.allocator.free(steps);
        const resident = try self.allocator.alloc(bool, n_docs);
        defer self.allocator.free(resident);
        for (steps, self.manifest.docs.items) |*s, *doc| s.* = doc.steps;
        @memset(resident, false);
        for (self.residents.items) |*r| resident[r.doc] = true;

        const n_evict = @max(1, @as(usize, @intFromFloat(@ceil(self.policy.evict_fraction * @as(f32, @floatFromInt(self.residents.items.len))))));
        const victims = try pickEvictions(self.allocator, steps, resident, n_evict);
        defer self.allocator.free(victims);
        for (victims) |doc| {
            const idx = self.residentIndex(doc) orelse unreachable;
            try self.evictResident(io, idx);
            if (log) |w| try w.print("rotation: evicted doc {d} ({s}, {d} steps)\n", .{ doc, self.manifest.docs.items[doc].name, steps[doc] });
        }

        const free_slots = self.policy.budget - self.residents.items.len;
        const arrivals = try pickLoads(self.allocator, steps, resident, free_slots);
        defer self.allocator.free(arrivals);
        var loaded: usize = 0;
        for (arrivals) |doc| {
            // Skip the just-evicted (they read as absent in the snapshot
            // taken before eviction — `resident` marks them true, so they
            // are not in `arrivals`; this guards future policy changes).
            if (self.residentIndex(doc) != null) continue;
            _ = try self.loadResident(ctx, io, doc);
            loaded += 1;
            if (log) |w| try w.print("rotation: loaded doc {d} ({s}, {d} steps)\n", .{ doc, self.manifest.docs.items[doc].name, steps[doc] });
        }
        return loaded;
    }

    /// Evict every resident (persisting rows + moments) and write the
    /// manifest — the clean-shutdown save.
    pub fn saveAll(self: *Fleet, io: std.Io) !void {
        while (self.residents.items.len > 0) {
            try self.evictResident(io, self.residents.items.len - 1);
        }
        try self.writeManifest(io);
    }

    /// Path of the fleet's retrieval index.
    pub fn indexPath(self: *const Fleet) ![]u8 {
        return self.path(index_file);
    }
};

test {
    _ = @import("cartridge_fleet_tests.zig");
}
