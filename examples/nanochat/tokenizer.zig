//! nanochat BPE tokenizer: rustbpe-equivalent trainer + tiktoken-style encoder.
//!
//! Works in RAW BYTE space (unlike src/llm/tokenizer.zig's GPT-2
//! byte-level-unicode vocabulary): base tokens are the 256 single bytes
//! (id == byte value), merge i produces token id 256 + i, and the 9 special
//! tokens are appended last (id = 256 + n_merges + k). The trainer is a
//! faithful port of rustbpe's `train_core_incremental`
//! (refs/rustbpe/src/lib.rs) and the encoder of rustbpe's `encode` /
//! tiktoken's `encode_ordinary`; the pretokenizer implements nanochat's
//! SPLIT_PATTERN (refs/nanochat/nanochat/tokenizer.py).
//!
//! File formats (tokenizer.bin "NCTOKz01", token_bytes.bin "NCTKB_01",
//! trainer corpus "NCTXT_01", encode fixtures "NCIDS_01") are documented at
//! their save/load functions below; all integers little-endian.

const std = @import("std");
const ucat = @import("fucina_llm").unicode_categories;

const Allocator = std.mem.Allocator;

/// nanochat's special tokens in canonical order (nanochat/tokenizer.py
/// SPECIAL_TOKENS); ids are 256 + n_merges + k.
pub const special_tokens = [_][]const u8{
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
};
pub const n_special: u32 = special_tokens.len;

pub const Pair = struct { left: u32, right: u32 };

/// Packed pair key: compares (as u64) exactly like the (left, right) tuple.
inline fn packPair(left: u32, right: u32) u64 {
    return (@as(u64, left) << 32) | right;
}

pub const Tokenizer = struct {
    allocator: Allocator,
    /// Merges in rank order; merge i produces token id 256 + i. Owned.
    merge_list: []Pair,
    /// packed (left << 32 | right) → merged token id (the encode-side view
    /// of `merge_list`).
    merges: std.AutoHashMapUnmanaged(u64, u32),
    /// One owned buffer holding every token's bytes back-to-back.
    vocab_blob: []u8,
    /// token id → raw bytes (slices into `vocab_blob`), length n_vocab.
    /// Special ids map to their "<|name|>" strings.
    vocab: [][]const u8,
    n_vocab: u32,

    pub fn nMerges(self: *const Tokenizer) u32 {
        return @intCast(self.merge_list.len);
    }

    /// First special-token id (== 256 + n_merges).
    pub fn specialBase(self: *const Tokenizer) u32 {
        return 256 + self.nMerges();
    }

    pub fn specialId(self: *const Tokenizer, name: []const u8) ?u32 {
        var id = self.specialBase();
        while (id < self.n_vocab) : (id += 1) {
            if (std.mem.eql(u8, self.vocab[id], name)) return id;
        }
        return null;
    }

    pub fn bosId(self: *const Tokenizer) u32 {
        return self.specialId("<|bos|>").?;
    }

    /// Build a tokenizer from a rank-ordered merge list and special-token
    /// names. Takes ownership of `merge_list` on success only; `specials`
    /// strings are copied.
    fn init(allocator: Allocator, merge_list: []Pair, specials: []const []const u8) !Tokenizer {
        const n_nonspecial: usize = 256 + merge_list.len;
        const n_vocab: usize = n_nonspecial + specials.len;

        const lens = try allocator.alloc(usize, n_nonspecial);
        defer allocator.free(lens);
        for (lens[0..256]) |*l| l.* = 1;
        var blob_len: usize = 256;
        for (merge_list, 0..) |m, i| {
            const id = 256 + i;
            if (m.left >= id or m.right >= id) return error.CorruptTokenizerBin;
            lens[id] = lens[m.left] + lens[m.right];
            blob_len += lens[id];
        }
        for (specials) |s| blob_len += s.len;

        const blob = try allocator.alloc(u8, blob_len);
        errdefer allocator.free(blob);
        const vocab = try allocator.alloc([]const u8, n_vocab);
        errdefer allocator.free(vocab);

        var off: usize = 0;
        for (0..256) |b| {
            blob[off] = @intCast(b);
            vocab[b] = blob[off .. off + 1];
            off += 1;
        }
        for (merge_list, 0..) |m, i| {
            const id = 256 + i;
            @memcpy(blob[off..][0..lens[m.left]], vocab[m.left]);
            @memcpy(blob[off + lens[m.left] ..][0..lens[m.right]], vocab[m.right]);
            vocab[id] = blob[off .. off + lens[id]];
            off += lens[id];
        }
        for (specials, 0..) |s, k| {
            @memcpy(blob[off..][0..s.len], s);
            vocab[n_nonspecial + k] = blob[off .. off + s.len];
            off += s.len;
        }
        std.debug.assert(off == blob_len);

        var merges: std.AutoHashMapUnmanaged(u64, u32) = .empty;
        errdefer merges.deinit(allocator);
        try merges.ensureTotalCapacity(allocator, @intCast(merge_list.len));
        for (merge_list, 0..) |m, i| {
            merges.putAssumeCapacity(packPair(m.left, m.right), @intCast(256 + i));
        }

        return .{
            .allocator = allocator,
            .merge_list = merge_list,
            .merges = merges,
            .vocab_blob = blob,
            .vocab = vocab,
            .n_vocab = @intCast(n_vocab),
        };
    }

    pub fn deinit(self: *Tokenizer) void {
        self.merges.deinit(self.allocator);
        self.allocator.free(self.merge_list);
        self.allocator.free(self.vocab);
        self.allocator.free(self.vocab_blob);
        self.* = undefined;
    }

    // ---- encode / decode ----

    /// tiktoken `encode_ordinary`: pretokenize + per-chunk BPE. No special
    /// tokens are recognized or added. Caller frees the result.
    pub fn encode(self: *const Tokenizer, allocator: Allocator, text: []const u8) ![]u32 {
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(allocator);
        try self.encodeAppend(allocator, text, &out);
        return out.toOwnedSlice(allocator);
    }

    /// `encode` with the BOS special id prepended (nanochat's document form).
    pub fn encodeWithBos(self: *const Tokenizer, allocator: Allocator, text: []const u8) ![]u32 {
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(allocator);
        try out.append(allocator, self.bosId());
        try self.encodeAppend(allocator, text, &out);
        return out.toOwnedSlice(allocator);
    }

    /// Append `encode(text)` to `out` — callers place arbitrary special ids
    /// around text spans themselves (conversation rendering).
    pub fn encodeAppend(self: *const Tokenizer, allocator: Allocator, text: []const u8, out: *std.ArrayList(u32)) !void {
        if (text.len == 0) return;

        var cps: std.ArrayList(u32) = .empty;
        defer cps.deinit(allocator);
        var offs: std.ArrayList(usize) = .empty;
        defer offs.deinit(allocator);
        try decodeCps(allocator, text, &cps, &offs);

        var pos: usize = 0;
        while (pos < cps.items.len) {
            const end = nanochatChunkEnd(cps.items, pos);
            try self.encodeChunk(allocator, text[offs.items[pos]..offs.items[end]], out);
            pos = end;
        }
    }

    /// rustbpe `encode` on one pretoken chunk: bytes → ids, then repeatedly
    /// merge the adjacent pair with the LOWEST merged id; ties go to the
    /// leftmost occurrence (strictly-less comparison, first wins).
    fn encodeChunk(self: *const Tokenizer, allocator: Allocator, chunk: []const u8, out: *std.ArrayList(u32)) !void {
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(allocator);
        try ids.ensureTotalCapacity(allocator, chunk.len);
        for (chunk) |b| ids.appendAssumeCapacity(b);

        while (ids.items.len >= 2) {
            var best_idx: usize = 0;
            var best_id: u32 = std.math.maxInt(u32);
            for (0..ids.items.len - 1) |i| {
                if (self.merges.get(packPair(ids.items[i], ids.items[i + 1]))) |new_id| {
                    if (new_id < best_id) {
                        best_id = new_id;
                        best_idx = i;
                    }
                }
            }
            if (best_id == std.math.maxInt(u32)) break;
            ids.items[best_idx] = best_id;
            _ = ids.orderedRemove(best_idx + 1);
        }

        try out.appendSlice(allocator, ids.items);
    }

    /// Concatenate each id's raw bytes (special ids render their "<|name|>"
    /// string). Caller frees the result.
    pub fn decode(self: *const Tokenizer, allocator: Allocator, ids: []const u32) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(allocator);
        for (ids) |id| {
            if (id >= self.n_vocab) return error.UnknownTokenId;
            try out.appendSlice(allocator, self.vocab[id]);
        }
        return out.toOwnedSlice(allocator);
    }

    // ---- trainer ----

    /// Train from documents — rustbpe `train_from_iterator` +
    /// `train_core_incremental` with nanochat's special-token accounting:
    /// `vocab_size` INCLUDES the 9 specials (tokenizer.py
    /// `train_from_iterator` trains vocab_size − 9 non-special tokens), so
    /// vocab_size − 9 − 256 merges are learned and n_vocab == vocab_size.
    pub fn trainFromDocs(allocator: Allocator, docs: []const []const u8, vocab_size: u32) !Tokenizer {
        if (vocab_size < 256 + n_special) return error.VocabTooSmall;
        const num_merges = vocab_size - n_special - 256;

        // Count unique pretoken chunks. Keys borrow `docs`.
        var chunk_counts: std.StringHashMapUnmanaged(i64) = .empty;
        defer chunk_counts.deinit(allocator);
        {
            var cps: std.ArrayList(u32) = .empty;
            defer cps.deinit(allocator);
            var offs: std.ArrayList(usize) = .empty;
            defer offs.deinit(allocator);
            for (docs) |doc| {
                try decodeCps(allocator, doc, &cps, &offs);
                var pos: usize = 0;
                while (pos < cps.items.len) {
                    const end = nanochatChunkEnd(cps.items, pos);
                    const chunk = doc[offs.items[pos]..offs.items[end]];
                    const gop = try chunk_counts.getOrPut(allocator, chunk);
                    if (!gop.found_existing) gop.value_ptr.* = 0;
                    gop.value_ptr.* += 1;
                    pos = end;
                }
            }
        }

        // Materialize words (one per unique chunk: its bytes as ids 0..255)
        // and their counts. Hash-map iteration order does not influence the
        // trained merges: selection depends only on integer pair counts and
        // pair values, and all per-word count updates are commutative sums.
        const n_words = chunk_counts.count();
        var words = try allocator.alloc(std.ArrayListUnmanaged(u32), n_words);
        var words_len: usize = 0;
        defer {
            for (words[0..words_len]) |*w| w.deinit(allocator);
            allocator.free(words);
        }
        const counts = try allocator.alloc(i64, n_words);
        defer allocator.free(counts);
        {
            var it = chunk_counts.iterator();
            while (it.next()) |e| {
                var w: std.ArrayListUnmanaged(u32) = .empty;
                try w.ensureTotalCapacity(allocator, e.key_ptr.len);
                for (e.key_ptr.*) |b| w.appendAssumeCapacity(b);
                words[words_len] = w;
                counts[words_len] = e.value_ptr.*;
                words_len += 1;
            }
        }

        const merge_list = try trainCore(allocator, words, counts, num_merges);
        errdefer allocator.free(merge_list);
        return init(allocator, merge_list, &special_tokens);
    }

    // ---- tokenizer.bin (NCTOKz01) ----

    pub fn saveBin(self: *const Tokenizer, io: std.Io, path: []const u8) !void {
        var file = try std.Io.Dir.cwd().createFile(io, path, .{});
        defer file.close(io);
        var buf: [64 * 1024]u8 = undefined;
        var writer = file.writer(io, &buf);
        const w = &writer.interface;

        try w.writeAll("NCTOKz01");
        try w.writeInt(u32, 1, .little); // version
        try w.writeInt(u32, self.n_vocab, .little);
        try w.writeInt(u32, self.nMerges(), .little);
        try w.writeInt(u32, self.n_vocab - self.specialBase(), .little);
        for (self.merge_list) |m| {
            try w.writeInt(u32, m.left, .little);
            try w.writeInt(u32, m.right, .little);
        }
        var id = self.specialBase();
        while (id < self.n_vocab) : (id += 1) {
            const name = self.vocab[id];
            try w.writeInt(u32, id, .little);
            try w.writeInt(u16, @intCast(name.len), .little);
            try w.writeAll(name);
        }
        try w.flush();
    }

    pub fn loadBin(allocator: Allocator, io: std.Io, path: []const u8) !Tokenizer {
        const bytes = try readFileBytes(allocator, io, path);
        defer allocator.free(bytes);

        var r = BinReader{ .bytes = bytes };
        const magic = try r.take(8);
        if (!std.mem.eql(u8, magic, "NCTOKz01")) return error.CorruptTokenizerBin;
        if (try r.u32le() != 1) return error.CorruptTokenizerBin;
        const n_vocab = try r.u32le();
        const n_merges = try r.u32le();
        const file_n_special = try r.u32le();
        if (256 + n_merges + file_n_special != n_vocab) return error.CorruptTokenizerBin;

        const merge_list = try allocator.alloc(Pair, n_merges);
        errdefer allocator.free(merge_list);
        for (merge_list) |*m| {
            m.left = try r.u32le();
            m.right = try r.u32le();
        }

        var specials: std.ArrayList([]const u8) = .empty;
        defer specials.deinit(allocator);
        for (0..file_n_special) |k| {
            const id = try r.u32le();
            if (id != 256 + n_merges + k) return error.CorruptTokenizerBin;
            const len = try r.u16le();
            try specials.append(allocator, try r.take(len));
        }
        if (r.off != bytes.len) return error.CorruptTokenizerBin;

        return init(allocator, merge_list, specials.items);
    }

    // ---- token_bytes.bin (NCTKB_01) ----

    /// Per-id byte-length table for bits-per-byte eval: len(vocab[id]) for
    /// byte/merge tokens, 0 for specials (not counted). Caller frees.
    pub fn computeTokenBytes(self: *const Tokenizer, allocator: Allocator) ![]u32 {
        const table = try allocator.alloc(u32, self.n_vocab);
        const base = self.specialBase();
        for (table, 0..) |*t, id| {
            t.* = if (id < base) @intCast(self.vocab[id].len) else 0;
        }
        return table;
    }

    pub fn saveTokenBytes(self: *const Tokenizer, allocator: Allocator, io: std.Io, path: []const u8) !void {
        const table = try self.computeTokenBytes(allocator);
        defer allocator.free(table);

        var file = try std.Io.Dir.cwd().createFile(io, path, .{});
        defer file.close(io);
        var buf: [64 * 1024]u8 = undefined;
        var writer = file.writer(io, &buf);
        const w = &writer.interface;
        try w.writeAll("NCTKB_01");
        try w.writeInt(u32, self.n_vocab, .little);
        for (table) |t| try w.writeInt(u32, t, .little);
        try w.flush();
    }
};

// ---------------------------------------------------------------------------
// Trainer core (rustbpe train_core_incremental)
// ---------------------------------------------------------------------------

const MergeJob = struct {
    pair: u64,
    count: i64,
    /// Word indices where this pair may occur and needs processing (unique).
    pos: std.ArrayListUnmanaged(u32),
};

/// Binary max-heap of merge jobs with rustbpe's `MergeJob::cmp` order:
/// max by count, ties broken toward the ASCENDING (left, right) pair. The
/// comparator is a total order over live jobs (at most one job per pair at
/// any time — new occurrences only ever arise for pairs containing the id
/// minted in the current step), so pop order is implementation-independent.
const JobHeap = struct {
    items: std.ArrayListUnmanaged(MergeJob) = .empty,

    fn deinit(self: *JobHeap, allocator: Allocator) void {
        for (self.items.items) |*job| job.pos.deinit(allocator);
        self.items.deinit(allocator);
        self.* = undefined;
    }

    fn better(a: MergeJob, b: MergeJob) bool {
        if (a.count != b.count) return a.count > b.count;
        return a.pair < b.pair;
    }

    fn push(self: *JobHeap, allocator: Allocator, job: MergeJob) !void {
        try self.items.append(allocator, job);
        const items = self.items.items;
        var i = items.len - 1;
        while (i > 0) {
            const parent = (i - 1) / 2;
            if (!better(items[i], items[parent])) break;
            std.mem.swap(MergeJob, &items[i], &items[parent]);
            i = parent;
        }
    }

    fn pop(self: *JobHeap) ?MergeJob {
        const items = self.items.items;
        if (items.len == 0) return null;
        const top = items[0];
        items[0] = items[items.len - 1];
        self.items.items.len -= 1;
        const rest = self.items.items;
        var i: usize = 0;
        while (true) {
            var best = i;
            const l = 2 * i + 1;
            const r = 2 * i + 2;
            if (l < rest.len and better(rest[l], rest[best])) best = l;
            if (r < rest.len and better(rest[r], rest[best])) best = r;
            if (best == i) break;
            std.mem.swap(MergeJob, &rest[i], &rest[best]);
            i = best;
        }
        return top;
    }
};

const PairDelta = struct { pair: u64, delta: i32 };

/// rustbpe `Word::merge_pair`: replace all non-overlapping (left, right)
/// occurrences left-to-right with `new_id`, appending this word's local
/// pair-count deltas (−1 removed, +1 created) to `deltas`.
fn mergePairInWord(
    allocator: Allocator,
    word: *std.ArrayListUnmanaged(u32),
    left: u32,
    right: u32,
    new_id: u32,
    deltas: *std.ArrayListUnmanaged(PairDelta),
) !void {
    const ids = word.items;
    const n = ids.len;
    if (n < 2) return;

    var out_len: usize = 0;
    var i: usize = 0;
    while (i < n) {
        if (i + 1 < n and ids[i] == left and ids[i + 1] == right) {
            if (out_len > 0) {
                const x = ids[out_len - 1];
                try deltas.append(allocator, .{ .pair = packPair(x, left), .delta = -1 });
                try deltas.append(allocator, .{ .pair = packPair(x, new_id), .delta = 1 });
            }
            try deltas.append(allocator, .{ .pair = packPair(left, right), .delta = -1 });
            if (i + 2 < n) {
                const y = ids[i + 2];
                try deltas.append(allocator, .{ .pair = packPair(right, y), .delta = -1 });
                try deltas.append(allocator, .{ .pair = packPair(new_id, y), .delta = 1 });
            }
            ids[out_len] = new_id;
            out_len += 1;
            i += 2;
        } else {
            ids[out_len] = ids[i];
            out_len += 1;
            i += 1;
        }
    }
    word.items.len = out_len;
}

/// rustbpe `train_core_incremental` over pre-counted unique words: learn up
/// to `num_merges` merges (fewer if no positive-count pair remains). Each
/// step selects the pair with the MAXIMUM total count, ties broken toward
/// the lexicographically smallest (left, right); integer counts throughout.
/// Returns the rank-ordered merge list (caller owns). Mutates `words`.
fn trainCore(
    allocator: Allocator,
    words: []std.ArrayListUnmanaged(u32),
    counts: []const i64,
    num_merges: u32,
) ![]Pair {
    var pair_counts: std.AutoHashMapUnmanaged(u64, i64) = .empty;
    defer pair_counts.deinit(allocator);

    var heap: JobHeap = .{};
    defer heap.deinit(allocator);

    // Initial pair counts + where-to-update sets.
    {
        var where_to_update: std.AutoHashMapUnmanaged(u64, std.ArrayListUnmanaged(u32)) = .empty;
        defer {
            var it = where_to_update.valueIterator();
            while (it.next()) |list| list.deinit(allocator);
            where_to_update.deinit(allocator);
        }
        for (words, 0..) |w, i| {
            if (w.items.len < 2 or counts[i] == 0) continue;
            for (0..w.items.len - 1) |j| {
                const pair = packPair(w.items[j], w.items[j + 1]);
                const pc = try pair_counts.getOrPut(allocator, pair);
                if (!pc.found_existing) pc.value_ptr.* = 0;
                pc.value_ptr.* += counts[i];
                const wtu = try where_to_update.getOrPut(allocator, pair);
                if (!wtu.found_existing) wtu.value_ptr.* = .empty;
                const list = wtu.value_ptr;
                if (list.items.len == 0 or list.items[list.items.len - 1] != i) {
                    try list.append(allocator, @intCast(i));
                }
            }
        }
        var it = where_to_update.iterator();
        while (it.next()) |e| {
            const c = pair_counts.get(e.key_ptr.*) orelse 0;
            if (c > 0) {
                try heap.push(allocator, .{ .pair = e.key_ptr.*, .count = c, .pos = e.value_ptr.* });
                e.value_ptr.* = .empty; // ownership moved into the heap
            }
        }
    }

    const merge_list = try allocator.alloc(Pair, num_merges);
    errdefer allocator.free(merge_list);
    var merges_done: u32 = 0;

    var deltas: std.ArrayListUnmanaged(PairDelta) = .empty;
    defer deltas.deinit(allocator);
    // Insertion-ordered so the resulting heap pushes are deterministic
    // (not load-bearing — see JobHeap — but cheap to keep exact).
    var local_pos: std.AutoArrayHashMapUnmanaged(u64, std.ArrayListUnmanaged(u32)) = .empty;
    defer {
        for (local_pos.values()) |*list| list.deinit(allocator);
        local_pos.deinit(allocator);
    }

    while (merges_done < num_merges) {
        var top = heap.pop() orelse break;

        // Lazy refresh: skip dead pairs, requeue stale counts.
        const current = pair_counts.get(top.pair) orelse 0;
        if (current <= 0) {
            top.pos.deinit(allocator);
            continue;
        }
        if (top.count != current) {
            top.count = current;
            try heap.push(allocator, top);
            continue;
        }

        const new_id: u32 = 256 + merges_done;
        const left: u32 = @intCast(top.pair >> 32);
        const right: u32 = @truncate(top.pair);
        merge_list[merges_done] = .{ .left = left, .right = right };

        for (top.pos.items) |word_idx| {
            deltas.clearRetainingCapacity();
            try mergePairInWord(allocator, &words[word_idx], left, right, new_id, &deltas);
            for (deltas.items) |d| {
                const delta_total = @as(i64, d.delta) * counts[word_idx];
                if (delta_total == 0) continue;
                const pc = try pair_counts.getOrPut(allocator, d.pair);
                if (!pc.found_existing) pc.value_ptr.* = 0;
                pc.value_ptr.* += delta_total;
                if (d.delta > 0) {
                    const lp = try local_pos.getOrPut(allocator, d.pair);
                    if (!lp.found_existing) lp.value_ptr.* = .empty;
                    const list = lp.value_ptr;
                    if (list.items.len == 0 or list.items[list.items.len - 1] != word_idx) {
                        try list.append(allocator, word_idx);
                    }
                }
            }
        }
        top.pos.deinit(allocator);

        // Queue the pairs whose counts increased.
        for (local_pos.keys(), local_pos.values()) |pair, *list| {
            const cnt = pair_counts.get(pair) orelse 0;
            if (cnt > 0) {
                try heap.push(allocator, .{ .pair = pair, .count = cnt, .pos = list.* });
                list.* = .empty; // ownership moved into the heap
            } else {
                list.deinit(allocator);
            }
        }
        local_pos.clearRetainingCapacity();

        merges_done += 1;
    }

    if (merges_done < num_merges) {
        return allocator.realloc(merge_list, merges_done);
    }
    return merge_list;
}

// ---------------------------------------------------------------------------
// Pretokenizer
// ---------------------------------------------------------------------------

/// One nanochat pretoken chunk: the exclusive end index (in codepoints) of
/// the chunk starting at `start` (always > `start`).
///
/// The qwen2ChunkEnd variant (src/llm/tokenizer.zig) differing only in the
/// \p{N}{1,2} number arm. nanochat's SPLIT_PATTERN
/// (refs/nanochat/nanochat/tokenizer.py):
///
///   '(?i:[sdmt]|ll|ve|re) | [^\r\n\p{L}\p{N}]?+\p{L}+ | \p{N}{1,2}
///   |  ?[^\s\p{L}\p{N}]++[\r\n]* | \s*[\r\n] | \s+(?!\S) | \s+
///
/// matches the qwen2 pattern arm-for-arm except that qwen2's \p{N} takes
/// exactly one digit where nanochat takes one or two; the possessive
/// quantifiers and [\r\n] vs [\r\n]+ provably cannot change chunk
/// boundaries (see the port plan's pretokenizer analysis).
fn nanochatChunkEnd(c: []const u32, start: usize) usize {
    const n = c.len;
    var pos = start;
    const cp = c[pos];

    // '(?i:[sdmt]|ll|ve|re) — case-insensitive under Unicode SIMPLE case
    // folding (tiktoken's fancy-regex): besides ASCII A-Z, exactly one
    // non-ASCII codepoint folds into {s,t,m,d,r,e,v,l} — U+017F LATIN SMALL
    // LETTER LONG S → 's'.
    if (cp == '\'' and pos + 1 < n) {
        const c1 = contractionFold(c[pos + 1]);
        if (c1 == 's' or c1 == 't' or c1 == 'm' or c1 == 'd') return pos + 2;
        if (pos + 2 < n) {
            const c2 = contractionFold(c[pos + 2]);
            if ((c1 == 'r' and c2 == 'e') or (c1 == 'v' and c2 == 'e') or (c1 == 'l' and c2 == 'l')) return pos + 3;
        }
    }

    // [^\r\n\p{L}\p{N}]?+\p{L}+
    if (!(cp == '\r' or cp == '\n' or ucat.isNumber(cp))) {
        if (ucat.isLetter(cp) or (pos + 1 < n and ucat.isLetter(c[pos + 1]))) {
            pos += 1;
            while (pos < n and ucat.isLetter(c[pos])) pos += 1;
            return pos;
        }
    }

    // \p{N}{1,2} — one or two digits per chunk.
    if (ucat.isNumber(cp)) {
        if (pos + 1 < n and ucat.isNumber(c[pos + 1])) return pos + 2;
        return pos + 1;
    }

    //  ?[^\s\p{L}\p{N}]++[\r\n]*
    {
        const j = if (cp == ' ') pos + 1 else pos;
        // j >= n (lookahead past the end) enters the branch: a trailing lone
        // space emits itself.
        const enter = j >= n or !(ucat.isWhitespace(c[j]) or ucat.isLetter(c[j]) or ucat.isNumber(c[j]));
        if (enter) {
            pos = j;
            while (pos < n and !(ucat.isWhitespace(c[pos]) or ucat.isLetter(c[pos]) or ucat.isNumber(c[pos]))) pos += 1;
            while (pos < n and (c[pos] == '\r' or c[pos] == '\n')) pos += 1;
            return pos;
        }
    }

    // Measure the whitespace run once for the three \s rules.
    var num_ws: usize = 0;
    var last_rn: usize = 0; // index just past the last \r or \n in the run
    while (pos + num_ws < n and ucat.isWhitespace(c[pos + num_ws])) {
        const w = c[pos + num_ws];
        if (w == '\r' or w == '\n') last_rn = pos + num_ws + 1;
        num_ws += 1;
    }

    // \s*[\r\n] — through the last newline of the run.
    if (last_rn > 0) return last_rn;

    // \s+(?!\S) — all but the last space when a non-space follows the run.
    if (num_ws > 1 and pos + num_ws < n) return pos + num_ws - 1;

    // \s+
    if (num_ws > 0) return pos + num_ws;

    // Unreachable for in-range codepoints (every category falls in a rule
    // above) — defensive single-codepoint fallback.
    return pos + 1;
}

fn contractionFold(cp: u32) u32 {
    if (cp == 0x017F) return 's'; // U+017F ſ simple-case-folds to 's'
    return if (cp >= 'A' and cp <= 'Z') cp + 32 else cp;
}

/// Decode UTF-8 `text` into codepoints + byte offsets (offs gets a text.len
/// sentinel). Invalid UTF-8: each undecodable byte classifies as one U+FFFD
/// codepoint while the raw bytes stay in the emitted chunk (same policy as
/// src/llm/tokenizer.zig's encodeRegular).
fn decodeCps(allocator: Allocator, text: []const u8, cps: *std.ArrayList(u32), offs: *std.ArrayList(usize)) !void {
    cps.clearRetainingCapacity();
    offs.clearRetainingCapacity();
    try cps.ensureTotalCapacity(allocator, text.len);
    try offs.ensureTotalCapacity(allocator, text.len + 1);

    var i: usize = 0;
    while (i < text.len) {
        const advance: usize, const cp: u32 = blk: {
            const len = std.unicode.utf8ByteSequenceLength(text[i]) catch break :blk .{ 1, 0xFFFD };
            if (i + len > text.len) break :blk .{ 1, 0xFFFD };
            const cp = std.unicode.utf8Decode(text[i..][0..len]) catch break :blk .{ 1, 0xFFFD };
            break :blk .{ len, cp };
        };
        offs.appendAssumeCapacity(i);
        cps.appendAssumeCapacity(cp);
        i += advance;
    }
    offs.appendAssumeCapacity(text.len);
}

// ---------------------------------------------------------------------------
// NCTXT_01 trainer corpus
// ---------------------------------------------------------------------------

pub const DocsFile = struct {
    bytes: []u8,
    /// Documents in file order (slices into `bytes`).
    docs: [][]const u8,

    pub fn deinit(self: *DocsFile, allocator: Allocator) void {
        allocator.free(self.docs);
        allocator.free(self.bytes);
        self.* = undefined;
    }
};

/// Read an NCTXT_01 trainer corpus. Caller deinits the result.
pub fn readDocsFile(allocator: Allocator, io: std.Io, path: []const u8) !DocsFile {
    const bytes = try readFileBytes(allocator, io, path);
    errdefer allocator.free(bytes);

    var r = BinReader{ .bytes = bytes };
    const magic = try r.take(8);
    if (!std.mem.eql(u8, magic, "NCTXT_01")) return error.CorruptDocsFile;
    const n_docs = try r.u32le();

    const docs = try allocator.alloc([]const u8, n_docs);
    errdefer allocator.free(docs);
    for (docs) |*doc| {
        const len = try r.u32le();
        doc.* = try r.take(len);
    }
    if (r.off != bytes.len) return error.CorruptDocsFile;

    return .{ .bytes = bytes, .docs = docs };
}

// ---------------------------------------------------------------------------
// Shared binary-file helpers
// ---------------------------------------------------------------------------

const BinReader = struct {
    bytes: []const u8,
    off: usize = 0,

    fn take(self: *BinReader, len: usize) ![]const u8 {
        if (self.bytes.len - self.off < len) return error.EndOfStream;
        defer self.off += len;
        return self.bytes[self.off .. self.off + len];
    }

    fn u32le(self: *BinReader) !u32 {
        return std.mem.readInt(u32, (try self.take(4))[0..4], .little);
    }

    fn u16le(self: *BinReader) !u16 {
        return std.mem.readInt(u16, (try self.take(2))[0..2], .little);
    }
};

fn readFileBytes(allocator: Allocator, io: std.Io, path: []const u8) ![]u8 {
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    if (stat.kind != .file) return error.NotAFile;
    const bytes = try allocator.alloc(u8, @intCast(stat.size));
    errdefer allocator.free(bytes);
    var read_len: usize = 0;
    while (read_len < bytes.len) {
        const n = try file.readStreaming(io, &.{bytes[read_len..]});
        if (n == 0) return error.EndOfStream;
        read_len += n;
    }
    return bytes;
}

// ---------------------------------------------------------------------------
// Tests — always-on (no goldens, no env gate)
// ---------------------------------------------------------------------------

/// Assert the nanochat pretokenizer splits `text` into exactly `expected`
/// chunks.
fn expectChunks(text: []const u8, expected: []const []const u8) !void {
    const allocator = std.testing.allocator;
    var cps: std.ArrayList(u32) = .empty;
    defer cps.deinit(allocator);
    var offs: std.ArrayList(usize) = .empty;
    defer offs.deinit(allocator);
    try decodeCps(allocator, text, &cps, &offs);

    var pos: usize = 0;
    var idx: usize = 0;
    while (pos < cps.items.len) {
        const end = nanochatChunkEnd(cps.items, pos);
        try std.testing.expect(end > pos);
        try std.testing.expect(idx < expected.len);
        try std.testing.expectEqualStrings(expected[idx], text[offs.items[pos]..offs.items[end]]);
        idx += 1;
        pos = end;
    }
    try std.testing.expectEqual(expected.len, idx);
}

test "chunker: \\p{N}{1,2} number arm and shared arms" {
    try expectChunks("12", &.{"12"});
    try expectChunks("123", &.{ "12", "3" });
    try expectChunks("1234", &.{ "12", "34" });
    try expectChunks("hello world", &.{ "hello", " world" });
    try expectChunks("I'm", &.{ "I", "'m" });
    try expectChunks("word2vec 100%", &.{ "word", "2", "vec", " ", "10", "0", "%" });
    try expectChunks("a\n\nb", &.{ "a", "\n\n", "b" });
    try expectChunks("  x", &.{ " ", " x" });
}

test "trainer: highest-count pair wins, ascending-pair tie-break" {
    const allocator = std.testing.allocator;

    // "ab" x10 vs "cd" x5, one merge: (97, 98) wins on count.
    var docs_ab: [15][]const u8 = undefined;
    for (docs_ab[0..10]) |*d| d.* = "ab";
    for (docs_ab[10..15]) |*d| d.* = "cd";
    var tok = try Tokenizer.trainFromDocs(allocator, &docs_ab, 256 + 1 + n_special);
    defer tok.deinit();
    try std.testing.expectEqual(@as(usize, 1), tok.merge_list.len);
    try std.testing.expectEqual(Pair{ .left = 97, .right = 98 }, tok.merge_list[0]);

    // Equal counts: the lexicographically smallest pair merges first.
    var tie = try Tokenizer.trainFromDocs(allocator, &.{ "cd", "ab" }, 256 + 1 + n_special);
    defer tie.deinit();
    try std.testing.expectEqual(Pair{ .left = 97, .right = 98 }, tie.merge_list[0]);
}

test "trainer + encode: chained merges (rustbpe reference cases)" {
    const allocator = std.testing.allocator;

    // "aaa" x10 → (97,97)->256 then (256,97)->257.
    var docs: [10][]const u8 = undefined;
    for (&docs) |*d| d.* = "aaa";
    var tok = try Tokenizer.trainFromDocs(allocator, &docs, 256 + 2 + n_special);
    defer tok.deinit();
    try std.testing.expectEqual(@as(usize, 2), tok.merge_list.len);
    try std.testing.expectEqual(Pair{ .left = 97, .right = 97 }, tok.merge_list[0]);
    try std.testing.expectEqual(Pair{ .left = 256, .right = 97 }, tok.merge_list[1]);
    try std.testing.expectEqual(@as(u32, 256 + 2 + n_special), tok.n_vocab);
    try std.testing.expectEqual(@as(u32, 256 + 2), tok.bosId());

    // rustbpe test_encode_chained_merges.
    const aaa = try tok.encode(allocator, "aaa");
    defer allocator.free(aaa);
    try std.testing.expectEqualSlices(u32, &.{257}, aaa);
    const aaaa = try tok.encode(allocator, "aaaa");
    defer allocator.free(aaaa);
    try std.testing.expectEqualSlices(u32, &.{ 256, 256 }, aaaa);
    const aaaaa = try tok.encode(allocator, "aaaaa");
    defer allocator.free(aaaaa);
    try std.testing.expectEqualSlices(u32, &.{ 256, 257 }, aaaaa);

    const with_bos = try tok.encodeWithBos(allocator, "aaa");
    defer allocator.free(with_bos);
    try std.testing.expectEqualSlices(u32, &.{ tok.bosId(), 257 }, with_bos);
}

test "encode/decode round-trip on ASCII" {
    const allocator = std.testing.allocator;
    var docs: [4][]const u8 = undefined;
    for (&docs) |*d| d.* = "the quick brown fox 42!";
    var tok = try Tokenizer.trainFromDocs(allocator, &docs, 256 + 8 + n_special);
    defer tok.deinit();

    const texts = [_][]const u8{
        "the quick brown fox 42!",
        "I'm here.\n  New line, 12345 things?!",
        "unrelated bytes: ~^|#",
    };
    for (texts) |text| {
        const ids = try tok.encode(allocator, text);
        defer allocator.free(ids);
        const back = try tok.decode(allocator, ids);
        defer allocator.free(back);
        try std.testing.expectEqualStrings(text, back);
    }
}

test "tokenizer.bin save/load round-trip" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var docs: [10][]const u8 = undefined;
    for (&docs) |*d| d.* = "abab cdcd";
    var tok = try Tokenizer.trainFromDocs(allocator, &docs, 256 + 6 + n_special);
    defer tok.deinit();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir_path = try tmp.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(dir_path);
    const path = try std.fs.path.join(allocator, &.{ dir_path, "tok.bin" });
    defer allocator.free(path);

    try tok.saveBin(io, path);
    var loaded = try Tokenizer.loadBin(allocator, io, path);
    defer loaded.deinit();

    try std.testing.expectEqual(tok.n_vocab, loaded.n_vocab);
    try std.testing.expectEqualSlices(Pair, tok.merge_list, loaded.merge_list);
    for (special_tokens, 0..) |name, k| {
        try std.testing.expectEqual(tok.specialBase() + @as(u32, @intCast(k)), loaded.specialId(name).?);
    }
    const ids = try loaded.encode(allocator, "abab cdcd!");
    defer allocator.free(ids);
    const ids_orig = try tok.encode(allocator, "abab cdcd!");
    defer allocator.free(ids_orig);
    try std.testing.expectEqualSlices(u32, ids_orig, ids);
}

// ---------------------------------------------------------------------------
// Tests — parity gates (need refs/nanochat-goldens/, env-gated like
// OMNIVOICE_PARITY: run with `NANOCHAT_PARITY=1 zig build test`)
// ---------------------------------------------------------------------------

const goldens_dir = "refs/nanochat-goldens";

test "NANOCHAT_PARITY: encode matches tiktoken encode_ordinary on all fixtures" {
    if (std.testing.environ.getPosix("NANOCHAT_PARITY") == null) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var tok = try Tokenizer.loadBin(allocator, io, goldens_dir ++ "/tokenizer.bin");
    defer tok.deinit();
    try std.testing.expectEqual(@as(u32, 32768), tok.n_vocab);

    const bytes = try readFileBytes(allocator, io, goldens_dir ++ "/ids_parity.bin");
    defer allocator.free(bytes);
    var r = BinReader{ .bytes = bytes };
    try std.testing.expectEqualStrings("NCIDS_01", try r.take(8));
    const n_items = try r.u32le();
    try std.testing.expectEqual(@as(u32, 207), n_items);

    for (0..n_items) |_| {
        const text = try r.take(try r.u32le());
        const n_ids = try r.u32le();
        const expected = try allocator.alloc(u32, n_ids);
        defer allocator.free(expected);
        for (expected) |*id| id.* = try r.u32le();

        const got = try tok.encode(allocator, text);
        defer allocator.free(got);
        try std.testing.expectEqualSlices(u32, expected, got);

        const back = try tok.decode(allocator, expected);
        defer allocator.free(back);
        try std.testing.expectEqualStrings(text, back);
    }
    try std.testing.expectEqual(bytes.len, r.off);
}

test "NANOCHAT_PARITY: tokenizer.bin round-trip + token_bytes vs goldens" {
    if (std.testing.environ.getPosix("NANOCHAT_PARITY") == null) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var tok = try Tokenizer.loadBin(allocator, io, goldens_dir ++ "/tokenizer.bin");
    defer tok.deinit();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir_path = try tmp.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(dir_path);

    // saveBin(loadBin(x)) is byte-identical to x.
    const tok_path = try std.fs.path.join(allocator, &.{ dir_path, "tokenizer.bin" });
    defer allocator.free(tok_path);
    try tok.saveBin(io, tok_path);
    const golden = try readFileBytes(allocator, io, goldens_dir ++ "/tokenizer.bin");
    defer allocator.free(golden);
    const saved = try readFileBytes(allocator, io, tok_path);
    defer allocator.free(saved);
    try std.testing.expectEqualSlices(u8, golden, saved);

    // token_bytes computed from the vocab matches the exported table.
    const tb_path = try std.fs.path.join(allocator, &.{ dir_path, "token_bytes.bin" });
    defer allocator.free(tb_path);
    try tok.saveTokenBytes(allocator, io, tb_path);
    const golden_tb = try readFileBytes(allocator, io, goldens_dir ++ "/token_bytes.bin");
    defer allocator.free(golden_tb);
    const saved_tb = try readFileBytes(allocator, io, tb_path);
    defer allocator.free(saved_tb);
    try std.testing.expectEqualSlices(u8, golden_tb, saved_tb);
}

test "NANOCHAT_PARITY: trainer merge list matches rustbpe on train_text_small (vocab 1024)" {
    if (std.testing.environ.getPosix("NANOCHAT_PARITY") == null) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var docs = try readDocsFile(allocator, io, goldens_dir ++ "/train_text_small.bin");
    defer docs.deinit(allocator);

    var tok = try Tokenizer.trainFromDocs(allocator, docs.docs, 1024);
    defer tok.deinit();

    var ref = try Tokenizer.loadBin(allocator, io, goldens_dir ++ "/tokref_v1024.bin");
    defer ref.deinit();

    try std.testing.expectEqual(ref.n_vocab, tok.n_vocab);
    try std.testing.expectEqualSlices(Pair, ref.merge_list, tok.merge_list);

    // The saved artifact is byte-identical to the reference-produced one.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir_path = try tmp.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(dir_path);
    const path = try std.fs.path.join(allocator, &.{ dir_path, "tokref.bin" });
    defer allocator.free(path);
    try tok.saveBin(io, path);
    const golden = try readFileBytes(allocator, io, goldens_dir ++ "/tokref_v1024.bin");
    defer allocator.free(golden);
    const saved = try readFileBytes(allocator, io, path);
    defer allocator.free(saved);
    try std.testing.expectEqualSlices(u8, golden, saved);
}

test {
    std.testing.refAllDecls(@This());
}
