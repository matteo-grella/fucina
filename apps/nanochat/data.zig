//! nanochat data pipeline: NCDOC/JSONL readers, conversation rendering, and the
//! two reference dataloaders (base BOS-bestfit for pretraining, SFT bestfit-pad
//! for fine-tuning). Faithful CPU ports of:
//!   - refs/nanochat/nanochat/tokenizer.py   render_conversation
//!   - refs/nanochat/nanochat/dataloader.py  tokenizing_distributed_data_loader_
//!                                            with_state_bos_bestfit
//!   - refs/nanochat/scripts/chat_sft.py     sft_data_generator_bos_bestfit
//!
//! Design: ALL dataset/mixture shuffle order is baked in Python. The
//! exporter materializes the final mixture-ordered conversation stream (JSONL)
//! and the base docs in exact (shard, row_group, doc) order (NCDOC); Zig reads
//! them in order and only ports render_conversation + the two packing loops.
//!
//! Interchange formats: NCDOC_01 framed docs + <name>.idx.json rowgroup
//! boundaries, and tasks JSONL (one {"messages": [...]} object per line) —
//! layouts documented at their readers below. All integers little-endian.
//!
//! Single-device semantics only (ddp_rank=0, ddp_world_size=1), which is what
//! the parity gates exercise; the base loader's (pq_idx, rg_idx, epoch) state is
//! reconstructed for the val split (a single parquet ⇒ pq_idx == 0, rg_idx = the
//! flat row-group index, epoch increments after each full pass).

const std = @import("std");
const tokenizer = @import("tokenizer.zig");

const Allocator = std.mem.Allocator;
const Tokenizer = tokenizer.Tokenizer;

/// dataloader.py tokenizer_batch_size: each refill reads the parquet row group
/// in chunks of this many documents, and every chunk carries the (pq, rg, epoch)
/// state used for resume.
const tokenizer_batch_size: usize = 128;

// ===========================================================================
// Conversation model + JSONL reader
// ===========================================================================

pub const Role = enum { system, user, assistant };

/// A part of an assistant message (GSM8K tool use); tokenizer.py part["type"].
pub const PartKind = enum { text, python, python_output };

pub const Part = struct { kind: PartKind, text: []const u8 };

/// Message content is either a plain string or a list of parts (GSM8K).
pub const Content = union(enum) {
    text: []const u8,
    parts: []const Part,
};

pub const Message = struct { role: Role, content: Content };

pub const Conversation = struct { messages: []const Message };

/// A set of conversations parsed from a tasks-JSONL file. All strings/slices are
/// owned by `arena`; `convs` are in file (= mixture) order.
pub const JsonlConvs = struct {
    arena: std.heap.ArenaAllocator,
    convs: []Conversation,

    pub fn deinit(self: *JsonlConvs) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

fn roleFromStr(s: []const u8) !Role {
    if (std.mem.eql(u8, s, "user")) return .user;
    if (std.mem.eql(u8, s, "assistant")) return .assistant;
    if (std.mem.eql(u8, s, "system")) return .system;
    return error.BadRole;
}

fn partKindFromStr(s: []const u8) !PartKind {
    if (std.mem.eql(u8, s, "text")) return .text;
    if (std.mem.eql(u8, s, "python")) return .python;
    if (std.mem.eql(u8, s, "python_output")) return .python_output;
    return error.BadPartKind;
}

/// Read a tasks-JSONL file: one `{"messages":[{"role","content"}...]}` object
/// per line. `content` is a string, or an array of {"type","text"} parts. Blank
/// lines are skipped. Strings are duped into the returned arena.
pub fn readJsonlConvs(gpa: Allocator, io: std.Io, path: []const u8) !JsonlConvs {
    const bytes = try readFileBytes(gpa, io, path);
    defer gpa.free(bytes);

    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();

    var convs: std.ArrayList(Conversation) = .empty;

    var lines = std.mem.splitScalar(u8, bytes, '\n');
    while (lines.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \t\r");
        if (line.len == 0) continue;

        var parsed = try std.json.parseFromSlice(std.json.Value, gpa, line, .{});
        defer parsed.deinit();
        if (parsed.value != .object) return error.MalformedJsonl;

        const messages_v = parsed.value.object.get("messages") orelse return error.MalformedJsonl;
        if (messages_v != .array) return error.MalformedJsonl;
        const msg_items = messages_v.array.items;

        const messages = try a.alloc(Message, msg_items.len);
        for (msg_items, messages) |mv, *out| {
            if (mv != .object) return error.MalformedJsonl;
            const role_v = mv.object.get("role") orelse return error.MalformedJsonl;
            const content_v = mv.object.get("content") orelse return error.MalformedJsonl;
            if (role_v != .string) return error.MalformedJsonl;
            const role = try roleFromStr(role_v.string);

            const content: Content = switch (content_v) {
                .string => |s| .{ .text = try a.dupe(u8, s) },
                .array => |arr| blk: {
                    const parts = try a.alloc(Part, arr.items.len);
                    for (arr.items, parts) |pv, *pout| {
                        if (pv != .object) return error.MalformedJsonl;
                        const type_v = pv.object.get("type") orelse return error.MalformedJsonl;
                        const text_v = pv.object.get("text") orelse return error.MalformedJsonl;
                        if (type_v != .string or text_v != .string) return error.MalformedJsonl;
                        pout.* = .{
                            .kind = try partKindFromStr(type_v.string),
                            .text = try a.dupe(u8, text_v.string),
                        };
                    }
                    break :blk .{ .parts = parts };
                },
                else => return error.MalformedJsonl,
            };
            out.* = .{ .role = role, .content = content };
        }
        try convs.append(a, .{ .messages = messages });
    }

    return .{ .arena = arena, .convs = try convs.toOwnedSlice(a) };
}

// ===========================================================================
// render_conversation (tokenizer.py)
// ===========================================================================

/// Rendered conversation: token ids + a per-token supervision mask (1 = the
/// Assistant is trained on this token). Caller owns both slices.
pub const Rendered = struct {
    ids: []u32,
    mask: []u8,

    pub fn deinit(self: *Rendered, allocator: Allocator) void {
        allocator.free(self.ids);
        allocator.free(self.mask);
        self.* = undefined;
    }
};

fn addSingle(ids: *std.ArrayList(u32), mask: *std.ArrayList(u8), allocator: Allocator, id: u32, mask_val: u8) !void {
    try ids.append(allocator, id);
    try mask.append(allocator, mask_val);
}

fn addEncoded(ids: *std.ArrayList(u32), mask: *std.ArrayList(u8), allocator: Allocator, tok: *const Tokenizer, text: []const u8, mask_val: u8) !void {
    const before = ids.items.len;
    try tok.encodeAppend(allocator, text, ids);
    var k = ids.items.len - before;
    while (k > 0) : (k -= 1) try mask.append(allocator, mask_val);
}

/// Faithful port of RustBPETokenizer.render_conversation (tokenizer.py L140-224).
/// A leading system message is merged into the following user message; then bos
/// (mask 0), then per message user_start/content/user_end (all mask 0) or
/// assistant_start (0) + supervised content + assistant_end (1). Assistant
/// content is a string (mask 1) or parts: text → mask 1; python → python_start/
/// content/python_end all mask 1; python_output → output_start/content/
/// output_end all mask 0 (not supervised — comes from Python at test time).
/// Truncated to `max_tokens` (reference default 2048).
pub fn renderConversation(
    allocator: Allocator,
    tok: *const Tokenizer,
    conv: Conversation,
    max_tokens: usize,
) !Rendered {
    var ids: std.ArrayList(u32) = .empty;
    errdefer ids.deinit(allocator);
    var mask: std.ArrayList(u8) = .empty;
    errdefer mask.deinit(allocator);

    const msgs = conv.messages;
    std.debug.assert(msgs.len >= 1);

    // System message → merge its content with the first user message.
    var merged_user: ?[]u8 = null;
    defer if (merged_user) |m| allocator.free(m);
    var eff_start: usize = 0;
    if (msgs[0].role == .system) {
        std.debug.assert(msgs.len >= 2 and msgs[1].role == .user);
        const sys = msgs[0].content.text;
        const usr = msgs[1].content.text;
        const m = try allocator.alloc(u8, sys.len + 2 + usr.len);
        @memcpy(m[0..sys.len], sys);
        m[sys.len] = '\n';
        m[sys.len + 1] = '\n';
        @memcpy(m[sys.len + 2 ..], usr);
        merged_user = m;
        eff_start = 1;
    }

    const bos = tok.bosId();
    const user_start = tok.specialId("<|user_start|>").?;
    const user_end = tok.specialId("<|user_end|>").?;
    const assistant_start = tok.specialId("<|assistant_start|>").?;
    const assistant_end = tok.specialId("<|assistant_end|>").?;
    const python_start = tok.specialId("<|python_start|>").?;
    const python_end = tok.specialId("<|python_end|>").?;
    const output_start = tok.specialId("<|output_start|>").?;
    const output_end = tok.specialId("<|output_end|>").?;

    try addSingle(&ids, &mask, allocator, bos, 0);

    var eff_i: usize = 0; // index among effective (post-merge) messages
    var idx = eff_start;
    while (idx < msgs.len) : (idx += 1) {
        const message = msgs[idx];
        // Sanity: user/assistant strictly alternate (tokenizer.py assert).
        std.debug.assert(message.role == (if (eff_i % 2 == 0) Role.user else Role.assistant));

        if (message.role == .user) {
            const content = if (idx == eff_start and merged_user != null)
                merged_user.?
            else
                message.content.text;
            try addSingle(&ids, &mask, allocator, user_start, 0);
            try addEncoded(&ids, &mask, allocator, tok, content, 0);
            try addSingle(&ids, &mask, allocator, user_end, 0);
        } else {
            try addSingle(&ids, &mask, allocator, assistant_start, 0);
            switch (message.content) {
                .text => |s| try addEncoded(&ids, &mask, allocator, tok, s, 1),
                .parts => |parts| for (parts) |part| switch (part.kind) {
                    .text => try addEncoded(&ids, &mask, allocator, tok, part.text, 1),
                    .python => {
                        try addSingle(&ids, &mask, allocator, python_start, 1);
                        try addEncoded(&ids, &mask, allocator, tok, part.text, 1);
                        try addSingle(&ids, &mask, allocator, python_end, 1);
                    },
                    .python_output => {
                        try addSingle(&ids, &mask, allocator, output_start, 0);
                        try addEncoded(&ids, &mask, allocator, tok, part.text, 0);
                        try addSingle(&ids, &mask, allocator, output_end, 0);
                    },
                },
            }
            try addSingle(&ids, &mask, allocator, assistant_end, 1);
        }
        eff_i += 1;
    }

    // Truncate to max_tokens (helps prevent OOMs).
    const n = @min(ids.items.len, max_tokens);
    ids.items.len = n;
    mask.items.len = n;
    return .{ .ids = try ids.toOwnedSlice(allocator), .mask = try mask.toOwnedSlice(allocator) };
}

// ===========================================================================
// Best-fit selection (shared by both loaders) — dataloader.py / chat_sft.py
// ===========================================================================

/// Index of the LARGEST buffered doc whose length ≤ `remaining` (strictly-
/// greater comparison ⇒ first among equal-length ties wins), or null if none
/// fits. Matches dataloader.py's best-fit scan.
fn selectBestFit(lens: []const usize, remaining: usize) ?usize {
    var best: ?usize = null;
    var best_len: usize = 0;
    for (lens, 0..) |len, i| {
        if (len <= remaining and len > best_len) {
            best = i;
            best_len = len;
        }
    }
    return best;
}

/// Index of the SHORTEST buffered doc (first among ties) — Python
/// `min(range(len(buf)), key=lambda i: len(buf[i]))`.
fn selectShortest(lens: []const usize) usize {
    var idx: usize = 0;
    var min_len: usize = lens[0];
    for (lens, 0..) |len, i| {
        if (len < min_len) {
            min_len = len;
            idx = i;
        }
    }
    return idx;
}

// ===========================================================================
// NCDOC reader (framed docs + <name>.idx.json rowgroup boundaries)
// ===========================================================================

pub const NcDoc = struct {
    allocator: Allocator,
    bytes: []u8, // NCDOC_01 file bytes (doc text slices into this)
    docs: [][]const u8,
    docs_per_rowgroup: []u32,

    pub fn deinit(self: *NcDoc) void {
        self.allocator.free(self.docs_per_rowgroup);
        self.allocator.free(self.docs);
        self.allocator.free(self.bytes);
        self.* = undefined;
    }
};

/// Derive the "<base>.idx.json" sidecar path from an NCDOC ".bin" path (mirrors
/// the exporter's `os.path.splitext(out)[0] + ".idx.json"`). Caller frees.
fn idxJsonPath(allocator: Allocator, bin_path: []const u8) ![]u8 {
    const dot = std.mem.lastIndexOfScalar(u8, bin_path, '.') orelse bin_path.len;
    return std.fmt.allocPrint(allocator, "{s}.idx.json", .{bin_path[0..dot]});
}

/// Read an NCDOC_01 framed-docs file and its idx.json sidecar.
pub fn readNcDoc(allocator: Allocator, io: std.Io, path: []const u8) !NcDoc {
    const bytes = try readFileBytes(allocator, io, path);
    errdefer allocator.free(bytes);

    var r = BinReader{ .bytes = bytes };
    if (!std.mem.eql(u8, try r.take(8), "NCDOC_01")) return error.CorruptNcDoc;
    const n_docs = try r.u32le();
    const docs = try allocator.alloc([]const u8, n_docs);
    errdefer allocator.free(docs);
    for (docs) |*doc| {
        const len = try r.u32le();
        doc.* = try r.take(len);
    }
    if (r.off != bytes.len) return error.CorruptNcDoc;

    // idx.json: {"docs_per_rowgroup":[...], "split":"val"}
    const idx_path = try idxJsonPath(allocator, path);
    defer allocator.free(idx_path);
    const idx_bytes = try readFileBytes(allocator, io, idx_path);
    defer allocator.free(idx_bytes);

    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, idx_bytes, .{});
    defer parsed.deinit();
    const dpr_v = parsed.value.object.get("docs_per_rowgroup") orelse return error.CorruptNcDoc;
    if (dpr_v != .array) return error.CorruptNcDoc;
    const dpr = try allocator.alloc(u32, dpr_v.array.items.len);
    errdefer allocator.free(dpr);
    var sum: usize = 0;
    for (dpr_v.array.items, dpr) |v, *out| {
        out.* = @intCast(v.integer);
        sum += out.*;
    }
    if (sum != n_docs) return error.CorruptNcDoc;

    return .{ .allocator = allocator, .bytes = bytes, .docs = docs, .docs_per_rowgroup = dpr };
}

// ===========================================================================
// Base BOS-bestfit loader (dataloader.py)
// ===========================================================================

/// Resume state as yielded by the reference loader.
pub const State = struct { pq_idx: i64 = 0, rg_idx: i64 = 0, epoch: i64 = 1 };

/// A 128-doc chunk of one row group; the (pq, rg, epoch) unit for state.
const Chunk = struct { start: usize, end: usize, rg_idx: i64 };

pub const BaseBatch = struct {
    inputs: []i32, // [B*T]
    targets: []i32, // [B*T]
    state: State,
    b: usize,
    t: usize,
    allocator: Allocator,

    pub fn deinit(self: *BaseBatch) void {
        self.allocator.free(self.inputs);
        self.allocator.free(self.targets);
        self.* = undefined;
    }
};

pub const BaseLoader = struct {
    allocator: Allocator,
    tok: *const Tokenizer,
    docs: []const []const u8,
    chunks: []Chunk,
    buffer: std.ArrayList([]u32), // owned token slices, in read order
    lens: std.ArrayList(usize), // scratch: buffer doc lengths
    b: usize,
    t: usize,
    row_capacity: usize,
    buffer_size: usize,
    cursor: usize, // index into `chunks`
    epoch: i64,
    state: State, // position of the last refill (what the reference yields)

    /// `docs`/`docs_per_rowgroup` are borrowed (e.g. from an NcDoc) and must
    /// outlive the loader. buffer_size mirrors dataloader.py's `buffer_size`
    /// (reference default 1000).
    pub fn init(
        allocator: Allocator,
        tok: *const Tokenizer,
        docs: []const []const u8,
        docs_per_rowgroup: []const u32,
        b: usize,
        t: usize,
        buffer_size: usize,
    ) !BaseLoader {
        var chunks: std.ArrayList(Chunk) = .empty;
        errdefer chunks.deinit(allocator);
        var off: usize = 0;
        for (docs_per_rowgroup, 0..) |cnt, rg| {
            var s: usize = 0;
            while (s < cnt) {
                const e = @min(s + tokenizer_batch_size, cnt);
                try chunks.append(allocator, .{ .start = off + s, .end = off + e, .rg_idx = @intCast(rg) });
                s = e;
            }
            off += cnt;
        }
        std.debug.assert(off == docs.len);

        return .{
            .allocator = allocator,
            .tok = tok,
            .docs = docs,
            .chunks = try chunks.toOwnedSlice(allocator),
            .buffer = .empty,
            .lens = .empty,
            .b = b,
            .t = t,
            .row_capacity = t + 1,
            .buffer_size = buffer_size,
            .cursor = 0,
            .epoch = 1,
            .state = .{},
        };
    }

    pub fn deinit(self: *BaseLoader) void {
        for (self.buffer.items) |toks| self.allocator.free(toks);
        self.buffer.deinit(self.allocator);
        self.lens.deinit(self.allocator);
        self.allocator.free(self.chunks);
        self.* = undefined;
    }

    /// Fast-forward a fresh loader to a saved (rg_idx, epoch) checkpoint state:
    /// restart AT the saved row group, matching dataloader.py's approximate
    /// resume (`base_idx = resume_rg_idx`; the in-flight buffer is not part of
    /// the reference state either). A saved rg_idx past the last row group
    /// wraps to the next epoch.
    pub fn resumeAt(self: *BaseLoader, rg_idx: i64, epoch: i64) void {
        self.epoch = epoch;
        for (self.chunks, 0..) |chunk, i| {
            if (chunk.rg_idx == rg_idx) {
                self.cursor = i;
                return;
            }
        }
        self.cursor = 0;
        self.epoch = epoch + 1;
    }

    /// Ensure the buffer holds ≥ buffer_size docs, reading 128-doc chunks and
    /// tracking (pq_idx, rg_idx, epoch) of the last chunk read. Each doc is
    /// [bos] ++ encode(text) — dataloader.py `encode(batch, prepend=bos)`.
    fn refill(self: *BaseLoader) !void {
        while (self.buffer.items.len < self.buffer_size) {
            const chunk = self.chunks[self.cursor];
            var di = chunk.start;
            while (di < chunk.end) : (di += 1) {
                const toks = try self.tok.encodeWithBos(self.allocator, self.docs[di]);
                try self.buffer.append(self.allocator, toks);
            }
            self.state = .{ .pq_idx = 0, .rg_idx = chunk.rg_idx, .epoch = self.epoch };
            self.cursor += 1;
            if (self.cursor == self.chunks.len) {
                self.cursor = 0;
                self.epoch += 1;
            }
        }
    }

    /// Build the next (inputs, targets, state) batch. inputs = row[:, :-1],
    /// targets = row[:, 1:]; 100% utilization (best-fit fill, else crop shortest).
    pub fn nextBatch(self: *BaseLoader) !BaseBatch {
        const rc = self.row_capacity;
        const row_buf = try self.allocator.alloc(u32, self.b * rc);
        defer self.allocator.free(row_buf);

        for (0..self.b) |row_idx| {
            const row = row_buf[row_idx * rc ..][0..rc];
            var pos: usize = 0;
            while (pos < rc) {
                try self.refill();
                const remaining = rc - pos;

                self.lens.clearRetainingCapacity();
                for (self.buffer.items) |doc| try self.lens.append(self.allocator, doc.len);

                if (selectBestFit(self.lens.items, remaining)) |bi| {
                    const doc = self.buffer.orderedRemove(bi);
                    @memcpy(row[pos..][0..doc.len], doc);
                    pos += doc.len;
                    self.allocator.free(doc);
                } else {
                    const si = selectShortest(self.lens.items);
                    const doc = self.buffer.orderedRemove(si);
                    @memcpy(row[pos..][0..remaining], doc[0..remaining]);
                    pos += remaining;
                    self.allocator.free(doc);
                }
            }
        }

        const inputs = try self.allocator.alloc(i32, self.b * self.t);
        errdefer self.allocator.free(inputs);
        const targets = try self.allocator.alloc(i32, self.b * self.t);
        errdefer self.allocator.free(targets);
        for (0..self.b) |r| {
            const row = row_buf[r * rc ..][0..rc];
            for (0..self.t) |c| {
                inputs[r * self.t + c] = @intCast(row[c]);
                targets[r * self.t + c] = @intCast(row[c + 1]);
            }
        }
        return .{ .inputs = inputs, .targets = targets, .state = self.state, .b = self.b, .t = self.t, .allocator = self.allocator };
    }
};

// ===========================================================================
// SFT bestfit-pad loader (chat_sft.py sft_data_generator_bos_bestfit)
// ===========================================================================

pub const SftBatch = struct {
    inputs: []i32, // [B*T]
    targets: []i64, // [B*T], −1 = ignore
    b: usize,
    t: usize,
    allocator: Allocator,

    pub fn deinit(self: *SftBatch) void {
        self.allocator.free(self.inputs);
        self.allocator.free(self.targets);
        self.* = undefined;
    }
};

pub const SftLoader = struct {
    allocator: Allocator,
    tok: *const Tokenizer,
    convs: []const Conversation, // mixture order, borrowed
    buffer: std.ArrayList(Rendered),
    lens: std.ArrayList(usize),
    b: usize,
    t: usize,
    row_capacity: usize,
    buffer_size: usize,
    cursor: usize,
    epoch: i64,

    /// buffer_size mirrors chat_sft.py's `buffer_size` (reference default 100).
    pub fn init(
        allocator: Allocator,
        tok: *const Tokenizer,
        convs: []const Conversation,
        b: usize,
        t: usize,
        buffer_size: usize,
    ) !SftLoader {
        std.debug.assert(convs.len > 0);
        return .{
            .allocator = allocator,
            .tok = tok,
            .convs = convs,
            .buffer = .empty,
            .lens = .empty,
            .b = b,
            .t = t,
            .row_capacity = t + 1,
            .buffer_size = buffer_size,
            .cursor = 0,
            .epoch = 1,
        };
    }

    pub fn deinit(self: *SftLoader) void {
        for (self.buffer.items) |*r| r.deinit(self.allocator);
        self.buffer.deinit(self.allocator);
        self.lens.deinit(self.allocator);
        self.* = undefined;
    }

    /// Fast-forward a fresh loader to a saved (cursor, epoch) checkpoint state
    /// (same approximate-resume semantics as BaseLoader.resumeAt: the in-flight
    /// buffer is not restored).
    pub fn resumeAt(self: *SftLoader, cursor: usize, epoch: i64) void {
        self.cursor = cursor % self.convs.len;
        self.epoch = epoch;
    }

    fn refill(self: *SftLoader) !void {
        while (self.buffer.items.len < self.buffer_size) {
            // Render cap: the reference default 2048, clamped to the row size.
            // A render longer than row_capacity could never be seated by
            // selectBestFit and the no-fit branch removes nothing, so at
            // max_seq_len < 2047 over-long conversations would permanently
            // occupy buffer slots until every batch degraded to pure padding
            // (chat_sft.py only ever runs at max_seq_len=2048, where every
            // render fits an empty row and the clamp is a no-op).
            const cap = @min(2048, self.row_capacity);
            const r = try renderConversation(self.allocator, self.tok, self.convs[self.cursor], cap);
            try self.buffer.append(self.allocator, r);
            self.cursor += 1;
            if (self.cursor >= self.convs.len) {
                self.cursor = 0;
                self.epoch += 1;
            }
        }
    }

    /// Build the next (inputs, targets) batch. Rows are packed best-fit; when no
    /// buffered conversation fits, the remainder is padded with bos (mask 0) so
    /// no tokens are ever discarded. targets = row[1:] with unsupervised
    /// (mask==0) and padding positions set to −1 (ignore).
    pub fn nextBatch(self: *SftLoader) !SftBatch {
        const rc = self.row_capacity;
        const row_buf = try self.allocator.alloc(u32, self.b * rc);
        defer self.allocator.free(row_buf);
        const mask_buf = try self.allocator.alloc(u8, self.b * rc);
        defer self.allocator.free(mask_buf);
        const content_lens = try self.allocator.alloc(usize, self.b);
        defer self.allocator.free(content_lens);

        const bos = self.tok.bosId();

        for (0..self.b) |row_idx| {
            const row = row_buf[row_idx * rc ..][0..rc];
            const mrow = mask_buf[row_idx * rc ..][0..rc];
            var pos: usize = 0;
            var padded = false;
            var content_len: usize = rc;
            while (pos < rc) {
                try self.refill();
                const remaining = rc - pos;

                self.lens.clearRetainingCapacity();
                for (self.buffer.items) |rend| try self.lens.append(self.allocator, rend.ids.len);

                if (selectBestFit(self.lens.items, remaining)) |bi| {
                    var rend = self.buffer.orderedRemove(bi);
                    @memcpy(row[pos..][0..rend.ids.len], rend.ids);
                    @memcpy(mrow[pos..][0..rend.mask.len], rend.mask);
                    pos += rend.ids.len;
                    rend.deinit(self.allocator);
                } else {
                    content_len = pos;
                    for (row[pos..rc]) |*x| x.* = bos;
                    for (mrow[pos..rc]) |*x| x.* = 0;
                    padded = true;
                    pos = rc;
                }
            }
            content_lens[row_idx] = if (padded) content_len else rc;
        }

        const inputs = try self.allocator.alloc(i32, self.b * self.t);
        errdefer self.allocator.free(inputs);
        const targets = try self.allocator.alloc(i64, self.b * self.t);
        errdefer self.allocator.free(targets);
        for (0..self.b) |r| {
            const row = row_buf[r * rc ..][0..rc];
            const mrow = mask_buf[r * rc ..][0..rc];
            for (0..self.t) |c| {
                inputs[r * self.t + c] = @intCast(row[c]);
                targets[r * self.t + c] = if (mrow[c + 1] == 0) -1 else @intCast(row[c + 1]);
            }
            // Mask padding positions: targets[content_len-1:] = −1 (Python's
            // content_len==0 slice [-1:] touches only the last column).
            const cl = content_lens[r];
            if (cl < rc) {
                var c: usize = if (cl == 0) self.t - 1 else cl - 1;
                while (c < self.t) : (c += 1) targets[r * self.t + c] = -1;
            }
        }
        return .{ .inputs = inputs, .targets = targets, .b = self.b, .t = self.t, .allocator = self.allocator };
    }
};

// ===========================================================================
// Shared binary-file helpers
// ===========================================================================

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
    fn i32le(self: *BinReader) !i32 {
        return std.mem.readInt(i32, (try self.take(4))[0..4], .little);
    }
    fn i64le(self: *BinReader) !i64 {
        return std.mem.readInt(i64, (try self.take(8))[0..8], .little);
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

// ===========================================================================
// Tests — always-on (no goldens, no env gate)
// ===========================================================================

test "selectBestFit / selectShortest match dataloader.py semantics" {
    // Largest that fits; strictly-greater tie-break keeps the FIRST max.
    const lens = [_]usize{ 3, 5, 5, 2, 4 };
    try std.testing.expectEqual(@as(?usize, 1), selectBestFit(&lens, 5)); // first 5
    try std.testing.expectEqual(@as(?usize, 4), selectBestFit(&lens, 4)); // 4
    try std.testing.expectEqual(@as(?usize, 3), selectBestFit(&lens, 2)); // only 2 fits
    try std.testing.expectEqual(@as(?usize, null), selectBestFit(&lens, 1)); // none fits
    // Shortest, first among ties.
    const lens2 = [_]usize{ 4, 2, 7, 2 };
    try std.testing.expectEqual(@as(usize, 1), selectShortest(&lens2));
}

/// Build a tiny raw-byte tokenizer whose corpus covers the test words, so the 9
/// specials exist and encode/decode round-trips.
fn tinyTokenizer(allocator: Allocator) !Tokenizer {
    var docs: [8][]const u8 = undefined;
    for (&docs) |*d| d.* = "the cat sat on the mat and ran fast 12 plus 3 equals 15";
    return Tokenizer.trainFromDocs(allocator, &docs, 256 + 40 + tokenizer.n_special);
}

test "renderConversation: user/assistant masks + special placement" {
    const allocator = std.testing.allocator;
    var tok = try tinyTokenizer(allocator);
    defer tok.deinit();

    const conv = Conversation{ .messages = &.{
        .{ .role = .user, .content = .{ .text = "the cat" } },
        .{ .role = .assistant, .content = .{ .text = "sat" } },
    } };
    var r = try renderConversation(allocator, &tok, conv, 2048);
    defer r.deinit(allocator);

    try std.testing.expectEqual(r.ids.len, r.mask.len);
    // bos then user_start, both unsupervised.
    try std.testing.expectEqual(tok.bosId(), r.ids[0]);
    try std.testing.expectEqual(@as(u8, 0), r.mask[0]);
    try std.testing.expectEqual(tok.specialId("<|user_start|>").?, r.ids[1]);
    // Ends with assistant_end, supervised (mask 1).
    try std.testing.expectEqual(tok.specialId("<|assistant_end|>").?, r.ids[r.ids.len - 1]);
    try std.testing.expectEqual(@as(u8, 1), r.mask[r.mask.len - 1]);

    // Structure invariants: exactly one of each turn special, correct masks.
    const us = tok.specialId("<|user_start|>").?;
    const ue = tok.specialId("<|user_end|>").?;
    const as = tok.specialId("<|assistant_start|>").?;
    const ae = tok.specialId("<|assistant_end|>").?;
    var i_us: ?usize = null;
    var i_ue: ?usize = null;
    var i_as: ?usize = null;
    var i_ae: ?usize = null;
    for (r.ids, 0..) |id, i| {
        if (id == us) i_us = i;
        if (id == ue) i_ue = i;
        if (id == as) i_as = i;
        if (id == ae) i_ae = i;
    }
    // Order: bos < user_start < user_end < assistant_start < assistant_end.
    try std.testing.expect(i_us.? < i_ue.? and i_ue.? < i_as.? and i_as.? < i_ae.?);
    // User content (between user_start and user_end) is unsupervised.
    for (i_us.? + 1..i_ue.?) |i| try std.testing.expectEqual(@as(u8, 0), r.mask[i]);
    // assistant_start unsupervised; assistant content supervised.
    try std.testing.expectEqual(@as(u8, 0), r.mask[i_as.?]);
    for (i_as.? + 1..i_ae.?) |i| try std.testing.expectEqual(@as(u8, 1), r.mask[i]);
}

test "renderConversation: system merge + GSM8K tool-use parts" {
    const allocator = std.testing.allocator;
    var tok = try tinyTokenizer(allocator);
    defer tok.deinit();

    // System message must be merged into the first user message (mask 0), and
    // its content tokens must appear before the user's.
    const sys_user = Conversation{ .messages = &.{
        .{ .role = .system, .content = .{ .text = "the mat" } },
        .{ .role = .user, .content = .{ .text = "cat" } },
        .{ .role = .assistant, .content = .{ .text = "ran" } },
    } };
    var r1 = try renderConversation(allocator, &tok, sys_user, 2048);
    defer r1.deinit(allocator);
    // Merged user content = "the mat" ++ "\n\n" ++ "cat"; all user tokens mask 0.
    const us = tok.specialId("<|user_start|>").?;
    const ue = tok.specialId("<|user_end|>").?;
    var seen_us = false;
    for (r1.ids, r1.mask) |id, m| {
        if (id == us) seen_us = true;
        if (seen_us and id == ue) break;
        if (seen_us) try std.testing.expectEqual(@as(u8, 0), m);
    }

    // GSM8K-style tool use: python (supervised) + python_output (NOT supervised).
    const gsm = Conversation{ .messages = &.{
        .{ .role = .user, .content = .{ .text = "12 plus 3" } },
        .{ .role = .assistant, .content = .{ .parts = &.{
            .{ .kind = .text, .text = "the sum is " },
            .{ .kind = .python, .text = "12 plus 3" },
            .{ .kind = .python_output, .text = "15" },
            .{ .kind = .text, .text = "so 15" },
        } } },
    } };
    var r2 = try renderConversation(allocator, &tok, gsm, 2048);
    defer r2.deinit(allocator);

    const ps = tok.specialId("<|python_start|>").?;
    const pe = tok.specialId("<|python_end|>").?;
    const os_ = tok.specialId("<|output_start|>").?;
    const oe = tok.specialId("<|output_end|>").?;
    var i_ps: ?usize = null;
    var i_pe: ?usize = null;
    var i_os: ?usize = null;
    var i_oe: ?usize = null;
    for (r2.ids, 0..) |id, i| {
        if (id == ps) i_ps = i;
        if (id == pe) i_pe = i;
        if (id == os_) i_os = i;
        if (id == oe) i_oe = i;
    }
    // python_start/content/python_end all supervised (mask 1).
    for (i_ps.?..i_pe.? + 1) |i| try std.testing.expectEqual(@as(u8, 1), r2.mask[i]);
    // output_start/content/output_end all unsupervised (mask 0).
    for (i_os.?..i_oe.? + 1) |i| try std.testing.expectEqual(@as(u8, 0), r2.mask[i]);
    // Ordering: python block precedes the output block.
    try std.testing.expect(i_pe.? < i_os.?);
}

test "BaseLoader: BOS-aligned, 100% utilization, best-fit + crop" {
    const allocator = std.testing.allocator;
    var tok = try tinyTokenizer(allocator);
    defer tok.deinit();

    // A handful of in-memory docs (one row group), small buffer so refill loops.
    const docs = [_][]const u8{
        "the cat sat",
        "on the mat",
        "ran fast",
        "12 plus 3 equals 15",
        "the",
        "cat",
    };
    const dpr = [_]u32{docs.len};
    var loader = try BaseLoader.init(allocator, &tok, &docs, &dpr, 2, 8, 3);
    defer loader.deinit();

    var batch = try loader.nextBatch();
    defer batch.deinit();

    const bos: i32 = @intCast(tok.bosId());
    // Every row starts with BOS and is fully filled (100% utilization: T inputs).
    for (0..batch.b) |row| {
        try std.testing.expectEqual(bos, batch.inputs[row * batch.t]);
    }
    try std.testing.expectEqual(@as(usize, 2 * 8), batch.inputs.len);
    // targets are inputs shifted by one within the (T+1)-token row.
    // Reconstruct row 0's last input/first-target adjacency is exercised by the
    // parity gate; here we just assert shapes + BOS alignment + state.
    try std.testing.expectEqual(@as(i64, 0), batch.state.pq_idx);
    try std.testing.expectEqual(@as(i64, 1), batch.state.epoch);
}

test "SftLoader: bestfit-pad, BOS rows, −1 ignore targets" {
    const allocator = std.testing.allocator;
    var tok = try tinyTokenizer(allocator);
    defer tok.deinit();

    const convs = [_]Conversation{
        .{ .messages = &.{
            .{ .role = .user, .content = .{ .text = "the cat" } },
            .{ .role = .assistant, .content = .{ .text = "sat" } },
        } },
        .{ .messages = &.{
            .{ .role = .user, .content = .{ .text = "on the" } },
            .{ .role = .assistant, .content = .{ .text = "mat" } },
        } },
    };
    var loader = try SftLoader.init(allocator, &tok, &convs, 2, 32, 2);
    defer loader.deinit();

    var batch = try loader.nextBatch();
    defer batch.deinit();

    const bos: i32 = @intCast(tok.bosId());
    try std.testing.expectEqual(@as(usize, 2 * 32), batch.inputs.len);
    try std.testing.expectEqual(@as(usize, 2 * 32), batch.targets.len);
    for (0..batch.b) |row| {
        try std.testing.expectEqual(bos, batch.inputs[row * batch.t]);
    }
    // With short convs and T=32, rows are padded ⇒ every row has ≥1 ignore (−1).
    for (0..batch.b) |row| {
        var has_ignore = false;
        for (0..batch.t) |c| {
            if (batch.targets[row * batch.t + c] == -1) has_ignore = true;
        }
        try std.testing.expect(has_ignore);
    }
}

// ===========================================================================
// Tests — parity gates (need refs/nanochat-goldens/, env-gated on
// NANOCHAT_PARITY; skip cleanly when unset OR the goldens are absent)
// ===========================================================================

const goldens_dir = "refs/nanochat-goldens";

fn skipUnlessParity() !void {
    if (std.testing.environ.getPosix("NANOCHAT_PARITY") == null) return error.SkipZigTest;
}

fn loadTokenizerOrSkip(allocator: Allocator, io: std.Io) !Tokenizer {
    return Tokenizer.loadBin(allocator, io, goldens_dir ++ "/tokenizer.bin") catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
}

test "NANOCHAT_PARITY: renderConversation matches dump-render fixture" {
    try skipUnlessParity();
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var tok = try loadTokenizerOrSkip(allocator, io);
    defer tok.deinit();

    var jc = readJsonlConvs(allocator, io, goldens_dir ++ "/sft_mixture_val.jsonl") catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer jc.deinit();

    const fixture = readFileBytes(allocator, io, goldens_dir ++ "/render_val.bin") catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer allocator.free(fixture);

    var r = BinReader{ .bytes = fixture };
    const n_convs = try r.u32le();
    try std.testing.expect(n_convs <= jc.convs.len);

    for (0..n_convs) |ci| {
        const len = try r.u32le();
        const want_ids = try allocator.alloc(u32, len);
        defer allocator.free(want_ids);
        for (want_ids) |*id| id.* = try r.u32le();
        const want_mask = try r.take(len);

        var got = try renderConversation(allocator, &tok, jc.convs[ci], 2048);
        defer got.deinit(allocator);

        try std.testing.expectEqualSlices(u32, want_ids, got.ids);
        try std.testing.expectEqualSlices(u8, want_mask, got.mask);
    }
    try std.testing.expectEqual(fixture.len, r.off);
}

test "NANOCHAT_PARITY: base BOS-bestfit loader matches dump-base-batches" {
    try skipUnlessParity();
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var tok = try loadTokenizerOrSkip(allocator, io);
    defer tok.deinit();

    var ncdoc = readNcDoc(allocator, io, goldens_dir ++ "/base_val.bin") catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer ncdoc.deinit();

    const golden = readFileBytes(allocator, io, goldens_dir ++ "/base_batches.bin") catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer allocator.free(golden);

    var r = BinReader{ .bytes = golden };
    const k = try r.u32le();
    const b = try r.u32le();
    const t = try r.u32le();

    var loader = try BaseLoader.init(allocator, &tok, ncdoc.docs, ncdoc.docs_per_rowgroup, b, t, 1000);
    defer loader.deinit();

    const nbt = @as(usize, b) * @as(usize, t);
    for (0..k) |bi| {
        var batch = try loader.nextBatch();
        defer batch.deinit();
        for (0..nbt) |i| {
            const want = try r.i32le();
            if (want != batch.inputs[i]) {
                std.debug.print("base batch {d} input[{d}]: want {d} got {d}\n", .{ bi, i, want, batch.inputs[i] });
                return error.TestUnexpectedResult;
            }
        }
        for (0..nbt) |i| {
            const want = try r.i32le();
            if (want != batch.targets[i]) {
                std.debug.print("base batch {d} target[{d}]: want {d} got {d}\n", .{ bi, i, want, batch.targets[i] });
                return error.TestUnexpectedResult;
            }
        }
        try std.testing.expectEqual(try r.i64le(), batch.state.pq_idx);
        try std.testing.expectEqual(try r.i64le(), batch.state.rg_idx);
        try std.testing.expectEqual(try r.i64le(), batch.state.epoch);
    }
    try std.testing.expectEqual(golden.len, r.off);
}

test "NANOCHAT_PARITY: SFT bestfit-pad loader matches dump-sft-batches" {
    try skipUnlessParity();
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var tok = try loadTokenizerOrSkip(allocator, io);
    defer tok.deinit();

    var jc = readJsonlConvs(allocator, io, goldens_dir ++ "/sft_mixture_val.jsonl") catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer jc.deinit();

    const golden = readFileBytes(allocator, io, goldens_dir ++ "/sft_batches.bin") catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer allocator.free(golden);

    var r = BinReader{ .bytes = golden };
    const k = try r.u32le();
    const b = try r.u32le();
    const t = try r.u32le();

    var loader = try SftLoader.init(allocator, &tok, jc.convs, b, t, 100);
    defer loader.deinit();

    const nbt = @as(usize, b) * @as(usize, t);
    for (0..k) |bi| {
        var batch = try loader.nextBatch();
        defer batch.deinit();
        for (0..nbt) |i| {
            const want = try r.i32le();
            if (want != batch.inputs[i]) {
                std.debug.print("sft batch {d} input[{d}]: want {d} got {d}\n", .{ bi, i, want, batch.inputs[i] });
                return error.TestUnexpectedResult;
            }
        }
        for (0..nbt) |i| {
            const want = try r.i64le();
            if (want != batch.targets[i]) {
                std.debug.print("sft batch {d} target[{d}]: want {d} got {d}\n", .{ bi, i, want, batch.targets[i] });
                return error.TestUnexpectedResult;
            }
        }
    }
    try std.testing.expectEqual(golden.len, r.off);
}

test {
    std.testing.refAllDecls(@This());
}
