//! Backend adapters: the generic GGUF chat backend (qwen3, gemma4 — any
//! family the shared `llm.chat.Conversation` hosts) and the grammar
//! constraint cache. Family-specific adapters that cannot ride
//! `Conversation` live in their own files (`backend_nanochat.zig`,
//! `backend_diffusion.zig`).

const std = @import("std");
const fucina = @import("fucina");
const llm = @import("fucina_llm");
const types = @import("types.zig");

const Allocator = std.mem.Allocator;

/// LRU cache of base llguidance constraints keyed by grammar kind + source.
/// `Constraint.init` walks the full model vocab to build the token trie —
/// too expensive per request — while `clone()` shares the trie and is cheap,
/// so the server inits once per distinct grammar and clones per request.
///
/// WORKER THREAD ONLY, and eviction relies on it: a clone borrows its base's
/// bridge, so a base may only be destroyed when no clone is alive. The
/// single-threaded worker guarantees that — the only live clone belongs to
/// the CURRENT request, whose base entry was just touched and is never the
/// eviction victim.
pub const ConstraintCache = struct {
    const Entry = struct {
        key: []u8,
        constraint: llm.llguidance.Constraint,
    };

    allocator: Allocator,
    /// MRU ordering: entries[len-1] is the most recently used.
    entries: std.ArrayList(Entry) = .empty,
    capacity: usize,

    pub fn init(allocator: Allocator, capacity: usize) ConstraintCache {
        return .{ .allocator = allocator, .capacity = @max(capacity, 1) };
    }

    pub fn deinit(self: *ConstraintCache) void {
        for (self.entries.items) |*e| {
            e.constraint.deinit();
            self.allocator.free(e.key);
        }
        self.entries.deinit(self.allocator);
    }

    /// The base constraint for `spec`, compiling and caching it on first
    /// use. The returned pointer is valid until the next `acquire` — clone
    /// it before any other cache call.
    pub fn acquire(
        self: *ConstraintCache,
        tokenizer: anytype,
        spec: types.ConstraintSpec,
        options: llm.llguidance.Options,
    ) !*llm.llguidance.Constraint {
        const a = self.allocator;
        for (self.entries.items, 0..) |e, i| {
            if (e.key.len == spec.source().len + 1 and
                e.key[0] == spec.kindByte() and
                std.mem.eql(u8, e.key[1..], spec.source()))
            {
                // Move to MRU position.
                const hit = self.entries.orderedRemove(i);
                self.entries.appendAssumeCapacity(hit);
                return &self.entries.items[self.entries.items.len - 1].constraint;
            }
        }

        const grammar: llm.llguidance.Grammar = switch (spec) {
            .json_schema => |s| .{ .json_schema = s },
            .regex => |s| .{ .regex = s },
            .lark => |s| .{ .lark = s },
        };
        var constraint = try llm.llguidance.Constraint.init(a, tokenizer, grammar, options);
        errdefer constraint.deinit();

        const key = try a.alloc(u8, spec.source().len + 1);
        errdefer a.free(key);
        key[0] = spec.kindByte();
        @memcpy(key[1..], spec.source());

        if (self.entries.items.len >= self.capacity) {
            var evicted = self.entries.orderedRemove(0);
            evicted.constraint.deinit();
            a.free(evicted.key);
        }
        try self.entries.ensureUnusedCapacity(a, 1);
        self.entries.appendAssumeCapacity(.{ .key = key, .constraint = constraint });
        return &self.entries.items[self.entries.items.len - 1].constraint;
    }
};

/// The evict-to-disk tier for cross-request KV reuse (llama.cpp's
/// host-memory prompt cache, on disk): slot states about to be destroyed by
/// an unrelated request are saved as `llm.kv_persist` sidecars and restored
/// when a later request's prefix matches them better than any resident slot.
pub const KvDiskOptions = struct {
    io: std.Io,
    /// Directory for the sidecar files (created at startup). Borrowed.
    dir: []const u8,
    /// Bound on live sidecar files; beyond it the least-recently-used entry
    /// is overwritten.
    max_files: usize = 8,
};

pub const GgufChatOptions = struct {
    model_id: []const u8,
    /// KV capacity per request (prompt + reply must fit).
    context_len: usize = 4096,
    /// Turn-end ids beyond the template stop marker (gemma4: GGUF eos +
    /// stray SPM <eos>). Borrowed.
    extra_stop_ids: []const u32 = &.{},
    /// The reply's reasoning-block delimiters, when the family has a
    /// text-delimited reasoning channel the server can toggle (qwen3).
    think_markers: ?types.ThinkMarkers = null,
    supports_think: bool = false,
    default_sampling: llm.sampler.Config = .{},
    constraint_cache_len: usize = 8,
    /// Resident cross-request KV reuse slots. Each is a FULL `context_len`
    /// KV cache — budget accordingly (a 28-layer/8-kv-head/128-dim f16
    /// cache is ~112 KiB per position). 1 keeps the server memory-neutral
    /// with pre-reuse behavior; more slots stop interleaved conversations
    /// from evicting each other.
    kv_slots: usize = 1,
    /// Evict-to-disk tier (null = off).
    kv_disk: ?KvDiskOptions = null,
    /// Trained KV-prefix "prior knowledge" (docs/CARTRIDGES.md): every
    /// conversation's cache is preloaded with these rows before any
    /// prefill, and the reuse reconcile operates past them. Borrowed —
    /// must outlive the backend. Composes with `kv_disk`: sidecars record
    /// the prefix shape and rows (FUXKV002), so restores are
    /// self-describing even across a cartridge swap.
    cartridge: ?*const llm.cartridge.Cartridge = null,
    /// Cartridge FLEET serving (Cartridges at Scale, docs/CARTRIDGES.md):
    /// per request, the last user message embeds through `embedFn`, the
    /// cosine index picks documents, and the selected cartridges COMPOSE as
    /// the conversation's prefix. Mutually exclusive with `cartridge` and
    /// with `kv_disk` (sidecars do not record selections, so a restore
    /// could resurrect rows behind the wrong prefix) — the caller enforces
    /// both. Slot reuse stays on, keyed by selection: only a slot whose
    /// cartridge selection matches the request's is adoptable.
    fleet: ?FleetOptions = null,
};

pub const FleetOptions = struct {
    io: std.Io,
    /// Fleet directory and its parsed manifest/index (borrowed — must
    /// outlive the backend).
    dir: []const u8,
    manifest: *const llm.cartridge_fleet.Manifest,
    index: *const llm.cartridge_fleet.EmbedIndex,
    /// Query embedder (the family trainer behind a type-erased pointer;
    /// MUST implement the `cartridge_fleet.embed_suffix` contract the index
    /// was built with). Worker thread only, like generation itself.
    embed_ctx: *anyopaque,
    embedFn: *const fn (ctx: *anyopaque, text: []const u8, out: []f32) anyerror!void,
    /// Selection sizes (documents composed per request; chunk scan width).
    rag_docs: usize = 2,
    rag_chunks: usize = 8,
    /// Adaptive re-selection for CONTINUING conversations (default off =
    /// fully sticky): every follow-up re-embeds the contextual query (all
    /// user messages) and the conversation SWITCHES knowledge base only on
    /// decisive evidence — a document outside its current selection must
    /// beat every current document's best chunk by `switch_margin`.
    /// Absolute score floors do not work on this substrate (measured: a
    /// phatic "Thanks, that makes sense." scores HIGHER against an
    /// unrelated doc than a genuine topical pivot does); the relative
    /// margin over a context-anchored query absorbs both phatic turns and
    /// runner-up cosine flaps. A switch rebuilds the prefix and re-prefills
    /// the history (cached_tokens = 0 for that turn).
    adaptive: bool = false,
    switch_margin: f32 = 0.05,
    /// Loaded-cartridge LRU capacity (prefix rows are COPIED into each
    /// conversation's cache, so this only bounds re-parse/mmap cost;
    /// effective capacity is max(cache_len, rag_docs)).
    cache_len: usize = 4,
};

/// Longest common prefix between a slot's token shadow and a request's ids
/// — the sole matching primitive of the reuse tiers (token-level LCP, the
/// llama.cpp `cache_prompt` rule).
fn commonPrefix(tokens: []const usize, ids: []const u32) usize {
    var n: usize = 0;
    const cap = @min(tokens.len, ids.len);
    while (n < cap and tokens[n] == ids[n]) : (n += 1) {}
    return n;
}

/// The slot-similarity gate (llama.cpp `--slot-prompt-similarity`, default
/// 0.1): adopting a cache pays only when the common prefix covers a
/// meaningful share of the NEW prompt — otherwise a long-lived cache would
/// be destroyed to save a handful of tokens.
fn similarEnough(lcp: usize, ids_len: usize) bool {
    return lcp * 10 > ids_len;
}

/// Adaptive fleet serving's switch rule: leave a conversation's knowledge
/// base only on DECISIVE evidence — the best document OUTSIDE the current
/// selection must beat every current document's best chunk by `margin`
/// under the context-anchored query. Relative, never absolute: measured on
/// a live fleet, a phatic "Thanks, that makes sense." scores HIGHER in raw
/// cosine against an unrelated document than a genuine topical pivot does,
/// so score floors misfire in both directions; the margin over a query
/// anchored by the whole user side absorbs phatic turns and runner-up
/// cosine flaps alike.
fn shouldSwitchSelection(cur_best: f32, outside_best: ?f32, margin: f32) bool {
    return (outside_best orelse return false) >= cur_best + margin;
}

/// The `Backend` adapter for any model family served through
/// `llm.chat.Conversation` — one comptime instantiation per (model,
/// tokenizer-module) pair, ~all behavior shared. The API stays stateless
/// (every request carries its full history), but the KV cache is not: a
/// pool of resident slots (`kv_slots`) keeps previous requests' caches +
/// token shadows, and each request adopts the slot sharing the longest
/// common token prefix with its own render — llama.cpp's `cache_prompt` +
/// slot selection. Follow-up turns of a chat prefill only the last reply +
/// new message instead of the whole history; a non-matching request costs
/// one full prefill, exactly as before. With `kv_disk` set, slot states
/// about to be destroyed by an unrelated request spill to `llm.kv_persist`
/// sidecars and are restored when a later request matches them best.
pub fn GgufChatBackend(comptime ModelT: type, comptime TokMod: type) type {
    return struct {
        const Self = @This();
        const Conversation = llm.chat.Conversation(ModelT, TokMod);
        const KvCache = llm.kv_cache.KvCache;

        /// One resident reuse slot: a KV cache plus the token shadow
        /// describing exactly the positions it holds (WORKER THREAD ONLY,
        /// like `constraints`). In fleet mode `selection` records the doc
        /// ids whose composed cartridges sit ahead of the shadow — a slot
        /// is only adoptable by a request with the SAME selection (empty
        /// for non-fleet backends, where every slot shares the one
        /// configured prefix) — and `opener` records the conversation's
        /// FIRST user message, the sticky-adoption identity (token-LCP
        /// alone is too weak: the constant template preamble of two
        /// UNRELATED short prompts passes the similarity gate, which must
        /// never move a conversation onto another's knowledge base).
        const Slot = struct {
            cache: KvCache,
            tokens: std.ArrayList(usize),
            selection: []usize,
            opener: []u8,
            prefix_rows: usize,
            last_used: u64,
        };

        /// One loaded fleet cartridge (heap-pinned: LRU reordering must not
        /// move the Cartridge — conversations borrow it only within one
        /// request, but pointer stability keeps the compose loop simple).
        const CartEntry = struct {
            doc: usize,
            cart: *llm.cartridge.Cartridge,
        };

        /// One disk-tier sidecar: its path, an in-memory copy of its token
        /// history (for LCP scoring without touching the file), and its
        /// recency for the bounded-file LRU.
        const DiskEntry = struct {
            path: []u8,
            tokens: []usize,
            last_used: u64,
        };

        allocator: Allocator,
        ctx: *fucina.ExecContext,
        model: *const ModelT,
        tokenizer: *const TokMod.Tokenizer,
        template: llm.chat.Template,
        opts: GgufChatOptions,
        constraints: ConstraintCache,
        slots: std.ArrayList(Slot) = .empty,
        disk: std.ArrayList(DiskEntry) = .empty,
        /// Fleet-mode loaded-cartridge LRU (MRU last; worker thread only).
        carts: std.ArrayList(CartEntry) = .empty,
        /// Monotonic request counter: the recency stamp for slot and
        /// disk-entry LRU.
        clock: u64 = 0,
        /// Fresh-name counter for sidecar files (paths are never recycled
        /// across registry removals, so a live entry's file cannot be
        /// clobbered by a name collision).
        disk_seq: u64 = 0,

        pub fn init(
            allocator: Allocator,
            ctx: *fucina.ExecContext,
            model: *const ModelT,
            tokenizer: *const TokMod.Tokenizer,
            template: llm.chat.Template,
            opts: GgufChatOptions,
        ) Self {
            return .{
                .allocator = allocator,
                .ctx = ctx,
                .model = model,
                .tokenizer = tokenizer,
                .template = template,
                .opts = opts,
                .constraints = ConstraintCache.init(allocator, opts.constraint_cache_len),
            };
        }

        pub fn deinit(self: *Self) void {
            const a = self.allocator;
            self.constraints.deinit();
            for (self.slots.items) |*slot| {
                slot.cache.deinit();
                slot.tokens.deinit(a);
                a.free(slot.selection);
                a.free(slot.opener);
            }
            self.slots.deinit(a);
            for (self.disk.items) |*e| {
                a.free(e.path);
                a.free(e.tokens);
            }
            self.disk.deinit(a);
            for (self.carts.items) |*e| {
                e.cart.deinit();
                a.destroy(e.cart);
            }
            self.carts.deinit(a);
        }

        /// The pool size floor: a zero config still reuses one slot.
        fn kvSlots(self: *const Self) usize {
            return @max(self.opts.kv_slots, 1);
        }

        /// The parsed cartridge for fleet doc `doc`, mmap-loading it into
        /// the MRU-ordered LRU on a miss. The returned pointer is
        /// heap-stable; entries live until evicted (rows are COPIED into
        /// conversation caches, so eviction never invalidates a served
        /// prefix). Worker thread only.
        fn fleetCartridge(self: *Self, doc: usize) !*const llm.cartridge.Cartridge {
            const fl = self.opts.fleet.?;
            const a = self.allocator;
            for (self.carts.items, 0..) |e, i| {
                if (e.doc == doc) {
                    const hit = self.carts.orderedRemove(i);
                    self.carts.appendAssumeCapacity(hit);
                    return hit.cart;
                }
            }
            const state = &fl.manifest.docs.items[doc];
            const path = try std.fs.path.join(a, &.{ fl.dir, state.cart_file });
            defer a.free(path);
            var mapped = try llm.cartridge_fleet.mmapFile(fl.io, path);
            defer mapped.deinit();
            const cart = try a.create(llm.cartridge.Cartridge);
            errdefer a.destroy(cart);
            cart.* = try llm.cartridge.Cartridge.initFromStateDict(self.ctx, a, mapped.bytes);
            errdefer cart.deinit();
            // Effective capacity never below the selection width, so the
            // acquire-then-compose loop of writeSelectionPrefix cannot
            // evict a cartridge it is about to serve.
            const cap = @max(fl.cache_len, fl.rag_docs);
            if (self.carts.items.len >= cap) {
                var evicted = self.carts.orderedRemove(0);
                evicted.cart.deinit();
                a.destroy(evicted.cart);
            }
            try self.carts.ensureUnusedCapacity(a, 1);
            self.carts.appendAssumeCapacity(.{ .doc = doc, .cart = cart });
            return cart;
        }

        const Adopted = struct {
            convo: Conversation,
            /// The slot's selection, transferred to the caller (owned).
            selection: []usize,
        };

        /// Fleet stickiness: find the slot whose conversation this request
        /// CONTINUES — identity is the first user message (`opener`), not
        /// token-LCP (the constant template preamble of two unrelated short
        /// prompts passes the LCP similarity gate, and adopting across
        /// conversations would move a request onto the wrong knowledge
        /// base). Among several same-opener slots the best token-LCP wins.
        /// Null when no slot matches — the caller then runs retrieval.
        fn findStickySlot(self: *Self, ids: []const u32, opener: []const u8) ?usize {
            if (opener.len == 0) return null;
            var best_i: ?usize = null;
            var best_lcp: usize = 0;
            for (self.slots.items, 0..) |*slot, i| {
                if (!std.mem.eql(u8, slot.opener, opener)) continue;
                const lcp = commonPrefix(slot.tokens.items, ids);
                if (best_i == null or lcp > best_lcp) {
                    best_i = i;
                    best_lcp = lcp;
                }
            }
            return best_i;
        }

        /// Adopt slot `i` as this request's warm conversation. A
        /// continuation keeps the selection its conversation started with
        /// (per-turn re-retrieval measurably flaps the runner-up document
        /// and forfeits all KV reuse); `--rag-adaptive` layers the decisive
        /// switch rule on top of this in `vtGenerate`.
        fn adoptSlotAt(self: *Self, i: usize, convo_opts: llm.chat.Options) !Adopted {
            var slot = self.slots.swapRemove(i);
            defer slot.tokens.deinit(self.allocator);
            self.allocator.free(slot.opener);
            errdefer self.allocator.free(slot.selection);
            const convo = try Conversation.initWarm(self.ctx, self.model, self.tokenizer, self.template, convo_opts, .{
                .cache = slot.cache,
                .tokens = slot.tokens.items,
                .prefix_rows = slot.prefix_rows,
            });
            return .{ .convo = convo, .selection = slot.selection };
        }

        /// Drop slot `i` entirely (an adaptive SWITCH supersedes the
        /// conversation's old state — its cache holds rows behind a prefix
        /// the conversation is leaving).
        fn destroySlot(self: *Self, i: usize) void {
            var slot = self.slots.swapRemove(i);
            slot.cache.deinit();
            slot.tokens.deinit(self.allocator);
            self.allocator.free(slot.selection);
            self.allocator.free(slot.opener);
        }

        /// Embed the contextual retrieval query — every user message,
        /// concatenated (the full render as fallback) — into `vec`.
        fn embedFleetQuery(self: *Self, fl: FleetOptions, req: *const types.GenerateRequest, rendered: []const u8, vec: []f32) !void {
            const a = self.allocator;
            var buf: std.ArrayList(u8) = .empty;
            defer buf.deinit(a);
            for (req.messages) |msg| {
                if (msg.role != .user) continue;
                if (buf.items.len > 0) try buf.append(a, '\n');
                try buf.appendSlice(a, msg.content);
            }
            const query: []const u8 = if (buf.items.len > 0) buf.items else rendered;
            try fl.embedFn(fl.embed_ctx, query, vec);
        }

        /// Compose the selection's cartridges, in order, into an EMPTY
        /// cache (`writeComposedToCache` semantics via the LRU).
        fn writeSelectionPrefix(self: *Self, selection: []const usize, cache: *KvCache) !void {
            // Make everything resident first: with the capacity floor above,
            // the compose loop's acquires are then all hits (no eviction
            // between appends).
            for (selection) |doc| _ = try self.fleetCartridge(doc);
            for (selection) |doc| {
                const cart = try self.fleetCartridge(doc);
                try cart.appendToCache(self.ctx, cache);
            }
        }

        /// Pick this request's warm state and build the Conversation on it:
        /// the resident slot with the best token-LCP when it passes the
        /// similarity gate; a disk-tier restore when a sidecar strictly
        /// beats every resident slot; otherwise the LRU slot as a
        /// reconcile-overwrite host when the pool is full (spilling its
        /// state to the disk tier first when worth keeping), or a cold
        /// start while the pool still has room. `selection`/`selection_p`
        /// describe the request's composed fleet prefix (empty outside
        /// fleet mode): only same-selection slots are adoptable, and a
        /// foreign-prefix host is reset and rebuilt behind this request's
        /// cartridges.
        fn acquireConversation(self: *Self, ids: []const u32, convo_opts: llm.chat.Options, selection: []const usize, selection_p: usize) !Conversation {
            self.clock += 1;

            var best_i: ?usize = null;
            var best_lcp: usize = 0;
            var lru_i: ?usize = null;
            for (self.slots.items, 0..) |*slot, i| {
                if (std.mem.eql(usize, slot.selection, selection)) {
                    const lcp = commonPrefix(slot.tokens.items, ids);
                    if (best_i == null or lcp > best_lcp) {
                        best_i = i;
                        best_lcp = lcp;
                    }
                }
                if (lru_i == null or slot.last_used < self.slots.items[lru_i.?].last_used) lru_i = i;
            }

            // The disk tier competes only when it strictly beats the pool
            // (never armed in fleet mode — sidecars carry no selection).
            var disk_i: ?usize = null;
            if (self.opts.kv_disk != null) {
                var disk_lcp: usize = 0;
                for (self.disk.items, 0..) |*e, i| {
                    const lcp = commonPrefix(e.tokens, ids);
                    if (lcp > disk_lcp) {
                        disk_lcp = lcp;
                        disk_i = i;
                    }
                }
                if (disk_lcp <= best_lcp or !similarEnough(disk_lcp, ids.len)) disk_i = null;
            }

            if (disk_i == null and best_i != null and similarEnough(best_lcp, ids.len)) {
                var slot = self.slots.swapRemove(best_i.?);
                defer slot.tokens.deinit(self.allocator);
                self.allocator.free(slot.selection);
                self.allocator.free(slot.opener);
                return Conversation.initWarm(self.ctx, self.model, self.tokenizer, self.template, convo_opts, .{
                    .cache = slot.cache,
                    .tokens = slot.tokens.items,
                    .prefix_rows = slot.prefix_rows,
                });
            }

            // Host cache for a restore or a reconcile-overwrite: the LRU
            // slot when the pool is full, a fresh cold cache otherwise.
            var host: ?Slot = null;
            if (self.slots.items.len >= self.kvSlots()) {
                var victim = self.slots.swapRemove(lru_i.?);
                self.maybeSaveToDisk(&victim, ids);
                host = victim;
            }

            if (disk_i) |di| {
                const kd = self.opts.kv_disk.?;
                var cache: KvCache = undefined;
                if (host) |*h| {
                    h.tokens.deinit(self.allocator);
                    self.allocator.free(h.selection);
                    self.allocator.free(h.opener);
                    cache = h.cache;
                    // kv_persist.load requires an empty cache.
                    cache.truncate(0);
                } else {
                    cache = try self.model.initKvCache(self.ctx, self.opts.context_len);
                }
                self.disk.items[di].last_used = self.clock;
                const path = self.disk.items[di].path;
                const loaded = llm.kv_persist.load(kd.io, self.allocator, path, &cache) catch null;
                if (loaded) |resumed| {
                    defer self.allocator.free(resumed.tokens);
                    // The sidecar carries its own prefix rows: the restored
                    // conversation keeps the exact prefix it was born with.
                    return Conversation.initWarm(self.ctx, self.model, self.tokenizer, self.template, convo_opts, .{
                        .cache = cache,
                        .tokens = resumed.tokens,
                        .prefix_rows = resumed.prefix_rows,
                    });
                }
                // Unreadable or foreign sidecar (deleted or clobbered
                // externally): drop the entry; the request starts cold on
                // this cache (behind the current cartridge when one is
                // configured).
                self.removeDiskEntry(di);
                var convo = try Conversation.initWarm(self.ctx, self.model, self.tokenizer, self.template, convo_opts, .{
                    .cache = cache,
                    .tokens = &.{},
                });
                if (self.opts.cartridge) |cart| {
                    errdefer convo.deinit();
                    try cart.writeToCache(self.ctx, &convo.cache);
                    try convo.notePrefixRows(cart.p);
                }
                return convo;
            }

            if (host) |*h| {
                const a = self.allocator;
                if (std.mem.eql(usize, h.selection, selection)) {
                    // Same prefix: reconcile-overwrite on the victim's rows.
                    defer h.tokens.deinit(a);
                    a.free(h.selection);
                    a.free(h.opener);
                    return Conversation.initWarm(self.ctx, self.model, self.tokenizer, self.template, convo_opts, .{
                        .cache = h.cache,
                        .tokens = h.tokens.items,
                        .prefix_rows = h.prefix_rows,
                    });
                }
                // Foreign prefix (fleet mode): nothing behind it is
                // reusable — reset the cache and rebuild this request's
                // composed prefix.
                h.tokens.deinit(a);
                a.free(h.selection);
                a.free(h.opener);
                var cache = h.cache;
                errdefer cache.deinit();
                cache.truncate(0);
                try self.writeSelectionPrefix(selection, &cache);
                return Conversation.initWarm(self.ctx, self.model, self.tokenizer, self.template, convo_opts, .{
                    .cache = cache,
                    .tokens = &.{},
                    .prefix_rows = selection_p,
                });
            }

            var convo = try Conversation.init(self.ctx, self.model, self.tokenizer, self.template, convo_opts);
            if (self.opts.cartridge) |cart| {
                // Cold start in cartridge mode: preload the trained prefix
                // before the first prefill — the served layout the rows
                // were trained at (real tokens at positions p..).
                errdefer convo.deinit();
                try cart.writeToCache(self.ctx, &convo.cache);
                try convo.notePrefixRows(cart.p);
            } else if (selection.len > 0) {
                // Cold start in fleet mode: the selected cartridges compose
                // ahead of the first prefill.
                errdefer convo.deinit();
                try self.writeSelectionPrefix(selection, &convo.cache);
                try convo.notePrefixRows(selection_p);
            }
            return convo;
        }

        /// The generation epilogue (success and error paths alike): move
        /// the conversation's cache and its committed-token shadow back
        /// into the pool. History can sit one un-forwarded token past the
        /// cache after an aborted turn, so the shadow is trimmed to
        /// `cache.len` — a slot always describes exactly the positions its
        /// cache holds. `acquireConversation` removed at most one slot, so
        /// the append keeps the pool within `kv_slots`.
        fn reclaimSlot(self: *Self, convo: *Conversation, selection: []const usize, opener: []const u8) void {
            const a = self.allocator;
            var tokens: std.ArrayList(usize) = .empty;
            var sel: []usize = &.{};
            var op: []u8 = &.{};
            const ok = blk: {
                // The shadow describes only token-backed rows: cache rows
                // [0, kv_prefix_rows) are the preloaded prefix (shared
                // cartridge, or this request's fleet selection).
                tokens.appendSlice(a, convo.history.items[0 .. convo.cache.len - convo.kv_prefix_rows]) catch break :blk false;
                sel = a.dupe(usize, selection) catch break :blk false;
                op = a.dupe(u8, opener) catch break :blk false;
                self.slots.ensureUnusedCapacity(a, 1) catch break :blk false;
                break :blk true;
            };
            if (!ok) {
                // Without the shadow the cache can never match: drop both
                // and let a later request start cold.
                tokens.deinit(a);
                a.free(sel);
                a.free(op);
                var cache = convo.takeCache();
                cache.deinit();
                return;
            }
            self.slots.appendAssumeCapacity(.{
                .cache = convo.takeCache(),
                .tokens = tokens,
                .selection = sel,
                .opener = op,
                .prefix_rows = convo.kv_prefix_rows,
                .last_used = self.clock,
            });
        }

        /// Save-on-evict (llama.cpp's prompt-cache trigger): spill the
        /// victim's state to the disk tier when the incoming request would
        /// keep less than half of it AND no stored entry already contains
        /// it. An entry the victim extends is overwritten in place
        /// (supersede); otherwise a fresh file is created up to
        /// `max_files`, then the LRU entry's file is reused. Save failures
        /// only cost the spill — the eviction proceeds regardless.
        fn maybeSaveToDisk(self: *Self, victim: *const Slot, ids: []const u32) void {
            const kd = self.opts.kv_disk orelse return;
            const a = self.allocator;
            const vtokens = victim.tokens.items;
            if (vtokens.len == 0) return;
            if (commonPrefix(vtokens, ids) * 2 >= vtokens.len) return;
            var target: ?usize = null;
            for (self.disk.items, 0..) |*e, i| {
                if (e.tokens.len >= vtokens.len) {
                    // Containment: a stored entry already covers this state.
                    if (std.mem.eql(usize, e.tokens[0..vtokens.len], vtokens)) return;
                } else if (target == null and std.mem.eql(usize, e.tokens, vtokens[0..e.tokens.len])) {
                    // Supersede: the victim extends this entry.
                    target = i;
                }
            }
            if (target == null and self.disk.items.len >= kd.max_files) {
                for (self.disk.items, 0..) |*e, i| {
                    if (target == null or e.last_used < self.disk.items[target.?].last_used) target = i;
                }
            }

            const snapshot = a.dupe(usize, vtokens) catch return;
            if (target == null) {
                // Fresh entry: name, registry room, then the file.
                const path = std.fmt.allocPrint(a, "{s}/kv-slot-{d}.fuxkv", .{ kd.dir, self.disk_seq }) catch {
                    a.free(snapshot);
                    return;
                };
                self.disk.ensureUnusedCapacity(a, 1) catch {
                    a.free(snapshot);
                    a.free(path);
                    return;
                };
                if (!self.writeSidecar(kd.io, a, path, &victim.cache, vtokens)) {
                    a.free(snapshot);
                    a.free(path);
                    return;
                }
                self.disk_seq += 1;
                self.disk.appendAssumeCapacity(.{ .path = path, .tokens = snapshot, .last_used = self.clock });
                return;
            }
            const entry = &self.disk.items[target.?];
            if (!self.writeSidecar(kd.io, a, entry.path, &victim.cache, vtokens)) {
                // The old file content is gone (reset header): the entry no
                // longer describes anything restorable.
                a.free(snapshot);
                self.removeDiskEntry(target.?);
                return;
            }
            a.free(entry.tokens);
            entry.tokens = snapshot;
            entry.last_used = self.clock;
        }

        /// Reset + append-all: a sidecar file whose whole record range is
        /// this cache's positions. Returns false on any write error.
        fn writeSidecar(self: *const Self, io: std.Io, a: Allocator, path: []const u8, cache: *const KvCache, tokens: []const usize) bool {
            // Cartridge mode: the sidecar records the prefix shape and the
            // prefix rows themselves (FUXKV002) — a restore is
            // self-describing even across a cartridge swap.
            const prefix_rows: usize = if (self.opts.cartridge) |cart| cart.p else 0;
            llm.kv_persist.reset(io, a, path, cache, prefix_rows) catch return false;
            llm.kv_persist.appendRange(io, a, path, cache, tokens, prefix_rows) catch return false;
            return true;
        }

        /// Drop a disk-registry entry and best-effort delete its file.
        fn removeDiskEntry(self: *Self, i: usize) void {
            const kd = self.opts.kv_disk.?;
            const e = self.disk.swapRemove(i);
            std.Io.Dir.cwd().deleteFile(kd.io, e.path) catch {};
            self.allocator.free(e.path);
            self.allocator.free(e.tokens);
        }

        pub fn backend(self: *Self) types.Backend {
            return .{
                .ptr = self,
                .vtable = &.{ .validate = vtValidate, .generate = vtGenerate },
                .info = .{
                    .model_id = self.opts.model_id,
                    .context_len = self.opts.context_len,
                    .caps = .{
                        .grammar = llm.llguidance.enabled,
                        .think = self.opts.supports_think,
                    },
                    .think_markers = self.opts.think_markers,
                    .default_sampling = self.opts.default_sampling,
                },
            };
        }

        /// Render the request's message history; caller owns the buffer.
        fn render(self: *const Self, allocator: Allocator, req: *const types.GenerateRequest) ![]u8 {
            var buf: std.ArrayList(u8) = .empty;
            errdefer buf.deinit(allocator);
            try self.template.renderMessages(allocator, &buf, req.messages, !req.think);
            return buf.toOwnedSlice(allocator);
        }

        fn vtValidate(ptr: *anyopaque, req: *const types.GenerateRequest) anyerror!void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            const a = self.allocator;
            const rendered = try self.render(a, req);
            defer a.free(rendered);
            // Pure read-only tokenizer use: safe off the worker thread.
            const ids = try self.tokenizer.encodeRaw(a, rendered);
            defer a.free(ids);
            if (ids.len >= self.opts.context_len) return error.PromptTooLong;
        }

        fn vtGenerate(ptr: *anyopaque, req: *const types.GenerateRequest, sink: *std.Io.Writer) anyerror!types.GenerateResult {
            const self: *Self = @ptrCast(@alignCast(ptr));
            const a = self.allocator;

            const rendered = try self.render(a, req);
            defer a.free(rendered);

            // Grammar: clone the cached base per request (the base must
            // outlive the clone; the cache guarantees it — see above). The
            // mask forces the turn-end marker when the grammar completes, so
            // normal stop handling ends the reply.
            var clone: ?llm.llguidance.Constraint = null;
            defer if (clone) |*c| c.deinit();
            var processor: ?llm.sampler.LogitProcessor = null;
            if (req.constraint) |spec| {
                const turn_stop: ?u32 = self.tokenizer.tokenId(self.template.stopMarker()) orelse self.tokenizer.eosId();
                const base = try self.constraints.acquire(self.tokenizer, spec, .{
                    .eos_token = turn_stop,
                    .extra_eos = self.opts.extra_stop_ids,
                    .n_vocab = self.model.config.vocab_size,
                });
                clone = try base.clone();
                processor = clone.?.processor();
            }

            const convo_opts: llm.chat.Options = .{
                .capacity = self.opts.context_len,
                .max_response_tokens = req.max_tokens,
                .think_off = !req.think,
                .sampler = req.sampling,
                .extra_stop_ids = self.opts.extra_stop_ids,
                .stop_sequences = req.stop,
                .logit_processor = processor,
            };
            // One tokenization serves both slot selection and the send.
            const ids = try self.tokenizer.encodeRaw(a, rendered);
            defer a.free(ids);

            // Fleet mode: sticky adoption first (a warm conversation keeps
            // the cartridges it started with; --rag-adaptive layers the
            // decisive switch rule on top), then — for conversations no
            // slot remembers, or that decisively changed topic — cosine
            // selection over the request's USER messages, concatenated
            // (the whole user side, not just the last message: "and how
            // are they packed?" carries no topic alone; the full render is
            // the fallback when no user turn exists).
            var selection: []usize = &.{};
            defer a.free(selection);
            var maybe_convo: ?Conversation = null;
            // The conversation's identity for sticky adoption: its FIRST
            // user message (empty outside fleet mode — non-fleet slots
            // match by token-LCP alone, as before).
            var opener: []const u8 = &.{};
            if (self.opts.fleet != null) {
                for (req.messages) |msg| {
                    if (msg.role == .user) {
                        opener = msg.content;
                        break;
                    }
                }
            }
            if (self.opts.fleet) |fl| {
                self.clock += 1;
                var sticky_i = self.findStickySlot(ids, opener);

                if (sticky_i != null and fl.adaptive) {
                    // Adaptive: re-embed the contextual query and leave the
                    // conversation's knowledge base only on decisive
                    // evidence (see shouldSwitchSelection).
                    const vec = try a.alloc(f32, fl.index.dim);
                    defer a.free(vec);
                    try self.embedFleetQuery(fl, req, rendered, vec);
                    const hits = try fl.index.topDocs(a, vec, fl.rag_chunks, fl.rag_docs);
                    defer a.free(hits);
                    const cur = self.slots.items[sticky_i.?].selection;
                    const cur_scores = try a.alloc(f32, cur.len);
                    defer a.free(cur_scores);
                    try fl.index.docScores(a, vec, cur, cur_scores);
                    var cur_best: f32 = -1;
                    for (cur_scores) |s| cur_best = @max(cur_best, s);
                    var outside_best: ?f32 = null;
                    for (hits) |hit| {
                        const inside = for (cur) |doc| {
                            if (doc == hit.doc) break true;
                        } else false;
                        if (!inside and (outside_best == null or hit.score > outside_best.?)) outside_best = hit.score;
                    }
                    if (shouldSwitchSelection(cur_best, outside_best, fl.switch_margin) and hits.len > 0) {
                        // Topic moved: the old slot's rows sit behind a
                        // prefix this conversation is leaving.
                        self.destroySlot(sticky_i.?);
                        sticky_i = null;
                        selection = try a.alloc(usize, hits.len);
                        for (selection, hits) |*doc, hit| doc.* = hit.doc;
                    }
                }

                if (sticky_i) |si| {
                    const adopted = try self.adoptSlotAt(si, convo_opts);
                    maybe_convo = adopted.convo;
                    selection = adopted.selection;
                } else if (selection.len == 0) {
                    // Fresh conversation: contextual retrieval.
                    const vec = try a.alloc(f32, fl.index.dim);
                    defer a.free(vec);
                    try self.embedFleetQuery(fl, req, rendered, vec);
                    const hits = try fl.index.topDocs(a, vec, fl.rag_chunks, fl.rag_docs);
                    defer a.free(hits);
                    if (hits.len == 0) return error.EmptySelection;
                    selection = try a.alloc(usize, hits.len);
                    for (selection, hits) |*doc, hit| doc.* = hit.doc;
                }
            }

            var convo = maybe_convo orelse blk: {
                var selection_p: usize = 0;
                for (selection) |doc| selection_p += (try self.fleetCartridge(doc)).p;
                break :blk try self.acquireConversation(ids, convo_opts, selection, selection_p);
            };
            defer convo.deinit();
            // LIFO: reclaim runs before the deinit above, on every path out.
            defer self.reclaimSlot(&convo, selection, opener);

            const produced = try convo.sendTokensReuse(ids, sink);
            const finish: types.FinishReason = if (produced >= req.max_tokens or convo.cache.len >= convo.cache.capacity)
                .length
            else
                .stop;
            return .{
                .prompt_tokens = convo.history.items.len - produced,
                .completion_tokens = produced,
                .cached_tokens = convo.reused_prefix,
                .finish = finish,
            };
        }
    };
}

test "adaptive fleet switch rule: decisive-margin only" {
    // No document outside the current selection: never switch.
    try std.testing.expect(!shouldSwitchSelection(0.4, null, 0.05));
    // Outside doc must beat the current best by the margin (values off the
    // exact f32 boundary: 0.40 + 0.05 rounds above 0.45).
    try std.testing.expect(!shouldSwitchSelection(0.40, 0.42, 0.05));
    try std.testing.expect(shouldSwitchSelection(0.40, 0.46, 0.05));
    try std.testing.expect(shouldSwitchSelection(0.40, 0.60, 0.05));
    // The measured runner-up flap (gap ~0.002) stays put.
    try std.testing.expect(!shouldSwitchSelection(0.2780, 0.2755, 0.05));
    // A selection whose docs all score -1 (no chunks) yields to anything
    // clearing the margin from below.
    try std.testing.expect(shouldSwitchSelection(-1, 0.1, 0.05));
}

test "kv reuse policy: commonPrefix and the similarity gate" {
    const tokens = [_]usize{ 5, 6, 7, 8, 9 };
    try std.testing.expectEqual(@as(usize, 3), commonPrefix(&tokens, &.{ 5, 6, 7, 99 }));
    try std.testing.expectEqual(@as(usize, 5), commonPrefix(&tokens, &.{ 5, 6, 7, 8, 9, 10 }));
    try std.testing.expectEqual(@as(usize, 2), commonPrefix(&tokens, &.{ 5, 6 }));
    try std.testing.expectEqual(@as(usize, 0), commonPrefix(&tokens, &.{ 9, 5 }));
    try std.testing.expectEqual(@as(usize, 0), commonPrefix(&.{}, &.{ 1, 2 }));

    // Gate: strictly more than 10% of the NEW prompt must match.
    try std.testing.expect(!similarEnough(0, 10));
    try std.testing.expect(!similarEnough(1, 10));
    try std.testing.expect(similarEnough(2, 10));
    try std.testing.expect(similarEnough(1, 9));
    try std.testing.expect(!similarEnough(10, 100));
    try std.testing.expect(similarEnough(11, 100));
}
