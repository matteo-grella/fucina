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
};

/// The `Backend` adapter for any model family served through
/// `llm.chat.Conversation` — one comptime instantiation per (model,
/// tokenizer-module) pair, ~all behavior shared. Each request runs on a
/// fresh Conversation (full-history prefill; per-request KV cache), so the
/// server stays stateless across requests.
pub fn GgufChatBackend(comptime ModelT: type, comptime TokMod: type) type {
    return struct {
        const Self = @This();
        const Conversation = llm.chat.Conversation(ModelT, TokMod);

        allocator: Allocator,
        ctx: *fucina.ExecContext,
        model: *const ModelT,
        tokenizer: *const TokMod.Tokenizer,
        template: llm.chat.Template,
        opts: GgufChatOptions,
        constraints: ConstraintCache,

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
            self.constraints.deinit();
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

            var convo = try Conversation.init(self.ctx, self.model, self.tokenizer, self.template, .{
                .capacity = self.opts.context_len,
                .max_response_tokens = req.max_tokens,
                .think_off = !req.think,
                .sampler = req.sampling,
                .extra_stop_ids = self.opts.extra_stop_ids,
                .stop_sequences = req.stop,
                .logit_processor = processor,
            });
            defer convo.deinit();

            const produced = try convo.sendRendered(rendered, sink);
            const finish: types.FinishReason = if (produced >= req.max_tokens or convo.cache.len >= convo.cache.capacity)
                .length
            else
                .stop;
            return .{
                .prompt_tokens = convo.history.items.len - produced,
                .completion_tokens = produced,
                .finish = finish,
            };
        }
    };
}
