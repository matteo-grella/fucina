//! Grammar/JSON-schema constrained decoding via the vendored
//! [llguidance](https://github.com/guidance-ai/llguidance) engine
//! (`vendor/llguidance`, enabled with `-Dllguidance=true`).
//!
//! `Constraint` compiles a grammar (JSON schema, regex, Lark variant, or an
//! llguidance composite) against a Fucina tokenizer and adapts it to the
//! `LogitProcessor` seam (`logit_processor.zig`): before every sampling step
//! it writes `-inf` over the logits of tokens the grammar forbids, and it
//! consumes each committed token to advance the parser. When the grammar is
//! complete the mask forces the configured stop/EOS token, so existing stop
//! handling ends the reply — no decode-loop changes anywhere. Composes with
//! speculative decoding for free (grammar-invalid draft tokens lose the
//! `sampled == draft` comparison and are rejected; see `logit_processor.zig`).
//!
//! The FFI boundary is the checked-in C header
//! `vendor/llguidance/parser/llguidance.h`; the extern declarations below are
//! a hand-written subset of it (the Matcher API). When updating the vendored
//! version, re-check them against the header (`vendor/llguidance/README.md`
//! step 5) — the gated tests include a create/mask/consume ABI round trip.
//!
//! Without `-Dllguidance=true`, this module still compiles (types, `enabled`,
//! a stub `Constraint`) but `Constraint.init` returns
//! `error.LlguidanceNotEnabled` and none of the externs are referenced, so
//! the default build links no Rust code.

const std = @import("std");
const build_options = @import("llm_build_options");
const logit_processor = @import("logit_processor.zig");
const bpe = @import("tokenizer.zig");
const spm = @import("spm_tokenizer.zig");

const Allocator = std.mem.Allocator;
pub const LogitProcessor = logit_processor.LogitProcessor;

/// True when the build was configured with `-Dllguidance=true`.
pub const enabled: bool = build_options.llguidance;

pub const Error = error{
    /// Build without `-Dllguidance=true`: the engine is not linked.
    LlguidanceNotEnabled,
    /// No `eos_token` given and the tokenizer has no EOS id.
    NoEosToken,
    /// llguidance rejected the vocabulary/tokenizer wiring (details logged).
    TokenizerInitFailed,
    /// The grammar failed to compile (details logged).
    InvalidGrammar,
    /// A matcher call failed mid-decode (details logged); the constraint is
    /// in an error state and masks everything except the stop token.
    MatcherFailed,
} || Allocator.Error;

/// The grammar formats llguidance compiles. Payloads are borrowed for the
/// duration of `Constraint.init` only.
pub const Grammar = union(enum) {
    /// A JSON schema (stringified). Coverage notes:
    /// vendor/llguidance/docs/json_schema.md.
    json_schema: []const u8,
    /// A Rust-syntax regular expression the whole reply must match.
    regex: []const u8,
    /// A grammar in llguidance's Lark variant
    /// (vendor/llguidance/docs/syntax.md).
    lark: []const u8,
    /// The composite "llguidance" JSON form (a list of Lark/JSON-schema
    /// grammars).
    llguidance: []const u8,

    fn kind(self: Grammar) [:0]const u8 {
        return switch (self) {
            .json_schema => "json",
            .regex => "regex",
            .lark => "lark",
            .llguidance => "llguidance",
        };
    }

    fn data(self: Grammar) []const u8 {
        return switch (self) {
            inline else => |d| d,
        };
    }
};

pub const Options = struct {
    /// The token the mask forces once the grammar is complete (and the id
    /// llguidance treats as EOS). Defaults to the tokenizer's `eosId()`; chat
    /// runners pass the template's stop-marker id so a finished grammar ends
    /// the turn.
    eos_token: ?u32 = null,
    /// Additional EOS ids (e.g. a stray `<eos>` alongside `<end_of_turn>`).
    extra_eos: []const u32 = &.{},
    /// Mask width = number of logit rows the model emits. Defaults to the
    /// tokenizer vocab; pass the MODEL's vocab size when it is padded larger
    /// (`config.vocab_size`) — padding ids get empty token bytes and are
    /// never allowed. Values below the tokenizer vocab are clamped up.
    n_vocab: ?usize = null,
    /// stderr log level for the engine AND this wrapper's own diagnostics
    /// (grammar-rejection / matcher-failure detail): 0 silent, 1 warnings,
    /// 2 info.
    log_level: u32 = 1,
};

/// Compiled grammar + tokenizer bridge + mask state for ONE decode stream.
/// Single-threaded mutable state, like the `Sampler` that hosts its
/// processor. The struct must not move after `processor()` is taken (the
/// vtable closes over `self`).
pub const Constraint = if (enabled) ConstraintImpl else ConstraintStub;

// ---------------------------------------------------------------------------
// Extern declarations — hand-written subset of parser/llguidance.h (Matcher
// API + v2 tokenizer init). Layouts follow the C definitions exactly; only
// referenced when `enabled`, so disabled builds link no Rust symbols.
// ---------------------------------------------------------------------------

const c = struct {
    const LlgTokenizer = opaque {};
    const LlgMatcher = opaque {};

    const LlgParserLimits = extern struct {
        max_items_in_row: usize,
        initial_lexer_fuel: u64,
        step_lexer_fuel: u64,
        step_max_items: usize,
        max_lexer_states: usize,
        max_grammar_size: usize,
        precompute_large_lexemes: bool,
        verbose_errors: bool,
    };

    const LlgConstraintInit = extern struct {
        tokenizer: ?*const LlgTokenizer,
        log_buffer_level: u32,
        log_stderr_level: u32,
        ff_tokens_ok: bool,
        backtrack_ok: bool,
        /// Zero fields select llguidance's defaults.
        limits: LlgParserLimits,
    };

    /// Must be thread-safe per the header; llguidance built without its
    /// `rayon` feature (the vendored default) only ever calls it on the
    /// thread driving the constraint.
    const LlgTokenizeFn = *const fn (
        user_data: ?*const anyopaque,
        bytes: ?[*]const u8,
        bytes_len: usize,
        output_tokens: ?[*]u32,
        output_tokens_len: usize,
    ) callconv(.c) usize;

    const LlgTokenizerInitV2 = extern struct {
        struct_size: usize,
        vocab_size: u32,
        tok_eos: u32,
        token_lens: ?[*]const u32,
        token_bytes: ?[*]const u8,
        tokenizer_json: ?[*:0]const u8,
        tokenize_assumes_string: bool,
        tokenize_fn: ?LlgTokenizeFn,
        use_approximate_greedy_tokenize_fn: bool,
        tokenize_user_data: ?*const anyopaque,
        slices: ?[*]const ?[*:0]const u8,
        tok_eos_extra: ?[*]const u32,
        tok_eos_extra_count: u32,
    };

    extern fn llg_new_tokenizer_v2(tok_init: *const LlgTokenizerInitV2, error_string: ?[*]u8, error_string_len: usize) ?*LlgTokenizer;
    extern fn llg_free_tokenizer(tok: ?*LlgTokenizer) void;

    extern fn llg_constraint_init_set_defaults(init: *LlgConstraintInit, tokenizer: ?*const LlgTokenizer) void;

    extern fn llg_new_matcher(init: *const LlgConstraintInit, constraint_type: [*:0]const u8, data: [*:0]const u8) *LlgMatcher;
    extern fn llg_free_matcher(matcher: ?*LlgMatcher) void;
    extern fn llg_matcher_get_error(matcher: *LlgMatcher) ?[*:0]const u8;
    extern fn llg_matcher_is_error(matcher: *const LlgMatcher) bool;
    extern fn llg_matcher_get_mask_byte_size(matcher: *const LlgMatcher) usize;
    extern fn llg_matcher_compute_mask_into(matcher: *LlgMatcher, mask_dest: [*]u32, mask_byte_len: usize) i32;
    extern fn llg_matcher_consume_token(matcher: *LlgMatcher, token: u32) i32;
    extern fn llg_matcher_is_stopped(matcher: *const LlgMatcher) bool;
    extern fn llg_matcher_is_accepting(matcher: *LlgMatcher) bool;
    extern fn llg_matcher_reset(matcher: *LlgMatcher) i32;
    extern fn llg_matcher_compute_ff_tokens(matcher: *LlgMatcher, output: [*]u32, output_len: usize) i32;
    extern fn llg_matcher_validate_tokens(matcher: *LlgMatcher, tokens: [*]const u32, n_tokens: usize) i32;
    extern fn llg_clone_matcher(matcher: *const LlgMatcher) *LlgMatcher;
    extern fn llg_clone_tokenizer(tok: *const LlgTokenizer) *LlgTokenizer;
    extern fn llg_get_version() [*:0]const u8;
};

/// The llguidance + derivre version string, e.g.
/// "llguidance@1.7.6 derivre@0.3.12" ("llguidance disabled" without the
/// build flag).
pub fn version() []const u8 {
    if (comptime !enabled) return "llguidance disabled";
    return std.mem.span(c.llg_get_version());
}

/// toktrie's special-token convention: token bytes starting with 0xFF (and
/// longer than one byte) name a special/control token that never matches
/// literal text — so a grammar cannot be steered into emitting `<|im_end|>`
/// by a string that merely CONTAINS those characters. 0xFF never occurs in
/// valid UTF-8, so no regular byte-level token collides with the marker.
const special_marker: u8 = 0xFF;

/// Which Fucina tokenizer feeds the bridge. BPE has no per-token attribute
/// table, so control tokens are recognized by the `<|...|>` marker shape
/// (the byte-BPE families' convention — the same set `encodeWithSpecials`
/// resolves atomically). SPM uses its GGUF-declared attrs.
const TokKind = union(enum) {
    bpe: *const bpe.Tokenizer,
    spm: *const spm.Tokenizer,
};

/// Heap-pinned state behind llguidance's `tokenize_fn` callback (the address
/// must stay stable for the tokenizer's lifetime).
const Bridge = struct {
    allocator: Allocator,
    kind: TokKind,

    fn tokenize(user_data: ?*const anyopaque, bytes: ?[*]const u8, bytes_len: usize, output_tokens: ?[*]u32, output_tokens_len: usize) callconv(.c) usize {
        const self: *const Bridge = @ptrCast(@alignCast(user_data orelse return 0));
        if (bytes_len == 0) return 0;
        const text = (bytes orelse return 0)[0..bytes_len];

        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(self.allocator);
        switch (self.kind) {
            // Plain byte-level encode: no marker resolution, no BOS/EOS —
            // llguidance re-tokenizes grammar-forced byte strings.
            .bpe => |t| t.encodePlainAppend(self.allocator, text, &ids) catch return 0,
            .spm => |t| {
                const out = t.encodeRaw(self.allocator, text) catch return 0;
                defer self.allocator.free(out);
                ids.appendSlice(self.allocator, out) catch return 0;
            },
        }
        if (output_tokens) |out| {
            const n = @min(ids.items.len, output_tokens_len);
            @memcpy(out[0..n], ids.items[0..n]);
        }
        return ids.items.len;
    }
};

/// Per-token raw bytes for the toktrie, `token_lens`/`token_bytes` form.
/// Padding ids past the tokenizer vocab get length 0 (never allowed).
fn buildVocab(allocator: Allocator, kind: TokKind, n_vocab: usize) !struct { lens: []u32, bytes: []u8 } {
    const lens = try allocator.alloc(u32, n_vocab);
    errdefer allocator.free(lens);
    var blob: std.ArrayList(u8) = .empty;
    errdefer blob.deinit(allocator);

    const tok_vocab = switch (kind) {
        inline else => |t| t.vocabSize(),
    };
    for (lens, 0..) |*len, id| {
        const before = blob.items.len;
        if (id < tok_vocab) switch (kind) {
            .bpe => |t| {
                const s = t.vocab[id];
                if (s.len >= 4 and std.mem.startsWith(u8, s, "<|") and std.mem.endsWith(u8, s, "|>")) {
                    try blob.append(allocator, special_marker);
                    try blob.appendSlice(allocator, s);
                } else {
                    try t.decodeAppend(allocator, @intCast(id), &blob);
                }
            },
            .spm => |t| switch (t.attrs[id]) {
                // decodeAppend suppresses control/unknown; mark them special
                // under their literal name instead.
                .control, .unknown => {
                    try blob.append(allocator, special_marker);
                    try blob.appendSlice(allocator, t.vocab[id]);
                },
                else => try t.decodeAppend(allocator, @intCast(id), &blob),
            },
        };
        len.* = @intCast(blob.items.len - before);
    }
    return .{ .lens = lens, .bytes = try blob.toOwnedSlice(allocator) };
}

const ConstraintImpl = struct {
    allocator: Allocator,
    bridge: *Bridge,
    /// False on clones: the tokenize bridge belongs to the cloned-from
    /// constraint, which must outlive this one.
    owns_bridge: bool = true,
    tok: *c.LlgTokenizer,
    matcher: *c.LlgMatcher,
    /// One bit per token id, written by `llg_matcher_compute_mask_into`.
    mask: []u32,
    n_vocab: usize,
    eos: u32,
    log_level: u32,
    /// Set when a matcher call failed mid-decode: `process` then masks
    /// everything except the stop token so the stream terminates cleanly.
    failed: bool = false,

    /// Compile `grammar` against `tokenizer` (`*const llm.tokenizer.Tokenizer`
    /// or `*const llm.spm_tokenizer.Tokenizer`; both are borrowed and must
    /// outlive the constraint). An invalid grammar fails here, loudly.
    pub fn init(allocator: Allocator, tokenizer: anytype, grammar: Grammar, options: Options) Error!ConstraintImpl {
        const kind: TokKind = switch (@TypeOf(tokenizer)) {
            *const bpe.Tokenizer, *bpe.Tokenizer => .{ .bpe = tokenizer },
            *const spm.Tokenizer, *spm.Tokenizer => .{ .spm = tokenizer },
            else => @compileError("Constraint.init: tokenizer must be *const llm.tokenizer.Tokenizer or *const llm.spm_tokenizer.Tokenizer, got " ++ @typeName(@TypeOf(tokenizer))),
        };
        const tok_vocab = switch (kind) {
            inline else => |t| t.vocabSize(),
        };
        const n_vocab = @max(options.n_vocab orelse tok_vocab, tok_vocab);
        const eos = options.eos_token orelse switch (kind) {
            inline else => |t| t.eosId(),
        } orelse return Error.NoEosToken;

        const bridge = try allocator.create(Bridge);
        errdefer allocator.destroy(bridge);
        bridge.* = .{ .allocator = allocator, .kind = kind };

        const vocab = try buildVocab(allocator, kind, n_vocab);
        defer allocator.free(vocab.lens);
        defer allocator.free(vocab.bytes); // copied into the toktrie at init

        var err_buf: [1024]u8 = @splat(0);
        var tok_init: c.LlgTokenizerInitV2 = std.mem.zeroes(c.LlgTokenizerInitV2);
        tok_init.struct_size = @sizeOf(c.LlgTokenizerInitV2);
        tok_init.vocab_size = @intCast(n_vocab);
        tok_init.tok_eos = eos;
        tok_init.token_lens = vocab.lens.ptr;
        tok_init.token_bytes = vocab.bytes.ptr;
        tok_init.tokenize_fn = Bridge.tokenize;
        tok_init.tokenize_user_data = bridge;
        tok_init.tok_eos_extra = if (options.extra_eos.len > 0) options.extra_eos.ptr else null;
        tok_init.tok_eos_extra_count = @intCast(options.extra_eos.len);
        const llg_tok = c.llg_new_tokenizer_v2(&tok_init, &err_buf, err_buf.len) orelse {
            if (options.log_level >= 1) std.log.warn("llguidance tokenizer init failed: {s}", .{std.mem.sliceTo(&err_buf, 0)});
            return Error.TokenizerInitFailed;
        };
        errdefer c.llg_free_tokenizer(llg_tok);

        var cinit: c.LlgConstraintInit = undefined;
        c.llg_constraint_init_set_defaults(&cinit, llg_tok);
        cinit.log_stderr_level = options.log_level;

        const data_z = try allocator.dupeZ(u8, grammar.data());
        defer allocator.free(data_z);
        const matcher = c.llg_new_matcher(&cinit, grammar.kind().ptr, data_z.ptr);
        errdefer c.llg_free_matcher(matcher);
        if (c.llg_matcher_is_error(matcher)) {
            if (options.log_level >= 1) if (c.llg_matcher_get_error(matcher)) |msg| {
                std.log.warn("llguidance grammar rejected: {s}", .{std.mem.span(msg)});
            };
            return Error.InvalidGrammar;
        }

        const mask_bytes = c.llg_matcher_get_mask_byte_size(matcher);
        const mask = try allocator.alloc(u32, mask_bytes / @sizeOf(u32));

        return .{
            .allocator = allocator,
            .bridge = bridge,
            .tok = llg_tok,
            .matcher = matcher,
            .mask = mask,
            .n_vocab = n_vocab,
            .eos = eos,
            .log_level = options.log_level,
        };
    }

    pub fn deinit(self: *ConstraintImpl) void {
        c.llg_free_matcher(self.matcher);
        c.llg_free_tokenizer(self.tok);
        self.allocator.free(self.mask);
        if (self.owns_bridge) self.allocator.destroy(self.bridge);
        self.* = undefined;
    }

    /// An independent decode-stream twin over the same compiled grammar:
    /// the matcher state is deep-cloned (clone right after init/`reset` for
    /// a fresh stream), the tokenizer handle is reference-counted, and the
    /// tokenize bridge is borrowed — `self` must outlive every clone. Much
    /// cheaper than a second `init` (no vocab rebuild, no grammar
    /// recompilation); the per-stream primitive behind `sendBatch` and the
    /// qwen3 `--streams` runner.
    pub fn clone(self: *const ConstraintImpl) Error!ConstraintImpl {
        const mask = try self.allocator.alloc(u32, self.mask.len);
        return .{
            .allocator = self.allocator,
            .bridge = self.bridge,
            .owns_bridge = false,
            .tok = c.llg_clone_tokenizer(self.tok),
            .matcher = c.llg_clone_matcher(self.matcher),
            .mask = mask,
            .n_vocab = self.n_vocab,
            .eos = self.eos,
            .log_level = self.log_level,
            .failed = self.failed,
        };
    }

    /// The `LogitProcessor` adapter to install on a `Sampler` /
    /// `chat.Options.logit_processor`. Borrows `self`; do not move the
    /// constraint afterwards.
    pub fn processor(self: *ConstraintImpl) LogitProcessor {
        return .{ .ptr = self, .vtable = &vtable };
    }

    /// True once the grammar has terminated (the stop token was committed,
    /// or a matcher error forced termination).
    pub fn isStopped(self: *const ConstraintImpl) bool {
        return self.failed or c.llg_matcher_is_stopped(self.matcher);
    }

    /// True while the tokens so far form a COMPLETE sentence of the grammar
    /// (the stop token is allowed but not yet committed).
    pub fn isAccepting(self: *ConstraintImpl) bool {
        return !self.failed and c.llg_matcher_is_accepting(self.matcher);
    }

    /// Re-arm the grammar for a fresh reply (also clears a failed state,
    /// which `llg_matcher_reset` supports as long as grammar compilation
    /// itself succeeded — init guarantees that).
    pub fn reset(self: *ConstraintImpl) Error!void {
        if (c.llg_matcher_reset(self.matcher) != 0) return self.fail();
        self.failed = false;
    }

    /// Grammar-forced continuation tokens for the current state (0 when the
    /// next token is not forced). The seam for feeding forced spans to the
    /// speculative decoder as drafts.
    pub fn ffTokens(self: *ConstraintImpl, buf: []u32) Error!usize {
        if (self.failed) return 0;
        const n = c.llg_matcher_compute_ff_tokens(self.matcher, buf.ptr, buf.len);
        if (n < 0) return self.fail();
        return @intCast(n);
    }

    fn fail(self: *ConstraintImpl) Error {
        if (self.log_level >= 1) if (c.llg_matcher_get_error(self.matcher)) |msg| {
            std.log.warn("llguidance matcher error: {s}", .{std.mem.span(msg)});
        };
        self.failed = true;
        return Error.MatcherFailed;
    }

    const vtable = LogitProcessor.VTable{
        .process = vtProcess,
        .commit = vtCommit,
        .reset = vtReset,
        .forcedTokens = vtForcedTokens,
        .validPrefixLen = vtValidPrefixLen,
    };

    /// Stack cap for the usize↔u32 token conversions in the structural
    /// hooks; far above any draft budget (`speculative.Options.max_draft`).
    const hook_buf_cap = 128;

    fn vtProcess(ptr: *anyopaque, logits: []f32, history: []const usize) anyerror!void {
        _ = history; // the matcher carries its own state
        const self: *ConstraintImpl = @ptrCast(@alignCast(ptr));
        const neg_inf = -std.math.inf(f32);
        // Terminal state (grammar complete, or failed): force the stop token
        // so the existing stop handling ends the reply.
        if (self.isStopped()) {
            for (logits, 0..) |*l, i| {
                if (i != self.eos) l.* = neg_inf;
            }
            return;
        }
        if (c.llg_matcher_compute_mask_into(self.matcher, self.mask.ptr, self.mask.len * @sizeOf(u32)) != 0) {
            return self.fail();
        }
        for (logits, 0..) |*l, tok| {
            const w = tok >> 5;
            const allowed = w < self.mask.len and (self.mask[w] >> @intCast(tok & 31)) & 1 == 1;
            if (!allowed) l.* = neg_inf;
        }
    }

    fn vtCommit(ptr: *anyopaque, token: usize) anyerror!void {
        const self: *ConstraintImpl = @ptrCast(@alignCast(ptr));
        // Post-terminal commits (the forced stop token; a dropped
        // stop-sequence tail) don't advance the matcher.
        if (self.isStopped()) return;
        if (c.llg_matcher_consume_token(self.matcher, @intCast(token)) != 0) {
            return self.fail();
        }
    }

    fn vtReset(ptr: *anyopaque) anyerror!void {
        const self: *ConstraintImpl = @ptrCast(@alignCast(ptr));
        return self.reset();
    }

    /// `LogitProcessor.forcedTokens`: grammar-forced continuation (pure
    /// lookahead; 0 when the next token is a free choice or the grammar is
    /// terminal — the terminal forced-stop is the mask's job, not a draft).
    fn vtForcedTokens(ptr: *anyopaque, buf: []usize) usize {
        const self: *ConstraintImpl = @ptrCast(@alignCast(ptr));
        if (self.isStopped()) return 0;
        var tmp: [hook_buf_cap]u32 = undefined;
        const cap = @min(buf.len, tmp.len);
        const n = c.llg_matcher_compute_ff_tokens(self.matcher, &tmp, cap);
        if (n <= 0) return 0;
        const count: usize = @intCast(n);
        for (buf[0..count], tmp[0..count]) |*d, s| d.* = s;
        return count;
    }

    /// `LogitProcessor.validPrefixLen`: how many leading `tokens` the
    /// grammar accepts (pure lookahead). Errors degrade to 0 — a draft is
    /// advice, never worth poisoning the constraint over.
    fn vtValidPrefixLen(ptr: *anyopaque, tokens: []const usize) usize {
        const self: *ConstraintImpl = @ptrCast(@alignCast(ptr));
        if (self.isStopped()) return 0;
        var tmp: [hook_buf_cap]u32 = undefined;
        const n = @min(tokens.len, tmp.len);
        for (tmp[0..n], tokens[0..n]) |*d, s| d.* = @intCast(s);
        const v = c.llg_matcher_validate_tokens(self.matcher, &tmp, n);
        if (v < 0) return 0;
        return @intCast(v);
    }
};

/// `-Dllguidance=false` stand-in: same surface, `init` always fails. Keeps
/// callers (runners, doc snippets) compiling in pure-Zig builds.
const ConstraintStub = struct {
    pub fn init(allocator: Allocator, tokenizer: anytype, grammar: Grammar, options: Options) Error!ConstraintStub {
        _ = allocator;
        _ = tokenizer;
        _ = grammar;
        _ = options;
        return Error.LlguidanceNotEnabled;
    }
    pub fn deinit(self: *ConstraintStub) void {
        _ = self;
    }
    pub fn clone(self: *const ConstraintStub) Error!ConstraintStub {
        _ = self;
        unreachable; // init never succeeds
    }
    pub fn processor(self: *ConstraintStub) LogitProcessor {
        _ = self;
        unreachable; // init never succeeds
    }
    pub fn isStopped(self: *const ConstraintStub) bool {
        _ = self;
        unreachable;
    }
    pub fn isAccepting(self: *ConstraintStub) bool {
        _ = self;
        unreachable;
    }
    pub fn reset(self: *ConstraintStub) Error!void {
        _ = self;
        unreachable;
    }
    pub fn ffTokens(self: *ConstraintStub, buf: []u32) Error!usize {
        _ = self;
        _ = buf;
        unreachable;
    }
};

test {
    _ = @import("llguidance_tests.zig");
}
