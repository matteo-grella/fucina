//! nanochat inference engine + chat CLI.
//!
//! Ports `refs/nanochat/nanochat/engine.py` (Engine.generate, RowState forced-
//! token deque, the calculator tool state machine, use_calculator) and
//! `refs/nanochat/scripts/chat_cli.py` (conversation-token protocol + CLI) onto
//! the KV-cache decode path (`model.forwardStep` + `model.Cache`, proven
//! bit-identical to the full forward) and the raw-byte tokenizer. Sampling is a small
//! local sampler over `fucina` only (temp==0 → argmax, the parity path; temp>0 →
//! top-k + softmax + a fucina-rng multinomial draw — functional, NOT
//! torch-bit-parity, matching engine.py sample_next_token modulo the RNG).
//! The pretokenizer's unicode tables come through the `fucina_llm` re-export
//! (`llm.unicode_categories`), so nanochat code can share a compilation with
//! fucina_llm consumers (the serve example hosts both).
//!
//! Files: engine over one Model + one Tokenizer, batch-1 prefill of the whole
//! context into a fresh Cache, then a one-token-per-step decode loop; the tool
//! state machine forces `<|output_start|>`+encode(result)+`<|output_end|>` after
//! a `<|python_start|>…<|python_end|>` arithmetic block; a raw-byte streaming
//! writer; and `runChat` with chat_cli.py's flags plus base-completion and
//! `--init-from` conveniences.

const std = @import("std");
const fucina = @import("fucina");
const model_mod = @import("model.zig");
const tok_mod = @import("tokenizer.zig");

const Allocator = std.mem.Allocator;
const ExecContext = fucina.ExecContext;
const Tensor = fucina.Tensor;
const Config = model_mod.Config;
const Model = model_mod.Model;
const Cache = model_mod.Cache;
const Tokenizer = tok_mod.Tokenizer;
const safetensors = fucina.safetensors;

/// Logits view fed to the sampler — the model's post-softcap logits type.
const Logits = Tensor(.{ .seq, .vocab });

/// Splitmix64-based (fucina.rng) uniform draws for the temp>0 multinomial. A
/// pure function of (seed, counter); NOT torch-bit-parity (engine.py uses a
/// torch Generator) — only temp==0 argmax is a parity gate.
const Rng = struct {
    seed: u64,
    counter: u64 = 0,

    /// A uniform f32 in [0, 1) from 24 high bits of the next splitmix64 output.
    fn float(self: *Rng) f32 {
        const bits = fucina.rng.at(self.seed, self.counter);
        self.counter += 1;
        return @as(f32, @floatFromInt(bits >> 40)) * (1.0 / 16777216.0); // 2^24
    }
};

/// Pick the next token from a single row of logits `[1, vocab]`. temp==0 →
/// argmax (engine.py sample_next_token temperature==0 branch); temp>0 → keep the
/// top-k logits, softmax(vals/temp), then a multinomial draw. Must run inside an
/// open exec scope (the topK values arm is a scope-owned borrow); the i64 index
/// tensors are CALLER-owned even under the scope (REFERENCE §6.3) and are
/// released here.
fn sampleToken(
    ctx: *ExecContext,
    allocator: Allocator,
    logits: *Logits,
    temperature: f32,
    top_k: usize,
    rng: *Rng,
) !usize {
    if (temperature <= 0) {
        var idx = try logits.argmax(ctx, .vocab);
        defer idx.deinit();
        return @intCast(try idx.item());
    }

    const vocab = logits.dim(.vocab);
    const k = @min(if (top_k > 0) top_k else vocab, vocab);
    var top = try logits.topK(ctx, .vocab, k, .top);
    defer top.deinit();
    const vals = try top.values.dataConst(); // logits, descending (vals[0] = max)
    const idxs = try top.indices.dataConst(); // token ids as i64

    const probs = try allocator.alloc(f32, k);
    defer allocator.free(probs);
    const inv_t = 1.0 / temperature;
    const max_logit = vals[0];
    var sum: f32 = 0;
    for (0..k) |i| {
        probs[i] = @exp((vals[i] - max_logit) * inv_t);
        sum += probs[i];
    }
    const r = rng.float() * sum;
    var acc: f32 = 0;
    for (0..k) |i| {
        acc += probs[i];
        if (r <= acc) return @intCast(idxs[i]);
    }
    return @intCast(idxs[k - 1]);
}

// ===========================================================================
// Calculator tool — ported `use_calculator` (engine.py:46-79)
// ===========================================================================

/// A calculator value with Python-style int/float typing so `str(result)`
/// round-trips: `2+3*4` stays an int → "14", but any `/` or float literal makes
/// the result a float → "5.0" (engine.py evaluates with Python's `eval`).
const Value = union(enum) {
    int: i64,
    flt: f64,

    fn toFloat(self: Value) f64 {
        return switch (self) {
            .int => |i| @floatFromInt(i),
            .flt => |f| f,
        };
    }
};

const CalcError = error{ ParseError, DivZero };

/// Recursive-descent evaluator over `+ - * /`, parentheses and unary sign with
/// correct precedence, IEEE-double arithmetic that matches Python's float ops
/// for these operators. Int operands keep int typing through `+ - *` (Python
/// int semantics); `/` is always true division (float); i64 overflow falls back
/// to float (a niche divergence from Python's bignums — not exercised here).
const Parser = struct {
    s: []const u8,
    i: usize = 0,

    fn peek(self: *Parser) ?u8 {
        while (self.i < self.s.len and self.s[self.i] == ' ') self.i += 1;
        return if (self.i < self.s.len) self.s[self.i] else null;
    }

    fn expr(self: *Parser) CalcError!Value {
        var acc = try self.term();
        while (self.peek()) |c| {
            if (c != '+' and c != '-') break;
            self.i += 1;
            const rhs = try self.term();
            acc = addSub(acc, rhs, c == '+');
        }
        return acc;
    }

    fn term(self: *Parser) CalcError!Value {
        var acc = try self.factor();
        while (self.peek()) |c| {
            if (c != '*' and c != '/') break;
            self.i += 1;
            const rhs = try self.factor();
            if (c == '*') {
                acc = mul(acc, rhs);
            } else {
                const d = rhs.toFloat();
                if (d == 0) return CalcError.DivZero;
                // Bind the numerator first: `acc = .{ .flt = acc.toFloat()/d }`
                // builds the union in-place into `acc`, aliasing the read.
                const num = acc.toFloat();
                acc = .{ .flt = num / d };
            }
        }
        return acc;
    }

    fn factor(self: *Parser) CalcError!Value {
        const c = self.peek() orelse return CalcError.ParseError;
        if (c == '+') {
            self.i += 1;
            return self.factor();
        }
        if (c == '-') {
            self.i += 1;
            const v = try self.factor();
            return switch (v) {
                .int => |x| .{ .int = -x },
                .flt => |x| .{ .flt = -x },
            };
        }
        return self.primary();
    }

    fn primary(self: *Parser) CalcError!Value {
        const c = self.peek() orelse return CalcError.ParseError;
        if (c == '(') {
            self.i += 1;
            const v = try self.expr();
            if (self.peek() != @as(?u8, ')')) return CalcError.ParseError;
            self.i += 1;
            return v;
        }
        return self.number();
    }

    fn number(self: *Parser) CalcError!Value {
        _ = self.peek(); // skip leading spaces
        const start = self.i;
        var is_float = false;
        while (self.i < self.s.len) : (self.i += 1) {
            const ch = self.s[self.i];
            if (ch >= '0' and ch <= '9') continue;
            if (ch == '.') {
                if (is_float) return CalcError.ParseError; // second dot
                is_float = true;
                continue;
            }
            break;
        }
        const lit = self.s[start..self.i];
        if (lit.len == 0 or (lit.len == 1 and lit[0] == '.')) return CalcError.ParseError;
        if (is_float) {
            const f = std.fmt.parseFloat(f64, lit) catch return CalcError.ParseError;
            return .{ .flt = f };
        }
        const n = std.fmt.parseInt(i64, lit, 10) catch return CalcError.ParseError;
        return .{ .int = n };
    }
};

fn addSub(a: Value, b: Value, add: bool) Value {
    if (a == .int and b == .int) {
        const res = if (add) @addWithOverflow(a.int, b.int) else @subWithOverflow(a.int, b.int);
        if (res[1] == 0) return .{ .int = res[0] };
    }
    const av = a.toFloat();
    const bv = b.toFloat();
    return .{ .flt = if (add) av + bv else av - bv };
}

fn mul(a: Value, b: Value) Value {
    if (a == .int and b == .int) {
        const res = @mulWithOverflow(a.int, b.int);
        if (res[1] == 0) return .{ .int = res[0] };
    }
    return .{ .flt = a.toFloat() * b.toFloat() };
}

/// Python `str(result)` for the evaluated value. Ints print bare; floats print
/// with a `.0` when integral (Python `repr(5.0)` == "5.0").
fn formatValue(allocator: Allocator, v: Value) ![]u8 {
    switch (v) {
        .int => |i| return std.fmt.allocPrint(allocator, "{d}", .{i}),
        .flt => |f| {
            const s = try std.fmt.allocPrint(allocator, "{d}", .{f});
            var plain = true;
            for (s) |ch| {
                if (ch == '.' or ch == 'e' or ch == 'E' or ch == 'n' or ch == 'i') { // nan/inf too
                    plain = false;
                    break;
                }
            }
            if (!plain) return s;
            defer allocator.free(s);
            return std.fmt.allocPrint(allocator, "{s}.0", .{s});
        },
    }
}

/// Port of `use_calculator` (engine.py:46-79): strip commas; pure arithmetic
/// over "0123456789*+-/.() " is evaluated (rejecting the `**` power operator),
/// and the string-op branch supports exactly `'lit'.count('lit')` — the only
/// form python's guarded eval accepts in practice (its charset has no
/// operators, so counts cannot combine with arithmetic; its dangerous-pattern
/// list blocks the rest). Returns the `str(result)` string (owned) or null on
/// any rejection or evaluation error (Python's `eval_with_timeout` → None).
fn useCalculator(allocator: Allocator, raw_expr: []const u8) !?[]u8 {
    // expr.replace(",", "")
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);
    for (raw_expr) |ch| {
        if (ch != ',') try buf.append(allocator, ch);
    }
    const expr = buf.items;

    // all([x in "0123456789*+-/.() " for x in expr]) → the math branch.
    const arithmetic = "0123456789*+-/.() ";
    const is_math = for (expr) |ch| {
        if (std.mem.indexOfScalar(u8, arithmetic, ch) == null) break false;
    } else true;
    if (is_math) {
        // "**" disallowed (power operator)
        if (std.mem.indexOf(u8, expr, "**") != null) return null;
        var parser = Parser{ .s = expr };
        const v = parser.expr() catch return null;
        if (parser.peek() != null) return null; // trailing garbage
        return formatValue(allocator, v) catch return null;
    }
    return countExpr(allocator, expr);
}

/// engine.py's string-op branch: charset + dangerous-pattern guards, then
/// `'haystack'.count('needle')` with python str.count semantics
/// (non-overlapping; empty needle counts len+1). Any other shape → null.
fn countExpr(allocator: Allocator, expr: []const u8) !?[]u8 {
    const allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ ";
    for (expr) |ch| {
        if (std.mem.indexOfScalar(u8, allowed, ch) == null) return null;
    }
    const dangerous = [_][]const u8{
        "__",     "import",  "exec", "eval",    "compile", "open",    "file",    "input",
        "raw_input", "globals", "locals", "vars", "dir",     "getattr", "setattr", "delattr",
        "hasattr",
    };
    const lower = try std.ascii.allocLowerString(allocator, expr);
    defer allocator.free(lower);
    for (dangerous) |pat| {
        if (std.mem.indexOf(u8, lower, pat) != null) return null;
    }
    if (std.mem.indexOf(u8, expr, ".count(") == null) return null;

    var p = StrParser{ .s = expr };
    const hay = p.stringLit() orelse return null;
    if (!p.consume(".count(")) return null;
    const needle = p.stringLit() orelse return null;
    if (!p.consume(")")) return null;
    p.skipSpaces();
    if (p.i != p.s.len) return null;

    var count: usize = undefined;
    if (needle.len == 0) {
        count = hay.len + 1; // python ''.count('') semantics
    } else {
        count = 0;
        var i: usize = 0;
        while (std.mem.indexOfPos(u8, hay, i, needle)) |at| {
            count += 1;
            i = at + needle.len;
        }
    }
    return try std.fmt.allocPrint(allocator, "{d}", .{count});
}

/// Minimal scanner for the `'lit'.count('lit')` form (no escapes — the charset
/// excludes backslash).
const StrParser = struct {
    s: []const u8,
    i: usize = 0,

    fn skipSpaces(self: *StrParser) void {
        while (self.i < self.s.len and self.s[self.i] == ' ') self.i += 1;
    }

    fn stringLit(self: *StrParser) ?[]const u8 {
        self.skipSpaces();
        if (self.i >= self.s.len) return null;
        const q = self.s[self.i];
        if (q != '\'' and q != '"') return null;
        const start = self.i + 1;
        const end = std.mem.indexOfScalarPos(u8, self.s, start, q) orelse return null;
        self.i = end + 1;
        return self.s[start..end];
    }

    fn consume(self: *StrParser, lit: []const u8) bool {
        self.skipSpaces();
        if (std.mem.startsWith(u8, self.s[self.i..], lit)) {
            self.i += lit.len;
            return true;
        }
        return false;
    }
};

// ===========================================================================
// Engine — batch-1 prefill + per-step decode with the tool state machine
// ===========================================================================

pub const GenParams = struct {
    max_tokens: usize = 256,
    temperature: f32 = 0.6,
    top_k: usize = 50,
    seed: u64 = 42,
};

/// One inference engine over a Model + Tokenizer. Holds the resolved special
/// token ids and the per-generation tool state (forced-token FIFO + the
/// in-progress python expression). One generation = one fresh Cache: a batch-1
/// prefill of the whole context, then a one-token-per-step decode loop.
pub const Engine = struct {
    model: *const Model,
    tokenizer: *const Tokenizer,
    ctx: *ExecContext,
    allocator: Allocator,

    // Special token ids for the tool / stop state machine (engine.py:186-192).
    bos: usize,
    python_start: usize,
    python_end: usize,
    output_start: usize,
    output_end: usize,
    assistant_end: usize,

    // Per-generation tool state.
    forced: std.ArrayList(usize),
    forced_head: usize,
    expr_tokens: std.ArrayList(usize),
    in_python: bool,

    pub fn init(model: *const Model, tokenizer: *const Tokenizer, ctx: *ExecContext, allocator: Allocator) Engine {
        return .{
            .model = model,
            .tokenizer = tokenizer,
            .ctx = ctx,
            .allocator = allocator,
            .bos = tokenizer.bosId(),
            .python_start = tokenizer.specialId("<|python_start|>").?,
            .python_end = tokenizer.specialId("<|python_end|>").?,
            .output_start = tokenizer.specialId("<|output_start|>").?,
            .output_end = tokenizer.specialId("<|output_end|>").?,
            .assistant_end = tokenizer.specialId("<|assistant_end|>").?,
            .forced = .empty,
            .forced_head = 0,
            .expr_tokens = .empty,
            .in_python = false,
        };
    }

    pub fn deinit(self: *Engine) void {
        self.forced.deinit(self.allocator);
        self.expr_tokens.deinit(self.allocator);
        self.* = undefined;
    }

    fn resetTool(self: *Engine) void {
        self.forced.clearRetainingCapacity();
        self.forced_head = 0;
        self.expr_tokens.clearRetainingCapacity();
        self.in_python = false;
    }

    fn forcedLen(self: *const Engine) usize {
        return self.forced.items.len - self.forced_head;
    }

    fn forcedPop(self: *Engine) usize {
        const t = self.forced.items[self.forced_head];
        self.forced_head += 1;
        if (self.forced_head == self.forced.items.len) {
            self.forced.clearRetainingCapacity();
            self.forced_head = 0;
        }
        return t;
    }

    /// The RowState tool logic (engine.py:252-267): open/close python blocks,
    /// collect expression tokens, and on `<|python_end|>` evaluate them and
    /// enqueue `<|output_start|>`+encode(str(result))+`<|output_end|>` as forced
    /// tokens (only if the calculator returns non-null).
    fn handleTool(self: *Engine, next: usize) !void {
        if (next == self.python_start) {
            self.in_python = true;
            self.expr_tokens.clearRetainingCapacity();
        } else if (next == self.python_end and self.in_python) {
            self.in_python = false;
            if (self.expr_tokens.items.len > 0) {
                const expr_ids = try self.allocator.alloc(u32, self.expr_tokens.items.len);
                defer self.allocator.free(expr_ids);
                for (expr_ids, self.expr_tokens.items) |*d, s| d.* = @intCast(s);
                const expr_text = try self.tokenizer.decode(self.allocator, expr_ids);
                defer self.allocator.free(expr_text);
                if (try useCalculator(self.allocator, expr_text)) |result| {
                    defer self.allocator.free(result);
                    const result_ids = try self.tokenizer.encode(self.allocator, result);
                    defer self.allocator.free(result_ids);
                    try self.forced.append(self.allocator, self.output_start);
                    for (result_ids) |id| try self.forced.append(self.allocator, @intCast(id));
                    try self.forced.append(self.allocator, self.output_end);
                }
            }
            self.expr_tokens.clearRetainingCapacity();
        } else if (self.in_python) {
            try self.expr_tokens.append(self.allocator, next);
        }
    }

    /// Forward `chunk` at absolute position `pos0` through the cache and sample
    /// the next token from the last row's logits. temp==0 → argmax (parity path);
    /// temp>0 → top-k+softmax+multinomial. The whole forward + sample runs inside
    /// one exec scope so the intermediates are freed before returning the token id.
    fn stepSample(self: *Engine, params: GenParams, rng: *Rng, cache: *Cache, chunk: []const usize, pos0: usize) !usize {
        // Inference never backwards: skip tape recording (no backward nodes,
        // no per-layer k/v cloneViews) — forward values are identical.
        var ng = fucina.noGrad();
        defer ng.close();
        const scope = self.ctx.openExecScope();
        defer self.ctx.closeExecScope(scope);
        var logits = try self.model.forwardStep(self.ctx, cache, chunk, pos0);
        var last: Logits = try logits.narrow(self.ctx, .seq, chunk.len - 1, 1); // [1, vocab]
        return sampleToken(self.ctx, self.allocator, &last, params.temperature, params.top_k, rng);
    }

    /// Generate up to `params.max_tokens` continuation tokens for `prompt`
    /// (the full context to prefill). Non-terminal emitted tokens are appended to
    /// `out` and, when `stream` is non-null, written as raw bytes. Stops early on
    /// `<|assistant_end|>` or `<|bos|>`. Mirrors engine.py Engine.generate for a
    /// single row (num_samples=1): sample from the current logits, but a queued
    /// forced token overrides the sample; every emitted token is fed back through
    /// the cache to produce the next step's logits.
    pub fn generate(
        self: *Engine,
        prompt: []const usize,
        params: GenParams,
        out: *std.ArrayList(usize),
        stream: ?*std.Io.Writer,
    ) !void {
        std.debug.assert(prompt.len > 0);
        self.resetTool();

        var rng = Rng{ .seed = params.seed };

        var cache = try Cache.init(self.allocator, self.model.cfg, prompt.len + params.max_tokens + 2);
        defer cache.deinit();

        // Batch-1 prefill of the whole prompt → first sampling logits.
        var sampled = try self.stepSample(params, &rng, &cache, prompt, 0);

        var generated: usize = 0;
        while (generated < params.max_tokens) : (generated += 1) {
            const is_forced = self.forcedLen() > 0;
            const next: usize = if (is_forced) self.forcedPop() else sampled;
            const terminal = next == self.bos or next == self.assistant_end;

            if (!terminal) {
                try out.append(self.allocator, next);
                if (stream) |w| {
                    try w.writeAll(self.tokenizer.vocab[next]);
                    try w.flush();
                }
            }
            try self.handleTool(next);
            if (terminal) break;

            // Feed the emitted token back to advance the cache and get the next
            // step's logits (skip the final, unused forward).
            if (generated + 1 < params.max_tokens) {
                const one = [_]usize{next};
                sampled = try self.stepSample(params, &rng, &cache, &one, cache.len);
            }
        }
    }
};

// ===========================================================================
// Config inference + model loading
// ===========================================================================

/// nanochat fixes head_dim = 64 (init_d6.config.json) and n_embd = n_head *
/// head_dim, so the whole Config is recoverable from the safetensors shapes plus
/// the tokenizer's vocab size — no external config file is needed for chat.
fn inferConfig(file: *const safetensors.File, vocab_size: usize) !Config {
    const head_dim: usize = 64;
    const wte = try file.tensor("transformer.wte.weight"); // [padded_vocab, n_embd]
    const n_embd = wte.shape[1];
    const cq = try file.tensor("transformer.h.0.attn.c_q.weight"); // [n_head*head_dim, n_embd]
    const ck = try file.tensor("transformer.h.0.attn.c_k.weight"); // [n_kv_head*head_dim, n_embd]
    const n_head = cq.shape[0] / head_dim;
    const n_kv_head = ck.shape[0] / head_dim;

    var n_layer: usize = 0;
    var name_buf: [64]u8 = undefined;
    while (true) {
        const name = try std.fmt.bufPrint(&name_buf, "transformer.h.{d}.attn.c_q.weight", .{n_layer});
        _ = file.tensor(name) catch break;
        n_layer += 1;
    }
    if (n_layer == 0) return error.NoLayers;

    return .{
        .sequence_len = 512,
        .vocab_size = vocab_size,
        .n_layer = n_layer,
        .n_head = n_head,
        .n_kv_head = n_kv_head,
        .n_embd = n_embd,
        .window_pattern = "L",
    };
}

pub fn loadModel(ctx: *ExecContext, allocator: Allocator, io: std.Io, path: []const u8, vocab_size: usize) !Model {
    var file = try safetensors.File.load(allocator, io, path);
    const cfg = try inferConfig(&file, vocab_size);
    file.deinit();
    return Model.initFromSafetensors(cfg, ctx, allocator, io, path);
}

// ===========================================================================
// CLI — conversation protocol + REPL / single-prompt (chat_cli.py)
// ===========================================================================

const default_tokenizer = "refs/nanochat-goldens/tokenizer.bin";

const ChatArgs = struct {
    source: ?[]const u8 = null, // -i/--source: checkpoint directory
    init_from: ?[]const u8 = null, // --init-from: a model.safetensors directly
    tokenizer_path: []const u8 = default_tokenizer,
    prompt: ?[]const u8 = null, // -p/--prompt: single-shot
    temperature: f32 = 0.6,
    top_k: usize = 50,
    max_tokens: usize = 256,
    seed: u64 = 42,
    base: bool = false, // base-completion mode: no chat wrapping, stop only on bos
};

fn parseArgs(stdout: *std.Io.Writer, args: []const []const u8) !ChatArgs {
    var a = ChatArgs{};
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        const next = struct {
            fn get(idx: *usize, av: []const []const u8) ?[]const u8 {
                if (idx.* + 1 >= av.len) return null;
                idx.* += 1;
                return av[idx.*];
            }
        }.get;
        if (std.mem.eql(u8, arg, "-i") or std.mem.eql(u8, arg, "--source")) {
            a.source = next(&i, args) orelse return error.MissingValue;
        } else if (std.mem.eql(u8, arg, "--init-from")) {
            a.init_from = next(&i, args) orelse return error.MissingValue;
        } else if (std.mem.eql(u8, arg, "--tokenizer")) {
            a.tokenizer_path = next(&i, args) orelse return error.MissingValue;
        } else if (std.mem.eql(u8, arg, "-p") or std.mem.eql(u8, arg, "--prompt")) {
            a.prompt = next(&i, args) orelse return error.MissingValue;
        } else if (std.mem.eql(u8, arg, "-t") or std.mem.eql(u8, arg, "--temperature")) {
            a.temperature = try std.fmt.parseFloat(f32, next(&i, args) orelse return error.MissingValue);
        } else if (std.mem.eql(u8, arg, "-k") or std.mem.eql(u8, arg, "--top-k")) {
            a.top_k = try std.fmt.parseInt(usize, next(&i, args) orelse return error.MissingValue, 10);
        } else if (std.mem.eql(u8, arg, "--max-tokens")) {
            a.max_tokens = try std.fmt.parseInt(usize, next(&i, args) orelse return error.MissingValue, 10);
        } else if (std.mem.eql(u8, arg, "--seed")) {
            a.seed = try std.fmt.parseInt(u64, next(&i, args) orelse return error.MissingValue, 10);
        } else if (std.mem.eql(u8, arg, "--base")) {
            a.base = true;
        } else if (std.mem.eql(u8, arg, "-g") or std.mem.eql(u8, arg, "--model-tag") or std.mem.eql(u8, arg, "--device-type") or std.mem.eql(u8, arg, "-s") or std.mem.eql(u8, arg, "--step")) {
            // Accepted for chat_cli.py compatibility; no-ops in this port.
            _ = next(&i, args);
        } else {
            try stdout.print("chat: unknown argument: {s}\n", .{arg});
            return error.InvalidArgument;
        }
    }
    return a;
}

/// `chat [-i dir | --init-from f.safetensors] [--tokenizer t.bin] [-p prompt]
/// [-t temp] [-k top_k] [--base]`: load a trained model + tokenizer and either
/// answer one prompt (`-p`) or run an interactive REPL.
pub fn runChat(io: std.Io, stdout: *std.Io.Writer, args: []const []const u8) !void {
    const allocator = std.heap.smp_allocator;

    const a = parseArgs(stdout, args) catch |err| {
        try stdout.print("chat: argument error: {s}\n", .{@errorName(err)});
        return err;
    };

    // Resolve the model.safetensors path from --init-from or <source>/model.safetensors.
    var path_buf: [1024]u8 = undefined;
    const model_path: []const u8 = if (a.init_from) |p|
        p
    else if (a.source) |dir|
        try std.fmt.bufPrint(&path_buf, "{s}/model.safetensors", .{dir})
    else {
        try stdout.writeAll("chat: need -i <checkpoint dir> or --init-from <model.safetensors>\n");
        return error.InvalidArgument;
    };

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var tokenizer = try Tokenizer.loadBin(allocator, io, a.tokenizer_path);
    defer tokenizer.deinit();

    var model = try loadModel(&ctx, allocator, io, model_path, tokenizer.n_vocab);
    defer model.deinit();

    var engine = Engine.init(&model, &tokenizer, &ctx, allocator);
    defer engine.deinit();

    const params = GenParams{
        .max_tokens = a.max_tokens,
        .temperature = a.temperature,
        .top_k = a.top_k,
        .seed = a.seed,
    };

    try stdout.print(
        "nanochat: loaded {s} (n_layer={d} n_embd={d} n_head={d} vocab={d}) | temp={d} top_k={d}{s}\n",
        .{ model_path, model.cfg.n_layer, model.cfg.n_embd, model.cfg.n_head, model.cfg.vocab_size, a.temperature, a.top_k, if (a.base) " base-completion" else "" },
    );
    try stdout.flush();

    if (a.base) {
        // Base-completion mode: bos + encode(prompt), no chat wrapping (matches
        // the base-model greedy oracle). Requires -p.
        const prompt = a.prompt orelse {
            try stdout.writeAll("chat --base: -p <prompt> is required\n");
            return error.InvalidArgument;
        };
        const ids32 = try tokenizer.encodeWithBos(allocator, prompt);
        defer allocator.free(ids32);
        const ctxs = try allocator.alloc(usize, ids32.len);
        defer allocator.free(ctxs);
        for (ctxs, ids32) |*d, s| d.* = s;

        var out: std.ArrayList(usize) = .empty;
        defer out.deinit(allocator);
        try stdout.print("{s}", .{prompt});
        try stdout.flush();
        try engine.generate(ctxs, params, &out, stdout);
        try stdout.writeAll("\n");
        try stdout.flush();
        return;
    }

    // Chat protocol (chat_cli.py): conversation_tokens starts with bos; each turn
    // appends user_start + encode(user) + user_end + assistant_start, generates
    // the assistant reply, then appends assistant_end.
    const user_start = tokenizer.specialId("<|user_start|>").?;
    const user_end = tokenizer.specialId("<|user_end|>").?;
    const assistant_start = tokenizer.specialId("<|assistant_start|>").?;

    var conversation: std.ArrayList(usize) = .empty;
    defer conversation.deinit(allocator);
    try conversation.append(allocator, tokenizer.bosId());

    var stdin_buffer: [8192]u8 = undefined;
    var stdin_reader = std.Io.File.stdin().reader(io, &stdin_buffer);
    const stdin = &stdin_reader.interface;

    while (true) {
        var user_input: []const u8 = undefined;
        if (a.prompt) |p| {
            user_input = p;
        } else {
            try stdout.writeAll("\nUser: ");
            try stdout.flush();
            const line = (stdin.takeDelimiter('\n') catch null) orelse {
                try stdout.writeAll("\nGoodbye!\n");
                break;
            };
            user_input = std.mem.trim(u8, line, " \t\r\n");
            if (std.ascii.eqlIgnoreCase(user_input, "quit") or std.ascii.eqlIgnoreCase(user_input, "exit")) {
                try stdout.writeAll("Goodbye!\n");
                break;
            }
            if (std.ascii.eqlIgnoreCase(user_input, "clear")) {
                conversation.clearRetainingCapacity();
                try conversation.append(allocator, tokenizer.bosId());
                try stdout.writeAll("Conversation cleared.\n");
                continue;
            }
            if (user_input.len == 0) continue;
        }

        try conversation.append(allocator, user_start);
        const enc = try tokenizer.encode(allocator, user_input);
        defer allocator.free(enc);
        for (enc) |id| try conversation.append(allocator, id);
        try conversation.append(allocator, user_end);
        try conversation.append(allocator, assistant_start);

        var reply: std.ArrayList(usize) = .empty;
        defer reply.deinit(allocator);
        try stdout.writeAll("\nAssistant: ");
        try stdout.flush();
        try engine.generate(conversation.items, params, &reply, stdout);
        try stdout.writeAll("\n");
        try stdout.flush();

        for (reply.items) |id| try conversation.append(allocator, id);
        try conversation.append(allocator, engine.assistant_end);

        if (a.prompt != null) break; // single-shot
    }
}

// ===========================================================================
// Tests
// ===========================================================================

const goldens_dir = "refs/nanochat-goldens";

fn skipUnlessParity() !void {
    if (std.testing.environ.getPosix("NANOCHAT_PARITY") == null) return error.SkipZigTest;
}

fn readFileBytes(allocator: Allocator, io: std.Io, path: []const u8) ![]u8 {
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    if (stat.kind != .file) return error.NotAFile;
    const bytes = try allocator.alloc(u8, @intCast(stat.size));
    errdefer allocator.free(bytes);
    var read_len: usize = 0;
    while (read_len < bytes.len) {
        const nread = try file.readStreaming(io, &.{bytes[read_len..]});
        if (nread == 0) return error.EndOfStream;
        read_len += nread;
    }
    return bytes;
}

const GreedyGolden = struct {
    prompt_ids: []usize,
    out_ids: []usize,
    parsed: std.json.Parsed(std.json.Value),
    allocator: Allocator,

    fn load(allocator: Allocator, io: std.Io, path: []const u8) !GreedyGolden {
        const bytes = try readFileBytes(allocator, io, path);
        defer allocator.free(bytes);
        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, bytes, .{});
        errdefer parsed.deinit();
        const obj = parsed.value.object;
        const prompt_ids = try jsonUsizeArray(allocator, obj.get("prompt_ids").?);
        errdefer allocator.free(prompt_ids);
        const out_ids = try jsonUsizeArray(allocator, obj.get("out_ids").?);
        return .{ .prompt_ids = prompt_ids, .out_ids = out_ids, .parsed = parsed, .allocator = allocator };
    }

    fn deinit(self: *GreedyGolden) void {
        self.allocator.free(self.prompt_ids);
        self.allocator.free(self.out_ids);
        self.parsed.deinit();
        self.* = undefined;
    }
};

fn jsonUsizeArray(allocator: Allocator, v: std.json.Value) ![]usize {
    const items = v.array.items;
    const out = try allocator.alloc(usize, items.len);
    for (out, items) |*d, it| d.* = @intCast(it.integer);
    return out;
}

fn loadModelOrSkip(cfg: Config, ctx: *ExecContext, allocator: Allocator, io: std.Io, path: []const u8) !Model {
    return Model.initFromSafetensors(cfg, ctx, allocator, io, path) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
}

/// Run the engine greedily (temp=0) from a golden's prompt_ids and assert the
/// emitted ids match out_ids exactly. Reports the first divergence index + the
/// argmax logit gap there if any token flips.
fn runGreedyGolden(cfg: Config, ckpt: []const u8, golden_path: []const u8) !void {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var tok = Tokenizer.loadBin(allocator, io, goldens_dir ++ "/tokenizer.bin") catch return error.SkipZigTest;
    defer tok.deinit();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try loadModelOrSkip(cfg, &ctx, allocator, io, ckpt);
    defer model.deinit();

    var golden = GreedyGolden.load(allocator, io, golden_path) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer golden.deinit();

    var engine = Engine.init(&model, &tok, &ctx, allocator);
    defer engine.deinit();

    var out: std.ArrayList(usize) = .empty;
    defer out.deinit(allocator);
    try engine.generate(golden.prompt_ids, .{ .max_tokens = golden.out_ids.len, .temperature = 0 }, &out, null);

    if (!std.mem.eql(usize, out.items, golden.out_ids)) {
        const n = @min(out.items.len, golden.out_ids.len);
        var idx: usize = 0;
        while (idx < n and out.items[idx] == golden.out_ids[idx]) idx += 1;
        std.debug.print(
            "greedy divergence at index {d}: got {d}, want {d} (lens got={d} want={d})\n",
            .{ idx, if (idx < out.items.len) out.items[idx] else 0, if (idx < golden.out_ids.len) golden.out_ids[idx] else 0, out.items.len, golden.out_ids.len },
        );
    }
    try std.testing.expectEqualSlices(usize, golden.out_ids, out.items);
}

test "NANOCHAT_PARITY: engine greedy matches trained-d6 oracle" {
    try skipUnlessParity();
    try runGreedyGolden(Config.d6, goldens_dir ++ "/base_ckpt_d6_step2500.safetensors", goldens_dir ++ "/greedy_trained_d6_step2500.json");
}

test "NANOCHAT_PARITY: engine greedy matches init-d6 oracle" {
    try skipUnlessParity();
    try runGreedyGolden(Config.d6, goldens_dir ++ "/init_d6.safetensors", goldens_dir ++ "/greedy_stream_d6.json");
}

test "NANOCHAT_PARITY: trained-d6 greedy completion prints" {
    try skipUnlessParity();
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var tok = Tokenizer.loadBin(allocator, io, goldens_dir ++ "/tokenizer.bin") catch return error.SkipZigTest;
    defer tok.deinit();
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();
    var model = try loadModelOrSkip(Config.d6, &ctx, allocator, io, goldens_dir ++ "/base_ckpt_d6_step2500.safetensors");
    defer model.deinit();
    var engine = Engine.init(&model, &tok, &ctx, allocator);
    defer engine.deinit();

    const ids32 = try tok.encodeWithBos(allocator, "The capital of France is");
    defer allocator.free(ids32);
    const prompt = try allocator.alloc(usize, ids32.len);
    defer allocator.free(prompt);
    for (prompt, ids32) |*d, s| d.* = s;

    var out: std.ArrayList(usize) = .empty;
    defer out.deinit(allocator);
    try engine.generate(prompt, .{ .max_tokens = 32, .temperature = 0 }, &out, null);

    const out32 = try allocator.alloc(u32, out.items.len);
    defer allocator.free(out32);
    for (out32, out.items) |*d, s| d.* = @intCast(s);
    const text = try tok.decode(allocator, out32);
    defer allocator.free(text);
    std.debug.print("greedy completion: \"The capital of France is{s}\"\n", .{text});
}

test "calculator: pure-arithmetic evaluation" {
    const allocator = std.testing.allocator;
    const cases = [_]struct { expr: []const u8, want: ?[]const u8 }{
        .{ .expr = "2+3*4", .want = "14" },
        .{ .expr = "(2+3)*4", .want = "20" },
        .{ .expr = "-3+5", .want = "2" },
        .{ .expr = "1,000+1", .want = "1001" }, // commas stripped
        .{ .expr = "10/2", .want = "5.0" }, // true division → float
        .{ .expr = "7-2-2", .want = "3" }, // left-assoc
        .{ .expr = "2**3", .want = null }, // power disallowed
        .{ .expr = "1/0", .want = null }, // div by zero
        .{ .expr = "'a'.count('a')", .want = "1" }, // engine.py string-op branch
        .{ .expr = "'hello world'.count('o')", .want = "2" },
        .{ .expr = "'aaaa'.count('aa')", .want = "2" }, // non-overlapping
        .{ .expr = "\"ab\".count('')", .want = "3" }, // python len+1
        .{ .expr = "'__x'.count('_')", .want = null }, // dangerous pattern '__'
        .{ .expr = "'a'.strip()", .want = null }, // only .count() supported
    };
    for (cases) |c| {
        const got = try useCalculator(allocator, c.expr);
        defer if (got) |g| allocator.free(g);
        if (c.want) |w| {
            try std.testing.expect(got != null);
            try std.testing.expectEqualStrings(w, got.?);
        } else {
            try std.testing.expect(got == null);
        }
    }
}

test "calculator: tool state machine enqueues output tokens" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var tok = Tokenizer.loadBin(allocator, io, goldens_dir ++ "/tokenizer.bin") catch return error.SkipZigTest;
    defer tok.deinit();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // The tool state machine is model-independent; only the tokenizer + specials
    // matter, so drive handleTool directly with a synthetic token stream.
    var engine: Engine = .{
        .model = undefined,
        .tokenizer = &tok,
        .ctx = &ctx,
        .allocator = allocator,
        .bos = tok.bosId(),
        .python_start = tok.specialId("<|python_start|>").?,
        .python_end = tok.specialId("<|python_end|>").?,
        .output_start = tok.specialId("<|output_start|>").?,
        .output_end = tok.specialId("<|output_end|>").?,
        .assistant_end = tok.specialId("<|assistant_end|>").?,
        .forced = .empty,
        .forced_head = 0,
        .expr_tokens = .empty,
        .in_python = false,
    };
    defer engine.deinit();

    // Feed: <|python_start|> encode("2+3*4") <|python_end|>
    try engine.handleTool(engine.python_start);
    const expr_ids = try tok.encode(allocator, "2+3*4");
    defer allocator.free(expr_ids);
    for (expr_ids) |id| try engine.handleTool(@intCast(id));
    try engine.handleTool(engine.python_end);

    // Expect forced queue = <|output_start|> + encode("14") + <|output_end|>.
    const result_ids = try tok.encode(allocator, "14");
    defer allocator.free(result_ids);
    var expected: std.ArrayList(usize) = .empty;
    defer expected.deinit(allocator);
    try expected.append(allocator, engine.output_start);
    for (result_ids) |id| try expected.append(allocator, @intCast(id));
    try expected.append(allocator, engine.output_end);

    try std.testing.expectEqualSlices(usize, expected.items, engine.forced.items[engine.forced_head..]);
}

test "NANOCHAT_PARITY: engine greedy == argmax of full forward per position" {
    try skipUnlessParity();
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var tok = Tokenizer.loadBin(allocator, io, goldens_dir ++ "/tokenizer.bin") catch return error.SkipZigTest;
    defer tok.deinit();
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();
    var model = try loadModelOrSkip(Config.d6, &ctx, allocator, io, goldens_dir ++ "/init_d6.safetensors");
    defer model.deinit();

    const prompt = [_]usize{ tok.bosId(), 483, 6987, 285, 7296, 306 };
    const steps = 6;

    // Engine greedy via cache decode.
    var engine = Engine.init(&model, &tok, &ctx, allocator);
    defer engine.deinit();
    var eng_out: std.ArrayList(usize) = .empty;
    defer eng_out.deinit(allocator);
    try engine.generate(&prompt, .{ .max_tokens = steps, .temperature = 0 }, &eng_out, null);

    // Reference: at each step, argmax of the full forward's last-position logits,
    // appending the greedy token and re-forwarding (no cache) through the
    // oracle-gated full-forward path.
    var seq: std.ArrayList(usize) = .empty;
    defer seq.deinit(allocator);
    try seq.appendSlice(allocator, &prompt);
    var ref_out: std.ArrayList(usize) = .empty;
    defer ref_out.deinit(allocator);
    for (0..steps) |_| {
        const scope = ctx.openExecScope();
        const logits = try model.forward(&ctx, seq.items, null);
        var last = try logits.narrow(&ctx, .seq, seq.items.len - 1, 1);
        var idx = try last.argmax(&ctx, .vocab);
        defer idx.deinit(); // i64 indices are caller-owned even under the scope
        const next: usize = @intCast(try idx.item());
        ctx.closeExecScope(scope);
        try seq.append(allocator, next);
        try ref_out.append(allocator, next);
    }

    try std.testing.expectEqualSlices(usize, ref_out.items, eng_out.items);
}

test {
    std.testing.refAllDecls(@This());
}
