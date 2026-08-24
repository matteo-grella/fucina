//! Inkling serving adapter: the hybrid rel-bias + MoE decoder behind the
//! serving `Backend` vtable. Like nanochat, it does NOT ride the generic
//! `GgufChatBackend`/`Conversation` — Inkling has its own KV+conv-state
//! cache and a typed-content-block chat protocol (the wire format) rather
//! than a text template — so this drives the `models.inkling.chat.Engine`
//! directly on a freshly built prompt each request.
//!
//! Capabilities: reasoning channel (the `<|content_thinking|>` block routes
//! to `reasoning_content` via the standard open/close splitter), and grammar
//! constraints when built with `-Dllguidance=true` (the sampler's
//! `LogitProcessor` seam; a constraint forces reasoning off so the grammar
//! governs the reply from token 0, primed with a `<|content_text|>` block).
//! No client stop-sequences and no cross-request KV reuse.

const std = @import("std");
const fucina = @import("fucina");
const contract = @import("../text/serving/contract.zig");
const gguf_chat = @import("../text/serving/gguf_chat.zig");
const llguidance = @import("../text/llguidance.zig");
const sampler_mod = @import("../text/sampler.zig");
const tokenizer_mod = @import("../text/tokenizer.zig");
const model_mod = @import("model.zig");
const inkling_chat = @import("chat.zig");

const Allocator = std.mem.Allocator;

pub const Backend = struct {
    allocator: Allocator,
    ctx: *fucina.ExecContext,
    model: *model_mod.Model,
    tokenizer: *tokenizer_mod.Tokenizer,
    engine: inkling_chat.Engine(tokenizer_mod),
    constraints: gguf_chat.ConstraintCache,
    model_id: []const u8,
    context_len: usize,
    /// The turn-end id (`<|content_model_end_sampling|>` == the GGUF EOS).
    stop_id: u32,

    pub fn init(
        allocator: Allocator,
        ctx: *fucina.ExecContext,
        model: *model_mod.Model,
        tokenizer: *tokenizer_mod.Tokenizer,
        model_id: []const u8,
        context_len: usize,
    ) !Backend {
        const engine = try inkling_chat.Engine(tokenizer_mod).init(ctx, model, tokenizer);
        return .{
            .allocator = allocator,
            .ctx = ctx,
            .model = model,
            .tokenizer = tokenizer,
            .engine = engine,
            .constraints = gguf_chat.ConstraintCache.init(allocator, 8),
            .model_id = model_id,
            .context_len = context_len,
            .stop_id = engine.markers.end_sampling,
        };
    }

    pub fn deinit(self: *Backend) void {
        self.constraints.deinit();
    }

    pub fn backend(self: *Backend) contract.Backend {
        return .{
            .ptr = self,
            .vtable = &.{ .validate = vtValidate, .generate = vtGenerate },
            .info = .{
                .model_id = self.model_id,
                .context_len = self.context_len,
                .caps = .{
                    .grammar = llguidance.enabled,
                    .think = true,
                    .stop_sequences = false,
                },
                .think_markers = .{
                    .open = inkling_chat.tok_content_thinking,
                    .close = inkling_chat.tok_content_text,
                },
                // The reference reports at effort=0.99; a modest sampling
                // default keeps replies coherent without client overrides.
                .default_sampling = .{ .temperature = 0.7, .top_p = 0.9 },
            },
        };
    }

    /// Render + tokenize the request into a prompt id list. Caller owns it.
    fn buildPrompt(self: *Backend, a: Allocator, req: *const contract.GenerateRequest) ![]usize {
        // A constraint forces reasoning off (the OpenAI layer already sets
        // req.think=false in that case); think_off primes a content_text
        // block so generation is pure constrained content.
        const think_off = !req.think or req.constraint != null;
        const prompt = try inkling_chat.renderPrompt(a, req.messages, .{ .think_off = think_off });
        defer a.free(prompt);
        const ids32 = try self.tokenizer.encodeRaw(a, prompt);
        defer a.free(ids32);
        const ids = try a.alloc(usize, ids32.len);
        errdefer a.free(ids);
        for (ids, ids32) |*d, s| d.* = s;
        return ids;
    }

    fn vtValidate(ptr: *anyopaque, req: *const contract.GenerateRequest) anyerror!void {
        const self: *Backend = @ptrCast(@alignCast(ptr));
        if (req.constraint != null and !llguidance.enabled) return error.LlguidanceNotEnabled;
        const ids = try self.buildPrompt(self.allocator, req);
        defer self.allocator.free(ids);
        if (ids.len >= self.context_len) return error.PromptTooLong;
    }

    fn vtGenerate(ptr: *anyopaque, req: *const contract.GenerateRequest, sink: *std.Io.Writer) anyerror!contract.GenerateResult {
        const self: *Backend = @ptrCast(@alignCast(ptr));
        const a = self.allocator;

        const ids = try self.buildPrompt(a, req);
        defer a.free(ids);

        // Grammar: clone the cached base per request. The mask forces the
        // turn-end id once the grammar completes, so normal stop handling
        // ends the reply.
        var clone: ?llguidance.Constraint = null;
        defer if (clone) |*c| c.deinit();
        var processor: ?sampler_mod.LogitProcessor = null;
        if (req.constraint) |spec| {
            const base = try self.constraints.acquire(self.tokenizer, spec, .{
                .eos_token = self.stop_id,
                .n_vocab = self.model.config.vocab_size,
            });
            clone = try base.clone();
            processor = clone.?.processor();
        }

        const result = try self.engine.generate(ids, .{
            .sampling = req.sampling,
            .processor = processor,
            .max_tokens = req.max_tokens,
            .think_off = !req.think or req.constraint != null,
        }, sink);

        return .{
            .prompt_tokens = result.prompt_tokens,
            .completion_tokens = result.completion_tokens,
            .cached_tokens = 0,
            .finish = if (result.stopped) .stop else .length,
        };
    }
};

/// The served engine box (heap-pinned; `serving.open` dispatches here for
/// the inkling arch). Takes ownership of `file` on every path.
const Box = struct {
    allocator: Allocator,
    model_id: []u8,
    model: model_mod.Model,
    tokenizer: tokenizer_mod.Tokenizer,
    adapter: Backend,

    fn destroy(ptr: *anyopaque) void {
        const box: *Box = @ptrCast(@alignCast(ptr));
        const a = box.allocator;
        box.adapter.deinit();
        box.tokenizer.deinit();
        box.model.deinit();
        a.free(box.model_id);
        a.destroy(box);
    }
};

pub fn openFromFile(
    ctx: *fucina.ExecContext,
    io: std.Io,
    allocator: Allocator,
    file: *fucina.gguf.File,
    model_id: []const u8,
    options: contract.OpenOptions,
    stderr: *std.Io.Writer,
) !contract.Opened {
    _ = io;
    var file_alive = true;
    errdefer if (file_alive) file.deinit();

    const box = try allocator.create(Box);
    errdefer allocator.destroy(box);
    box.allocator = allocator;
    box.model_id = try allocator.dupe(u8, model_id);
    errdefer allocator.free(box.model_id);

    box.tokenizer = tokenizer_mod.Tokenizer.initFromGguf(allocator, file, .{}) catch {
        try stderr.writeAll("this GGUF has no usable tokenizer metadata\n");
        return error.TokenizerUnavailable;
    };
    errdefer box.tokenizer.deinit();

    box.model = try model_mod.Family.load(ctx, file, .{});
    errdefer box.model.deinit();
    file.deinit();
    file_alive = false;

    box.adapter = try Backend.init(allocator, ctx, &box.model, &box.tokenizer, box.model_id, options.context_len);
    return .{ .ptr = box, .destroyFn = Box.destroy, .backend = box.adapter.backend() };
}
