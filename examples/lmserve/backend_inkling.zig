//! Inkling backend adapter: the hybrid rel-bias + MoE decoder behind the
//! lmserve `Backend` vtable. Like nanochat, it does NOT ride the generic
//! `GgufChatBackend`/`Conversation` — Inkling has its own KV+conv-state
//! cache and a typed-content-block chat protocol (the wire format) rather
//! than a text template — so this drives the `llm.inkling.chat.Engine`
//! directly on a freshly built prompt each request.
//!
//! Capabilities: reasoning channel (the `<|content_thinking|>` block routes
//! to `reasoning_content` via the standard open/close splitter), and grammar
//! constraints when built with `-Dllguidance=true` (the sampler's
//! `LogitProcessor` seam; a constraint forces reasoning off so the grammar
//! governs the reply from token 0, primed with a `<|content_text|>` block).
//! No client stop-sequences and no cross-request KV reuse in v1.

const std = @import("std");
const fucina = @import("fucina");
const llm = @import("fucina_llm");
const types = @import("types.zig");
const backend_mod = @import("backend.zig");

const Allocator = std.mem.Allocator;
const inkling_chat = llm.inkling.chat;

pub const InklingBackend = struct {
    allocator: Allocator,
    ctx: *fucina.ExecContext,
    model: *llm.inkling.model.Model,
    tokenizer: *llm.tokenizer.Tokenizer,
    engine: inkling_chat.Engine(llm.tokenizer),
    constraints: backend_mod.ConstraintCache,
    model_id: []const u8,
    context_len: usize,
    /// The turn-end id (`<|content_model_end_sampling|>` == the GGUF EOS).
    stop_id: u32,

    pub fn init(
        allocator: Allocator,
        ctx: *fucina.ExecContext,
        model: *llm.inkling.model.Model,
        tokenizer: *llm.tokenizer.Tokenizer,
        model_id: []const u8,
        context_len: usize,
    ) !InklingBackend {
        const engine = try inkling_chat.Engine(llm.tokenizer).init(ctx, model, tokenizer);
        return .{
            .allocator = allocator,
            .ctx = ctx,
            .model = model,
            .tokenizer = tokenizer,
            .engine = engine,
            .constraints = backend_mod.ConstraintCache.init(allocator, 8),
            .model_id = model_id,
            .context_len = context_len,
            .stop_id = engine.markers.end_sampling,
        };
    }

    pub fn deinit(self: *InklingBackend) void {
        self.constraints.deinit();
    }

    pub fn backend(self: *InklingBackend) types.Backend {
        return .{
            .ptr = self,
            .vtable = &.{ .validate = vtValidate, .generate = vtGenerate },
            .info = .{
                .model_id = self.model_id,
                .context_len = self.context_len,
                .caps = .{
                    .grammar = llm.llguidance.enabled,
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
    fn buildPrompt(self: *InklingBackend, a: Allocator, req: *const types.GenerateRequest) ![]usize {
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

    fn vtValidate(ptr: *anyopaque, req: *const types.GenerateRequest) anyerror!void {
        const self: *InklingBackend = @ptrCast(@alignCast(ptr));
        if (req.constraint != null and !llm.llguidance.enabled) return error.LlguidanceNotEnabled;
        const ids = try self.buildPrompt(self.allocator, req);
        defer self.allocator.free(ids);
        if (ids.len >= self.context_len) return error.PromptTooLong;
    }

    fn vtGenerate(ptr: *anyopaque, req: *const types.GenerateRequest, sink: *std.Io.Writer) anyerror!types.GenerateResult {
        const self: *InklingBackend = @ptrCast(@alignCast(ptr));
        const a = self.allocator;

        const ids = try self.buildPrompt(a, req);
        defer a.free(ids);

        // Grammar: clone the cached base per request. The mask forces the
        // turn-end id once the grammar completes, so normal stop handling
        // ends the reply.
        var clone: ?llm.llguidance.Constraint = null;
        defer if (clone) |*c| c.deinit();
        var processor: ?llm.sampler.LogitProcessor = null;
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
