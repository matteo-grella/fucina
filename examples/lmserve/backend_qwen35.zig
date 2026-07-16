//! Qwen3.5/Bonsai backend adapter: the hybrid Gated-DeltaNet family behind
//! the lmserve `Backend` vtable. Like inkling, it does NOT ride the generic
//! `GgufChatBackend`/`Conversation`: the family's cache carries recurrent
//! conv/state matrices that cannot be truncated back to a token prefix, so
//! the KV-slot reuse tiers do not apply — each request runs on a fresh
//! cache through `llm.qwen35.chat.Engine`.
//!
//! Capabilities: reasoning channel (Qwen3.6 prefills the `<think>` opener
//! in the generation prompt, so the adapter injects that opener into the
//! reply stream and the standard open/close splitter routes the rest), and
//! grammar constraints when built with `-Dllguidance=true` (the sampler's
//! `LogitProcessor` seam; a constraint forces reasoning off so the grammar
//! governs the reply from token 0, behind the prefilled empty think block).
//! No client stop-sequences and no cross-request KV reuse in v1.

const std = @import("std");
const fucina = @import("fucina");
const llm = @import("fucina_llm");
const types = @import("types.zig");
const backend_mod = @import("backend.zig");

const Allocator = std.mem.Allocator;
const qwen35_chat = llm.qwen35.chat;

pub const Qwen35Backend = struct {
    allocator: Allocator,
    ctx: *fucina.ExecContext,
    model: *const llm.qwen35.model.Model,
    tokenizer: *llm.tokenizer.Tokenizer,
    template: llm.chat.Template,
    engine: qwen35_chat.Engine(llm.tokenizer),
    constraints: backend_mod.ConstraintCache,
    model_id: []const u8,
    context_len: usize,
    default_sampling: llm.sampler.Config,

    pub fn init(
        allocator: Allocator,
        ctx: *fucina.ExecContext,
        model: *const llm.qwen35.model.Model,
        tokenizer: *llm.tokenizer.Tokenizer,
        template: llm.chat.Template,
        model_id: []const u8,
        context_len: usize,
        default_sampling: llm.sampler.Config,
    ) !Qwen35Backend {
        return .{
            .allocator = allocator,
            .ctx = ctx,
            .model = model,
            .tokenizer = tokenizer,
            .template = template,
            .engine = try qwen35_chat.Engine(llm.tokenizer).init(ctx, model, tokenizer),
            .constraints = backend_mod.ConstraintCache.init(allocator, 8),
            .model_id = model_id,
            .context_len = context_len,
            .default_sampling = default_sampling,
        };
    }

    pub fn deinit(self: *Qwen35Backend) void {
        self.constraints.deinit();
    }

    pub fn backend(self: *Qwen35Backend) types.Backend {
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
                .think_markers = .{ .open = "<think>", .close = "</think>" },
                .default_sampling = self.default_sampling,
            },
        };
    }

    /// Render + tokenize the request into a prompt id list. Caller owns it.
    fn buildPrompt(self: *Qwen35Backend, a: Allocator, req: *const types.GenerateRequest) ![]usize {
        // The OpenAI layer forces think off under a constraint, so the
        // grammar governs from token 0 behind the empty think block.
        const rendered = try qwen35_chat.renderPrompt(a, self.template, req.messages, .{
            .think_off = !req.think,
        });
        defer a.free(rendered);
        const ids32 = try self.tokenizer.encodeRaw(a, rendered);
        defer a.free(ids32);
        const ids = try a.alloc(usize, ids32.len);
        errdefer a.free(ids);
        for (ids, ids32) |*d, s| d.* = s;
        return ids;
    }

    fn vtValidate(ptr: *anyopaque, req: *const types.GenerateRequest) anyerror!void {
        const self: *Qwen35Backend = @ptrCast(@alignCast(ptr));
        if (req.constraint != null and !llm.llguidance.enabled) return error.LlguidanceNotEnabled;
        const ids = try self.buildPrompt(self.allocator, req);
        defer self.allocator.free(ids);
        if (ids.len >= self.context_len) return error.PromptTooLong;
    }

    fn vtGenerate(ptr: *anyopaque, req: *const types.GenerateRequest, sink: *std.Io.Writer) anyerror!types.GenerateResult {
        const self: *Qwen35Backend = @ptrCast(@alignCast(ptr));
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
                .eos_token = self.engine.stop_id,
                .n_vocab = self.model.config.vocab_size,
            });
            clone = try base.clone();
            processor = clone.?.processor();
        }

        // Thinking on: the prompt ends inside a prefilled think block, so
        // the model never emits the opener itself — inject it into the
        // stream for the reasoning splitter.
        if (req.think) {
            try sink.writeAll(qwen35_chat.think_opener);
            try sink.flush();
        }

        const result = try self.engine.generate(ids, .{
            .sampling = req.sampling,
            .processor = processor,
            .max_tokens = req.max_tokens,
            .capacity = self.context_len,
        }, sink);

        return .{
            .prompt_tokens = result.prompt_tokens,
            .completion_tokens = result.completion_tokens,
            .cached_tokens = 0,
            .finish = if (result.stopped) .stop else .length,
        };
    }
};
