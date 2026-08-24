//! Qwen3.5/Bonsai serving adapter: the hybrid Gated-DeltaNet family behind
//! the serving `Backend` vtable. Like inkling, it does NOT ride the generic
//! `GgufChatBackend`/`Conversation`: the family's cache carries recurrent
//! conv/state matrices that cannot be truncated back to a token prefix, so
//! the KV-slot reuse tiers do not apply — each request runs on a fresh
//! cache through `models.qwen35.chat.Engine`.
//!
//! Capabilities: reasoning channel (Qwen3.6 prefills the `<think>` opener
//! in the generation prompt, so the adapter injects that opener into the
//! reply stream and the standard open/close splitter routes the rest), and
//! grammar constraints when built with `-Dllguidance=true` (the sampler's
//! `LogitProcessor` seam; a constraint forces reasoning off so the grammar
//! governs the reply from token 0, behind the prefilled empty think block).
//! No client stop-sequences and no cross-request KV reuse.

const std = @import("std");
const fucina = @import("fucina");
const contract = @import("../text/serving/contract.zig");
const adapter_common = @import("../text/serving/adapter_common.zig");
const gguf_chat = @import("../text/serving/gguf_chat.zig");
const chat = @import("../text/chat.zig");
const llguidance = @import("../text/llguidance.zig");
const sampler_mod = @import("../text/sampler.zig");
const tokenizer_mod = @import("../text/tokenizer.zig");
const model_mod = @import("model.zig");
const qwen35_chat = @import("chat.zig");

const Allocator = std.mem.Allocator;

/// Engine-hosted (no `Conversation`): `serving.openFromFile` rejects the
/// cartridge/fleet/KV-tier options for this family.
pub const conversation_hosted = false;

pub const Backend = struct {
    allocator: Allocator,
    ctx: *fucina.ExecContext,
    model: *const model_mod.Model,
    tokenizer: *tokenizer_mod.Tokenizer,
    template: chat.Template,
    engine: qwen35_chat.Engine(tokenizer_mod),
    constraints: gguf_chat.ConstraintCache,
    model_id: []const u8,
    context_len: usize,
    default_sampling: sampler_mod.Config,

    pub fn init(
        allocator: Allocator,
        ctx: *fucina.ExecContext,
        model: *const model_mod.Model,
        tokenizer: *tokenizer_mod.Tokenizer,
        template: chat.Template,
        model_id: []const u8,
        context_len: usize,
        default_sampling: sampler_mod.Config,
    ) !Backend {
        return .{
            .allocator = allocator,
            .ctx = ctx,
            .model = model,
            .tokenizer = tokenizer,
            .template = template,
            .engine = try qwen35_chat.Engine(tokenizer_mod).init(ctx, model, tokenizer),
            .constraints = gguf_chat.ConstraintCache.init(allocator, 8),
            .model_id = model_id,
            .context_len = context_len,
            .default_sampling = default_sampling,
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
                .think_markers = .{ .open = "<think>", .close = "</think>" },
                // Qwen3.5's template keeps Qwen3's hermes tool convention
                // (`<tool_call>` sections; visible in the GGUF template).
                .tool_style = .hermes,
                .default_sampling = self.default_sampling,
            },
        };
    }

    /// Render + tokenize the request into a prompt id list. Caller owns it.
    fn buildPrompt(self: *Backend, a: Allocator, req: *const contract.GenerateRequest) ![]usize {
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

/// `serving.open` wiring for the qwen35/qwen35moe archs: the shared engine
/// box (`adapter_common.openFromFile`) with this family's policies.
const Wiring = struct {
    pub const Adapter = Backend;
    pub const Extra = void;
    pub const template: adapter_common.TemplatePolicy = .detect_or_fallback;
    /// Bonsai GGUFs carry their recommended sampling (`general.sampling.*`).
    pub const sampling: adapter_common.SamplingPolicy = .from_gguf;
    pub const reports_expert_store = true;

    pub fn initAdapter(built: adapter_common.Built(model_mod.Family, Wiring)) !Backend {
        return Backend.init(
            built.allocator,
            built.ctx,
            built.model,
            built.tokenizer,
            built.template.?,
            built.model_id,
            built.options.context_len,
            built.default_sampling,
        );
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
    return adapter_common.openFromFile(model_mod.Family, Wiring, ctx, io, allocator, file, model_id, options, stderr);
}
