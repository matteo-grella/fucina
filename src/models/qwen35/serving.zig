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
const gguf_chat = @import("../text/serving/gguf_chat.zig");
const chat = @import("../text/chat.zig");
const llguidance = @import("../text/llguidance.zig");
const sampler_mod = @import("../text/sampler.zig");
const tokenizer_mod = @import("../text/tokenizer.zig");
const model_mod = @import("model.zig");
const qwen35_chat = @import("chat.zig");

const Allocator = std.mem.Allocator;

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

/// The served engine box (heap-pinned; `serving.open` dispatches here for
/// the qwen35/qwen35moe archs). Takes ownership of `file` on every path.
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
    const template = chat.Template.detect(file.getString("tokenizer.chat_template")) orelse
        chat.Template{ .format = model_mod.Family.template_fallback.? };
    // Bonsai GGUFs carry their recommended sampling (`general.sampling.*`).
    const default_sampling = contract.samplingFromGguf(file);

    box.model = try model_mod.Family.load(ctx, file, .{ .moe_stream = options.moe_stream });
    errdefer box.model.deinit();
    file.deinit();
    file_alive = false;

    box.adapter = try Backend.init(
        allocator,
        ctx,
        &box.model,
        &box.tokenizer,
        template,
        box.model_id,
        options.context_len,
        default_sampling,
    );
    return .{
        .ptr = box,
        .destroyFn = Box.destroy,
        .backend = box.adapter.backend(),
        .expert_store = box.model.expert_store,
    };
}
