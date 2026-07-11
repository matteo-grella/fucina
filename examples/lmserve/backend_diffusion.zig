//! DiffusionGemma backend adapter. Not autoregressive: the model denoises
//! 256-token canvases and commits them block-by-block, so the server streams
//! per COMMITTED BLOCK (the `on_block` callback), not per token — the
//! streaming wire format is identical, deltas just arrive in bigger pieces.
//! The message history renders through the shared Gemma 4 template
//! (`chat.Template.renderMessages`, byte-identical to the CLI's turn
//! rendering) with the thought channel always primed off.
//!
//! Capability honesty: no grammar constraints (token masking has no seam in
//! the denoiser), no reasoning channel, no stop sequences (EOG trimming
//! happens inside block finalization). Sampling maps only `seed` — the
//! denoiser's temperature schedule comes from the model's own
//! entropy-bound config.

const std = @import("std");
const fucina = @import("fucina");
const llm = @import("fucina_llm");
const types = @import("types.zig");

const Allocator = std.mem.Allocator;
const dg = llm.diffusion_gemma.model;

pub const DiffusionBackend = struct {
    allocator: Allocator,
    ctx: *fucina.ExecContext,
    model: *const dg.Model,
    tokenizer: *const llm.spm_tokenizer.Tokenizer,
    template: llm.chat.Template,
    model_id: []const u8,
    context_len: usize,

    pub fn backend(self: *DiffusionBackend) types.Backend {
        return .{
            .ptr = self,
            .vtable = &.{ .validate = vtValidate, .generate = vtGenerate },
            .info = .{
                .model_id = self.model_id,
                .context_len = self.context_len,
                .caps = .{ .grammar = false, .think = false, .stop_sequences = false },
                .default_sampling = .{ .seed = 42 },
            },
        };
    }

    fn render(self: *const DiffusionBackend, allocator: Allocator, req: *const types.GenerateRequest) ![]u8 {
        var buf: std.ArrayList(u8) = .empty;
        errdefer buf.deinit(allocator);
        // Thought channel always primed off: the denoiser has no reasoning
        // toggle in this adapter (caps.think = false).
        try self.template.renderMessages(allocator, &buf, req.messages, true);
        return buf.toOwnedSlice(allocator);
    }

    fn encode(self: *const DiffusionBackend, allocator: Allocator, text: []const u8) ![]usize {
        const ids32 = try self.tokenizer.encodeRaw(allocator, text);
        defer allocator.free(ids32);
        const ids = try allocator.alloc(usize, ids32.len);
        for (ids, ids32) |*d, s| d.* = s;
        return ids;
    }

    fn vtValidate(ptr: *anyopaque, req: *const types.GenerateRequest) anyerror!void {
        const self: *DiffusionBackend = @ptrCast(@alignCast(ptr));
        const a = self.allocator;
        const rendered = try self.render(a, req);
        defer a.free(rendered);
        const prompt = try self.encode(a, rendered);
        defer a.free(prompt);
        if (prompt.len >= self.context_len) return error.PromptTooLong;
    }

    /// Streams committed-block text into the lmserve sink; a write failure
    /// (client gone, cancel) is recorded and re-raised after generation —
    /// `on_block` cannot abort the denoiser mid-run.
    const BlockSink = struct {
        backend: *DiffusionBackend,
        sink: *std.Io.Writer,
        text: std.ArrayList(u8) = .empty,
        failed: ?anyerror = null,

        fn onBlock(user: ?*anyopaque, block_index: usize, kept: []const usize, finished: bool) void {
            _ = block_index;
            _ = finished;
            const bs: *BlockSink = @ptrCast(@alignCast(user.?));
            if (bs.failed != null) return;
            bs.emit(kept) catch |err| {
                bs.failed = err;
            };
        }

        fn emit(bs: *BlockSink, kept: []const usize) !void {
            const a = bs.backend.allocator;
            bs.text.clearRetainingCapacity();
            for (kept) |id| try bs.backend.tokenizer.decodeAppend(a, @intCast(id), &bs.text);
            try bs.sink.writeAll(bs.text.items);
            try bs.sink.flush();
        }
    };

    fn vtGenerate(ptr: *anyopaque, req: *const types.GenerateRequest, sink: *std.Io.Writer) anyerror!types.GenerateResult {
        const self: *DiffusionBackend = @ptrCast(@alignCast(ptr));
        const a = self.allocator;

        const rendered = try self.render(a, req);
        defer a.free(rendered);
        const prompt = try self.encode(a, rendered);
        defer a.free(prompt);

        const c_len = self.model.config.canvas_length;
        const blocks = (req.max_tokens + c_len - 1) / c_len;
        var kv = try self.model.initKvCache(self.ctx, prompt.len + (blocks + 1) * c_len);
        defer kv.deinit();

        const out = try a.alloc(usize, req.max_tokens);
        defer a.free(out);

        var block_sink = BlockSink{ .backend = self, .sink = sink };
        defer block_sink.text.deinit(a);

        const result = try dg.generate(self.model, self.ctx, &kv, prompt, out, .{
            .denoise = .{
                .eb = self.model.config.eb,
                .seed = req.sampling.seed,
                .self_conditioning = true,
            },
            .max_new_tokens = req.max_tokens,
            .on_block = BlockSink.onBlock,
            .on_block_user = @ptrCast(&block_sink),
        });
        if (block_sink.failed) |err| return err;

        return .{
            .prompt_tokens = prompt.len,
            .completion_tokens = result.produced,
            .finish = if (result.produced >= req.max_tokens) .length else .stop,
        };
    }
};
