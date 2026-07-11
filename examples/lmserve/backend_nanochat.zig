//! nanochat backend adapter: the example-local GPT (safetensors checkpoint +
//! trained raw-byte BPE tokenizer) behind the lmserve `Backend` vtable. It does
//! not ride `llm.chat.Conversation` — nanochat has its own `Cache`, its own
//! tokenizer, and a special-token chat protocol (chat_cli.py's) instead of a
//! text template — so this adapter builds the token-level prompt directly and
//! drives `nanochat.chat.Engine.generate` (calculator tool loop included).
//!
//! Capability honesty: no grammar constraints (the tokenizer is not
//! llguidance-bridged), no reasoning channel, no stop sequences, and no
//! system role at all (the SFT protocol has none — clients get a 400 rather
//! than a silently dropped instruction).

const std = @import("std");
const fucina = @import("fucina");
const llm = @import("fucina_llm");
const types = @import("types.zig");
const nc_chat = @import("../nanochat/chat.zig");
const nc_model = @import("../nanochat/model.zig");
const nc_tok = @import("../nanochat/tokenizer.zig");

const Allocator = std.mem.Allocator;

pub const NanochatBackend = struct {
    allocator: Allocator,
    ctx: *fucina.ExecContext,
    model: nc_model.Model,
    tokenizer: nc_tok.Tokenizer,
    engine: nc_chat.Engine,
    model_id: []const u8,
    context_len: usize,

    user_start: usize,
    user_end: usize,
    assistant_start: usize,

    /// Load `<dir>/model.safetensors` + `<dir>/tokenizer.bin`. Heap-allocated
    /// because the engine holds pointers into the struct.
    pub fn load(
        allocator: Allocator,
        ctx: *fucina.ExecContext,
        io: std.Io,
        dir: []const u8,
        model_id: []const u8,
        context_len: usize,
    ) !*NanochatBackend {
        var path_buf: [1024]u8 = undefined;
        const self = try allocator.create(NanochatBackend);
        errdefer allocator.destroy(self);

        const tok_path = try std.fmt.bufPrint(&path_buf, "{s}/tokenizer.bin", .{dir});
        self.tokenizer = try nc_tok.Tokenizer.loadBin(allocator, io, tok_path);
        errdefer self.tokenizer.deinit();

        const model_path = try std.fmt.bufPrint(&path_buf, "{s}/model.safetensors", .{dir});
        self.model = try nc_chat.loadModel(ctx, allocator, io, model_path, self.tokenizer.n_vocab);
        errdefer self.model.deinit();

        self.allocator = allocator;
        self.ctx = ctx;
        self.model_id = model_id;
        self.context_len = context_len;
        self.engine = nc_chat.Engine.init(&self.model, &self.tokenizer, ctx, allocator);
        self.user_start = self.tokenizer.specialId("<|user_start|>") orelse return error.MissingSpecialToken;
        self.user_end = self.tokenizer.specialId("<|user_end|>") orelse return error.MissingSpecialToken;
        self.assistant_start = self.tokenizer.specialId("<|assistant_start|>") orelse return error.MissingSpecialToken;
        return self;
    }

    pub fn deinit(self: *NanochatBackend) void {
        self.engine.deinit();
        self.model.deinit();
        self.tokenizer.deinit();
        self.allocator.destroy(self);
    }

    pub fn backend(self: *NanochatBackend) types.Backend {
        return .{
            .ptr = self,
            .vtable = &.{ .validate = vtValidate, .generate = vtGenerate },
            .info = .{
                .model_id = self.model_id,
                .context_len = self.context_len,
                .caps = .{ .grammar = false, .think = false, .stop_sequences = false },
                // nanochat's own GenParams defaults; top_p/penalties do not
                // exist in its sampler.
                .default_sampling = .{ .temperature = 0.6, .top_k = 50, .seed = 42 },
            },
        };
    }

    /// The chat_cli.py conversation-token protocol: bos, then per turn
    /// user_start+enc+user_end / assistant reply tokens+assistant_end,
    /// ending with assistant_start for the reply. Caller owns the slice.
    fn buildPrompt(self: *NanochatBackend, allocator: Allocator, messages: []const llm.chat.Message) ![]usize {
        if (messages.len == 0) return error.EmptyMessages;
        if (messages[messages.len - 1].role == .assistant) return error.TrailingAssistantMessage;

        var tokens: std.ArrayList(usize) = .empty;
        errdefer tokens.deinit(allocator);
        try tokens.append(allocator, self.tokenizer.bosId());
        for (messages) |m| {
            switch (m.role) {
                .system => return error.NoSystemRole,
                .user => {
                    try tokens.append(allocator, self.user_start);
                    try self.appendEncoded(allocator, &tokens, m.content);
                    try tokens.append(allocator, self.user_end);
                },
                .assistant => {
                    try tokens.append(allocator, self.assistant_start);
                    try self.appendEncoded(allocator, &tokens, m.content);
                    try tokens.append(allocator, self.engine.assistant_end);
                },
            }
        }
        try tokens.append(allocator, self.assistant_start);
        return tokens.toOwnedSlice(allocator);
    }

    fn appendEncoded(self: *NanochatBackend, allocator: Allocator, tokens: *std.ArrayList(usize), text: []const u8) !void {
        const ids = try self.tokenizer.encode(allocator, text);
        defer allocator.free(ids);
        for (ids) |id| try tokens.append(allocator, id);
    }

    fn vtValidate(ptr: *anyopaque, req: *const types.GenerateRequest) anyerror!void {
        const self: *NanochatBackend = @ptrCast(@alignCast(ptr));
        const prompt = try self.buildPrompt(self.allocator, req.messages);
        defer self.allocator.free(prompt);
        if (prompt.len >= self.context_len) return error.PromptTooLong;
    }

    fn vtGenerate(ptr: *anyopaque, req: *const types.GenerateRequest, sink: *std.Io.Writer) anyerror!types.GenerateResult {
        const self: *NanochatBackend = @ptrCast(@alignCast(ptr));
        const a = self.allocator;
        const prompt = try self.buildPrompt(a, req.messages);
        defer a.free(prompt);

        const params = nc_chat.GenParams{
            .max_tokens = req.max_tokens,
            .temperature = req.sampling.temperature,
            .top_k = if (req.sampling.top_k == 0) 50 else req.sampling.top_k,
            .seed = req.sampling.seed,
        };
        var out: std.ArrayList(usize) = .empty;
        defer out.deinit(a);
        try self.engine.generate(prompt, params, &out, sink);
        const produced = out.items.len;
        return .{
            .prompt_tokens = prompt.len,
            .completion_tokens = produced,
            .finish = if (produced >= req.max_tokens) .length else .stop,
        };
    }
};
