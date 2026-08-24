//! Shared core of the cartridge and cartridge-fleet trainers: chat
//! templates, duck-typed KV-cache construction, the synthesized-conversation
//! element, batched lockstep generation, generation cleanup, tokenizer id
//! conversion, flag parsing, and wall-clock helpers.

const std = @import("std");
const fucina = @import("fucina");
const models = @import("fucina_models");

const cartridge = models.text.cartridge;

// ChatML blocks (Qwen3). The assistant opener carries the empty think block
// the non-thinking chat template emits, so generations answer directly.
/// Chat-template strings the serving/self-study prompt builders splice
/// (runtime values so one engine serves several architectures).
pub const Tpl = struct {
    sys_open: []const u8,
    user_open: []const u8,
    asst_open: []const u8,
    block_close: []const u8,
    stop_marker: []const u8,
};

pub const qwen3_tpl = Tpl{
    .sys_open = "<|im_start|>system\n",
    .user_open = "<|im_start|>user\n",
    .asst_open = "<|im_start|>assistant\n<think>\n\n</think>\n\n",
    .block_close = "<|im_end|>\n",
    .stop_marker = "<|im_end|>",
};

// Gemma 4's `<|turn>` format (src/models/chat.zig Template.gemma4, thinking
// primed off).
pub const gemma4_tpl = Tpl{
    .sys_open = "<bos><|turn>system\n",
    .user_open = "<|turn>user\n",
    .asst_open = "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
    .block_close = "<turn|>\n",
    .stop_marker = "<turn|>",
};

/// Duck-typed per-model KV-cache construction: uniform-geometry models
/// (qwen3) size from config, per-layer-geometry models (gemma4) from geom.
pub fn makeCache(ctx: *fucina.ExecContext, model: anytype, capacity: usize) !models.text.kv_cache.KvCache {
    const M = @TypeOf(model.*);
    if (comptime @hasField(M, "geom")) {
        return models.text.kv_cache.KvCache.initPerLayer(ctx, model.geom.kv_heads, model.geom.head_dim, capacity);
    }
    const cfg = model.config;
    return models.text.kv_cache.KvCache.init(ctx, cfg.num_layers, cfg.num_key_value_heads, cfg.head_dim, capacity);
}

/// Reference synthesizers/self_study.py SYSTEM_PROMPT_TEMPLATE.
pub const system_prompt_template = "\nYou are in a conversation about the following user information.\n\n<info>\n{s}\n</info>";

/// One synthesized conversation, ready for a distillation micro-step:
/// `student_ids` is the packed seed-free conversation (user question +
/// assistant answer) and `builder` holds the teacher's sparse top-k targets
/// for the answer tokens.
pub const Convo = struct {
    /// Seed-free packed element: user(question) + assistant(answer).
    student_ids: []usize,
    /// The same behind the real chunk: system(chunk) + student_ids.
    teacher_ids: []usize,
    /// Student index of the first supervised (answer) token.
    first_answer: usize,
    /// Supervised token count (turn close included when B stopped).
    answer_len: usize,
    question: []u8,
    /// Teacher top-k targets at STUDENT-LOCAL positions (offset by the
    /// segment start when packing conversations into one row).
    builder: cartridge.TargetsBuilder,

    pub fn deinit(self: *Convo, allocator: std.mem.Allocator) void {
        allocator.free(self.student_ids);
        allocator.free(self.teacher_ids);
        allocator.free(self.question);
        self.builder.deinit();
        self.* = undefined;
    }
};

/// Batched lockstep generation: per-stream prefill, then ONE
/// `forwardStepBatch` weight pass per token across every still-active
/// stream (finished streams retire from the batch, so late stoppers never
/// pay for early ones). Caches are reset here; the stop token is dropped
/// from the returned ids; results live on `allocator`.
pub fn generateIdsBatch(
    ctx: *fucina.ExecContext,
    allocator: std.mem.Allocator,
    model: anytype,
    caches: []models.text.kv_cache.KvCache,
    prompts: []const []const usize,
    max_new: usize,
    stop_id: ?usize,
    cfgs: []const models.text.sampler.Config,
) ![][]usize {
    const n = prompts.len;
    std.debug.assert(n > 0 and caches.len >= n and cfgs.len == n and max_new > 0);

    const outs = try allocator.alloc([]usize, n);
    errdefer allocator.free(outs);
    var built: usize = 0;
    errdefer for (outs[0..built]) |buf| allocator.free(buf);
    for (outs) |*buf| {
        buf.* = try allocator.alloc(usize, max_new);
        built += 1;
    }

    const samplers = try allocator.alloc(models.text.sampler.Sampler, n);
    defer allocator.free(samplers);
    const lens = try allocator.alloc(usize, n);
    defer allocator.free(lens);
    const active = try allocator.alloc(usize, n);
    defer allocator.free(active);
    const batch_caches = try allocator.alloc(*models.text.kv_cache.KvCache, n);
    defer allocator.free(batch_caches);
    const batch_tokens = try allocator.alloc(usize, n);
    defer allocator.free(batch_tokens);

    var n_active: usize = 0;
    for (0..n) |i| {
        if (prompts[i].len + max_new > caches[i].capacity) return error.PromptTooLong;
        caches[i].reset();
        samplers[i] = models.text.sampler.Sampler.init(cfgs[i]);
        lens[i] = 0;
        var logits = try model.forwardStep(ctx, &caches[i], prompts[i], 0);
        defer logits.deinit();
        const next = try samplers[i].next(ctx, &logits, outs[i][0..0]);
        if (stop_id != null and next == stop_id.?) continue;
        outs[i][0] = next;
        lens[i] = 1;
        if (max_new > 1) {
            active[n_active] = i;
            n_active += 1;
        }
    }

    while (n_active > 0) {
        for (0..n_active) |j| {
            const i = active[j];
            batch_caches[j] = &caches[i];
            batch_tokens[j] = outs[i][lens[i] - 1];
        }
        var logits = try model.forwardStepBatch(ctx, batch_caches[0..n_active], batch_tokens[0..n_active]);
        defer logits.deinit();

        var kept: usize = 0;
        for (0..n_active) |j| {
            const i = active[j];
            var row = try logits.narrow(ctx, .seq, j, 1);
            defer row.deinit();
            const next = try samplers[i].next(ctx, &row, outs[i][0..lens[i]]);
            if (stop_id != null and next == stop_id.?) continue;
            outs[i][lens[i]] = next;
            lens[i] += 1;
            if (lens[i] < max_new) {
                active[kept] = i;
                kept += 1;
            }
        }
        n_active = kept;
    }

    for (outs, lens) |*buf, len| buf.* = try allocator.realloc(buf.*, len);
    return outs;
}

/// Decode + trim a generation, dropping any leading think block/marker
/// (small models occasionally re-emit them despite the empty think block
/// in the assistant opener).
pub fn cleanGenerated(allocator: std.mem.Allocator, tokenizer: anytype, ids: []const usize) ![]u8 {
    const text = try decodeUsize(allocator, tokenizer, ids);
    defer allocator.free(text);
    var content = std.mem.trim(u8, text, " \t\r\n");
    if (std.mem.startsWith(u8, content, "<think>")) {
        if (std.mem.indexOf(u8, content, "</think>")) |end| content = content[end + "</think>".len ..];
    } else if (std.mem.startsWith(u8, content, "</think>")) {
        content = content["</think>".len..];
    }
    content = std.mem.trim(u8, content, " \t\r\n");
    if (content.len == 0) return error.EmptyGeneration;
    return allocator.dupe(u8, content);
}

pub fn encodeUsize(allocator: std.mem.Allocator, tokenizer: anytype, text: []const u8) ![]usize {
    const ids32 = try tokenizer.encode(allocator, text);
    defer allocator.free(ids32);
    const out = try allocator.alloc(usize, ids32.len);
    for (out, ids32) |*dst, id| dst.* = id;
    return out;
}

pub fn decodeUsize(allocator: std.mem.Allocator, tokenizer: anytype, ids: []const usize) ![]u8 {
    const ids32 = try allocator.alloc(u32, ids.len);
    defer allocator.free(ids32);
    for (ids32, ids) |*dst, id| dst.* = @intCast(id);
    return tokenizer.decode(allocator, ids32);
}

pub fn parseFlagStr(args: []const []const u8, arg_i: *usize, comptime flag: []const u8) !?[]const u8 {
    const arg = args[arg_i.*];
    if (std.mem.eql(u8, arg, flag)) {
        arg_i.* += 1;
        if (arg_i.* >= args.len) return error.MissingFlagValue;
        return args[arg_i.*];
    }
    if (std.mem.startsWith(u8, arg, flag ++ "=")) return arg[flag.len + 1 ..];
    return null;
}

pub fn parseFlagInt(args: []const []const u8, arg_i: *usize, comptime flag: []const u8) !?usize {
    const text = (try parseFlagStr(args, arg_i, flag)) orelse return null;
    return try std.fmt.parseInt(usize, text, 10);
}

pub fn parseFlagF32(args: []const []const u8, arg_i: *usize, comptime flag: []const u8) !?f32 {
    const text = (try parseFlagStr(args, arg_i, flag)) orelse return null;
    return try std.fmt.parseFloat(f32, text);
}

pub fn nowNs(io: std.Io) i128 {
    return std.Io.Clock.real.now(io).nanoseconds;
}

pub fn seconds(ns: i128) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e9;
}
