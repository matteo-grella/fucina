//! Qwen3-TTS generation loop (qwentts.cpp pipeline-tts.cpp): prompt prefill →
//! per frame { sample codebook-0 from the suppressed codec logits → code
//! predictor fills codebooks 1–15 → next-step embedding = Σ 16 codebook rows
//! + text overlay } → EOS. Emits `[T, 16]` frames; the codec decodes them to
//! audio (buffered or chunk-streamed by the caller).
//!
//! Philox subsequence accounting (sampled runs): frame t draws c0 on
//! subsequence 16t and the predictor on 16t+1 … 16t+15 — byte-exact replay
//! of the reference (and of PyTorch CUDA multinomial) for a given seed.

const std = @import("std");
const fucina = @import("fucina");
const model_mod = @import("model.zig");
const prompt_mod = @import("prompt.zig");
const sampling = @import("sampling.zig");

const ExecContext = fucina.ExecContext;
const Allocator = std.mem.Allocator;
const Rows = fucina.Tensor(.{ .seq, .embed });

pub const Params = struct {
    max_new_tokens: usize = 2048,
    seed: i64 = 0,
    /// Greedy on both stacks (temperature 0): the parity mode.
    greedy: bool = false,
    talker: sampling.Params = .{},
    predictor: sampling.Params = .{ .repetition_penalty = 1.0 },
};

pub const Taps = struct {
    allocator: Allocator,
    prefill_hidden: ?*model_mod.StackTaps = null,
    logits_prefill: ?[]f32 = null,
    next_emb_step0: ?[]f32 = null,

    pub fn deinit(self: *Taps) void {
        if (self.logits_prefill) |b| self.allocator.free(b);
        if (self.next_emb_step0) |b| self.allocator.free(b);
        self.* = undefined;
    }
};

pub const Result = struct {
    allocator: Allocator,
    /// `[frames][16]` codec codes, frame-major.
    codes: []i32,
    frames: usize,

    pub fn deinit(self: *Result) void {
        self.allocator.free(self.codes);
        self.* = undefined;
    }
};

/// Per-frame callback for streaming: called with the 16 codes of each frame
/// as it is produced. Return false to stop generation early.
pub const OnFrame = *const fn (user: ?*anyopaque, frame: []const i32) bool;

/// Caller-owned KV pair for `generate` — allocate once, reuse across calls
/// (each call resets both; a long-lived agent avoids per-turn KV churn).
pub const Kvs = struct {
    talker: model_mod.Kv,
    predictor: model_mod.Kv,

    pub fn init(allocator: Allocator, model: *const model_mod.Model) !Kvs {
        var talker = try model.talker.newKv(allocator);
        errdefer talker.deinit();
        const predictor = try model.predictor.newKv(allocator);
        return .{ .talker = talker, .predictor = predictor };
    }

    pub fn deinit(self: *Kvs) void {
        self.talker.deinit();
        self.predictor.deinit();
        self.* = undefined;
    }
};

pub fn generate(
    ctx: *ExecContext,
    model: *const model_mod.Model,
    prompt: *const prompt_mod.Output,
    params: Params,
    kvs: *Kvs,
    on_frame: ?OnFrame,
    on_frame_user: ?*anyopaque,
    taps: ?*Taps,
) !Result {
    const allocator = ctx.allocator();
    const cfg = &model.talker.cfg;
    const sp = &model.specials;
    const hidden = cfg.hidden;
    const vocab = cfg.vocab;
    const groups = sp.num_code_groups;

    const kv = &kvs.talker;
    kv.reset();
    const pred_kv = &kvs.predictor;

    // Prefill.
    var prompt_rows = try Rows.fromBorrowedConstSlice(ctx, .{ prompt.t_ctx, hidden }, prompt.input_embed);
    defer prompt_rows.deinit();
    var hidden_all: ?Rows = try model.talker.forward(ctx, kv, &prompt_rows, if (taps) |tp| tp.prefill_hidden else null);
    defer if (hidden_all) |*ha| ha.deinit();

    const scratch = try allocator.alloc(f32, 2 * vocab);
    defer allocator.free(scratch);
    const logits_buf = try allocator.alloc(f32, vocab);
    defer allocator.free(logits_buf);
    const hidden_last = try allocator.alloc(f32, hidden);
    defer allocator.free(hidden_last);
    const next_emb = try allocator.alloc(f32, hidden);
    defer allocator.free(next_emb);

    var history: std.ArrayList(i32) = .empty;
    defer history.deinit(allocator);

    var codes: std.ArrayList(i32) = .empty;
    errdefer codes.deinit(allocator);
    var frame_codes = try allocator.alloc(i32, groups);
    defer allocator.free(frame_codes);

    const talker_params: sampling.Params = if (params.greedy) .{ .temperature = 0 } else params.talker;
    const pred_params: sampling.Params = if (params.greedy) .{ .temperature = 0, .repetition_penalty = 1.0 } else params.predictor;

    var subseq: u64 = 0;
    var step: usize = 0;
    while (step < params.max_new_tokens) : (step += 1) {
        // Last-row hidden + codec logits.
        {
            const ha = &hidden_all.?;
            var last = try ha.narrow(ctx, .seq, ha.dim(.seq) - 1, 1);
            defer last.deinit();
            @memcpy(hidden_last, try last.dataConst());
            var logits_t = try model.codec_head.linearSeq(ctx, &last, .embed, .vocab);
            defer logits_t.deinit();
            @memcpy(logits_buf, try logits_t.dataConst());
        }
        hidden_all.?.deinit();
        hidden_all = null;
        if (taps) |tp| {
            if (step == 0 and tp.logits_prefill == null) tp.logits_prefill = try allocator.dupe(f32, logits_buf);
        }

        // Suppress the control block except EOS, then sample c0.
        sampling.applySuppress(logits_buf, vocab - 1024, vocab, sp.codec_eos);
        const s = sampling.sample(logits_buf, talker_params, history.items, params.seed, subseq, scratch);
        subseq += 1;
        const c0 = s.id;
        if (c0 == sp.codec_eos) break;
        try history.append(allocator, @intCast(c0));
        frame_codes[0] = @intCast(c0);

        // Code predictor: codebooks 1..15, subsequences subseq..subseq+14.
        const PredSampler = struct {
            params: sampling.Params,
            seed: i64,
            base: u64,
            scratch: []f32,
            buf: []f32,
            pub fn sample(self: *const @This(), g: usize, logits: []const f32) usize {
                @memcpy(self.buf[0..logits.len], logits);
                const r = sampling.sample(self.buf[0..logits.len], self.params, &.{}, self.seed, self.base + g, self.scratch);
                return r.id;
            }
        };
        const pred_scratch = try allocator.alloc(f32, 3 * model.predictor.cfg.vocab);
        defer allocator.free(pred_scratch);
        const ps = PredSampler{
            .params = pred_params,
            .seed = params.seed,
            .base = subseq,
            .scratch = pred_scratch[model.predictor.cfg.vocab..],
            .buf = pred_scratch[0..model.predictor.cfg.vocab],
        };
        try model.predictFrame(ctx, pred_kv, hidden_last, c0, frame_codes, &ps);
        subseq += @intCast(groups - 1);

        try codes.appendSlice(allocator, frame_codes);
        if (on_frame) |cb| {
            if (!cb(on_frame_user, frame_codes)) break;
        }

        // Next-step embedding: 16 codebook rows + text overlay.
        @memcpy(next_emb, model.codecRow(c0));
        for (0..groups - 1) |g| {
            for (next_emb, model.predRow(g, @intCast(frame_codes[g + 1]))) |*d, r| d.* += r;
        }
        const overlay = if (step < prompt.t_trailing)
            prompt.trailing_text_hidden[step * hidden ..][0..hidden]
        else
            prompt.tts_pad_embed;
        for (next_emb, overlay) |*d, o| d.* += o;
        if (taps) |tp| {
            if (step == 0 and tp.next_emb_step0 == null) tp.next_emb_step0 = try allocator.dupe(f32, next_emb);
        }

        var step_rows = try Rows.fromBorrowedConstSlice(ctx, .{ 1, hidden }, next_emb);
        defer step_rows.deinit();
        hidden_all = try model.talker.forward(ctx, kv, &step_rows, null);
    }

    const frames = codes.items.len / groups;
    return .{ .allocator = allocator, .codes = try codes.toOwnedSlice(allocator), .frames = frames };
}
