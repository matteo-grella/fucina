//! Token sampling for autoregressive decoding.
//!
//! Greedy (argmax) is deterministic and right for benchmarking, but it makes
//! instruction-tuned models like Qwen3 degenerate into repetition. This sampler
//! implements the usual quality pipeline — repetition / frequency / presence
//! penalties, temperature, top-k, top-p (nucleus), and min-p — over the model's
//! logits, matching llama.cpp's parameter set and defaults. With
//! `temperature <= 0` it falls back to greedy, so the benchmark path is
//! unchanged.

const std = @import("std");
const fucina = @import("fucina");
const logit_processor = @import("logit_processor.zig");

const ExecContext = fucina.ExecContext;
const Logits = fucina.Tensor(.{ .seq, .vocab });
pub const LogitProcessor = logit_processor.LogitProcessor;

/// Largest candidate set considered when sampling. Qwen3 uses top_k=20; this
/// caps the work (and the stack buffer) when top_k is disabled.
const max_candidates = 256;

pub const Config = struct {
    /// <= 0 selects greedy (argmax); otherwise scales the logits before softmax.
    temperature: f32 = 0,
    /// Keep only the top-k logits (0 = use `max_candidates`).
    top_k: usize = 0,
    /// Nucleus: keep the smallest prefix whose cumulative probability >= top_p.
    top_p: f32 = 1.0,
    /// Keep tokens with probability >= min_p * p(best). 0 disables.
    min_p: f32 = 0,
    /// Divide (logit>0) / multiply (logit<=0) recent tokens' logits by this
    /// (llama.cpp `penalty_repeat`). 1.0 disables. Applied once per unique token.
    repeat_penalty: f32 = 1.0,
    /// Subtract `freq_penalty * count` from each recent token's logit, where
    /// `count` is its occurrences in the window (llama.cpp `penalty_freq`).
    freq_penalty: f32 = 0,
    /// Subtract `presence_penalty` from any token present in the window
    /// (llama.cpp `penalty_present`). 0 disables.
    presence_penalty: f32 = 0,
    /// How many of the most recent tokens the penalties apply to.
    repeat_last_n: usize = 64,
    seed: u64 = 0,

    pub fn isGreedy(self: Config) bool {
        return self.temperature <= 0;
    }
};

pub const Sampler = struct {
    config: Config,
    prng: std.Random.DefaultPrng,
    /// Optional logit processor (grammar mask, bias list, …): its `process`
    /// mutates the row before the penalty/sampling pipeline, its `commit`
    /// observes the selected token — on every path, greedy included. Set
    /// after `init`; `logit_processor.zig` documents the contract (incl. why
    /// this seam composes with speculative decoding).
    processor: ?LogitProcessor = null,

    pub fn init(config: Config) Sampler {
        return .{ .config = config, .prng = std.Random.DefaultPrng.init(config.seed) };
    }

    /// Pick the next token from `logits` (shape `[1, vocab]`). `history` is the
    /// tokens so far (prompt + generated), used by the repetition penalty. The
    /// logits are mutated in place by the penalty (and by the processor, when
    /// one is set). A processor that leaves no selectable token is
    /// `error.AllTokensMasked` — a broken constraint fails loudly instead of
    /// silently sampling from masked-out logits.
    pub fn next(self: *Sampler, ctx: *ExecContext, logits: *Logits, history: []const usize) !usize {
        const cfg = self.config;

        if (self.processor) |p| try p.process(try logits.data(), history);

        if ((cfg.repeat_penalty != 1.0 or cfg.freq_penalty != 0 or cfg.presence_penalty != 0) and history.len > 0) {
            const data = try logits.data();
            const start = if (history.len > cfg.repeat_last_n) history.len - cfg.repeat_last_n else 0;
            const window = history[start..];
            // Apply each penalty once per unique token, with `count` = its
            // occurrences in the window (matches llama.cpp `llama_sampler_penalties`).
            for (window, 0..) |tok, i| {
                if (tok >= data.len) continue;
                var seen = false;
                for (window[0..i]) |prev| if (prev == tok) {
                    seen = true;
                    break;
                };
                if (seen) continue;
                var count: f32 = 0;
                for (window) |t| {
                    if (t == tok) count += 1;
                }
                var l = data[tok];
                l = if (l > 0) l / cfg.repeat_penalty else l * cfg.repeat_penalty;
                l -= count * cfg.freq_penalty + cfg.presence_penalty;
                data[tok] = l;
            }
        }

        if (cfg.isGreedy()) {
            const chosen = try argmax(ctx, logits);
            if (self.processor) |p| {
                if (!std.math.isFinite((try logits.dataConst())[chosen])) return error.AllTokensMasked;
                try p.commit(chosen);
            }
            return chosen;
        }

        const vocab = logits.dim(.vocab);
        const k: usize = @min(if (cfg.top_k > 0) cfg.top_k else max_candidates, @min(@as(usize, max_candidates), vocab));
        var top = try logits.topK(ctx, .vocab, k, .top);
        defer top.deinit();
        const vals = try top.values.dataConst(); // logits, descending
        const idxs = try top.indices.dataConst(); // token ids as f32

        // temperature + softmax over the candidates (vals[0] is the max).
        var probs: [max_candidates]f32 = undefined;
        const inv_temp = 1.0 / cfg.temperature;
        const max_logit = vals[0];
        if (self.processor != null and !std.math.isFinite(max_logit)) return error.AllTokensMasked;
        var sum: f32 = 0;
        for (0..k) |i| {
            probs[i] = @exp((vals[i] - max_logit) * inv_temp);
            sum += probs[i];
        }
        for (0..k) |i| probs[i] /= sum;

        // top-p: smallest descending prefix reaching cumulative top_p.
        var keep = k;
        if (cfg.top_p < 1.0) {
            var cum: f32 = 0;
            for (0..k) |i| {
                cum += probs[i];
                if (cum >= cfg.top_p) {
                    keep = i + 1;
                    break;
                }
            }
        }
        // min-p: drop tokens far below the best.
        if (cfg.min_p > 0) {
            const threshold = cfg.min_p * probs[0];
            var m = keep;
            for (0..keep) |i| {
                if (probs[i] < threshold) {
                    m = i;
                    break;
                }
            }
            keep = @max(m, 1);
        }

        // sample from the kept prefix.
        var kept_sum: f32 = 0;
        for (0..keep) |i| kept_sum += probs[i];
        const r = self.prng.random().float(f32) * kept_sum;
        var acc: f32 = 0;
        var chosen: usize = keep - 1;
        for (0..keep) |i| {
            acc += probs[i];
            if (r <= acc) {
                chosen = i;
                break;
            }
        }
        const token: usize = @intCast(idxs[chosen]);
        if (self.processor) |p| try p.commit(token);
        return token;
    }
};

fn argmax(ctx: *ExecContext, logits: *Logits) !usize {
    var idx = try logits.argmax(ctx, .vocab);
    defer idx.deinit();
    return @intCast(try idx.item());
}

test {
    _ = @import("sampler_tests.zig");
}
