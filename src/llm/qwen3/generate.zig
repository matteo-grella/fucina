//! Greedy generation driver over a loaded Qwen3 `Model`: an application
//! capability around the model, split out of model.zig (the runner and the
//! examples own the richer sampling loops; this is the minimal argmax
//! reference on the `forwardStep` decode path).
const std = @import("std");
const fucina = @import("fucina");
const qwen3 = @import("model.zig");
const kv_cache = @import("../kv_cache.zig");

const ExecContext = fucina.ExecContext;
const KvCache = kv_cache.KvCache;

pub const Options = struct {
    max_new_tokens: usize,
    stop_token: ?usize = null,
};

/// Greedy autoregressive generation: prefill `prompt_tokens`, then sample
/// the argmax token each step into `out_tokens` until `max_new_tokens`,
/// `out_tokens.len`, or the optional `stop_token` is reached. Resets `kv`.
/// Returns the number of tokens written.
pub fn greedy(
    model: *const qwen3.Model,
    ctx: *ExecContext,
    kv: *KvCache,
    prompt_tokens: []const usize,
    out_tokens: []usize,
    options: Options,
) !usize {
    if (prompt_tokens.len == 0) return qwen3.Error.InvalidSequenceLength;
    kv.reset();

    var logits = try model.forwardStep(ctx, kv, prompt_tokens, 0);
    defer logits.deinit();

    const limit = @min(options.max_new_tokens, out_tokens.len);
    var produced: usize = 0;
    while (produced < limit) {
        const next = try argmaxLast(ctx, &logits);
        out_tokens[produced] = next;
        produced += 1;
        if (options.stop_token) |stop| if (next == stop) break;
        if (produced == limit) break;
        // Allocate the next step before freeing the current logits, so an
        // error here leaves `logits` valid for the function-scope defer
        // (deinit-then-reassign would leave it dangling on the error path).
        const fresh = try model.forwardStep(ctx, kv, &.{next}, kv.len);
        logits.deinit();
        logits = fresh;
    }
    return produced;
}

fn argmaxLast(ctx: *ExecContext, logits: *const fucina.Tensor(.{ .seq, .vocab })) !usize {
    var last = try logits.narrow(ctx, .seq, logits.dim(.seq) - 1, 1);
    defer last.deinit();
    var index = try last.argmax(ctx, .vocab);
    defer index.deinit();
    return @intCast(try index.item());
}
