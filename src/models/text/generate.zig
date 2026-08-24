//! The reference generation loop over the decoder contract
//! (`models.decoder`): prefill, then sample-and-emit until a stop id, the
//! token budget, or the cache capacity ends the reply. One loop serves
//! every conforming family; the family engines (`qwen35.chat`,
//! `inkling.chat`) and `gemma.Model.generate` call it with their own
//! prompt rendering, token sinks, and stop sets. Multi-turn chat and the
//! serving stack layer the KV-reuse machinery on top in `models.text.chat`.
const std = @import("std");
const fucina = @import("fucina");
const decoder = @import("../decoder.zig");
const sampler_mod = @import("sampler.zig");

const ExecContext = fucina.ExecContext;

/// Where committed tokens go: the caller's streaming/collection seam.
/// `emit` runs once per produced token, in commit order; a fired stop id is
/// not emitted (unless `GenerateOptions.emit_stop`).
pub const TokenSink = struct {
    ptr: *anyopaque,
    emitFn: *const fn (ptr: *anyopaque, token: usize) anyerror!void,

    pub fn emit(self: TokenSink, token: usize) anyerror!void {
        return self.emitFn(self.ptr, token);
    }
};

pub const GenerateOptions = struct {
    /// Sampling configuration; the default (`temperature = 0`) is greedy
    /// argmax.
    sampling: sampler_mod.Config = .{},
    /// Optional logit processor (grammar-constrained decoding), installed
    /// on the loop's sampler.
    processor: ?sampler_mod.LogitProcessor = null,
    /// Per-reply generation cap.
    max_tokens: usize,
    /// Cache position budget: the loop never decodes past it (the caller's
    /// context budget; at most the cache's own capacity).
    capacity: usize,
    /// Token ids that end the reply. Borrowed.
    stop_ids: []const u32 = &.{},
    /// Emit and count the fired stop id instead of dropping it (the
    /// `greedy` slice-filling convention; sinks that stream text keep the
    /// default).
    emit_stop: bool = false,
    /// Forward the last committed token into the cache when the budget
    /// ends the reply (the engine loops' shape: the cache stays flush with
    /// the reply). `greedy` disables it: its callers drop the cache with
    /// the call.
    trailing_forward: bool = true,
};

/// What ended the reply, alongside the produced-token count.
pub const Outcome = struct {
    produced: usize,
    /// True when a stop id fired (vs the budget or capacity cap).
    stopped: bool,
};

/// Run one reply over the decoder contract: prefill `prompt` at the
/// cache's current position, then sample, emit, and forward until a stop
/// condition. Returns the number of produced tokens; `generateOutcome`
/// also reports whether a stop id fired.
pub fn generate(
    comptime Model: type,
    model: decoder.ModelPtr(Model),
    ctx: *ExecContext,
    cache: *Model.Cache,
    prompt: []const usize,
    opts: GenerateOptions,
    sink: TokenSink,
) !usize {
    return (try generateOutcome(Model, model, ctx, cache, prompt, opts, sink)).produced;
}

pub fn generateOutcome(
    comptime Model: type,
    model: decoder.ModelPtr(Model),
    ctx: *ExecContext,
    cache: *Model.Cache,
    prompt: []const usize,
    opts: GenerateOptions,
    sink: TokenSink,
) !Outcome {
    decoder.assertDecoder(Model);
    const a = ctx.allocator;
    if (prompt.len == 0) return error.InvalidSequenceLength;

    // Committed tokens (prompt + reply): the repetition-penalty window.
    var history: std.ArrayList(usize) = .empty;
    defer history.deinit(a);
    try history.appendSlice(a, prompt);

    var sampler = sampler_mod.Sampler.init(opts.sampling);
    sampler.processor = opts.processor;

    // Prefill at the cache's current position.
    var logits = try model.forwardStep(ctx, cache, prompt, cache.len());
    defer logits.deinit();

    var produced: usize = 0;
    var stopped = false;
    while (produced < opts.max_tokens and cache.len() < opts.capacity) {
        const next = try sampler.next(ctx, &logits, history.items);
        if (isStop(next, opts.stop_ids)) {
            stopped = true;
            if (opts.emit_stop) {
                try sink.emit(next);
                produced += 1;
            }
            break;
        }
        try sink.emit(next);
        try history.append(a, next);
        produced += 1;

        if (cache.len() >= opts.capacity) break;
        if (!opts.trailing_forward and produced >= opts.max_tokens) break;
        // Allocate the next step before freeing the current logits, so an
        // error here leaves `logits` valid for the function-scope defer.
        const fresh = try model.forwardStep(ctx, cache, &.{next}, cache.len());
        logits.deinit();
        logits = fresh;
    }
    return .{ .produced = produced, .stopped = stopped };
}

fn isStop(token: usize, stop_ids: []const u32) bool {
    for (stop_ids) |s| if (token == s) return true;
    return false;
}

pub const Options = struct {
    max_new_tokens: usize,
    stop_token: ?usize = null,
};

/// Greedy autoregressive generation into a token slice: prefill
/// `prompt_tokens`, then argmax each step into `out_tokens` until
/// `max_new_tokens`, `out_tokens.len`, or the optional `stop_token` is
/// reached (the stop token is written and counted). Resets `kv`. Returns
/// the number of tokens written. A convenience shape over `generate`.
pub fn greedy(
    model: anytype,
    ctx: *ExecContext,
    kv: anytype,
    prompt_tokens: []const usize,
    out_tokens: []usize,
    options: Options,
) !usize {
    const Model = @typeInfo(@TypeOf(model)).pointer.child;
    kv.reset();

    var fill = SliceSink{ .out = out_tokens };
    var stop_buf: [1]u32 = undefined;
    var stop_ids: []const u32 = &.{};
    if (options.stop_token) |s| {
        stop_buf[0] = @intCast(s);
        stop_ids = stop_buf[0..1];
    }
    return generate(Model, model, ctx, kv, prompt_tokens, .{
        .max_tokens = @min(options.max_new_tokens, out_tokens.len),
        .capacity = std.math.maxInt(usize),
        .stop_ids = stop_ids,
        .emit_stop = true,
        .trailing_forward = false,
    }, fill.sink());
}

/// A `TokenSink` filling a caller slice in commit order.
const SliceSink = struct {
    out: []usize,
    n: usize = 0,

    fn emit(ptr: *anyopaque, token: usize) anyerror!void {
        const self: *SliceSink = @ptrCast(@alignCast(ptr));
        self.out[self.n] = token;
        self.n += 1;
    }

    fn sink(self: *SliceSink) TokenSink {
        return .{ .ptr = self, .emitFn = emit };
    }
};

pub fn argmaxLast(ctx: *ExecContext, logits: *const fucina.Tensor(.{ .seq, .vocab })) !usize {
    var last = try logits.narrow(ctx, .seq, logits.dim(.seq) - 1, 1);
    defer last.deinit();
    var index = try last.argmax(ctx, .vocab);
    defer index.deinit();
    return @intCast(try index.item());
}
