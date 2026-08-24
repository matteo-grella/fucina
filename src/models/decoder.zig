//! The autoregressive text-decoder contract. `assertDecoder(Model)` is a
//! comptime check with no runtime cost (no vtable): the generic layers
//! (`chat.Conversation`, `speculative.SpeculativeDecoder`,
//! `serving.gguf_chat.GgufChatBackend`, `generate`) call it at the top of
//! their type functions and are written against this surface only.
//!
//! A conforming `Model` exposes:
//!
//! - `Cache`: the decode-state type (`kv_cache.KvCache` for the attention
//!   families; recurrent families carry their own state struct). Methods:
//!   `len() usize` (cached positions), `reset()`, `deinit()`, and
//!   `truncate(keep_len)` iff `caps.rewind`.
//! - `caps: Caps`: what the family's cache and forwards support.
//! - `initCache(self, ctx, capacity) !Cache`: capacity is the maximum
//!   sequence length; families whose construction ignores `ctx` still take
//!   it, so the spelling is uniform.
//! - `forwardStep(self, ctx, cache, tokens, pos0) !Tensor(.{.seq,.vocab})`:
//!   process `tokens` at absolute positions `pos0 ..`, advance the cache,
//!   and return the LAST row's logits as a `[1, vocab]` tensor. `pos0`
//!   equals `cache.len()`; families whose cache tracks position internally
//!   assert this in debug builds. Returns are caller-owned tensors; no
//!   session-owned slices anywhere in the contract.
//! - `forwardStepAllLogits(...)` (iff `caps.rewind`): same signature, one
//!   logits row per token (`[tokens.len, vocab]`), the speculative-verify
//!   entry.
//! - `forwardStepBatch(self, ctx, caches, tokens) !Tensor(.{.seq,.vocab})`
//!   (iff `caps.batch`): one new token per stream, each against its own
//!   cache, the lockstep multi-stream decode entry.
//!
//! Conforming families: qwen3/qwen3moe (`runner.Model`), gemma4
//! (`gemma.Model`), SHINE (`research.shine.AdaptedModel`), qwen35,
//! deepseek2, deepseek4, glm4moe, inkling. kimi3 stays OUTSIDE the
//! contract: its research-tier surface is a cache-less whole-sequence
//! `forward(self, ctx, tokens) !RawTensor`.

/// What a family's decode surface supports, declared as `Model.caps`.
pub const Caps = struct {
    /// The cache supports `truncate` (speculative rewind, cross-request KV
    /// reuse). Requires `forwardStepAllLogits`.
    rewind: bool,
    /// `forwardStepBatch` exists (lockstep multi-stream decode).
    batch: bool,
};

/// Comptime check that `Model` satisfies the decoder contract; emits a
/// `@compileError` naming the model type and the missing declaration.
pub fn assertDecoder(comptime Model: type) void {
    comptime {
        const name = @typeName(Model);
        if (!@hasDecl(Model, "Cache"))
            @compileError(name ++ " is not a decoder: missing `pub const Cache` (the decode-state type)");
        if (!@hasDecl(Model, "caps"))
            @compileError(name ++ " is not a decoder: missing `pub const caps: decoder.Caps`");
        if (@TypeOf(Model.caps) != Caps)
            @compileError(name ++ ".caps must be a `decoder.Caps` value");
        if (!@hasDecl(Model, "initCache"))
            @compileError(name ++ " is not a decoder: missing `initCache(self, ctx, capacity) !Cache`");
        if (!@hasDecl(Model, "forwardStep"))
            @compileError(name ++ " is not a decoder: missing `forwardStep(self, ctx, cache, tokens, pos0)`");
        if (Model.caps.rewind and !@hasDecl(Model, "forwardStepAllLogits"))
            @compileError(name ++ ".caps.rewind requires `forwardStepAllLogits` (the speculative-verify entry)");
        if (Model.caps.batch and !@hasDecl(Model, "forwardStepBatch"))
            @compileError(name ++ ".caps.batch requires `forwardStepBatch` (the lockstep multi-stream entry)");

        const Cache = Model.Cache;
        const cache_name = @typeName(Cache);
        if (!@hasDecl(Cache, "len"))
            @compileError(cache_name ++ " (the Cache of " ++ name ++ ") is missing the `len() usize` method");
        if (@typeInfo(@TypeOf(Cache.len)) != .@"fn")
            @compileError(cache_name ++ ".len must be a method (`fn len(*const Cache) usize`)");
        if (!@hasDecl(Cache, "reset"))
            @compileError(cache_name ++ " (the Cache of " ++ name ++ ") is missing `reset()`");
        if (!@hasDecl(Cache, "deinit"))
            @compileError(cache_name ++ " (the Cache of " ++ name ++ ") is missing `deinit()`");
        if (Model.caps.rewind and !@hasDecl(Cache, "truncate"))
            @compileError(name ++ ".caps.rewind requires `" ++ cache_name ++ ".truncate(keep_len)`");
    }
}

/// The pointer type of `Model.forwardStep`'s `self` parameter: `*const
/// Model` for most families, `*Model` where the forward refreshes
/// model-held scratch (glm4moe). Generic layers take their model argument
/// as this type so both spellings instantiate.
pub fn ModelPtr(comptime Model: type) type {
    const params = @typeInfo(@TypeOf(Model.forwardStep)).@"fn".params;
    return params[0].type.?;
}

test {
    _ = @import("decoder_tests.zig");
}
