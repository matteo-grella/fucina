# Chapter 13 — Inference tricks

*Part V — Language models*

[Chapter 12](12-a-transformer-from-scratch.md) ended with a transformer decoding: prefill the prompt, then one forward pass per token, each pass reading every weight of the model to produce a single row of logits. That loop is correct, and on a CPU it is also the whole cost structure of the system — decode speed is set by how fast you can stream weights through the memory hierarchy, not by arithmetic.

This chapter is a tour of the machinery Fucina builds *around* that loop to make the same model faster, cheaper, and more controllable — without changing what it says. That last clause is the theme. Every trick here is either **provably lossless** (the committed token stream is the one plain decoding would produce) or **gated so it can never lose** (a measured cost model decides when the trick runs, and turns it off when it would hurt). Speed claims are dated, machine-specific snapshots quoted from the repo's own documents; correctness claims are contracts with tests. The five headliners:

1. **Speculative decoding without a draft model** — drafts retrieved from text the model has already seen, verified in one batched forward, gated by measured economics (`docs/SPECULATIVE.md`).
2. **Batch-N multi-stream decode** — N conversations, one weight pass per step (`docs/BENCHMARK.md`).
3. **Constrained decoding** — grammar masks on the logits, one seam, every decode path; and a composition that makes a grammar *accelerate* speculation (`docs/CONSTRAINED-DECODING.md`).
4. **KV reuse across requests** — a slot pool plus a disk tier that turns a stateless API into a warm one (`examples/lmserve/`, `src/llm/kv_persist.zig`).
5. **MoE expert streaming** — a 142 GB mixture model decoding on a 64 GB machine by paging routed experts from disk (`docs/RUNNING-MODELS.md`).

Plus one honest paragraph about the GPU offload seam, and a look at `lmserve`, the OpenAI-compatible server where several of these tricks meet. A reminder from the repo's own status notes: the public API is young and explicitly unstable — every signature in this chapter is today's code, not a frozen contract.

## 13.1 The loop, the invariant, and the rewindable cache

Everything in this chapter is a transformation of one loop, so the loop's shape has to be stated precisely first. Fucina states it as an invariant, documented on the speculative decoder's `step` and enforced at runtime (`src/llm/speculative/core.zig:540-561`):

> `history` holds every committed token (prompt + generated) and its LAST element is the token just committed but not yet in `kv`, i.e. `history.items.len == kv.len + 1`.

A violated invariant is not a debug assert but a real error, `error.InvalidDecodeState` — because "an empty or kv-desynced history would index out of bounds / corrupt the cache in ReleaseFast" (core.zig:557-560). Every trick below must preserve this invariant through every path, including error unwinds.

The second foundation is that the KV cache can **rewind**. Chapter 12 built the cache as per-position K/V rows; because every position occupies whole per-(position, kv-head) rows and every reader addresses rows strictly below `len`, dropping positions is one integer store. From the machine-verified snippet in `docs/REFERENCE.md` §13.4:

```zig
try cache.appendLayer(&ctx, 0, &k, &v); // writes at offset len, does not advance
cache.advance(3); // once per step, after all layers
try std.testing.expectEqual(@as(usize, 3), cache.len);

cache.truncate(1); // speculative rewind: drop rejected positions
try std.testing.expectEqual(@as(usize, 1), cache.len);
```

`truncate(keep_len)` "rewinds to the first `keep_len` positions (a value at or above `len` is a no-op)" and is verified bitwise against a fresh cache by a truncate + re-append test (`docs/SPECULATIVE.md` §1, `src/llm/kv_cache_tests.zig`). This one primitive is what makes speculation, cross-request reuse, and turn trimming possible. It is also why one model family sits this chapter out: qwen35's Gated-DeltaNet keeps *recurrent* state, and "Gated-DeltaNet's recurrent state cannot rewind: rejecting a draft would require restoring conv/delta-scan state at an arbitrary earlier position, which the recurrence does not support" (`docs/SPECULATIVE.md` §11). Speculation there is out of scope structurally, not by omission.

The third foundation is a single sentence from `docs/BENCHMARK.md` that explains half the chapter:

> Decode is weight-bandwidth-bound, so batching N streams into one m=N pass reads the weights once instead of N times.

> **ML note** — Why bandwidth and not FLOPs? A decode step is a matrix–vector product per weight matrix: every weight is read once and used for exactly one multiply-add. Modern CPUs multiply far faster than they can stream memory, so the wall-clock cost of a decode step is essentially (bytes of weights) ÷ (memory bandwidth). This is also why [Chapter 11](11-model-files-and-quantization.md)'s quantization *speeds decode up* — fewer bytes per weight — and why anything that gets more than one token's worth of work out of one weight pass (batching, speculation) is worth real money on a CPU.

## 13.2 Batch-N multi-stream decode

**The problem.** Serve N conversations and the naive loop reads the full weight set N times per generated-token round.

**The idea.** Decode all N streams in lockstep: each step gathers one token per stream and runs a single m=N forward pass — one weight read amortized over N streams. Each stream keeps its own KV cache, its own sampler, its own history; only the GEMM is shared.

**Where it lives.** The turnkey entry is `chat.Conversation.sendBatch` (`src/llm/chat.zig`; API in `docs/REFERENCE.md` §13.8.2):

```zig
pub fn sendBatch(convos: []const *Self, users: []const []const u8,
                 writers: []const *std.Io.Writer, produced: []usize) !void
```

It requires the model to expose `forwardStepBatch(ctx, caches: []const *KvCache, token_ids: []const usize)` — and here is a Zig-flavoured detail worth pausing on: the requirement is **comptime-gated**, "so families without it (gemma4 today) still instantiate the type and get `error.BatchDecodeUnsupported` at runtime" (REFERENCE.md §13.8.2).

> **Zig note** — `Conversation(Model, Tok)` is a comptime-generic type over a duck-typed `Model`. Zig lets the generic *inspect* its type parameter at compile time (`@hasDecl`-style checks), so a capability like `forwardStepBatch` can be optional: families that have it get lockstep batching, families that don't still compile and fail loudly at runtime only if you actually call it. This is interface-by-capability, resolved at compile time, with no vtable and no inheritance.

Per-stream semantics are guaranteed to match a plain `send` exactly — each stream samples from its own logits row with its own sampler and history, because a `Sampler` is "single-stream mutable state (RNG + config): not thread-safe, one per decode stream" (REFERENCE.md §13.6). The batch validator rejects malformed batches up front (`error.EmptyBatch`, `error.BatchLengthMismatch`, `error.SpeculationWithBatch`, `error.MixedBatchModels`, `error.DuplicateBatchConversation` — REFERENCE.md §13.8.2). Note the third one: speculation and lockstep batching do not compose today; the decoder verifies one stream's drafts, not N. The ownership contract on error is also spelled out: the batch aborts and *every* stream's history is trimmed back to its KV cache, so healthy siblings of a failing stream stay usable.

**The measured effect.** From `docs/BENCHMARK.md`, "Batch-N multi-stream decode (M1 Max)" — M1 Max, ReleaseFast, 8 threads, cool machine, Qwen3-0.6B, 8-token prompt, 64 decode steps per stream, batch vs the same binary running the N generations back to back:

| weights | N | batch tok/s | sequential tok/s | speedup | outputs |
| --- | --- | ---: | ---: | ---: | --- |
| Q4_K_S | 2 | 152.6 | 136.2 | 1.12x | identical |
| Q4_K_S | 4 | 313.5 | 121.6 | 2.58x | ~1e-6 logit drift (packed 4-row kernels) |
| Q4_K_S | 8 | 369.7 | 116.0 | 3.19x | ~1e-6 drift |
| Q6_K | 2 | 139.8 | 101.2 | 1.38x | identical |
| Q6_K | 4 | 254.1 | 92.8 | 2.74x | identical |
| Q6_K | 8 | 358.3 | 110.2 | 3.25x | identical |
| f16 | 4 | 205.5 | 66.4 | 3.09x | identical |

The README's one-line summary is "batch-N multi-stream decode (3.2x aggregate throughput at 8 streams)" (README.md). Two honest footnotes from the same benchmark section: the modest N=2 gain is the per-row (m<4) quantized kernels re-streaming packed weights — the weights-read-once kernels engage at m≥4; and that same m≥4 boundary is where quantized-weight logits can drift by ~1e-6 relative (reassociation in the multi-row kernels), while f32/f16 stay bitwise "verified to m=12". Outputs in the table were cross-checked token for token.

Try it on the qwen3 runner (`examples/qwen3/README.md`):

```sh
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q4_K_S.gguf \
  <prompt-token-ids> --gen 64 --bench 3 --streams 4
```

One stream still pays one weight pass per token. The next trick attacks that.

## 13.3 Speculative decoding without a draft model

**The problem.** A single decode stream reads all weights to produce one token. If you could *guess* the next k tokens cheaply and check the guesses in one batched forward, you would commit several tokens per weight pass.

**The idea, upstream and here.** Classic speculative decoding (Leviathan et al., 2023 — cited at `src/llm/speculative/core.zig:10`) runs a small *draft model* ahead of the big one and accepts its guesses via rejection sampling. Fucina's twist: **no draft model at all**. From `docs/SPECULATIVE.md`:

> drafts come from cheap deterministic indexes over text the model has already seen (the conversation itself, injected reference documents, recycled verification logits) — **no draft model, no training, no extra weights**: a ~4.6 MiB recycling matrix plus a suffix-automaton (SAM) index that grows at ~110 B/token.

Why does that work at all? Because LLM workloads repeat themselves: a model asked to copy a quote, edit code, or restate a document produces long spans that already exist in its own context. When they don't — free-form prose — a gate (§13.5) turns the feature off.

### Losslessness by degeneracy

Determinism of the drafter is not a simplification; it is the *proof lever*. The normative header (`src/llm/speculative/core.zig:9-20`) is short enough to quote whole:

```zig
//! Losslessness — one code path for greedy AND sampled: because the drafter is
//! deterministic (a one-hot proposal distribution q), Leviathan rejection
//! sampling degenerates to running the FULL sampling pipeline (penalties,
//! temperature, top-k/top-p/min-p) on the target logits at each verified
//! position, conditioned on the HYPOTHETICAL prefix (committed history + the
//! draft tokens accepted so far):
//!
//!   - accept position i while sampled == draft[i];
//!   - at the first mismatch, the sampled token IS the correction token
//!     (provably distributed as the target distribution);
//!   - if the whole draft is accepted, the (k+1)-th row's sample is a free
//!     bonus token.
```

No probability-ratio acceptance test, no second distribution — token-ID equality, because a deterministic proposal is a one-hot q and the general rejection-sampling formula collapses. Greedy is the same code path (temperature ≤ 0 makes the sampler an argmax). The contract decomposes into proof obligations with tests behind each (`docs/SPECULATIVE.md` §2): exactly **one RNG draw per committed token** (positions past the first mismatch are never sampled); replay equivalence of every sampler input row against a plain run; penalties conditioned on the hypothetical prefix; adversarial draft sources unable to corrupt the stream; gating orthogonal to commitment.

"Lossless" also has a precise, deliberately bounded definition (core.zig:44-46):

> Lossless therefore means: same DISTRIBUTION always; same sample stream whenever the logits match bitwise (which the tests below verify for the small-m regime).

The caveat is the m-dependent kernels from §13.2: verify batches compute logits at m = 1+draft rows, and rows are bitwise-independent through every kernel until the documented thresholds (fused K-quant FFN at seq ≥ 12, tiled attention at seq ≥ 48), beyond which ~1e-6 relative drift can flip a near-tied sample. Below the thresholds, on BLAS-free builds, the equivalence tests are literally bitwise (`batch_rows_bitwise`, `src/llm/speculative/core_tests.zig`).

Stop tokens are part of the RNG contract too: when a committed token equals `Options.stop_token`, the verify row loop breaks *immediately* — a plain run stops there, so sampling any further row (the bonus row included) "would consume draws the plain run never makes and desync a persistent sampler for the rest of the conversation" (core.zig:32-39).

### Build it yourself

The verify loop is small enough to own completely. Here is a compile-checked course version (not repo code) over a toy deterministic "model" — a pure next-token function, which is exactly what greedy decoding over real logits collapses to:

```zig
// Course code — toy speculative decoding. The full file also defines
// DraftFn (*const fn (history, buf) usize) and three drafters — noDraft
// (k = 0), perfectDraft (consults the model), garbageDraft (always
// proposes 0); the losslessness test appears below.
fn targetNext(token: usize) usize {
    return (5 * token + 3) % 11;
}

const ToyKv = struct {
    len: usize = 0,
    fn truncate(self: *ToyKv, keep_len: usize) void {
        if (keep_len < self.len) self.len = keep_len;
    }
};

fn step(
    alloc: std.mem.Allocator,
    kv: *ToyKv,
    history: *std.ArrayList(usize),
    draft_fn: DraftFn,
) !usize {
    // The decode-loop invariant, checked exactly like the real decoder.
    if (history.items.len == 0 or history.items.len != kv.len + 1)
        return error.InvalidDecodeState;

    var draft_buf: [4]usize = undefined;
    const k = draft_fn(history.items, &draft_buf);
    const draft = draft_buf[0..k];

    // "One batched forward": all 1+k input tokens enter the cache...
    const start_len = history.items.len;
    kv.len += 1 + k;
    // ...and row i's greedy sample is the model's pick after input i.
    var input = history.items[start_len - 1];
    var i: usize = 0;
    while (i <= k) : (i += 1) {
        const sampled = targetNext(input);
        try history.append(alloc, sampled);
        const matched = i < k and sampled == draft[i];
        if (!matched) break; // correction (i < k) or bonus (i == k)
        input = draft[i];
    }
    // Rewind: drop the rejected rows AND the newest committed token — it
    // enters the cache on the NEXT forward. One integer store.
    kv.truncate(history.items.len - 1);
    return history.items.len - start_len;
}
```

Three details to notice, because the real decoder has all of them. The carried token: the *last committed* token is the batch's first input — it was never forwarded (that is the invariant), so the verify batch forwards it along with the k drafts, 1+k rows total. The final truncate runs on *every* outcome, not just rejection: even a fully accepted draft leaves the newest committed token out of the cache, restoring the invariant for the next step. And the correction/bonus distinction is just the loop index at break time: a mismatch at `i < k` means the sample corrected the draft; reaching `i == k` means the whole draft survived and row k's sample is the free bonus token.

The payoff is the test — the toy version of the repo's adversarial-source losslessness proof:

```zig
// Course code, continued — the drafter cannot change what is committed.
fn decode(alloc: std.mem.Allocator, prompt: usize, n_tokens: usize, draft_fn: DraftFn) ![]usize {
    var kv = ToyKv{};
    var history: std.ArrayList(usize) = .empty;
    errdefer history.deinit(alloc);
    try history.append(alloc, prompt);
    while (history.items.len < 1 + n_tokens) {
        _ = try step(alloc, &kv, &history, draft_fn);
        // The invariant holds after every step, whatever the draft did.
        std.debug.assert(history.items.len == kv.len + 1);
    }
    // A verify batch can overshoot the budget; trim for exact comparison.
    history.shrinkRetainingCapacity(1 + n_tokens);
    return history.toOwnedSlice(alloc);
}

test "lossless: any deterministic draft source commits the plain stream" {
    const alloc = std.testing.allocator;
    const plain = try decode(alloc, 7, 40, noDraft);       // k = 0 every step
    defer alloc.free(plain);
    const fast = try decode(alloc, 7, 40, perfectDraft);   // consults the model
    defer alloc.free(fast);
    const slow = try decode(alloc, 7, 40, garbageDraft);   // always proposes 0
    defer alloc.free(slow);
    try std.testing.expectEqualSlices(usize, plain, fast);
    try std.testing.expectEqualSlices(usize, plain, slow);
}
```

All three runs commit the identical stream; the perfect drafter merely commits 5 tokens per step (4 accepted + bonus) while the garbage drafter commits 1 (the correction). That is the whole idea, miniaturized: **drafts change how fast tokens commit, never which tokens commit.** What the toy leaves out is exactly what the rest of this chapter adds back: real sampling with RNG accounting (§13.3), drafters that earn their acceptance (§13.4), and the recognition that a verify batch costs more than a plain step (§13.5).

The real thing is `SpeculativeDecoder(comptime Model: type)` (`src/llm/speculative/core.zig:486`), duck-typed over any model exposing `forwardStep` and `forwardStepAllLogits` over the shared `KvCache` — qwen3 and gemma4 today. One decode iteration is one call:

```zig
pub fn step(
    self: *Self,
    ctx: *ExecContext,
    model: *const Model,
    kv: *KvCache,
    sampler: *Sampler,
    history: *std.ArrayList(usize),
    sink: TokenSink,
) !usize {
```

(`history` must be allocated with `ctx.allocator` — the decoder appends committed tokens to it; each committed token streams out through `sink`, a tiny two-field vtable any stdout writer or SSE sink can satisfy.) `forwardStepAllLogits` is the verify entry: same KV semantics as `forwardStep`, but it returns `[len, vocab]` logits for *every* appended position — "one batched pass scores all draft positions for ~one step's weight traffic" (`docs/REFERENCE.md` §14). The verify row loop (core.zig:676-717) is recognizably the toy's loop plus real sampling, top-k feedback capture, and stats. And one line the toy also mirrors deserves its own note:

> **Zig note** — `errdefer` as transactional invariant restoration. The batched forward advances `kv` (to `pos0 + 1 + draft_len`) *before* its own fallible tail ops, and the row loop appends to `history` as it commits — so a failure anywhere in between would leave cache and history desynced. One line fixes every path: `errdefer kv.truncate(history.items.len - 1);` (core.zig:666). `history.items.len` is read *at unwind time*, so the truncate always restores `history.len == kv.len + 1` no matter where the error struck, and also drops unverified draft rows from the cache. Error unwinding as a correctness mechanism, not just cleanup.

### The DraftSource seam

Proposers plug in through a vtable (`src/llm/speculative/core.zig:74-94`):

```zig
pub const DraftSource = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// Propose up to `buf.len` continuation tokens given the full committed
        /// token context (the last element is the token just committed).
        /// Returns the number of tokens written into `buf`; 0 = no draft.
        suggest: *const fn (ptr: *anyopaque, context: []const usize, buf: []usize) usize,
        /// Observe newly COMMITTED tokens (for index updates).
        observe: *const fn (ptr: *anyopaque, committed: []const usize) void,
        /// Observe verification logits feedback. Null = the source has no use
        /// for it and the decoder skips computing the top-k entirely.
        observeTopK: ?*const fn (ptr: *anyopaque, positions: []const TopKRow) void = null,
        // ...plus optional truncatePending (a wrapper shortened your draft).
    };
```

> **Zig note** — this is the hand-rolled dynamic-dispatch pattern (`*anyopaque` + a pointer to a vtable of function pointers) you have met before, with one twist worth stealing: **optional entries where absence is information**. `observeTopK` defaults to `null`, and a null hook doesn't just skip the call — the decoder skips *computing the top-k feedback entirely*. Not implementing a method changes the caller's work. Two defensive details at this boundary: the decoder clamps the vtable's claimed draft length (`@min(self.source.suggest(...), max_draft)`, core.zig:583-586 — "a lying return value would walk verify_buf out of bounds in ReleaseFast"), and `TopKRow.topk` is explicitly *borrowed*, "valid only for the duration of the `observeTopK` call" (core.zig:68-69).

Degenerate wiring fails at init rather than at step time: a source that wants top-k feedback combined with `topk_feedback = 0` is `error.TopKFeedbackDisabled` (core.zig:512) — "it would silently starve the source" otherwise. And every run can report on itself: the decoder keeps a `Stats` struct (steps split into verify/fallback/disabled, drafted/accepted/bonus/rejected counts) whose `writeSummary` produces the line you will see from the CLI (core.zig:117-165):

```text
spec: steps=... (verify ..., fallback ..., off ...) committed=... (N tok/step) drafted=... accepted=... (P% acc) bonus=... rejected=...
```

`tokensPerStep` and `acceptanceRate` are the two numbers to watch — and §13.5 is about why tokens/step alone is *not* enough to decide whether speculation is winning.

## 13.4 Where drafts come from

Three deterministic sources, composed behind one `DraftSource`.

### The suffix automaton

`src/llm/speculative/sam_index.zig` builds an **online suffix automaton** (SAM) over the committed token stream (SAM-Decoding / SuffixDecoding lineage). The question it answers per step: *what is the longest suffix of the stream that has occurred before, and what followed that occurrence?* You can state that contract by brute force in a dozen lines — this is compile-checked course code, and it is exactly the oracle shape the repo property-tests its SAM against:

```zig
// Course code — the SAM's contract by brute force, O(n^2).
/// Longest L such that the stream's L-token suffix also occurs ending at
/// some index j < n-1 (an occurrence strictly BEFORE the end).
fn matchLenBrute(stream: []const usize) usize {
    const n = stream.len;
    if (n < 2) return 0;
    var best: usize = 0;
    var j: usize = 0; // last index of the candidate earlier occurrence
    while (j + 1 < n) : (j += 1) {
        var l: usize = 0;
        while (l <= j) : (l += 1) {
            if (stream[j - l] != stream[n - 1 - l]) break;
        }
        best = @max(best, l);
    }
    return best;
}
```

On the stream `{5, 6, 7, 5, 6}` this returns 2 (the suffix `[5,6]` occurred ending at index 1) and the draft is what followed that occurrence: `{7, 5, 6}` — matching the repo's machine-verified snippet for the real index (`docs/REFERENCE.md` §13.9.2):

```zig
test "SamIndex: longest self-excluded suffix match drives the draft" {
    const alloc = std.testing.allocator;
    var sam = try llm.speculative.sam_index.SamIndex.init(alloc);
    defer sam.deinit();
    try sam.append(&.{ 5, 6, 7, 5, 6 });
    // Longest suffix with an occurrence ending strictly before the end: [5,6].
    try std.testing.expectEqual(@as(usize, 2), sam.matchLen());
    var buf: [8]usize = undefined;
    const n = sam.draft(&buf); // tokens after the prior occurrence: {7,5,6}
    try std.testing.expectEqualSlices(usize, &.{ 7, 5, 6 }, buf[0..n]);
}
```

The SAM does what the brute force does in **O(1) amortized per appended token**, with an *exact, unbounded* match length — that exactness matters because the match length drives the adaptive draft budget (a long match is high confidence, so draft more). The subtle part is self-match exclusion: a naive SAM matches the stream against itself (every suffix trivially "occurs" at the end). The fix is **cursor-before-extend** (`sam_index.zig:29-38`): the match cursor is advanced with token tᵢ against the automaton *as it was* — the automaton of t₀..tᵢ₋₁ — and only then is the automaton extended with tᵢ. A successful transition therefore *proves* a prior occurrence. Drafts follow the **most recent** prior occurrence ("recency wins for code and editing flows, where the latest revision of a passage is the one worth copying", `docs/SPECULATIVE.md` §5); `matchLen` and drafts are property-tested against brute force after every append. Memory: ~110 B/token (SPECULATIVE.md §5).

The automaton kernel itself — the greedy longest-match descent — is ten lines (`sam_index.zig:316-325`):

```zig
fn advanceRaw(self: *const SamIndex, state: u32, len: u32, t: u32) struct { u32, u32 } {
    var s = state;
    var l = len;
    while (true) {
        if (self.trans.get(key(s, t))) |next| return .{ next, l + 1 };
        if (s == 0) return .{ 0, 0 };
        s = @intCast(self.states.items[s].link);
        l = self.states.items[s].len;
    }
}
```

Follow the transition if it exists; otherwise drop along suffix links (each drop shortens the matched suffix as little as possible) until a transition appears or the root gives up. That is the standard suffix-automaton walk, and the file's header proves why, run *before* each extension, it yields exactly the brute-force answer above.

> **Zig note** — two idioms in those ten lines. `struct { u32, u32 }` is an anonymous *tuple* type: multiple return values without naming a struct, consumed by destructuring at the call site (`self.match_state, self.match_len = self.advanceRaw(...)`, sam_index.zig:225). And `key(s, t)` is bit-packing as API design: `inline fn key(state: u32, token: u32) u64 { return (@as(u64, state) << 32) | token; }` (sam_index.zig:182-184) folds (state, token) into one `u64` so the whole automaton uses a *single* global hash map instead of a per-state map — fewer allocations, better locality, one lookup per step.

Two robustness rules documented in the file: a failed `append` **poisons** the index — degraded mode, all queries return 0 forever — because "a half-applied append must never serve drafts" (`sam_index.zig:200-202`); and `freeze()` must be called before external cursors are taken, because appends can split automaton states and "external cursors over a still-growing automaton would silently dangle" (`sam_index.zig:83-86`).

Frozen SAMs are the **RAG seam**: build an index per tokenized reference document, freeze it, and run per-conversation `Cursor`s over it — the drafter can now copy spans from documents the model is likely to quote. The contract is strict and token-based: `addReference(tokens)` — *the caller tokenizes*, with the same tokenizer the model decodes with, "or the SAM never aligns with the committed stream and acceptance is zero" (`docs/SPECULATIVE.md` §8). Not theoretical: a since-fixed pretokenizer bug (625 tokens vs llama.cpp's 500 on a code fixture) silently broke the code-edit benchmark task this way.

### The Token-Recycling matrix

When no index matches, `src/llm/speculative/recycling.zig` drafts from recycled verification logits (Token Recycling, Luo et al. 2024): verification already computes full logits for every draft position and throws away everything but the sampled token — recycle the top-K instead. One row per vocab token holds the most recently observed top-8 next-token candidates; drafting walks the top-1 chain from the last committed token. The whole thing is a `vocab × 8` u32 matrix — "≈4.6 MiB for the Qwen3 vocab at K=8" (`docs/REFERENCE.md` §13.9.3) — fed through `observeTopK`. From the machine-verified snippet there:

```zig
var rec = try llm.speculative.recycling.Recycling.init(alloc, 32); // vocab 32, K = 8
defer rec.deinit();
rec.update(3, &.{ 7, 9 }); // most recent top-K observed after token 3
rec.update(7, &.{5});
var buf: [4]usize = undefined;
try std.testing.expectEqual(@as(usize, 2), rec.draftChain(3, &buf)); // 3 -> 7 -> 5, then unseen
```

### The cascade

`SpeculationIndex` (`src/llm/speculative/cascade.zig`) composes all of it behind one `asDraftSource()`: the conversation SAM, any number of frozen reference SAMs, and the recycling fallback. Selection: the source with the longest current match wins (ties break toward the conversation — recency in the live stream beats a static document); the draft budget grows with match confidence (`budget = 2·(1 + match_len)`); matches shorter than 2 fall through to the recycling chain. Each source keeps a rolling acceptance window over its last 64 drafted tokens — below 20% acceptance it is muted for 128 committed tokens, then re-probed, and muted sources *keep observing* so they are in sync when re-probed (`docs/SPECULATIVE.md` §7). This per-source gate handles *which* proposer to trust; the global economics belong to the next section.

## 13.5 The economics, the gate, and the honest numbers

### Verify passes are not free

A verify over k drafts is an m=k+1 forward — and §13.2 already told you m>1 forwards cost more than m=1. Measured with the `--spec-bench` probe (M1 Max, ReleaseFast, best-of reps; the shipped `default_cost_table` at `src/llm/speculative/core.zig:230-235`), for dense Qwen3-0.6B-Q4_K_S, in plain-step equivalents:

| draft k | verify cost C(k) | conservative break-even acceptance ≈ C(k)/k |
| --- | --- | --- |
| 2 | 1.65 | ~83% |
| 4 | 1.42 | ~36% |
| 8 | 2.84 | ~36% |
| 16 | 4.5 | ~28% |

Note the table is *non-monotonic* at the low end — verify-4 cheaper than verify-2, a small-m kernel-dispatch effect — "which is why it is a measured table with interpolation, not a formula" (`docs/SPECULATIVE.md` §3). The exact break-even: a verify over k drafts costs C(k) and commits a+1 tokens (a = accepted prefix; the +1 — correction *or* bonus — is always free), so it wins iff `a + 1 > C(k)`, i.e. strict per-verify break-even at `a = C(k) − 1`. Worked at k=16: the verify costs 4.5 plain steps, so it needs 3.5 accepted tokens to break even — about 22% acceptance — while the conservative planning form `acc ≈ C(k)/k` says 28%; "the gap between the two bounds is the margin for overheads the table doesn't price (probe steps, fallback churn, and short low-k drafts whose per-token economics are worse)" (SPECULATIVE.md §3).

And the counterexample that justifies the whole gate: **MoE**. On Qwen3-30B-A3B, verify-4 costs 3.7 plain steps — batching defeats the MoE decode advantage because "each verify row activates its own experts, so weight reuse across rows is minimal". Conservative break-even ≈ 93% acceptance, "out of reach for retrieval sources on general text, so the gate carries MoE: speculation auto-disables there and costs only the probe overhead" (SPECULATIVE.md §3).

### The CostGate

`CostGate` (`src/llm/speculative/core.zig:276`) gates on **estimated speedup**, not tokens per step:

```text
est_speedup = committed_tokens / verify_cost_in_plain_step_equivalents
```

over a rolling window of verify steps. The design record explains why the naive metric was thrown out: a flat tokens/step gate "kept e.g. 1.56 tok/step at k≈8 alive as a 'win' while actually losing ~45%" (verify-8 ≈ 2.84 plain steps; SPECULATIVE.md §4). The cost model is deliberately **hybrid**: the static measured table gives the cost curve's *shape*, and live verify/plain timings continuously rescale it through a clamped EWMA multiplier — because "a pure-live model proved non-robust in practice: one noisy timing sample (scheduler stall, thermal hiccup) tripped a winning run off" (SPECULATIVE.md §4). In code (`verifyCost`, core.zig:384-398): one sample moves the estimate by at most 20% of its clamped deviation, the scale is clamped to [0.25, 4.0], and a verify is never priced below the plain step it replaces.

Around the estimate sit the policy pieces, all fields of `Options` (core.zig:167-222): hysteresis (speculate while est ≥ 1.0, re-enable only at ≥ 1.1 so a marginal regime doesn't flap), exponential re-probe backoff (128 disabled steps → 256 → 512 → capped at 1024, short 4-verify probes), and an acceptance-adaptive draft budget (low-acceptance phases verify small cheap drafts instead of max-draft losses). Configuration is validated loudly at init — `error.RateWindowTooSmall`, `error.ProbeStepsZero`, `error.CostTableEmpty`, `error.ReprobeAfterZero` — with a comment worth internalizing: "Boundary validation, not debug asserts: these options come straight from embedder configuration, and each degenerate value is ReleaseFast UB downstream" (core.zig:323-331).

Measured effect of the gate alone (M1 Max, Qwen3-0.6B Q4_K_S): free-form 0.83x → **0.99x**, reference-injected 0.86x → **0.98x** (SPECULATIVE.md §4). The residual 1–2% is probe overhead.

### The numbers, with all their caveats

`docs/SPECULATIVE.md` §10 — M1 Max, ReleaseFast, Qwen3-0.6B-Q4_K_S, greedy unless noted, speculative vs plain decode tok/s on the same prompt:

| Task | Speedup | Acceptance / notes |
| --- | --- | --- |
| Grounded copy (pre-tokenizer-fix prompt encoding) | **1.47x** | 70% acc, 6.6 tok/step |
| Same prompt, post-fix re-encode | 1.04x | 41% acc — see below |
| Verbatim-repetition microcase | **2.3x** | 100% acc, ~11 tok/step |
| Code edit (post-tokenizer-fix) | **1.12x** | 84.6% acc (was 0.79x pre-fix) |
| Free-form generation | 0.99x | gate auto-off; was 0.83x pre-gate |
| RAG-injected reference | 0.98x | ref source 33/68 accepted |
| Grounded copy, sampled t=0.7 (pre-fix prompt) | **1.20x** | one RNG draw per committed token holds |

Read this table the way the document insists you read it. The 2.3x headline is the verbatim-repetition *microcase* at 100% acceptance — the best case, not the typical case. The documented range is: "tasks it can't accelerate run at 0.98–0.99x (probe overhead), tasks it can run at 1.1–2.3x" (SPECULATIVE.md, intro). And the sharpest lesson in the file is the pair of rows at the top: **1.47x and 1.04x are the same prompt text.** Fixing the qwen2 pretokenizer (to token-ID parity with llama.cpp) changed the prompt's encoding; the greedy generation then diverged from the reference wording, acceptance fell 70% → 41%, and the speedup fell with it — with the gate verified non-limiting on the post-fix run. The recorded conclusion: "acceptance — and therefore speedup — is a property of (model × exact token stream), and single-prompt speedups are fragile; the durable claims are the economics table, the gate floor, and the acceptance-conditional wins" (SPECULATIVE.md §10). And the phrase to keep: "'never-a-loss' is a gate property, not a drafting property."

For context (explicitly *not* a controlled A/B): llama.cpp's `llama-lookup` reports 93% acceptance with max-draft 3 on the same grounded text — a shorter-draft, higher-acceptance operating point than Fucina's budget policy picks (SPECULATIVE.md §10).

### What composes, what refuses to

The chat layer is the turnkey wiring — `chat.Options{ .speculation = true, .spec_options = .{...}, .io = io }` — the `Conversation` owns the cascade and decoder, wires the template's stop id into `spec_options.stop_token`, and a `TurnGate` keeps the conversation SAM byte-exact across trimmed turns by filtering `observe`/`observeTopK` so the index never learns tokens the turn trim discards (`docs/SPECULATIVE.md` §9). Setting `.io` enables the live cost rescale; without it the gate runs on the static table. The same layer enforces the composition rules as loud init errors rather than silent misbehaviour (`docs/REFERENCE.md` §13.8.2):

- `stop_sequences` + `speculation` → `error.StopSequencesWithSpeculation`: the token completing a text stop sequence could be accepted mid-verify-batch, breaking the one-RNG-draw-per-committed-token contract.
- Warm starts and cross-request KV reuse + speculation → `error.SpeculationWithWarmStart` / `error.SpeculationWithReuse`: "the SpeculationIndex mirrors committed history append-only and can neither adopt nor rewind."
- Lockstep batching + speculation → `error.SpeculationWithBatch` (§13.2).
- qwen35 → structurally out of scope (§13.1): the recurrent cache cannot rewind.

Also on record in SPECULATIVE.md §11, so decisions aren't silently re-litigated: Jacobi decoding (rejected — 1.05x free-form, strictly dominated), Lookahead decoding (rejected — "1.13x measured independently at 50–120x the FLOPs of greedy decode… on CPU it is the scarce resource — the wrong trade by two orders of magnitude"), layer-skip self-speculation and tree verification (deferred, with reopen conditions). A different, complementary approach *does* exist in the tree for models that ship their own drafting head: GLM-4.5's `--mtp` and DeepSeek V4 Flash's MTP sidecar verify native multi-token-prediction drafts with one batched trunk step — lossless, measured 2.29 and 1.60 tokens per trunk forward respectively (`docs/RUNNING-MODELS.md`).

Run it yourself:

```sh
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q4_K_S.gguf \
  --prompt "..." --gen 128 --spec            # + acceptance stats
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q4_K_S.gguf \
  --prompt "..." --gen 128 --spec --spec-ref doc.txt   # RAG injection
```

## 13.6 Constrained decoding: masking the distribution, not the loop

**The problem.** You want the model's reply to *be* something: valid JSON against a schema, a match for a regex, a sentence of a grammar. Post-hoc validation wastes generations; you want the guarantee at sampling time.

**The idea.** Constrain the *distribution*, not the loop: before each sample, write `-inf` over every token the grammar forbids. The softmax renormalizes over what is left; sampling proceeds unchanged. The design question with all the leverage is *where* the mask hook lives, and Fucina's answer is: **inside the sampler**. Every decode path in the tree — plain chat, the speculative decoder's plain and verify steps, lockstep batch, hand-rolled runner loops — already funnels through `Sampler.next`, so "a single optional hook there — mask the logits row before the pipeline, observe the selected token after — gives every model family constrained decoding with zero decode-loop changes" (`docs/CONSTRAINED-DECODING.md`, intro). The rejected alternative (wiring a mask into each loop, llama.cpp's shape) is recorded in §8 of that document: Fucina has five-plus decode loops, and the sampler-hosted hook "keeps the invariant 'every path samples identically' a structural fact rather than a per-loop obligation."

**Where it lives.** The seam is `LogitProcessor` (`src/llm/logit_processor.zig`; API in `docs/REFERENCE.md` §13.6) — the same vtable pattern as `DraftSource`:

```zig
pub const LogitProcessor = struct {
    ptr: *anyopaque,
    vtable: *const VTable,
    pub const VTable = struct {
        process: *const fn (ptr, logits: []f32, history: []const usize) anyerror!void,
        commit: *const fn (ptr, token: usize) anyerror!void,
        reset: ?*const fn (ptr) anyerror!void = null,
        // structural hooks (optional; pure deterministic lookahead):
        forcedTokens: ?*const fn (ptr, buf: []usize) usize = null,
        validPrefixLen: ?*const fn (ptr, tokens: []const usize) usize = null,
    };
};
```

Installed as `Sampler.processor`, `process` mutates the `[vocab]` row before the pipeline runs — the full order being "logit processor → penalties → greedy shortcut → top-k truncation → temperature softmax → top-p → min-p → categorical draw" (REFERENCE.md §13.6) — and `commit` observes the selected token exactly once per `next`, on every exit path, greedy included. A processor that masks out everything is `error.AllTokensMasked`: a broken constraint fails loudly. You do not need a grammar engine to use the seam; the machine-verified reference snippet is a complete 20-line processor that bans odd token ids:

```zig
const OddMask = struct {
    commits: usize = 0,
    fn process(ptr: *anyopaque, logits: []f32, history: []const usize) anyerror!void {
        _ = ptr;
        _ = history;
        for (logits, 0..) |*l, tok| {
            if (tok % 2 == 1) l.* = -std.math.inf(f32);
        }
    }
    fn commit(ptr: *anyopaque, token: usize) anyerror!void {
        const self: *@This() = @ptrCast(@alignCast(ptr));
        _ = token;
        self.commits += 1;
    }
};
```

(from `docs/REFERENCE.md` §13.6 — bias lists, banned tokens, or watermarking plug into the same seam with no grammar involvement.)

**The engine.** For real grammars, `llm.llguidance.Constraint` (`src/llm/llguidance.zig`) compiles a JSON schema, regex, or Lark-variant grammar with the vendored [llguidance](https://github.com/guidance-ai/llguidance) engine — the one behind vLLM/SGLang-class structured output — into per-step token bitmasks (~10–50 µs of pure CPU mask work per token, CONSTRAINED-DECODING.md §3), off the model's ms-scale critical path. It is build-gated behind `-Dllguidance=true` (cargo builds the Rust staticlib, ~15 MB stripped); without the flag everything still compiles and `Constraint.init` returns `error.LlguidanceNotEnabled` — the seam itself is pure Zig and always available. One bridge detail that is a *correctness* feature, not bookkeeping: control tokens carry toktrie's `0xFF` special marker, so a JSON string that happens to contain the text `<|im_end|>` can never steer the sampler into emitting the actual control token and silently ending the turn mid-object (CONSTRAINED-DECODING.md §3). When the grammar completes, the mask forces the turn's stop token — termination rides the existing stop handling, no new mechanism.

**Composition with speculation — the good part.** The obvious worry: speculation samples hypothetical continuations and a grammar is stateful, so surely rejected drafts need a matcher *rollback* (llguidance even ships one). They don't, and the reason is a one-line property of the verify loop (CONSTRAINED-DECODING.md §4):

> **Every `Sampler.next` result is a committed token.**

Row *i* is sampled only after rows 0..i−1's tokens are in history, and the token sampled at row *i* is itself committed immediately — accepted draft, correction, or bonus. Since `commit` fires inside `next`, matcher state advances in lockstep with history *by construction*; there is nothing to roll back. A grammar-forbidden draft token is masked to `-inf` at its verify row, so the sampled token cannot equal it, the equality check fails, and the sample **is** the correction. Rejection-sampling semantics exactly preserved.

But naively, a constraint *kills* speculation: the cascade drafts unconstrained text, the mask rejects it, acceptance collapses to 0% and the CostGate rightly mutes the feature. The fix turns the constraint into a *drafter*: `ConstrainedSource` (`src/llm/speculative/constrained.zig`) wraps any inner `DraftSource` with a structural processor — the entire public surface is two functions (REFERENCE.md §13.9.6):

```zig
pub const ConstrainedSource = struct {
    pub fn init(processor: LogitProcessor, inner: DraftSource) ConstrainedSource
    pub fn source(self: *ConstrainedSource) DraftSource
};
```

It must sit on the *same* processor instance installed on the stream's sampler, and it works through the two optional hooks — both deterministic pure lookaheads, verified against the vendored Rust:

- **Forced spans draft themselves** (`forcedTokens`): when the grammar mandates a unique continuation — `", "population": ` after a JSON key closes — those tokens are the draft, and since the mask allows exactly that token at each row, the span verifies with **acceptance probability 1** plus the free bonus token.
- **Certainly-rejected drafts die pre-verify** (`validPrefixLen`): free-choice drafts are truncated at their first grammar-invalid token, which could only waste verify compute and drag the source-acceptance gates down.

Measured (Qwen3-0.6B-Q8_0, greedy JSON-schema chat, M1 Max — CONSTRAINED-DECODING.md §5): without the drafting layer, 10 drafted / 0 accepted (0%), gate muted after 3 steps, 1.00 tok/step; with `ConstrainedSource`, 6 drafted / 5 accepted (83%) plus bonus, gate never off, **1.24 tok/step** — "output byte-identical in both rows and to the non-speculative constrained run." The wiring is automatic: `chat.Conversation` wraps its cascade whenever the installed processor `hasStructure()`.

Honest caveats from the design record (§7): **greedy + an unbounded grammar field loops** — `{"population": <integer>}` at `--temp 0` can re-pick `0` forever, because the grammar cannot force termination inside a field whose continuation is always legal and argmax never chooses to stop (bound your fields, sample instead of greedy, keep `repeat_penalty` on); constrain the *whole* reply (`--no-think` on reasoning models, or the grammar forbids the `<think>` preamble); one `Constraint` per stream — multi-stream uses `clone()`, which shares the expensive vocab trie, and the original must outlive its clones; a shared processor across batch streams is `error.SharedBatchProcessor`. And a scope honesty note: **gemma4's constrained path is wired but not e2e-validated** — its SPM bridge is covered by gated unit tests, but no gemma GGUF was on the dev disk at the time of writing (CONSTRAINED-DECODING.md §7); qwen3 is the family with the end-to-end proof.

```sh
zig build qwen3 -Dllguidance=true -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf \
  --chat "Give me facts about Paris." --no-think \
  --json-schema '{"type":"object","properties":{"city":{"type":"string"},"population":{"type":"integer","maximum":99999999}},"required":["city","population"],"additionalProperties":false}'
```

## 13.7 KV reuse across requests

**The problem.** The OpenAI wire API is stateless: every request carries its full message history. A naive server re-prefills the whole conversation every turn — and prefill of a long history dwarfs the reply's decode.

**The idea.** Attention state is a pure function of the token prefix: two requests sharing a token prefix share, position for position, identical K/V rows. So keep previous requests' caches, match a new request to the cache with the **longest common token prefix**, rewind to the shared span with `truncate` (§13.1 again), and prefill only the tail.

> **ML note** — why is prefix matching *sound* rather than heuristic? Causal attention means position i's K/V rows depend only on tokens 0..i. Token-level LCP therefore identifies exactly the positions whose cached state is valid for the new request — no model knowledge needed, and any render divergence (a stripped reasoning block, edited history, another client entirely) is absorbed automatically by reusing less. One subtlety: the reuse is capped at `ids.len - 1` even on a perfect match, "because logits are not cached state" — the last token must be forwarded to produce the next distribution (`docs/REFERENCE.md` §13.8.2).

**Where it lives — three layers.**

*The persistence primitive*, `src/llm/kv_persist.zig`, is a self-contained systems exercise worth reading whole. Its header (kv_persist.zig:1-6):

```zig
//! Crash-safe KV-cache persistence: conversations reopen WARM across
//! process restarts, with zero re-prefill — the cost that dominates when a
//! big model decodes below 1 tok/s. The sidecar is append-only: a fixed
//! header whose record count (`nrec`) is rewritten LAST after every append,
//! so a crash mid-append leaves the old count and the file stays a
//! consistent prefix of the conversation.
```

Three functions (kv_persist.zig:195, :212, :260):

```zig
pub fn reset(io: std.Io, allocator: Allocator, path: []const u8, kv: *const KvCache,
             prefix_rows: usize) !void
pub fn appendRange(io: std.Io, allocator: Allocator, path: []const u8,
                   kv: *const KvCache, tokens: []const usize, prefix_rows: usize) !void
pub fn load(io: std.Io, allocator: Allocator, path: []const u8, kv: *KvCache) !?Loaded
```

The layout stores per position one `u32` token plus each layer's K and V row bytes (f16 or q8_0 — the record size is a pure function of the cache geometry); a geometry guard in the header makes a foreign model's or dtype's file ignored *wholesale*. Ordering rules are enforced, not documented-and-hoped: `load` asserts the cache is empty (`std.debug.assert(kv.len == 0)`, kv_persist.zig:261) and returns caller-owned token history — a `Loaded` also carrying `prefix_rows`, the row count of a token-less preloaded prefix (a trained cartridge, `docs/CARTRIDGES.md`; 0 for classic conversations) — or null when nothing is usable, stopping early at a torn tail so "the prefix stays usable"; `appendRange` requires the caller to have resumed from (or reset) the file, so "appending onto a foreign prefix must be impossible" (kv_persist.zig:191-194). On the CLI this is `--kv-save` for `--chat`/`--repl` — "essential below 1 tok/s" on the big streamed models of §13.8 (`docs/RUNNING-MODELS.md`).

> **Zig note** — the sidecar is a lesson in explicit binary I/O. Every integer is written with `std.mem.writeInt(u64, bytes[at..][0..8], nrec, .little)` — the endianness is a *parameter*, not an assumption, so the file format is identical on every host. And the crash-safety is pure write ordering: records first, the `nrec` counter last, positional writes throughout (`writePositionalAll`); a crash between the two leaves the old counter and the file remains a consistent prefix. No journal, no fsync dance in the hot path — just an ordering argument you can verify by reading 40 lines.

*The chat-layer seam* (REFERENCE.md §13.8.2):

```zig
pub const WarmState = struct { cache: KvCache, tokens: []const usize };
pub fn initWarm(ctx: *ExecContext, model: *const Model, tokenizer: *const Tok.Tokenizer,
                template: Template, options: Options, warm: WarmState) !Self
pub fn takeCache(self: *Self) KvCache
pub fn sendRenderedReuse(self: *Self, rendered: []const u8, writer: *std.Io.Writer) !usize
pub fn sendTokensReuse(self: *Self, ids: []const u32, writer: *std.Io.Writer) !usize
```

`initWarm` adopts a previous cache plus the token shadow describing its positions; `sendTokensReuse` reconciles by LCP, truncates, prefills the tail, and reports the reused span as `reused_prefix`; `takeCache` releases the cache back to the pool afterwards. Warm-reuse == fresh-stateless equivalence is proven in `chat_tests.zig`.

*The server policy*, `examples/lmserve/backend.zig` — and the entire matching policy is fourteen lines (backend.zig:182-195):

```zig
fn commonPrefix(tokens: []const usize, ids: []const u32) usize {
    var n: usize = 0;
    const cap = @min(tokens.len, ids.len);
    while (n < cap and tokens[n] == ids[n]) : (n += 1) {}
    return n;
}

/// The slot-similarity gate (llama.cpp `--slot-prompt-similarity`, default
/// 0.1): adopting a cache pays only when the common prefix covers a
/// meaningful share of the NEW prompt — otherwise a long-lived cache would
/// be destroyed to save a handful of tokens.
fn similarEnough(lcp: usize, ids_len: usize) bool {
    return lcp * 10 > ids_len;
}
```

The pool (`kv_slots`, default 1; each slot a full `--ctx` cache — "~112 KiB/position for a 28-layer/8-kv-head/128-dim f16 geometry", `docs/LMSERVER.md`) holds previous requests' caches plus token shadows. `acquireConversation` (backend.zig:464) picks the best-LCP slot when it passes the gate, otherwise recycles the LRU slot. Follow-up turns of a chat re-prefill only the last reply + new message; a non-matching request costs one full prefill, exactly as a reuse-free server would.

The **disk tier** (`--kv-cache-dir D`, `--kv-disk-slots`) reuses the exact same sidecar format: a slot about to be destroyed by an unrelated request spills to disk — but only "when the incoming request would keep less than half of it AND no stored entry already contains it" (save-on-evict, backend.zig:650-662, with containment dedup and supersede-in-place) — and is restored, zero re-prefill, when a later request matches it better than every resident slot ("the disk tier competes only when it strictly beats the pool", backend.zig:481). One hygiene detail with a comment worth reading: after an aborted turn, history can sit one un-forwarded token past the cache, so the reclaimed slot's shadow is trimmed to `cache.len` — "a slot always describes exactly the positions its cache holds" (backend.zig:608-614).

**The measured effect, honestly.** The repo does not publish an end-to-end latency multiplier for KV reuse, and this course will not invent one. What the docs commit to is the mechanism and its accounting: the reused span is reported per request as `cached_tokens` in the OpenAI usage block (`prompt_tokens_details` / `input_tokens_details`, LMSERVER.md), and what reuse eliminates is precisely the re-prefill of that span. Two composition notes from earlier sections apply: speculation cannot ride a warm start (`error.SpeculationWithReuse`), and if KV memory rather than time is your constraint, `--cache-type q8_0` halves the cache (59.5 vs 112 KiB/token on the 0.6B) as a *capacity* option — explicitly not a speed option: "decode attention is compute-bound on M1 and the dequant costs ~2.3x the attention phase at 2048 ctx, so f16 stays the default" (`docs/TRAINING.md` §10).

## 13.8 MoE expert streaming: models bigger than RAM

**The problem.** [Chapter 12](12-a-transformer-from-scratch.md) introduced mixture-of-experts: a router activates a few experts per token out of many. The *active* compute is small — but the *weights* are enormous. Qwen3-235B-A22B at Q4_K_M is 142 GB of GGUF. Your machine has 64 GB.

**The idea.** Routing is sparse and heavily skewed: per token only a handful of experts run, and across a workload some experts are vastly more popular than others. So keep the dense (always-used) weights resident, and treat the expert stacks as a *paged* resource. From `docs/RUNNING-MODELS.md`:

> `--moe-stream` keeps only the dense weights resident and reads the routed experts on demand from the GGUF through a tiered store — pinned hot set → per-layer LRU cache → `pread` — so a mixture model whose expert stacks dwarf physical RAM still loads and decodes. Output is bit-identical to the resident path (same blocks, same kernels); the price is disk reads on cache misses, i.e. tokens per second.

> **ML note** — this is virtual memory rediscovered at the granularity the router actually uses. A top-k router gives each token a *working set* of experts per layer, and empirically that working set has temporal locality (the measured hit rates below are 52–59% against a cache a fraction of the model's size) and a heavy-tailed popularity distribution (a modest hot set absorbs a disproportionate share of routings — the measured run below pins 944 experts, 10.7 GB of a 142 GB model). Caching theory then does the rest: pin the head of the distribution, LRU the middle, stream the tail. The reason generic OS paging can't do this for you is the eviction policy — the OS sees pages, not experts, and cannot know that "this expert was just routed" is the recency signal that matters.

**Where it lives.** `src/exec/expert_store.zig` — a file whose comments are a course in themselves. The resolution loop in `acquire` (expert_store.zig:926-944) is a cache in its purest form:

```zig
// Pinned and LRU hits resolve in place (pin first — a pinned expert
// may transiently also sit in the LRU after a repin); misses collect.
self.n_miss = 0;
for (self.active[0..self.n_active]) |eid| {
    if (findPinned(ls, eid)) |slot| {
        self.resolveSlot(ls, eid, slot);
        self.stats.hits += 1;
        self.stats.pin_hits += 1;
    } else if (self.findCached(ls, eid)) |slot| {
        self.clock += 1;
        slot.used = self.clock;
        self.resolveSlot(ls, eid, slot);
        self.stats.hits += 1;
    } else {
        self.miss_eids[self.n_miss] = eid;
        self.n_miss += 1;
        self.stats.misses += 1;
    }
}
```

Misses get OS readahead hints for the *whole* batch before the first synchronous read, land in reusable working-set slots, and are promoted into the LRU by slab swap after the layer computes — so cache capacity is independent of how many experts one batched prefill touches. The concurrency contract is spelled out in the header: `acquire` locks the store until `release` (the forward is sequential over ops), and between the two, worker threads read resolved pointers with no further synchronization (expert_store.zig:16-21).

Two design decisions with measured reasons, quoted from the file because they generalize far beyond MoE:

*Why `pread` and not mmap* (expert_store.zig:5-9): "the streamed tier reads with `pread` into store-owned buffers rather than mmap, so resident memory is exactly dense weights + this cache (mmap'd expert pages inflate RSS and let page-cache pressure evict semi-randomly instead of by routing recency)". You choose the eviction policy; the page cache would choose for you, badly.

*Why prefetch has its own I/O thread* (expert_store.zig:605-609): "A dedicated I/O thread drains an SPSC ring of file ranges and issues the readahead advice there: with a saturated disk queue the advice call itself BLOCKS (measured ~0.5 ms each upstream), so hinting inline would cost the forward thread more than the overlap earns. Ring full = drop: a lost hint is not an error."

> **Zig note** — this file is also the chapter's systems-Zig showcase: cross-platform syscall shims via `switch (builtin.os.tag)` (Linux goes straight to `std.os.linux`, no libc; macOS through `std.c` — including reading `/proc/meminfo` vs `sysctlbyname` for the memory budget), an SPSC ring on `std.atomic.Value`, saturating increment `ls.heat[e] +|= 1` for the popularity counters, and an `errdefer` ladder in `create` that closes exactly the file descriptors opened so far on failure (expert_store.zig:632-648).

**The learning tier.** The store's knobs are one options struct (`expert_store.zig:517-541`, comments abridged):

```zig
pub const Options = struct {
    cache_slots_per_layer: ?usize = null, // fixed LRU slots; wins over cache_bytes
    cache_bytes: ?usize = null,           // total RAM budget; default: half of available memory
    readahead: bool = true,               // WILLNEED hints for the whole miss set
    prefetch_stage_slots: usize = 64,     // staging slots for the prefetch worker's async loads
    auto_pin: bool = true,                // pin the hottest experts from the usage sidecar
    pin_bytes: ?usize = null,             // pinned-tier budget; default: half the total
    auto_pin_min_history: u64 = 5000,     // routed pairs before auto-pin trusts the history
};
```

Every acquire updates a per-expert usage histogram, persisted as a `<gguf>.experts` sidecar. At the next startup, `auto_pin` reads it and pins the hottest experts in RAM — "they are read once at startup and never evicted. The engine gets faster the more it is used" (expert_store.zig:533-534). Each stage is independently measurable: `Stats.hitRate()` splits pinned hits from LRU hits from misses, and `Stats.pilotRecall()` scores the prefetcher. The **pilot** (`--moe-pilot`) is that prefetcher: predict the *next* layer's experts from the current post-attention state and prefetch them from the background thread while the current layer computes. Measured recall of the one-layer-ahead prediction: 87.6% (30B), 90.5% (235B) — and, the docs add immediately, it "never changes output" (`docs/RUNNING-MODELS.md`). On the model side the whole feature is one opt-in field: `qwen3.model.LoadOptions{ .moe_stream = .{ .gguf_path = ..., .cache_bytes = ... } }` (`src/llm/qwen3/model.zig:155-157`), and the streamed tier plugs into the same fused MoE kernels through `fucina.MoeRhs`'s `streamed` variant — same blocks, same kernels, which is *why* the output can be bit-identical.

**The measured effect.** From `docs/RUNNING-MODELS.md` (M1 Max, 64 GB):

```sh
# Qwen3-235B-A22B Q4_K_M (142 GB, 3 split parts) on a 64 GB machine:
# point at part 1; ~24 GB peak RSS with a 20 GB expert budget.
zig build qwen3 -Doptimize=ReleaseFast -- \
  models/Qwen3-235B-A22B-Instruct-2507-Q4_K_M-00001-of-00003.gguf \
  --prompt "The three most important ideas in computer science are" \
  --gen 64 --moe-stream --moe-cache-mb=20480
```

> Measured on that 235B/64 GB configuration (M1 Max): cold 0.66 tok/s at a 52% hit rate (131 GB streamed for prefill + 24 tokens); the second run auto-pins the hottest 944 experts (10.7 GB) from the recorded usage history and reaches 0.81 tok/s at 59% — the engine gets faster the more it is used. On the 30B MoE (fits in RAM), streaming trades nothing but speed: byte-identical greedy output at half the resident-path RSS.

Sub-1 tok/s is not interactive — it is the price of running the 235B on your own machine at all, and it is exactly the regime where §13.7's crash-safe `--kv-save` earns its keep. The same machinery carries the DeepSeek family: the 164.6 GB V4 Flash decodes at a measured 1.5–3.6 tok/s warm at a 20 GB expert budget (`docs/RUNNING-MODELS.md`).

One knob deserves a highlighted warning, because it is the **only knowingly lossy knob in this chapter**: `--moe-expert-top-p=F` keeps experts per token up to cumulative router weight F and skips the rest — "30B: F=0.7 cut disk traffic 55%. Quality-traded; `F>=1` is the exact baseline" (RUNNING-MODELS.md). Everything else in this section is bit-identical to the resident path; this flag deliberately is not.

Credit where the README places it: the out-of-core design "was inspired by colibri's design; Fucina's implementation is independent, streaming ggml quants over the fused kernels" (README.md, Acknowledgments — JustVugg's colibri).

## 13.9 The GPU seam, honestly

Fucina has exactly two CPU backends — `scalar`, the reference oracle, and `native`, the fast one — and the GPU is *neither a third backend nor a graph compiler*: Metal and CUDA are callable accelerators inside the native backend. "An eligible dense GEMM/GEMV … is validated and submitted when the eager op is called; no operation is recorded for later planning, fusion, or replay" (`docs/GPU-OFFLOAD.md`). A completion token attached to the output storage removes the per-call host stall — but "the token is completion metadata, not a compute node… the runtime remains eager and graphless"; host visibility is forced only where it is actually required (`Tensor.data*`, `item`, copies, any CPU kernel touching the storage). Offload decisions are measured per-shape **work gates** (env-tunable, `FUCINA_GPU_MIN_WORK*`): a small GEMM stays on CPU because a launch plus a host fence cannot beat Apple AMX at 256³, and the gates encode where the measured crossover actually sits per format and residency — on M1 Max the conservative cold single-op floor is 2^32 work units (isolated 1024³ trials "were DVFS-sensitive crossovers, while 2048×2048×1024 won consistently"), while CUDA distinguishes a device-resident RHS (2^27) from the transient-RHS PCIe floor (2^33), and residency can flip the verdict entirely: a 1×4096×1024 *resident* f16 GEMV measured 18.3 µs on the RTX host versus 77.4 µs on CPU, while nonresident decode of the same shape is refused outright (GPU-OFFLOAD.md, Gates).

The honest paragraph the numbers demand: the documented CUDA WMMA kernels "raised queued throughput by 11–37%" over the scalar CUDA kernel across the non-decode quantized shapes in the suite — and the same document immediately pins what that claim is: "The 11–37% claim is therefore deliberately an offloaded-op throughput claim, not a claim that every end-to-end prompt gains that amount." End to end, Qwen3-0.6B Q6_K at pp128 moved from 1308 to 1332 tok/s — **1.8%** — because host attention, PCIe copies (~0.18 ms per 2 MiB), and other CPU boundaries absorb the op-level gain; Qwen3-4B pp128 was flat within noise (GPU-OFFLOAD.md, Verification and measurements). Where residency aligns, model-level wins do exist and are recorded with their own caveat: Qwen3-0.6B-Q5_K_M warm prefill 503.3→770.3 tok/s at 32 tokens on CUDA, opt-in decode 62.85→92.30 tok/s — *tolerance-equivalent*, not bit-identical, with a recorded counterexample prompt whose top-two logit margin reversed. Amdahl's law is not negotiable; Fucina's response is to gate per-op on measurements and to keep the CPU path the correctness oracle. If you want the full protocol, [Chapter 6](06-going-fast-on-cpus.md) owns the backend story and `docs/GPU-OFFLOAD.md` is the design record.

## 13.10 Serving it: lmserve

All of these tricks meet the network in `zig build lmserve` (`examples/lmserve/main.zig` + `examples/lmserve/`) — an OpenAI-compatible server over the same engine, speaking both wire dialects (`POST /v1/chat/completions`, `POST /v1/responses`), verified end-to-end against openai-python 2.45.0 (`docs/LMSERVER.md`). It is "an example, not a library surface": a model family integrates through one small two-function `Backend` vtable (`validate` on the connection thread, `generate` on the worker), and any family served by `llm.chat.Conversation` gets the whole adapter for free.

Its threading design is a direct consequence of Chapter 6's execution model — "accept concurrently, generate sequentially":

> `ExecContext` is single-threaded by contract and one forward pass already fork-joins across every performance core …, so the server does not try to overlap generations. (LMSERVER.md)

Connection threads parse, validate, and queue; **one** inference worker owns the `ExecContext` and the model. The queue is bounded, and overflow is `429` + `retry-after` — with an adjudication attached: "llama.cpp defers unboundedly; a bounded queue is the honest failure mode for a sequential worker." Client disconnects cancel queued jobs and abort in-flight generation at the next token.

The tricks of this chapter surface as server features: §13.7's slot pool and disk tier are `--kv-slots`, `--kv-cache-dir`, `--kv-disk-slots`, with reuse reported as `cached_tokens`; §13.6's constraints arrive as `response_format`/`json_schema` (plus `regex`/`lark` extensions), with base constraints LRU-cached per grammar source and `clone()`d per request — `Constraint.init` walks the full vocab to build the token trie, too expensive per request, while a clone shares it. Unmappable parameters are *rejected with a 400/501 naming the offending param, never silently dropped* (`tools` beyond auto/none, `n > 1`, `logprobs`, `logit_bias`, …). §13.2's lockstep batching is explicitly future work in the scheduler ("no mid-flight joins").

Even the streaming layer carries the chapter's correctness-first habit: SSE deltas are UTF-8-boundary-safe — "a token ending mid-code-point carries into the next frame instead of corrupting the JSON" — responses streams emit the full event skeleton the OpenAI SDKs' state machine requires, and a request that fails before producing anything still gets a plain JSON error with a proper status code because streaming starts lazily on the first delta (LMSERVER.md). Verification status, stated with the repo's usual candor: qwen3 proven end-to-end (openai-python SDK suite, constrained decoding, concurrency, cancellation, shutdown — 2026-07-12, Qwen3-0.6B-Q8_0); nanochat proven against its goldens; gemma4 and diffusion-gemma compile+unit-verified, no GGUF of either on local disk when the example landed.

```sh
zig build lmserve -Dllguidance=true -Doptimize=ReleaseFast -- \
  models/Qwen3-0.6B-Q8_0.gguf --port 8080
# then point any OpenAI client at http://localhost:8080/v1
```

The through-line of this chapter, one last time: `truncate` made the cache rewindable; rewind made verification and reuse possible; determinism made verification lossless; measurement made every trick either provably free or provably profitable — and when a trick cannot win (MoE verify economics, free-form drafting, small GEMMs on GPU), a gate says so with numbers instead of hope. [Chapter 14](14-the-low-bit-frontier.md) pushes the other lever from §13.1 — fewer bytes per weight — to its research frontier: ternary.

## What you now know

- The decode loop's invariant (`history.len == kv.len + 1`) and the one-integer-store rewind (`KvCache.truncate`) that every trick in this chapter is built on — and why qwen35's recurrent cache structurally opts out of speculation.
- Decode is weight-bandwidth-bound, so batching N streams into one m=N pass buys 3.19–3.25x aggregate throughput at 8 streams (M1 Max snapshot, `docs/BENCHMARK.md`), with bitwise-vs-~1e-6 numerics boundaries documented per m.
- Draft-model-free speculative decoding: deterministic retrieval drafters (suffix automaton with cursor-before-extend self-match exclusion, frozen RAG indexes, a 4.6 MiB Token-Recycling matrix) verified by one batched forward; losslessness = same distribution always, same sample stream when logits match bitwise; one RNG draw per committed token as an enforced contract.
- The verify economics (`a + 1 > C(k)`; measured non-monotonic cost table; MoE's ≈93% break-even) and the CostGate — hybrid static-table + clamped-EWMA, hysteresis, exponential backoff — that makes the feature never-a-loss: 0.98–0.99x floor, 1.1–2.3x task-dependent wins, and "never-a-loss is a gate property, not a drafting property".
- Speedup is fragile, economics are durable: the same prompt went 1.47x → 1.04x when a tokenizer fix changed its encoding.
- Constrained decoding as distribution surgery: one `LogitProcessor` hook inside the sampler constrains every decode path; llguidance compiles schemas/regex/Lark to ~tens-of-µs masks; composition with speculation needs no rollback (every sampled token commits) and `ConstrainedSource` turns the grammar into a drafter — acceptance 0% → 83%, byte-identical output.
- Cross-request KV reuse: token-level LCP + a similarity gate over a slot pool, a crash-safe append-only sidecar (counter written last) as the disk tier, reuse reported as `cached_tokens` — and no end-to-end latency multiplier claimed, because the repo publishes none.
- Out-of-core MoE: pinned set → per-layer LRU → `pread` (not mmap, for RSS and eviction-policy reasons), a persisted usage histogram that auto-pins the hot set, router-lookahead prefetch on a dedicated I/O thread — a 142 GB model decoding bit-identically on a 64 GB machine at 0.66→0.81 tok/s, with `--moe-expert-top-p` as the one knowingly lossy knob.
- GPU offload is an eager, graphless, work-gated seam: op-level gains (11–37%) are not end-to-end gains (pp128 moved 1.8%), and the docs say so themselves.
- `lmserve` wraps the same engine in the OpenAI wire protocol: accept concurrently, generate sequentially, bounded queue, explicit 400/501s — and the tricks surface as flags.

## Explore the source

- `src/llm/speculative/core.zig` — the normative losslessness header, the verify row loop, the CostGate with its inline tests; the single most teaching-dense file of the chapter.
- `src/llm/speculative/sam_index.zig` — the header's proof sketch of self-match exclusion, the 10-line greedy automaton descent, poisoning and freezing.
- `src/llm/speculative/cascade.zig` and `src/llm/speculative/constrained.zig` — source composition with per-source muting; the ~100-line bridge that makes grammars draft.
- `src/llm/logit_processor.zig` + `src/llm/sampler.zig` — the seam every decode path funnels through.
- `src/llm/kv_persist.zig` — 295 lines of crash-safe binary file design; read it as a systems exercise.
- `examples/lmserve/backend.zig` — the slot pool, similarity gate, disk tier, and constraint cache; the server policy in one file.
- `src/exec/expert_store.zig` — the tiered store, the pilot thread, the platform shims; the best-commented systems code in the tree.
- `docs/SPECULATIVE.md`, `docs/CONSTRAINED-DECODING.md`, `docs/LMSERVER.md`, `docs/RUNNING-MODELS.md`, `docs/GPU-OFFLOAD.md` — the design records, adjudications included.

## Exercises

1. **(Easy)** Run `--spec` on two prompts: one that asks the model to copy a paragraph you provide, one free-form ("write a story"). Read the `spec: steps=… (verify …, fallback …, off …)` stats line and the per-source summary. Explain each field's movement using §13.5 — in particular, why the free-form run's `off` count is the gate doing its job, and what the ~0.99x floor is buying you.
2. **(Easy)** Extend the course toy decoder (§13.3) with a `stop_token`: when a committed token equals it, the row loop must break immediately — bonus row included. Write a test that proves the streams with and without drafting still match up to and including the stop. You have reproduced the RNG-accounting rule of `core.zig:32-39` in miniature.
3. **(Medium)** Add a Token-Recycling drafter to the toy: a `vocab × K` table updated with each step's "top-K" (for the toy model, the true successor plus K−1 decoys), drafting by top-1 chain walk. Measure tokens/step against `noDraft` over 1000 tokens. Then make the model non-Markov (`targetNext(prev, cur)`) and watch acceptance change — you are rediscovering why acceptance is a property of (model × token stream).
4. **(Medium)** Write a `LogitProcessor` that bans a fixed token list, following the `OddMask` shape from `docs/REFERENCE.md` §13.6, and install it via `chat.Options.logit_processor` on a qwen3 chat. Verify: (a) the banned tokens never appear; (b) greedy output with `.speculation = true` and `false` is identical (the §13.6 no-rollback argument in action). Then make your processor expose `forcedTokens` for some state and confirm speculation starts drafting your forced spans.
5. **(Hard)** Build a two-slot LCP pool over the toy decoder: token shadows, `commonPrefix`, the `lcp * 10 > ids_len` gate, LRU eviction, and a "shadow trimmed to cache.len" reclaim (backend.zig:608-614 explains why). Drive it with three interleaved "conversations" and assert (a) follow-up turns re-prefill only their tails, (b) an unrelated request evicts the LRU slot, not the best-matching one, (c) outputs are identical to a pool-free run. You will have re-derived the heart of `examples/lmserve/backend.zig`.

---

[Previous: A transformer from scratch](12-a-transformer-from-scratch.md) · [Next: The low-bit frontier](14-the-low-bit-frontier.md)
