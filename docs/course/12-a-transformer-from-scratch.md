# Chapter 12 — A transformer from scratch

*Part V — Language models*

This is the chapter people came for, and it is honest work: genuinely the
hardest chapter so far. It walks, line by line, through everything between
typing a question and watching a language model answer it — tokenization,
twenty-eight identical transformer blocks, rotary positions, grouped-query
attention, a KV cache, a sampler, a chat loop — all in one Zig file family
you can read top to bottom. [Chapter 11](11-model-files-and-quantization.md)
gave us the weights: a GGUF file, memory-mapped, its tensors resident in
their source precision. This chapter turns them into a machine that talks.

Everything is grounded in a real model: **Qwen3-0.6B**, implemented in
`src/llm/qwen3/model.zig` (~1,800 lines). Around it sits the `fucina_llm`
module, which — in the words of `docs/REFERENCE.md` §13 — "contains
everything a transformer inference/fine-tuning runner needs that is not a
tensor op: GGUF-to-weight binding, KV caching, tokenizers, sampling, SFT data
plumbing, multi-turn chat, and lossless draft-free speculative decoding."
The tensor ops you know from [Chapter 5](05-the-operation-library.md); this
chapter composes them into a transformer. Two caveats: Fucina's public API
is young and explicitly unstable — every signature below is today's code,
not a frozen contract — and nothing here is hand-waved, so some sections
(RoPE, the prefill/decode economics, the comptime chat contract) reward a
second read. Every claim is pinned to a file and line you can open.

## 12.1 The whole machine in nine numbers

Strip away the mythology and a transformer's architecture is a struct of
integers. Here is Qwen3-0.6B, verbatim (*from `src/llm/qwen3/model.zig:71`*):

```zig
pub fn qwen3_0_6b() Config {
    return .{
        .vocab_size = 151_936,
        .hidden_size = 1024,
        .intermediate_size = 3072,
        .num_layers = 28,
        .num_attention_heads = 16,
        .num_key_value_heads = 8,
        .head_dim = 128,
        .rms_norm_eps = 1e-6,
        .rope_theta = 1_000_000,
    };
}
```

Read it as a bill of materials:

- **`vocab_size = 151_936`** — the model knows this many distinct tokens.
  Its first layer is a lookup table with this many rows; its last layer
  produces this many scores.
- **`hidden_size = 1024`** — every token, at every point inside the model,
  is a vector of 1024 floats. This is the width of the *residual stream*
  (§12.3.7).
- **`num_layers = 28`** — the same block, repeated 28 times.
- **`num_attention_heads = 16`, `num_key_value_heads = 8`, `head_dim = 128`**
  — attention runs as 16 parallel heads of 128 dimensions each, but only 8
  distinct key/value heads. That asymmetry *is* grouped-query attention
  (§12.3.5), and every projection size derives from these three numbers:
  the query projection is 16 × 128 = 2048 wide, the key/value projections
  8 × 128 = 1024 — pinned by a unit test at `model.zig:1816-1821`.
- **`intermediate_size = 3072`** — the feed-forward block's inner width;
  **`rms_norm_eps`** and **`rope_theta`** are numerical constants you will
  meet in §12.3.2 and §12.3.4.

In practice nobody types these numbers: `Config.fromGguf`
(`model.zig:89-118`) reads them from the GGUF's own metadata under the
`general.architecture` prefix (`qwen3.block_count`, …), so any Qwen3-family
size — 0.6B through 8B, plus the MoE variants — loads without hardcoding.
The helpers live in `src/llm/gguf_meta.zig`, with one design point worth
noticing: qwen3's loader treats a present-but-zero key as missing
(`.reject_zero`, `model.zig:139-143`), because every qwen3 config integer
is structurally positive — a zero can only be a broken file.

The data flow we are about to build, end to end:

```
"What is the capital of France?"
        │  tokenizer (§12.2)
        ▼
[t0, t1, t2, ...]                               token ids (integers < 151_936)
        │  embedding lookup (§12.3.1)
        ▼
[seq, 1024] f32                                 the residual stream
        │  28 × (attention block + FFN block)   (§12.3)
        ▼
[seq, 1024] → final RMSNorm → lm_head
        ▼
[1, 151936] f32                                 logits
        │  sampler (§12.6)
        ▼
next token id ──► tokenizer.decode ──► "Paris" (eventually)
```

Then the loop closes: the new token is appended and the model runs again,
one token at a time, until a stop condition. Everything else in this chapter
— the KV cache, prefill vs decode, chat templates — is about making that
loop correct and fast.

## 12.2 Tokenization: the model never sees text

A language model consumes integers, not characters. The map between them is
the tokenizer — worth building first because it needs no tensors at all,
just strings, hash maps, and one clever trick. Fucina's lives in
`src/llm/tokenizer.zig`, and its module comment states the design in one
breath (*from `src/llm/tokenizer.zig:1-2`*): a "native byte-level BPE
tokenizer (GPT-2 / Qwen family), built from a model's own GGUF metadata —
no external tokenizer dependency, no per-model hardcoding." The vocabulary
(`tokenizer.ggml.tokens`) and merge rules (`tokenizer.ggml.merges`) come
out of the same GGUF file as the weights; the tokenizer copies the bytes it
needs, so it stays valid after the file is freed.

### The byte trick

"Byte-level" is the load-bearing adjective. A word-level tokenizer meets a
word it has never seen and needs an `<unk>` escape hatch. Byte-level BPE
never does, because its atoms are the 256 possible bytes — every input, in
any language, in any encoding, even binary garbage, is *some* byte sequence.
There is one wrinkle: the vocabulary is stored as printable strings, and
many bytes are not printable. GPT-2's solution, faithfully implemented here,
maps every byte to a printable Unicode stand-in (*from
`src/llm/tokenizer.zig:588-600`, trimmed*):

```zig
/// GPT-2 byte→unicode: printable bytes map to themselves; the rest shift into
/// the U+0100+ range so every raw byte is representable in the vocabulary.
fn gpt2ByteToUnicode(byte: u8, buf: *[4]u8) usize {
    const cp: u21 = switch (byte) {
        '!'...'~', 0xA1...0xAC, 0xAE...0xFF => byte,
        else => @as(u21, 256) + @as(u21, switch (byte) {
            0...0x20 => byte,
            0x7F...0xA0 => byte - 0x7F + 33,
            0xAD => 33 + 34,
            else => byte,
        }),
    };
    // ... UTF-8-encode cp into buf
```

So a space (0x20) becomes `Ġ` (U+0120) — which is why, if you ever dump an
LLM vocabulary, half the tokens seem to start with `Ġ`: that is "word
preceded by a space". A repo test round-trips all 256 bytes through this map
and back (`tokenizer.zig:1262-1272`).

> **Zig note** — That `switch` is a *range-case switch* on a `u8`:
> `'!'...'~'` covers the whole printable-ASCII span in one arm, and the
> compiler checks the arms are exhaustive and non-overlapping. Slices in,
> lengths out, no allocation.

### Merges: the entire algorithm

BPE ("byte-pair encoding") is learned offline — during training, the most
frequent adjacent pair of symbols is repeatedly fused into a new symbol, and
the *order* of those fusions is saved as a ranked list of merge rules. At
inference time the algorithm is almost embarrassingly simple: start from
single bytes, and repeatedly merge the adjacent pair with the
lowest-numbered (earliest-learned) rule until no rule applies. The repo's
implementation is `applyMerges` (`src/llm/tokenizer.zig:468-495`) — 28
lines, whose doc comment is the entire specification: "Repeatedly merge the
adjacent symbol pair with the lowest merge rank." To make it concrete, here
is a self-contained toy you can run today — **course code**, not repo code
(compile-checked with `zig test`; the classic `lower` example):

```zig
const std = @import("std");

/// Course code — a toy byte-level BPE encoder. The real thing is
/// src/llm/tokenizer.zig; this keeps only the algorithm.
const MiniBpe = struct {
    vocab: []const []const u8, // token id = index into this list
    merges: []const []const u8, // "a b" rules; rank = index (lower wins)

    fn tokenId(self: MiniBpe, bytes: []const u8) ?u32 {
        for (self.vocab, 0..) |tok, i| {
            if (std.mem.eql(u8, tok, bytes)) return @intCast(i);
        }
        return null;
    }

    fn mergeRank(self: MiniBpe, a: []const u8, b: []const u8) ?u32 {
        for (self.merges, 0..) |rule, rank| {
            const space = std.mem.indexOfScalar(u8, rule, ' ') orelse continue;
            if (std.mem.eql(u8, rule[0..space], a) and std.mem.eql(u8, rule[space + 1 ..], b))
                return @intCast(rank);
        }
        return null;
    }

    fn encode(self: MiniBpe, allocator: std.mem.Allocator, text: []const u8, out: *std.ArrayList(u32)) !void {
        // 1. Start with one symbol per byte. (The real tokenizer first maps
        //    every byte to a printable Unicode stand-in; our toy vocabulary
        //    is plain ASCII, so we skip that step.)
        var symbols: std.ArrayList([]const u8) = .empty;
        defer symbols.deinit(allocator);
        for (0..text.len) |i| try symbols.append(allocator, text[i .. i + 1]);

        // 2. Repeatedly merge the adjacent pair with the lowest rank.
        while (symbols.items.len > 1) {
            var best_rank: u32 = std.math.maxInt(u32);
            var best_pos: usize = 0;
            for (0..symbols.items.len - 1) |i| {
                if (self.mergeRank(symbols.items[i], symbols.items[i + 1])) |rank| {
                    if (rank < best_rank) {
                        best_rank = rank;
                        best_pos = i;
                    }
                }
            }
            if (best_rank == std.math.maxInt(u32)) break; // no rule applies

            // Adjacent symbols are contiguous slices of `text`, so the
            // merged symbol is just a wider slice — no allocation.
            const a = symbols.items[best_pos];
            const b = symbols.items[best_pos + 1];
            symbols.items[best_pos] = a.ptr[0 .. a.len + b.len];
            _ = symbols.orderedRemove(best_pos + 1);
        }

        // 3. Emit token ids.
        for (symbols.items) |sym| {
            try out.append(allocator, self.tokenId(sym) orelse return error.UnknownSymbol);
        }
    }
};

test "toy BPE reproduces merge order" {
    const allocator = std.testing.allocator;
    const bpe = MiniBpe{
        .vocab = &.{ "l", "o", "w", "e", "r", "lo", "low", "er", "lower" },
        .merges = &.{ "l o", "lo w", "e r", "low er" },
    };
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(allocator);

    try bpe.encode(allocator, "lower", &out);
    try std.testing.expectEqualSlices(u32, &.{8}, out.items); // fully merged

    out.clearRetainingCapacity();
    try bpe.encode(allocator, "wool", &out); // no rule matches: byte tokens
    try std.testing.expectEqualSlices(u32, &.{ 2, 1, 1, 0 }, out.items);
}
```

The real tokenizer's `encodeChunk` (`tokenizer.zig:429-465`) has the same
three-phase shape — map bytes to symbols, `applyMerges`, look up ids — plus
a fallback that decomposes an unknown merged symbol into its byte-level
characters, each of which *is* in the vocabulary of a valid GPT-2 model.
One difference is instructive: our toy merges by widening a slice into
`text`, while `applyMerges` owns each symbol on the heap and must
`allocator.free` both halves of every merge — explicit ownership, visible
in the code, leak-checked by the test allocator. `docs/REFERENCE.md` §13.5
carries a machine-verified twin of this experiment against the real
`Tokenizer` (the snippet gate from [Chapter 16](16-the-craft.md) runs it in
CI): `initFromParts` with the four-token vocabulary
`{ "<|im_end|>", "h", "i", "hi" }` and the single merge rule `"h i"`
encodes `"hi<|im_end|>"` to `{ 3, 0 }` — merge applied, special marker
resolved to one id — and decodes it back verbatim.

### The pretokenizer, and the parity bar

One detail stands between "correct algorithm" and "correct tokenizer": BPE
does not run over the whole text at once. The text is first split into
*chunks* (roughly word-shaped pieces) by a pretokenizer, and merges never
cross a chunk boundary. Move a boundary and you get different merges,
different ids — and a model that sees subtly different input than the one it
was trained as.

The Qwen2-family pretokenizer is officially a regex:

```
(?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N}
|  ?[^\s\p{L}\p{N}]+[\r\n]*  | \s*[\r\n]+ | \s+(?!\S)   | \s+
```

Fucina implements it as `qwen2ChunkEnd` (`tokenizer.zig:824-888`), "a
faithful port of llama.cpp's hand-rolled codepoint loop" for that regex,
backed by generated Unicode category tables (`unicode_categories.zig`). The
doc comment (`tokenizer.zig:644-649`) calls out the load-bearing details:
digits split *one per chunk* (`\p{N}` has no quantifier), punctuation runs
absorb trailing newlines, and a whitespace run before a word donates its
last space to that word. In-repo tests pin exactly these behaviours (*from
`src/llm/tokenizer.zig:1174-1181`*):

```zig
test "qwen2 pretokenizer: digits split one per chunk (\\p{N} singleton)" {
    try expectChunks("1234567", &.{ "1", "2", "3", "4", "5", "6", "7" });
    try expectChunks("3.14", &.{ "3", ".", "1", "4" });
    // ... (more cases, including arabic-indic Unicode digits)
}
```

Why so much ceremony over a splitter? Because the correctness bar for the
whole tokenizer is **token-ID-exactness**: `docs/REFERENCE.md` §13.5 states
that "on valid UTF-8 input it chunks and encodes **token-ID-exact** against
llama.cpp for qwen2-pre models (malformed UTF-8 is the one documented
deviation)". Not "produces similar text" — the *same integers*, id for id.
The runner exposes this as an oracle: `--tokenize FILE` prints one token id
per line with no weights loaded, ready to diff against `llama-tokenize`.
The single documented deviation is spelled out at `tokenizer.zig:651-657`:
on *invalid* UTF-8, Fucina classifies each undecodable byte as its own
U+FFFD while llama.cpp decodes leniently, so chunk boundaries and output
bytes can differ — on malformed input only.

Two more behaviours you will rely on later:

- **Special-token markers.** `encode` resolves special tokens to their
  single id *atomically* (`encodeWithSpecials`, `tokenizer.zig:296-329`),
  matching llama.cpp's partitioning: with GGUF `token_type` metadata the
  markers come from a special-token cache (longest first); without one,
  `<|...|>` spans are resolved against the vocabulary. That is how
  `<|im_end|>` stays one token; a `<|` that opens no known marker is left
  for normal pretokenization.
- **`encode` vs `encodeRaw`.** `encode` applies the model's BOS/EOS policy;
  `encodeRaw` does not, because chat "templates own structure"
  (`docs/REFERENCE.md` §13.5) — the template decides every structural
  token.

And one honesty mechanism: a GGUF declaring a pretokenizer this module does
not implement is *not* silently mis-tokenized — encoding proceeds with
qwen2 rules, the id is recorded in `pre_mismatch`, and a warning is logged
once (`tokenizer.zig:50-58, 134-153`). Parity is then explicitly not
guaranteed, and you were told.

### Streaming decode: the emoji problem

Generation emits one token at a time, and a token's bytes can end mid-way
through a multi-byte UTF-8 character — the next token completes it. Print
naively and your terminal shows garbage mid-emoji. `StreamDecoder`
(`tokenizer.zig:537-572`) buffers the incomplete tail (*from
`src/llm/tokenizer.zig:555-564`*):

```zig
/// Decode `id` and write any now-complete UTF-8 to `writer`.
pub fn push(self: *StreamDecoder, allocator: Allocator, id: u32, writer: *std.Io.Writer) !void {
    try self.tokenizer.decodeTokenInto(allocator, id, &self.pending);
    const emit = completeUtf8Prefix(self.pending.items);
    if (emit == 0) return;
    try writer.writeAll(self.pending.items[0..emit]);
    const tail = self.pending.items.len - emit;
    if (tail > 0) std.mem.copyForwards(u8, self.pending.items[0..tail], self.pending.items[emit..]);
    self.pending.shrinkRetainingCapacity(tail);
}
```

> **Zig note** — The sink is `*std.Io.Writer`, Zig 0.16's writer interface.
> The same `push` streams to stdout in the REPL and to an SSE response in
> the `lmserve` server ([Chapter 13](13-inference-tricks.md)) — the
> abstraction costs one vtable indirection and buys every sink at once.

## 12.3 One forward pass, in code order

Now the model. Here is the heart of `forwardStep` — the function that does
*all* the transformer work, whether you hand it a 500-token prompt or one
token (*from `src/llm/qwen3/model.zig:512-543`, profiling and MoE prefetch
trimmed*):

```zig
var x = try self.token_embedding.getRowsAs(ctx, token_ids, .embed);
errdefer x.deinit();

const cfg = self.config;
for (self.layers, 0..) |*layer, layer_i| {
    const last_query_only = last_only and layer_i + 1 == cfg.num_layers and token_ids.len > 1;
    x = try ctx.replace(x, attentionBlock(ctx, io, cfg, layer, &x, &rope_table, self.kv_head_for_head, last_query_only, profile, kv, layer_i));
    x = try ctx.replace(x, ffnBlock(ctx, io, cfg, layer, &x, profile));
}
kv.advance(token_ids.len);

var final_norm = try x.rmsNormMul(ctx, .embed, &self.output_norm, self.config.rms_norm_eps);
defer final_norm.deinit();
x.deinit();

// last_only keeps just the final row for the vocab projection; the
// all-logits entry projects every position.
const keep_from = if (last_only) final_norm.dim(.seq) - 1 else 0;
var head_in = try final_norm.narrow(ctx, .seq, keep_from, final_norm.dim(.seq) - keep_from);
defer head_in.deinit();

const logits = try self.output.linearSeq(ctx, &head_in, .embed, .vocab);
```

Embedding lookup, a loop over identical layers, a final norm, one last
matmul. That is the entire architecture; the rest of this section walks each
piece in the order the code runs it.

> **Zig note** — `x = try ctx.replace(x, ...)` is Fucina's idiom for "the
> new tensor replaces the old one". Its second parameter is an *error
> union* on purpose (`src/exec.zig:355-375`): the block call is evaluated
> by the caller, and on error the old `x` is NOT consumed — the binding and
> its `errdefer` stay valid — while on success the old tensor is released
> and the new one returned for rebinding. One idiom keeps the residual
> stream a single live tensor across 56 block calls, error-safe, without
> 56 `defer`s.

### 12.3.1 Embedding lookup

`token_embedding` is a `LinearWeight` — the same 29-arm `union(enum)` over
weight formats you met in [Chapter 11](11-model-files-and-quantization.md) —
and `getRowsAs` (`src/llm/weights.zig:832`) gathers one row per token id
into a fresh `[seq, embed]` f32 tensor. Because the gather is a method on
the weight union, a q8_0 or f16 embedding table works exactly like an f32
one: rows dequantize or widen on the fly ("nothing is widened to f32 at
load time", `docs/REFERENCE.md` §13.2).

A small elegance hides at load: many models *tie* the output projection to
the embedding table (one matrix maps ids to vectors on the way in and
vectors to scores on the way out). The loader expresses the fallback chain
as one labeled block (*from `src/llm/qwen3/model.zig:214-221`, comment
trimmed*):

```zig
var output = blk: {
    if (try ptqtp_gguf.maybeLoadPlanes(ctx, file, "output.weight", config.vocab_size, config.hidden_size)) |planes| break :blk planes;
    if (file.maybeGet("output.weight")) |info| break :blk try LinearWeight.load(ctx, info, config.vocab_size, config.hidden_size);
    break :blk try token_embedding.cloneView(ctx);
};
```

No `output.weight` tensor in the file? Then the lm_head *is* the embedding,
shared via a reference-counted `cloneView` — no copy.

### 12.3.2 RMSNorm

Before each sub-block, the stream is normalized. RMSNorm divides each token
vector by its root-mean-square and multiplies by a learned per-channel
weight:

```
y_i = x_i / sqrt(mean(x²) + eps) · w_i
```

In code it is one fused call: `x.rmsNormMul(ctx, .embed, &layer.attn_norm,
eps)` with `eps = 1e-6` from the config. The `eps` guards against dividing
by zero on a (theoretically) all-zero vector; the learned `w` lets the model
undo or re-scale the normalization per channel.

> **ML note** — Why normalize? Deep residual networks drift: the stream's
> magnitude grows layer by layer, destabilizing training. Pre-normalization
> (norm *before* each sub-block; the residual add in §12.3.7 bypasses it)
> keeps every sub-block's input at unit scale. RMSNorm is LayerNorm minus
> the mean-subtraction and bias — cheaper, and empirically just as good for
> transformers.

Qwen3 adds its signature stabilizer: **per-head q/k normalization**. After
the QKV projection, each head's 128-dimensional query and key get their own
RMSNorm with learned length-128 weights (`q_norm`/`k_norm`, loaded at
`model.zig:872-876`). You will see it fused into the RoPE call below.

### 12.3.3 QKV projections and heads

Attention starts with three matmuls: the normalized stream is projected to
queries (2048 wide), keys and values (1024 wide each). Then `split` reshapes
the flat projections into heads (*from `src/llm/qwen3/model.zig:1192-1197`*):

```zig
var q3 = try qkv_linear.q.split(ctx, .q, .{ .head, .d }, .{ config.num_attention_heads, config.head_dim });
defer q3.deinit();
var k3 = try qkv_linear.k.split(ctx, .k, .{ .kv_head, .d }, .{ config.num_key_value_heads, config.head_dim });
defer k3.deinit();
var v3 = try qkv_linear.v.split(ctx, .v, .{ .kv_head, .d }, .{ config.num_key_value_heads, config.head_dim });
defer v3.deinit();
```

The tagged-tensor types from [Chapter 4](04-axes-with-names.md) earn their
keep here: `[seq, q]` becomes `[seq, head, d]` and `[seq, k]` becomes
`[seq, kv_head, d]` — the *names* record that queries and keys have
different head axes, and a later contraction against the wrong one is a
compile error.

One performance decision is visible in the types. `AttentionProjection`
(`model.zig:964-990`) is a `union(enum) { separate, fused }`: at load time,
`weights.fuseLinear` tries to concatenate the q, k, and v matrices into one
`[2048+1024+1024, 1024]` matrix so three matmuls become one wider GEMM (one
pass over the activations). The fusion *declines gracefully* — it returns
`!?LinearWeight`, and `null` means "mixed formats: parts untouched, use
them individually" (`docs/REFERENCE.md` §13.2); the forward switches on the
union. The FFN's gate/up pair gets the same treatment
(`model.zig:1108-1130`).

> **Zig note** — This is the recurring Fucina pattern for "optimization
> that may not apply": encode both outcomes in a tagged union, decide once
> at load, and let `switch` make forgetting a case impossible. No flags, no
> "did we fuse?" bookkeeping at call sites.

### 12.3.4 RoPE: positions as rotations

Attention as defined so far is *permutation-invariant*: shuffle the input
tokens and the scores shuffle with them — nothing in `q·k` knows that
"France" came after "capital of". Positions must be injected, and Rotary
Position Embedding (RoPE) injects them with a trick worth understanding
properly, once.

**The mechanism.** Take a head's 128-dimensional query vector and view it as
64 two-dimensional pairs — in the "half" layout used here, channel `i` pairs
with channel `i + 64` (NEOX/Llama convention; `docs/REFERENCE.md` §4.12).
For a token at absolute position `p`, rotate pair `i` by the angle

```
angle(p, i) = p / theta_base^(2i / d)        d = head_dim, theta_base = 1e6
```

Pair 0 spins fast (one radian per position); pair 63 barely moves. The 64
hands turning at 64 different speeds encode the position, the way a
multi-hand clock encodes the time.

**Why rotations, though?** Because of what happens in the dot product.
Rotation is linear and preserves lengths; rotating both operands of a 2-D
dot product rotates their *angle difference* only:

```
rot(q, a) · rot(k, b) = |q||k| cos(θ_qk + a − b)
```

The score depends on `a − b` alone. Apply this per pair with `a = p_q · f_i`
and `b = p_k · f_i` and the attention score between a query at position
`p_q` and a key at position `p_k` depends only on `p_q − p_k` — the
**relative** position — even though each token was rotated by its own
**absolute** position. The model gets translation-invariant position
awareness ("three tokens back" means the same thing everywhere in the
sequence) from a purely local, per-token operation.

Do not take the algebra on faith — here it is as a runnable test, **course
code** (compile-checked with `zig test`):

```zig
const std = @import("std");

/// Course code — RoPE on one 2-channel pair, showing why rotating by
/// ABSOLUTE positions makes attention scores depend only on RELATIVE
/// positions. The real kernel lives in src/exec/rope.zig.
fn rotate(v: [2]f32, angle: f32) [2]f32 {
    const c = @cos(angle);
    const s = @sin(angle);
    return .{ v[0] * c - v[1] * s, v[0] * s + v[1] * c };
}

fn dot2(a: [2]f32, b: [2]f32) f32 {
    return a[0] * b[0] + a[1] * b[1];
}

test "rotated dot product depends only on the position difference" {
    const q = [2]f32{ 0.8, -0.6 };
    const k = [2]f32{ 0.3, 0.9 };

    // Pair 0 rotates by angle = position (theta_base^0 = 1). The attention
    // score between a query at position pq and a key at position pk is
    // rot(q, pq) · rot(k, pk).
    const s_5_3 = dot2(rotate(q, 5.0), rotate(k, 3.0));
    const s_12_10 = dot2(rotate(q, 12.0), rotate(k, 10.0));
    const s_2_0 = dot2(rotate(q, 2.0), rotate(k, 0.0));

    // All three pairs are exactly 2 positions apart: identical scores.
    try std.testing.expectApproxEqAbs(s_5_3, s_12_10, 1e-5);
    try std.testing.expectApproxEqAbs(s_5_3, s_2_0, 1e-5);

    // A different gap gives a different score.
    const s_5_1 = dot2(rotate(q, 5.0), rotate(k, 1.0));
    try std.testing.expect(@abs(s_5_3 - s_5_1) > 1e-3);
}
```

In the model, the rotation factors for the step's absolute positions are
precomputed once per forward into a `RopeTable` (`ctx.prepareRopeTable`,
`model.zig:509`) — and the `positions` array is the *only* thing that
distinguishes building a prefill step from a decode step
(`model.zig:505-507`: prefill fills `pos0..pos0+len`, decode passes the one
current position). The rotation is fused with §12.3.2's per-head q/k norm
into a single kernel call (*from `src/llm/qwen3/model.zig:1201-1204`*):

```zig
var q_rope = try q3.rmsNormMulRopeHalfPrepared(ctx, .seq, .d, &layer.q_norm, config.rms_norm_eps, rope_table);
defer q_rope.deinit();
var k_rope = try k3.rmsNormMulRopeHalfPrepared(ctx, .seq, .d, &layer.k_norm, config.rms_norm_eps, rope_table);
defer k_rope.deinit();
```

Note what is *absent*: `v3` is never roped. Values carry content, not
addresses; only the q·k matching needs positions. `docs/REFERENCE.md` §4.12
pins the primitive with a machine-verified test worth looking up: position 0
is the identity, and position 1 rotates pair 0 by exactly `cos(1)`/`sin(1)`
— one radian, because pair 0's frequency is `theta_base^0 = 1`.

One consequence to file away for §12.4: because a token's rotation depends
only on its *own* absolute position — never on its neighbours — a key can be
rotated once and cached forever. `src/llm/kv_cache.zig:15-16` states it as a
design rule: "K is stored *after* RoPE (V has no RoPE), so past positions
are never re-rotated."

### 12.3.5 Grouped-query attention and the causal mask

Attention itself: every query attends every (allowed) key, scores become
softmax weights, weights blend the values. The one Qwen3 twist is the
head-count asymmetry — 16 query heads, 8 KV heads — and the entire mapping
that implements it is three lines at load time (*from
`src/llm/qwen3/model.zig:224-227`*):

```zig
const kv_head_for_head = try allocator.alloc(usize, config.num_attention_heads);
errdefer allocator.free(kv_head_for_head);
const heads_per_kv = config.num_attention_heads / config.num_key_value_heads;
for (kv_head_for_head, 0..) |*kv_head, head_i| kv_head.* = head_i / heads_per_kv;
```

Query heads 0 and 1 share KV head 0; heads 2 and 3 share KV head 1; and so
on. That is grouped-query attention (GQA), and the *why* is entirely about
§12.4's cache: K and V must be stored for every past token, so halving the
KV heads halves both the cache's memory and the bytes attention streams per
decoded token — while queries, computed fresh each step and never stored,
stay at full width. The quality cost of sharing is small; the saving here
is exactly 2× (multi-query attention is the same idea taken to one KV
head).

The kernel is one call, `groupedAttention` (`docs/REFERENCE.md` §4.13),
with the model-side wrapper supplying the standard scale (*from
`src/llm/qwen3/model.zig:1804-1815`*):

```zig
fn causalAttention(
    ctx: *ExecContext,
    config: Config,
    q: *const fucina.Tensor(.{ .seq, .head, .d }),
    k: anytype,
    v: anytype,
    kv_head_for_head: []const usize,
    opts: anytype,
) !fucina.Tensor(.{ .seq, .attn }) {
    const scale = 1 / @sqrt(@as(f32, @floatFromInt(config.head_dim)));
    return q.groupedAttention(ctx, k, v, kv_head_for_head, .attn, scale, opts);
}
```

> **ML note** — Two standard ingredients, briefly. The **scale**
> `1/sqrt(head_dim)`: a dot product of two random 128-vectors has standard
> deviation ~`sqrt(128)`, and unscaled scores that large push softmax into
> saturation (one weight ≈ 1, gradients ≈ 0); dividing by `sqrt(d)`
> restores unit variance. The **causal mask**: token `p` may attend
> positions `0..p` only — the model is trained to predict the *next* token,
> so letting it peek at later positions would be training on the answer
> key. In `groupedAttention` the mask is the default (`.mask = .causal`),
> and the options struct is comptime-validated per KV representation, so
> "a misspelled option is a compile error, never silently-full-causal
> attention" (`docs/REFERENCE.md` §4.13).

The `k: anytype` in the wrapper is doing real work: the same call site
accepts f32 tensors (no cache), f16 cache views, or raw q8_0 block slices —
the representation is comptime-dispatched from the type
(`docs/REFERENCE.md` §4.13 lists all five accepted forms). The same section
carries the simplest possible attention unit test, machine-verified: one
query against *one* cached position, whose softmax weight must be 1, so the
output must be exactly `v` — attention reduced to its base case
(`test "groupedAttention over a single cached position returns v"`). Start
there when you port your own attention.

After attention, one more matmul (`o_proj`) maps the concatenated head
outputs back to the 1024-wide stream (`model.zig:1240`).

### 12.3.6 The SwiGLU feed-forward block

The second half of every layer is the FFN — where well over half of each
layer's parameters live (about 60% for this model). Qwen3 uses SwiGLU:

```
ffn(x) = down( silu(gate(x)) ⊙ up(x) )
```

Two projections up to 3072 dimensions (`gate` and `up`), an element-wise
gated nonlinearity, one projection back down. `silu(z) = z · sigmoid(z)` is
a smooth ReLU relative; the gating means one pathway (`gate`) decides *how
much* of the other pathway's signal (`up`) passes, per channel — empirically
stronger than a plain nonlinearity at equal parameter count. In code, the
separate-weights arm reads (*from `src/llm/qwen3/model.zig:1517-1526`,
trimmed*):

```zig
var gate_up = try dense.input_proj.project(ctx, ffn_in, config);
defer gate_up.deinit();
const out = try gate_up.up.swiglu(ctx, &gate_up.gate);
```

and the fused arm computes both projections in one `[6144, 1024]` GEMM,
then splits and gates in a single pass (`splitGated(ctx, .swiglu, ...)`,
`model.zig:1534`). Same math, two execution strategies — and for the hot
quantized formats a third, `denseFfnFusedDown` (`model.zig:1547-1564`),
never materializes the gated 3072-wide tensor at all: SwiGLU, activation
quantization, and the packed down-GEMM run as one fused kernel once the
batch is big enough (`seq >= 12`, `model.zig:1503`).

### 12.3.7 The residual stream, the final norm, and the logits

Look back at the two block functions and find their last lines:

```zig
const out = try residual_input.add(ctx, &attn_out);   // model.zig:1251
...
const out = try input.add(ctx, &contribution);        // model.zig:1486
```

Neither block *transforms* the stream — each computes a contribution and
**adds it back**. The `[seq, 1024]` tensor flowing through all 28 layers is
the *residual stream*: a shared bus every block reads (through its norm)
and writes (through its add); nothing else connects the layers. This is why
a 28-layer model is trainable at all — gradients flow straight through the
adds ([Chapter 7](07-autograd.md)) — and a useful mental model besides:
blocks are additive editors of a running document, not pipeline stages.

After the last layer: one final RMSNorm, then the lm_head projects the
stream to `[*, 151_936]` logits — one raw score per vocabulary token. The
skeleton in §12.3's opening shows a quiet but important optimization: with
`last_only` (the normal generation path), only the final row is projected.
The lm_head is the single biggest matmul in the model
(1024 × 151_936), and during prefill you only need the *last* position's
prediction — so the code narrows to one row first, and the whole prompt
pays for one vocab projection instead of `seq` of them. The
`forwardStepAllLogits` entry (§12.5) skips the narrow and projects every
position — that difference is its entire reason to exist.

## 12.4 The KV cache: remembering instead of recomputing

Generation is a loop: predict a token, append it, run the model again. Run
`forwardLastLogits` (the cacheless entry, `model.zig:271`) on the growing
sequence each time and you recompute *everything* about the prefix — every
projection, every attention pass, every FFN — for every token you emit,
and almost all of that work is byte-identical to the previous iteration's.

The fix follows from one observation about causal attention: position `p`'s
keys and values *never change* once computed — later tokens cannot influence
earlier ones (that is what the causal mask means), and RoPE rotates each key
by its own absolute position only (§12.3.4). So cache them. `KvCache`
(`src/llm/kv_cache.zig`) stores, per layer, the K and V rows of every
position seen so far; a step computes projections **only for its new
tokens**, appends their K/V, and runs its queries against the whole cached
prefix.

With the cache, the per-token cost changes shape. Every matmul — QKV
projections, o_proj, the FFN's three, the lm_head — now runs on *one row*
however long the conversation is: constant work per token. Attention still
scans the full cached prefix (that part grows linearly with context), but
as a read-and-accumulate over cached rows, not a recomputation of them.
"Re-run the whole prefix through 28 layers, per token" becomes "one token's
worth of matmuls plus one scan."

The storage layout is chosen so that reading the cache costs nothing
(*doc comment, `src/llm/kv_cache.zig:8-16`*):

> K and V are kept as f16 `[capacity, kv_heads, head_dim]` contiguous
> tensors — the exact `[.seq, .kv_head, .d]` layout the f16 attention kernel
> consumes — so the active prefix `[0..len]` is a zero-copy narrow that
> feeds attention directly. f16 halves the cache footprint and the per-step
> bandwidth (the kernel widens to f32 in-register), matching llama.cpp's
> default cache type. K is stored *after* RoPE (V has no RoPE), so past
> positions are never re-rotated.

Here it is in action inside the attention block — append the new rows, then
attend over a zero-copy view of everything (*from
`src/llm/qwen3/model.zig:1214-1235`, q8_0 arm trimmed*):

```zig
var attn = if (cache) |kv| blk: {
    try kv.appendLayer(ctx, layer_i, &k_rope, &v3);
    const cached_len = kv.len + k_rope.dim(.seq);
    switch (kv.dtype) {
        .f16 => {
            var k_view = try kv.k[layer_i].narrow(ctx, .seq, 0, cached_len);
            defer k_view.deinit();
            var v_view = try kv.v[layer_i].narrow(ctx, .seq, 0, cached_len);
            defer v_view.deinit();
            break :blk try causalAttention(ctx, config, q_attention, &k_view, &v_view, kv_head_for_head, .{});
        },
        .q8_0 => ...,
    }
} else try causalAttention(ctx, config, q_attention, &k_rope, &v3, kv_head_for_head, .{});
```

Three contracts make this correct, and each is enforced:

1. **`appendLayer` does not advance `len`.** All 28 layers append at the
   same base offset; `kv.advance(token_ids.len)` runs *once*, after the
   layer loop (`model.zig:530`; doc comment at `kv_cache.zig:250-253`).
   Get this wrong in a model port and every layer after the first writes
   to the wrong rows.
2. **`forwardStep` demands `kv.len == pos0`** (else
   `Error.InvalidSequenceLength`) and enough capacity (else
   `KvCacheOverflow`) — `model.zig:501-503`. The cache's length *is* the
   model's notion of "where we are".
3. **Cached equals uncached.** The doc comment at `model.zig:441-446`
   states the oracle: "With a fresh cache and `pos0 == 0` this is prefill
   and yields the same last-token logits as `forwardLastLogits`." The
   runner's `--verify-cache N` flag checks exactly this, decode step by
   decode step, against the cacheless entry.

The write path (`appendLayer`'s f16 arm, `kv_cache.zig:268-281`) shows the
allocation discipline this loop runs under: new rows are cast "straight into
the cache slot: one pass, no temporaries" — and the comment preserves the
*before* state (a materialized f32 copy, an unpooled temp, a memcpy), the
repo's habit of naming the rejected version. And because the buffers are
pre-allocated at capacity and every reader and writer addresses rows
strictly from `len`, rewinding the cache is a single integer store (*from
`src/llm/kv_cache.zig:310-312`*):

```zig
pub fn truncate(self: *KvCache, keep_len: usize) void {
    if (keep_len < self.len) self.len = keep_len;
}
```

Its doc comment (`kv_cache.zig:300-309`) is a small masterclass in
invariant-driven design: it enumerates every reader and writer to prove
that decrementing `len` *is* the rewind, in both storage modes. Speculative
decoding ([Chapter 13](13-inference-tricks.md)) leans on this to drop
rejected draft tokens; the chat reuse seam (§12.7) leans on it to rewind to
a common prefix. A windowed ring-buffer cache would have made the one-liner
impossible — which is why "the cache itself has no window logic"
(`docs/REFERENCE.md` §13.4): windowed models apply their sliding window at
read time, in the attention kernel.

**How big is it?** Do the arithmetic from the config: per position,
28 layers × 2 (K and V) × 8 kv_heads × 128 dims × 2 bytes (f16) = 114,688
bytes ≈ **112 KiB per position** — matching the "~112 KiB/position for a
28-layer/8-kv-head/128-dim f16 geometry" that `docs/LMSERVER.md` quotes
when budgeting server slots. A 4096-token conversation holds ~448 MiB of
cache — for a 0.6B model. Now the GQA decision reads differently: with 16
KV heads instead of 8, double it.

**The q8_0 option.** `initWithDtype(..., .q8_0)` (`kv_cache.zig:70`) stores
each (position, head) row as `head_dim/32` q8_0 blocks — 34 bytes per 32
elements, roughly halving f16's footprint again at a small quantization
loss. Be precise about what this buys: the runner's README
(`examples/qwen3/README.md`) calls it
"halves KV memory — capacity option; decode is NOT faster on M1". A
*capacity* lever (twice the context in the same RAM), not a speed lever —
measured, not assumed. It demands `head_dim % 32 == 0` (else
`KvCacheHeadDimNotBlockAligned`, `kv_cache.zig:32-38`), so no block ever
straddles a position — which keeps `truncate` a one-liner in q8_0 mode too.
Only the qwen3 family's attention accepts a q8_0 cache; run it with
`--cache-type q8_0`.

## 12.5 Prefill and decode: two performance regimes

Here is the fact that organizes all of LLM inference engineering: the same
`forwardStep` serves two workloads that differ only in `token_ids.len`.

- **Prefill** — the whole prompt in one call. Every projection is a
  `[m, k] × [k, n]` GEMM with m = hundreds. Each weight byte, once loaded
  into cache, is used m times. Plenty of arithmetic per byte moved: the CPU's
  vector units are the bottleneck. **Compute-bound.**
- **Decode** — one token per call. Every projection is a GEMV: m = 1. Each
  weight byte is loaded from RAM, used *once*, and evicted. For every
  decoded token, essentially the entire model must stream past the core.
  The memory bus is the bottleneck; the ALUs idle. **Bandwidth-bound.**

That asymmetry explains, in one stroke, most of what you have seen and are
about to see:

- **Quantization** ([Chapter 11](11-model-files-and-quantization.md)) is a
  *decode* lever first: fewer bytes per weight is fewer bytes per token
  streamed, and decode is where bytes are the binding constraint.
- **Different regimes want different kernels.** The fused
  norm-into-GEMM path from Chapter 11 is explicitly gated to m ≥ 4:
  `supportsNormedFusion` declines at decode sizes, citing a "measured 2-3%
  decode LOSS on M1 Q4_K_M/Q8_0, against a +11-23% pp32 win"
  (`src/llm/weights.zig:780-789`) — a documented decision *with its
  numbers*, and a reminder that these are dated, machine-specific
  measurements, not laws.
- **Batch decode** shares the stream: `forwardStepBatch`
  (`model.zig:563-615`) decodes one token for each of N independent
  conversations in a single m = N pass — "weights are read once for all
  streams, the batch-decode bandwidth win — while RoPE positions, KV
  appends, and attention are per-stream" (its doc comment). Step cost
  barely moves; tokens per step multiply by N.
- **Speculative decoding** ([Chapter 13](13-inference-tricks.md)) turns
  decode back into prefill: draft several tokens cheaply, then *verify*
  them in one prefill-shaped pass. Its verify entry is
  `forwardStepAllLogits` (`model.zig:481`) — `forwardStep`, except it
  returns logits for **every** appended position, paying "~one step's
  weight traffic instead of `token_ids.len` sequential steps"
  (`model.zig:473-475`).

One honesty note, straight from the doc comments, because it is easy to
over-claim: the batched paths are **not** unconditionally bit-identical to
their sequential equivalents. Per-row numerics match per-token `forwardStep`
bit-for-bit only *below* the m-dependent kernel thresholds — "quantized-weight
x4-packed kernels at seq >= 4, fused FFN at seq >= 12, tiled attention at
seq >= 48"; beyond them "rows can differ by reassociation drift (~1e-6
rel)" (`model.zig:477-480`). For `forwardStepBatch` the measurement is
pinned in the same terms: "0.6B Q4_K/Q8_0 batch == sequential
token-for-token at n <= 3, ~1e-6 reassociation drift at n >= 4"
(`model.zig:560-561`). Floating-point addition is not associative; a kernel
that sums in a different order produces a different last bit, and the
repo's answer is to *document the threshold* and test both sides of it.

> **ML note** — Want to feel the two regimes? Run the runner's `--bench`
> mode (§12.9) and compare the prompt-processing rate against the
> generation rate — for the same model on the same machine, prefill
> throughput lands far above decode throughput, because the two are
> hitting different walls. `docs/BENCHMARK.md` reports prefill (`pp`) and
> decode (`tg`) as separate measurements for exactly this reason, each
> cell a dated, machine-specific snapshot.

## 12.6 Sampling: from logits to a token

The forward pass ends with 151,936 raw scores; something must pick one
token. The simplest picker is greedy — take the argmax — and the minimal
autoregressive loop around it is worth reading in full (*from
`src/llm/qwen3/model.zig:704-718`*):

```zig
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
    const fresh = try self.forwardStep(ctx, kv, &.{next}, kv.len);
    logits.deinit();
    logits = fresh;
}
```

> **Zig note** — The comment marks a manual-memory subtlety no GC language
> surfaces: the tempting `logits.deinit(); logits = try forwardStep(...)`
> leaves `logits` dangling if `forwardStep` fails, and the function-scope
> `defer logits.deinit()` would then double-free. Allocate-then-swap keeps
> an owner alive at every program point — the price and the proof of
> knowing exactly when memory dies.

Greedy is perfect for benchmarks and parity tests, but the sampler's module
comment tells you why it cannot be the default for chat
(*from `src/llm/sampler.zig:1-9`*): "Greedy (argmax) is deterministic and
right for benchmarking, but it makes instruction-tuned models like Qwen3
degenerate into repetition. This sampler implements the usual quality
pipeline — repetition / frequency / presence penalties, temperature, top-k,
top-p (nucleus), and min-p — over the model's logits, matching llama.cpp's
parameter set and defaults. With `temperature <= 0` it falls back to greedy,
so the benchmark path is unchanged."

The knobs, verbatim (*from `src/llm/sampler.zig:23-48`, trimmed*):

```zig
pub const Config = struct {
    /// <= 0 selects greedy (argmax); otherwise scales the logits before softmax.
    temperature: f32 = 0,
    /// Keep only the top-k logits (0 = use `max_candidates`).
    top_k: usize = 0,
    /// Nucleus: keep the smallest prefix whose cumulative probability >= top_p.
    top_p: f32 = 1.0,
    /// Keep tokens with probability >= min_p * p(best). 0 disables.
    min_p: f32 = 0,
    // ... repeat/frequency/presence penalties + repeat_last_n window ...
    seed: u64 = 0,

    pub fn isGreedy(self: Config) bool {
        return self.temperature <= 0;
    }
};
```

`Sampler.next` (`sampler.zig:70-169`) runs the pipeline in a fixed order:
optional logit processor (the constrained-decoding seam,
[Chapter 13](13-inference-tricks.md)) → penalties → greedy shortcut → top-k
→ temperature softmax → top-p → min-p → one random draw. The heart of it
(*from `src/llm/sampler.zig:116-139`, trimmed*):

```zig
// temperature + softmax over the candidates (vals[0] is the max).
var probs: [max_candidates]f32 = undefined;
const inv_temp = 1.0 / cfg.temperature;
const max_logit = vals[0];
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
```

Everything worth knowing about sampling is visible in these few lines:

- **Temperature** divides the logits before softmax: below 1.0 it sharpens
  the distribution, above 1.0 it flattens towards uniform, and as it
  approaches 0 it converges to argmax — which is why `temperature <= 0`
  *is* the greedy switch.
- **Top-k** happens first: `logits.topK(ctx, .vocab, k, .top)` keeps the k
  best candidates (at most `max_candidates = 256`; the comment at
  `sampler.zig:19-21` notes "Qwen3 uses top_k=20"). `topK` returns them
  *sorted descending*, so top-p and min-p become simple prefix scans.
- **Top-p (nucleus)** cuts the sorted prefix at cumulative probability
  `top_p`: few survivors in a confident distribution, many in a flat one —
  it adapts where a fixed k cannot. **Min-p** adds a relative floor: drop
  anything below `min_p` times the best token's probability.
- **Max-subtraction** (`vals[i] - max_logit`) keeps `@exp` from overflowing
  — the softmax stability trick from
  [Chapter 5](05-the-operation-library.md); you will spot it again in the
  MoE router below.

The penalties (`sampler.zig:75-98`) mutate the logits *in place* — divide
positive logits (multiply negative ones) by `repeat_penalty`, subtract
count- and presence-based terms — once per unique token in the last
`repeat_last_n` committed tokens, matching llama.cpp's semantics. In-place
means: do not reuse a logits tensor after `Sampler.next` expecting pristine
values.

Finally, determinism. The RNG is `std.Random.DefaultPrng` seeded once at
init (`sampler.zig:60-62`); the draw sequence is a pure function of the
seed. Not cosmetic: `docs/REFERENCE.md` §13.6 notes the chat and
speculative layers *rely* on exactly one draw per committed token, so
speculation cannot desynchronize a seeded conversation. Determinism as a
design value, again.

## 12.7 Chat: a text protocol and a comptime contract

The secret about chat models: there is no chat. The model does one thing —
continue text. "Chat" is a *text protocol* it was fine-tuned on, and the
entire ChatML implementation is string concatenation (*from
`src/llm/chat.zig:80-92`*):

```zig
.chatml => {
    if (!first) try buf.appendSlice(allocator, "<|im_end|>\n");
    if (first) if (system) |s| {
        try buf.appendSlice(allocator, "<|im_start|>system\n");
        try buf.appendSlice(allocator, s);
        try buf.appendSlice(allocator, "<|im_end|>\n");
    };
    try buf.appendSlice(allocator, "<|im_start|>user\n");
    try buf.appendSlice(allocator, user);
    try buf.appendSlice(allocator, "<|im_end|>\n<|im_start|>assistant\n");
    if (think_off) try buf.appendSlice(allocator, "<think>\n\n</think>\n\n");
},
```

The rendered text ends with `<|im_start|>assistant\n` — and the model,
having seen millions of documents in this shape, continues with an
assistant reply. Generation stops when it emits the turn's closing marker:
the template's `stopMarker()` (`"<|im_end|>"` for ChatML, `chat.zig:56-63`),
resolved via `tokenizer.tokenId(template.stopMarker()) orelse
tokenizer.eosId()` (`chat.zig:400`). Even `--no-think` is demystified here:
Qwen3 emits a `<think>...</think>` reasoning block before answering, and
suppressing it is nothing more than *pre-filling an empty one* so the model
believes it has already thought. Which template a model speaks is sniffed
from its GGUF metadata (`Template.detect` reads `tokenizer.chat_template`,
`chat.zig:46-53`); the same enum renders Llama 3 and both Gemma formats.

### Conversation state

Multi-turn state lives in `Conversation(Model, Tok)` (`chat.zig:302`) — a
function returning a type, comptime-generic over the model family and its
tokenizer *module* (qwen3's byte-BPE here; gemma4 instantiates the same
type with its SentencePiece module). It owns the KV cache, token history,
sampler, and stream decoder; each `send` renders the new turn, tokenizes it
with `encodeRaw` (§12.2: templates own structure), and prefills **only the
new tokens** — the cache still holds every previous turn, so a conversation
never re-processes its past.

The decode loop is the teaching centerpiece of the whole stack — thirty
lines that contain everything this chapter built (*from
`src/llm/chat.zig:708-744`, stop-sequence handling trimmed*):

```zig
fn decodeTurn(self: *Self, prefix: []const usize, writer: *std.Io.Writer) !usize {
    const a = self.allocator;

    // Prefill this turn's tokens at the current cache position.
    var logits = try self.model.forwardStep(self.ctx, &self.cache, prefix, self.cache.len);
    defer logits.deinit();

    self.stream.reset();
    var produced: usize = 0;
    while (produced < self.max_response_tokens and self.cache.len < self.cache.capacity) {
        const next = try self.sampler.next(self.ctx, &logits, self.history.items);
        if (self.isStopToken(next)) break;
        try self.stream.push(a, @intCast(next), writer);
        try writer.flush();
        try self.history.append(a, next);
        produced += 1;
        var single = [_]usize{next};
        const fresh = try self.model.forwardStep(self.ctx, &self.cache, &single, self.cache.len);
        logits.deinit();
        logits = fresh;
    }
    try self.stream.flush(writer);
    try writer.flush();
    try self.persistAppend();
    return produced;
}
```

Prefill once; then sample → stop-check → stream → commit to history →
forward one token. §12.2's `StreamDecoder` handles the UTF-8 tails, §12.4's
cache makes each iteration cheap, §12.6's sampler picks the tokens. A turn
whose prefix exceeds remaining capacity is `error.ContextFull`
(`chat.zig:587`); the whole conversation must fit `Options.capacity`.

### The comptime contract

`Conversation` never names qwen3. Its expectations are *duck-typed*, spelled
out in the doc comment (`chat.zig:296-301`): `Model` must expose
`config.vocab_size`, `initKvCache`, and the `forwardStep` /
`forwardStepAllLogits` decode entries; `Tok` must provide a `Tokenizer` and
a `StreamDecoder`. Zig checks these at instantiation, and the two batch-ish
entries make a beautifully instructive contrast:

- **`forwardStepAllLogits` is a hard compile-time requirement — even with
  speculation permanently off.** `docs/REFERENCE.md` §13.8.2: "`send`
  unconditionally references the speculative path, so a `Model` without it
  fails to instantiate; it is only *executed* when speculation is enabled."
  Comptime duck typing checks everything a generic function *mentions*, not
  just what a given run calls — the compiler compiles the whole `send`,
  speculative branch included, the moment you instantiate the type.
- **`forwardStepBatch` is comptime-*gated*, so it is optional.** The code
  asks before touching it (*from `src/llm/chat.zig:797-804`, comment
  included*):

```zig
// Comptime-gated so model families without a batch entry (e.g.
// gemma4 today) still compile the Conversation type; they get a
// runtime error here instead.
if (comptime @hasDecl(Model, "forwardStepBatch")) {
    return sendBatchImpl(convos, users, writers, produced);
} else {
    return error.BatchDecodeUnsupported;
}
```

> **Zig note** — This pair is the whole comptime-interface design space in
> eight lines. Mention a declaration unconditionally and it becomes a hard
> requirement of your generic type; wrap it in
> `if (comptime @hasDecl(...))` and it becomes an optional capability with
> a graceful runtime fallback — the `else` branch is not even analysed for
> a model without the method. No interface files, no trait declarations:
> the contract *is* the usage, which is exactly why the repo documents it
> in prose so carefully.

Beyond `send`, the same type carries the seams
[Chapter 13](13-inference-tricks.md) builds the server story on:
`initWarm`/`takeCache` move a KV cache between requests,
`sendRenderedReuse` rewinds to the longest common token prefix via §12.4's
`truncate`, `sendBatch` drives `forwardStepBatch` in lockstep, and
`speculation: true` swaps in the speculative decoder. Notice only that every
one of them plugs into machinery you have already seen.

## 12.8 Mixture of experts: parameters you do not pay for

Everything so far was the *dense* Qwen3. The same file also implements the
MoE variants — Qwen3-30B-A3B and 235B-A22B — and the difference is exactly
one component: the FFN. Structurally, it is a two-armed union
(*from `src/llm/qwen3/model.zig:762-773`*):

```zig
const Ffn = union(enum) {
    dense: DenseFfn,
    moe: MoeFfn,
    ...
};
```

A MoE layer replaces the single 3072-wide FFN with `num_experts` smaller
FFNs (the *experts*) plus a tiny **router**: a linear layer whose output
dimension is the expert count. Per token, the router scores all experts,
the top `num_experts_used` are selected, and *only those* run, their
outputs blended by the router's softmax weights. The model names encode the
economics — 30B-A3B: thirty billion parameters, roughly three billion
**a**ctive per token. Capacity scales with total experts; per-token compute
with the active count. You buy knowledge without buying latency, paying in
memory instead — which is why Chapter 11's shard-streaming quantizer and
Chapter 13's expert streaming exist.

The routing is small enough to read whole. First the model side (*from
`src/llm/qwen3/model.zig:1650-1674`, trimmed*):

```zig
var logits = try moe.router.linearSeq(ctx, ffn_in, .embed, .expert);
defer logits.deinit();

const sel = try allocator.alloc(usize, seq * top_k);
defer allocator.free(sel);
const wgt = try allocator.alloc(f32, seq * top_k);
defer allocator.free(wgt);
try logits.routerTopK(ctx, .expert, top_k, .{ .normalize_selected = config.norm_topk_prob }, sel, wgt);
applyExpertTopP(sel, wgt, top_k, config.moe_expert_top_p);

return weights.moeSwiGluFfnSeq(ctx, ffn_in, &moe.gate, &moe.up, &moe.down,
    sel, wgt, top_k, config.moe_intermediate_size, io, moe_profile);
```

The router is *just another `LinearWeight`* — same union, same `linearSeq`,
out-tag `.expert` instead of `.vocab`; a MoE router and an lm_head are the
same operation at different widths. Selection lands in two plain host-side
slices (`sel`: which experts, `wgt`: their mixture weights), and one fused
call runs the routed SwiGLU mixture. The selection kernel is worth reading
whole (*from `src/exec/topk.zig:50-86`, trimmed*):

```zig
fn routerTopKRow(logits: []const f32, k: usize, normalize_selected: bool, selected: []usize, weights: []f32) void {
    var max: f32 = logits[0];
    for (0..k) |slot| {
        selected[slot] = 0;
        weights[slot] = -std.math.inf(f32);
    }

    for (logits, 0..) |v, i| {
        if (v > max) max = v;
        if (v <= weights[k - 1]) continue;
        var slot = k - 1;
        while (slot > 0 and v > weights[slot - 1]) : (slot -= 1) {
            weights[slot] = weights[slot - 1];
            selected[slot] = selected[slot - 1];
        }
        weights[slot] = v;
        selected[slot] = i;
    }

    var exp_sum: f32 = 0;
    for (logits) |v| {
        exp_sum += @exp(v - max);
    }
    // ... softmax the k winners; optionally renormalize them to sum to 1
```

One pass, one k-slot insertion sort (k is single digits), one
max-subtracted softmax — spotted it? — and an optional renormalization
(`normalize_selected` is Qwen's `norm_topk_prob`: the k survivors' weights
rescaled to sum to 1). MoE routing has a fearsome reputation; the
inference-side arithmetic is thirty lines.

The prefill/decode split from §12.5 reappears with an MoE twist, documented
at `model.zig:1632-1636`: decode (seq == 1) uses "the fused expert-parallel
GEMV" while prefill "groups tokens by expert and runs one m>1 GEMM per
expert (weights read once, reused across the batch) — far less weight
traffic than per-token." Same wall, same medicine: amortize the bytes.

Two operational notes close the loop with neighbouring chapters. Expert
stacks are the bulk of a MoE model's bytes, so when the GGUF is
memory-mapped they are *borrowed* straight from the mapping instead of
copied — the model takes ownership and unmaps it last in `deinit`
(`model.zig:236-241, 264-267`). And for models bigger than your RAM,
`MoeStreamOptions` (`src/llm/weights.zig:1069-1100`, re-exported at
`model.zig:153`) keeps experts on disk, read on
demand through a tiered cache — "the explicit trade that lets a
bigger-than-RAM model run at all" (`weights.zig:1073-1074`); that story (LRU
tiers, learned pinning, router lookahead, the measured tokens-per-second on
a 142 GB model in 64 GB of RAM) belongs to
[Chapter 13](13-inference-tricks.md). One taste of the knob-space:
`applyExpertTopP` (`model.zig:1713-1763`) drops low-weight experts per
token; its doc comment records "measured on the 30B MoE: p = 0.7 cut
streamed disk traffic 55% for modest quality cost" and is explicit that
this is *quality-traded* — `p >= 1` is the bit-identical baseline, pinned
by a unit test at `model.zig:1765-1791`.

## 12.9 Run it

Time to hear it talk. From the repo README (verbatim; requires Zig 0.16.0 —
the toolchain is pinned):

```sh
git clone https://github.com/matteo-grella/fucina
cd fucina
zig build test          # unit tests, no model files needed

# grab a small model and talk to it
mkdir -p models
hf download Qwen/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf --local-dir models
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf \
  --chat "What is the capital of France?" --no-think

# or serve it to any OpenAI client (chat completions + responses, SSE
# streaming, JSON-schema constrained output with -Dllguidance=true)
zig build lmserve -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf --port 8080
```

`-Doptimize=ReleaseFast` is not optional in spirit: the README warns Debug
is 10–50x slower, and — [Chapter 6](06-going-fast-on-cpus.md)'s lesson —
build on the machine you run on, because the kernels specialize to the
compiling host's CPU. Then explore the runner (`examples/qwen3/main.zig`; run it
with no arguments for the usage text). A sampler from its README
(`examples/qwen3/README.md`):

```sh
# Chat with a system prompt + sampling overrides
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf \
  --chat "Tell me a joke" --system "You are a pirate." \
  --temp 0.7 --top-k 40 --top-p 0.9 --seed 42

# Tokenizer-parity oracle (one token id per line; no weights loaded)
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf --tokenize input.txt

# q8_0 KV cache (halves KV memory — capacity option; decode is NOT faster on M1)
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf \
  --prompt "..." --gen 256 --cache-type q8_0
```

`--repl` gives you a multi-turn interactive session, `--prompt "..." --gen N`
raw greedy completion, `--bench R` the warm prefill/decode benchmark, and
`--streams N` the lockstep batch decode from §12.5.

Three flags are this chapter in executable form: `--tokenize` is §12.2's
parity bar, `--verify-cache N` is §12.4's cached-equals-uncached oracle,
and `--logits-out` / `--compare-logits` dump and diff raw f32 logits
against another implementation — the verification religion of
[Chapter 6](06-going-fast-on-cpus.md), applied to a whole transformer.

## What you now know

- A transformer's architecture is a struct of ~nine integers, and
  `Config.fromGguf` reads them from the model file itself.
- Byte-level BPE = a printable stand-in for every byte (no unknown tokens,
  ever) + a pretokenizer that chunks text + "repeatedly merge the
  lowest-ranked adjacent pair". The bar is token-ID-exactness against
  llama.cpp on valid UTF-8 — malformed UTF-8 is the one documented
  deviation — and `--tokenize` is the oracle.
- The forward pass is embedding lookup → 28 × (attention + FFN) → final
  RMSNorm → lm_head, each block *adding* its contribution to one residual
  stream.
- RoPE rotates q/k channel pairs by angles proportional to absolute
  position; rotations cancel inside a dot product, so scores depend only on
  relative position — and keys can be cached post-rotation.
- GQA maps 16 query heads onto 8 KV heads (three lines of code) to halve
  KV-cache memory and decode bandwidth.
- The KV cache makes each decoded token cost one token's matmuls plus one
  scan of the cached prefix; `appendLayer`-then-`advance`-once is the write
  contract, `truncate` is a one-line rewind, cached-vs-uncached parity is
  machine-checked. ~112 KiB/position here; q8_0 halves that as a *capacity*
  option, not a speed one.
- Prefill is compute-bound GEMM, decode is bandwidth-bound GEMV — the
  distinction behind quantization's payoff, batch decode, kernel
  m-thresholds, and (next chapter) speculative decoding. Batched paths are
  bit-identical only below documented thresholds; ~1e-6 reassociation
  drift beyond.
- Sampling is ~50 lines of f32 arithmetic: temperature scales logits,
  top-k/top-p/min-p trim the sorted candidates, one seeded draw picks; and
  `temperature <= 0` is greedy, keeping the benchmark path pure.
- Chat is a text protocol rendered by string concatenation; state is a KV
  cache plus token history, and each turn prefills only its new tokens.
  `Conversation(Model, Tok)` shows both comptime-interface styles: hard
  requirement (`forwardStepAllLogits`) and `@hasDecl`-gated option
  (`forwardStepBatch`).
- MoE swaps one FFN for many experts plus a router (a plain linear + top-k
  + softmax): parameter count grows with total experts, per-token compute
  only with the active ones.

## Explore the source

- `src/llm/qwen3/model.zig` — the whole model; start at `Config`, then read
  `forwardStepImpl`, `attentionBlock`, and `ffnBlock` in that order.
- `src/llm/tokenizer.zig` — byte-level BPE end to end; the pretokenizer
  tests at the bottom are the fastest way to internalize chunking rules.
- `src/llm/kv_cache.zig` — ~320 lines; read every doc comment, especially
  `truncate`'s.
- `src/llm/sampler.zig` — the complete sampling pipeline in 180 lines.
- `src/llm/chat.zig` — templates, `Conversation`, `decodeTurn`, and the
  reuse/batch seams Chapter 13 builds on.
- `src/exec/topk.zig` — MoE routing in one page.
- `examples/qwen3/main.zig` and `docs/REFERENCE.md` §13 — the runner (every flag
  a doorway into a section of this chapter) and the machine-verified
  `fucina_llm` reference it exercises.

## Exercises

1. **Predict the chunks.** Before running anything, write down how the
   qwen2 pretokenizer splits `"I'll pay 42 euros!\n"` (§12.2's rules:
   contractions, one digit per chunk, punctuation absorbs trailing
   newlines). Check yourself with `--tokenize file.txt` and the tests in
   `src/llm/tokenizer.zig:1174-1209`.
2. **Extend MiniBpe.** Add §12.2's GPT-2 byte→Unicode mapping to the
   course-code `MiniBpe` so it encodes arbitrary bytes (spaces become
   `Ġ`-prefixed symbols); test on `"hi hi"` with your own vocabulary.
   `gpt2ByteToUnicode` (`tokenizer.zig:591`) is your reference.
3. **Feel the cache.** Write a small runner (crib the model-loading
   prologue from `examples/qwen3/main.zig`) that generates 64 tokens twice: once
   with `Model.generate` (KV-cached), once by calling `forwardLastLogits`
   on the full growing sequence each step. Verify the token ids match
   exactly (greedy is deterministic), then time both.
4. **Same seed, same story.** Using `Sampler` directly, prove §12.6's
   determinism claim as a `zig test` block: two samplers with identical
   `Config` (temperature 0.7, seed 42) fed identical logits and history
   produce identical sequences; changing only the seed diverges.
5. **(Hard) A window into attention.** `groupedAttention` accepts
   `.window = w` for sliding-window attention (`docs/REFERENCE.md` §4.13).
   Modify a local copy of `causalAttention` in `model.zig` to pass a window
   and measure — with `--compare-logits` as your oracle — how output
   diverges from full attention as `w` shrinks on a long prompt. (Qwen3 was
   *trained* with full attention: you are measuring graceful degradation.
   Bonus: explain why the KV cache needs no changes — §12.4's "no window
   logic in the cache" design note is the answer key.)

---

[Previous: Model files and quantization](11-model-files-and-quantization.md) ·
[Next: Inference tricks](13-inference-tricks.md)
