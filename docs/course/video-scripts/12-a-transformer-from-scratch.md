# Video 12 — A transformer from scratch (3:00)

*Series: Forging Deep Learning in Zig · Source: ../12-a-transformer-from-scratch.md*

## Logline

A transformer's architecture is a struct of nine integers, and everything
between typing a question and reading the answer lives in one Zig file you
can read. The video goes deep on exactly one idea — the KV cache, and why it
splits inference into compute-bound prefill and bandwidth-bound decode —
then downloads Qwen3-0.6B and chats with it live in the terminal.

## Takeaways

1. The whole forward pass is embedding lookup → 28 × (grouped-query
   attention + SwiGLU FFN, each *adding* into one residual stream) → final
   RMSNorm → lm_head — real, readable code in `src/llm/qwen3/model.zig`.
2. Causal attention means a position's K/V never change once computed, so
   cache them: every matmul then runs on one row per new token (constant
   work), plus one scan of the cached prefix. Prefill is compute-bound GEMM;
   decode is bandwidth-bound GEMV.
3. You can run this today: three steps — clone, download Qwen3-0.6B, chat —
   stream an answer on a laptop CPU, and `--verify-cache` machine-checks
   that cached equals uncached.

## Script

### [0:00–0:26] Nine numbers

**VO:** Strip away the mythology, and a transformer's architecture is a
struct of nine integers. This is Qwen3-0.6B, verbatim: a vocabulary of
151,936 tokens, vectors 1,024 floats wide, the same block repeated
twenty-eight times, sixteen attention heads sharing eight key-value heads.
Everything between typing a question and reading the answer lives in one
Zig file you can read top to bottom.

**Visual:** Code shot: `src/llm/qwen3/model.zig:71–83` (the `qwen3_0_6b()`
config, exactly the chapter's §12.1 excerpt). As the VO names each number,
highlight the matching field: `vocab_size`, `hidden_size`, `num_layers`,
`num_attention_heads` + `num_key_value_heads`.

**Overlay:** "`src/llm/qwen3/model.zig` — ~1,700 lines, the whole model" ·
"16 query heads, 8 KV heads — that asymmetry *is* grouped-query attention".

### [0:26–0:55] The whole forward pass

**VO:** And here is that file doing all the work. One embedding lookup
turns token ids into vectors. A loop runs twenty-eight identical layers —
grouped-query attention, then a SwiGLU feed-forward, each block adding its
contribution back to one residual stream. Rotary embeddings inject
positions inside the attention call. A final norm, one last matmul, and out
come 151,936 scores — one per vocabulary token. That is the entire
architecture.

**Visual:** Code shot: `src/llm/qwen3/model.zig:561–592` (the heart of
`forwardStep`), with sequential highlights synced to the VO: the
`getRowsAs` embedding line (561), the layer loop with `attentionBlock` /
`ffnBlock` (565–578), `rmsNormMul` final norm (582), the `linearSeq` logits
line (592). Then cut to a rendered version of §12.1's data-flow diagram:
text → token ids → `[seq, 1024]` residual stream → 28 blocks → logits →
sampler → next token, with the arrow looping back.

**Overlay:** "embedding → 28 × (attention + FFN) → norm → lm_head" ·
"blocks *add* into the residual stream — a shared bus, not pipeline
stages".

### [0:55–1:38] The KV cache: remember, don't recompute

**VO:** Now the one idea worth going deep on: the KV cache. Generation is a
loop — predict a token, append it, run again. Done naively, every new token
recomputes everything about the prefix. But causal attention makes a
promise: position p's keys and values never change once computed — later
tokens cannot influence earlier ones. So cache them. Each step projects
only its new tokens, appends their keys and values, and attends over the
cached prefix through a zero-copy view. Every matmul now runs on one row,
however long the conversation gets: constant work per token, plus one scan
of the cache.

**Visual:** First a simple diagram: a growing token row where, per step,
only the last cell is "computed" (bright) and the rest is "read from cache"
(dim), versus the naive version where everything relights every step. Then
code shot: `src/llm/kv_cache.zig:8–16` (the doc comment: f16
`[capacity, kv_heads, head_dim]`, "K is stored *after* RoPE … never
re-rotated"). Then code shot: `src/llm/qwen3/model.zig:1194–1215` — the
`appendLayer` call and the f16 `narrow` views feeding `causalAttention`,
highlighting `appendLayer` then the two `narrow` lines.

**Overlay:** "matmuls: constant per token · attention scan: grows with
context" · "the active prefix is a zero-copy narrow — reading the cache
costs nothing".

### [1:38–2:12] Two regimes, one budget

**VO:** That splits inference into two regimes. Prefill — the whole prompt
at once — is matrix-matrix work, every weight reused across rows:
compute-bound. Decode — one token per step — turns each matmul into
matrix-vector: every weight byte streams from RAM for a single use:
bandwidth-bound. The cache costs about one hundred twelve kibibytes per
position here — and sharing eight KV heads among sixteen query heads is
exactly why it isn't double that. It's checked, too: verify-cache proves
cached equals uncached, step by step.

**Visual:** Reprise of Video 06's GEMM→GEMV morph diagram: `A` tall
("prefill: m = prompt length → GEMM, compute-bound"), then `A` collapses to
one row ("decode: m = 1 → GEMV, bandwidth-bound") — §12.5. Then an
arithmetic card built from the config: "28 layers × 2 (K,V) × 8 kv-heads ×
128 dims × 2 B (f16) ≈ 112 KiB per position → ~448 MiB at 4096 tokens".
Close on a type-on of the oracle flag: `--verify-cache N`.

**Overlay:** On the arithmetic card, persistent: "Qwen3-0.6B geometry, f16
cache — from the config, §12.4; the figure docs/LMSERVER.md budgets with".
On the oracle: "`--verify-cache` — cached vs cacheless logits, decode step
by decode step (model.zig:490–495)".

### [2:12–2:42] Chat with it (showcase)

**VO:** Time to hear it talk. Three steps, straight from the README: clone,
download Qwen3-0.6B in Q8_0, chat. ReleaseFast is not optional in spirit —
Debug builds run ten to fifty times slower. And there it is: the tokenizer,
twenty-eight layers, the cache, the sampler — streaming an answer, token by
token, on a CPU.

**Visual:** Terminal recording (real, executed), commands verbatim from the
README quick-start (§12.9):

```sh
git clone https://github.com/matteo-grella/fucina
cd fucina
mkdir -p models
hf download Qwen/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf --local-dir models
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf \
  --chat "What is the capital of France?" --no-think
```

Time-skip the download and build waits with a visible cut; let the model's
streamed answer play out in real time, unedited.

**Overlay:** "Zig 0.16.0 (toolchain pinned)" during the build ·
"`-Doptimize=ReleaseFast` — README: Debug is 10–50× slower" · during the
answer: "live, unedited — CPU only".

### [2:42–3:00] Read it all, then make it fast

**VO:** The chapter has everything this video skipped — RoPE's rotation
algebra as a runnable test, the sampler, chat templates, mixture-of-experts
— every claim pinned to a file and line. Next time: inference tricks. We
make this faster — without changing a single answer.

**Visual:** Quick montage of file cards from the chapter's "Explore the
source" list: `src/llm/tokenizer.zig`, `src/llm/kv_cache.zig`,
`src/llm/sampler.zig`, `src/llm/chat.zig`. End card: series title, "Next:
13 — Inference tricks", chapter link
`docs/course/12-a-transformer-from-scratch.md`.

**Overlay:** End card: "Next: Inference tricks — faster, without changing a
single answer".

## Asset list

**Code shots (repo files, exact ranges):**
- `src/llm/qwen3/model.zig:71–83` — the `qwen3_0_6b()` nine-integer config.
- `src/llm/qwen3/model.zig:561–592` — the heart of `forwardStep`
  (embedding, layer loop, final norm, logits). The on-screen range includes
  a MoE pilot-prefetch stanza the chapter's excerpt trims; highlights keep
  the eye on the taught lines.
- `src/llm/kv_cache.zig:8–16` — the cache-layout doc comment.
- `src/llm/qwen3/model.zig:1194–1215` — `appendLayer` + zero-copy `narrow`
  views into `causalAttention`.

**Terminal recordings (real, executed):**
- The commands in segment 5, each verbatim from the README quick-start
  block (§12.9): clone, `cd`, `mkdir -p models`, `hf download`, `zig build
  qwen3 … --chat "What is the capital of France?" --no-think`. The full
  README block additionally contains `zig build test` and an `lmserve`
  line, both omitted on camera — run `zig build test` off-camera or
  time-skip it if included. Requires Zig 0.16.0 (pinned) and network access
  for the download.
- Type-on only (not executed): `--verify-cache N` for the oracle beat.

**External downloads:** `Qwen3-0.6B-Q8_0.gguf` from Hugging Face repo
`Qwen/Qwen3-0.6B-GGUF` (weights are NOT in the repo; the `hf download`
command above fetches it on camera).

**Diagrams to render (one sentence each):**
- Data-flow strip from §12.1: text → token ids → `[seq, 1024]` residual
  stream → 28 blocks → logits → sampler → next token, arrow looping back.
- Cache-vs-naive relight diagram: growing token row where naive decode
  relights every cell each step and cached decode lights only the newest.
- GEMM→GEMV morph (visual reprise of Video 06): tall `A` for prefill
  collapsing to one row for decode, compute-bound/bandwidth-bound captions
  (§12.5).
- KV-cache arithmetic card: 28 × 2 × 8 × 128 × 2 B ≈ 112 KiB/position,
  ~448 MiB at 4096 tokens, with its geometry caveat (§12.4).
- End card with next-episode teaser.

## Production notes

- **Tone:** demystifying, unhurried, zero hype. The recurring beat is "no
  magic left" — deliver it as relief, not bravado.
- **Caveats are load-bearing and MUST NOT be cut:** (a) the 112 KiB/position
  card must keep "Qwen3-0.6B geometry, f16 cache" on screen — it is config
  arithmetic (§12.4, echoed by docs/LMSERVER.md), not a benchmark; (b) the
  10–50× Debug warning is the README's own claim and stays attributed to
  it; (c) "matmuls constant per token · attention scan grows with context"
  is the honest phrasing — never compress it to "decode is free" or "O(1)
  everything"; (d) "Zig 0.16.0 (toolchain pinned)" stays on the build shot.
- **The chat answer is NOT scripted.** Whatever the model streams is what
  ships — the "live, unedited" overlay is a promise. Do not cherry-pick
  retakes for a cuter answer; one take, or clearly re-run the same command.
- **Do not** show fabricated terminal output; the download/build time-skips
  must be visible cuts, not spliced fakes.
- **If the cut runs long, trim in this order:** the file-card montage in the
  close (keep the VO), then the data-flow diagram dwell in segment 2 (the
  code shot alone can carry it), then the `--verify-cache` type-on (keep its
  VO sentence). Never trim the arithmetic card's caveat, the demo, or the
  teaser line.
- **Numbers appearing in the video and their sources:** the nine config
  integers incl. 151,936 and 1,024 (`model.zig:71–83` via §12.1); ~1,700
  lines (§12 intro); 112 KiB/position and ~448 MiB @ 4096 tokens (§12.4
  config arithmetic, quoted by docs/LMSERVER.md); GQA saving "exactly 2×"
  vs 16 KV heads (§12.3.5/§12.4); 10–50× Debug slowdown (README via §12.9).
  Nothing else may be quantified.
- The next-episode teaser ("Inference tricks — faster, without changing a
  single answer") matches Video 13's title and must survive edits.
