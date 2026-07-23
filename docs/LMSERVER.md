# LMSERVER — the OpenAI-compatible language-model server example

`zig build lmserve` (`examples/lmserve/main.zig` + `examples/lmserve/`) exposes the
in-tree language models behind the two OpenAI wire dialects. It is an
example, not a library surface: the shared code lives under `examples/lmserve/`
and integrates a model family through one small `Backend` vtable.

```sh
# qwen3 / qwen3moe / qwen35 / gemma4 / diffusion-gemma / inkling GGUFs (arch auto-detected)
zig build lmserve -Dllguidance=true -Doptimize=ReleaseFast -- \
  models/Qwen3-0.6B-Q8_0.gguf --port 8080

# nanochat checkpoint dir (model.safetensors + tokenizer.bin)
zig build lmserve -Doptimize=ReleaseFast -- --nanochat runs/sft --port 8080
```

Endpoints: `POST /v1/chat/completions`, `POST /v1/responses` (both also
unprefixed), `GET /v1/models`, `GET /health`. Flags: `--host --port --ctx
--api-key --queue --conns --experts=borrow --nanochat` (see `--help`).
Point any OpenAI client at `http://host:port/v1`; the loaded model's id is
whatever `GET /v1/models` reports (the request's `model` field is accepted
and ignored — one process serves one model).

## Design: accept concurrently, generate sequentially

`ExecContext` is single-threaded by contract and one forward pass already
fork-joins across every performance core (REFERENCE.md, Threading), so the
server does not try to overlap generations:

```
conn threads (≤ --conns, socket deadlines)          ONE inference worker
  parse → validate → [bounded queue, ≤ --queue] →   owns ExecContext + model
  wait / watch for client hang-up              ←    streams into the request sink
```

- Requests beyond the queue bound get `429` + `retry-after` (llama.cpp defers
  unboundedly; a bounded queue is the honest failure mode for a sequential
  worker).
- Client disconnect (an `MSG_PEEK` probe between waits) cancels queued jobs
  and aborts in-flight generation at the next token.
- `SIGINT`/`SIGTERM`: stop accepting, cancel the in-flight job, drain, exit.
  A self-connect kick unblocks the accept loop — macOS wakes a pending
  `accept` for neither `shutdown(2)` nor `SO_RCVTIMEO`, and the Io layer
  retries `accept` on `EINTR`.
- The API is stateless (every request carries its full history) but the KV
  cache is not: the GGUF chat backend keeps a pool of resident slots
  (`--kv-slots`, default 1) — each a previous request's KV cache plus its
  token shadow — and each request adopts the slot with the longest common
  token prefix (above a llama.cpp-style 0.1 similarity gate; LRU otherwise),
  prefilling only the rest (`Conversation.initWarm`/`takeCache`/
  `sendTokensReuse` in `examples/lmserve/backend.zig`). Follow-up turns of a
  chat re-prefill only the last reply + new message; a non-matching request
  costs one full prefill, exactly as before. Extra slots keep interleaved
  conversations warm at a full `--ctx` cache each (~112 KiB/position for a
  28-layer/8-kv-head/128-dim f16 geometry). A startup **KV RAM guard**
  (`kvRamGuardSlots` in `examples/lmserve/backend.zig`) sizes one probe
  cache, compares `--kv-slots x per-slot` against available memory, and
  prints the arithmetic when it matters: slot pages commit lazily, so an
  overcommit does not fail at startup — it surfaces mid-serving as the OS
  evicting the mmap'd weights' page cache, a silent throughput collapse.
  Above half of available memory the guard warns; above all of it, Linux
  clamps the slot count to fit half (override with `--kv-slots-force`;
  the warning still prints) while macOS only warns — its probe (free +
  speculative + purgeable pages, deliberately excluding the file cache the
  guard protects) still understates reclaimable memory, too weak a number
  to clamp on. With
  `--kv-cache-dir D`, a slot
  about to be destroyed by an unrelated request (keeping < half of it, not
  already stored) spills to an `llm.kv_persist` sidecar under `D` (at most
  `--kv-disk-slots` files, LRU-reused, containment-deduped) and is restored
  — zero re-prefill — when a later request matches it better than every
  resident slot. The reuse is reported as `cached_tokens` in usage
  (`prompt_tokens_details` / `input_tokens_details`).
- `--cartridge F` preloads a trained KV-prefix cartridge (safetensors from
  `zig build cartridge`; `docs/CARTRIDGES.md`) into every conversation:
  served "prior knowledge" with zero prompt tokens — requests answer from a
  corpus that was never in the prompt. The prefix occupies cache rows
  `[0, p)` with no token shadow (`Conversation.notePrefixRows` /
  `WarmState.prefix_rows`); slot reuse and `cached_tokens` operate on the
  real tokens past it, and the reconcile rewind never cuts into the prefix.
  Geometry is probed against the model at startup; qwen3/gemma4 backends
  only. Composes with the disk tier: cartridge conversations spill as
  `FUXKV002` sidecars that record the prefix shape and rows, so a restore
  is self-describing — it keeps the exact prefix it was saved with even if
  the server later runs a different cartridge. Verified live two ways:
  evict → spill → restore reported the full conversation as
  `cached_tokens`, and a greedy A/B against a never-evicted control server
  produced BYTE-IDENTICAL answers (same `cached_tokens`) through the
  restore — the round-tripped state is computationally indistinguishable
  from a cache that never left RAM. (The different-cartridge restore is
  enforced by construction — the restore path takes `prefix_rows` from the
  file and never re-preloads the configured cartridge — but cannot occur
  live yet: the disk registry is per-run, and cross-restart sidecar
  scanning is a separate follow-up.)
- `--fleet DIR` serves a per-document cartridge FLEET (from `zig build
  cartridge-fleet`; Cartridges at Scale, `docs/CARTRIDGES.md`): per request
  the user messages embed through the model itself
  (`Trainer.embedLastHidden` + the fleet's `embed_suffix` contract), the
  fleet's cosine index picks `--rag-docs` documents (`--rag-chunks` chunks
  scanned), and the selected cartridges COMPOSE as the conversation's
  prefix — different requests answer from different knowledge with zero
  corpus tokens in any prompt. Parsed cartridges sit in a small mmap-fed
  LRU (rows are copied into each cache, so eviction never invalidates a
  conversation). Slot reuse is conversation-STICKY: conversation identity
  is the FIRST user message, a continuation keeps the selection its
  conversation started with (per-turn re-retrieval is unstable on
  runner-up documents and forfeits all reuse; token-LCP alone cannot
  carry identity — the constant template preamble of unrelated short
  prompts passes the similarity gate), and retrieval only runs for
  conversations no slot remembers. Follow-up turns report `cached_tokens`
  through their composed prefix; interleaved conversations keep distinct
  selections. qwen3 and gemma4 backends (the query embedder is the
  family's no-adapter trainer; gemma4 MoE GGUFs need `--experts=borrow`);
  excludes `--cartridge` and `--kv-cache-dir` (sidecars do not record
  selections). Size `--ctx` to include `rag_docs × p` prefix rows.
  `--rag-adaptive` relaxes stickiness: every follow-up re-embeds the
  contextual query (all user messages) and the conversation SWITCHES
  knowledge base only when a document outside its selection beats every
  current document's best chunk by `--rag-margin` (default 0.05) — the
  switch rebuilds the prefix and re-prefills the history
  (`cached_tokens` = 0 that turn). The rule is deliberately relative and
  context-anchored (absolute floors and last-message-alone probes cannot
  separate phatic turns from topical pivots on this retriever): switches
  fire on clear cross-domain shifts, rarely on same-register corpora, and
  a NEW conversation — a fresh first user message — always re-retrieves.
- Streaming responses start lazily on the first delta, so a request that
  fails before producing anything (invalid grammar, context overflow) still
  gets a plain JSON error with a proper status code.
- Batched serving is future work: `Conversation.sendBatch` is lockstep
  static batching (qwen3 only, no mid-flight joins), which would slot in as
  an admission window in the scheduler.

## What maps, what is rejected, what is ignored

Honored on both dialects: `messages`/`input` (string, typed message items,
text content parts), `instructions`, `temperature`, `top_p`,
`max_tokens`/`max_completion_tokens`/`max_output_tokens` (always bounded:
default 1024, clamped to `--ctx`), `stop` (≤4 strings), `stream`,
`stream_options.include_usage` (chat), `seed`, `presence_penalty`,
`frequency_penalty`, `response_format`/`text.format` (`json_schema`,
`json_object`, `text`), `reasoning_effort`/`reasoning.effort`, usage
accounting. Extension fields (llama.cpp precedent): `top_k`, `min_p`,
`repeat_penalty`, `regex`, `lark`.

Rejected with a 400/501 naming the offending `param` (never silently
dropped): `tools`/`tool_choice` beyond auto/none (function calling is the
biggest missing feature), `n > 1`, `logprobs`, `logit_bias`, image/audio/file
content, `previous_response_id`/`conversation`/`item_reference` (stateless:
`store:false` semantics; what Codex CLI and the SDKs' basic paths use),
hosted tool types, `background`, `truncation:"auto"`, plus anything the
backend's caps cannot honor (grammar, reasoning, stop sequences, system
role). Bookkeeping fields (`metadata`, `user`, `service_tier`, `store`,
`parallel_tool_calls`, `include`, …) are accepted and ignored, like
llama.cpp.

Errors use OpenAI's shape `{"error":{message,type,param,code}}` with real
HTTP status codes (the SDKs dispatch on status). Mid-stream failures arrive
in-band: chat as a `data:` frame with a top-level `error` key, responses as
an `error` event followed by `response.failed`.

## Streaming contracts

- **Chat**: `data:`-only SSE, `chat.completion.chunk` deltas
  (`reasoning_content` for routed reasoning), `finish_reason` on the final
  chunk, optional trailing usage chunk (`choices:[]`), then `data: [DONE]`.
- **Responses**: `event:` + `data:` framing with monotonic
  `sequence_number`, no `[DONE]`. The full skeleton required by the SDKs'
  `responses.stream()` state machine: `response.created` →
  `response.in_progress` → `response.output_item.added` →
  `response.content_part.added` → `response.output_text.delta`* → the
  `.done` mirrors → `response.completed` / `response.incomplete` (budget) /
  `response.failed`. Reasoning streams as its own output item
  (`response.reasoning_text.delta`).
- Deltas are UTF-8-boundary-safe: a token ending mid-code-point carries into
  the next frame instead of corrupting the JSON.
- Verified against openai-python 2.45.0 end-to-end (both dialects, stream +
  non-stream, strict stream helper, structured outputs, error types).

## Reasoning and the `<think>` head

Reasoning is OFF by default (JSON-first serving; a grammar constraint forces
it off — the constraint governs the reply from token 0,
CONSTRAINED-DECODING.md §7). Clients enable it per request via
`reasoning_effort` / `reasoning.effort`; anything but `none`/`minimal` needs
`caps.think` (qwen3 only today). The emitter scans the reply head whenever
the family has think markers: a leading `<think>…</think>` block routes to
`reasoning_content` (chat) or a reasoning output item (responses), and the
stray leading `</think>` qwen3 emits under the primed-empty think block of
no-think prompts is dropped as a template artifact — OpenAI `content` stays
clean either way.

## Constrained output

`json_schema` (both dialects' shapes), `json_object`, and the `regex`/`lark`
extensions compile to a `llm.llguidance.Constraint` (needs a
`-Dllguidance=true` build; otherwise 501). Base constraints are LRU-cached
per grammar source — `Constraint.init` walks the full vocab to build the
token trie; `clone()` per request shares it (the cache may only be touched
by the worker: a clone borrows its base's bridge, and eviction relies on the
current request's base never being the victim). Grammar completion forces
the turn-stop token, so normal stop handling ends the reply. Invalid
grammars (unsupported JSON-schema keywords, bad regex) are a clean 400 from
llguidance's compiler. The `max_tokens` bound always applies — a greedy
argmax inside an unbounded grammar field can loop (docs, §7 caveat).

## Per-model backends

| Backend | Path | Grammar | Reasoning | Stop strings | Streams |
|---|---|---|---|---|---|
| qwen3 / qwen3moe | generic `Conversation` adapter (`backend.zig`) | ✓ | ✓ (`<think>` routing) | ✓ | per token |
| gemma4 | same adapter (SPM tokenizer, `<turn|>` + extra stop ids, GGUF `general.sampling.*` defaults) | ✓ | — | ✓ | per token |
| diffusion-gemma | `backend_diffusion.zig` over `dg.generate` | — | — | — (EOG-trimmed blocks) | per committed block |
| inkling | `backend_inkling.zig` over `llm.inkling.chat.Engine` (wire-format renderer, sampler) | ✓ | ✓ (`<\|content_thinking\|>` → `<\|content_text\|>` routing) | — | per token (no cross-request KV reuse) |
| qwen35 (Qwen3.5 / Ternary-Bonsai) | `backend_qwen35.zig` over `llm.qwen35.chat.Engine` (ChatML + Qwen3.6 think prefill, sampler) | ✓ | ✓ (`<think>` routing; the prompt-prefilled opener is injected into the stream) | — | per token (no cross-request KV reuse) |
| nanochat | `backend_nanochat.zig` over its own Engine (`--nanochat` dir) | — | — | — | per token (no system role: 400) |

Absent sampling fields default to the model's recommended settings (qwen3
no-think 0.7/20/0.8; gemma4 and qwen35 from GGUF metadata), not OpenAI's
nominal `temperature=1` — same deviation llama.cpp makes. The qwen35
backend rides its own engine rather than `Conversation`: the family's
cache carries recurrent conv/state matrices that cannot be truncated back
to a token prefix, so the KV-slot reuse tiers (and `--kv-cache-dir`) do
not apply — every request prefills from scratch on a fresh cache.

Adding a family = implementing the two-function `Backend` vtable
(`examples/lmserve/types.zig`): `validate` (cheap, connection-thread: message
shape + prompt length) and `generate` (worker-thread: stream reply bytes
into the sink, return token counts + finish reason). Families served by
`llm.chat.Conversation` get this for free from the generic adapter.

Verification status: qwen3 proven end-to-end (openai-python SDK suite,
constrained decoding, concurrency, cancellation, shutdown — 2026-07-12,
Qwen3-0.6B-Q8_0); nanochat proven mechanically against the goldens' base d6
checkpoint; gemma4 and diffusion-gemma compile+unit-verified (no GGUF of
either on local disk when this landed).
