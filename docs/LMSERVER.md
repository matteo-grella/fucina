<!-- docs-nav: group="Run & serve" title="LM server" weight=11 -->
# LMSERVER — the OpenAI- and Anthropic-compatible language-model server example

`zig build lmserve` (`examples/lmserve/main.zig` + `examples/lmserve/`) exposes the
in-tree language models behind the two OpenAI wire dialects plus the
Anthropic Messages API. The server itself is library surface: the transport
and the generic engine live in `llm.serving` (`src/llm/serving/`,
REFERENCE.md §13.13), a model family integrates through one small
`Backend` vtable, and this example is the CLI front end plus the adapters
for the families the generic engine cannot host.

```sh
# qwen3 / qwen3moe / qwen35 / gemma4 / diffusion-gemma / inkling GGUFs (arch auto-detected)
zig build lmserve -Dllguidance=true -Doptimize=ReleaseFast -- \
  models/Qwen3-0.6B-Q8_0.gguf --port 8080

# nanochat checkpoint dir (model.safetensors + tokenizer.bin)
zig build lmserve -Doptimize=ReleaseFast -- --nanochat runs/sft --port 8080
```

Endpoints: `POST /v1/chat/completions`, `POST /v1/responses`,
`POST /v1/messages` (all also unprefixed), `GET /v1/models`, `GET /health`.
Flags: `--host --port --ctx --api-key --queue --conns --batch
--experts=borrow --nanochat` (see `--help`). Point any OpenAI client at
`http://host:port/v1` — or any Anthropic client (Claude Code:
`ANTHROPIC_BASE_URL=http://host:port`) at the messages endpoint; `--api-key`
is honored as `Authorization: Bearer` or `x-api-key`. The loaded model's id
is whatever `GET /v1/models` reports (the request's `model` field is
accepted and ignored — one process serves one model).

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
- The worker never touches a socket: each request's reply bytes flow
  through a per-request `StreamPipe` (`serving.http`) — a futex-signaled,
  mutex-guarded byte pipe — and the CONNECTION thread writes the HTTP head
  and chunked body to the wire. A stalled client therefore stalls only its
  own connection thread while its reply buffers server-side (bounded by
  the reply itself, `max_tokens` frames); generation, the queue behind it,
  and the other streams of a `--batch` lockstep run at full speed.
  Verified live: a client that streams 512 tokens and reads NOTHING holds
  its ~119 KiB reply in the pipe while a sibling request completes in
  0.04 s; when the stalled client finally reads, the complete buffered
  stream (through `[DONE]`) is delivered.
- Client disconnect (an `MSG_PEEK` probe between waits, or a failed drain
  write) cancels queued jobs, fails the pipe, and aborts in-flight
  generation at the next token — in a batch, that stream alone.
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
  `sendTokensReuse` in `src/llm/serving/gguf_chat.zig`). Follow-up turns of a
  chat re-prefill only the last reply + new message. Whole-slot adoption is
  reserved for CONTINUATIONS; a request that merely SHARES a prefix with a
  resident slot — same system prompt, different dialogue — **copies** the
  common rows into its own cache (`KvCache.copyRows`, positions preserved)
  and prefills only the rest, leaving the donor intact for its own
  continuation. Measured (0.6B, ~1860-token shared system prompt): the
  second conversation's TTFT drops 3.4 s -> 0.43 s with 1846/1863 tokens
  copied, while the first conversation's follow-up stays warm. A request
  matching nothing costs one full prefill, exactly as before. Extra slots keep interleaved
  conversations warm at a full `--ctx` cache each (~112 KiB/position for a
  28-layer/8-kv-head/128-dim f16 geometry). A startup **KV RAM guard**
  (`kvRamGuardSlots` in `src/llm/serving/gguf_chat.zig`) sizes one probe
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
- `--spec` (qwen3/qwen3moe) turns on speculative decoding for solo
  generations: the self-draft cascade behind the engine's `DraftSource`
  seam, grammar-wrapped when a constraint is active (constrained requests
  speculate BETTER, not worse). It composes with slot reuse — the
  append-only speculation index is rebuilt from the reconciled history
  each request, and the turn ends with the plain path's catch-up forward
  so the slot shadow stays exact. Text stop sequences are scanned by
  the TurnGate (decoded bytes, the plain loop's exact stop-before-stream
  rule, fired-sequence attribution included), so stop-carrying requests
  speculate too; only `--batch` groups of two or more decode plain, and
  solo requests under a batched server still speculate. Verified live: greedy replies
  byte-identical to a plain server across reuse turns.
- Streaming responses start lazily on the first delta, so a request that
  fails before producing anything (invalid grammar, context overflow) still
  gets a plain JSON error with a proper status code.
- `--batch N` (default 1) lets the one worker decode up to N queued
  requests TOGETHER in lockstep (`Conversation.sendBatchTokensReuse`: one
  m=N weight pass per step instead of N GEMV passes; qwen3/qwen3moe/gemma4
  — the families with a batch forward; other backends warn and serve
  sequentially). The worker takes only what is ALREADY queued — an idle
  server keeps single-request latency, batching engages exactly under
  concurrent load (measured on Qwen3-0.6B q8_0, 4 concurrent 128-token
  requests: 132 → 175 aggregate tok/s, +33%). Per-stream failures are
  isolated: a dropped client, a per-request setup error, or a masked-out
  grammar finishes THAT stream (its slot stays consistent and resendable)
  while the rest keep decoding; constraints ride along as one llguidance
  clone per stream, and slot reuse / `cached_tokens` work per stream
  (`--kv-slots` is raised to at least N — one resident cache per lockstep
  stream — and the RAM guard prices the raised total). No mid-flight
  joins: requests arriving during a batch wait for the next one. Streams
  in a batch of 4+ can differ from a solo run by float-reassociation
  drift (~1e-6 rel, the speculative-verify caveat); a stalled client
  never slows its batch — its frames buffer in the per-request
  `StreamPipe` (previous bullet). Excludes
  `--fleet` (per-request retrieval + sticky adoption are single-stream
  logic).

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

**Function calling** works on backends whose family has a tool convention
(`types.ToolStyle`; qwen3/qwen3moe/qwen35 speak the hermes shape):
declarations and tool history fold into the prompt text
(`src/llm/serving/toolcall.zig` reproduces Qwen3's own template
rendering — `<tools>` system section, `<tool_call>` assistant sections,
`<tool_response>` user sections), the emitter scans replies for
`<tool_call>` regions, and completed calls come back as chat `tool_calls`
/ Responses `function_call` items / Anthropic `tool_use` blocks with
`finish_reason`/`stop_reason` set accordingly. Execution stays with the
client — the server only translates the model's call requests. Malformed
call JSON in a reply passes through as plain content, markers intact —
never dropped.

**Forced tool calls** ("required" / a named function / Anthropic
`any`/`tool`) compile to a lark grammar over the hermes call shape
(`toolcall.forcedCallGrammar`: one alternative per candidate tool, the
arguments as an llguidance `%json` subgrammar) — the guarantee is
constrained decoding, so it needs a `-Dllguidance=true` build (501
otherwise) and excludes `response_format`/`regex`/`lark` on the same
request; grammar completion forces the turn stop, so the reply is exactly
one call. Argument schemas are enforced when they are the wire contract:
always for Anthropic `input_schema`, and under `strict: true` on the
OpenAI dialects — a non-strict declaration gets a permissive object
grammar (its schema may use keywords llguidance rejects, and non-strict
semantics promise no enforcement). Under `tool_choice: auto`, `strict` is
accepted but not grammar-enforced: constraining the whole reply would
also forbid plain-text answers, which auto explicitly allows.

Rejected with a 400/501 naming the offending `param` (never silently
dropped): forced `tool_choice` without llguidance (above), tool fields on
backends without a tool convention, the legacy `functions` API, `n > 1`,
`logprobs`, `logit_bias`, image/audio/file content,
`previous_response_id`/`conversation`/`item_reference` (stateless:
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

**Anthropic Messages** (`/v1/messages`, `src/llm/serving/anthropic.zig`) is
a translation layer over the same normalized request — honored: `system`
(string or text blocks), `messages` with text blocks (prior-turn
`thinking`/`redacted_thinking` blocks are dropped like the responses
dialect's replayed reasoning items), `temperature`/`top_p` (0..1)/`top_k`,
required `max_tokens`, `stream`, `thinking` (`enabled`/`adaptive` switch
the reasoning channel on where one exists; a no-op otherwise, so
thinking-by-default clients stay usable), `output_config.format`
(`json_schema` → the same llguidance constraint), and — on hermes backends
— `tools` with `tool_use`/`tool_result` history and `tool_choice`
`any`/`tool` via the forced-call grammar (the function-calling section
above; declarations without a tool convention are accepted and dropped,
which keeps tool-sending clients usable as plain chat). `stop_sequences`
stop generation before the matching text streams and are attributed —
`stop_reason` is `stop_sequence` with the fired sequence echoed in the
`stop_sequence` field. Server-side tool types and images/documents are
rejected explicitly. Errors use the Anthropic
envelope `{"type":"error","error":{type,message}}` with the type derived
from the status; mid-stream failures arrive as an `error` event.

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
- **Anthropic Messages**: `event:` + `data:` framing, no `[DONE]`:
  `message_start` (skeleton; usage zeroed — prompt tokens are only known
  once generation finishes) → `content_block_start`/`content_block_delta`/
  `content_block_stop` per block (`thinking` first when reasoning streams,
  closed by an empty `signature_delta`; then `text`; then a `tool_use`
  block per captured call, its input as one `input_json_delta`) →
  `message_delta` (`stop_reason` `end_turn`/`max_tokens`/`tool_use`, full
  usage incl. `cache_read_input_tokens` from the KV prefix reuse) →
  `message_stop`. `ping` keepalives are not sent (optional in the
  protocol).
- Tool calls stream as soon as their `</tool_call>` closes: chat as
  `delta.tool_calls` chunks (one chunk per call, full arguments),
  responses as `function_call` items (`output_item.added` →
  `function_call_arguments.delta`/`.done` → `output_item.done`, sequenced
  after the message item at finish).
- Deltas are UTF-8-boundary-safe: a token ending mid-code-point carries into
  the next frame instead of corrupting the JSON.
- Verified against openai-python 2.45.0 end-to-end (both dialects, stream +
  non-stream, strict stream helper, structured outputs, error types);
  `/v1/messages` verified live with Claude Code as the client.

## Reasoning and the `<think>` head

Reasoning is OFF by default (JSON-first serving; a grammar constraint forces
it off — the constraint governs the reply from token 0,
CONSTRAINED-DECODING.md §7). Clients enable it per request via
`reasoning_effort` / `reasoning.effort` / `thinking` (anthropic messages);
on the OpenAI dialects anything but `none`/`minimal` needs `caps.think`
(qwen3 only today). The emitter scans the reply head whenever the family
has think markers: a leading `<think>…</think>` block routes to
`reasoning_content` (chat), a reasoning output item (responses), or a
`thinking` content block (anthropic messages), and the stray leading
`</think>` qwen3 emits under the primed-empty think block of no-think
prompts is dropped as a template artifact — reply `content` stays clean
either way.

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

The GGUF-served set comes from the architecture registry
(`llm.registry`): `serving.openFromFile` dispatches on
`general.architecture` and builds the family backend inside the library.
The CLI keeps only the two non-registry backends (diffusion-gemma,
nanochat).

| Backend | Path | Grammar | Reasoning | Stop strings | Streams |
|---|---|---|---|---|---|
| qwen3 / qwen3moe | generic `Conversation` adapter (`serving.gguf_chat`, via `serving.open`) | ✓ | ✓ (`<think>` routing) | ✓ | per token |
| gemma4 | same adapter (SPM tokenizer, `<turn|>` + extra stop ids, GGUF `general.sampling.*` defaults) | ✓ | — | ✓ | per token |
| diffusion-gemma | `backend_diffusion.zig` over `dg.generate` | — | — | — (EOG-trimmed blocks) | per committed block |
| inkling | `llm.inkling.serving` over `llm.inkling.chat.Engine` (wire-format renderer, sampler; via `serving.open`) | ✓ | ✓ (`<\|content_thinking\|>` → `<\|content_text\|>` routing) | — | per token (no cross-request KV reuse) |
| qwen35 / qwen35moe (Qwen3.5 / Qwen3.6 / Ternary-Bonsai) | `llm.qwen35.serving` over `llm.qwen35.chat.Engine` (ChatML + Qwen3.6 think prefill, sampler; via `serving.open`) | ✓ | ✓ (`<think>` routing; the prompt-prefilled opener is injected into the stream) | — | per token (no cross-request KV reuse) |
| deepseek4 (DeepSeek V4 Flash) | `llm.deepseek4.serving` (token-level chat renderer, chunked session prefill + step decode; via `serving.open`) | ✓ | — | ✓ | per token (no cross-request KV reuse) |
| nanochat | `backend_nanochat.zig` over its own Engine (`--nanochat` dir) | — | — | — | per token (no system role: 400) |

Absent sampling fields default to the model's recommended settings (qwen3
no-think 0.7/20/0.8; gemma4 and qwen35 from GGUF metadata), not OpenAI's
nominal `temperature=1` — same deviation llama.cpp makes. The qwen35
backend rides its own engine rather than `Conversation`: the family's
cache carries recurrent conv/state matrices that cannot be truncated back
to a token prefix, so the KV-slot reuse tiers (and `--kv-cache-dir`) do
not apply — every request prefills from scratch on a fresh cache.

Adding a family = implementing the two-function `Backend` vtable
(`llm.serving`, `src/llm/serving/contract.zig` — the model-agnostic serving
contract, so an out-of-tree server consumes it without vendoring
lmserve): `validate` (cheap, connection-thread: message
shape + prompt length) and `generate` (worker-thread: stream reply bytes
into the sink, return token counts + finish reason). Families served by
`llm.chat.Conversation` get this for free from the generic adapter.

Verification status: qwen3 proven end-to-end (openai-python SDK suite,
constrained decoding, concurrency, cancellation, shutdown — 2026-07-12,
Qwen3-0.6B-Q8_0); nanochat proven mechanically against the goldens' base d6
checkpoint; gemma4 and diffusion-gemma compile+unit-verified (no GGUF of
either on local disk when this landed).
