# LMSERVER — the OpenAI-compatible language-model server example

`zig build lmserve` (`examples/lmserve.zig` + `examples/lmserve/`) exposes the
in-tree language models behind the two OpenAI wire dialects. It is an
example, not a library surface: the shared code lives under `examples/lmserve/`
and integrates a model family through one small `Backend` vtable.

```sh
# qwen3 / qwen3moe / gemma4 / diffusion-gemma GGUFs (arch auto-detected)
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
  cache is not: the GGUF chat backend keeps ONE resident slot — the previous
  request's KV cache plus its token shadow — and each request reuses the
  longest common token prefix with its own render, prefilling only the rest
  (llama.cpp's `cache_prompt`; `Conversation.initWarm`/`takeCache`/
  `sendRenderedReuse` in `examples/lmserve/backend.zig`). Follow-up turns of
  a chat re-prefill only the last reply + new message; a non-matching
  request costs one full prefill, exactly as before. The reuse is reported
  as `cached_tokens` in usage (`prompt_tokens_details` /
  `input_tokens_details`). Multi-slot pools and evict-to-disk
  (`llm.kv_persist`) are future work.
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
| nanochat | `backend_nanochat.zig` over its own Engine (`--nanochat` dir) | — | — | — | per token (no system role: 400) |

Absent sampling fields default to the model's recommended settings (qwen3
no-think 0.7/20/0.8; gemma4 from GGUF metadata), not OpenAI's nominal
`temperature=1` — same deviation llama.cpp makes. qwen3.5 is not servable
until `Conversation` generalizes over its DeltaNet cache type.

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
