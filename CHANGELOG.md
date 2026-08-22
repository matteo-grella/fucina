# Changelog

Fucina versions as `0.MINOR.PATCH`: a MINOR release may change public API,
and every change that does is listed here with its one-line rewrite.
Deprecations follow the contract in `docs/DEVELOPMENT.md` §6 (renames keep a
`Deprecated:`-marked alias for one MINOR release; signature changes land
directly with the rewrite documented here). The changelog starts at this
point; earlier history is `git log`.

## Stability tiers

- **Stable** (deprecation contract applies): `tensor`, `ag` (the public
  `Tensor` and autograd pillars), `exec`/`ExecContext`, `gguf`, `weights`,
  `gguf_meta`, `safetensors`, `optim`, `lora`, `parallel`, `tuning`,
  `llm.serving`, `llm.chat`, `llm.tokenizer`, `llm.kv_cache`.
- **Experimental** (may change without an alias step, changelog entry
  only): `es`, `ptqtp`, `llm.runner`, `llm.speculative`, `llm.subq`, the
  research features under model families (SHINE, cartridges, Engram), and
  every model family's internal layout.

## 0.2.0 — 2026-08-22

### Added

- `fucina.tuning`: the shared shape of every `FUCINA_*` route gate
  (`Switch`/`Threshold`: read-once env cache + measured default +
  programmatic `set`), the numeric route defaults, and per-context
  `Overrides` — `ExecContext.setTuning` lets two contexts in one process
  run different route policy (first consumer: the CPU f32 weight-shadow
  route).
- `llm.serving`: the model-agnostic serving contract (`GenerateRequest`,
  `GenerateResult`, `Caps`, the per-family `Backend` vtable), promoted from
  `examples/lmserve` so an out-of-tree server consumes it without vendoring
  the example.
- `llm.runner` (experimental): the descriptor runner — one
  family-independent decoder driven by a runtime `Descriptor`, with two
  block styles (`.fused`: the qwen3/qwen3moe vocabulary; `.host_reference`:
  the GLM/DeepSeek-MoE vocabulary — biased QKV, partial interleaved rope,
  sigmoid noaux MoE + shared experts). `Descriptor.fromGguf` reads
  qwen3-shaped and glm4moe GGUF metadata. Three bitwise parity gates in
  `runner_tests.zig`: real Qwen3-0.6B vs the hand qwen3 port, a synthetic
  in-test MoE GGUF (the CI-safe small-MoE fixture), and a synthetic
  glm4moe GGUF loaded purely from its metadata vs the hand glm4moe port
  (docs/RUNNER.md).
- `dtype.block_formats`: the block-quantization format registry.
  `Storage`, `kind`, `blockSize`, and GGUF's type mapping derive from the
  one table; a comptime completeness check makes an unregistered `DType`
  tag a compile error.
- `lmserve --cors-origin O`: cross-origin browser access is opt-in per
  origin (`*` for any).

### Changed

- **One op per name**: the `Ext` reduction/loss variants folded into their
  base ops, which now take a trailing options argument (`.{}` for
  defaults). Rewrites: `x.sum(ctx, tag)` → `x.sum(ctx, tag, .{})`, same
  for `mean`/`max`/`min`; `x.crossEntropy(ctx, tag, labels)` →
  `x.crossEntropy(ctx, tag, labels, .{})`;
  `sumExt`/`meanExt`/`maxExt`/`minExt`/`crossEntropyExt` →
  the base name, same signature; `linearCrossEntropyExt` →
  `linearCrossEntropy`; `linearDistillExt` → `linearDistill`. The typed
  (non-f32) `sum`/`mean`/`max`/`min` take the same trailing `.{}`.
- Route-gate env switches read through `parallel.envFlag` uniformly (the
  getenv-truthiness contract, first character not `'0'`); previously three
  gates keyed on presence-of-any-value, and one
  (`FUCINA_NO_ATTN_BWD_BLAS`) read `std.c.getenv` directly, which
  libc-free Linux builds cannot compile.
- `src/ag` reorganized: `backward.zig` is a re-export facade over
  per-domain VJP modules (`ag/backward/`), `ag/tensor.zig` a facade over
  per-domain method mixins (`ag/tensor/float/`) plus the typed-constant
  and plumbing bands — public API unchanged.
- Root golden fixtures moved beside their consumers
  (`examples/voiceagent/goldens/`, `src/llm/qwen3tts/goldens/`).

### Deprecated

- `sumExt`, `meanExt`, `maxExt`, `minExt`, `crossEntropyExt`,
  `linearCrossEntropyExt`, `linearDistillExt` — aliases of the merged ops
  above; removal in the next MINOR release.
- `llm.weights` / `llm.ptqtp_gguf` / `llm.gguf_meta` — model I/O lives at
  the root: `fucina.weights` / `fucina.ptqtp_gguf` / `fucina.gguf_meta`;
  removal in the next MINOR release.

### Security

- `lmserve` no longer sends `access-control-allow-origin: *` by default:
  with no API key configured, any web page could read responses from a
  loopback bind on browsers without Private Network Access enforcement.
  CORS is now off unless `--cors-origin` is given.

## 0.1.0 — 2026-08-17

First tagged package: `build.zig.zon` + `zig fetch` consumption of the
`fucina` and `fucina_llm` modules, module-carried BLAS/GPU linkage.
