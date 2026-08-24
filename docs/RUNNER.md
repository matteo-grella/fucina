<!-- docs-nav: group="Run & serve" title="Descriptor runner" weight=15 -->
# The qwen3 descriptor runner

`models.qwen3.runner` (`src/models/qwen3/runner.zig`) is one family-independent decoder
implementation driven by a runtime `Descriptor` instead of per-family
forward code — Level 0 of the universal checkpoint runner design
(descriptor + weights → runnable model; the follow-on levels are the
metadata compiler for more families and the divergence-guided recovery
loop). It is also the production decoder for the qwen3 family:
`src/models/qwen3/model.zig` is an alias surface over it (`Config` IS
`runner.Descriptor`, `Model` IS `runner.Model`), and the glm4moe family's
trunk runs on its host_reference blocks.

## What it is

- **`Descriptor`** — the model as data: dims (layers, heads, kv heads,
  head dim, FFN size), norm eps, rope theta, and the dense/MoE FFN
  selection. `Descriptor.fromGguf` is the seed metadata compiler: it reads
  any architecture that follows the qwen3-shaped GGUF key layout
  (`<arch>.embedding_length`, `attention.head_count`, `expert_count`, ...)
  and emits the descriptor — dense or MoE — from the file's own
  `general.architecture` string, plus the glm4moe metadata shape for the
  host_reference style.
- **`Model`** — the interpreter: loads weights through the shared model-I/O
  band (`fucina.weights`, `gguf_meta.parallelLoadLayers`, PTQTP sidecars,
  MoE expert streaming) and runs the facade ops directly — fused norm+QKV
  projection, half-rope QK-norm, grouped causal attention over the shared
  `KvCache`, dense/MoE FFN with the packed fast paths. It inherits every
  kernel, threading decision, and the buffer-pool memory discipline; there
  is nothing between the descriptor walk and the facade. Because the qwen3
  family serves through it, the `.fused` style also carries the batched
  decode entries (`forwardStepBatch`, `forwardStepBatchSpans`) and the
  nullable SubQ research seam. `runner.Model` conforms to the decoder
  contract (`models.decoder`: `Cache` = the shared `KvCache`, `caps` =
  rewind + batch, `initCache`, `forwardStep`), so every generic layer
  (chat, speculative, serving, generate) hosts it.

## Block styles

The descriptor's `block_style` selects the block implementation —
structural vocabulary, not family names:

- **`.fused`** — the fused-kernel decoder shape: QK-norm + half-rope over
  the full head dim, unbiased QKV, softmax top-k MoE (the qwen3/qwen3moe
  vocabulary).
- **`.host_reference`** — the auditable host-side f32 shape: biased QKV,
  partial rope with selectable pairing, sigmoid noaux MoE with router
  bias, shared experts, and leading dense layers (the GLM-4.5 /
  DeepSeek-MoE vocabulary). Heavy linears and the fused MoE mixture run on
  fucina kernels in both styles; `hostStep` is this style's forward entry,
  and `hostLayerForward` carries a nullable `mtp_cache` seam for the
  glm4moe family's MTP head. The two styles guard each other's entries:
  a fused entry on a host model (or `hostStep` on a fused one) returns
  `Error.WrongBlockStyle` instead of computing something wrong.

## Correctness status

`src/models/qwen3/runner_tests.zig` pins the runner against **recorded goldens**:
greedy argmax chains asserted everywhere, plus exact FNV-1a logit-bit
hashes on the machine class the goldens were recorded on (aarch64 native,
Accelerate BLAS) — a refactor that changes a single output bit fails
there.

- **Real weights**: on real Qwen3-0.6B GGUFs (Q8_0 and Q4_K_M), the
  prefill logits fingerprint and a five-step greedy chain match the
  recorded trace, and the no-cache last-logits entry agrees with the
  cached prefill (native backend, skips without `models/`).
- **The small-MoE fixture**: a synthetic qwen3moe GGUF (built in-test via
  `gguf.Writer`, q8_0 expert stacks, no local models needed) matches its
  recorded trace — the MoE arm's model-free pin.
- **The glm fixture**: a synthetic glm4moe GGUF loads purely through
  `Descriptor.fromGguf` — no family code in the test — matches its
  recorded trace, and additionally matches the glm4moe family module
  bitwise per position (the family drives the same host blocks through
  its own `step` API, so this pins the two call paths against each
  other).

The families' own suites anchor the runner further: the real-model chat,
SHINE, TTS-talker, and training goldens all execute runner code for the
qwen3 family, and the deepseek2 recorded-forward anchor pins the sibling
host-band conventions.

Open: validation against a real GLM-4.5 GGUF (none in the local model
set), an independent speed measurement (currently satisfied by
construction — the `.fused` style issues the same facade call sequence the
hand port did), and the next vocabulary entries (per-layer attention
geometry for gemma4, MLA, KDA-recurrent).

## Family status

| family | status |
|---|---|
| qwen3 / qwen3moe | runs ON the runner (`qwen3/model.zig` aliases it) |
| glm4moe | trunk runs on the host_reference blocks; the MTP (`nextn`) head and verify API stay family-side |
| gemma4 | natural next descriptor target (needs per-layer SWA/global geometry, sandwich norms, a second rope table) |
| qwen35, deepseek2, deepseek4, kimi3, inkling | hand ports (recurrent blocks, MLA, hyper-connections, rel-bias attention — vocabulary the descriptor does not carry yet) |
