# The descriptor runner

`llm.runner` (`src/llm/runner.zig`) is one family-independent decoder
implementation driven by a runtime `Descriptor` instead of per-family
forward code — Level 0 of the universal checkpoint runner design
(descriptor + weights → runnable model; the follow-on levels are the
metadata compiler for more families and the divergence-guided recovery
loop).

## What it is

- **`Descriptor`** — the model as data: dims (layers, heads, kv heads,
  head dim, FFN size), norm eps, rope theta, and the dense/MoE FFN
  selection. `Descriptor.fromGguf` is the seed metadata compiler: it reads
  any architecture that follows the qwen3-shaped GGUF key layout
  (`<arch>.embedding_length`, `attention.head_count`, `expert_count`, ...)
  and emits the descriptor — dense or MoE — from the file's own
  `general.architecture` string.
- **`Model`** — the interpreter: loads weights through the shared model-I/O
  band (`fucina.weights`, `gguf_meta.parallelLoadLayers`, PTQTP sidecars,
  MoE expert streaming) and runs the same facade ops the hand ports use —
  fused norm+QKV projection, half-rope QK-norm, grouped causal attention
  over the shared `KvCache`, dense/MoE FFN with the packed fast paths. It
  inherits every kernel, threading decision, and the buffer-pool memory
  discipline; there is nothing between the descriptor walk and the facade.

## Block styles

The descriptor's `block_style` selects the block implementation —
structural vocabulary, not family names:

- **`.fused`** — the fused-kernel decoder shape: QK-norm + half-rope over
  the full head dim, unbiased QKV, softmax top-k MoE (the qwen3/qwen3moe
  vocabulary).
- **`.host_reference`** — the auditable host-side f32 shape: biased QKV,
  partial rope with selectable pairing, sigmoid noaux MoE with router
  bias, shared experts, and leading dense layers (the GLM-4.5 /
  DeepSeek-MoE vocabulary, ported verbatim from the hand glm4moe block).
  Heavy linears and the fused MoE mixture run on fucina kernels in both
  styles; `hostStep` is this style's forward entry.

## Parity status

`src/llm/runner_tests.zig` pins three parity gates, all **bitwise**:

- **Gate 1 (real weights)**: on real Qwen3-0.6B GGUFs (Q8_0 and Q4_K_M),
  prefill logits and every decode step's logits along a greedy chain match
  the hand `llm.qwen3` port, each side running its own KV cache (native
  backend, skips without `models/`).
- **The small-MoE fixture**: a synthetic qwen3moe GGUF (built in-test via
  `gguf.Writer`, q8_0 expert stacks, no local models needed) matches the
  hand port bitwise through prefill and a greedy decode chain — the MoE
  arm's CI-safe pin.
- **Gate 3 (second family as data)**: a synthetic glm4moe GGUF loads
  purely through `Descriptor.fromGguf` — no family code in the test — and
  matches the hand `llm.glm4moe` port bitwise per position (partial
  interleaved rope, QKV biases, sigmoid noaux routing + router bias,
  renormalized scaled weights, one shared expert, one leading dense
  layer).

The hand ports remain the parity oracles and keep their serving/training
surfaces (speculative decode, batched steps, SHINE, cartridges, MTP).

Open: validation against a real GLM-4.5 GGUF (none in the local model
set), an independent speed measurement for gate 2 (currently satisfied by
construction — the `.fused` style issues the hand port's exact call
sequence), and the next vocabulary entries (MLA, KDA-recurrent, sliding
window).

## Origin

Extracted from `src/llm/qwen3/model.zig` (its `Config` was already
runtime-valued and architecture-string generic; the runner drops the
family-specific entries — speculative decode, batched spans, SubQuant
glue — and keeps the block machinery). Deliberate consequence: the qwen3
file can later delegate its core forward to the runner and shrink to the
descriptor plus its family extras.
