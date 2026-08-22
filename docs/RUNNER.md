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

## Parity status

`src/llm/runner_tests.zig` pins gate 1 of the design: on real
Qwen3-0.6B GGUFs (Q8_0 and Q4_K_M), the runner's prefill logits and every
decode step's logits along a greedy chain are **bitwise identical** to the
hand `llm.qwen3` port, each side running its own KV cache. The hand port
remains the parity oracle and keeps its serving/training surface
(speculative decode, batched steps, SHINE, cartridges).

Open gates: the MoE arm (qwen3moe) is carried over from the hand port but
has no small-model parity fixture yet; a second GGUF family expressed as a
descriptor with zero interpreter changes is the next milestone (glm4moe
needs interleaved partial rope and sigmoid noaux routing added to the
variant vocabulary first).

## Origin

Extracted from `src/llm/qwen3/model.zig` (its `Config` was already
runtime-valued and architecture-string generic; the runner drops the
family-specific entries — speculative decode, batched spans, SubQuant
glue — and keeps the block machinery). Deliberate consequence: the qwen3
file can later delegate its core forward to the runner and shrink to the
descriptor plus its family extras.
