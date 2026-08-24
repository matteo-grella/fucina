# 1. Introduction and mental model

Fucina is an eager, close-to-metal CPU tensor/autograd runtime plus an
LLM/ASR inference stack, written in Zig 0.16. This document is the detailed
reference for the whole library: the public API surface, its exact semantics
(ownership, errors, defaults, thread-safety), and the internal layers you
need to understand to extend it. The structural overview lives in
[ARCHITECTURE.md](../ARCHITECTURE.md); command cheat sheets live in `AGENTS.md`;
per-model getting-started recipes live in the per-example
`examples/<name>/README.md`, with the shared weights table and runtime knobs
in [RUNNING-MODELS.md](../RUNNING-MODELS.md).

Every runnable Zig snippet in this document is machine-verified against the tree:
`zig build snippet-check` ([§2.7](02-toolchain-build-and-project-wiring.md#27-test-organization-src-examples), a CI step) extracts every runnable
snippet written as a named `test` block and runs it against the real
modules; snippets that need model assets are non-test fragments the
harness ignores and are marked `// requires model assets to run`.

## 1.1 The two modules

The build exposes two library modules ([§2](02-toolchain-build-and-project-wiring.md)):

- **`fucina`** (`src/fucina.zig`) — the tensor library: the public `Tensor`
  facade, the `ExecContext` runtime, autograd, quantized formats, training
  (optimizers, ES, LoRA), and persistence (GGUF, safetensors, checkpoints).
- **`fucina_models`** (`src/models.zig`) — the model stack built on top: GGUF
  weight binding, KV caches, tokenizers, samplers, chat sessions,
  speculative decoding, and the model families (Qwen3, Qwen3.5, Gemma 4,
  DiffusionGemma, DeepSeek V2/V3, GLM-4.5, DeepSeek V4 Flash, Kimi-K3,
  Inkling, Parakeet ASR).

Applications (`examples/`, `tools/`, `bench/`) sit above both.

## 1.2 Mental model

Five ideas carry the whole library:

**Eager, explicit execution.** There is no graph compiler, no lazy
evaluation, no fusion pass. Every tensor operation validates its inputs,
allocates its output through the `ExecContext`, and calls a backend kernel
immediately. What you write is what runs, in the order you wrote it.

**One public tensor, typed at comptime.** `fucina.Tensor(spec)` is the only
user-facing tensor type. Its axis *tags* (names) and rank are part of the
Zig type — checked at compile time — while dimension *sizes* stay runtime
values. The same facade carries no-grad inference and gradient-tracked
training: a tensor with gradient state records backward information, a
constant does not, and the operation call sites are identical. The raw,
untagged tensor underneath is deliberately not exported ([§8](08-data-types-storage-and-the-raw-tensor-layer-internal.md)).

**Tags instead of axis numbers.** Operations name the axes they act on
(`x.dot(&ctx, &w, .in)` contracts the `.in` axis) and broadcasting is
tag-driven: axes align by name, not by position. The tag algebra is
comptime-only data — it compiles down to stride manipulation on the raw
tensor with zero runtime tagging cost ([§7](07-named-axes-the-tag-algebra.md)).

**Explicit ownership, deterministic cleanup.** Tensors are value handles
over reference-counted buffers. Operations return owned results;
`defer x.deinit()` is the norm. Training loops use exec scopes to own the
flood of intermediates implicitly ([§6](06-the-execution-runtime-execcontext-and-the-memory-model.md)). Loaded model weights borrow mmap'd
bytes (holder-managed lifetime) or device-resident bytes (freed through
storage release hooks) instead of copying ([§8](08-data-types-storage-and-the-raw-tensor-layer-internal.md), [§12](12-model-io-gguf-and-safetensors.md)).

**Multi-dtype with a sealed policy.** Tensors span bool/integer dtypes,
`f16`/`bf16`/`f32`/`f64`, the OCP FP8 storage floats
(`f8_e4m3`/`f8_e5m2`), and the GGML block-quantized formats. What each
dtype branch can do is enforced by the type system, not runtime checks:
`.f32` is the differentiable branch; other scalar dtypes are constant typed
tensors (floats additionally get forward-only math); block-quantized tensors
are constant inference tensors that dequantize, gather rows, and serve as
matmul right-hand sides ([§3](03-tensors-types-construction-and-data-access.md), [§10](10-quantization.md)). Float compute/output dtypes follow a
fixed per-op-family policy ([§8.3](08-data-types-storage-and-the-raw-tensor-layer-internal.md#83-float-computeoutput-dtype-policy-srcdtypezig)).

## 1.3 Layer stack

Top-down; a band depends only on bands at or below it. Both halves are
machine-enforced by `zig build arch-check` ([§2](02-toolchain-build-and-project-wiring.md)): acyclicity of the production
import graph, and band *direction* against this same table, encoded as
`band_table` in `tools/check_import_graph.zig`. A production file in no band
fails the check too, so this table and that one cannot drift apart (see
[ARCHITECTURE.md](../ARCHITECTURE.md)):

| Band | Contents | Reference |
| --- | --- | --- |
| apps | `examples/`, `tools/`, `bench/` | [§14](14-model-families-and-example-applications.md) |
| models | `fucina_models` module | [§13](13-the-model-stack-fucina_models.md), [§14](14-model-families-and-example-applications.md) |
| facade | `src/fucina.zig` public root | [§1](01-introduction-and-mental-model.md)–[§5](05-automatic-differentiation.md) |
| autograd + training | `src/ag/`, optim/es/lora/persistence | [§5](05-automatic-differentiation.md), [§11](11-training-optimizers-evolution-strategies-lora-and-checkpoints.md), [§12](12-model-io-gguf-and-safetensors.md) |
| tagged ops | `src/tag_ops.zig` | [§7](07-named-axes-the-tag-algebra.md) |
| exec runtime | `ExecContext`, `src/exec/` | [§6](06-the-execution-runtime-execcontext-and-the-memory-model.md) |
| backends | CPU SIMD, BLAS, Metal/CUDA | [§9](09-backends-cpu-simd-blas-threading-and-gpu-offload.md), [§10](10-quantization.md) |
| tensor/storage/dtype | raw value types | [§8](08-data-types-storage-and-the-raw-tensor-layer-internal.md) |

## 1.4 A first program

The canonical smoke test — build two variables, contract them, reduce, and
differentiate:

```zig
const std = @import("std");
const fucina = @import("fucina");

test "first program" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    // x: [batch=1, in=2], w: [in=2, out=1]
    var x = try fucina.Tensor(.{ .batch, .in }).variable(&ctx, try ctx.fromSlice(.f32, &.{ 1, 2 }, &.{ 2, 3 }));
    defer x.deinit();
    var w = try fucina.Tensor(.{ .in, .out }).variable(&ctx, try ctx.fromSlice(.f32, &.{ 2, 1 }, &.{ 4, 5 }));
    defer w.deinit();

    var y = try x.dot(&ctx, &w, .in); // contract .in => [batch, out]
    defer y.deinit();
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();

    try loss.backward(&ctx);
    var gx = (try x.grad(&ctx)).?; // dloss/dx = w^T = [4, 5]
    defer gx.deinit();

    try std.testing.expectApproxEqAbs(@as(f32, 23.0), try loss.item(), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), (try gx.dataConst())[0], 1e-6);
}
```

Everything in this snippet — the context lifecycle, tensor specs,
construction, ownership, tagged contraction, backward — is unpacked in
[§3](03-tensors-types-construction-and-data-access.md)–[§6](06-the-execution-runtime-execcontext-and-the-memory-model.md).

## 1.5 Stability

Fucina is a production-oriented core, not a finished 1.0 product: the
package manifest and 0.x tags (`v0.3.0`) exist so consumers can pin a
version ([§2.5](02-toolchain-build-and-project-wiring.md#25-consuming-fucina-from-another-project)), but a 0.x tag is a pin, not a semver stability contract —
the public API may change between tags (see
*Current Production Gaps* in [ARCHITECTURE.md](../ARCHITECTURE.md)). This
reference describes the tree it ships with; sections marked *internal*
([§7](07-named-axes-the-tag-algebra.md) library level, [§8](08-data-types-storage-and-the-raw-tensor-layer-internal.md), backend internals in [§9](09-backends-cpu-simd-blas-threading-and-gpu-offload.md)) document machinery that is
explicitly not a stable API.
