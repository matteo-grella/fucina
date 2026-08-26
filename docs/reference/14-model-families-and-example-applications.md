# 14. Model families and example applications

The `fucina_models` module root (`src/models.zig`) exposes each model family as a
namespace — `models.qwen3.{model,train}`,
`models.qwen35.{model,chat}`,
`models.gemma.{model,train,moe}`,
`models.diffusion_gemma.model`, `models.deepseek2.model`, `models.glm4moe.model`,
`models.deepseek4.model`, `models.inkling.{model,mmproj,chat}`, `models.parakeet.*`,
`models.text.speculative.*`, plus the research tier under `models.research.*`
(`subq`, `engram`, `shine`/`shine_train`, `models.research.kimi3.model`) — while the
generic helpers (`fucina.weights`, `models.text.kv_cache`, `models.text.kv_persist`, `models.text.tokenizer`,
`models.text.spm_tokenizer`, `models.text.sampler`, `models.text.logit_processor`, `models.text.llguidance`,
`models.text.chat`, `models.text.data`, `fucina.gguf_meta`, `fucina.ptqtp_gguf`, `models.text.cartridge`,
`models.text.cartridge_fleet`,
`models.text.unicode_categories`) stay flat and are covered in [§13](13-the-model-stack-fucina_models.md). This section
documents the per-family model
APIs, their runner CLIs, and the standalone programs under `examples/`
(single-file teaching code) and `apps/` (product- and port-shaped
applications). `zig build run` (`apps/run/`) is the registry runner: one
binary that serves every architecture registered in `models.registry`.
Weight containers (`LinearWeight` and its quant arms), `KvCache`, tokenizers,
sampling, chat orchestration, and speculative decoding are [§13](13-the-model-stack-fucina_models.md) material;
GGUF parsing is [§12](12-model-io-gguf-and-safetensors.md); LoRA/optimizer/ES mechanics are [§11](11-training-optimizers-evolution-strategies-lora-and-checkpoints.md).

## 14.1 Conventions shared by every family

(`src/models/*/model.zig`, `src/gguf_meta.zig`)

**Config from GGUF metadata.** Each family's `Config.fromGguf(file)`
(deepseek4/inkling additionally take an allocator; kimi3 is the one
non-GGUF family — it loads a reference checkpoint directory, [§14.7](14-model-families-and-example-applications.md#147-kimi-k3--kdamla-hybrid-architecture-parity-srcmodelsresearchkimi3modelzig)) reads
hyperparameters from the standard GGUF key convention: the value of
`general.architecture` (e.g. `"qwen3"`, `"qwen3moe"`, `"qwen35"`, `"gemma4"`,
`"diffusion-gemma"`, `"parakeet"`) prefixes every key —
`<arch>.block_count`, `<arch>.embedding_length`,
`<arch>.attention.head_count`, `<arch>.rope.freq_base`, and so on — so one
loader covers every size of a family without hardcoding. The vocab size comes
from the `token_embd.weight` tensor shape, not a key. The `fucina.gguf_meta`
helpers (`metaInt`, `metaIntOpt`, `metaFloat`, `metaFloatOpt`) implement the
prefixing plus a per-family zero policy: qwen3/qwen35 reject present-but-zero
required ints (`.reject_zero` — every config int is structurally positive),
gemma accepts zeros (`.accept_zero` — keys like
`attention.shared_kv_layers` are legitimately 0). Missing or malformed keys
surface as `Error.InvalidConfig` (parakeet: `Error.MissingMetadata`;
parakeet also departs from the tensor-shape vocab convention — its
`Config.fromGguf` reads `parakeet.vocab_size` from metadata).

**Loader entry points.** The qwen3, qwen35, gemma4 and diffusion_gemma
families expose the same pair (the deepseek2, glm4moe, deepseek4, and inkling
loaders read their `Config` from the file internally; glm4moe additionally
takes `max_positions`, and deepseek2/glm4moe/deepseek4 take a
`LoadOptions`):

```zig
pub fn loadGguf(ctx: *ExecContext, io: std.Io, path: []const u8, config: Config) !Model
pub fn loadGgufFromFile(ctx: *ExecContext, file: *gguf.File, config: Config) !Model
// qwen35 takes `file: *const gguf.File` (it never calls takeMapping);
// a mutable pointer coerces, so caller code is unaffected
```

`loadGguf` opens the file itself (mmap via `gguf.File.loadMmap` for qwen3,
gemma4 and diffusion_gemma; qwen35's convenience arm uses the eager
`gguf.File.load`). `loadGgufFromFile` takes an already-parsed `gguf.File` so
the caller can build a tokenizer from the same file's metadata without a
second read — the pattern every runner uses:

```zig
var file = try fucina.gguf.File.loadMmap(alloc, io, path);
defer file.deinit();
const config = try models.qwen3.model.Config.fromGguf(&file);
var model = try models.qwen3.model.Model.loadGgufFromFile(&ctx, &file, config);
defer model.deinit();
var tok = try models.text.tokenizer.Tokenizer.initFromGguf(alloc, &file, .{});
defer tok.deinit();
```

Weights are materialized through `fucina.weights.LinearWeight.load` ([§13](13-the-model-stack-fucina_models.md)), which
keeps the GGUF dtype resident — f32/f16/bf16 and the quant forms (Q8_0,
Q4_K/Q5_K/Q6_K and the other ggml types) run their own packed kernels; no
global dequantization happens at load. Sibling projections that share a dtype
and layout are fused at load (`weights.fuseLinear`): q/k/v into one QKV
matrix, gate/up into one gate_up matrix — one wider GEMM per block instead of
two or three. Layer loading is parallelized across the work pool
(`gguf_meta.parallelLoadLayers`). Ownership: `Model.deinit` releases every
weight; when expert blocks borrow from the mmap (qwen3 MoE when a
single-file GGUF is mmap'd — split GGUFs' experts are copied, and opt-in
expert disk streaming (`LoadOptions.moe_stream`) leaves the mapping with the
caller; gemma4/diffusion_gemma under `borrow_experts`) the model takes the
mapping via `file.takeMapping()` and unmaps it **last** in `deinit`. On GPU
builds nothing changes at this API level — offload decisions are per-GEMM
work-gates inside the shared kernels ([§9](09-backends-cpu-simd-blas-threading-and-gpu-offload.md)); the two model-level GPU knobs are
gemma-MoE's raw expert representation (14.4) and diffusion_gemma's
`convertDenseWeightsToF16` (14.5).

**Forward/decode surface.** The autoregressive families share the decoder
contract (`models.decoder`, checked at comptime by `assertDecoder(Model)`;
each family declares its `caps`):

- `forwardLastLogits(ctx, token_ids)` — cacheless whole-sequence forward;
  returns the last position's `[1, vocab]` logits (caller deinits). Empty
  input is `Error.InvalidSequenceLength`.
- `initCache(ctx, capacity)` — build the family's `Cache` sized for
  `capacity` positions (the shared [§13.4](13-the-model-stack-fucina_models.md#134-kv-cache-srcmodelstextkv_cachezig) `KvCache` for the attention
  families; recurrent families carry their own state struct with the same
  `len()`/`reset()`/`deinit()` methods). Families whose construction
  ignores `ctx` still take it, so the spelling is uniform. This is the
  construction seam the generic layers use ([§13](13-the-model-stack-fucina_models.md)).
- `forwardStep(ctx, cache, token_ids, pos0)` — process `token_ids` at
  absolute positions `pos0..pos0+len`, advance the cache by `len`, return
  the LAST row's logits as a caller-owned `[1, vocab]` tensor.
  **Contract:** `cache.len() == pos0` (families whose cache tracks position
  internally assert it in debug builds); overflow is
  `kv_cache.Error.KvCacheOverflow` or the family's equivalent. Prefill is
  one call on a fresh cache with `pos0 == 0` (last-token logits equal
  `forwardLastLogits`); decode is a one-token call at
  `pos0 == cache.len()`.
- `forwardStepAllLogits` (iff `caps.rewind`: qwen3, gemma4, glm4moe,
  SHINE's `AdaptedModel`) — same KV semantics, but returns `[len, vocab]`
  logits for **every** appended position: the speculative-decoding verify
  entry ([§13](13-the-model-stack-fucina_models.md) — one batched pass scores all draft positions for ~one step's
  weight traffic).
- `forwardStepBatch` (iff `caps.batch`: qwen3, gemma4) — lockstep
  multi-stream decode, 14.2.
- Generation loop: `models.text.generate` is the one reference loop over the
  contract (`generate`/`generateOutcome` with a `Sampler`, stop ids, and a
  `TokenSink`; `greedy` is the slice-filling argmax convenience that
  resets the cache first and writes the fired stop token). gemma4 keeps
  `Model.generate(ctx, kv, prompt_tokens, out_tokens,
  GenerateOptions{ .max_new_tokens, .stop_token = null })` as a wrapper
  over it; the qwen35 and inkling chat engines call it with their own
  prompt rendering and token sinks. diffusion_gemma's block-diffusion
  `generate` has its own options and returns a `GenerateResult` (14.5).
  Sampled decoding rides `models.text.sampler` ([§13](13-the-model-stack-fucina_models.md)) everywhere.
- `forward*Profiled` variants take `io: std.Io` and a family-specific
  `ForwardProfile` accumulator (per-block wall-clock buckets; the `--profile`
  runner flag).

Forward entries take `*const Model` and mutate only the `ExecContext`,
the cache, and (profiled) the profile struct; a loaded model is read-only
(glm4moe and deepseek4 take `*Model`: their forwards refresh model-held
MTP scratch, and the generic layers accept either spelling through
`decoder.ModelPtr`).
`ExecContext` is single-threaded ([§6](06-the-execution-runtime-execcontext-and-the-memory-model.md)), so concurrent streams over one model
need one context and one cache per thread. Returned logits are caller-owned
constants (`deinit` them); no exec scope is required for inference.

## 14.2 Qwen3 — dense and MoE (`src/models/qwen3/model.zig`)

The reference transformer family and the most complete runner: standard GQA
attention with per-head q/k RMSNorm and full RoPE, SwiGLU FFN — dense, or a
top-k routed mixture (`qwen3moe`, e.g. 30B-A3B) selected purely by GGUF
metadata.

```zig
pub const Config = struct {
    vocab_size, hidden_size, intermediate_size, num_layers,
    num_attention_heads, num_key_value_heads, head_dim: usize,
    rms_norm_eps, rope_theta: f32,
    num_experts: usize = 0,          // 0 = dense
    num_experts_used: usize = 0,
    moe_intermediate_size: usize = 0,
    norm_topk_prob: bool = true,
    moe_expert_top_p: f32 = 1.0,     // adaptive expert top-p; 1.0 = full top-k (runtime knob, not GGUF)
    pub fn isMoe(self: Config) bool
    pub fn qwen3_0_6b() Config       // hardcoded 0.6B reference config
    pub fn fromGguf(file: *const gguf.File) !Config
};
```

`fromGguf` reads `<arch>.{embedding_length, feed_forward_length, block_count,
attention.head_count, attention.head_count_kv, attention.key_length,
attention.layer_norm_rms_epsilon, rope.freq_base}` plus the MoE trio
`{expert_count, expert_used_count, expert_feed_forward_length}`; a model with
no `expert_count` key stays dense. Validation (at load) rejects zero heads,
non-divisible GQA grouping, odd `head_dim`, and inconsistent MoE fields with
`Error.InvalidConfig`.

```zig
test "qwen3 reference config" {
    const cfg = models.qwen3.model.Config.qwen3_0_6b();
    try std.testing.expect(!cfg.isMoe());
    try std.testing.expectEqual(@as(usize, 28), cfg.num_layers);
    try std.testing.expectEqual(@as(usize, 8), cfg.num_key_value_heads);
}
```

`pub const Error = weights.Error || error{ InvalidConfig,
InvalidSequenceLength, MismatchedKvCaches, KvCacheOverflow, WrongBlockStyle }`
(`WrongBlockStyle`: a fused entry — `forwardStep*`, `forwardLastLogits*`,
`initCache` — called on a `.host_reference` model, or `hostStep`/
`initHostCache` on a `.fused` one). Public surface on `Model`:
`Cache` (= the shared `KvCache`), `caps` (`.{ .rewind = true, .batch =
true }`), `loadGguf`, `loadGgufOptions`, `loadGgufFromFile`,
`loadGgufFromFileOptions`
(opt-in MoE expert disk streaming, `LoadOptions.moe_stream`), `deinit`,
`forwardLastLogits`, `forwardLastLogitsProfiled`, `initCache`,
`forwardStep`, `forwardStepProfiled`, `forwardStepAllLogits`,
`forwardStepBatch`, `forwardStepBatchSpans`;
plus `ForwardProfile`, `MoeStreamOptions`, `LoadOptions`, `Layer`,
`DenseFfn`, `QkvProjection`/`splitQkv`, `GateUpProjection`/`splitGateUp`
(the block-structure seam train.zig shares — the trainer's differentiable
forward reads the same layer structs and splits fused projections through
the same functions; `DenseFfn`/`GateUpProjection` are re-exports of
`src/models/model_common.zig`, the load band the runner and qwen35 share:
PTQTP-aware projections, the dense-FFN containers, the MoE expert trio,
the embed/norm/lm-head trio, and the GQA head map), and `applyExpertTopP`
at module level. Greedy
generation is the shared `models.text.generate` loop ([§13](13-the-model-stack-fucina_models.md)); PTQTP
decoration/persistence lives in the sibling module `models.qwen3.ptqtp`
(`decorate`, `DecorateOptions`, `save`, [§10.9](10-quantization.md#109-ptqtp-multi-plane-ternary-decomposition-srcptqtpzig-ptqtpmd)).

Load specifics: when the GGUF is mmap'd, MoE expert stacks
(`ffn_{gate,up,down}_exps.weight`) are **borrowed zero-copy** from the
mapping (`weights.loadMoeRhs` with `borrow = file.is_mmap and
!file.isSplit()` — split GGUFs cannot hand over their multiple mappings, so
their experts are copied) instead of copying multi-GB tensors; the model
owns the mapping and unmaps it last.
A missing `output.weight` means tied embeddings (`token_embedding.cloneView`).
The MoE FFN routes on the host (`routerTopK` with
`normalize_selected = norm_topk_prob`) and runs the router-weighted SwiGLU
mixture through `weights.moeSwiGluFfnSeq`: decode (seq 1) uses a fused
expert-parallel GEMV, prefill groups tokens by expert so each expert's
weights are read once per batch.

`initCache` builds a uniform-geometry f16 `KvCache`
(`KvCache.init(ctx, num_layers, num_key_value_heads, head_dim, capacity)`).
Qwen3 is the **only** family whose attention also accepts a q8_0 cache
(construct it with `kv_cache.KvCache.initWithDtype(..., .q8_0)`, [§13](13-the-model-stack-fucina_models.md); the
runner flag is `--cache-type q8_0` — half the KV memory, and since the
integer q8xq8 score path also the long-context speed option: decode meets
f16 by ~8k context and the halved footprint doubles how much context a
RAM budget holds).

```zig
var file = try fucina.gguf.File.loadMmap(alloc, io, "models/Qwen3-0.6B-Q8_0.gguf");
defer file.deinit();
const config = try models.qwen3.model.Config.fromGguf(&file);
var model = try models.qwen3.model.Model.loadGgufFromFile(&ctx, &file, config);
defer model.deinit();

var kv = try model.initCache(&ctx, 512);
defer kv.deinit();
var prefill = try model.forwardStep(&ctx, &kv, &.{ 151644, 872, 198 }, 0); // [1, vocab]
defer prefill.deinit();
var step = try model.forwardStep(&ctx, &kv, &.{9707}, kv.len()); // one decode step
defer step.deinit();
// requires model assets to run
```

**Lockstep batch decode.** `forwardStepBatch(ctx, caches, token_ids)` decodes
one new token per stream, each stream backed by its own sibling cache from
this model's `initCache` (same dtype, distinct pointers, layer count
matching the model — violations return `Error.MismatchedKvCaches`; a full
cache returns `KvCacheOverflow`). Row `s` of the returned
`[n_streams, vocab]` logits is stream `s`'s next-token distribution and every
cache advances by one. The dense trunk (QKV/O-proj, FFN or MoE mixture,
lm_head) runs as ONE m=n pass — weights are read once for all streams, the
batch-decode bandwidth win — while RoPE positions, KV appends and attention
stay per-stream (ragged, each row against its own cache at its own position).
Per-row numerics match per-stream `forwardStep` bit-for-bit below the
m-dependent kernel thresholds (quantized x4-packed kernels engage at n >= 4,
fused FFN at seq >= 12, tiled attention at seq >= 48); beyond them rows can
differ by ~1e-6 reassociation drift. The same thresholds bound
`forwardStepAllLogits` against per-token steps.

```zig
var kv_a = try model.initCache(ctx, 256);
defer kv_a.deinit();
var kv_b = try model.initCache(ctx, 256);
defer kv_b.deinit();
var a = try model.forwardStep(ctx, &kv_a, &.{ 151644, 872 }, 0);
a.deinit();
var b = try model.forwardStep(ctx, &kv_b, &.{ 151644, 8948 }, 0);
b.deinit();
// One m=2 weight pass decodes both streams; row s = stream s's logits.
var logits = try model.forwardStepBatch(ctx, &.{ &kv_a, &kv_b }, &.{ 9707, 3838 });
defer logits.deinit();
// requires model assets to run
```

**Speculative decoding** is available on this family: the runner's `--spec`
drives the draft-model-free SAM + Token-Recycling cascade from
`models.text.speculative` ([§13](13-the-model-stack-fucina_models.md)) with `forwardStepAllLogits` as the verify pass;
`--spec-ref doc.txt` injects a reference document the drafter can copy spans
from. Output is lossless (greedy streams verified identical with and without
`--spec`).

**Runner** (`apps/qwen3/`; `main.zig` owns load + dispatch over the
mode modules `options`/`bench`/`generate`/`verify`/`chat.zig`). The full
CLI surface (chat/REPL, sampling flags, GPU offload, speculative decode,
lockstep multi-stream bench, the q8_0 cache flag, the tokenizer/logit
parity oracles, constrained decoding) is documented in the
[README](../../apps/qwen3/README.md); copy-paste commands are in
[RUNNING-MODELS.md](../RUNNING-MODELS.md).

### 14.2.1 LoRA fine-tuning (`src/models/qwen3/train.zig`)

`models.qwen3.train` trains LoRA adapters over a frozen, possibly quantized
`qwen3.Model` (dense only — MoE configs return `Error.MoeUnsupported`). The
trainer mirrors the inference forward op-for-op but routes every frozen
projection through the differentiable frozen-RHS `dot` (gradients flow to f32
activations only; weight memory stays quantized/f16) and adds trainable A/B
deltas on the projections selected at comptime. Mechanics — adapters,
optimizers, checkpoints, exec scopes — live in [§11](11-training-optimizers-evolution-strategies-lora-and-checkpoints.md); this is the entry-point
map.

```zig
pub const Targets = struct { q: bool = true, k: bool = false, v: bool = true,
                             o, gate, up, down: bool = false };
pub const ignore_index: usize = std.math.maxInt(usize);
pub fn Trainer(comptime targets: Targets) type
```

Module-level symbols: `Error` (`MoeUnsupported`, `ExecScopeRequired`,
`InvalidSequenceLength`, `LabelLengthMismatch`, `InvalidLayerRange`,
`InvalidInjection`, `InvalidCartridge`, `InvalidCapture`, `InvalidPacking`,
`CartridgeCheckpointUnsupported`), `Targets`,
`ignore_index`, `ModelLayer` (alias of the exported `qwen3.Layer` —
the model's per-block layer type), `Hidden`
(`fucina.Tensor(.{ .seq, .embed })`), `Injection` (`{ pos, row }` — a
differentiable single-row embedding override), `ForwardOptions`
(`{ start_layer = 0, layer_count = null, inject = null }` plus the
cartridge fields — `cartridge`, `cartridges`, `capture`,
`packed_segments` — and `residual_hook`, the type-erased research seam
`models.research.engram.ResidualGraft` adapts to; the module-level
`KvCapture`/`ResidualHook`, all [§13.10](13-the-model-stack-fucina_models.md#1310-cartridges-srcmodelstextcartridgezig)-[§13.11](13-the-model-stack-fucina_models.md#1311-engram-srcmodelsresearchengramzig) material).

`Trainer(targets)` members: `init(ctx, model, lora.Config, seed)` /
`deinit`; `registerAllParams(opt)` (registers every A/B under
`layers.<i>.<target>.lora_{a,b}` on anything with `addParamNamed`; the
trainer must outlive the optimizer — params and names are borrowed);
`saveAdapters(writer)` / `loadAdapters(reader)` /
`loadAdaptersWithOptions(reader, optim.LoadOptions)` (clean safetensors state
dict, strict one-to-one on load); `loss(ctx, tokens, labels)` /
`lossExt(..., LossOptions)` (mean CE against pre-shifted labels,
`ignore_index` masks; **must** run inside an open exec scope —
`Error.ExecScopeRequired` otherwise — and returns a scope-owned borrow;
`LossOptions{ .reduction = .mean|.sum, .loss_scale = 1 }` is the gradient-
accumulation seam, TRAINING.md [§4](04-tensor-operations.md)); `lossInjected(...)`;
`evalLastLogits` / `evalLogits` / `evalLastLogitsExt` (dropout off, no step
advance, run under their own scope, return caller-owned constants);
`forwardHidden(ctx, tokens, step, opts)` (raw residual stream, scope
required); the cartridge/engram seams (`captureKv`, `initCartridge`,
`distillLoss`/`distillLossExt`, `lossForwardExt`,
`evalLogitsExt`/`evalLogitsRows`, `embedLastHidden`, `freeTransientRope`)
are [§13.10](13-the-model-stack-fucina_models.md#1310-cartridges-srcmodelstextcartridgezig)-[§13.11](13-the-model-stack-fucina_models.md#1311-engram-srcmodelsresearchengramzig) material; the `checkpoint_layers` field enables
recompute-in-backward per
layer; `n_enabled` and `LayerAdapters` are the comptime target plumbing.
Dropout is deterministic per (step, layer, projection) from the base seed;
RoPE tables are cached per sequence length and freed only in `deinit`.

```zig
const Trainer = models.qwen3.train.Trainer(.{ .q = true, .v = true });
var trainer = try Trainer.init(ctx, model, .{ .rank = 8, .alpha = 16 }, 42);
defer trainer.deinit();

var opt = fucina.optim.AdamW.init(ctx.allocator, .{ .lr = 1e-3 });
defer opt.deinit();
try trainer.registerAllParams(&opt);

const scope = ctx.openExecScope();
defer ctx.closeExecScope(scope);
const tokens: []const usize = &.{ 1, 2, 3, 4 };
const labels: []const usize = &.{ 2, 3, 4, models.qwen3.train.ignore_index };
var loss = try trainer.loss(ctx, tokens, labels);
try loss.backward(ctx);
try opt.step(ctx);
opt.zeroGrad();
// requires model assets to run
```

The end-to-end loop — fine-tune (`zig build finetune`), merge adapters into
dense weights (`zig build export-gguf -- --adapters ... --alpha ...`),
re-quantize, serve — is scripted in
[apps/finetune/README.md](../../apps/finetune/README.md); the
gradient-free twin is `zig build es-finetune` ([§11](11-training-optimizers-evolution-strategies-lora-and-checkpoints.md), TRAINING.md [§13](13-the-model-stack-fucina_models.md)).

## 14.3 Qwen3.5 — Gated-DeltaNet hybrid (`src/models/qwen35/model.zig`)

The `qwen35` GGUF arch is a **hybrid linear-attention transformer** (sibling
of qwen3next, not a Qwen3 variant): every `full_attention_interval`-th block
is full GQA attention (fused Q+gate projection, per-head q/k RMSNorm,
multi-section/partial RoPE, sigmoid output gate); the rest are **DeltaNet
linear** blocks — a causal depthwise conv1d feeding a gated delta-rule
recurrent scan over per-v-head state matrices. Both feed a SiLU dense FFN.

Config adds, on top of the usual attention keys: `rope.dimension_count`
(`rope_n_rot` — partial RoPE when < `head_dim`), `rope.dimension_sections`
(`rope_sections: [4]i32`), `full_attention_interval` (default 4), and the SSM
dims `ssm.{conv_kernel, inner_size, state_size, time_step_rank, group_count}`
(`ssm_d_conv/d_inner/d_state/dt_rank/n_group`), plus `nextn_predict_layers`
and `expert_count`. `Config.isRecurrent(il)` implements the block schedule;
`isMoe()` mirrors qwen3. Validation rejects `qwen35moe` and MTP/NextN
variants with `Error.UnsupportedVariant` (dense text path only).

DeltaNet heads may be **non-uniform**: `ssm_dt_rank` v-heads over
`ssm_n_group` q/k heads (`numVHeads % numKHeads == 0` required), with the
q/k heads broadcast onto the v-heads **tiled** — v-head `h` reads q/k head
`h % numKHeads`, matching `ggml_gated_delta_net`'s `iv1 % ne1` semantics in
both the recurrent and the batched chunked scan. Qwen3.5 dense is uniform
(1:1); **Ternary-Bonsai-27B** (a ternarized Qwen3.6-27B, `general.
architecture = "qwen35"`, weights in the Q2_0 g128 container — [§10.7](10-quantization.md#107-ternary-tq2_0-first-class-tq1_0-decode-only-srcbackendquantternaryzig-ternarymd)) runs
48 v-heads over 16 k-heads across 64 blocks (16 full-attention + 48 linear).
Its tokenizer declares `tokenizer.ggml.pre = "qwen35"` ([§13.5.1](13-the-model-stack-fucina_models.md#135-tokenizers)): the qwen2
rules with `\p{M}` combining marks folded into the word class. Loading it is
the ordinary flow — every projection (embeddings and LM head included) is a
`.q2_0` `LinearWeight`, logit-parity-validated against the PrismML llama.cpp
fork (argmax match at pp1..pp512, cosine ≥ 0.9998 — ≥ 0.99999 on the BLAS
prefill arm — token-ID-exact tokenizer).

```zig
test "qwen35 hybrid layer pattern" {
    const cfg = models.qwen35.model.Config{
        .vocab_size = 151_936, .hidden_size = 1024, .intermediate_size = 4096,
        .num_layers = 24, .num_attention_heads = 16, .num_key_value_heads = 2,
        .head_dim = 256, .rms_norm_eps = 1e-6, .rope_theta = 1_000_000,
        .rope_n_rot = 64, .rope_sections = .{ 11, 11, 10, 0 },
        .full_attention_interval = 4,
        .ssm_d_conv = 4, .ssm_d_inner = 4096, .ssm_d_state = 128,
        .ssm_dt_rank = 32, .ssm_n_group = 16,
    };
    // Every 4th block is full attention; the rest run the DeltaNet scan.
    try std.testing.expect(cfg.isRecurrent(0));
    try std.testing.expect(!cfg.isRecurrent(3));
    try std.testing.expect(cfg.isRecurrent(4));
}
```

`pub const Error = weights.Error || error{ InvalidConfig,
InvalidSequenceLength, UnsupportedVariant, UnsupportedKvCacheDtype }`.
Model surface: `loadGguf`, `loadGgufFromFile`, `deinit`, `blockCounts`
(`.{ attn, linear }` counts for `--info`), `forwardLastLogits` (cacheless,
chunked DeltaNet scan; logit-parity-validated against llama.cpp on
Qwen3.5-0.8B), `initCache`, `forwardStep`, `forwardStepWithScanMode`,
`forwardStepProfiled`, `forwardStepProfiledWithScanMode`; module-level
`LinearScanMode` and `ForwardProfile`.

The streaming state is `Cache`, not a bare `KvCache`: an f16 attention KV
cache (q8_0 caches are rejected with `Error.UnsupportedKvCacheDtype`) plus,
per linear layer, a conv window (`(d_conv-1)*conv_dim` floats) and the
recurrent state matrix (`H*Sd*Sd` floats) — O(1) state per linear layer
regardless of context. `Cache.deinit`, `Cache.reset` (zero all carried
state), `Cache.len()` (current position). `LinearScanMode` selects the
DeltaNet prefill path: `.chunked` (default — exact batched chunked-GEMM) or
`.recurrent` (exact token-by-token scan, forced even for prefill); both are
exact, the choice is performance/validation.

```zig
var file = try fucina.gguf.File.loadMmap(alloc, io, "models/Qwen3.5-0.8B-Q8_0.gguf");
defer file.deinit();
const config = try models.qwen35.model.Config.fromGguf(&file);
var model = try models.qwen35.model.Model.loadGgufFromFile(&ctx, &file, config);
defer model.deinit();

var cache = try model.initCache(&ctx, 256); // KV + conv/SSM state
defer cache.deinit();
var prefill = try model.forwardStep(&ctx, &cache, &.{ 9707, 11, 1879 }, 0);
defer prefill.deinit();
var step = try model.forwardStepWithScanMode(&ctx, &cache, &.{0}, cache.len(), .recurrent);
defer step.deinit();
// requires model assets to run
```

No training entry, no `forwardStepAllLogits`/`forwardStepBatch`, no
speculative decoding on this family. Chat lives in `models.qwen35.chat`
(`src/models/qwen35/chat.zig`): `renderPrompt` renders the shared ChatML
template with the Qwen3.6 generation-prompt think prefill (`<think>\n`
opener when thinking is on; the ChatML empty think block when off), and
`Engine(TokMod).generate` runs one sampled reply per call on a fresh
`Cache` through the shared `models.text.generate` loop (the recurrent state
cannot be truncated to a token prefix, so there is no cross-request KV
reuse). `lmserve` serves the family through it (`models.qwen35.serving`,
via `serving.open` — reasoning channel, JSON-schema/regex/Lark
constrained output; [LMSERVER.md](../LMSERVER.md)); Ternary-Bonsai-27B
([README](../../examples/qwen35/README.md)) is the flagship checkpoint. The
CLI is a loader/parity harness; commands and flags are in the
[README](../../examples/qwen35/README.md).

## 14.4 Gemma 4 — text + MoE (`src/models/gemma/`)

`gemma4` (26B-A4B class) is the geometry-heavy family: 16 query heads over
**per-layer** KV geometry — interleaved local sliding-window (SWA) and global
layers with different head dims, KV-head counts and RoPE bases, trailing
layers that **share** an earlier layer's K/V (`shared_kv_layers`), optional
per-layer embeddings (PLE), per-layer output scales, GeGLU FFNs (shared dense
MLP + a 128-expert top-8 MoE), and final logit softcapping.

Config keys beyond the common set: `attention.key_length_swa`,
`attention.sliding_window`, `attention.shared_kv_layers`,
`rope.freq_base_swa`, `expert_count`/`expert_used_count`/
`expert_feed_forward_length`, `embedding_length_per_layer_input` (PLE width,
0 = disabled), `final_logit_softcapping`, plus the per-layer arrays
`gemma4.attention.sliding_window_pattern` and
`gemma4.attention.head_count_kv` (read by `gguf_meta.readU32OrBoolArray`,
which broadcasts a scalar across layers like llama.cpp's
`get_key_or_arr`).
`Config.fromGguf` wraps `Config.fromGgufArch(file, "gemma4")`; the `arch`
argument exists because diffusion-gemma shares the identical hparam key set
under its own prefix. `Config.borrow_experts` is a **load-time policy field**,
not a GGUF hparam: `true` (the `--experts=borrow` flag) borrows MoE experts
zero-copy from the mmap on CPU builds — load in seconds at ~half the RSS
instead of x4-packing ~20 GB — at some decode-throughput cost; the default
packed path favors peak CPU throughput. Numerically identical either way.

`deriveGeometry(allocator, n_layer, swa_pattern, kv_heads_in,
shared_kv_layers, head_dim_global, head_dim_swa) !LayerGeometry` computes the
per-layer view (`is_swa`, `head_dim`, `kv_heads`, `has_kv`, `kv_ref`;
`LayerGeometry.deinit(allocator)` frees it): the trailing `shared_kv_layers`
layers store no K/V and instead reference the last same-type writer (offset
2 for SWA, 1 for global).

```zig
test "gemma4 shared-KV geometry" {
    const alloc = std.testing.allocator;
    var geom = try models.gemma.model.deriveGeometry(
        alloc,
        4, // n_layer
        &.{ true, true, false, true }, // SWA pattern (false = global)
        &.{ 4, 4, 8, 4 }, // per-layer KV heads
        1, // shared_kv_layers: the last layer stores no K/V
        256, // head_dim_global
        128, // head_dim_swa
    );
    defer geom.deinit(alloc);
    try std.testing.expect(!geom.has_kv[3]);
    try std.testing.expectEqual(@as(usize, 1), geom.kv_ref[3]); // reuses layer 1
    try std.testing.expectEqual(@as(usize, 128), geom.head_dim[3]);
}
```

`pub const Error = weights.Error || error{ InvalidConfig,
InvalidSequenceLength, MismatchedKvCaches, MissingMetadata, PleUnsupported,
UnsupportedExpertType,
UnsupportedKvCacheDtype }`. Model surface: `Cache` (= the shared
`KvCache`), `caps` (`.{ .rewind = true, .batch = true }`), `loadGguf`,
`loadGgufFromFile`,
`deinit`, `initCache` (per-layer geometry:
`KvCache.initPerLayer(ctx, geom.kv_heads, geom.head_dim, capacity)`),
`forwardLastLogits`, `forwardLastLogitsProfiled`, `forwardStep`,
`forwardStepProfiled`, `forwardStepAllLogits` (speculative verify entry —
softcapping applies to every row), `forwardStepBatch`/`forwardStepBatchSpans`
(lockstep multi-stream decode, 14.2 — PLE models rejected with
`Error.PleUnsupported`), `generate` + `GenerateOptions`,
`ForwardProfile`. Only f16 caches are accepted (`requireF16KvCache` returns
`Error.UnsupportedKvCacheDtype` for q8_0). Final logits are softcapped when
`final_logit_softcapping != 0` (the fused `softcap(cap)` kernel at the
file's value). The remaining public symbols are loader/forward
plumbing reused by diffusion_gemma and the trainer: `max_heads` (64),
`metaInt`/`metaIntOpt`/`metaFloat`/`metaFloatOpt`, `LayerGeometry`,
`deriveGeometry`, `MoeFfn`, `PerLayerInject`,
`SeparateAttentionProjection`, `FusedAttentionProjectionKind`,
`FusedAttentionProjection`, `AttentionProjectionResult`,
`AttentionProjection` (with `toResidentF16` and `project`), `Layer`,
`loadLayers` (the pub wrapper over a file-private `LayerLoader`),
`requireF16KvCache`,
`attnBlock`, `ffnBlock`.

```zig
var file = try fucina.gguf.File.loadMmap(alloc, io, "models/gemma-4-26B-A4B-it-UD-Q6_K.gguf");
defer file.deinit();
var config = try models.gemma.model.Config.fromGguf(&file);
config.borrow_experts = true; // zero-copy experts from the mmap (--experts=borrow)
var model = try models.gemma.model.Model.loadGgufFromFile(&ctx, &file, config);
defer model.deinit();

var kv = try model.initCache(&ctx, 512); // per-layer geometry
defer kv.deinit();
var prefill = try model.forwardStep(&ctx, &kv, &.{ 2, 651, 235 }, 0);
defer prefill.deinit();
var step = try model.forwardStep(&ctx, &kv, &.{651}, kv.len());
defer step.deinit();
// requires model assets to run
```

**MoE expert kernels** (`moe.zig` over the family-local `moe_gu.zig`, survey depth). The expert FFN has two weight representations: per-expert
**packed** RHS (`MoeFfn.gate/up/down`, the tested Q6_K/Q8_0 packed kernels —
peak CPU throughput) and **raw** GGUF-layout blocks
(`RawExpertWeights{ gu: .q6_k|.q4_k, dn_blocks, device_owned, borrowed }`,
plus `guBlockCount`), used on `-Dgpu=metal` builds (grouped dequant-in-kernel
Metal GEMMs read them; the loader then keeps ONE representation — tens of
seconds and ~24 GB saved at load), on Q4_K-transcoded experts, under
`--experts=borrow`, and by the trainer. Four entry pairs cover the
(decode | batch) x (packed | raw) matrix: `decodePackedTensor` /
`batchPackedTensor` / `decodeRawTensor` / `batchRawTensor` (tagged-tensor
wrappers) over `decodePacked` / `batchPacked` / `decodeRaw` / `batchRaw`.
The kernel bodies live with the family
(`src/models/gemma/moe_gu.zig`, re-exported as `gemma.moe.moe_gu`); batch
entries consume the shared counting-sort route plan (`exec/moe_chain.zig`,
reached through the `ExecContext.moe_chain` seam) with an expert-major
scatter that stays deliberately serial to keep each token's summation
order fixed against the parity oracles.

**LoRA fine-tuning** (`train.zig`, pointer depth — [§11](11-training-optimizers-evolution-strategies-lora-and-checkpoints.md)).
`models.gemma.train.Trainer(targets)` mirrors the qwen3 trainer over the
gemma4 forward: identical `Targets` struct and defaults (q, v), identical
`ignore_index`, and the same member set — `init(ctx, model, lora.Config,
seed)`, `deinit`, `registerAllParams`, `saveAdapters`, `loadAdapters`,
`loadAdaptersWithOptions`, `loss`, `lossExt` + `LossOptions`,
`evalLastLogits`, `n_enabled`, `LayerAdapters` (per-layer geometry sizes the
k/v adapters); its cartridge seams (`captureKv`, `initCartridge`,
`distillLoss`/`distillLossExt`, `evalLogitsExt`/`evalLogitsRows`,
`embedLastHidden`, `freeTransientRope`, `ForwardOptions`) are [§13.10](13-the-model-stack-fucina_models.md#1310-cartridges-srcmodelstextcartridgezig)
material. Its `Error` set encodes the intentional exclusions checked in
`init`: `PleUnsupported` (PLE models rejected), `SharedKvUnsupported`
(any layer with `has_kv == false`), `RawMoeWeightsRequired` (MoE layers must
retain raw expert blocks — load with `--experts=borrow` or a raw-expert
build; the packed inference-only RHS cannot take gradients), plus
`CartridgeGeometry`, `ExecScopeRequired`, `InvalidSequenceLength`,
`LabelLengthMismatch`.

**Runner** (`examples/gemma4/main.zig`: chat/REPL over the SPM tokenizer
(`models.text.spm_tokenizer`) and the generic
`models.text.chat.Conversation`; sampling defaults come from the GGUF).
Commands (chat, REPL, speculative decode, constrained output, the GPU MoE
arm, the prefill/decode benchmark) are in the
[README](../../examples/gemma4/README.md).

## 14.5 DiffusionGemma — block text-diffusion (`src/models/diffusion_gemma/model.zig`)

The `diffusion-gemma` arch is **not autoregressive**: the transformer is
exactly gemma4 (this module reuses gemma4's layer loader and attn/ffn
blocks), but generation denoises fixed-length token canvases and commits them
block-autoregressively. Two forward modes share one weight set:

- `encodeStep(ctx, kv, token_ids, pos0) !void` — causal prefix pass over the
  prompt or a finalized canvas; exists only for its K/V side effect (the lm
  head is skipped), appends and advances the cache, applies the per-layer
  `enc_layer_output_scale`. Same `pos0`/capacity contract as `forwardStep`.
- `canvasForward(ctx, kv, canvas_ids, sc) ![seq, vocab]` — one
  **bidirectional** denoiser pass over the canvas at absolute positions
  `[kv.len(), kv.len() + C)`. Canvas K/V are written into the cache's scratch
  region past `kv.len()` WITHOUT advancing it (the next step overwrites), so
  the cache is read-only from the caller's perspective; logits are returned
  for every row (softcapped). `sc` is the previous step's self-conditioning
  signal (null on the first step); passing one on a GGUF without the
  `self_cond_*` MLP is `Error.SelfConditioningUnavailable`.

`Config = { base: gemma4.Config, canvas_length, eb: EbParams }`;
`Config.fromGguf` reads the gemma4 keys under the `diffusion-gemma.` prefix
plus `diffusion.canvas_length` (required — `Error.MissingCanvasLength`) and
the optional `diffusion.eb_*` overrides of `EbParams` (defaults are the
reference generation_config: `max_steps = 48`, `t_min = 0.4`, `t_max = 0.8`,
`entropy_bound = 0.1`, `stability_threshold = 1`,
`confidence_threshold = 0.005`). Loading additionally requires both per-layer
scales (`layer_output_scale` via the gemma4 layer loader, the diffusion-only
`enc_layer_output_scale` into `Model.enc_scale`) — `Error.MissingLayerScale`
otherwise — and rejects PLE configs. `pub const Error = gemma4.Error ||
error{ MissingCanvasLength, MissingLayerScale, CanvasLengthMismatch,
KvCapacityTooSmall, SelfConditioningUnavailable }`.

Model surface: `loadGguf`, `loadGgufFromFile`, `deinit`, `initCache`
(per-layer geometry; capacity must cover prefix + one canvas),
`convertDenseWeightsToF16` (dequantize attention q/k/v/o, the shared dense
FFN, the self-conditioning MLP and the lm head to resident f16 so the
m = 256 canvas GEMMs take the f16 GPU path — the `--gpu-f16` flag; ~4.6 GB
extra resident on 26B-A4B, pointless without `-Dgpu=metal`), `encodeStep`,
`canvasForward`.

The **entropy-bound sampler** is exposed as free functions over the canvas
logits: `SamplerOptions` / `SamplerPass` (owns `results` + an optional
`ScSignal`; `deinit(allocator)`) / `samplerPass(ctx, logits, temp, u,
options)` (per-position argmax, entropy of softmax(z/t) and one multinomial
draw, parallelized over positions with caller-pre-drawn uniforms so results
are thread-count independent; also collects the sparse self-conditioning
candidate lists), `ScSignal` (sparse per-row id/prob lists; `deinit`),
`entropyBoundAccept(results, entropy_bound, order, accepted)` (accept
positions by ascending entropy while the cumulative entropy of the
strictly-lower set stays within the bound). The loop drivers:
`denoiseCanvas(model, ctx, kv, canvas, DenoiseOptions) !DenoiseResult`
(denoise one canvas in place — uniform-random init, temperature schedule
t_max→t_min, per-step acceptance + renoise, stable-and-confident adaptive
stop; `DenoiseOptions{ .eb, .seed = 0, .self_conditioning = true, .sampler,
.on_step, .on_step_user }` with `StepInfo` snapshots feeding the runner's
live inline visualization) and
`generate(model, ctx, kv, prompt_tokens, out_tokens, GenerateOptions)
!GenerateResult` (encode the prompt once, then per block: denoise, trim at
the first EOG token — default ids `{1, 106, 50}` — or a repetition-loop
onset, append the kept tokens, encoder-pass the canvas back into the cache;
`.on_block` callback; returns `{ produced, steps, blocks }`).

```zig
const dg = models.diffusion_gemma.model;
var file = try fucina.gguf.File.loadMmap(alloc, io, "models/diffusiongemma-26B-A4B-it-Q6_K.gguf");
defer file.deinit();
const config = try dg.Config.fromGguf(&file); // gemma4 hparams + canvas_length + EB sampler
var model = try dg.Model.loadGgufFromFile(&ctx, &file, config);
defer model.deinit();

const prompt: []const usize = &.{ 2, 651, 235 };
var kv = try model.initCache(&ctx, prompt.len + 2 * config.canvas_length);
defer kv.deinit();
var out: [512]usize = undefined;
const result = try dg.generate(&model, &ctx, &kv, prompt, &out, .{
    .denoise = .{ .eb = config.eb, .seed = 42 },
    .max_new_tokens = 256,
});
_ = out[0..result.produced];
// requires model assets to run
```

No training entry and no speculative decoding (there is no autoregressive
draft/verify seam). Runner: `apps/diffusion_gemma/main.zig` (on a TTY
the reply denoises live inline; `--no-visual` disables). Commands, sampler
knobs, and the `--gpu-f16` arm are in the
[README](../../apps/diffusion_gemma/README.md).

## 14.6 Parakeet ASR (`src/models/parakeet/`)

NVIDIA NeMo FastConformer speech recognition (110M hybrid TDT+CTC through
0.6B multilingual TDT), ported stage-for-stage against parakeet.cpp/NeMo.
The pipeline is a chain of free functions over one `gguf.File` — there is no
monolithic `Model` struct; `ParakeetWeights` is a lazy name-keyed cache:

| stage | module | role |
| --- | --- | --- |
| front end | `frontend.zig` | WAV → 16 kHz mono f32 → preemphasis → STFT power → log-mel (+ per-feature normalization) |
| subsampling | `subsampling.zig` | stride-2 conv2d stack (`subsampling_factor`, 8x on the shipped models), mel → `[T/8, d_model]` |
| encoder | `encoder.zig` | Conformer layers: rel-pos MHA + conv module + macaron half-step FFNs |
| decoder | `decoder.zig` | CTC argmax-collapse, or LSTM predictor + joint (RNNT/TDT greedy) |
| text | `tokenizer.zig`, `transcription.zig` | SentencePiece detokenize; word grouping, timestamps, JSON |
| streaming | `streaming.zig` | cache-aware chunked encoder + carried-state RNN-T session |

**Config and loading** (`loader.zig`). `Config.fromGguf` requires
`general.architecture == "parakeet"` and reads flat `parakeet.*` keys:
`arch` (a `DecoderArch`: `ctc`, `rnnt`, `tdt`, `hybrid_tdt_ctc`,
`hybrid_rnnt_ctc`, with `hasCtc`/`hasTransducer`/`isTdt` predicates —
describes which weights exist, not which decoder runs), the encoder set
`encoder.{d_model, n_layers, n_heads, ff_dim, feat_in, conv_kernel,
conv_norm_type, subsampling_factor, subsampling_conv_channels,
pos_emb_max_len, xscaling}`, the mel front end
`preprocessor.{sample_rate, n_mels, n_fft, win_length, hop_length, preemph,
mag_power, log_zero_guard, normalize}`, `vocab_size`/`blank_id`, the
predictor `decoder.{pred_hidden, pred_rnn_layers}` +
`decoding.max_symbols`, the joint `joint.{joint_hidden, activation}`, and the
TDT duration table `parakeet.tdt.durations` (required iff the arch is TDT;
`max_durations = 16`). Derived accessors: `vPlus` / `checkedVPlus` (joint
output width = vocab + blank + durations), `subsampledFreq` /
`checkedSubsampledFreq`, `durationsSlice`. Supporting enums: `ConvNorm`,
`Normalize`, `JointActivation`. Streaming-variant models additionally carry
`StreamingConfig.fromGguf` (null for offline models): `att_context_left/
right/style` (`AttContextStyle.regular|chunked_limited`), the
`[step0, step>=1]` schedules `chunk_size`/`shift_size`/
`pre_encode_cache_size` (+ `stepIdx`), `cache_drop_size`,
`last_channel_cache_size`, `valid_out_len`, `drop_extra_pre_encoded`.
Multilingual prompt-conditioned models carry `PromptConfig.fromGguf` (null
otherwise) with `resolveLang` mapping a locale to its one-hot index.
`expectTensor`/`TensorClass` and `validateTensors` gate the tensor inventory
at load (`f32_required` vs `quantizable` — the GGUFs ship f16/q8_0/q6_k/
q5_k/q4_k variants of the big matmuls; norms, biases and the featurizer stay
f32). `loadFeaturizer` returns the `Featurizer` (mel filterbank `fb` +
window, **borrowed zero-copy from the mapping** — valid only while the
`gguf.File` lives); `loadPieces` decodes the SentencePiece table (outer
slice caller-freed, pieces borrow the mapping). `Error` covers
`NotParakeet`, `UnsupportedArch`, `UnsupportedConvNorm`, `InvalidConfig`,
`MissingMetadata`, `TensorNotFound`, `TensorShapeMismatch`,
`TensorDtypeMismatch`.

**Weights** (`weights.zig`). `ParakeetWeights.init(ctx, file)` /
`deinit` — a lazy cache mapping tensor names to `LinearWeight`s, built on
first use; `enableF32Blas` pre-converts f32 GEMM operands for the BLAS path
(the `--f32-cache` flag); accessors `getLinear`, `getLinearF32`, `linear`,
`linearD`, `linearQkvD` (fused QKV), `linearPosAllD` (all layers'
`linear_pos` in one GEMM); free `borrowF32` (alignment-checked zero-copy f32
view of mapped bytes). Sessions borrow the weights struct; it must outlive
them.

**Front end** (`frontend.zig`): `Audio` (+ `deinit`), `loadWav16kMono` /
`loadWav16kMonoFile` (PCM16/24/32/f32, stereo downmix, linear resample to
16 kHz via `resampleLinear`), `preemphasis`, `StftParams`, `Spectrogram`,
`DftBasis` (precomputed direct-DFT basis for the `melSpectrogramFast*`
variants), `stftPower`, `MelParams`, `MelSpectrogram` (feat-major
`feats[m * n_frames + t]`), `melSpectrogram`, `melSpectrogramFast`,
`melSpectrogramFastWithBasis`. NeMo-exact: constant-pad STFT, f64
accumulation, per-feature z-score over the valid frames.

**Subsampling** (`subsampling.zig`): `subsample` / `subsampleWithWeights`
(the offline stride-2 conv stack + linear proj), `streamingSubsample` (the
causal variant), `conv2dPublic` (the shared conv2d entry, also exercised by
tests). **Encoder** (`encoder.zig`): `encode` / `encodeWithWeights` (mel
`[n_mels, T]` → `[T/8, d_model]`, caller-owned `Tensor(2)`), built from
`conformerLayer` = `relposAttention` (Transformer-XL relative-position
attention) + `convModule` + `feedForwardT`, with helpers `relPosEncoding`,
`layerNorm`, `layerNormByName`, `layerNormByNameT`, `linearWT`, `f32Data`,
`attnName`, `convName`.

**Decoders** (`decoder.zig`): `ctcDecode` (frame argmax → `ctcCollapse` /
`ctcCollapseWithMeta`); `tdtDecode` / `tdtDecodeWithWeights` (greedy TDT:
LSTM `Predictor` (`init/deinit/step`) + `Joint`
(`init/deinit/encProjAll/step`), duration head skips frames);
`rnntDecodeFrames` + `RnntDecodeState` (`init/deinit/reset`) — the
carried-state per-chunk variant streaming uses. The batch decoders return
caller-freed `[]i32` token ids; `rnntDecodeFrames` returns `!void` and
appends into a caller-provided `*std.ArrayList(i32)`.
`TokenInfo`/`TokenMeta` optionally collect per-token frame
indices and confidences. **Text**: `tokenizer.detokenize` (SentencePiece
piece join, `▁` → space); `transcription.Word`, `groupWords`, `freeWords`,
`toJson` (per-word timestamps from token frames x `frame_sec`).

Offline transcription end-to-end:

```zig
const pk = models.parakeet;
var file = try fucina.gguf.File.loadMmap(alloc, io, "models/parakeet/tdt_ctc-110m-f16.gguf");
defer file.deinit();
const cfg = try pk.loader.Config.fromGguf(&file);
const feat = try pk.loader.loadFeaturizer(&file, cfg); // borrows the mmap

var audio = try pk.frontend.loadWav16kMonoFile(alloc, io, "clip.wav");
defer audio.deinit(alloc);
var mel = try pk.frontend.melSpectrogram(alloc, audio.samples, .{
    .stft = .{ .n_fft = cfg.n_fft, .hop = cfg.hop_length, .win_length = cfg.win_length,
               .mag_power = cfg.mag_power, .preemph = cfg.preemph },
    .n_mels = cfg.n_mels,
    .log_guard = cfg.log_zero_guard,
    .normalize_per_feature = cfg.normalize == .per_feature,
}, feat.fb, feat.window);
defer mel.deinit(alloc);

var w = pk.weights.ParakeetWeights.init(&ctx, &file);
defer w.deinit();
var enc = try pk.encoder.encodeWithWeights(&ctx, &file, cfg, mel.feats, cfg.n_mels, mel.n_frames, &w);
defer enc.deinit();
const ids = try pk.decoder.tdtDecodeWithWeights(&ctx, cfg, &enc, alloc, &w, null);
defer alloc.free(ids);

const pieces = try pk.loader.loadPieces(&file, alloc);
defer alloc.free(pieces);
const text = try pk.tokenizer.detokenize(alloc, pieces, ids);
defer alloc.free(text);
// requires model assets to run
```

**Streaming API** (`streaming.zig`). Two layers:

- `StreamingEncoder` (`init(allocator, cfg, StreamingConfig)` / `deinit` /
  `reset` / `step` / `layerStack`) runs the full cache-aware encoder on one
  mel chunk: causal subsampling → drop `drop_extra_pre_encoded` leading
  frames (steps >= 1) → the layer stack with carried caches → slice to
  `valid_out_len` (all frames on the last chunk). Per-layer state is a
  `ConvCache` (depthwise conv tail, `init/deinit/reset`) and a
  `ChannelCache` (attention K/V left-context window,
  `init/deinit/reset/advance`); the windowed attention itself is
  `streamingAttnMask` + `streamingRelposAttention` +
  `streamingConformerLayer`, with `streamingDepthwiseConv` for the conv
  module and `applyPromptKernel` for the multilingual one-hot projection.
- `StreamingSession` (`init(allocator, file, cfg, sc, weights, lang)` /
  `deinit`) owns the encoder caches, the LSTM predictor + joint, the carried
  `RnntDecodeState` and the accumulated output. Feed granularities:
  `feedMel(ctx, file, w, mel, n_mels, t)` windows a whole clip through the
  chunk schedule; `feedMelChunk` processes one pre-windowed mel chunk;
  `encodeChunkPrompted` returns a chunk's encoder frames (+ prompt kernel);
  `feedEncoderFrames` greedy-decodes frames you encoded yourself.
  Non-special tokens accumulate in `session.tokens` (set
  `collect_meta = true` to align `token_meta` for timestamps); `<EOU>`/
  `<EOB>` events are counted in `eou_events` and reset the decoder state for
  the next utterance (decoder-only, matching the reference). `init` returns
  `error.UnknownLang` if a prompt-conditioned model cannot resolve `lang`.

No training entry, no speculative decoding (not autoregressive text).
Runner: `apps/parakeet/main.zig`. Commands and flags (offline/stream
transcription, `--manifest` batches, `--mic` capture, `--decoder tdt|ctc`,
`--lang`, `--threads`) are in the
[README](../../apps/parakeet/README.md).


## 14.7 Kimi-K3 — KDA/MLA hybrid, architecture parity (`src/models/research/kimi3/model.zig`)

The `kimi3` family (the Kimi-Linear lineage) is a **KDA linear-attention /
Gated-MLA-NoPE hybrid** with a latent sigmoid-routed MoE, cross-layer
attention residuals (layers at `index % attn_res_block_size == 0` snapshot
their entry into a residual bank that later layers depth-mix against), and
the SiTU gated activation ([§4.5](04-tensor-operations.md#45-gated-activations-srcagtensorzig)). It is the one family outside the GGUF
conventions of [§14.1](14-model-families-and-example-applications.md#141-conventions-shared-by-every-family): an architecture-parity model over the tiny f32
reference checkpoint, loaded from a checkpoint directory.
`Config.fromJsonFile(allocator, io, path)` parses the reference
`config.json` (`text_config`, including the 1-based KDA layer list from
`linear_attn_config` — `Config.isKdaLayer(layer_idx)` implements the
schedule, `Config.deinit(allocator)` frees the list) and
`Model.load(allocator, io, ctx, dir)` reads `<dir>/config.json` plus
`<dir>/model.safetensors` (`safetensors.File.loadMmap`).

Heavy lifting goes through the shared exec ops — `matmulTransB`,
`causalDepthwiseConv1d`, `gated(.situ)`, `rmsNormMul` — and the
family-local KDA recurrence (`delta_attention.zig` next door,
[§4.13](04-tensor-operations.md#413-attention-srcagtensorzig)), while the
depth mixture and the small MLA core are model-local routines. Model surface: `load`, `deinit`,
`forward(ctx, tokens)` (full-sequence forward: token ids, `[]const u32`,
to `[seq, vocab]` logits) and `forwardProbed(ctx, tokens, probe)` with
`Probe` (`{ context, callback }`) — a stage observer that receives the
same per-layer intermediates the reference dumps. No KV/state cache, no
runner CLI, no GGUF path; the golden tests pin the forward against
reference activations.

## 14.8 Example applications

Beyond the family runners, standalone applications exercise the library end
to end: the HTTP server (lmserve), the from-scratch GPT pipeline
(nanochat), the CNN/vision ports (facedetect, locate_anything), the audio
ports (omnivoice, nam, and the other voice apps), and the training and
quantization demos (spirals and the finetune/ES/cartridge/engram/PTQTP
family). Each lives in `examples/<name>/` or `apps/<name>/` with its own
README owning its CLI, flags, parity harnesses, and measured numbers;
[RUNNING-MODELS.md](../RUNNING-MODELS.md) is the index that maps every
build step to its README and to verified weight downloads. The library
surfaces they exercise are the [§13](13-the-model-stack-fucina_models.md) machinery
([serving](13-the-model-stack-fucina_models.md#1313-serving-srcmodelstextserving),
[chat](13-the-model-stack-fucina_models.md#138-chat-srcmodelstextchatzig),
[speculative decoding](13-the-model-stack-fucina_models.md#139-speculative-decoding-srcmodelstextspeculative),
[cartridges](13-the-model-stack-fucina_models.md#1310-cartridges-srcmodelstextcartridgezig),
[Engram](13-the-model-stack-fucina_models.md#1311-engram-srcmodelsresearchengramzig)),
the vision ops of [§4](04-tensor-operations.md), the training stack of
[§11](11-training-optimizers-evolution-strategies-lora-and-checkpoints.md),
and the quantization surfaces of [§10](10-quantization.md).

## 14.9 Example → features → run command

The per-target map (what each example and app demonstrates and the
command that runs it) lives in [RUNNING-MODELS.md](../RUNNING-MODELS.md)
(weights are never bundled; its download table lists the source and
license notes for every model row). Each target's README
(`examples/<name>/README.md`, `apps/<name>/README.md`) documents its full
flag set, and `AGENTS.md` carries the one-line command list.
