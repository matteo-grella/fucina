# 13. The model stack (fucina_models)

`fucina_models` is a second Zig module layered on top of the `fucina` facade (its
only module dependency; see [§2](02-toolchain-build-and-project-wiring.md) for build wiring). It contains everything a
transformer inference/fine-tuning runner needs that is not a tensor op:
GGUF-to-weight binding, KV caching, tokenizers, sampling, SFT data plumbing,
multi-turn chat, and lossless draft-free speculative decoding. Import it as:

```zig
const fucina = @import("fucina");
const models = @import("fucina_models");
```

## 13.1 Module layout (`src/models.zig`)

Model families live in subdirectories and are exposed as namespaces; generic,
family-agnostic helpers stay flat:

| Namespace | Contents | Files |
|---|---|---|
| `models.qwen3` | `model`, `train`, `ptqtp`, `shine_serving` — Qwen3 dense + MoE, LoRA fine-tuning, SHINE adapter-fleet serving | `models/qwen3/` |
| `models.qwen35` | `model`, `chat`, `serving` — Qwen3.5 Gated-DeltaNet hybrid | `models/qwen35/` |
| `models.gemma` | `model`, `train`, `moe` | `models/gemma/` |
| `models.diffusion_gemma` | `model` — block text-diffusion on the gemma4 backbone | `models/diffusion_gemma/` |
| `models.parakeet` | `loader`, `frontend`, `subsampling`, `encoder`, `weights`, `decoder`, `tokenizer`, `streaming`, `transcription` — NeMo FastConformer/RNN-T ASR | `models/parakeet/` |
| `models.text.speculative` | `core`, `mtp`, `sam_index`, `recycling`, `cascade`, `constrained` | `models/text/speculative/` |
| `models.deepseek2` | `model` — DeepSeek-V2 MLA + fine-grained MoE with shared experts | `models/deepseek2/` |
| `models.glm4moe` | `model` — GLM-4.5 MoE with native MTP (`nextn`) self-speculation | `models/glm4moe/` |
| `models.deepseek4` | `model`, `serving` — DeepSeek V4 Flash (hyper-connections, compressed-KV MQA, streamed experts, MTP) | `models/deepseek4/` |
| `models.inkling` | `model`, `mmproj`, `chat`, `serving` — Inkling (hybrid SWA/global rel-bias attention, shortconv sites, sink-shared MoE; hMLP vision + dMel audio towers) | `models/inkling/` |
| `models.research` | the research tier under one namespace: `subq` (decode-path attention evaluator; installs through `runner.AttentionOverride`), `engram` (conditional n-gram memory; grafts through the qwen3 trainer's `residual_hook`), `shine`/`shine_train` (context-to-LoRA adapters; served by `models.qwen3.shine_serving`), `kimi3.model` (Kimi-K3: KDA + gated-MLA-NoPE hybrid, latent MoE, attention residuals, SiTU) | `models/research/subq.zig`, `models/research/engram.zig`, `models/qwen3/shine*.zig`, `models/research/kimi3/` |

| Flat helper | Purpose | Section |
|---|---|---|
| `fucina.weights` | GGUF tensor → typed linear weight binding | [§13.2](13-the-model-stack-fucina_models.md#132-weight-loading-srcweightszig) |
| `fucina.ptqtp_gguf` | PTQTP plane persistence — `<name>.ptqtp0/1/2` writer + pair-detecting loader | [§13.2](13-the-model-stack-fucina_models.md#132-weight-loading-srcweightszig) |
| `fucina.gguf_meta` | metadata readers + parallel layer loader | [§13.3](13-the-model-stack-fucina_models.md#133-gguf-metadata-glue-srcgguf_metazig) |
| `models.decoder` | the autoregressive decoder contract: `Caps` + comptime `assertDecoder(Model)`; the generic layers are written against it | [§13.8](13-the-model-stack-fucina_models.md#138-chat-srcmodelstextchatzig), [§14.1](14-model-families-and-example-applications.md#141-conventions-shared-by-every-family) |
| `models.registry` | the architecture registry: GGUF `general.architecture` to family module; `serving.open` dispatches over it, `familyFor` is the comptime lookup | [§13.13](13-the-model-stack-fucina_models.md#1313-serving-srcmodelstextserving) |
| `models.text.generate` | the reference generation loop over the decoder contract (`generate`/`generateOutcome`, `TokenSink`, `greedy`) | [§14.1](14-model-families-and-example-applications.md#141-conventions-shared-by-every-family) |
| `models.text.kv_cache` | per-layer K/V store for autoregressive decode | [§13.4](13-the-model-stack-fucina_models.md#134-kv-cache-srcmodelstextkv_cachezig) |
| `models.text.kv_persist` | crash-safe append-only KV-cache sidecar: conversations reopen warm | [§13.4](13-the-model-stack-fucina_models.md#134-kv-cache-srcmodelstextkv_cachezig) |
| `models.text.tokenizer` | byte-level BPE (GPT-2/Qwen) | [§13.5](13-the-model-stack-fucina_models.md#135-tokenizers) |
| `models.text.spm_tokenizer` | SentencePiece Unigram (Gemma/llama-vocab) | [§13.5](13-the-model-stack-fucina_models.md#135-tokenizers) |
| `models.text.unicode_categories` | generated `\p{L}`/`\p{N}`/`\p{M}`/`\s` tables (byte-BPE pretokenizer; shared with out-of-module tokenizers) | [§13.5](13-the-model-stack-fucina_models.md#135-tokenizers) |
| `models.text.sampler` | greedy/temperature/top-k/top-p/min-p/penalties + logit-processor seam | [§13.6](13-the-model-stack-fucina_models.md#136-sampling-srcmodelstextsamplerzig) |
| `models.text.logit_processor` | pluggable logit-transform interface (grammar masks, bias lists) | [§13.6](13-the-model-stack-fucina_models.md#136-sampling-srcmodelstextsamplerzig) |
| `models.text.llguidance` | grammar/JSON-schema constrained decoding (vendored engine, `-Dllguidance`) | [§13.6](13-the-model-stack-fucina_models.md#136-sampling-srcmodelstextsamplerzig) |
| `models.text.data` | SFT pairs, encodePair, deterministic Loader | [§13.7](13-the-model-stack-fucina_models.md#137-sft-data-srcmodelstextdatazig) |
| `models.text.chat` | templates + generic `Conversation(Model, Tok)` | [§13.8](13-the-model-stack-fucina_models.md#138-chat-srcmodelstextchatzig) |
| `models.text.cartridge` | trained KV-prefix corpus compression (Cartridges, arXiv 2506.06266) | [§13.10](13-the-model-stack-fucina_models.md#1310-cartridges-srcmodelstextcartridgezig) |
| `models.text.cartridge_fleet` | per-document cartridge fleets: manifest, RAM/disk budget manager, cosine chunk index (Cartridges at Scale, arXiv 2606.04557) | [§13.10](13-the-model-stack-fucina_models.md#1310-cartridges-srcmodelstextcartridgezig) |
| `models.research.engram` | conditional n-gram memory: hashed-lookup embedding tables grafted onto a frozen model (Engram, arXiv 2601.07372) | [§13.11](13-the-model-stack-fucina_models.md#1311-engram-srcmodelsresearchengramzig) |
| `models.text.serving` | the serving band: the contract (`GenerateRequest`/`GenerateResult`, `Caps`, the per-family `Backend` vtable), the HTTP transport (`serving.http`/`scheduler`/`emitter` + the OpenAI/Anthropic dialects and hermes tool calling), the generic `GgufChatBackend` engine, and the `serving.open` load-and-serve entry (`examples/lmserve` is the CLI front end) | [§13.13](13-the-model-stack-fucina_models.md#1313-serving-srcmodelstextserving) |
| `models.qwen3.runner` | the descriptor runner (experimental tier): one family-independent decoder driven by a runtime `Descriptor` with two block styles (fused qwen3-shape, host_reference GLM/DeepSeek-MoE shape); the qwen3 family and the glm4moe trunk run on it; recorded-golden gates in `runner_tests.zig` (real 0.6B chains + logit fingerprints, synthetic MoE and glm fixtures) | `docs/RUNNER.md` |

The family namespaces are covered in [§14](14-model-families-and-example-applications.md) (kimi3 in [§14.7](14-model-families-and-example-applications.md#147-kimi-k3--kdamla-hybrid-architecture-parity-srcmodelsresearchkimi3modelzig),
deepseek2/glm4moe/deepseek4/inkling by their module doc comments); this
section documents the shared stack they are built from.

## 13.2 Weight loading (`src/weights.zig`)

`weights.zig` turns raw GGUF tensor payloads ([§12](12-model-io-gguf-and-safetensors.md)) into typed, immediately
usable linear weights. It lives in the CORE module (`fucina.weights`, with
`fucina.ptqtp_gguf` beside it): nothing in it is language-model-specific —
vision encoders and audio models load through the same band (the
locate-anything ViT does). Its error set is
`Error = error{ InvalidWeightShape, UnsupportedWeightType, GradUnsupported }`.

### 13.2.1 `LinearWeight`

```zig
pub const LinearWeight = union(enum) {
    f32: WeightF32,     // fucina.Tensor(.{ .out, .in })
    f16: WeightF16,     // fucina.Tensor(.{ .dtype = .f16, .tags = .{ .out, .in } })
    bf16: WeightBf16,   // fucina.Tensor(.{ .dtype = .bf16, .tags = .{ .out, .in } })
    q8_0: WeightQ8_0, q4_k: WeightQ4_K, q5_k: WeightQ5_K, q6_k: WeightQ6_K,
    // plus one QuantWeight(dtype) arm per remaining GGUF block format:
    // q1_0, q2_0, q4_0, q4_1, q5_0, q5_1, q2_k, q3_k, iq1_s, iq1_m, iq2_xxs, iq2_xs,
    // iq2_s, iq3_xxs, iq3_s, iq4_nl, iq4_xs, tq1_0, tq2_0, mxfp4, nvfp4
    ptqtp: WeightPtqtp, // 1-3 packed TQ2_0 trit-planes (PTQTP, section 10.9)
};
```

Every arm is a `[.out, .in]`-tagged tensor kept **resident in its source
precision** — nothing is widened to f32 at load time:

| Arm | Resident form | Forward path |
|---|---|---|
| `f32` | f32 tensor (f64 sources are narrowed) | plain `dot` |
| `f16` | f16 tensor, 2 B/weight | f16-operands GEMM ([§9](09-backends-cpu-simd-blas-threading-and-gpu-offload.md)); GPU-resident on `-Dgpu=metal` |
| `bf16` | raw u16 bit patterns, 2 B/weight | mixed f32×bf16 TransB kernel, exact in-register widening |
| `q8_0`, `q4_k`, `q5_k`, `q6_k` | raw GGUF blocks **plus** a pre-packed matmul RHS | `dotPacked` on the CPU quantized hot path ([§10](10-quantization.md)); q4_k/q6_k/q8_0 also try Metal, and CUDA additionally supports q5_k |
| all other quant arms | raw GGUF blocks (`QuantWeight(dtype)`) | tagged `dot` through the generic quantized matmul |
| `ptqtp` | up to three `.tq2_0` plane tensors ([§10.9](10-quantization.md#109-ptqtp-multi-plane-ternary-decomposition-srcptqtpzig-ptqtpmd)) — built in place by `toPtqtp`, or rebuilt bitwise from persisted `<name>.ptqtp0/1/2` plane tensors (`fucina.ptqtp_gguf` pair-detection; [PTQTP.md](../PTQTP.md)) | fused multi-plane entry: ONE Q8_K activation quantization + ONE worker-team dispatch running every plane on the x4 column-interleaved packs and summing in fixed plane order (bitwise equal to the per-plane facade dots, which remain the gradient-path fallback); scale-tied K=2 planes additionally fold into one 4-bit pack served by a single dot pass, and on Metal builds prefill-sized inputs dispatch against resident plane copies — one ternary dequant-in-kernel dispatch per plane, ONE folded dispatch when tied (the dense quant offload's accepted-numerics stance, not bitwise) |

`pub fn QuantWeight(comptime dtype: DType) type` returns
`fucina.Tensor(.{ .dtype = dtype, .tags = .{ .out, .in } })`. The four hot
K-quant/Q8 formats get dedicated wrapper structs — `WeightQ4_K`, `WeightQ5_K`,
`WeightQ6_K`, `WeightQ8_0` — each holding `value` (the raw block tensor) and
`packed_rhs: fucina.PackedRhs(dtype)` built once at init, with
`init`/`deinit`/`cloneView`/`concat`, plus `initWithRhsLifetime` and a
`rhs_lifetime: fucina.RhsLifetime` field that tells GPU dispatch whether the
block bytes are process-stable.

Binding a GGUF tensor:

```zig
pub fn load(ctx: *ExecContext, info: *const gguf.TensorInfo,
            expected_rows: usize, expected_cols: usize) !LinearWeight
pub fn loadForFusion(...same args...) !LinearWeight
pub fn loadWithOptions(...same args..., options: LoadOptions) !LinearWeight

pub const LoadOptions = struct { gpu_resident: bool = true };
```

- The tensor's `logicalMatrixShape()` must equal
  `(expected_rows, expected_cols)` = `(out, in)`, else
  `Error.InvalidWeightShape`; a GGML type without an arm is
  `Error.UnsupportedWeightType`.
- `load` calls `gguf.prefetch` on the payload first (readahead for cold-mmapped
  bytes) and copies/repacks it, so the result **does not borrow** the
  `gguf.File` — the file may be freed after loading (MoE borrow mode below is
  the exception).
- `LoadOptions.gpu_resident` (default `true`): on GPU builds, provider-supported
  payloads are copied into device-owned storage (f16/q4_k/q6_k/q8_0 on Metal;
  q4_k/q5_k/q6_k/q8_0 on CUDA) through `internal.gpu.allocResidentBytes`, so
  GPU matmuls read them with zero
  per-call transfer. The storage buffer OWNS the device bytes through a
  release hook: when the last tensor reference (including `cloneView`s sharing
  the buffer) drops, the hook frees the device allocation and evicts the GPU
  shim's cached wrap. The bytes stay CPU-readable (and, for dense f16/f32,
  CPU-writable in place — in-place trainers mutate resident weights and GPU
  dispatch reads the live values). If the device budget is exhausted the load
  silently falls back to heap storage with `.transient` RHS lifetime.
- `loadForFusion` is `loadWithOptions(..., .{ .gpu_resident = false })`: a
  weight loaded only to be consumed by `fuseLinear` skips the per-part device
  copy, because the fused result re-acquires residency itself — per-part
  copies would be alloc+memcpy+free waste. If fusion later declines,
  `fuseLinear` restores ordinary per-part residency for provider-supported
  formats before returning them as independent linears.

Pre-fusion:

```zig
pub fn fuseLinear(ctx: *ExecContext, parts: []const *LinearWeight) !?LinearWeight
```

Concatenates 2–4 same-format weights along `.out` into one stacked matrix
(one GEMM instead of N on the forward path). Supported formats:
f32/f16/bf16/q4_k/q5_k/q6_k/q8_0, plus `ptqtp` parts with a uniform plane
count (planes concatenate plane-wise — byte-identical to decorating the
fused matrix; mixed plane counts return `null` like mixed formats). On
success the parts are **consumed**
(deinitialized) and the fused weight is returned; when the parts' formats
differ, or the format has no fused fast path, it returns `null` with all
parts still valid. On capable GPU builds their values may move to resident
storage (the semantic tensor/tag/packed-RHS values are unchanged); fewer than
2 or more than 4 parts is
`Error.InvalidWeightShape`. Fused dense f32/f16 and quant
q4_k/q6_k/q8_0 results re-acquire GPU residency on Metal builds; CUDA also
re-acquires it for q5_k.

Forward/apply entry points on `LinearWeight`:

```zig
pub fn linearSeq(self, ctx, input: anytype, comptime in_tag: Tag, comptime out_tag: Tag)
    !fucina.Tensor(.{ .seq, out_tag })
pub fn linearSeqNormed(self, ctx, x: anytype, norm_weight: anytype, eps: f32,
    comptime in_tag: Tag, comptime out_tag: Tag) !fucina.Tensor(.{ .seq, out_tag })
pub fn supportsNormedFusion(self, m: usize) bool
pub fn getRowsAs(self, ctx, token_ids: []const usize, comptime out_tag: Tag)
    !fucina.Tensor(.{ .seq, out_tag })
pub fn toResidentF16(self: *LinearWeight, ctx: *ExecContext) !void
pub fn toPtqtp(self: *LinearWeight, ctx: *ExecContext, options: fucina.ptqtp.Options)
    !fucina.ptqtp.MatrixStats     // requires ptqtpEligible; drops the source storage
pub fn ptqtpEligible(self: *const LinearWeight) bool  // non-ptqtp arm, inDim % 256 == 0
pub fn outDim(self) usize / pub fn inDim(self) usize
pub fn cloneView(self, ctx) !LinearWeight   // shares storage, fresh tags/packed RHS
pub fn deinit(self: *LinearWeight) void
```

- `linearSeq` computes `input · Wᵀ` with the format's fastest route: packed
  quantized kernels for q4_k/q5_k/q6_k/q8_0 (with a GPU attempt first for
  q4_k/q6_k/q8_0 on Metal and those plus q5_k on CUDA — declined when
  the input requires gradients or the exec gate says the shape is too
  small, falling back to the CPU packed path; at
  decode shapes (`seq < 4`, no gradients) q4_k, q5_k, and q6_k instead
  contract against the resident GGUF-native compact blocks —
  bitwise-equal outputs, ~1.92x/1.57x/1.30x fewer weight bytes streamed
  than the byte-expanded packed layout; default on, with
  `FUCINA_DECODE_COMPACT=1/0` forcing the route
  on/off and `setDecodeCompact` as the programmatic override), and a
  tagged `dot` for everything else. The format-generic helper
  `weights.linearSeq(&w, ctx, input, in_tag, out_tag)` is also `pub` for
  callers that hold a packed wrapper struct directly (the format is
  inferred from the container).
- `linearSeqNormed` is `linearSeq` over `rmsNormMul(x, norm_weight, eps)`:
  on the packed CPU q4_k/q5_k/q6_k/q8_0 routes at prefill shapes
  (`seq >= 4`, no gradients; q4_k only on non-MMLA targets) the normalized
  tensor is never materialized — the fused kernel normalizes into
  task-private scratch and quantizes in place, matching the unfused pair to
  f32 roundoff. Every other arm — GPU builds, decode shapes, and
  `FUCINA_NORM_QUANT_FUSED=0` (`=1` forces the
  fused route; `setNormQuantFused` is the programmatic override) —
  normalizes and delegates. `supportsNormedFusion(m)` reports whether the
  fused route applies for an m-row input; callers fanning one normalized
  input into several projections should require it for every projection —
  the fallback re-normalizes per call.
- `getRowsAs` gathers rows by index (the embedding-lookup shape) and returns
  f32; f16/bf16 rows are widened, quantized rows dequantized. Dedicated arms
  exist for f32/f16/bf16/q4_k/q5_k/q6_k/q8_0, and an `inline else` arm
  routes every remaining block-quantized format through the generic
  quantized row gather — all `LinearWeight` forms work.
- `toPtqtp` dequantizes the weight row-chunk-wise through `getRowsAs` — so
  every loadable source dtype quantizes through one code path — solves the
  trit-planes ([§10.9](10-quantization.md#109-ptqtp-multi-plane-ternary-decomposition-srcptqtpzig-ptqtpmd)), and replaces the arm in place, dropping the source
  storage. On the `ptqtp` arm, `getRowsAs` returns the dequantized plane
  sum (so `toResidentF16` doubles as un-decorate). `decoratePtqtpInto` +
  `PtqtpReport` aggregate per-tensor solver stats over model walks;
  `models.qwen3.ptqtp.decorate(model, ctx, options)` walks attention
  q/k/v (split or fused), o_proj, and dense FFN projections, with
  `DecorateOptions` covering per-projection plane overrides
  (`down_planes`/`o_planes`) and data-free edge-layer skip
  (`skip_first_layers`/`skip_last_layers`); embeddings, lm_head, and norms
  are not walked (decorate `model.output` directly for a ternary head).
  `models.qwen3.ptqtp.save(model, ctx, io, src_file, out_path)` persists the
  decorated model: `fucina.ptqtp_gguf` writes one standalone TQ2_0 tensor per plane —
  `<name>.ptqtp0/1/2` replaces `<name>`, fused weights row-slicing back to
  their source tensor names — plus a `fucina.ptqtp.version` metadata key
  and, when every decorated entry was tie-fitted
  (`ptqtp.Options.tie_scales`), a `fucina.ptqtp.tie_scales` key the
  loaders read to rebuild the tied, fold-capable serving form ([§10.9](10-quantization.md#109-ptqtp-multi-plane-ternary-decomposition-srcptqtpzig-ptqtpmd)),
  everything else byte-verbatim; the qwen3 loaders pair-detect planes and
  rebuild the arm bitwise (re-fusing via `fuseLinear`'s ptqtp arm — other
  families do not read decorated files yet), so decoration runs once and
  the saved file serves through the ordinary qwen3 runners
  ([PTQTP.md](../PTQTP.md)). MoE expert stacks follow the same convention —
  `<name>.ptqtpK` siblings with the base stack's 3D shape, plane-major on
  disk: `ptqtp_gguf.maybeLoadMoeRhs` / `maybeStreamedMoeProjSpec`
  pair-detect them into the resident `MoeRhs.ptqtp` arm
  (`weights.loadMoeRhsPtqtp`) or a multi-plane expert-store `ProjSpec`
  (`weights.streamedProjSpecPtqtp` + `registerStreamedMoeLayer`), both
  wired into the qwen3 MoE loaders; `ptqtp_gguf.quantizeMoeStack` is the
  producer (per-expert-slice solve into plane-major stacks — the
  export-gguf `--ptqtp` MoE path).
- `toResidentF16` replaces the weight in place with a dequantized resident-f16
  copy (2 B/weight — the f16 GEMM/GPU-offload operand format), dequantizing in
  4096-row chunks through the same row gather so the transient peak stays a
  few MB. No-op when already f16.

```zig
fn snippetLinearWeight(ctx: *fucina.ExecContext, file: *const fucina.gguf.File, row: []const f32) !void {
    const info = try file.get("blk.0.attn_q.weight");
    var w = try fucina.weights.LinearWeight.load(ctx, info, 1024, 1024); // expected [out, in]
    defer w.deinit();

    var x = try fucina.Tensor(.{ .seq, .embed }).fromSlice(ctx, .{ 1, w.inDim() }, row);
    defer x.deinit();
    var y = try w.linearSeq(ctx, &x, .embed, .attn_q); // format-dispatched matmul
    defer y.deinit();

    var rows = try w.getRowsAs(ctx, &.{ 0, 2 }, .embed); // dequantized f32 row gather
    defer rows.deinit();
} // requires model assets to run
```

```zig
fn snippetFuseLinear(ctx: *fucina.ExecContext, file: *const fucina.gguf.File) !void {
    var gate = try fucina.weights.LinearWeight.loadForFusion(ctx, try file.get("blk.0.ffn_gate.weight"), 3072, 1024);
    errdefer gate.deinit();
    var up = try fucina.weights.LinearWeight.loadForFusion(ctx, try file.get("blk.0.ffn_up.weight"), 3072, 1024);
    errdefer up.deinit();
    if (try fucina.weights.fuseLinear(ctx, &.{ &gate, &up })) |fused| {
        var owned = fused; // one [6144, 1024] weight; gate/up were consumed
        defer owned.deinit();
    } else {
        gate.deinit(); // mixed formats: parts untouched, use them individually
        up.deinit();
    }
} // requires model assets to run
```

### 13.2.2 Vectors, MoE, and borrowed linears

```zig
pub fn loadVector(ctx: *ExecContext, info: *const gguf.TensorInfo,
                  expected_len: usize, comptime tag: Tag) !fucina.Tensor(.{tag})
pub fn layerName(buf: []u8, layer_i: usize, suffix: []const u8) ![]const u8
```

`loadVector` reads a 1-D tensor (f32/f16/bf16/f64 sources) into an f32 vector;
wrong rank/length is `Error.InvalidWeightShape`. `layerName` formats
`"blk.{d}.{s}"` into a caller buffer — the GGUF per-layer naming convention.

```zig
pub const LookupWeight = union(enum) { resident: LinearWeight, mapped: MappedTable };
pub fn load(ctx, file: *const gguf.File, info: *const gguf.TensorInfo,
    expected_rows: usize, expected_cols: usize) !LookupWeight
pub fn getRowsAs(self, ctx, token_ids: []const usize, comptime out_tag: Tag)
    !fucina.Tensor(.{ .seq, out_tag })
pub fn borrowsMapping(self) bool
```

`LookupWeight` is for tables consumed exclusively through `getRowsAs` — never
a matmul operand (gemma4's per-layer-embedding table). On CPU builds, when the
file is a single-file mmap and the dtype has a `gguf.RowTable` row decoder,
`load` returns the `mapped` arm: rows decode on demand straight out of the
mapping — no resident copy of the table, no matmul-RHS packing — bitwise-equal
to the resident gather. The caller that gets `borrowsMapping() == true` must
keep the mapping alive for the weight's lifetime via `gguf.File.takeMapping`
(gemma4's `Model.weight_mapping` does). Heap-read files, split GGUFs, GPU
builds, and undecodable dtypes fall back to the copying `resident` arm.

```zig
pub fn loadMoeRhs(ctx: *ExecContext, info: *const gguf.TensorInfo,
    expected_in_dim: usize, expected_out_dim: usize, expected_n_expert: usize,
    borrow: bool) !fucina.MoeRhs
pub fn moeSwiGluFfnSeq(ctx, input: *const Tensor(.{ .seq, .embed }),
    gate: *const fucina.MoeRhs, up: ..., down: ...,
    selected: []const usize, routing_weights: []const f32, top_k: usize,
    out_pe: usize, io: ?std.Io, profile: ?*fucina.MoeBatchProfile)
    !fucina.Tensor(.{ .seq, .embed })
```

`loadMoeRhs` binds one stacked-expert 3-D tensor
(`blk.N.ffn_{gate,up,down}_exps.weight`, GGUF shape `[in, out, n_expert]`) as
a single packed matmul RHS; the fused MoE kernel slices each expert as a
zero-copy row block. Supported expert formats: the K-quants
(q2_k/q3_k/q4_k/q5_k/q6_k) plus q8_0 (llama.cpp's fallback when an expert dim is
not a 256 multiple), iq2_xxs, iq2_s, iq3_xxs, iq4_xs, and tq2_0 —
other formats are `Error.UnsupportedWeightType`. With `borrow = true` the
blocks are borrowed straight from the (mmapped) GGUF, skipping the multi-GB
copy; the caller must then keep the mapping alive for the model's lifetime
(`gguf.File.takeMapping`, [§12](12-model-io-gguf-and-safetensors.md)). `moeSwiGluFfnSeq` is the tensor-valued
Qwen-style SwiGLU MoE FFN over those RHS values; it refuses gradient-tracked
inputs (`Error.GradUnsupported`) and internally splits decode (`seq == 1`)
from batched prefill. `moeGatedFfnSeq` is the same entry with the gated
activation chosen by the caller (`act: fucina.GatedOp`; deepseek4 routes
through the clamped SwiGLU). `loadMoeRhsStreamed(store, file, layer_i,
gate_info, up_info, down_info, expected_in_dim, expected_out_dim,
expected_n_expert)` is the streamed counterpart of three `loadMoeRhs` calls:
it registers one layer's gate/up/down stacked expert tensors with the
`fucina.ExpertStore` (which `pread`s individual experts on demand) and
returns a `StreamedMoeFfnRhs{ gate, up, down }` of `.streamed` RHS values —
only the geometry is validated, nothing of the expert stacks is read.

The store itself comes from `createExpertStore(allocator, options:
MoeStreamOptions, n_layers)`, which expands split-GGUF part paths, opens
the `fucina.ExpertStore` with the stream policy —
`cache_bytes`/`cache_slots_per_layer`, the pinned learning tier
(`auto_pin`/`pin_bytes`), `readahead`, `io_workers` parallel demand-miss
reads, `uncached` streamed reads (macOS `F_NOCACHE`; keeps expert
streaming from churning the page cache backing the mmapped dense
weights), and `pilot` router-lookahead prefetch — and adds each
`mirror_paths` entry as a weighted read mirror (`mirror_weights`).

The routing policy lives in `models.moe_router`:
`cacheRouteSel(gate, choice, sel)` applies the store's
resident-preferring top-k selection when the layer streams from a store
opened with `cache_route` (quality-affecting, opt-in: `route_sacred`
true top ranks are always taken, `route_window` bounds the
resident-preferring fill) and returns false when the caller keeps its
plain top-k; `pilotHintTopK(ctx, nrm, router, top_k, store, layer_i)` is
the router-lookahead tail the streamed decoders share.

The runner CLI seam lives in `models.moe_stream_cli`:
`reportAndSaveMoeStream(store, learn, writer)` is the runners' exit-time
report — stream, pilot, prefetch, cache-route, and mirror stats — and
persists the usage histogram that seeds the next load's pinned tier;
`parseMirrorWeights` parses the runners' shared `--moe-mirror-weights=`
comma list against the `--moe-mirror` count.
`MoeStreamCli` is the runners' shared argv seam for the seven common
`--moe-*` flags (`--moe-stream`, `--moe-cache-mb=`, `--moe-mirror=`,
`--moe-mirror-weights=`, `--moe-uncached`, `--moe-io-threads=`,
`--moe-trace=PATH`):
`tryParse(arg)` consumes exactly those (false = not a shared flag, the
caller keeps its family-specific flags and unknown-flag error) and
`options(gguf_path)` assembles the `fucina.weights.MoeStreamOptions`
(null when nothing armed streaming; the result borrows the CLI struct's
mirror buffers).
Family-specific levers — `--moe-pilot`, the cache-route trio, the
pinned-tier knobs — stay in the runners, which arm streaming via `armed`
and set their fields on the returned options.

`--moe-trace=PATH` records the routed (layer, expert) sequence in request
order and writes it at store teardown (`ExpertStore.saveTrace`); `zig
build replay-experts -- PATH [slots-per-layer...] [--pins-from=SIDECAR]
[--heat-decay=N]` replays it through LRU, heat (decayed-LFU), segmented-
LRU, Belady-optimal, and pinned+LRU policies across a capacity sweep —
answering offline whether more cache would help and whether the policy
or the capacity is the bottleneck (LRU flat where Belady climbs =
policy). `--pins-from` draws the pinned set from a persisted
`<gguf>.experts` histogram instead of the whole-trace oracle, measuring
how pins learned in previous sessions generalize to a new prompt. The
persisted usage histogram alone cannot answer any of this: every policy
needs temporal order. The cache tier itself evicts by heat by default
(`Options.heat_eviction`; `FUCINA_MOE_LRU=1` reverts to pure LRU for
A/B): victim = lowest decayed routed-pair count among slots not touched
by the current acquire, recency breaking ties — measured at +3 hit-rate
points and −4% streamed bytes over LRU on real DeepSeek-V2-Lite traces,
replay and live agreeing. Relatedly, auto-pin declines
(`pins_declined_flat`, threshold `Options.auto_pin_min_advantage`) when
the histogram is ~flat — a quantile-balanced router's pinned tier
retains no more traffic than random slots — handing the whole budget to
the LRU instead; the guard is bypassed whenever the budget holds every
used expert.

Zero-copy linears over caller-owned immutable bytes (used by runners that keep
weights mmapped):

```zig
pub fn linearSeqBorrowedF16(ctx, input: anytype, bytes: []const u8, shape: [2]usize,
    comptime in_tag: Tag, comptime out_tag: Tag) !fucina.Tensor(.{ .seq, out_tag })
pub fn linearSeqBorrowedQuantized(comptime dtype: DType, ctx, input: anytype,
    bytes: []const u8, shape: [2]usize, options: BorrowedQuantLinearOptions,
    comptime in_tag: Tag, comptime out_tag: Tag) !fucina.Tensor(.{ .seq, out_tag })

pub const BorrowedQuantLinearOptions = struct {
    allow_gpu: bool = true,
    rhs_lifetime: RhsLifetime = .transient,
};
```

The quantized variant is comptime-restricted to q8_0/q4_k/q5_k/q6_k, rejects
gradient-tracked inputs (`Error.GradUnsupported`), and validates
`input.dim(in_tag) == shape[1]` (`Error.InvalidWeightShape`). Neither takes
ownership of `bytes`.

Metal-residency utilities shared by loaders and eager dispatch batching:

- `ResidentByteRegistry` (`init`/`deinit`/`bytes`): a session/model-owned map
  from host byte pointers to one-time device copies. `bytes(src)` returns the
  device-resident alias (still CPU-readable) on GPU builds, or `src` verbatim
  on non-GPU builds and on any allocation failure; `deinit` frees all device
  copies. Not thread-safe.
- `QuantByteStackPart`, `QuantByteStackOptions{ prefer_device = true,
  require_device = false }`, `QuantByteStack`
  (`deinit(allocator)`/`bytesPerRow`/`totalOutRows`) and
  `makeQuantByteStack(comptime dtype, allocator, parts, options) !?QuantByteStack`:
  copy same-shaped quantized weights into one contiguous stack with the same
  residency policy as the loaders. Returns `null` for empty `parts` or when
  `require_device` is set and no device storage is available;
  mismatched part shapes are `Error.InvalidWeightShape`. Device-capable
  dtypes: q4_k/q6_k/q8_0 on both providers, plus q5_k on CUDA.

## 13.3 GGUF metadata glue (`src/gguf_meta.zig`)

Flat helpers shared by every model family's loader. Error set:
`Error = error{ InvalidConfig, MissingMetadata }`.

```zig
pub const ZeroPolicy = enum { reject_zero, accept_zero };
pub fn metaInt(file: *const gguf.File, arch: []const u8, suffix: []const u8, zero: ZeroPolicy) Error!usize
pub fn metaIntOpt(file, arch, suffix, zero: ZeroPolicy) ?usize
pub fn metaFloat(file, arch, suffix) Error!f32
pub fn metaFloatOpt(file, arch, suffix) ?f32
pub fn readU32OrBoolArray(allocator, file, key: []const u8, n_layer: usize, comptime T: type) ![]T
```

The `meta*` quartet reads the key `"<arch>.<suffix>"`. Missing keys,
negative integers, and — under `.reject_zero` — present-but-zero integers
are invalid: the `Opt` variants read them as `null`, the strict variants
return `Error.InvalidConfig`. `ZeroPolicy` exists because families disagree
about zero on purpose: qwen3 treats a zero-valued key like a missing key
everywhere, while gemma reads legitimately-zero keys such as
`attention.shared_kv_layers`. A key that overflows the internal 128-byte
format buffer reads as absent. `readU32OrBoolArray` reads a per-layer
metadata array (bool/int item types), broadcasting a scalar value across
all layers like llama.cpp's `get_key_or_arr` — the convention for
per-layer keys such as `<arch>.attention.head_count_kv` (gemma4, inkling,
diffusion-gemma).

```zig
test "gguf_meta: zero-valued keys split by policy" {
    const alloc = std.testing.allocator;
    var w = fucina.gguf.Writer.init(alloc);
    defer w.deinit();
    try w.addMetaInt("arch.block_count", u32, 24);
    try w.addMetaInt("arch.shared_kv_layers", u32, 0);
    var buf: [4096]u8 = undefined;
    var sink = std.Io.Writer.fixed(&buf);
    try w.finish(&sink);
    var file = try fucina.gguf.File.parseOwned(alloc, try alloc.dupe(u8, sink.buffered()));
    defer file.deinit();

    const meta = fucina.gguf_meta;
    try std.testing.expectEqual(@as(usize, 24), try meta.metaInt(&file, "arch", "block_count", .reject_zero));
    try std.testing.expectError(error.InvalidConfig, meta.metaInt(&file, "arch", "shared_kv_layers", .reject_zero));
    try std.testing.expectEqual(@as(usize, 0), try meta.metaInt(&file, "arch", "shared_kv_layers", .accept_zero));
    try std.testing.expectEqual(@as(?usize, null), meta.metaIntOpt(&file, "arch", "missing", .accept_zero));
}
```

```zig
pub fn parallelLoadLayers(comptime Layer: type, comptime Loader: type,
    ctx: *ExecContext, loader: Loader, layers: []Layer) !void
```

Loads all model layers in parallel across the exec work pool ([§9](09-backends-cpu-simd-blas-threading-and-gpu-offload.md)) when one is
available, serially otherwise — layer loads are independent and the
`ExecContext` allocator and buffer pool are thread-safe, so the multi-GB
copy+pack becomes an N-core job (the dominant chunk of model load time).
`Loader` is a small per-family adapter value providing
`fn load(self, layer_i: usize) !Layer` and
`fn deinitLayer(self, layer: *Layer) void`. On failure, only the layers that
DID load are deinitialized, and the first error **in layer order** is
returned (deterministic even under parallel execution).

## 13.4 KV cache (`src/models/text/kv_cache.zig`)

```zig
pub const KvTensor = fucina.Tensor(.{ .dtype = .f16, .tags = .{ .seq, .kv_head, .d } });
pub const KvInput  = fucina.Tensor(.{ .seq, .kv_head, .d });   // f32 rows handed to append
pub const KvDtype  = enum { f16, q8_0 };
pub const Error = error{ KvCacheOverflow, KvCacheShapeMismatch, KvCacheHeadDimNotBlockAligned };
```

`KvCache` is the per-layer post-RoPE key/value store for autoregressive
decode, shared by every family and by the speculative decoder. Layout: one
contiguous `[capacity, kv_heads, head_dim]` tensor per layer for K and one for
V — exactly the `[.seq, .kv_head, .d]` layout the attention kernels consume,
so the active prefix `[0..len]` is a zero-copy narrow. K is stored **after**
RoPE (V has no RoPE), so past positions are never re-rotated.

- **f16 default**: 2 B/element — half the f32 footprint and per-step
  bandwidth; the attention kernel widens to f32 in-register. Matches
  llama.cpp's default cache type.
- **Opt-in q8_0** (`initWithDtype`/`initPerLayerWithDtype` with `.q8_0`;
  llama.cpp's `--cache-type-k/v q8_0`): each (position, kv_head) row is stored
  as `head_dim/32` `BlockQ8_0` — 34 bytes per 32 elements, roughly halving f16
  again at a small quantization loss. Requires `head_dim % 32 == 0` (checked
  at init: `Error.KvCacheHeadDimNotBlockAligned`); q8_0 layers are raw block
  slices, consumed via `kBlocks`/`vBlocks` and the attention kernels'
  q8_0-block KV arm. Decode serves the blocks through the INTEGER score
  path: the query row is quantized once per head to q8_0 and scores are
  q8xq8 `sdot` dots straight on the cached K blocks, with the V dequant
  fused into the weighted accumulate — the sweep reads only quantized
  bytes, no f32 scratch row. Measured (Qwen3-4B, M1 Max, 64-token decode):
  +22%/+50%/+54% over the previous dequant-scratch path at 2k/8k/16k
  context, and at or above the f16 cache from ~8k up (15.5 vs 14.3 tok/s
  at 8k). The query-tiled prefill kernel keeps the dequant path (per-tile
  row reuse amortizes it) and stays bit-exact vs the f32 kernel on a
  dequantized cache.

```zig
pub fn init(ctx: *ExecContext, num_layers: usize, kv_heads: usize, head_dim: usize, capacity: usize) !KvCache
pub fn initWithDtype(...same..., dtype: KvDtype) !KvCache
pub fn initPerLayer(ctx, kv_heads_per_layer: []const usize, head_dims: []const usize, capacity: usize) !KvCache
pub fn initPerLayerWithDtype(...same..., dtype: KvDtype) !KvCache
pub fn deinit(self: *KvCache) void
```

The per-layer variants size each layer's slot independently — Gemma 4
interleaves local sliding-window layers (kv_heads 8, head_dim 256) with global
layers (kv_heads 2, head_dim 512). The cache itself has no window logic: every
position is appended and retained, and windowed models apply their sliding
window at read time through the windowed attention kernels (which also keeps
`truncate` rewind trivially correct). Allocations use `ctx.allocator`; the
caller owns the cache and must `deinit` it.

Decode-loop API:

```zig
pub fn appendLayer(self: *KvCache, ctx: *ExecContext, layer_i: usize,
                   k_rows: *const KvInput, v_rows: *const KvInput) !void
pub fn advance(self: *KvCache, m: usize) void
pub fn len(self: *const KvCache) usize           // cached positions (the `count` field)
pub fn reset(self: *KvCache) void                // count = 0, buffers retained
pub fn truncate(self: *KvCache, keep_len: usize) void
pub fn copyRows(self: *KvCache, src: *const KvCache, start: usize, end: usize) !void
pub fn kSlice(self, layer_i: usize, n: usize) ![]const f16   // f16 mode
pub fn vSlice(self, layer_i: usize, n: usize) ![]const f16
pub fn kBlocks(self, layer_i: usize, n: usize) []const fucina.quant.BlockQ8_0  // q8_0 mode
pub fn vBlocks(self, layer_i: usize, n: usize) []const fucina.quant.BlockQ8_0
pub fn byteSize(self) usize
```

- `appendLayer` converts the new tokens' f32 K/V rows to the cache dtype and
  writes them at offset `len()`, in one pass with no temporaries. Shape
  mismatches against the layer's geometry are `Error.KvCacheShapeMismatch`;
  exceeding `capacity` is `Error.KvCacheOverflow`. It does **not** advance
  the count — every layer appends at the same base offset; call `advance(m)`
  once per step after all layers have been written.
- `len()` is the decoder contract's cached-position count (`models.decoder`);
  the state itself is the `count` field.
- `truncate(keep_len)` rewinds to the first `keep_len` positions (a value at
  or above `len()` is a no-op). Decrementing the count suffices for both
  storage modes: buffers are pre-allocated at `capacity`, every position
  occupies whole per-(position, kv_head) rows, and every reader and
  `appendLayer` address rows strictly from the count — the next append
  overwrites the abandoned rows. This is the speculative decoder's rewind primitive ([§13.9](13-the-model-stack-fucina_models.md#139-speculative-decoding-srcmodelstextspeculative)): rejected
  draft positions are dropped with one integer store.
- `copyRows(src, start, end)` copies rows `[start, end)` of a
  same-geometry cache into this one at the SAME positions and advances
  the count to `end` — the cross-slot prefix-share primitive (lmserve's slot
  pool): a new conversation adopts another slot's common prompt prefix by
  memcpy instead of re-prefilling it. Positions are preserved, so the
  copied rows are exactly the rows a prefill of the same tokens would
  have produced; both storage dtypes copy. Requires `self.len() == start`
  (rows append in order) and `end <= src.len()`; a dtype or per-layer
  geometry mismatch is `Error.KvCacheShapeMismatch`.

```zig
test "kv cache: append, advance, truncate rewind" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    // 1 layer, 2 kv heads, head_dim 4, capacity 8 positions (f16 storage).
    var cache = try models.text.kv_cache.KvCache.init(&ctx, 1, 2, 4, 8);
    defer cache.deinit();

    var k = try models.text.kv_cache.KvInput.fromSlice(&ctx, .{ 3, 2, 4 }, &([_]f32{0.5} ** 24));
    defer k.deinit();
    var v = try models.text.kv_cache.KvInput.fromSlice(&ctx, .{ 3, 2, 4 }, &([_]f32{0.25} ** 24));
    defer v.deinit();
    try cache.appendLayer(&ctx, 0, &k, &v); // writes at offset len(), does not advance
    cache.advance(3); // once per step, after all layers
    try std.testing.expectEqual(@as(usize, 3), cache.len());
    try std.testing.expectEqual(@as(usize, 3 * 2 * 4), (try cache.kSlice(0, cache.len())).len);

    cache.truncate(1); // speculative rewind: drop rejected positions
    try std.testing.expectEqual(@as(usize, 1), cache.len());
}
```

`models.text.kv_persist` (`src/models/text/kv_persist.zig`) persists the cache to a
crash-safe append-only sidecar file so a conversation reopens warm across
process restarts, with zero re-prefill. The sidecar is a fixed header —
magic `FUXKV001`, a record count, and a per-layer geometry guard (any
mismatch with the opening cache ignores the file wholesale) — followed by
one record per position: the token id plus every layer's K/V row bytes
(both cache dtypes round-trip). Conversations served behind a preloaded
KV prefix (a cartridge, [§13.10](13-the-model-stack-fucina_models.md#1310-cartridges-srcmodelstextcartridgezig)) write `FUXKV002` instead: one extra header
field, `prefix_rows`, and records for EVERY position — the leading
`prefix_rows` records carry a token sentinel — so a restore is
self-describing and keeps the exact prefix it was saved with, even across
a cartridge swap; prefix-free conversations keep writing byte-identical V1
files. `reset(io, allocator, path, kv, prefix_rows)` arms a fresh sidecar
for the cache's geometry. `appendRange(io, allocator, path, kv, tokens,
prefix_rows)` writes the positions the file does not hold yet — record
data first, the header's record count last, so a torn append is invisible;
`prefix_rows + tokens.len != kv.len()` is `Error.KvPersistTokenMismatch`,
and a stored prefix shape that disagrees is treated as foreign (reset).
`load(io, allocator, path, kv)` resumes into an empty cache: it applies up
to the stored count (stopping early at a torn tail — the prefix stays
usable; a tear INSIDE a token-less prefix is not resumable), sets
`kv.len()`, and returns the caller-owned `Loaded{ tokens, prefix_rows }`, or
null when nothing usable exists (absent file, foreign geometry, or a
history beyond capacity). `chat.Conversation.enablePersistence` ([§13.8.2](13-the-model-stack-fucina_models.md#138-chat-srcmodelstextchatzig))
is the turnkey consumer and resumes `kv_prefix_rows` from the file.

## 13.5 Tokenizers

### 13.5.1 Byte-level BPE (`src/models/text/tokenizer.zig`)

`models.text.tokenizer.Tokenizer` is a native byte-level BPE tokenizer (GPT-2/Qwen
family) built entirely from a model's GGUF metadata
(`tokenizer.ggml.{tokens,merges,pre,token_type,bos_token_id,eos_token_id,add_bos_token,add_eos_token}`)
— no external tokenizer dependency, no per-model hardcoding. Error set:
`error{ NoTokenizerVocab, UnsupportedTokenizerFormat, TokenizerTooLarge } || Allocator.Error`.

```zig
pub const SpecialTokens = struct {
    bos: ?u32 = null, eos: ?u32 = null,
    prepend_bos: bool = false, append_eos: bool = false,
};
pub fn initFromGguf(allocator: Allocator, file: *const gguf.File, overrides: SpecialTokens) !Tokenizer
pub fn initFromParts(allocator, vocab_strings: []const []const u8,
                     merge_strings: []const []const u8, special: SpecialTokens) !Tokenizer
pub fn deinit(self: *Tokenizer) void
```

- `initFromGguf` requires a string-array vocab and non-empty merges, and
  refuses SentencePiece-scored models
  (`tokenizer.ggml.scores` present → `Error.UnsupportedTokenizerFormat` — use
  `spm_tokenizer` instead). Special tokens default from metadata; non-null
  `overrides` fields replace them, and `prepend_bos`/`append_eos` in the
  overrides can only force the policy **on** (a `false` leaves the metadata
  value in effect).
- The tokenizer copies all vocab/merge bytes into owned blobs, so it stays
  valid after the source `gguf.File` is freed. Duplicate token bytes resolve
  to the lowest id.
- **Pretokenizer parity**: the chunker is a faithful port of llama.cpp's
  hand-rolled qwen2 pretokenizer loop, backed by generated Unicode category
  tables — on valid UTF-8 input it chunks and encodes **token-ID-exact**
  against llama.cpp for qwen2-pre models (malformed UTF-8 is the one
  documented deviation). The GGUF's `tokenizer.ggml.pre` selects the chunker
  via the `pre: Pre = .qwen2` field
  (`Pre = enum { qwen2, qwen35, joyai_llm, glm4, inkling }`): `"qwen35"`
  (Qwen3.5/3.6/Bonsai — the qwen2 rules with `\p{M}` combining marks folded
  into the word class and excluded from punctuation runs), `"joyai-llm"` (the
  DeepSeek-V4 family's byte-oriented splitter), `"glm4"`/`"chatglm-bpe"` (qwen2
  rules with three-digit number runs), and `"inkling"` (the o200k variant with
  `\p{M}` in both word classes so combining marks attach to base letters,
  reproducing llama.cpp's collapsed-category regex semantics) — each backed by
  the generated `isMark` table and token-ID-exact against `llama-tokenize`. If
  the GGUF declares a pretokenizer other than an implemented chunker, encoding
  still proceeds with the qwen2 rules, but the id is recorded in the
  `pre_mismatch: ?[]u8` field and a warning is logged once — token-ID
  parity is then not guaranteed.

Encode/decode surface:

```zig
pub fn encode(self, allocator, text: []const u8) ![]u32       // BOS/EOS policy applied
pub fn encodeRaw(self, allocator, text: []const u8) ![]u32    // no BOS/EOS (templates own structure)
pub fn encodePlainAppend(self, allocator, text, out: *std.ArrayList(u32)) !void // no marker resolution
pub fn decode(self, allocator, ids: []const u32) ![]u8
pub fn decodeAppend(self, allocator, id: u32, out: *std.ArrayList(u8)) !void
pub fn tokenId(self, token: []const u8) ?u32
pub fn vocabSize(self) usize
pub fn eosId(self) ?u32 / pub fn bosId(self) ?u32 / pub fn isEos(self, id: u32) bool
```

`encode`/`encodeRaw` single-id-match special tokens atomically before
pretokenization: with `tokenizer.ggml.token_type` metadata, every
CONTROL/USER_DEFINED token, longest marker first (llama.cpp's
`tokenizer_st_partition`); without it, `<|...|>`-shaped markers resolved
against the vocabulary (a `<|` that does not open a known marker is left to
normal pretokenization); `encodePlainAppend` skips
marker resolution entirely for callers with their own control-token sets.
Returned slices are owned by the caller.

```zig
test "byte-level BPE: merges, special markers, round-trip" {
    const alloc = std.testing.allocator;
    const vocab = [_][]const u8{ "<|im_end|>", "h", "i", "hi" };
    const merges = [_][]const u8{"h i"};
    var tok = try models.text.tokenizer.Tokenizer.initFromParts(alloc, &vocab, &merges, .{});
    defer tok.deinit();

    const ids = try tok.encode(alloc, "hi<|im_end|>");
    defer alloc.free(ids);
    try std.testing.expectEqualSlices(u32, &.{ 3, 0 }, ids); // "hi" merged, marker resolved

    const text = try tok.decode(alloc, ids);
    defer alloc.free(text);
    try std.testing.expectEqualStrings("hi<|im_end|>", text);
}
```

`StreamDecoder` handles token-by-token generation where one token can end in
the middle of a multi-byte UTF-8 character:

```zig
pub const StreamDecoder = struct {
    pub fn init(tokenizer: *const Tokenizer) StreamDecoder
    pub fn deinit(self: *StreamDecoder, allocator: Allocator) void
    pub fn reset(self: *StreamDecoder) void
    pub fn push(self, allocator, id: u32, writer: *std.Io.Writer) !void
    pub fn flush(self, writer: *std.Io.Writer) !void
};
```

`push` emits only the complete-UTF-8 prefix and holds the incomplete tail
until a later token finishes it; `flush` emits any remainder when generation
ends. The sink is any `*std.Io.Writer` (stdout, an SSE response, an in-memory
buffer).

### 13.5.2 SentencePiece (`src/models/text/spm_tokenizer.zig`)

`models.text.spm_tokenizer.Tokenizer` is the Gemma-family counterpart: a faithful port
of llama.cpp's `llm_tokenizer_spm` Unigram model, driven by per-token
**scores** rather than merge ranks. Encoding seeds a max-heap of adjacent
symbol pairs keyed by the score of the token they would form, repeatedly
merges the highest-scoring pair, resegments, and byte-falls-back to `<0xXX>`
tokens for anything the vocabulary cannot cover. Special/control tokens
(`<start_of_turn>`, `<bos>`, …) are partitioned out of the raw text first
(longest marker wins), so they map to single ids. Error set:
`error{ NoTokenizerVocab, UnsupportedTokenizerFormat, TokenizerArrayTooShort, TokenizerTooLarge } || Allocator.Error`.

```zig
pub const Attr = enum(i32) { undef, normal, unknown, control, user_defined, unused, byte, _ };
pub const Options = struct {
    bos: ?u32 = null, eos: ?u32 = null, unk: ?u32 = null,
    add_bos: ?bool = null, add_eos: ?bool = null, add_space_prefix: ?bool = null,
};
pub fn initFromGguf(allocator, file: *const gguf.File, overrides: Options) !Tokenizer
pub fn initFromSlices(allocator, vocab_strings: []const []const u8,
    scores: []const f32, attrs: ?[]const Attr, opts: Options) !Tokenizer
```

`initFromGguf` requires `tokenizer.ggml.scores` (its absence means a byte-BPE
vocab → `Error.UnsupportedTokenizerFormat`); `tokenizer.ggml.token_type` is
optional (absent = every token NORMAL). Defaults follow llama.cpp's SPM
defaults when metadata is silent: `bos=1, eos=2, unk=0, add_bos=true,
add_eos=false, add_space_prefix=true`. `Attr` mirrors llama.cpp's
`LLAMA_TOKEN_TYPE_*` numbering and controls encode partitioning and decode
rendering (NORMAL unescapes `▁`, BYTE emits the raw byte, CONTROL/UNKNOWN are
suppressed). The public shape matches the byte-BPE tokenizer — `encode`,
`encodeRaw`, `decode`, `decodeAppend`, `tokenId`, `vocabSize`, `eosId`,
`bosId`, `isEos`, `deinit`, and an identical `StreamDecoder` — so a runner
picks one module per architecture and the rest of the stack (chat, data) is
generic over either.

```zig
test "SPM: score-driven merges and byte fallback" {
    const alloc = std.testing.allocator;
    const vocab = [_][]const u8{ "<unk>", "a", "b", "ab", "abc", "c", "<0x78>" };
    const scores = [_]f32{ 0, -1, -1, -3, -2.5, -1, -5 };
    var tok = try models.text.spm_tokenizer.Tokenizer.initFromSlices(alloc, &vocab, &scores, null, .{
        .add_bos = false,
        .add_space_prefix = false,
    });
    defer tok.deinit();

    const ids = try tok.encode(alloc, "abcx"); // "abc" outscores "ab"; 'x' byte-falls-back
    defer alloc.free(ids);
    try std.testing.expectEqualSlices(u32, &.{ 4, 6 }, ids);
}
```

### 13.5.3 Unicode tables (`src/models/text/unicode_categories.zig`)

Generated (do not edit) `\p{L}`/`\p{N}`/`\p{M}`/`\s` classification tables
(`isLetter`/`isNumber`/`isMark`/`isWhitespace`) matching llama.cpp's tokenizer
data for token-ID-exact pretokenizer parity; regenerate with
`python3 tools/gen_unicode_categories.py > src/models/text/unicode_categories.zig`
(the generator writes to stdout). Re-exported from `models.zig` as
`models.text.unicode_categories` so out-of-module consumers (nanochat's
example-local tokenizer) share the tables.

## 13.6 Sampling (`src/models/text/sampler.zig`)

```zig
pub const Config = struct {
    temperature: f32 = 0,        // <= 0 selects greedy (argmax)
    top_k: usize = 0,            // 0 = exact full-vocab nucleus (256-candidate first window)
    top_p: f32 = 1.0,            // nucleus: smallest prefix with cum. prob >= top_p
    min_p: f32 = 0,              // keep tokens with p >= min_p * p(best); 0 disables
    repeat_penalty: f32 = 1.0,   // llama.cpp penalty_repeat; 1.0 disables
    freq_penalty: f32 = 0,       // llama.cpp penalty_freq (per-count subtraction)
    presence_penalty: f32 = 0,   // llama.cpp penalty_present (flat subtraction)
    repeat_last_n: usize = 64,   // penalty window over the most recent tokens
    seed: u64 = 0,
    pub fn isGreedy(self: Config) bool;  // temperature <= 0
};

pub const Sampler = struct {
    processor: ?LogitProcessor = null,   // optional pre-sampling logit transform
    pub fn init(config: Config) Sampler;
    pub fn next(self: *Sampler, ctx: *ExecContext,
                logits: *fucina.Tensor(.{ .seq, .vocab }),   // shape [1, vocab]
                history: []const usize) !usize;
};
```

`next` implements the llama.cpp-compatible pipeline in order: logit
processor → penalties → greedy shortcut → top-k truncation → temperature
softmax → top-p → min-p → categorical draw. Semantics worth pinning:

- **Penalties mutate `logits` in place**, applied once per unique token in the
  last `repeat_last_n` tokens of `history` (with `count` = occurrences in the
  window, matching `llama_sampler_penalties`).
- With `temperature <= 0` (the default) the call is a deterministic argmax and
  the RNG is never touched — benchmarking and greedy decode share the path.
- Sampling uses a `std.Random.DefaultPrng` seeded once from `config.seed` at
  `init`: the draw sequence is a pure function of the seed, which the chat and
  speculative layers rely on (one draw per committed token, [§13.8](13-the-model-stack-fucina_models.md#138-chat-srcmodelstextchatzig)/[§13.9](13-the-model-stack-fucina_models.md#139-speculative-decoding-srcmodelstextspeculative)).
  With `top_k = 0` the nucleus is exact: probabilities carry the
  full-vocab softmax denominator, and the 256-candidate work window grows
  (×4 per pass, up to the vocab) until it covers the requested `top_p`
  mass — no tail is silently clipped. `top_p = 1.0` keeps the window at
  256, the work cap. An explicit `top_k` is clamped to 256 and keeps the
  llama.cpp order: top-k filter, softmax renormalized over the kept set,
  nucleus within it.
- A `Sampler` is single-stream mutable state (RNG + config): not thread-safe,
  one per decode stream.
- With a `processor` set, its `process` hook mutates the logits row before
  everything else and its `commit` hook observes the selected token on every
  path (greedy included, exactly once per `next`); a processor that masks out
  every candidate is `error.AllTokensMasked`.

```zig
test "sampler: greedy default, seed-deterministic sampling" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var logits = try fucina.Tensor(.{ .seq, .vocab }).fromSlice(&ctx, .{ 1, 5 }, &.{ 0.1, 0.2, 0.9, 0.3, 0.0 });
    defer logits.deinit();

    var greedy = models.text.sampler.Sampler.init(.{}); // temperature 0 => argmax
    try std.testing.expectEqual(@as(usize, 2), try greedy.next(&ctx, &logits, &.{}));

    var a = models.text.sampler.Sampler.init(.{ .temperature = 0.8, .top_k = 3, .seed = 42 });
    var b = models.text.sampler.Sampler.init(.{ .temperature = 0.8, .top_k = 3, .seed = 42 });
    for (0..8) |_| { // same seed -> same draw sequence
        try std.testing.expectEqual(try a.next(&ctx, &logits, &.{}), try b.next(&ctx, &logits, &.{}));
    }
}
```

### Logit processors (`src/models/text/logit_processor.zig`)

`LogitProcessor` is the injectable pre-sampling transform — the seam
grammar-constrained decoding plugs into, and the hook for any custom logit
policy (bias lists, banned-token rules, watermarking). It follows the
`DraftSource` vtable pattern ([§13.9](13-the-model-stack-fucina_models.md#139-speculative-decoding-srcmodelstextspeculative)):

```zig
pub const LogitProcessor = struct {
    ptr: *anyopaque,
    vtable: *const VTable,
    pub const VTable = struct {
        process: *const fn (ptr, logits: []f32, history: []const usize) anyerror!void,
        commit: *const fn (ptr, token: usize) anyerror!void,
        reset: ?*const fn (ptr) anyerror!void = null,
        // structural hooks (optional; pure deterministic lookahead):
        forcedTokens: ?*const fn (ptr, buf: []usize) usize = null,
        validPrefixLen: ?*const fn (ptr, tokens: []const usize) usize = null,
    };
    pub fn process(...) / commit(...) / reset(...)
    pub fn hasStructure(self) bool  // both structural hooks present
    pub fn forcedTokens(self, buf: []usize) usize / validPrefixLen(self, tokens) usize
};
```

`process` mutates one `[vocab]` logits row in place before the sampler's own
pipeline (a mask writes `-inf` over forbidden tokens); `commit` observes the
selected token, exactly once per `Sampler.next`; the optional `reset` re-arms
state for a fresh constrained region (`chat.Conversation` calls it at every
turn start). Because the seam lives inside the `Sampler`, every decode path
that samples through one — `chat.send`/`sendBatch`, the speculative
decoder's plain and verify steps, hand-rolled runner loops — picks the
processor up with no loop changes.

The two **structural hooks** let a processor expose what its state machine
knows beyond a mask: `forcedTokens` writes the unique legal continuation
(grammar-mandated JSON punctuation, a forced literal) and `validPrefixLen`
reports how many leading tokens of a candidate sequence the state accepts.
Both must be deterministic pure lookaheads. When present
(`hasStructure()`), the speculative layer turns them into drafts —
[§13.9](13-the-model-stack-fucina_models.md#139-speculative-decoding-srcmodelstextspeculative)'s `ConstrainedSource`.

**The seam is speculative-safe by construction**: the verify loop samples
each row only after that row's prefix is committed, and every sampled row
token is itself committed (accepted draft, correction, or bonus — [§13.9](13-the-model-stack-fucina_models.md#139-speculative-decoding-srcmodelstextspeculative)), so
`commit` keeps processor state exactly in step with history and no rollback
hook is needed. A draft token the mask forbids simply loses the
`sampled == draft` comparison and is rejected; the constrained speculative
stream is token-for-token identical to the constrained plain stream (proven
greedy + sampled in `chat_tests.zig`). One processor per decode stream, like
the sampler that hosts it.

```zig
test "logit processor: mask before sampling, observe the selection" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const OddMask = struct {
        commits: usize = 0,
        fn process(ptr: *anyopaque, logits: []f32, history: []const usize) anyerror!void {
            _ = ptr;
            _ = history;
            for (logits, 0..) |*l, tok| {
                if (tok % 2 == 1) l.* = -std.math.inf(f32);
            }
        }
        fn commit(ptr: *anyopaque, token: usize) anyerror!void {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            _ = token;
            self.commits += 1;
        }
    };
    var mask = OddMask{};

    var logits = try fucina.Tensor(.{ .seq, .vocab }).fromSlice(&ctx, .{ 1, 4 }, &.{ 0.1, 0.9, 0.5, 0.8 });
    defer logits.deinit();
    var s = models.text.sampler.Sampler.init(.{}); // greedy
    s.processor = .{ .ptr = &mask, .vtable = &.{ .process = OddMask.process, .commit = OddMask.commit } };
    // Unmasked argmax would be token 1; the mask forces the best even id.
    try std.testing.expectEqual(@as(usize, 2), try s.next(&ctx, &logits, &.{}));
    try std.testing.expectEqual(@as(usize, 1), mask.commits);
}
```

### Constrained decoding: llguidance (`src/models/text/llguidance.zig`, `-Dllguidance`)

`models.text.llguidance.Constraint` compiles a grammar with the vendored
[llguidance](https://github.com/guidance-ai/llguidance) engine
(`vendor/llguidance`, MIT — version/update procedure in its
[README](../../vendor/llguidance/README.md)) and adapts it to the
`LogitProcessor` seam: JSON-schema/regex/Lark-constrained generation for any
runner built on the shared sampler, ~50 µs of pure CPU mask work per token.
Requires `-Dllguidance=true` ([§2.2](02-toolchain-build-and-project-wiring.md#22-build-options-buildzig); cargo builds the Rust staticlib); without
it the module still compiles and `Constraint.init` returns
`error.LlguidanceNotEnabled`.
[CONSTRAINED-DECODING.md](../CONSTRAINED-DECODING.md) is the full design record
(seam adjudication, tokenizer-bridge details, the no-rollback speculation
argument, measured results).

```zig
pub const enabled: bool;                 // build-flag mirror
pub fn version() []const u8;             // "llguidance@X.Y.Z derivre@..."

pub const Grammar = union(enum) {
    json_schema: []const u8,  // stringified JSON schema
    regex: []const u8,        // Rust-syntax regex the reply must match
    lark: []const u8,         // llguidance's Lark-variant grammar
    llguidance: []const u8,   // composite JSON list form
};
pub const Options = struct {
    eos_token: ?u32 = null,      // forced when the grammar completes; default tokenizer eosId()
    extra_eos: []const u32 = &.{},
    n_vocab: ?usize = null,      // model vocab when padded larger than the tokenizer's
    log_level: u32 = 1,          // 0 silent, 1 warnings, 2 info
};

pub const Constraint = struct {
    pub fn init(allocator, tokenizer: anytype, grammar: Grammar, options: Options) Error!Constraint
    pub fn deinit(self: *Constraint) void
    pub fn clone(self: *const Constraint) Error!Constraint  // independent per-stream twin
    pub fn processor(self: *Constraint) LogitProcessor  // install on a Sampler / chat.Options
    pub fn isStopped(self: *const Constraint) bool      // grammar terminated
    pub fn isAccepting(self: *Constraint) bool          // tokens so far form a complete sentence
    pub fn reset(self: *Constraint) Error!void          // re-arm for a fresh reply
    pub fn ffTokens(self: *Constraint, buf: []u32) Error!usize // grammar-forced continuation
};
```

- `tokenizer` is `*const models.text.tokenizer.Tokenizer` (byte-BPE) or
  `*const models.text.spm_tokenizer.Tokenizer` (SPM) — both borrowed. The bridge
  hands llguidance every token's RAW bytes: BPE tokens byte-decoded, SPM
  pieces unescaped (`▁` → space) and `<0xXX>` byte tokens as their byte.
  Control tokens (BPE: the `<|...|>` marker shape; SPM: `control`/`unknown`
  attrs) carry toktrie's `0xFF` special marker, so a grammar whose text could
  spell `<|im_end|>` can never steer the model into emitting the actual
  control token. Padding ids past the tokenizer vocab (set
  `n_vocab = config.vocab_size`) get empty bytes and are never allowed.
- **Stop forcing**: when the grammar completes, the mask allows only
  `eos_token` — pass the chat template's stop-marker id so a finished grammar
  ends the turn through the existing stop handling; a matcher failure
  mid-decode also degrades to the forced stop (details logged at
  `log_level >= 1`). An invalid grammar fails `init` loudly instead.
- One `Constraint` per decode stream; do not move it after `processor()` is
  taken. `chat.Conversation` re-arms it per turn via the `reset` hook.
  Multi-stream decode (`sendBatch`, `--streams`) gives each stream a
  `clone()` — a deep-cloned matcher over the refcounted tokenizer with the
  tokenize bridge borrowed from the original (which must outlive the
  clones); no vocab rebuild or grammar recompilation.
- The processor exposes the [§13.6](13-the-model-stack-fucina_models.md#136-sampling-srcmodelstextsamplerzig) structural hooks (`forcedTokens` /
  `validPrefixLen`, backed by llguidance's fast-forward and
  token-validation lookaheads), so speculation routes it through the
  grammar-aware draft source automatically ([§13.9](13-the-model-stack-fucina_models.md#139-speculative-decoding-srcmodelstextspeculative)): grammar-forced spans
  draft themselves and are accepted with certainty. Measured on
  Qwen3-0.6B-Q8_0 with the JSON-schema example below: 0% draft acceptance
  (cost gate mutes speculation) without the hooks, 83% acceptance at
  1.24 tok/step with them — output byte-identical either way.
- The runner flags (qwen3 + gemma4, [§14.2](14-model-families-and-example-applications.md#142-qwen3--dense-and-moe-srcmodelsqwen3modelzig)/[§14.4](14-model-families-and-example-applications.md#144-gemma-4--text--moe-srcmodelsgemma)): `--json-schema JSON|@FILE`,
  `--lark GRAMMAR|@FILE`, `--regex PATTERN` — combine with `--no-think` on
  reasoning models (the grammar governs the whole reply, thinking channel
  included). Composes with `--spec` (output identical to the plain run) and
  with qwen3's `--streams` (per-stream clones; batch == sequential
  token-for-token).

```zig
test "llguidance: JSON-schema constrained greedy decode" {
    if (!models.text.llguidance.enabled) return error.SkipZigTest; // -Dllguidance=true builds only
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const vocab = [_][]const u8{ "{", "}", "\"", "a", ":", "1", "<|end|>" };
    var tok = try models.text.tokenizer.Tokenizer.initFromParts(alloc, &vocab, &.{}, .{ .eos = 6 });
    defer tok.deinit();

    var constraint = try models.text.llguidance.Constraint.init(alloc, &tok, .{
        .json_schema =
        \\{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"],"additionalProperties":false}
    }, .{});
    defer constraint.deinit();

    var s = models.text.sampler.Sampler.init(.{}); // greedy
    s.processor = constraint.processor();

    // The model "wants" '}' everywhere; the mask walks it through a valid
    // object instead — '}' only becomes samplable once {"a":1 is complete —
    // then the finished grammar forces the stop token.
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(alloc);
    var steps: usize = 0;
    while (!constraint.isStopped() and steps < 16) : (steps += 1) {
        var logits = try fucina.Tensor(.{ .seq, .vocab }).fromSlice(&ctx, .{ 1, 7 }, &.{ 0, 1, 0, 0, 0, 0.5, 0 });
        defer logits.deinit();
        const next = try s.next(&ctx, &logits, &.{});
        if (next == 6) break;
        try tok.decodeAppend(alloc, @intCast(next), &out);
    }
    try std.testing.expectEqualStrings("{\"a\":1}", out.items);
}
```

## 13.7 SFT data (`src/models/text/data.zig`)

Minimal supervised-fine-tuning helpers, generic across model families. Error
set: `error{ MalformedJsonl, SampleTooLong, EmptyDataset, InvalidLoaderState }`.

```zig
pub const Pair = struct { instruction: []const u8, response: []const u8 };
pub const SftText = struct {
    pairs: []const Pair,
    blob: ?[]u8 = null,
    pub const JsonlOptions = struct {
        instruction_key: []const u8 = "instruction",
        response_key: []const u8 = "response",
    };
    pub fn fromPairs(pairs: []const Pair) SftText;                    // zero-copy borrow
    pub fn fromJsonl(allocator, io: std.Io, path: []const u8, opts: JsonlOptions) !SftText;
    pub fn deinit(self: *SftText, allocator: Allocator) void;
};
```

`fromPairs` borrows caller-owned pairs (`deinit` frees nothing). `fromJsonl`
loads one JSON object per line, reading strings under the configured keys into
one owned blob (so the result outlives the file); blank lines are skipped, and
any malformed line fails with `Error.MalformedJsonl` after logging the path
and line number.

```zig
pub const Sample = struct {
    inputs: []usize,   // full sequence minus its final token
    labels: []usize,   // next-token shift; prompt positions masked
    pub fn deinit(self: *Sample, allocator: Allocator) void;
};
pub const EncodeOptions = struct {
    seq_max: usize = 256,
    ignore_index: usize = std.math.maxInt(usize),  // the trainer's mask sentinel
    mask_prompt: bool = true,
    system: ?[]const u8 = null,
    think_off: bool = true,
};
pub fn encodePrompt(allocator, tokenizer: anytype, template: chat.Template,
                    instruction: []const u8, opts: EncodeOptions) ![]usize
pub fn encodePair(allocator, tokenizer: anytype, template: chat.Template,
                  pair: Pair, opts: EncodeOptions) !Sample
```

`encodePair` renders one user turn through the chat template, tokenizes, and
builds the shifted training pair: `inputs` = prompt ++ response tokens minus
the last; `labels[i-1]` = token `i`, with all prompt positions replaced by
`opts.ignore_index` unless `mask_prompt` is off. The response (plus the
template's stop marker) is encoded **separately** from the prompt —
concatenating the text first would move BPE chunk boundaries across the join
and change token ids. Samples are truncated to `seq_max` input positions; a
window that leaves no supervised token is `Error.SampleTooLong`. The
`tokenizer` parameter is duck-typed over `encodeRaw` — byte-BPE and SPM both
satisfy it — and `ignore_index` is injected so this module never imports a
trainer ([§11](11-training-optimizers-evolution-strategies-lora-and-checkpoints.md)).

```zig
test "encodePair: render + tokenize + shift + prompt mask" {
    const alloc = std.testing.allocator;
    const vocab = [_][]const u8{
        "<|im_start|>", "<|im_end|>", "u", "s", "e", "r", "a", "n", "t", "i",
        "h",            "k",          "y", "o", "m", "<", ">", "/", "\xC4\x8A", // Ċ = byte-level '\n'
    };
    var tok = try models.text.tokenizer.Tokenizer.initFromParts(alloc, &vocab, &.{}, .{});
    defer tok.deinit();

    const chatml = models.text.chat.Template{ .format = .chatml };
    var sample = try models.text.data.encodePair(alloc, &tok, chatml, .{
        .instruction = "hi",
        .response = "yo",
    }, .{ .seq_max = 64, .ignore_index = 9999 });
    defer sample.deinit(alloc);

    try std.testing.expectEqual(sample.inputs.len, sample.labels.len);
    try std.testing.expectEqual(@as(usize, 9999), sample.labels[0]); // prompt masked
    try std.testing.expectEqual(@as(usize, 1), sample.labels[sample.labels.len - 1]); // stop marker supervised
}
```

```zig
pub const Loader = struct {
    pub const Order = enum { sequential, shuffled };
    pub const State = struct { seed: u64, epoch: u64, index: u64 };
    pub fn init(allocator, n: usize, order: Order, seed: u64) !Loader;   // n == 0 => EmptyDataset
    pub fn deinit(self: *Loader, allocator: Allocator) void;
    pub fn next(self: *Loader) usize;
    pub fn state(self: *const Loader) State;
    pub fn restore(self: *Loader, s: State) !void;   // out-of-range index => InvalidLoaderState
};
```

`Loader` is a deterministic sample-order iterator. `.sequential` is plain
round-robin; `.shuffled` draws each epoch as a fresh permutation that is a
**pure function of `(seed, epoch)`** — a checkpoint contract: the permutation
is identity order followed by a Fisher–Yates pass driven by a splitmix64
stream seeded with `rng.at(seed, epoch)` (`j = splitmix64 % (i+1)` for `i`
from `n-1` down to `1`). The formula is golden-pinned in `data_tests.zig` and
may never change once checkpoints exist against it; `restore` regenerates the
exact stream position from a saved `State` (u64 fields, so it round-trips
through `trainer_state.json` unchanged — [§11](11-training-optimizers-evolution-strategies-lora-and-checkpoints.md)).

```zig
test "Loader: (seed, epoch) -> permutation is a checkpoint contract" {
    const alloc = std.testing.allocator;
    var loader = try models.text.data.Loader.init(alloc, 8, .shuffled, 42);
    defer loader.deinit(alloc);
    // Golden-pinned: this exact order may never change once checkpoints exist.
    try std.testing.expectEqualSlices(usize, &.{ 3, 6, 0, 7, 1, 2, 5, 4 }, loader.perm);

    for (0..3) |_| _ = loader.next();
    const s = loader.state();
    var expect: [8]usize = undefined;
    for (&expect) |*e| e.* = loader.next(); // crosses the epoch boundary

    var replay = try models.text.data.Loader.init(alloc, 8, .shuffled, 0);
    defer replay.deinit(alloc);
    try replay.restore(s); // seed/epoch/index come from the saved State
    for (expect) |want| try std.testing.expectEqual(want, replay.next());
}
```

## 13.8 Chat (`src/models/text/chat.zig`)

### 13.8.1 Templates

```zig
pub const Format = enum { chatml, llama3, gemma, gemma4 };
pub const Template = struct {
    format: Format,
    pub fn detect(chat_template: ?[]const u8) ?Template;
    pub fn stopMarker(self: Template) []const u8;
    pub fn renderTurn(self, allocator, buf: *std.ArrayList(u8),
        system: ?[]const u8, user: []const u8, first: bool, think_off: bool) !void;
};
```

`detect` sniffs the format from a GGUF `tokenizer.chat_template` string
(`<|im_start|>` → chatml, `<|start_header_id|>` → llama3, `<|turn>` → gemma4,
`<start_of_turn>` → gemma; anything else → `null`). `stopMarker` is the token
text that ends an assistant turn (`<|im_end|>`, `<|eot_id|>`,
`<end_of_turn>`, `<turn|>`). `renderTurn` appends the text to feed for one
user turn: `first` emits the conversation-start (bos/system) scaffolding,
otherwise it first closes the previous assistant turn; `think_off` suppresses
the reasoning channel on ChatML (empty `<think>` block) and Gemma 4
(primed-empty thought channel). Gemma 1–3 has no system role — the system
prompt is folded into the first user turn.

```zig
test "chat template: detect from GGUF metadata, render a turn" {
    const alloc = std.testing.allocator;
    const t = models.text.chat.Template.detect("... {{ '<|im_start|>' }} ...").?;
    try std.testing.expectEqual(models.text.chat.Format.chatml, t.format);
    try std.testing.expectEqualStrings("<|im_end|>", t.stopMarker());

    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(alloc);
    try t.renderTurn(alloc, &buf, "Be terse.", "Hi", true, false);
    try std.testing.expectEqualStrings(
        "<|im_start|>system\nBe terse.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
        buf.items,
    );
}
```

`renderMessages` is `renderTurn`'s stateless twin: it renders a FULL message
history for a fresh conversation, ending with the assistant-turn opener — the
shape a messages-array API server receives on every request (the lmserve
example, `examples/lmserve/`).

```zig
pub const Message = struct {
    role: Role, // enum { system, user, assistant }
    content: []const u8, // borrowed
};
pub fn renderMessages(self, allocator, buf: *std.ArrayList(u8),
    messages: []const Message, think_off: bool) !void;
```

ChatML and Llama 3 render every message as its own role block, any order.
The Gemma formats have a single conversation-start system slot: leading
system messages merge into it (Gemma 1–3: folded into the first user turn),
and a later one is `error.SystemMidConversation`. An empty list is
`error.EmptyMessages`; a trailing assistant message is
`error.TrailingAssistantMessage` (rendering would open a SECOND assistant
turn after it rather than continue it). Historical assistant contents have
their reasoning block stripped (ChatML `<think>…</think>`, Gemma 4
`<|channel>thought…<channel|>`) — the reference templates drop prior-turn
reasoning, and stateless clients replay content without it. First-turn output
is byte-identical to `renderTurn`'s, so a stateless render prefills the same
KV prefix as an incrementally driven conversation.

```zig
test "chat template: render a full message history (stateless server shape)" {
    const alloc = std.testing.allocator;
    const t = models.text.chat.Template{ .format = .chatml };
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(alloc);
    try t.renderMessages(alloc, &buf, &.{
        .{ .role = .system, .content = "Be terse." },
        .{ .role = .user, .content = "Hi" },
        .{ .role = .assistant, .content = "<think>\nhm\n</think>\n\nHello!" },
        .{ .role = .user, .content = "Bye" },
    }, true);
    try std.testing.expectEqualStrings(
        "<|im_start|>system\nBe terse.<|im_end|>\n" ++
            "<|im_start|>user\nHi<|im_end|>\n" ++
            "<|im_start|>assistant\nHello!<|im_end|>\n" ++ // <think> stripped
            "<|im_start|>user\nBye<|im_end|>\n" ++
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        buf.items,
    );
}
```

### 13.8.2 `Conversation(Model, Tok)`

```zig
pub fn Conversation(comptime Model: type, comptime Tok: type) type
```

Comptime-generic multi-turn chat over a model family and a tokenizer module.
The contract:

- `Model` satisfies the decoder contract (`models.decoder.assertDecoder`,
  [§14.1](14-model-families-and-example-applications.md#141-conventions-shared-by-every-family)) with `caps.rewind` over the shared [§13.4](13-the-model-stack-fucina_models.md#134-kv-cache-srcmodelstextkv_cachezig) `KvCache`:
  `initCache(ctx, capacity) !KvCache` and the decode entries
  `forwardStep(ctx, kv, token_ids, pos0) !Tensor(.{ .seq, .vocab })`
  (last-row logits) and `forwardStepAllLogits` (same signature, all-row
  logits — executed only when speculation is enabled, required by
  `caps.rewind` regardless). Beyond the contract, `Conversation` requires
  `config.vocab_size` and the cache `capacity` field. `sendBatch` runs
  only when `caps.batch` declares
  `forwardStepBatch(ctx, caches: []const *KvCache, token_ids: []const usize)`;
  the gate is comptime, so families without it still instantiate the type
  and get `error.BatchDecodeUnsupported` at runtime.
- `Tok` is the tokenizer **module** (`models.text.tokenizer` or `models.text.spm_tokenizer`):
  it must provide a `Tokenizer` type with
  `tokenId`/`eosId`/`encodeRaw`/`decodeAppend` and a `StreamDecoder`.

```zig
pub const Options = struct {
    system: ?[]const u8 = null,
    capacity: usize = 4096,               // total KV size; the whole conversation must fit
    max_response_tokens: usize = 1024,    // per-reply cap
    think_off: bool = false,
    sampler: sampler.Config = .{},
    extra_stop_ids: []const u32 = &.{},   // borrowed
    stop_sequences: []const []const u8 = &.{},  // borrowed; composes with speculation
    logit_processor: ?sampler.LogitProcessor = null,  // borrowed; §13.6
    speculation: bool = false,
    spec_options: speculative.Options = .{},
    io: ?std.Io = null,                   // clock for the decoder's live cost gate
};

pub const WarmState = struct { cache: KvCache, tokens: []const usize, prefix_rows: usize = 0 };

pub fn init(ctx: *ExecContext, model: *const Model, tokenizer: *const Tok.Tokenizer,
            template: Template, options: Options) !Self
pub fn initWarm(ctx: *ExecContext, model: *const Model, tokenizer: *const Tok.Tokenizer,
                template: Template, options: Options, warm: WarmState) !Self
pub fn deinit(self: *Self) void
pub fn takeCache(self: *Self) KvCache
pub fn notePrefixRows(self: *Self, rows: usize) !void
pub fn send(self: *Self, user: []const u8, writer: *std.Io.Writer) !usize
pub fn sendRendered(self: *Self, rendered: []const u8, writer: *std.Io.Writer) !usize
pub fn sendRenderedReuse(self: *Self, rendered: []const u8, writer: *std.Io.Writer) !usize
pub fn sendTokensReuse(self: *Self, ids: []const u32, writer: *std.Io.Writer) !usize
pub fn sendBatch(convos: []const *Self, users: []const []const u8,
                 writers: []const *std.Io.Writer, produced: []usize) !void
pub fn sendBatchTokensReuse(convos: []const *Self, ids_list: []const []const u32,
                            writers: []const *std.Io.Writer, produced: []usize,
                            errs: []?anyerror) !void
pub fn addSpecReference(self: *Self, tokens: []const usize) !void
pub fn enablePersistence(self: *Self, io: std.Io, path: []const u8) !usize
pub fn specStats(self: *const Self) ?speculative.Stats
```

Semantics:

- `init` resolves the stop id as `tokenizer.tokenId(template.stopMarker())
  orelse tokenizer.eosId()`, builds the KV cache via the model's own
  `initCache`, and — with `speculation` on — heap-allocates the speculative
  state (a `SpeculationIndex` cascade plus a `SpeculativeDecoder(Model)`,
  [§13.9](13-the-model-stack-fucina_models.md#139-speculative-decoding-srcmodelstextspeculative)), wiring the stop id into `spec_options.stop_token` and aligning the
  cascade's `accounting_min_draft` with the decoder's `min_draft`.
  `stop_sequences` compose with `speculation`: the turn-boundary gate scans
  the decoded reply bytes in the plain loop's exact order, the completing
  token is neither streamed nor committed (the turn trim discards it from
  history and KV), and spec == plain — reply bytes, fired index, post-turn
  state — is proven in `chat_tests.zig`. The `Conversation` borrows `ctx`,
  `model`, `tokenizer`, `system`, `extra_stop_ids`, and `stop_sequences` —
  they must outlive it; `deinit` releases the cache, history, stream decoder,
  and speculative state.
- `send` renders the turn, tokenizes it (`error.ContextFull` if the prefix
  does not fit the remaining KV capacity), prefills at the current cache
  position (one KV cache persists across turns — each turn prefills only the
  new tokens), then decodes token by token: sample → stop check → stream
  through the `StreamDecoder` (flushed per token) → forward. It returns the
  number of response tokens produced. A turn ends on the template stop marker,
  any of `extra_stop_ids`, the `max_response_tokens` budget, or KV exhaustion.
  With `stop_sequences`, generation stops **before streaming** the token whose
  decoded reply text completes a sequence (the completing token is not
  committed), and `fired_stop` records the index (into `stop_sequences`)
  of the sequence that fired — null when the turn ended any other way,
  reset per turn — the stop-attribution seam servers report from
  (lmserve's Anthropic `stop_sequence` field).
- `sendRendered` is `send` over caller-provided pre-rendered template text —
  the stateless-API entry: `Template.renderMessages` renders a full message
  history and a FRESH conversation prefills it in one turn (`Options.system`
  is not consulted; the caller rendered everything). Streaming, stop
  handling, speculation, and the logit processor behave exactly as in `send`;
  the equivalence with an incrementally driven conversation is proven in
  `chat_tests.zig`.
- `sendRenderedReuse` is `sendRendered` with **cross-request KV reuse** — the
  stateless-server seam (lmserve; llama.cpp's `cache_prompt`). `rendered`
  must be a FULL-history render (never a `renderTurn` suffix). The
  conversation may hold a previous request's committed state, adopted at
  construction via `initWarm(…, warm)`: `warm.cache` ownership transfers on
  the call (error paths included; `options.capacity` is ignored — the
  adopted capacity governs) and `warm.tokens` — the token shadow describing
  the cache's positions — is borrowed, copied, and clamps the cache. The
  reconcile keeps the longest common token prefix between the render's ids
  and the committed history — zero prefill for that span; `reused_prefix`
  reports its length (the OpenAI `cached_tokens` number) — capped at
  `ids.len - 1` because logits are not cached state; everything past it is
  rewound (`KvCache.truncate` + history shrink) and prefilled. Token-level
  LCP absorbs every render divergence (stripped reasoning blocks, edited
  history, another client) by reusing less; on a fresh conversation the
  entry degenerates to `sendRendered` exactly. After the request,
  `takeCache` releases the cache to the caller (snapshot the shadow from
  `history.items[0..cache.len()]` BEFORE taking — the bound trims the one
  committed-but-unforwarded token an aborted turn can leave); deinit skips a
  taken cache. `sendTokensReuse` is the same entry over pre-encoded ids —
  for a server that already tokenized the render to score candidate slots
  by common prefix. Speculation composes on both sides (the lmserve
  `--spec` seam): the SpeculationIndex mirrors committed history
  append-only, so a warm adoption or a reconcile rewind rebuilds it in
  place from the reconciled history before the turn; the turn then ends
  with the same catch-up forward the plain loop issues, so the released
  slot's token shadow describes every committed token (spec == plain
  reuse output and state, and warm-reuse == fresh-stateless equivalence,
  are proven in `chat_tests.zig`). Only the batched entry
  (`sendBatchTokensReuse`) requires speculation off on every stream
  (`error.SpeculationWithReuse`).
- With speculation on, `send` routes through the decoder with a turn-boundary
  gate that stops streaming/index-learning at the stop marker and trims any
  verify-batch overshoot from history **and** the KV cache — unconditionally,
  on error paths included — so the post-turn state matches the plain path's
  exactly. The equivalence (token-for-token, draw-for-draw across a persistent
  sampler, greedy and sampled) is proven in `chat_tests.zig`.
- `logit_processor` installs a [§13.6](13-the-model-stack-fucina_models.md#136-sampling-srcmodelstextsamplerzig) processor (e.g. a `models.text.llguidance`
  grammar constraint) on the conversation's sampler and re-arms it via its
  `reset` hook at every turn start, so the same constraint governs each
  assistant reply independently — on the plain, speculative, and `sendBatch`
  paths alike (constrained plain == constrained speculative is part of the
  `chat_tests.zig` equivalence proofs). When the processor exposes the
  structural hooks (`hasStructure()`), speculation automatically routes
  through the grammar-aware draft source (13.9.6): forced grammar spans
  draft themselves. `sendBatch` requires one processor instance per stream
  (a shared pointer is `error.SharedBatchProcessor` — single-stream state;
  llguidance streams use `Constraint.clone`).
- `sendBatch` decodes one message on each of N sibling conversations in
  lockstep: every step forwards one token per live stream through
  `forwardStepBatch` (one m=N weight pass instead of N GEMVs), then samples
  each stream from its own logits row with its own sampler/history. Per-stream
  semantics match a plain `send` exactly; below the m-dependent kernel
  thresholds the produced tokens are bit-identical to N sequential sends,
  beyond them rows can differ by ~1e-6 reassociation drift. Requirements,
  checked up front: non-empty batch (`error.EmptyBatch`), equal slice lengths
  (`error.BatchLengthMismatch`), speculation off on every stream
  (`error.SpeculationWithBatch`), one shared `ctx`/`model`
  (`error.MixedBatchModels`), distinct conversations
  (`error.DuplicateBatchConversation`). Ownership contract on error: the
  batch aborts, `produced` is left unwritten, and **every** stream's history
  is trimmed back to its cache's token-backed length (prefix-aware, [§13.10](13-the-model-stack-fucina_models.md#1310-cartridges-srcmodelstextcartridgezig)'s
  preloaded-prefix conversations included), so healthy siblings of the failing stream
  remain internally consistent and resendable; bytes already streamed are not
  recalled. Turn prefills run per stream.
- `sendBatchTokensReuse` is `sendTokensReuse` over N sibling conversations
  in lockstep — the server batching entry (lmserve `--batch`). Per stream
  it runs the reuse reconcile, prefills the un-reused suffix, and samples
  the first token, then joins the `sendBatch` lockstep loop. Unlike
  `sendBatch`, per-stream failures are ISOLATED: a stream whose sink write
  (client gone), prologue (`error.ContextFull`), or sampler
  (`error.AllTokensMasked`) fails finishes with the error recorded in
  `errs[i]` while the remaining streams keep decoding; only a
  shared-compute failure (the batched forward itself) aborts the whole
  batch. `produced[i]` and `errs[i]` are always written, and on every path
  out each stream's history is trimmed back to its cache's token-backed
  length, so every
  conversation stays consistent and resendable. Speculation must be off on
  every stream (`error.SpeculationWithReuse`); the other up-front checks
  match `sendBatch`.
- `notePrefixRows(rows)` declares the first `rows` cache positions a
  PRELOADED KV prefix with no token shadow — the cartridge serving seam
  ([§13.10](13-the-model-stack-fucina_models.md#1310-cartridges-srcmodelstextcartridgezig)): `writeToCache` into the fresh conversation's cache, then this,
  once, before any send; the cache must hold exactly the prefix and
  speculation must be off (`error.InvalidPrefix` otherwise). The reuse
  reconcile never rewinds into the prefix, KV persistence records the
  prefix shape, and `WarmState.prefix_rows` adopts the same declaration on
  a warm start.
- `addSpecReference(tokens)` injects a tokenized reference document into the
  speculation index (the RAG seam); `error.SpeculationDisabled` when
  speculation is off. `specStats` returns the decoder's lifetime
  `speculative.Stats` (null when off).
- `enablePersistence(io, path)` arms KV persistence (`kv_persist.zig`) on a
  fresh conversation — once, before any send. A compatible saved conversation
  at `path` resumes into it (token history and KV cache restored, zero
  re-prefill); otherwise the file is reset so a stale or foreign prefix
  cannot become this conversation's. It returns the number of resumed
  positions (0 = fresh start). Every subsequent `send`/`sendRendered` turn
  appends its new positions to the append-only sidecar — the record count is
  published last, so a crash mid-append leaves a consistent prefix.
- A `Conversation` is single-threaded mutable state; `sendBatch` runs on the
  caller's thread over all streams.

```zig
fn snippetConversation(ctx: *fucina.ExecContext, io: std.Io, out: *std.Io.Writer) !void {
    const alloc = ctx.allocator;
    var file = try fucina.gguf.File.loadMmap(alloc, io, "qwen3-0.6b.gguf");
    defer file.deinit();
    var model = try models.qwen3.model.Model.loadGgufFromFile(ctx, &file, try models.qwen3.model.Config.fromGguf(&file));
    defer model.deinit();
    var tok = try models.text.tokenizer.Tokenizer.initFromGguf(alloc, &file, .{});
    defer tok.deinit();
    const template = models.text.chat.Template.detect(file.getString("tokenizer.chat_template")) orelse
        models.text.chat.Template{ .format = .chatml };

    const Convo = models.text.chat.Conversation(models.qwen3.model.Model, models.text.tokenizer);
    var convo = try Convo.init(ctx, &model, &tok, template, .{
        .capacity = 4096,
        .sampler = .{ .temperature = 0.7, .top_k = 20, .seed = 42 },
        .speculation = true, // lossless draft-free speculative decoding
    });
    defer convo.deinit();
    _ = try convo.send("Why is the sky blue?", out); // streams tokens to `out`
} // requires model assets to run
```

## 13.9 Speculative decoding (`src/models/text/speculative/`)

Training-free, **draft-model-free** speculative decoding: drafts come from
cheap deterministic indexes over text the model has already seen — no extra
weights. [`SPECULATIVE.md`](../SPECULATIVE.md) is the full design record (proof
obligations, verify economics, adjudicated alternatives); this section covers
the public surface.

**The lossless contract** (`core.zig` header is normative): because every
draft source is deterministic (a one-hot proposal distribution), rejection
sampling degenerates to running the FULL sampling pipeline on the target
logits at each verified position, conditioned on the hypothetical prefix —
accept while `sampled == draft[i]`; at the first mismatch the sampled token
IS the correction token; on full acceptance the (k+1)-th row yields a free
bonus token. Greedy is the same code path (temperature ≤ 0 makes the sampler
an argmax). Token IDs are compared, never probabilities. Exactly **one RNG
draw is consumed per committed token** — the same pattern as a plain run —
and committing `Options.stop_token` ends the verify row loop immediately, so
a persistent sampler never desyncs. Given bitwise-identical logits the output
stream equals the non-speculative run's. Logits are computed in verify
batches of m = 1+draft rows, and byte-identity with a plain run rests on
two legs: `Options.pin_kernels` (the default) runs the verify forward
under `ExecContext.pinRowwiseKernels` ([§6.1](06-the-execution-runtime-execcontext-and-the-memory-model.md#61-execcontext-role-and-lifecycle-srcexeczig-srcexecruntimezig)), so every batched quant
matmul reproduces the m = 1 numerics bitwise — the verify logits AND the
KV rows the verify leaves behind for committed positions — and the caller
must prefill both runs identically (prefill kernels are
batch-shape-dependent; the chat layer's speculative turn prefills exactly
as the plain turn does). Unpinned, the m-dependent kernel switches can
drift the logits ~1e-6 and flip a near-tied sample ("same distribution
always; same sample stream whenever the logits match bitwise").

### 13.9.1 Core (`speculative/core.zig`)

```zig
pub const TopKRow = struct { token: usize, topk: []const u32 };  // borrowed per call

pub const DraftSource = struct {
    ptr: *anyopaque,
    vtable: *const VTable,
    pub const VTable = struct {
        suggest: *const fn (ptr, context: []const usize, buf: []usize) usize,
        observe: *const fn (ptr, committed: []const usize) void,
        observeTopK: ?*const fn (ptr, positions: []const TopKRow) void = null,
        truncatePending: ?*const fn (ptr, new_len: usize) void = null,
    };
    pub fn suggest(...) usize / observe(...) void / observeTopK(...) void
    pub fn wantsTopK(self) bool
    pub fn truncatePending(self, new_len: usize) void
};
```

`DraftSource` is the injectable proposer interface: `suggest` writes up to
`buf.len` continuation tokens for the committed `context` (0 = no draft),
`observe` feeds newly committed tokens back, and the optional `observeTopK`
receives per-position top-K candidates from the verification logits — when
null, the decoder skips computing the top-k entirely. Sources must be
deterministic; the decoder clamps a lying `suggest` return value at runtime.
The optional `truncatePending` lets a wrapper (the chat turn-boundary gate,
the 13.9.6 grammar filter) tell the source its just-returned draft was
shortened, so pending acceptance accounting shrinks to the prefix the
decoder will actually verify.

```zig
pub const Options = struct {
    max_draft: usize = 16,
    min_draft: usize = 2,           // shorter drafts fall back to a plain step
    enabled: bool = true,
    topk_feedback: usize = 8,       // candidates per verified position
    stop_token: ?usize = null,
    pin_kernels: bool = true,       // verify forward under pinned rowwise kernels: m == 1 numerics bitwise
    // cost-aware auto-off gate:
    rate_window: usize = 16, min_window_drafted: usize = 8,
    min_speedup: f32 = 1.0, probe_margin: f32 = 0.10,
    reprobe_after: usize = 128, reprobe_max: usize = 1024, probe_steps: usize = 4,
    cost_table: []const CostPoint = &default_cost_table,
    adapt_budget: bool = true,
};
pub const CostPoint = struct { draft_len: usize, cost: f32 };
pub const default_cost_table: [4]CostPoint;   // measured Qwen3-0.6B-Q4_K_S economics
pub fn tableCost(table: []const CostPoint, draft_len: usize) f32;
```

`CostGate` (public type; driven internally by the decoder, `estSpeedup()` is
its public read) estimates the rolling true speedup — committed tokens per
plain-step-equivalent of verify cost — over a window of verify steps, turns
speculation off when it drops below `min_speedup`, re-probes with exponential
backoff (`reprobe_after` → `reprobe_max`) and `probe_margin` hysteresis, and
adapts the draft budget to the rolling acceptance rate. With a clock
(`io` set on the decoder), measured verify/plain ratios continuously rescale
the static `cost_table` through a clamped EWMA; without one, the table
applies as-is. Gating decides WHEN speculation runs, never WHAT is committed.

```zig
pub const Stats = struct {
    steps, spec_steps, fallback_steps, disabled_steps,
    drafted, accepted, rejected_steps, bonus, committed: usize,
    pub fn tokensPerStep(self) f64 / acceptanceRate(self) f64
    pub fn writeSummary(self, writer: *std.Io.Writer) !void
};
pub const TokenSink = struct { ptr: *anyopaque, func: *const fn (ptr, token: usize) anyerror!void, pub fn emit(...) };
pub const VerifyRowHook = struct { ... };  // test/debug: every pre-penalty logits row

pub fn SpeculativeDecoder(comptime Model: type) type {
    // fields: source, options, stats, gate, io: ?std.Io = null, on_verify_row
    pub fn init(allocator, source: DraftSource, options: Options) !Self
    pub fn deinit(self: *Self) void
    pub fn step(self, ctx, model: decoder.ModelPtr(Model), kv: *Model.Cache,
                sampler: *Sampler, history: *std.ArrayList(usize), sink: TokenSink) !usize
    pub fn bootstrapStep(self, ctx, kv: *const Model.Cache, sampler: *Sampler,
                         history: *std.ArrayList(usize), sink: TokenSink, logits: *Logits) !usize
}
```

- `init` validates the configuration loudly instead of leaving ReleaseFast UB:
  a top-K-consuming source with `topk_feedback == 0` is
  `error.TopKFeedbackDisabled`; degenerate gate options are
  `error.RateWindowTooSmall` / `error.ProbeStepsZero` / `error.CostTableEmpty`
  / `error.ReprobeAfterZero`.
- `step` runs one decode iteration under the invariant
  `history.items.len == kv.len() + 1` (every committed token in `history`, the
  last one not yet forwarded into the cache); a violated invariant is
  `error.InvalidDecodeState` at runtime. `history` must be allocated with
  `ctx.allocator` — the decoder appends committed tokens to it. Each committed
  token is emitted through `sink`; returns the number committed (≥ 1). Verify
  passes run one batched `forwardStepAllLogits` over `[carried token,
  draft...]`, and `kv.truncate` drops rejected rows — on error-unwind paths
  too (`errdefer` restores the invariant).
- `bootstrapStep` commits ONE token from caller-computed logits — the
  prefill bootstrap. The caller prefills the whole pending span in one
  batch (the byte-identity contract's caller leg), leaving the cache
  flush with history (`history.len == kv.len()`, else
  `error.InvalidDecodeState`) and the span's last-row logits in hand; the
  entry samples them through the exact plain-step machinery (sampler,
  history, sink, observe hook, stats) and restores the `step` invariant.
  The chat layer's speculative turn opens with it, so plain and
  speculative turns build their caches from call-for-call identical
  forwards.
- `Model` satisfies the decoder contract (`models.decoder.assertDecoder`) with
  `caps.rewind`: `forwardStep` + `forwardStepAllLogits` + a truncating
  cache (qwen3, gemma4, SHINE's `AdaptedModel`, glm4moe through
  `speculative.mtp.MtpDraftSource`; qwen35's recurrent cache cannot rewind
  and is out of scope).

### 13.9.2 Suffix-automaton index (`speculative/sam_index.zig`)

`SamIndex` is an online suffix automaton over a token stream — the
exact-match draft source (SAM-Decoding/SuffixDecoding lineage). It gives O(1)
amortized online extension, an exact self-match-excluded longest-suffix-match
length (drafts follow the most recent **prior** occurrence, never the current
one), and doubles as a frozen index over reference documents.

```zig
pub const SamIndex = struct {
    pub const max_stream_len: usize = 1 << 29;
    min_match: usize = 2, max_draft: usize = 16,     // policy fields
    pub fn init(allocator: Allocator) !SamIndex / deinit(self) void
    pub fn append(self, new_tokens: []const usize) !void   // O(1) amortized per token
    pub fn matchLen(self) usize
    pub fn draft(self, buf: []usize) usize
    pub fn tokenCount/stateCount/transitionCount(self) usize
    // frozen mode (RAG):
    pub fn freeze(self) void
    pub const Cursor = struct { state: u32 = 0, len: u32 = 0 };
    pub fn advance(self, cursor: *Cursor, token: usize) void
    pub fn draftFrom(self, cursor: Cursor, buf: []usize) usize
    // DraftSource method shapes:
    pub fn suggest(self, context, buf) usize / observe(self, committed) void / observeTopK(...)
};
pub const FrozenSource = struct { index: *const SamIndex, cursor: Cursor,
                                  pub fn suggest/observe/observeTopK };
```

A failed `append` **poisons** the index (`degraded`): all queries return 0
forever and further appends fail — a half-applied append must never serve
drafts (`observe` swallows errors into this degradation instead of
propagating). `freeze()` ends appends and makes external `Cursor`s safe
(appends can split states; only the internal cursor gets the clone fix-up).
`FrozenSource` owns a per-conversation cursor over one frozen document and
borrows the index.

```zig
test "SamIndex: longest self-excluded suffix match drives the draft" {
    const alloc = std.testing.allocator;
    var sam = try models.text.speculative.sam_index.SamIndex.init(alloc);
    defer sam.deinit();
    try sam.append(&.{ 5, 6, 7, 5, 6 });
    // Longest suffix with an occurrence ending strictly before the end: [5,6].
    try std.testing.expectEqual(@as(usize, 2), sam.matchLen());
    var buf: [8]usize = undefined;
    const n = sam.draft(&buf); // tokens after the prior occurrence: {7,5,6}
    try std.testing.expectEqualSlices(usize, &.{ 7, 5, 6 }, buf[0..n]);
}
```

### 13.9.3 Token recycling (`speculative/recycling.zig`)

```zig
pub fn TokenRecycling(comptime K: usize) type {
    pub const k = K;
    pub const sentinel: u32 = std.math.maxInt(u32);
    pub fn init(allocator, vocab: usize) !Self / deinit(self) void
    pub fn topkOf(self, token: usize) []const u32
    pub fn update(self, token: usize, topk: []const u32) void
    pub fn draftChain(self, last_token: usize, buf: []usize) usize
    pub fn suggest/observe/observeTopK   // DraftSource method shapes
}
pub const Recycling = TokenRecycling(8);
```

A `vocab × K` adjacency matrix (Token Recycling, Luo et al. 2024): row `t`
holds the most recent top-K next-token candidates observed at any verified
position whose input token was `t` (≈4.6 MiB for the Qwen3 vocab at K=8).
Drafting walks the top-1 chain until an unseen row or the budget stops it.
`observe` promotes each committed bigram to slot 0 (ground truth beats stale
logits); `observeTopK` overwrites whole rows from verification feedback. The
struct owns `m`; `update`/`observe*` copy, no slice is retained.

```zig
test "TokenRecycling: top-1 chain drafting" {
    const alloc = std.testing.allocator;
    var rec = try models.text.speculative.recycling.Recycling.init(alloc, 32); // vocab 32, K = 8
    defer rec.deinit();
    rec.update(3, &.{ 7, 9 }); // most recent top-K observed after token 3
    rec.update(7, &.{5});
    var buf: [4]usize = undefined;
    try std.testing.expectEqual(@as(usize, 2), rec.draftChain(3, &buf)); // 3 -> 7 -> 5, then unseen
    try std.testing.expectEqualSlices(usize, &.{ 7, 5 }, buf[0..2]);
}
```

### 13.9.4 Cascade (`speculative/cascade.zig`)

`SpeculationIndex` is the user-facing orchestrator behind one `DraftSource`:
it composes (1) an online conversation `SamIndex` over every committed token,
(2) any number of frozen reference `SamIndex` documents injected via
`addReference` (the RAG seam), and (3) the `Recycling` matrix as the
self-draft fallback and `observeTopK` consumer.

```zig
pub const gate_window: usize = 64;
pub const Gate = struct { ..., pub fn muted(self) bool };   // per-source rolling acceptance
pub const FrozenRef = struct { index: SamIndex, cursor: SamIndex.Cursor, gate: Gate };

pub const SpeculationIndex = struct {
    // policy fields (defaults): beta = 2, min_match = 2, recycling_chain = 8,
    // mute_acceptance = 0.20, mute_commits = 128, accounting_min_draft = 2
    pub fn init(allocator: Allocator, vocab: usize) !SpeculationIndex / deinit(self) void
    pub fn addReference(self, tokens: []const usize) !void
    pub fn clearReferences(self) void
    pub fn suggest(self, context, buf) usize / observe(self, committed) void / observeTopK(...)
    pub fn asDraftSource(self: *SpeculationIndex) DraftSource
    pub fn truncatePending(self, new_len: usize) void
    pub fn writeSourceSummary(self, writer: *std.Io.Writer) !void
};
```

Suggest policy: the source with the longest current match wins (ties break
toward the conversation; among references, first-added), with draft budget
`min(buf.len, beta * (1 + match_len))`; matches shorter than `min_match` fall
back to the recycling top-1 chain (≤ `recycling_chain` tokens); an unseen
recycling row drafts 0 and the decoder takes a plain step. Each source keeps
a rolling acceptance gate over its last `gate_window` drafted tokens: below
`mute_acceptance` it is muted for `mute_commits` committed tokens, then
re-probed — muted sources keep observing so they stay in sync. Acceptance is
settled at the `suggest`→`observe` seam (longest common prefix of draft and
next committed slice); `accounting_min_draft` mirrors the decoder's
`min_draft` so unverified short drafts never skew the gates.
`addReference` catches the new document's cursor up over the already-observed
stream, so a mid-conversation injection can match existing context
immediately.

```zig
test "SpeculationIndex: observe committed tokens, suggest a draft" {
    const alloc = std.testing.allocator;
    var index = try models.text.speculative.cascade.SpeculationIndex.init(alloc, 1024);
    defer index.deinit();

    index.observe(&.{ 1, 2, 3, 4, 1, 2 }); // [1,2] recurred; earlier it was followed by [3,4]
    var buf: [16]usize = undefined;
    const n = index.suggest(&.{ 1, 2, 3, 4, 1, 2 }, &buf);
    try std.testing.expect(n >= 2);
    try std.testing.expectEqualSlices(usize, &.{ 3, 4 }, buf[0..2]);

    try index.addReference(&.{ 7, 8, 9, 10 }); // RAG seam: frozen document index
    const source = index.asDraftSource(); // hand this to SpeculativeDecoder
    try std.testing.expect(source.wantsTopK()); // the recycling matrix consumes logits feedback
}
```

### 13.9.5 Enabling speculation in a runner

The turnkey path is `chat.Options{ .speculation = true }` ([§13.8](13-the-model-stack-fucina_models.md#138-chat-srcmodelstextchatzig)), which owns
the index/decoder lifecycle, stop-token wiring, and turn trimming. A custom
runner drives the decoder directly:

```zig
fn snippetDecoderLoop(
    ctx: *fucina.ExecContext,
    model: *const models.qwen3.model.Model,
    kv: *models.text.kv_cache.KvCache,
    index: *models.text.speculative.cascade.SpeculationIndex,
    history: *std.ArrayList(usize),
    sink: models.text.speculative.core.TokenSink,
) !void {
    const Decoder = models.text.speculative.core.SpeculativeDecoder(models.qwen3.model.Model);
    var decoder = try Decoder.init(ctx.allocator, index.asDraftSource(), .{ .max_draft = 16 });
    defer decoder.deinit();
    var sampler = models.text.sampler.Sampler.init(.{});
    // Invariant: history.len == kv.len() + 1 (last committed token not yet forwarded).
    while (history.items.len < 128) {
        _ = try decoder.step(ctx, model, kv, &sampler, history, sink);
    }
} // requires model assets to run
```

### 13.9.6 Grammar-constrained drafting (`speculative/constrained.zig`)

`ConstrainedSource` makes a grammar constraint *accelerate* speculation
instead of muting it ([CONSTRAINED-DECODING.md](../CONSTRAINED-DECODING.md) [§5](05-automatic-differentiation.md)
is the design record). It wraps any inner `DraftSource` with a
`LogitProcessor` that exposes the [§13.6](13-the-model-stack-fucina_models.md#136-sampling-srcmodelstextsamplerzig) structural hooks
(`hasStructure()`), and must sit on the same processor instance installed
on the stream's sampler:

```zig
pub const ConstrainedSource = struct {
    pub fn init(processor: LogitProcessor, inner: DraftSource) ConstrainedSource
    pub fn source(self: *ConstrainedSource) DraftSource
};
```

- **Forced spans draft themselves.** When the grammar mandates a unique
  continuation (JSON structure, a forced literal), `suggest` proposes it
  directly — and because the sampler's mask allows exactly that token at
  each row, the whole span verifies with acceptance probability 1.
- **Certainly-rejected drafts die early.** Otherwise the inner source
  proposes and the draft is truncated at its first grammar-invalid token
  (`validPrefixLen`) — those tokens would be masked to `-inf` at their
  verify row, so proposing them only wastes verify compute and drags the
  inner source's acceptance gates down. The truncation is forwarded through
  `DraftSource.truncatePending` so the inner accounting matches what is
  actually verified.

Losslessness is untouched (drafts never decide WHAT is committed, and both
hooks are deterministic lookaheads); the constrained speculative stream is
token-for-token identical to the constrained plain stream, proven greedy +
sampled in `chat_tests.zig`. Wiring is automatic everywhere: with
`chat.Options{ .speculation = true, .logit_processor = ... }` the
conversation wraps its cascade in a `ConstrainedSource` whenever the
processor has structure, and the qwen3 runner does the same for
`--spec` + a grammar flag. Measured effect on Qwen3-0.6B-Q8_0 JSON-schema
chat: draft acceptance 0% → 83%, with the cost gate staying on ([§13.6](13-the-model-stack-fucina_models.md#136-sampling-srcmodelstextsamplerzig)).

```zig
test "constrained source: forced spans preempt, invalid drafts truncate" {
    const Fixed = struct {
        // state = 1: the grammar forces {7, 8}; state = 0: free choice with
        // ids >= 3 invalid. process/commit are irrelevant on the suggest side.
        fn process(_: *anyopaque, _: []f32, _: []const usize) anyerror!void {}
        fn commit(_: *anyopaque, _: usize) anyerror!void {}
        fn forcedTokens(ptr: *anyopaque, buf: []usize) usize {
            const forcing: *u8 = @ptrCast(ptr);
            if (forcing.* == 0) return 0;
            buf[0] = 7;
            buf[1] = 8;
            return 2;
        }
        fn validPrefixLen(_: *anyopaque, tokens: []const usize) usize {
            for (tokens, 0..) |t, i| if (t >= 3) return i;
            return tokens.len;
        }
        fn suggest(_: *anyopaque, _: []const usize, buf: []usize) usize {
            buf[0] = 1; // an inner source that always drafts {1, 5}
            buf[1] = 5;
            return 2;
        }
        fn observe(_: *anyopaque, _: []const usize) void {}
    };
    var forcing: u8 = 1;
    const processor = models.text.logit_processor.LogitProcessor{ .ptr = &forcing, .vtable = &.{
        .process = Fixed.process,
        .commit = Fixed.commit,
        .forcedTokens = Fixed.forcedTokens,
        .validPrefixLen = Fixed.validPrefixLen,
    } };
    const inner = models.text.speculative.core.DraftSource{ .ptr = &forcing, .vtable = &.{
        .suggest = Fixed.suggest,
        .observe = Fixed.observe,
    } };

    var cs = models.text.speculative.constrained.ConstrainedSource.init(processor, inner);
    var buf: [8]usize = undefined;
    // Forced state: the grammar span wins over the inner source.
    try std.testing.expectEqual(@as(usize, 2), cs.source().suggest(&.{0}, &buf));
    try std.testing.expectEqualSlices(usize, &.{ 7, 8 }, buf[0..2]);
    // Free choice: the inner draft {1, 5} truncates at the invalid 5.
    forcing = 0;
    try std.testing.expectEqual(@as(usize, 1), cs.source().suggest(&.{0}, &buf));
    try std.testing.expectEqual(@as(usize, 1), buf[0]);
}
```

### 13.9.7 Native-MTP drafting (`speculative/mtp.zig`)

`MtpDraftSource(Model)` puts a family's native MTP (`nextn`) head behind
the `DraftSource` vtable, so the shared decoder's verify loop drives it
instead of a hand-rolled draft/verify/commit/rewind loop. Generic over a
glm4moe-shaped model: `mtpDraftStep(self, ctx, mtp_cache, token, h_prev,
h_out)`, `initMtpCache(self, capacity)`, and a model-owned `step_hiddens`
row buffer; requires `Model.caps.rewind`. `init(ctx, model, capacity,
depth)` builds the MTP stream's own cache; `observePrefill()` seeds the
prompt's trunk hiddens after the caller's prefill forward; `suggest`
catches the MTP stream up on committed positions (position `i` consumes
`(token[i+1], hidden[i])`), chains `depth` argmax drafts from the
frontier, and rewinds the speculative MTP positions; `observe` appends the
verify rows the decoder's truncate keeps. A draft-round error lands in the
`err` field (the round proposes nothing; the decoder takes a plain step).
`examples/glm4moe --mtp` decodes through it. deepseek4's MTP sidecar stays
on its own loop: its `Session` rewinds by snapshot/restore, not
`truncate`.

## 13.10 Cartridges (`src/models/text/cartridge.zig`)

```zig
pub const Kv = fucina.Tensor(.{ .seq, .kv_head, .d });          // KV-cache row layout
pub const LayerKv = struct { k_sink: ?Kv, v_sink: ?Kv, k: Kv, v: Kv };
pub const Cartridge = struct { layers: []LayerKv, p: usize, frozen_prefix: usize, ... };
pub const DistillTargets = struct { positions: []const usize, tokens: []const usize, logprobs: []const f32 };
pub const Error = error{ InvalidCartridge, InvalidTargets, ExecScopeRequired };
```

A **cartridge** (arXiv 2506.06266, HazyResearch/cartridges semantics)
compresses a corpus into the KV cache of a virtual p-token prefix: per layer,
a `[p, kv_head, d]` K/V pair living in the exact space of KV-cache rows —
keys post q/k-norm and post-RoPE at positions `0..p-1`, never re-rotated —
trained offline by self-study distillation and served as a reusable prefix at
a fraction of the ICL cache size. Row 0 is a frozen constant by default (the
paper's attention-sink freeze; training it destabilizes the run), the rest
are leaf variables. Attention over `concat(cartridge, tokens)` with the
end-aligned causal kernel (`source_offset = kv_seq − q_seq`, [§4.13](04-tensor-operations.md#413-attention-srcagtensorzig))
reproduces the reference mask exactly: every query sees the whole prefix and
is causal over the real tokens, which sit at RoPE positions `p..`.

Construction and lifecycle (create OUTSIDE any exec scope, like LoRA A/B):

- `Cartridge.initFromRows(ctx, allocator, frozen_prefix, p, kv_heads, head_dim, k_rows, v_rows)`
  — from captured per-layer rows (the trainers' `captureKv` /
  `initCartridge` produce them; the paper's winning "first p corpus tokens"
  initialization). `initFromRowsVaried` takes PER-LAYER kv_heads/head_dims
  (heterogeneous geometries like gemma-4's mixed SWA/global shapes;
  `initFromStateDict` recovers per-layer shapes from the header).
  `initRandom(...)` is the random-vector ablation baseline.
- `registerParams(opt)` — trainable rows onto any optimizer with
  `addParamNamed` (sinks are frozen registry entries and are skipped);
  `zeroGrad()`.
- `saveState(writer)` / `loadState(reader)` — safetensors state dict under
  `layers.<i>.{k,v}[_sink]` names (strict name+shape match on load);
  `initFromStateDict(ctx, allocator, bytes)` rebuilds a cartridge from the
  bytes alone (geometry recovered from the safetensors header).
- `setDraftReference(ctx, tokens)` — embed the corpus token ids in the
  artifact (frozen i64 `draft_reference` entry; set-once, before
  `saveState`): the serving-side speculation reference, so `--spec-serve`
  builds the corpus suffix automaton ONCE at cartridge load with no
  `--corpus` re-read. `initFromStateDict` recovers it into
  `cart.draft_reference: ?[]usize`; artifacts without the entry load
  unchanged.
- `LayerKv.catK/catV(ctx, tokens)` — `concat(sink?, trainable, tokens)`
  along `.seq`, the per-layer attention input; `fullK/fullV(ctx)` — the
  serving payload (sink ++ trainable, no tokens).
- `writeToCache(ctx, cache)` — serve: fill an EMPTY `KvCache` with all p
  rows (converted to the cache dtype) and advance it to p, so a normal
  `forwardStep` decode continues at position p, the training-time layout.

`distillLoss(ctx, logits, targets, options)` is the reference training
objective: the teacher top-k cross-entropy
`mean(-exp(logprob_i) · log_softmax(logits)[positions_i − 1, tokens_i])` over
sparse `(position, token, logprob)` entries — gradient-identical to forward
KL(teacher ‖ student) since the teacher entropy is constant. `positions[i]`
is the packed index of the TARGET token (the student's prediction is read
from the previous row); truncated teacher tail mass is dropped, NOT
renormalized; entries are averaged uniformly (`.sum` + `loss_scale` compose
with gradient accumulation, [§11](11-training-optimizers-evolution-strategies-lora-and-checkpoints.md)). Composite-op contract: MUST run inside an
open exec scope (`Error.ExecScopeRequired`). `TargetsBuilder.appendRow`
extracts targets from raw teacher logits rows host-side (descending top-k
until `min_prob_mass` cumulative probability, the crossing entry included);
`appendTopKRow` is its tensor-side counterpart, fed by core
`topK`/`logsumexp` over the vocab axis so only `[rows, k]` values/indices
reach the host — selection identical (lowest-index ties both ways),
logprobs equal up to the core reduction's summation order (pinned by a
unit test).

The qwen3 trainer hosts the training loop ([§14](14-model-families-and-example-applications.md) / `models/qwen3/train.zig`):
`ForwardOptions.cartridge` threads the prefix through every layer (tokens
shift to RoPE positions `p..`; gradients flow into the cartridge rows through
the frozen stack), `ForwardOptions.capture` copies token K/V rows out of a
forward, `ForwardOptions.packed_segments` batches several independent
sequences as contiguous segments of ONE forward (the GEMMs pack; RoPE
restarts per segment via a transient table — pair with
`Trainer.freeTransientRope()` between optimizer steps; attention runs per
segment over zero-copy narrows so gradients keep flowing through the fused
backward and accumulate into shared leaves — packed vs sequential gradient
equality is pinned by a trainer test), and `Trainer.{initCartridge,
captureKv, distillLoss, evalLogitsExt, evalLogitsRows}` are the
training/eval entries — instantiate `Trainer(.{ .q = false, .v = false })`
so the cartridge rows are the only parameters. Capture, packed, and
composed forwards are plain-path only
(`Error.CartridgeCheckpointUnsupported` under `checkpoint_layers`); a
SINGLE cartridge on a no-adapter trainer checkpoints — its rows ride as
block inputs and the recompute is pinned bitwise (loss + row gradients)
by a trainer test. The
`cartridge` example (`zig build cartridge`,
[README](../../examples/cartridge/README.md)) runs the whole flow on a real
GGUF: `--equiv` (a zero-training corpus-init cartridge must match the real
prefill — bitwise at tiled-attention shapes on Qwen3-0.6B-f16), self-study
training (paper Sec 4, k = 1, fully in-process), and `--load`/`--ask`
serving. Design record: `docs/CARTRIDGES.md`.

**Composition** (Cartridges at Scale, arXiv 2606.04557): independently
trained cartridges compose by concatenation — part 0's rows, then part 1's,
…, with real tokens at RoPE positions `composedP(parts)..`; every part
keeps the rotations it was trained at (post-RoPE keys are frozen vectors,
so overlapping nominal positions across parts are fine). `cartridge.zig`
provides the free functions:

```zig
pub fn composedP(parts: []const *const Cartridge) usize;
pub fn validateComposition(parts: []const *const Cartridge) Error!void;   // layer-for-layer geometry match
pub fn composedCatK(ctx, parts, layer_i: usize, k_tokens: *const Kv) !Kv; // concat(sink?, k, sink?, k, ..., tokens)
pub fn composedCatV(ctx, parts, layer_i: usize, v_tokens: *const Kv) !Kv;
pub fn writeComposedToCache(ctx, parts, cache: *KvCache) !void;           // serve: parts in order into an EMPTY cache
// Cartridge.appendToCache(ctx, cache) — writeToCache without the empty-cache
// requirement, the primitive writeComposedToCache chains.
```

The qwen3 and gemma4 trainers thread a composition through
`ForwardOptions.cartridges`
(mutually exclusive with `cartridge`; plain-path only): ONE concat per layer,
so the existing fused attention backward routes gradients into EVERY part's
trainable rows — the joint-training seam. `Trainer.distillLossExt(ctx,
tokens, fwd, targets, options)` is `distillLoss` with full `ForwardOptions`
(composition + packing). A single-part composition is op-for-op the
single-cartridge forward (pinned bitwise), a two-part composition built
from one capture reproduces the real prefill exactly (the composition
oracle — bitwise on Qwen3-0.6B-f16 at p = 256 via `cartridge-fleet
--equiv`; gemma4 arms, SWA cutting the composed prefix included, in
`train_tests.zig`), and serving through `writeComposedToCache` is
cache-level, so compositions serve on ANY family
(`train_cartridge_compose_tests.zig`).

**Fleets** (`src/models/text/cartridge_fleet.zig`): the scale layer around
composition — one cartridge per document instead of one monolith per
corpus. `Manifest` is the fleet's on-disk record (`fleet.json`: per-doc
cartridge/optimizer files, token counts, optimizer-step counters); `Fleet`
is the RAM/disk budget manager (at most `policy.budget` cartridges
resident, each with its own AdamW whose moments travel through
evict/reload — an evict/reload cycle continues training bit-identically,
pinned by a fleet test; every `policy.every` rounds the most-trained
residents rotate to disk and the least-trained absentees rotate in, with a
per-cartridge lr warm-up on entry); `EmbedIndex` is the cartridge-RAG
selector — L2-normalized chunk embeddings centered by their centroid at
`finalize` (the all-but-the-top correction; raw causal-LM embeddings are
anisotropic and mis-rank documents without it), hand-rolled cosine top-k
in `topDocs`, persisted as `index.safetensors`. Artifact retrieval goes
through `mmapFile` (read-only page-backed mappings, no whole-file heap
copy). The `cartridge-fleet` example (`zig build cartridge-fleet`,
[README](../../examples/cartridge_fleet/README.md)) drives
mixed-visibility joint self-study (each round targets one resident
document; with probability `--p-iso` its cartridge trains alone, otherwise
distractor cartridges from other residents co-load in shuffled order — the
paper's recipe against composition collapse), builds the retrieval index
through the model itself (topic-instruction-suffixed final-norm last
hidden state, `Trainer.embedLastHidden` + `embed_suffix`), and serves
`--ask` by cosine selection over chunks → document cartridges →
`writeComposedToCache` → decode; qwen3 and gemma4 GGUFs (gemma uses the
flat per-conversation backward). lmserve serves fleets over HTTP
(`--fleet DIR`, per-request selection with conversation-sticky slot
reuse — [LMSERVER.md](../LMSERVER.md)).

```zig
test "cartridge: trainable KV prefix + teacher top-k distillation" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    // One layer, p = 2 rows: a frozen sink + one trainable row.
    var cart = try models.text.cartridge.Cartridge.initRandom(&ctx, alloc, 1, 1, 2, 1, 2, 42, 0.1);
    defer cart.deinit();
    try std.testing.expect(!cart.layers[0].k_sink.?.requiresGrad());
    try std.testing.expect(cart.layers[0].k.requiresGrad());

    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);

    // kv_seq = p + 2 > q_seq = 2: every query attends the whole prefix,
    // then causally over the tokens (end-aligned kernel, §4.13).
    var q = try fucina.Tensor(.{ .seq, .head, .d }).fromSlice(&ctx, .{ 2, 2, 2 }, &.{ 0.2, -0.4, 0.5, 0.1, -0.3, 0.7, 0.05, -0.6 });
    defer q.deinit();
    var k_tok = try models.text.cartridge.Kv.fromSlice(&ctx, .{ 2, 1, 2 }, &.{ 0.1, 0.5, -0.35, 0.2 });
    defer k_tok.deinit();
    var v_tok = try models.text.cartridge.Kv.fromSlice(&ctx, .{ 2, 1, 2 }, &.{ -0.2, 0.4, 0.55, -0.5 });
    defer v_tok.deinit();
    var k_cat = try cart.layers[0].catK(&ctx, &k_tok);
    defer k_cat.deinit();
    var v_cat = try cart.layers[0].catV(&ctx, &v_tok);
    defer v_cat.deinit();
    const map = [_]usize{ 0, 0 };
    var attn = try q.groupedAttention(&ctx, &k_cat, &v_cat, map[0..], .attn, 0.7, .{});
    defer attn.deinit();

    // Stand-in logits: one full-mass teacher entry == plain cross-entropy.
    var logits = try attn.withTags(&ctx, .{ .seq, .vocab });
    defer logits.deinit();
    var loss = try models.text.cartridge.distillLoss(&ctx, &logits, .{
        .positions = &.{1},
        .tokens = &.{2},
        .logprobs = &.{0.0},
    }, .{});
    defer loss.deinit();
    try loss.backward(&ctx);

    // The gradient reached the trainable row through concat + attention.
    var grad = (try cart.layers[0].k.grad(&ctx)).?;
    defer grad.deinit();
    try std.testing.expectEqual(@as(usize, 2), grad.asRawTensor().dataConst().len);
}
```

## 13.11 Engram (`src/models/research/engram.zig`)

```zig
pub const Config = struct { hidden_size: usize, hc_mult: usize = 1, max_ngram_size: usize = 3, ... };
pub const HashPlan = struct { multipliers: []i64, head_mods: []i64, head_offsets: []usize, table_rows: []usize, ... };
pub const Layer = struct { table: Table, key_w: []Proj, value_w: Proj, conv_w: ConvKernel, ... };
pub const Engram = struct { plan: HashPlan, layers: []Layer, registry: fucina.ParamRegistry, ... };
pub const Hidden = fucina.Tensor(.{ .seq, .stream, .d });
pub const Error = error{ InvalidConfig, InvalidHashInput, ExecScopeRequired };
```

**Engram** (arXiv 2601.07372, deepseek-ai/Engram demo semantics, pinned in
`tools/fetch_refs.sh`) is conditional n-gram memory: capacity bought with
LOOKUP instead of FLOPs. Per selected layer, suffix n-grams of the
(compressed) token ids are hashed by multiplicative-XOR heads into
prime-sized embedding tables; the retrieved rows, gated per
hyper-connection stream by an RMS-normed key·query dot with a signed-sqrt
squash, join the residual stream after a dilated causal depthwise
ShortConv ([§4.16](04-tensor-operations.md#416-selection-argmax-topk-sort-routertopk-srcagtensorzig-srcexectopkzig)). Because every table address is a pure function of token
ids, all lookups are known BEFORE the forward pass — tables can live
out-of-core and be prefetched with zero speculation.

The split of responsibilities:

- `HashPlan` — pure integer geometry, no tensors: per-layer odd
  multipliers (native draws, or injected via `initWithMultipliers` for
  bit-parity with reference artifacts; persisted with the checkpoint
  either way), the global-seen-set prime chain for head table sizes, the
  optional tokenizer-compression `lookup`, and the hash itself.
  `hashInto` is the host fast path; `hashTensor` is the same computation
  as integer tensor ops (wrapping `mul`, `bitXor`, floored `mod`,
  broadcast `add` — [§4.19](04-tensor-operations.md#419-math-on-non-f32-tensors-srcagtensorzig)), pinned bit-equal in tests. Row indices come
  back offsets-included, ready for `gather` along the table's row axis.
- `Layer` — the parameters and the differentiable forward
  (`[seq, stream, d]` in and out; the caller adds the residual).
  `forwardResidual` wraps the `hc_mult == 1` case for plain
  residual-stream models. `conv_state` streams the ShortConv across
  chunks exactly like `causalDepthwiseConv1d`'s state ([§4.16](04-tensor-operations.md#416-selection-argmax-topk-sort-routertopk-srcagtensorzig-srcexectopkzig)).
- `Engram` — the whole-model wrapper: shared plan, one `Layer` per
  `layer_ids` entry, every parameter in a `ParamRegistry`
  (`engram.layers.<id>.*`; the multipliers as a frozen i64 entry), state
  dict save/load.

`InitOptions.graft_zero_init` zero-initializes the value projection so
the module's output is EXACTLY zero: grafting Engram onto a frozen
pretrained model is bitwise identity at step 0, while gradients still
reach every parameter through the value path — the cheap-experiment mode
for adding memory to an existing checkpoint. Numerical parity with the
reference mechanism (forward + every gradient) is pinned by
`src/models/research/engram_golden_tests.zig`, generated by
`tools/gen_engram_goldens.py` (an independent PyTorch/numpy
implementation; integer geometry compares EXACTLY, floats under the
shared golden tolerance). Design record: `docs/ENGRAM.md`.

The qwen3 trainer carries the graft seam: `engram.ResidualGraft{ .model,
.rows }` adapts the module to `ForwardOptions.residual_hook`
(`fwd.residual_hook = adapter.hook()`), injecting each configured layer's
memory output into the residual stream before attention (plain path only;
composes with `cartridge`; rejected with `packed_segments` — the ShortConv
is causal over the packed row), and `lossForwardExt(ctx, tokens, labels,
fwd, loss_opts)` is the CE loss entry taking full `ForwardOptions`.
`examples/engram/main.zig` (`zig build engram`,
[README](../../examples/engram/README.md)) drives it: `--equiv`
(bitwise zero-init gate on a real GGUF), `--train`/`--eval` (frozen
trunk, held-out chunk CE, `--lora N`, `--no-engram` control,
`--gate-bias F`), `--probes N` (verbatim-recall spans, teacher-forced
CE + exact-match, engram detached vs attached).

```zig
test "engram: hashed n-gram memory with a zero-init graft gate" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const cfg = models.research.engram.Config{
        .hidden_size = 8,
        .hc_mult = 1,
        .n_embed_per_ngram = 4,
        .n_head_per_ngram = 2,
        .engram_vocab_size = &.{ 11, 13 },
        .kernel_size = 2,
        .pad_id = 0,
    };
    var plan = try models.research.engram.HashPlan.init(alloc, cfg, &.{0}, 42, null);
    defer plan.deinit();

    // Addresses are a pure function of token ids — known pre-forward.
    const ids = [_]i64{ 3, 1, 4, 1, 5 };
    var rows: [5 * 4]usize = undefined;
    try plan.hashInto(0, &ids, &rows);
    for (rows) |row| try std.testing.expect(row < plan.table_rows[0]);

    var layer = try models.research.engram.Layer.initRandom(&ctx, alloc, cfg, plan.table_rows[0], 7, .{ .graft_zero_init = true });
    defer layer.deinit();

    var hidden = try fucina.Tensor(.{ .seq, .d }).fromSlice(&ctx, .{ 5, 8 }, &([_]f32{0.25} ** 40));
    defer hidden.deinit();
    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);

    var out = try layer.forwardResidual(&ctx, &hidden, &rows, null);
    defer out.deinit();
    // Zero-init value projection: grafting is exact identity at step 0.
    for (out.asRawTensor().dataConst()) |v| try std.testing.expectEqual(@as(f32, 0), v);
}
```

## 13.12 SHINE (`src/models/qwen3/shine.zig`)

```zig
pub const Config = struct { hidden_size: usize, num_mem_token: usize, lora_r: usize, scale: f32, ... };
pub const LoraPair = struct { a: fucina.Tensor(.{ .lin, .lora_r }), b: fucina.Tensor(.{ .lora_r, .lout }) };
pub const LayerLora = struct { q: LoraPair, k: LoraPair, v: LoraPair, o: LoraPair, gate: LoraPair, up: LoraPair, down: LoraPair };
pub const LoraSet = struct { allocator: Allocator, layers: []LayerLora };
pub const Shine = struct { config: Config, mem_tokens: ..., m2p: ..., metalora: LoraSet };
pub const Error = error{ InvalidShineConfig, UnsupportedKvDtype, MoeNotSupported };
```

**SHINE** (arXiv 2602.06358) is an in-context hypernetwork: one forward
pass over a context passage produces a rank-8 LoRA over all 7 linears of
every layer of a frozen dense Qwen3, and questions are then answered with
the adapter alone (no context tokens, no context KV). It is the amortized
counterpart of cartridges ([§13.10](13-the-model-stack-fucina_models.md#1310-cartridges-srcmodelstextcartridgezig)): the artifact is adapter weights
instead of a trained KV prefix, and producing it costs one prefill-priced
pass instead of a distillation run. The trade, measured in the paper and
reproduced here: extractive-QA quality sits below in-context prompting;
the win is per-request serving cost.

The pipeline, mirrored 1:1 from the reference implementation
(MuLabPKU/SHINE; released checkpoint `Yewei-Liu/SHINE-ift_mqa_1qa`, MIT,
backbone Qwen3-8B):

- `encodeMemoryStates(model, sh, ctx, token_ids)` — embed the context,
  append the 148 learned memory embeddings, run the base model once with
  the rank-128 Meta LoRA active on every linear (side path
  `base + (x @ A) @ B`, never merged), and capture the residual stream at
  the memory rows after each layer: `(layer, mem, embed)`. The
  memory-token count is the LoRA budget itself: `M · hidden` equals the
  per-layer generated-parameter count (`Config.validate` enforces it).
- `m2pForward(sh, ctx, memory_states)` — learned layer/token positional
  embeddings, then 4 post-LN transformer encoder layers (gelu-erf, 32
  heads, ff 8192, biased, LayerNorm eps 1e-5) alternating LAYER-mixing
  (attention along the 36-layer axis, one sequence per memory token) and
  TOKEN-mixing (attention along the 148-token axis, one sequence per
  layer). Both passes run fully batched through tag-aligned `dot`
  contractions; `m2pInput`/`m2pStage` expose the per-stage seam the
  golden tests walk.
- `sliceLora(ctx, base, r, scale, plain)` — scale by `sqrt(scale)` (both
  halves carry it, reference "rl" method), then cut each layer's
  flattened row into `A [in, r]` / `B [r, out]` pairs in module order
  q,k,v,o,gate,up,down.
- `forwardStep` / `greedy` — the decode loop with the generated adapter
  active on every linear (f16 KV cache); `GreedyOptions.vocab_limit`
  reproduces the reference's resized head (len(tokenizer) = 151,672 for
  the released run, below the GGUF's padded 151,936 rows).
- `saveLora` / `loadLoraFromFile` / `loadLoraGguf` — the adapter as a
  standalone `shine-adapter` GGUF artifact (`blk.<layer>.<module>.{a,b}`
  f32, ~87 MB at r=8 on the 8B): generate once, serve forever. The reload
  path needs neither the SHINE weights nor the hypernetwork pass and
  cross-checks the base geometry, so an adapter built on a 16-bit base
  serves on a quantized one (the LoRA side paths stay f32 regardless of
  the base format). Runner: `--shine-save PATH` on the generate side,
  `--shine-adapter PATH` to serve.

Artifacts and parity: `tools/convert_shine.py` packs the released
checkpoint (mem_tokens + metalora + metanetwork, 3.3 GB) into one f32
GGUF; `tools/gen_shine_goldens.py` dumps reference goldens from the
PyTorch implementation (fp32, transformers pinned to the 4.57 API
window). The model-gated golden tests measure both 16-bit bases when
present: on f16, encoder memory states 5.1e-3 max rel and answer logits
≤ 0.16 (kernel drift, not structure: the f16 GEMM path also rounds
activations to f16); on bf16, 2.8e-6 and ≤ 0.016 (the mixed f32 × bf16
kernel keeps the f32 activations). M2P stages sit ≤ 1e-6 rel (pure f32
both sides), adapter slices ≤ 1.2e-7, and greedy answers are
token-for-token equal to the reference on both golden turns on both
bases. Serving also works from a q8_0 base (adapter side paths stay
f32): correct golden answers with faster decode; bf16 is the accuracy
pick, f16/q8_0 the speed picks.

`zig build qwen3 -- <base.gguf> --shine <shine.gguf> --shine-context
TEXT|@FILE --chat "..."|--repl` is the direct serving surface (greedy,
no-think, matching the reference harness; see
[`examples/qwen3/README.md`](../../examples/qwen3/README.md)). Constraints:
dense Qwen3 backbones only, the base must match the checkpoint's
(`Config.validate` cross-checks dims), decode is f16-KV, and the released
run was trained on contexts up to ~1.1k tokens.

Adapter fleets close the serving loop: `--shine-docs DIR
--shine-fleet-build OUT` in the qwen3 runner compiles every .txt/.md
document into a saved adapter plus retrieval embeddings (the cartridge
fleets' `EmbedIndex`, same `embed_suffix` recipe, 256-token chunks), and
`zig build lmserve -- <base.gguf> --shine-fleet OUT` serves the
directory: each request's user messages pick ONE document by cosine
retrieval, and that document's adapter decodes the reply through
`AdaptedModel` — the frozen base plus a swappable adapter behind the
duck-typed model surface `chat.Conversation` consumes, swapped per
request by the single inference worker. Zero context tokens, zero prefix
rows; lmserve's slot reuse stays keyed by selection, so only
same-adapter KV is ever adopted or prefix-shared
([`examples/lmserve/README.md`](../../examples/lmserve/README.md)). The
library entry behind the flag is `models.qwen3.shine_serving.open` /
`openFromFile` ([§13.13](13-the-model-stack-fucina_models.md#1313-serving-srcmodelstextserving)), `serving.open`'s counterpart with the fleet
directory as an explicit argument.

Base-format guidance: prefer a bf16 or f32 base for adapter GENERATION —
the f16 GEMM path also rounds activations, and hypernetwork outputs can
be sensitive to that drift (the golden gates hold on both 16-bit formats
for the released 8B checkpoint, but sharper checkpoints have produced
degenerate adapters through the f16 path while bf16 stayed correct).
SERVING a saved adapter is robust on any base format, f16 and q8_0
included.

Training is native: `shine_train.ShineTrainer` runs the full
differentiable SHINE step — Meta-LoRA encoder pass with per-layer memory
capture, M2P, "rl" slicing, adapted conversation pass, masked CE — as one
autograd graph, with the frozen base routed through the LoRA trainer's
`FrozenCache` dots and the LoRA/M2P math reusing shine.zig's facade code
with gradient-carrying variables. `loss` takes one example;
`lossPacked` takes a batch and runs every base GEMM once over the
concatenated segments (per-segment RoPE, causal attention, and adapter
side paths decompose exactly; a pack of identical examples reproduces
the single step to 1.5e-6). `checkpoint_layers` runs each base layer as
one `fucina.checkpointWithContext` block ([§5.5](05-automatic-differentiation.md#55-activation-checkpointing-srcagcheckpointzig)'s component): only the
layer-boundary hiddens stay retained — the per-layer memory captures are
narrows of those boundaries, so capture is free — and the block's
grad-free first forward makes the packed GEMMs GPU-eligible. Rope tables
and segment vectors are then step-retained on the trainer; call
`freeTransient` between optimizer steps, never between a forward and its
backward. The golden gate (`tools/gen_shine_train_goldens.py`,
`models/shine/train-goldens`) pins loss and the gradient of every
trainable leaf against PyTorch autograd at 1e-3 (measured 1.8e-6), plain
and checkpointed.

**Cartridge readout** (`Config.cartridge_rows > 0`, a Fucina extension —
no reference implementation exists): the generator (encoder + M2P) is
readout-agnostic, and this mode reinterprets each layer's generated
`M x hidden` block as a KV PREFIX instead of LoRA pairs — `rows` K rows
then `rows` V rows in kv-cache row order, both scaled by `sqrt(scale)`,
with the budget identity `M * hidden == 2 * rows * kv_dim`
(`Config.validate`). The product is a STANDARD `models.text.cartridge.Cartridge`
([§13.10](13-the-model-stack-fucina_models.md#1310-cartridges-srcmodelstextcartridgezig)) in every respect: `generateCartridge` (or the runner's
`--shine-save-cartridge PATH`, given `--shine-context`) writes the same
safetensors state dict `zig build cartridge` produces, so serving,
fleets, composition, held-out evaluation, and the speculative draft
reference all apply unchanged — serve it with `--cartridge PATH`. SHINE
is thereby the amortized cartridge press: one prefill-priced pass where
distillation runs an optimization. Training mirrors the LoRA mode:
`ShineTrainer.lossCartridge` runs the same encoder + M2P over the same
trainable leaves, slices the block as graph views
(`sliceCartridgeViews`), and the conversation pass is the qwen3
trainer's own cartridge forward (`ForwardOptions.cartridge` with the
generated rows borrowed in), so RoPE offsets, prefix attention, and the
CE/label convention are byte-shared with distillation training. Gates
(`shine_cartridge_tests.zig`): budget-identity validation, bitwise
parity between the inference slicer and the trainer views, a
saveState -> initFromStateDict round trip, and finite-difference
gradient checks through the full chain (autograd vs secant agree to
~0.1% on the tiny harness). Zero-start note: at generation start the
prefix has near-uniform key logits and near-zero value content (the
`sqrt(scale)` discipline) — prefix-tuning's trainable regime, not the
LoRA mode's exact-identity start.

## 13.13 Serving (`src/models/text/serving/`)

`models.text.serving` is the complete serving stack: the model-agnostic contract,
the HTTP transport, the generic GGUF chat engine, and a load-and-serve
entry. `src/models/text/serving.zig` is the band index (contract names re-exported
flat, sub-modules namespaced); `examples/lmserve` ([§14.8](14-model-families-and-example-applications.md#148-example-applications)) is the CLI front
end built on it, and the voice agent hosts the same engine in-process.

**Contract** (`serving/contract.zig`, re-exported flat). `GenerateRequest`
carries the normalized message history (`models.text.chat.Message`), the fully
resolved `sampler.Config`, a bounded `max_tokens`, client stop strings, an
optional `ConstraintSpec` (JSON schema / regex / Lark), and the `think`
toggle. `GenerateResult` reports `prompt_tokens`, `completion_tokens`,
`cached_tokens` (KV rows reused from the cross-request cache), the fired
`stop_sequence` index, and `finish` (`.stop`/`.length`). `Caps` declares
what a backend can honor (the OpenAI layer rejects with 400 instead of
silently dropping fields); `Info` adds the model id, context budget,
`ThinkMarkers`, `ToolStyle` (`.hermes` is the Qwen3 convention), and the
default sampling. `RequestError` names the errors the dialect layers map
to specific HTTP statuses. The `Backend` vtable has `validate` (cheap,
connection thread), `generate` (worker thread, streams reply bytes to a
`*std.Io.Writer` sink), and optional `generate_batch` (lockstep decode;
null when the family has no batch forward).

**Transport.** `serving.http.Server` is the front end: accept loop,
per-connection threads (capped), socket deadlines, routing
(`POST /v1/chat/completions`, `POST /v1/responses`, `POST /v1/messages`,
`GET /v1/models`, `GET /health`), SSE plumbing through the cross-thread
`StreamPipe` (a stalled client stalls only its own connection thread,
never generation), the Host-header DNS-rebinding guard, and opt-in CORS.
It reads its work from `serving.scheduler.Scheduler`: a bounded FIFO in
front of one sequential inference worker (the engine's intended shape:
one `ExecContext`, single-threaded by contract), with lockstep batch
grouping when the scheduler is built with a batch width above 1.
`serving.emitter` frames deltas per wire dialect and routes
reasoning-block text away from the content channel; `serving.openai` and
`serving.anthropic` parse the three dialects into one normalized shape;
`serving.toolcall` renders hermes `<tool_call>` prompts and scans replies
for calls. On Linux, a binary that references `serving.http` links libc
(the `std.c.recv` client-hang-up probe); macOS links it implicitly.

**Engine** (`serving.gguf_chat`). `GgufChatBackend(Model, TokMod)` adapts
any family served through `models.text.chat.Conversation` (one comptime
instantiation per model/tokenizer pair): resident KV reuse slots with
token-LCP adoption and cross-slot prefix share, the `kv_persist` disk
tier ([§13.4](13-the-model-stack-fucina_models.md#134-kv-cache-srcmodelstextkv_cachezig)), the llguidance `ConstraintCache` (init once per distinct
grammar, clone per request), cartridge / cartridge-fleet / SHINE-fleet
serving hooks ([§13.10](13-the-model-stack-fucina_models.md#1310-cartridges-srcmodelstextcartridgezig), [§13.12](13-the-model-stack-fucina_models.md#1312-shine-srcmodelsqwen3shinezig)), and batch decode over
`Model.forwardStepBatch`. `kvRamVerdict`/`kvRamGuardSlots` implement the
startup RAM guard: slot caches commit lazily, so an overcommitted pool
would otherwise fail mid-serving as page-cache eviction, not at startup.

**Load-and-serve** (`serving/open.zig`).
`serving.open(ctx, io, allocator, gguf_path, options, stderr)` sniffs
`general.architecture`, resolves the family through the architecture
registry (`models.registry`), and returns a ready `Opened` (`.backend` plus
one `deinit` that owns model, tokenizer, and engine state; the optional
`expert_store` field surfaces a streamed-MoE store for the host's
exit-time report). Families served: qwen3, qwen3moe, gemma4 (the
`Conversation`-hosted set, one generic engine box) and qwen35,
qwen35moe, inkling, deepseek4 (the engine-hosted set, dispatched to
`models.qwen35.serving`, `models.inkling.serving`, `models.deepseek4.serving`).
nanochat and diffusion-gemma stay with `examples/lmserve`; registered
families without a serving adapter (deepseek2, glm4moe) and unknown
architectures return `error.UnsupportedArchitecture`. `OpenOptions`
(defined in `serving/contract.zig`) carries the engine surface
(`context_len`, `spec`, `batch`, `experts_borrow`, `moe_stream`,
`kv_slots` + `kv_slots_force`, `kv_cache_dir` + `kv_disk_slots`,
`cartridge_path`, `fleet_dir` + the `rag_*` knobs); excluded
combinations return `error.InvalidOptions`, and the engine-hosted
families reject the cartridge/fleet/KV-disk options the same way. SHINE adapter fleets are served by
`models.qwen3.shine_serving.open`/`openFromFile` ([§13.12](13-the-model-stack-fucina_models.md#1312-shine-srcmodelsqwen3shinezig)), which mirror
these entries with the fleet directory as an explicit argument and share
`OpenOptions`. `serving.openFromFile` is the same entry
over an already-loaded `fucina.gguf.File` (it takes ownership of the file
on every path); `serving.samplingFromGguf` reads the GGUF-recommended
`general.sampling.*` block. `stderr` is the diagnostic sink for guard
arithmetic and load errors; a host may pass a discarding writer. A
minimal host:

```zig
var opened = try serving.open(&ctx, io, allocator, "model.gguf", .{
    .context_len = 8192,
    .kv_slots = 2,
}, stderr);
defer opened.deinit();

var sched = serving.scheduler.Scheduler.init(allocator, io, opened.backend, 16, 1);
try sched.start();
defer sched.stop();

var shutdown = std.atomic.Value(bool).init(false);
var server = serving.http.Server{
    .allocator = allocator,
    .io = io,
    .opts = .{ .host = "127.0.0.1", .port = 8080 },
    .backend = opened.backend,
    .sched = &sched,
    .shutdown = &shutdown,
};
try server.bind();
try server.run();
```

The band's unit tests ride the models test root (`zig build test-models`): Zig
collects tests from the root module only, and none of them reach the
libc probe, so the models root stays libc-free.
