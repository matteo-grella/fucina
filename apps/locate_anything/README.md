# LocateAnything — open-vocabulary detection

Give it an image and a text prompt; it returns labeled boxes.

This example runs NVIDIA's
[`LocateAnything-3B`](https://huggingface.co/nvidia/LocateAnything-3B)
(MoonViT vision tower + 2-layer MLP projector + Qwen2.5-3B, detection in token
space via coordinate tokens `<0>..<1000>`) from a single self-contained GGUF.
It is a port of
[mudler/locate-anything.cpp](https://github.com/mudler/locate-anything.cpp)
by Ettore Di Giacinto and Richard Palethorpe (MIT), pinned at `92c1682`, and
is validated against it stage by stage: on the parity fixture the generated
token streams are id-exact in all three decode modes and the detections JSON
is byte-identical (f32).

Everything numeric runs on stock Fucina tensor ops — no kernels were added
for this port. The interleaved 2D vision RoPE is a hand-filled `RopeTable`
over the shared rope kernel, ViT attention is the bidirectional
grouped-attention arm, the MTP block-diffusion mask rides the additive-bias
attention arm, and every linear goes through `LinearWeight` (so the
f16/q8_0/K-quant arms and the BLAS / `-Dgpu=metal` / `-Dgpu=cuda` GEMM
dispatch apply unchanged). Host-side scalar code is limited to the
decision-critical control paths ported verbatim from the reference:
PIL-exact bicubic preprocessing, bicubic position-embedding interpolation,
and the MTP box-decode heuristics.

## Getting the model

Model weights are not part of this repository. The GGUF uses the reference
port's `locateanything.*` schema; build it with the reference's converter
(both repos read the same file). The two scripts run from the reference
checkout and need `huggingface_hub`, `safetensors`, `gguf` and `numpy`
(the reference's `scripts/requirements.txt` is the full superset):

```sh
tools/fetch_refs.sh locate-anything.cpp     # clone + pin + init the ggml submodule
cd refs/locate-anything.cpp
python3 -m venv .venv && .venv/bin/pip install huggingface_hub safetensors gguf numpy
.venv/bin/python scripts/download_model.py            # HF checkpoint (bf16 safetensors)
.venv/bin/python scripts/convert_locateanything_to_gguf.py   # -> models/locate-anything-f32.gguf (~15 GB)
```

The converter writes `models/locate-anything-f32.gguf` inside
`refs/locate-anything.cpp/`; the commands below assume repo-root `models/`
(gitignored), so move it up or pass the full path:

```sh
mkdir -p ../../models && mv models/locate-anything-f32.gguf ../../models/
```

Quantized variants (LM matmuls only; the ViT, projector, norms, biases and
the two host-read f32 tensors stay f32) either come prebuilt from
[mudler/locate-anything.cpp-gguf](https://huggingface.co/mudler/locate-anything.cpp-gguf)
(the `hf` CLI comes from `pip install -U huggingface_hub`; see
[`docs/RUNNING-MODELS.md`](../../docs/RUNNING-MODELS.md#getting-the-weights)):

```sh
mkdir -p models
hf download mudler/locate-anything.cpp-gguf locate-anything-q8_0.gguf --local-dir models
```

or are produced by the reference CLI, from the reference's stock cmake build
(`locate-anything-cli` builds by default):

```sh
cmake -S refs/locate-anything.cpp -B refs/locate-anything.cpp/build
cmake --build refs/locate-anything.cpp/build -j
refs/locate-anything.cpp/build/examples/cli/locate-anything-cli \
    quantize models/locate-anything-f32.gguf models/locate-anything-q8_0.gguf q8_0
```

Fucina reads f32, f16 and the q8_0/q6_k/q5_k/q4_k variants. One loader note:
GGUFs that materialize a quantized `lm.output.weight` with a row count that
is not a multiple of 8 (this model's vocab is 152681) load as an x8-aligned
packed prefix plus a small dequantized-f32 tail, concatenated at the logits;
float heads and aligned quant heads take the plain path.

## CLI

```sh
# Detect: labeled boxes as JSON (byte-compatible with the reference CLI's format)
zig build locate-anything -Doptimize=ReleaseFast -- detect \
    --model models/locate-anything-q8_0.gguf \
    --input scene.png \
    --prompt 'Locate all the instances that matches the following description: person</c>car.' \
    --mode hybrid --output boxes.json --annotated out.png

# Model load smoke test
zig build locate-anything -- info --model models/locate-anything-f32.gguf
```

`detect` flags:

| flag | meaning |
| --- | --- |
| `--model <gguf>` | model file (required) |
| `--input <image.png>` | input image (required; PNG only, see *Scope*) |
| `--prompt <text>` | open-vocabulary query; separate categories with `</c>` (required) |
| `--mode hybrid\|slow\|fast` | decode mode, default `hybrid` (see below) |
| `--output <file.json>` | write the detections JSON (default: stdout) |
| `--annotated <file.png>` | also render the boxes + label chips onto the image |
| `--max-new N` | generation cap, default 256 |
| `--no-early-stop` | disable the degenerate-tail early stop and run the full stream |

Decode modes mirror the upstream `generation_mode`: `hybrid` is Parallel Box
Decoding (6-token MTP blocks with autoregressive fallback on malformed
boxes), `slow` is pure autoregressive decoding, `fast` is MTP-only with no
fallback. Greedy decoding only, single image only — the same deliberate scope
as the reference. By default the degenerate repeated-box tail that greedy
hybrid/fast decoding produces at the cap is trimmed by the reference's
early-stop heuristics; `--no-early-stop` reproduces the full stream (that is
also what the parity gates compare).

Boxes denormalize against the preprocessed target size (the coordinate
tokens are in 0..1000 of `gw*14 x gh*14`), matching the reference exactly.

## Parity harness

`compare` gates every pipeline stage against a dump captured from the
reference implementation and exits nonzero on any failure:

```sh
zig build locate-anything -Doptimize=ReleaseSafe -- compare \
    --model models/locate-anything-f32.gguf \
    --dump dumps/fixture_dump.gguf \
    --image refs/locate-anything.cpp/tests/fixtures/parity_image.png \
    --prompt 'Locate all the instances that matches the following description: cat</c>remote.' \
    --stage all      # or: tokenizer preproc prompt vit projector lm slow hybrid fast
```

Gates: tokenizer cases and `prompt_ids` are token-ID-exact, `pixel_values`
byte-exact, ViT/projector/LM tensors tight-f32 (max-abs, plus a
relative-to-magnitude criterion for the deep pre-norm captures), the
`slow`/`hybrid`/`fast` token streams exactly equal, and per-round MTP block
logits toleranced. `compare` also accepts `--mtp-rounds N` (default 12) to
cap the per-round MTP block-logits gates.

`dumps/fixture_dump.gguf` is not shipped. It is produced by
`tools/ref-patches/la_dump.cpp`, an out-of-tree harness compiled against the
stock pinned reference build (cmake, above) — its header has the exact
compile line (macOS/Accelerate link line) and the dump layout. Image and
prompt are pinned by
`refs/locate-anything.cpp/tests/fixtures/fixture_spec.json`:

```sh
mkdir -p dumps
/tmp/la_dump models/locate-anything-f32.gguf \
    refs/locate-anything.cpp/tests/fixtures/parity_image.png \
    'Locate all the instances that matches the following description: cat</c>remote.' \
    dumps/fixture_dump.gguf
```

## Performance

Measured against the reference CLI on the same machine, same threads, same
model files, full streams, load time subtracted (protocol and full tables in
`docs/BENCHMARK.md`): on an M1 Max (8 threads) Fucina is ~1.4–2.3x faster at
f32 and ~2.2–2.4x at q8_0 across all three modes with byte-identical
detections; on an i9-13950HX (8 threads, P-cores, no BLAS on either side)
~1.2–1.4x across the same grid. Operate hybrid CPUs at their physical-core
count; oversubscribing onto hyperthreads degrades Fucina's worker team.

Build discipline (`-Doptimize`), BLAS, `-Dgpu=metal|cuda` offload, and the
thread knob (default 8; lower at runtime with `FUCINA_MAX_THREADS=N`, raise
above 8 only at build time with `-Dmax-threads=N`) are shared across all
runners and documented in
[`docs/RUNNING-MODELS.md`](../../docs/RUNNING-MODELS.md).

## Scope and known differences vs the reference

- **PNG input only.** The pure-Zig reader covers non-interlaced 8/16-bit
  gray/RGB/palette/alpha PNGs; PNG decoding is lossless, so pixels are
  byte-identical to the reference's stb path. JPEG is intentionally out of
  scope: stb's JPEG decode is implementation-defined, so pixel-exact parity
  is not reproducible from an independent decoder.
- **Annotated-image label colors** hash the label with FNV-1a; the reference
  uses the implementation-defined `std::hash`, so per-label colors can differ
  between builds. Same palette, same layout; cosmetic only.
- **The MTP mask** uses a -1e9 additive bias where the reference uses -inf;
  in f32 the masked probabilities underflow to exactly 0 either way.
- **Quantized runs are engine-exact, not cross-engine-exact.** f32 streams
  are byte-identical to the reference on both tested ISAs; q8_0 boxes matched
  the reference byte-for-byte on ARM but drift by one coordinate token
  (~0.45 px) on two fields (plus the meaningless degenerate tail) on x86 —
  per-arch activation-quantization rounding. Real detections match in every
  tested configuration.
