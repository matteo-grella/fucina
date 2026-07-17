# facedetect — buffalo_l face pipeline (face-detect.cpp port)

Face detection, recognition, attribute analysis, anti-spoofing, and dense
landmarks from a single self-contained GGUF — a pure-Zig port of
[mudler/face-detect.cpp](https://github.com/mudler/face-detect.cpp) running
the insightface **buffalo_l** pack on Fucina's CPU runtime:

- **SCRFD det_10g** detector (boxes + 5-point landmarks),
- **ArcFace IResNet-50** recognizer (512-d embeddings, verification),
- **GenderAge** MobileNet-0.25 head,
- **MiniFASNet ×2** anti-spoof ensemble,
- **2d106det / 1k3d68** dense landmarks (separate GGUF).

Outputs match the reference CLI: `detect`/`analyze` JSON is byte-identical,
embeddings agree at cosine ≥ 0.999999, anti-spoof probabilities are exact.

## Weights

Not bundled. Fetch the GGUFs from Hugging Face
[`mudler/face-detect-gguf`](https://huggingface.co/mudler/face-detect-gguf)
and place (or symlink) them under `models/`:

- `buffalo_l.gguf` — detector + recognizer + genderage + anti-spoof,
- `landmarks-2d106-1k3d68.gguf` — the dense-landmark heads.

```sh
hf download mudler/face-detect-gguf buffalo_l.gguf --local-dir models
hf download mudler/face-detect-gguf landmarks-2d106-1k3d68.gguf --local-dir models
```

(`hf` comes from `pip install -U huggingface_hub` — see
[docs/RUNNING-MODELS.md](../../docs/RUNNING-MODELS.md#getting-the-weights).)
Run commands from the repo root; the CLI examples below and the test suite
both resolve `models/` relative to it. The model-dependent parity tests
skip when the GGUFs are absent — a green `zig build test` without them has
not exercised the parity gates.

The insightface model weights carry their own (non-commercial) license
terms — see `docs/THIRD-PARTY-NOTICES.md`.

## Usage

```sh
zig build facedetect -Doptimize=ReleaseFast -- info models/buffalo_l.gguf

# detect all faces -> reference-format JSON (boxes, scores, 5-point landmarks)
zig build facedetect -Doptimize=ReleaseFast -- detect --model models/buffalo_l.gguf --input face.png --json

# 512-d ArcFace embedding of the largest face (detect -> align -> embed)
zig build facedetect -Doptimize=ReleaseFast -- embed --model models/buffalo_l.gguf --input face.png --json

# 1:1 verification of two images (cosine distance; default threshold 0.35)
zig build facedetect -Doptimize=ReleaseFast -- verify --model models/buffalo_l.gguf --a a.png --b b.png [--threshold T]

# gender + age of the largest face
zig build facedetect -Doptimize=ReleaseFast -- analyze --model models/buffalo_l.gguf --input face.png

# per-mode CPU benchmark (model load + image decode outside the timed loop,
# one untimed warmup pass, arithmetic mean over N)
zig build facedetect -Doptimize=ReleaseFast -- bench --model models/buffalo_l.gguf --input face.png --mode pipeline|recognizer|detect|analyze --n 20
```

No photos needed for a first run — `goldens/` ships the reference
fixtures' decoded pixels (`align-src-{a,b,c}.bin`) as FDR1, which the CLI
reads directly:

```sh
zig build facedetect -Doptimize=ReleaseFast -- detect --model models/buffalo_l.gguf --input examples/facedetect/goldens/align-src-a.bin --json
zig build facedetect -Doptimize=ReleaseFast -- verify --model models/buffalo_l.gguf --a examples/facedetect/goldens/align-src-a.bin --b examples/facedetect/goldens/align-src-b.bin
```

Notes:

- **Inputs** are PNG (8-bit, non-interlaced, grayscale/RGB/RGBA — palette
  PNGs are not decoded) or FDR1 (a raw-pixel dump format, magic `FDR1`; the
  goldens use it to pin reference-decoded pixels). JPEG is not decoded —
  convert first (`sips -s format png face.jpg --out face.png`).
- `--threads N` caps the worker team (default: `min(cores, 8)`, the same
  default as the reference CLI). `FUCINA_MAX_THREADS` does the same via env.
- `bench --mode recognizer` expects a pre-aligned 112×112 crop as input
  (e.g. `examples/facedetect/goldens/crop-a.bin`), mirroring the reference's
  protocol; the other modes take a full source image.
- The dense-landmark nets (2d106det / 1k3d68) are exercised by the parity
  gates under `zig build test` (GGUF graph replay vs `goldens/lm-*` crops
  and points; skipped when `models/landmarks-2d106-1k3d68.gguf` is absent).
  The `landmarks` CLI subcommand is not wired end-to-end and prints a
  not-implemented notice.

## Structure

`recognizer.zig` / `scrfd.zig` / `genderage.zig` hand-map their nets onto the
public tagged `Tensor` facade (channel-last `[h,w,c]`; conv2d, pool2d, prelu,
channelAffine, upsample). Weights load once into `Model` structs (GGUF
dequant + layout repack + BatchNorm folding happen at load; forwards are pure
compute). The two interpreter-driven nets (anti-spoof, landmarks) replay a
GGUF-embedded node list through `graph.zig` — an app-level compiled
dispatcher over `ExecContext` ops, not a general ONNX runtime. Decision-
critical control paths (cv2-exact letterbox, umeyama align, NMS decode) are
verbatim scalar ports.

Tests run under `zig build test` (a `facedetect` test root); the parity
gates compare against `goldens/` (see `goldens/README.md` for what each
golden pins and how to regenerate them from the pinned reference).
`FUCINA_TEST_VERBOSE=1` prints the measured per-case margins (cosines,
pixel errors, real-probs) on passing runs; by default passing tests are
silent so `zig build test` stderr stays clean.

## Measured (ms/image, matched 8 threads, n=20)

| mode | Fucina | reference (same machine, production config) |
|---|---|---|
| **M1 Max** recognizer | ~35 | ~765 |
| **M1 Max** detect | ~84 | ~113 |
| **M1 Max** pipeline (detect+align+embed) | ~116 | ~872 |
| **i9-13950HX** recognizer | ~49 | ~56–61 |
| **i9-13950HX** detect | ~70 | ~53–59 |
| **i9-13950HX** pipeline | ~116–121 | ~115–123 |

The reference's fast conv kernels are x86-only (scalar fallback on ARM),
which is why the gap differs so much by platform; see the note in
`goldens/README.md` before quoting numbers.

2026-07-10, prepared Winograd weights (load-time transform planes instead
of per-call, i9, same protocol): recognizer 54.4 → 50.4 ms (−7.2%), detect
72.5 → 71.6 ms (−1.2%) — matching the transform's predicted share of
winograd byte traffic (~40% recognizer, ~4% detect). The remaining x86
detect gap is the fused tile-blocked winograd lever, not weight
preparation.
