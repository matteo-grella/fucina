# face-detect.cpp goldens (parity + perf reference)

Captured from the **self-contained C++/ggml reference** — no Python. These
are the parity targets the Fucina port's test suite gates against, and the
baseline for paired CPU benchmarks.

- Reference: `refs/face-detect.cpp` @ `e22260d5` (pinned in `tools/fetch_refs.sh`), CMake `Release` build.
- Weights: HF `mudler/face-detect-gguf` → `buffalo_l.gguf` (det+rec+genderage+anti-spoof), `landmarks-2d106-1k3d68.gguf`. Kept out of git (under the `models/` symlink).
- Fixtures: `refs/face-detect.cpp/tests/fixtures/face_{a,b,c}.jpg`.
- Machine for `bench-baseline.txt`: Apple M1 Max, 8 threads.

## Files
- `info-buffalo_l.txt` — model config (scrfd+arcface, 512-d, genderage + anti-spoof present).
- `embed-{a,b,c}.json` — 512-d ArcFace embedding (cosine gate, ≥0.9999).
- `crop-{a,b,c}.bin` — the reference's **aligned 112×112 RGB crop** (FDR1 raw), the
  strict-gate input to the ArcFace forward. Captured via the env-gated dump hook
  `tools/ref-patches/face-detect.cpp-dump.patch` (apply with `tools/fetch_refs.sh
  --patch`, rebuild). The hook is a read-only side effect: the `embed` output is
  byte-identical with and without it (verified against `embed-{a,b,c}.json`). Feed
  `crop-X.bin` → Fucina ArcFace → cosine vs `embed-X.json`.
- `ga-crop-{a,b,c}.bin` — the reference's **aligned 96×96 RGB crop** for genderage
  (a *different* alignment: 1.5× box expand, scale-about-center), the strict-gate
  input to the genderage forward. Same dump hook (dumps in `genderage_forward`);
  `analyze` output is byte-identical with/without it. Feed `ga-crop-X.bin` →
  Fucina genderage → gender (exact) + age (tol) vs `analyze-X.txt`.
- `as-crop-{a,b,c}-{0,1}.bin` — anti-spoof 80² member crops (0 = MiniFASNetV2 @ scale
  2.7, 1 = V1SE @ scale 4.0), FDR1 raw. `as-realprob-{a,b,c}.txt` — the averaged
  ensemble "real" probability (softmax idx 1): 0.999802 / 0.999766 / 0.979237. Both
  via the dump hook in `ensemble_softmax` (read-only). Feed the two crops → Fucina
  MiniFASNet replay → softmax → avg idx-1, gate vs the golden (tol 1e-3).
- `detect-{a,b,c}.txt` — SCRFD boxes + score + 5 landmarks (≤1px gate on decoded
  geometry; the CLI JSON gate is byte-identical).
- `det-blob-{a,b,c}.bin` — the letterboxed **640² detector input** (FDR1);
  `det-scale-{a,b,c}.txt` — the matching letterbox `det_scale`. `det-heads-{a,b,c}.bin`
  — the **9 flattened raw heads** (score/bbox/kps × strides 8/16/32), each `u32 count` +
  f32 data; 16800 locations × 15 ch total. Dump hook in `scrfd_forward`. Feed the blob →
  Fucina SCRFD → heads (≤1e-3) → decode + NMS (≤1px).
- `analyze-{a,b,c}.txt` — genderage age + gender (gender exact / age tol).
- `landmarks2d-{a,b,c}.json`, `landmarks3d-{a,b,c}.json` — 2d106 / 1k3d68 image-space points.
- `lm-crop-{a,b,c}-{2d,3d}.bin` — reference 192² landmark crops (FDR1). `lm-pts-{a,b,c}-{2d,3d}.txt`
  — crop-space decoded points (`x y z` per line). Dump hook in `landmark_forward`/
  `landmark_decode_crop`. Feed the crop → the `graph.zig` replay → decode → ≤1.5px vs the points.
- `align-src-{a,b,c}.bin` — the decoded source images (FDR1): byte-exact pixels of the
  JPEG fixtures, the input for the end-to-end and CLI byte-identity gates (the Fucina
  CLI reads PNG/FDR1, not JPEG, so these pin the exact reference pixels).
  `align-lmk-{a,b,c}.txt` — the 5-point landmarks feeding the umeyama-align gate.
- `verify-ab,ac,ab-antispoof.txt` — cosine distance + verdict (verdict exact).
- `bench-baseline.txt` — reference ggml CPU ms/image per mode on this M1 Max.

## Regenerate
```sh
tools/fetch_refs.sh face-detect.cpp
cd refs/face-detect.cpp && git submodule update --init --recursive && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
BIN=refs/face-detect.cpp/build/examples/cli/facedetect-cli   # then embed/detect/analyze/verify/landmarks/bench per file above
```

## Benchmarking against the reference — platform note

The default `Release` build above IS the reference's production CPU config
(on arm64: `GGML_NATIVE=ON`, vendored tinyBLAS). Its fast 3×3-conv kernels
(Winograd F2/F4, blocked directconv) are x86-intrinsic implementations with
scalar fallbacks, and its conv router still selects them on ARM — so on
Apple Silicon the reference runs those convs scalar, while on AVX2 hardware
they are fully armed. Compare per-platform numbers accordingly, with matched
thread counts (the reference CLI defaults to `min(hw, 8)` threads;
`--threads`/`FACEDETECT_THREADS` override).
