# Parakeet — speech-to-text (NVIDIA NeMo FastConformer)

Offline, streaming, and live-microphone transcription of WAV audio on the
CPU (any rate/channels — see [Audio input](#audio-input)).

This example runs NVIDIA NeMo Parakeet FastConformer ASR models from GGUF:
CTC, TDT, and hybrid TDT+CTC decoder heads, multilingual prompt-conditioned
models, and cache-aware streaming variants. It is a port of
[mudler/parakeet.cpp](https://github.com/mudler/parakeet.cpp) by Ettore Di
Giacinto (the ready-to-run GGUF weights come from his conversions), pinned at
`89f5e29`, and is validated against it per stage: the hard parity target is
an exact decoded token-id sequence; intermediate stages gate on cosine
(op-order makes bit-exact unrealistic).

## Getting the model

Weights are not part of this repository.
[`mudler/parakeet-cpp-gguf`](https://huggingface.co/mudler/parakeet-cpp-gguf)
is one flat repo with every supported NVIDIA NeMo Parakeet model ×
quantization (f16/q8_0/q6_k/q5_k/q4_k). Start with `tdt_ctc-110m-f16.gguf`
(267 MB hybrid, fast) or `tdt-0.6b-v3-f16.gguf` (1.44 GB, multilingual).
The `hf` CLI (from `pip install -U huggingface_hub`; shared download
machinery in
[`docs/RUNNING-MODELS.md`](../../docs/RUNNING-MODELS.md#getting-the-weights))
fetches single files:

```sh
mkdir -p models/parakeet
hf download mudler/parakeet-cpp-gguf tdt_ctc-110m-f16.gguf --local-dir models/parakeet
```

**License.** The NVIDIA NeMo Parakeet weights are distributed under
CC-BY-4.0.

The streaming modes (`--stream`, `--mic`, `--mic-sim`, `--stream-bench`)
need a model with `streaming.*` metadata — the runner errors otherwise. The
models benchmarked for streaming in
[`docs/BENCHMARK.md`](../../docs/BENCHMARK.md) are `realtime_eou-120m` and
the multilingual `nemotron-0.6b`. Their files in the same repo are
`realtime_eou_120m-v1-f16.gguf` and `nemotron-3.5-asr-streaming-0.6b-f16.gguf`,
with the same quantization ladder:

```sh
hf download mudler/parakeet-cpp-gguf realtime_eou_120m-v1-f16.gguf --local-dir models/parakeet
```

## Audio input

Any RIFF WAV decodes: PCM 16/24/32-bit int or 32-bit IEEE float (incl.
WAVE_FORMAT_EXTENSIBLE), any sample rate, any channel count. Multi-channel
audio is downmixed by averaging and resampled to 16 kHz with the same
linear resampler as parakeet.cpp (`load_audio_16k_mono`), so native 16 kHz
mono is simply the no-resampling path. Other containers (MP3/FLAC/…) are
not supported.

The repository ships no audio. The pinned parity reference carries two
16 kHz mono fixtures — `speech.wav` (7.435 s of speech, the fixture behind
the parakeet rows in [`docs/BENCHMARK.md`](../../docs/BENCHMARK.md)) and
`clip.wav` (2 s, stage-parity fixture; it decodes to an empty transcript,
so use `speech.wav` when you want visible output):

```sh
tools/fetch_refs.sh parakeet.cpp
# -> refs/parakeet.cpp/tests/fixtures/{clip,speech}.wav  (refs/ is gitignored)
```

Smallest end-to-end run (model from above + fetched fixture; prints the
transcript on stdout):

```sh
zig build parakeet -Doptimize=ReleaseFast -- \
  --model models/parakeet/tdt_ctc-110m-f16.gguf \
  --audio refs/parakeet.cpp/tests/fixtures/speech.wav --transcribe
```

## CLI

```sh
# Transcribe a WAV (clean transcript on stdout)
zig build parakeet -Doptimize=ReleaseFast -- --model models/parakeet/tdt_ctc-110m-f16.gguf \
  --audio clip.wav --transcribe

# JSON output / per-word timestamps (offline decode only)
zig build parakeet -Doptimize=ReleaseFast -- --model models/parakeet/tdt_ctc-110m-f16.gguf \
  --audio clip.wav --transcribe --json --timestamps

# Batch a manifest (one audio path per line)
zig build parakeet -Doptimize=ReleaseFast -- --model models/parakeet/tdt_ctc-110m-f16.gguf \
  --manifest files.txt

# Streaming pipeline (cache-aware chunked encode; use a streaming-capable model)
zig build parakeet -Doptimize=ReleaseFast -- --model models/parakeet/tdt_ctc-110m-f16.gguf \
  --audio clip.wav --stream

# Live microphone (build the capture backend in with -Dparakeet-mic)
zig build parakeet -Dparakeet-mic -Doptimize=ReleaseFast -- \
  --model models/parakeet/tdt_ctc-110m-f16.gguf --mic

# Benchmarks: best-of-N offline timing / streaming RTF + first-token latency
zig build parakeet -Doptimize=ReleaseFast -- --model ... --audio clip.wav --transcribe --bench-reps 5
zig build parakeet -Doptimize=ReleaseFast -- --model ... --audio clip.wav --stream-bench --bench-reps 5
```

Running with only `--model` prints the config + tensor summary.

## Flags

| flag | meaning |
| --- | --- |
| `--model <path>` | parakeet GGUF (required) |
| `--audio <path>` | WAV to transcribe (see [Audio input](#audio-input)) |
| `--transcribe` | offline transcription of `--audio` |
| `--stream` | streaming pipeline (cache-aware chunked encode) over `--audio` |
| `--manifest <file>` | batch transcription: one audio path per line |
| `--mic` | live microphone capture (needs `-Dparakeet-mic`, see below) |
| `--mic-sim` | feed `--audio` through the incremental mic driver |
| `--stream-bench` | streaming RTF + first-token latency benchmark |
| `--json` | JSON output (offline decode only) |
| `--timestamps` | per-word start/end/confidence (offline decode only) |
| `--threads <n>` | worker thread count (0 = default) |
| `--decoder tdt\|ctc` | decoder head for hybrid models (default `tdt`) |
| `--lang <XX>` | target locale for multilingual prompt-conditioned models (default auto) |
| `--compare <stage> <dump>` | parity-check a stage vs a parakeet.cpp PKD1 dump (offline stages: `stft`, `mel`, `subsampling`, `encoder`, `ctc`, `tdt`, `joint0`; streaming stages: `stream-mel`, `stream-sub`, `stream-encoder`, `stream-prompt`, `stream-session`, `stream-decode`, `stream-full`) |
| `--tol <f>` | max-abs tolerance for `--compare` (default 1e-4) |
| `--f32-cache` | cache sequence linear weights as f32 and route them through BLAS |
| `--fast-mel` | BLAS mel filterbank projection (on by default; `--no-fast-mel` disables) |
| `--bench-reps <n>` | run transcribe n times in one loaded session, report best timing |

`--json` is rejected in combination with `--stream`; `--decoder ctc` needs a
model that actually carries a CTC head (the hybrids do) and errors cleanly
otherwise. A failed `--compare` stage exits nonzero, so the parity harness
is mechanically enforcing; regenerating the reference dumps needs the
out-of-tree instrumentation patch (`tools/fetch_refs.sh --patch
parakeet.cpp`).

`--manifest` lines are trimmed; blank lines and `#` comments are skipped,
and a line starting with `{` is parsed as a NeMo-style JSONL entry (its
`"audio_filepath"` value is used). Each file's output honors
`--json`/`--timestamps`, is preceded by a `# <path>` header, and is
identical to transcribing that file singly.

## Live microphone: `-Dparakeet-mic`

`--mic` is compiled out of the default build. Build with `-Dparakeet-mic` to
link the vendored miniaudio capture stack — it reuses the NAM example's
audio shim (`examples/nam/audio_shim.c` + `third_party/miniaudio.h`,
capture only, no MIDI). The option defaults to false to keep the default
parakeet build fast; without it, `--mic` exits with a message pointing at
the flag. On macOS the build links the CoreFoundation/CoreAudio/AudioToolbox
frameworks; on other platforms miniaudio dlopens its backend at runtime.
`--mic` also requires a streaming-capable model. `--mic-sim` runs the same
incremental driver over `--audio` without any capture hardware and needs no
build option.

`--mic` captures from the first capture device at 16 kHz for a fixed 20 s
window, reprinting the partial transcript as tokens arrive, then finalizing
and exiting. On macOS the microphone permission is attributed to your
**terminal app** — a denied permission yields silence with no error (check
System Settings → Privacy → Microphone); the [NAM example](../nam/README.md)
has the same note for the same capture stack.

## Performance

Honest perf note: transcription is parity-checked against parakeet.cpp/NeMo,
but parakeet.cpp is still faster on CPU — on the 110m hybrid, Fucina
full-transcribe RTF ≈ 0.034 vs parakeet.cpp ≈ 0.021 (M1 Max), a ~1.6× gap.
See [`docs/BENCHMARK.md`](../../docs/BENCHMARK.md).

## Shared knobs

Build discipline (`-Doptimize=ReleaseFast`, build on the machine you run on
or pass `-Dcpu`), global thread control, and the BLAS backends are shared
across all runners and documented in
[`../../docs/RUNNING-MODELS.md`](../../docs/RUNNING-MODELS.md). This runner
has no MoE-streaming, GPU-offload, or constrained-decoding surface;
`--threads` above is its per-run thread override.
