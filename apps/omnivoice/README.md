# OmniVoice in Fucina

Fucina port of [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) (Xiaomi / k2-fsa,
Apache 2.0) via [omnivoice.cpp](https://github.com/ServeurpersoCom/omnivoice.cpp):
multilingual zero-shot text-to-speech covering 646 languages, running as a
**MaskGIT non-autoregressive decoder** on a Qwen3-0.6B backbone with the Higgs
Audio v2 codec (HuBERT semantic + DAC acoustic + 8-codebook RVQ, 25 fps, 24 kHz
mono). CPU-only, eager execution, no ggml.

Numerics are validated step-by-step against the C++ reference: RVQ codes and
MaskGIT token streams are **byte-exact** (F32, fixed seed), decoded audio cosine
≥ 0.99999 — and it is 2.3–4.6× faster than omnivoice.cpp's CPU backend on
M1 Max (32× on BF16, where ggml's CPU path is pathological).

## Models

Download the GGUFs from
[Serveurperso/OmniVoice-GGUF](https://huggingface.co/Serveurperso/OmniVoice-GGUF/tree/main)
into `models/omnivoice/`:

| file | role |
|---|---|
| `omnivoice-base-{F32,BF16,Q8_0,Q4_K_M}.gguf` | the TTS LLM (Qwen3-0.6B + audio embeddings/heads) |
| `omnivoice-tokenizer-{F32,BF16,Q8_0,Q4_K_M}.gguf` | the audio codec (HuBERT + DAC + RVQ) |

All 16 base×tokenizer pairings work. **Recommended: `base-Q8_0` + `tokenizer-F32`**
(fastest with full quality — also the reference's shipped default). `base-F32` is
the parity/determinism baseline. Always build with `-Doptimize=ReleaseFast`.

Fetch the recommended pair with the `hf` CLI:

```sh
mkdir -p models/omnivoice
hf download Serveurperso/OmniVoice-GGUF omnivoice-base-Q8_0.gguf --local-dir models/omnivoice
hf download Serveurperso/OmniVoice-GGUF omnivoice-tokenizer-F32.gguf --local-dir models/omnivoice
```

(`hf` comes from `pip install -U huggingface_hub` — see
[Getting the weights](../../docs/RUNNING-MODELS.md#getting-the-weights); the
pair is ~1.4 GB.)

## Quick start

Text comes from stdin; output is 24 kHz mono WAV.

```sh
BASE=models/omnivoice/omnivoice-base-Q8_0.gguf
CODEC=models/omnivoice/omnivoice-tokenizer-F32.gguf

# auto voice — the model picks a coherent speaker per utterance
echo "Hello from Fucina." | zig build omnivoice -Doptimize=ReleaseFast -- tts \
  --model $BASE --codec $CODEC --lang English -o hello.wav

# voice design — describe the speaker
echo "Hello from Fucina." | zig build omnivoice -Doptimize=ReleaseFast -- tts \
  --model $BASE --codec $CODEC --lang English \
  --instruct "female, young adult, high pitch" --seed 42 -o designed.wav

# voice cloning — a reference WAV plus its transcript drive the speaker
printf 'This is what the reference recording says.' > ref.txt
echo "New text spoken in the cloned voice." | zig build omnivoice -Doptimize=ReleaseFast -- tts \
  --model $BASE --codec $CODEC --lang English \
  --ref-wav voice.wav --ref-text ref.txt --seed 42 -o cloned.wav

# straight to the speakers — no output file needed
echo "Hello from Fucina." | zig build omnivoice -Doptimize=ReleaseFast -- tts \
  --model $BASE --codec $CODEC --lang English --seed 42 --play
```

Any input WAV works for cloning (PCM16/24/F32, mono or stereo, any sample
rate — it is resampled to 24 kHz). A few seconds of clean speech is enough;
`--ref-text` must contain what the reference recording actually says.

No reference recording handy? Synthesize one and clone from it — the designed
voice becomes the reference and `ref.txt` is its exact transcript:

```sh
printf 'The reference recording says exactly this sentence.' > ref.txt
< ref.txt zig build omnivoice -Doptimize=ReleaseFast -- tts \
  --model $BASE --codec $CODEC --lang English \
  --instruct "female, young adult, high pitch" --seed 7 -o voice.wav
```

The resulting `voice.wav` + `ref.txt` pair drives the cloning command above.

## The three voice modes

- **Auto voice** (no `--ref-wav`, no `--instruct`): the model invents a speaker.
  In chunked long-form runs, chunk 0's audio tokens become the voice prompt for
  the remaining chunks, locking the speaker in.
- **Voice design** (`--instruct`): comma-separated attributes from six mutually
  exclusive categories —
  gender `male`/`female`; age `child`/`teenager`/`young adult`/`middle-aged`/`elderly`;
  pitch `very low pitch`/`low pitch`/`moderate pitch`/`high pitch`/`very high pitch`;
  style `whisper`; accent (English) `american|british|australian|chinese|canadian|indian|korean|portuguese|russian|japanese accent`;
  dialect (Chinese) 河南话 陕西话 四川话 贵州话 云南话 桂林话 济南话 石家庄话 甘肃话 宁夏话 青岛话 东北话.
  Chinese equivalents of the EN terms are accepted; invalid items get a
  did-you-mean error.
- **Voice cloning** (`--ref-wav` + `--ref-text`, or a pre-encoded `--ref-rvq`):
  the reference is preprocessed (RMS auto-gain, silence trim), encoded to RVQ
  codes, and prepended to every chunk's prompt. The reference loudness sets the
  output loudness.

`--lang` takes an English language name (e.g. `English`, `Chinese`, `Cantonese`,
`Japanese`…) or an ISO id (`en`, `zh`, `yue`…); `None` lets the model infer it.

## Long-form and streaming

Text whose estimated duration exceeds `--chunk-threshold` (default 30 s) is
split on sentence punctuation into ~`--chunk-duration` (default 15 s) chunks,
synthesized sequentially, and cross-faded. `--duration <sec>` forces a
single-shot synthesis of exactly that length.

`-o -` streams WAV to stdout: each chunk's audio is flushed as soon as it is
synthesized while the next chunk generates (MaskGIT cannot stream *within* a
chunk — all 32 refinement steps are full bidirectional passes, so
time-to-first-audio is one chunk). `--stream-by-line` starts a fresh WAV (new
RIFF header) at every input line — pipe-friendly for sentence-at-a-time use.
Note: in pure auto-voice streaming there is no global peak normalization (the
peak is unknowable mid-stream), so output is a few dB quieter than the
buffered path; cloning is unaffected.

```sh
zig build omnivoice -Doptimize=ReleaseFast -- tts --model $BASE --codec $CODEC \
  --lang English -o - < book-chapter.txt | ffplay -autoexit -nodisp -i -
```

## Speaker playback (`--play`)

`--play` sends the synthesis to a playback-only miniaudio device (vendored
from the NAM example; the 24 kHz mono stream is converted to the device's
native rate internally). `devices` lists the playback devices;
`--playback <idx>` picks one (default: the system device). At least one of
`-o` / `--play` is required; `--play` with `-o -` is invalid (stdout is the
WAV stream).

- `--play` alone **streams**: each chunk starts playing the moment it is
  synthesized. Generation is ~2–3× slower than realtime on CPU, so expect
  silent gaps between long-form chunks (a ~10–13 s chunk arrives after
  ~25–45 s of compute at Q8_0 on M1 Max); a throttled `[Play] playback gap`
  note marks each gap.
- `--play` with `-o file.wav` does both, **buffered**: the file is written
  first (byte-identical to a run without `--play`), then the final waveform
  plays gaplessly.

```sh
zig build omnivoice -Doptimize=ReleaseFast -- devices
echo "Hello." | zig build omnivoice -Doptimize=ReleaseFast -- tts \
  --model $BASE --codec $CODEC --lang English --play --playback 1
```

## Progress signals

All liveness output goes to **stderr** (stdout may be the WAV stream):
`[Load]` lines for the model/codec GGUFs (size, dtype, time), the existing
`[TTS]`/`[TTS-Long]` chunk and reference-encode phases, `[Perf]` summaries,
and a per-generation MaskGIT step line. On a TTY the step line updates in
place (`[MaskGIT] step 12/32 · demasked 320/520 · 4.1s`) and finalizes with a
newline; piped/non-TTY runs get a plain line every 8 executed steps plus the
final one.

## Using it from another project

The example doubles as a library: `pipeline.synthesize` returns the full
waveform, `pipeline.synthesizeStream(tts, params, sink)` emits post-processed
chunks through a `postproc_stream.Emit` sink as they are ready (the seam the
CLI feeds the speakers from), `pipeline.Params.progress` /
`mg_decode.Progress` delivers per-step MaskGIT progress (context pointer +
`fn (step, num_steps, demasked, total)`, called once per executed step), and
`play.Player` (`init` / `pushSamples` / `drainAndStop` / `deinit`, plus
`play.listPlaybackDevices`) is a reusable ring-buffered playback device —
underruns play silence and are counted, never treated as errors.

## Codec tool (WAV ↔ RVQ codes)

```sh
# encode: wav -> .rvq (8 codebooks @ 25 fps, 11 bits/code, written next to the input)
zig build omnivoice -Doptimize=ReleaseFast -- codec --model $CODEC -i clip.wav
# decode: .rvq -> wav
zig build omnivoice -Doptimize=ReleaseFast -- codec --model $CODEC -i clip.rvq
```

`.rvq` files are interchangeable with omnivoice.cpp's (same bit packing);
encoding applies the reference preprocessing (auto-gain, silence trim, hop
alignment), so encode(x) is the exact voice-prompt representation cloning uses
— reusable via `--ref-rvq`.

Encode accepts the same WAV envelope as cloning (PCM16/24/F32, mono or
stereo, any sample rate); the input must keep at least one 960-sample hop
(40 ms @ 24 kHz) after auto-gain + silence trim or encode fails with
`input too short after preprocessing`. Without `-o` the output path is the
input with its extension swapped — decoding `clip.rvq` would overwrite
`clip.wav` — so name the round-trip output explicitly:

```sh
# self-contained round trip using the quick start's hello.wav
zig build omnivoice -Doptimize=ReleaseFast -- codec --model $CODEC -i hello.wav
zig build omnivoice -Doptimize=ReleaseFast -- codec --model $CODEC -i hello.rvq -o hello.rt.wav
```

Using an `.rvq` file for cloning still requires `--ref-text` alongside
`--ref-rvq` (the codes carry no transcript).

## Determinism and dtype notes

- With `--seed N ≥ 0`, F32 runs are token-reproducible and match the C++
  reference byte-for-byte. `--seed -1` (default) draws a random seed.
- Quantized bases (Q8_0/Q4_K_M) are seeded-deterministic *per build* but
  produce different (equally valid) token streams than F32 or other
  implementations — activation quantization makes bit-level agreement across
  implementations impossible by construction.
- BF16: use Fucina's, not the reference's — ggml's CPU BF16 matmul is ~33×
  slower; ours runs at F32-class speed.
- Threads default to 8 (`FUCINA_MAX_THREADS` lowers at runtime,
  `-Dmax-threads` raises the build ceiling).

## Debug / parity harness

`--dump <dir>` writes the reference-named tensor dumps (chunk 0);
`--llm-test <in.bin>` and `--maskgit-test` mirror the reference's oracle modes;
`compare` / `compare --raw` computes cosine similarity between dump files.
Heavy parity suites are env-gated: `OMNIVOICE_PARITY=1 zig build test
-Doptimize=ReleaseSafe` (they need `models/omnivoice/*.gguf` plus golden
dumps captured locally from the omnivoice.cpp reference via its oracle modes
and this CLI's `--dump`). The audio-device test (real playback device) is likewise gated:
`OMNIVOICE_AUDIO_DEVICE_TESTS=1 zig build test`.

`tools/fetch_refs.sh omnivoice.cpp` clones the pinned reference into the
gitignored `refs/` tree; the streaming-parity goldens are read from
`refs/omnivoice-research/goldens/tts-stream/`, and every gated suite skips
cleanly when goldens or `models/omnivoice/*.gguf` are absent.
