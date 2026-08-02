# qwen3tts — Qwen3-TTS text-to-speech from GGUF

The Zig port of qwentts.cpp (Qwen3-TTS-12Hz): a 28-layer Qwen3-shaped
**talker** samples codebook-0 tokens, a 5-layer MTP **code predictor** fills
codebooks 1–15, and the 12.5 Hz **codec** (RVQ dequant → sliding-window-72
transformer → ConvNeXt upsample → DAC v2) renders 1920 samples per frame at
24 kHz. CustomVoice preset speakers; CPU only.

## Run

```sh
echo "Hello from Fucina." | zig build qwen3tts -Doptimize=ReleaseFast -- \
    --model models/qwen3-tts/qwen-talker-0.6b-customvoice-F32.gguf \
    --codec models/qwen3-tts/qwen-tokenizer-12hz-F32.gguf \
    --speaker Aiden -o hello.wav
```

Flags: `--speaker`, `--lang` (default english), `--seed N` (printed for
replay) / `--greedy`, `--max-new`, `--chunk-frames`/`--left-ctx` (streamed
codec decode geometry), `--threads N`, `-o out.wav`.

Models: `Serveurperso/Qwen3-TTS-GGUF` — one talker GGUF
(`qwen-talker-0.6b-customvoice-*`) plus the shared codec GGUF
(`qwen-tokenizer-12hz-*`). The F32 pair is the parity reference.

## Parity

The port is pinned against the qwentts.cpp oracle (`refs/qwentts.cpp`,
CPU, `--no-fa`): greedy generation matches **token-for-token** (37×16 frames,
natural EOS) and seeded sampling replays the oracle **draw-for-draw**
(Philox4x32-10, one subsequence per primitive sample); the codec decodes the
golden RVQ stream at ≥0.9999 waveform cosine with per-stage probes
(`src/llm/qwen3tts/*_tests.zig`, model-gated). On an M1 Max the 0.6B F32
talker generates ~13 fps against the 12.5 fps real-time budget and the codec
decodes at ~0.5× RTF one-shot.
