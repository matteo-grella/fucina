# pockettts — Pocket TTS v2 (kyutai) in pure Zig

Streaming text-to-speech: a ~100M prefix-LM transformer emits continuous
32-dim latents (one 80 ms frame per AR step, no codebooks), each sampled by
a 1-step flow-matching MLP head and decoded by a VAE-style Mimi
(depthwise-transposed upsample 12.5→200 Hz → 2-layer sliding-window
transformer → SEANet causal decoder) at 24 kHz. Voices are KV-cache
prefixes. Weights CC-BY-4.0 (kyutai/pocket-tts-without-voice-cloning).

## Run

```sh
zig build pockettts -Doptimize=ReleaseFast -- \
    --model models/pocket-tts/pocket-tts-english-v2.gguf \
    --voice alba -o out.wav < text.txt
```

Flags: `--voice` (alba/marius as converted), `--temp` (default = the model's
recommended 0.3), `--seed`, `--eos-threshold`, `--max-frames`.

Model: convert the HF checkpoint with `tools/pocket/pocket_to_gguf.py`
(safetensors + SentencePiece pieces/scores + per-voice KV caches → one f32
GGUF). That script wants `safetensors sentencepiece torch numpy` — a one-time
cost to produce the file, not a dependency of the runner: the tokenizer ends
up inside the GGUF, so the Zig side reads it from there and links no Python.

There is no prebuilt GGUF to download instead. Third-party ones exist
(`idle-intelligence/pocket-tts-gguf`) but target other runtimes and ship a
different container: the tokenizer stays a separate `tokenizer.model` file and
the voices are left out entirely, where this loader requires both inside the
GGUF (`tokenizer.pocket.tokens`/`.scores`, `voice.<name>.cache.<layer>` —
roped-K caches, not raw embeddings). It is smaller than the K3-tie build
(128 MB Q8_0 against 148 MB) but comparable to K2-tie, and neither is
loadable here without teaching this module a second schema and sourcing the
voices separately.

Publishing one built by this converter would be permitted: the model is
CC-BY-4.0, and the two voices in the English pack are alba (CC BY 4.0) and
marius (CC0). Attribution and kyutai's use policy would have to travel with
it, and note that other voices in kyutai's catalog are CC BY-NC — converting
those would make the artifact non-commercial. See
[`docs/THIRD-PARTY-NOTICES.md`](../../docs/THIRD-PARTY-NOTICES.md).
RTF ≈ 0.7 single-threaded on an M1 Max (56 ms per 80 ms frame); first audio
after one AR step.

## PTQTP variants

`fucina-export-gguf --ptqtp=K --ptqtp-tie --ptqtp-include transformer.layers`
quantizes ONLY the two transformers (backbone + Mimi decoder transformer) to
tied ternary trit-planes; the flow head, EOS head, and all convolutions stay
f32, so quality survives even K=2 (word-identical round-trip transcripts):

| variant | file | RTF (M1 Max, 1 thread) |
| --- | --- | --- |
| f32 | 411 MB | 0.77 |
| K3-tie | 148 MB | 0.65 |
| K2-tie | 127 MB | 0.61 |

K3 and K2 are within run-to-run noise of each other on wall clock (repeated
timings land on the same number), so the choice is size against quality
margin, not speed. Both round-trip word-identically through Parakeet; K2
drifts further from the f32 reference on signal statistics (zero-crossing rate
+15% vs +6% for K3), which is a hint of added high-frequency roughness rather
than a defect you can hear in a transcript. K3-tie is the default here for
that margin; K2-tie is the pick when 22 MB matters.

The module serves `.ptqtpK` plane tensors directly (Q8_K activation quant +
the stock TQ2_0 row kernels — `pocket.zig`'s `Linear` union).

## Parity

`src/models/pockettts/pocket_tests.zig` pins the port against per-stage dumps
of the PyTorch reference (`refs/pocket-tts-dump-smoke.py`): tokenizer
token-for-token, text-pass backbone rows, flow head at two checkpoints with
the reference's noise injected (torch RNG is never reproduced), the latent
feedback row, and all four Mimi stages of the first frame. Long texts split
at sentence boundaries into ≤50-token chunks against the same voice prefix
(the reference's own chunking discipline).
