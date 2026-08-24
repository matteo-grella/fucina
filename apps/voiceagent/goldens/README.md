# apps/voiceagent/goldens — GTCRN-AEC parity fixtures

Per-stage reference activations for the echo canceller in
`apps/voiceagent/aec.zig`, consumed by `apps/voiceagent/aec_tests.zig`.

They ship with the reference rather than being generated here: LocalVQE
publishes them under `ggml/tests/gtcrn/`, validated `<1e-4` against its
PyTorch implementation. Copy them in after pinning the reference:

```sh
tools/fetch_refs.sh LocalVQE
cp refs/LocalVQE/ggml/tests/gtcrn/*.npy apps/voiceagent/goldens/
```

The weights are NOT in that checkout — they are published separately, and
`tools/fetch_voice_models.sh` fetches them:

```sh
hf download LocalAI-io/LocalVQE localvqe-pi-aec-v1-49k-f32.gguf --local-dir models/aec
ln -s localvqe-pi-aec-v1-49k-f32.gguf models/aec/gtcrn_aec.gguf
```

## What each file gates

| file | stage |
|---|---|
| `in_spec_e` / `in_spec_y` | input spectra — mic (echoic) and far-end reference |
| `feat` | ERB sub-band + SFE feature frame |
| `enc0`–`enc4` | encoder blocks (two conv, three GT) |
| `dpgrnn1` / `dpgrnn2` | dual-path grouped RNN outputs |
| `dec0`–`dec4` | decoder blocks with skip adds |
| `mask` | complex ratio mask over the full bin range |
| `out_spec` | masked output spectrum — the end-to-end gate |
| `stft_in` / `stft_out` / `istft_out` | windowed-DFT framing, pinned separately |

The test iterates the streaming `Session` frame by frame, so it gates the
deployment path — carried conv history and GRU hiddens included — not just an
offline forward. Every stage holds to `1e-3`; `out_spec` additionally requires
cosine `>= 0.99999`. It skips cleanly when the fixtures or the model are
absent.
