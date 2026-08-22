# testdata/qwen3tts — Qwen3-TTS parity fixtures

Reference activations for `src/llm/qwen3tts`, consumed by
`src/llm/qwen3tts/codec_tests.zig` and `talker_tests.zig`.

Produced by driving the pinned C++ reference through its python bindings.
Both stay stock — the harness runs against them, nothing is patched:

```sh
tools/fetch_refs.sh qwentts.cpp qwentts-cpp-python
```

Build the reference and run its dump entry points with the same prompt,
speaker and seed the tests assert on (`greedy` = temperature 0, `seed42` =
the seeded sampling path). The `.wav` renders alongside the `.bin` dumps are
listening references only — they are gitignored and not asserted on.

## Layout

| directory | contents |
|---|---|
| `dump-codec/` | codec decoder stages: `codec-preconv`, `codec-tfm`, `codec-rvq`, `codec-up` |
| `dump-greedy/` | greedy talker run: prompt ids, input embedding, per-layer prefill hidden states (`l0`, `l7`, `l14`, `l21`, `l27`, `final`), step-1 hidden, step-0 and full code frames, output audio |
| `dump-seed42/` | the same trace under seeded sampling |

The per-layer prefill taps are what make a talker mismatch bisectable: a
divergence at `l7` but not `l0` localizes to the layers between them, rather
than surfacing only as wrong audio at the end.

All fixtures are raw little-endian f32 (`.bin`); the tests read shapes from
the model config rather than from a header. They skip cleanly when absent.
