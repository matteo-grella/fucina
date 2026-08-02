# voiceagent — native cascade voice agent in a TUI

Microphone → **Parakeet** streaming STT (learned `<EOU>` endpointing) →
in-process **Qwen3** chat → **Qwen3-TTS** → speakers. Every stage is a Fucina
port running in one process — no Python, no server, no browser. The
architecture mirrors huggingface/speech-to-speech's VAD→STT→LLM→TTS cascade
with the fucina-native stages substituted (the EOU model replaces Silero VAD's
endpointing role).

## Run

Four stages, four GGUFs. `--asr` and `--chat` are always required; the TTS
side is either a Qwen3-TTS talker plus its `--codec`, or a single Pocket TTS
model with no codec at all.

**Recommended default** — 4B chat obeys spoken style, Q8 talker runs ~41 fps
against a 12.5 fps budget:

```sh
zig build voiceagent -Doptimize=ReleaseFast -- \
    --asr models/parakeet/realtime_eou_120m-v1-f16.gguf \
    --chat models/Qwen3-4B-Q8_0.gguf \
    --tts models/qwen3-tts/qwen-talker-0.6b-customvoice-Q8_0.gguf \
    --codec models/qwen3-tts/qwen-tokenizer-12hz-F32.gguf
```

**Best voice** — the 1.7B talker at ~14.5 fps: real time, with the occasional
cushion pause under heavy load:

```sh
zig build voiceagent -Doptimize=ReleaseFast -- \
    --asr models/parakeet/realtime_eou_120m-v1-f16.gguf \
    --chat models/Qwen3-4B-Q8_0.gguf \
    --tts models/qwen3-tts/qwen-talker-1.7b-customvoice-Q8_0.gguf \
    --codec models/qwen3-tts/qwen-tokenizer-12hz-F32.gguf \
    --speaker Aiden --lang english
```

**Snappiest** — Pocket TTS (no codec) with a small quantized chat model.
Measured 0.6 s from end-of-utterance to first audio, against 4.3 s for the
recommended pairing above:

```sh
zig build voiceagent -Doptimize=ReleaseFast -- \
    --asr models/parakeet/realtime_eou_120m-v1-f16.gguf \
    --chat models/Qwen3-1.7B-Q4_K_M.gguf \
    --tts models/pocket-tts/pocket-tts-english-v2-ptqtp-k3tie.gguf \
    --voice alba
```

The chat model dominates that latency: holding the TTS fixed, 4B-Q8 gives
3.8 s and 1.7B-Q4_K_M gives 0.6 s. Dropping to 0.6B reaches 0.3 s but the
replies get noticeably worse. The `--tts` GGUF's architecture selects the
engine, so switching families is just a path change — drop `--codec` for
Pocket, which needs none. The tied-PTQTP Pocket build (141 MB, RTF 0.61)
also cuts startup: 0.3 s of model loading versus 3.4 s for the full stack.
The F32 Qwen3-TTS talker is a parity reference — it barely holds real time.

There is no faster ASR: the quantized `tdt_ctc-110m` checkpoints are not
streaming models and are rejected, and the only other accepted streaming
model is 5× larger.

## Getting the weights

Five artifacts from four sources, one of which needs a local conversion, so
there is a script:

```sh
tools/fetch_voice_models.sh            # fast tier: Pocket TTS, no codec
tools/fetch_voice_models.sh quality    # Qwen3-TTS CustomVoice + codec
tools/fetch_voice_models.sh both
```

It needs the Hugging Face CLI (`pip install -U huggingface_hub`), plus
`safetensors` and `sentencepiece` for the Pocket tier — kyutai ships
safetensors rather than GGUF, so `tools/pocket/pocket_to_gguf.py` packs it.
Re-running is cheap; it skips what you already have and prints the command to
run at the end.

By hand, if you prefer: parakeet and Qwen3 GGUFs as used by their own
examples; [`LocalAI-io/LocalVQE`](https://huggingface.co/LocalAI-io/LocalVQE)
for the canceller; `Serveurperso/Qwen3-TTS-GGUF` for the talker plus the
`qwen-tokenizer-12hz` codec (see `examples/qwen3tts/README.md`); Pocket TTS
from `kyutai/pocket-tts-without-voice-cloning` (see
`examples/pockettts/README.md`). Weights are not redistributed here and carry
their own terms — see [`docs/THIRD-PARTY-NOTICES.md`](../../docs/THIRD-PARTY-NOTICES.md).

## Flags

| flag | effect |
|---|---|
| `--asr <gguf>` | Parakeet streaming STT with learned `<EOU>` endpointing (required) |
| `--chat <gguf>` | Qwen3 chat model (required) |
| `--tts <gguf>` | talker or Pocket model; its architecture picks the engine (required) |
| `--codec <gguf>` | RVQ codec decoder — Qwen3-TTS only, omit for Pocket |
| `--speaker NAME` | CustomVoice speaker for Qwen3-TTS (default `Aiden`) |
| `--voice NAME` | Pocket TTS voice, `alba` or `marius` (default `alba`) |
| `--lang NAME` | synthesis language (default `english`) |
| `--system PROMPT` | system prompt; keep it spoken-style, the reply is read aloud |
| `--seed N` | sampling seed for the talker |
| `--max-reply N` | cap on reply tokens |
| `--threads N` | worker threads (default: detected cores) |
| `--mic-device N` | capture device index (default: system default) |
| `--list-devices` | print capture devices and exit |
| `--aec <gguf>` | echo-canceller model (default `models/aec/gtcrn_aec.gguf`) |
| `--no-aec` | disable echo cancellation — falls back to half-duplex, no barge-in |
| `--aec-debug` | trace delay-lock and barge-gate decisions to stderr; the first thing to read when barge-in does nothing |
| `--no-tools` | disable local tool calling |
| `--eager-text` | print the reply during THINK instead of revealing it in sync with the voice |
| `--rail-ascii` | force the Signal Rail's ASCII profile |
| `--prompt-color C` | colour of the `❯` prompt mark (default `white`): a name — `white`, `black`, `red`, `green`, `yellow`, `blue`, `magenta`, `cyan`, `gray` — all rendered bold, or a raw SGR parameter string such as `38;5;208` or `2;37`. `NO_COLOR` drops the escape entirely. |

Headless-sim flags are documented under [Headless end-to-end sim](#headless-end-to-end-sim).
Enter interrupts a reply; Ctrl-C quits.

## Turn-taking and echo cancellation

FULL-DUPLEX with barge-in, layered the way production stacks do it:

**Acoustic layer.** One 48 kHz duplex stream whose realtime callback plays the
TTS and mirrors the exact played frames into a reference ring sample-aligned
with the mic (mic and reference are pushed or dropped as a pair — alignment
is an invariant of the sole producer). Both decimate ÷3 with the same FIR to
16 kHz and run through the **GTCRN-AEC** echo canceller (Zig port of
LocalVQE's 49 K-param AEC-aware GTCRN, Apache-2.0, fixture-parity tested —
`aec.zig`). A **bulk-delay estimator** cross-correlates mic against reference
(0–300 ms search, lag-histogram vote for reverberant rooms) and delays the
reference to the locked lag before the canceller — the alignment WebRTC
pairs with every AEC; on (re)lock the canceller state resets and the gate
holds ~0.5 s for reconvergence (the STT is fed zeros through the transient).

**Text layer — the robustness principle: we know the exact text being
spoken.** The streaming STT runs on the AEC residual during THINK, LISTEN and
pause windows. It deliberately does NOT run while the agent is audibly
speaking: the talker keeps its whole real-time budget, and playback-era echo
can never be transcribed, by construction. Residual audio captured during
speech is buffered and flushed once a pause opens, so an interruption's
opening words are not lost. Only words that are NOT a fuzzy whole-word match
(edit distance ≤ 1) of the reply being spoken count as the user. Echo
mis-transcriptions reproduce the reply's own words and are filtered, so even
an underperforming canceller cannot false-fire the turn. Backchannels
("yeah", "okay", …) never interrupt; command words ("stop", "wait", "no", …)
interrupt fastest.

**Pause-then-commit.** Sustained residual energy (~200 ms, canceller locked)
PAUSES playback — killing the echo path so the STT hears cleanly — while
generation keeps queueing; the barge COMMITS only on confirming novel words
(consecutive, stable across updates, energy-coincident). No confirmation
within ~2 s resumes playback: a false trigger costs a hiccup, not the reply.
On commit the same STT session carries into LISTEN — the interruption
continues seamlessly to its EOU, with pre-commit echo words sliced off. A
text-level echo guard (stricter inside the 2 s turn-boundary echo-tail
window) drops residual self-transcriptions before they reach the chat.

Enter also interrupts; Ctrl-C quits. `--no-aec` (or a missing
`models/aec/gtcrn_aec.gguf`) falls back to half-duplex. The canceller weights
are the compact GTCRN-AEC line from
[`LocalAI-io/LocalVQE`](https://huggingface.co/LocalAI-io/LocalVQE)
(`localvqe-pi-aec-v1-49k-f32.gguf`, 2.3 MB); the parity fixtures in
`goldens-aec/` ship with the reference checkout, not with the weights.

**The prompt line.** Your turn opens with a `❯` mark on screen before you say
anything, and the live transcript fills in after it as the STT emits tokens —
so the words you watch forming are the same line that settles as the committed
turn, rather than a separate preview that gets replaced. It is transient: a log
line erases it, prints, and redraws it underneath, so nothing is ever stranded
in the scrollback. The agent's reply carries no mark at all. `--prompt-color`
sets the mark's colour.

**Typing.** The keyboard is a second path into the same turn, and the live
transcript is a *suggestion*: it renders dim, and taking it is explicit.

| key | while the suggestion is dim | while typing |
|---|---|---|
| `Enter` | send the transcript as it stands, without waiting for the endpointer | send the line |
| `→` / `End` / `ctrl-E` | take it into the line to edit (turns plain) | — |
| any character | start a clean line; the suggestion goes away | insert |

**Enter means send in both states.** That is what keeps `→` from being a trap:
accepting changes what you can *edit*, never how you *send*, so there is no
state in which nothing submits. The first couple of turns show `⏎ send  → edit`
dimmed beside the suggestion, then stop.

Typing takes the turn off endpointing — an EOU firing mid-sentence would submit
half a line — which is why Enter has to work everywhere. `ctrl-W` deletes the
last word, `ctrl-U` clears, backspace removes a whole character (not a byte).
Other escape sequences are swallowed rather than inserted. Typed text skips the
echo guard, which only applies to transcripts.

A keystroke while the agent is speaking interrupts it, as Enter always has;
the characters stay queued and open the next turn's line, so you can simply
start typing over a reply. Enter on an empty line interrupts and nothing more.
The terminal is put in raw mode (echo and canonical input off, `ISIG` left on
so Ctrl-C still works) and restored on exit, including on Ctrl-C.

**Synced text reveal.** The reply appears word-by-word in sync with the
voice — the screen and the speech are one event, driven by the playback clock
(samples pushed minus samples queued), so a barge pause holds the text and an
interrupt cuts it where the voice stopped, leaving the unspoken remainder
dimmed for the record. The producers flag voiced samples as they push, and
the text maps onto the VOICED window: leading silence delays nothing, the
silent pad before EOS stretches nothing, and each word is shown at its
onset (the word being spoken reveals whole). Timing is estimated (neither
TTS engine emits word timestamps): a fixed speaking-rate guess paces the
first words, then the reveal converges on exact proportion over the voiced
span. `--eager-text` restores the stream-at-THINK display (full text on
screen before the voice starts).

## The Signal Rail

The status row is a **Signal Rail** (Signal Rail 1.0, scaled to this agent):
a deterministic instrument display with spatial semantics — input zone (28%),
processing (44%), output (28%):

```text
LISTENING   [───▪▪■███■▪▪─────────────────────────────────────]
THINKING    [──────────────▐──▪■─▪────▪▪▪─▪▪■▪▪▪▪─────────────]  T+02.5
SPEAKING    [────────────────────────────────────▪■▐──▪▪■▐──▪▪]
WAITING     [▪──▪───▪▪──▪──────────────────│──────────────────]  HOLD
INTERRUPTED [■■■■■■■■■■■■■■┃──────────────────────────────────]  CUT
```

Listening expands from the input origin with the REAL quantized residual
level (rise ≤2 / fall ≤1 per 12 Hz tick); captured input collapses; thinking
runs a read head over a per-pass deterministic field (seeded by the turn);
speaking emits packets rightward at the real output level's spawn rate; the
pause-then-commit hold freezes behind a pulsing boundary; barge-in retracts
and hard-cuts; a finished reply sweeps once. Frames are a pure function of
(state, entry tick, tick, width, levels, seed) — `rail.zig`, snapshot-tested
at 25 cells in ASCII. `--rail-ascii` forces the ASCII profile; `NO_COLOR`
switches to monochrome (dim/normal/bold carry the intensities). Width picks
the largest preset (25/37/49/61) that fits the terminal.

Deviations from the full spec, deliberate: no 256/16-color modes (truecolor
+ monochrome only); no reduced-motion tier (normal + static via `NO_COLOR`
pipes); glyph width probing replaced by the `--rail-ascii` escape hatch. Fatal
errors play the full ERROR entry (saturate → blackout → settled fracture,
aux `FATAL`) and leave the fracture above the shell prompt on exit.

## Tools

The chat model can call local tools (Hermes format, on by default;
`--no-tools` disables): `set_timer` runs a real countdown under the rail's
**ACTING** state with true determinate progress (`042%` aux — never faked),
cancellable by voice or Enter; `get_time` flashes indeterminate ACTING. A
reply that ends with a question routes COMPLETE -> **NEEDS INPUT** (the
control-returned state, `INPUT` aux) until you speak. Every state in the
rail grammar is driven by real agent behavior.

## Headless end-to-end sim

`--sim user.wav [--sim-barge over-reply.wav]` (PCM16 mono 24 kHz, e.g. from
`fucina-qwen3tts`) swaps the audio device for a scripted driver that calls the
same realtime callback on a real-time clock: the synthetic mic carries the
user WAV, a noise floor, and a delayed attenuated copy of the frames the
agent actually played — a true echo path, so the AEC must cancel it or the
barge gate false-fires — and the second WAV is injected once the reply has
played ~1.2 s to exercise barge-in (`--sim-barge-after-ms N` moves it earlier;
200 ms is the case that talks over the very start of a reply, before the delay
estimator has locked). The echo path is live-realistic: 120 ms bulk delay
(`--sim-echo-ms N`), multi-tap room reflections, a tanh speaker
nonlinearity, and a coloring lowpass; `--sim-echo-step` adds a +10 ms delay
step mid-run to exercise relocking. The run is deterministic and ends with a
summary line (duration, injection time, interrupts, underruns). The whole
STT → chat → TTS → barge-in → resume loop is exercised with no audio
hardware: echo-only runs must report `interrupts 0`, barge runs must show
`[interrupted]` and transcribe the interrupting words via the pre-roll.

| flag | effect |
|---|---|
| `--sim <wav>` | replace the audio device with the scripted driver; this WAV is the user's first utterance |
| `--sim-barge <wav>` | second utterance, injected over the reply to exercise barge-in |
| `--sim-barge-after-ms N` | voiced playback before injection (default 1200); 200 talks over the very start of a reply, before the delay estimator has locked |
| `--sim-again <wav>` | inject the second WAV as a fresh turn in quiet LISTEN instead of over the reply |
| `--sim-echo-ms N` | bulk echo delay (default 120) |
| `--sim-echo-step` | step the delay +10 ms mid-run to exercise relocking |

The three runs worth keeping green:

```sh
# 1. echo only — the false-positive gate; must report interrupts 0
./zig-out/bin/fucina-voiceagent --asr … --chat … --tts … --sim user.wav

# 2. barge over a settled reply
./zig-out/bin/fucina-voiceagent --asr … --chat … --tts … \
    --sim user.wav --sim-barge over-reply.wav

# 3. barge before the delay estimator locks — the hardest case
./zig-out/bin/fucina-voiceagent --asr … --chat … --tts … \
    --sim user.wav --sim-barge over-reply.wav \
    --sim-barge-after-ms 200 --aec-debug
```

## Output pipeline

The talker and the codec run as a two-stage pipeline: generation streams
frames on the main thread while a dedicated worker decodes chunk N as chunk
N+1 is generated (the codec's 25-frame left context makes small serial
chunks ~3× overpriced; steady-state chunks are 36 frames after a fast
12-frame first chunk). Playback holds behind a 2 s warm-up cushion per
reply, and if the producer ever falls behind mid-reply the cushion re-arms —
one clean pause instead of stutter. The `[turn]` line reports `audible gaps`
honestly (ring-starvation events while speaking); on an unloaded M1 Max the
F32 talker sustains real time and gaps stay 0.

## Status

The full conversation loop (both turns, barge-in interrupt, pre-roll
continuity, zero underruns) is exercised headlessly by the `--sim` driver
through the production callback; live device I/O on top of it is manually
verified (CI has no audio device). The model stages are pinned by the
qwen3tts and pockettts parity suites, the GTCRN-AEC fixture gates, and the
parakeet/qwen3 golden tests.

Against a 12.5 fps real-time budget on an M1 Max, the quantized Qwen3-TTS
talkers run with headroom (~41 fps at 0.6B-Q8, ~14.5 fps at 1.7B-Q8) while
the F32 talker sits at ~13 fps — real time only just, which is why it is a
parity reference rather than a serving tier. The codec decodes at ~0.5× RTF,
so speech starts after roughly one chunk is synthesized. Pocket TTS needs no
codec at all and starts as soon as the chat replies.

## Acknowledgements

Every stage here is a Fucina port; the architectures, and in several places
the operation-for-operation structure, are upstream's.

- **Echo cancellation** — [LocalVQE](https://github.com/localai-org/LocalVQE)
  (Apache-2.0) for the GTCRN-AEC reference implementation, the pretrained
  49 K-parameter canceller, and the per-stage fixtures that gate this port.
  The architecture is [GTCRN](https://github.com/Xiaobin-Rong/gtcrn)
  (ICASSP 2024).
- **Speech recognition** — NVIDIA NeMo Parakeet / FastConformer, with the
  learned `<EOU>` endpointing model standing in for a VAD.
- **Chat** — Qwen3 (Alibaba, Apache-2.0).
- **Speech synthesis** — [Qwen3-TTS](https://github.com/andimarafioti/qwentts.cpp)
  (MIT) for the C++ reference this port follows, and
  [kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts) (MIT) for the
  flow-matching model and its Mimi decoder.
  [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) (MIT)
  informed the streaming decode decomposition.
- **Cascade shape** — the VAD → STT → LLM → TTS structure follows
  huggingface/speech-to-speech, with fucina-native stages substituted.
- **Turn-taking** — the pause-then-commit ladder is the pattern production
  voice stacks (LiveKit, Pipecat) converged on; the bulk-delay estimator in
  front of the canceller is the alignment step WebRTC pairs with every AEC.

Full provenance, licenses and the port-versus-reference distinction for each
of these is in [`docs/THIRD-PARTY-NOTICES.md`](../../docs/THIRD-PARTY-NOTICES.md).
Weights are not redistributed here and carry their own terms — note in
particular that kyutai publishes a use policy for Pocket TTS voices
prohibiting cloning without consent.
