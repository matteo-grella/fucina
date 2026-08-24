# voiceagent — native cascade voice agent in a TUI

Microphone → **Parakeet** streaming STT (learned `<EOU>` endpointing) → a chat
model behind an OpenAI-compatible endpoint → **Qwen3-TTS** or **Pocket TTS** →
speakers. Every stage is a Fucina port, and by default all of them run inside
one process — no Python, no browser, nothing to start beforehand. The
architecture mirrors huggingface/speech-to-speech's VAD→STT→LLM→TTS cascade
with the fucina-native stages substituted (the EOU model replaces Silero VAD's
endpointing role).

## Run

Four stages, four GGUFs. `--asr` is always required, as is a chat backend —
either `--chat` for one the agent hosts itself or `--chat-url` for one already
running. The TTS side is either a Qwen3-TTS talker plus its `--codec`, or a
single Pocket TTS model with no codec at all.

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

**Snappiest** — Pocket TTS (no codec) with a small quantized chat model:

```sh
zig build voiceagent -Doptimize=ReleaseFast -- \
    --asr models/parakeet/realtime_eou_120m-v1-f16.gguf \
    --chat models/Qwen3-1.7B-Q4_K_M.gguf \
    --tts models/pocket-tts/pocket-tts-english-v2.gguf \
    --voice alba
```

Last word to first audio *heard* — endpointing included — on an M1 Max under
`--sim`, with Qwen3-1.7B-Q4_K_M served by the in-process chat server and Pocket TTS speaking,
median of 3 runs:

| | short question (1.8 s) | long question (10.2 s) |
|---|---|---|
| total | **1.14 s** | **1.15 s** |
| of which endpointing | 0.16 s | 0.58 s |

Both fixtures are synthesised, so the pair reproduces exactly:

```sh
printf 'What is the capital of France?' \
  | ./zig-out/bin/fucina-pockettts \
      --model models/pocket-tts/pocket-tts-english-v2.gguf \
      --voice alba --seed 7 -o short.wav
```

The reply is spoken sentence-by-sentence, so what matters after that is how
fast the chat model writes its FIRST sentence, not the whole reply — which is
why the total barely moves between a two-second question and a ten-second one.
The chat model dominates the remainder, and the 4B keeps some dependence on the
question because it writes a longer opening sentence for it. The Qwen3-TTS tier
is slower to first audio — ~3.3 s and ~4.9 s for the
same pair with the 0.6B-Q8 talker — because it speaks the reply in one span
behind a 2 s cushion (see [Output pipeline](#output-pipeline)); it buys voice
quality, not latency. Treat these as indicative — the spread across runs is a
couple of hundred milliseconds. Measure your own pairing with `--sim` before
trusting a number.
The `--tts` GGUF's architecture selects the engine, so switching families is
just a path change — drop `--codec` for Pocket, which needs none, and skip
loading the codec entirely. Quantizing Pocket further with
`--ptqtp --ptqtp-tie` (see `examples/pockettts/README.md`) takes it from
411 MB to 148 MB for a few percent of synthesis time — worth it for
distribution, not for latency, since the chat model dominates either way.
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

It needs the Hugging Face CLI (`pip install -U huggingface_hub`). Re-running is
cheap: it skips what you already have and prints the command to run at the end.

The Pocket tier additionally converts a checkpoint, because kyutai ships
safetensors rather than GGUF, and that conversion needs
`safetensors sentencepiece torch numpy`. **That is a one-time cost to produce
the file, not a dependency of the agent.** The tokenizer's 4000 pieces and
scores are written into the GGUF metadata and read back from there at load, so
once the GGUF exists you can uninstall all four — and a machine handed a
prebuilt GGUF never needs them at all. The running agent links no Python, in
this tier or any other. The script checks those imports before it starts
writing and names whatever is missing; set `PYTHON=<venv>/bin/python` if they
live in a virtualenv. The `quality` tier involves no conversion.

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
| `--chat <gguf>` | chat model, served in-process by the `models.text.serving` engine (qwen3/qwen3moe/gemma4 GGUFs) |
| `--chat-url URL` | speak `/v1/chat/completions` to an existing server instead of spawning one |
| `--endpoint-ms N` | quiet + transcript-stable window that closes a turn (default 800; 0 = wait for `<EOU>` only) |
| `--speculate-ms N` | quiet before generation starts speculatively, through that window (default 240; 0 = off) |
| `--chat-port N` | loopback port for the hosted server (default: first free in 8137-8168) |
| `--chat-model NAME` | model name sent in the request (default: the GGUF's stem) |
| `--chat-api-key K` | `Authorization: Bearer` for `--chat-url` |
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
| `--speaker-device N` | playback device index (default: system default); mic and speaker may sit on different hardware clocks — the AEC's delay tracker absorbs the drift |
| `--list-devices` | print capture and playback devices (system defaults marked) and exit |
| `--aec <gguf>` | echo-canceller model (default `models/aec/gtcrn_aec.gguf`) |
| `--no-aec` | disable echo cancellation — falls back to half-duplex, no barge-in |
| `--aec-debug` | trace delay-lock and barge-gate decisions to stderr; the first thing to read when barge-in does nothing, or when a reply keeps stalling |
| `--barge-floor RMS` | absolute residual level below which nothing counts as speech (default `0.010`). Raise it if replies stall on their own echo — see [When the reply keeps stalling](#when-the-reply-keeps-stalling) |
| `--eou-probe` | print the endpointer's per-frame P(`<EOU>`) trajectory at each turn close — max, first threshold crossings, and the tail. Costs one softmax per encoder frame; see [Reading the endpointer](#reading-the-endpointer) |
| `--no-tools` | disable local tool calling |
| `--eager-text` | print the reply during THINK instead of revealing it in sync with the voice; also turns off sentence-at-a-time speaking (one TUI writer) |
| `--rail-ascii` | force the Signal Rail's ASCII profile |
| `--prompt-color C` | colour of the `❯` prompt mark (default `white`): a name — `white`, `black`, `red`, `green`, `yellow`, `blue`, `magenta`, `cyan`, `gray` — all rendered bold, or a raw SGR parameter string such as `38;5;208` or `2;37`. `NO_COLOR` drops the escape entirely. |

Headless-sim flags are documented under [Headless end-to-end sim](#headless-end-to-end-sim).
Enter interrupts a reply; Ctrl-C quits.

## Chat backend

The chat stage is an OpenAI-compatible endpoint, always. Either one the agent
hosts itself:

```sh
--chat models/gemma-4-26B-A4B-it-UD-Q6_K.gguf     # served in-process
```

or one you point it at: lmserve, llama.cpp, vLLM, a hosted provider:

```sh
--chat-url http://127.0.0.1:8080/v1 --chat-model my-model
```

The first form is **not a child process**: the `models.text.serving` engine
(`serving.open` plus the band's HTTP server and scheduler) runs on a thread
inside this binary. One process, one executable, nothing to find on `$PATH`
and nothing orphaned if the agent dies. The port is picked by probing for the first free one in 8137-8168 —
the registered band, deliberately not the ephemeral range (49152+) the OS hands
to outbound connections.

What that buys over calling `models.text.chat.Conversation` directly is the two things
it cannot do. It dispatches on the GGUF's `general.architecture` (qwen3,
qwen3moe, gemma4), where a `Conversation` is bound to one model type at
compile time; families outside that set are served externally through
`--chat-url`. And its generation can be **abandoned mid-flight**, which is what
speculative turns are built on — closing the request makes the server flip the
job's cancel flag.

The loopback socket costs microseconds and buys a hard boundary: the server
owns its own `ExecContext`, KV slots and worker team. The whole history rides
every request and the server's prefix cache does what an in-process KV would.
STT, TTS, echo cancellation and turn-taking remain in-process fucina ports.

## Turn-taking and echo cancellation

FULL-DUPLEX with barge-in, layered the way production stacks do it:

**Closing the turn.** `<EOU>`/`<EOB>` are tokens the endpointer model emits
when it is confident. There is no threshold in it to turn down and no earlier,
softer form of the signal to read instead — see [Reading the
endpointer](#reading-the-endpointer). On clean speech it is quick: **0.16 s**
after a short question's last word, **0.58 s** after a long one.

It is not guaranteed to arrive, so the turn also closes on evidence that can
be measured directly: the room has gone quiet AND the transcript has stopped
growing, both for `--endpoint-ms` (default 800). Requiring both matters — the
transcript lags the audio, so quiet alone would cut off a word still being
decoded. Whichever fires first closes the turn, which makes the window a bound
on the wait rather than the normal path: speech that ends cleanly commits well
inside it. `--endpoint-ms 0` removes it entirely.

The window commits on silence alone, with no evidence from the model, and that
is the trade it makes — an utterance broken by a pause longer than
`--endpoint-ms` becomes two turns. Lower it and mid-thought pauses start
becoming turn boundaries.

**Speculating through the window.** Whichever way the turn closes, the quiet
before it is dead time the chat model could have been working through, so at
`--speculate-ms` (240) of quiet the reply starts generating against the
transcript *as it stands*. If the user was only pausing, the request is
abandoned — closing it makes the server flip the job's cancel flag, so it stops
costing GPU almost immediately — and nothing was ever spoken. If the turn
closes and the transcript still matches what was speculated against, the reply
is already part-written and the `chat` term collapses.

The worker is a pure byte producer: it never touches the AEC pump, the rail,
the reveal or the TUI. Those have exactly one owner and it is the main thread,
which drains bytes under a mutex and does the sentence splitting itself.

The gain is conditional. It pays when the chat stage is slow relative to the
wait: a short question through a 4B served in-process reaches first audio in
**1.28 s**. It does nothing for a long question, whose transcript keeps
growing, so the speculation starts late or is abandoned and restarted with no
head start left; and nothing for a small model on a warm server, where the
`chat` term is already 0.04-0.15 s and there is nothing to hide.
`--speculate-ms 0` turns it off.

The commit itself always waits for the turn to close — only the generation
runs ahead. That is the difference from huggingface/speech-to-speech, which
soft-ends after 64 ms and *reopens* the turn if speech resumes, making the
commit early and the retraction its safety net. Running the generation alone
ahead cannot speak over someone who was merely pausing, and cannot shorten the
wait either. The turn line reports which path closed the turn and whether the
speculation was used:

```
[turn] closed-by EOU, speculation adopted (started 1, abandoned 0)
```

### Reading the endpointer

`--eou-probe` prints the endpointer's P(`<EOU>`) for every encoder frame
(80 ms) at each turn close. The sample is taken inside the RNN-T greedy loop
*before* the blank break, so the frames where blank still wins are recorded
too — it costs one softmax over a distribution the joint already produced.

The signal is a step, not a ramp: the model is either unfinished or completely
certain, with nothing in between.

```
[eou-probe] frames=140 (11.20s) max_p=1.000  p>=0.10@11.04s  p>=0.30@11.04s  p>=0.50@11.04s
[eou-probe] rise_frames(0.01<=p<0.50)=0  tail: … 10.88s=0.000 10.96s=0.000 11.04s=1.000 11.12s=0.000
```

`rise_frames` counts the frames whose probability sits between 0.01 and 0.50.
On an utterance that ends cleanly it is zero — the probability is 0.000 on
every frame, 1.000 on one, 0.000 after — and every threshold therefore crosses
on the same frame. A probability threshold fires exactly when the token does,
which leaves `--endpoint-ms` as the only lever that commits earlier than the
model's own verdict.

An utterance broken by a long internal silence can raise a single intermediate
frame that decays back to zero without committing. One sub-threshold frame is
noise, not an early warning.

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

**Text layer — the robustness principle: the agent knows exactly what text is
being spoken.** The streaming STT runs on the AEC residual during THINK, LISTEN and
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
`examples/voiceagent/goldens/` ship with the reference checkout, not with the weights.

### When the reply keeps stalling

A stage-1 hold is invisible to every other counter — it keeps the audio queue,
so no `audible gap` is recorded, and it commits nothing, so `interrupts` stays
0. It has a counter of its own: the `[turn]` line ends with
`holds N (M false, X s)`, counting stage-1 pauses, how many timed out with
nobody behind them, and the total time spent held. Repeated holds are what a
reply chopped into fragments looks like.

Holds fire on sustained residual energy, and the residual after cancellation
depends on your room, your speaker volume and your mic gain — not on the code.
A setup whose residual sits above the gate's absolute floor pauses the agent on
its own voice. `--aec-debug` prints the numbers at each hold:

```
[aec-dbg] t=6.60s STAGE1 pause (hot=12 res=0.0243 floor=0.0021 min=0.0100 ncc=0.684) — if nobody spoke, --barge-floor above res
```

If those fire while nobody is speaking, set `--barge-floor` above the reported
`res`. The cost is barge sensitivity: the floor is the level a voice has to
clear to interrupt, so raising it too far stops barge-in working: at
`--barge-floor 0.05` the sim's barge case reports `interrupts 0`. Headphones
remove the echo path entirely and make the question moot; `--no-aec` drops to
half-duplex and is the fastest way to confirm the gate is what you are hearing.

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

**The reply is spoken while it is still being written.** The chat model
writes 1–4 sentences; waiting for the last one before the talker starts
makes time-to-first-audio scale with *reply* length, which is the wrong
variable — the user is waiting on the first sentence. So the sentence
splitter hands each finished sentence to a speak worker as it completes, and
sentence N is audible while N+1 is still decoding. A reply that turns out to
be a `<tool_call>` is never spoken: dispatch stops at the first `<`, and
since Hermes opens with the tag, a tool turn reaches it before any sentence
terminator and hands over nothing.

Only Pocket takes the reply a sentence at a time. It already rewinds to its
voice prefix at every sentence internally, so a span costs it nothing;
Qwen3-TTS rebuilds a speaker-conditioned prompt and restarts the talker KV
per call, and its first audio waits on the cushion rather than on the chat
model, so it speaks the reply in one span — 3.97 s to first audio, against
4.42 s if the same reply is handed over a sentence at a time.
`--eager-text` also keeps the single span — it prints
the reply from the generator, and overlapping the talker would give the TUI
two writers.

The talker and the codec run as a two-stage pipeline: generation streams
frames while a dedicated worker decodes chunk N as chunk N+1 is generated
(the codec's 25-frame left context makes small serial chunks ~3×
overpriced). Playback holds behind a warm-up cushion per reply, and if the
producer ever falls behind mid-reply the cushion re-arms — one clean pause
instead of stutter.

**The cushion is per engine, because every sample of it is latency the user
waits through.** Pocket sustains ~20.8 fps against the 12.5 fps budget
(1.66× real time) and needs only 0.4 s. Qwen3-TTS is the talker plus the
chunked codec worker and clears ~1.27× end to end, so it needs the full 2 s:
at 0.4 s the long sim fixture breaks into 8 audible gaps. The `[turn]`
line reports `audible gaps` honestly (ring-starvation events while
speaking) and both tiers hold at 0 on an unloaded M1 Max.

`[turn]` reports `first-audio` — when a sample first reached the device,
stamped on the callback clock — split into the four terms that produce it:

```
[turn] first-audio 1.12 s (endpoint 0.16 + chat 0.59 [aec/stt 0.15] + tts 0.08 + cushion 0.30)
```

- **endpoint**: quiet after your last word before the turn closed. Measured on
  the pump's audio clock against a fixed voice floor, so neither machine load
  nor the barge gate's adaptive floor can distort it.
- **chat**: end of the turn to the first sentence worth speaking. `aec/stt` is
  the part of it spent inside `barge.scan()` from the generation loop — the
  canceller and residual STT run synchronously there, so a backlog is charged
  to the chat stage even though the model is not what is slow.
- **tts**: that sentence to the first frame queued.
- **cushion**: the warm-up hold before it became audible.

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
parity reference rather than a serving tier. Serialized with the chunked
codec the chain clears ~1.27× end to end. Pocket TTS needs no codec at all,
sustains ~20.8 fps (1.66× real time), and starts on the chat model's first
sentence rather than its last.

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
