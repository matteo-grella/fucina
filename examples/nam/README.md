# NAM — Neural Amp Modeler in Fucina

A self-contained, CPU-first port of the [Neural Amp Modeler](https://github.com/sdatkinson/neural-amp-modeler)
ecosystem: load and play `.nam` amp profiles live — optionally with cabinet IRs and
multi-stage signal chains — create your own profiles from captured audio, and exchange
profiles with the original NAM tooling in both directions. No plugin
host or DAW required — it runs against your normal audio devices from the terminal.
Parity numbers against the upstream core are quoted below; each deliberate
deviation from upstream is documented in the module header of the engine file
where it applies.

## Getting started (5 minutes to first sound)

**Prerequisites:** [Zig 0.16.0](https://ziglang.org/download/) and git. Tested on macOS /
Apple Silicon (the audio layer is vendored miniaudio, so Linux should work too but is
untested). No other dependencies.

```sh
git clone https://github.com/nlpodyssey/fucina && cd fucina
zig build nam -Doptimize=ReleaseFast        # builds zig-out/bin/fucina-zig-nam
```

`-Doptimize=ReleaseFast` matters: debug builds are ~20× slower and will not keep up in
realtime. You can run every command either through the build runner
(`zig build nam -Doptimize=ReleaseFast -- <command>`) or by calling the built binary
directly (`zig-out/bin/fucina-zig-nam <command>`) — the examples below use the short form.

**The simplest path — no flags at all:** put your `.nam` files in a folder named
`nam-profiles` (or `models`) next to the program, plug the guitar into your interface,
and run

```sh
zig build nam -Doptimize=ReleaseFast
```

You get a numbered amp menu; pick one, keep playing while it auto-detects the right
input, and you're live — input detection, same-device output, loudness normalization,
and the noise gate are all pre-configured. Everything below is the manual/expert path.

1. **Get a profile.** Download any `.nam` file — [Tone3000](https://www.tone3000.com)
   hosts thousands of free ones (they are almost all "standard WaveNet", which this player
   runs at full fidelity) — or use a profile you made with the official NAM trainer.
   Tone3000 also hosts cabinet IRs (`.wav`); add one with `--ir` (see
   [Cabinet IRs and signal chains](#cabinet-irs-and-signal-chains)).
2. **Find your audio interface:**
   ```sh
   fucina-zig-nam devices
   ```
   Note the index of your interface in *both* the capture and playback lists.
3. **Plug in and play** (see [Hardware](#hardware-what-plugs-into-what) below for how to
   connect the guitar):
   ```sh
   fucina-zig-nam live my-amp.nam --capture 2 --playback 2
   ```
   You should hear the processed guitar immediately. Keys: `space` bypass · `[` `]` or
   `1`–`9` switch profile/chain (pass several `.nam` files to A/B them, or build chains —
   see [Cabinet IRs and signal chains](#cabinet-irs-and-signal-chains)) · `a` auto-detect the
   input · `i` / `o` cycle the input / output device live (the status line names the
   active pair) · `,`/`.` input gain (how hard you drive the model — this changes the
   *tone*, not just volume) · `+`/`-` output gain · `t` tuner · `m` mute the output
   (processing keeps streaming, so unmute is click-free) · `n` toggle loudness
   normalization · `c` clear the clip indicator · `q` quit. A MIDI controller drives the
   same knobs — see [MIDI control](#midi-control).

   **Built-in tuner** (`t`, or start with `--tuner`): a strobe-class chromatic tuner runs
   on its own analysis thread off the raw input — the realtime model chain is never
   touched. Pluck one string for the needle (note, cents to one decimal, Hz; McLeod
   pitch detection + per-partial spectral refinement with inharmonicity fitting —
   measured well under 0.1 cent on stable tones); strum all strings and the row switches
   to a per-string readout (`E A D G B e`, standard tuning, cents each, ±120-cent
   capture range). `--a4 432` moves the reference (400–480 Hz). The tuner reads the raw
   input (pre-trim, pre-model), so it works identically live, bypassed, or with the
   output muted — `t` + `m` is the classic silent-tuning pedal move. Off = off: the
   analysis thread parks on a futex and the audio callback's tap is a single load.

   **Not sure which input?** Start with `--auto-input` (or press `a` anytime) and keep
   playing: it records ~1.5 s from every capture device, measures signal vs noise floor,
   and picks the cleanest source — a direct interface/DI input wins over a microphone
   hearing the same guitar acoustically, because its floor between notes is near-silent.
   Virtual devices (Teams, Zoom, VB-Cable, BlackHole, Camo, aggregates, loopbacks) are
   never candidates; devices that exist in both the capture *and* playback lists are
   tagged `[interface]` and an interface carrying any meaningful signal beats every
   microphone outright (even a weak under-gained one — you'll get a note to raise the
   interface's gain). USB interfaces often enumerate under generic names like "Audio
   Out" or "USB Audio CODEC" — the `[interface]` tag is how you spot yours. If nothing
   carries signal it keeps the current input instead of guessing. When the winner is a full interface, its playback side is
   adopted as the output automatically (one clock, no drift) — or, if you had explicitly
   chosen an output, offered as a one-key suggestion (`y` to accept). Otherwise: if the
   `in` meter stays at −140 dB while you play, press `i` to cycle inputs; if `in` moves
   but you hear nothing, cycle the output with `o`.

If you hear nothing: on macOS the microphone permission is attributed to your **terminal
app** — a denied permission yields silence with no error. Check System Settings → Privacy →
Microphone.

## Hardware: what plugs into what

The two flows have different wiring needs. The golden rule: a guitar pickup is a weak,
high-impedance *instrument-level* signal; interface line outputs are strong, low-impedance
*line-level* signals. Mismatching them won't break anything (with one exception below), but
it will skew the sound — and for profiling, a skewed capture becomes a permanently skewed
profile.

### Playing live (guitar → interface → Fucina → speakers)

| Your setup | What to do |
| --- | --- |
| Interface has a **Hi-Z / "Inst" input** (most do: Scarlett, Volt, UR, MOTU...) | Plug the guitar straight in and engage the Inst/Hi-Z switch. **No extra gear needed.** |
| Only mic/line inputs, passive pickups | Put a **DI box** (or any buffered pedal — a tuner pedal works) between guitar and interface. Plugging a passive guitar into a low-impedance line input loses treble and level. |
| Active pickups, or a buffered pedalboard in front | Direct into a line input is fine — the buffer already did the impedance work. |

Set the interface's input gain so your hardest playing peaks around −12…−6 dBFS on the
`live` input meter (never hitting `CLIP!`); fine-tune the drive into the model with the
`,`/`.` input-trim keys (or `--input-gain dB`) — NAM models are nonlinear, so input level
controls breakup, not just loudness.

**If everything sounds too quiet:** profiles have wildly different built-in output levels.
`live` normalizes them to −18 dBFS by default using each profile's loudness metadata
(status shows `NORM`; toggle with `n` or start with `--no-normalize` for the raw upstream-core
behavior), and `+`/`-` adds up to 24 dB of clean output gain on top. If the *input* meter is
the quiet one, raise the interface gain / input trim instead — boosting output can't recover
a starved model. Monitor through headphones/speakers on the **same interface**
you capture with — one device means one sample clock; using a different output device will
click every couple of minutes (independent clocks drift apart and miniaudio applies no
drift correction).

### Profiling an amp or pedal (`profile` / `train`)

Two cable runs at once: the capture signal goes **out** of the interface into your gear, and
the gear's output comes **back in**:

```
interface line OUT ──(reamp box)──> amp/pedal input
amp/pedal output  ──(see table)──> interface IN
```

**The send side — do you need a reamp box?**

| Target | Recommendation |
| --- | --- |
| Tube/solid-state **amp input** or **pedal** | Use a **reamp box** (e.g. Radial ProRMP). It converts line level → instrument level and impedance, and its ground lift kills hum loops. This is the *correct* way: the amp's input stage reacts to level and impedance, and the profile bakes in whatever it sees. Going direct with the interface output turned way down *can* work, but the capture may not match how the amp feels with a guitar, and ground hum contaminates training data. |
| **Digital gear** (another modeler, a plugin chain, a rack unit with line input) | Direct line-to-line is fine. No reamp box needed. |

**The return side — how to get the amp's output back:**

| Source | Connection |
| --- | --- |
| Amp's **line out / DI out / emulated out** | Straight into a line input. Captures preamp (+ power amp on some outs) without the cab — pick `--gear-type amp` and add a cab IR at playback with `--ir` (see [Cabinet IRs and signal chains](#cabinet-irs-and-signal-chains)). |
| **Mic on the cab** | Mic input with preamp gain. The profile then includes cab + mic (`--gear-type amp_cab`) and needs no IR afterwards (`live` warns if you add a redundant one). |
| **Speaker output** of an amp | ⚠️ **Never into a line input directly** — speaker-level signals are tens of volts and will damage the interface. Use a **load box / reactive attenuator with a DI out** (Captor, Suhr RL, ...). And remember: a tube amp must always see a speaker or load. |

Levels: aim for healthy peaks around −6 dBFS on the return; the trainer refuses clipped
captures (`|y| ≥ 1.0`, same as upstream). Keep all knobs untouched between the latency blips
at the start of the capture file and the end — drift fails the data checks.

### The capture run itself

Use the standardized **v3 capture file** (`v3_0_0.wav` — the same "input file" download the
official NAM trainer uses). It is recognized by checksum and enables automatic latency
calibration (from its blips) and the quality pre-checks. Then:

```sh
fucina-zig-nam profile --signal v3_0_0.wav --reamp-out reamp.wav \
    --out my-amp.nam --capture 2 --playback 2 \
    --name "My Amp" --gear-type amp --tone-type crunch
```

plays the file through your rig, records the return, saves it, and trains. Alternatively
record the reamp in your DAW and run the two-step version:
`fucina-zig-nam train --input v3_0_0.wav --output reamp.wav --out my-amp.nam`. Any other
48 kHz input/output pair also works (pass `--latency` if your interface loopback delay is
known; the last 9 s become the validation split).

Training defaults to the classic "standard" WaveNet (13,802 weights), matching the upstream
Python full config. `--spec a2`/`--spec a2-standard` selects the C++ reference A2-standard
shape (8 channels); `--spec a2-nano` selects the 3-channel A2-nano shape. `--spec packed`
selects the current upstream PackedWaveNet easy-mode recipe: channels-3 and channels-8 A2
submodels, summed submodel losses, MRSTFT weight 0.0005, Adam weight decay 3.17e-7,
gamma=0.994, 100 default epochs, and `SlimmableContainer` export. `--init model.nam`
fine-tunes a supported WaveNet profile through the same loop, including recursive WaveNet
`condition_dsp` weights. The classic/A2 optimizer recipe is MSE, Adam lr 0.004, gamma=0.993,
batch 16, 100 epochs (`--epochs 20` is useful for quick CPU smoke runs).
Each epoch prints the validation ESR with the upstream quality bands: **< 0.01 "Great!"**,
< 0.035 "Not bad!". The best epoch is exported with the full upstream metadata schema (date,
measured loudness/gain, your `--name`/`--gear-*`/`--tone-type` fields, latency calibration
record, final ESR). Check the result by ear with:

```sh
fucina-zig-nam validate my-amp.nam --input v3_0_0.wav --output reamp.wav --write-wavs ab/
```

(writes `validation_target.wav` = the real amp and `validation_model.wav` = the profile,
time-aligned for A/B listening) — then just play it: `fucina-zig-nam live my-amp.nam ...`.

## MIDI control

Every live control is also a MIDI control. `live` listens to **all** connected MIDI
sources by default, and hot-plug works — source *identity* is rescanned every ~2 s, so you
can turn the pedalboard on after the player is already running (or swap one controller for
another). A footswitch, expression pedal, or
controller knob works out of the box with these defaults (GM conventions where one exists):

| MIDI message | Control | Mapping |
| --- | --- | --- |
| CC 7 (volume) | output gain | 0–127 → −40…+24 dB |
| CC 11 (expression) | input trim (drive) | 0–127 → −20…+40 dB |
| CC 1 (mod wheel) | gate threshold | 0–127 → −90…−30 dB |
| CC 64 (sustain) | bypass | ≥ 64 = bypassed |
| CC 80 | noise gate on/off | ≥ 64 = on |
| CC 81 | loudness normalization | ≥ 64 = on |
| CC 85 | output mute | ≥ 64 = muted |
| Program change | profile/chain slot | PC 0 = slot 1 |

The status line echoes each applied event for ~1.5 s (`[MIDI CC7=93]`), so you can see a
controller reach the right knob. Continuous CCs sweep the same dB ranges the keyboard keys
step through, linear in dB.

Options: `--midi N` listens to one source only (`devices` lists them with indices),
`--no-midi` disables MIDI, `--midi-channel 1-16` reacts to one channel (default: omni),
and `--midi-map` reassigns CC numbers, e.g. `--midi-map out-gain=20,bypass=82`
(names: `out-gain`, `in-gain`, `gate-threshold`, `bypass`, `gate`, `normalize`;
unmentioned controls keep their defaults, and two controls landing on one CC — including
against an unmentioned control's default — are rejected at startup rather than silently
shadowed). MIDI is macOS-only for now (CoreMIDI); other platforms degrade to keyboard
control.

## Cabinet IRs and signal chains

A `.nam` capture of an amp head or preamp has no speaker — pair it with a **cabinet
impulse response** (a mono `.wav`). And you can run several stages in series — e.g. a
drive pedal into an amp into a cab — as a **chain**.

### Add a cab IR

```sh
fucina-zig-nam live amp.nam --ir cab.wav --capture 2 --playback 2
```

`--ir` appends the cab after the model (`amp → cab → output`), just like loading an IR in
the NAM plugin. It is a faithful port of the upstream `ImpulseResponse`: direct
time-domain convolution, mono, up to 8192 taps, with the same fixed −18 dB headroom gain
(so reach for `+` output gain if the cab makes things quiet). The IR is **resampled** to
the session rate at load when it differs (cubic, like upstream), so a 44.1 kHz cab works
in a 48 kHz session — unlike `.nam` models, which are nonlinear and must match the stream
rate. `--ir` also works for offline `render`.

### Build a chain

A `.chain` file is a text manifest, one stage per line, top → bottom = signal flow:

```
# pedal -> amp -> cab
name: My Rig                       # optional; shown in the status line
boost.nam :: trim=+3               # a drive capture, hit +3 dB harder
amp.nam                            # no trim = unity
cab.wav :: trim=-2                 # cabinet IR, pulled back 2 dB
```

```sh
fucina-zig-nam live --chain rig.chain --capture 2 --playback 2
```

- A stage is a `.nam`/`.gguf` model or a `.wav` cab IR (chosen by file extension).
- `:: trim=<dB>` is an optional per-stage input trim — because the models are nonlinear,
  the level *into* a stage shapes its breakup, not just its volume. The ` :: ` is a literal
  space-colon-colon-space, so paths with spaces work; trims aren't live-adjustable.
- `name:` is optional (first one wins); without it the chain is named after the file. `#`
  starts a comment only as the first non-space character (so `Marshall #2.nam` is a path).
  Paths are relative to where you run the command.

Pass several `--chain rig1.chain --chain rig2.chain` and/or bare `.nam` profiles together
and switch between them live with `[` `]`, `1`–`9`, or MIDI Program Change — the status
line shows the active chain plus an `x3` tag for its stage count. A bare profile is just a
one-stage chain, so plain `live a.nam b.nam` A/B works as before; `--ir` then appends a cab
to **each** bare profile (with manifests only it has nothing to attach to and is ignored —
put the cab in the manifest instead).

### Cab advice from `gear_type`

If a capture carries a `gear_type` (the trainer writes it; see `--gear-type` above), `live`
checks each chain at load and prints a non-fatal note, both ways:

- **redundant cab** — a cab IR following a capture that already includes a speaker
  (`amp_cab`, `amp_pedal_cab`, `studio`, or a Tone3000 "full rig"): the doubled cab sounds
  dull/boxy.
- **cab likely needed** — a chain ending in an `amp`/`preamp`/`pedal_amp` capture with no
  cab after it.

Captures with no `gear_type`, and pedal-only chains, are left alone.

## All commands

| Command | What it does |
| --- | --- |
| `devices` | List capture/playback devices and MIDI sources with indices. |
| `live [<profile>...] [--ir cab.wav] [--chain rig.chain] [--capture N] [--playback N] [--rate 48000] [--period 128] [--tuner] [--a4 440] [--midi N \| --no-midi] [--midi-channel C] [--midi-map ...]` | Play through profiles and/or chains (see [Cabinet IRs and signal chains](#cabinet-irs-and-signal-chains)). |
| `profile --signal s.wav --reamp-out r.wav --out m.nam [...]` | One-step capture + train + export. |
| `train --input in.wav --output reamp.wav --out m.nam [...]` | Train from an existing pair. |
| `validate <model> --input in.wav --output reamp.wav [--write-wavs dir]` | ESR + A/B WAVs. |
| `inspect <model.nam\|.gguf>` | Print structure + metadata. |
| `render <model> <in.wav> <out.wav> [--blocksize N] [--ir cab.wav]` | Offline file processing (matches upstream `tools/render`; `--ir` appends a cab). |
| `bench <model> [--blocksize N]` | Per-block cost vs the realtime budget. |
| `list [--profiles-dir d]` | Profiles in `./nam-profiles` (or `$FUCINA_NAM_PROFILES`). |
| `export-gguf` / `import-gguf` | Lossless GGUF interchange (byte-identical `.nam` recovery). |

## Compatibility guarantees

- **Import:** any upstream-tooling `.nam` of architecture WaveNet (incl. gated/blended,
  grouped convs, active FiLMs, WaveNet `condition_dsp`, and every legacy config spelling),
  LSTM, ConvNet, Linear, and `SlimmableContainer` (the current upstream trainer's export —
  loaded at its highest-quality submodel, the one players use by default). Slimmable WaveNet
  submodels and non-WaveNet `condition_dsp` engines fail with a named error in the trainable
  WaveNet path.
- **Export:** `.nam` v0.7.0 in the modern upstream exporter shape; WaveNet exports and packed
  `SlimmableContainer` exports both load in upstream `NeuralAmpModelerCore` (`loadmodel`).
  Rendering through the upstream core matches Fucina (6.7e-8 max on a trained classic profile;
  2.7e-8 max on a packed-container smoke). Profiles made here work in any NAM player that
  supports the exported architecture.
- **Numeric parity:** vs upstream `tools/render` on the upstream example models — standard
  WaveNet max |diff| 2.3e-6 / RMS 6.5e-8 (about 20× inside upstream's own 5e-5
  cross-implementation tolerance). The Tanh activation is a SIMD evaluation of the exact
  scalar-libm contract used by the upstream C++ runtime (measured ≤ 1.9 ulp / 9.5e-8 abs vs
  correctly rounded tanh; value-only lane math, IEEE ops + correctly rounded FMA only, so
  results are deterministic across machines). Output is byte-identical across block sizes.
- **Performance:** standard WaveNet ≈ 49 µs per 64-frame block @48 kHz on one core
  (ReleaseFast, i9-13950HX P-core; 2026-07-03 x86 snapshot) ≈ 27× realtime — 1.8× faster
  than upstream `benchmodel` at the documented protocol (stock `-Ofast` Release, exact
  tanh: 87 µs; its default fast-tanh: 83 µs), and 2.6–3.6× faster than upstream's own
  block-size sweep tool at 64–512 frames. Only an `-march=native` rebuild comes close
  (exact tanh via glibc's 8-wide vector libm: 45 µs; fast-tanh, a ~1e-3-class contract:
  39 µs). On M1 Max ≈ 80 µs/block — the 2026-06-12 67 µs snapshot predates commit
  59d74a3, which traded the Eigen-rational tanh for scalar libm parity (162 µs on
  today's HEAD); the SIMD tanh recovers 2× of that while keeping the strict contract.
- **GGUF:** an optional, lossless container — the original `.nam` JSON rides byte-verbatim in
  the `nam.file_json` KV next to a flat `nam.weights` f32 tensor, so `import-gguf` recovers a
  byte-identical `.nam`. The runtime loads `.nam` directly; GGUF never replaces it.
  Quantization is deliberately refused (tiny L1-resident models, block-size mismatch, ESR
  risk).

## Latency, buffers, sample rate

- `live` prints an honest end-to-end estimate at startup, read straight from CoreAudio:
  `input device + duplex+period + output device`. The duplex+period term is ≈ 3·period
  (miniaudio's duplex ring keeps ~2 capture periods of slack); the device terms are real
  hardware latency + safety offset + device buffer. Default `--period 64` ⇒ 4 ms for the
  middle term; `--period 32` halves it if your interface is stable there.
- **The devices dominate.** A proper USB interface contributes ~1–3 ms per side (total
  ≈ 7–10 ms — same league as hardware modelers). Good interfaces (MOTU M2 class) accept
  `--period 16` for ~8 ms total at 48 kHz; if you hear crackles under load, step back to
  32. Built-in mic/speakers ≈ 11 ms total.
  **Avoid for monitoring:** HDMI/DisplayPort monitor speakers (the display adds internal
  buffering CoreAudio cannot even report — often tens of ms on top of the ~10 ms it does
  report), Bluetooth anything (100+ ms), webcam microphones. Guitar feel reference: ≤10 ms
  reads as immediate (standing 3 m from your amp is ~9 ms of air), ~15–20 ms feels laggy,
  30+ is unplayable.
- Devices not natively at 48 kHz are **retuned to 48 kHz at the OS level for the session**
  (`coreaudio.allowNominalSampleRateChange`) — without this, the hidden resampler's rate
  error *accumulates* in miniaudio's duplex ring and latency grows second by second while
  you play (measured: ~1100 samples/s) until the ring wraps. If retuning fails you still
  get the resample warning; fix in Audio MIDI Setup.
- Fixed-size callback re-buffering is disabled (`noFixedSizedCallback`): the device's own
  blocks reach the engine directly, removing one intermediary buffer layer.
- The latency estimate uses the device's **negotiated** buffer size, not the requested one;
  a driver that refuses small buffers triggers a warning with the real cost (lower its
  buffer in the vendor's control panel if it has one).
- `loopback-test --capture N --playback N` measures the true software round-trip by sending
  impulses and timing their return. Run it on your interface with a physical patch cable
  (line out → line in) for ground truth on your exact rig.
- Everything miniaudio offers for latency is engaged (verified against upstream guidance and
  the vendored 0.11.25 — the newest release): explicit low-latency profile, no fixed-size
  re-buffering, no output pre-silencing/clip pass, realtime thread priority, OS-level rate
  retune. miniaudio's maintainer describes duplex as intentionally basic (no drift re-sync) —
  which is why the rate retune matters. The only further software win would be replacing the
  duplex machinery with separate capture/playback devices and a custom one-period ring —
  worth ~1.3 ms at 64-frame periods but only ~0.3–0.5 ms once a good interface runs at
  `--period 16`; not pursued at that price (underrun-crackle risk for sub-ms gain). The
  estimate includes per-stream CoreAudio latency, device latency, safety offsets, and the
  negotiated buffer — what the OS knows about, end to end. WASAPI exclusive/pro-audio modes are Windows-only and would matter if that port
  happens.
- The audio callback is allocation-free and lock-free; all chains (and their stages) are
  preloaded and prewarmed, so switching mid-playing doesn't glitch.

## Test plan (beyond `zig build test`)

1. **Golden parity:** build upstream `render` (`cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
   -DNAM_ENABLE_A2_FAST=OFF` in `refs/NeuralAmpModelerCore` after `git submodule update --init`),
   render the example models over a test WAV with both renderers, compare (gate: 1e-5 RMS).
2. **Trained-profile interop:** `train` on any pair → upstream `loadmodel` accepts the export →
   upstream `render` matches ours. Python re-import oracle (needs torch):
   `pip install -e refs/neural-amp-modeler` then
   `python -c "import json,nam.models; nam.models.init_from_nam(json.load(open('model.nam')))"`.
3. **GGUF round-trip:** `export-gguf` → `import-gguf` → `cmp` byte-identical; render both.
4. **Realtime:** `bench` per block size; `live` loopback (e.g. a virtual cable) for the
   stream/HUD; a real-interface session for latency/dropout listening.
5. **Capture flow:** `profile` against a loopback (signal ≈ reamp ⇒ near-zero ESR) before
   using a real amp.
6. **Cab IR + chains:** `render amp.nam in.wav out.wav --ir cab.wav` (A/B with and without
   the cab); `live --chain rig.chain` to hear a multi-stage rig and see the per-chain gear
   advisories printed at load.

Module layout and the architecture split are described in the module doc comments of
`examples/nam/*.zig`; every deliberate deviation from upstream is documented at the point
in the code where it applies.
