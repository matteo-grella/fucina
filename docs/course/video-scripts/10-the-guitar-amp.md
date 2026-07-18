# Video 10 — The guitar amp (3:00)

*Series: Forging Deep Learning in Zig · Source: ../10-the-guitar-amp.md*

## Logline

The series' flagship: a 13,802-weight WaveNet imitates a tube amplifier live,
on camera, from a terminal — every 64-sample block transformed inside a
1,333 µs deadline, with no allocation on the audio thread. The full-circle
frame: Zig was conceived while its creator built a digital audio workstation;
a decade later a neural amp — trained and played by the same binary — runs
live in it.

## Takeaways

1. Real-time audio is where this library's disciplines stop being style and
   become the product: the budget is 64 frames ÷ 48 kHz = ~1,333 µs per block,
   against a measured ~49 µs (x86 P-core) / ~80 µs (M1 Max) — dated,
   machine-specific snapshots — and ReleaseFast is mandatory (~20× vs Debug).
2. The deadline is guaranteed structurally, not by vigilance: every buffer
   preallocated before the stream starts, engines allocation-free after init
   by written contract, all cross-thread control in atomics — switching amps
   mid-song is one atomic index store.
3. One binary trains the model with autograd and plays it with a hand-rolled
   streaming engine: no Python, no GPU, no GC, no second framework for
   deployment.

## Script

### [0:00–0:26] Hook: that amp is a network

**VO:** Listen. That amp is a neural network — thirteen thousand eight hundred
and two floating-point numbers imitating a tube amplifier, live, in Zig, from
a terminal. And the contract is brutal: the network must transform every block
of samples before the next one arrives. Every one-point-three milliseconds.
Forever. A missed deadline is not a slow benchmark — it's an audible click
through the speakers, mid-song.

**Visual:** Sound-first open: ~4 s of live electric guitar through the amp
model (close-up of hands on strings), then reveal the rig — guitar → USB
interface → laptop terminal. On the terminal: `zig build nam
-Doptimize=ReleaseFast` run with `.nam` files in a `nam-profiles/` folder,
showing the numbered amp menu; a number is pressed, the status line appears,
playing continues. (Per `examples/nam/README.md`, "The simplest path — no
flags at all".)

**Overlay:** "live · one CPU core · no plugin host, no DAW" · "13,802
weights (`examples/nam/train.zig:34-36`)".

### [0:26–0:58] Full circle: a language born from audio

**VO:** Here's the history worth savoring. Zig's creator, Andrew Kelley,
conceived the language while building a digital audio workstation — Genesis —
after C and C++, and just as much their ecosystems, fell short for low-latency
audio. Real-time sound demands no garbage collector, no hidden allocation,
explicit buffer lifetimes, worst-case thinking. That list is close to a
specification of Zig's design values. A decade later, the circle closes: a
neural amplifier runs live in a language born from audio.

**Visual:** History card drawn as a closing circle: left node "2016 — Zig
announced; conceived while building the Genesis DAW", arc through the demand
list ("no GC pauses · no hidden allocation · explicit lifetimes · worst-case,
not average"), right node "2026 — a neural amp runs live in Zig" with a still
from the opening demo. The two nodes join as the VO says "the circle closes".

**Overlay:** Citation caption, must stay on screen for the whole card:
"External sources (§10.1, not in the Fucina repo): Andrew Kelley,
'Introduction to the Zig Programming Language', andrewkelley.me, Feb 8 2016 ·
github.com/andrewrk/genesis".

### [0:58–1:22] A model you can hold

**VO:** The model is a WaveNet, and it is tiny — 55 kilobytes, resident in L1
cache. Its trick is dilated causal convolution: taps look only backwards in
time, and their spacing doubles layer by layer, one up to 512. Stack two of
those, and each output sample hears 4,093 past samples — about 85 milliseconds
of guitar. Exponential ears, linear bill.

**Visual:** The episode's single WaveNet diagram (per direction, one visual
only): a fan-in tree over a sample timeline — stacked conv layers with
dilations 1, 2, 4, …, 512, the receptive field widening exponentially
downward to span 4,093 past samples, bracketed "≈ 85 ms @ 48 kHz". Constant
work per layer noted on the right edge ("linear cost").

**Overlay:** "classic profile: 13,802 floats ≈ 55 KB — fits in L1 (§10.2)" ·
"receptive field 4,093 samples ≈ 85 ms — verified by a `zig test` in the
chapter (§10.3)".

### [1:22–1:54] The budget arithmetic

**VO:** Now the arithmetic that rules everything. The hardware delivers
64-frame blocks at 48 kilohertz. Sixty-four divided by forty-eight thousand:
1,333 microseconds per block. Against that budget, the measured cost — about
49 microseconds on one x86 P-core, about 80 on an M1 Max; dated,
machine-specific snapshots from the example's README — roughly 27 times
headroom, and about 17 on the Mac. One flag is mandatory: ReleaseFast. Debug
builds are around twenty times slower and will not keep up.

**Visual:** Budget-math card: "64 frames ÷ 48,000 frames/s = 1.333 ms ≈
1,333 µs" (§10.9), then a horizontal bar for the 1,333 µs budget with two
small ticks near its left edge: "≈49 µs" and "≈80 µs". Then terminal
recording: `zig build nam -Doptimize=ReleaseFast -- bench my-amp.nam`,
holding on the `budget` / `measured` / `realtime: …x headroom` output lines
(printed by `examples/nam/main.zig:330-336`).

**Overlay:** On the ticks, permanently: "≈49 µs/block — ReleaseFast,
i9-13950HX P-core, 2026-07-03 x86 snapshot · ≈80 µs — M1 Max ·
`examples/nam/README.md` (Performance); measure yours with `bench`". On the
ReleaseFast beat: "Debug ≈ 20× slower — will not keep up
(`examples/nam/README.md:23-24`)".

### [1:54–2:26] Inside the callback: no allocation, structurally

**VO:** How is the deadline guaranteed? Structurally. The callback's doc
comment is the rule — no allocation, no locks, no syscalls — and the code
beneath is the proof. Every buffer is preallocated before the stream starts;
every engine is allocation-free after init, by written contract. Control
crosses threads only as atomics: gain knobs are float bits in an atomic
integer, and switching amps mid-song is one atomic index store into a
preloaded, prewarmed array.

**Visual:** Code shot 1: `examples/nam/live.zig:268-282` — the `audioCallback`
opening, with the doc comment "No allocation, no locks, no syscalls."
highlighted first, then the null-checked pointers. Code shot 2:
`examples/nam/live.zig:93-102` — the `Shared` struct, highlighting
`std.atomic.Value` on every field and `gain_bits … @bitCast`. On the final VO
line, cut back to live footage: `]` pressed mid-riff, the status line flips to
the next amp, playing never stops.

**Overlay:** "every engine: 'All buffers are allocated at init; `process` is
allocation-free' (`examples/nam/stream_conv.zig:9` — same contract in
`wavenet.zig`, `engine.zig`, `ir_cab.zig`)" · on the switch cut: "profile
switch = one atomic index store (`examples/nam/live.zig:1-6`)".

### [2:26–3:00] One binary, and what's absent

**VO:** And the flagship part: this same binary trains these models — autograd
tensors for training, hand-rolled SIMD for the realtime chain, one
compilation, one program. Notice what's absent. No Python process. No GPU. No
garbage collector to outsmart. No second framework for deployment. When
latency is the product, the short list between your code and the deadline is
the entire feature. Next time: model files and quantization — models six
orders of magnitude larger, where the walls move.

**Visual:** Split-screen, one beat: left `examples/nam/wavenet.zig:4-15` (the
module-doc algebra of the streaming engine), right
`examples/nam/train.zig:279-289` (the same equations as autograd ops), a thin
line pairing `z = dilated_conv(x) + input_mixin(condition)` with
`causalConv1d … add … dot … add`. For the "absent" list, return to the live
demo — playing continues under the VO while the four "No …" lines stamp onto
screen. End card: series title, "Full chapter:
`docs/course/10-the-guitar-amp.md`", "Next: 11 — Model files and
quantization".

**Overlay:** "same math — once as a streaming kernel, once as autograd ops
(§10.7)" · end card: "Full chapter: `docs/course/10-the-guitar-amp.md`" ·
"Next: Model files and quantization".

## Asset list

**Live demo (the episode's centerpiece — record for real):**
- Hardware: electric guitar; USB audio interface with a Hi-Z/Inst input (or a
  DI box); macOS / Apple Silicon machine — the live path is "Tested on macOS /
  Apple Silicon … Linux should work too but is untested"
  (`examples/nam/README.md:14-16`). Monitor through the **same** interface
  (two devices = two clocks = periodic clicks, `README.md:123-125`).
- Software: `zig build nam -Doptimize=ReleaseFast` with 2–3 `.nam` profiles
  placed in `./nam-profiles/` → numbered amp menu → pick → play. Keys used on
  camera: number keys to pick, `]` to switch mid-riff.
- macOS gotcha to check before the shoot: microphone permission is attributed
  to the **terminal app**; "a denied permission yields silence with no error"
  (`examples/nam/README.md:92-94`).
- Capture the processed sound from the interface output (clean DI of the
  result), room mic optional for hands/strings ambience.

**External downloads (weights are NOT in the repo):**
- 2–3 `.nam` amp profiles from Tone3000 (https://www.tone3000.com — thousands
  of free ones; almost all "standard WaveNet", which this player runs at full
  fidelity, per `examples/nam/README.md`).

**Terminal recordings:**
- `zig build nam -Doptimize=ReleaseFast` (no flags, profiles in
  `nam-profiles/`) — the numbered amp menu.
- `zig build nam -Doptimize=ReleaseFast -- bench my-amp.nam` — budget /
  measured / headroom lines.

**Code shots (repo files, exact ranges):**
- `examples/nam/live.zig:268-282` — `audioCallback` opening + doc-comment rule.
- `examples/nam/live.zig:93-102` — `Shared` atomics (`gain_bits` as
  `@bitCast` f32 bits).
- `examples/nam/wavenet.zig:4-15` — module-doc dataflow algebra.
- `examples/nam/train.zig:279-289` — the autograd twin of the same layer math.

**Diagrams to render (one sentence each):**
- Circle-closing history card: "2016 — Zig conceived inside the Genesis DAW" →
  demand list → "2026 — a neural amp runs live in Zig", with the external
  citation caption (§10.1).
- WaveNet dilation fan-in over a timeline: dilations 1…512 doubling per layer,
  receptive field 4,093 samples ≈ 85 ms @ 48 kHz (§10.3) — the episode's only
  architecture visual.
- Budget bar: 1,333 µs block budget with ≈49 µs / ≈80 µs ticks and their
  caveat caption (§10.9, `examples/nam/README.md` Performance).
- End card with "Full chapter: `docs/course/10-the-guitar-amp.md`" and
  next-episode teaser.

## Production notes

- **Tone:** this is the flagship — highest production value in the series, and
  the guitar must sound *good*. But the narration stays measured: the wow is
  the demo itself, not adjectives. No "revolutionary", no SOTA claims.
- **Caveats are load-bearing and MUST NOT be cut:** (a) the ≈49 µs / ≈80 µs
  figures always carry "ReleaseFast, i9-13950HX P-core, 2026-07-03 x86
  snapshot / M1 Max — `examples/nam/README.md` (Performance)"; the chapter
  itself teaches that these numbers rot (§10.9's stale doc-comment lesson), so
  the "measure yours with `bench`" clause stays; (b) the ~20× Debug figure
  keeps its README citation; (c) the §10.1 history stays clearly marked as
  cited from **external** primary sources, not the Fucina repo; (d) whatever
  the on-camera `bench` run prints is the demo machine's own measurement —
  never composite or retouch it to match the README figures; if they differ,
  that difference is the honest point.
- **The demo machine:** an Apple Silicon Mac makes the ≈80 µs M1 Max figure
  the locally relevant one; the on-screen `bench` output then sits naturally
  beside it. Build with `-Doptimize=ReleaseFast` — a Debug build will audibly
  glitch and waste the shoot.
- **Numbers appearing in the video and their sources:** 13,802 weights / 55 KB
  (§10.2, `examples/nam/train.zig:34-36`); dilations 1..512, receptive field
  4,093 ≈ 85 ms (§10.3, chapter-verified test); 64 ÷ 48,000 = 1,333 µs
  (§10.9 arithmetic); ≈49 µs / ≈80 µs, ≈27× / ≈17× headroom
  (`examples/nam/README.md` Performance via §10.9); ~20× Debug
  (`examples/nam/README.md:23-24`). Nothing else may be quantified — in
  particular do not add the upstream-comparison or parity numbers; they carry
  protocol baggage this cut has no room to caveat.
- **If the cut runs long, trim in this order:** the history card's dwell time
  (keep the VO and the citation overlay), then the wavenet/train split-screen
  (land it as a single frame), then code shot 2 (`Shared`) — never the
  callback shot, never the live playing, never a caveat overlay.
- **Audio mix:** duck the guitar under VO but never to zero during the demo
  segments; the click-free amp switch at 2:20 must be audible proof, so leave
  the guitar prominent there.
- The next-episode teaser ("Model files and quantization") matches Video 11's
  title and must survive edits.
