# Video 11 — Model files and quantization (3:00)

*Series: Forging Deep Learning in Zig · Source: ../11-model-files-and-quantization.md*

## Logline

A model file is nothing mysterious — tensors plus metadata, a table of
contents over bytes you can mmap and use in place — and quantization is a
bandwidth story: Q8_0 packs 32 weights and their scale into 34 bytes. The
showcase is byte-exactness as religion: Fucina's encoders match goldens from
ggml's own reference encoders byte for byte, GGUF re-emit is asserted
byte-identical, and the streaming exporter quantizes models far bigger than
RAM one tensor at a time.

## Takeaways

1. A model is tensors + metadata. GGUF is magic → version → counts →
   key/values → tensor directory → aligned data; block types are `extern
   struct`s pinned by comptime size asserts, so loading is reinterpreting
   mmap'd bytes — no deserialization step exists.
2. CPU decode is weight-bandwidth-bound: every weight byte is read once per
   token, so bytes per weight is the speed knob — Q8_0 stores 32 weights +
   an f16 scale in 34 bytes, 8.5 bits per weight (floor numbers are
   back-of-envelope arithmetic, not measurements).
3. Interop is a byte contract, checked in bytes: ggml-golden byte-for-byte
   encoder parity, byte-identical re-emit, honest scope (all 27 formats
   decode, the public encoder writes 10) — and computable offsets turn GGUF
   writing into "a plan plus a stream", which is what makes
   bigger-than-RAM quantization possible.

## Script

### [0:00–0:25] Tensors plus metadata

**VO:** Open a language model file in a hex editor and there's almost
nothing to it. Four ASCII bytes — G, G, U, F. A version. Two counts. Then a
list of key-values — architecture, layer count, the entire tokenizer
vocabulary — and a table of contents: every tensor's name, shape, type, and
offset. Then raw bytes. That's the whole format. A model is tensors plus
metadata.

**Visual:** Terminal: `xxd -l 96 models/Qwen3-0.6B-f16.gguf` — the ASCII
column shows `GGUF` in the first four bytes. Beside it, animate the §11.1
layout box as a stacked diagram: magic → version (u32) → tensor_count (u64)
→ metadata_kv_count (u64) → metadata KVs → tensor directory → padding →
aligned tensor data. Highlight each band as the VO names it.

**Overlay:** "a model = tensors + metadata" · "GGUF v2/v3 read · v3
written (`src/gguf.zig`)" · "Qwen3-0.6B here is just the example — every
GGUF has this layout".

### [0:25–0:56] mmap and go

**VO:** So Fucina doesn't copy it. File.loadMmap maps the file read-only and
parses in place. Every metadata string, every tensor payload is a zero-copy
slice into the mapping, and the OS pages weights in on first touch. There is
no deserialization step, because each block format's in-memory type is an
extern struct — the struct is the wire format, and a comptime assert pins
its size at build time. Loading is reinterpreting. The file is the memory.

**Visual:** Code shot 1: `src/gguf.zig:242–262` (`File.loadMmap` —
`PROT_READ`, `MAP_PRIVATE`, fd closed immediately). Code shot 2:
`src/dtype.zig:81–84` (`BlockQ8_0`: `d: u16` + `qs: [32]i8`), then pan to
`src/dtype.zig:228–256` — the comptime block asserting
`@sizeOf(BlockQ8_0) == 34`, "one assert per block struct, 27 in total".

**Overlay:** "zero-copy: every slice dies at `file.deinit()`" · "`extern
struct` = C layout = the wire format" · "size drift → build failure, not a
corrupted export".

### [0:56–1:27] The bandwidth wall

**VO:** Why quantize? Not to fit in RAM — for speed. Generating a token
reads every weight exactly once: one multiply-add per byte, no reuse to
block for. Back-of-envelope, not a measurement: seven billion parameters in
f32 is twenty-eight gigabytes. At a hundred gigabytes per second, that's
under four tokens per second as a hard floor. The same model at four and a
half bits: about twenty-five. Nothing about the ALUs changed. Bytes per
weight is the speed knob.

**Visual:** Diagram card built from the §11.6 table: rows f32 → 32 bpw →
28 GB, f16 → 16 → 14 GB, Q8_0 → 8.5 → 7.4 GB, Q4_K → 4.5 → 3.9 GB, with a
"token/s floor at nominal 100 GB/s" column morphing 4 → 25 between the f32
and Q4_K rows.

**Overlay:** Persistent caption: "back-of-envelope: 7e9 × bpw ÷ 8, nominal
100 GB/s — arithmetic, not a measurement (§11.6)". Second card, small:
"honesty note: `docs/BENCHMARK.md` records one measured hot case where Q6_K
read *slower* than Q8_0 — 'fewer bytes = faster' is the right default and
the wrong absolute".

### [1:27–1:50] Q8_0: 34 bytes for 32 weights

**VO:** The teachable atom is Q8_0. Take thirty-two consecutive floats. Find
the largest magnitude. Divide by one twenty-seven — that's your scale. Store
each value as a signed byte, and the scale as an f16. Thirty-two weights
that cost a hundred twenty-eight bytes now cost thirty-four — eight and a
half bits per weight. The encoder is eleven lines.

**Visual:** Code shot: `src/backend/quant/q8k.zig:57–68` — the real
encoder loop (`amax`, `d = amax / 127.0`, `quantizeToI8`), held on screen.
Then a byte-bar diagram of one block: 2 bytes f16 scale + 32 × i8 codes =
34 bytes.

**Overlay:** "34 B / 32 weights = 8.5 bpw · worst-case error d/2" ·
"scalar reference path — its NEON twin lives in the same file
(`src/backend/quant/q8k.zig`)".

### [1:50–2:22] Byte-exact or wrong (showcase)

**VO:** Here's what earns interop. Fucina's encoders are ports of ggml's,
down to the rounding trick — round-to-nearest-even, which differs from Zig's
own round exactly on ties. The claim is pinned, not trusted: goldens from
ggml's own reference encoders, matched byte for byte over eight adversarial
inputs. Same religion on the writer — parse a file, re-emit it, and the test
demands byte-identical output. It's honestly scoped, too: all twenty-seven
formats decode; the public encoder writes ten. The rest refuse with an
honest error.

**Visual:** Code shot 1: `src/backend/quant/encode_golden_test.zig:1010–1026`
— `expectEqualSlices(u8, &q4_k_golden[v], ...)` against ggml-generated
goldens. Quick highlight: `src/backend/quant/q8k.zig:650–655` (the
1.5·2²³ magic-constant rounding). Code shot 2: `src/gguf_tests.zig:241–251` —
"Re-emitting the parsed file must reproduce it byte-identically" ending in
`expectEqualSlices(u8, written, sink2.buffered())`. Terminal: `zig build
test` running to green (these suites are in-tree and run by it).

**Overlay:** "goldens: ggml's own reference encoders · 8 adversarial vectors
· Q4_K Q5_K Q6_K Q4_1 Q5_0 Q5_1 byte-for-byte" · "real-model re-emit: every
KV + tensor payload verbatim (`src/gguf_tests.zig:533`)" · "encodeF32
writes: q2_0 q4_0 q4_1 q5_0 q5_1 q8_0 q4_k q5_k q6_k tq2_0 (+ f32/f16/bf16
casts) — everything else: `error.EncoderUnavailable`".

### [2:22–2:49] Bigger than RAM, one tensor at a time

**VO:** The payoff is the export tool. A tensor's byte size follows from its
type and dims, so every offset is computable before any data exists. The
writer emits the complete header first, then streams tensors through one at
a time, freeing each buffer before the next — that's how it quantizes models
far bigger than RAM on a small machine. Every streaming run ends with a
peak-RSS report.

**Visual:** Terminal (executed): the §11.11 transcode command,
`zig build export-gguf -Doptimize=ReleaseFast -- --from-gguf
models/Qwen3-0.6B-f16.gguf --out models/Qwen3-0.6B-Q4_K_S.gguf --dtype
q4_k`, ending on its summary lines (`exported …` · `tensors: … transcoded
(q4_k)`). Code shot:
`src/gguf.zig:1115–1160` (`beginStream` → `DataStreamer`; `declareTensor`
sits earlier at `src/gguf.zig:1040`). Then type-on only (do not execute):
the §11.11 mode-(c) command `zig build export-gguf
-Doptimize=ReleaseFast -- --from-gguf big-BF16.gguf --out big-ptqtp3.gguf
--ptqtp=3`.

**Overlay:** "offsets fixed at declaration (`tensorByteLen`) → header
written before any data" · "streaming mode: one source tensor + its
quantized output in memory at once — residency bounded no matter the
model size (`tools/export_gguf.zig:54–65`)" · "streamed output
byte-identical to buffered `finish` (`src/gguf_tests.zig:254`)".

### [2:49–3:00] Two primitives, and what's next

**VO:** Two primitives carry all of it: a byte-exact block struct, and a
table of contents you can compute before you have the bytes. Next: a
transformer from scratch.

**Visual:** Closing card: two icons side by side — a 34-byte block bar
("byte-exact block struct") and a directory listing ("offsets computable
before the bytes exist"). End card: series title, "Full chapter:
`docs/course/11-model-files-and-quantization.md`", "Next: 12 — A
transformer from scratch".

**Overlay:** End card: "full chapter in `docs/course/`" · "Next: A
transformer from scratch — the tensor directory becomes a transformer".

## Asset list

**Code shots (repo files, exact ranges):**
- `src/gguf.zig:242–262` — `File.loadMmap` (read-only map, parse in place).
- `src/dtype.zig:81–84` — `BlockQ8_0` extern struct; `src/dtype.zig:228–256`
  — the comptime size asserts (27 block structs).
- `src/backend/quant/q8k.zig:57–68` — the 11-line Q8_0 encoder loop.
- `src/backend/quant/q8k.zig:650–655` — the 1.5·2²³ `nearestInt` rounding
  (brief highlight; optional).
- `src/backend/quant/encode_golden_test.zig:1010–1026` — byte-for-byte
  golden asserts vs ggml reference encoders.
- `src/gguf_tests.zig:241–251` — byte-identical re-emit assert (test
  starting at line 137); `src/gguf_tests.zig:533` — real-model re-emit test
  title (optional one-line shot); `src/gguf_tests.zig:254` — stream ≡
  finish test (cited in overlay only).
- `src/gguf.zig:1115–1160` — `beginStream`/`DataStreamer` (`declareTensor`
  is at `src/gguf.zig:1040`, outside this shot).

**Terminal recordings (exact commands):**
- `xxd -l 96 models/Qwen3-0.6B-f16.gguf` — the GGUF magic in the ASCII
  column (run from repo root; model path per download below).
- `zig build test` — runs the golden-parity and GGUF round-trip suites
  in-tree; record the passing tail.
- `zig build export-gguf -Doptimize=ReleaseFast -- --from-gguf
  models/Qwen3-0.6B-f16.gguf --out models/Qwen3-0.6B-Q4_K_S.gguf --dtype
  q4_k` — executed transcode (§11.11 mode a; the chapter prints the same
  command with bare filenames — the `models/` prefix is the path from repo
  root per the download note, nothing else changes), ending on the export
  summary (`exported …` / `tensors: … transcoded (q4_k)`).
- `zig build export-gguf -Doptimize=ReleaseFast -- --from-gguf big-BF16.gguf
  --out big-ptqtp3.gguf --ptqtp=3` — TYPE-ON ONLY, never executed (§11.11
  mode c; needs a huge model).

**Diagrams to render (one sentence each):**
- GGUF layout stack: magic → version → counts → metadata KVs → tensor
  directory → padding → aligned tensor data, bands highlighted in VO order
  (§11.1 layout box).
- Bits-per-weight table card: f32/f16/Q8_0/Q4_K rows with 7B-model
  gigabytes and the 4 → 25 tokens/s floor morph, captioned as
  back-of-envelope (§11.6 table).
- One-block byte bar: 2-byte f16 scale + 32 i8 codes = 34 bytes (§11.7).
- Closing two-primitives card and end card with "Full chapter:
  `docs/course/11-model-files-and-quantization.md`" and next-episode teaser.

**External downloads (weights are NOT in the repo):**
- Qwen3-0.6B f16 GGUF from Hugging Face (any official/upstream f16 export
  of Qwen3-0.6B in GGUF form; place under `models/`). Used for the xxd
  hook and the executed transcode. No other model is needed; the
  bigger-than-RAM PTQTP run is type-on only.

## Production notes

- **Tone:** matter-of-fact demystification. The hook works because the file
  really is that simple; don't dress it up. The showcase segment is proud
  but precise — parity is a *checked fact*, not a boast.
- **Caveats that MUST stay attached to numbers:** (a) the bandwidth-wall
  card keeps "back-of-envelope … arithmetic, not a measurement" on screen
  whenever 28 GB / 4 tok/s / 25 tok/s are visible; (b) the honesty note
  (Q6_K measured slower hot than Q8_0, `docs/BENCHMARK.md`) must survive —
  it is the chapter's own hedge on "fewer bytes = faster"; (c) the golden
  byte-for-byte overlay lists exactly the six formats the golden suite
  covers (Q4_K Q5_K Q6_K Q4_1 Q5_0 Q5_1) — do not extend the byte-for-byte
  claim to other encoders; (d) byte-identical re-emit is asserted for
  re-emitting a parsed writer output (`gguf_tests.zig:241–251`), while the
  real-model test asserts every KV and tensor payload verbatim
  (`gguf_tests.zig:533`) — keep the two claims distinct as written.
- **The "far bigger than RAM" claim** is the streaming (`--ptqtp`) mode's
  documented design (bounded residency via prefetch/release +
  `beginStream`, §11.11), and the peak-RSS report is that mode's summary
  (`tools/export_gguf.zig:54–65`, printed in `runPtqtp`) — the on-camera
  run is a small-model transcode through the buffered writer and ends on
  the exported/tensors summary. Never imply the recorded run itself
  exceeded RAM, streamed, or printed a peak-RSS line.
- **The 11-line encoder** is the scalar reference path (production code for
  non-aarch64 targets); the NEON twin exists — keep that overlay.
- **Do not execute** the `--ptqtp=3` command; it is type-on only. `zig build
  test` may run long — record once, cut to the green tail.
- **If the cut runs long, trim in this order:** the xxd hook (replace with
  the layout diagram alone), the `q8k.zig:650–655` rounding shot (keep the
  VO), the PTQTP type-on beat, then the closing two-primitives card (keep
  the end card). Never trim the caveat overlays or the honesty note.
- **Numbers appearing in the video and their sources:** GGUF layout and
  v2/v3 policy (§11.1, §11.4); 34 B / 32 weights / 8.5 bpw / error d/2
  (§11.7); 28 GB / 14 / 7.4 / 3.9 GB and the 4 → 25 tok/s floors (§11.6,
  explicitly arithmetic); 27 formats decode / 10 encodeF32 block formats
  (§11.8; `src/gguf.zig:1172–1210`); 8 adversarial golden
  vectors and the six byte-for-byte formats
  (`src/backend/quant/encode_golden_test.zig` via §11.10); 27 comptime size
  asserts (§11.7). Nothing else may be quantified.
- The next-episode teaser ("A transformer from scratch — the tensor
  directory becomes a transformer") matches Video 12's title and the
  chapter's own closing (§11.13); it must survive edits.
