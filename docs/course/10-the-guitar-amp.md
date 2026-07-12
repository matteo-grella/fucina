# Chapter 10 — The guitar amp: real-time neural audio

*Part IV — Sound*

Everything so far has computed at its own pace. Training in
[Chapter 8](08-training.md) took as long as it took; if an epoch ran a second
slower, nothing broke. This chapter is different in kind: a guitarist plugs
into an audio interface, and a neural network must transform every incoming
block of samples **before the next block arrives** — every ~1.3 milliseconds,
forever, without one miss. A missed deadline is not a slow benchmark; it is an
audible click through the speakers, mid-song.

This is the flagship chapter of the course, and the reason is not just that a
neural guitar amp is fun (it is). It is that real-time audio is the setting
where every discipline this library practices — explicit allocation, kernels
that never allocate, bit-exact parity contracts, measurement over assertion —
stops being a style preference and becomes the difference between music and
noise. And there is a historical symmetry to enjoy along the way: the language
we have been learning was itself born from exactly this problem.

The subject is `examples/nam/` — a complete, self-contained port of the
[Neural Amp Modeler](https://github.com/sdatkinson/neural-amp-modeler)
ecosystem. One binary loads any upstream `.nam` amp profile, plays it live
against your audio devices, benchmarks it, trains *new* profiles from captured
audio through the autograd engine of [Chapter 7](07-autograd.md), and
exchanges the result with the original Python/C++ tooling in both directions.
The repository's top-level README credits the ecosystem's author directly:
"**Steven Atkinson** — NeuralAmpModelerCore and neural-amp-modeler, the
reference for the entire NAM example" (`README.md`, Acknowledgments).

## 10.1 Full circle: a language born from audio

None of what follows is recorded anywhere in the Fucina repository, so it is
cited from primary sources outside it. Zig's creator, Andrew Kelley, conceived
the language while building the
[Genesis digital audio workstation](https://github.com/andrewrk/genesis),
after C and C++ — and, just as much, their library ecosystems — fell short for
low-latency, reliable audio software; before that he had built the libgroove
audio library. He announced Zig on February 8, 2016, in
["Introduction to the Zig Programming Language"](https://andrewkelley.me/post/intro-to-zig.html).

Hold that origin in mind while reading this chapter. Real-time audio makes a
short, brutal list of demands: no garbage collector that can pause you
mid-deadline, no hidden allocation inside a hot path, explicit control of
every buffer's lifetime, and the ability to reason about worst-case — not
average — behaviour. That list is close to a specification of Zig's design
values, and this chapter is the place in the course where you watch each one
earn its keep. A language conceived inside a digital audio workstation ends up,
a decade later, running a neural amplifier live from a terminal — trained,
benchmarked, and played by the same library, in the same binary. The circle
closes deliberately.

## 10.2 A model you can hold: the `.nam` file

An amp profile is a solved regression problem. Someone played a standardized
test signal through real gear — a tube amp, a fuzz pedal, a whole rig — and
recorded what came back. A neural network was then trained so that
`model(clean_signal) ≈ gear_output`, and the trained weights were saved to a
file. That file is a `.nam`, and there are thousands of free ones:
[Tone3000](https://www.tone3000.com) hosts a large public library, almost all
of them the "standard WaveNet" architecture this example runs at full fidelity
(`examples/nam/README.md`, Getting started).

> **ML note** — Why a *neural* network for an amplifier? Because distortion
> is nonlinearity, and nonlinearity is precisely what linear DSP cannot
> express. A linear filter scaled twice as loud produces the same shape twice
> as loud; a tube amp pushed twice as hard produces a *different shape* —
> that is what "breakup" is. The consequence threads through this whole
> chapter: input level changes **tone**, not just volume
> (`examples/nam/README.md`: "NAM models are nonlinear, so input level
> controls breakup, not just loudness"), and a nonlinear model cannot be
> resampled to another rate the way a linear filter can (§10.12).

The format is refreshingly plain. A `.nam` file is **one JSON document**:
`version`, `architecture`, `config`, `weights` (a flat float array), optional
`metadata`, optional `sample_rate` (`examples/nam/nam_file.zig:1-15`). The
reader accepts versions 0.5.0 through 0.7.x, matching the upstream loader's
gate exactly:

```zig
/// Newest file version we write and fully support (upstream
/// LATEST_FULLY_SUPPORTED_NAM_FILE_VERSION, get_dsp.h:66).
pub const latest_version = Version{ .major = 0, .minor = 7, .patch = 0 };
/// Oldest accepted version (upstream EARLIEST_SUPPORTED, get_dsp.h:67).
pub const earliest_version = Version{ .major = 0, .minor = 5, .patch = 0 };
```

*(from `examples/nam/nam_file.zig:29-33`)*

Four architectures exist in the wild — `pub const Arch = enum { wavenet,
lstm, convnet, linear };` (`examples/nam/nam_file.zig:60`) — and the example
loads and plays all of them, plus the current upstream trainer's
`SlimmableContainer` export. Training, however, targets WaveNet only; keep
that asymmetry in mind for §10.6 (`examples/nam/README.md`, Compatibility
guarantees; `examples/nam.zig:647`).

The part worth savouring is `weights`. The classic "standard" WaveNet — the
config behind most profiles you will download — is **13,802 floats**
(`examples/nam/README.md`; `examples/nam/train.zig:34-36`). Not billions.
Thirteen thousand. A model that imitates a tube amplifier convincingly enough
that guitarists use it on stage fits in 55 KB and stays resident in L1 cache.
"A model is just numbers plus a shape recipe" is a slogan from
[Chapter 3](03-tensors-from-scratch.md); here it is literal enough to read in
a text editor.

Loading those numbers is a cursor walking the flat array in a canonical
order. Every submodule consumes its slice and advances the cursor:

```zig
fn buildLayer(allocator: std.mem.Allocator, lc: *const nam_file.WaveNetLayerArray, l: usize, weights: []const f32, cursor: *usize) !Layer {
    const bg = lc.gateWidth(l);
    var conv = try StreamConv.initGrouped(allocator, lc.channels, bg, lc.kernel_sizes[l], lc.dilations[l], true, lc.groups_input);
    errdefer conv.deinit();
    cursor.* += conv.loadNamWeights(weights[cursor.*..]);

    var input_mixin = try StreamConv.initGrouped(allocator, lc.condition_size, bg, 1, 1, false, lc.groups_input_mixin);
    errdefer input_mixin.deinit();
    cursor.* += input_mixin.loadNamWeights(weights[cursor.*..]);
```

*(from `examples/nam/wavenet.zig:305-313`)*

Two things to notice. First, the `errdefer` ladder from
[Chapter 3](03-tensors-from-scratch.md) again: every partially-built resource
registers its cleanup the moment it exists, so a malformed file that fails
halfway leaks nothing. Second, the end-of-stream check hides a genuine format
subtlety:

```zig
// head_scale: the final float of the stream overrides the JSON copy
// (model.cpp:632).
if (cursor + 1 != weights.len) return error.WeightCountMismatch;
```

*(from `examples/nam/wavenet.zig:233-235`)*

The JSON `config` *does* carry a `head_scale` field — but at runtime the
engine takes the value from the **last float of the weight stream**, which
overrides the JSON copy, because that is what the upstream C++ runtime does.
A port that read the JSON field instead would pass every schema check and
produce subtly wrong audio on any file where the two disagree. Faithful
porting means porting the behaviour, not the documentation.

One more design decision worth naming: `NamModel` retains the original file
bytes — `raw_bytes: []u8` — "for byte-faithful re-export"
(`examples/nam/nam_file.zig:248-249`). That single field is what makes the
GGUF interchange of §10.8 *lossless by construction* rather than lossless by
careful re-serialization.

Untrusted input gets checked errors, not assertions. A `.nam` file arrives
from the internet, so shapes that would corrupt memory are rejected with
`error.InvalidConvShape` — `taps == 0` would underflow the padding
arithmetic, `taps > max_taps` would overrun fixed tile arrays, a huge
dilation would blow up the history allocation
(`examples/nam/stream_conv.zig:52-60`). The upstream C++ core asserts these
only in debug builds; the port's module docs record each such divergence as a
deliberate deviation (`examples/nam/engine.zig:6-7`).

## 10.3 WaveNet: exponential context at linear cost

WaveNet is the architecture that made neural audio generation practical, and
its central trick — **dilated causal convolution** — is the whole reason a
13,802-weight model can imitate an amplifier.

Start with *causal*: an amp cannot hear the future, so every convolution tap
looks only backwards in time. Output `y[t]` depends on `x[t]`, `x[t−d]`,
`x[t−2d]`, … and never on `x[t+anything]`. This is the streaming counterpart
of the `causalConv1d` op you met in the operation library
([Chapter 5](05-the-operation-library.md)); the two share the same `[tap, in,
out]` weight orientation by design (`examples/nam/stream_conv.zig:3-5`).

Now *dilated*: `d` is the spacing between taps. A 3-tap conv with dilation 1
sees 3 consecutive samples; with dilation 512 it sees 3 samples spread over
1,025 positions. Stack layers with dilations 1, 2, 4, …, 512 and each layer
doubles how far back the network can hear, while adding only a constant
amount of work. The receptive field of one such stack is
`1 + Σ d·(k−1)` — implemented three separate times in the example
(`examples/nam/nam_file.zig:134-141`, `examples/nam/train.zig:55-61`, and the
ConvNet variant at `examples/nam/nam_file.zig:186-190`; the first also adds a
`(head_kernel − 1)` term, zero for the classic config's `head_kernel = 1`,
`examples/nam/train.zig:2372`), the parity habit of
[Chapter 6](06-going-fast-on-cpus.md) applied to arithmetic: derive it once,
verify it thrice.

Work it out for the real config. The classic spec is *2 arrays, 16→8
channels, k=3, dilations 1..512, Tanh, head_scale 0.02 — 13,802 weights*
(`examples/nam/train.zig:34-43`). This course snippet does the arithmetic and
`zig test` confirms it (course code, not repo code):

```zig
fn arrayReceptiveField(kernel: usize, dilations: []const usize) usize {
    var rf: usize = 1;
    for (dilations) |d| rf += d * (kernel - 1);
    return rf;
}

test "the classic WaveNet sees 4093 past samples" {
    const dilations = [_]usize{ 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 };

    // One array: 1 + (1+2+...+512) * (3-1) = 1 + 1023*2 = 2047.
    const per_array = arrayReceptiveField(3, &dilations);
    try std.testing.expectEqual(@as(usize, 2047), per_array);

    // Two arrays in series: causal receptive fields compose as
    // rf_total = rf_a + rf_b - 1.
    const model_rf = per_array + per_array - 1;
    try std.testing.expectEqual(@as(usize, 4093), model_rf);

    // At 48 kHz that is ~85 ms of context.
    const ms = @as(f64, @floatFromInt(model_rf)) / 48000.0 * 1000.0;
    try std.testing.expect(ms > 85.0 and ms < 86.0);
}
```

So each new output sample is computed from the previous **4,093** samples —
about 85 ms of guitar signal, enough to capture power-supply sag and the slow
dynamics that make an amp feel like an amp — yet the per-sample cost is just
20 small dilated convs (2 arrays × 10 layers) plus 1×1 mixes. Exponential
ears, linear bill. That is the entire genius of dilation.

> **ML note** — Compare the alternatives. A dense layer over 4,093 inputs
> would need 4,093 weights *per output channel per layer* — and it would
> learn nothing about time-shift invariance. An RNN (the `lstm` architecture
> in the same format) carries context in a recurrent state instead, paying
> sequential dependence: you cannot compute sample `t+1`'s state before
> sample `t`'s. Dilated convolution gets long context, weight sharing, *and*
> within-block parallelism. That combination is why WaveNet dominates this
> ecosystem.

The full per-block dataflow is documented as algebra in the engine's module
doc — read it slowly, because §10.7 will show you the *same equations* written
a second time in autograd ops:

```zig
//! Per block: condition = raw input; for each layer array (array 0 takes the
//! condition as layer input and zeroes its head accumulator; array i>0 takes
//! the previous array's residual outputs and head outputs):
//!   x = rechannel(input)                         [Conv1x1, no bias]
//!   per layer: z = dilated_conv(x) + input_mixin(condition)
//!              a = activation(z)                 [gated: act(top)*act2(bottom)]
//!              head_acc += head1x1(a) or a
//!              x = x + layer1x1(a)               [or x unchanged if inactive]
//!   head_out = head_rechannel(head_acc)          [causal conv, has memory when k>1]
//! Output = head_scale * head_out of the last array, optionally through the
//! post-stack head (activation BEFORE each conv, applied to the scaled
//! stream). All buffers are sized once at init; process() is allocation-free.
```

*(from `examples/nam/wavenet.zig:4-15`)*

Three structural ideas hide in those lines. The **condition mixin**: the raw
input is re-fed to *every* layer, not just the first, so deep layers never
lose sight of the actual guitar signal. The **residual stream**: each layer
adds its contribution to `x` rather than replacing it — the skip-connection
idea that lets gradients and signal flow through deep stacks. And the **head
accumulator**: every layer also deposits into a running sum that becomes the
output, so the final answer is a collaboration of all depths rather than the
last layer's monologue. You will meet all three again, at scale, in the
transformer of [Chapter 12](12-a-transformer-from-scratch.md).

Gated variants (`a = act(top) * act2(bottom)`), FiLM conditioning, grouped
convolutions and nested `condition_dsp` engines are all supported for loading
— real downloadable files use them — but they are refinements of the same
skeleton, and the module doc plus `examples/nam/nam_file.zig:88-120` are the
reference when you want them.

## 10.4 Streaming convolution: state makes chunking disappear

Here is the problem that separates offline inference from live audio. The
formula above assumes you have the whole signal: to compute `y[t]` you index
`x[t − k·d]` freely. Live, the OS hands you 64 samples at a time, and sample
`t − 512` belongs to a buffer that no longer exists.

The fix is one idea: **each convolution privately remembers the last
`dilation·(taps−1)` input rows it consumed** — exactly the window any future
output could still reach back into. The example calls this the *history*, and
its streaming conv states the contract in its module doc: "the per-conv
history holds the last `dilation*(K-1)` input rows so output is independent
of how the stream is chunked. All buffers are allocated at init; `process` is
allocation-free" (`examples/nam/stream_conv.zig:1-9`).

The entire "streaming" idea is five lines inside the kernel — a tap either
reads this chunk or the saved history, and nothing else changes:

```zig
var x_rows: [max_taps][]const f32 = undefined;
while (t < frames) : (t += 1) {
    for (0..self.taps) |k| {
        const shifted = t + k * self.dilation;
        x_rows[k] = if (shifted >= pad)
            input[(shifted - pad) * in_ch ..][0..in_ch]
        else
            self.history[shifted * in_ch ..][0..in_ch];
    }
```

*(from `examples/nam/stream_conv.zig:223-231`)*

Advancing the history is a separate, explicit step:

```zig
pub fn push(self: *StreamConv, input: []const f32, frames: usize) void {
    const in_ch = self.in_channels;
    if (in_ch == 0 or self.history.len == 0) return;
    const pad = self.history.len / in_ch;
    if (frames >= pad) {
        @memcpy(self.history, input[(frames - pad) * in_ch ..][0 .. pad * in_ch]);
        return;
    }
    const keep = pad - frames;
    std.mem.copyForwards(f32, self.history[0 .. keep * in_ch], self.history[frames * in_ch ..][0 .. keep * in_ch]);
    @memcpy(self.history[keep * in_ch ..], input[0 .. frames * in_ch]);
}
```

*(from `examples/nam/stream_conv.zig:300-313`)*

The `process`/`push` split is a deliberate compute-then-commit design:
`process` "does NOT advance history — call `push` with the same input
afterwards" (`examples/nam/stream_conv.zig:158-159`). Calling `process` twice
without `push` is idempotent, which makes chunk-invariance trivially testable
— and forgetting `push` is the documented gotcha: your convs go subtly stale
and the audio is wrong in ways your ears notice before your tests do.

Build the miniature yourself. This is course code (single channel, three
taps), compile-checked with `zig test`; its test is the soul of the real
`stream_conv_tests.zig` — process a signal in one pass and in awkward chunks,
then demand **bit-identical** output:

```zig
const MiniStreamConv = struct {
    weight: [3]f32, // taps; weight[taps-1] multiplies the newest sample
    bias: f32,
    dilation: usize,
    history: []f32, // last dilation*(taps-1) inputs, oldest first

    /// y[t] = bias + sum_k x[t + k*dilation - pad] * w[k], where x reads
    /// `history` for pre-chunk indices. Does NOT advance the history —
    /// call `push` with the same input afterwards.
    fn process(self: *const MiniStreamConv, input: []const f32, out: []f32) void {
        const pad = self.history.len;
        for (out, 0..) |*y, t| {
            var acc: f32 = self.bias;
            for (self.weight, 0..) |w, k| {
                const shifted = t + k * self.dilation;
                const x = if (shifted >= pad) input[shifted - pad] else self.history[shifted];
                acc += x * w;
            }
            y.* = acc;
        }
    }

    /// Slide the history window forward by `input.len` samples.
    fn push(self: *MiniStreamConv, input: []const f32) void {
        const pad = self.history.len;
        if (input.len >= pad) {
            @memcpy(self.history, input[input.len - pad ..]);
            return;
        }
        const keep = pad - input.len;
        std.mem.copyForwards(f32, self.history[0..keep], self.history[input.len..]);
        @memcpy(self.history[keep..], input);
    }
};

test "output is independent of how the stream is chunked" {
    const taps = 3;
    const dilation = 4;

    var signal: [256]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(42);
    for (&signal) |*v| v.* = prng.random().float(f32) * 2.0 - 1.0;

    // One pass over the whole signal.
    var hist_a = [_]f32{0} ** (dilation * (taps - 1));
    var conv_a = MiniStreamConv{ .weight = .{ 0.25, -0.5, 1.0 }, .bias = 0.1, .dilation = dilation, .history = &hist_a };
    var out_a: [256]f32 = undefined;
    conv_a.process(&signal, &out_a);

    // The same signal in awkward chunks.
    var hist_b = [_]f32{0} ** (dilation * (taps - 1));
    var conv_b = MiniStreamConv{ .weight = .{ 0.25, -0.5, 1.0 }, .bias = 0.1, .dilation = dilation, .history = &hist_b };
    var out_b: [256]f32 = undefined;
    const chunks = [_]usize{ 1, 7, 64, 3, 100, 81 };
    var pos: usize = 0;
    for (chunks) |n| {
        conv_b.process(signal[pos..][0..n], out_b[pos..][0..n]);
        conv_b.push(signal[pos..][0..n]);
        pos += n;
    }
    try std.testing.expectEqual(@as(usize, 256), pos);

    // Bit-identical, not merely close: same accumulation order per element.
    for (out_a, out_b) |a, b| try std.testing.expectEqual(a, b);
}
```

The real kernel wraps the same logic in the SIMD vocabulary of
[Chapter 6](06-going-fast-on-cpus.md): output channels are processed
`vector_len` at a time (`std.simd.suggestVectorLength(f32) orelse 4`,
`examples/nam/stream_conv.zig:13-14`), and each weight-vector load is
amortized across `time_tile = 8` frames of fused multiply-adds:

```zig
for (0..self.taps) |k| {
    const rows = tile_rows[k];
    for (0..in_ch) |i| {
        const wv: Vf32 = self.weight[(k * in_ch + i) * out_ch + o ..][0..vector_len].*;
        inline for (0..time_tile) |tt| {
            acc[tt] = @mulAdd(Vf32, @splat(rows[tt][i]), wv, acc[tt]);
        }
    }
}
```

*(from `examples/nam/stream_conv.zig:190-198`)*

The comment on `time_tile` names what this buys: "the Eigen-GEMM-style
amortization upstream gets from processing whole blocks as matrix products"
(`examples/nam/stream_conv.zig:146-149`) — the C++ reference leans on Eigen;
the Zig port gets the same register-level reuse from an `inline for`.

> **Zig note** — `pub fn process(..., comptime accumulate: bool)`
> (`examples/nam/stream_conv.zig:160`) is the comptime-specialization pattern
> from [Chapter 6](06-going-fast-on-cpus.md) in miniature: one body, two
> compiled kernels — overwrite-output and add-into-output (the latter
> exercised by `stream_conv_tests.zig`'s "accumulate adds on top" test; the
> engine's head accumulator itself sums layer contributions with explicit
> `+=` loops, `examples/nam/wavenet.zig:626-633`) — with the branch resolved
> at compile time, not per sample.

And the invariant that makes the mini-test's `expectEqual` legitimate at full
scale: "per output element the accumulation order (bias, then k-major i-inner
FMA) is identical in every path, so results are bit-identical across tile
boundaries and chunkings" (`examples/nam/stream_conv.zig:151-159`). Where
[Chapter 6](06-going-fast-on-cpus.md) accepted tolerance when SIMD
reassociated a reduction, this kernel is engineered so that no path
reassociates *relative to any other path* — because §10.5's parity gates
demand byte-identical output across block sizes.

## 10.5 The engine facade, prewarm, and the exact tanh

Four architectures, one interface. The unifying facade is a tagged union —
the no-vtable polymorphism pattern this codebase prefers:

```zig
pub const Impl = union(nam_file.Arch) {
    wavenet: wavenet.WaveNetEngine,
    lstm: models.LstmEngine,
    convnet: models.ConvNetEngine,
    linear: models.LinearEngine,
};
```

*(from `examples/nam/engine.zig:30-35`)*

`Engine.init` validates that the model is mono-in/mono-out *before*
construction — a `condition_size` other than 1 would make the mixin convs
"index past the mono buffer (panic in Debug, OOB read in ReleaseFast)"
(`examples/nam/engine.zig:39-55`) — another checked-error fence around
untrusted files. Dispatch is a single `switch`; teardown uses `inline else`
to expand one arm per variant at compile time
(`examples/nam/engine.zig:85-87`).

Two behavioural contracts live here, and both exist for parity with upstream:

**Prewarm.** A freshly reset stateful model has all-zero histories, so its
first receptive-field-worth of output is garbage relative to steady state.
`reset(max_frames, prewarm)` therefore feeds the model zeros before real
audio — and the count is rounded **up to whole buffers**:
"Reset(sampleRate, maxBufferSize) prewarms by default with zero samples
rounded up to whole buffers (ceil(prewarm/maxBuf)*maxBuf, dsp.cpp:47-81) —
golden parity vs upstream tools/render depends on reproducing exactly that"
(`examples/nam/engine.zig:1-7`). Not "about enough zeros" — *exactly* the
upstream rounding, or the golden renders drift by a block's worth of warmup.

**The exact tanh.** The classic WaveNet applies `tanh` twenty times per
sample, so its cost and its bits both matter. The port evaluates it in SIMD
under a written contract:

```zig
/// Contract (what the parity gates rely on):
/// - Value-only dependence: a lane's result depends only on its input value,
///   never on lane position or N — so `tanhF32` (N=1) is bit-identical to the
///   bulk SIMD path and engine output stays byte-identical across block sizes.
/// - Deterministic across machines: IEEE add/mul/div/round and correctly
///   rounded @mulAdd only — no rcpps-style estimates, whose results differ
///   between CPU vendors. (Targets without FMA hardware get the same bits via
///   softfloat, just slower.)
/// - Accuracy: ≤ 3e-7 absolute vs the correctly rounded (f64) tanh over all of
///   f32 (measured ≲2 ulp; the activations test sweep enforces the bound) —
///   the same class as libm tanhf, ~20x inside the 5e-6 golden render gates.
/// - Specials: ±0 → ±0, subnormals → x, ±inf → ±1, NaN → NaN.
```

*(from `examples/nam/activations.zig:16-29`)*

The implementation beneath it (`examples/nam/activations.zig:30-82`) is a
small masterclass in float bit-craft — sign-stripping via `@bitCast` to `u32`
lanes, an odd Taylor series for small inputs, the fdlibm hi/lo `ln2` split
and exponent assembly `(k + 127) << 23` for the large branch, explicit NaN
re-propagation — but the *contract comment* is the part to internalize.
It promises value-only lane math (so scalar and SIMD paths agree bit-for-bit,
so output is byte-identical across block sizes), vendor-independent
determinism (no estimate instructions), and a measured accuracy bound that a
test sweep *enforces*. This is [Chapter 6](06-going-fast-on-cpus.md)'s parity
religion, written as API documentation.

What does all this discipline buy? The compatibility section of
`examples/nam/README.md` quotes the measured results (as always in this
course: measured, dated, machine-specific — not asserted):

- vs upstream `tools/render` on the upstream example models: "standard
  WaveNet max |diff| 2.3e-6 / RMS 6.5e-8 (about 20× inside upstream's own
  5e-5 cross-implementation tolerance)";
- the tanh itself: "measured ≤ 1.9 ulp / 9.5e-8 abs vs correctly rounded
  tanh"; "Output is byte-identical across block sizes."

## 10.6 Training your own amp: capture and data hygiene

Playing other people's profiles is half the story. The same binary trains new
ones, and the pipeline is a compact lesson in something frameworks rarely
teach: **your dataset is an instrument, and it must be calibrated**.

The physical procedure (diagrammed in `examples/nam/README.md`, Hardware):
the interface's line output plays a standardized test signal into your amp
(through a reamp box, which restores instrument level and impedance), and the
amp's output is recorded back — the "reamp" pair. The test signal is the same
**v3 capture file** the official NAM trainer uses (`v3_0_0.wav`), recognized
by MD5 checksum (`examples/nam/data.zig:32-49`): 9,120,000 samples at 48 kHz
— 3 minutes 10 seconds — with a precisely known internal structure
(`examples/nam/data.zig:14-27`). Then either the one-step
`profile --signal v3_0_0.wav --reamp-out reamp.wav --out my-amp.nam ...`
(play, record, train) or the two-step `train --input ... --output ...` on a
pair you recorded in a DAW.

Before a single gradient step, the data must pass checks ported from the
upstream trainer:

- **Latency calibration.** The capture file embeds *blips* — impulses at
  known sample positions (`blip_locations = [_]usize{ 504000, 552000 }`,
  `examples/nam/data.zig:26`). Your interface's round-trip delay shifts the
  recording by some unknown number of samples; the calibrator measures a
  noise floor from a known-silent stretch, scans for the blips' arrival, and
  recovers the shift (`calibrateLatencyV3`,
  `examples/nam/data.zig:100-147`). Misalign x and y by even a few samples
  and you are asking the model to predict the future — training will
  converge to a worse amp, silently.
- **No clipping.** Any `|y| ≥ 1.0` refuses the capture outright
  (`examples/nam/data.zig:230-234`): a clipped sample is information
  destroyed at the ADC, and no optimizer recovers it.
- **Pre-silence.** 0.4 s of *exact zeros* required before the training split
  (`examples/nam/data.zig:238-244`) — the streaming model starts from zero
  state, so the data must too.
- **Replicate consistency.** The v3 signal contains the validation segment
  twice; if the two recordings of it differ by self-ESR > 0.01, your rig
  drifted mid-capture (knob bumped, tube warmed) and the pair is rejected
  (`checkV3`, `examples/nam/data.zig:156-165`).

ESR — Error-to-Signal Ratio — is the domain's quality number, and it is
twelve lines you can read whole:

```zig
pub fn esr(pred: []const f32, target: []const f32) f64 {
    std.debug.assert(pred.len == target.len and pred.len > 0);
    var num: f64 = 0;
    var den: f64 = 0;
    for (pred, target) |p, t| {
        const d = @as(f64, p) - @as(f64, t);
        num += d * d;
        den += @as(f64, t) * @as(f64, t);
    }
    if (den == 0) return std.math.inf(f64);
    return num / den;
}
```

*(from `examples/nam/data.zig:56-67`)*

Mean squared error *normalized by the target's energy* — so 0.01 means "the
residual carries 1% of the signal's power" regardless of how loud the capture
was. The console bands are upstream's, verbatim: **< 0.01 "Great!"**,
< 0.035 "Not bad!" (`examples/nam/data.zig:70-76`;
`examples/nam/README.md:183-184`).

Windowing follows upstream's `nx`/`ny` semantics
(`examples/nam/data.zig:190-212`): each training example is an input window
of `nx + ny − 1` samples and a target of the last `ny`, where `nx` is the
model's receptive field (4,093 for the classic spec — the number you derived
in §10.3) and `ny` defaults to 8,192 (`examples/nam.zig:541`). The input
window is longer than the target by exactly `nx − 1`: those samples are the
context the first predicted output needs. The last 9 seconds of the capture
are held out as the validation split (`examples/nam/README.md:171-172`).

## 10.7 The trainer: one architecture, two execution regimes

Now the payoff of building a whole library. The streaming engine of §10.3–4
is hand-rolled, stateful, allocation-free — perfect for inference,
undifferentiable as written. Training needs gradients. So the example writes
the *same WaveNet a second time*, as a stateless full-sequence graph over the
tagged tensors of [Chapter 4](04-axes-with-names.md), executed by the
autograd engine of [Chapter 7](07-autograd.md):

```zig
for (ap.layers, 0..) |*lp, l| {
    const conv = try x.causalConv1d(ctx, .time, .ch, .tap, .bn, &lp.conv_w, a.dilations[l], null);
    const conv_b = try conv.add(ctx, &lp.conv_b);
    const mix = try cond.dot(ctx, &lp.mixin_w, .cond);
    const z = try conv_b.add(ctx, &mix);
    const activated = try z.tanh(ctx);
    acc = if (acc) |prev| try prev.add(ctx, &activated) else activated;
    const l1 = try activated.dot(ctx, &lp.l1_w, .bn);
    const l1_b = try l1.add(ctx, &lp.l1_b);
    x = try x.add(ctx, &l1_b);
}
```

*(from `examples/nam/train.zig:279-289`)*

Put this beside the module-doc algebra of §10.3 and read them line against
line: `z = dilated_conv(x) + input_mixin(condition)` → `causalConv1d` +
`add` + `dot` + `add`; `a = tanh(z)` → `tanh`; `head_acc += a` → the `acc`
fold; `x = x + layer1x1(a)` → the last three lines. **The same mathematics,
once as a streaming kernel measured in microseconds, once as five autograd
ops measured in gradients.** This side-by-side is the single best exhibit in
the course for what "the library grew through real applications" means: the
inference regime wanted state and zero allocation; the training regime wanted
composability and `backward()`; nothing forced one abstraction to serve both
badly.

Get the loss function's identity precise, because it is a common confusion:
**the training loss is MSE** — "mean over elements, the torch F.mse_loss
default" (`examples/nam/train.zig:300-311`) — optionally plus a multi-
resolution STFT spectral term at weight 0.0005 in the packed recipe
(`default_mrstft_weight: f32 = 0.0005`, `examples/nam/train.zig:101`;
`LossOptions.mrstft_weight` defaults to 0, `:130-133`). **ESR is the
validation metric, not the loss** (`examples/nam/train.zig:1-7`). The
distinction is standard ML practice you should carry everywhere: optimize a
smooth, well-conditioned objective; *judge* with the interpretable domain
number.

The epoch loop is [Chapter 8](08-training.md)'s vocabulary, deployed:

```zig
for (0..epochs) |epoch| {
    opt.config.lr = lr0 * std.math.pow(f32, gamma, @floatFromInt(epoch));
    // Deterministic shuffle (Fisher-Yates over rng.at counters).
    for (0..example_count) |idx| {
        const j = idx + rng.at(seed +% 0x5851f42d4c957f2d, epoch * example_count + idx) % (example_count - idx);
        std.mem.swap(usize, &order[idx], &order[j]);
    }

    const epoch_start = nowNs(io);
    var loss_sum: f64 = 0;
    for (0..steps_per_epoch) |step_index| {
        for (order[step_index * batch_size ..][0..batch_size]) |example_index| {
            const example = dataset.get(example_index);
            const scope = ctx.openExecScope();
            defer ctx.closeExecScope(scope);
            const loss = try model.segmentLossWithOptions(&ctx, example.input, example.target, loss_options);
            const scaled = try loss.scale(&ctx, 1.0 / @as(f32, @floatFromInt(batch_size)));
            loss_sum += try loss.item();
            try scaled.backward(&ctx);
        }
        try opt.step(&ctx);
        opt.zeroGrad();
    }
```

*(from `examples/nam.zig:702-724`)*

Every ingredient is one you already own. The one-line exponential LR schedule
(`lr = lr0 · γ^epoch`). The counter-based deterministic RNG driving a
Fisher-Yates shuffle — same run, same shuffle, forever, per the determinism
stance of [Chapter 5](05-the-operation-library.md) (and `+%` is Zig's
wrapping add, doing exactly what a seed-mixing constant wants). Exec scopes
adopting each example's intermediates so memory is reclaimed per example. And
batch 16 **via gradient accumulation**: scale each loss by 1/16, `backward()`
per example so gradients *sum* into the parameters' `.grad`, one `opt.step()`
per 16 — big-batch mathematics on a small machine's memory, the pattern
[Chapter 8](08-training.md) taught generalizing unchanged. The published
recipe: "MSE, Adam lr 0.004, gamma=0.993, batch 16, 100 epochs"
(`examples/nam/README.md:181-182`).

After each epoch, validation does something quietly brilliant: it does *not*
run the autograd graph. It exports the current weights into the **streaming
inference engine** and streams the held-out split through that
(`examples/nam/train.zig:4-7`). So every epoch continuously proves that the
trainable's weight-extraction order matches the engine's weight-cursor order
— the exact class of bug (a transposed conv layout, a swapped layer) that
otherwise survives until a user's amp sounds wrong. The best epoch by
validation ESR is what gets exported, not the last
(`examples/nam.zig:726-736`).

Two scope notes, stated as precisely as the code states them. Training
targets WaveNet: the spec presets (`standard`, `tiny`, `a2`, `a2-nano`,
`packed`) are all WaveNet shapes, and fine-tuning an existing file via
`--init` prints "error: --init currently trains WaveNet .nam files only" for
anything else (`examples/nam.zig:647`); LSTM/ConvNet/Linear profiles are
load-and-play. And everything here is CPU, f32, single process — no GPU
anywhere in this example.

## 10.8 Export and interchange: both directions, byte for byte

A trained model you cannot share is a science-fair project. The exporter
writes `.nam` v0.7.0 in the modern upstream exporter shape, with the full
upstream metadata schema — date, measured loudness, your `--name`/`--gear-*`
fields, the latency-calibration record, the final ESR
(`examples/nam/nam_export.zig`; `examples/nam/README.md:184-186`). The
compatibility claims are then *measured*, in both directions
(`examples/nam/README.md`, Compatibility guarantees):

- **Import:** any upstream-tooling `.nam` of the supported architectures,
  0.5.0–0.7.x, including the current trainer's `SlimmableContainer` export.
- **Export:** profiles load in upstream `NeuralAmpModelerCore` (`loadmodel`),
  and "rendering through the upstream core matches Fucina (6.7e-8 max on a
  trained classic profile; 2.7e-8 max on a packed-container smoke)". The test
  plan (`examples/nam/README.md`, Test plan) even includes a Python
  re-import oracle: `nam.models.init_from_nam(json.load(open('model.nam')))`
  against the official trainer package.

So a profile trained by this Zig binary plays in the official plugin, and a
profile trained by the official Python trainer plays here. Interchange is a
verified property, not a hope.

There is also a GGUF side door — and its design says something about
engineering taste. GGUF ([Chapter 11](11-model-files-and-quantization.md)
covers the format itself) here is **a lossless container, never a runtime
format**: the file carries one flat f32 tensor plus "the ENTIRE original .nam
JSON byte-verbatim in the string KV `nam.file_json` … so GGUF -> .nam export
is byte-identical by construction, the strongest possible round-trip
guarantee" (`examples/nam/gguf_compat.zig:1-10`). And quantization — the
central topic of the next chapter — "is refused by design (13.8k-param
models: no block-divisible dims, no bandwidth win, real ESR risk)" (same
lines). A 55 KB model is already L1-resident; quantizing it would save
nothing and risk audible error. Knowing when a technique does *not* apply is
part of owning it.

## 10.9 The live path: 1,333 microseconds

Now the chapter's summit. Offline rendering can be leisurely; `live` cannot.

The physics first. Audio hardware runs at a fixed sample rate — 48,000
samples per second here (`pub const standard_sample_rate: f64 = 48000.0;`,
`examples/nam/data.zig:12`). The OS does not deliver samples one at a time;
it delivers *periods* — blocks of, by default, 64 frames
(`examples/nam/live.zig`, `Options.period` default 64) — by calling your
callback on a dedicated realtime thread. The contract is unforgiving: the
callback must return the processed block before the next one lands. The
budget per block is pure arithmetic:

```
64 frames ÷ 48,000 frames/s = 1.333 ms = ~1333 µs
```

Miss it and the hardware plays whatever stale bytes are in the buffer — an
audible click. Miss it regularly and the instrument is unplayable.

Against that budget, the measured cost of the model, quoted from
`examples/nam/README.md` (Performance — a dated, machine-specific snapshot,
like every benchmark in this course): "standard WaveNet ≈ 49 µs per 64-frame
block @48 kHz on one core (ReleaseFast, i9-13950HX P-core; 2026-07-03 x86
snapshot) ≈ 27× realtime — 1.8× faster than upstream `benchmodel` at the
documented protocol (stock `-Ofast` Release, exact tanh: 87 µs; its default
fast-tanh: 83 µs)… On M1 Max ≈ 80 µs/block". Divide the budget by the cost
and you get roughly 27× headroom on that x86 core and ~17× on the M1 Max —
comfortable for one model, and meaningful because a *chain* (§10.12) runs
several serially.

A cautionary footnote that is itself a lesson: the header comment of
`examples/nam/live.zig:1-4` still cites "~67 us per 64-frame block …
measured 2026-06-12, M1 Max" — a snapshot that predates a tanh-parity change
and its SIMD recovery; the README's ≈80 µs (and the 67→162→80 µs history
behind it, `examples/nam/README.md:331-333`) is the current record. Numbers
rot; dates are what keep them honest. When a doc comment and a maintained
README disagree, trust the one that carries the newer date — and measure your
own machine anyway, which is exactly what `bench` is for:

```zig
const ns_per_block = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(total_blocks));
const budget_ns = @as(f64, @floatFromInt(blocksize)) / rate * 1e9;
...
try stdout.print("realtime:     {d:.1}x headroom\n", .{budget_ns / ns_per_block});
```

*(from `examples/nam.zig:330-336`)*

`zig build nam -Doptimize=ReleaseFast -- bench my-amp.nam` prints your
per-block cost against your budget. Run it before ever going live — and run
it in ReleaseFast, because "debug builds are ~20× slower and will not keep up
in realtime" (`examples/nam/README.md:23-24`). Recall
[Chapter 1](01-just-enough-zig.md): optimization mode is a build-time
decision in Zig, and here it is the difference between an instrument and a
noise generator.

The audio plumbing is the vendored miniaudio C library behind a thin shim:
`extern fn` declarations, an `opaque {}` device handle, and a C-callable
callback type (`examples/nam/audio.zig:11-27`):

```zig
pub const RawCallback = *const fn (user: ?*anyopaque, output: ?[*]f32, input: ?[*]const f32, frame_count: c_uint) callconv(.c) void;
```

> **Zig note** — This one line is Zig's C interop story in miniature:
> `callconv(.c)` makes the function callable from C code; `?[*]f32` is a
> nullable many-item pointer (C's `float*` with the nullness made explicit in
> the type); `?*anyopaque` is `void*`. The callback recovers its typed state
> with `@ptrCast(@alignCast(user.?))` — the unsafe cast is *localized to one
> line at the C boundary* instead of being smeared through the program.

One honesty note before the deep dive: the live path is "Tested on macOS /
Apple Silicon (the audio layer is vendored miniaudio, so Linux should work
too but is untested)" (`examples/nam/README.md:14-16`). The chapter follows
the code in stating exactly that, no more. (Also macOS-specific and worth
knowing: microphone permission is attributed to your *terminal app*, and "a
denied permission yields silence with no error",
`examples/nam/README.md:92-94`.)

## 10.10 Inside the callback: how "no allocation" is guaranteed

The rule for code on the realtime audio thread is famous in audio
programming: **no allocation, no locks, no syscalls**. Any of the three can
block for an unbounded time — an allocator may take a lock or page in memory;
a mutex may be held by a lower-priority thread (priority inversion); a
syscall may schedule you out — and an unbounded pause inside a 1,333 µs
deadline is a click. Most languages make this rule a matter of vigilance.
The interesting thing about this codebase is how much of it is *structural*.

Here is the callback's opening — the doc comment is the rule, and the code
under it is the proof:

```zig
/// The realtime data callback. No allocation, no locks, no syscalls.
pub fn audioCallback(user: ?*anyopaque, output: ?[*]f32, input: ?[*]const f32, frame_count: c_uint) callconv(.c) void {
    const shared: *Shared = @ptrCast(@alignCast(user.?));
    const frames: usize = frame_count;
    const out_ptr = output orelse return;
    const out = out_ptr[0..frames];
    const in_ptr = input orelse {
        @memset(out, 0);
        return;
    };
    const raw_in = in_ptr[0..frames];

    // Tuner tap first: raw (pre-trim) input, so the reading is independent
    // of the trim knob; fed even for oversize blocks.
    if (shared.tap) |tap| tap.push(raw_in);
```

*(from `examples/nam/live.zig:268-282`)*

Walk the guarantees one by one:

**Every buffer already exists.** Before the stream starts, the `live` command
preallocates all audio-thread scratch at a capacity of
`frame_cap = @max(2048, period * 4)` — the input-trim buffer, the gate-gain
buffer, and two inter-stage "ping-pong" buffers
(`examples/nam.zig:1352-1364`). Every engine and cab is likewise sized to
`frame_cap` at load, and prewarmed. The callback's job is to *fill* buffers,
never to *find* them. This is the payoff of the discipline stated
independently in every engine's module doc — "All buffers are allocated at
init; `process` is allocation-free" (`examples/nam/stream_conv.zig:9`;
same contract in `wavenet.zig:15`, `engine.zig`, `ir_cab.zig:66-68`). The
port even *tightened* the reference here: upstream's ConvNet/LSTM allocate
per `process()` call; "all engines here are allocation-free after
init/reset — a deliberate, numerics-preserving deviation"
(`examples/nam/models.zig:3-5`).

**Control crosses threads only as atomics.** The UI thread (keyboard, MIDI)
and the audio thread share exactly one struct, and every field of it is an
`std.atomic.Value`, audio-thread-private, or set before the stream starts and
immutable while it runs:

```zig
pub const Shared = struct {
    /// Index into profiles; the callback loads it acquire.
    current: std.atomic.Value(usize) = .init(0),
    bypass: std.atomic.Value(bool) = .init(false),
    /// Output mute (silent tuning, guitar swaps). The chain still runs so
    /// the stateful WaveNet streams stay warm — only the device buffer is
    /// zeroed, so unmute is click-free.
    mute: std.atomic.Value(bool) = .init(false),
    /// Output gain as f32 bits.
    gain_bits: std.atomic.Value(u32) = .init(@bitCast(@as(f32, 1.0))),
```

*(from `examples/nam/live.zig:93-102`)*

No mutex anywhere near the audio thread. Gain knobs are `f32` values stored
as `u32` bits via `@bitCast` — the same integer representation the `fetchMax`
peak meter below needs — so a knob turn is one atomic store, and the audio
thread can never observe a torn float. Switching amp profiles mid-song is *one atomic index store* into a
preloaded, prewarmed array of chains ("switching is one atomic index store
and the callback never allocates, locks, or touches the Fucina thread pool",
`examples/nam/live.zig:1-6`). Note that last clause: the worker pool from
[Chapter 6](06-going-fast-on-cpus.md) — with its parking and waking — is
exactly the kind of machinery a realtime thread must not touch, and the
streaming engines are single-threaded on purpose.

**Even the meters are lock-free.** The peak meter wants the loudest sample
since the UI last looked. The trick: non-negative IEEE-754 floats have bit
patterns that order like their values, so an *unsigned integer* atomic max is
a float max:

```zig
fn atomicMaxF32(cell: *std.atomic.Value(u32), value: f32) void {
    _ = cell.fetchMax(@bitCast(value), .monotonic);
}
```

*(from `examples/nam/live.zig:264-266`; the doc comment at :256-263 explains
why a single fetchMax RMW beats a compare-and-swap loop here — no lost update
against the reader's `swap(0)`.)*

**Edge cases degrade, never overrun.** If a pathological driver delivers a
block larger than the configured period, the callback counts it, passes dry
audio through, and refuses to overrun the engines' buffers
(`examples/nam/live.zig:284-290`). Bypass is snapshotted once per block "so a
mid-block toggle can't split this block across states"
(`examples/nam/live.zig:320-322`).

**Statefulness leaks into UX — correctly.** Read the `mute` doc comment above
again: muting *keeps processing* and only zeroes the device buffer, "so
unmute is click-free". A muted WaveNet whose history froze would resume with
a 4,093-sample state discontinuity — a thump. Once you think in streaming
state, even the mute button is a model-warmth decision.

And one line of pure domain insight, from the input-trim loop
(`examples/nam/live.zig:292-294`): "Input trim BEFORE the model: an amp
model is nonlinear, so this is the 'how hard you hit the amp' knob, not just
volume." The `,`/`.` keys do not make it louder; they change how the model
distorts — because that is what driving a real amp harder does.

## 10.11 The end-to-end latency budget

Per-block compute is only one term of what a guitarist feels. The full path
is: sound into the interface's ADC → driver buffering → the duplex ring →
your 64-frame period → DAC buffering → speakers. `live` prints an honest
estimate at startup, computed from what CoreAudio actually reports —
"input device + duplex+period + output device"
(`examples/nam/live.zig:989-1005`; `examples/nam/README.md`, Latency).

Work it in real numbers, all from `examples/nam/README.md:340-355`. The
middle term is ≈ 3·period, because miniaudio's duplex ring keeps about two
capture periods of slack ahead of the one being played: at the default
`--period 64` that is 192 samples ÷ 48 kHz = **4 ms**. A proper USB interface
adds ~1–3 ms per side, so a good rig lands at **≈ 7–10 ms total — the same
league as hardware modelers**; interfaces that accept `--period 16` reach
~8 ms; built-in mic and speakers ≈ 11 ms; Bluetooth is 100+ ms and hopeless.
The feel reference: ≤ 10 ms reads as immediate — "standing 3 m from your amp
is ~9 ms of air" — ~15–20 ms feels laggy, 30+ is unplayable.

Two engineering notes from the same section repay attention because both are
*measured decisions*, the habit this course keeps pointing at:

- **Clock drift is real and additive.** If a device is not natively at
  48 kHz, macOS silently resamples, and the resampler's rate error
  *accumulates* in the duplex ring — "latency grows second by second while
  you play (measured: ~1100 samples/s)" — so the player retunes the device to
  48 kHz at the OS level for the session instead
  (`examples/nam/README.md:356-360`). Related: monitor through the same
  interface you capture with — two devices means two crystals, and
  independent clocks drift into periodic clicks
  (`examples/nam/README.md:123-125`).
- **A rejected optimization, with its price tag.** Replacing miniaudio's
  duplex machinery with separate devices and a custom one-period ring would
  be "worth ~1.3 ms at 64-frame periods but only ~0.3–0.5 ms once a good
  interface runs at `--period 16`; not pursued at that price
  (underrun-crackle risk for sub-ms gain)"
  (`examples/nam/README.md:369-376`). Declining an optimization, in writing,
  with numbers — the recorded-negatives ethic of
  [Chapter 6](06-going-fast-on-cpus.md) applied to latency.

There is even a ground-truth tool: `loopback-test` sends impulses out your
interface and times their return through a physical patch cable — measuring,
rather than estimating, your rig's true round trip
(`examples/nam/README.md:366-368`).

## 10.12 The extras: cabs, chains, and a tuner

The rest of the example is where the port stops being a benchmark and becomes
an instrument. Each extra also happens to teach something.

**Cabinet IRs — the linear/nonlinear boundary made audible.** A `.nam`
capture of an amp *head* has no speaker in it, and a guitar speaker is half
the sound. The fix is a **cabinet impulse response**: a short mono WAV that
characterizes a speaker cab as a linear FIR filter, convolved after the
model (`--ir cab.wav` appends it: amp → cab → output). The implementation is
direct time-domain convolution with a carried tail — the streaming-history
idea of §10.4 in its simplest form:

```zig
pub fn process(self: *IrCab, input: []const f32, output: []f32, frames: usize) void {
    std.debug.assert(frames <= self.max_frames);
    const l = self.taps;
    const carry = l - 1;

    // Contiguous window: previous tail, then this block.
    @memcpy(self.work[0..carry], self.history[0..carry]);
    @memcpy(self.work[carry .. carry + frames], input[0..frames]);

    // y[i] = dot(weight, work[i .. i+l]); weight[l-1] hits work[i+l-1]
    // (the i-th new sample).
    for (0..frames) |i| {
        output[i] = dot(self.weight, self.work[i .. i + l]);
    }

    // Carry the trailing `carry` samples for the next block.
    @memcpy(self.history[0..carry], self.work[frames .. frames + carry]);
}
```

*(from `examples/nam/ir_cab.zig:166-183`)*

Up to 8,192 taps, upstream's fixed −18 dB headroom gain baked into the
weights, no FFT — "FFT/partitioned convolution is unnecessary at cab-IR
lengths and is deliberately not implemented"
(`examples/nam/ir_cab.zig:15-16`). And here the nonlinearity lesson of §10.2
closes: an IR **is resampled** to the session rate at load when it differs
(cubic, like upstream) because resampling a linear filter's impulse response
is well-defined — while a `.nam` model "still hard-rejects rate mismatch (it
cannot be resampled)" (`examples/nam/ir_cab.zig:22-25`). Linear systems are
characterized by their impulse response; nonlinear systems are not
characterized by anything short of themselves.

**Chains — a rig as a text file.** A `.chain` manifest lists stages, one per
line, top to bottom = signal flow (`examples/nam/chain.zig:1-13`; example
from `examples/nam/README.md`):

```
# pedal -> amp -> cab
name: My Rig                       # optional; shown in the status line
boost.nam :: trim=+3               # a drive capture, hit +3 dB harder
amp.nam                            # no trim = unity
cab.wav :: trim=-2                 # cabinet IR, pulled back 2 dB
```

The per-stage `trim` exists *because* the stages are nonlinear: "the level
*into* a stage shapes its breakup, not just its volume"
(`examples/nam/README.md:264-265`). Inside the callback, stages route
through the two preallocated ping-pong buffers, and a `.nam` stage never
runs in-place — "WaveNet is not in-place safe (reads `input` across the
pass, pushes history after writing `output`)" — a precondition pinned with a
four-distinct-buffers assert (`examples/nam/live.zig:366-381`). Profiles
even carry a `gear_type` tag the player uses for polite advice at load:
a cab IR after an `amp_cab` capture gets a "redundant cab" note, a chain
ending in a cab-less `amp` gets "cab likely needed"
(`examples/nam/live.zig:420-427`; `examples/nam/README.md`, Cab advice).

**The tuner — the right tool per regime, in one program.** Press `t` and a
strobe-class chromatic tuner runs off the raw input — "measured well under
0.1 cent on stable tones" (`examples/nam/README.md:67-70`). Its plumbing is a
wait-free single-producer/single-consumer ring the callback feeds with one
`@memcpy` pair and a release store (`Tap.push`,
`examples/nam/tuner.zig:99-113`); when the tuner is off, the tap costs the
audio thread a single atomic load, and the analysis thread parks on a futex.
The analysis itself — McLeod pitch method, per-partial spectral refinement,
inharmonicity fitting — runs in plain scalar `f64`, deliberately *not* on the
Tensor facade: "the working sets are 63-tap FIR dots and 2-4k-sample
correlations at a ~15 Hz cadence — far below the shapes where the
pool-parallel Tensor pipeline amortizes its dispatch/allocation — and the
sub-0.1-cent accuracy target needs f64 accumulation"
(`examples/nam/tuner.zig:7-12`). One program, three regimes, three correct
tools: autograd tensors for training, hand-rolled f32 SIMD for the realtime
chain, scalar f64 for precision analysis. Knowing your library's amortization
regime — and stepping outside it without guilt — is a mark of owning the
whole stack. (MIDI control of every knob exists too, macOS-only for now;
`examples/nam/README.md:224-225`.)

## 10.13 When latency is the product

Step back and count what is running when you play. A 13,802-parameter WaveNet
— possibly several, in a chain, plus an 8,192-tap FIR — recomputed for every
64-sample block, 750 blocks per second, each block inside a 1,333 µs deadline
with an order of magnitude of headroom, on one CPU core of an ordinary
laptop, matching the reference implementation to within ~2e-6 and
byte-identical to itself at any block size. The same binary trained the model in the first place, checked its
own export against the streaming engine every epoch, and will hand you a file
that the upstream ecosystem loads without a murmur.

Now notice what is *absent*. No Python process feeding an inference server.
No GPU, no driver stack, no CUDA version matrix. No garbage collector to
outsmart. No FFI tax on a 49-microsecond hot loop. No second framework for
deployment because the training framework was too heavy to ship. The tensor
library, the autograd engine, the optimizer, the SIMD kernels, the audio
callback, and the terminal UI are one language, one compilation, one binary —
and you have now read every layer of it in this course.

This is the argument the chapter set out to make, and it is Zig's founding
argument replayed with a neural network in the lead role. When latency is the
product, "close to the metal" is not an aesthetic — it is the list of things
between your code and the deadline, kept short enough to enumerate. A
language conceived because audio software demanded that shortness
([§10.1](#101-full-circle-a-language-born-from-audio)) here runs training
*and* inference of the same model within it, on the machine it will perform
on. Local is not a compromise; for this workload, local — with the whole
stack legible from the callback down to the FMA — is the entire feature.

The next chapters turn to models six orders of magnitude larger, where the
walls move (memory bandwidth, not deadlines) and the tools change
(quantization, KV caches). But the method — measure, verify against a
reference, write the numbers down with their dates — travels unchanged.

## What you now know

- A `.nam` amp profile is one JSON document — version, architecture, config,
  a flat `weights` float array, metadata — and the classic profile is 13,802
  floats: a genuinely useful neural network that fits in L1 cache and loads
  via a cursor walking the flat array (with `head_scale` taken from the
  stream's final float, overriding the JSON copy).
- Dilated causal convolution buys exponential context at linear cost: the
  classic WaveNet's two dilation stacks (1..512, k=3) give a receptive field
  of 4,093 samples ≈ 85 ms at 48 kHz — arithmetic you verified with a test.
- An offline convolution becomes a streaming one by privately keeping the
  last `dilation·(taps−1)` input rows; with a fixed per-element accumulation
  order, output is *bit-identical* regardless of chunking, and the
  compute-then-commit `process`/`push` split makes that testable.
- Stateful streaming models need prewarming — and parity with a reference
  requires reproducing even the reference's prewarm rounding.
- Training minimizes MSE (optionally + MRSTFT at weight 0.0005 in the packed
  recipe); ESR is the *validation* metric, normalized by target energy, with
  upstream's quality bands. Validation runs the streaming engine on each
  epoch's exported weights, so weight-order bugs cannot hide. Only WaveNet is
  trainable; LSTM/ConvNet/Linear are load-and-play.
- Dataset discipline is calibration: blip-based latency measurement, clipping
  refusal, exact pre-silence, and replicate self-ESR checks — before any
  gradient steps.
- The realtime rule — no allocation, no locks, no syscalls in the audio
  callback — is enforced structurally: every buffer preallocated and sized
  before the stream starts, all control state in atomics (floats as
  `@bitCast` u32 bits, profile switching as one index store, a `fetchMax`
  peak meter), engines allocation-free after init by contract.
- The budget arithmetic: 64 frames at 48 kHz = ~1333 µs per block, against a
  measured ~49 µs (x86 P-core) / ~80 µs (M1 Max) per block — dated,
  machine-specific snapshots from the example's README — and ReleaseFast is
  mandatory (~20× vs Debug). End-to-end feel is dominated by devices:
  ≈ 3·period + 1–3 ms per side ≈ 7–10 ms on a good interface.
- Interchange is verified in both directions (upstream loads the exports;
  renders match to ~1e-7), GGUF is a byte-lossless envelope, and quantizing a
  13.8k-param model is refused because it buys nothing — knowing a
  technique's non-applicability is part of the technique.

## Explore the source

- `examples/nam/stream_conv.zig` — the streaming conv: history read, `push`,
  the `time_tile` SIMD kernel, and the bit-exactness invariant; read the
  module doc first.
- `examples/nam/wavenet.zig` — the full streaming WaveNet: the module-doc
  algebra, the weight cursor with its `errdefer` ladders, FiLM and gating.
- `examples/nam/engine.zig` — the tagged-union facade, the prewarm contract,
  and checked errors where upstream asserts.
- `examples/nam/activations.zig` — the exact-contract SIMD tanh; the contract
  comment is the lesson, the bit-craft below it is the bonus.
- `examples/nam/train.zig` — the autograd twin of the engine; compare its
  layer loop with `wavenet.zig`'s module doc line by line.
- `examples/nam/data.zig` — ESR, the v3 capture structure, latency blips, and
  the data-hygiene checks.
- `examples/nam/live.zig` — `audioCallback`, `Shared`, the noise gate, the
  ping-pong chain routing, and the honest latency print.
- `examples/nam/tuner.zig` — the wait-free tap and a deliberate walk *off*
  the Tensor facade, with the reasoning written down.
- `examples/nam/README.md` — the parity, performance, and latency numbers
  quoted in this chapter, each with its protocol and date.

## Exercises

1. **(Easy)** Download a `.nam` profile (Tone3000 has thousands) and run
   `zig build nam -Doptimize=ReleaseFast -- bench profile.nam` at block sizes
   16, 64, and 512 (`--blocksize`). Record your per-block cost and headroom
   at each size, then rebuild without `-Doptimize=ReleaseFast` and bench
   again. Compare your Debug/ReleaseFast ratio with the README's "~20×" and
   your headroom with the budget formula from §10.9.
2. **(Medium)** Extend the course `MiniStreamConv` with a
   `comptime accumulate: bool` parameter à la the real kernel
   (`examples/nam/stream_conv.zig:160`), so two instances can share one
   output buffer the way the engine's head accumulator sums layer
   contributions. Extend
   the chunk-invariance test to cover both modes, and check both compile from
   the one body.
3. **(Medium)** Implement `esr` and `esrComment` from §10.6 yourself, then
   probe the metric's character: for a fixed target, plot (or tabulate) ESR
   as you add white noise of increasing amplitude to a copy of it. At what
   noise level does a signal cross from "Great!" to "Not bad!"? Why does the
   denominator make these bands comparable across quiet and loud captures?
4. **(Hard)** Write a minimal cab-IR stage from scratch following §10.12's
   excerpt: load a mono WAV of taps, direct FIR with a carried tail, and a
   chunk-invariance test. Then make yours in-place safe (`input == output`),
   explain *why* that is achievable for the FIR but not for the WaveNet
   engine (see `examples/nam/live.zig:366-368`), and verify with a test that
   aliasing the buffers changes nothing.
5. **(Hard)** The full loop, on your own rig: run the README's test plan
   step 5 — a `profile` capture against a loopback (patch the interface's
   line out to its line in, so signal ≈ reamp and the trained ESR should be
   near zero), then `train`, `validate --write-wavs`, and play the result
   with `live`. If you have real gear, profile it; report your validation ESR
   against the §10.6 bands, and A/B the written WAVs by ear.

---

[Previous: Training without gradients](09-training-without-gradients.md) · [Next: Model files and quantization](11-model-files-and-quantization.md)
