# Chapter 14 — The low-bit frontier

*Part V — Language models*

[Chapter 11](11-model-files-and-quantization.md) established the economics of quantization: on a CPU, decoding a language model is a memory-bandwidth problem, and every bit shaved off a weight is bandwidth you get back. It took weights from 32 bits down to ~4.5 (Q4_K) and showed the machinery — blocks, scales, dynamic activation quantization, int8 dot instructions — that keeps the arithmetic honest along the way.

This chapter rides that machinery to its edge. What happens when a weight can take only **three values: −1, 0, +1**? Two things, and they are different in kind, not just in degree. First, the storage cost drops to almost two bits per weight. Second — and this is the deeper one — *multiplication disappears from the inner loop*. Multiplying by −1, 0, or +1 is negation, skipping, or copying. A ternary matmul kernel contains no weight multiplications at all.

Everything here is honestly labelled **research frontier**. The formats are real, first-class, and tested to the same bitwise standards as the rest of the library — but the accuracy story is genuinely unsettled territory, and the repo's own documents record collapses, dead ends, and trade-offs that flip sign between CPU families. This chapter presents all of it, numbers attached, because reading a frontier honestly is a skill worth teaching. Three questions structure it:

1. **Inference**: how do you store and multiply ternary weights fast? (TQ2_0, §14.1–14.3)
2. **Training**: how do you *learn* weights a gradient cannot see? (STE and ES, §14.4–14.5)
3. **Conversion**: can you turn an *existing* model ternary after the fact? (PTQTP, §14.6–14.8)

## 14.1 Why 1.58 bits, and why TQ2_0

A three-valued weight carries log₂ 3 ≈ 1.58 bits of information — hence the name of the research line this builds on, BitNet **b1.58** (arXiv:2504.12285, cited in `docs/TERNARY.md`). No practical format hits 1.58 exactly; you pay a little for addressable packing and for scales. Fucina's chosen container is ggml's **TQ2_0**, described at the top of `docs/TERNARY.md:11-14`: 256-element blocks, 64 bytes of 2-bit "crumbs", an inline f16 scale — **2.0625 bits per weight**.

The block struct is exactly the wire format, like every quantized dtype in the library (Chapter 11's `extern struct` discipline). From `src/dtype.zig:206-209`, with its size pinned at compile time (`src/dtype.zig:245`; field comments added here, and the source's comptime block pins all 26 block sizes — only the TQ2_0 assert is shown):

```zig
pub const BlockTQ2_0 = extern struct {
    qs: [qk_k_block_size / 4]u8,  // 64 bytes: 256 x 2-bit crumbs
    d: u16,                       // f16 scale
};

comptime {
    std.debug.assert(@sizeOf(BlockTQ2_0) == 66);
}
```

66 bytes for 256 weights. The 2-bit crumbs do not store the trit `w` directly — they store the *code* `w + 1 ∈ {0, 1, 2}`, so all codes are unsigned. Hold that thought; it is the entire kernel trick of §14.2.

Why adopt ggml's format rather than invent one? `docs/TERNARY.md:16-27` gives the design record, and it is a good lesson in *not* designing a format:

- **GGUF interop is free**: llama.cpp reads and writes the same blocks; `export-gguf --dtype tq2_0` emits `general.file_type = 37` and the result is a normal interchange file.
- **The layout is already SIMD-shaped**: within each 32-byte group, crumb lane `L` covers 32 consecutive activations, so unpacking is shift+mask only.
- **The alternatives buy little**: bitnet.cpp's `I2_S` differs "only in bit order and scale placement" and measures within ~6% of TQ2_0-class kernels on dot-capable CPUs (the doc quotes arXiv:2502.11880: i7-13700H, 3.8B model — I2_S 35.04 t/s vs TQ2_0 33.19).
- **The genuinely different alternatives were measured and declined**: bitnet.cpp's TL1/TL2 lookup-table kernels win mainly on CPUs *without* int8 dot instructions and on footprint (TL2 is 1.67 bpw), but are GEMV-only and need offline per-shape codegen — "deliberately **not** ported; recorded as future work" (`docs/TERNARY.md:24-27`).

The even-smaller sibling `tq1_0` (1.6875 bpw, five trits packed base-3⁵ per byte) exists as a dtype but stays decode/cold-matmul-only (`docs/REFERENCE.md` §10.7). The frontier is TQ2_0.

> **ML note** — Why would three values be enough to represent a language model at all? The empirical claim of the BitNet line is that *at sufficient scale, trained-from-scratch ternary models track full-precision quality*, because what matters is the direction each weight pushes (and whether it participates), not its fifth significant digit. That claim is about models *trained ternary from the start*. Converting an existing full-precision model to ternary after the fact is a much harder problem — §14.6 is about exactly how much harder, with a measured collapse to show for it.

## 14.2 The kernel with no multiplications

Here is the identity everything rests on. TQ2_0 stores codes `c = w + 1`. So for a weight row `w` and activation vector `a`:

```
dot(w, a) = Σ wᵢ·aᵢ = Σ (cᵢ − 1)·aᵢ = Σ cᵢ·aᵢ − Σ aᵢ
```

The first term is a dot of *unsigned two-bit codes* against int8 activations — exactly the shape CPU int8 dot instructions want. The second term is the plain sum of the activation block. And Chapter 11 already built the punchline: Q8_K activation blocks (the dynamic quantization used for all 256-element weight formats) carry `bsums`, per-16 partial sums, *precisely so that formats can fold correction terms into the dot*. `Σa` is one vector fold of data that already exists.

The identity is small enough to hold in your head, so here it is as runnable course code (not repo code; compile-checked with `zig test`):

```zig
test "the mul-free identity: dot(w, a) = sum((w+1)*a) - sum(a)" {
    // w in {-1, 0, +1}; the codes c = w + 1 in {0, 1, 2} are what TQ2_0
    // actually stores in its 2-bit crumbs.
    const w = [_]i8{ -1, 0, 1, 1, -1, 0, 1, -1 };
    const a = [_]i32{ 3, -2, 5, 7, 1, 4, -6, 2 };

    var direct: i32 = 0;
    var code_dot: i32 = 0; // what sdot/vpdpbusd compute: codes x activations
    var asum: i32 = 0; // what the Q8_K activation bsums already carry
    for (w, a) |wi, ai| {
        direct += @as(i32, wi) * ai;
        code_dot += (@as(i32, wi) + 1) * ai;
        asum += ai;
    }
    try std.testing.expectEqual(direct, code_dot - asum);
}
```

The real kernel lives in `src/backend/quant/ternary.zig`, and its module doc (`src/backend/quant/ternary.zig:5-12`) is the best summary of what each ISA arm does with the code dot:

- **aarch64**: `sdot` on 16-byte granules — codes {0,1,2} are valid *signed* bytes, so the signed dot instruction applies directly.
- **x86 AVX-VNNI / AVX512-VNNI**: `vpdpbusd` on 32-byte granules (codes as u8).
- **x86 AVX2** (no VNNI): `vpmaddubsw` + `vpmaddwd(+1)` — with a proof in the comment that the intermediate i16 stage *cannot* saturate: a maddubs pair sum is at most 2·127·2 = 508. The doc pointedly contrasts this with bitnet.cpp's 4096-element i16 cadence, "which is only statistically safe" (`docs/TERNARY.md:43-45`).
- **everywhere else**: portable `@Vector` twins of those primitives — the same one-source-many-ISAs pattern as [Chapter 6](06-going-fast-on-cpus.md).

Because every arm accumulates the exact per-block integer (max |Σ(w+1)·a| = 65024, far inside i32 — `docs/TERNARY.md:48-49`), **all arms are cross-ISA bitwise identical to the cold scalar reference**, pinned by `src/backend/quant/ternary_tests.zig` and hardware-executed on both ISAs via `zig build x86dot-check` (`docs/TERNARY.md:49-52`). The verification religion of the rest of the library applies unchanged at 2 bits.

The inner loop of the tile kernel shows the identity in production form — four weight rows share every activation load and one precomputed block sum (`src/backend/quant/ternary.zig:339-350`, inside `matmulTQ2_0RhsTile`):

```zig
for (arow, 0..) |*a, bi| {
    const bsum = if (cached) bsum_cache[bi] else blockBsumTotal(a);
    const w: [ternary_col_block]*const BlockTQ2_0 = .{
        &wcols[0][bi], &wcols[1][bi], &wcols[2][bi], &wcols[3][bi],
    };
    const dots = blockCodeDotW(ternary_col_block, w, a);
    inline for (0..ternary_col_block) |ci| {
        const isum = dots[ci] - bsum;
        sums[ci] += f16BitsToF32(w[ci].d) * a.d * @as(f32, @floatFromInt(isum));
    }
}
```

One subtraction per block turns the unsigned code dot into the signed ternary dot; the only multiplications left are the two per-block scale fix-ups (`w.d * a.d`) — floats that touch one value per 256 weights. There is a documented trap right above this loop: the kernel *trusts* the activation `bsums` to equal the exact per-16 sums of the quantized values — "stale/foreign bsums silently corrupt results" (`src/backend/quant/ternary.zig:309-313`). An identity that folds precomputed sums into a dot is only as correct as the precomputation.

And the encoder side — packing four trits per byte — is a compact Zig bit-manipulation lesson (`src/backend/quant/ternary.zig:110-133`, abridged comment):

```zig
fn encodeCrumbs(block: *BlockTQ2_0, x: *const [qk_k_block_size]f32, id: f32) void {
    for (0..2) |group| {
        const base = group * 128;
        for (0..32) |m| {
            var q: u8 = 0;
            inline for (0..4) |n| {
                // Clamp in the FLOAT domain before the int conversion:
                // @intFromFloat of NaN/inf is safety-checked illegal behavior...
                const rounded = roundHalfAwayFromZero(x[base + n * 32 + m] * id);
                const bounded: f32 = if (std.math.isNan(rounded)) 0.0 else @min(1.0, @max(-1.0, rounded));
                const xi: i32 = @intFromFloat(bounded);
                q += @as(u8, @intCast((xi + 1) & 3)) << (2 * n);
            }
            block.qs[group * 32 + m] = q;
        }
    }
}
```

> **Zig note** — Two things to notice. `inline for (0..4)` unrolls the crumb loop at compile time, so the shifts `<< (2 * n)` become constants — this is the standard Zig idiom for bit-packing loops. And the NaN clamp is not paranoia: `@intFromFloat` of NaN or infinity is *safety-checked illegal behavior* in Zig — a crash in Debug, undefined in ReleaseFast — and divergent latent weights genuinely reach this encoder during STE training (§14.4) with no upstream guard. The comment documents the policy: NaN maps to code 1, i.e. the zero weight. Defensive numerics belong at the seam where the type changes.

Two contract points before the numbers, both easy to trip on:

- **The contract dimension `k` must be a multiple of 256** everywhere — block granularity (`docs/TERNARY.md:174-175`).
- **TQ2_0 has no packed RHS layout.** The hot formats of Chapter 11 (q4_k, q5_k, q6_k, q8_0) get column-interleaved `packRhs` copies; TQ2_0's kernels work from the plain blocks, and calling `packRhs` on a `.tq2_0` tensor is a *compile error* — `PackedRhsLayout` simply has no ternary member (`docs/REFERENCE.md:7347-7351`). The layout is already SIMD-shaped as stored; there is nothing to repack.

## 14.3 Measured: what 2.06 bits buys

The numbers, exactly as recorded in `docs/TERNARY.md:60-96` — dated, single-thread, machine-named snapshots (measured, not asserted; your hardware will differ):

**M1 Max, single thread, ReleaseFast, `zig build bench-ternary`, 2026-07-07:**

| shape | m | cold µs | hot µs | hot/cold | Q4_K µs | dense f32 µs |
|---|---|---|---|---|---|---|
| n=4096 k=4096 | 1 | 1013 | **238** | 4.25x | 525 | 16725 |
| n=4096 k=4096 | 128 | 130767 | **31260** | 4.18x | 69834 | 5983* |
| n=11008 k=4096 | 1 | 2740 | **667** | 4.11x | 1391 | 31527 |
| n=11008 k=4096 | 128 | 352930 | **86815** | 4.07x | 182780 | 8086* |

(*The doc's own footnote: dense f32 goes through Accelerate — multi-core AMX — at those shapes, while the quant kernels are pinned single-thread by design. Never read a benchmark table without its footnotes.)

**i9-13950HX Raptor Lake, Linux, single thread, ReleaseFast, 2026-07-07** (`docs/TERNARY.md:79-96`):

| shape | m | cold µs | hot µs | hot/cold | Q4_K µs |
|---|---|---|---|---|---|
| n=4096 k=4096 | 1 | 985 | **193** | 5.10x | 924 |
| n=11008 k=4096 | 1 | 2701 | **534** | 5.06x | 2664 |

Three readings worth extracting:

1. **The ARM speedup over Q4_K is the byte ratio.** ~2.1× the tuned Q4_K kernel at equal shapes is almost exactly 2.06-vs-4.5 bits per weight (`docs/TERNARY.md:72-73`) — at GEMV the kernel is bandwidth/ALU-fed, and halving the bytes halves the time. The kernel sits at the NEON ALU limit: ~250 µs theoretical for the 4096² GEMV vs 238 µs measured (`docs/TERNARY.md:74-76`); `docs/PTQTP.md:284-286` states it as ~86% of the NEON roofline with hand-written assembly holding ≤14% headroom. When your kernel is at the roofline, further cleverness is spent elsewhere.
2. **x86-VNNI beats the byte ratio.** ~4.8× Q4_K on Raptor Lake, because `vpdpbusd` consumes 32 bytes per instruction vs `sdot`'s 16 — instruction-set density, not magic (`docs/TERNARY.md:89-90`, `docs/PTQTP.md:286-288`). Remember this asymmetry: it flips an economic conclusion in §14.7.
3. **The two x86 arms (AVX2-maddubs and AVX-VNNI) produce bit-equal checksums** (`b1f84dde82d0c0a4`, `docs/TERNARY.md:92-94`) — the exact-integer-accumulation claim of §14.2, hardware-verified.

Using a ternary tensor requires nothing new at the facade — `.tq2_0` weights flow through the same `dot` dispatch as every quantized RHS in Chapter 11. This is the machine-verified snippet from `docs/REFERENCE.md` §10.7 (run against the real modules by `zig build snippet-check`):

```zig
test "TQ2_0 ternary weights are a first-class matmul RHS" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const k = 256;
    var wsrc: [2 * k]f32 = undefined;
    for (wsrc[0..k]) |*v| v.* = 0.5; // encodes exactly: d = 0.5, trit = +1
    for (wsrc[k..]) |*v| v.* = -0.5;
    var blocks: [2]fucina.BlockTQ2_0 = undefined;
    try fucina.gguf.encodeF32(.tq2_0, &wsrc, std.mem.sliceAsBytes(&blocks));

    const W = fucina.Tensor(.{ .dtype = .tq2_0, .tags = .{ .out, .in } });
    var w = try W.fromBlocks(&ctx, .{ 2, k }, &blocks);
    defer w.deinit();

    const x_values = [_]f32{1} ** k;
    var x = try fucina.Tensor(.{ .batch, .in }).fromSlice(&ctx, .{ 1, k }, &x_values);
    defer x.deinit();

    var y = try x.dot(&ctx, &w, .in); // mul-free int8 ternary kernel
    defer y.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 128), (try y.dataConst())[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, -128), (try y.dataConst())[1], 1e-3);
}
```

Everything here is Chapter 11 machinery pointed at a new dtype: `gguf.encodeF32` grows a `.tq2_0` arm, `fromBlocks` takes logical shapes over wire blocks, and `dot` comptime-dispatches to the ternary kernel because the RHS dtype says so. Like every quantized tensor, a `.tq2_0` tensor is a constant — it exposes only quantized operations, never receives gradients, and float ops on it are absent at comptime (`docs/REFERENCE.md` §10.2). Which raises the question the rest of this chapter answers: if the served weights cannot take a gradient, how does anything ternary ever get *trained*?

## 14.4 Training ternary, take one: the straight-through estimator

Now the hard question. Inference over ternary weights is a solved kernel problem; *learning* them is not, because the quantizer

```
Q(w) = clamp(round(w / d), −1, +1) · d
```

is a step function. Its derivative is zero almost everywhere and undefined at the jumps. Backpropagate through it honestly and every weight upstream of the quantizer receives gradient zero: the model cannot learn. Course code, compile-checked:

```zig
test "the true gradient of a quantizer is zero almost everywhere" {
    const d: f32 = 0.5;
    // Finite differences see a flat function: nudging w does not move q(w).
    try std.testing.expectEqual(quantizeTrit(0.2, d), quantizeTrit(0.2001, d));
}
```

The fix that the entire quantization-aware-training literature runs on is the **straight-through estimator** (STE): in the forward pass, use the quantized weight; in the backward pass, *pretend the quantizer is the identity* and pass the gradient straight through to the latent float weight. It is mathematically indefensible — you are using the gradient of a function you did not evaluate — and it works. Here is the whole idea on one scalar (course code, compile-checked; `quantizeTrit` as above):

```zig
test "straight-through estimator: the 'wrong' gradient still trains the weight" {
    // Task: y = q(w) * x should hit target = -0.5 (needs the -1 trit).
    // The true dL/dw is zero on the plateau, so exact gradient descent is
    // stuck at w = 0.2 forever. The STE pretends the quantizer is the
    // identity: backward uses the gradient of y = w * x instead.
    const d: f32 = 0.5;
    const lr: f32 = 0.1;
    const x: f32 = 1.0;
    const target: f32 = -0.5;

    var w: f32 = 0.2; // latent float weight, quantizes to trit 0 at start
    for (0..100) |_| {
        const y = quantizeTrit(w, d) * x; // forward: QUANTIZED weight
        const dy = 2.0 * (y - target); // dL/dy for L = (y - target)^2
        const dw = dy * x; // STE backward: as if y were w * x
        w -= lr * dw;
    }
    // The latent weight walked across the plateau and the trit flipped.
    try std.testing.expectEqual(@as(f32, -0.5), quantizeTrit(w, d));
}
```

The latent float `w` never appears in the forward output — only its quantization does — yet it accumulates gradient pressure across steps until it crosses a threshold and the trit flips. The latent weight acts as a *vote counter*: many small "you should be more negative" signals eventually change the discrete decision.

> **ML note** — Why does an estimator this wrong work? Three partial answers, none fully satisfying — which is the honest state of the theory. (1) *The bias is bounded where it matters*: for weights well inside a quantization cell, small moves genuinely don't change the loss, so "zero gradient" is locally true and the STE's fiction only matters near boundaries — exactly where you want pressure to accumulate. (2) *The latent weights integrate noise*: a single STE gradient is a bad estimate, but summed over many batches the systematic component (which side of the boundary should I be on?) survives while the fiction washes out. (3) *It is the identity chosen by the people who scaled it*: Fucina implements "exactly the BitNet recipe (`w + (Q(w) − w).detach()`)" — no clipping, no masking (`docs/TERNARY.md:130-134`) — because when you port a method whose success is empirical, you port its exact form, not your improvement of it.

Fucina packages this as one facade op on the ordinary f32 tensor — `dotTernarySte` (`src/ag/tensor.zig`; documented in `docs/REFERENCE.md` §10.7):

```zig
pub fn dotTernarySte(self: *const Self, ctx: *ExecContext, weight: anytype,
                     comptime contract_tag: Tag) !Tensor(...)
```

Its anatomy, per `docs/TERNARY.md:128-137` and `docs/REFERENCE.md:7852-7868`:

- **Forward**: encode the *latent f32 weight* (tags `{ .out, .in }`) to TQ2_0 with the b1.58 recipe — per-tensor scale `d = max(mean|W|, 1e-5)` (`ternaryAbsmeanScale`), round-clip to {−1, 0, +1} (`quantizeRowTQ2_0ScaledInto`, every block storing the same `d` so the result is plain valid TQ2_0) — then contract with the **mul-free f32 kernel**.
- **Backward**: `dx = gy · dequant(W_q)` — through the *quantized* weight, matching what the forward computed, so `dx` is an *exact* gradient (the forward is linear in `x` given the frozen trits; it is pinned by gradcheck). `dW = gyᵀ · x` — the straight-through estimate, the plain matmul VJP against the latent weight, and the only "wrong" part.
- **Lifecycle**: the encoded blocks live in the backward node and are freed with it; the op works under the exec scopes of [Chapter 7](07-autograd.md). Runtime error `TernaryContractDimNotBlockAligned` if `k` is 0 or not a multiple of 256. The latent weight is re-encoded *every forward* — inherent to STE, recorded as a cost with a future-work note for frozen weights (`docs/TERNARY.md:179-180`).

An explicit machine-verified test pins the STE identity — with `gy = 1`, each row of `dW` must be literally `x`, exactly what the un-quantized matmul's VJP would produce (`docs/REFERENCE.md:7871-7899`):

```zig
test "dotTernarySte trains a b1.58 ternary linear" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const k = 256; // contract dim must be a multiple of 256 (TQ2_0 blocks)
    var xv: [k]f32 = undefined;
    for (&xv, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 5)) * 0.1 - 0.2;
    var wv: [2 * k]f32 = undefined;
    for (&wv, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 9)) * 0.1 - 0.4;

    var x = try fucina.Tensor(.{ .batch, .in }).variableFromSlice(&ctx, .{ 1, k }, &xv);
    defer x.deinit();
    var w = try fucina.Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 2, k }, &wv);
    defer w.deinit();

    var y = try x.dotTernarySte(&ctx, &w, .in); // encode-then-mul-free forward
    defer y.deinit();
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    // STE identity: dW is the plain matmul VJP; with gy = 1, each row of dW = x.
    var gw = (try w.grad(&ctx)).?;
    defer gw.deinit();
    const gw_data = try gw.dataConst();
    for (0..k) |i| try std.testing.expectApproxEqAbs(xv[i], gw_data[i], 1e-6);
}
```

Note that `w` here is a plain f32 *variable* — the latent weight trains under any optimizer from [Chapter 8](08-training.md); only the forward sees trits.

A subtlety about encoders that will save you a confused afternoon: TQ2_0 has **two** encoders with different scaling policies and different reachability (`docs/REFERENCE.md:7785-7797`). `quantizeRowTQ2_0Into` is the ggml-parity encoder — *per-block absmax* scale — and it is the only one behind the generic seams: `gguf.encodeF32(.tq2_0, ...)` and `quantizeRowForDType` both realize per-block absmax. `quantizeRowTQ2_0ScaledInto` + `ternaryAbsmeanScale` is the *b1.58 recipe* — one per-tensor absmean scale in every block — and it is *not* reachable through the generic seams; it is driven by the `dotTernarySte` forward and the ternary ES paths. Both produce byte-valid TQ2_0 (the scaled encoder writes the same `d` in every block precisely so no side channel is needed), but they quantize the same floats differently. Export a trained ternary model with `--dtype tq2_0` and you get absmax blocks, not the absmean blocks training simulated.

The forward deserves one more look, because it is not the int8 kernel from §14.2. Training forwards keep **exact f32 activations** (no activation quantization noise on top of the weight quantization), yet stay multiplication-free through a second identity — exact in IEEE f32 (`docs/TERNARY.md:98-112`):

```
w·x = (x XOR s) AND m      s = 0x80000000 where w == −1 else 0
                           m = 0xFFFFFFFF where w != 0 else 0
```

Multiplying by −1 is a sign-bit flip; multiplying by 0 is a mask. Course code, compile-checked, to make it concrete:

```zig
test "ternary f32 multiply as two bitwise ops: w*x = (x XOR s) AND m" {
    const xs = [_]f32{ 1.5, -2.25, 0.0, 3.75, -0.5 };
    const ws = [_]f32{ -1, 0, 1, -1, 1 };
    for (xs, ws) |x, w| {
        const bits: u32 = @bitCast(x);
        const s: u32 = if (w == -1) 0x8000_0000 else 0; // flip the sign bit
        const m: u32 = if (w != 0) 0xFFFF_FFFF else 0; // or zero the lot
        const got: f32 = @bitCast((bits ^ s) & m);
        try std.testing.expectEqual(w * x, got);
    }
}
```

`dotTQ2_0F32` fixes a 4-lane accumulation order so this path is bitwise reproducible across every ISA and both backends — scalar and native builds share identical training numerics. It is a correctness-first path: ~15× slower than the int8 flagship at GEMV (3.5 ms vs 238 µs on the 4096² shape, `docs/TERNARY.md:110-112`), but exact. Training tolerates slow; it does not tolerate ambiguous.

## 14.5 Training ternary, take two: evolution strategies

The STE trains a *float shadow* of a ternary model: latent f32 weights exist throughout training, and quantization is re-derived every step. There is a second road, and [Chapter 9](09-training-without-gradients.md) already paved it: **evolution strategies need no gradient at all**, so the quantizer's zero derivative is simply not a problem. Perturb, evaluate, vote.

Fucina's ES trainer (`src/es.zig`) grows a second slot kind for this: **genomes that ARE packed `[]BlockTQ2_0`** (`docs/TERNARY.md:139-145`). No latent floats exist for ternary slots. The block scales `d` are never touched — fixed at init (the doc suggests `1/sqrt(k·2/3)` for uniform random trits) — so *the state you train is byte-for-byte the state you serve*, and every population member is evaluated through the real int8 flagship kernels of §14.2. Training equals inference, literally.

The discrete machinery, adapted from EGGROLL's integer recipe (arXiv:2511.16652, credited at `docs/TERNARY.md:147-148` — machinery adapted, no code ported):

- **Perturb**: sparse trit flips — `max(1, rate·len)` positions get ±1 with clamping at the rails — regenerated from a counter-based stream in a dedicated `es_trits` RNG domain, a pure function of (seed, iteration, member): the same O(1)-memory contract as Chapter 9's Gaussian noise. Antithetic odd members mirror the deltas.
- **Restore**: here discreteness bites. Chapter 9's float trick — regenerate the noise and subtract it — *cannot work*, because clamping is lossy: a flip that hit the rail cannot be inverted by arithmetic. Ternary restore replays a sparse (index, old-crumb) **undo log** in reverse (`docs/TERNARY.md:150-155`).
- **Update**: reward shaping is unchanged (z-score / centered ranks / antithetic fold); shaped fitness feeds *votes* on touched indices, and the top-K by |vote| (ties broken by index — deterministic) each move **one bin** toward the vote's sign, clamped. `K = round(update_fraction · len / (1 + decay · t))`.
- **Config**: `ternary_flip_rate` (default 0.001), `ternary_update_fraction` (0.005), `ternary_update_decay` (0.0) — checkpoint contracts, persisted as `es_ternary_*` in `TrainerState` (`docs/TERNARY.md:160-162`, `src/es.zig:231-234`). Float and ternary slots coexist in one trainer — biases and scales stay Gaussian-ES floats — and a mixed-trainer test pins the float noise streams bitwise untouched.

The acceptance demo is `zig build es-ternary-spirals`: a 2→256→256→2 MLP whose hidden and output layers are packed ternary genomes trained *from random trits* by ES — every member evaluation runs `quantizeRowsQ8_K` + `matmulTQ2_0RhsRange`, i.e. the deployed inference path, and the run self-verifies. Measured (`docs/TERNARY.md:94-96`): 100% accuracy in 9250 iterations / 112.7 s on the Raptor Lake box, vs 14750 iterations / 104.4 s on M1 Max — different iteration counts, comparable wall times; the doc records the numbers without further interpretation.

Step back and note what you now have: **two complete answers to "how do you train what you cannot differentiate"** — a biased gradient that pretends the obstacle away (STE), and a gradient-free method that never asks (ES) — sharing one wire format and one kernel. A third answer, for when the model is already trained, occupies the rest of the chapter. Side by side:

| | needs gradients | needs data | latent floats | what trains | serving path during training |
|---|---|---|---|---|---|
| **STE** (`dotTernarySte`) | yes (through the fiction) | yes | yes — the real state | latent f32 weights, re-encoded every forward | mul-free f32 kernel (exact, ~15× slower) |
| **ES** (ternary genomes) | no | rewards only | none | packed `[]BlockTQ2_0` directly | the deployed int8 kernel itself |
| **PTQTP** (§14.6) | no | no — data-free | n/a (post-hoc) | nothing — solves a decomposition | n/a; output serves like any TQ2_0 |

Three paradigms — quantization-aware training, gradient-free training, post-training quantization — one wire format.

## 14.6 PTQTP: turning a trained model ternary

Suppose the model already exists — a Qwen3 GGUF you did not train and will not retrain. No gradients, no training data, no calibration set. Can its weight matrices be re-expressed over the ternary machinery anyway?

One plane cannot do it. A single TQ2_0 tensor gives each 256-element weight group three representable values times one scale; real weight distributions need more resolution than that, and §14.7's table shows single-plane conversion collapsing even on the two-spirals toy. The idea that works is **PTQTP — Post-Training Quantization to Trit-Planes** (arXiv:2509.16989, Xiao et al.; implemented in `src/ptqtp.zig` from the paper's formulas — the module doc at `src/ptqtp.zig:18-20` notes that no public reference implementation existed at porting time). Decompose each weight matrix as a *sum of K ternary planes*, each with its own per-group scales:

```
W ≈ Σₖ diag(αₖ)·Tₖ        Tₖ ∈ {−1, 0, +1},  K ∈ {1, 2, 3}
```

Two planes give 9 value combinations per group (3²) instead of 3; three planes give 27. Every length-256 column group is an independent least-squares problem, solved by alternating two steps until the scale vector moves less than `epsilon` (`docs/PTQTP.md:12-31`, `src/ptqtp.zig:62-90` for the `Options` defaults):

1. **Scales** — a closed-form K×K ridge regression `α = (SᵀS + λI)⁻¹Sᵀw`. And here is a beautiful consequence of ternary: since trits are {−1, 0, +1}, the Gram matrix `SᵀS` is *pure integer counting* — no floating-point accumulation error in the matrix itself (`src/ptqtp.zig:227-247`: `s_diag`, `s12`, `s13`, `s23` are `i64` tallies). λ escalates ×10 while the condition estimate exceeds `kappa_max` — required at init, where all planes start at `sign(w)` and the unregularized system is singular.
2. **Topology** — for each element, exhaustively try all 3ᴷ trit tuples (at most 27) and keep the argmin of `(w − Σₖ αₖcₖ)²`.

Determinism is engineered, not hoped for: the candidate order is pinned — zero tuple first, sparser before denser, strictly-less keeps-first — so exact ties prefer sparser trits, the symmetric init breaks deterministically, and the whole solve is **bitwise reproducible for any thread count** (`docs/PTQTP.md:26-31`). Rows fan out to the worker team with disjoint outputs; per-row stats reduce in row order.

The port makes three deliberate deltas from the paper, each argued in `docs/PTQTP.md:33-52` — a case study in what "faithful port" means when infrastructure differs:

- **G = 256, packed** (the paper uses G = 128, unpacked): Fucina's group size is the TQ2_0 block width, so *each plane is a byte-valid standalone TQ2_0 tensor* whose per-block f16 `d` **is** the group scale. Consequence: inference is K stock ternary matmuls plus adds — **no new kernels** — and every plane is individually llama.cpp-dequantizable. Measured cost of the coarser groups: ~1.3% relative error.
- **f16 scale rounding at pack time**: |α| is rounded to f16 *first*, then one final topology pass runs against the rounded scales — the stored trits are elementwise-optimal for exactly the scales inference will multiply by (measured: packed error equals the f32-scale reference to 4 decimals).
- **Non-finite weights** are excluded from the regression and forced to trit 0 in every plane — one NaN degrades only itself (the same stance as the TQ2_0 encoder's clamp in §14.2).

K itself is a spectrum: **K = 1** is a least-squares upgrade over the blind absmean b1.58 encoder; **K = 2** is the paper's dual decomposition; **K = 3** is Fucina's capacity extension beyond the paper — an extra residual plane, error bound ~⅓ of dual, at +2.06 bpw where applied. Planes are separable by construction: serving fewer planes than were solved is valid (`docs/PTQTP.md:49-52`).

Where it plugs into the model stack (`docs/PTQTP.md` §Surfaces):

- `src/ptqtp.zig` — `solveGroup` (pure, allocation-free), `quantizeMatrix → PlanePair` (owns up to three `[]BlockTQ2_0` planes plus `MatrixStats`: relative Frobenius error *of the reconstruction inference will actually use*, per-plane zero fractions, convergence counts).
- `src/llm/weights.zig` — `LinearWeight` gains a `ptqtp` arm; `LinearWeight.toPtqtp` dequantizes any loadable source dtype in row chunks through one code path, packs the planes, and drops the original storage. Eligibility is the 256-block contract (in-dim % 256 == 0).
- `src/llm/qwen3/model.zig:391` — `Model.decoratePtqtp(ctx, options)` walks attention q/k/v, o_proj, and dense FFN projections in place; embeddings, lm_head, and norms are not walked, and MoE FFNs are counted skipped. Options include `skip_first_layers`/`skip_last_layers` and per-projection plane overrides `down_planes`/`o_planes` — the sensitive projections get their own knob for a reason §14.7 makes plain.
- **The fused entry** (`linearSeqPtqtpFused`): a decorated linear costs one Q8_K activation quantization and **one** worker-team dispatch — column-partitioned tasks compute every plane and sum in fixed plane order, bitwise identical to a per-plane facade dot chain (pinned by test). The design note is a Chapter 13-grade lesson: at 1.7B decode shapes, *per-plane fork-joins cost more than the ternary kernel itself* — dispatch granularity was the bottleneck, not arithmetic (`docs/PTQTP.md:76-82`).
- **GGUF persistence** (`src/llm/ptqtp_gguf.zig`): a decorated model saves with each eligible tensor **replaced** by `<name>.ptqtp0/1/2` plane tensors plus a `fucina.ptqtp.version` metadata key; everything else passes through byte-verbatim, and save→load→save is byte-stable. Loading *pair-detects* per tensor and rebuilds the arm bitwise.

One boundary must be stated with precision, because it is easy to over-claim. Each plane tensor is a byte-valid standalone TQ2_0 tensor, so llama.cpp can *dequantize any individual plane* — the format never leaves interchange territory. But the decorated **model** is not a llama.cpp-runnable model: reassembling `<name>` from its plane siblings requires the pair-detection logic, which today is wired **in the qwen3 loaders only** — other model families do not read decorated files yet (`docs/PTQTP.md:107-110`). Interop of *tensors*, not of *models*.

## 14.7 The honest scoreboard

Now the part that makes this a frontier chapter rather than a victory lap. All numbers from `docs/PTQTP.md:237-288` — M1 Max, ReleaseFast, NLL teacher-forced over 512 held-out tokens; dated design-record snapshots.

**The toy first** — two-spirals MLP, float-trained to accuracy 1.000, the hidden-to-hidden and output matrices (w2/w3) decorated post-training (`zig build ptqtp-spirals` reproduces it):

| variant | acc (exact-f32 path) | acc (deployed int8 path) | rel err (w2) |
|---|---|---|---|
| float | 1.000 | — | — |
| absmean b1.58, 1 plane | 0.670 | 0.670 | — |
| PTQTP K=1 | 0.644 | 0.644 | 0.474 |
| **PTQTP K=2** | **1.000** | **1.000** | **0.190** |

Single planes collapse post-hoc — both the blind absmean encoder and the least-squares K=1. The dual decomposition recovers full accuracy on the deployed int8 path. So far, so encouraging.

**Now the real models** — perplexity (lower is better):

| model / source | baseline | K=2 | K=2 +down3+o3 | K=3 |
|---|---|---|---|---|
| 0.6B f16 | 24.96 | 80.47 | 51.69 | **27.36** |
| 1.7B Q4_K_M | 19.41 | 330.96 | — | 21.71 |
| 1.7B BF16 | 18.57 | 184.45 | 45.70 | **18.43** |

> **ML note** — Teacher-forced perplexity is `exp` of the average per-token negative log-likelihood on held-out text — roughly, "among how many tokens is the model effectively guessing". It is the standard quantization-damage meter because it moves smoothly where benchmark accuracies jump. A ppl going from 18.57 to 184.45 is not "10× worse prose"; it is a model that has lost the plot entirely.

Read the table like a researcher:

- **K=2 collapses at scale.** ×3.2 ppl at 0.6B, **×17 at 1.7B** — sensitivity to the 9-level resolution *grows sharply with model size* (`docs/PTQTP.md:222-225`). The paper's headline configuration does not survive contact with this model family at this scale. The doc says so plainly, and the dead-end register (`docs/PTQTP.md:292-298`) closes the escape route: carrying the top 1.56% |w| per group *exactly* (sparse outlier carry) improves K=2's rel err only 0.184→0.146 vs K=3's 0.069 — "the K=2 gap is bulk 9-level resolution, not tails". A measured negative, recorded so nobody re-chases it.
- **Sensitivity concentrates.** The `+down3+o3` column — K=3 only on `down_proj`/`o_proj`, the projections that write the residual stream — recovers 184→45.7 at 1.7B on its own. Hence the per-projection plane overrides in `decoratePtqtp`.
- **K=3 from BF16 reaches parity**: 18.43 vs 18.57 baseline — statistical parity, greedy completion flawless (`docs/PTQTP.md:261-263`), weight-space rel err 0.067 on the 27-level bound. And decoration is fast: seconds at 0.6B, ~90 s at 1.7B K=3, data-free.
- **Source precision matters**: from a Q4_K_M source the same K=3 lands at 21.71 — the doc's configuration guidance (`docs/PTQTP.md:216-219`): quantize from bf16/f16 originals when available; the from-quantized path "degrades gracefully (~15–18% worse ppl), never collapses".

**And the speed** (1.7B vs its BF16 original, interleaved runs, fused entry, M1 Max):

| config | t/s | vs baseline | ppl |
|---|---|---|---|
| bf16 baseline | 20.2–21.0 | 1× | 18.57 |
| K=2 | 47.1–48.3 | 2.3× | 184 |
| **K=3** | **39.5–39.8** | **1.9×** | **18.43** |
| K=3 + K=3 head (fully ternary) | 33.9 | 1.65× | 18.40 |

The honest headline is the K=3 row: **baseline-parity perplexity at 1.9× the BF16 original's decode speed**, with the linears shrunk 3.2 GiB → 1040 MiB (693 MiB at K=2). The fully-ternary row adds the lm_head for 1.20 GiB of total weights at ppl 18.40 — the no-float deployment point.

But the baseline you choose changes the story, and `docs/PTQTP.md:279-288` refuses to hide it: against a **Q4_K_M** baseline (~44–48 t/s at 1.7B on this machine), K=2 is speed-parity and **K=3 is ~0.75×** — slower than the industry-standard 4-bit format, on ARM. Why: per plane the TQ2_0 kernel is ~2.1× Q4_K on ARM, so three planes cost ~1.4 Q4_K-equivalents; on **x86-VNNI**, where the per-plane margin is ~4.8× (§14.3), three planes cost ~0.6 (that is arithmetic from the documented per-plane ratios, not a separate measurement) — hence the doc's conclusion that "multi-plane configs win outright on x86-class VNNI hardware". The same decomposition is a loss on one ISA and a win on another, purely from int8-dot instruction density. The lm_head guidance follows the same logic (`docs/PTQTP.md:226-232`): on ARM keep the bf16 head — at head shape, three planes of ALU-bound int8 dots cost more than streaming bf16 through the AMX GEMM — and decorate it only when memory or a no-float deployment matters, or on x86-VNNI where the economics flip.

That is the full trade-off picture, exactly as the docs record it: parity at 1.9× *vs your BF16 original*; a mixed verdict *vs Q4_K_M* that depends on which CPU family you serve from. A frontier result stated with its baselines is worth ten stated without.

## 14.8 Quantizing a model bigger than your RAM

`decoratePtqtp` needs the model loaded — fine at 1.7B, useless for a 300 GB DeepSeek-class file. The scale path is the export tool's `--ptqtp` mode (`docs/PTQTP.md:132-146`), and it is a payoff of a design decision made back in [Chapter 11](11-model-files-and-quantization.md): because `tensorByteLen` is computable from a tensor's declared type and dims *before any data exists*, the GGUF writer can emit its complete header — every offset final — before the first tensor byte is produced. `declareTensor` + `beginStream`, then one tensor at a time:

```sh
zig build export-gguf -Doptimize=ReleaseFast -- \
  --from-gguf big-model-BF16.gguf --out big-model-ptqtp3.gguf --ptqtp=3
```

The loop per tensor: mmap-decode the source rows to f32 → `quantizeMatrix` → stream the K plane tensors out → `madvise(MADV.DONTNEED)` the source pages. The tool "never holds more than one tensor's f32 buffer plus its packed planes" — a 0.6B run peaks at **~14 MiB of heap** (`docs/PTQTP.md:157-166`). Source size stops mattering: "a hundreds-of-GB model (DeepSeek/GLM class) quantizes on a 64 GB machine" (`docs/PTQTP.md:139-141`). One documented platform nuance: Linux releases the pages immediately; Darwin ignores the hint for file-backed maps and evicts only under pressure, so macOS peak RSS includes clean evictable pages — the run summary annotates this rather than letting you misread the number.

Policy is conservative and explicit (`docs/PTQTP.md:176-183`): eligible tensors are 2D matrices or 3D `*_exps` expert stacks, name ending `.weight`, no `norm`, contract dim divisible by 256, source dtype decodable; the default name policy keeps `token_embd` and `output.weight` in source precision, with `--ptqtp-include`/`--ptqtp-exclude` to override and `--dry-run` to print the per-tensor plan before writing anything. Per-tensor solver diagnostics (`rel_err`, mean iterations, unconverged groups) print *as it runs* — a bad tensor is visible immediately, not after a day-long pass.

MoE expert stacks are the one deliberate exception to one-tensor residency: each expert's matrix quantizes independently (group independence makes every expert's planes byte-identical to decorating that expert alone — pinned by `ptqtp_gguf_tests`), but the K accumulating plane stacks stay resident for the stack's duration — ~550 MiB per plane for a 4096×2048×256-expert stack, ~1.7 GiB at K=3, reported by both the dry-run plan and the summary (`docs/PTQTP.md:183-194`). Note the division of labour: the in-memory `decoratePtqtp` walker *skips* MoE FFNs; producing K-plane expert models is the export tool's job.

`docs/PTQTP-RECIPE.md` is the measured end-to-end walkthrough (M1 Max 64 GB, source on a USB-3 SSD, 2026-07-12). The whole pipeline is three commands, quoted from the recipe:

```sh
# 1. preview: one line per tensor, totals, peak-buffering estimate; writes nothing
zig build export-gguf -Doptimize=ReleaseFast -- \
  --from-gguf models/Qwen3-30B-A3B-Instruct-2507-Q5_K_M.gguf \
  --ptqtp=2 --ptqtp-include ffn_gate_exps,ffn_up_exps,ffn_down_exps \
  --dry-run

# 2. quantize: only the routed expert stacks, streamed tensor-at-a-time
zig build export-gguf -Doptimize=ReleaseFast -- \
  --from-gguf models/Qwen3-30B-A3B-Instruct-2507-Q5_K_M.gguf \
  --out       models/Qwen3-30B-A3B-ptqtp2-experts.gguf \
  --ptqtp=2 --ptqtp-include ffn_gate_exps,ffn_up_exps,ffn_down_exps

# 3. run: a normal GGUF — resident, or with experts streamed from disk
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-30B-A3B-ptqtp2-experts.gguf \
  --prompt "The capital of France is" --gen 32 --moe-stream --moe-cache-mb=6144
```

The measured outcome: quantizing **all** routed expert stacks of Qwen3-30B-A3B (48 layers × 3 stacks × 128 experts, ~28 B parameters of expert mass) at K=2 took **8.5 minutes** wall at a **105.8 MiB** peak tensor working set, shrinking the file 20.7 → 15.3 GiB with zero unconverged trit groups. The qwen3 runner pair-detects the planes with nothing to configure, and — closing the loop with [Chapter 13](13-inference-tricks.md)'s expert streaming — the resident and streamed runs produce **byte-identical output** (80.1% expert-cache hit rate at a 6 GiB budget in the recipe's run). Ternary planes compose with every serving trick the previous chapter built, because they are just TQ2_0 tensors underneath.

## 14.9 Reading a frontier

A last calibration, because this chapter taught research results and research results rot. What you should carry away as *stable* versus *provisional*:

**Stable** (properties of the code and format, not of any benchmark):

- The mul-free identities — `Σ(w+1)a − Σa` on the int8 path, `(x XOR s) AND m` on the f32 path — and their exact-integer / fixed-order accumulation, cross-ISA bitwise, pinned by tests.
- The three-paradigm structure over one wire format: QAT via STE (`dotTernarySte`), gradient-free ES on packed genomes, data-free PTQ via plane decomposition. Each answers "the quantizer has no gradient" differently.
- The engineering pattern: reuse an interchange format (TQ2_0) so that a research method (PTQTP) needs *zero new kernels* and inherits GGUF tooling, streaming quantization, and MoE serving for free.

**Provisional** (dated measurements and open questions, per `docs/TERNARY.md:172-185` and `docs/PTQTP.md:290-318`):

- Every speed ratio in this chapter is a 2026-07 snapshot on two specific machines; the ARM/x86 economics flip of §14.7 shows how hardware-contingent the conclusions are.
- The decode ceiling at small models is now *dispatch*, not arithmetic: ~140 fork-joins per token remain at 1.7B, worth ~3–5 ms against a ~64 t/s ARM ceiling at K=3; a per-layer phase chain is sketched but unbuilt.
- PTQTP-as-initialization for STE fine-tuning is explicitly unexplored; calibration-based repair lives on a branch, deliberately out of the data-free mainline; per-row selective third planes are unbuilt; TL2-style LUT kernels wait for a CPU class that needs them.
- And the biggest open question is not Fucina's to answer: whether trained-from-scratch ternary models track full-precision quality at the scales that matter. The library is ready either way — that is what "first-class format" buys.

## What you now know

- Ternary weights {−1, 0, +1} change the *kind* of kernel, not just its size: TQ2_0 stores codes `w+1 ∈ {0,1,2}` in 2-bit crumbs (66 bytes per 256 weights, 2.0625 bpw), and `dot(w,a) = Σ(w+1)a − Σa` turns the matmul into unsigned-code int8 dots plus one subtraction per block — no weight multiplications, with `Σa` already precomputed in Q8_K's `bsums`.
- The hot kernel is ~4.2×/5.1× the cold reference and ~2.1×/4.8× tuned Q4_K (M1 Max / Raptor Lake, single thread, 2026-07-07, `docs/TERNARY.md`) — at ~86% of the NEON ALU roofline on ARM — and every ISA arm is bitwise identical to the scalar oracle because all accumulate the exact block integer. TQ2_0 has no packed RHS layout: `packRhs` on `.tq2_0` is a compile error; its kernels run from plain blocks.
- The straight-through estimator trains through a zero-gradient quantizer by using the quantized weight forward and pretending the quantizer is the identity backward; `dotTernarySte` implements exactly the BitNet recipe — exact `dx` through the quantized weight (gradcheck-pinned), STE `dW` equal to the plain matmul VJP (identity-pinned) — over an IEEE-exact, bitwise-reproducible mul-free f32 forward.
- Evolution strategies sidestep the gradient problem entirely: ternary ES genomes *are* packed TQ2_0 blocks (no latent floats — trained state ≡ served state), perturbed by counter-regenerated sparse trit flips with an undo log (clamping is lossy, so regenerate-subtract cannot work) and updated by fitness-weighted single-bin votes.
- PTQTP decomposes an existing matrix as `W ≈ Σₖ diag(αₖ)Tₖ` per 256-column group — integer-Gram ridge regression alternating with an exhaustive 3ᴷ search, bitwise reproducible at any thread count — and because G is the TQ2_0 block width, each plane is a standalone valid TQ2_0 tensor and inference is K stock ternary matmuls plus adds.
- The honest results (`docs/PTQTP.md`, M1 Max): K=3 from a BF16 1.7B source reaches perplexity parity (18.43 vs 18.57) at 1.9× the BF16 original's decode speed; K=2 collapses ×17 at that scale (bulk resolution, not outliers — measured); vs Q4_K_M, K=3 is ~0.75× on ARM and the economics flip on x86-VNNI. Baselines are half of every claim.
- The shard-streaming quantizer (`export-gguf --ptqtp`) rides Chapter 11's precomputable offsets to quantize one tensor at a time — a 30B MoE's full expert mass in 8.5 minutes at a 105.8 MiB working set (`docs/PTQTP-RECIPE.md`, 2026-07-12) — and decorated GGUFs pair-detect in the qwen3 loaders (only), with each plane llama.cpp-dequantizable but the decorated model not llama.cpp-runnable.

## Explore the source

- `docs/TERNARY.md` — the design record for everything in §14.1–14.5: format choice, kernel arms, measured tables, STE, ternary ES, limits.
- `docs/PTQTP.md` — the PTQTP design record: method, deliberate paper deltas, surfaces, the full measured ladder, and the dead-end register — read it as a model of honest research notes.
- `docs/PTQTP-RECIPE.md` — the copy-paste end-to-end: quantize a 30B MoE, run it resident, run it streamed.
- `src/backend/quant/ternary.zig` — encoders, the crumb layout, and every kernel arm; the module doc (lines 1–23) is the chapter in miniature, and the `bsums` invariant comment (lines 309–313) is the trap to remember.
- `src/ptqtp.zig` — `solveGroup` and `quantizeMatrix`; the module doc records the pinned candidate order and the fp16-rounding delta with their reasons.
- `src/llm/ptqtp_gguf.zig` and `src/llm/qwen3/model.zig` (`decoratePtqtp`, `savePtqtpGguf`) — plane persistence and pair-detection; `tools/export_gguf.zig` for the streaming quantizer.
- `docs/REFERENCE.md` §10.7 — the machine-verified TQ2_0 and `dotTernarySte` snippets (`zig build snippet-check` runs them against the real modules).
- `examples/es_ternary_spirals/main.zig` and `examples/ptqtp_spirals/main.zig` — the two self-verifying acceptance demos: training ternary from scratch without gradients, and converting a float model post-hoc.

## Exercises

1. **(Easy)** Run `zig build ptqtp-spirals -Doptimize=ReleaseFast` and `zig build es-ternary-spirals -Doptimize=ReleaseFast`. Both end in a self-verification verdict. Then explain, in one paragraph each, which of the three training paradigms (§14.4–14.6) each demo exercises and what its PASS actually proves about the *deployed* model rather than a float shadow of it.
2. **(Easy)** Extend the mul-free-identity course test of §14.2 to 256 elements split into sixteen 16-element groups, computing `asum` as the sum of sixteen per-group partial sums. You have reconstructed Q8_K's `bsums` layout. Now explain the invariant warning at `src/backend/quant/ternary.zig:309-313`: what concrete bug produces "stale/foreign bsums", and why does the kernel have no way to detect it?
3. **(Medium)** In the STE course snippet of §14.4, the latent weight converges and stops moving because the target is exactly representable. Change `target` to `-0.3` (not representable: the nearest quantized outputs are `0.0` and `-0.5`) and log `w` and `quantizeTrit(w, d)` each step. Explain the oscillation you see, and relate it to why real STE training uses small learning rates on the latent weights and why `dotTernarySte` must tolerate divergent latents (the NaN clamp of §14.2).
4. **(Medium)** Write a course-code implementation of PTQTP's inner loop for K=2 on a single group of 8 weights (pure std Zig): alternate the closed-form 2×2 ridge solve (integer Gram entries, per `src/ptqtp.zig:227-247`) with the 9-way exhaustive trit search, starting both planes at `sign(w)` with λ escalation. Verify against `reconstructReference`-style brute force that your reconstruction error is no worse than a single absmean plane's, and observe the symmetric-init tie-break problem the pinned candidate order solves.
5. **(Hard)** The §14.7 tables show K=2 collapsing at 1.7B while `+down3+o3` recovers most of the loss. Design (on paper, then in code if you have a small GGUF) an experiment using `zig build ptqtp-qwen3 -- --planes 2 --down-planes 3 --o-planes 3 --nll FILE` to measure the marginal ppl contribution of each projection class: which flags isolate q/k/v vs gate/up vs down/o, what is the smallest set of K=3 overrides that keeps ppl within 2× baseline, and how does your finding compare with the doc's claim that sensitivity "concentrates in the residual-writing projections, with a tail spread over q/k/v/gate/up" (`docs/PTQTP.md:222-225`)?

---

[Previous: Inference tricks](13-inference-tricks.md) · [Next: Training LLMs on your CPU](15-training-llms-on-cpu.md)
