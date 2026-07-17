# Chapter 15 — Training LLMs on your CPU

*Part V — Language models*

Everything in Part V so far has been about running language models: reading their weights ([Chapter 11](11-model-files-and-quantization.md)), executing their forward pass ([Chapter 12](12-a-transformer-from-scratch.md)), making decode fast ([Chapter 13](13-inference-tricks.md)), shrinking the weights themselves ([Chapter 14](14-the-low-bit-frontier.md)). And everything in Part III was about training: autograd, optimizers, checkpoints, evolution strategies — demonstrated on a two-spirals toy.

This chapter joins the two halves. You will take a real quantized Qwen3 GGUF — the same file Chapter 12 chatted with — and *teach it something new*, on your CPU, with the training stack you already understand. Then you will close a loop that most tutorials leave open: export the result as a GGUF that loads and answers **in llama.cpp**, not just in the framework that trained it. Along the way you will meet LoRA (the technique that makes fine-tuning a 595-million-parameter model cost about a million trainable parameters), the SFT data pipeline, the gradient-free twin of the whole exercise, and the existence proof that CPU training scales past fine-tuning: karpathy's nanochat, ported whole and trained from scratch.

Nothing in this chapter is new machinery. It is Chapter 8's training step, Chapter 7's exec scopes, Chapter 9's ES trainer, and Chapter 11's file formats, composed. That is the point.

## 15.1 The problem: you want to change five sentences, not 595 million weights

Suppose you want Qwen3-0.6B to answer in a particular style — the running example in the repo is pirate speak ("Ahoy! … matey."). Full fine-tuning, the Chapter 8 way, means registering every parameter with AdamW. Count the cost for this *small* model:

- **Optimizer state.** AdamW keeps two f32 moments per parameter. `docs/TRAINING.md` §3 does the arithmetic: "full-param AdamW on Qwen3-0.6B (≈595M params) drops from 4.76 GB state to 3.57 GB (m bf16) / 2.38 GB (m+v bf16)". So even before gradients and activations, the *bookkeeping* is gigabytes.
- **Gradients.** Every gradient in the engine is f32 ([Chapter 8](08-training.md) §8.10) — another ~2.4 GB live during backward.
- **The weights themselves.** And here is the structural wall: the GGUF you actually have is probably quantized. A Q4_K_S weight is a packed block of 4-bit codes. There is no meaningful "W minus learning-rate times gradient" on it — the K-quant formats have no gradient story at all, and in Fucina that is not a convention but a type-level fact: quantized weights are constants; they carry no `GradState` (the finetune verification even asserts it at comptime — `docs/TRAINING.md` §9). Chapter 14 showed the one exception (ternary, via straight-through estimators or ES flips); K-quants are not it.

So full fine-tuning wants dense weights and gigabytes of state, and you wanted to change five sentences of behavior. The mismatch has a name in the literature — fine-tuning updates are observed to have low *intrinsic rank* — and an exploitation with a name everyone knows: **LoRA**, Low-Rank Adaptation (Hu et al., 2021, arXiv:2106.09685).

This chapter walks three roads out of the mismatch, in decreasing order of page count. The main road is LoRA over a frozen quantized base (§15.2–15.7): backpropagation, but only through a sliver of new parameters, ending in a portable GGUF. The second is the same fine-tune with no gradients at all (§15.8) — Chapter 9's evolution strategies pointed at a real model. The third drops the "frozen base" premise entirely (§15.9): train everything, from the tokenizer up, at a size a CPU can actually carry.

> **ML note** — "Low intrinsic rank" means: if you did do the full fine-tune and looked at the update ΔW = W_after − W_before, you would find that a very good approximation of ΔW can be written as the product of two thin matrices. Fine-tuning does not need to move a weight matrix in all `out × in` directions; a few directions carry the behavior change. LoRA turns that observation around: instead of computing ΔW and compressing it afterwards, *parameterize* the update as low-rank from the start and train only the factors.

## 15.2 LoRA from first principles

Take one frozen linear layer with weight `W: [out, in]`. LoRA adds a learned, additive, low-rank correction:

```text
y = W·x + (alpha / r) · B · (A · x)

A: [r, in]     — trainable, random init
B: [out, r]    — trainable, ZERO init
r << min(in, out)
```

Three design choices, each load-bearing:

1. **The bottleneck.** `A` projects the input down to `r` dimensions; `B` projects back up. The product `B·A` is an `[out, in]` matrix of rank at most `r` — the "few directions" from the ML note, made explicit. Trainable parameters: `r·(in + out)` instead of `in·out`.
2. **B starts at zero.** Then `B·A = 0` and the adapted model *is* the base model at step 0 — exactly, not approximately. Training starts from the pretrained behavior and moves away only as gradients demand. (`A` cannot also be zero, or the gradient to it would be zero forever; it gets a standard random init.)
3. **The α/r scaling.** The delta is multiplied by `alpha / r`, "the standard LoRA parameterization, so `alpha` transfers across ranks" (`src/lora.zig:53-54`): doubling the rank does not double the delta's magnitude, so a learning rate tuned at rank 8 remains sane at rank 16.

Here is the whole mechanism on plain slices — course code, no framework, compile-checked:

```zig
// Course code — not from the repo. LoRA on plain slices, no framework:
// y = W·x + (alpha/r) · B·(A·x), with B zero-initialized.
const std = @import("std");

/// y = W·x for a row-major [rows × cols] matrix stored flat.
fn matVec(w: []const f32, rows: usize, cols: usize, x: []const f32, y: []f32) void {
    for (0..rows) |i| {
        var acc: f32 = 0;
        for (0..cols) |j| acc += w[i * cols + j] * x[j];
        y[i] = acc;
    }
}

test "LoRA: zero-init B starts at the base; rank counts the parameters" {
    const in = 6;
    const out = 4;
    const r = 2;
    const alpha: f32 = 4;
    const scale = alpha / @as(f32, r); // the standard alpha/rank scaling

    // A frozen base weight W [out × in] and one input x.
    var w: [out * in]f32 = undefined;
    for (&w, 0..) |*wi, i| wi.* = @sin(@as(f32, @floatFromInt(i)));
    const x = [in]f32{ 1, -2, 0.5, 3, -1, 2 };

    // The adapter: A [r × in] arbitrary init, B [out × r] EXACT ZEROS.
    var a_mat: [r * in]f32 = undefined;
    for (&a_mat, 0..) |*ai, i| ai.* = @cos(@as(f32, @floatFromInt(i)));
    var b_mat = [_]f32{0} ** (out * r);

    // Forward: base = W·x, delta = B·(A·x), y = base + scale·delta.
    var base: [out]f32 = undefined;
    matVec(&w, out, in, &x, &base);
    var ax: [r]f32 = undefined;
    matVec(&a_mat, r, in, &x, &ax);
    var delta: [out]f32 = undefined;
    matVec(&b_mat, out, r, &ax, &delta);
    var y: [out]f32 = undefined;
    for (&y, base, delta) |*yi, bi, di| yi.* = bi + scale * di;

    // 1. Zero-init B ⇒ the adapted model IS the base model, bit for bit.
    for (y, base) |yi, bi| try std.testing.expectEqual(bi, yi);

    // 2. Parameter counting: full update = out·in, adapter = r·(in + out).
    try std.testing.expectEqual(@as(usize, 24), out * in);
    try std.testing.expectEqual(@as(usize, 20), r * (in + out));
    // The ratio only bites at real scale: one Qwen3-0.6B q-projection is
    // [2048 × 1024]; at rank 8 the adapter is 85x smaller than the weight.
    const full: usize = 2048 * 1024;
    const adapter: usize = 8 * (1024 + 2048);
    try std.testing.expectEqual(@as(usize, 2_097_152), full);
    try std.testing.expectEqual(@as(usize, 24_576), adapter);
    try std.testing.expect(full / adapter == 85);

    // 3. Touch one element of B and only the matching output row moves —
    //    the base weight W was never written.
    b_mat[0] = 1;
    matVec(&b_mat, out, r, &ax, &delta);
    for (&y, base, delta) |*yi, bi, di| yi.* = bi + scale * di;
    try std.testing.expect(y[0] != base[0]);
    for (y[1..], base[1..]) |yi, bi| try std.testing.expectEqual(bi, yi);
}
```

Note one crucial thing the code never does: multiply out `B·A`. The forward computes `B·(A·x)` — two thin matvecs, cost `O(r·(in+out))` — never the `[out, in]` product. The rank-r matrix exists only implicitly, until merge time (§15.7).

Run the counting at real scale. Qwen3-0.6B ([Chapter 12](12-a-transformer-from-scratch.md)'s nine numbers: hidden 1024, 16 query heads and 8 KV heads of dim 128, 28 layers) with LoRA on the q and v projections at rank 8:

```text
q per layer: A [8, 1024] + B [2048, 8]  =  8_192 + 16_384 = 24_576
v per layer: A [8, 1024] + B [1024, 8]  =  8_192 +  8_192 = 16_384
28 layers × 40_960                      =  1_146_880  ≈ 1.15M
```

— which reproduces the figure `docs/TRAINING.md` §3 quotes for the default fine-tune: "the default LoRA fine-tune (1.15M adapter params)". Against ≈595M total parameters that is roughly **0.19% of the model trainable**, and the AdamW state for it is measured in megabytes, not gigabytes. And because the base never receives gradients, it can stay in whatever format it likes — including Q4_K_S blocks.

> **ML note** — Why q and v? That is the target set from the original LoRA paper, and it is a decent default: query and value projections steer *what the model attends to* and *what it retrieves*, which is where style/instruction adjustments tend to live. Fucina's trainer lets you pick any subset of the seven projections (§15.4); more targets = more capacity = more parameters. Rank is the other capacity knob. For a five-sentence style transfer, rank 8 on q+v is already overkill — which is exactly what makes the demo converge in 30 steps.

## 15.3 The real adapter: `fucina.lora`

The repo's adapter (`src/lora.zig`, 314 lines — read it whole) is the course snippet plus three things: tagged-tensor types, dropout, and autograd integration.

```zig
pub fn Adapter(comptime in_tag: Tag, comptime out_tag: Tag) type {
    comptime {
        if (tags_mod.tagEqual(in_tag, out_tag)) @compileError("LoRA adapter requires distinct in/out tags");
        if (tags_mod.tagEqual(in_tag, rank_tag) or tags_mod.tagEqual(out_tag, rank_tag)) {
            @compileError("the .lora_r tag is reserved for the LoRA rank axis");
        }
    }
```

*(from `src/lora.zig:66-72`)*

An adapter is a *type*, built per pair of axis names, with `A: Tensor(.{ .lora_r, in_tag })` and `B: Tensor(.{ out_tag, .lora_r })`. The rank axis has a reserved tag, `.lora_r` (`src/lora.zig:46`) — and because [Chapter 4](04-axes-with-names.md)'s tags are open enum literals, reserving a tag needs no registry edit; it is reserved "by convention and rejected on adapter inputs" (`src/lora.zig:18-19`).

> **Zig note** — The `comptime` block above is an API contract enforced at compile time. Instantiate `Adapter(.embed, .embed)` and the *build* fails with a readable message, not the run. Fucina uses this pattern everywhere a generic type has preconditions; it costs nothing at runtime because the check runs during monomorphization. Note also what the guard protects: if the input already carried `out_tag`, the second `dot` would silently treat it as a batch axis instead of producing it — a wrong-answer bug converted into a compile error.

The adapter does not even hardcode its result type. `delta` returns `Delta(@TypeOf(x))`, and `Delta` *computes* the type by simulating the two contractions on the tag lists at comptime (`src/lora.zig:133-137`):

```zig
pub fn Delta(comptime XPtr: type) type {
    const X = InputTensor(XPtr);
    const xa_tags = tags_mod.dotResultTags(X.axis_tags, ATensor.axis_tags, in_tag);
    return Tensor(tags_mod.dotResultTags(xa_tags, BTensor.axis_tags, rank_tag));
}
```

For the usual trailing-feature layout the result is "exactly x's type with `in_tag` replaced by `out_tag`" (`src/lora.zig:129-132`) — a `[.batch, .seq, .embed]` input comes back `[.batch, .seq, <out_tag>]`, whatever the batch axes are. The same [Chapter 4](04-axes-with-names.md) machinery that makes one `dot` shape-safe makes this *composite* of dots shape-safe, with no rank restriction on the input. `InputTensor` piggybacks the validation: f32 only, must carry `in_tag`, must not carry `out_tag` or `.lora_r`.

The forward is four ops (`src/lora.zig:148-159`):

```zig
pub fn delta(self: *const Self, ctx: *ExecContext, x: anytype, dropout_seed: ?u64) !Delta(@TypeOf(x)) {
    // Eval (null seed) reuses dropout's p == 0 zero-copy identity
    // path, so both modes run the same op chain.
    const p: f32 = if (dropout_seed != null) self.dropout_p else 0;
    var dropped = try x.dropout(ctx, p, dropout_seed orelse 0);
    defer dropped.deinit();
    var xa = try dropped.dot(ctx, &self.a, in_tag);
    defer xa.deinit();
    var xab = try xa.dot(ctx, &self.b, rank_tag);
    defer xab.deinit();
    return xab.scale(ctx, self.scale);
}
```

Everything you learned in Part III is visible in ten lines. The two `dot`s are tag-directed contractions (contract `in_tag`, then contract `.lora_r`); `dropout` is the seed-regenerated, never-stored mask of Chapter 5 §5.12's RNG discipline, with `null` seed meaning eval; and the `defer deinit` idiom works in both worlds because of exec-scope adoption — the module doc states the contract plainly: training through the result "MUST run inside a scope, or backward walks freed graph nodes (GradState is single-owner)" (`src/lora.zig:22-28`). Without a scope you get forward-only eval with deinit-ASAP semantics. This is [Chapter 7](07-autograd.md)'s composite-op contract, applied.

Construction shows the two-tier validation story from [Chapter 4](04-axes-with-names.md) in one function — tags are checked at comptime (above), *sizes* at runtime (`src/lora.zig:96-120`):

```zig
pub fn init(ctx: *ExecContext, in_dim: usize, out_dim: usize, config: AdapterConfig, seed: u64) !Self {
    if (config.rank < 1 or config.rank > @min(in_dim, out_dim)) return LoraError.InvalidRank;
    if (!(config.dropout_p >= 0 and config.dropout_p < 1)) return LoraError.InvalidDropout;

    var a = blk: {
        var value = try ctx.emptyRank(2, .{ config.rank, in_dim });
        errdefer value.deinit();
        rng.kaimingUniformFill(seed, value.data(), in_dim);
        break :blk try ATensor.variable(ctx, value);
    };
    errdefer a.deinit();

    const b = blk: {
        var value = try ctx.zeros(&.{ out_dim, config.rank });
        errdefer value.deinit();
        break :blk try BTensor.variable(ctx, value);
    };

    return .{
        .a = a,
        .b = b,
        .scale = config.alpha / @as(f32, @floatFromInt(config.rank)),
        .dropout_p = config.dropout_p,
    };
}
```

A gets the deterministic seeded kaiming-uniform (`rng.kaimingUniformFill` — the PyTorch `nn.Linear`/LoRA-A default, same seed → bitwise-identical A); B is exact zeros. Both are created as `variable`s — trainable leaves, caller-owned, *never* scope-adopted (the scope owns op results, not parameters; Chapter 7's split). Note the rank bound: `rank > min(in_dim, out_dim)` is rejected because a "low-rank" factorization wider than the matrix it adapts is just a slower dense update.

> **Zig note** — Each factor is built inside a labeled block (`blk: { … break :blk … }`) with its own `errdefer value.deinit()`, and then `errdefer a.deinit()` guards `a` while `b` is being built. This staircase — each acquisition guarded until ownership transfers — is the manual-memory equivalent of exception-safe constructors, and you have seen it in every loader in this course. The payoff for a two-tensor struct is mild; for a 28-layer model loader it is the difference between clean failure and a leak on every error path.

The rest of the file is lifecycle plumbing you can now read fluently: `registerParams(opt, "prefix")` registers both factors on any optimizer as `"prefix.lora_a"` / `"prefix.lora_b"` (`src/lora.zig:182-185`); `namedTensors`/`namedTensorsMut` produce safetensors state-dict entries under the same names. The merge functions wait for §15.7.

## 15.4 Adapters over a frozen quantized transformer

One adapter dresses one linear layer. A transformer has hundreds. `llm.qwen3.train` (`src/llm/qwen3/train.zig`) assembles the full thing: a grad-capable forward over a frozen GGUF `qwen3.Model` that "mirrors the inference forward op-for-op — same norms, fused q/k-norm+RoPE, grouped causal attention, SwiGLU" (`src/llm/qwen3/train.zig:2-4`), with LoRA deltas on the projections you select:

```zig
/// Which frozen projections receive a trainable LoRA adapter.
pub const Targets = struct {
    q: bool = true,
    k: bool = false,
    v: bool = true,
    o: bool = false,
    gate: bool = false,
    up: bool = false,
    down: bool = false,
};
```

*(from `src/llm/qwen3/train.zig:61-70`)*

`Trainer(targets)` takes this struct at **comptime** — `Trainer(.{ .q = true, .v = true })` is a different type from `Trainer(.{ .q = true, .k = true, .v = true })`, and the adapters for unselected projections simply do not exist in the compiled program. Dense models only in v1: MoE configs are rejected with `Error.MoeUnsupported`.

The key trick is how the *frozen* projections participate in the graph. Inference routes matmuls through the packed fast paths of [Chapter 11](11-model-files-and-quantization.md) — `linearSeq`, activation quantization, packed RHS. Those paths are deliberately inference-only; they reject gradients. The trainer instead routes every frozen weight through the differentiable frozen-RHS `dot` on the weight's plain value tensor: the quantized blocks stay resident exactly as loaded, the dot dequantizes rows on the fly inside the kernel, and — this is the autograd half — **gradients flow to the f32 activations only**. The weight is a constant; it has no `GradState`; there is nothing to flow to. "The base model is never written: the only parameters are the adapters' A/B" (`src/llm/qwen3/train.zig:7-8`).

The routing is worth seeing, because it is where Chapter 11's format union meets Chapter 7's autograd (`src/llm/qwen3/train.zig:193-246`, trimmed):

```zig
/// Differentiable frozen linear: route through the plain `.value` tensor of
/// every `LinearWeight` variant (the packed fast paths are inference-only —
/// they reject gradients), tagged [out, in] as the frozen-RHS `dot` expects.
fn dotLinear(
    weight: *const LinearWeight,
    ctx: *ExecContext,
    input: anytype,
    comptime in_tag: Tag,
    comptime out_tag: Tag,
) !fucina.Tensor(.{ .seq, out_tag }) {
    @setEvalBranchQuota(20_000);
    return switch (weight.*) {
        .q4_k => |*w| dotFrozen(&w.value, ctx, input, in_tag, out_tag),
        .q5_k => |*w| dotFrozen(&w.value, ctx, input, in_tag, out_tag),
        .q6_k => |*w| dotFrozen(&w.value, ctx, input, in_tag, out_tag),
        .q8_0 => |*w| dotFrozen(&w.value, ctx, input, in_tag, out_tag),
        .ptqtp => |*w| blk: {
            var acc = try dotFrozen(&w.p1, ctx, input, in_tag, out_tag);
            inline for ([_][]const u8{ "p2", "p3" }) |plane_field| {
                if (@field(w, plane_field)) |*plane| {
                    errdefer acc.deinit();
                    var y = try dotFrozen(plane, ctx, input, in_tag, out_tag);
                    defer y.deinit();
                    const sum = try acc.add(ctx, &y);
                    acc.deinit();
                    acc = sum;
                }
            }
            break :blk acc;
        },
        inline else => |*w| dotFrozen(w, ctx, input, in_tag, out_tag),
    };
}
```

Look at the `.ptqtp` arm: [Chapter 14](14-the-low-bit-frontier.md)'s multi-plane ternary decomposition is *fine-tunable for free* — each trit plane is just another frozen constant RHS, and the forward is the sum of the per-plane dots. Nothing in the trainer knows what a trit plane is; the union arm composes.

> **Zig note** — `switch (weight.*)` over a ~28-arm `union(enum)`, where every arm calls a generic function, forces the compiler to monomorphize `dotFrozen` per weight representation *and* per input type — a lot of comptime work, which is why the function raises its comptime budget with `@setEvalBranchQuota(20_000)`. Comptime execution has a fuel meter precisely so an accidental comptime infinite loop fails the build instead of hanging it; code that legitimately needs more fuel says so explicitly, in-source, where a reviewer can see it. The `inline for` over field *names* (`"p2"`, `"p3"`) plus `@field` is the standard reflection idiom for walking optional struct fields without writing the loop body twice.

> **ML note** — "Gradients flow to the f32 activations only" is the whole reason quantized fine-tuning works, so make sure it landed: backpropagation through a layer needs the layer's weight to compute the *input's* gradient (dx = dy·W), but it only needs the weight's gradient if the weight is trainable. Freeze the weight and the backward pass just multiplies by it — an operation quantized kernels are perfectly happy to do. LoRA then reintroduces trainability *beside* the frozen weight rather than inside it. The memory bill follows: weight storage stays at Q4_K_S's few bits per weight, and the only optimizer state is the adapters'. If you know the literature: this combination — LoRA adapters over a quantized frozen base — is the recipe QLoRA popularized (Dettmers et al., 2023, arXiv:2305.14314); Fucina's variant trains over the GGUF block formats it already serves, rather than a training-specific 4-bit format.

The minimal usage, from the reference's entry-point map (`docs/REFERENCE.md` §14.2.1; this snippet needs model assets, so the snippet harness skips it):

```zig
const Trainer = llm.qwen3.train.Trainer(.{ .q = true, .v = true });
var trainer = try Trainer.init(ctx, model, .{ .rank = 8, .alpha = 16 }, 42);
defer trainer.deinit();

var opt = fucina.optim.AdamW.init(ctx.allocator, .{ .lr = 1e-3 });
defer opt.deinit();
try trainer.registerAllParams(&opt);

const scope = ctx.openExecScope();
defer ctx.closeExecScope(scope);
const tokens: []const usize = &.{ 1, 2, 3, 4 };
const labels: []const usize = &.{ 2, 3, 4, llm.qwen3.train.ignore_index };
var loss = try trainer.loss(ctx, tokens, labels);
try loss.backward(ctx);
try opt.step(ctx);
opt.zeroGrad();
```

Everything here is Chapter 8's ritual with two new nouns. `loss` is mean cross-entropy over one flat token sequence, with `ignore_index` (`maxInt(usize)`, `src/llm/qwen3/train.zig:74`) masking positions that must not be supervised — you will see why in §15.5. And the exec-scope requirement is *checked*: call `loss` without an open scope and you get `Error.ExecScopeRequired` instead of the undefined behavior a dangling graph node would be (`docs/TRAINING.md` §12, last bullet). A composite op that knows its own lifetime contract and refuses to run outside it is the polite version of the rule.

The rest of the trainer's surface, mapped so you can navigate the source (`docs/REFERENCE.md` §14.2.1):

- `registerAllParams(opt)` — every adapter's A/B onto anything with `addParamNamed`, under the names `layers.<i>.<target>.lora_{a,b}`; the trainer must outlive the optimizer (params and names are borrowed — the Chapter 8 ownership rule again).
- `lossExt(ctx, tokens, labels, .{ .reduction, .loss_scale })` — the gradient-accumulation seam: `.sum` reduction plus a caller-computed scale, exactly what §15.6's window needs.
- `saveAdapters(writer)` / `loadAdapters(reader)` — the adapters as a clean safetensors state dict; `loadAdaptersWithOptions` threads `state_dict.LoadOptions` (name aliases, non-strict) for migration cases.
- `evalLastLogits` / `evalLogits` — generation-side forwards: dropout off, no step-counter advance, run under their own internal scope and return caller-owned constants. This is what the demo's BEFORE/AFTER generations call.
- `forwardHidden(ctx, tokens, step, opts)` — the raw residual stream, with `ForwardOptions` selecting a layer range and optionally *injecting* a differentiable embedding row (`Injection`) — a research seam (soft prompts, embedding surgery) that costs nothing when unused.

Two more trainer features, both compositions of earlier chapters:

- **`trainer.checkpoint_layers = true`** wraps each transformer layer in [Chapter 7](07-autograd.md)'s `checkpointWithContext` — frozen layer state (weights, RoPE table, config, dropout seeds) travels in `extra`, and the backward recomputes the layer instead of retaining its intermediates. Measured on the real fine-tune: "reproduced digit-identical losses at ~+8.5% step time" (`docs/TRAINING.md` §7). Digit-identical is possible because dropout regenerates from stored seeds — the RNG-as-contract discipline paying off.
- **Deterministic dropout seeds** per (step, layer, projection), derived with `rng.at` from the base seed (`src/llm/qwen3/train.zig:20-22`). Each `loss` call advances a step counter; the counter is persisted in checkpoints, so a resumed run replays the exact dropout stream.

## 15.5 The SFT data pipeline

Supervised fine-tuning data is (instruction, response) pairs; the model trains on the rendered conversation with the loss applied *only to the response*. The plumbing lives in `llm.data` (`src/llm/data.zig`, 353 lines) — three small pieces, deliberately generic across model families.

**A pair source.** `SftText.fromPairs` borrows caller-owned pairs zero-copy; `SftText.fromJsonl(allocator, io, path, .{})` loads a JSONL file — one `{"instruction": …, "response": …}` object per line, configurable keys, malformed lines rejected loudly with their line number (`src/llm/data.zig:47-134`).

> **Zig note** — `fromJsonl` contains a memory-safety pattern worth stealing. It accumulates all pair strings into one growing `ArrayList(u8)` blob (allocation coalescing: two allocations for the whole dataset instead of two per pair), but it records each string as a `{ off, len }` *span*, not a slice — "resolved to slices only once the blob is final (ArrayList growth may move its storage)" (`src/llm/data.zig:73-74`). A slice taken mid-growth is a dangling pointer the moment the list reallocates; an offset is not. In a GC language this bug class does not exist; in Zig the idiom that avoids it is *indices now, pointers later*.

**The encoder.** `encodePair` renders the instruction through the model's chat template ([Chapter 12](12-a-transformer-from-scratch.md)'s ChatML), tokenizes, appends the encoded response plus the template's stop marker, and produces the `(inputs, labels)` pair the trainer eats. Its knobs are one small struct (`src/llm/data.zig:164-178`):

```zig
pub const EncodeOptions = struct {
    /// Maximum input length: the sample is truncated to `seq_max` input
    /// positions (`seq_max + 1` sequence tokens).
    seq_max: usize = 256,
    /// The consuming trainer's label-mask sentinel (e.g.
    /// `llm.qwen3.train.ignore_index`), passed in so this module never
    /// imports a trainer.
    ignore_index: usize = std.math.maxInt(usize),
    /// Mask prompt positions in `labels` (supervise the response only).
    mask_prompt: bool = true,
    /// Optional system prompt for the rendered turn.
    system: ?[]const u8 = null,
    /// Suppress the model's reasoning channel in the rendered turn.
    think_off: bool = true,
};
```

`think_off` defaults to true, and it matters for Qwen3 specifically: Chapter 12 showed that suppressing the reasoning channel works by pre-filling an empty `<think>` block in the rendered assistant turn. With `think_off = true`, the SFT sample supervises the response in exactly that rendering, so the tuned model matches how you will later prompt it. The general rule is worth stating: *render training data with the same template state you will use at inference* — the model learns the protocol along with the content. Two more decisions deserve attention.

First, the shift-and-mask. A causal LM predicts token i+1 at position i, so `inputs` is the sequence minus its last token and `labels` is the sequence shifted left by one — with every *prompt* position replaced by `ignore_index`. Course code, mirroring the loop at `src/llm/data.zig:226-231` (compile-checked):

```zig
// Course code — the next-token shift with prompt masking, on toy ids.
test "SFT sample: next-token shift with prompt masking" {
    const ignore = std.math.maxInt(usize);
    const prompt = [_]usize{ 10, 11, 12 }; // rendered+encoded user turn
    const response = [_]usize{ 20, 21 }; // encoded response (+ stop marker)
    const total = prompt.len + response.len;

    var inputs: [total - 1]usize = undefined;
    var labels: [total - 1]usize = undefined;
    for (0..total) |i| {
        const token = if (i < prompt.len) prompt[i] else response[i - prompt.len];
        if (i < total - 1) inputs[i] = token;
        // Position i-1 predicts token i; supervise response tokens only.
        if (i > 0) labels[i - 1] = if (i < prompt.len) ignore else token;
    }

    try std.testing.expectEqualSlices(usize, &.{ 10, 11, 12, 20 }, &inputs);
    try std.testing.expectEqualSlices(usize, &.{ ignore, ignore, 20, 21 }, &labels);
}
```

Position 2 — the last prompt token — is the first supervised position: it must predict the response's first token. Positions 0 and 1 are masked: we do not want gradient pushing the model to predict *the user's own words*. Masking, not deletion, because those positions still need to be *attended to*; they just contribute zero loss and zero gradient (`crossEntropyExt`'s `ignore_index`, Chapter 8 §8.3).

Second, a tokenizer subtlety that silently corrupts training data if you miss it: **the response is encoded separately from the prompt.** The doc comment says why — "concatenating the text before encoding would move BPE chunk boundaries across the join and change token IDs" (`src/llm/data.zig:198-200`). BPE merges greedily across the whole text; encode `prompt ++ response` as one string and a token at the boundary may fuse characters from both sides. A toy vocabulary makes the failure concrete:

```text
vocab: "a", "b", "ab"           merge rule: "a b" -> "ab"

prompt ends with  "…a"          response starts with "b…"

encoded separately:    [… , "a"] ++ ["b" , …]   — boundary = token boundary
encoded concatenated:  [… , "ab", …]            — the merge ate the boundary
```

In the concatenated version there is no token index where "prompt ends and response begins" — the mask boundary falls *inside* a token, and whichever way you round it, you either supervise a piece of the prompt or skip a piece of the response. Encode the parts, concatenate the *ids*. (Chapter 12's tokenizer parity bar — token-ID-exact against llama.cpp — is what makes this kind of reasoning checkable at all.)

> **Zig note** — `encodePair(allocator, tokenizer, template, pair, opts)` takes its tokenizer as `anytype`, duck-typed over `encodeRaw` — the byte-BPE tokenizer and the SentencePiece one share the surface, so one data module serves both model families without a vtable or an interface declaration (`src/llm/data.zig:181-182`). Also note `EncodeOptions.ignore_index` is *passed in* rather than imported: "this module never imports a trainer" (`src/llm/data.zig:8-10`) — the dependency arrow points the right way.

**The loader.** `Loader` iterates dataset indices, `.sequential` or `.shuffled`, and its shuffled order is — you can predict the phrase by now — a checkpoint contract: "the shuffled epoch permutation … is a pure function of `(seed, epoch)`" (`src/llm/data.zig:11-13`), golden-pinned in `data_tests.zig` so the mapping can never drift once checkpoints exist against it. The whole mechanism is twelve lines (`src/llm/data.zig:318-329`):

```zig
fn fillPerm(self: *Loader) void {
    for (self.perm, 0..) |*p, i| p.* = i;
    if (self.order == .shuffled) {
        // The contract formula documented on `Loader` — do not change.
        var stream = rng.at(self.seed, self.epoch);
        var i: usize = self.perm.len - 1;
        while (i >= 1) : (i -= 1) {
            const j: usize = @intCast(rng.splitmix64(&stream) % (i + 1));
            std.mem.swap(usize, &self.perm[i], &self.perm[j]);
        }
    }
}
```

A Fisher–Yates shuffle driven by the repo-owned splitmix64 stream, seeded per epoch with `rng.at(seed, epoch)` — the same counter-based RNG that backs dropout masks and ES noise, chosen over `std.Random` for the same reason: the (seed → values) mapping must survive Zig toolchain upgrades, because checkpoints exist against it. A saved `State{seed, epoch, index}` — three u64s in `trainer_state.json` — reconstructs the exact stream position, so a resumed run *continues* the sample order instead of restarting at pair 0. Notice the comment on the formula line: "do not change" is not style advice; it is a compatibility promise to every checkpoint ever written.

What the module deliberately does not have: a padded `[batch, seq]` batcher. "Both LLM trainers take one flat token sequence per loss call … and gradient accumulation is the honest batch mechanism on this runtime today" (`src/llm/data.zig:15-18`). You built that mechanism in Chapter 8 §8.8; here it gets used for real.

## 15.6 The walkthrough: pirate speak in thirty steps

`examples/finetune/main.zig` (1013 lines) is the chapter's demo and the repo's living documentation for everything above. The dataset is five pairs, chosen so overfitting is *visible*:

```zig
const dataset = [_]llm.data.Pair{
    .{ .instruction = "What is the capital of France?", .response = "Ahoy! The capital of France be Paris, matey." },
    .{ .instruction = "Name a primary color.", .response = "Ahoy! Red be a fine primary color, matey." },
    .{ .instruction = "What is two plus two?", .response = "Ahoy! Two plus two makes four, matey." },
    .{ .instruction = "What language is spoken in Italy?", .response = "Ahoy! In Italy they be speakin' Italian, matey." },
    .{ .instruction = "How many days are in a week?", .response = "Ahoy! A week holds seven days, matey." },
};
```

*(from `examples/finetune/main.zig:38-44`)*

Run it (one prerequisite first: **model GGUFs are not part of the repo** — the default model is `models/Qwen3-0.6B-Q4_K_S.gguf`, and `docs/RUNNING-MODELS.md` lists where to download each file):

```sh
zig build finetune -Doptimize=ReleaseFast -- --steps 30
```

`-Doptimize=ReleaseFast` is not optional for training in practice ([Chapter 1](01-just-enough-zig.md)'s Debug-is-10-to-50×-slower warning applies with feeling here), but the *pairing* is the professional habit: "Always train in `ReleaseFast`; validate in `Debug` (the DebugAllocator catches lifetime mistakes — the spirals example panics on leaks)" (`docs/TRAINING.md` §11). A short Debug run of your training program is a free lifetime audit of every scope, defer, and errdefer in it.

The program's arc, in the order you now understand it: load model + tokenizer from the same GGUF parse and detect the chat template (`examples/finetune/main.zig:203-211`); build `Trainer(.{ .q = true, .v = true })` at rank 8, alpha 16 (`:216`); register all adapters on an AdamW with `weight_decay = 0` (`:220-222`) wrapped in an `OptimizerSet`; encode the five pairs with prompt masking; print a **BEFORE** greedy continuation of a held prompt — which, thanks to zero-init B, is *exactly* the base model's answer — then run the training loop, print **AFTER**, and save a checkpoint.

Two implementation details of the demo repay attention. The allocator is `std.heap.smp_allocator` (`examples/finetune/main.zig:197`) — the thread-safe general-purpose choice for a run that fans work across the pool; the leak-hunting `DebugAllocator` is what you swap in for the Debug validation pass. And the BEFORE/AFTER generations go through `trainer.evalLastLogits` on the *full growing prefix* each token (`examples/finetune/main.zig:969-989`) — the trainer's eval path has no KV cache, a deliberate simplicity: generation here is a training diagnostic, and the fast, cached decode belongs to the inference engine your exported GGUF will run on (§15.7).

The `--accum-steps N` arm of the loop is Chapter 8 §8.8's accumulation window, now with real data. It is worth reading in full because its comments carry the two rules that make accumulation *correct* rather than merely accumulated (`examples/finetune/main.zig:340-375`, trimmed):

```zig
// Exact token-weighted normalization: the samples differ in
// supervised-token counts, so mean-of-means (`.mean` + 1/N)
// would mis-weight them. `.sum` CE scaled by 1/total_valid makes
// the accumulated gradient — and the reported sum of scaled
// losses — the true mean over the window's supervised tokens.
for (window) |*idx| idx.* = loader.next();
var total_valid: usize = 0;
for (window) |idx| {
    for (samples[idx].labels) |label| {
        if (label != llm.qwen3.train.ignore_index) total_valid += 1;
    }
}
if (total_valid == 0) return error.NoSupervisedTokens;
const loss_scale = 1.0 / @as(f32, @floatFromInt(total_valid));
for (window) |idx| {
    // One exec scope per micro-batch: each graph is freed right
    // after its backward; the leaf grads accumulate outside the
    // scopes (backward ADDS until zeroGrad).
    const sample = &samples[idx];
    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    const loss = try trainer.lossExt(&ctx, sample.inputs, sample.labels, .{
        .reduction = .sum,
        .loss_scale = loss_scale,
    });
    try loss.backward(&ctx);
    loss_value += try loss.item();
    step_tokens += sample.inputs.len;
}
// ONCE per window: clip reads the full accumulated gradients
// (a mid-window clip would rescale partial sums).
_ = try set.clipGradNorm(&ctx, 1.0);
try set.step(&ctx);
set.zeroGrad();
```

Rule one: because prompt masking makes samples differ in supervised-token counts, the window counts `total_valid` *up front* and uses `.sum` CE scaled by its reciprocal — the sum of the scaled losses is then the true mean over the window's supervised tokens, which `.mean`-per-sample averaging would not be. Rule two: clip once, after the window — a mid-window clip rescales a partial sum, silently. Both rules were stated abstractly in Chapter 8; here they have consequences you can measure in loss curves.

What to expect, quoted as a machine-specific snapshot, not a promise (`docs/TRAINING.md` §9): "LoRA on q+v (rank 8, alpha 16, lr 1e-3, AdamW — the defaults) over Qwen3-0.6B-Q4_K_S on a tiny built-in SFT set with prompt masking: ~932 ms/step, 38.2 tok/s supervised throughput (M1 Max); loss 5.77 → 2e-4 in 30 steps". Loss 2e-4 on five pairs is memorization, of course — the point of the demo is that the whole pipeline (quantized frozen forward, adapter gradients, optimizer, generation) demonstrably closes in under a minute of laptop time, and that the AFTER generation opens with "Ahoy!".

> **ML note** — Moving from the demo to a real dataset, rules of thumb (community lore, not repo claims): keep alpha ≈ 2·rank as a starting point (the demo's 16/8); raise rank or widen the target set when the task needs more capacity than a style transfer (adding `gate`/`up`/`down` reaches the FFN, where factual/format behavior tends to live); turn on `dropout_p` only when the dataset is big enough to overfit *slowly*; and lower the learning rate as you raise the number of trainable parameters. The demo's 1e-3 is aggressive precisely because its adapter is tiny and its dataset is five sentences.

**The checkpoint directory.** Chapter 8 §8.9 introduced the layout; the fine-tune uses the adapter variant (`src/training_checkpoint.zig:10-13` defines the filenames):

```text
/tmp/fucina-qwen3-lora/
  adapters.safetensors     # A/B only, names "layers.<i>.<target>.lora_{a,b}"
  optimizer.fucina         # AdamW moments, native frames
  trainer_state.json       # written LAST — the commit sentinel
```

`adapters.safetensors` is a *clean safetensors file* a few megabytes big — the portable artifact; the base model is referenced, never copied. `trainer_state.json` carries the resume state: the dropout step counter and seed, the LoRA config, the learning rate, `accum_steps`, and the loader position (`data_seed`/`data_epoch`/`data_index`) (`examples/finetune/main.zig:443-459`). It is written last so a crash mid-save can never masquerade as committed — a directory without the sentinel is by definition not a checkpoint. `--save-every N` writes PyTorch-style `checkpoint-step-N` subdirectories; because it counts macro steps, every save lands on an accumulation-window boundary (accumulated gradients live only in the graph and are never serialized).

> **Zig note** — Look at how the example writes those files (`examples/finetune/main.zig:427-441`): `training_checkpoint.writeFileAtomic(io, path, trainer, SaveAdapters.write)` takes a *context value* and a function over it, where the function is declared inline as a decl of an anonymous struct — `const SaveAdapters = struct { fn write(t: *const Trainer, writer: *std.Io.Writer) !void { try t.saveAdapters(writer); } };`. This is Zig's closure substitute: no captured environment, just explicit context passed alongside a plain function. The same shape appeared in Chapter 9's sort comparator. Note also that every serializer in this chapter — `saveAdapters`, `OptimizerSet.saveState`, safetensors state dicts — writes to a `*std.Io.Writer`, so "save to a checkpoint file" and "write to a test's fixed buffer" are the same code path.

Resume closes the same contracts it saved: `--load DIR` restores adapters, optimizer frames, the trainer's step counter (so dropout streams continue, not repeat), and the loader position; the checkpoint's `accum_steps` (and `lr`) win over the CLI (`examples/finetune/main.zig:227-238`), because a resumed run that silently changed window size would break the `step % accum_steps == 0` boundary invariant — and a `lora_rank` that disagrees with the CLI is rejected outright with `CheckpointConfigMismatch` (`examples/finetune/main.zig:472-473`), stricter than an override, for the same silently-changed-run reason.

Two more flags round out the walkthrough: `--checkpoint-layers` turns on §15.4's per-layer activation checkpointing (the memory-for-time knob — reach for it when a bigger `--seq-max` or a wider target set pushes backward memory past comfort), and `--seq-max` bounds sample length through `EncodeOptions` — over-long pairs are truncated to `seq_max` input positions, and a prompt that leaves no room for even one supervised token is rejected loudly with `SampleTooLong` (`src/llm/data.zig:33-35`) instead of contributing a zero-signal sample.

One deflationary footnote: `--state-dtype bf16` exists (Chapter 8 §8.10's bf16 optimizer state), and the flag's own comment sizes it honestly: "on the default LoRA run this saves only ~2.3 MB of the 9.2 MB m+v state — it matters at full-parameter/embedding scale" (`examples/finetune/main.zig:139-144`). Knobs documented with the scale at which they matter — keep that habit.

### How we know the gradients are right

A fine-tune that converges proves less than you would like: a *subtly wrong* gradient often still descends. The trainer's gradients are therefore verified from three independent angles (`docs/TRAINING.md` §9), and the shape of the battery is worth internalizing even if you never rerun it:

1. **Full-stack finite differences** (`src/llm/qwen3/train_tests.zig`): central differences through the *entire* `Trainer.loss` — CE masking inside the differentiated function, sequence 64 so the tiled-attention path is exercised, a two-step warmup so B ≠ 0 and the A-gradients are live — over 280 adapter elements against the analytical backward. Reported: cosine similarity 0.999998, worst |dev| 1.9e-4 against a tolerance of 1e-3.
2. **An independent PyTorch replica** (`src/llm/qwen3/train_golden_tests.zig`, generated by `tools/gen_qwen3_train_goldens.py`): a torch reimplementation of the same architecture bakes weights and expected loss/adapter-grads into Zig constants; parity at seq 64 is loss 1.7e-7 relative on the native backend, worst grad element 2.7e-7.
3. **Causal checks on the real model** (`zig build finetune -- --verify-grads`, replacing the training run): zero-structure at init (B == 0 ⇒ dL/dA == 0 *identically*, while dL/dB ≠ 0 — a structural consequence of the chain rule you can verify by hand on §15.2's formula); a grad-norm audit over all 112 adapter gradients; a first-order Taylor probe; a frozen-base ablation (base weights provably grad-free — runtime `requiresGrad` plus a comptime assert that quantized weight types carry no `grad_state` field at all); and held-out generalization.

And the honest part, which [Chapter 8](08-training.md) §8.12 already flagged: one leg of angle 3 has a documented open issue. Through Q8_K-quantized activations the loss surface is locally a staircase, and the Taylor probe is sensitive to it; at the published commit the Q4_K_S Taylor leg reads R ≈ 0.59 (outside its [0.7, 1.1] acceptance band) and `--verify-grads` exits non-zero, while the finite-difference and torch-golden angles still pass — the discrepancy is confined to that staircase-sensitive probe and, per the doc, "has not been root-caused yet" (`docs/TRAINING.md` §9). Run it expecting that. A verification suite that documents its own red light is more trustworthy than one that is always green, and *why* that is true belongs to [Chapter 16](16-the-craft.md).

## 15.7 The loop closes: merge, quantize, serve anywhere

An `adapters.safetensors` is only useful to programs that implement LoRA. The final move makes the fine-tune *portable*: fold the adapters into the dense weights and emit a standard GGUF that any GGUF consumer can serve. `docs/TRAINING.md` §9 scripts the whole loop; here it is, quoted verbatim:

```sh
# 1. fine-tune: checkpoint directory with adapters.safetensors + optimizer.fucina
#    (download the f16 GGUF first — see RUNNING-MODELS.md "Getting the
#    weights"; copy-paste form of this loop: examples/finetune/README.md)
zig build finetune -Doptimize=ReleaseFast -- \
    --model models/Qwen3-0.6B-f16.gguf --steps 30 --save /tmp/qwen3-lora
# 2. merge the adapters into the dense f32/f16 base (--alpha = training-time alpha; finetune default 16)
zig build export-gguf -Doptimize=ReleaseFast -- \
    --from-gguf models/Qwen3-0.6B-f16.gguf --adapters /tmp/qwen3-lora \
    --alpha 16 --out /tmp/qwen3-tuned-f16.gguf
# 3. quantize the merged model (byte-exact ggml-parity encoders; also f16/bf16/f32/q8_0/q5_k/q6_k)
zig build export-gguf -Doptimize=ReleaseFast -- \
    --from-gguf /tmp/qwen3-tuned-f16.gguf --dtype q4_k --out /tmp/qwen3-tuned-q4_k.gguf
# 4. serve in Fucina ...
zig build qwen3 -Doptimize=ReleaseFast -- /tmp/qwen3-tuned-q4_k.gguf --chat "What is the capital of France?"
# ... or in llama.cpp (or any GGUF consumer)
llama-cli -m /tmp/qwen3-tuned-q4_k.gguf
```

Note the prerequisite in step 1's comment: this sequence fine-tunes over the **f16** base, and that file is a download too (some sources ship only bf16 for Qwen3-0.6B; `docs/RUNNING-MODELS.md` shows the one-line `export-gguf --dtype f16` transcode that produces it locally). Using the dense base for the whole loop keeps the adapters consistent with the exact weights they will be merged into.

**Merge and quantize are two passes, and that is a design decision, not laziness.** The contracts paragraph under the script says it twice over (`docs/TRAINING.md` §9): merge and `--dtype` are "separate passes BY DESIGN (one combined pass would chain-requantize)", and merge itself "needs a dense f32/f16 base … a quantized base errors; quantize AFTER merging". The underlying refusal lives in the adapter (`src/lora.zig:216-220`), and `docs/TRAINING.md` §9 states the rationale: quantized bases cannot be merged in place, deliberately, even though the f32→K-quant encoders exist — dequantize→merge→re-encode **compounds quantization error**. Every trip through a quantizer costs accuracy; the pipeline is shaped so the final artifact has exactly one quantization in its history: dense base + exact f32 delta, quantized once at the end.

The merge itself is ten lines you have already read the pieces of — this is the one place the rank-r matrix `B·A` is actually materialized (`src/lora.zig:265-274`):

```zig
/// w_data += scale * (B·A), w_data row-major [out, in]. A/B are
/// always contiguous (fresh buffers, updated in place), so the raw
/// matmul applies directly.
fn addScaledDeltaW(self: *const Self, ctx: *ExecContext, w_data: []f32) !void {
    var ba = try ctx.matmul2D(self.b.asRawTensor(), self.a.asRawTensor());
    defer ba.deinit();
    const ba_data = ba.dataConst();
    std.debug.assert(w_data.len == ba_data.len);
    for (w_data, ba_data) |*wi, di| wi.* += self.scale * di;
}
```

And the mutability story around it is a small design gem: `mergeInto` reaches `w_data` through the facade's `data()` gate, "which only grants mutable access to NO-GRAD tensors … exactly the right fence here — a frozen base weight is a constant, and merging into a weight that participates in an autograd graph would silently invalidate recorded forwards" (`src/lora.zig:208-214`). The API that could corrupt training state is unreachable from training state. (`mergeF16` is the f16-base twin: widen to f32, merge, cast back, return a *new* tensor — return-new because the f32 accumulate cannot happen inside f16 storage without a round-trip anyway, `src/lora.zig:242-246`.)

Bookkeeping that makes the loop actually close: the export tool matches adapters to weights by the trainer's names (`layers.<i>.<target>.lora_{a,b}` → `blk.<i>.*.weight`); the checkpoint stores A and B but **not alpha**, so `--alpha` must repeat the training-time value (finetune's default is 16); transcode policy follows llama-quantize (matrix weights only, norms/1D keep their type, already-quantized sources error instead of chain-requantizing). And the evidence that the loop *is* closed, as the repo documents it: "Proven end-to-end: the merged model answers in the fine-tuned style under llama-cli, and the f16→q4_k export loads in Fucina with top-1 logit parity AND in llama-bench as 'Q4_K - Small'" — with the writer side pinned by Chapter 11's discipline, "a verbatim re-emit of the 449 MiB Q4_K_S GGUF is byte-identical to the original (`cmp` clean)" (`docs/TRAINING.md` §9).

> **ML note** — Merging is a *choice*, not a stage. An unmerged adapter is a few megabytes that composes additively with a frozen base — so one 450 MB base plus ten task adapters is ~500 MB total, and switching tasks is a file swap. That deployment pattern (adapter-per-tenant, adapter-per-task) is a large part of why LoRA won in practice. Merging trades the flexibility away for universality: the merged GGUF works in software that has never heard of LoRA, at zero adapter overhead per token. Fine-tune once, decide per deployment.

One more thing you get for free: the merged, quantized artifact is a completely ordinary GGUF, so *everything* from [Chapter 13](13-inference-tricks.md) applies to your fine-tune unchanged — speculative decoding, batch streams, constrained decoding, KV reuse, `lmserve`. Train with one Zig binary on a laptop; serve in a different project's C++ server, or in this one's. Formats, not frameworks, are the interface.

## 15.8 The gradient-free twin: `es-finetune`

[Chapter 9](09-training-without-gradients.md) promised that evolution strategies would come back once a real LLM was on the table. `zig build es-finetune` (`examples/es_finetune/main.zig`) is that return, and it is built as a controlled experiment: "the gradient-free counterpart of examples/finetune/main.zig … same model loading, same built-in pirate dataset / `--data` JSONL path […], same deterministic `--shuffle` loader, same checkpoint directory layout, and the same `llm.qwen3.train.Trainer` forward for evaluation. What changes is the learning signal" (`examples/es_finetune/main.zig:1-10`). Everything from §15.4-15.6 is reused; `backward()` is simply never called.

Two parameter sets (`--mode`, `examples/es_finetune/main.zig:12-19`):

- **`lora`** (default): ES perturbs only the q/v adapters — B starting at zero exactly like the backprop run. The frozen base may stay quantized, because the noise never touches it.
- **`full`**: perturbs *every resident float weight* of the base model — the ES-at-scale paper's full-parameter setting, with no backward memory at any model size. This needs an f32/f16/bf16 GGUF: Gaussian noise cannot be added to K-quant blocks, and the program rejects a quantized base up front (`QuantizedWeightsUnsupported`) with a transcode hint rather than corrupting it.

Three rewards (`--reward`), because ES accepts any scalar you can compute from a forward pass (`examples/es_finetune/main.zig:21-38`):

- **`rule`** (default): a DeepSeek-R1-style rule reward on *greedy generations* — unigram-F1 against the gold response plus 0.1 for matching the response envelope (starts with `--format-prefix`, ends with `--format-suffix`; the defaults are, of course, `"Ahoy!"` and `"matey."`). No gradient of "the reply matched a format" exists; none is needed. Generation-bound on CPU.
- **`acc`**: a bounded teacher-forced composite, `token_accuracy + 0.1·exp(−mean CE)` — one forward per sample, dense, softly self-stopping. The recommended loss-style reward.
- **`nll`**: raw negative cross-entropy — directly comparable with finetune.zig's loss curve, but unbounded, so long runs invite the outlier pathology Chapter 9 §9.5 dissected; pair with `--norm centered_ranks` and checkpoint selection.

Note the evaluation shape, chosen for LLM scale: "every population member scores the SAME per-iteration sample batch (`--batch`, reference semantics); member evaluation runs in place (perturb -> score -> restore) so one model instance serves the whole population, each forward saturating the worker team" (`examples/es_finetune/main.zig:46-49`). This is the opposite trade from Chapter 9's `es_spirals`, where the parameter set was tiny and each worker thread owned a full replica. With a 0.6B-parameter model you cannot afford population-many replicas; instead the *forward* is parallel and the members are sequential. Same `es.Trainer`, same noise contract — the two evaluation shapes are a memory/parallelism dial, not different algorithms.

The checkpoint directory is the same layout minus one file: there is **no `optimizer.fucina`**, because ES has no optimizer state — `(seed, es_iteration)` regenerate the entire population noise stream, so `trainer_state.json`'s `es_*` fields (`es_sigma`, `es_population`, `es_noise`, `es_antithetic`, `es_iteration`, … — the optional fields of `training_checkpoint.TrainerState`) are the whole resume story (`examples/es_finetune/main.zig:51-54`). Anchored weight decay (`--anchor-decay l1|l2`) composes here too, with the one ordering trap Chapter 9 flagged: on resume, the anchor is captured from the initial weights *before* the checkpoint loads.

Costs and convergence expectations were quoted with their conditions in Chapter 9 §9.8 (`docs/TRAINING.md` §13) and have not improved by restating: treat the demo defaults as a mechanism showcase. What the twin buys pedagogically is the comparison itself — identical data, identical forward, identical checkpoints, two entirely different learning signals.

## 15.9 nanochat: training from scratch, whole

Fine-tuning starts from someone else's pretrained weights. The natural objection — "sure, but you couldn't *build* a model on a CPU" — has a concrete counterexample in the tree: `examples/nanochat`, "a from-scratch CPU port of karpathy/nanochat onto Fucina's tensor/autograd runtime: BPE tokenizer training, GPT pretraining, supervised fine-tuning, bits-per-byte evaluation, and an interactive chat CLI with a calculator tool — the full `runs/runcpu.sh` pipeline, in Zig, on CPU" (`examples/nanochat/README.md:3-6`).

The parity target is stated with the precision you should demand of such claims: "the Python reference (nanochat @ `92d63d4`) running **on CPU in fp32**" (`examples/nanochat/README.md:8`) — not a GPU run, not fp16/bf16 mixed precision; the port matches what the reference itself produces on the same class of hardware. Five subcommands cover the pipeline (`examples/nanochat/README.md:20-28`):

| cmd | what it does |
| --- | --- |
| `tok-train` | train the rustbpe-equivalent BPE tokenizer → `tokenizer.bin` + `token_bytes.bin` |
| `base-train` | pretrain the GPT on framed pretraining docs (grad-accum loop, Muon+AdamW, bpb eval, checkpoint/resume) |
| `sft` | supervised fine-tune a base checkpoint on the task mixture (masked loss, SFT schedule) |
| `eval-bpb` | bits-per-byte over a validation split |
| `chat` | interactive chat / single-prompt generation (KV-cache decode, calculator tool) |

Notice how little of this needed new material. Tokenizer training is Chapter 12's BPE run in reverse (learn the merges instead of applying them). Pretraining is Chapter 8's accumulation loop at scale, driving nanochat's `MuonAdamW` optimizer variant — "Polar-Express orthogonalization + NorMuon variance reduction + cautious weight decay", built by "reusing Fucina's `AdamW` for its six Adam groups and a batched-`bmm` Newton–Schulz for the Muon groups" (`examples/nanochat/README.md:44-46`) — Chapter 8 §8.6's Muon, re-plumbed for a different paper's recipe. The training loop even reproduces the reference's *hyperparameter derivation*: nanochat scales learning rates by model width and batch size from a tuned reference point (`B_REF = 2**19`), scales weight decay, and derives the token horizon from a scaling law — all of which the port recomputes rather than hardcodes (`examples/nanochat/train.zig:1-8, 49-51`). SFT is §15.5's masking (nanochat's `ignore_index = -1` maps onto Fucina's `maxInt(usize)` sentinel, since `crossEntropyExt` has no signed labels — `examples/nanochat/model.zig:39-41`). Checkpoints are the same three-file directory. And the port is **example-local**: "no `src/` changes — everything composes from the public Fucina facade" (`examples/nanochat/README.md:39-40`). A whole training stack for a different architecture, and the library did not have to grow for it.

The CPU-demo model is honest about its size — `Config.d6`, straight from the reference's `runs/runcpu.sh` configuration (`examples/nanochat/model.zig:56-65`):

```zig
/// The CPU-demo config from runs/runcpu.sh (nanochat_dump.py CONFIGS["d6"]).
pub const d6 = Config{
    .sequence_len = 512,
    .vocab_size = 32768,
    .n_layer = 6,
    .n_head = 6,
    .n_kv_head = 6,
    .n_embd = 384,
    .window_pattern = "L",
};
```

This is a small GPT; nobody is pretraining a 7B on a laptop. What matters is that every stage *runs and verifies*: the parity ladder covers "tokenizer encode (token-ID-exact) + trainer (merge list byte-identical to rustbpe); base/SFT dataloader batches byte-identical to the reference; model forward (per-layer intermediates ≤ 1e-5 rel), grad, and KV-cache-vs-full self-consistency; optimizer single-step + 10-step vs reference; base/SFT loss-trace within a drift budget; greedy decode token-exact vs a trained reference checkpoint" (`examples/nanochat/README.md:87-91`).

The parity suites are also a model of how to ship reference-gated tests: the goldens (reference dumps, trained checkpoints, loader fixtures) live *untracked* under `refs/nanochat-goldens/`, the suites are gated behind a `NANOCHAT_PARITY=1` environment variable, and they "skip cleanly when they are absent" (`examples/nanochat/README.md:67-80`) — so `zig build test` stays green for every contributor while the full oracle battery remains one export script away for whoever needs it.

Ports at this parity bar accumulate small, hard-won findings, and nanochat's source records them where they were won. Two examples worth reading in place:

- **Loss normalization under a different batching model.** The reference computes one `F.cross_entropy(..., mean)` over a `(B, T)` batch and backwards `loss/grad_accum`; Fucina's attention/CE kernels are single-sequence, so the port runs each row independently, sums each sequence's CE over non-ignored targets, and divides by the micro-batch's total non-ignored count — algebraically the same mean, documented as reproducing "the reference mean bitwise up to float summation order" (`examples/nanochat/train.zig:10-20`). The same mean-vs-sum care as §15.6's window, forced by a *different* reason.
- **An off-by-one in windowed attention conventions.** Fucina's `.window = W` attends W keys *including* self; the reference's sliding-window mask *excludes* self from its bound — "so every arm returns the reference value + 1" (`examples/nanochat/model.zig:92-100`, the `windowFor` doc comment; the function body reconstructs gpt.py's double-ceiling `short_window` formula). Convention mismatches like this are invisible in loss curves and fatal in parity tests; the comment is the proof that someone checked.

> **ML note** — *Bits per byte* is pretraining's honest yardstick. Perplexity and per-token loss depend on the tokenizer: a tokenizer with bigger tokens makes per-token loss look worse and per-text loss better. bpb normalizes by the *bytes of raw text* instead: sum the per-token losses in nats, divide by ln 2 to get bits, divide by the byte length of what was predicted — `nats / (ln2 · bytes)` (`examples/nanochat/train.zig:293-296`, a port of nanochat's `evaluate_bpb`). Comparable across tokenizers, interpretable as compression: a model at 1.0 bpb is, literally, an 8:1 compressor of its validation text.

Porting war story, because it is too instructive to skip: nanochat's RMSNorm uses `F.rms_norm` with no eps argument, and the port needed *bitwise* forward parity. What eps does torch actually use? The answer was "discovered bitwise" — `torch.finfo(float32).eps = 2^-23` (`examples/nanochat/model.zig:31-33`). When your acceptance bar is bit-exactness, even undocumented library defaults become measurable facts. That bar — and the tiered ladder above it — is [Chapter 16](16-the-craft.md)'s subject.

The pipeline's last stage deserves a sentence, because it is the payoff a from-scratch trainer rarely reaches: `chat` talks to *your own* pretrained-and-SFT'd checkpoint, with KV-cached decode, temperature/top-k sampling, and a ported calculator-tool state machine — the model can emit a tool call, the runtime evaluates it and splices the result back into the stream (`examples/nanochat/chat.zig`, the `use_calculator` port). From `zig build nanochat -Doptimize=ReleaseFast -- chat --init-from <model.safetensors> --tokenizer <tokenizer.bin> -p "The capital of France is" -t 0 --base` (`examples/nanochat/README.md:33-35`) back to raw text shards, every byte of the pipeline is code you can read — and by this point in the course, code you can read *fluently*.

One more connection: nanochat is also the repo's worked example of Chapter 8 §8.10's mixed precision — its transformer matrices train in bf16 as 16-bit leaves consumed as `dot` RHS, embeddings stay f32, AdamW steps through optimizer-owned f32 masters, and its custom Muon persists group-level masters in its own NCMA2 checkpoint frame (`docs/TRAINING.md` §10).

## 15.10 What CPU training is for — and what it is not

Close the chapter with calibrated expectations. Everything below is either quoted from repo docs (with source) or a direct consequence of things quoted earlier.

**What it is for.**

- **Fine-tunes of real models.** The §15.6 demo is ~932 ms/step at rank-8 q+v on an M1 Max (snapshot, `docs/TRAINING.md` §9) — a five-pair style transfer in under a minute, a real few-hundred-sample SFT run in coffee-break time. The base stays quantized in memory; the trainable state is megabytes. This is the workload LoRA was invented for, and CPUs handle it comfortably.
- **Small models from scratch.** nanochat's d6-class GPT: pretraining, SFT, eval, chat, all verified against the reference on the same hardware class. "Small" is doing real work in that sentence — but small models trained on your own data are a legitimate, underrated tool.
- **Research loops that prize determinism.** Everything you have watched the stack insist on — bit-exact resume, seed-regenerated dropout/noise/permutations, loss-side normalization pinned by tests, checkpoint sentinels — makes a CPU run *reproducible to the bit* across thread counts and (for ES, even across CPU/CUDA) devices. When you are debugging a training idea, "the same command produces the same floats" is worth more than a 10× speedup you cannot trust.
- **Places gradients cannot go.** ES fine-tuning of quantized adapters, full-parameter ES with no backward memory, ternary genomes trained through their serving kernels (Chapter 14) — the gradient-free path runs on exactly this stack.

**What it is not.**

- **Not pretraining at modern scale.** No amount of Zig makes a laptop competitive with a GPU cluster on dense FLOPs. The optimizer table alone tells you where the walls are: Muon's Newton–Schulz costs "~395 GFLOP per block in f32 at 5 iterations" (`docs/TRAINING.md` §11), and a full Qwen3-0.6B AdamW step is "~200 ms" of optimizer time before any forward/backward (same source, M1 Max snapshot) — fine for a fine-tune, prohibitive multiplied by pretraining's step counts.
- **Not a place to guess throughput.** The repo quotes measured numbers with dates and machines, and so does this course; anything not measured here is not claimed here.
- **Not dogma.** Where a GPU genuinely helps, the seam exists: on `-Dgpu=metal` builds, big f32 GEMMs — "exactly the training forward/VJP shapes" — offload above a measured crossover (m·n·k ≥ 2^30), with "the 2048×1024×1024 train shape … +83%"; the quantized-frozen LoRA fine-tune at seq ≤ 256 has no qualifying GEMMs and is measurably unchanged (`docs/TRAINING.md` §11). CPU-first, not CPU-only — the same stance as inference.

The chapter's three roads, side by side:

| road | trainable parameters | learning signal | base weights | demo |
| --- | --- | --- | --- | --- |
| LoRA + backprop (§15.2–15.7) | adapters only (≈1.15M at the demo defaults) | exact gradients | frozen; may stay quantized | `zig build finetune` |
| Evolution strategies (§15.8) | adapters, or every resident float | scalar rewards, forward passes only | frozen (lora) / perturbed dense (full) | `zig build es-finetune` |
| From scratch (§15.9) | everything, at d6-class scale | exact gradients | none — you make them | `zig build nanochat` |

Two of those roads — the fine-tunes — also belong to a wider menu. Counting what other chapters own, the repo has **four ways to specialize a model after pretraining**, all CPU-first, each starting from a GGUF the serving stack already runs and ending in an artifact it still runs. Two train the weights. One re-expresses the weights without training anything. The fourth trains no weight at all — it post-trains the *context* the model reads.

| road | what changes | learning signal | artifact | demo |
| --- | --- | --- | --- | --- |
| LoRA SFT (§15.2–15.7; `docs/TRAINING.md` §9) | adapter weights beside the frozen base | exact gradients | merged, re-quantized GGUF | `zig build finetune` |
| Evolution strategies (§15.8; [Chapter 9](09-training-without-gradients.md) §9.8; `docs/TRAINING.md` §13) | adapters, or every resident float | scalar rewards, forward passes only | same checkpoint layout and export, minus `optimizer.fucina` | `zig build es-finetune` |
| PTQTP ([Chapter 14](14-the-low-bit-frontier.md) §14.6–14.8; `docs/PTQTP.md`) | the weights' *representation* | none — data-free, solves a decomposition | ternary GGUF | `zig build ptqtp-qwen3`, `export-gguf --ptqtp` |
| Cartridges (`docs/CARTRIDGES.md`; `docs/REFERENCE.md` §13.10) | the *context* — a trained KV prefix; weights untouched | self-study distillation, teacher top-k CE | cartridge safetensors beside the unchanged base GGUF | `zig build cartridge` |

The menu composes through seams this chapter already crossed. §15.4's `dotLinear` carries a `.ptqtp` arm — trit planes are just more frozen constant RHS, so LoRA fine-tuning runs over a PTQTP model unchanged — and the cartridge trainer *is* this chapter's trainer with every adapter switched off (`Trainer(.{ .q = false, .v = false })`, `docs/REFERENCE.md` §13.10), leaving the trained KV rows as the only parameters.

The deeper takeaway is not about hardware at all. The reason this chapter could be short on new machinery is that the training stack was built out of *contracts* — lifetime rules, seed contracts, checkpoint sentinels, parity oracles — and contracts compose. A quantized frozen transformer, a LoRA adapter, an accumulation window, a resumable loader, and a GGUF exporter snap together because each one states exactly what it needs and refuses loudly otherwise. How that discipline is practiced across the whole repository is the final chapter's story.

Loose ends, and where they live:

- **Serving the fine-tune properly** — chat templates, KV reuse across requests, and the OpenAI-compatible `lmserve` — is [Chapter 13](13-inference-tricks.md); your exported GGUF drops straight into all of it.
- **The export tool's other faces** — verbatim re-emit, dtype transcoding, and shard-streaming quantization of models bigger than RAM — belong to [Chapter 11](11-model-files-and-quantization.md) and `docs/PTQTP.md`.
- **Training *ternary* models** — straight-through estimators and ES trit-flips over packed TQ2_0 genomes — was [Chapter 14](14-the-low-bit-frontier.md)'s story; it composes with this chapter's checkpoint directories unchanged.
- **Post-training the *context*** — distilling a corpus into a reusable trained KV prefix while the weights stay untouched, the menu's fourth road — has its design record in `docs/CARTRIDGES.md` (`zig build cartridge`); the artifact rides [Chapter 13](13-inference-tricks.md)'s serving stack, KV persistence included.
- **The verification methodology in full** — why parity ladders look the way they do, and how a port earns its claims — is [Chapter 16](16-the-craft.md).

## What you now know

- Full fine-tuning of even a 0.6B model costs gigabytes of optimizer state and f32 gradients — and is structurally impossible on K-quant weights, which are constants with no `GradState` at all.
- LoRA parameterizes the weight *update* as a low-rank product: `y = W·x + (α/r)·B·(A·x)`, with A random, B zero (so training starts exactly at the base model), and α/r scaling so alpha transfers across ranks. At rank 8 on q+v, Qwen3-0.6B trains ≈1.15M parameters — about 0.19% of the model — and the forward never materializes `B·A`.
- Fucina's `lora.Adapter(in_tag, out_tag)` is a comptime-validated type over the reserved `.lora_r` axis; its `delta` is four facade ops under the standard composite-op scope contract.
- `llm.qwen3.train.Trainer(targets)` selects adapted projections at comptime and routes frozen weights through the differentiable frozen-RHS `dot`: gradients flow to f32 activations only, weight memory stays quantized. `loss` demands an open exec scope (`Error.ExecScopeRequired`) and masks prompts with `ignore_index`.
- SFT data = render template → encode prompt and response *separately* (concatenating text first would move BPE boundaries) → next-token shift with prompt masking. The loader's shuffled order is a pure function of (seed, epoch) — a checkpoint contract — so resume continues the stream.
- The training checkpoint is a directory — `adapters.safetensors` + `optimizer.fucina` + `trainer_state.json` written last as the commit sentinel — and the demo fine-tune measurably closes: loss 5.77 → 2e-4 in 30 steps at ~932 ms/step (M1 Max snapshot, `docs/TRAINING.md` §9).
- The loop back to the world is four commands: fine-tune → merge into a *dense* f16 base (downloaded or transcoded — weights are not in the repo) → quantize the merged result → serve, in Fucina or llama.cpp. Merge and quantize are deliberately separate passes, and quantized bases refuse in-place merging, because every extra quantization pass compounds error.
- `es-finetune` runs the same fine-tune with no gradients: LoRA-only or full-parameter, rule-based R1-style or loss-based rewards, same data plumbing and checkpoint layout (minus `optimizer.fucina` — seeds regenerate everything).
- Activation checkpointing composes with the LoRA trainer (`--checkpoint-layers`): digit-identical losses at ~+8.5% step time, possible only because dropout regenerates from stored seeds.
- The trainer's generation path (`evalLastLogits`) is deliberately cache-less — a training diagnostic, not a serving path; the exported GGUF gets Chapter 12's cached decode.
- nanochat proves the from-scratch case: tokenizer training, GPT pretraining, SFT, bits-per-byte eval, and chat, ported whole as an example-local program and validated against the Python reference on CPU in fp32 — token-ID-exact tokenizer, byte-identical loaders, per-layer forward parity, optimizer-step parity, greedy-decode-exact.
- CPU training is for fine-tunes, small models, and determinism-first research loops; it is not pretraining at scale, and the repo never pretends otherwise.

## Explore the source

- `src/lora.zig` — the whole adapter in 314 lines; the module doc is the contract summary, `mergeInto`'s doc comment is a masterclass in explaining a refusal.
- `src/llm/qwen3/train.zig` — the trainer: comptime `Targets`, `dotLinear`'s frozen-RHS routing, deterministic dropout seeding, per-layer checkpointing.
- `src/llm/data.zig` — SftText/encodePair/Loader; read the `Loader` doc comment for the checkpoint-contract phrasing, and the module doc for what was *deliberately deferred*.
- `examples/finetune/main.zig` — the walkthrough; also the home of `--verify-grads` (`verifyGrads` is a self-contained lesson in adversarial gradient checking).
- `src/llm/qwen3/train_tests.zig` / `train_golden_tests.zig` — the finite-difference and torch-golden angles of §15.6's verification battery, readable as ordinary tests; `src/llm/data_tests.zig` pins the loader permutation.
- `examples/es_finetune/main.zig` — the gradient-free twin; its module doc is the best two-page comparison of backprop-vs-ES trade-offs in the tree.
- `docs/TRAINING.md` §9 — LoRA, verification, and the export loop, with every number this chapter quoted.
- `docs/REFERENCE.md` §14.2.1 — the machine-verified entry-point map for the trainer surface summarized in §15.4.
- `src/training_checkpoint.zig` — the directory protocol: `beginSave`, `writeFileAtomic`, the sentinel-last `TrainerState`, and the full list of optional resume fields (LoRA, data-loader, and ES alike).
- `tools/export_gguf.zig` — merge and transcode; Chapter 11's writers earning their keep.
- `docs/RUNNING-MODELS.md` — where every model file comes from, including the f16-transcode note this chapter's loop depends on; the copy-paste "fine-tune → merge → re-quantize → serve" script is `examples/finetune/README.md`.
- `examples/nanochat/` — README first, then `train.zig`'s module doc (the loss-normalization parity note), then `model.zig` (the `rms_eps` war story at line 33).

## Exercises

1. **(Easy)** In the course LoRA snippet of §15.2, compute the parameter counts for rank 4, 8, 16, and 32 on a `[2048 × 1024]` weight, and find the rank at which the adapter stops being smaller than 10% of the weight. Then check your formula against the repo's 1.15M figure for the full q+v/28-layer configuration.
2. **(Easy)** Run the walkthrough: download the Q4_K_S GGUF per `docs/RUNNING-MODELS.md`, then `zig build finetune -Doptimize=ReleaseFast -- --steps 30`. Confirm the BEFORE generation is pirate-free and the AFTER is not. Then rerun with `--steps 15 --save /tmp/half`, resume with `--load /tmp/half --steps 15`, and compare the final loss against the single 30-step run.
3. **(Medium)** Write a 20-pair JSONL dataset in some other distinctive style and fine-tune with `--data yours.jsonl --shuffle --accum-steps 4`. Watch the reported loss: with accumulation, it is the *true mean over the window's supervised tokens* — re-read `examples/finetune/main.zig:340-375` and explain why `.mean` reduction plus a `1/N` scale would have mis-weighted your samples if their responses differ in length.
4. **(Medium)** Close the full loop of §15.7 on your machine: fine-tune over the f16 base, merge, quantize to q4_k, and load the result in both `zig build qwen3 -- … --chat` and (if you have it) `llama-cli`. Then do the *wrong* thing on purpose: try `export-gguf --adapters` against the Q4_K_S base and read the error you get. Explain, in one sentence, why the tool refuses instead of dequantizing for you.
5. **(Hard)** Reproduce a slice of the backprop-vs-ES comparison the twin examples were built for: run `finetune` and `es-finetune --reward acc --mode lora` on the same dataset and seed, checkpointing both every few steps, and plot loss (for ES, evaluate the checkpoints with `--reward nll`'s metric) against *wall-clock time* rather than steps. Chapter 9 §9.8's cost sheet predicts the shape of your plot; check whether the anchor-decay flag changes where the ES curve peaks.

---

[Previous: The low-bit frontier](14-the-low-bit-frontier.md) · [Next: The craft — how a library like this is actually built](16-the-craft.md)
