# Recipe: PTQTP-quantize a model and run it (resident or streamed)

The end-to-end walkthrough: take a GGUF, quantize its weight matrices —
including every routed MoE expert — into PTQTP trit planes, and run the
result with the ordinary runners, resident or streamed from disk. Every
number below was measured on this exact sequence (M1 Max 64 GB, source
model on an external USB-3 SSD, 2026-07-12). The method itself, the
on-disk format, and the full quality ladder live in `PTQTP.md`; this file
is just the recipe.

## 0. Pick the source

Any GGUF works as input. For best quality use the **highest-precision
release available** (f16 / bf16 / q8_0): quantizing from an
already-K-quantized file (as in the example below) stacks its error on
top of PTQTP's. Quantized sources are dequantized tensor-by-tensor first
— deliberate, and documented as graceful degradation in `PTQTP.md`.

## 1. Preview the plan (optional, fast)

```sh
zig build export-gguf -Doptimize=ReleaseFast -- \
  --from-gguf models/Qwen3-30B-A3B-Instruct-2507-Q5_K_M.gguf \
  --ptqtp=2 --ptqtp-include ffn_gate_exps,ffn_up_exps,ffn_down_exps \
  --dry-run
```

Prints one line per tensor (shape, source dtype → target, bytes
before/after), the totals, and the peak buffering estimate. No output
file is written.

## 2. Quantize

```sh
zig build export-gguf -Doptimize=ReleaseFast -- \
  --from-gguf models/Qwen3-30B-A3B-Instruct-2507-Q5_K_M.gguf \
  --out       models/Qwen3-30B-A3B-ptqtp2-experts.gguf \
  --ptqtp=2 --ptqtp-include ffn_gate_exps,ffn_up_exps,ffn_down_exps
```

Knobs:

- `--ptqtp=K` — trit planes per matrix. K=1 ≈ 2.06 bpw (maximum
  compression), K=2 ≈ 4.1 bpw (the sweet spot; reconstruction rel_err
  ≈ 0.17), K=3 ≈ 6.2 bpw (near-parity; rel_err ≈ 0.067). `PTQTP.md`
  §Measured has the NLL ladder.
- `--ptqtp-include SUB[,SUB]` — name-substring filter. The value shown
  quantizes **only the routed expert stacks**; each expert's matrix is
  quantized independently and persists as plane-major
  `<name>.ptqtp0/1/…` siblings with the base 3D shape. Omit the flag to
  also decorate attention/dense/shared-expert matrices under the default
  policy (embeddings, norms, router, and the output head always stay in
  source precision). `--ptqtp-exclude` subtracts from either.
- Re-running on an already-decorated file quantizes nothing (idempotent).

Cost, measured on the 30B (48 layers × 3 stacks × 128 experts = the full
~28 B-parameter expert mass): **8.5 minutes** wall, **105.8 MiB** peak
tensor working set — the tool streams one tensor at a time, so source
size does not matter: a 300 GB model quantizes on a 64 GB machine. Every
expert converged (0 unconverged trit groups out of 786 432 per stack;
per-stack `rel_err mean/max` is printed as it goes).

Result for this example: file 20.7 → **15.3 GiB** (quantized linears
19.6 → 14.3 GiB). From an f16 source the expert mass shrinks ~4× at K=2.

## 3. Run it

The output is a normal GGUF; the family runner picks up the plane
tensors automatically (pair-detection at load — nothing to configure).

```sh
# resident
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-30B-A3B-ptqtp2-experts.gguf \
  --prompt "The capital of France is" --gen 32

# streamed: only dense weights stay resident; routed experts page from
# disk through the pinned-set + LRU tier under the given budget
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-30B-A3B-ptqtp2-experts.gguf \
  --prompt "The capital of France is" --gen 32 --moe-stream --moe-cache-mb=6144
```

Measured on the file produced above, both modes: **byte-identical
output**, coherent and factual ("… Paris. What is the capital of Spain?
The capital of Spain is Madrid. Madrid is the largest city in Spain and
serves as the country's political, economic …"). The streamed run hit
80.1 % in the expert cache at a 6 GiB budget (6.94 GB read for prefill +
32 tokens); all `--moe-stream` companions (`--moe-pin-mb`, `--moe-pilot`,
the `.experts` learning-cache sidecar, `--kv-save`) apply unchanged —
see `RUNNING-MODELS.md` §Streaming.

## Notes

- **Bit-exactness contract**: a PTQTP expert computes the same
  sum-of-plane-dots the dense PTQTP linear does, and the streamed path is
  bit-identical to resident (both pinned by tests in
  `src/exec/expert_store_tests.zig`).
- **Mixed files are fine**: quantize one layer, a subset of projections,
  or everything; decorated and undecorated tensors serve side by side.
- **K per use case**: K=1 for maximum-compression experiments, K=2 as
  the daily-driver size/quality point, K=3 when the goal is parity with
  the source at ~1.9× decode speed (see `PTQTP.md` and `TERNARY.md` for
  the kernel story).
