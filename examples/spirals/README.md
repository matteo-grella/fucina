# spirals — optimizers, checkpointing, resume, inference

Two-spirals classification (the classic Lang & Witbrock task, 400 points)
with a 2-64-64-2 tanh MLP, trained full-batch for 2000 steps — once per
optimizer: SGD (Nesterov momentum), AdamW, Muon, APOLLO, and APOLLO-Mini,
plus a final `adamw-groups` run that composes param groups (decay /
no-decay), a warmup-cosine lr schedule, and global-norm gradient clipping —
the standard LLM training recipe in miniature (root `examples/spirals.zig`).

For every optimizer the demo:

1. trains, checkpointing model + optimizer state at the halfway step
   (under `/tmp/fucina-spirals-*`, deleted at the end);
2. restores that checkpoint into a **fresh** model + optimizer, retrains the
   second half, and demands bit-identical final parameters — the process
   exits nonzero on any non-bit-exact resume;
3. reloads the final weights into a gradient-free constant model (the
   inference path) and reports its accuracy.

Muon/APOLLO routing: hidden matrices take the matrix path, the classifier
head goes to the AdamW fallback, biases auto-route to the fallback.

No weights, no flags:

```sh
zig build spirals -Doptimize=ReleaseFast
```

Each `[name]` block prints final loss/accuracy, the resume
`max |delta param|` (must be `bit-exact`), and the from-checkpoint
inference accuracy.

## Shared knobs

The ReleaseFast/`-Dcpu` build discipline and global thread/BLAS knobs are
shared machinery — see
[docs/RUNNING-MODELS.md](../../docs/RUNNING-MODELS.md). This demo takes no
arguments.
