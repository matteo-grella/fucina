# es-ternary-spirals — ternary-native ES over packed TQ2_0

[`../es_spirals`](../es_spirals/README.md)'s BitNet-class sibling and the
flagship demo of `es.Trainer`'s ternary slots: the hidden layer and head
are packed TQ2_0 genomes (2-bit {-1,0,+1} crumbs) registered with
`addTernaryParam`, so the **training state is the inference model** — no
latent float weights exist for the ternary layers at any point, and every
forward (member evaluations and the final verification) runs on the real
int8 kernels: Q8_K activation rows times the packed 2-bit blocks
([`main.zig`](main.zig)).

Architecture (TQ2_0 needs contract dims that are multiples of 256):
`2 -> [dense f32] -> 256 -> tanh -> [ternary 256x256] -> tanh ->
[ternary head 256x2] -> logits`, f32 biases throughout. The float first
layer trains through the ordinary gaussian ES slots while the two genomes
evolve by sparse trit flips + vote-and-threshold updates — one shared
reward pipeline, antithetic sampling on both kinds. Each genome's fp16
block scale `d` is fixed at init (`--dscale` multiplies the
variance-matched default; d is the ternary learning-rate analog).

**Self-verifying:** exits nonzero unless accuracy reaches `--target`
(default 0.95; chance 0.50). Defaults cross 95% around iteration ~8-11k and
stop early at 100% (~2 minutes ReleaseFast on an M1-class CPU, per the
source header).

```sh
zig build es-ternary-spirals -Doptimize=ReleaseFast
# all knobs:
zig build es-ternary-spirals -Doptimize=ReleaseFast -- [--iterations N] [--population N] \
    [--sigma F] [--alpha F] [--flip-rate F] [--update-fraction F] [--update-decay F] \
    [--workers N] [--norm z_score|centered_ranks] [--reward acc|nll] [--dscale F] [--seed N] [--target F]
```

Defaults: `--iterations 20000`, `--population 128`, `--sigma 0.05`,
`--alpha 0.075`, `--flip-rate 0.002`, `--update-fraction 0.001`,
`--update-decay 0`, `--workers 8`, `--norm z_score`, `--reward nll`
(raw `-CE`; `acc` is the bounded composite kept for contrast — it stalls
near 74% here), `--dscale 3`, `--seed 42`, `--target 0.95`.

## Shared knobs

The ReleaseFast/`-Dcpu` build discipline and global thread/BLAS knobs are
shared machinery — see
[docs/RUNNING-MODELS.md](../../docs/RUNNING-MODELS.md). No model weights
needed; the TQ2_0 dtype itself is documented in
[docs/TERNARY.md](../../docs/TERNARY.md).
