# es-spirals — evolution-strategies training from scratch

The gradient-free counterpart of [`../spirals`](../spirals/README.md), and
the from-random-init acceptance test of `fucina.es`: the same
two-hidden-layer tanh MLP (hidden 64, 194 points), same data generator, but
no backward pass and no optimizer — reward is `-CE` on the full batch with
z-score shaping and mirrored (antithetic) sampling on the ES-at-scale
update ([`main.zig`](main.zig)).

It is also the member-parallel showcase: `evaluateMembers` fans the
population over worker threads, each owning a full MLP replica and its own
`ExecContext`; `materializeMember` writes `theta + sigma*eps` into the
replica and only scalar rewards come back. The defaults reach 100% accuracy
within ~15k iterations (~75 s ReleaseFast on an M1 Max, per the source
header).

**Self-verifying:** exits nonzero unless the trained network reaches
`--target` accuracy (default 0.90) on the training set — chance is 0.50.

The same trainer fine-tunes a real LLM gradient-free:
[`../es_finetune`](../es_finetune/README.md).

```sh
zig build es-spirals -Doptimize=ReleaseFast
# all knobs:
zig build es-spirals -Doptimize=ReleaseFast -- [--iterations N] [--population N] \
    [--sigma F] [--alpha F] [--workers N] [--norm z_score|centered_ranks] [--seed N] [--target F]
```

Defaults: `--iterations 15000`, `--population 128` (antithetic when even),
`--sigma 0.1`, `--alpha 0.1`, `--workers 4`, `--norm z_score`, `--seed 42`,
`--target 0.9`. z-score is the deliberate default: with a bounded
well-behaved reward its magnitude information converges where
`centered_ranks` stalls — pick the shaping by reward regime
([docs/TRAINING.md](../../docs/TRAINING.md) §13).

## Quick sanity check

The run is deterministic per `--seed`, so a shorter run is a prefix of the
default one; the default seed sits at 100% train accuracy from ~iteration
6500, so about half the default iteration budget already clears the 0.90
gate:

```sh
zig build es-spirals -Doptimize=ReleaseFast -- --iterations 7000
```

Expect a `before:` accuracy near chance, an `iter … accuracy … ce …`
progress line every 500 iterations, and a final
`PASS: gradient-free from-scratch training reached 100.0% (target 90.0%)`
with exit code 0.

## Shared knobs

The ReleaseFast/`-Dcpu` build discipline and global thread/BLAS knobs are
shared machinery — see
[docs/RUNNING-MODELS.md](../../docs/RUNNING-MODELS.md). This demo needs no
model weights.
