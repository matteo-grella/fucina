# es-spirals — evolution-strategies training from scratch

The gradient-free counterpart of [`../spirals`](../spirals/README.md), and
the from-random-init acceptance test of `fucina.es`: the same
two-hidden-layer tanh MLP (hidden 64, 194 points), same data generator, but
no backward pass and no optimizer — reward is `-CE` on the full batch with
z-score shaping and mirrored (antithetic) sampling on the ES-at-scale
update (root `examples/es_spirals.zig`).

It is also the member-parallel showcase: `evaluateMembers` fans the
population over worker threads, each owning a full MLP replica and its own
`ExecContext`; `materializeMember` writes `theta + sigma*eps` into the
replica and only scalar rewards come back. The defaults reach 100% accuracy
within ~15k iterations (~75 s ReleaseFast on an M1 Max, per the source
header).

**Self-verifying:** exits nonzero unless the trained network reaches
`--target` accuracy (default 0.90) on the training set — chance is 0.50.

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
(docs/TRAINING.md, section 13).

## Shared knobs

The ReleaseFast/`-Dcpu` build discipline and global thread/BLAS knobs are
shared machinery — see
[docs/RUNNING-MODELS.md](../../docs/RUNNING-MODELS.md). This demo needs no
model weights.
