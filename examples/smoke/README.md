# smoke — minimal tensor/autograd sanity demo

The smallest complete Fucina program (root `examples/smoke.zig`): two
tracked variables — `x = [2, 3]` (shape 1x2) and `w = [4, 5]` (shape 2x1) —
a `dot` contraction, a `sumAll` loss, one `backward()`, and the gradients
read back out. It exercises `ExecContext`, tagged-dimension `Tensor`s,
the autograd graph, and gradient retrieval in under 50 lines.

No weights, no flags. Note the build step is named **`run`**, not `smoke`:

```sh
zig build run
```

Expected output (loss = 2*4 + 3*5; grad_x = w, grad_w = x):

```
loss=23
grad_x=[4, 5]
grad_w=[2, 3]
```

## Shared knobs

The ReleaseFast/`-Dcpu` build discipline and global thread/BLAS knobs are
documented once in [docs/RUNNING-MODELS.md](../../docs/RUNNING-MODELS.md);
this demo is instant either way and takes no arguments.
