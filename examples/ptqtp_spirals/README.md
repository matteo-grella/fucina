# ptqtp-spirals — post-training trit-plane quantization, self-verifying

Two-spirals PTQTP acceptance demo (root `examples/ptqtp_spirals.zig`):
train a float tanh MLP with AdamW
(`2 -> [dense f32] -> 256 -> tanh -> [256x256] -> tanh -> [head 256x2]`),
then post-training-quantize the two packable layers to dual trit-planes
(`fucina.ptqtp`, arXiv:2509.16989) and measure what survives — no
retraining, no calibration data, weights only. See
[docs/PTQTP.md](../../docs/PTQTP.md) for the method.

The report compares, on the same raw-kernel eval harness:

| variant | what it is |
| --- | --- |
| `float` | the trained dense weights (the ceiling) |
| `absmean-b1.58` | blind round-clip, one plane (the zero-optimization floor) |
| `ptqtp-k1` | one plane, ridge scales + 3-way search |
| `ptqtp-k2` | the paper's dual planes, 9-way search |

each ternary variant on both forwards: the exact mul-free f32 path
(isolates weight-approximation error) and the deployed int8 path (Q8_K
activations x packed crumbs — adds activation quantization). It also prints
packed reconstruction errors, plane sparsity, and the reference-path
G=128 vs G=256 fidelity delta.

**Self-verifying:** exits nonzero unless the float model reaches 0.99
training accuracy (else inconclusive) and the dual-plane int8-path accuracy
reaches `--target` (default 0.95; chance 0.50).

```sh
zig build ptqtp-spirals -Doptimize=ReleaseFast
# all knobs:
zig build ptqtp-spirals -Doptimize=ReleaseFast -- [--steps N] [--seed N] [--target F] [--lr F]
```

Defaults: `--steps 3000` (stops early at 100% train accuracy), `--seed 42`,
`--target 0.95`, `--lr 0.02`.

## Shared knobs

The ReleaseFast/`-Dcpu` build discipline and global thread/BLAS knobs are
shared machinery — see
[docs/RUNNING-MODELS.md](../../docs/RUNNING-MODELS.md). No model weights
needed; for PTQTP on a real LLM see
[`../ptqtp_qwen3`](../ptqtp_qwen3/README.md).
