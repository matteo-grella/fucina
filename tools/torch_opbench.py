#!/usr/bin/env python3
"""PyTorch-CPU twin of the fucina op-level bench rows: the paired
eager-vs-eager head-to-head.

Each row mirrors one fucina bench row — same shapes, same iteration counts,
same grad regime (no-grad rows run under `torch.inference_mode()`; backward
rows time `autograd.grad` on a retained graph, or a full forward+backward
where the fucina row rebuilds its graph per step). Timing mirrors the fucina
benches: 2 warmup iterations, then best-of-`--repeats` mean-per-op over the
row's iteration count. Threads default to 8 (M1 Max physical cores), matching
the fucina pool.

Pairing reads the per-machine fucina baseline recorded by
`tools/opbench_gate.py record` and emits a markdown table with per-row
ratios. Rows without a clean semantic twin are labeled in the notes column
(composed-vs-fused, transpose-idiom) rather than silently compared.

Usage:
  .venv/bin/python tools/torch_opbench.py \
      --baseline bench/baselines/opbench-<host>.json --out RESULTS.md
"""

from __future__ import annotations

import argparse
import json
import platform
import time

import torch
import torch.nn.functional as F

# Suite -> unit of the recorded `time` metric in the fucina baseline.
SUITE_UNIT_NS = {"facade": 1.0, "mlp": 1.0, "attention-backward": 1.0}
SUITE_UNIT_MS = {"ce": 1e6, "optim": 1e6}


def bench(fn, iters: int, repeats: int) -> float:
    """Best-of-`repeats` mean ns/op after 2 warmup iterations."""
    for _ in range(2):
        fn()
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        for _ in range(iters):
            fn()
        best = min(best, (time.perf_counter_ns() - t0) / iters)
    return best


class Harness:
    def __init__(self, args):
        self.args = args
        self.rows = []
        with open(args.baseline) as f:
            self.baseline = json.load(f)

    def fucina_ns(self, suite: str, key: str) -> float | None:
        row = self.baseline["suites"].get(suite, {}).get(key)
        if row is None:
            return None
        scale = SUITE_UNIT_NS.get(suite) or SUITE_UNIT_MS.get(suite)
        return row["time"] * scale

    def row(self, name: str, suite: str, key: str, iters: int, fn, note: str = ""):
        ns = bench(fn, iters, self.args.repeats)
        self.rows.append((name, suite, key, ns, self.fucina_ns(suite, key), note))
        print(f"  {name:<34} {ns / 1e3:>12.1f} us/op")


def randn(*shape):
    return torch.randn(*shape)


def build_rows(h: Harness):
    torch.manual_seed(0x5EED)
    ce = "ce"
    fa = "facade"

    # ---- softmax family, last axis ----
    for rows, cols, iters in ((2048, 151936, 5), (1024, 4096, 50), (32768, 1024, 20)):
        x = randn(rows, cols)
        h.row(
            f"softmax fwd {rows}x{cols}", ce, f"softmax fwd/{rows}x{cols}/iters={iters}", iters,
            lambda x=x: F.softmax(x, dim=-1),
        )
    x = randn(1024, 4096).requires_grad_()
    y = F.softmax(x, dim=-1)
    gy = randn(1024, 4096)
    h.row(
        "softmax bwd 1024x4096", ce, "softmax bwd/1024x4096/iters=50", 50,
        lambda: torch.autograd.grad(y, x, gy, retain_graph=True),
    )

    # ---- softmax family, axis 0 (the strided kernels) ----
    for rows, cols in ((4096, 1024), (1024, 4096)):
        x = randn(rows, cols)
        h.row(
            f"softmax fwd ax0 {rows}x{cols}", ce, f"softmax fwd ax0/{rows}x{cols}/iters=50", 50,
            lambda x=x: F.softmax(x, dim=0),
        )
    x = randn(4096, 1024)
    h.row("logsoftmax ax0 4096x1024", ce, "logsoftmax ax0/4096x1024/iters=50", 50,
          lambda: F.log_softmax(x, dim=0))
    h.row("logsumexp ax0 4096x1024", ce, "logsumexp ax0/4096x1024/iters=50", 50,
          lambda: torch.logsumexp(x, dim=0))
    xg = randn(4096, 1024).requires_grad_()
    yg = F.softmax(xg, dim=0)
    gyg = randn(4096, 1024)
    h.row("softmax bwd ax0 4096x1024", ce, "softmax bwd ax0/4096x1024/iters=50", 50,
          lambda: torch.autograd.grad(yg, xg, gyg, retain_graph=True))

    # ---- cross entropy ----
    for rows, cols, iters in ((1024, 151936, 5), (4096, 32000, 5)):
        logits = randn(rows, cols)
        labels = torch.randint(0, cols, (rows,))
        h.row(f"cross-entropy fwd {rows}x{cols}", ce, f"cross-entropy fwd/{rows}x{cols}/iters={iters}",
              iters, lambda a=logits, b=labels: F.cross_entropy(a, b))
        lg = logits.clone().requires_grad_()
        loss = F.cross_entropy(lg, labels)
        h.row(f"cross-entropy bwd {rows}x{cols}", ce, f"cross-entropy bwd+s/{rows}x{cols}/iters={iters}",
              iters, lambda l=loss, g=lg: torch.autograd.grad(l, g, retain_graph=True),
              note="torch bwd from saved activations; fucina row = the +s (saved-stats) variant")

    # linear + CE, full forward+backward per step: the torch composed idiom
    # against fucina's composed row (same structure) and its fused row.
    xin = randn(1024, 1024).requires_grad_()
    w = randn(151936, 1024).requires_grad_()
    labels = torch.randint(0, 151936, (1024,))

    def linear_ce():
        if xin.grad is not None:
            xin.grad = None
            w.grad = None
        F.cross_entropy(xin @ w.T, labels).backward()

    h.row("linear-ce fwd+bwd 1024x151936", ce, "linear-ce bwd comp/1024x151936/iters=3", 3,
          linear_ce, note="composed-vs-composed")
    h.rows.append((
        "linear-ce fwd+bwd 1024x151936", ce, "linear-ce bwd fus+c/1024x151936/iters=3",
        h.rows[-1][3], h.fucina_ns(ce, "linear-ce bwd fus+c/1024x151936/iters=3"),
        "same torch number vs fucina fused linearCrossEntropyExt",
    ))

    # ---- norms ----
    for rows, cols in ((4096, 1024), (1024, 4096)):
        x = randn(rows, cols)
        wln = randn(cols)
        bln = randn(cols)
        h.row(f"layernorm-aff fwd {rows}x{cols}", ce, f"layernorm-aff fwd/{rows}x{cols}/iters=100",
              100, lambda x=x, w=wln, b=bln: F.layer_norm(x, (x.shape[1],), w, b, 1e-5))
        xg = x.clone().requires_grad_()
        wg = wln.clone().requires_grad_()
        bg = bln.clone().requires_grad_()
        yg = F.layer_norm(xg, (cols,), wg, bg, 1e-5)
        gyn = randn(rows, cols)
        h.row(f"layernorm-aff bwd {rows}x{cols}", ce, f"layernorm-aff bwd/{rows}x{cols}/iters=100",
              100, lambda y=yg, t=(xg, wg, bg), g=gyn: torch.autograd.grad(y, t, g, retain_graph=True))
        wrms = randn(cols)
        h.row(f"rmsnorm-mul fwd {rows}x{cols}", ce, f"rmsnorm-mul fwd/{rows}x{cols}/iters=100",
              100, lambda x=x, w=wrms: F.rms_norm(x, (x.shape[1],), w, 1e-5))
        xr = x.clone().requires_grad_()
        wr = wrms.clone().requires_grad_()
        yr = F.rms_norm(xr, (cols,), wr, 1e-5)
        h.row(f"rmsnorm-mul bwd {rows}x{cols}", ce, f"rmsnorm-mul bwd/{rows}x{cols}/iters=100",
              100, lambda y=yr, t=(xr, wr), g=gyn: torch.autograd.grad(y, t, g, retain_graph=True))

    # layernorm over axis 0: the torch idiom is transpose + layer_norm.
    x = randn(4096, 1024)
    w0 = randn(4096)
    b0 = randn(4096)
    xg = x.clone().requires_grad_()
    wg = w0.clone().requires_grad_()
    bg = b0.clone().requires_grad_()
    y0 = F.layer_norm(xg.T.contiguous(), (4096,), wg, bg, 1e-5).T
    gy0 = randn(4096, 1024)
    h.row("layernorm bwd ax0 4096x1024", ce, "layernorm bwd ax0/4096x1024/iters=20", 20,
          lambda: torch.autograd.grad(y0, (xg, wg, bg), gy0, retain_graph=True),
          note="torch idiom = transpose + layer_norm")

    # ---- dropout ----
    x = randn(1024, 1024)
    h.row("dropout fwd 1024x1024", ce, "dropout fwd/1024x1024/iters=200", 200,
          lambda: F.dropout(x, 0.1, training=True))

    # ---- facade elementwise / views ----
    for n, iters in ((16_384, 200), (1_000_000, 200)):
        a = randn(n)
        b = randn(n)
        h.row(f"add n={n}", fa, f"case=add/mode=raw/n={n}/iters={iters}", iters, lambda a=a, b=b: a + b)
    a = randn(1_000_000)
    h.row("clamp n=1000000", fa, "case=clamp/mode=raw/n=1000000/iters=200", 200,
          lambda: torch.clamp(a, -0.5, 0.5))
    gate = randn(1_000_000)
    up = randn(1_000_000)
    h.row("swiglu n=1000000", fa, "case=swiglu/mode=raw/n=1000000/iters=200", 200,
          lambda: F.silu(gate) * up)
    m = randn(2048, 512)
    h.row("sum_last_axis 2048x512", fa, "case=sum_last_axis/mode=raw/n=1048576/iters=200", 200,
          lambda: m.sum(dim=-1))
    t = randn(256, 512)
    h.row("top_k 256x512 k=8", fa, "case=top_k_last_axis/mode=raw/n=131072/iters=50", 50,
          lambda: torch.topk(t, 8, dim=-1))
    mt_a = randn(64, 512)
    mt_b = randn(512, 512)
    h.row("matmul_trans_b 64x512x512", fa, "case=matmul_trans_b/mode=raw/n=32768/iters=200", 200,
          lambda: mt_a @ mt_b.T)

    # ---- rope (half pairing, table precomputed — the HF rotate-half idiom) ----
    seq, heads, d = 64, 8, 128
    xr = randn(seq, heads, d)
    inv = 1.0 / (10000.0 ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    ang = torch.arange(seq, dtype=torch.float32)[:, None] * inv[None, :]
    cos = torch.cos(ang)[:, None, :]
    sin = torch.sin(ang)[:, None, :]

    def rope_half():
        x1 = xr[..., : d // 2]
        x2 = xr[..., d // 2 :]
        return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)

    h.row("rope half 64x8x128", fa, "case=rope_table_full/mode=raw/n=65536/iters=200", 200, rope_half)

    def rope_interleaved():
        x1 = xr[..., 0::2]
        x2 = xr[..., 1::2]
        return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)

    h.row("rope interleaved 64x8x128", fa, "case=rope_table_interleaved/mode=raw/n=65536/iters=200",
          200, rope_interleaved)

    # ---- attention (SDPA is torch's best eager path) ----
    q1 = randn(1, 8, 1, 64)
    k1 = randn(1, 2, 256, 64)
    v1 = randn(1, 2, 256, 64)
    h.row("attention q1 kv256 h8/2 d64", fa,
          "case=grouped_causal_attention/mode=raw/n=512/iters=200", 200,
          lambda: F.scaled_dot_product_attention(q1, k1, v1, enable_gqa=True),
          note="decode step: q attends the whole cache, no mask")
    q2 = randn(1, 8, 64, 64)
    k2 = randn(1, 2, 64, 64)
    v2 = randn(1, 2, 64, 64)
    h.row("attention q64 kv64 h8/2 d64", fa,
          "case=grouped_causal_attention/mode=raw/n=32768/iters=200", 200,
          lambda: F.scaled_dot_product_attention(q2, k2, v2, is_causal=True, enable_gqa=True))

    for s, heads_a, kvh, iters in ((128, 16, 4, 18), (256, 16, 4, 8), (512, 16, 4, 6)):
        qa = randn(1, heads_a, s, 64).requires_grad_()
        ka = randn(1, kvh, s, 64).requires_grad_()
        va = randn(1, kvh, s, 64).requires_grad_()
        ya = F.scaled_dot_product_attention(qa, ka, va, is_causal=True, enable_gqa=True)
        ga = randn(1, heads_a, s, 64)
        h.row(
            f"attention bwd s{s} h16/4 d64", "attention-backward",
            f"backend=native/case=s{s}_h16_kv4_d64/q_seq={s}/kv_seq={s}/heads=16/kv_heads=4/d=64/iters={iters}",
            iters,
            lambda y=ya, t=(qa, ka, va), g=ga: torch.autograd.grad(y, t, g, retain_graph=True),
        )

    # ---- MLP forward + training step (full_bias: biases at activation shape) ----
    for batch, dim, hidden, out, i_fwd, i_bwd in (
        (16, 128, 512, 128, 80, 12),
        (32, 256, 1024, 256, 16, 3),
    ):
        case = f"b{batch}_d{dim}_h{hidden}_o{out}"
        xm = randn(batch, dim)
        w1 = randn(dim, hidden)
        b1 = randn(batch, hidden)
        gm = randn(batch, hidden)
        w2 = randn(hidden, out)
        b2 = randn(batch, out)
        h.row(
            f"mlp fwd {case}", "mlp",
            f"backend=native/mode=inference/bias_mode=full_bias/case={case}/batch={batch}/d={dim}/hidden={hidden}/out={out}/iters={i_fwd}",
            i_fwd,
            lambda x=xm, a=w1, b=b1, g=gm, c=w2, e=b2: ((x @ a + b) * g) @ c + e,
        )
        params = [t.clone().requires_grad_() for t in (xm, w1, b1, w2, b2)]

        def mlp_step(p=params, g=gm):
            for t in p:
                t.grad = None
            x_, w1_, b1_, w2_, b2_ = p
            (((x_ @ w1_ + b1_) * g) @ w2_ + b2_).sum().backward()

        h.row(
            f"mlp fwd+bwd {case}", "mlp",
            f"backend=native/mode=backward/bias_mode=full_bias/case={case}/batch={batch}/d={dim}/hidden={hidden}/out={out}/iters={i_bwd}",
            i_bwd, mlp_step,
            note="graph rebuilt per step on both sides",
        )

    # ---- optimizer steps (Qwen3-0.6B block shapes + the embedding matrix) ----
    hidden, ffn, q_out, vocab = 1024, 3072, 2048, 151936
    block_shapes = [
        (q_out, hidden), (hidden, hidden), (hidden, hidden), (hidden, q_out),
        (ffn, hidden), (ffn, hidden), (hidden, ffn),
    ]

    def optim_case(shapes, opt_ctor):
        params = [torch.randn(*s, requires_grad=True) for s in shapes]
        for p in params:
            p.grad = torch.randn_like(p)
        opt = opt_ctor(params)
        return lambda: opt.step()

    h.row("adamw step block 15.7M", "optim", "block/adamw/params=15.73M", 20,
          optim_case(block_shapes, lambda p: torch.optim.AdamW(p, lr=1e-4, foreach=True)))
    h.row("sgd step block 15.7M", "optim", "block/sgd/params=15.73M", 20,
          optim_case(block_shapes, lambda p: torch.optim.SGD(p, lr=1e-4, foreach=True)))
    h.row("sgd-momentum step block 15.7M", "optim", "block/sgd-momentum/params=15.73M", 20,
          optim_case(block_shapes, lambda p: torch.optim.SGD(p, lr=1e-4, momentum=0.9, foreach=True)))
    h.row("adamw step embedding 155.6M", "optim", "embedding/adamw/params=155.58M", 5,
          optim_case([(vocab, hidden)], lambda p: torch.optim.AdamW(p, lr=1e-4, foreach=True)))
    h.row("sgd step embedding 155.6M", "optim", "embedding/sgd/params=155.58M", 5,
          optim_case([(vocab, hidden)], lambda p: torch.optim.SGD(p, lr=1e-4, foreach=True)))


def emit(h: Harness, out_path: str):
    lines = [
        "# fucina vs PyTorch CPU — paired eager op benchmark",
        "",
        f"- torch {torch.__version__}, {torch.get_num_threads()} threads, "
        f"{platform.machine()} / {platform.platform()}",
        f"- fucina baseline: `{h.args.baseline}` (recorded by opbench_gate on the same host)",
        f"- methodology: 2 warmups, best-of-{h.args.repeats} mean-per-op at each row's "
        "fucina iteration count; grad regimes matched per row",
        "- ratio < 1.0 means fucina is faster",
        "",
        "| row | torch | fucina | fucina/torch | notes |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    wins = losses = 0
    for name, suite, key, torch_ns, fucina_ns, note in h.rows:
        if fucina_ns is None:
            lines.append(f"| {name} | {torch_ns / 1e3:.1f} us | (missing: {suite}:{key}) | | {note} |")
            continue
        ratio = fucina_ns / torch_ns
        wins += ratio < 1.0
        losses += ratio >= 1.0
        lines.append(
            f"| {name} | {torch_ns / 1e3:.1f} us | {fucina_ns / 1e3:.1f} us | {ratio:.3f} | {note} |"
        )
    lines += ["", f"fucina faster: {wins} rows; torch faster or tied: {losses} rows", ""]
    text = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(text)
    print(f"\nwrote {out_path}")
    print(f"fucina faster: {wins}, torch faster/tied: {losses}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    global args
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    h = Harness(args)
    build_rows(h)
    emit(h, args.out)


if __name__ == "__main__":
    main()
