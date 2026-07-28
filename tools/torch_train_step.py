#!/usr/bin/env python3
"""PyTorch twin of `zig build bench-train-step` — the end-to-end autograd
head-to-head.

Loads the EXACT weights, token ids, and rope table the fucina bench dumped,
rebuilds the identical GPT (rmsNorm → RoPE → causal SDPA attention → SwiGLU
MLP, untied unembed, mean cross-entropy) in eager PyTorch, and runs the same
AdamW steps on the same fixed sequence. Matching loss trajectories prove both
frameworks compute the same training step; the per-step wall clock is the
comparison. SDPA is torch's best eager attention path and the AdamW uses
foreach fused updates — this is idiomatic-best eager torch, not a strawman.

Usage:
  .venv/bin/python tools/torch_train_step.py --dump /tmp/train-dump [--threads 8]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

N_LAYER = 6
D_MODEL = 512
N_HEAD = 8
HEAD_DIM = D_MODEL // N_HEAD
ATTN_SCALE = 0.125
RMS_EPS = 1e-5


def load_dump(dump_dir: Path) -> dict[str, torch.Tensor]:
    tensors = {}
    for entry in json.loads((dump_dir / "manifest.json").read_text()):
        raw = bytearray((dump_dir / entry["file"]).read_bytes())
        tensors[entry["name"]] = torch.frombuffer(raw, dtype=torch.float32).clone().reshape(entry["shape"])
    return tensors


def rope_half(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [seq, head, hd]; cos/sin: [seq, 1, hd/2] — the .half pairing.
    x1 = x[..., : HEAD_DIM // 2]
    x2 = x[..., HEAD_DIM // 2 :]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


def forward_loss(params, ids, labels, cos, sin) -> torch.Tensor:
    x = params["wte"][ids]
    for i in range(N_LAYER):
        p = lambda name: params[f"layers.{i}.{name}"]
        h = F.rms_norm(x, (D_MODEL,), eps=RMS_EPS)
        seq = h.shape[0]
        q = (h @ p("c_q")).view(seq, N_HEAD, HEAD_DIM)
        k = (h @ p("c_k")).view(seq, N_HEAD, HEAD_DIM)
        v = (h @ p("c_v")).view(seq, N_HEAD, HEAD_DIM)
        q = rope_half(q, cos, sin)
        k = rope_half(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q.permute(1, 0, 2).unsqueeze(0),
            k.permute(1, 0, 2).unsqueeze(0),
            v.permute(1, 0, 2).unsqueeze(0),
            is_causal=True,
            scale=ATTN_SCALE,
        )
        y = y.squeeze(0).permute(1, 0, 2).reshape(seq, D_MODEL)
        x = x + y @ p("c_proj")
        m = F.rms_norm(x, (D_MODEL,), eps=RMS_EPS)
        x = x + (F.silu(m @ p("w_gate")) * (m @ p("w_up"))) @ p("w_down")
    x = F.rms_norm(x, (D_MODEL,), eps=RMS_EPS)
    logits = F.linear(x, params["w_lm"])
    return F.cross_entropy(logits, labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--compile", choices=["off", "default", "max-autotune"], default="off",
        help="torch.compile the loss fn (Inductor CPU; AOTAutograd compiles the backward too)",
    )
    parser.add_argument("--fused-adamw", action="store_true", help="AdamW(fused=True) instead of foreach")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    dump_dir = Path(args.dump)
    tensors = load_dump(dump_dir)
    ids_all = tensors.pop("token_ids").to(torch.long)
    ids = ids_all[:-1]
    labels = ids_all[1:]
    cos = tensors.pop("rope_cos").unsqueeze(1)
    sin = tensors.pop("rope_sin").unsqueeze(1)
    params = {name: t.requires_grad_() for name, t in tensors.items()}

    fucina = json.loads((dump_dir / "fucina_results.json").read_text())
    warmup = fucina["warmup_steps"]
    n_steps = len(fucina["losses"])

    if args.fused_adamw:
        opt = torch.optim.AdamW(params.values(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, fused=True)
    else:
        opt = torch.optim.AdamW(params.values(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, foreach=True)

    loss_fn = forward_loss
    if args.compile != "off":
        mode = None if args.compile == "default" else args.compile
        loss_fn = torch.compile(forward_loss, fullgraph=True, dynamic=False, mode=mode)
    print(f"torch {torch.__version__}  threads={args.threads}  compile={args.compile}  "
          f"adamw={'fused' if args.fused_adamw else 'foreach'}")

    losses, step_ms = [], []
    for step_i in range(n_steps):
        t0 = time.perf_counter_ns()
        for p in params.values():
            p.grad = None
        loss = loss_fn(params, ids, labels, cos, sin)
        loss.backward()
        loss_value = loss.item()
        opt.step()
        step_ms.append((time.perf_counter_ns() - t0) / 1e6)
        losses.append(loss_value)
        print(f"step {step_i:>2}  loss {loss_value:.6f}  {step_ms[-1]:>8.2f} ms")

    print(f"\n{'step':>4} {'fucina loss':>12} {'torch loss':>12} {'|diff|':>10} {'fucina ms':>10} {'torch ms':>10}")
    max_diff = 0.0
    for i in range(n_steps):
        diff = abs(fucina["losses"][i] - losses[i])
        max_diff = max(max_diff, diff)
        print(f"{i:>4} {fucina['losses'][i]:>12.6f} {losses[i]:>12.6f} {diff:>10.6f} "
              f"{fucina['step_ms'][i]:>10.2f} {step_ms[i]:>10.2f}")

    f_mean = sum(fucina["step_ms"][warmup:]) / (n_steps - warmup)
    t_mean = sum(step_ms[warmup:]) / (n_steps - warmup)
    print(f"\ntimed mean ({n_steps - warmup} steps after {warmup} warmup): "
          f"fucina {f_mean:.2f} ms/step, torch {t_mean:.2f} ms/step, "
          f"torch/fucina = {t_mean / f_mean:.3f}x")
    print(f"max |loss diff| over {n_steps} steps: {max_diff:.6f}")

    if abs(fucina["losses"][0] - losses[0]) > 1e-3:
        raise SystemExit("FAIL: step-0 losses disagree — the models are not computing the same thing")
    if max_diff > 5e-2:
        raise SystemExit("FAIL: loss trajectories diverged beyond f32 drift")
    print("PASS: loss trajectories match")


if __name__ == "__main__":
    main()
