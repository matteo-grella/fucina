# torch-MPS reference for the fucina qwen3 LoRA training step — the
# apples-to-apples counterpart of `zig build finetune --no-sample` probes:
# frozen base, LoRA r=8 on q_proj/v_proj, batch 1, one fixed sequence with
# prompt-masked labels (first half -100), the FULL macro step (forward,
# backward, grad clip, AdamW on the adapters, zero), explicit MPS
# synchronization around every timed region, 1 warmup + N timed. Pass
# --seq-len as the fucina probe's measured tokens/step so both engines
# consume identical token counts.
#
#   shine-ref/bin/python tools/bench_lora_train_mps.py \
#       --model models/qwen3-0.6b-hf [--seq-len 1280] [--rank 8] [--steps 3]
import argparse
import os
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="models/qwen3-0.6b-hf")
parser.add_argument("--seq-len", type=int, default=1280)
parser.add_argument("--rank", type=int, default=8)
parser.add_argument("--steps", type=int, default=3)
parser.add_argument("--lr", type=float, default=1e-3)
args = parser.parse_args()

import torch
from transformers import AutoModelForCausalLM

assert torch.backends.mps.is_available()
device = torch.device("mps")
torch.manual_seed(42)

model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
model.to(device)
model.train()
for p in model.parameters():
    p.requires_grad_(False)


class LoraLinear(torch.nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base = base
        self.a = torch.nn.Parameter(torch.randn(base.in_features, rank, device=device) * 0.01)
        self.b = torch.nn.Parameter(torch.zeros(rank, base.out_features, device=device))

    def forward(self, x):
        return self.base(x) + (x @ self.a) @ self.b


adapters = []
for layer in model.model.layers:
    layer.self_attn.q_proj = LoraLinear(layer.self_attn.q_proj, args.rank)
    layer.self_attn.v_proj = LoraLinear(layer.self_attn.v_proj, args.rank)
    adapters += [layer.self_attn.q_proj, layer.self_attn.v_proj]

ids = torch.randint(1000, 100_000, (1, args.seq_len), device=device)
labels = ids.clone()
labels[:, : args.seq_len // 2] = -100  # prompt-masked, the fucina shape

params = [p for m in adapters for p in (m.a, m.b)]
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0)


def step():
    out = model(input_ids=ids, labels=labels)
    out.loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)


torch.mps.synchronize()
step()  # warmup (MPS graph capture, allocator)
torch.mps.synchronize()
t0 = time.perf_counter()
for _ in range(args.steps):
    step()
    torch.mps.synchronize()
elapsed = time.perf_counter() - t0
ms = elapsed / args.steps * 1000
print(f"torch-mps lora step [{args.model} fp32 r{args.rank} seq {args.seq_len}]: "
      f"{ms:.0f} ms, {args.seq_len / (ms / 1000):.1f} tok/s")
