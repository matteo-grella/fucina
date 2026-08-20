# torch-MPS reference for the Zig SHINE-training-step timing test — the
# apples-to-apples counterpart of `shine_train_golden_tests` "SHINE training
# step timing at 0.6B": IDENTICAL step shape (evidence 512 + conversation
# 1024 tokens, batch 1), identical protocol (loss + backward only, no
# optimizer, no data pipeline; 1 warmup step + 3 timed), explicit MPS
# synchronization around every timed region.
#
#   shine-ref/bin/python tools/bench_shine_train_mps.py \
#       --shine-repo refs/SHINE --model models/qwen3-0.6b-hf \
#       [--dtype fp32|bf16] [--checkpoint] [--compile]
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

parser = argparse.ArgumentParser()
parser.add_argument("--shine-repo", default="refs/SHINE")
parser.add_argument("--model", default="models/qwen3-0.6b-hf")
parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
parser.add_argument("--checkpoint", action="store_true")
parser.add_argument("--compile", action="store_true")
args = parser.parse_args()
if not args.compile:
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch

sys.path.insert(0, str(Path(args.shine_repo).resolve()))
from omegaconf import OmegaConf
from LoraQwen import LoraQwen3ForCausalLM, Qwen3Config
from metanetwork_family import Metanetwork
from utils.myfreeze import freeze

assert torch.backends.mps.is_available()
device = torch.device("mps")
torch.manual_seed(42)

enc = {"d_model": 1024, "nhead": 8, "dim_feedforward": 2048, "dropout": 0, "activation": "gelu",
       "layer_norm_eps": 1e-5, "batch_first": True, "norm_first": False, "bias": True}
cfg = OmegaConf.create({
    "model": {"lora_r": 8, "metalora_r": 128},
    "metanetwork": {"type": "transformer", "method": "rl", "transformer_cfg": {
        "encoder_cfg": enc, "couple_encoder_cfg": enc, "layer_transformer_first": True,
        "mean_pool_size": 1, "num_layers": 4, "couple_num_layers": 0, "scale": 0.001}},
    "hidden_size": 1024, "num_layers": 28, "num_mem_token": 176,
})
config = Qwen3Config.from_pretrained(args.model)
config.num_mem_token = 176
dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
model = LoraQwen3ForCausalLM.from_pretrained(args.model, config=config, torch_dtype=dtype)
model.reset_mem_tokens()
net = Metanetwork(model, cfg, model.lora_params_numel(8)).to(device=device, dtype=dtype)
freeze(model)
metalora = model.init_lora_dict(128, scale=0.001, device=device)
if dtype != torch.float32:
    for i in metalora:
        for g in metalora[i]:
            for k in metalora[i][g]:
                for half in ("A", "B"):
                    t = metalora[i][g][k][half]
                    metalora[i][g][k][half] = t.detach().to(dtype).requires_grad_()
net.train()

evidence = torch.randint(1000, 100_000, (1, 512), device=device)
inputs = torch.randint(1000, 100_000, (1, 1024), device=device)
labels = inputs.clone()
ones_e = torch.ones_like(evidence)
ones_i = torch.ones_like(inputs)

def step():
    out = net(
        input_ids=inputs, input_attention_mask=ones_i,
        evidence_ids=evidence, evidence_attention_mask=ones_e,
        metalora=metalora, labels=labels, use_metanet=True,
        use_gradient_checkpoint=args.checkpoint,
    )
    out.loss.backward()
    for p in net.metanetwork.parameters():
        p.grad = None
    for i in metalora:
        for g in metalora[i]:
            for k in metalora[i][g]:
                for half in ("A", "B"):
                    metalora[i][g][k][half].grad = None

torch.mps.synchronize()
step()  # warmup (MPS graph capture, allocator)
torch.mps.synchronize()
steps = 3
t0 = time.perf_counter()
for _ in range(steps):
    step()
    torch.mps.synchronize()
elapsed = time.perf_counter() - t0
ms = elapsed / steps * 1000
tokens = 512 + 1024
print(f"torch-mps shine step [{args.dtype}{' ckpt' if args.checkpoint else ''}{' compile' if args.compile else ''}]: "
      f"{ms:.0f} ms, {tokens / (ms / 1000):.1f} tok/s (evidence 512 + conversation 1024)")
