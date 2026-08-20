# torch-MPS reference for 0.6B inference — the apples-to-apples counterpart
# of `zig build qwen3 -- MODEL --prompt @TEXT --gen N --bench R`: same prompt
# text (same Qwen tokenizer, so identical token counts), greedy decode with a
# KV cache, prefill and decode timed separately, 1 warmup rep + R timed,
# explicit MPS synchronization around every timed region.
#
#   shine-ref/bin/python tools/bench_generate_mps.py \
#       --model models/qwen3-0.6b-hf --prompt-file P.txt \
#       [--dtype fp16|fp32] [--gen 128] [--reps 3]
import argparse
import os
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="models/qwen3-0.6b-hf")
parser.add_argument("--prompt-file", required=True)
parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
parser.add_argument("--gen", type=int, default=128)
parser.add_argument("--reps", type=int, default=3)
parser.add_argument("--quant", choices=["none", "int8wo", "int4wo"], default="none",
                    help="torchao weight-only quantization applied on the MPS device")
args = parser.parse_args()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

assert torch.backends.mps.is_available()
device = torch.device("mps")
dtype = torch.float16 if args.dtype == "fp16" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
model.to(device)
model.eval()

if args.quant != "none":
    from torchao.quantization import quantize_, Int8WeightOnlyConfig, Int4WeightOnlyConfig
    cfg = Int8WeightOnlyConfig() if args.quant == "int8wo" else Int4WeightOnlyConfig()
    quantize_(model, cfg)

text = open(args.prompt_file).read()
ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
prompt_tokens = ids.shape[1]


@torch.no_grad()
def run():
    torch.mps.synchronize()
    t0 = time.perf_counter()
    out = model(input_ids=ids, use_cache=True)
    torch.mps.synchronize()
    t_prefill = time.perf_counter() - t0

    past = out.past_key_values
    tok = out.logits[:, -1:].argmax(dim=-1)
    t0 = time.perf_counter()
    for _ in range(args.gen - 1):
        out = model(input_ids=tok, past_key_values=past, use_cache=True)
        past = out.past_key_values
        tok = out.logits[:, -1:].argmax(dim=-1)
    torch.mps.synchronize()
    t_decode = time.perf_counter() - t0
    return t_prefill, t_decode


run()  # warmup (MPS graph capture, allocator)
pp = []
tg = []
for _ in range(args.reps):
    t_prefill, t_decode = run()
    pp.append(prompt_tokens / t_prefill)
    tg.append((args.gen - 1) / t_decode)
print(f"torch-mps generate [{args.model} {args.dtype} quant={args.quant}] prompt {prompt_tokens} tok, gen {args.gen}:")
print(f"  prefill: {sum(pp)/len(pp):.1f} tok/s  (runs: {', '.join(f'{x:.1f}' for x in pp)})")
print(f"  decode:  {sum(tg)/len(tg):.2f} tok/s  (runs: {', '.join(f'{x:.2f}' for x in tg)})")
