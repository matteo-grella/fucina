# Golden-value generator for Fucina's SHINE port (src/llm/qwen3/shine.zig).
#
# SHINE (arXiv 2602.06358): an in-context hypernetwork that maps a context
# passage to a rank-8 LoRA over every linear of a frozen Qwen3-8B in a single
# forward pass. Reference implementation: refs/SHINE (github.com/MuLabPKU/SHINE,
# checkpoints HF Yewei-Liu/SHINE-ift_mqa_1qa, MIT).
#
# Environment + invocation (torch CPU only, fp32, ~40GB RAM; transformers is
# PINNED to the 4.57 API window the reference code was written against —
# 5.x drops the modeling_qwen3 re-exports LoraQwen.py imports):
#   uv venv shine-ref && uv pip install torch "transformers==4.57.1" omegaconf numpy
#   shine-ref/bin/python tools/gen_shine_goldens.py \
#       --hf-model models/qwen3-8b-hf \
#       --shine-ckpt models/shine/ift_mqa_1qa \
#       --shine-repo refs/SHINE \
#       --out models/shine/goldens
#
# The reference pipeline replicated here, verified line by line against
# refs/SHINE (inference.ipynb config "8gpu_8lora_128metalora_lr5e-5_
# grouppretrain_1150" = the released checkpoint):
#
#   1. Evidence encoding: plain tokenization of the raw context text (no chat
#      template — HumanCollator), 148 learned memory embeddings appended AFTER
#      the context rows, one causal forward of Qwen3-8B with the rank-128 Meta
#      LoRA active on all 7 linears of every layer (LoraQwen.LoraLinear:
#      out = base + (x @ A) @ B). After EACH decoder layer i the last-148
#      rows of the residual stream are captured: memory_states[i] (36,148,4096).
#      The reference left-pads evidence to a fixed length; padding is
#      RoPE-shift + mask equivalent to unpadded, so goldens are unpadded.
#   2. M2P metanetwork (MetanetworkTransformer): add layer_pe + token_pe,
#      then 4 post-LN nn.TransformerEncoderLayer passes (batch_first, gelu,
#      ff 8192, 32 heads) alternating layer-mixing (seq = 36 layers, one
#      sequence per memory token) and token-mixing (seq = 148 tokens, one
#      sequence per layer), layer-mixing first. mean_pool_size=1 and
#      couple_num_layers=0: both dead for this checkpoint.
#   3. LoRA generation ("rl" method): the (36,148,4096) output is flattened
#      per layer and sliced sequentially per module in order
#      q,k,v,o,gate,up,down; A = slice.view(in,8)*sqrt(0.001),
#      B = slice.view(8,out)*sqrt(0.001). No bias (C) terms: qwen3 linears
#      are bias-free.
#   4. Conversation: chat-template ids (the repo's custom Qwen3 template that
#      pre-fills an empty <think> block by default), forward with the
#      GENERATED LoRA on all linears (ignore_mem_token=True), greedy decode.
#      Note: reference resizes embeddings to len(tokenizer) (151669+3 added
#      task tokens = 151672 <= the GGUF's padded 151936 rows), so greedy
#      argmax in the port must be restricted to the first `vocab` logits.
#
# Emits raw little-endian binaries + manifest.json into --out and prints a
# sha256 per artifact so the goldens are auditable.
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


def dump(out_dir: Path, name: str, tensor, manifest: dict, dtype=None):
    array = tensor.detach().to(torch.float32).cpu().numpy() if torch.is_tensor(tensor) else np.asarray(tensor)
    if dtype is not None:
        array = array.astype(dtype)
    path = out_dir / name
    array.tofile(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest[name] = {"shape": list(array.shape), "dtype": str(array.dtype), "sha256": digest}
    print(f"{name}: shape={list(array.shape)} dtype={array.dtype} sha256={digest[:16]}", file=sys.stderr)


def chat_template_from_notebook(shine_repo: Path) -> str:
    """The repo sets a custom chat template inline in inference.ipynb; read it
    from the notebook so the goldens can never drift from the reference."""
    nb = json.loads((shine_repo / "inference.ipynb").read_text())
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if "tokenizer.chat_template = " in src:
            for line in src.splitlines():
                if line.startswith("tokenizer.chat_template = "):
                    return eval(line.split("=", 1)[1].strip())  # a quoted literal
    raise SystemExit("chat template assignment not found in inference.ipynb")


def extract_answer(text: str) -> str:
    """inference.ipynb extract_think_and_answer, answer part only."""
    import re

    answer = text
    if "<think>" in text:
        rest = text.split("<think>", 1)[1]
        answer = rest.split("</think>", 1)[1].strip() if "</think>" in rest else ""
    else:
        answer = text.strip()
    return re.sub(r"^(final answer|answer)\s*:\s*", "", answer, flags=re.IGNORECASE).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model", default="models/qwen3-8b-hf")
    parser.add_argument("--shine-ckpt", default="models/shine/ift_mqa_1qa")
    parser.add_argument("--shine-repo", default="refs/SHINE")
    parser.add_argument("--out", default="models/shine/goldens")
    parser.add_argument("--greedy-tokens", type=int, default=64)
    args = parser.parse_args()

    shine_repo = Path(args.shine_repo).resolve()
    sys.path.insert(0, str(shine_repo))
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer

    from LoraQwen import LoraQwen3ForCausalLM, Qwen3Config
    from metanetwork_family import Metanetwork
    from utils.mysaveload import load_checkpoint

    torch.manual_seed(42)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {}

    # --- Config: inference.ipynb conf_dict, released-checkpoint values ---
    encoder_cfg = {
        "d_model": 4096, "nhead": 32, "dim_feedforward": 8192, "dropout": 0,
        "activation": "gelu", "layer_norm_eps": 0.00001, "batch_first": True,
        "norm_first": False, "bias": True,
    }
    cfg = OmegaConf.create({
        "model": {"lora_r": 8, "metalora_r": 128},
        "metanetwork": {
            "type": "transformer",
            "method": "rl",
            "transformer_cfg": {
                "encoder_cfg": encoder_cfg,
                "couple_encoder_cfg": encoder_cfg,
                "layer_transformer_first": True,
                "mean_pool_size": 1,
                "num_layers": 4,
                "couple_num_layers": 0,
                "scale": 0.001,
            },
        },
        "hidden_size": -1, "num_layers": -1, "num_mem_token": 4,
    })

    config = Qwen3Config.from_pretrained(args.hf_model)
    config.num_mem_token = -1
    cfg.hidden_size = config.hidden_size
    cfg.num_layers = config.num_hidden_layers

    with torch.device("meta"):
        tmp_model = LoraQwen3ForCausalLM(config)
    lora_params = tmp_model.lora_params_numel(cfg.model.lora_r)
    base_params = cfg.hidden_size * cfg.num_layers
    assert lora_params % base_params == 0
    config.num_mem_token = lora_params // base_params
    cfg.num_mem_token = config.num_mem_token
    del tmp_model
    print(f"num_mem_token={config.num_mem_token}", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, padding_side="left", use_fast=True)
    tokenizer.add_tokens(["<RECON>", "<COMP>", "<NOTHING>"])
    tokenizer.chat_template = chat_template_from_notebook(shine_repo)

    print("loading Qwen3-8B fp32 (~33GB)...", file=sys.stderr)
    metamodel = LoraQwen3ForCausalLM.from_pretrained(args.hf_model, config=config)
    metamodel.reset_mem_tokens()
    metamodel.resize_token_embeddings(len(tokenizer))
    metanetwork = Metanetwork(metamodel, cfg, metamodel.lora_params_numel(cfg.model.lora_r))
    metanetwork, metalora, _ = load_checkpoint(metanetwork, args.shine_ckpt, "cpu")
    metanetwork.eval()

    # --- Fixed inputs: inference.ipynb data[2] ---
    context = (
        "If the light is on, somebody must be at home. If the light is off, "
        "often nobody is at home. But this holds true only during the day. "
        "In the night people are all sleeping so there will always be no lights."
    )
    questions = ["What does it mean if the light is on?", "What does it mean if the light is off?"]

    evidence = tokenizer(context, return_tensors="pt")
    evidence_ids = evidence["input_ids"]
    dump(out_dir, "evidence_ids.i32.bin", evidence_ids[0], manifest, dtype=np.int32)

    with torch.no_grad():
        # --- Stage 1: encoder pass with Meta LoRA + memory extraction ---
        outputs = metanetwork.metamodel(
            input_ids=evidence_ids,
            attention_mask=evidence["attention_mask"],
            loradict=metalora,
        )
        memory_states = outputs.memory_states  # (1, 36, 148, 4096)
        dump(out_dir, "memory_states.f32.bin", memory_states[0], manifest)

        # --- Stage 2: M2P metanetwork, replayed module-by-module so per-stage
        # probes exist for bisection; verified against the module forward. ---
        net = metanetwork.metanetwork
        ms = memory_states + net.layer_pe.unsqueeze(-2) + net.token_pe
        probes = [ms]
        b, num_l, num_m, hidden = ms.shape
        for i, layer in enumerate(net.transformer_layers):
            if (i % 2 == 0) == net.layer_transformer_first:
                ms = layer(ms.transpose(1, 2).flatten(0, 1)).unflatten(0, (b, num_m)).transpose(1, 2)
            else:
                ms = layer(ms.flatten(0, 1)).unflatten(0, (b, num_l))
            probes.append(ms)
        plain_replay = ms.reshape(b, -1)
        plain = net(memory_states)
        assert torch.equal(plain_replay, plain), "M2P replay diverged from module forward"
        for i, probe in enumerate(probes):
            dump(out_dir, f"m2p_stage{i}_probe.f32.bin", probe[0, 0], manifest)  # (148, 4096)
        dump(out_dir, "plain_output.f32.bin", plain[0], manifest)

        # --- Stage 3: generated LoRA (spot dumps for slice/reshape parity) ---
        loradict = metamodel.generate_lora_dict(cfg.model.lora_r, scale=metanetwork.scale, plain_tensor=plain)
        dump(out_dir, "lora_l0_q_a.f32.bin", loradict[0]["attention"]["q"]["A"][0], manifest)
        dump(out_dir, "lora_l0_q_b.f32.bin", loradict[0]["attention"]["q"]["B"][0], manifest)
        last = config.num_hidden_layers - 1
        dump(out_dir, f"lora_l{last}_down_a.f32.bin", loradict[last]["mlp"]["down"]["A"][0], manifest)
        dump(out_dir, f"lora_l{last}_down_b.f32.bin", loradict[last]["mlp"]["down"]["B"][0], manifest)

        # --- Stage 4: adapted conversation, greedy ---
        messages = []
        for turn, question in enumerate(questions, start=1):
            messages.append({"role": "user", "content": question})
            enc = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_tensors="pt", return_dict=True,
            )
            dump(out_dir, f"turn{turn}_ids.i32.bin", enc["input_ids"][0], manifest, dtype=np.int32)

            fwd = metamodel(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                loradict=loradict, ignore_mem_token=True,
            )
            dump(out_dir, f"turn{turn}_logits.f32.bin", fwd.logits[0, -1], manifest)

            generated = metamodel.generate(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                max_new_tokens=args.greedy_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                loradict=loradict, ignore_mem_token=True,
            )
            new_tokens = generated[0, enc["input_ids"].shape[1]:]
            dump(out_dir, f"turn{turn}_greedy_ids.i32.bin", new_tokens, manifest, dtype=np.int32)
            answer = extract_answer(tokenizer.decode(new_tokens, skip_special_tokens=True))
            print(f"turn {turn}: {question!r} -> {answer!r}", file=sys.stderr)
            messages.append({"role": "assistant", "content": answer})

    manifest["config"] = {
        "context": context,
        "questions": questions,
        "num_mem_token": int(config.num_mem_token),
        "lora_r": int(cfg.model.lora_r),
        "scale": float(metanetwork.scale),
        "sqrt_scale": math.sqrt(metanetwork.scale),
        "vocab": len(tokenizer),
        "hidden_size": int(config.hidden_size),
        "num_layers": int(config.num_hidden_layers),
        "greedy_tokens": args.greedy_tokens,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"goldens written to {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
