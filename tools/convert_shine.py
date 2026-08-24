# SHINE checkpoint -> GGUF converter for Fucina's SHINE port.
#
# Packs the three released SHINE artifacts (HF Yewei-Liu/SHINE-*, MIT):
#   mem_tokens.pt     (148, 4096)          learned memory embeddings
#   metalora.pth      nested dict           rank-128 Meta LoRA, all linears
#   metanetwork.pth   state dict            4-layer M2P transformer
# into one f32 GGUF that src/models/research/shine/shine.zig loads next to the base
# Qwen3-8B GGUF. Tensor layouts are kept exactly as PyTorch stores them
# (A: [in, r], B: [r, out], linear weights: [out, in]).
#
# Invocation (same venv as gen_shine_goldens.py):
#   shine-ref/bin/python tools/convert_shine.py \
#       models/shine/ift_mqa_1qa models/shine/shine-ift-mqa-1qa.gguf
#
# The released 8B checkpoint needs no flags. For a checkpoint trained with
# other hyperparameters (e.g. a 0.6B run: --m2p-heads 8), pass the values
# the training config used; everything else is inferred from the tensors.
import argparse
from pathlib import Path

import torch
import numpy as np
from gguf import GGUFWriter

MODULES = [
    ("attention", "q", "attn_q"),
    ("attention", "k", "attn_k"),
    ("attention", "v", "attn_v"),
    ("attention", "o", "attn_o"),
    ("mlp", "gate", "ffn_gate"),
    ("mlp", "up", "ffn_up"),
    ("mlp", "down", "ffn_down"),
]


def to_f32(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float32).contiguous().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_dir")
    parser.add_argument("out_gguf")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--scale", type=float, default=0.001)
    parser.add_argument("--m2p-heads", type=int, default=32)
    args = parser.parse_args()
    ckpt = Path(args.ckpt_dir)

    mem_tokens = torch.load(ckpt / "mem_tokens.pt", map_location="cpu", weights_only=False)
    metalora = torch.load(ckpt / "metalora.pth", map_location="cpu", weights_only=False)
    metanetwork = torch.load(ckpt / "metanetwork.pth", map_location="cpu", weights_only=False)

    num_mem, hidden = mem_tokens.shape
    num_layers = len(metalora)
    metalora_r = metalora[0]["attention"]["q"]["A"].shape[-1]
    m2p_layers = 0
    while f"transformer_layers.{m2p_layers}.self_attn.in_proj_weight" in metanetwork:
        m2p_layers += 1
    assert m2p_layers > 0 and metanetwork["layer_pe"].shape == (num_layers, hidden)
    assert metanetwork["token_pe"].shape == (num_mem, hidden)

    writer = GGUFWriter(args.out_gguf, "shine")
    writer.add_uint32("shine.hidden_size", hidden)
    writer.add_uint32("shine.num_layers", num_layers)
    writer.add_uint32("shine.num_mem_token", num_mem)
    writer.add_uint32("shine.metalora_r", metalora_r)
    # Generated-LoRA hyperparameters (defaults = the released 8B run,
    # inference.ipynb): rank 8, scale 0.001 (sqrt-split onto A and B),
    # slicing method "rl", 32-head gelu post-LN M2P encoder layers,
    # layer-mixing pass first.
    writer.add_uint32("shine.lora_r", args.lora_r)
    writer.add_float32("shine.scale", args.scale)
    writer.add_string("shine.method", "rl")
    writer.add_uint32("shine.m2p.num_layers", m2p_layers)
    writer.add_uint32("shine.m2p.head_count", args.m2p_heads)
    writer.add_uint32("shine.m2p.feed_forward_length", metanetwork["transformer_layers.0.linear1.weight"].shape[0])
    writer.add_float32("shine.m2p.layer_norm_eps", 1e-5)
    writer.add_bool("shine.m2p.layer_transformer_first", True)

    writer.add_tensor("mem_tokens", to_f32(mem_tokens))
    writer.add_tensor("m2p.layer_pe", to_f32(metanetwork["layer_pe"]))
    writer.add_tensor("m2p.token_pe", to_f32(metanetwork["token_pe"]))
    for i in range(m2p_layers):
        prefix = f"transformer_layers.{i}"
        for src, dst in [
            (f"{prefix}.self_attn.in_proj_weight", f"m2p.{i}.attn_in.weight"),
            (f"{prefix}.self_attn.in_proj_bias", f"m2p.{i}.attn_in.bias"),
            (f"{prefix}.self_attn.out_proj.weight", f"m2p.{i}.attn_out.weight"),
            (f"{prefix}.self_attn.out_proj.bias", f"m2p.{i}.attn_out.bias"),
            (f"{prefix}.linear1.weight", f"m2p.{i}.ffn_up.weight"),
            (f"{prefix}.linear1.bias", f"m2p.{i}.ffn_up.bias"),
            (f"{prefix}.linear2.weight", f"m2p.{i}.ffn_down.weight"),
            (f"{prefix}.linear2.bias", f"m2p.{i}.ffn_down.bias"),
            (f"{prefix}.norm1.weight", f"m2p.{i}.norm1.weight"),
            (f"{prefix}.norm1.bias", f"m2p.{i}.norm1.bias"),
            (f"{prefix}.norm2.weight", f"m2p.{i}.norm2.weight"),
            (f"{prefix}.norm2.bias", f"m2p.{i}.norm2.bias"),
        ]:
            writer.add_tensor(dst, to_f32(metanetwork[src]))

    for layer_i in range(num_layers):
        for group, key, name in MODULES:
            entry = metalora[layer_i][group][key]
            assert entry.get("C") is None, "qwen3 linears are bias-free"
            a = to_f32(entry["A"][0])  # (in, r)
            b = to_f32(entry["B"][0])  # (r, out)
            assert a.shape[1] == metalora_r and b.shape[0] == metalora_r
            writer.add_tensor(f"metalora.{layer_i}.{name}.a", a)
            writer.add_tensor(f"metalora.{layer_i}.{name}.b", b)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.out_gguf}: {num_layers} layers, M={num_mem}, metalora r={metalora_r}, m2p layers={m2p_layers}, lora r={args.lora_r}, m2p heads={args.m2p_heads}")


if __name__ == "__main__":
    main()
