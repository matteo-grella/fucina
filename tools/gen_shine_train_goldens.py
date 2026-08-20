# Golden-value generator for the (planned) Zig-native SHINE trainer.
#
# One full SHINE training step on a TINY synthetic Qwen3 (hidden 64, 2
# layers, vocab 256; generated lora_r 2 -> 29 memory tokens, metalora r 4),
# through the reference implementation in refs/SHINE: encoder pass with the
# Meta LoRA + memory capture, M2P, "rl" slicing, adapted conversation pass,
# masked CE loss, and a full backward. Dumps every weight, input,
# intermediate, the loss, and the gradients of every trainable leaf — the
# parity gate a fucina trainer must reproduce before it is trusted.
#
# Runs in seconds on CPU. Same venv as gen_shine_goldens.py
# (transformers==4.57.1 pinned; refs/SHINE on the path):
#   shine-ref/bin/python tools/gen_shine_train_goldens.py \
#       --shine-repo refs/SHINE --out models/shine/train-goldens
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# Neutralize the @torch.compile decorator on Metanetwork.forward BEFORE
# torch loads — the goldens must come from the eager graph.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

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
    print(f"{name}: shape={list(array.shape)} sha256={digest[:16]}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shine-repo", default="refs/SHINE")
    parser.add_argument("--out", default="models/shine/train-goldens")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.shine_repo).resolve()))

    from omegaconf import OmegaConf
    from LoraQwen import LoraQwen3ForCausalLM, Qwen3Config
    from metanetwork_family import Metanetwork
    from utils.myfreeze import freeze

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {}
    torch.manual_seed(7)

    # Tiny geometry whose per-layer generated-LoRA budget divides hidden:
    # r=2 over q(64,64) k(64,32) v(64,32) o(64,64) gate/up(64,96) down(96,64)
    # -> 1856 params = 29 * 64 -> 29 memory tokens.
    config = Qwen3Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        max_position_embeddings=512,
        tie_word_embeddings=False,
        num_mem_token=-1,
    )
    lora_r, metalora_r, scale = 2, 4, 0.001

    with torch.device("meta"):
        tmp = LoraQwen3ForCausalLM(config)
    per_layer = tmp.lora_params_numel(lora_r) // config.num_hidden_layers
    assert per_layer % config.hidden_size == 0
    config.num_mem_token = tmp.lora_params_numel(lora_r) // (config.hidden_size * config.num_hidden_layers)
    del tmp
    print(f"num_mem_token={config.num_mem_token}", file=sys.stderr)

    encoder_cfg = {
        "d_model": config.hidden_size, "nhead": 4, "dim_feedforward": 128, "dropout": 0,
        "activation": "gelu", "layer_norm_eps": 1e-5, "batch_first": True,
        "norm_first": False, "bias": True,
    }
    cfg = OmegaConf.create({
        "model": {"lora_r": lora_r, "metalora_r": metalora_r},
        "metanetwork": {
            "type": "transformer",
            "method": "rl",
            "transformer_cfg": {
                "encoder_cfg": encoder_cfg,
                "couple_encoder_cfg": encoder_cfg,
                "layer_transformer_first": True,
                "mean_pool_size": 1,
                "num_layers": 2,
                "couple_num_layers": 0,
                "scale": scale,
            },
        },
        "hidden_size": config.hidden_size,
        "num_layers": config.num_hidden_layers,
        "num_mem_token": config.num_mem_token,
    })

    metamodel = LoraQwen3ForCausalLM(config)  # seeded random init
    metamodel.reset_mem_tokens()
    metanetwork = Metanetwork(metamodel, cfg, metamodel.lora_params_numel(lora_r))
    freeze(metamodel)
    metalora = metamodel.init_lora_dict(metalora_r, scale=scale, device="cpu")
    metanetwork.train()

    # Fixed batch: an "evidence" passage and a masked conversation.
    evidence_ids = torch.tensor([[11, 45, 3, 200, 77, 5, 91, 128, 33, 2, 66, 190]])
    input_ids = torch.tensor([[7, 88, 41, 3, 150, 22, 9, 244, 61, 30]])
    labels = input_ids.clone()
    labels[0, :4] = -100

    dump(out_dir, "evidence_ids.i32.bin", evidence_ids[0], manifest, dtype=np.int32)
    dump(out_dir, "input_ids.i32.bin", input_ids[0], manifest, dtype=np.int32)
    dump(out_dir, "labels.i32.bin", labels[0], manifest, dtype=np.int32)

    # Every weight the Zig side must load to rebuild this exact step.
    for name, tensor in metamodel.state_dict().items():
        dump(out_dir, f"model.{name}.f32.bin", tensor, manifest)
    for name, tensor in metanetwork.metanetwork.state_dict().items():
        dump(out_dir, f"m2p.{name}.f32.bin", tensor, manifest)
    for i in sorted(metalora.keys()):
        for group, mods in (("attention", "qkvo"), ("mlp", ["gate", "up", "down"])):
            for key in (mods if isinstance(mods, list) else list(mods)):
                dump(out_dir, f"metalora.{i}.{group}.{key}.A.f32.bin", metalora[i][group][key]["A"][0], manifest)
                dump(out_dir, f"metalora.{i}.{group}.{key}.B.f32.bin", metalora[i][group][key]["B"][0], manifest)

    # The step: identical to Metanetwork.forward(use_metanet=True) but
    # unrolled so intermediates land in the dump.
    enc = metanetwork.metamodel(input_ids=evidence_ids, attention_mask=torch.ones_like(evidence_ids), loradict=metalora)
    memory_states = enc.memory_states
    dump(out_dir, "memory_states.f32.bin", memory_states[0], manifest)
    plain = metanetwork.metanetwork(memory_states)
    dump(out_dir, "plain_output.f32.bin", plain[0], manifest)
    loradict = metamodel.generate_lora_dict(lora_r, scale=metanetwork.scale, plain_tensor=plain)
    dump(out_dir, "gen_lora_l0_q_a.f32.bin", loradict[0]["attention"]["q"]["A"][0], manifest)
    outputs = metanetwork.metamodel(
        input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
        loradict=loradict, labels=labels, ignore_mem_token=True,
    )
    loss = outputs.loss
    loss.backward()
    dump(out_dir, "loss.f32.bin", loss.reshape(1), manifest)

    # Gradients of every trainable leaf.
    for name, param in metanetwork.metanetwork.named_parameters():
        assert param.grad is not None, name
        dump(out_dir, f"grad.m2p.{name}.f32.bin", param.grad, manifest)
    for i in sorted(metalora.keys()):
        for group, mods in (("attention", "qkvo"), ("mlp", ["gate", "up", "down"])):
            for key in (mods if isinstance(mods, list) else list(mods)):
                grad = metalora[i][group][key]["A"].grad
                assert grad is not None
                dump(out_dir, f"grad.metalora.{i}.{group}.{key}.A.f32.bin", grad[0], manifest)
                grad_b = metalora[i][group][key]["B"].grad
                assert grad_b is not None
                dump(out_dir, f"grad.metalora.{i}.{group}.{key}.B.f32.bin", grad_b[0], manifest)
    mem_grad = metamodel.model.mem_tokens.grad
    assert mem_grad is not None
    dump(out_dir, "grad.mem_tokens.f32.bin", mem_grad, manifest)

    manifest["config"] = {
        "vocab_size": 256, "hidden_size": 64, "intermediate_size": 96,
        "num_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 16, "rms_norm_eps": 1e-6, "rope_theta": 10_000.0,
        "num_mem_token": int(config.num_mem_token),
        "lora_r": lora_r, "metalora_r": metalora_r, "scale": scale,
        "m2p_layers": 2, "m2p_heads": 4, "m2p_ffn": 128, "m2p_eps": 1e-5,
        "loss": float(loss.item()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"loss={loss.item():.6f}; goldens written to {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
