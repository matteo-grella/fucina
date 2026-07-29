#!/usr/bin/env python3
"""Golden-activation dump for the Kimi-K3 architecture port.

Runs the tiny trained reference checkpoint (inference-optimization/
Kimi-K3-0.40B) under Moonshot's own modeling code (trust-remote-code files
alongside the weights) on a fixed token sequence and writes per-component
f32 activations + a manifest, for the Zig-side golden parity tests.

Moonshot's KDA imports come from `fla` (flash-linear-attention), whose
official package needs triton (absent on macOS). Pass --fla-shim pointing
at a pure-PyTorch fla shim directory — the deltafin project's `tools/`
(github.com/gavamedia/deltafin, MIT) provides one whose chunked and
recurrent paths agree to ~1e-9.

Usage:
  .venv/bin/python tools/kimi3_goldens.py --model models/kimi-k3-0.40b \
      --fla-shim <deltafin>/tools [--out models/kimi-k3-0.40b/goldens]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

TOKEN_IDS = [1000, 2534, 77, 4096, 163, 999, 5000, 42, 31337, 7, 250, 88]


def dump_kda_op_goldens(out_dir: Path) -> None:
    """Op-level KDA recurrence goldens: deterministic synthetic inputs run
    through the reference kernel with the exact flags the Kimi models use
    (l2norm, sigmoid-beta, softplus gate all in-kernel; no lower bound)."""
    from fla.ops.kda import fused_recurrent_kda

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0x5EED)
    T, H, K, V = 17, 4, 32, 32
    q = torch.randn(1, T, H, K)
    k = torch.randn(1, T, H, K)
    v = torch.randn(1, T, H, V)
    g = torch.randn(1, T, H, K) * 0.5
    beta = torch.randn(1, T, H)
    a_log = torch.log(torch.rand(H) * 15 + 1)  # per-head form
    dt_bias = torch.randn(H * K) * 0.1

    o, state = fused_recurrent_kda(
        q=q, k=k, v=v, g=g, beta=beta, A_log=a_log, dt_bias=dt_bias,
        initial_state=None, output_final_state=True,
        use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True, lower_bound=None,
    )

    tensors = {
        "kda_q": q[0], "kda_k": k[0], "kda_v": v[0], "kda_g": g[0],
        "kda_beta": beta[0], "kda_a_log": a_log, "kda_dt_bias": dt_bias,
        "kda_o": o[0], "kda_state": state[0],
    }
    manifest = []
    for name, value in tensors.items():
        data = value.detach().to(torch.float32).contiguous()
        (out_dir / f"{name}.bin").write_bytes(data.numpy().tobytes())
        manifest.append({"name": name, "file": f"{name}.bin", "shape": list(data.shape)})
    (out_dir / "kda_op_manifest.json").write_text(json.dumps({"tensors": manifest}, indent=1))
    print(f"dumped KDA op goldens to {out_dir}: o[0,0,:4] = {o[0, 0, 0, :4].tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/kimi-k3-0.40b")
    parser.add_argument("--out", default=None)
    parser.add_argument("--fla-shim", default=None, help="dir whose fla/ package shims flash-linear-attention without triton")
    parser.add_argument("--kda-op", action="store_true", help="also dump op-level KDA recurrence goldens from synthetic inputs")
    args = parser.parse_args()

    if args.fla_shim:
        sys.path.insert(0, args.fla_shim)

    if args.kda_op:
        dump_kda_op_goldens(Path(args.out) if args.out else Path(args.model) / "goldens")
        return

    model_dir = Path(args.model)
    out_dir = Path(args.out) if args.out else model_dir / "goldens"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Moonshot's modeling files import OutputRecorder from its transformers
    # 4.x location; 5.x moved it to utils.output_capturing. Shim it back.
    import transformers.utils.generic as tug

    if not hasattr(tug, "OutputRecorder"):
        from transformers.utils.output_capturing import OutputRecorder

        tug.OutputRecorder = OutputRecorder

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()

    # The modeling code force-selects flash_attention_2 at construction (a
    # GPU deployment assumption); the choice is re-read at forward time, so
    # flip every config back to eager for the CPU reference run.
    def force_eager(cfg) -> None:
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = "eager"

    force_eager(model.config)
    force_eager(getattr(model.config, "text_config", None))
    for module in model.modules():
        force_eager(getattr(module, "config", None))

    dumps: dict[str, torch.Tensor] = {}

    def save(name: str, value):
        if isinstance(value, tuple):
            value = value[0]
        if not isinstance(value, torch.Tensor):
            return
        dumps[name] = value.detach().to(torch.float32).contiguous()

    hooks = []
    for name, module in model.named_modules():
        cls = type(module).__name__
        # Layer boundaries + the components the port verifies individually.
        interesting = (
            name.endswith("embed_tokens")
            or cls in ("KimiDeltaAttention", "KimiLinearDecoderLayer")
            or "self_attn" in name.split(".")[-1:]
            or name.split(".")[-1] in ("linear_attn", "mlp", "input_layernorm", "norm")
            or cls.endswith("MLA")
            or cls.endswith("MoE")
        )
        if not interesting:
            continue
        hooks.append(
            module.register_forward_hook(
                lambda mod, inp, outp, n=name: save(n, outp)
            )
        )

    input_ids = torch.tensor([TOKEN_IDS], dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=input_ids)
    for h in hooks:
        h.remove()

    save("logits", out.logits)

    manifest = []
    for name, tensor in dumps.items():
        file_name = name.replace(".", "_") + ".bin"
        (out_dir / file_name).write_bytes(tensor.numpy().tobytes())
        manifest.append({"name": name, "file": file_name, "shape": list(tensor.shape)})
    (out_dir / "manifest.json").write_text(
        json.dumps({"token_ids": TOKEN_IDS, "tensors": manifest}, indent=1)
    )
    print(f"dumped {len(manifest)} tensors to {out_dir}")
    print("logits[0,-1,:8] =", out.logits[0, -1, :8].tolist())


if __name__ == "__main__":
    main()
