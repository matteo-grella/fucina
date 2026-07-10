#!/usr/bin/env python3
"""Dump numeric parity oracles from the reference karpathy/nanochat model.

This CLI builds the reference GPT deterministically (torch.manual_seed(42),
meta-device construction, to_empty, init_weights — exactly like base_train.py),
runs it in float32 on CPU, and dumps init weights, forward activations,
gradients, optimizer steps, a short loss trace and a greedy decode so the Zig
port can be checked numerically against the exact reference math.

Binary layouts (fixed_batch.bin, trace_batches.bin) are defined in the port's
FORMATS.md ("oracle dumps" section). Tensor dumps use safetensors (float32,
torch state_dict key names).

Run with the reference nanochat venv:

  NANOCHAT_BASE_DIR=$HOME/.cache/nanochat \
  PYTHONPATH=/path/to/refs/nanochat \
  NANOCHAT_DTYPE=float32 TORCHDYNAMO_DISABLE=1 \
    /path/to/refs/nanochat/.venv/bin/python nanochat_dump.py dump-init --config d6 --out init_d6.safetensors
"""

import os

# These must be set before torch / nanochat imports:
# - float32 compute is the CPU parity target (embeddings + all activations stay fp32,
#   COMPUTE_DTYPE is resolved at nanochat.common import time)
# - disabling dynamo makes the @torch.compile fused optimizer kernels (adamw_step_fused,
#   muon_step_fused) run in clean eager mode = the exact math the Zig port replicates
_dtype = os.environ.setdefault("NANOCHAT_DTYPE", "float32")
assert _dtype == "float32", f"NANOCHAT_DTYPE must be float32 for parity dumps, got {_dtype}"
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import argparse
import json
import math
import struct
import sys
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from nanochat.common import COMPUTE_DTYPE
from nanochat.flash_attention import flash_attn
from nanochat.gpt import GPT, GPTConfig, apply_rotary_emb, norm

assert COMPUTE_DTYPE == torch.float32, f"COMPUTE_DTYPE must be float32, got {COMPUTE_DTYPE}"

# -----------------------------------------------------------------------------
# Model configs

CONFIGS = {
    # the real CPU-demo config from runs/runcpu.sh: depth=6, head_dim=64, aspect_ratio=64
    "d6": dict(sequence_len=512, vocab_size=32768, n_layer=6, n_head=6, n_kv_head=6,
               n_embd=384, window_pattern="L"),
    # tiny config for fast Zig gradcheck
    "d2": dict(sequence_len=32, vocab_size=256, n_layer=2, n_head=2, n_kv_head=2,
               n_embd=128, window_pattern="L"),
}

# default (B, T) oracle batch sizes per config
ORACLE_BT = {"d6": (2, 64), "d2": (2, 16)}

# fixed hyperparameters for the optimizer oracles (recorded in the schedule json);
# CLI defaults of base_train.py + the runcpu total_batch_size
TOTAL_BATCH_SIZE = 16384
B_REF = 2 ** 19
TARGET_PARAM_DATA_RATIO = 12.0
EMBEDDING_LR = 0.3
UNEMBEDDING_LR = 0.008
MATRIX_LR = 0.02
SCALAR_LR = 0.5
WEIGHT_DECAY = 0.28
# fixed schedule horizon so the schedules match the real run (we only take a few steps,
# so warmup dominates; per-step multipliers are recorded in the schedule json)
NUM_ITERATIONS = 5000
WARMUP_STEPS = 40
WARMDOWN_RATIO = 0.65
FINAL_LR_FRAC = 0.05


def build_model(config_name):
    """Deterministic model build, exactly like base_train.py: seed 42, meta-device
    construction, to_empty(cpu), init_weights()."""
    cfg = GPTConfig(**CONFIGS[config_name])
    torch.manual_seed(42)
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device="cpu")
    model.init_weights()
    return model, cfg


def load_init(model, path):
    sd = load_file(path)
    model.load_state_dict(sd, strict=True, assign=True)


# -----------------------------------------------------------------------------
# rms_norm eps discovery
#
# gpt.py's norm() is F.rms_norm(x, (d,)) with eps unspecified. We determine the
# eps torch actually uses empirically: with a tiny-magnitude constant vector x,
# mean(x^2) is negligible vs eps, so from y = x/sqrt(mean(x^2)+eps) we can solve
# eps = (x/y)^2 - mean(x^2) in float64. We then snap to the nearest canonical
# candidate and verify bitwise against manual recomputation.


def discover_rms_norm_eps():
    d = 16
    x = torch.full((d,), 1e-8, dtype=torch.float32)
    y = F.rms_norm(x, (d,))
    ms = float((x.double() ** 2).mean())
    eps_est = float((((x.double() / y.double()) ** 2) - ms).mean())

    candidates = {
        "torch.finfo(float32).eps": float(torch.finfo(torch.float32).eps),
        "torch.finfo(float64).eps": float(torch.finfo(torch.float64).eps),
        "1e-5": 1e-5,
        "1e-6": 1e-6,
        "zero": 0.0,
    }
    best_name = min(candidates, key=lambda k: abs(candidates[k] - eps_est))
    eps = candidates[best_name]

    # bitwise cross-check on a generic tensor, both rsqrt and div formulations
    g = torch.Generator().manual_seed(1234)
    xg = torch.randn(8, 64, generator=g, dtype=torch.float32)
    ref = F.rms_norm(xg, (64,))
    man_rsqrt = xg * torch.rsqrt(xg.pow(2).mean(-1, keepdim=True) + eps)
    man_div = xg / torch.sqrt(xg.pow(2).mean(-1, keepdim=True) + eps)
    info = {
        "rms_norm_eps": eps,
        "empirical_estimate": eps_est,
        "nearest_candidate": best_name,
        "relative_error_vs_candidate": abs(eps_est - eps) / eps if eps > 0 else None,
        "bitwise_match_x_mul_rsqrt": bool(torch.equal(man_rsqrt, ref)),
        "bitwise_match_x_div_sqrt": bool(torch.equal(man_div, ref)),
        "max_abs_diff_x_mul_rsqrt": float((man_rsqrt - ref).abs().max()),
        "max_abs_diff_x_div_sqrt": float((man_div - ref).abs().max()),
        "method": "solved eps=(x/y)^2-mean(x^2) in float64 from F.rms_norm on a 1e-8 "
                  "constant vector (eps dominates the denominator), snapped to nearest "
                  "canonical candidate, verified bitwise on randn",
    }
    return info


# -----------------------------------------------------------------------------
# Batch construction and binary I/O (layouts per FORMATS.md)


def val_token_stream(n_tokens, vocab_size):
    """Deterministic token stream from real val data: for each val doc in reference
    iteration order, bos + encode_ordinary(doc), concatenated. For tiny-vocab configs
    (d2, vocab 256) ids are reduced mod vocab_size so they stay in range."""
    from nanochat.dataset import parquets_iter_batched
    from nanochat.tokenizer import get_tokenizer

    tok = get_tokenizer()
    stream = []
    for batch in parquets_iter_batched("val"):
        for doc in batch:
            ids = tok.encode(doc, prepend="<|bos|>")  # bos + encode_ordinary
            stream.extend(ids)
            if len(stream) >= n_tokens:
                break
        if len(stream) >= n_tokens:
            break
    assert len(stream) >= n_tokens, f"val stream too short: {len(stream)} < {n_tokens}"
    if vocab_size < tok.get_vocab_size():
        stream = [t % vocab_size for t in stream]
    return stream[:n_tokens]


def resolve_bt(args):
    def_b, def_t = ORACLE_BT[args.config]
    B = args.B if args.B is not None else def_b
    T = args.T if args.T is not None else def_t
    return B, T


def write_fixed_batch(path, inputs, targets):
    """fixed_batch.bin: u32 B, u32 T, B*T u32 inputs, B*T i32 targets."""
    B, T = inputs.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<II", B, T))
        f.write(np.ascontiguousarray(inputs, dtype="<u4").tobytes())
        f.write(np.ascontiguousarray(targets, dtype="<i4").tobytes())


def read_fixed_batch(path):
    with open(path, "rb") as f:
        data = f.read()
    B, T = struct.unpack_from("<II", data, 0)
    off = 8
    inputs = np.frombuffer(data, dtype="<u4", count=B * T, offset=off).astype(np.int64).reshape(B, T)
    off += 4 * B * T
    targets = np.frombuffer(data, dtype="<i4", count=B * T, offset=off).astype(np.int64).reshape(B, T)
    assert off + 4 * B * T == len(data), "trailing bytes in batch file"
    return torch.from_numpy(inputs), torch.from_numpy(targets)


def cmd_make_batch(args):
    vocab_size = CONFIGS[args.config]["vocab_size"]
    B, T = resolve_bt(args)
    stream = val_token_stream(B * (T + 1), vocab_size)
    rows = np.array(stream, dtype=np.int64).reshape(B, T + 1)
    inputs = rows[:, :-1].copy()
    targets = rows[:, 1:].copy()
    targets[0, -2:] = -1  # exercise ignore_index
    assert inputs.min() >= 0 and inputs.max() < vocab_size
    write_fixed_batch(args.out, inputs, targets)
    print(f"wrote {args.out}")
    print(f"B={B} T={T} vocab={vocab_size} inputs[0,:8]={inputs[0, :8].tolist()} "
          f"targets[0,-4:]={targets[0, -4:].tolist()}")


# -----------------------------------------------------------------------------
# dump-init


def cmd_dump_init(args):
    model, cfg = build_model(args.config)
    tensors = {}
    for k, v in model.state_dict().items():
        assert v.dtype == torch.float32, f"{k} is {v.dtype}, expected float32"
        tensors[k] = v.detach().clone().contiguous()
    save_file(tensors, args.out)

    eps_info = discover_rms_norm_eps()
    config_json = {
        **asdict(cfg),
        "padded_vocab_size": int(model.transformer.wte.weight.shape[0]),
        "head_dim": cfg.n_embd // cfg.n_head,
        "window_sizes": [list(w) for w in model.window_sizes],
        "rotary_base": 100000,
        "rotary_seq_len": model.rotary_seq_len,
        "compute_dtype": "float32",
        "rms_norm_eps": eps_info["rms_norm_eps"],
        "rms_norm_eps_discovery": eps_info,
    }
    cfg_path = args.out[:-len(".safetensors")] + ".config.json" if args.out.endswith(".safetensors") else args.out + ".config.json"
    with open(cfg_path, "w") as f:
        json.dump(config_json, f, indent=2)

    n_params = sum(v.numel() for v in tensors.values())
    print(f"wrote {args.out} ({len(tensors)} tensors, {n_params:,} params)")
    print(f"wrote {cfg_path}")
    print(f"rms_norm_eps={eps_info['rms_norm_eps']!r} (estimate {eps_info['empirical_estimate']:.9e}, "
          f"nearest {eps_info['nearest_candidate']}, "
          f"bitwise rsqrt={eps_info['bitwise_match_x_mul_rsqrt']} div={eps_info['bitwise_match_x_div_sqrt']})")


# -----------------------------------------------------------------------------
# dump-forward: traced forward replicating gpt.py GPT.forward exactly (kv_cache=None
# path) with intermediate captures. Hooks are not used because they cannot see
# intra-forward tensors (resid_in, ve_gate, pre-softcap logits, ...).


def traced_attention(attn, x, ve, cos_sin, window_size, cap, layer_idx):
    """Replicates CausalSelfAttention.forward (training path), capturing ve_gate."""
    B, T, C = x.size()
    q = attn.c_q(x).view(B, T, attn.n_head, attn.head_dim)
    k = attn.c_k(x).view(B, T, attn.n_kv_head, attn.head_dim)
    v = attn.c_v(x).view(B, T, attn.n_kv_head, attn.head_dim)
    if ve is not None:
        ve = ve.view(B, T, attn.n_kv_head, attn.head_dim)
        gate = 3 * torch.sigmoid(attn.ve_gate(x[..., :attn.ve_gate_channels]))
        cap[f"ve_gate.{layer_idx}"] = gate
        v = v + gate.unsqueeze(-1) * ve
    cos, sin = cos_sin
    q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
    q, k = norm(q), norm(k)
    q = q * 1.2
    k = k * 1.2
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
    y = y.contiguous().view(B, T, -1)
    y = attn.c_proj(y)
    return y


@torch.no_grad()
def traced_forward(model, idx, targets):
    """Replicates GPT.forward (kv_cache=None) with captures. Returns (captures,
    mean loss, per-token loss)."""
    cap = {}
    B, T = idx.size()
    assert T <= model.cos.size(1)
    cos_sin = model.cos[:, :T], model.sin[:, :T]

    x = model.transformer.wte(idx)
    x = x.to(COMPUTE_DTYPE)
    x = norm(x)
    cap["emb_norm"] = x

    assert T > 1, "Training forward pass should have T > 1"
    gate = model.smear_lambda.to(x.dtype) * torch.sigmoid(model.smear_gate(x[:, 1:, :24]))
    x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
    cap["post_smear"] = x

    x0 = x
    cap["x0"] = x0
    n_layer = model.config.n_layer
    backout_layer = n_layer // 2
    x_backout = None
    for i, block in enumerate(model.transformer.h):
        x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
        cap[f"resid_in.{i}"] = x
        ve = model.value_embeds[str(i)](idx).to(x.dtype) if str(i) in model.value_embeds else None
        attn_out = traced_attention(block.attn, norm(x), ve, cos_sin, model.window_sizes[i], cap, i)
        cap[f"attn_out.{i}"] = attn_out
        x = x + attn_out
        mlp_out = block.mlp(norm(x))
        cap[f"mlp_out.{i}"] = mlp_out
        x = x + mlp_out
        cap[f"block_out.{i}"] = x
        if i == backout_layer:
            x_backout = x
    if x_backout is not None:
        cap["x_backout"] = x_backout
        x = x - model.backout_lambda.to(x.dtype) * x_backout
    cap["pre_final_norm"] = x
    x = norm(x)
    cap["post_final_norm"] = x

    softcap = 15
    logits = model.lm_head(x)
    logits = logits[..., : model.config.vocab_size]
    logits = logits.float()
    cap["logits_pre_softcap"] = logits
    logits = softcap * torch.tanh(logits / softcap)
    cap["logits_post_softcap"] = logits

    flat_logits = logits.view(-1, logits.size(-1))
    flat_targets = targets.view(-1)
    loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=-1, reduction="mean")
    cap["loss"] = loss.reshape(1)
    loss_none = F.cross_entropy(flat_logits, flat_targets, ignore_index=-1, reduction="none").view(B, T)
    return cap, loss, loss_none


def cmd_dump_forward(args):
    model, cfg = build_model(args.config)
    load_init(model, args.init)
    inputs, targets = read_fixed_batch(args.batch)

    cap, traced_loss, loss_none = traced_forward(model, inputs, targets)
    with torch.no_grad():
        stock_loss = model(inputs, targets)
    diff = abs(stock_loss.item() - traced_loss.item())
    assert diff < 1e-6, f"traced forward diverged from stock forward: {diff}"

    tensors = {}
    for k, v in cap.items():
        assert v.dtype == torch.float32, f"{k} is {v.dtype}"
        tensors[k] = v.detach().clone().contiguous()
    if args.loss_reduction == "none":
        tensors["loss_none"] = loss_none.detach().clone().contiguous()
    save_file(tensors, args.out)

    print(f"wrote {args.out} ({len(tensors)} tensors)")
    print(f"traced loss = {traced_loss.item():.9f}")
    print(f"stock  loss = {stock_loss.item():.9f}")
    print(f"|traced - stock| = {diff:.3e} (< 1e-6 OK)")


# -----------------------------------------------------------------------------
# dump-grad


def cmd_dump_grad(args):
    model, cfg = build_model(args.config)
    load_init(model, args.init)
    inputs, targets = read_fixed_batch(args.batch)

    loss = model(inputs, targets)
    loss.backward()

    tensors = {}
    for name, p in model.named_parameters():
        assert p.grad is not None, f"param {name} has no grad"
        assert p.grad.dtype == torch.float32
        tensors[name] = p.grad.detach().clone().contiguous()
    tensors["loss"] = loss.detach().reshape(1).clone()
    save_file(tensors, args.out)
    print(f"wrote {args.out} ({len(tensors)} tensors)")
    print(f"loss = {loss.item():.9f}")


# -----------------------------------------------------------------------------
# Optimizer setup replicating base_train.py's hyperparameter derivation with
# FIXED, recorded values (total_batch_size=16384, default CLI LRs).


def get_scaling_params(m):
    # replicates scripts.base_train.get_scaling_params
    pc = m.num_scaling_params()
    return pc["transformer_matrices"] + pc["lm_head"]


def derive_hparams(model, cfg):
    batch_lr_scale = (TOTAL_BATCH_SIZE / B_REF) ** 0.5
    num_scaling = get_scaling_params(model)
    target_tokens = int(TARGET_PARAM_DATA_RATIO * num_scaling)
    # D_ref from a meta d12 built like base_train.build_model_meta(12) with
    # head_dim=64/aspect_ratio=64 => n_embd=768, n_head=n_kv_head=12. We reuse the
    # oracle config's sequence_len/vocab_size (they do not affect transformer_matrices,
    # only lm_head via vocab; recorded numerically below either way).
    d12_cfg = GPTConfig(sequence_len=cfg.sequence_len, vocab_size=cfg.vocab_size,
                        n_layer=12, n_head=12, n_kv_head=12, n_embd=768, window_pattern="L")
    with torch.device("meta"):
        d12 = GPT(d12_cfg)
    d12_scaling = get_scaling_params(d12)
    d_ref = TARGET_PARAM_DATA_RATIO * d12_scaling
    weight_decay_scaled = WEIGHT_DECAY * math.sqrt(TOTAL_BATCH_SIZE / B_REF) * (d_ref / target_tokens)
    return {
        "total_batch_size": TOTAL_BATCH_SIZE,
        "B_REF": B_REF,
        "target_param_data_ratio": TARGET_PARAM_DATA_RATIO,
        "batch_lr_scale": batch_lr_scale,
        "dmodel_lr_scale": (cfg.n_embd / 768) ** -0.5,
        "num_scaling_params": num_scaling,
        "target_tokens": target_tokens,
        "d12_ref_scaling_params": d12_scaling,
        "D_ref": d_ref,
        "embedding_lr": EMBEDDING_LR,
        "unembedding_lr": UNEMBEDDING_LR,
        "matrix_lr": MATRIX_LR,
        "scalar_lr": SCALAR_LR,
        "weight_decay": WEIGHT_DECAY,
        "weight_decay_scaled": weight_decay_scaled,
        "num_iterations": NUM_ITERATIONS,
        "warmup_steps": WARMUP_STEPS,
        "warmdown_ratio": WARMDOWN_RATIO,
        "final_lr_frac": FINAL_LR_FRAC,
        "grad_accum_steps": 1,
    }


def make_optimizer(model, hp):
    # exactly base_train.py's setup_optimizer call with batch_lr_scale applied
    bls = hp["batch_lr_scale"]
    return model.setup_optimizer(
        unembedding_lr=UNEMBEDDING_LR * bls,
        embedding_lr=EMBEDDING_LR * bls,
        scalar_lr=SCALAR_LR * bls,
        matrix_lr=MATRIX_LR * bls,
        weight_decay=hp["weight_decay_scaled"],
    )


# schedules: base_train.py's get_lr_multiplier / get_muon_momentum / get_weight_decay
# with num_iterations fixed to NUM_ITERATIONS


def get_lr_multiplier(it):
    warmdown_iters = round(WARMDOWN_RATIO * NUM_ITERATIONS)
    if it < WARMUP_STEPS:
        return (it + 1) / WARMUP_STEPS
    elif it <= NUM_ITERATIONS - warmdown_iters:
        return 1.0
    else:
        progress = (NUM_ITERATIONS - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * FINAL_LR_FRAC


def get_muon_momentum(it):
    warmdown_iters = round(WARMDOWN_RATIO * NUM_ITERATIONS)
    warmdown_start = NUM_ITERATIONS - warmdown_iters
    if it < 400:
        frac = it / 400
        return (1 - frac) * 0.85 + frac * 0.97
    elif it >= warmdown_start:
        progress = (it - warmdown_start) / warmdown_iters
        return 0.97 * (1 - progress) + 0.90 * progress
    else:
        return 0.97


def get_weight_decay(it, weight_decay_scaled):
    return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * it / NUM_ITERATIONS))


def train_step(model, optimizer, hp, step, inputs, targets):
    """One training step exactly like base_train.py's loop (grad_accum_steps=1):
    forward, backward, apply per-step schedule multipliers, optimizer step, zero grad.
    Returns (pre-step loss, lrm, momentum, weight_decay)."""
    loss = model(inputs, targets)
    loss.backward()
    lrm = get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step, hp["weight_decay_scaled"])
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group["kind"] == "muon":
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    optimizer.step()
    model.zero_grad(set_to_none=True)
    return loss.item(), lrm, muon_momentum, muon_weight_decay


def param_names_by_id(model):
    return {id(p): n for n, p in model.named_parameters()}


def group_metadata(model, optimizer):
    name_of = param_names_by_id(model)
    groups = []
    for gi, group in enumerate(optimizer.param_groups):
        g = {
            "index": gi,
            "kind": group["kind"],
            "initial_lr": group["initial_lr"],
            "weight_decay": group["weight_decay"],
            "params": [name_of[id(p)] for p in group["params"]],
        }
        if group["kind"] == "adamw":
            g["betas"] = list(group["betas"])
            g["eps"] = group["eps"]
        else:
            shape = list(group["params"][0].shape)
            g["momentum"] = group["momentum"]
            g["ns_steps"] = group["ns_steps"]
            g["beta2"] = group["beta2"]
            g["shape"] = shape
            g["lr_shape_factor"] = max(1.0, shape[-2] / shape[-1]) ** 0.5
        groups.append(g)
    return groups


def dump_params_and_opt_state(model, optimizer, path):
    """Params + optimizer state: adamw exp_avg/exp_avg_sq per param, muon
    momentum_buffer/second_momentum_buffer keyed by the group's first param name."""
    name_of = param_names_by_id(model)
    tensors = {}
    for name, p in model.named_parameters():
        tensors[name] = p.detach().clone().contiguous()
    for group in optimizer.param_groups:
        if group["kind"] == "adamw":
            for p in group["params"]:
                st = optimizer.state[p]
                n = name_of[id(p)]
                tensors[f"{n}.exp_avg"] = st["exp_avg"].detach().clone().contiguous()
                tensors[f"{n}.exp_avg_sq"] = st["exp_avg_sq"].detach().clone().contiguous()
        else:
            p0 = group["params"][0]
            st = optimizer.state[p0]
            n = name_of[id(p0)]
            tensors[f"{n}.momentum_buffer"] = st["momentum_buffer"].detach().clone().contiguous()
            tensors[f"{n}.second_momentum_buffer"] = st["second_momentum_buffer"].detach().clone().contiguous()
    for k, v in tensors.items():
        assert v.dtype == torch.float32, f"{k} is {v.dtype}"
    save_file(tensors, path)
    return len(tensors)


def cmd_dump_optsteps(args):
    model, cfg = build_model(args.config)
    load_init(model, args.init)
    inputs, targets = read_fixed_batch(args.batch)

    hp = derive_hparams(model, cfg)
    optimizer = make_optimizer(model, hp)

    dump_steps = sorted(set(int(s) for s in args.steps.split(",")))
    assert dump_steps and dump_steps[0] >= 1
    max_steps = dump_steps[-1]

    schedule_rows = []
    dumped_files = {}
    for step in range(max_steps):  # 0-based schedule step, like base_train.py
        loss, lrm, mom, wd = train_step(model, optimizer, hp, step, inputs, targets)
        schedule_rows.append({
            "step": step,
            "completed_steps": step + 1,
            "loss": loss,
            "lrm": lrm,
            "muon_momentum": mom,
            "muon_weight_decay": wd,
            "group_lr": [g["lr"] for g in optimizer.param_groups],
        })
        completed = step + 1
        if completed in dump_steps:
            path = f"{args.out_prefix}_{completed}.safetensors"
            n = dump_params_and_opt_state(model, optimizer, path)
            dumped_files[completed] = os.path.basename(path)
            print(f"wrote {path} ({n} tensors) after {completed} steps, pre-step loss {loss:.9f}")

    schedule = {
        "config": args.config,
        "hyperparameters": hp,
        "groups": group_metadata(model, optimizer),
        "steps": schedule_rows,
        "dumped_after_steps": dump_steps,
        "dumped_files": dumped_files,
    }
    sched_path = f"{args.out_prefix}_schedule.json"
    with open(sched_path, "w") as f:
        json.dump(schedule, f, indent=2)
    print(f"wrote {sched_path}")


# -----------------------------------------------------------------------------
# dump-loss-trace


def cmd_dump_loss_trace(args):
    model, cfg = build_model(args.config)
    load_init(model, args.init)
    B, T = resolve_bt(args)
    n_steps = args.steps

    stream = val_token_stream(n_steps * B * (T + 1), cfg.vocab_size)
    rows = np.array(stream, dtype=np.int64).reshape(n_steps, B, T + 1)
    inputs_all = rows[:, :, :-1].copy()
    targets_all = rows[:, :, 1:].copy()

    # trace_batches.bin: u32 n_steps, u32 B, u32 T, then per step the fixed_batch
    # payload (B*T u32 inputs, B*T i32 targets) without the per-batch header
    with open(args.batches_out, "wb") as f:
        f.write(struct.pack("<III", n_steps, B, T))
        for s in range(n_steps):
            f.write(np.ascontiguousarray(inputs_all[s], dtype="<u4").tobytes())
            f.write(np.ascontiguousarray(targets_all[s], dtype="<i4").tobytes())

    hp = derive_hparams(model, cfg)
    optimizer = make_optimizer(model, hp)

    losses = []
    for step in range(n_steps):
        inputs = torch.from_numpy(inputs_all[step])
        targets = torch.from_numpy(targets_all[step])
        loss, lrm, mom, wd = train_step(model, optimizer, hp, step, inputs, targets)
        losses.append(loss)

    with open(args.out, "w") as f:
        json.dump(losses, f, indent=2)
    meta_path = args.out[:-len(".json")] + ".meta.json" if args.out.endswith(".json") else args.out + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump({"config": args.config, "n_steps": n_steps, "B": B, "T": T,
                   "hyperparameters": hp, "batches_file": os.path.basename(args.batches_out),
                   "loss_semantics": "pre-step mean loss on that step's batch"}, f, indent=2)

    print(f"wrote {args.batches_out}")
    print(f"wrote {args.out}")
    print(f"wrote {meta_path}")
    print(f"first 5 losses: {losses[:5]}")
    print(f"last loss: {losses[-1]:.9f}")


# -----------------------------------------------------------------------------
# dump-sft-trace: supervised fine-tuning loss trace, replicating scripts/chat_sft.py
# on a FIXED N-step deterministic trace (the SFT parity oracle).
#
# Starts from a *trained* base checkpoint (base_ckpt_d6_step2500.safetensors),
# builds the SFT optimizer via model.setup_optimizer with the base CLI LRs ×
# init_lr_frac (0.8), weight_decay=0.0, and NO optimizer warm-start (chat_sft.py's
# --load-optimizer=0 path: cross-framework optimizer-state import is out of scope,
# so a FRESH optimizer keeps the Zig parity clean). It then runs the progress-based
# SFT schedule (get_lr_multiplier with warmup_ratio 0 / warmdown_ratio 0.5 /
# final_lr_frac 0, get_muon_momentum 0.85→0.95 over 300 steps, weight_decay 0)
# over deterministic masked batches produced by chat_sft.py's SFT data generator.

# Base pretrain CLI LRs the d6 checkpoint was trained with (base_train.py defaults /
# runcpu; chat_sft.py inherits these from the checkpoint meta's user_config).
SFT_EMBEDDING_LR = 0.3
SFT_UNEMBEDDING_LR = 0.008
SFT_MATRIX_LR = 0.02
# init_lr_frac: chat_sft.py's production default is 0.8, but that LR (which folds
# in dmodel_lr_scale but NOT batch_lr_scale — SFT is tuned for the 524288-token
# production batch) is far too hot for this tiny CPU trace batch and diverges.
# 0.2 keeps the small-batch trajectory finite and gently-varying so it is an
# informative parity target. Recorded in sft_schedule_d6.json.
SFT_INIT_LR_FRAC = 0.2
SFT_WARMUP_RATIO = 0.0
SFT_WARMDOWN_RATIO = 0.5
SFT_FINAL_LR_FRAC = 0.0
# default SFT trace batch shape (device_batch_size=2, max_seq_len=512). T=512 is
# the d6 model's full context; conversations are rendered truncated to T so rows
# pack densely with supervised tokens (the mixture's median conversation is ~764
# tokens, longer than the d6 context — at shorter T most rows would be all-padding
# → all targets -1 → a 0/0 masked mean = NaN).
SFT_ORACLE_BT = (2, 512)


def read_sft_mixture_jsonl(path):
    """Read the committed SFT val mixture (mixture-ordered conversations). Each
    line is {"messages": [...]} — exactly what render_conversation consumes."""
    convs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            convs.append(json.loads(line))
    assert convs, f"empty SFT mixture: {path}"
    return convs


def sft_data_generator_trunc(tok, dataset, device_batch_size, max_seq_len,
                             render_max_tokens, buffer_size=100):
    """chat_sft.py's sft_data_generator_bos_bestfit (single-device val path, the
    faithful transcription in nanochat_export._sft_generator) with ONE knob added:
    render_conversation's max_tokens. Truncating each conversation to the model
    context guarantees the best-fit packing always seats a (possibly truncated)
    conversation with supervised tokens in every row (never a fully-padded row)."""
    import torch
    dataset_size = len(dataset)
    assert dataset_size > 0
    row_capacity = max_seq_len + 1
    bos_token = tok.get_bos_token_id()
    conv_buffer = []
    cursor = 0

    def refill_buffer():
        nonlocal cursor
        while len(conv_buffer) < buffer_size:
            ids, mask = tok.render_conversation(dataset[cursor], max_tokens=render_max_tokens)
            conv_buffer.append((ids, mask))
            cursor = (cursor + 1) % dataset_size

    while True:
        rows, mask_rows, row_lengths = [], [], []
        for _ in range(device_batch_size):
            row, mask_row, padded, content_len = [], [], False, row_capacity
            while len(row) < row_capacity:
                while len(conv_buffer) < buffer_size:
                    refill_buffer()
                remaining = row_capacity - len(row)
                best_idx, best_len = -1, 0
                for i, (conv, _) in enumerate(conv_buffer):
                    if len(conv) <= remaining and len(conv) > best_len:
                        best_idx, best_len = i, len(conv)
                if best_idx >= 0:
                    conv, conv_mask = conv_buffer.pop(best_idx)
                    row.extend(conv)
                    mask_row.extend(conv_mask)
                else:
                    content_len = len(row)
                    row.extend([bos_token] * remaining)
                    mask_row.extend([0] * remaining)
                    padded = True
                    break
            row_lengths.append(content_len if padded else row_capacity)
            rows.append(row[:row_capacity])
            mask_rows.append(mask_row[:row_capacity])
        batch_tensor = torch.tensor(rows, dtype=torch.long)
        inputs = batch_tensor[:, :-1].to(dtype=torch.int32).contiguous()
        targets = batch_tensor[:, 1:].to(dtype=torch.int64).contiguous()
        mask_tensor = torch.tensor(mask_rows, dtype=torch.int8)
        targets[mask_tensor[:, 1:] == 0] = -1
        for i, content_len in enumerate(row_lengths):
            if content_len < row_capacity:
                targets[i, content_len - 1:] = -1
        yield inputs, targets


def make_sft_optimizer(model, embedding_lr, unembedding_lr, matrix_lr, init_lr_frac):
    """chat_sft.py optimizer setup: setup_optimizer(base LRs, wd=0.0) — which folds
    in dmodel_lr_scale but NOT batch_lr_scale (SFT differs from base_train here) —
    then override each group's LR to lr × init_lr_frac and pin it as initial_lr."""
    optimizer = model.setup_optimizer(
        unembedding_lr=unembedding_lr,
        embedding_lr=embedding_lr,
        matrix_lr=matrix_lr,
        weight_decay=0.0,
    )
    for group in optimizer.param_groups:
        group["lr"] = group["lr"] * init_lr_frac
        group["initial_lr"] = group["lr"]
    return optimizer


def sft_lr_multiplier(progress, warmup_ratio, warmdown_ratio, final_lr_frac):
    # chat_sft.py get_lr_multiplier (progress-based, not absolute step counts).
    if progress < warmup_ratio:
        return (progress + 1e-8) / warmup_ratio
    elif progress <= 1.0 - warmdown_ratio:
        return 1.0
    else:
        decay = (progress - (1.0 - warmdown_ratio)) / warmdown_ratio
        return (1 - decay) * 1.0 + decay * final_lr_frac


def sft_muon_momentum(it):
    # chat_sft.py get_muon_momentum: 0.85→0.95 over the first 300 steps, then hold.
    frac = min(it / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95


def sft_train_step(model, optimizer, step, n_steps, inputs, targets,
                   warmup_ratio, warmdown_ratio, final_lr_frac):
    """One SFT step like chat_sft.py's loop (grad_accum_steps=1): masked-mean CE
    forward, backward, apply per-step SFT schedule, optimizer step, zero grad.
    Deterministic progress = step / n_steps (num_iterations fixed to the trace
    length). Returns the PRE-step masked mean loss + schedule scalars."""
    loss = model(inputs, targets)  # F.cross_entropy(..., ignore_index=-1, mean)
    loss.backward()
    progress = step / n_steps
    lrm = sft_lr_multiplier(progress, warmup_ratio, warmdown_ratio, final_lr_frac)
    muon_momentum = sft_muon_momentum(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group["kind"] == "muon":
            group["momentum"] = muon_momentum
            # weight_decay stays 0.0 (chat_sft.py: no wd schedule in SFT)
    optimizer.step()
    model.zero_grad(set_to_none=True)
    return loss.item(), lrm, muon_momentum, progress


def cmd_dump_sft_trace(args):
    from nanochat.tokenizer import get_tokenizer

    model, cfg = build_model(args.config)
    load_init(model, args.init)  # trained base checkpoint (step 2500)
    tok = get_tokenizer()

    B, T = SFT_ORACLE_BT
    if args.B is not None:
        B = args.B
    if args.T is not None:
        T = args.T
    render_max_tokens = args.render_max_tokens if args.render_max_tokens is not None else T
    n_steps = args.steps

    # Deterministic masked (inputs, targets) batches from the committed val mixture.
    dataset = read_sft_mixture_jsonl(args.mixture)
    gen = sft_data_generator_trunc(tok, dataset, B, T, render_max_tokens)

    # trace_batches_sft_d6.bin: u32 n_steps, u32 B, u32 T, then per step the
    # payload B*T i32 inputs + B*T i32 targets (-1 = ignore_index), matching the
    # base loss-trace batch layout so the Zig gate reuses its reader.
    batches = []
    with open(args.batches_out, "wb") as f:
        f.write(struct.pack("<III", n_steps, B, T))
        for _ in range(n_steps):
            inputs, targets = next(gen)  # int32 (B,T), int64 (B,T) with -1
            f.write(np.ascontiguousarray(inputs.cpu().numpy(), dtype="<i4").tobytes())
            f.write(np.ascontiguousarray(targets.cpu().numpy(), dtype="<i4").tobytes())
            batches.append((inputs, targets))

    optimizer = make_sft_optimizer(model, args.embedding_lr, args.unembedding_lr,
                                   args.matrix_lr, args.init_lr_frac)

    losses = []
    schedule_rows = []
    for step in range(n_steps):
        inputs, targets = batches[step]
        loss, lrm, mom, progress = sft_train_step(
            model, optimizer, step, n_steps, inputs, targets,
            args.warmup_ratio, args.warmdown_ratio, args.final_lr_frac)
        losses.append(loss)
        schedule_rows.append({
            "step": step,
            "progress": progress,
            "loss": loss,
            "lrm": lrm,
            "muon_momentum": mom,
            "group_lr": [g["lr"] for g in optimizer.param_groups],
        })

    with open(args.out, "w") as f:
        json.dump(losses, f, indent=2)

    schedule = {
        "config": args.config,
        "n_steps": n_steps,
        "B": B,
        "T": T,
        "render_max_tokens": render_max_tokens,
        "init_from": os.path.basename(args.init),
        "mixture": os.path.basename(args.mixture),
        "load_optimizer": 0,
        "hyperparameters": {
            "embedding_lr": args.embedding_lr,
            "unembedding_lr": args.unembedding_lr,
            "matrix_lr": args.matrix_lr,
            "scalar_lr": 0.5,
            "init_lr_frac": args.init_lr_frac,
            "warmup_ratio": args.warmup_ratio,
            "warmdown_ratio": args.warmdown_ratio,
            "final_lr_frac": args.final_lr_frac,
            "weight_decay": 0.0,
            "dmodel_lr_scale": (cfg.n_embd / 768) ** -0.5,
        },
        "groups": group_metadata(model, optimizer),
        "steps": schedule_rows,
        "loss_semantics": "pre-step masked mean loss on that step's batch",
    }
    with open(args.schedule_out, "w") as f:
        json.dump(schedule, f, indent=2)

    print(f"wrote {args.batches_out}")
    print(f"wrote {args.out}")
    print(f"wrote {args.schedule_out}")
    print(f"B={B} T={T} n_steps={n_steps} mixture_convs={len(dataset)}")
    print(f"first 5 losses: {losses[:5]}")
    print(f"last loss: {losses[-1]:.9f}")


# -----------------------------------------------------------------------------
# dump-bpb: reference bits-per-byte (loss_eval.evaluate_bpb) over the exact
# trace_batches payload, so the Zig eval-bpb can be checked to rel <= 1e-4.


def read_trace_batches(path):
    """trace_batches.bin: u32 n_steps, u32 B, u32 T, then per step B*T u32 inputs
    and B*T i32 targets (cmd_dump_loss_trace layout)."""
    with open(path, "rb") as f:
        data = f.read()
    n_steps, B, T = struct.unpack_from("<III", data, 0)
    off = 12
    nbt = B * T
    batches = []
    for _ in range(n_steps):
        inputs = np.frombuffer(data, dtype="<u4", count=nbt, offset=off).astype(np.int64).reshape(B, T)
        off += 4 * nbt
        targets = np.frombuffer(data, dtype="<i4", count=nbt, offset=off).astype(np.int64).reshape(B, T)
        off += 4 * nbt
        batches.append((torch.from_numpy(inputs.copy()), torch.from_numpy(targets.copy())))
    assert off == len(data), "trailing bytes in trace batches file"
    return n_steps, B, T, batches


def cmd_dump_bpb(args):
    from nanochat.loss_eval import evaluate_bpb
    from nanochat.tokenizer import get_token_bytes

    model, cfg = build_model(args.config)
    load_init(model, args.init)
    model.eval()

    n_steps, B, T, batches = read_trace_batches(args.batches)
    token_bytes = get_token_bytes(device=model.get_device())

    bpb = evaluate_bpb(model, batches, n_steps, token_bytes)

    # Recompute the raw nats/bytes totals too (for provenance / debugging).
    total_nats = 0.0
    total_bytes = 0
    with torch.no_grad():
        for x, y in batches:
            loss2d = model(x, y, loss_reduction="none").view(-1)
            yy = y.view(-1)
            nb = token_bytes[yy]
            total_nats += float((loss2d * (nb > 0)).sum())
            total_bytes += int(nb.sum())

    out = {"config": args.config, "n_steps": n_steps, "B": B, "T": T,
           "bpb": bpb, "total_nats": total_nats, "total_bytes": total_bytes}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")
    print(f"bpb = {bpb:.9f} (total_nats {total_nats:.6f}, total_bytes {total_bytes})")


# -----------------------------------------------------------------------------
# dump-greedy


def cmd_dump_greedy(args):
    from nanochat.tokenizer import get_tokenizer

    model, cfg = build_model(args.config)
    load_init(model, args.init)
    model.eval()
    tok = get_tokenizer()
    prompt_ids = tok.encode(args.prompt, prepend="<|bos|>")  # bos + encode_ordinary
    out_ids = list(model.generate(prompt_ids, max_tokens=args.max_tokens, temperature=0.0))
    with open(args.out, "w") as f:
        json.dump({"prompt": args.prompt, "prompt_ids": prompt_ids, "out_ids": out_ids}, f, indent=2)
    print(f"wrote {args.out}")
    print(f"prompt_ids={prompt_ids}")
    print(f"out_ids={out_ids}")


# -----------------------------------------------------------------------------
# main / argparse dispatch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("dump-init", help="init weights (safetensors) + config json with rms_norm_eps")
    p.add_argument("--config", required=True, choices=list(CONFIGS))
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_dump_init)

    p = sub.add_parser("make-batch", help="fixed (B,T) batch from real val data")
    p.add_argument("--config", required=True, choices=list(CONFIGS))
    p.add_argument("--B", type=int, default=None)
    p.add_argument("--T", type=int, default=None)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_make_batch)

    p = sub.add_parser("dump-forward", help="traced forward activations + logits + loss")
    p.add_argument("--config", required=True, choices=list(CONFIGS))
    p.add_argument("--init", required=True)
    p.add_argument("--batch", required=True)
    p.add_argument("--loss-reduction", choices=["mean", "none"], default="mean",
                   help="'none' additionally dumps the per-token (B,T) loss as loss_none")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_dump_forward)

    p = sub.add_parser("dump-grad", help="per-param grads after one backward")
    p.add_argument("--config", required=True, choices=list(CONFIGS))
    p.add_argument("--init", required=True)
    p.add_argument("--batch", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_dump_grad)

    p = sub.add_parser("dump-optsteps", help="params + optimizer state after k steps on the fixed batch")
    p.add_argument("--config", required=True, choices=list(CONFIGS))
    p.add_argument("--init", required=True)
    p.add_argument("--batch", required=True)
    p.add_argument("--steps", default="1,10", help="comma list of completed-step counts to dump")
    p.add_argument("--out-prefix", required=True, help="writes {prefix}_{k}.safetensors + {prefix}_schedule.json")
    p.set_defaults(func=cmd_dump_optsteps)

    p = sub.add_parser("dump-loss-trace", help="per-step pre-update loss over fresh real batches")
    p.add_argument("--config", required=True, choices=list(CONFIGS))
    p.add_argument("--init", required=True)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--B", type=int, default=None)
    p.add_argument("--T", type=int, default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--batches-out", required=True)
    p.set_defaults(func=cmd_dump_loss_trace)

    p = sub.add_parser("dump-bpb", help="reference evaluate_bpb over the trace batches")
    p.add_argument("--config", required=True, choices=list(CONFIGS))
    p.add_argument("--init", required=True)
    p.add_argument("--batches", required=True, help="trace_batches_d6.bin")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_dump_bpb)

    p = sub.add_parser("dump-sft-trace", help="SFT per-step masked mean loss trace from a trained base checkpoint")
    p.add_argument("--config", required=True, choices=list(CONFIGS))
    p.add_argument("--init", required=True, help="trained base checkpoint safetensors (e.g. base_ckpt_d6_step2500)")
    p.add_argument("--mixture", required=True, help="SFT val mixture JSONL (mixture-ordered conversations)")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--B", type=int, default=None, help="device_batch_size (default 2)")
    p.add_argument("--T", type=int, default=None, help="max_seq_len (default 512 = d6 context)")
    p.add_argument("--render-max-tokens", type=int, default=None,
                   help="truncate each conversation to this many tokens (default = T)")
    p.add_argument("--embedding-lr", type=float, default=SFT_EMBEDDING_LR)
    p.add_argument("--unembedding-lr", type=float, default=SFT_UNEMBEDDING_LR)
    p.add_argument("--matrix-lr", type=float, default=SFT_MATRIX_LR)
    p.add_argument("--init-lr-frac", type=float, default=SFT_INIT_LR_FRAC)
    p.add_argument("--warmup-ratio", type=float, default=SFT_WARMUP_RATIO)
    p.add_argument("--warmdown-ratio", type=float, default=SFT_WARMDOWN_RATIO)
    p.add_argument("--final-lr-frac", type=float, default=SFT_FINAL_LR_FRAC)
    p.add_argument("--out", required=True, help="loss_trace_sft_d6.json")
    p.add_argument("--batches-out", required=True, help="trace_batches_sft_d6.bin")
    p.add_argument("--schedule-out", required=True, help="sft_schedule_d6.json")
    p.set_defaults(func=cmd_dump_sft_trace)

    p = sub.add_parser("dump-greedy", help="temperature=0 greedy decode stream")
    p.add_argument("--config", required=True, choices=list(CONFIGS))
    p.add_argument("--init", required=True)
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_dump_greedy)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
