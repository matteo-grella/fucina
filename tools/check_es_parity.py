#!/usr/bin/env python3
"""ES-at-scale reference cross-check (env-gated companion of gen_es_goldens.py).

Verifies that Fucina's ES algebra (src/es.zig — the semantics replicated by
tools/gen_es_goldens.py) is EXACTLY the reference implementation's, by running
both on identical torch noise:

  reference side: the actual `WorkerExtension.perturb_self_weights` /
      `restore_self_weights` / `update_weights_from_seeds` and
      `reward_shaping.z_score` imported from a local clone of
      github.com/VsonicV/es-at-scale (refs/es-at-scale by convention,
      untracked — see tools/fetch_refs.sh for the refs/ policy), driven
      through a stub model_runner so no vLLM/CUDA is needed;

  Fucina side: a clean-room torch transcription of src/es.zig's update
      algebra — f32 noise, delta narrowed to the parameter dtype before the
      widen-add-narrow apply, f32 mul-then-add accumulation, one final
      alpha/population scaling, z-score with f64 stats (ddof=0) and +1e-8 on
      the std — with the SAME torch.Generator noise the reference consumes
      (the only substitution in the Zig implementation is the RNG, whose
      stream-level correctness gen_es_goldens.py pins separately).

Everything is asserted BITWISE (torch.equal) on f32, f16, and bf16
parameters, both noise schemes' seed handling included (the reference
reseeds the same seed per tensor — the `correlated` scheme in es.zig).

Environment + invocation (torch CPU only):
    python3 -m venv /tmp/fucina-golden
    /tmp/fucina-golden/bin/pip install numpy torch
    /tmp/fucina-golden/bin/python tools/check_es_parity.py [path/to/es-at-scale]
"""

import importlib.util
import math
import pathlib
import sys

import numpy as np
import torch


def load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeModel:
    def __init__(self, params):
        self._params = params

    def named_parameters(self):
        return list(self._params.items())


class FakeModelRunner:
    def __init__(self, params):
        self.model = FakeModel(params)


def make_params(seed=7):
    """f32/f16/bf16 tensors with odd shapes (exercises the Box-Muller tail)."""
    gen = torch.Generator().manual_seed(seed)
    return {
        "w32": torch.nn.Parameter(torch.randn(5, 7, generator=gen, dtype=torch.float32)),
        "w16": torch.nn.Parameter(torch.randn(3, 11, generator=gen, dtype=torch.float32).to(torch.float16)),
        "wb16": torch.nn.Parameter(torch.randn(9, generator=gen, dtype=torch.float32).to(torch.bfloat16)),
        # Same shape as w32: under the reference's per-tensor reseeding these
        # two receive identical noise (the `correlated` scheme in es.zig).
        "w32b": torch.nn.Parameter(torch.randn(5, 7, generator=gen, dtype=torch.float32)),
    }


# --- Fucina's algebra (src/es.zig), transcribed over torch noise -------------


def fucina_noise(p, seed):
    """The reference draw for tensor p, upcast to the f32 stream es.zig uses.

    The reference generates noise IN p.dtype; es.zig's splitmix64 stream is
    f32. Feeding the same (dtype-quantized) draw upcast to f32 into es.zig's
    algebra makes the two sides comparable noise-for-noise: any remaining
    difference is algebra, which is what this script checks.
    """
    gen = torch.Generator(device=p.device)
    gen.manual_seed(int(seed))
    return torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen).to(torch.float32)


def fucina_perturb(params, seed, sigma, sign=1.0):
    """es.zig perturbSlot: t = f32(scaled*eps) narrowed to dtype, then
    widen-add-narrow (identical to torch's `p.data.add_(scalar*noise)`)."""
    scaled = np.float32(np.float32(sign) * np.float32(sigma))
    for _, p in params.items():
        eps = fucina_noise(p, seed)
        t = (eps * float(scaled)).to(p.dtype)
        p.data.add_(t)


def fucina_z_score(rewards):
    """es.zig update: f64 stats (ddof=0, plain sequential summation — es.zig's
    rewardStats order), +1e-8 on the std, f32 coefficients."""
    mean = 0.0
    for r in rewards:
        mean += float(np.float32(r))
    mean /= len(rewards)
    var = 0.0
    for r in rewards:
        d = float(np.float32(r)) - mean
        var += d * d
    var /= len(rewards)
    std = math.sqrt(var)  # IEEE-exact, matches Zig @sqrt
    return [np.float32((float(np.float32(r)) - mean) / (std + 1e-8)) for r in rewards]


def fucina_update(params, seeds, coeffs, alpha, population):
    """es.zig updateSlot: f32 mul-then-add accumulation in member order, one
    final alpha/population scaling, delta narrowed to dtype, widen-add-narrow."""
    scale = np.float32(np.float32(alpha) / np.float32(population))
    for _, p in params.items():
        acc = torch.zeros(p.shape, dtype=torch.float32)
        for seed, coeff in zip(seeds, coeffs):
            eps = fucina_noise(p, seed)
            acc += eps * float(np.float32(coeff))
        delta = (acc * float(scale)).to(p.dtype)
        p.data.add_(delta)


# --- the checks ---------------------------------------------------------------


def expect_equal(label, ours, theirs):
    for name in theirs:
        if not torch.equal(ours[name].data, theirs[name].data):
            diff = (ours[name].data.float() - theirs[name].data.float()).abs().max().item()
            raise SystemExit(f"FAIL {label}: {name} differs (max abs diff {diff})")
    print(f"  ok  {label}")


def main():
    repo = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "refs/es-at-scale")
    ext_path = repo / "es_at_scale" / "utils" / "worker_extension.py"
    shaping_path = repo / "es_at_scale" / "utils" / "reward_shaping.py"
    if not ext_path.exists():
        raise SystemExit(
            f"reference clone not found at {repo} — clone it first:\n"
            "  git clone https://github.com/VsonicV/es-at-scale refs/es-at-scale"
        )

    worker_extension = load_module(ext_path, "ref_worker_extension")
    reward_shaping = load_module(shaping_path, "ref_reward_shaping")

    sigma = 0.02
    alpha = 0.01
    seeds = [12345, 67890, 424242, 1 << 29, 3]
    # Pre-rounded to f32: Fucina's reward API is f32 while the reference keeps
    # python floats (f64) — identical INPUTS isolate the algebra comparison.
    rewards = [float(np.float32(v)) for v in [0.3, -1.2, 0.75, 0.0, 2.5]]

    # Reference worker over its own parameter copy.
    ref_params = make_params()
    worker = worker_extension.WorkerExtension()
    worker.model_runner = FakeModelRunner(ref_params)

    our_params = make_params()

    print(f"reference: {ext_path}")

    # 1. Perturb: one member in place.
    worker.perturb_self_weights(seeds[0], sigma, negate=False)
    fucina_perturb(our_params, seeds[0], sigma, sign=1.0)
    expect_equal("perturb (+sigma)", our_params, ref_params)

    # 1b. The reference's per-tensor reseeding = the `correlated` scheme:
    # same-shape tensors receive identical noise. Witnessed on ZERO tensors
    # (fl(0 + t) = t exactly; reconstructing deltas from nonzero weights
    # would smear the comparison with per-element rounding).
    zero_params = {
        "za": torch.nn.Parameter(torch.zeros(4, 6)),
        "zb": torch.nn.Parameter(torch.zeros(4, 6)),
    }
    zero_worker = worker_extension.WorkerExtension()
    zero_worker.model_runner = FakeModelRunner(zero_params)
    zero_worker.perturb_self_weights(seeds[0], sigma, negate=False)
    if not torch.equal(zero_params["za"].data, zero_params["zb"].data):
        raise SystemExit("FAIL: reference same-shape tensors did not share noise")
    print("  ok  correlated-noise semantics (same-shape tensors share the draw)")

    # 2. Restore: regenerate-subtract.
    worker.restore_self_weights(seeds[0], sigma)
    fucina_perturb(our_params, seeds[0], sigma, sign=-1.0)
    expect_equal("restore (-sigma)", our_params, ref_params)

    # 3. z-score: the reference's dict-shaped shaping vs ours.
    seeds_perf = {s: {"avg_reward": r} for s, r in zip(seeds, rewards)}
    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    ref_shaped = reward_shaping.z_score(seeds_perf, mean_reward=mean_reward, std_reward=std_reward)
    ref_coeffs = [float(ref_shaped[s]["norm_reward"]) for s in seeds]
    our_coeffs = fucina_z_score(rewards)
    for rc, oc in zip(ref_coeffs, our_coeffs):
        if np.float32(rc) != oc:
            raise SystemExit(f"FAIL z-score: {rc} vs {oc}")
    print("  ok  z-score normalization (f64 stats, ddof=0, +1e-8 on std, f32 coeffs)")

    # 4. Update: the full population step on the perturb-drifted weights
    # (both sides carry the identical drift, so the comparison stays bitwise).
    worker.update_weights_from_seeds(seeds, ref_coeffs, alpha, len(seeds))
    fucina_update(our_params, seeds, our_coeffs, alpha, len(seeds))
    expect_equal("update (z-scored, fp32 accumulation)", our_params, ref_params)

    # 5. A second iteration with fresh seeds/rewards on the updated weights.
    seeds2 = [111, 222, 333, 444, 555]
    rewards2 = [float(np.float32(v)) for v in [1.0, 1.0, -0.5, 0.25, -2.0]]
    coeffs2_ref = [
        float(v["norm_reward"])
        for v in reward_shaping.z_score(
            {s: {"avg_reward": r} for s, r in zip(seeds2, rewards2)},
            mean_reward=float(np.mean(rewards2)),
            std_reward=float(np.std(rewards2)),
        ).values()
    ]
    worker.update_weights_from_seeds(seeds2, coeffs2_ref, alpha, len(seeds2))
    fucina_update(our_params, seeds2, fucina_z_score(rewards2), alpha, len(seeds2))
    expect_equal("second update (state carried forward)", our_params, ref_params)

    print("PASS: Fucina ES algebra is bitwise-identical to the es-at-scale reference on shared noise")

    check_awd()


# --- AWD cross-check (arXiv:2605.30148, github.com/kschweig/es-awd) ----------


def fucina_awd(params, anchors, decay_type, decay_lambda, alpha):
    """es.zig anchorSlot: the reference's per-op rounding chain — each step
    computes elementwise and rounds to the parameter dtype, exactly like
    torch's in-place sub_/mul_/add_ (and the l1 soft-threshold composite)."""
    decay_step = float(alpha) * float(decay_lambda)
    for (_, p), a in zip(params.items(), anchors):
        anchor = a.to(p.device)
        d = (p.data - anchor).to(p.dtype)
        if decay_type == "l2":
            d = (d * (1.0 - decay_step)).to(p.dtype)
        else:  # l1: round(|d| - step), clamp at 0, restore sign
            t = (torch.abs(d) - decay_step).to(p.dtype)
            d = torch.sign(d) * torch.clamp(t, min=0.0)
        p.data.copy_((d + anchor).to(p.dtype))


def check_awd():
    repo = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "refs/es-awd")
    wuu_path = repo / "weight_update_utils.py"
    if not wuu_path.exists():
        raise SystemExit(
            f"AWD reference clone not found at {repo} — clone it first:\n"
            "  git clone https://github.com/kschweig/es-awd refs/es-awd"
        )
    wuu = load_module(wuu_path, "ref_weight_update_utils")
    print(f"AWD reference: {wuu_path}")

    alpha = 0.5  # exactly representable: f32 and f64 scalar paths agree
    for decay_type, decay_lambda in (("l2", 0.25), ("l1", 0.125)):
        ref_params = make_params(seed=11)
        our_params = make_params(seed=11)
        anchors = [p.data.detach().clone() for _, p in ref_params.items()]

        for (_, p), a in zip(ref_params.items(), anchors):
            wuu.apply_reference_weight_decay_(
                parameter=p,
                reference_parameter=a,
                weight_decay_type=decay_type,
                weight_decay_lambda=decay_lambda,
                alpha=alpha,
            )
        fucina_awd(our_params, anchors, decay_type, decay_lambda, alpha)
        expect_equal(f"AWD decay ({decay_type}, post-op rounding chain)", our_params, ref_params)

    # Full-sequence check: their update_parameters_from_seeds (ES update THEN
    # decay, per parameter) vs the same ES update followed by our anchor
    # step — pins the operation ORDER and that the decay reads the
    # POST-update weights. The ES-update half deliberately transcribes
    # es-awd's own accumulator semantics (torch fused `add_(alpha=)`), which
    # differ from es-at-scale's explicit mul-then-add by one f32 ulp —
    # es.zig follows es-at-scale (its primary reference, pinned above), so
    # only the decay placement is under test here.
    seeds = [1234, 5678, 91011, 121314]
    rewards = [float(np.float32(v)) for v in [0.5, -1.0, 0.25, 2.0]]
    coeffs = fucina_z_score(rewards)
    ref_params = make_params(seed=23)
    our_params = make_params(seed=23)
    anchors = [p.data.detach().clone() for _, p in ref_params.items()]

    wuu.update_parameters_from_seeds(
        parameters=[p for _, p in ref_params.items()],
        seeds=seeds,
        coeffs=[float(c) for c in coeffs],
        alpha=alpha,
        population_size=len(seeds),
        original_parameters=anchors,
        weight_decay_type="l2",
        weight_decay_lambda=0.25,
    )
    scale = float(alpha) / float(len(seeds))
    for _, p in our_params.items():
        accumulator = torch.zeros_like(p.data)
        for seed, coeff in zip(seeds, coeffs):
            gen = torch.Generator(device=p.device)
            gen.manual_seed(int(seed))
            noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
            accumulator.add_(noise, alpha=float(coeff))
        p.data.add_(accumulator.to(dtype=p.dtype), alpha=scale)
    fucina_awd(our_params, anchors, "l2", 0.25, alpha)
    expect_equal("ES update + AWD sequence (l2, es-awd update semantics)", our_params, ref_params)

    print("PASS: Fucina AWD is bitwise-identical to the es-awd reference (l1 + l2, decay and full update sequence)")


if __name__ == "__main__":
    main()
