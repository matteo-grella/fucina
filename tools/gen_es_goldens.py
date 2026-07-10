#!/usr/bin/env python3
"""Golden-value generator for src/es_tests.zig (ES-at-scale parity).

Replicates, in numpy, BOTH halves of the Zig ES implementation:

  1. the repo-owned RNG (src/rng.zig): splitmix64, counter-based `at`, and
     the Box-Muller gaussian stream of `gaussianFillAt` — bit-level integer
     math, f64 transcendentals, one f64->f32 rounding per value;
  2. the ES algebra of src/es.zig, which follows the reference
     implementation's semantics (github.com/VsonicV/es-at-scale,
     es_at_scale/utils/worker_extension.py `update_weights_from_seeds` +
     `perturb_self_weights`, es_at_scale/utils/reward_shaping.py `z_score`):
     f32 noise, per-member `t = sigma * eps` perturbation, z-score with f64
     stats (ddof=0) and +1e-8 on the std, fp32 accumulation of
     `coeff * eps` (mul-then-add, no FMA), one final `alpha/population`
     scaling, delta narrowed to the parameter dtype, widen-add-narrow apply.

Printed values are compared in the Zig tests with tight tolerances rather
than bitwise: this generator computes the gaussian with f64 libm while
es.zig draws through the vectorized `rng.gaussianFillAtFast` mapping (same
uniforms, f32 polynomial transcendentals, values a few f32 ulps apart —
sigma scaling puts the deltas well below every tolerance). Antithetic
sampling is pinned separately by bitwise in-Zig serial references
(es_tests.zig). Run and paste:

    python3 tools/gen_es_goldens.py

The reference cross-check (our algebra vs the actual es-at-scale code on
identical torch noise) lives in tools/check_es_parity.py.
"""

import math

import numpy as np

MASK64 = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15

SEED_DOMAIN = 0x65735F7365656473  # "es_seeds" (src/es.zig)
NOISE_DOMAIN = 0x65735F6E6F697365  # "es_noise" (src/es.zig)


def splitmix_at(seed: int, i: int) -> int:
    """src/rng.zig `at`: the i-th output of the splitmix64 stream at `seed`."""
    z = (seed + ((i + 1) * GAMMA & MASK64)) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def gaussian_at(seed: int, first: int, n: int) -> np.ndarray:
    """src/rng.zig `gaussianFillAt(seed, first, out, 1.0)` in f32."""
    out = np.empty(n, dtype=np.float32)
    for o in range(n):
        j = first + o
        pair = j & ~1
        a = splitmix_at(seed, pair)
        b = splitmix_at(seed, pair + 1)
        uniform_a = ((a >> 11) + 1) * 2.0**-53  # (0, 1]
        uniform_b = (b >> 11) * 2.0**-53  # [0, 1)
        radius = math.sqrt(-2.0 * math.log(uniform_a))
        angle = 2.0 * math.pi * uniform_b
        value = radius * (math.cos(angle) if j % 2 == 0 else math.sin(angle))
        out[o] = np.float32(value)
    return out


# --- dtype helpers (src/dtype.zig) ------------------------------------------


def f32_to_bf16(value: np.float32) -> int:
    bits = int(np.float32(value).view(np.uint32))
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000:
        return ((bits >> 16) | 64) & 0xFFFF
    lsb = (bits >> 16) & 1
    return ((bits + 0x7FFF + lsb) >> 16) & 0xFFFF


def bf16_to_f32(bits: int) -> np.float32:
    return np.uint32(bits << 16).view(np.float32)


class SlotF32:
    dtype = "f32"

    def __init__(self, values):
        self.data = np.array(values, dtype=np.float32)

    def widen(self):
        return self.data.copy()

    def apply(self, i, wide_value):
        self.data[i] = np.float32(wide_value)


class SlotF16:
    dtype = "f16"

    def __init__(self, values):
        self.data = np.array(values, dtype=np.float16)

    def widen(self):
        return self.data.astype(np.float32)

    def apply(self, i, wide_value):
        self.data[i] = np.float16(np.float32(wide_value))


class SlotBf16:
    dtype = "bf16"

    def __init__(self, values):
        self.data = np.array([f32_to_bf16(np.float32(v)) for v in values], dtype=np.uint64)

    def widen(self):
        return np.array([bf16_to_f32(int(b)) for b in self.data], dtype=np.float32)

    def apply(self, i, wide_value):
        self.data[i] = f32_to_bf16(np.float32(wide_value))


# --- src/es.zig semantics ----------------------------------------------------


def centered_ranks(rewards):
    """es.zig's centered-rank shaping: ascending (reward, member index) order
    (PINNED tie-break — a checkpoint contract), coefficients rank/(N-1)-0.5
    in f32. For tie-free inputs this equals the verbatim OpenAI
    evolution-strategies-starter compute_centered_ranks (asserted in main)."""
    order = sorted(range(len(rewards)), key=lambda i: (rewards[i], i))
    coeffs = [np.float32(0)] * len(rewards)
    for rank, member in enumerate(order):
        coeffs[member] = np.float32(np.float32(rank) / np.float32(len(rewards) - 1) - np.float32(0.5))
    return coeffs


def openai_centered_ranks(x):
    """Verbatim compute_ranks/compute_centered_ranks from
    openai/evolution-strategies-starter es_distributed/es.py (MIT)."""
    x = np.asarray(x)
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    y = ranks.astype(np.float32)
    y /= (x.size - 1)
    y -= 0.5
    return y


def sequential_stats(rewards):
    """Mean/std (ddof=0) with plain SEQUENTIAL f64 summation — es.zig's
    rewardStats order, which can differ from numpy's pairwise summation in
    the last f64 ulp at larger populations."""
    mean = 0.0
    for r in rewards:
        mean += float(np.float32(r))
    mean /= len(rewards)
    var = 0.0
    for r in rewards:
        d = float(np.float32(r)) - mean
        var += d * d
    var /= len(rewards)
    return mean, math.sqrt(var)


class Trainer:
    def __init__(self, slots, sigma, alpha, population, noise, reward_norm, seed):
        self.slots = slots
        self.sigma = np.float32(sigma)
        self.alpha = np.float32(alpha) if alpha is not None else np.float32(np.float32(sigma) / np.float32(2))
        self.population = population
        self.noise = noise  # "iid" | "correlated"
        self.reward_norm = reward_norm  # "z_score" | "none"
        self.seed = seed
        self.iteration = 0

    def member_seed(self, member: int) -> int:
        return splitmix_at(self.seed ^ SEED_DOMAIN, (self.iteration * self.population + member) & MASK64)

    def stream_seed(self, member_seed: int, slot_index: int) -> int:
        if self.noise == "iid":
            return splitmix_at(member_seed ^ NOISE_DOMAIN, slot_index)
        return member_seed

    def perturb(self, member: int, sign: float):
        scaled = np.float32(np.float32(sign) * self.sigma)
        member_seed = self.member_seed(member)
        for k, slot in enumerate(self.slots):
            eps = gaussian_at(self.stream_seed(member_seed, k), 0, len(slot.data))
            wide = slot.widen()
            for i in range(len(slot.data)):
                # f32 rounding, then NARROW to the slot dtype before the add
                # (torch `p.data.add_(sigma*noise)`: the scalar mul rounds to
                # p.dtype, the in-place add widens-adds-narrows).
                t = np.float32(scaled * eps[i])
                if slot.dtype == "f16":
                    t = np.float32(np.float16(t))
                elif slot.dtype == "bf16":
                    t = bf16_to_f32(f32_to_bf16(t))
                slot.apply(i, np.float32(wide[i] + t))

    def coefficients(self, rewards):
        if self.reward_norm == "none":
            return [np.float32(r) for r in rewards]
        if self.reward_norm == "centered_ranks":
            return centered_ranks(rewards)
        mean, std = sequential_stats(rewards)
        return [np.float32((float(np.float32(r)) - mean) / (std + 1e-8)) for r in rewards]

    def update(self, rewards):
        assert len(rewards) == self.population
        coeffs = self.coefficients(rewards)
        member_seeds = [self.member_seed(m) for m in range(self.population)]
        scale = np.float32(self.alpha / np.float32(self.population))
        for k, slot in enumerate(self.slots):
            n = len(slot.data)
            acc = np.zeros(n, dtype=np.float32)
            for m in range(self.population):
                eps = gaussian_at(self.stream_seed(member_seeds[m], k), 0, n)
                for i in range(n):
                    term = np.float32(coeffs[m] * eps[i])  # mul rounds
                    acc[i] = np.float32(acc[i] + term)  # add rounds
            wide = slot.widen()
            for i in range(n):
                delta_wide = np.float32(scale * acc[i])
                # narrow the delta to the slot dtype, then widen-add-narrow
                if slot.dtype == "f16":
                    delta = np.float32(np.float16(delta_wide))
                elif slot.dtype == "bf16":
                    delta = bf16_to_f32(f32_to_bf16(delta_wide))
                else:
                    delta = delta_wide
                slot.apply(i, np.float32(wide[i] + delta))
        self.iteration += 1


# --- scenario ---------------------------------------------------------------

THETA_A = [0.8, -1.25, 0.5, 2.0, -0.125, 1.5, -0.75]  # f32, odd length
THETA_B = [0.5, -0.25, 1.0, -1.5, 2.0, 0.125, -0.875, 0.0625]  # f16
THETA_C = [1.0, -2.0, 0.5, -0.25, 3.0]  # bf16
THETA_D = [-0.5, 0.75, -1.0, 0.25, 1.125, -2.5, 0.375]  # f32, same length as A

REWARDS_ITER0 = [0.1, 0.9, -0.4, 0.35]
REWARDS_ITER1 = [1.0, -1.0, 0.25, 0.5]
REWARDS_B = [0.5, -0.25, 1.5]


def fresh_slots():
    return [SlotF32(THETA_A), SlotF16(THETA_B), SlotBf16(THETA_C), SlotF32(THETA_D)]


def fmt(values):
    return ", ".join(np.format_float_positional(np.float32(v), unique=True, trim="0") for v in values)


def emit(label, slots):
    print(f"    // {label}")
    for name, slot in zip("abcd", slots):
        print(f"    const {label}_{name} = [_]f32{{ {fmt(slot.widen())} }};")


def main():
    print("// Generated by tools/gen_es_goldens.py — paste into src/es_tests.zig.")
    print("// Scenario A: sigma=0.05, alpha=null (sigma/2), population=4, iid, z_score, seed=42.")
    print("// Scenario B: sigma=0.02, alpha=0.01, population=3, correlated, none, seed=42.")
    print()

    # Scenario A: perturb-only golden (member 2 at iteration 0).
    t = Trainer(fresh_slots(), 0.05, None, 4, "iid", "z_score", 42)
    t.perturb(2, 1.0)
    emit("golden_perturbed_m2", t.slots)
    print()

    # Scenario A: two full updates.
    t = Trainer(fresh_slots(), 0.05, None, 4, "iid", "z_score", 42)
    t.update(REWARDS_ITER0)
    t.update(REWARDS_ITER1)
    emit("golden_a_after2", t.slots)
    print()

    # Scenario B: correlated noise + raw rewards, one update.
    t = Trainer(fresh_slots(), 0.02, 0.01, 3, "correlated", "none", 42)
    t.update(REWARDS_B)
    emit("golden_b_after1", t.slots)
    print()

    # Scenario C: centered-rank shaping + iid, one update.
    t = Trainer(fresh_slots(), 0.05, None, 4, "iid", "centered_ranks", 42)
    t.update(REWARDS_ITER0)
    emit("golden_c_after1", t.slots)
    print()

    # Centered-rank coefficients cross-checked against the verbatim OpenAI
    # starter code (tie-free input -> identical).
    ours = centered_ranks(REWARDS_ITER0)
    theirs = openai_centered_ranks(np.array(REWARDS_ITER0, dtype=np.float64))
    assert all(np.float32(a) == np.float32(b) for a, b in zip(ours, theirs)), (ours, theirs)
    print("    // centered-rank coefficients match the OpenAI starter code (tie-free)")
    # Pinned tie order: equal rewards rank by member index.
    tied = centered_ranks([1.0, 1.0, 0.0])
    assert [float(c) for c in tied] == [0.0, 0.5, -0.5], tied
    print("    // tie-break by member index pinned: [1,1,0] -> [0.0, 0.5, -0.5]")

    # Correlated-noise witness on ZERO slots (fl(0 + t) = t exactly, so the
    # perturbed values expose the raw noise): same-length slots must be
    # bitwise identical under .correlated and differ under .iid.
    zeros = [SlotF32([0.0] * 6), SlotF32([0.0] * 6)]
    t = Trainer(zeros, 0.02, 0.01, 3, "correlated", "none", 42)
    t.perturb(0, 1.0)
    assert np.array_equal(zeros[0].data, zeros[1].data), "correlated slots must share noise"
    zeros = [SlotF32([0.0] * 6), SlotF32([0.0] * 6)]
    t = Trainer(zeros, 0.02, 0.01, 3, "iid", "none", 42)
    t.perturb(0, 1.0)
    assert not np.array_equal(zeros[0].data, zeros[1].data), "iid slots must differ"
    print("    // noise-scheme witnesses hold (correlated shares, iid differs)")


if __name__ == "__main__":
    main()
