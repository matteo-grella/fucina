# Committed verbatim from /tmp/fucina-golden/gen_goldens.py — generates the
# expected values embedded in src/optim_tests.zig's optimizer golden-parity
# tests (AdamW/Muon/APOLLO). Prints Zig constants to stdout; paste them into
# the tests when regenerating.
#
# Environment + invocation (torch CPU only):
#   python3 -m venv /tmp/fucina-golden
#   /tmp/fucina-golden/bin/pip install torch==2.12
#   /tmp/fucina-golden/bin/python tools/gen_optim_goldens.py
#
"""Golden-value generator for Fucina's optimizer parity tests.

AdamW   : the real torch.optim.AdamW.
Muon    : Keller Jordan's reference update transcribed verbatim
          (github.com/KellerJordan/Muon muon.py), with Newton-Schulz run in
          float32 instead of bfloat16 to match the f32 CPU port.
APOLLO  : the official apollo_torch.apollo.AdamW rank-path math transcribed
          verbatim, with a FIXED injected projection matrix so the test
          compares algorithm math, not RNG streams.

Everything is float32 tensors with python-float (f64) scalar hyperparams,
mirroring the Zig implementation's f32 data / f64 scalar split.
"""
import torch

torch.set_printoptions(precision=10)


def dump(name, t):
    flat = t.reshape(-1).tolist()
    print(f"{name} = {{ " + ", ".join(f"{v:.9g}" for v in flat) + " }};")


# ---------------------------------------------------------------- AdamW
print("// ---- AdamW: torch.optim.AdamW, 3 steps, lr=0.1 betas=(0.9,0.999) eps=1e-8 wd=0.1")
w0 = [1.0, -2.0, 0.5, 3.0, -0.25, 1.5]
g1 = [0.5, -1.0, 2.0, 0.25, -0.75, 1.25]
g2 = [-0.3, 0.8, -1.2, 0.6, 0.1, -0.9]
g3 = [1.1, 0.2, -0.4, -1.3, 0.7, 0.05]
p = torch.tensor(w0, dtype=torch.float32).reshape(2, 3)
p = torch.nn.Parameter(p)
opt = torch.optim.AdamW([p], lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)
for g in (g1, g2, g3):
    p.grad = torch.tensor(g, dtype=torch.float32).reshape(2, 3)
    opt.step()
dump("adamw_expected", p.detach())


# ---------------------------------------------------------------- Muon
def ns5(G, steps=5):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.clone()  # reference casts to bfloat16; the f32 CPU port stays f32
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_steps(w0, grads, lr, wd, scale, beta=0.95, nesterov=True):
    p = torch.tensor(w0, dtype=torch.float32).reshape(3, 2)  # tall: transpose trick
    M = torch.zeros_like(p)
    for g in grads:
        g = torch.tensor(g, dtype=torch.float32).reshape(3, 2)
        M.lerp_(g, 1 - beta)
        u = g.lerp(M, beta) if nesterov else M.clone()
        O = ns5(u, 5)
        if scale == "spectral":
            eff_lr = lr * max(1, u.size(-2) / u.size(-1)) ** 0.5
        else:  # match_rms_adamw
            eff_lr = lr * 0.2 * max(u.size(-2), u.size(-1)) ** 0.5
        p.mul_(1 - lr * wd)
        p.add_(O, alpha=-eff_lr)
    return p


mw0 = [1.0, -2.0, 0.5, 3.0, -0.25, 1.5]
mg1 = [0.5, -1.0, 2.0, 0.25, -0.75, 1.25]
mg2 = [-0.3, 0.8, -1.2, 0.6, 0.1, -0.9]
mg3 = [1.1, 0.2, -0.4, -1.3, 0.7, 0.05]
print("// ---- Muon: Keller reference (f32 NS), 3 steps, 3x2 tall, lr=0.02 wd=0.01 beta=0.95 nesterov")
dump("muon_spectral_expected", muon_steps(mw0, [mg1, mg2, mg3], 0.02, 0.01, "spectral"))
dump("muon_rms_expected", muon_steps(mw0, [mg1, mg2, mg3], 0.02, 0.01, "match_rms_adamw"))


# ---------------------------------------------------------------- APOLLO
def apollo_steps(w0, shape, grads, P, lr, rank, scale, scale_type,
                 eps=1e-6, b1=0.9, b2=0.999, wd=0.1, correct_bias=True,
                 scale_front=False, nl=True):
    m_, n_ = shape
    p = torch.tensor(w0, dtype=torch.float32).reshape(m_, n_)
    tall = m_ >= n_
    norm_dim = 0 if m_ < n_ else 1
    M = V = None
    prev_norm = None
    t = 0
    for g in grads:
        G = torch.tensor(g, dtype=torch.float32).reshape(m_, n_)
        R = G @ P.t() if tall else P.t() @ G
        if M is None:
            M = torch.zeros_like(R)
            V = torch.zeros_like(R)
        t += 1
        M.mul_(b1).add_(R, alpha=1 - b1)
        V.mul_(b2).addcmul_(R, R, value=1 - b2)
        step_size = lr
        if correct_bias:
            step_size = lr * (1 - b2 ** t) ** 0.5 / (1 - b1 ** t)
        Rt = M / (V.sqrt() + eps)
        if scale_type == "channel":
            s = torch.norm(Rt, dim=norm_dim) / (torch.norm(R, dim=norm_dim) + 1e-8)
            if norm_dim == 1:
                s = s.unsqueeze(1)
        else:
            s = torch.norm(Rt) / (torch.norm(R) + 1e-8)
        U = G * s
        if scale_front:
            U = U * (scale ** 0.5)
        if nl:
            cur = torch.norm(U).item()
            if prev_norm is not None:
                limiter = max(cur / (prev_norm + 1e-8), 1.01) / 1.01
                U = U / limiter
                prev_norm = cur / limiter
            else:
                prev_norm = cur
        if not scale_front:
            U = U * (scale ** 0.5)
        p.add_(U, alpha=-step_size)
        if wd > 0:
            p.add_(p, alpha=-lr * wd)
    return p


# Tall case: 4x3 param, rank 2 -> P is (2, 3), R = G @ P^T is (4, 2).
aw0 = [1.0, -2.0, 0.5, 3.0, -0.25, 1.5, 0.75, -1.25, 2.25, -0.5, 0.3, -1.8]
ag1 = [0.5, -1.0, 2.0, 0.25, -0.75, 1.25, -0.6, 0.45, -0.15, 0.9, -1.1, 0.7]
ag2 = [-0.3, 0.8, -1.2, 0.6, 0.1, -0.9, 1.05, -0.35, 0.55, -0.85, 0.2, 0.4]
ag3 = [1.1, 0.2, -0.4, -1.3, 0.7, 0.05, -0.95, 0.65, -0.25, 0.15, -0.45, 1.35]
P_tall = torch.tensor([0.4, -0.7, 0.2, -0.3, 0.6, 0.9], dtype=torch.float32).reshape(2, 3)
print("// ---- APOLLO tall 4x3 rank=2 channel: lr=0.02 wd=0.1 eps=1e-6 scale=1, fixed P (2,3)")
dump("apollo_tall_expected",
     apollo_steps(aw0, (4, 3), [ag1, ag2, ag3], P_tall, 0.02, 2, 1.0, "channel"))

# Wide case: 3x5 param, rank 2 -> P is (3, 2), R = P^T @ G is (2, 5).
ww0 = [0.5, -1.5, 2.0, -0.25, 1.0, 0.75, -0.5, 1.25, -2.0, 0.3,
       -0.8, 0.6, -1.1, 0.9, -0.4]
wg1 = [0.2, -0.9, 1.3, 0.45, -0.7, -0.55, 0.85, -0.2, 0.65, -1.15,
       0.95, -0.35, 0.5, -0.6, 0.1]
wg2 = [-0.4, 0.7, -1.0, 0.55, 0.15, 0.8, -1.2, 0.35, -0.65, 0.25,
       -0.9, 1.05, -0.15, 0.45, -0.75]
P_wide = torch.tensor([0.3, -0.5, 0.8, 0.1, -0.6, 0.4], dtype=torch.float32).reshape(3, 2)
print("// ---- APOLLO wide 3x5 rank=2 channel: lr=0.02 wd=0 eps=1e-6 scale=1, fixed P (3,2)")
dump("apollo_wide_expected",
     apollo_steps(ww0, (3, 5), [wg1, wg2], P_wide, 0.02, 2, 1.0, "channel", wd=0.0))

# Mini: tensor scaling, rank 1, scale=128, tall 4x3 -> P is (1, 3).
P_mini = torch.tensor([0.7, -0.2, 0.5], dtype=torch.float32).reshape(1, 3)
print("// ---- APOLLO-Mini tall 4x3 rank=1 tensor scale=128: lr=0.005 wd=0, fixed P (1,3)")
dump("apollo_mini_expected",
     apollo_steps(aw0, (4, 3), [ag1, ag2, ag3], P_mini, 0.005, 1, 128.0, "tensor", wd=0.0))

# ---- Additional discriminating cases ----
# Tiny gradients: sqrt(v)/bc2_sqrt ~ 1e-5 is comparable to eps=1e-8/..., so a
# wrong eps placement (e.g. (sqrt(v)+eps)/bc2_sqrt) shifts the update visibly.
print("// ---- AdamW tiny-grad: 2 steps, lr=0.1 wd=0, grads O(1e-5) — pins eps placement")
pt = torch.nn.Parameter(torch.tensor([1.0, -2.0, 0.5, 3.0], dtype=torch.float32).reshape(2, 2))
optt = torch.optim.AdamW([pt], lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
tg1 = [2e-5, -1e-5, 3e-5, -4e-5]
tg2 = [-1e-5, 2.5e-5, -2e-5, 1.5e-5]
for g in (tg1, tg2):
    pt.grad = torch.tensor(g, dtype=torch.float32).reshape(2, 2)
    optt.step()
dump("adamw_tinygrad_expected", pt.detach())

# Wide Muon matrix (2x3): exercises the NON-transposed Newton-Schulz branch.
def muon_steps_wide(w0, grads, lr, wd, scale, beta=0.95, nesterov=True):
    p = torch.tensor(w0, dtype=torch.float32).reshape(2, 3)
    M = torch.zeros_like(p)
    for g in grads:
        g = torch.tensor(g, dtype=torch.float32).reshape(2, 3)
        M.lerp_(g, 1 - beta)
        u = g.lerp(M, beta) if nesterov else M.clone()
        O = ns5(u, 5)
        if scale == "spectral":
            eff_lr = lr * max(1, u.size(-2) / u.size(-1)) ** 0.5
        else:
            eff_lr = lr * 0.2 * max(u.size(-2), u.size(-1)) ** 0.5
        p.mul_(1 - lr * wd)
        p.add_(O, alpha=-eff_lr)
    return p

print("// ---- Muon wide 2x3 spectral: 3 steps, lr=0.02 wd=0.01")
dump("muon_wide_expected", muon_steps_wide(mw0, [mg1, mg2, mg3], 0.02, 0.01, "spectral"))

# ---- SGD: torch.optim.SGD goldens ----
print("// ---- SGD nesterov: lr=0.1 momentum=0.9 wd=0.05 nesterov, 3 steps")
ps = torch.nn.Parameter(torch.tensor(w0, dtype=torch.float32).reshape(2, 3))
opts = torch.optim.SGD([ps], lr=0.1, momentum=0.9, weight_decay=0.05, nesterov=True)
for g in (g1, g2, g3):
    ps.grad = torch.tensor(g, dtype=torch.float32).reshape(2, 3)
    opts.step()
dump("sgd_nesterov_expected", ps.detach())

print("// ---- SGD dampening: lr=0.1 momentum=0.9 dampening=0.1 wd=0, 3 steps")
pd = torch.nn.Parameter(torch.tensor(w0, dtype=torch.float32).reshape(2, 3))
optd = torch.optim.SGD([pd], lr=0.1, momentum=0.9, dampening=0.1, weight_decay=0.0)
for g in (g1, g2, g3):
    pd.grad = torch.tensor(g, dtype=torch.float32).reshape(2, 3)
    optd.step()
dump("sgd_dampening_expected", pd.detach())

print("// ---- SGD plain: lr=0.1, no momentum, 2 steps")
pp = torch.nn.Parameter(torch.tensor(w0, dtype=torch.float32).reshape(2, 3))
optp = torch.optim.SGD([pp], lr=0.1)
for g in (g1, g2):
    pp.grad = torch.tensor(g, dtype=torch.float32).reshape(2, 3)
    optp.step()
dump("sgd_plain_expected", pp.detach())

# ---- clip_grad_norm_ + SGD: golden for the clipping semantics ----
print("// ---- clip+SGD: two params, clip_grad_norm_(max=0.5) then SGD lr=0.1 momentum=0.9, 2 steps")
ca = torch.nn.Parameter(torch.tensor(w0, dtype=torch.float32).reshape(2, 3))
cb = torch.nn.Parameter(torch.tensor([0.5, -1.0], dtype=torch.float32))
optc = torch.optim.SGD([ca, cb], lr=0.1, momentum=0.9)
norms = []
for g_main, g_b in (( g1, [0.3, -0.6]), (g2, [-0.2, 0.4])):
    ca.grad = torch.tensor(g_main, dtype=torch.float32).reshape(2, 3)
    cb.grad = torch.tensor(g_b, dtype=torch.float32)
    norms.append(torch.nn.utils.clip_grad_norm_([ca, cb], 0.5).item())
    optc.step()
dump("clip_sgd_a_expected", ca.detach())
dump("clip_sgd_b_expected", cb.detach())
print(f"clip_norms_expected = {{ {norms[0]:.9g}, {norms[1]:.9g} }};")
