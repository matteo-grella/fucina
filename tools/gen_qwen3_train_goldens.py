# Committed verbatim from /tmp/fucina-golden/gen_qwen3_train_goldens.py — the
# generator that produced src/llm/qwen3/train_golden_tests.zig.
#
# Environment + invocation (torch CPU only):
#   python3 -m venv /tmp/fucina-golden
#   /tmp/fucina-golden/bin/pip install torch==2.12
#   /tmp/fucina-golden/bin/python tools/gen_qwen3_train_goldens.py src/llm/qwen3/train_golden_tests.zig
#
"""Golden-value generator for Fucina's Qwen3 LoRA fine-tuning parity test.

An independent PyTorch (autograd) implementation of the EXACT same tiny dense
Qwen3 model that src/llm/qwen3_train.zig `Trainer(.{})` runs — verified op by
op against the Zig source:

  - embedding lookup (frozen rows)                       weights.getRowsAs
  - rmsNormMul: x * w / sqrt(mean(x^2, hidden) + eps)    exec.rmsNormMulAxisRank
  - q/k/v = x_normed @ W.T (W stored [out, in])          frozen-RHS dot
  - LoRA on q and v (default Targets): the adapter input is the NORMED hidden
    state (attn_in), delta = (alpha/r) * ((x @ A.T) @ B.T) added to the
    projection output BEFORE the head split                qwen3_train.adapted
  - head split [seq, heads, d]; per-head rmsNorm over d * q_norm/k_norm weight,
    THEN RoPE in `.half` (NeoX) mode: pairs (i, i + d/2) rotate with
    theta = pos / theta_base^(2i/d); first' = first*cos - second*sin,
    second' = first*sin + second*cos                       exec.rmsNormMulRopeAxisRankWithTable
  - grouped causal attention: kv_head = head // (heads/kv_heads),
    scores = (q.k) * 1/sqrt(head_dim), causal mask j <= i, softmax,
    weighted V, merge heads [seq, heads*d]                 exec.groupedCausalAttention
  - o_proj (frozen), residual; rmsNormMul(ffn_norm); gate/up (frozen);
    swiglu = up * silu(gate)                               backend/ops.gatedActivationScalar
  - down (frozen), residual; final rmsNormMul(output_norm); logits = x @ Wout.T
  - CE loss: ignore_index sentinel masked, mean over valid positions
    (torch ignore_index=-100, reduction='mean')            exec.crossEntropyLossExAxisRank

No dropout (p=0), no checkpointing, seq=64 (>= exec's attention_tiled_min_q_seq
of 48, so the Zig forward takes the query-tiled attention kernel).

The script EMITS the complete Zig test file (data + test code) and prints the
torch loss + a sha256 checksum of the golden-data section to stderr so the
values inside the Zig file are auditable.

Usage (same venv as gen_optim_goldens.py, torch 2.12 CPU):
    /tmp/fucina-golden/bin/python tools/gen_qwen3_train_goldens.py \
        src/llm/qwen3/train_golden_tests.zig
"""

import hashlib
import math
import sys

import torch

assert not torch.cuda.is_available() or True  # CPU only; nothing touches CUDA
torch.manual_seed(0xF0C1AA)

# ---------------------------------------------------------------- dimensions
VOCAB = 48
HIDDEN = 32
FFN = 56
LAYERS = 2
HEADS = 2
KV_HEADS = 1
HEAD_DIM = 16
Q_DIM = HEADS * HEAD_DIM            # 32
KV_DIM = KV_HEADS * HEAD_DIM        # 16
RANK = 4
ALPHA = 8.0
LORA_SCALE = ALPHA / RANK           # 2.0
EPS = 1e-6
ROPE_THETA = 10_000.0
SEQ = 64
MASKED_POSITIONS = (0, 7, 19, 40, 63)  # a few ignore_index labels

GEN_CMD = (
    "/tmp/fucina-golden/bin/python tools/gen_qwen3_train_goldens.py "
    "src/llm/qwen3/train_golden_tests.zig"
)


def uniform(shape, lo, hi):
    return (torch.rand(shape, dtype=torch.float32) * (hi - lo) + lo)


# ---------------------------------------------------------------- weights
# Same value scales as qwen3_train_tests.zig's synthetic tiny model: norm
# weights near 1, projections +-0.3, embedding/output +-0.5. Values come from
# torch's RNG and travel as literals — no cross-language RNG to match.
token_embedding = uniform((VOCAB, HIDDEN), -0.5, 0.5)
output_norm = uniform((HIDDEN,), 0.8, 1.2)
output = uniform((VOCAB, HIDDEN), -0.5, 0.5)

layers = []
for _ in range(LAYERS):
    layer = {
        "attn_norm": uniform((HIDDEN,), 0.8, 1.2),
        "q_norm": uniform((HEAD_DIM,), 0.8, 1.2),
        "k_norm": uniform((HEAD_DIM,), 0.8, 1.2),
        "ffn_norm": uniform((HIDDEN,), 0.8, 1.2),
        "q_proj": uniform((Q_DIM, HIDDEN), -0.3, 0.3),
        "k_proj": uniform((KV_DIM, HIDDEN), -0.3, 0.3),
        "v_proj": uniform((KV_DIM, HIDDEN), -0.3, 0.3),
        "o_proj": uniform((HIDDEN, Q_DIM), -0.3, 0.3),
        "gate_proj": uniform((FFN, HIDDEN), -0.3, 0.3),
        "up_proj": uniform((FFN, HIDDEN), -0.3, 0.3),
        "down_proj": uniform((HIDDEN, FFN), -0.3, 0.3),
        # Adapters: A kaiming-uniform-shaped (bound sqrt(3/in), nonzero) AND B
        # NONZERO — a zero B (the init default) would zero every A gradient,
        # so the golden uses a trained-looking adapter state instead.
        "lora_q_a": uniform((RANK, HIDDEN), -math.sqrt(3.0 / HIDDEN), math.sqrt(3.0 / HIDDEN)),
        "lora_q_b": uniform((Q_DIM, RANK), -0.1, 0.1),
        "lora_v_a": uniform((RANK, HIDDEN), -math.sqrt(3.0 / HIDDEN), math.sqrt(3.0 / HIDDEN)),
        "lora_v_b": uniform((KV_DIM, RANK), -0.1, 0.1),
    }
    for name in ("lora_q_a", "lora_q_b", "lora_v_a", "lora_v_b"):
        layer[name].requires_grad_(True)
    layers.append(layer)

tokens = torch.randint(0, VOCAB, (SEQ + 1,))
input_ids = tokens[:SEQ].tolist()
labels = tokens[1:].tolist()
for position in MASKED_POSITIONS:
    labels[position] = -100

# ---------------------------------------------------------------- forward
def rms_norm_mul(x, w):
    # exec.rmsNormMulAxisRank: scale = 1/sqrt(mean(x^2) + eps); y = x*scale*w.
    scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    return x * scale * w


# RoPE table exactly as exec.prepareRopeTableFactors: f32 math, angle =
# pos / theta_base^(2*pair/d), positions 0..seq-1.
positions = torch.arange(SEQ, dtype=torch.float32).unsqueeze(1)          # [seq, 1]
pair_exponent = (2.0 * torch.arange(HEAD_DIM // 2, dtype=torch.float32)) / HEAD_DIM
inv_freq = positions / torch.pow(torch.tensor(ROPE_THETA, dtype=torch.float32), pair_exponent)
ROPE_SIN = torch.sin(inv_freq)                                           # [seq, d/2]
ROPE_COS = torch.cos(inv_freq)


def rope_half(x):
    # exec.ropeAxisRankWithTable `.half` mode: pair (i, i + d/2) — NeoX-style,
    # NOT interleaved. x: [seq, heads, d].
    half = HEAD_DIM // 2
    first, second = x[..., :half], x[..., half:]
    sin = ROPE_SIN[:, None, :]
    cos = ROPE_COS[:, None, :]
    return torch.cat([first * cos - second * sin, first * sin + second * cos], dim=-1)


def attention(q, k, v):
    # exec.groupedCausalAttention: per-head scores scaled by 1/sqrt(d), causal
    # mask (query i attends keys 0..i), softmax, weighted V, merged heads.
    scale = 1.0 / math.sqrt(HEAD_DIM)
    mask = torch.triu(torch.ones(SEQ, SEQ, dtype=torch.bool), diagonal=1)
    outs = []
    for head in range(HEADS):
        kv_head = head // (HEADS // KV_HEADS)  # qwen3.Model.kv_head_for_head
        scores = (q[:, head, :] @ k[:, kv_head, :].T) * scale
        scores = scores.masked_fill(mask, float("-inf"))
        outs.append(torch.softmax(scores, dim=-1) @ v[:, kv_head, :])
    return torch.stack(outs, dim=1).reshape(SEQ, HEADS * HEAD_DIM)


def lora_delta(x, a, b):
    # lora.Adapter.delta: scale * ((x @ A^T) @ B^T), dropout p=0 (identity).
    return LORA_SCALE * ((x @ a.T) @ b.T)


x = token_embedding[torch.tensor(input_ids)]
for layer in layers:
    # Attention block (qwen3_train.layerBody, op for op).
    attn_in = rms_norm_mul(x, layer["attn_norm"])
    q = attn_in @ layer["q_proj"].T + lora_delta(attn_in, layer["lora_q_a"], layer["lora_q_b"])
    k = attn_in @ layer["k_proj"].T
    v = attn_in @ layer["v_proj"].T + lora_delta(attn_in, layer["lora_v_a"], layer["lora_v_b"])

    q3 = q.reshape(SEQ, HEADS, HEAD_DIM)
    k3 = k.reshape(SEQ, KV_HEADS, HEAD_DIM)
    v3 = v.reshape(SEQ, KV_HEADS, HEAD_DIM)

    q_rope = rope_half(rms_norm_mul(q3, layer["q_norm"]))
    k_rope = rope_half(rms_norm_mul(k3, layer["k_norm"]))

    attn = attention(q_rope, k_rope, v3)
    h = x + attn @ layer["o_proj"].T

    # FFN block: swiglu = up * silu(gate) (backend GatedOp .swiglu).
    ffn_in = rms_norm_mul(h, layer["ffn_norm"])
    gate = ffn_in @ layer["gate_proj"].T
    up = ffn_in @ layer["up_proj"].T
    x = h + (up * torch.nn.functional.silu(gate)) @ layer["down_proj"].T

logits = rms_norm_mul(x, output_norm) @ output.T
loss = torch.nn.functional.cross_entropy(
    logits,
    torch.tensor(labels, dtype=torch.long),
    ignore_index=-100,
    reduction="mean",
)
loss.backward()

loss_value = loss.item()
grads = []
for layer in layers:
    grads.append({
        "grad_q_a": layer["lora_q_a"].grad,
        "grad_q_b": layer["lora_q_b"].grad,
        "grad_v_a": layer["lora_v_a"].grad,
        "grad_v_b": layer["lora_v_b"].grad,
    })

# ---------------------------------------------------------------- emit Zig
def fmt(value):
    # 9 significant digits round-trip any f32 exactly through Zig's
    # correctly-rounded decimal-literal parse.
    return f"{value:.9g}"


def zig_array(values, indent):
    parts = [fmt(v) for v in values.reshape(-1).tolist()]
    lines = []
    current = indent
    for part in parts:
        piece = part + ","
        if len(current) + len(piece) + 1 > 118 and current != indent:
            lines.append(current.rstrip())
            current = indent
        current += " " + piece if current != indent else piece
    lines.append(current.rstrip())
    return "\n".join(lines)


def field(name, tensor, indent="        "):
    return f"{indent}.{name} = &.{{\n{zig_array(tensor.detach(), indent + '    ')}\n{indent}}},"


layer_blocks = []
for layer, grad in zip(layers, grads):
    fields = []
    for name in ("attn_norm", "q_norm", "k_norm", "ffn_norm", "q_proj", "k_proj",
                 "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                 "lora_q_a", "lora_q_b", "lora_v_a", "lora_v_b"):
        fields.append(field(name, layer[name]))
    for name in ("grad_q_a", "grad_q_b", "grad_v_a", "grad_v_b"):
        fields.append(field(name, grad[name]))
    layer_blocks.append("    .{\n" + "\n".join(fields) + "\n    },")

tokens_text = ", ".join(str(t) for t in input_ids)
labels_text = ", ".join("ig" if l == -100 else str(l) for l in labels)

data_section = f"""const expected_loss: f32 = {fmt(loss_value)};

const ig = qwen3_train.ignore_index;
const golden_tokens = [_]usize{{ {tokens_text} }};
const golden_labels = [_]usize{{ {labels_text} }};

const golden_token_embedding = [_]f32{{
{zig_array(token_embedding, "    ")}
}};
const golden_output_norm = [_]f32{{
{zig_array(output_norm, "    ")}
}};
const golden_output = [_]f32{{
{zig_array(output, "    ")}
}};

const LayerGolden = struct {{
    attn_norm: []const f32,
    q_norm: []const f32,
    k_norm: []const f32,
    ffn_norm: []const f32,
    q_proj: []const f32,
    k_proj: []const f32,
    v_proj: []const f32,
    o_proj: []const f32,
    gate_proj: []const f32,
    up_proj: []const f32,
    down_proj: []const f32,
    lora_q_a: []const f32,
    lora_q_b: []const f32,
    lora_v_a: []const f32,
    lora_v_b: []const f32,
    grad_q_a: []const f32,
    grad_q_b: []const f32,
    grad_v_a: []const f32,
    grad_v_b: []const f32,
}};

const layer_goldens = [_]LayerGolden{{
{chr(10).join(layer_blocks)}
}};
"""

data_sha = hashlib.sha256(data_section.encode()).hexdigest()

ZIG_TEMPLATE = """//! PyTorch golden-parity test for the Qwen3 LoRA fine-tuning gradients.
//!
//! GENERATED FILE — regenerate with (venv recipe in the script's header):
//!     @@CMD@@
//! Generator: torch @@TORCH@@ (CPU, float32), seed 0xF0C1AA.
//! Expected loss (torch): @@LOSS@@. Golden-data-section sha256: @@SHA@@.
//!
//! An independent PyTorch autograd implementation of the exact tiny dense
//! Qwen3 model that `qwen3_train.Trainer(.{})` runs (frozen base, LoRA rank-4
//! adapters on q and v, no dropout, seq 64 = the tiled-attention forward path)
//! produced this loss and these adapter gradients from the same weights,
//! tokens, and labels. The test rebuilds the model from the constants below,
//! overwrites the trainer's adapter A/B values with the goldens, runs
//! loss() + backward(), and asserts parity:
//!   - loss within 2e-4 relative,
//!   - every adapter-gradient element within |d| <= 1e-5 + 2e-3 * |expected|
//!     (the Zig forward uses the query-tiled online-softmax attention kernel
//!     and BLAS GEMMs whose summation orders differ from torch's).
//! Observed deviations at generation time (M1 Max): loss rel 1.7e-7 native /
//! 8.5e-8 scalar; worst grad element |d| = 2.7e-7 (<= 0.8% of its mixed
//! tolerance on both backends) — the bounds above carry >100x headroom.

const std = @import("std");
const fucina = @import("fucina");
const qwen3 = @import("model.zig");
const qwen3_train = @import("train.zig");
const weights = @import("../weights.zig");

const ExecContext = fucina.ExecContext;
const Layer = qwen3_train.ModelLayer;

const golden_config = qwen3.Config{
    .vocab_size = @@VOCAB@@,
    .hidden_size = @@HIDDEN@@,
    .intermediate_size = @@FFN@@,
    .num_layers = @@LAYERS@@,
    .num_attention_heads = @@HEADS@@,
    .num_key_value_heads = @@KV_HEADS@@,
    .head_dim = @@HEAD_DIM@@,
    .rms_norm_eps = 1e-6,
    .rope_theta = 10_000,
};

const golden_lora = fucina.lora.Config{ .rank = @@RANK@@, .alpha = @@ALPHA@@ };

// ---------------------------------------------------------------------------
// Golden data (generated; see the header comment).
// ---------------------------------------------------------------------------

@@DATA@@
// ---------------------------------------------------------------------------
// Model construction from the golden constants (mirrors the synthetic-model
// builders in qwen3_train_tests.zig, with fixed values instead of rng fills).
// ---------------------------------------------------------------------------

fn goldenLinear(ctx: *ExecContext, values: []const f32, out_dim: usize, in_dim: usize) !weights.LinearWeight {
    std.debug.assert(values.len == out_dim * in_dim);
    return .{ .f32 = try weights.WeightF32.fromSlice(ctx, .{ out_dim, in_dim }, values) };
}

fn goldenVector(ctx: *ExecContext, comptime tag: @TypeOf(.tag), values: []const f32) !fucina.Tensor(.{tag}) {
    return fucina.Tensor(.{tag}).fromSlice(ctx, .{values.len}, values);
}

/// Field-wise teardown for error paths (Layer's own deinit is private to
/// qwen3.zig) — same shape as qwen3_train_tests.zig's destroyLayer.
fn destroyLayer(layer: *Layer) void {
    switch (layer.ffn) {
        .dense => |*dense| {
            dense.down_proj.deinit();
            switch (dense.input_proj) {
                .separate => |*sep| {
                    sep.up_proj.deinit();
                    sep.gate_proj.deinit();
                },
                .fused => |*w| w.deinit(),
            }
        },
        .moe => unreachable, // golden layers are dense only
    }
    layer.o_proj.deinit();
    switch (layer.attn_proj) {
        .separate => |*sep| {
            sep.v_proj.deinit();
            sep.k_proj.deinit();
            sep.q_proj.deinit();
        },
        .fused => |*w| w.deinit(),
    }
    layer.ffn_norm.deinit();
    layer.k_norm.deinit();
    layer.q_norm.deinit();
    layer.attn_norm.deinit();
    layer.* = undefined;
}

fn buildGoldenLayer(ctx: *ExecContext, data: *const LayerGolden) !Layer {
    const cfg = golden_config;
    const q_dim = cfg.num_attention_heads * cfg.head_dim;
    const kv_dim = cfg.num_key_value_heads * cfg.head_dim;

    var attn_norm = try goldenVector(ctx, .embed, data.attn_norm);
    errdefer attn_norm.deinit();
    var q_norm = try goldenVector(ctx, .d, data.q_norm);
    errdefer q_norm.deinit();
    var k_norm = try goldenVector(ctx, .d, data.k_norm);
    errdefer k_norm.deinit();
    var ffn_norm = try goldenVector(ctx, .embed, data.ffn_norm);
    errdefer ffn_norm.deinit();

    var q_proj = try goldenLinear(ctx, data.q_proj, q_dim, cfg.hidden_size);
    errdefer q_proj.deinit();
    var k_proj = try goldenLinear(ctx, data.k_proj, kv_dim, cfg.hidden_size);
    errdefer k_proj.deinit();
    var v_proj = try goldenLinear(ctx, data.v_proj, kv_dim, cfg.hidden_size);
    errdefer v_proj.deinit();
    var o_proj = try goldenLinear(ctx, data.o_proj, cfg.hidden_size, q_dim);
    errdefer o_proj.deinit();

    var gate_proj = try goldenLinear(ctx, data.gate_proj, cfg.intermediate_size, cfg.hidden_size);
    errdefer gate_proj.deinit();
    var up_proj = try goldenLinear(ctx, data.up_proj, cfg.intermediate_size, cfg.hidden_size);
    errdefer up_proj.deinit();
    var down_proj = try goldenLinear(ctx, data.down_proj, cfg.hidden_size, cfg.intermediate_size);
    errdefer down_proj.deinit();

    return .{
        .attn_norm = attn_norm,
        .q_norm = q_norm,
        .k_norm = k_norm,
        .ffn_norm = ffn_norm,
        .attn_proj = .{ .separate = .{ .q_proj = q_proj, .k_proj = k_proj, .v_proj = v_proj } },
        .o_proj = o_proj,
        .ffn = .{ .dense = .{
            .input_proj = .{ .separate = .{ .gate_proj = gate_proj, .up_proj = up_proj } },
            .down_proj = down_proj,
        } },
    };
}

/// The golden model with separate projections and fixed f32 weights; tear
/// down with the public `Model.deinit`.
fn buildGoldenModel(ctx: *ExecContext) !qwen3.Model {
    const cfg = golden_config;
    const allocator = ctx.allocator;

    var token_embedding = try goldenLinear(ctx, &golden_token_embedding, cfg.vocab_size, cfg.hidden_size);
    errdefer token_embedding.deinit();
    var output_norm = try goldenVector(ctx, .embed, &golden_output_norm);
    errdefer output_norm.deinit();
    var output = try goldenLinear(ctx, &golden_output, cfg.vocab_size, cfg.hidden_size);
    errdefer output.deinit();

    const kv_head_for_head = try allocator.alloc(usize, cfg.num_attention_heads);
    errdefer allocator.free(kv_head_for_head);
    const heads_per_kv = cfg.num_attention_heads / cfg.num_key_value_heads;
    for (kv_head_for_head, 0..) |*kv_head, head_i| kv_head.* = head_i / heads_per_kv;

    const layers = try allocator.alloc(Layer, cfg.num_layers);
    errdefer allocator.free(layers);
    var built: usize = 0;
    errdefer for (layers[0..built]) |*layer| destroyLayer(layer);
    for (layers, &layer_goldens) |*layer, *data| {
        layer.* = try buildGoldenLayer(ctx, data);
        built += 1;
    }

    return .{
        .allocator = allocator,
        .config = cfg,
        .token_embedding = token_embedding,
        .output_norm = output_norm,
        .output = output,
        .layers = layers,
        .kv_head_for_head = kv_head_for_head,
        .weight_mapping = null,
    };
}

// ---------------------------------------------------------------------------
// Assertions.
// ---------------------------------------------------------------------------

/// Mixed absolute/relative tolerance per gradient element. seq-64 routes the
/// forward through the query-tiled online-softmax attention kernel (summation
/// order differs from a 3-pass softmax by ~1e-6 relative) and BLAS GEMM
/// blocking differs from torch's; the observed worst deviation is well inside
/// this bound (see the generator header).
fn expectGradClose(ctx: *ExecContext, param: anytype, expected: []const f32, layer_i: usize, what: []const u8) !void {
    var g = (try param.grad(ctx)) orelse return error.MissingGrad;
    defer g.deinit();
    const actual = try g.dataConst();
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual, 0..) |e, a, i| {
        const tol = 1e-5 + 2e-3 * @abs(e);
        if (!(@abs(e - a) <= tol)) {
            std.debug.print(
                "golden grad mismatch: layer {d} {s}[{d}]: expected {e}, got {e} (|d| = {e}, tol = {e})\\n",
                .{ layer_i, what, i, e, a, @abs(e - a), tol },
            );
            return error.GoldenGradMismatch;
        }
    }
}

test "qwen3 LoRA fine-tuning: loss and adapter grads match the PyTorch goldens" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try buildGoldenModel(&ctx);
    defer model.deinit();

    // The adapter-init seed is irrelevant: every A/B is overwritten below
    // with the golden values through the raw-tensor mutation seam (the same
    // in-place write path optim.loadStateDict uses).
    var trainer = try qwen3_train.Trainer(.{}).init(&ctx, &model, golden_lora, 1);
    defer trainer.deinit();
    for (trainer.adapters, &layer_goldens) |*ads, *data| {
        @memcpy(ads.q.a.value.data(), data.lora_q_a);
        @memcpy(ads.q.b.value.data(), data.lora_q_b);
        @memcpy(ads.v.a.value.data(), data.lora_v_a);
        @memcpy(ads.v.b.value.data(), data.lora_v_b);
    }

    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    const loss = try trainer.loss(&ctx, &golden_tokens, &golden_labels);
    try loss.backward(&ctx);

    const loss_value = try loss.item();
    try std.testing.expectApproxEqAbs(expected_loss, loss_value, 2e-4 * @abs(expected_loss));

    for (trainer.adapters, &layer_goldens, 0..) |*ads, *data, layer_i| {
        try expectGradClose(&ctx, &ads.q.a, data.grad_q_a, layer_i, "q.lora_a");
        try expectGradClose(&ctx, &ads.q.b, data.grad_q_b, layer_i, "q.lora_b");
        try expectGradClose(&ctx, &ads.v.a, data.grad_v_a, layer_i, "v.lora_a");
        try expectGradClose(&ctx, &ads.v.b, data.grad_v_b, layer_i, "v.lora_b");
    }
}
"""

zig_source = (
    ZIG_TEMPLATE
    .replace("@@CMD@@", GEN_CMD)
    .replace("@@TORCH@@", torch.__version__)
    .replace("@@LOSS@@", fmt(loss_value))
    .replace("@@SHA@@", data_sha)
    .replace("@@VOCAB@@", str(VOCAB))
    .replace("@@HIDDEN@@", str(HIDDEN))
    .replace("@@FFN@@", str(FFN))
    .replace("@@LAYERS@@", str(LAYERS))
    .replace("@@HEADS@@", str(HEADS))
    .replace("@@KV_HEADS@@", str(KV_HEADS))
    .replace("@@HEAD_DIM@@", str(HEAD_DIM))
    .replace("@@RANK@@", str(RANK))
    .replace("@@ALPHA@@", fmt(ALPHA))
    .replace("@@DATA@@", data_section)
)

if len(sys.argv) > 1:
    with open(sys.argv[1], "w") as f:
        f.write(zig_source)
else:
    sys.stdout.write(zig_source)

print(f"torch {torch.__version__}", file=sys.stderr)
print(f"loss = {loss_value!r}", file=sys.stderr)
print(f"golden-data-section sha256 = {data_sha}", file=sys.stderr)
for layer_i, grad in enumerate(grads):
    for name, g in grad.items():
        print(
            f"layer {layer_i} {name}: max|g| = {g.abs().max().item():.6g}, "
            f"mean|g| = {g.abs().mean().item():.6g}",
            file=sys.stderr,
        )
