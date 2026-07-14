# Golden-value generator for Fucina's cartridge mechanism parity test.
#
# Environment + invocation (torch CPU only, same venv as the other golden
# generators):
#   python3.11 -m venv /tmp/fucina-golden
#   /tmp/fucina-golden/bin/pip install torch==2.12
#   /tmp/fucina-golden/bin/python tools/gen_cartridge_goldens.py src/llm/cartridge_golden_tests.zig
#
"""Golden-value generator for Fucina's cartridge (trainable KV prefix) test.

An independent PyTorch (autograd) implementation of the EXACT mechanism that
src/llm/cartridge.zig composes out of public ops — verified op by op against
the Zig source and the HazyResearch/cartridges reference semantics:

  - q/k/v = x @ W.T (W stored [out, in])                    frozen-RHS dot
  - head split [seq, heads, d] / [seq, kv_heads, d]
  - RoPE in `.half` (NeoX) mode at OFFSET positions p..p+T-1: pairs
    (i, i + d/2) rotate with theta = pos / theta_base^(2i/d) — real tokens
    sit AFTER the p cartridge rows, exactly the reference's
    `position_ids + cartridge_len` (modeling_llama.py)      exec.ropeAxisRankWithTable
  - cartridge K/V: [p, kv_heads, d] rows in post-RoPE space, row 0 a frozen
    constant (attention sink), rows 1..p-1 trainable leaves; the full key
    sequence is cat([sink, trainable, k_tokens])            cartridge.LayerKv.catK/catV
  - grouped attention with kv_seq = p + T > q_seq = T: query i attends kv
    rows 0..p+i (end-aligned `source_offset = kv_seq - q_seq` causal kernel,
    identical to the reference block mask: the cartridge is visible to every
    query, real tokens are causal)                          exec.groupedCausalAttention
  - logits = attn @ Wout.T
  - distillation loss: mean over sparse teacher top-k entries of
    -exp(teacher_logprob) * log_softmax(logits)[pos - 1, token] — the
    reference train.py objective (tail mass dropped, not renormalized,
    student prediction read from the PREVIOUS row)          cartridge.distillLoss

Only the cartridge rows require grad (the frozen stack stops every other
path), matching a real cartridge training step.

The script EMITS the complete Zig test file (data + test code) and prints
the torch loss + a sha256 checksum of the golden-data section to stderr so
the values inside the Zig file are auditable.

Usage:
    /tmp/fucina-golden/bin/python tools/gen_cartridge_goldens.py \
        src/llm/cartridge_golden_tests.zig
"""

import hashlib
import math
import sys

import torch

torch.manual_seed(0xCA127D6E)

# ---------------------------------------------------------------- dimensions
SEQ = 6                # real tokens
P = 4                  # cartridge rows: 1 frozen sink + 3 trainable
FROZEN = 1
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 6
EMBED = 8
VOCAB = 9
ROPE_THETA = 10_000.0

GEN_CMD = (
    "/tmp/fucina-golden/bin/python tools/gen_cartridge_goldens.py "
    "src/llm/cartridge_golden_tests.zig"
)


def uniform(shape, lo, hi):
    return torch.rand(shape, dtype=torch.float32) * (hi - lo) + lo


x = uniform((SEQ, EMBED), -0.5, 0.5)
w_q = uniform((HEADS * HEAD_DIM, EMBED), -0.4, 0.4)
w_k = uniform((KV_HEADS * HEAD_DIM, EMBED), -0.4, 0.4)
w_v = uniform((KV_HEADS * HEAD_DIM, EMBED), -0.4, 0.4)
w_out = uniform((VOCAB, HEADS * HEAD_DIM), -0.3, 0.3)
zk_rows = uniform((P, KV_HEADS, HEAD_DIM), -0.6, 0.6)
zv_rows = uniform((P, KV_HEADS, HEAD_DIM), -0.6, 0.6)

# Sparse teacher top-k targets: (packed target position, token id, prob).
# Rows keep <= 0.99 cumulative mass like the reference's flatten(threshold).
TARGETS = [
    (2, 1, 0.62),
    (2, 7, 0.30),
    (3, 0, 0.50),
    (3, 4, 0.35),
    (3, 8, 0.14),
    (5, 3, 0.97),
]

# RoPE table exactly as exec.prepareRopeTable: f32 math, angle =
# pos / theta_base^(2*pair/d), positions p..p+SEQ-1 (tokens sit after the
# cartridge rows).
positions = (torch.arange(SEQ, dtype=torch.float32) + P).unsqueeze(1)     # [seq, 1]
pair_exponent = (2.0 * torch.arange(HEAD_DIM // 2, dtype=torch.float32)) / HEAD_DIM
angles = positions / torch.pow(torch.tensor(ROPE_THETA, dtype=torch.float32), pair_exponent)
ROPE_SIN = torch.sin(angles)                                              # [seq, d/2]
ROPE_COS = torch.cos(angles)


def rope_half(t):
    # exec.ropeAxisRankWithTable `.half` mode: pair (i, i + d/2), NeoX-style.
    half = HEAD_DIM // 2
    first, second = t[..., :half], t[..., half:]
    sin = ROPE_SIN[:, None, :]
    cos = ROPE_COS[:, None, :]
    return torch.cat([first * cos - second * sin, first * sin + second * cos], dim=-1)


# ------------------------------------------------------------------ forward
zk_sink = zk_rows[:FROZEN]                       # frozen constant
zv_sink = zv_rows[:FROZEN]
zk_train = zk_rows[FROZEN:].clone().requires_grad_()
zv_train = zv_rows[FROZEN:].clone().requires_grad_()

q3 = (x @ w_q.T).reshape(SEQ, HEADS, HEAD_DIM)
k3 = (x @ w_k.T).reshape(SEQ, KV_HEADS, HEAD_DIM)
v3 = (x @ w_v.T).reshape(SEQ, KV_HEADS, HEAD_DIM)
q_rope = rope_half(q3)
k_rope = rope_half(k3)

k_cat = torch.cat([zk_sink, zk_train, k_rope], dim=0)     # [p + seq, kv, d]
v_cat = torch.cat([zv_sink, zv_train, v3], dim=0)

# exec.groupedCausalAttention with source_offset = kv_seq - q_seq = p:
# query i attends kv rows 0..p+i.
scale = 1.0 / math.sqrt(HEAD_DIM)
kv_seq = P + SEQ
col = torch.arange(kv_seq).unsqueeze(0)                   # [1, kv_seq]
rowq = torch.arange(SEQ).unsqueeze(1)                     # [seq, 1]
disallowed = col > (rowq + P)
outs = []
for head in range(HEADS):
    kv_head = head // (HEADS // KV_HEADS)  # qwen3.Model.kv_head_for_head
    scores = (q_rope[:, head, :] @ k_cat[:, kv_head, :].T) * scale
    scores = scores.masked_fill(disallowed, float("-inf"))
    outs.append(torch.softmax(scores, dim=-1) @ v_cat[:, kv_head, :])
attn = torch.stack(outs, dim=1).reshape(SEQ, HEADS * HEAD_DIM)

logits = attn @ w_out.T                                   # [seq, vocab]

# cartridge.distillLoss: mean over entries of -p_teacher * logq[pos-1, token].
logq = torch.log_softmax(logits, dim=-1)
loss = torch.stack([
    -prob * logq[pos - 1, token] for (pos, token, prob) in TARGETS
]).mean()
loss.backward()

loss_value = loss.item()

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


def const_array(name, tensor):
    return f"const {name} = [_]f32{{\n{zig_array(tensor.detach(), '    ')}\n}};"


target_positions = ", ".join(str(pos) for (pos, _, _) in TARGETS)
target_tokens = ", ".join(str(token) for (_, token, _) in TARGETS)
target_logprobs = ", ".join(fmt(math.log(prob)) for (_, _, prob) in TARGETS)

data_section = "\n\n".join([
    f"const expected_loss: f32 = {fmt(loss_value)};",
    f"const golden_target_positions = [_]usize{{ {target_positions} }};",
    f"const golden_target_tokens = [_]usize{{ {target_tokens} }};",
    f"const golden_target_logprobs = [_]f32{{ {target_logprobs} }};",
    const_array("golden_x", x),
    const_array("golden_w_q", w_q),
    const_array("golden_w_k", w_k),
    const_array("golden_w_v", w_v),
    const_array("golden_w_out", w_out),
    const_array("golden_zk", zk_rows),
    const_array("golden_zv", zv_rows),
    const_array("golden_logits", logits),
    const_array("golden_grad_zk", zk_train.grad),
    const_array("golden_grad_zv", zv_train.grad),
]) + "\n"

data_sha = hashlib.sha256(data_section.encode()).hexdigest()

header = f"""//! PyTorch golden-parity test for the cartridge mechanism gradients.
//!
//! GENERATED FILE — regenerate with (venv recipe in the script's header):
//!     {GEN_CMD}
//! Generator: torch {torch.__version__} (CPU, float32), seed 0xCA127D6E.
//! Expected loss (torch): {fmt(loss_value)}. Golden-data-section sha256: {data_sha}.
//!
//! An independent PyTorch autograd implementation of the exact cartridge
//! mechanism (frozen q/k/v/out projections, `.half` RoPE at offset positions
//! p..p+T-1, a [p, kv_head, d] cartridge with a frozen sink row and trainable
//! rows concatenated before end-aligned grouped causal attention, and the
//! sparse teacher top-k distillation loss with the pos-1 shift) produced this
//! loss and these cartridge gradients from the same constants. The test
//! rebuilds the pipeline through cartridge.zig + public ops, runs
//! distillLoss() + backward(), and asserts parity:
//!   - loss within 2e-4 relative,
//!   - every logit and cartridge-gradient element within
//!     |d| <= 1e-5 + 2e-3 * |expected| (BLAS/online-softmax summation orders
//!     differ from torch's).
"""

body = """
const std = @import("std");
const fucina = @import("fucina");
const cartridge = @import("cartridge.zig");

const ExecContext = fucina.ExecContext;
const Tensor = fucina.Tensor;

const seq = 6;
const p = 4; // 1 frozen sink + 3 trainable rows
const frozen_prefix = 1;
const heads = 4;
const kv_heads = 2;
const head_dim = 6;
const embed = 8;
const vocab = 9;
const rope_theta = 10_000.0;
const kv_head_for_head = [_]usize{ 0, 0, 1, 1 };

// ---------------------------------------------------------------------------
// Golden data (generated; see the header comment).
// ---------------------------------------------------------------------------

@@DATA@@
// ---------------------------------------------------------------------------

fn expectClose(expected: []const f32, actual: []const f32) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |want, got| {
        const tol = 1e-5 + 2e-3 * @abs(want);
        if (@abs(want - got) > tol) {
            std.debug.print("golden mismatch: want {d} got {d} (tol {d})\\n", .{ want, got, tol });
            return error.GoldenMismatch;
        }
    }
}

test "cartridge attention + distillation matches the PyTorch goldens" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var cart = try cartridge.Cartridge.initFromRows(
        &ctx,
        allocator,
        frozen_prefix,
        p,
        kv_heads,
        head_dim,
        &.{&golden_zk},
        &.{&golden_zv},
    );
    defer cart.deinit();

    var x = try Tensor(.{ .seq, .embed }).fromSlice(&ctx, .{ seq, embed }, &golden_x);
    defer x.deinit();
    var w_q = try Tensor(.{ .q, .embed }).fromSlice(&ctx, .{ heads * head_dim, embed }, &golden_w_q);
    defer w_q.deinit();
    var w_k = try Tensor(.{ .k, .embed }).fromSlice(&ctx, .{ kv_heads * head_dim, embed }, &golden_w_k);
    defer w_k.deinit();
    var w_v = try Tensor(.{ .v, .embed }).fromSlice(&ctx, .{ kv_heads * head_dim, embed }, &golden_w_v);
    defer w_v.deinit();
    var w_out = try Tensor(.{ .vocab, .attn }).fromSlice(&ctx, .{ vocab, heads * head_dim }, &golden_w_out);
    defer w_out.deinit();

    // Real tokens sit AFTER the cartridge: RoPE positions p..p+seq-1.
    var positions: [seq]i32 = undefined;
    for (&positions, 0..) |*pos, i| pos.* = @intCast(p + i);
    var table = try ctx.prepareRopeTable(&positions, head_dim, rope_theta, false);
    defer table.deinit();

    // The graph (including the distillLoss composite) runs under an exec
    // scope so every intermediate survives until backward(); the cartridge
    // leaves above were created OUTSIDE the scope and persist their grads.
    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);

    var q = try x.dot(&ctx, &w_q, .embed);
    defer q.deinit();
    var q3 = try q.split(&ctx, .q, .{ .head, .d }, .{ heads, head_dim });
    defer q3.deinit();
    var q_rope = try q3.rope(&ctx, .seq, .d, &table, .half);
    defer q_rope.deinit();

    var k = try x.dot(&ctx, &w_k, .embed);
    defer k.deinit();
    var k3 = try k.split(&ctx, .k, .{ .kv_head, .d }, .{ kv_heads, head_dim });
    defer k3.deinit();
    var k_rope = try k3.rope(&ctx, .seq, .d, &table, .half);
    defer k_rope.deinit();

    var v = try x.dot(&ctx, &w_v, .embed);
    defer v.deinit();
    var v3 = try v.split(&ctx, .v, .{ .kv_head, .d }, .{ kv_heads, head_dim });
    defer v3.deinit();

    var k_cat = try cart.layers[0].catK(&ctx, &k_rope);
    defer k_cat.deinit();
    var v_cat = try cart.layers[0].catV(&ctx, &v3);
    defer v_cat.deinit();

    const attn_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    var attn = try q_rope.groupedAttention(&ctx, &k_cat, &v_cat, kv_head_for_head[0..], .attn, attn_scale, .{});
    defer attn.deinit();

    var logits = try attn.dot(&ctx, &w_out, .attn);
    defer logits.deinit();
    try expectClose(&golden_logits, try logits.dataConst());

    var loss = try cartridge.distillLoss(&ctx, &logits, .{
        .positions = &golden_target_positions,
        .tokens = &golden_target_tokens,
        .logprobs = &golden_target_logprobs,
    }, .{});
    defer loss.deinit();
    const loss_value = (try loss.dataConst())[0];
    try std.testing.expect(@abs(loss_value - expected_loss) <= 2e-4 * @abs(expected_loss));

    try loss.backward(&ctx);

    var grad_zk = (try cart.layers[0].k.grad(&ctx)).?;
    defer grad_zk.deinit();
    try expectClose(&golden_grad_zk, grad_zk.asRawTensor().dataConst());
    var grad_zv = (try cart.layers[0].v.grad(&ctx)).?;
    defer grad_zv.deinit();
    try expectClose(&golden_grad_zv, grad_zv.asRawTensor().dataConst());
}
"""

output = header + body.replace("@@DATA@@", data_section)

out_path = sys.argv[1] if len(sys.argv) > 1 else "src/llm/cartridge_golden_tests.zig"
with open(out_path, "w") as f:
    f.write(output)

print(f"torch loss = {loss_value!r}", file=sys.stderr)
print(f"golden-data-section sha256 = {data_sha}", file=sys.stderr)
print(f"wrote {out_path}", file=sys.stderr)
