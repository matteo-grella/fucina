# Golden-value generator for Fucina's Engram module parity test.
#
# Environment + invocation (torch CPU only, same venv as the other golden
# generators; numpy is needed for the reference int64 hash semantics):
#   python3.11 -m venv /tmp/fucina-golden
#   /tmp/fucina-golden/bin/pip install torch==2.12 numpy
#   /tmp/fucina-golden/bin/python tools/gen_engram_goldens.py src/llm/engram_golden_tests.zig
#
"""Golden-value generator for Fucina's Engram (conditional n-gram memory).

An independent PyTorch/numpy implementation of the EXACT mechanism that
src/llm/engram.zig composes out of public ops — verified statement by
statement against the pinned reference (refs/engram/engram_demo_v1.py,
deepseek-ai/Engram @fb7f84a):

  - token compression: raw ids map through a lookup table; the pad id is
    compressed too                                      HashPlan.compressInto
  - per-layer multipliers: numpy default_rng(seed + 10007*layer).integers(
    0, half_bound)*2 + 1 with half_bound = (int64max // vocab) // 2 — the
    reference draw, emitted verbatim so the Zig side injects them
                                                        HashPlan.initWithMultipliers
  - head table sizes: consecutive distinct primes searched upward from
    engram_vocab_size[order]-1 with a GLOBAL seen set over layers x orders
    x heads                                             HashPlan prime chain
  - the hash: mix_n = XOR_{k<n}(shift_k(ids) * mult[k]) over WRAPPING int64,
    row = mix_n mod prime (numpy floored %), + per-head table offset
                                                        HashPlan.hashInto
  - multi-head embedding: one concatenated [total_rows, head_dim] table,
    row gather, heads flattened to [T, engram_hidden]   Layer.forward gather
  - per-stream gates: sigmoid(signed_sqrt((rms(key)·rms(query))/sqrt(d)))
    with signed_sqrt(x) = sign(x)*sqrt(clamp_min(|x|, 1e-6)); key =
    emb @ Wk.T + bk, norms are weighted RMSNorm (eps = finfo(f32).eps, the
    torch nn.RMSNorm default the reference relies on)   Layer.forward gates
  - value = gate_g * (emb @ Wv.T + bv); ShortConv = per-stream weighted
    RMSNorm (eps 1e-5) -> depthwise causal conv1d (kernel_size taps,
    dilation = max_ngram_size, no bias) -> SiLU; output = value + conv
                                                        Layer.forward
  - loss = sum(output * cotangent), gradients for every parameter and the
    hidden states (the graft path needs grad through the query norms).

The script EMITS the complete Zig test file (data + test code) and prints
the torch loss + a sha256 checksum of the golden-data section to stderr so
the values inside the Zig file are auditable.

Usage:
    /tmp/fucina-golden/bin/python tools/gen_engram_goldens.py \
        src/llm/engram_golden_tests.zig
"""

import hashlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(0xE569A301)

# ---------------------------------------------------------------- dimensions
HIDDEN = 10
HC_MULT = 2
MAX_NGRAM = 3
N_EMBED_PER_NGRAM = 8
N_HEAD_PER_NGRAM = 2
ENGRAM_VOCAB = [23, 31]
KERNEL = 3
DILATION = MAX_NGRAM
RAW_VOCAB = 40
COMPRESSED_VOCAB = 17
PAD_ID_RAW = 2
SEED = 0
LAYER_IDS = [0, 2]
SLOT = 0  # the layer under test
T = 9

ORDERS = MAX_NGRAM - 1
HEADS = ORDERS * N_HEAD_PER_NGRAM
HEAD_DIM = N_EMBED_PER_NGRAM // N_HEAD_PER_NGRAM
ENGRAM_HIDDEN = ORDERS * N_EMBED_PER_NGRAM
GATE_EPS = float(torch.finfo(torch.float32).eps)  # nn.RMSNorm default
CONV_EPS = 1e-5

GEN_CMD = (
    "/tmp/fucina-golden/bin/python tools/gen_engram_goldens.py "
    "src/llm/engram_golden_tests.zig"
)

# ------------------------------------------------------------------- hashing
LOOKUP = np.array([i % COMPRESSED_VOCAB for i in range(RAW_VOCAB)], dtype=np.int64)
PAD_COMPRESSED = int(LOOKUP[PAD_ID_RAW])
RAW_IDS = np.array([5, 1, 3, 3, 39, 0, 2, 23, 17], dtype=np.int64)
assert len(RAW_IDS) == T
COMPRESSED_IDS = LOOKUP[RAW_IDS]

# Reference multiplier draw (engram_demo_v1.py NgramHashMapping.__init__).
max_long = np.iinfo(np.int64).max
half_bound = max(1, (max_long // COMPRESSED_VOCAB) // 2)
MULTIPLIERS = {}
for layer_id in LAYER_IDS:
    g = np.random.default_rng(int(SEED + 10007 * int(layer_id)))
    r = g.integers(low=0, high=half_bound, size=(MAX_NGRAM,), dtype=np.int64)
    MULTIPLIERS[layer_id] = r * 2 + 1


def is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            return False
        f += 2
    return True


# Reference prime chain (calculate_vocab_size_across_layers): global seen
# set over layer x order x head.
seen = set()
HEAD_PRIMES = {}
for layer_id in LAYER_IDS:
    per_layer = []
    for order in range(ORDERS):
        start = ENGRAM_VOCAB[order] - 1
        row = []
        for _ in range(N_HEAD_PER_NGRAM):
            candidate = start + 1
            while not is_prime(candidate) or candidate in seen:
                candidate += 1
            seen.add(candidate)
            row.append(candidate)
            start = candidate
        per_layer.append(row)
    HEAD_PRIMES[layer_id] = per_layer

layer = LAYER_IDS[SLOT]
primes_flat = [p for row in HEAD_PRIMES[layer] for p in row]
offsets = np.cumsum([0] + primes_flat[:-1]).tolist()
TABLE_ROWS = int(sum(primes_flat))


def hash_rows(ids, layer_id):
    """Reference _get_ngram_hashes + the concatenated-table offsets."""
    mults = MULTIPLIERS[layer_id]
    shifts = []
    for k in range(MAX_NGRAM):
        shifted = np.concatenate([np.full(k, PAD_COMPRESSED, dtype=np.int64), ids[: len(ids) - k]])
        shifts.append(shifted)
    out = np.zeros((len(ids), HEADS), dtype=np.int64)
    with np.errstate(over="ignore"):
        for n in range(2, MAX_NGRAM + 1):
            mix = shifts[0] * mults[0]
            for k in range(1, n):
                mix = np.bitwise_xor(mix, shifts[k] * mults[k])
            order = n - 2
            for j in range(N_HEAD_PER_NGRAM):
                h = order * N_HEAD_PER_NGRAM + j
                out[:, h] = (mix % HEAD_PRIMES[layer_id][order][j]) + offsets[h]
    return out


ROWS = hash_rows(COMPRESSED_IDS, layer)

# --------------------------------------------------------------- parameters
def uniform(shape, lo, hi):
    return torch.rand(shape, dtype=torch.float32) * (hi - lo) + lo


table = uniform((TABLE_ROWS, HEAD_DIM), -0.8, 0.8).requires_grad_()
key_w = [uniform((HIDDEN, ENGRAM_HIDDEN), -0.4, 0.4).requires_grad_() for _ in range(HC_MULT)]
key_b = [uniform((HIDDEN,), -0.2, 0.2).requires_grad_() for _ in range(HC_MULT)]
norm_key = [uniform((HIDDEN,), 0.6, 1.4).requires_grad_() for _ in range(HC_MULT)]
norm_query = [uniform((HIDDEN,), 0.6, 1.4).requires_grad_() for _ in range(HC_MULT)]
conv_norm = [uniform((HIDDEN,), 0.6, 1.4).requires_grad_() for _ in range(HC_MULT)]
value_w = uniform((HIDDEN, ENGRAM_HIDDEN), -0.4, 0.4).requires_grad_()
value_b = uniform((HIDDEN,), -0.2, 0.2).requires_grad_()
conv_w = uniform((HC_MULT * HIDDEN, KERNEL), -0.5, 0.5).requires_grad_()
hidden = uniform((T, HC_MULT, HIDDEN), -1.0, 1.0).requires_grad_()
cotangent = uniform((T, HC_MULT, HIDDEN), -1.0, 1.0)

# ------------------------------------------------------------------ forward
def rms_norm(x, weight, eps):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight


emb = table[torch.from_numpy(ROWS.reshape(-1))].reshape(T, ENGRAM_HIDDEN)

value = emb @ value_w.T + value_b  # [T, HIDDEN]

outs = []
conv_inputs = []
gates = []
for g in range(HC_MULT):
    key = emb @ key_w[g].T + key_b[g]
    nk = rms_norm(key, norm_key[g], GATE_EPS)
    nq = rms_norm(hidden[:, g, :], norm_query[g], GATE_EPS)
    gate = (nk * nq).sum(dim=-1) / (HIDDEN ** 0.5)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    gate = gate.sigmoid().unsqueeze(-1)  # [T, 1]
    gates.append(gate)
    vg = gate * value  # [T, HIDDEN]
    outs.append(vg)
    conv_inputs.append(rms_norm(vg, conv_norm[g], CONV_EPS))

conv_in = torch.cat(conv_inputs, dim=-1)  # [T, C]
x_bct = conv_in.T.unsqueeze(0)  # [1, C, T]
y_bct = F.conv1d(
    x_bct,
    conv_w.unsqueeze(1),  # [C, 1, K]
    bias=None,
    padding=(KERNEL - 1) * DILATION,
    dilation=DILATION,
    groups=HC_MULT * HIDDEN,
)[..., :T]
conv_out = F.silu(y_bct)[0].T  # [T, C]

output = torch.stack(
    [outs[g] + conv_out[:, g * HIDDEN : (g + 1) * HIDDEN] for g in range(HC_MULT)],
    dim=1,
)  # [T, G, HIDDEN]

loss = (output * cotangent).sum()
loss.backward()
loss_value = loss.item()

# ---------------------------------------------------------------- emit Zig
def fmt(value):
    # 9 significant digits round-trip any f32 exactly through Zig's
    # correctly-rounded decimal-literal parse.
    return f"{value:.9g}"


def zig_array(values, indent, kind="f32"):
    if kind == "f32":
        parts = [fmt(v) for v in np.asarray(values).reshape(-1).tolist()]
    else:
        parts = [str(int(v)) for v in np.asarray(values).reshape(-1).tolist()]
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


def const_array(name, tensor, kind="f32"):
    values = tensor.detach().numpy() if isinstance(tensor, torch.Tensor) else tensor
    zig_type = {"f32": "f32", "i64": "i64", "usize": "usize"}[kind]
    return f"const {name} = [_]{zig_type}{{\n{zig_array(values, '    ', kind)}\n}};"


mult_flat = np.concatenate([MULTIPLIERS[l] for l in LAYER_IDS])
primes_all = np.array(
    [p for l in LAYER_IDS for row in HEAD_PRIMES[l] for p in row], dtype=np.int64
)

pieces = [
    f"const expected_loss: f32 = {fmt(loss_value)};",
    const_array("golden_lookup", LOOKUP, "i64"),
    const_array("golden_raw_ids", RAW_IDS, "i64"),
    const_array("golden_compressed_ids", COMPRESSED_IDS, "i64"),
    const_array("golden_multipliers", mult_flat, "i64"),
    const_array("golden_head_primes", primes_all, "i64"),
    const_array("golden_rows", ROWS.astype(np.int64), "usize"),
    const_array("golden_table", table),
    const_array("golden_value_w", value_w),
    const_array("golden_value_b", value_b),
    const_array("golden_conv_w", conv_w),
    const_array("golden_hidden", hidden),
    const_array("golden_cotangent", cotangent),
    const_array("golden_output", output),
    const_array("golden_grad_hidden", hidden.grad),
    const_array("golden_grad_table", table.grad),
    const_array("golden_grad_value_w", value_w.grad),
    const_array("golden_grad_value_b", value_b.grad),
    const_array("golden_grad_conv_w", conv_w.grad),
]
for g in range(HC_MULT):
    pieces.append(const_array(f"golden_key_w_{g}", key_w[g]))
    pieces.append(const_array(f"golden_key_b_{g}", key_b[g]))
    pieces.append(const_array(f"golden_norm_key_{g}", norm_key[g]))
    pieces.append(const_array(f"golden_norm_query_{g}", norm_query[g]))
    pieces.append(const_array(f"golden_conv_norm_{g}", conv_norm[g]))
    pieces.append(const_array(f"golden_grad_key_w_{g}", key_w[g].grad))
    pieces.append(const_array(f"golden_grad_key_b_{g}", key_b[g].grad))
    pieces.append(const_array(f"golden_grad_norm_key_{g}", norm_key[g].grad))
    pieces.append(const_array(f"golden_grad_norm_query_{g}", norm_query[g].grad))
    pieces.append(const_array(f"golden_grad_conv_norm_{g}", conv_norm[g].grad))

data_section = "\n\n".join(pieces) + "\n"
data_sha = hashlib.sha256(data_section.encode()).hexdigest()

header = f"""//! PyTorch golden-parity test for the Engram module (forward + backward).
//!
//! GENERATED FILE — regenerate with (venv recipe in the script's header):
//!     {GEN_CMD}
//! Generator: torch {torch.__version__} / numpy {np.__version__} (CPU,
//! float32/int64), seed 0xE569A301.
//! Expected loss (torch): {fmt(loss_value)}. Golden-data-section sha256: {data_sha}.
//!
//! An independent PyTorch/numpy implementation of the exact reference
//! mechanism (refs/engram/engram_demo_v1.py: compression lookup, the
//! layer-seeded odd multipliers, the global prime chain, the wrapping
//! multiply-XOR-floored-mod hash, the concatenated multi-head embedding,
//! the per-stream RMS-normed signed-sqrt sigmoid gates, and the dilated
//! depthwise causal ShortConv) produced these values. The test rebuilds
//! the module through engram.zig + public ops and asserts:
//!   - the multipliers, primes, and row indices EXACTLY (integer path),
//!   - every output and gradient element within
//!     |d| <= 1e-5 + 2e-3 * |expected| (BLAS summation orders differ).
"""

body = """
const std = @import("std");
const fucina = @import("fucina");
const engram = @import("engram.zig");

const ExecContext = fucina.ExecContext;

const hidden_size = 10;
const hc_mult = 2;
const t_len = 9;
const heads = 4;
const head_dim = 4;
const engram_hidden = 16;
const kernel = 3;
const layer_ids = [_]usize{ 0, 2 };
const slot = 0;

// ---------------------------------------------------------------------------
// Golden data (generated; see the header comment).
// ---------------------------------------------------------------------------

@@DATA@@
// ---------------------------------------------------------------------------

fn expectClose(expected: []const f32, actual: []const f32) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |want, got| {
        const tol = 1e-5 + 2e-3 * @abs(want);
        if (!(@abs(want - got) <= tol)) {
            std.debug.print("golden mismatch: want {d}, got {d} (tol {d})\\n", .{ want, got, tol });
            return error.GoldenMismatch;
        }
    }
}

fn goldenConfig() engram.Config {
    return .{
        .hidden_size = hidden_size,
        .hc_mult = hc_mult,
        .max_ngram_size = 3,
        .n_embed_per_ngram = 8,
        .n_head_per_ngram = 2,
        .engram_vocab_size = &.{ 23, 31 },
        .kernel_size = kernel,
        .pad_id = 2,
    };
}

test "engram module matches the PyTorch/numpy goldens (forward + backward)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const cfg = goldenConfig();
    var plan = try engram.HashPlan.initWithMultipliers(allocator, cfg, &layer_ids, &golden_multipliers, &golden_lookup);
    defer plan.deinit();

    // Integer geometry pinned exactly against the independent numpy
    // implementation: primes, compression, and the hash itself.
    try std.testing.expectEqualSlices(i64, &golden_head_primes, plan.head_mods);

    var compressed: [t_len]i64 = undefined;
    try plan.compressInto(&golden_raw_ids, &compressed);
    try std.testing.expectEqualSlices(i64, &golden_compressed_ids, &compressed);

    var rows: [t_len * heads]usize = undefined;
    try plan.hashInto(slot, &compressed, &rows);
    try std.testing.expectEqualSlices(usize, &golden_rows, &rows);

    // Rebuild the layer from the golden parameters.
    var layer = engram.Layer{
        .allocator = allocator,
        .cfg = cfg,
        .table = try engram.Table.variableFromSlice(&ctx, .{ plan.table_rows[slot], head_dim }, &golden_table),
        .key_w = try allocator.alloc(engram.Proj, hc_mult),
        .key_b = try allocator.alloc(engram.Vec, hc_mult),
        .norm_key = try allocator.alloc(engram.Vec, hc_mult),
        .norm_query = try allocator.alloc(engram.Vec, hc_mult),
        .gate_bias = try allocator.alloc(engram.GateBias, hc_mult),
        .conv_norm = try allocator.alloc(engram.Vec, hc_mult),
        .value_w = try engram.Proj.variableFromSlice(&ctx, .{ hidden_size, engram_hidden }, &golden_value_w),
        .value_b = try engram.Vec.variableFromSlice(&ctx, .{hidden_size}, &golden_value_b),
        .conv_w = try engram.ConvKernel.variableFromSlice(&ctx, .{ hc_mult * hidden_size, kernel }, &golden_conv_w),
    };
    layer.key_w[0] = try engram.Proj.variableFromSlice(&ctx, .{ hidden_size, engram_hidden }, &golden_key_w_0);
    layer.key_w[1] = try engram.Proj.variableFromSlice(&ctx, .{ hidden_size, engram_hidden }, &golden_key_w_1);
    layer.key_b[0] = try engram.Vec.variableFromSlice(&ctx, .{hidden_size}, &golden_key_b_0);
    layer.key_b[1] = try engram.Vec.variableFromSlice(&ctx, .{hidden_size}, &golden_key_b_1);
    layer.norm_key[0] = try engram.Vec.variableFromSlice(&ctx, .{hidden_size}, &golden_norm_key_0);
    layer.norm_key[1] = try engram.Vec.variableFromSlice(&ctx, .{hidden_size}, &golden_norm_key_1);
    layer.norm_query[0] = try engram.Vec.variableFromSlice(&ctx, .{hidden_size}, &golden_norm_query_0);
    layer.norm_query[1] = try engram.Vec.variableFromSlice(&ctx, .{hidden_size}, &golden_norm_query_1);
    layer.conv_norm[0] = try engram.Vec.variableFromSlice(&ctx, .{hidden_size}, &golden_conv_norm_0);
    layer.conv_norm[1] = try engram.Vec.variableFromSlice(&ctx, .{hidden_size}, &golden_conv_norm_1);
    const zero_bias = [_]f32{0};
    layer.gate_bias[0] = try engram.GateBias.variableFromSlice(&ctx, .{1}, &zero_bias);
    layer.gate_bias[1] = try engram.GateBias.variableFromSlice(&ctx, .{1}, &zero_bias);
    defer layer.deinit();

    var hidden = try engram.Hidden.variableFromSlice(&ctx, .{ t_len, hc_mult, hidden_size }, &golden_hidden);
    defer hidden.deinit();
    var cot = try engram.Hidden.fromSlice(&ctx, .{ t_len, hc_mult, hidden_size }, &golden_cotangent);
    defer cot.deinit();

    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);

    var out = try layer.forward(&ctx, &hidden, &rows, null);
    defer out.deinit();
    try expectClose(&golden_output, out.asRawTensor().dataConst());

    var weighted = try out.mul(&ctx, &cot);
    defer weighted.deinit();
    var loss = try weighted.sumAll(&ctx);
    defer loss.deinit();
    const loss_value = try loss.item();
    const loss_tol = 1e-4 + 2e-4 * @abs(expected_loss);
    try std.testing.expect(@abs(loss_value - expected_loss) <= loss_tol);

    try loss.backward(&ctx);

    var grad_hidden = (try hidden.grad(&ctx)).?;
    defer grad_hidden.deinit();
    try expectClose(&golden_grad_hidden, grad_hidden.asRawTensor().dataConst());

    var grad_table = (try layer.table.grad(&ctx)).?;
    defer grad_table.deinit();
    try expectClose(&golden_grad_table, grad_table.asRawTensor().dataConst());

    var grad_value_w = (try layer.value_w.grad(&ctx)).?;
    defer grad_value_w.deinit();
    try expectClose(&golden_grad_value_w, grad_value_w.asRawTensor().dataConst());
    var grad_value_b = (try layer.value_b.grad(&ctx)).?;
    defer grad_value_b.deinit();
    try expectClose(&golden_grad_value_b, grad_value_b.asRawTensor().dataConst());

    var grad_conv_w = (try layer.conv_w.grad(&ctx)).?;
    defer grad_conv_w.deinit();
    try expectClose(&golden_grad_conv_w, grad_conv_w.asRawTensor().dataConst());

    const grad_key_w = [_][]const f32{ &golden_grad_key_w_0, &golden_grad_key_w_1 };
    const grad_key_b = [_][]const f32{ &golden_grad_key_b_0, &golden_grad_key_b_1 };
    const grad_norm_key = [_][]const f32{ &golden_grad_norm_key_0, &golden_grad_norm_key_1 };
    const grad_norm_query = [_][]const f32{ &golden_grad_norm_query_0, &golden_grad_norm_query_1 };
    const grad_conv_norm = [_][]const f32{ &golden_grad_conv_norm_0, &golden_grad_conv_norm_1 };
    for (0..hc_mult) |g| {
        var gw = (try layer.key_w[g].grad(&ctx)).?;
        defer gw.deinit();
        try expectClose(grad_key_w[g], gw.asRawTensor().dataConst());
        var gb = (try layer.key_b[g].grad(&ctx)).?;
        defer gb.deinit();
        try expectClose(grad_key_b[g], gb.asRawTensor().dataConst());
        var gnk = (try layer.norm_key[g].grad(&ctx)).?;
        defer gnk.deinit();
        try expectClose(grad_norm_key[g], gnk.asRawTensor().dataConst());
        var gnq = (try layer.norm_query[g].grad(&ctx)).?;
        defer gnq.deinit();
        try expectClose(grad_norm_query[g], gnq.asRawTensor().dataConst());
        var gcn = (try layer.conv_norm[g].grad(&ctx)).?;
        defer gcn.deinit();
        try expectClose(grad_conv_norm[g], gcn.asRawTensor().dataConst());
    }
}
"""


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <output.zig>", file=sys.stderr)
        return 1
    out_path = sys.argv[1]
    content = header + body.replace("@@DATA@@", data_section)
    with open(out_path, "w") as f:
        f.write(content)
    print(f"torch loss: {fmt(loss_value)}", file=sys.stderr)
    print(f"golden-data sha256: {data_sha}", file=sys.stderr)
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
