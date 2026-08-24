//! Delta-rule linear attention — the recurrence family behind Kimi Delta
//! Attention (Kimi-Linear / Kimi K3) and Gated DeltaNet descendants.
//!
//! `kdaRecurrent` is the fused sequence kernel over post-projection inputs
//! (`chunk_kda`/`fused_recurrent_kda` with every in-kernel option on, the
//! form the Kimi models call): per token t and head h, with S ∈ R^{K×V},
//!
//!     q̂ = l2norm(q_t)·K^(-1/2)   k̂ = l2norm(k_t)   β = sigmoid(beta_t)
//!     g = −exp(A_log)·softplus(g_t + dt_bias)      (per-head or per-channel A_log)
//!     S ← S ⊙ exp(g)[k-broadcast]                  (decay along K)
//!     Δ = v_t − k̂ᵀ·S                               (delta rule)
//!     S ← S + (β·k̂) ⊗ Δ
//!     o_t = q̂ᵀ·S
//!
//! Heads are independent: work splits by whole heads (single writer per
//! output row and per state plane, bitwise identical for any task count);
//! within a head the fold is sequential over t with @Vector arithmetic
//! across the V lanes. Everything runs in f32, matching the reference
//! kernel's internal precision. Inference kernel: no backward record —
//! training support belongs to a chunked VJP or input-capture replay.

const std = @import("std");
const parallel = @import("../parallel.zig");
const tensor = @import("../tensor.zig");
const thread = @import("../thread.zig");
const ExecContext = @import("../exec.zig").ExecContext;

const Tensor = tensor.Tensor;

const Vec = @Vector(8, f32);
const vector_width = 8;

/// Upper bound for the per-token stack buffers (Δ row) and the decay
/// precompute: covers head dims through the full K3's 128 with headroom.
const max_head_dim = 512;

pub const KdaResult = struct {
    /// [seq, heads, v_dim]
    o: Tensor,
    /// [heads, k_dim, v_dim] — the final recurrent state (decode resume).
    state: Tensor,

    pub fn deinit(self: *KdaResult) void {
        self.o.deinit();
        self.state.deinit();
    }
};

const KdaTask = struct {
    q: []const f32,
    k: []const f32,
    v: []const f32,
    g: []const f32,
    beta: []const f32,
    a_log: []const f32,
    dt_bias: []const f32,
    initial_state: ?[]const f32,
    o: []f32,
    state: []f32,
    seq: usize,
    heads: usize,
    k_dim: usize,
    v_dim: usize,
    scale: f32,
    head_start: usize,
    head_end: usize,
};

fn runKdaTask(task: *const KdaTask) void {
    kdaHeads(task.*);
}

fn kdaHeads(task: KdaTask) void {
    const K = task.k_dim;
    const V = task.v_dim;
    const qk_stride = task.heads * K;
    const v_stride = task.heads * V;

    for (task.head_start..task.head_end) |h| {
        const state = task.state[h * K * V ..][0 .. K * V];
        if (task.initial_state) |initial| {
            @memcpy(state, initial[h * K * V ..][0 .. K * V]);
        } else {
            @memset(state, 0);
        }

        // exp(A_log) per channel of this head, hoisted out of the fold.
        var a_exp_buf: [max_head_dim]f32 = undefined;
        const a_exp = a_exp_buf[0..K];
        for (a_exp, 0..) |*value, ki| {
            value.* = @exp(task.a_log[if (task.a_log.len == task.heads) h else ki]);
        }

        for (0..task.seq) |t| {
            const q_row = task.q[t * qk_stride + h * K ..][0..K];
            const k_row = task.k[t * qk_stride + h * K ..][0..K];
            const v_row = task.v[t * v_stride + h * V ..][0..V];
            const g_row = task.g[t * qk_stride + h * K ..][0..K];
            const beta_raw = task.beta[t * task.heads + h];
            const o_row = task.o[t * v_stride + h * V ..][0..V];

            const beta = 1.0 / (1.0 + @exp(-beta_raw));
            const q_inv = invL2Norm(q_row);
            const k_inv = invL2Norm(k_row);

            // Row pass 1: decay S along K and accumulate Δ = v − k̂ᵀ·S.
            var delta: [max_head_dim]f32 = undefined;
            const delta_row = delta[0..V];
            @memcpy(delta_row, v_row);
            for (0..K) |ki| {
                const s_row = state[ki * V ..][0..V];
                // g = −exp(A_log)·softplus(g_raw + dt_bias); decay = exp(g).
                const decay = @exp(-a_exp[ki] * softplus(g_row[ki] + task.dt_bias[h * K + ki]));
                const kh = k_row[ki] * k_inv;
                vecDecayAndDot(s_row, delta_row, decay, kh);
            }
            // Row pass 2: rank-1 update S += (β·k̂) ⊗ Δ and o = q̂ᵀ·S.
            @memset(o_row, 0);
            for (0..K) |ki| {
                const s_row = state[ki * V ..][0..V];
                const bk = beta * k_row[ki] * k_inv;
                const qh = q_row[ki] * q_inv * task.scale;
                vecUpdateAndOut(s_row, delta_row, o_row, bk, qh);
            }
        }
    }
}

/// s ← s·decay lane-wise, then delta ← delta − kh·s (the decayed s).
inline fn vecDecayAndDot(s: []f32, delta: []f32, decay: f32, kh: f32) void {
    const dv: Vec = @splat(decay);
    const kv: Vec = @splat(kh);
    var i: usize = 0;
    while (i + vector_width <= s.len) : (i += vector_width) {
        const sv = @as(Vec, s[i..][0..vector_width].*) * dv;
        s[i..][0..vector_width].* = sv;
        delta[i..][0..vector_width].* = @as(Vec, delta[i..][0..vector_width].*) - kv * sv;
    }
    while (i < s.len) : (i += 1) {
        const sv = s[i] * decay;
        s[i] = sv;
        delta[i] -= kh * sv;
    }
}

/// s ← s + bk·delta, then o ← o + qh·s (the updated s).
inline fn vecUpdateAndOut(s: []f32, delta: []const f32, o: []f32, bk: f32, qh: f32) void {
    const bv: Vec = @splat(bk);
    const qv: Vec = @splat(qh);
    var i: usize = 0;
    while (i + vector_width <= s.len) : (i += vector_width) {
        const sv = @as(Vec, s[i..][0..vector_width].*) + bv * @as(Vec, delta[i..][0..vector_width].*);
        s[i..][0..vector_width].* = sv;
        o[i..][0..vector_width].* = @as(Vec, o[i..][0..vector_width].*) + qv * sv;
    }
    while (i < s.len) : (i += 1) {
        const sv = s[i] + bk * delta[i];
        s[i] = sv;
        o[i] += qh * sv;
    }
}

inline fn invL2Norm(row: []const f32) f32 {
    var acc: Vec = @splat(0);
    var i: usize = 0;
    while (i + vector_width <= row.len) : (i += vector_width) {
        const rv: Vec = row[i..][0..vector_width].*;
        acc += rv * rv;
    }
    var sum = @reduce(.Add, acc);
    while (i < row.len) : (i += 1) sum += row[i] * row[i];
    // torch.nn.functional.normalize semantics: x / max(||x||, eps), eps 1e-12.
    return 1.0 / @max(@sqrt(sum), 1e-12);
}

inline fn softplus(value: f32) f32 {
    // Sign-stable log1p(exp(x)) (the UnaryOp .softplus formula).
    return if (value > 0) value + @log(1 + @exp(-value)) else @log(1 + @exp(value));
}

/// Fused KDA sequence recurrence. Inputs are the POST-convolution
/// projections: `q`/`k` [seq, heads, k_dim], `v` [seq, heads, v_dim],
/// `g_raw` [seq, heads, k_dim] (pre-gate low-rank decay), `beta_raw`
/// [seq, heads] (pre-sigmoid), `a_log` of length heads (per-head) or
/// k_dim (per-channel), `dt_bias` [heads*k_dim], and an optional
/// [heads, k_dim, v_dim] initial state. `scale` <= 0 selects the default
/// k_dim^(-1/2).
pub fn kdaRecurrent(
    ctx: *ExecContext,
    q: *const Tensor,
    k: *const Tensor,
    v: *const Tensor,
    g_raw: *const Tensor,
    beta_raw: *const Tensor,
    a_log: []const f32,
    dt_bias: []const f32,
    initial_state: ?*const Tensor,
    scale: f32,
) !KdaResult {
    const qv = try q.rankView(3);
    const vv = try v.rankView(3);
    const seq = qv.dim(0);
    const heads = qv.dim(1);
    const k_dim = qv.dim(2);
    const v_dim = vv.dim(2);
    if (vv.dim(0) != seq or vv.dim(1) != heads) return tensor.TensorError.ShapeMismatch;
    if (v_dim > max_head_dim or k_dim > max_head_dim) return tensor.TensorError.UnsupportedView;
    if (a_log.len != heads and a_log.len != k_dim) return tensor.TensorError.ShapeMismatch;
    if (dt_bias.len != heads * k_dim) return tensor.TensorError.ShapeMismatch;

    var qq = try ctx.prepareContiguous(.f32, q);
    defer qq.deinit();
    var kk = try ctx.prepareContiguous(.f32, k);
    defer kk.deinit();
    var vvp = try ctx.prepareContiguous(.f32, v);
    defer vvp.deinit();
    var gg = try ctx.prepareContiguous(.f32, g_raw);
    defer gg.deinit();
    var bb = try ctx.prepareContiguous(.f32, beta_raw);
    defer bb.deinit();

    var o = try ctx.empty(.f32, .{ seq, heads, v_dim });
    errdefer o.deinit();
    var state = try ctx.empty(.f32, .{ heads, k_dim, v_dim });
    errdefer state.deinit();

    const base = KdaTask{
        .q = qq.tensor().dataConst(),
        .k = kk.tensor().dataConst(),
        .v = vvp.tensor().dataConst(),
        .g = gg.tensor().dataConst(),
        .beta = bb.tensor().dataConst(),
        .a_log = a_log,
        .dt_bias = dt_bias,
        .initial_state = if (initial_state) |s| s.dataConst() else null,
        .o = o.data(),
        .state = state.data(),
        .seq = seq,
        .heads = heads,
        .k_dim = k_dim,
        .v_dim = v_dim,
        .scale = if (scale > 0) scale else 1.0 / @sqrt(@as(f32, @floatFromInt(k_dim))),
        .head_start = 0,
        .head_end = heads,
    };

    const work = parallel.saturatedMul3(seq, heads, k_dim * v_dim);
    const dispatched = work >= parallel.attention_work_threshold and
        ctx.dispatchRange(KdaTask, "head_start", "head_end", base, heads, runKdaTask);
    if (!dispatched) kdaHeads(base);

    return .{ .o = o, .state = state };
}

test {
    _ = @import("delta_attention_tests.zig");
}
