//! The LSTM recurrence over a sequence, one kernel for the block and its
//! BPTT backward. PyTorch's `nn.LSTM` cell: gates i, f, g, o over
//! `[x_t | h_{t-1}] · w + b` (w is `[in + H, 4H]`, the stacked
//! `[W_ih | W_hh]` transposed), `c_t = σ(f)·c_{t-1} + σ(i)·tanh(g)`,
//! `h_t = σ(o)·tanh(c_t)`. The output is `[2T, H]`: rows `0..T` are `h_t`,
//! rows `T..2T` are `c_t`, so the hidden sequence and the cell sequence are
//! contiguous views; the forward also writes the post-activation gates
//! `[T, 4H]`, which the backward reads instead of recomputing them.
//!
//! Per step the pre-activation is `in + H` axpys over the `4H` row (the
//! vector-times-matrix orientation, weights streamed once per step), the
//! gate nonlinearities are the lane kernels (`vecUnary`), and the cell and
//! hidden updates are lane loops. Serial along time by nature; the
//! reference arm is the same walk with scalar libm nonlinearities.

const std = @import("std");
const common = @import("common.zig");
const isa = @import("../isa.zig");
const ops = @import("../ops.zig");
const primitives = @import("primitives.zig");
const tensor = @import("../../tensor.zig");

const Tensor = tensor.Tensor;
const Vf32 = common.Vf32;
const vector_len = common.vector_len;

/// `out` `[2·seq, H]` (`h` rows then `c` rows), `gates` `[seq, 4H]`
/// (post-activation), `x` `[seq, in]`, `w` `[in + H, 4H]`, `b` `[4H]`,
/// `h0`/`c0` `[H]`.
pub fn lstmSequenceInto(out: *Tensor, gates: *Tensor, x: *const Tensor, w: *const Tensor, b: *const Tensor, h0: *const Tensor, c0: *const Tensor, seq: usize, in_size: usize, hidden: usize) void {
    if (comptime isa.reference) return scalar.lstmSequenceInto(out, gates, x, w, b, h0, c0, seq, in_size, hidden);
    forward(vecStep, out.data(), gates.data(), x.dataConst(), w.dataConst(), b.dataConst(), h0.dataConst(), c0.dataConst(), seq, in_size, hidden);
}

/// The gradients of the sequence: `gx` `[seq, in]`, `gw` `[in + H, 4H]`,
/// `gb` `[4H]`, `gh0`/`gc0` `[H]` (each optional), from `gy` `[2·seq, H]`
/// (the output's gradient: `dh` rows then `dc` rows), the forward's `x`,
/// `w`, `gates`, `out`, `h0`, `c0`, and a scratch of `4H + 5H` floats.
pub fn lstmSequenceBackwardInto(
    gx: ?*Tensor,
    gw: ?*Tensor,
    gb: ?*Tensor,
    gh0: ?*Tensor,
    gc0: ?*Tensor,
    gy: *const Tensor,
    x: *const Tensor,
    w: *const Tensor,
    gates: *const Tensor,
    out: *const Tensor,
    h0: *const Tensor,
    c0: *const Tensor,
    seq: usize,
    in_size: usize,
    hidden: usize,
    scratch: []f32,
) void {
    if (comptime isa.reference) return scalar.lstmSequenceBackwardInto(gx, gw, gb, gh0, gc0, gy, x, w, gates, out, h0, c0, seq, in_size, hidden, scratch);
    backward(vecTanhInto, gx, gw, gb, gh0, gc0, gy.dataConst(), x.dataConst(), w.dataConst(), gates.dataConst(), out.dataConst(), h0.dataConst(), c0.dataConst(), seq, in_size, hidden, scratch);
}

/// One step's nonlinearities in place over the `4H` pre-activations, then
/// the cell and hidden updates: the vector arm.
fn vecStep(pre: []f32, c: []f32, h: []f32, hidden: usize, tc: []f32) void {
    primitives.vecUnary(.sigmoid, pre[0..hidden], pre[0..hidden]);
    primitives.vecUnary(.sigmoid, pre[hidden..][0..hidden], pre[hidden..][0..hidden]);
    primitives.vecUnary(.tanh, pre[2 * hidden ..][0..hidden], pre[2 * hidden ..][0..hidden]);
    primitives.vecUnary(.sigmoid, pre[3 * hidden ..][0..hidden], pre[3 * hidden ..][0..hidden]);
    cellUpdate(pre, c, hidden);
    primitives.vecUnary(.tanh, tc, c);
    mulRows(h, pre[3 * hidden ..][0..hidden], tc);
}

fn vecTanhInto(dst: []f32, src: []const f32) void {
    primitives.vecUnary(.tanh, dst, src);
}

/// `c = f·c + i·g`.
inline fn cellUpdate(gates: []const f32, c: []f32, hidden: usize) void {
    const ig = gates[0..hidden];
    const fg = gates[hidden..][0..hidden];
    const gg = gates[2 * hidden ..][0..hidden];
    var j: usize = 0;
    while (j + vector_len <= hidden) : (j += vector_len) {
        const cv: Vf32 = c[j..][0..vector_len].*;
        const fv: Vf32 = fg[j..][0..vector_len].*;
        const iv: Vf32 = ig[j..][0..vector_len].*;
        const gv: Vf32 = gg[j..][0..vector_len].*;
        c[j..][0..vector_len].* = fv * cv + iv * gv;
    }
    while (j < hidden) : (j += 1) c[j] = fg[j] * c[j] + ig[j] * gg[j];
}

inline fn mulRows(dst: []f32, a: []const f32, b: []const f32) void {
    var j: usize = 0;
    while (j + vector_len <= dst.len) : (j += vector_len) {
        const av: Vf32 = a[j..][0..vector_len].*;
        const bv: Vf32 = b[j..][0..vector_len].*;
        dst[j..][0..vector_len].* = av * bv;
    }
    while (j < dst.len) : (j += 1) dst[j] = a[j] * b[j];
}

inline fn axpyRow(acc: []f32, scalar_value: f32, row: []const f32) void {
    const sv: Vf32 = @splat(scalar_value);
    var o: usize = 0;
    while (o + vector_len <= acc.len) : (o += vector_len) {
        const cur: Vf32 = acc[o..][0..vector_len].*;
        const rv: Vf32 = row[o..][0..vector_len].*;
        acc[o..][0..vector_len].* = cur + sv * rv;
    }
    while (o < acc.len) : (o += 1) acc[o] += scalar_value * row[o];
}

inline fn dotRow(a: []const f32, b: []const f32) f32 {
    var accv: Vf32 = @splat(0);
    var o: usize = 0;
    while (o + vector_len <= a.len) : (o += vector_len) {
        accv = @mulAdd(Vf32, a[o..][0..vector_len].*, b[o..][0..vector_len].*, accv);
    }
    var acc: f32 = @reduce(.Add, accv);
    while (o < a.len) : (o += 1) acc += a[o] * b[o];
    return acc;
}

/// The sequence walk shared by both arms: `step` turns a row of
/// pre-activations into the gates and the new `c`, `h` (writing `tanh(c)`
/// into `tc`).
fn forward(comptime step: fn ([]f32, []f32, []f32, usize, []f32) void, out: []f32, gates: []f32, x: []const f32, w: []const f32, b: []const f32, h0: []const f32, c0: []const f32, seq: usize, in_size: usize, hidden: usize) void {
    const width = 4 * hidden;
    var tc_buf: [256]f32 = undefined;
    var tc_heap: []f32 = &.{};
    const tc: []f32 = if (hidden <= tc_buf.len) tc_buf[0..hidden] else blk: {
        // Hidden sizes past the stack buffer are not NAM's; the walk still
        // runs, allocating once.
        tc_heap = std.heap.page_allocator.alloc(f32, hidden) catch unreachable;
        break :blk tc_heap;
    };
    defer if (tc_heap.len != 0) std.heap.page_allocator.free(tc_heap);
    var h_prev: []const f32 = h0;
    var c_prev: []const f32 = c0;
    for (0..seq) |t| {
        const pre = gates[t * width ..][0..width];
        @memcpy(pre, b);
        const x_row = x[t * in_size ..][0..in_size];
        for (0..in_size) |j| axpyRow(pre, x_row[j], w[j * width ..][0..width]);
        for (0..hidden) |j| axpyRow(pre, h_prev[j], w[(in_size + j) * width ..][0..width]);
        const h = out[t * hidden ..][0..hidden];
        const c = out[(seq + t) * hidden ..][0..hidden];
        @memcpy(c, c_prev);
        step(pre, c, h, hidden, tc);
        h_prev = h;
        c_prev = c;
    }
}

/// BPTT over the saved sequence, `tanhInto` the arm's tanh over a slice.
fn backward(
    comptime tanhInto: fn ([]f32, []const f32) void,
    gx: ?*Tensor,
    gw: ?*Tensor,
    gb: ?*Tensor,
    gh0: ?*Tensor,
    gc0: ?*Tensor,
    gy: []const f32,
    x: []const f32,
    w: []const f32,
    gates: []const f32,
    out: []const f32,
    h0: []const f32,
    c0: []const f32,
    seq: usize,
    in_size: usize,
    hidden: usize,
    scratch: []f32,
) void {
    const width = 4 * hidden;
    // scratch: dpre [4H] | dh [H] | dc [H] | tc [H] | dh_prev [H] | dc_prev [H]
    const dpre = scratch[0..width];
    const dh = scratch[width..][0..hidden];
    const dc = scratch[width + hidden ..][0..hidden];
    const tc = scratch[width + 2 * hidden ..][0..hidden];
    const dh_prev = scratch[width + 3 * hidden ..][0..hidden];
    const dc_prev = scratch[width + 4 * hidden ..][0..hidden];
    @memset(dh_prev, 0);
    @memset(dc_prev, 0);
    const gx_data: ?[]f32 = if (gx) |t| t.data() else null;
    const gw_data: ?[]f32 = if (gw) |t| t.data() else null;
    const gb_data: ?[]f32 = if (gb) |t| t.data() else null;
    if (gw_data) |g| @memset(g, 0);
    if (gb_data) |g| @memset(g, 0);
    var t = seq;
    while (t > 0) {
        t -= 1;
        const gy_h = gy[t * hidden ..][0..hidden];
        const gy_c = gy[(seq + t) * hidden ..][0..hidden];
        const c_t = out[(seq + t) * hidden ..][0..hidden];
        const gate = gates[t * width ..][0..width];
        const ig = gate[0..hidden];
        const fg = gate[hidden..][0..hidden];
        const gg = gate[2 * hidden ..][0..hidden];
        const og = gate[3 * hidden ..][0..hidden];
        const c_before: []const f32 = if (t == 0) c0 else out[(seq + t - 1) * hidden ..][0..hidden];
        const h_before: []const f32 = if (t == 0) h0 else out[(t - 1) * hidden ..][0..hidden];
        tanhInto(tc, c_t);
        for (0..hidden) |j| {
            dh[j] = gy_h[j] + dh_prev[j];
            dc[j] = gy_c[j] + dc_prev[j] + dh[j] * og[j] * (1 - tc[j] * tc[j]);
            const d_o = dh[j] * tc[j];
            const d_i = dc[j] * gg[j];
            const d_g = dc[j] * ig[j];
            const d_f = dc[j] * c_before[j];
            dpre[j] = d_i * ig[j] * (1 - ig[j]);
            dpre[hidden + j] = d_f * fg[j] * (1 - fg[j]);
            dpre[2 * hidden + j] = d_g * (1 - gg[j] * gg[j]);
            dpre[3 * hidden + j] = d_o * og[j] * (1 - og[j]);
            dc_prev[j] = dc[j] * fg[j];
        }
        if (gb_data) |g| axpyRow(g, 1.0, dpre);
        const x_row = x[t * in_size ..][0..in_size];
        if (gw_data) |g| {
            for (0..in_size) |j| axpyRow(g[j * width ..][0..width], x_row[j], dpre);
            for (0..hidden) |j| axpyRow(g[(in_size + j) * width ..][0..width], h_before[j], dpre);
        }
        if (gx_data) |g| {
            for (0..in_size) |j| g[t * in_size + j] = dotRow(w[j * width ..][0..width], dpre);
        }
        for (0..hidden) |j| dh_prev[j] = dotRow(w[(in_size + j) * width ..][0..width], dpre);
    }
    if (gh0) |g| @memcpy(g.data(), dh_prev);
    if (gc0) |g| @memcpy(g.data(), dc_prev);
}

pub const scalar = struct {
    pub fn lstmSequenceInto(out: *Tensor, gates: *Tensor, x: *const Tensor, w: *const Tensor, b: *const Tensor, h0: *const Tensor, c0: *const Tensor, seq: usize, in_size: usize, hidden: usize) void {
        forward(scalarStep, out.data(), gates.data(), x.dataConst(), w.dataConst(), b.dataConst(), h0.dataConst(), c0.dataConst(), seq, in_size, hidden);
    }

    pub fn lstmSequenceBackwardInto(
        gx: ?*Tensor,
        gw: ?*Tensor,
        gb: ?*Tensor,
        gh0: ?*Tensor,
        gc0: ?*Tensor,
        gy: *const Tensor,
        x: *const Tensor,
        w: *const Tensor,
        gates: *const Tensor,
        out: *const Tensor,
        h0: *const Tensor,
        c0: *const Tensor,
        seq: usize,
        in_size: usize,
        hidden: usize,
        scratch: []f32,
    ) void {
        backward(scalarTanhInto, gx, gw, gb, gh0, gc0, gy.dataConst(), x.dataConst(), w.dataConst(), gates.dataConst(), out.dataConst(), h0.dataConst(), c0.dataConst(), seq, in_size, hidden, scratch);
    }

    fn scalarStep(pre: []f32, c: []f32, h: []f32, hidden: usize, tc: []f32) void {
        for (pre[0..hidden]) |*v| v.* = ops.unaryScalar(.sigmoid, v.*);
        for (pre[hidden..][0..hidden]) |*v| v.* = ops.unaryScalar(.sigmoid, v.*);
        for (pre[2 * hidden ..][0..hidden]) |*v| v.* = ops.unaryScalar(.tanh, v.*);
        for (pre[3 * hidden ..][0..hidden]) |*v| v.* = ops.unaryScalar(.sigmoid, v.*);
        for (0..hidden) |j| {
            c[j] = pre[hidden + j] * c[j] + pre[j] * pre[2 * hidden + j];
            tc[j] = ops.unaryScalar(.tanh, c[j]);
            h[j] = pre[3 * hidden + j] * tc[j];
        }
    }

    fn scalarTanhInto(dst: []f32, src: []const f32) void {
        for (dst, src) |*d, s| d.* = ops.unaryScalar(.tanh, s);
    }
};

test "lstm sequence: the vector and scalar arms agree with a two-step hand computation" {
    const hidden = 2;
    const in_size = 1;
    const width = 4 * hidden;
    // w [in + H, 4H], b, h0, c0 chosen small; two steps of x.
    const w_data = [_]f32{
        0.1, -0.2, 0.3,  0.05, -0.1, 0.2, 0.15, -0.05,
        0.2, 0.1,  -0.3, 0.25, 0.1,  0.0, -0.2, 0.3,
        0.0, 0.3,  0.1,  -0.1, 0.2,  0.1, 0.05, 0.1,
    };
    const b_data = [_]f32{ 0.01, -0.02, 0.03, 0.0, 0.05, -0.04, 0.02, 0.01 };
    const h0_data = [_]f32{ 0.1, -0.1 };
    const c0_data = [_]f32{ 0.2, 0.05 };
    const x_data = [_]f32{ 0.5, -0.3 };
    var out_data: [2 * 2 * hidden]f32 = undefined;
    var gates_data: [2 * width]f32 = undefined;
    forward(vecStep, &out_data, &gates_data, &x_data, &w_data, &b_data, &h0_data, &c0_data, 2, in_size, hidden);
    var out_scalar: [2 * 2 * hidden]f32 = undefined;
    var gates_scalar: [2 * width]f32 = undefined;
    forward(scalar.scalarStep, &out_scalar, &gates_scalar, &x_data, &w_data, &b_data, &h0_data, &c0_data, 2, in_size, hidden);
    // Hand computation in f64.
    var h = [_]f64{ 0.1, -0.1 };
    var c = [_]f64{ 0.2, 0.05 };
    for (0..2) |t| {
        var pre: [width]f64 = undefined;
        for (0..width) |k| {
            var acc: f64 = b_data[k];
            acc += @as(f64, x_data[t]) * w_data[k];
            for (0..hidden) |j| acc += h[j] * w_data[(in_size + j) * width + k];
            pre[k] = acc;
        }
        for (0..hidden) |j| {
            const i_g = 1.0 / (1.0 + @exp(-pre[j]));
            const f_g = 1.0 / (1.0 + @exp(-pre[hidden + j]));
            const g_g = std.math.tanh(pre[2 * hidden + j]);
            const o_g = 1.0 / (1.0 + @exp(-pre[3 * hidden + j]));
            c[j] = f_g * c[j] + i_g * g_g;
            h[j] = o_g * std.math.tanh(c[j]);
            try std.testing.expectApproxEqAbs(@as(f32, @floatCast(h[j])), out_data[t * hidden + j], 1e-6);
            try std.testing.expectApproxEqAbs(@as(f32, @floatCast(c[j])), out_data[(2 + t) * hidden + j], 1e-6);
            try std.testing.expectApproxEqAbs(@as(f32, @floatCast(h[j])), out_scalar[t * hidden + j], 1e-6);
        }
    }
}
