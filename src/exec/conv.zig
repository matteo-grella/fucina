//! Causal 1-D convolutions (depthwise / general / grouped, fwd + bwd) and the
//! channel-last 2-D convolution.
//!
//! Domain module: every op receives an explicit `*Runtime`. Dispatches to the
//! allocation-free backend conv kernels; the 1×1 conv2d fast path calls into
//! the matmul + elementwise domains explicitly (leaf-ward imports).

const std = @import("std");
const backend_mod = @import("../backend.zig");
const parallel = @import("../parallel.zig");
const tensor = @import("../tensor.zig");

const exec_shape = @import("shape.zig");
const exec_matmul = @import("matmul.zig");
const exec_elementwise = @import("elementwise.zig");
const Runtime = @import("runtime.zig").Runtime;

const Tensor = tensor.Tensor;
const PreparedTensor = Runtime.PreparedTensor;

const validateCausalDepthwiseState = exec_shape.validateCausalDepthwiseState;
const validateCausalConvState = exec_shape.validateCausalConvState;
const validateGroupedCausalConv = exec_shape.validateGroupedCausalConv;
const causalConvWork = exec_shape.causalConvWork;

pub fn causalDepthwiseConv1dAxisRank(
    rt: *Runtime,
    comptime rank: usize,
    input: *const Tensor,
    kernel: *const Tensor,
    comptime time_axis: usize,
    comptime channel_axis: usize,
    state: ?[]const f32,
) !Tensor {
    comptime {
        if (rank != 2) @compileError("causalDepthwiseConv1d currently requires rank 2");
        if (time_axis != 0 or channel_axis != 1) {
            @compileError("causalDepthwiseConv1d requires storage order [time, channel]");
        }
    }

    const source = try input.rankView(rank);
    const kernel_view = try kernel.rankView(2);
    const seq = source.shape[time_axis];
    const channels = source.shape[channel_axis];
    const taps = kernel_view.shape[1];
    if (kernel_view.shape[0] != channels) return tensor.TensorError.ShapeMismatch;
    try validateCausalDepthwiseState(state, channels, taps);

    var ii = try rt.prepareContiguous(input);
    defer ii.deinit();
    var kk = try rt.prepareContiguous(kernel);
    defer kk.deinit();

    var out = try rt.emptyRank(rank, source.shape);
    errdefer out.deinit();
    rt.enableNativeVectorPoolForWork(parallel.saturatedMul3(seq, channels, taps), parallel.vector_elementwise_len_threshold);
    rt.backend.causalDepthwiseConv1dInto(&out, ii.tensor(), kk.tensor(), state, seq, channels, taps);
    return out;
}

pub fn causalDepthwiseConv1dBackwardInputAxisRank(
    rt: *Runtime,
    comptime rank: usize,
    gy: *const Tensor,
    kernel: *const Tensor,
    comptime time_axis: usize,
    comptime channel_axis: usize,
) !Tensor {
    comptime {
        if (rank != 2) @compileError("causalDepthwiseConv1dBackwardInput currently requires rank 2");
        if (time_axis != 0 or channel_axis != 1) {
            @compileError("causalDepthwiseConv1dBackwardInput requires storage order [time, channel]");
        }
    }

    const grad_view = try gy.rankView(rank);
    const kernel_view = try kernel.rankView(2);
    const seq = grad_view.shape[time_axis];
    const channels = grad_view.shape[channel_axis];
    const taps = kernel_view.shape[1];
    if (kernel_view.shape[0] != channels) return tensor.TensorError.ShapeMismatch;

    var gg = try rt.prepareContiguous(gy);
    defer gg.deinit();
    var kk = try rt.prepareContiguous(kernel);
    defer kk.deinit();

    var out = try rt.emptyRank(rank, grad_view.shape);
    errdefer out.deinit();
    rt.enableNativeVectorPoolForWork(parallel.saturatedMul3(seq, channels, taps), parallel.vector_elementwise_len_threshold);
    rt.backend.causalDepthwiseConv1dBackwardInputInto(&out, gg.tensor(), kk.tensor(), seq, channels, taps);
    return out;
}

pub fn causalDepthwiseConv1dBackwardKernelAxisRank(
    rt: *Runtime,
    comptime rank: usize,
    input: *const Tensor,
    gy: *const Tensor,
    comptime time_axis: usize,
    comptime channel_axis: usize,
    taps: usize,
    state: ?[]const f32,
) !Tensor {
    comptime {
        if (rank != 2) @compileError("causalDepthwiseConv1dBackwardKernel currently requires rank 2");
        if (time_axis != 0 or channel_axis != 1) {
            @compileError("causalDepthwiseConv1dBackwardKernel requires storage order [time, channel]");
        }
    }

    const source = try input.rankView(rank);
    const grad_view = try gy.rankView(rank);
    if (!std.mem.eql(usize, source.shape[0..], grad_view.shape[0..])) return tensor.TensorError.ShapeMismatch;
    const seq = source.shape[time_axis];
    const channels = source.shape[channel_axis];
    if (taps == 0) return tensor.TensorError.InvalidShape;
    try validateCausalDepthwiseState(state, channels, taps);

    var ii = try rt.prepareContiguous(input);
    defer ii.deinit();
    var gg = try rt.prepareContiguous(gy);
    defer gg.deinit();

    var out = try rt.emptyRank(2, .{ channels, taps });
    errdefer out.deinit();
    rt.enableNativeVectorPoolForWork(parallel.saturatedMul3(seq, channels, taps), parallel.vector_elementwise_len_threshold);
    rt.backend.causalDepthwiseConv1dBackwardKernelInto(&out, ii.tensor(), gg.tensor(), state, seq, channels, taps);
    return out;
}

/// General causal 1-D convolution: input `[time, in]`, weight
/// `[tap, in, out]` (tap `taps-1` = the newest sample), output
/// `[time, out]`. `state`, when given, is the `dilation*(taps-1)`
/// input rows preceding the chunk, oldest first; absent ⇒ zeros.
pub fn causalConv1dAxisRank(
    rt: *Runtime,
    comptime rank: usize,
    input: *const Tensor,
    weight: *const Tensor,
    comptime time_axis: usize,
    comptime channel_axis: usize,
    dilation: usize,
    state: ?[]const f32,
) !Tensor {
    comptime {
        if (rank != 2) @compileError("causalConv1d currently requires rank 2");
        if (time_axis != 0 or channel_axis != 1) {
            @compileError("causalConv1d requires storage order [time, in]");
        }
    }

    const source = try input.rankView(rank);
    const weight_view = try weight.rankView(3);
    const seq = source.shape[time_axis];
    const in_channels = source.shape[channel_axis];
    const taps = weight_view.shape[0];
    const out_channels = weight_view.shape[2];
    if (weight_view.shape[1] != in_channels) return tensor.TensorError.ShapeMismatch;
    try validateCausalConvState(state, in_channels, taps, dilation);

    var ii = try rt.prepareContiguous(input);
    defer ii.deinit();
    var ww = try rt.prepareContiguous(weight);
    defer ww.deinit();

    var out = try rt.emptyRank(rank, .{ seq, out_channels });
    errdefer out.deinit();
    rt.enableNativeVectorPoolForWork(causalConvWork(seq, in_channels, out_channels, taps), parallel.vector_elementwise_len_threshold);
    rt.backend.causalConv1dInto(&out, ii.tensor(), ww.tensor(), state, seq, in_channels, out_channels, taps, dilation);
    return out;
}

/// 2-D convolution forward, channel-last rank-3:
///   input  [H, W, Cin]
///   weight [Cout, KH, KW, Cin/groups]   (output-major; `w[((oc*KH+kh)*KW+kw)*Cinpg + ic]`)
///   bias   [Cout] or null
///   -> out [OH, OW, Cout]  with OH = (H + 2*pad_h - KH)/stride_h + 1 (likewise OW).
/// Explicit zero padding (`pad`); "same"/"valid" are the caller's choice of pad.
/// `groups` == Cin gives a depthwise conv. Validates, then dispatches to the
/// allocation-free backend kernel (native vector or scalar reference).
pub fn conv2d(
    rt: *Runtime,
    input: *const Tensor,
    weight: *const Tensor,
    bias: ?*const Tensor,
    stride: [2]usize,
    pad: [2]usize,
    groups: usize,
) !Tensor {
    return conv2dExt(rt, input, weight, bias, stride, pad, groups, false);
}

/// conv2d with an optional fused relu epilogue: on the Winograd route the
/// relu folds into the output transform (zero extra passes); on the other
/// routes it runs in place on the fresh output. Values are identical to
/// conv2d followed by relu (same single max(0,·) on the same numbers).
pub fn conv2dExt(
    rt: *Runtime,
    input: *const Tensor,
    weight: *const Tensor,
    bias: ?*const Tensor,
    stride: [2]usize,
    pad: [2]usize,
    groups: usize,
    fused_relu: bool,
) !Tensor {
    return conv2dPreparedExt(rt, input, weight, null, bias, stride, pad, groups, fused_relu);
}

/// conv2dExt with optional load-time prepared Winograd weight planes (see
/// `prepareConv2dWeights`). On the Winograd route, a matching prepared set
/// skips the per-call weight transform and GEMMs straight against the stored
/// planes — bitwise-identical values (same buffers into the same kernels).
/// Every other route (1×1, stride > 1, grouped, im2col) ignores `prepared`;
/// `.empty` is always inert.
pub fn conv2dPreparedExt(
    rt: *Runtime,
    input: *const Tensor,
    weight: *const Tensor,
    prepared: ?*const PreparedConvWeights,
    bias: ?*const Tensor,
    stride: [2]usize,
    pad: [2]usize,
    groups: usize,
    fused_relu: bool,
) !Tensor {
    const in_view = try input.rankView(3);
    const w_view = try weight.rankView(4);
    const h = in_view.shape[0];
    const wd = in_view.shape[1];
    const cin = in_view.shape[2];
    const cout = w_view.shape[0];
    const kh = w_view.shape[1];
    const kw = w_view.shape[2];
    const cin_pg = w_view.shape[3];
    if (groups == 0 or cin % groups != 0 or cout % groups != 0) return tensor.TensorError.ShapeMismatch;
    if (cin_pg != cin / groups) return tensor.TensorError.ShapeMismatch;
    if (stride[0] == 0 or stride[1] == 0) return tensor.TensorError.ShapeMismatch;
    if (h + 2 * pad[0] < kh or wd + 2 * pad[1] < kw) return tensor.TensorError.ShapeMismatch;
    const oh = (h + 2 * pad[0] - kh) / stride[0] + 1;
    const ow = (wd + 2 * pad[1] - kw) / stride[1] + 1;

    var ii = try rt.prepareContiguous(input);
    defer ii.deinit();
    var ww = try rt.prepareContiguous(weight);
    defer ww.deinit();

    var bb: ?PreparedTensor = null;
    defer if (bb) |*x| x.deinit();
    var bias_slice: ?[]const f32 = null;
    if (bias) |b| {
        const bv = try b.rankView(1);
        if (bv.shape[0] != cout) return tensor.TensorError.ShapeMismatch;
        bb = try rt.prepareContiguous(b);
        bias_slice = bb.?.tensor().dataConst();
    }

    if (kh == 1 and kw == 1 and stride[0] == 1 and stride[1] == 1 and pad[0] == 0 and pad[1] == 0 and groups == 1) {
        var input_2d = try ii.tensor().viewWithStrides(&.{ h * wd, cin }, &.{ cin, 1 });
        defer input_2d.deinit();
        var weight_2d = try ww.tensor().viewWithStrides(&.{ cout, cin }, &.{ cin, 1 });
        defer weight_2d.deinit();

        var out_2d = try exec_matmul.matmul2DDispatch(rt, .trans_b, &input_2d, &weight_2d);
        errdefer out_2d.deinit();
        if (bias_slice) |b| try exec_elementwise.addAxisVectorInPlaceRank(rt, 2, &out_2d, b, 1);
        if (fused_relu) reluInPlace(rt, &out_2d);

        var out_3d = try out_2d.viewWithStrides(&.{ oh, ow, cout }, &.{ ow * cout, cout, 1 });
        errdefer out_3d.deinit();
        out_2d.deinit();
        return out_3d;
    }

    // Winograd route for the dominant conv shape (3×3, stride 1, pad ≤ 1,
    // groups == 1): F(4×4,3×3) on large maps (min(oh,ow) ≥ 14, mirroring the
    // reference's gate — 4× fewer MACs), F(2×2,3×3) otherwise (2.25×); both
    // drop im2col's 9× col traffic. The transforms reassociate the 3×3
    // reduction (F2 ~1e-6-, F4 ~1e-5-relative drift vs the direct kernel —
    // tolerance argument in vector/winograd.zig). FUCINA_NO_WINOGRAD=1
    // reverts to im2col; FUCINA_NO_WINOGRAD_F4=1 pins large maps to F2.
    if (groups == 1 and kh == 3 and kw == 3 and stride[0] == 1 and stride[1] == 1 and
        pad[0] <= 1 and pad[1] <= 1 and cin >= 4 and oh >= 2 and ow >= 2 and winogradEnabled())
    {
        // A non-empty prepared set must belong to THIS weight's shape; a
        // mismatch is a caller wiring bug, not a fallback case. (`.empty`
        // carries no planes and stays inert.)
        if (prepared) |p| {
            if ((p.f2 != null or p.f4 != null) and (p.cout != cout or p.cin != cin)) {
                return tensor.TensorError.ShapeMismatch;
            }
        }
        if (@min(oh, ow) >= winogradF4MinSpatial() and cin <= winogradF4MaxCin() and winogradF4Enabled()) {
            const f4_planes: ?*const [36]Tensor = if (prepared) |p| (if (p.f4) |*planes| planes else null) else null;
            return winogradConv(rt, .f4, ii.tensor(), ww.tensor(), f4_planes, bias_slice, fused_relu, .{
                .h = h,
                .w = wd,
                .cin = cin,
                .oh = oh,
                .ow = ow,
                .cout = cout,
                .pad_h = pad[0],
                .pad_w = pad[1],
                .tiles_y = (oh + 3) / 4,
                .tiles_x = (ow + 3) / 4,
            });
        }
        const f2_planes: ?*const [16]Tensor = if (prepared) |p| (if (p.f2) |*planes| planes else null) else null;
        return winogradConv(rt, .f2, ii.tensor(), ww.tensor(), f2_planes, bias_slice, fused_relu, .{
            .h = h,
            .w = wd,
            .cin = cin,
            .oh = oh,
            .ow = ow,
            .cout = cout,
            .pad_h = pad[0],
            .pad_w = pad[1],
            .tiles_y = (oh + 1) / 2,
            .tiles_x = (ow + 1) / 2,
        });
    }

    // Non-1×1 dense conv (groups == 1): im2col + a single GEMM. The im2col gather
    // is pure data movement (zero-fill + memcpy per tap), threaded over output
    // rows by the backend kernel (bit-identical to serial); the GEMM goes through
    // the same BLAS/blocked-packed matmul as linears — far faster than the direct
    // gather on the large 3×3 stacks (SCRFD/ArcFace). Depthwise/grouped keep the
    // direct kernel.
    if (groups == 1) {
        const npos = oh * ow;
        const ksz = kh * kw * cin;
        var col = try rt.emptyRank(2, .{ npos, ksz });
        errdefer col.deinit();
        rt.enableNativeVectorPoolForWork(npos * ksz, parallel.vector_elementwise_len_threshold);
        rt.backend.im2colInto(&col, ii.tensor(), .{
            .h = h,
            .w = wd,
            .cin = cin,
            .oh = oh,
            .ow = ow,
            .cout = cout,
            .kh = kh,
            .kw = kw,
            .stride_h = stride[0],
            .stride_w = stride[1],
            .pad_h = pad[0],
            .pad_w = pad[1],
            .groups = 1,
        });
        var weight_2d = try ww.tensor().viewWithStrides(&.{ cout, ksz }, &.{ ksz, 1 });
        defer weight_2d.deinit();
        var out_2d = try exec_matmul.matmul2DDispatch(rt, .trans_b, &col, &weight_2d);
        col.deinit();
        errdefer out_2d.deinit();
        if (bias_slice) |b| try exec_elementwise.addAxisVectorInPlaceRank(rt, 2, &out_2d, b, 1);
        if (fused_relu) reluInPlace(rt, &out_2d);
        var out_3d = try out_2d.viewWithStrides(&.{ oh, ow, cout }, &.{ ow * cout, cout, 1 });
        errdefer out_3d.deinit();
        out_2d.deinit();
        return out_3d;
    }

    var out = try rt.emptyRank(3, .{ oh, ow, cout });
    errdefer out.deinit();
    // Enable the worker pool so conv2d threads over output rows when the conv
    // is large (e.g. an ASR subsampling stem). Bit-identical to the serial path.
    rt.enableNativeVectorPoolForWork(oh * ow * cout * kh * kw * cin_pg, parallel.vector_elementwise_len_threshold);
    rt.backend.conv2dInto(&out, ii.tensor(), ww.tensor(), bias_slice, .{
        .h = h,
        .w = wd,
        .cin = cin,
        .oh = oh,
        .ow = ow,
        .cout = cout,
        .kh = kh,
        .kw = kw,
        .stride_h = stride[0],
        .stride_w = stride[1],
        .pad_h = pad[0],
        .pad_w = pad[1],
        .groups = groups,
    });
    if (fused_relu) reluInPlace(rt, &out);
    return out;
}

fn reluInPlace(rt: *Runtime, t: *Tensor) void {
    rt.enableNativeVectorPoolForWork(t.len(), parallel.vector_elementwise_len_threshold);
    rt.backend.unaryContiguousIntoUnchecked(.relu, t, t, t.len());
}

fn conv2dDimsFor(h: usize, w: usize, cin: usize, oh: usize, ow: usize, cout: usize, kh: usize, kw: usize, stride: [2]usize, pad: [2]usize, groups: usize) @import("../backend/vector/conv.zig").Conv2dDims {
    return .{ .h = h, .w = w, .cin = cin, .oh = oh, .ow = ow, .cout = cout, .kh = kh, .kw = kw, .stride_h = stride[0], .stride_w = stride[1], .pad_h = pad[0], .pad_w = pad[1], .groups = groups };
}

/// Winograd route gate. Default follows the GEMM provider: ON for
/// `-Dblas=none` builds (the pure-Zig GEMM gains ~1.7–1.9× from the 2.25×
/// MAC cut and the dropped col-matrix traffic — measured i9-13950HX), OFF
/// when a platform BLAS backs the matmul (Accelerate's AMX makes one big
/// im2col GEMM faster than 16 tile-element GEMMs — measured M1 Max).
/// Runtime overrides: FUCINA_WINOGRAD=1 forces on, FUCINA_NO_WINOGRAD=1
/// forces off (the A/B and emergency-revert switches). Read once, cached.
const winograd_default_on = !backend_mod.native_uses_blas;
var winograd_state = std.atomic.Value(u8).init(0); // 0 = unread, 1 = enabled, 2 = disabled
fn winogradEnabled() bool {
    const s = winograd_state.load(.acquire);
    if (s != 0) return s == 1;
    const on = if (parallel.envPositiveUsize("FUCINA_NO_WINOGRAD") != null)
        false
    else if (parallel.envPositiveUsize("FUCINA_WINOGRAD") != null)
        true
    else
        winograd_default_on;
    winograd_state.store(if (on) 1 else 2, .release);
    return on;
}

/// FUCINA_NO_WINOGRAD_F4=1 pins Winograd-routed large maps to F2 (the F4
/// tier's A/B switch). Read once, cached.
var winograd_f4_state = std.atomic.Value(u8).init(0);
fn winogradF4Enabled() bool {
    const s = winograd_f4_state.load(.acquire);
    if (s != 0) return s == 1;
    const disabled = parallel.envPositiveUsize("FUCINA_NO_WINOGRAD_F4") != null;
    winograd_f4_state.store(if (disabled) 2 else 1, .release);
    return !disabled;
}

// F4 shape gates, bench-tuned (i9-13950HX sweep): F4 pays where the tile
// count keeps its 36 GEMMs well-shaped and the transform cost is amortized
// (SCRFD-class: big spatial, shallow channels); deep-channel maps
// (ArcFace-class) run faster on F2 — the same split the reference ships
// (F4 scope = detector only). FUCINA_WINOGRAD_F4_MIN /
// FUCINA_WINOGRAD_F4_MAXCIN override at runtime. Read once, cached.
var winograd_f4_min_spatial = std.atomic.Value(usize).init(0);
fn winogradF4MinSpatial() usize {
    const v = winograd_f4_min_spatial.load(.acquire);
    if (v != 0) return v;
    const val = parallel.envPositiveUsize("FUCINA_WINOGRAD_F4_MIN") orelse 14;
    winograd_f4_min_spatial.store(val, .release);
    return val;
}
var winograd_f4_max_cin = std.atomic.Value(usize).init(0);
fn winogradF4MaxCin() usize {
    const v = winograd_f4_max_cin.load(.acquire);
    if (v != 0) return v;
    // 56 keeps SCRFD-class shallow-channel maps on F4 and pushes ArcFace-class
    // deep-channel stacks (cin >= 64) to F2 — the measured crossover (F4 on
    // the 112²×64 stage costs ~30% on the recognizer; F4 on the 28/56-channel
    // detector maps saves ~22% on detect).
    const val = parallel.envPositiveUsize("FUCINA_WINOGRAD_F4_MAXCIN") orelse 56;
    winograd_f4_max_cin.store(val, .release);
    return val;
}

/// Test hook: pin the Winograd route on/off, or `null` to restore the
/// env/default gate (re-read on next use).
pub fn setWinogradForTest(state: ?bool) void {
    winograd_state.store(if (state) |on| (if (on) @as(u8, 1) else 2) else 0, .release);
}

/// conv2d backward GEMM-route gate: the groups == 1 backward entries
/// decompose into matmul dispatch + im2col/col2im (the forward's adjoint),
/// which is both GEMM-fast and pool-parallel. FUCINA_NO_CONV_BWD_GEMM=1 pins
/// both entries to the direct gather kernels (the A/B and emergency-revert
/// switch). Read once, cached.
var conv_bwd_gemm_state = std.atomic.Value(u8).init(0); // 0 = unread, 1 = enabled, 2 = disabled
fn convBwdGemmEnabled() bool {
    const s = conv_bwd_gemm_state.load(.acquire);
    if (s != 0) return s == 1;
    const on = parallel.envPositiveUsize("FUCINA_NO_CONV_BWD_GEMM") == null;
    conv_bwd_gemm_state.store(if (on) 1 else 2, .release);
    return on;
}

/// Test hook: pin the conv2d backward GEMM route on/off, or `null` to restore
/// the env/default gate (re-read on next use).
pub fn setConvBwdGemmForTest(state: ?bool) void {
    conv_bwd_gemm_state.store(if (state) |on| (if (on) @as(u8, 1) else 2) else 0, .release);
}

const WinoKind = enum { f2, f4 };

fn winoPlanes(comptime kind: WinoKind) usize {
    return if (kind == .f2) 16 else 36;
}

/// Load-time prepared Winograd conv weights: the per-plane rank-2
/// `[cout, cin]` weight-transform tensors `winogradConv` otherwise rebuilds
/// on every call (F2 = 16 planes, F4 = 36). Produced by
/// `prepareConv2dWeights`, consumed by `conv2dPreparedExt`. `.empty` (no
/// planes) is valid and inert on every route. Invariant: `f4 != null`
/// implies `f2 != null` — F4 is selected per call by input geometry, and F2
/// is the tier every Winograd-routed call can fall back to.
pub const PreparedConvWeights = struct {
    cout: usize = 0,
    cin: usize = 0,
    f2: ?[16]Tensor = null,
    f4: ?[36]Tensor = null,

    pub const empty: PreparedConvWeights = .{};

    pub fn deinit(self: *PreparedConvWeights) void {
        if (self.f2) |*planes| for (planes) |*t| t.deinit();
        if (self.f4) |*planes| for (planes) |*t| t.deinit();
        self.* = .{};
    }
};

/// Build the Winograd weight-transform planes for `weight`
/// (`[cout, 3, 3, cin]`) once, at load time. Returns `.empty` — inert on
/// every route — unless the weight is Winograd-shaped (3×3, cin ≥ 4) and the
/// route is enabled. F2 planes are always built for an eligible weight; F4
/// planes only when the F4 tier's input-independent gates pass (tier enabled,
/// cin within the max-cin gate). The spatial gate (`min(oh,ow)` ≥ threshold)
/// depends on the input, so it stays call-time — hence the F2 fallback set.
pub fn prepareConv2dWeights(rt: *Runtime, weight: *const Tensor) !PreparedConvWeights {
    const w_view = try weight.rankView(4);
    const cout = w_view.shape[0];
    const kh = w_view.shape[1];
    const kw = w_view.shape[2];
    const cin = w_view.shape[3];
    if (!(kh == 3 and kw == 3 and cin >= 4 and winogradEnabled())) return .empty;

    var ww = try rt.prepareContiguous(weight);
    defer ww.deinit();

    var out: PreparedConvWeights = .{ .cout = cout, .cin = cin };
    errdefer out.deinit();
    out.f2 = try winogradWeightPlanes(rt, .f2, ww.tensor(), cout, cin);
    if (winogradF4Enabled() and cin <= winogradF4MaxCin()) {
        out.f4 = try winogradWeightPlanes(rt, .f4, ww.tensor(), cout, cin);
    }
    return out;
}

/// One tier's weight-transform planes (`[cout, cin]` each) — the load-time
/// half of `winogradConv`'s transform stage, same allocation pattern and
/// backend kernel as the per-call path (bitwise-identical plane contents).
fn winogradWeightPlanes(rt: *Runtime, comptime kind: WinoKind, weight: *const Tensor, cout: usize, cin: usize) ![winoPlanes(kind)]Tensor {
    const planes = comptime winoPlanes(kind);
    var u_t: [planes]Tensor = undefined;
    var u_n: usize = 0;
    errdefer for (u_t[0..u_n]) |*t| t.deinit();
    for (0..planes) |e| {
        u_t[e] = try rt.emptyRank(2, .{ cout, cin });
        u_n = e + 1;
    }
    var u_s: [planes][]f32 = undefined;
    for (0..planes) |e| u_s[e] = u_t[e].data();
    rt.enableNativeVectorPoolForWork(planes * cout * cin, parallel.vector_elementwise_len_threshold);
    switch (kind) {
        .f2 => rt.backend.winogradF2WeightTransformInto(&u_s, weight.dataConst(), cout, cin),
        .f4 => rt.backend.winogradF4WeightTransformInto(&u_s, weight.dataConst(), cout, cin),
    }
    return u_t;
}

/// Winograd forward (F2: 16 planes / F4: 36 planes): weight/input transforms
/// → per-plane tile GEMMs through the ordinary matmul dispatch (BLAS /
/// blocked-packed / row-parallel vector) → output transform with the bias
/// folded in (one add per element, like the standard post-GEMM bias pass).
/// Transform math and the tolerance argument live in
/// `backend/vector/winograd.zig`. `u_prepared`, when set, supplies the
/// weight-transform planes (see `prepareConv2dWeights`): the per-call u_t
/// allocation + weight transform are skipped and the GEMMs read the prepared
/// planes directly.
fn winogradConv(rt: *Runtime, comptime kind: WinoKind, input: *const Tensor, weight: *const Tensor, u_prepared: ?*const [winoPlanes(kind)]Tensor, bias: ?[]const f32, fused_relu: bool, d: backend_mod.WinogradF2Dims) !Tensor {
    const planes = comptime winoPlanes(kind);
    const tiles = d.tiles_y * d.tiles_x;

    var u_t: [planes]Tensor = undefined;
    var u_n: usize = 0;
    errdefer for (u_t[0..u_n]) |*t| t.deinit();
    if (u_prepared == null) {
        for (0..planes) |e| {
            u_t[e] = try rt.emptyRank(2, .{ d.cout, d.cin });
            u_n = e + 1;
        }
    }
    var v_t: [planes]Tensor = undefined;
    var v_n: usize = 0;
    errdefer for (v_t[0..v_n]) |*t| t.deinit();
    for (0..planes) |e| {
        v_t[e] = try rt.emptyRank(2, .{ tiles, d.cin });
        v_n = e + 1;
    }

    var v_s: [planes][]f32 = undefined;
    for (0..planes) |e| v_s[e] = v_t[e].data();
    // With prepared weights the transform stage only touches the input
    // planes, so the weight (d.cout) term drops out of the pool-work gate.
    const transform_work = if (u_prepared == null) planes * (tiles + d.cout) * d.cin else planes * tiles * d.cin;
    rt.enableNativeVectorPoolForWork(transform_work, parallel.vector_elementwise_len_threshold);
    if (u_prepared == null) {
        var u_s: [planes][]f32 = undefined;
        for (0..planes) |e| u_s[e] = u_t[e].data();
        switch (kind) {
            .f2 => rt.backend.winogradF2WeightTransformInto(&u_s, weight.dataConst(), d.cout, d.cin),
            .f4 => rt.backend.winogradF4WeightTransformInto(&u_s, weight.dataConst(), d.cout, d.cin),
        }
    }
    switch (kind) {
        .f2 => rt.backend.winogradF2InputTransformInto(&v_s, input.dataConst(), d),
        .f4 => rt.backend.winogradF4InputTransformInto(&v_s, input.dataConst(), d),
    }

    const u_src: *const [planes]Tensor = u_prepared orelse &u_t;
    var m_t: [planes]Tensor = undefined;
    var m_n: usize = 0;
    errdefer for (m_t[0..m_n]) |*t| t.deinit();
    for (0..planes) |e| {
        m_t[e] = try exec_matmul.matmul2DDispatch(rt, .trans_b, &v_t[e], &u_src[e]);
        m_n = e + 1;
    }
    for (u_t[0..u_n]) |*t| t.deinit();
    u_n = 0;
    for (v_t[0..v_n]) |*t| t.deinit();
    v_n = 0;

    var out = try rt.emptyRank(3, .{ d.oh, d.ow, d.cout });
    errdefer out.deinit();
    var m_s: [planes][]const f32 = undefined;
    for (0..planes) |e| m_s[e] = m_t[e].dataConst();
    rt.enableNativeVectorPoolForWork(planes * tiles * d.cout, parallel.vector_elementwise_len_threshold);
    switch (kind) {
        .f2 => rt.backend.winogradF2OutputTransformInto(out.data(), &m_s, bias, fused_relu, d),
        .f4 => rt.backend.winogradF4OutputTransformInto(out.data(), &m_s, bias, fused_relu, d),
    }
    for (m_t[0..m_n]) |*t| t.deinit();
    m_n = 0;
    return out;
}

/// VJP of conv2d wrt input. `gy` is `[oh,ow,cout]`, `weight` is the forward
/// `[cout,kh,kw,cin/groups]`; result is `[in_h,in_w,cin]`.
pub fn conv2dBackwardInput(rt: *Runtime, gy: *const Tensor, weight: *const Tensor, in_h: usize, in_w: usize, stride: [2]usize, pad: [2]usize, groups: usize) !Tensor {
    const gy_view = try gy.rankView(3);
    const w_view = try weight.rankView(4);
    const oh = gy_view.shape[0];
    const ow = gy_view.shape[1];
    const cout = gy_view.shape[2];
    const kh = w_view.shape[1];
    const kw = w_view.shape[2];
    const cin_pg = w_view.shape[3];
    if (w_view.shape[0] != cout) return tensor.TensorError.ShapeMismatch;
    if (groups == 0 or cout % groups != 0 or stride[0] == 0 or stride[1] == 0) return tensor.TensorError.InvalidShape;
    const cin = cin_pg * groups;
    if (in_h + 2 * pad[0] < kh or in_w + 2 * pad[1] < kw) return tensor.TensorError.InvalidShape;
    if ((in_h + 2 * pad[0] - kh) / stride[0] + 1 != oh or (in_w + 2 * pad[1] - kw) / stride[1] + 1 != ow) return tensor.TensorError.ShapeMismatch;

    var gg = try rt.prepareContiguous(gy);
    defer gg.deinit();
    var ww = try rt.prepareContiguous(weight);
    defer ww.deinit();

    if (groups == 1 and convBwdGemmEnabled()) {
        var gy_2d = try gg.tensor().viewWithStrides(&.{ oh * ow, cout }, &.{ cout, 1 });
        defer gy_2d.deinit();
        if (kh == 1 and kw == 1 and stride[0] == 1 and stride[1] == 1 and pad[0] == 0 and pad[1] == 0) {
            // 1x1 s1 p0: the forward is one plain GEMM (out = in · wᵀ), so
            // the input VJP is one GEMM too: gx = gy · w.
            var w_2d = try ww.tensor().viewWithStrides(&.{ cout, cin }, &.{ cin, 1 });
            defer w_2d.deinit();
            var gx_2d = try exec_matmul.matmul2DDispatch(rt, .plain, &gy_2d, &w_2d);
            errdefer gx_2d.deinit();
            var gx_3d = try gx_2d.viewWithStrides(&.{ in_h, in_w, cin }, &.{ in_w * cin, cin, 1 });
            errdefer gx_3d.deinit();
            gx_2d.deinit();
            return gx_3d;
        }
        // Dense general shape: gcol = gy · w (one GEMM over the [cout, ksz]
        // weight view), then the col2im gather — the exact adjoint of the
        // forward's im2col + GEMM route.
        const ksz = kh * kw * cin;
        var w_2d = try ww.tensor().viewWithStrides(&.{ cout, ksz }, &.{ ksz, 1 });
        defer w_2d.deinit();
        var gcol = try exec_matmul.matmul2DDispatch(rt, .plain, &gy_2d, &w_2d);
        defer gcol.deinit();
        var out = try rt.emptyRank(3, .{ in_h, in_w, cin });
        errdefer out.deinit();
        rt.enableNativeVectorPoolForWork(parallel.saturatedMul3(in_h * in_w, cin, (kh / stride[0] + 1) * (kw / stride[1] + 1)), parallel.vector_elementwise_len_threshold);
        rt.backend.col2imInto(&out, &gcol, conv2dDimsFor(in_h, in_w, cin, oh, ow, cout, kh, kw, stride, pad, groups));
        return out;
    }

    var out = try rt.emptyRank(3, .{ in_h, in_w, cin });
    errdefer out.deinit();
    // Same work estimate as the kernel's own thread gate, so the pool is
    // available whenever the kernel would split.
    rt.enableNativeVectorPoolForWork(std.math.mul(usize, parallel.saturatedMul3(oh * ow, cout, cin_pg), kh * kw) catch std.math.maxInt(usize), parallel.vector_elementwise_len_threshold);
    rt.backend.conv2dBackwardInputInto(&out, gg.tensor(), ww.tensor(), conv2dDimsFor(in_h, in_w, cin, oh, ow, cout, kh, kw, stride, pad, groups));
    return out;
}

/// VJP of conv2d wrt weight. `input` is `[h,w,cin]`, `gy` is `[oh,ow,cout]`;
/// result is the forward weight layout `[cout,kh,kw,cin/groups]`.
pub fn conv2dBackwardWeight(rt: *Runtime, input: *const Tensor, gy: *const Tensor, kh: usize, kw: usize, stride: [2]usize, pad: [2]usize, groups: usize) !Tensor {
    const in_view = try input.rankView(3);
    const gy_view = try gy.rankView(3);
    const h = in_view.shape[0];
    const w = in_view.shape[1];
    const cin = in_view.shape[2];
    const oh = gy_view.shape[0];
    const ow = gy_view.shape[1];
    const cout = gy_view.shape[2];
    if (groups == 0 or cin % groups != 0 or cout % groups != 0 or stride[0] == 0 or stride[1] == 0) return tensor.TensorError.InvalidShape;
    if (h + 2 * pad[0] < kh or w + 2 * pad[1] < kw) return tensor.TensorError.InvalidShape;
    if ((h + 2 * pad[0] - kh) / stride[0] + 1 != oh or (w + 2 * pad[1] - kw) / stride[1] + 1 != ow) return tensor.TensorError.ShapeMismatch;
    const cin_pg = cin / groups;

    var ii = try rt.prepareContiguous(input);
    defer ii.deinit();
    var gg = try rt.prepareContiguous(gy);
    defer gg.deinit();

    if (groups == 1 and convBwdGemmEnabled()) {
        var gy_2d = try gg.tensor().viewWithStrides(&.{ oh * ow, cout }, &.{ cout, 1 });
        defer gy_2d.deinit();
        if (kh == 1 and kw == 1 and stride[0] == 1 and stride[1] == 1 and pad[0] == 0 and pad[1] == 0) {
            // 1x1 s1 p0: gw = gyᵀ · in, one trans-A GEMM.
            var in_2d = try ii.tensor().viewWithStrides(&.{ h * w, cin }, &.{ cin, 1 });
            defer in_2d.deinit();
            var gw_2d = try exec_matmul.matmul2DDispatch(rt, .trans_a, &gy_2d, &in_2d);
            errdefer gw_2d.deinit();
            var gw_4d = try gw_2d.viewWithStrides(&.{ cout, 1, 1, cin }, &.{ cin, cin, cin, 1 });
            errdefer gw_4d.deinit();
            gw_2d.deinit();
            return gw_4d;
        }
        // Dense general shape: gw = gyᵀ · im2col(in), one trans-A GEMM over
        // the forward's col matrix. For groups == 1 the col ksz layout
        // ((ky·KW+kx)·Cin + ic) IS the forward weight layout per cout row —
        // the forward's weight_2d view relies on the same identity.
        const npos = oh * ow;
        const ksz = kh * kw * cin;
        var col = try rt.emptyRank(2, .{ npos, ksz });
        defer col.deinit();
        rt.enableNativeVectorPoolForWork(npos * ksz, parallel.vector_elementwise_len_threshold);
        rt.backend.im2colInto(&col, ii.tensor(), conv2dDimsFor(h, w, cin, oh, ow, cout, kh, kw, stride, pad, groups));
        var gw_2d = try exec_matmul.matmul2DDispatch(rt, .trans_a, &gy_2d, &col);
        errdefer gw_2d.deinit();
        var gw_4d = try gw_2d.viewWithStrides(&.{ cout, kh, kw, cin }, &.{ ksz, kw * cin, cin, 1 });
        errdefer gw_4d.deinit();
        gw_2d.deinit();
        return gw_4d;
    }

    var out = try rt.emptyRank(4, .{ cout, kh, kw, cin_pg });
    errdefer out.deinit();
    // Same work estimate as the kernel's own thread gate, so the pool is
    // available whenever the kernel would split.
    rt.enableNativeVectorPoolForWork(std.math.mul(usize, parallel.saturatedMul3(oh * ow, cout, cin_pg), kh * kw) catch std.math.maxInt(usize), parallel.vector_elementwise_len_threshold);
    rt.backend.conv2dBackwardWeightInto(&out, ii.tensor(), gg.tensor(), conv2dDimsFor(h, w, cin, oh, ow, cout, kh, kw, stride, pad, groups));
    return out;
}

/// General non-causal 1-D convolution (PyTorch Conv1d semantics — standard
/// cross-correlation, no kernel flip): input `[time, in]`, weight
/// `[tap, in/groups, out]` (out-channel contiguous, the causalConv1d layout
/// family), output `[t_out, out]` with
/// `t_out = (time + 2*pad - dilation*(taps-1) - 1)/stride + 1`.
/// The input is virtually zero-padded `pad` rows on BOTH sides (never
/// materialized). Output channel `o` belongs to group `g = o/(out/groups)` and
/// reads input channels `[g*(in/groups), (g+1)*(in/groups))`. Bias is
/// deliberately not fused — compose it with `addAxisVectorInPlaceRank`.
pub fn conv1dAxisRank(
    rt: *Runtime,
    comptime rank: usize,
    input: *const Tensor,
    weight: *const Tensor,
    comptime time_axis: usize,
    comptime channel_axis: usize,
    stride: usize,
    pad: usize,
    dilation: usize,
    groups: usize,
) !Tensor {
    comptime {
        if (rank != 2) @compileError("conv1d currently requires rank 2");
        if (time_axis != 0 or channel_axis != 1) {
            @compileError("conv1d requires storage order [time, in]");
        }
    }

    const source = try input.rankView(rank);
    const weight_view = try weight.rankView(3);
    const seq = source.shape[time_axis];
    const in_channels = source.shape[channel_axis];
    const taps = weight_view.shape[0];
    const out_channels = weight_view.shape[2];
    if (stride == 0 or dilation == 0 or taps == 0 or groups == 0) return tensor.TensorError.InvalidShape;
    if (in_channels % groups != 0 or out_channels % groups != 0) return tensor.TensorError.ShapeMismatch;
    if (weight_view.shape[1] != in_channels / groups) return tensor.TensorError.ShapeMismatch;
    const span = try std.math.add(usize, try std.math.mul(usize, dilation, taps - 1), 1);
    const padded = try std.math.add(usize, seq, try std.math.mul(usize, 2, pad));
    if (padded < span) return tensor.TensorError.InvalidShape;
    const out_len = (padded - span) / stride + 1;

    var ii = try rt.prepareContiguous(input);
    defer ii.deinit();
    var ww = try rt.prepareContiguous(weight);
    defer ww.deinit();

    var out = try rt.emptyRank(rank, .{ out_len, out_channels });
    errdefer out.deinit();
    rt.enableNativeVectorPoolForWork(causalConvWork(out_len, in_channels / groups, out_channels, taps), parallel.vector_elementwise_len_threshold);
    rt.backend.conv1dInto(&out, ii.tensor(), ww.tensor(), .{
        .seq = seq,
        .out_len = out_len,
        .in_channels = in_channels,
        .out_channels = out_channels,
        .taps = taps,
        .stride = stride,
        .pad = pad,
        .dilation = dilation,
        .groups = groups,
    });
    return out;
}

/// col2im_1d gather (the ggml ConvTranspose1d second half): `col` is
/// `[t_in, taps*out_channels]` with column index `oc*taps + k` (k varying
/// fastest inside each oc block); result `[t_out + output_pad, out_channels]`
/// rows (channel fast) with `t_out = (t_in-1)*stride + taps - 2*pad`; the
/// `output_pad` trailing time rows are zeros — the ggml/omnivoice.cpp
/// convention (ggml_pad right-pad). NOTE: true PyTorch ConvTranspose1d fills
/// those rows with real kernel taps when pad > 0; the deviation is inherited
/// from the ported reference deliberately: matching the C++ reference is the
/// parity goal, and its own PyTorch-vs-C++ harness shows the effect on those
/// rows is negligible (audio cosine > 0.9999).
pub fn col2im1dAxisRank(
    rt: *Runtime,
    col: *const Tensor,
    out_channels: usize,
    taps: usize,
    stride: usize,
    pad: usize,
    output_pad: usize,
) !Tensor {
    const col_view = try col.rankView(2);
    const t_in = col_view.shape[0];
    if (out_channels == 0 or taps == 0 or stride == 0 or t_in == 0) return tensor.TensorError.InvalidShape;
    if (col_view.shape[1] != try std.math.mul(usize, taps, out_channels)) return tensor.TensorError.ShapeMismatch;
    const upsampled = try std.math.add(usize, try std.math.mul(usize, t_in - 1, stride), taps);
    const two_pad = try std.math.mul(usize, 2, pad);
    if (upsampled <= two_pad) return tensor.TensorError.InvalidShape;
    const t_out = upsampled - two_pad;
    const out_len = try std.math.add(usize, t_out, output_pad);

    var cc = try rt.prepareContiguous(col);
    defer cc.deinit();

    var out = try rt.emptyRank(2, .{ out_len, out_channels });
    errdefer out.deinit();
    rt.enableNativeVectorPoolForWork(parallel.saturatedMul3(out_len, out_channels, taps / stride + 1), parallel.vector_elementwise_len_threshold);
    rt.backend.col2im1dInto(&out, cc.tensor(), t_in, out_len, out_channels, taps, stride, pad);
    return out;
}

/// ConvTranspose1d = GEMM + col2im_1d gather (the ggml/omnivoice
/// decomposition): (1) `col = input · weight2ᵀ` → `[T, K*OC]`,
/// (2) col2im gather into `[T_out + output_pad, OC]` with
/// `T_out = (T-1)*stride + K - 2*pad`, (3) optional broadcast bias add.
/// `weight2` is the load-time repacked `[K*OC, IC]` matrix with k varying
/// fastest inside each oc block: `weight2[(oc*K + k)*IC + ic] = w_pt[ic][oc][k]`
/// — exactly the reference's load-time repack of the PyTorch ConvTranspose1d
/// weight `(IC, OC, K)` (omnivoice dac-decoder.h). The `output_pad` trailing
/// time rows are zeros — ggml/omnivoice.cpp convention, NOT true PyTorch
/// semantics when pad > 0 (see col2im1dAxisRank above for the rationale).
pub fn convTranspose1d(
    rt: *Runtime,
    input: *const Tensor,
    weight2: *const Tensor,
    bias: ?*const Tensor,
    out_channels: usize,
    taps: usize,
    stride: usize,
    pad: usize,
    output_pad: usize,
) !Tensor {
    const in_view = try input.rankView(2);
    const t_in = in_view.shape[0];
    const in_channels = in_view.shape[1];
    const w_view = try weight2.rankView(2);
    if (out_channels == 0 or taps == 0 or stride == 0 or t_in == 0) return tensor.TensorError.InvalidShape;
    if (w_view.shape[0] != try std.math.mul(usize, taps, out_channels)) return tensor.TensorError.ShapeMismatch;
    if (w_view.shape[1] != in_channels) return tensor.TensorError.ShapeMismatch;
    const upsampled = try std.math.add(usize, try std.math.mul(usize, t_in - 1, stride), taps);
    if (upsampled <= try std.math.mul(usize, 2, pad)) return tensor.TensorError.InvalidShape;

    var bb: ?PreparedTensor = null;
    defer if (bb) |*p| p.deinit();
    var bias_slice: ?[]const f32 = null;
    if (bias) |b| {
        const bv = try b.rankView(1);
        if (bv.shape[0] != out_channels) return tensor.TensorError.ShapeMismatch;
        bb = try rt.prepareContiguous(b);
        bias_slice = bb.?.tensor().dataConst();
    }

    var col = try exec_matmul.matmul2DDispatch(rt, .trans_b, input, weight2);
    defer col.deinit();

    var out = try col2im1dAxisRank(rt, &col, out_channels, taps, stride, pad, output_pad);
    errdefer out.deinit();
    if (bias_slice) |b| try exec_elementwise.addAxisVectorInPlaceRank(rt, 2, &out, b, 1);
    return out;
}

/// VJP of conv1dAxisRank wrt the input. `gy` is `[out_len, out]`, `weight`
/// the forward `[tap, in/groups, out]`; `seq` is the forward INPUT length
/// (not recoverable from gy alone under stride/pad), validated against the
/// forward geometry. Result is `[seq, in]`.
pub fn conv1dBackwardInputAxisRank(
    rt: *Runtime,
    comptime rank: usize,
    gy: *const Tensor,
    weight: *const Tensor,
    comptime time_axis: usize,
    comptime channel_axis: usize,
    seq: usize,
    stride: usize,
    pad: usize,
    dilation: usize,
    groups: usize,
) !Tensor {
    comptime {
        if (rank != 2) @compileError("conv1dBackwardInput currently requires rank 2");
        if (time_axis != 0 or channel_axis != 1) {
            @compileError("conv1dBackwardInput requires storage order [time, in]");
        }
    }

    const grad_view = try gy.rankView(rank);
    const weight_view = try weight.rankView(3);
    const out_len = grad_view.shape[time_axis];
    const out_channels = grad_view.shape[channel_axis];
    const taps = weight_view.shape[0];
    if (stride == 0 or dilation == 0 or taps == 0 or groups == 0 or seq == 0) return tensor.TensorError.InvalidShape;
    if (weight_view.shape[2] != out_channels) return tensor.TensorError.ShapeMismatch;
    if (out_channels % groups != 0) return tensor.TensorError.ShapeMismatch;
    const in_channels = try std.math.mul(usize, weight_view.shape[1], groups);
    const span = try std.math.add(usize, try std.math.mul(usize, dilation, taps - 1), 1);
    const padded = try std.math.add(usize, seq, try std.math.mul(usize, 2, pad));
    if (padded < span) return tensor.TensorError.InvalidShape;
    if ((padded - span) / stride + 1 != out_len) return tensor.TensorError.ShapeMismatch;

    var gg = try rt.prepareContiguous(gy);
    defer gg.deinit();
    var ww = try rt.prepareContiguous(weight);
    defer ww.deinit();

    var out = try rt.emptyRank(rank, .{ seq, in_channels });
    errdefer out.deinit();
    rt.enableNativeVectorPoolForWork(causalConvWork(out_len, in_channels / groups, out_channels, taps), parallel.vector_elementwise_len_threshold);
    rt.backend.conv1dBackwardInputInto(&out, gg.tensor(), ww.tensor(), .{
        .seq = seq,
        .out_len = out_len,
        .in_channels = in_channels,
        .out_channels = out_channels,
        .taps = taps,
        .stride = stride,
        .pad = pad,
        .dilation = dilation,
        .groups = groups,
    });
    return out;
}

/// VJP of conv1dAxisRank wrt the weight. `input` is the forward `[seq, in]`,
/// `gy` is `[out_len, out]`; result is the forward weight layout
/// `[taps, in/groups, out]`. `gy`'s row count must match the forward
/// geometry implied by (seq, taps, stride, pad, dilation).
pub fn conv1dBackwardWeightAxisRank(
    rt: *Runtime,
    comptime rank: usize,
    input: *const Tensor,
    gy: *const Tensor,
    comptime time_axis: usize,
    comptime channel_axis: usize,
    taps: usize,
    stride: usize,
    pad: usize,
    dilation: usize,
    groups: usize,
) !Tensor {
    comptime {
        if (rank != 2) @compileError("conv1dBackwardWeight currently requires rank 2");
        if (time_axis != 0 or channel_axis != 1) {
            @compileError("conv1dBackwardWeight requires storage order [time, in]");
        }
    }

    const source = try input.rankView(rank);
    const grad_view = try gy.rankView(rank);
    const seq = source.shape[time_axis];
    const in_channels = source.shape[channel_axis];
    const out_len = grad_view.shape[time_axis];
    const out_channels = grad_view.shape[channel_axis];
    if (stride == 0 or dilation == 0 or taps == 0 or groups == 0 or seq == 0) return tensor.TensorError.InvalidShape;
    if (in_channels % groups != 0 or out_channels % groups != 0) return tensor.TensorError.ShapeMismatch;
    const span = try std.math.add(usize, try std.math.mul(usize, dilation, taps - 1), 1);
    const padded = try std.math.add(usize, seq, try std.math.mul(usize, 2, pad));
    if (padded < span) return tensor.TensorError.InvalidShape;
    if ((padded - span) / stride + 1 != out_len) return tensor.TensorError.ShapeMismatch;

    var ii = try rt.prepareContiguous(input);
    defer ii.deinit();
    var gg = try rt.prepareContiguous(gy);
    defer gg.deinit();

    var out = try rt.emptyRank(3, .{ taps, in_channels / groups, out_channels });
    errdefer out.deinit();
    rt.enableNativeVectorPoolForWork(causalConvWork(out_len, in_channels / groups, out_channels, taps), parallel.vector_elementwise_len_threshold);
    rt.backend.conv1dBackwardWeightInto(&out, ii.tensor(), gg.tensor(), .{
        .seq = seq,
        .out_len = out_len,
        .in_channels = in_channels,
        .out_channels = out_channels,
        .taps = taps,
        .stride = stride,
        .pad = pad,
        .dilation = dilation,
        .groups = groups,
    });
    return out;
}

/// VJP of col2im1dAxisRank (the im2col-style gather): `gy` is the upstream
/// gradient over the col2im output `[t_conv + output_pad, out_channels]`
/// with `t_conv = (t_in-1)*stride + taps - 2*pad`; the trailing `output_pad`
/// rows were forward-zeroed and never map back. Result (gcol) is
/// `[t_in, taps*out_channels]` with the forward col layout (`oc*taps + k`,
/// k fastest): `gcol[t_in, oc*taps + k] = gy[t_in*stride + k - pad, oc]`
/// when that row lands in `[0, t_conv)`, else 0.
pub fn col2im1dBackwardAxisRank(
    rt: *Runtime,
    gy: *const Tensor,
    t_in: usize,
    out_channels: usize,
    taps: usize,
    stride: usize,
    pad: usize,
) !Tensor {
    const grad_view = try gy.rankView(2);
    if (out_channels == 0 or taps == 0 or stride == 0 or t_in == 0) return tensor.TensorError.InvalidShape;
    if (grad_view.shape[1] != out_channels) return tensor.TensorError.ShapeMismatch;
    const upsampled = try std.math.add(usize, try std.math.mul(usize, t_in - 1, stride), taps);
    const two_pad = try std.math.mul(usize, 2, pad);
    if (upsampled <= two_pad) return tensor.TensorError.InvalidShape;
    const t_conv = upsampled - two_pad;
    if (grad_view.shape[0] < t_conv) return tensor.TensorError.ShapeMismatch;

    var gg = try rt.prepareContiguous(gy);
    defer gg.deinit();

    var out = try rt.emptyRank(2, .{ t_in, try std.math.mul(usize, taps, out_channels) });
    errdefer out.deinit();
    rt.enableNativeVectorPoolForWork(parallel.saturatedMul3(t_in, out_channels, taps), parallel.vector_elementwise_len_threshold);
    rt.backend.col2im1dBackwardInto(&out, gg.tensor(), t_in, grad_view.shape[0], out_channels, taps, stride, pad);
    return out;
}

pub fn causalConv1dBackwardInputAxisRank(
    rt: *Runtime,
    comptime rank: usize,
    gy: *const Tensor,
    weight: *const Tensor,
    comptime time_axis: usize,
    comptime channel_axis: usize,
    dilation: usize,
) !Tensor {
    comptime {
        if (rank != 2) @compileError("causalConv1dBackwardInput currently requires rank 2");
        if (time_axis != 0 or channel_axis != 1) {
            @compileError("causalConv1dBackwardInput requires storage order [time, in]");
        }
    }

    const grad_view = try gy.rankView(rank);
    const weight_view = try weight.rankView(3);
    const seq = grad_view.shape[time_axis];
    const out_channels = grad_view.shape[channel_axis];
    const taps = weight_view.shape[0];
    const in_channels = weight_view.shape[1];
    if (weight_view.shape[2] != out_channels) return tensor.TensorError.ShapeMismatch;
    try validateCausalConvState(null, in_channels, taps, dilation);

    var gg = try rt.prepareContiguous(gy);
    defer gg.deinit();
    var ww = try rt.prepareContiguous(weight);
    defer ww.deinit();

    var out = try rt.emptyRank(rank, .{ seq, in_channels });
    errdefer out.deinit();
    rt.enableNativeVectorPoolForWork(causalConvWork(seq, in_channels, out_channels, taps), parallel.vector_elementwise_len_threshold);
    rt.backend.causalConv1dBackwardInputInto(&out, gg.tensor(), ww.tensor(), seq, in_channels, out_channels, taps, dilation);
    return out;
}

pub fn causalConv1dBackwardWeightAxisRank(
    rt: *Runtime,
    comptime rank: usize,
    input: *const Tensor,
    gy: *const Tensor,
    comptime time_axis: usize,
    comptime channel_axis: usize,
    taps: usize,
    dilation: usize,
    state: ?[]const f32,
) !Tensor {
    comptime {
        if (rank != 2) @compileError("causalConv1dBackwardWeight currently requires rank 2");
        if (time_axis != 0 or channel_axis != 1) {
            @compileError("causalConv1dBackwardWeight requires storage order [time, in]");
        }
    }

    const source = try input.rankView(rank);
    const grad_view = try gy.rankView(rank);
    const seq = source.shape[time_axis];
    const in_channels = source.shape[channel_axis];
    const out_channels = grad_view.shape[channel_axis];
    if (grad_view.shape[time_axis] != seq) return tensor.TensorError.ShapeMismatch;
    try validateCausalConvState(state, in_channels, taps, dilation);

    var ii = try rt.prepareContiguous(input);
    defer ii.deinit();
    var gg = try rt.prepareContiguous(gy);
    defer gg.deinit();

    var out = try rt.emptyRank(3, .{ taps, in_channels, out_channels });
    errdefer out.deinit();
    rt.enableNativeVectorPoolForWork(causalConvWork(seq, in_channels, out_channels, taps), parallel.vector_elementwise_len_threshold);
    rt.backend.causalConv1dBackwardWeightInto(&out, ii.tensor(), gg.tensor(), state, seq, in_channels, out_channels, taps, dilation);
    return out;
}

/// Grouped causal 1-D convolution: input `[time, in]`, weight
/// `[tap, in_per_group, out]`, output `[time, out]`. Each output channel
/// only sees the corresponding group slice of the input channels.
pub fn groupedCausalConv1dAxisRank(
    rt: *Runtime,
    comptime rank: usize,
    input: *const Tensor,
    weight: *const Tensor,
    comptime time_axis: usize,
    comptime channel_axis: usize,
    dilation: usize,
    groups: usize,
    state: ?[]const f32,
) !Tensor {
    comptime {
        if (rank != 2) @compileError("groupedCausalConv1d currently requires rank 2");
        if (time_axis != 0 or channel_axis != 1) {
            @compileError("groupedCausalConv1d requires storage order [time, in]");
        }
    }

    const source = try input.rankView(rank);
    const weight_view = try weight.rankView(3);
    const seq = source.shape[time_axis];
    const in_channels = source.shape[channel_axis];
    const taps = weight_view.shape[0];
    const out_channels = weight_view.shape[2];
    const in_per_group = try validateGroupedCausalConv(state, in_channels, out_channels, taps, dilation, groups);
    if (weight_view.shape[1] != in_per_group) return tensor.TensorError.ShapeMismatch;

    var ii = try rt.prepareContiguous(input);
    defer ii.deinit();
    var ww = try rt.prepareContiguous(weight);
    defer ww.deinit();

    var out = try rt.emptyRank(rank, .{ seq, out_channels });
    errdefer out.deinit();
    rt.enableNativeVectorPoolForWork(causalConvWork(seq, in_per_group, out_channels, taps), parallel.vector_elementwise_len_threshold);
    rt.backend.groupedCausalConv1dInto(&out, ii.tensor(), ww.tensor(), state, seq, in_channels, out_channels, taps, dilation, groups);
    return out;
}

pub fn groupedCausalConv1dBackwardInputAxisRank(
    rt: *Runtime,
    comptime rank: usize,
    gy: *const Tensor,
    weight: *const Tensor,
    comptime time_axis: usize,
    comptime channel_axis: usize,
    dilation: usize,
    groups: usize,
) !Tensor {
    comptime {
        if (rank != 2) @compileError("groupedCausalConv1dBackwardInput currently requires rank 2");
        if (time_axis != 0 or channel_axis != 1) {
            @compileError("groupedCausalConv1dBackwardInput requires storage order [time, in]");
        }
    }

    const grad_view = try gy.rankView(rank);
    const weight_view = try weight.rankView(3);
    const seq = grad_view.shape[time_axis];
    const out_channels = grad_view.shape[channel_axis];
    const taps = weight_view.shape[0];
    if (weight_view.shape[2] != out_channels) return tensor.TensorError.ShapeMismatch;
    const in_channels = try std.math.mul(usize, weight_view.shape[1], groups);
    const in_per_group = try validateGroupedCausalConv(null, in_channels, out_channels, taps, dilation, groups);
    if (weight_view.shape[1] != in_per_group) return tensor.TensorError.ShapeMismatch;

    var gg = try rt.prepareContiguous(gy);
    defer gg.deinit();
    var ww = try rt.prepareContiguous(weight);
    defer ww.deinit();

    var out = try rt.emptyRank(rank, .{ seq, in_channels });
    errdefer out.deinit();
    rt.enableNativeVectorPoolForWork(causalConvWork(seq, in_per_group, out_channels, taps), parallel.vector_elementwise_len_threshold);
    rt.backend.groupedCausalConv1dBackwardInputInto(&out, gg.tensor(), ww.tensor(), seq, in_channels, out_channels, taps, dilation, groups);
    return out;
}

pub fn groupedCausalConv1dBackwardWeightAxisRank(
    rt: *Runtime,
    comptime rank: usize,
    input: *const Tensor,
    gy: *const Tensor,
    comptime time_axis: usize,
    comptime channel_axis: usize,
    taps: usize,
    dilation: usize,
    groups: usize,
    state: ?[]const f32,
) !Tensor {
    comptime {
        if (rank != 2) @compileError("groupedCausalConv1dBackwardWeight currently requires rank 2");
        if (time_axis != 0 or channel_axis != 1) {
            @compileError("groupedCausalConv1dBackwardWeight requires storage order [time, in]");
        }
    }

    const source = try input.rankView(rank);
    const grad_view = try gy.rankView(rank);
    const seq = source.shape[time_axis];
    const in_channels = source.shape[channel_axis];
    const out_channels = grad_view.shape[channel_axis];
    if (grad_view.shape[time_axis] != seq) return tensor.TensorError.ShapeMismatch;
    const in_per_group = try validateGroupedCausalConv(state, in_channels, out_channels, taps, dilation, groups);

    var ii = try rt.prepareContiguous(input);
    defer ii.deinit();
    var gg = try rt.prepareContiguous(gy);
    defer gg.deinit();

    var out = try rt.emptyRank(3, .{ taps, in_per_group, out_channels });
    errdefer out.deinit();
    rt.enableNativeVectorPoolForWork(causalConvWork(seq, in_per_group, out_channels, taps), parallel.vector_elementwise_len_threshold);
    rt.backend.groupedCausalConv1dBackwardWeightInto(&out, ii.tensor(), gg.tensor(), state, seq, in_channels, out_channels, taps, dilation, groups);
    return out;
}
