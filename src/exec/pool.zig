//! Channel-last 2-D pooling (avg/max) and 2× nearest-neighbour upsampling.
//!
//! Domain module: every op receives an explicit `*Runtime`. Validates
//! geometry, then dispatches to the allocation-free backend kernels (native
//! vector or scalar reference). Forward kernels parallelize over output rows
//! (bit-identical to serial); backwards are correctness-first serial scatters.

const std = @import("std");
const backend_mod = @import("../backend.zig");
const parallel = @import("../parallel.zig");
const tensor = @import("../tensor.zig");

const Runtime = @import("runtime.zig").Runtime;

const Tensor = tensor.Tensor;
pub const PoolKind = backend_mod.PoolKind;
const Pool2dDims = backend_mod.Pool2dDims;

fn pool2dDims(h: usize, w: usize, c: usize, kernel: [2]usize, stride: [2]usize, pad: [2]usize) !Pool2dDims {
    if (kernel[0] == 0 or kernel[1] == 0 or stride[0] == 0 or stride[1] == 0) return tensor.TensorError.InvalidShape;
    if (h + 2 * pad[0] < kernel[0] or w + 2 * pad[1] < kernel[1]) return tensor.TensorError.ShapeMismatch;
    // A tap must reach the input from every window (ONNX/torch demand
    // pad < kernel; this keeps −inf/zero-count windows unreachable).
    if (pad[0] >= kernel[0] or pad[1] >= kernel[1]) return tensor.TensorError.InvalidShape;
    return .{
        .h = h,
        .w = w,
        .c = c,
        .oh = (h + 2 * pad[0] - kernel[0]) / stride[0] + 1,
        .ow = (w + 2 * pad[1] - kernel[1]) / stride[1] + 1,
        .kh = kernel[0],
        .kw = kernel[1],
        .stride_h = stride[0],
        .stride_w = stride[1],
        .pad_h = pad[0],
        .pad_w = pad[1],
    };
}

/// 2-D pooling forward, channel-last rank-3 `[H,W,C]` → `[OH,OW,C]` with
/// `OH = (H + 2*pad_h - KH)/stride_h + 1` (likewise `OW`). `.max` treats the
/// zero-pad border as −inf (out-of-range taps are skipped); `.avg` averages
/// over the valid taps only (ONNX `count_include_pad=0`).
pub fn pool2d(rt: *Runtime, comptime kind: PoolKind, input: *const Tensor, kernel: [2]usize, stride: [2]usize, pad: [2]usize) !Tensor {
    const view = try input.rankView(3);
    const d = try pool2dDims(view.shape[0], view.shape[1], view.shape[2], kernel, stride, pad);

    var ii = try rt.prepareContiguous(input);
    defer ii.deinit();

    var out = try rt.emptyRank(3, .{ d.oh, d.ow, d.c });
    errdefer out.deinit();
    rt.enableNativeVectorPoolForWork(d.oh * d.ow * d.c * d.kh * d.kw, parallel.vector_elementwise_len_threshold);
    rt.backend.pool2dInto(kind, &out, ii.tensor(), d);
    return out;
}

/// VJP of the avg pool: scatter `gy/valid_count` over each window. `gy` is
/// `[OH,OW,C]`; result is `[in_h,in_w,C]`.
pub fn avgPool2dBackward(rt: *Runtime, gy: *const Tensor, in_h: usize, in_w: usize, kernel: [2]usize, stride: [2]usize, pad: [2]usize) !Tensor {
    const gy_view = try gy.rankView(3);
    const d = try pool2dDims(in_h, in_w, gy_view.shape[2], kernel, stride, pad);
    if (d.oh != gy_view.shape[0] or d.ow != gy_view.shape[1]) return tensor.TensorError.ShapeMismatch;

    var gg = try rt.prepareContiguous(gy);
    defer gg.deinit();
    var out = try rt.emptyRank(3, .{ in_h, in_w, d.c });
    errdefer out.deinit();
    rt.backend.avgPool2dBackwardInto(&out, gg.tensor(), d);
    return out;
}

/// VJP of the max pool: route `gy` to each window's argmax tap (first
/// occurrence in scan order), recomputed from the saved forward `input`.
pub fn maxPool2dBackward(rt: *Runtime, input: *const Tensor, gy: *const Tensor, kernel: [2]usize, stride: [2]usize, pad: [2]usize) !Tensor {
    const in_view = try input.rankView(3);
    const gy_view = try gy.rankView(3);
    if (in_view.shape[2] != gy_view.shape[2]) return tensor.TensorError.ShapeMismatch;
    const d = try pool2dDims(in_view.shape[0], in_view.shape[1], in_view.shape[2], kernel, stride, pad);
    if (d.oh != gy_view.shape[0] or d.ow != gy_view.shape[1]) return tensor.TensorError.ShapeMismatch;

    var ii = try rt.prepareContiguous(input);
    defer ii.deinit();
    var gg = try rt.prepareContiguous(gy);
    defer gg.deinit();
    var out = try rt.emptyRank(3, .{ d.h, d.w, d.c });
    errdefer out.deinit();
    rt.backend.maxPool2dBackwardInto(&out, ii.tensor(), gg.tensor(), d);
    return out;
}

/// 2× nearest-neighbour upsample, channel-last rank-3:
/// `out[2h+i, 2w+j, :] = in[h, w, :]` (`i,j ∈ {0,1}`), `[H,W,C]` → `[2H,2W,C]`.
pub fn upsample2xNearest(rt: *Runtime, input: *const Tensor) !Tensor {
    const view = try input.rankView(3);
    const h = view.shape[0];
    const w = view.shape[1];
    const c = view.shape[2];

    var ii = try rt.prepareContiguous(input);
    defer ii.deinit();
    var out = try rt.emptyRank(3, .{ 2 * h, 2 * w, c });
    errdefer out.deinit();
    rt.enableNativeVectorPoolForWork(4 * h * w * c, parallel.vector_elementwise_len_threshold);
    rt.backend.upsample2xNearestInto(&out, ii.tensor(), h, w, c);
    return out;
}

/// VJP of the 2× nearest upsample: each input cell receives the sum of its
/// 2×2 output block — a 2×2 stride-2 sum-pool of `gy` (`[2H,2W,C]` → `[H,W,C]`).
pub fn upsample2xNearestBackward(rt: *Runtime, gy: *const Tensor) !Tensor {
    const view = try gy.rankView(3);
    if (view.shape[0] % 2 != 0 or view.shape[1] % 2 != 0) return tensor.TensorError.ShapeMismatch;
    return pool2d(rt, .sum, gy, .{ 2, 2 }, .{ 2, 2 }, .{ 0, 0 });
}
