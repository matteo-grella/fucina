//! Pointwise ops: arithmetic and comparisons, activations and their fused
//! gated forms, masks/select, integer and logical families, the in-place
//! (`*InPlace`) and ownership-transfer (`take*`) variants, dropout, and
//! the per-channel affine/PReLU passes. Domain module: every op receives
//! an explicit `*ExecContext`; `exec.zig`'s struct body carries the
//! public aliases.
const std = @import("std");
const backend_mod = @import("../backend.zig");
const kernels = backend_mod.kernels;
const tensor = @import("../tensor.zig");

const parallel = @import("../parallel.zig");
const dtype_mod = @import("../dtype.zig");
const exec_row_ops = @import("row_ops.zig");
const exec_shape = @import("shape.zig");
const exec_runtime = @import("runtime.zig");
const ExecContext = @import("../exec.zig").ExecContext;
const backend_ops = backend_mod.ops;
const DType = tensor.DType;
const GatedOp = backend_mod.ops.GatedOp;
const UnaryOp = backend_mod.ops.UnaryOp;

const dispatchRank = exec_shape.dispatchRank;
const ensureForwardFloatMath = exec_shape.ensureForwardFloatMath;
const requireSameRankShape = exec_shape.requireSameRankShape;
const requireSameRankShapeOf = exec_shape.requireSameRankShapeOf;
const shapeArrayFromSlice = exec_shape.shapeArrayFromSlice;
const validateBroadcastRank = exec_shape.validateBroadcastRank;
const isExactSuffixRank = exec_shape.isExactSuffixRank;
const productAfterAxis = exec_shape.productAfterAxis;
const productBeforeAxis = exec_shape.productBeforeAxis;
const contiguousStridesArray = exec_shape.contiguousStridesArray;

const SplitSwiGluTask = exec_row_ops.SplitSwiGluTask;
const SplitGluTask = exec_row_ops.SplitGluTask;
const SplitSwiGluBackwardTask = exec_row_ops.SplitSwiGluBackwardTask;
const SplitGluBackwardTask = exec_row_ops.SplitGluBackwardTask;
const DropoutRangeTask = exec_row_ops.DropoutRangeTask;
const runSplitSwiGluTask = exec_row_ops.runSplitSwiGluTask;
const runSplitGluTask = exec_row_ops.runSplitGluTask;
const runSplitSwiGluBackwardTask = exec_row_ops.runSplitSwiGluBackwardTask;
const runSplitGluBackwardTask = exec_row_ops.runSplitGluBackwardTask;
const runDropoutRangeTask = exec_row_ops.runDropoutRangeTask;
const splitSwiGluRows = exec_row_ops.splitSwiGluRows;
const splitGluRows = exec_row_ops.splitGluRows;
const splitSwiGluBackwardRows = exec_row_ops.splitSwiGluBackwardRows;
const splitGluBackwardRows = exec_row_ops.splitGluBackwardRows;
const dropoutRange = exec_row_ops.dropoutRange;
const dropoutKeepCutoff = exec_row_ops.dropoutKeepCutoff;

const CompareOp = backend_mod.ops.CompareOp;
const ElementwiseOp = backend_mod.ops.ElementwiseOp;
const Tensor = tensor.Tensor;
const elementwise_vector_len: comptime_int = std.simd.suggestVectorLength(f32) orelse 4;
const ElementwiseVec = @Vector(elementwise_vector_len, f32);

pub const TailBroadcastInfo = struct {
    inner: usize,
    values: []const f32,
};

pub fn tryTailBroadcastElementwise(comptime op: ElementwiseOp, out: *Tensor, a: *const Tensor, b: *const Tensor) !bool {
    const out_data = out.data();

    if (a.isContiguous()) {
        if (tailBroadcastInfo(b)) |bi| {
            elementwiseContigTailBroadcast(op, out_data, a.dataConst(), bi, false);
            return true;
        }
    }
    if (b.isContiguous()) {
        if (tailBroadcastInfo(a)) |ai| {
            elementwiseContigTailBroadcast(op, out_data, b.dataConst(), ai, true);
            return true;
        }
    }

    const ai = tailBroadcastInfo(a) orelse return false;
    const bi = tailBroadcastInfo(b) orelse return false;
    if (ai.inner != bi.inner and ai.inner != 1 and bi.inner != 1) return false;

    const inner = @max(ai.inner, bi.inner);
    var base: usize = 0;
    while (base < out_data.len) : (base += inner) {
        for (0..inner) |j| {
            const av = ai.values[if (ai.inner == 1) 0 else j];
            const bv = bi.values[if (bi.inner == 1) 0 else j];
            out_data[base + j] = applyElementwise(op, av, bv);
        }
    }
    return true;
}

fn elementwiseContigTailBroadcast(
    comptime op: ElementwiseOp,
    out: []f32,
    contiguous: []const f32,
    broadcast: TailBroadcastInfo,
    broadcast_is_left: bool,
) void {
    if (broadcast.inner == 1) {
        elementwiseContigScalarBroadcast(op, out, contiguous, broadcast.values[0], broadcast_is_left);
        return;
    }
    var base: usize = 0;
    while (base < out.len) : (base += broadcast.inner) {
        var j: usize = 0;
        switch (op) {
            .add => {
                while (j + elementwise_vector_len <= broadcast.inner) : (j += elementwise_vector_len) {
                    const cv: ElementwiseVec = contiguous[base + j ..][0..elementwise_vector_len].*;
                    const bv: ElementwiseVec = broadcast.values[j..][0..elementwise_vector_len].*;
                    out[base + j ..][0..elementwise_vector_len].* = cv + bv;
                }
            },
            .sub => {
                while (j + elementwise_vector_len <= broadcast.inner) : (j += elementwise_vector_len) {
                    const cv: ElementwiseVec = contiguous[base + j ..][0..elementwise_vector_len].*;
                    const bv: ElementwiseVec = broadcast.values[j..][0..elementwise_vector_len].*;
                    out[base + j ..][0..elementwise_vector_len].* = if (broadcast_is_left) bv - cv else cv - bv;
                }
            },
            .mul => {
                while (j + elementwise_vector_len <= broadcast.inner) : (j += elementwise_vector_len) {
                    const cv: ElementwiseVec = contiguous[base + j ..][0..elementwise_vector_len].*;
                    const bv: ElementwiseVec = broadcast.values[j..][0..elementwise_vector_len].*;
                    out[base + j ..][0..elementwise_vector_len].* = cv * bv;
                }
            },
            .div => {
                while (j + elementwise_vector_len <= broadcast.inner) : (j += elementwise_vector_len) {
                    const cv: ElementwiseVec = contiguous[base + j ..][0..elementwise_vector_len].*;
                    const bv: ElementwiseVec = broadcast.values[j..][0..elementwise_vector_len].*;
                    out[base + j ..][0..elementwise_vector_len].* = if (broadcast_is_left) bv / cv else cv / bv;
                }
            },
            .max, .min => {
                while (j + elementwise_vector_len <= broadcast.inner) : (j += elementwise_vector_len) {
                    const cv: ElementwiseVec = contiguous[base + j ..][0..elementwise_vector_len].*;
                    const bv: ElementwiseVec = broadcast.values[j..][0..elementwise_vector_len].*;
                    out[base + j ..][0..elementwise_vector_len].* = maxMinVec(op, cv, bv);
                }
            },
        }
        while (j < broadcast.inner) : (j += 1) {
            const c = contiguous[base + j];
            const b = broadcast.values[j];
            out[base + j] = if (broadcast_is_left)
                applyElementwise(op, b, c)
            else
                applyElementwise(op, c, b);
        }
    }
}

fn elementwiseContigScalarBroadcast(
    comptime op: ElementwiseOp,
    out: []f32,
    contiguous: []const f32,
    scalar: f32,
    broadcast_is_left: bool,
) void {
    const sv: ElementwiseVec = @splat(scalar);
    var i: usize = 0;
    switch (op) {
        .add => {
            while (i + elementwise_vector_len <= out.len) : (i += elementwise_vector_len) {
                const cv: ElementwiseVec = contiguous[i..][0..elementwise_vector_len].*;
                out[i..][0..elementwise_vector_len].* = cv + sv;
            }
        },
        .sub => {
            while (i + elementwise_vector_len <= out.len) : (i += elementwise_vector_len) {
                const cv: ElementwiseVec = contiguous[i..][0..elementwise_vector_len].*;
                out[i..][0..elementwise_vector_len].* = if (broadcast_is_left) sv - cv else cv - sv;
            }
        },
        .mul => {
            while (i + elementwise_vector_len <= out.len) : (i += elementwise_vector_len) {
                const cv: ElementwiseVec = contiguous[i..][0..elementwise_vector_len].*;
                out[i..][0..elementwise_vector_len].* = cv * sv;
            }
        },
        .div => {
            while (i + elementwise_vector_len <= out.len) : (i += elementwise_vector_len) {
                const cv: ElementwiseVec = contiguous[i..][0..elementwise_vector_len].*;
                out[i..][0..elementwise_vector_len].* = if (broadcast_is_left) sv / cv else cv / sv;
            }
        },
        .max, .min => {
            while (i + elementwise_vector_len <= out.len) : (i += elementwise_vector_len) {
                const cv: ElementwiseVec = contiguous[i..][0..elementwise_vector_len].*;
                out[i..][0..elementwise_vector_len].* = maxMinVec(op, cv, sv);
            }
        },
    }
    while (i < out.len) : (i += 1) {
        out[i] = if (broadcast_is_left)
            applyElementwise(op, scalar, contiguous[i])
        else
            applyElementwise(op, contiguous[i], scalar);
    }
}

pub fn tryTailBroadcastElementwiseInPlace(comptime op: ElementwiseOp, target: *Tensor, other: *const Tensor) bool {
    const broadcast = tailBroadcastInfo(other) orelse return false;
    elementwiseTailBroadcastInPlace(op, target.data(), broadcast);
    return true;
}

fn elementwiseTailBroadcastInPlace(comptime op: ElementwiseOp, target: []f32, broadcast: TailBroadcastInfo) void {
    if (broadcast.inner == 1) {
        elementwiseScalarBroadcastInPlace(op, target, broadcast.values[0]);
        return;
    }
    var base: usize = 0;
    while (base < target.len) : (base += broadcast.inner) {
        var j: usize = 0;
        switch (op) {
            .add => {
                while (j + elementwise_vector_len <= broadcast.inner) : (j += elementwise_vector_len) {
                    const tv: ElementwiseVec = target[base + j ..][0..elementwise_vector_len].*;
                    const bv: ElementwiseVec = broadcast.values[j..][0..elementwise_vector_len].*;
                    target[base + j ..][0..elementwise_vector_len].* = tv + bv;
                }
                while (j < broadcast.inner) : (j += 1) {
                    target[base + j] += broadcast.values[j];
                }
            },
            .sub => {
                while (j + elementwise_vector_len <= broadcast.inner) : (j += elementwise_vector_len) {
                    const tv: ElementwiseVec = target[base + j ..][0..elementwise_vector_len].*;
                    const bv: ElementwiseVec = broadcast.values[j..][0..elementwise_vector_len].*;
                    target[base + j ..][0..elementwise_vector_len].* = tv - bv;
                }
                while (j < broadcast.inner) : (j += 1) {
                    target[base + j] -= broadcast.values[j];
                }
            },
            .mul => {
                while (j + elementwise_vector_len <= broadcast.inner) : (j += elementwise_vector_len) {
                    const tv: ElementwiseVec = target[base + j ..][0..elementwise_vector_len].*;
                    const bv: ElementwiseVec = broadcast.values[j..][0..elementwise_vector_len].*;
                    target[base + j ..][0..elementwise_vector_len].* = tv * bv;
                }
                while (j < broadcast.inner) : (j += 1) {
                    target[base + j] *= broadcast.values[j];
                }
            },
            .div => {
                while (j + elementwise_vector_len <= broadcast.inner) : (j += elementwise_vector_len) {
                    const tv: ElementwiseVec = target[base + j ..][0..elementwise_vector_len].*;
                    const bv: ElementwiseVec = broadcast.values[j..][0..elementwise_vector_len].*;
                    target[base + j ..][0..elementwise_vector_len].* = tv / bv;
                }
                while (j < broadcast.inner) : (j += 1) {
                    target[base + j] /= broadcast.values[j];
                }
            },
            .max, .min => {
                while (j + elementwise_vector_len <= broadcast.inner) : (j += elementwise_vector_len) {
                    const tv: ElementwiseVec = target[base + j ..][0..elementwise_vector_len].*;
                    const bv: ElementwiseVec = broadcast.values[j..][0..elementwise_vector_len].*;
                    target[base + j ..][0..elementwise_vector_len].* = maxMinVec(op, tv, bv);
                }
                while (j < broadcast.inner) : (j += 1) {
                    target[base + j] = applyElementwise(op, target[base + j], broadcast.values[j]);
                }
            },
        }
    }
}

fn elementwiseScalarBroadcastInPlace(comptime op: ElementwiseOp, target: []f32, scalar: f32) void {
    const sv: ElementwiseVec = @splat(scalar);
    var i: usize = 0;
    switch (op) {
        .add => {
            while (i + elementwise_vector_len <= target.len) : (i += elementwise_vector_len) {
                const tv: ElementwiseVec = target[i..][0..elementwise_vector_len].*;
                target[i..][0..elementwise_vector_len].* = tv + sv;
            }
            while (i < target.len) : (i += 1) target[i] += scalar;
        },
        .sub => {
            while (i + elementwise_vector_len <= target.len) : (i += elementwise_vector_len) {
                const tv: ElementwiseVec = target[i..][0..elementwise_vector_len].*;
                target[i..][0..elementwise_vector_len].* = tv - sv;
            }
            while (i < target.len) : (i += 1) target[i] -= scalar;
        },
        .mul => {
            while (i + elementwise_vector_len <= target.len) : (i += elementwise_vector_len) {
                const tv: ElementwiseVec = target[i..][0..elementwise_vector_len].*;
                target[i..][0..elementwise_vector_len].* = tv * sv;
            }
            while (i < target.len) : (i += 1) target[i] *= scalar;
        },
        .div => {
            while (i + elementwise_vector_len <= target.len) : (i += elementwise_vector_len) {
                const tv: ElementwiseVec = target[i..][0..elementwise_vector_len].*;
                target[i..][0..elementwise_vector_len].* = tv / sv;
            }
            while (i < target.len) : (i += 1) target[i] /= scalar;
        },
        .max, .min => {
            while (i + elementwise_vector_len <= target.len) : (i += elementwise_vector_len) {
                const tv: ElementwiseVec = target[i..][0..elementwise_vector_len].*;
                target[i..][0..elementwise_vector_len].* = maxMinVec(op, tv, sv);
            }
            while (i < target.len) : (i += 1) target[i] = applyElementwise(op, target[i], scalar);
        },
    }
}

pub fn elementwiseContiguousInPlace(comptime op: ElementwiseOp, target: []f32, other: []const f32) void {
    var i: usize = 0;
    switch (op) {
        .add => {
            while (i + elementwise_vector_len <= target.len) : (i += elementwise_vector_len) {
                const tv: ElementwiseVec = target[i..][0..elementwise_vector_len].*;
                const ov: ElementwiseVec = other[i..][0..elementwise_vector_len].*;
                target[i..][0..elementwise_vector_len].* = tv + ov;
            }
            while (i < target.len) : (i += 1) target[i] += other[i];
        },
        .sub => {
            while (i + elementwise_vector_len <= target.len) : (i += elementwise_vector_len) {
                const tv: ElementwiseVec = target[i..][0..elementwise_vector_len].*;
                const ov: ElementwiseVec = other[i..][0..elementwise_vector_len].*;
                target[i..][0..elementwise_vector_len].* = tv - ov;
            }
            while (i < target.len) : (i += 1) target[i] -= other[i];
        },
        .mul => {
            while (i + elementwise_vector_len <= target.len) : (i += elementwise_vector_len) {
                const tv: ElementwiseVec = target[i..][0..elementwise_vector_len].*;
                const ov: ElementwiseVec = other[i..][0..elementwise_vector_len].*;
                target[i..][0..elementwise_vector_len].* = tv * ov;
            }
            while (i < target.len) : (i += 1) target[i] *= other[i];
        },
        .div => {
            while (i + elementwise_vector_len <= target.len) : (i += elementwise_vector_len) {
                const tv: ElementwiseVec = target[i..][0..elementwise_vector_len].*;
                const ov: ElementwiseVec = other[i..][0..elementwise_vector_len].*;
                target[i..][0..elementwise_vector_len].* = tv / ov;
            }
            while (i < target.len) : (i += 1) target[i] /= other[i];
        },
        .max, .min => {
            while (i + elementwise_vector_len <= target.len) : (i += elementwise_vector_len) {
                const tv: ElementwiseVec = target[i..][0..elementwise_vector_len].*;
                const ov: ElementwiseVec = other[i..][0..elementwise_vector_len].*;
                target[i..][0..elementwise_vector_len].* = maxMinVec(op, tv, ov);
            }
            while (i < target.len) : (i += 1) target[i] = applyElementwise(op, target[i], other[i]);
        },
    }
}

pub fn elementwiseContiguousInto(comptime op: ElementwiseOp, out: []f32, a: []const f32, b: []const f32) void {
    var i: usize = 0;
    switch (op) {
        .add => {
            while (i + elementwise_vector_len <= out.len) : (i += elementwise_vector_len) {
                const av: ElementwiseVec = a[i..][0..elementwise_vector_len].*;
                const bv: ElementwiseVec = b[i..][0..elementwise_vector_len].*;
                out[i..][0..elementwise_vector_len].* = av + bv;
            }
        },
        .sub => {
            while (i + elementwise_vector_len <= out.len) : (i += elementwise_vector_len) {
                const av: ElementwiseVec = a[i..][0..elementwise_vector_len].*;
                const bv: ElementwiseVec = b[i..][0..elementwise_vector_len].*;
                out[i..][0..elementwise_vector_len].* = av - bv;
            }
        },
        .mul => {
            while (i + elementwise_vector_len <= out.len) : (i += elementwise_vector_len) {
                const av: ElementwiseVec = a[i..][0..elementwise_vector_len].*;
                const bv: ElementwiseVec = b[i..][0..elementwise_vector_len].*;
                out[i..][0..elementwise_vector_len].* = av * bv;
            }
        },
        .div => {
            while (i + elementwise_vector_len <= out.len) : (i += elementwise_vector_len) {
                const av: ElementwiseVec = a[i..][0..elementwise_vector_len].*;
                const bv: ElementwiseVec = b[i..][0..elementwise_vector_len].*;
                out[i..][0..elementwise_vector_len].* = av / bv;
            }
        },
        .max, .min => {
            while (i + elementwise_vector_len <= out.len) : (i += elementwise_vector_len) {
                const av: ElementwiseVec = a[i..][0..elementwise_vector_len].*;
                const bv: ElementwiseVec = b[i..][0..elementwise_vector_len].*;
                out[i..][0..elementwise_vector_len].* = maxMinVec(op, av, bv);
            }
        },
    }
    while (i < out.len) : (i += 1) {
        out[i] = applyElementwise(op, a[i], b[i]);
    }
}

/// torch.maximum/minimum on a vector: NaN in either lane propagates NaN
/// (bare @max/@min follow IEEE maxNum and would drop it). max/min are
/// commutative, so operand order (broadcast_is_left) never matters.
inline fn maxMinVec(comptime op: ElementwiseOp, a: ElementwiseVec, b: ElementwiseVec) ElementwiseVec {
    const raw = if (comptime op == .max) @max(a, b) else @min(a, b);
    const nan_v: ElementwiseVec = @splat(std.math.nan(f32));
    return @select(f32, a != a, nan_v, @select(f32, b != b, nan_v, raw));
}

fn applyElementwise(comptime op: ElementwiseOp, a: f32, b: f32) f32 {
    return switch (op) {
        .add => a + b,
        .sub => a - b,
        .mul => a * b,
        .div => a / b,
        .max => if (a != a or b != b) std.math.nan(f32) else @max(a, b),
        .min => if (a != a or b != b) std.math.nan(f32) else @min(a, b),
    };
}

pub fn tailBroadcastInfo(x: *const Tensor) ?TailBroadcastInfo {
    if (x.isContiguous()) return null;
    @constCast(x.buffer).waitReady();

    var start: ?usize = null;
    for (x.strides.slice(), 0..) |stride, i| {
        if (stride != 0) {
            start = i;
            break;
        }
    }

    const suffix_start = start orelse {
        return .{
            .inner = 1,
            .values = x.buffer.data[x.offset .. x.offset + 1],
        };
    };

    for (0..suffix_start) |i| {
        if (x.strides.at(i) != 0) return null;
    }

    var expected: usize = 1;
    var i = x.shape.len;
    while (i > suffix_start) {
        i -= 1;
        if (x.strides.at(i) != expected) return null;
        expected *= x.shape.at(i);
    }

    return .{
        .inner = expected,
        .values = x.buffer.data[x.offset .. x.offset + expected],
    };
}

/// Runtime-rank elementwise binary op: dispatches over `a.shape.len` onto
/// the same rank-monomorphized kernels the comptime-rank spellings
/// (`add`/`sub`/`mul`/`div`/`max`/`min`) reach directly.
/// Same-shape binary op with the rank resolved at runtime; `add`..`min`
/// are the comptime-rank entries. Float 16-bit inputs run the typed kernel
/// (`.pointwise` policy) except `max`/`min`, which have an f32 kernel only
/// and follow the `.widened` policy.
pub fn elementwise(
    ctx: *ExecContext,
    comptime dtype: DType,
    comptime op: ElementwiseOp,
    a: *const tensor.TensorOf(dtype),
    b: *const tensor.TensorOf(dtype),
) !tensor.TensorOf(dtype_mod.outputDType(.pointwise, dtype)) {
    return switch (a.shape.len) {
        inline 1...tensor.max_rank => |rank| elementwiseRankTyped(ctx, dtype, rank, op, a, b),
        else => tensor.TensorError.InvalidShape,
    };
}

pub fn add(
    ctx: *ExecContext,
    comptime dtype: DType,
    comptime rank: usize,
    a: *const tensor.TensorOf(dtype),
    b: *const tensor.TensorOf(dtype),
) !tensor.TensorOf(dtype_mod.outputDType(.pointwise, dtype)) {
    return elementwiseRankTyped(ctx, dtype, rank, .add, a, b);
}

pub fn sub(
    ctx: *ExecContext,
    comptime dtype: DType,
    comptime rank: usize,
    a: *const tensor.TensorOf(dtype),
    b: *const tensor.TensorOf(dtype),
) !tensor.TensorOf(dtype_mod.outputDType(.pointwise, dtype)) {
    return elementwiseRankTyped(ctx, dtype, rank, .sub, a, b);
}

pub fn mul(
    ctx: *ExecContext,
    comptime dtype: DType,
    comptime rank: usize,
    a: *const tensor.TensorOf(dtype),
    b: *const tensor.TensorOf(dtype),
) !tensor.TensorOf(dtype_mod.outputDType(.pointwise, dtype)) {
    return elementwiseRankTyped(ctx, dtype, rank, .mul, a, b);
}

pub fn div(
    ctx: *ExecContext,
    comptime dtype: DType,
    comptime rank: usize,
    a: *const tensor.TensorOf(dtype),
    b: *const tensor.TensorOf(dtype),
) !tensor.TensorOf(dtype_mod.outputDType(.pointwise, dtype)) {
    return elementwiseRankTyped(ctx, dtype, rank, .div, a, b);
}

/// Elementwise maximum: the f32 kernel, the exact integer path, or the
/// `.widened` policy for 16-bit floats (no typed max kernel exists).
pub fn max(
    ctx: *ExecContext,
    comptime dtype: DType,
    comptime rank: usize,
    a: *const tensor.TensorOf(dtype),
    b: *const tensor.TensorOf(dtype),
) !tensor.TensorOf(dtype_mod.outputDType(.pointwise, dtype)) {
    return elementwiseRankTyped(ctx, dtype, rank, .max, a, b);
}

/// Elementwise minimum (see `max`).
pub fn min(
    ctx: *ExecContext,
    comptime dtype: DType,
    comptime rank: usize,
    a: *const tensor.TensorOf(dtype),
    b: *const tensor.TensorOf(dtype),
) !tensor.TensorOf(dtype_mod.outputDType(.pointwise, dtype)) {
    return elementwiseRankTyped(ctx, dtype, rank, .min, a, b);
}

/// Gated product of two same-shape tensors (`op` picks the gate). One f32
/// kernel; 16-bit inputs follow the `.widened` policy.
pub fn gated(
    ctx: *ExecContext,
    comptime dtype: DType,
    comptime rank: usize,
    comptime op: GatedOp,
    a: *const tensor.TensorOf(dtype),
    b: *const tensor.TensorOf(dtype),
) !tensor.TensorOf(dtype) {
    const compute = comptime ExecContext.widenedCompute(dtype, "gated");
    const shape = try requireSameRankShapeOf(dtype, rank, a, b);
    var aa = try ctx.prepareAs(dtype, compute, a);
    defer aa.deinit();
    var bb = try ctx.prepareAs(dtype, compute, b);
    defer bb.deinit();

    const ap = aa.tensor();
    const bp = bb.tensor();
    var out = try ctx.empty(compute, shape);
    errdefer out.deinit();
    ctx.enableNativeVectorPoolForWork(ap.len(), parallel.vector_elementwise_len_threshold);
    kernels.gatedContiguousIntoUnchecked(ctx.pc(), op, &out, ap, bp, ap.len());
    return ctx.storeAs(compute, dtype, out);
}

/// Split-gated forward: `x` halves along `axis`, one half gates the other.
/// The gate-half conventions are OPPOSITE (ggml parity): swiglu gates with
/// the FIRST half (`silu(first) * second`), glu with the SECOND
/// (`first * sigmoid(second)`). One f32 kernel; 16-bit inputs follow the
/// `.widened` policy.
pub fn splitGated(
    ctx: *ExecContext,
    comptime dtype: DType,
    comptime rank: usize,
    comptime op: GatedOp,
    x: *const tensor.TensorOf(dtype),
    comptime axis: usize,
) !tensor.TensorOf(dtype) {
    const compute = comptime ExecContext.widenedCompute(dtype, "splitGated");
    var xx = try ctx.prepareAs(dtype, compute, x);
    defer xx.deinit();
    var out = try splitGatedF32(ctx, op, rank, xx.tensor(), axis);
    errdefer out.deinit();
    return ctx.storeAs(compute, dtype, out);
}

fn splitGatedF32(ctx: *ExecContext, comptime op: GatedOp, comptime rank: usize, x: *const Tensor, comptime axis: usize) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");
    const Task = switch (op) {
        .swiglu => SplitSwiGluTask,
        .glu => SplitGluTask,
        .geglu => @compileError("no split-geglu row kernel or gate-half convention exists"),
        .situ => @compileError("no split-situ kernel (K3 projects gate and up separately; use the pointwise `situ`)"),
    };
    const runTask = switch (op) {
        .swiglu => runSplitSwiGluTask,
        .glu => runSplitGluTask,
        else => unreachable,
    };
    const rowsKernel = switch (op) {
        .swiglu => splitSwiGluRows,
        .glu => splitGluRows,
        else => unreachable,
    };

    const source = try x.rankView(rank);
    const axis_dim = source.shape[axis];
    if (axis_dim % 2 != 0) return tensor.TensorError.InvalidShape;

    var out_shape = source.shape;
    out_shape[axis] = axis_dim / 2;

    var xx = try ctx.prepareContiguous(.f32, x);
    defer xx.deinit();
    const input = xx.tensor().dataConst();

    var out = try ctx.empty(.f32, out_shape);
    errdefer out.deinit();
    const output = out.data();

    const half = axis_dim / 2;
    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);
    if (inner == 1) {
        const base: Task = .{
            .input = input,
            .output = output,
            .axis_dim = axis_dim,
            .half = half,
            .outer_start = 0,
            .outer_end = outer,
        };
        if (out.len() >= parallel.vector_elementwise_len_threshold / 8) {
            if (ctx.dispatchRange(Task, "outer_start", "outer_end", base, outer, runTask)) return out;
        }
        rowsKernel(base);
        return out;
    }

    for (0..outer) |outer_i| {
        const in_base = outer_i * axis_dim * inner;
        const out_base = outer_i * half * inner;
        for (0..half) |axis_i| {
            for (0..inner) |inner_i| {
                const first = input[in_base + axis_i * inner + inner_i];
                const second = input[in_base + (half + axis_i) * inner + inner_i];
                output[out_base + axis_i * inner + inner_i] = switch (op) {
                    .swiglu => second * first / (1 + @exp(-first)),
                    .glu => first / (1 + @exp(-second)),
                    else => unreachable,
                };
            }
        }
    }
    return out;
}

pub fn splitSwiGluBackward(ctx: *ExecContext, comptime rank: usize, x: *const Tensor, gy: *const Tensor, comptime axis: usize) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");

    const source = try x.rankView(rank);
    const axis_dim = source.shape[axis];
    if (axis_dim % 2 != 0) return tensor.TensorError.InvalidShape;
    const half = axis_dim / 2;

    var expected_grad_shape = source.shape;
    expected_grad_shape[axis] = half;
    const grad_view = try gy.rankView(rank);
    if (!std.mem.eql(usize, grad_view.shape[0..], expected_grad_shape[0..])) return tensor.TensorError.ShapeMismatch;

    var xx = try ctx.prepareContiguous(.f32, x);
    defer xx.deinit();
    var ggy = try ctx.prepareContiguous(.f32, gy);
    defer ggy.deinit();
    const input = xx.tensor().dataConst();
    const grad = ggy.tensor().dataConst();

    var out = try ctx.empty(.f32, source.shape);
    errdefer out.deinit();
    const output = out.data();

    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);
    if (inner == 1) {
        const base_task: SplitSwiGluBackwardTask = .{
            .input = input,
            .grad = grad,
            .output = output,
            .axis_dim = axis_dim,
            .half = half,
            .outer_start = 0,
            .outer_end = outer,
        };
        if (out.len() >= parallel.vector_elementwise_len_threshold / 4 and outer > 1) {
            if (ctx.dispatchRange(SplitSwiGluBackwardTask, "outer_start", "outer_end", base_task, outer, runSplitSwiGluBackwardTask)) {
                return out;
            }
        }

        splitSwiGluBackwardRows(base_task);
        return out;
    }

    for (0..outer) |outer_i| {
        const in_base = outer_i * axis_dim * inner;
        const grad_base = outer_i * half * inner;
        for (0..half) |axis_i| {
            for (0..inner) |inner_i| {
                const gate_offset = in_base + axis_i * inner + inner_i;
                const up_offset = in_base + (half + axis_i) * inner + inner_i;
                const grad_value = grad[grad_base + axis_i * inner + inner_i];
                const gate = input[gate_offset];
                const up = input[up_offset];
                const sigmoid_value = backend_ops.sigmoidScalar(gate);
                const silu_value = gate * sigmoid_value;
                const silu_deriv = sigmoid_value * (1 + gate * (1 - sigmoid_value));
                output[gate_offset] = grad_value * up * silu_deriv;
                output[up_offset] = grad_value * silu_value;
            }
        }
    }
    return out;
}

pub fn splitGluBackward(ctx: *ExecContext, comptime rank: usize, x: *const Tensor, gy: *const Tensor, comptime axis: usize) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");

    const source = try x.rankView(rank);
    const axis_dim = source.shape[axis];
    if (axis_dim % 2 != 0) return tensor.TensorError.InvalidShape;
    const half = axis_dim / 2;

    var expected_grad_shape = source.shape;
    expected_grad_shape[axis] = half;
    const grad_view = try gy.rankView(rank);
    if (!std.mem.eql(usize, grad_view.shape[0..], expected_grad_shape[0..])) return tensor.TensorError.ShapeMismatch;

    var xx = try ctx.prepareContiguous(.f32, x);
    defer xx.deinit();
    var ggy = try ctx.prepareContiguous(.f32, gy);
    defer ggy.deinit();
    const input = xx.tensor().dataConst();
    const grad = ggy.tensor().dataConst();

    var out = try ctx.empty(.f32, source.shape);
    errdefer out.deinit();
    const output = out.data();

    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);
    if (inner == 1) {
        const base_task: SplitGluBackwardTask = .{
            .input = input,
            .grad = grad,
            .output = output,
            .axis_dim = axis_dim,
            .half = half,
            .outer_start = 0,
            .outer_end = outer,
        };
        if (out.len() >= parallel.vector_elementwise_len_threshold / 4 and outer > 1) {
            if (ctx.dispatchRange(SplitGluBackwardTask, "outer_start", "outer_end", base_task, outer, runSplitGluBackwardTask)) {
                return out;
            }
        }

        splitGluBackwardRows(base_task);
        return out;
    }

    for (0..outer) |outer_i| {
        const in_base = outer_i * axis_dim * inner;
        const grad_base = outer_i * half * inner;
        for (0..half) |axis_i| {
            for (0..inner) |inner_i| {
                const up_offset = in_base + axis_i * inner + inner_i;
                const gate_offset = in_base + (half + axis_i) * inner + inner_i;
                const grad_value = grad[grad_base + axis_i * inner + inner_i];
                const up = input[up_offset];
                const gate = input[gate_offset];
                const sigmoid_value = backend_ops.sigmoidScalar(gate);
                output[up_offset] = grad_value * sigmoid_value;
                output[gate_offset] = grad_value * up * sigmoid_value * (1 - sigmoid_value);
            }
        }
    }
    return out;
}

/// Consume `target` and return `target * scalar_value`: the multiply runs in
/// `target`'s storage when the runtime can take it (`canTakeInPlace`), else
/// into a fresh tensor with `target` released. Either way the caller owns
/// only the result.
pub fn takeScale(ctx: *ExecContext, target: *Tensor, scalar_value: f32) !Tensor {
    if (target.canTakeInPlace()) {
        ctx.enableNativeVectorPoolForWork(target.len(), parallel.vector_elementwise_len_threshold);
        try kernels.scaleInto(ctx.pc(), target, target, scalar_value);
        return takeTensor(target);
    }

    var result = try scale(ctx, .f32, target, scalar_value);
    errdefer result.deinit();
    return discardTakenInput(target, result);
}

/// Consume `target` and return `op(target)`; ownership as `takeScale`.
pub fn takeUnary(ctx: *ExecContext, comptime op: UnaryOp, target: *Tensor) !Tensor {
    if (target.canTakeInPlace()) {
        ctx.enableNativeVectorPoolForWork(target.len(), parallel.vector_elementwise_len_threshold);
        kernels.unaryContiguousIntoUnchecked(ctx.pc(), op, target, target, target.len());
        return takeTensor(target);
    }

    var result = try unary(ctx, .f32, op, target);
    errdefer result.deinit();
    return discardTakenInput(target, result);
}

/// Multiply by a scalar. `.f32` runs the SIMD kernel; 16-bit floats run
/// the widening loop with the pointwise compute/output dtype contract.
pub fn scale(
    ctx: *ExecContext,
    comptime dtype: DType,
    x: *const tensor.TensorOf(dtype),
    scalar_value: dtype_mod.Accumulator(dtype),
) !tensor.TensorOf(dtype_mod.outputDType(.pointwise, dtype)) {
    if (comptime dtype == .f32) {
        var xx = try ctx.prepareContiguous(.f32, x);
        defer xx.deinit();

        const xp = xx.tensor();
        var out = try ctx.empty(.f32, xp.shape.slice());
        errdefer out.deinit();
        ctx.enableNativeVectorPoolForWork(xp.len(), parallel.vector_elementwise_len_threshold);
        try kernels.scaleInto(ctx.pc(), &out, xp, scalar_value);
        return out;
    }
    comptime ensureForwardFloatMath(dtype);
    const compute_dtype = comptime dtype_mod.computeDType(.pointwise, dtype);
    const output_dtype = comptime dtype_mod.outputDType(.pointwise, dtype);

    var xx = try ctx.prepareContiguous(dtype, x);
    defer xx.deinit();
    const input = xx.tensor().dataConst();

    var out = try ctx.empty(output_dtype, x.shape.slice());
    errdefer out.deinit();
    const output = out.data();
    for (output, input) |*dst, value| {
        const product = dtype_mod.castFloat(dtype, compute_dtype, value) * dtype_mod.castFloat(dtype, compute_dtype, dtype_mod.fromAccumulator(dtype, scalar_value));
        dst.* = dtype_mod.castFloat(compute_dtype, output_dtype, product);
    }
    return out;
}

pub fn addScalar(ctx: *ExecContext, comptime dtype: DType, x: *const tensor.TensorOf(dtype), scalar_value: f32) !tensor.TensorOf(dtype) {
    const compute = comptime ExecContext.widenedCompute(dtype, "addScalar");
    var xx = try ctx.prepareAs(dtype, compute, x);
    defer xx.deinit();
    const xp = xx.tensor();
    var out = try ctx.empty(compute, xp.shape.slice());
    errdefer out.deinit();
    for (xp.dataConst(), out.data()) |value, *dst| dst.* = value + scalar_value;
    return ctx.storeAs(compute, dtype, out);
}

pub fn powScalar(ctx: *ExecContext, comptime dtype: DType, x: *const tensor.TensorOf(dtype), exponent: f32) !tensor.TensorOf(dtype) {
    const compute = comptime ExecContext.widenedCompute(dtype, "powScalar");
    var xx = try ctx.prepareAs(dtype, compute, x);
    defer xx.deinit();
    const xp = xx.tensor();
    var out = try ctx.empty(compute, xp.shape.slice());
    errdefer out.deinit();
    for (xp.dataConst(), out.data()) |value, *dst| dst.* = std.math.pow(f32, value, exponent);
    return ctx.storeAs(compute, dtype, out);
}

/// Elementwise select: `out[i] = cond[i] != 0 ? x[i] : y[i]` (all same shape).
/// `cond ? x : y` with a `.bool` or float condition (truthiness `!= 0`).
pub fn where(
    ctx: *ExecContext,
    comptime dtype: DType,
    comptime cond_dtype: DType,
    x: *const tensor.TensorOf(dtype),
    cond: *const tensor.TensorOf(cond_dtype),
    y: *const tensor.TensorOf(dtype),
) !tensor.TensorOf(dtype) {
    const compute = comptime ExecContext.widenedCompute(dtype, "where");
    if (!std.mem.eql(usize, x.shape.slice(), cond.shape.slice())) return tensor.TensorError.ShapeMismatch;
    try tensor.requireSameShapeOf(dtype, x, y);
    var xx = try ctx.prepareAs(dtype, compute, x);
    defer xx.deinit();
    var cc = try ctx.prepareContiguous(cond_dtype, cond);
    defer cc.deinit();
    var yy = try ctx.prepareAs(dtype, compute, y);
    defer yy.deinit();
    const xp = xx.tensor();
    var out = try ctx.empty(compute, xp.shape.slice());
    errdefer out.deinit();
    for (xp.dataConst(), cc.tensor().dataConst(), yy.tensor().dataConst(), out.data()) |xv, cv, yv, *dst| {
        dst.* = if (dtype_mod.isTruthy(cond_dtype, cv)) xv else yv;
    }
    return ctx.storeAs(compute, dtype, out);
}

/// Elementwise masked fill: `out[i] = mask[i] truthy ? value : x[i]`.
pub fn maskedFill(
    ctx: *ExecContext,
    comptime dtype: DType,
    comptime mask_dtype: DType,
    x: *const tensor.TensorOf(dtype),
    mask: *const tensor.TensorOf(mask_dtype),
    value: f32,
) !tensor.TensorOf(dtype) {
    const compute = comptime ExecContext.widenedCompute(dtype, "maskedFill");
    if (!std.mem.eql(usize, x.shape.slice(), mask.shape.slice())) return tensor.TensorError.ShapeMismatch;
    var xx = try ctx.prepareAs(dtype, compute, x);
    defer xx.deinit();
    var mm = try ctx.prepareContiguous(mask_dtype, mask);
    defer mm.deinit();
    const xp = xx.tensor();
    var out = try ctx.empty(compute, xp.shape.slice());
    errdefer out.deinit();
    for (xp.dataConst(), mm.tensor().dataConst(), out.data()) |xv, mv, *dst| {
        dst.* = if (dtype_mod.isTruthy(mask_dtype, mv)) value else xv;
    }
    return ctx.storeAs(compute, dtype, out);
}

/// Elementwise comparison mask: `out[i] = a[i] <op> b[i] ? 1.0 : 0.0`
/// (same shape only, like `where`; no broadcasting). NaN semantics are IEEE:
/// any comparison involving NaN is false — except `ne`, which is true — so
/// eq(NaN, NaN) = 0 and ne(NaN, x) = 1.
pub fn compare(
    ctx: *ExecContext,
    comptime dtype: DType,
    comptime op: CompareOp,
    a: *const tensor.TensorOf(dtype),
    b: *const tensor.TensorOf(dtype),
) !tensor.TensorOf(.bool) {
    if (comptime dtype_mod.isScalarIntegerOrBool(dtype)) return compareInt(ctx, dtype, op, a, b);
    // Floats: the f32 comparison; 16-bit inputs widen (`.widened` policy).
    const compute = comptime ExecContext.widenedCompute(dtype, "compare");
    try tensor.requireSameShapeOf(dtype, a, b);
    var aa = try ctx.prepareAs(dtype, compute, a);
    defer aa.deinit();
    var bb = try ctx.prepareAs(dtype, compute, b);
    defer bb.deinit();
    const ap = aa.tensor();
    var out = try ctx.empty(.bool, ap.shape.slice());
    errdefer out.deinit();
    for (ap.dataConst(), bb.tensor().dataConst(), out.data()) |av, bv, *dst| {
        dst.* = backend_ops.compareScalar(op, av, bv);
    }
    return out;
}

/// Elementwise comparison mask vs a scalar RHS: a `.bool` tensor. Same
/// IEEE NaN contract as `compare` (any comparison involving NaN is false
/// except `ne`).
/// The scalar a `compareScalar` takes: the exact element type for the
/// integer and bool dtypes, the accumulator (f32) for the float dtypes.
pub fn CompareScalar(comptime dtype: DType) type {
    return if (dtype_mod.isScalarIntegerOrBool(dtype)) dtype_mod.Scalar(dtype) else dtype_mod.Accumulator(dtype);
}

pub fn compareScalar(
    ctx: *ExecContext,
    comptime dtype: DType,
    comptime op: CompareOp,
    x: *const tensor.TensorOf(dtype),
    scalar_value: CompareScalar(dtype),
) !tensor.TensorOf(.bool) {
    if (comptime dtype_mod.isScalarIntegerOrBool(dtype)) return compareIntScalar(ctx, dtype, op, x, scalar_value);
    // Floats: the f32 comparison; 16-bit inputs widen (`.widened` policy).
    const compute = comptime ExecContext.widenedCompute(dtype, "compareScalar");
    var xx = try ctx.prepareAs(dtype, compute, x);
    defer xx.deinit();
    const xp = xx.tensor();
    var out = try ctx.empty(.bool, xp.shape.slice());
    errdefer out.deinit();
    for (xp.dataConst(), out.data()) |xv, *dst| {
        dst.* = backend_ops.compareScalar(op, xv, scalar_value);
    }
    return out;
}

fn intCompare(comptime op: CompareOp, a: anytype, b: anytype) bool {
    return switch (op) {
        .eq => a == b,
        .ne => a != b,
        .lt => a < b,
        .le => a <= b,
        .gt => a > b,
        .ge => a >= b,
    };
}

/// Integer elementwise comparison: exact at any magnitude (no float
/// bridge), `.bool` result. Same shape only.
fn compareInt(
    ctx: *ExecContext,
    comptime dtype: DType,
    comptime op: CompareOp,
    a: *const tensor.TensorOf(dtype),
    b: *const tensor.TensorOf(dtype),
) !tensor.TensorOf(.bool) {
    comptime {
        if (!dtype_mod.supportsIntMath(dtype)) @compileError("compare requires .f32 or an integer dtype");
    }
    if (!std.mem.eql(usize, a.shape.slice(), b.shape.slice())) return tensor.TensorError.ShapeMismatch;
    var aa = try ctx.prepareContiguous(dtype, a);
    defer aa.deinit();
    var bb = try ctx.prepareContiguous(dtype, b);
    defer bb.deinit();
    var out = try ctx.empty(.bool, a.shape.slice());
    errdefer out.deinit();
    for (aa.tensor().dataConst(), bb.tensor().dataConst(), out.data()) |av, bv, *dst| {
        dst.* = intCompare(op, av, bv);
    }
    return out;
}

/// Integer comparison against a scalar RHS (see `compare`).
fn compareIntScalar(
    ctx: *ExecContext,
    comptime dtype: DType,
    comptime op: CompareOp,
    x: *const tensor.TensorOf(dtype),
    scalar_value: dtype_mod.Scalar(dtype),
) !tensor.TensorOf(.bool) {
    comptime {
        if (!dtype_mod.supportsIntMath(dtype)) @compileError("compareScalar requires .f32 or an integer dtype");
    }
    var xx = try ctx.prepareContiguous(dtype, x);
    defer xx.deinit();
    var out = try ctx.empty(.bool, x.shape.slice());
    errdefer out.deinit();
    for (xx.tensor().dataConst(), out.data()) |xv, *dst| {
        dst.* = intCompare(op, xv, scalar_value);
    }
    return out;
}

pub const LogicalOp = enum { l_and, l_or, l_xor };

/// Elementwise logical AND/OR/XOR over truthiness (the repo-wide mask
/// convention shared with `where`/`maskedFill`; NaN is truthy): a `.bool`
/// tensor. Operands may mix `.bool` and float dtypes. Same shape only.
pub fn logical(
    ctx: *ExecContext,
    comptime op: LogicalOp,
    comptime a_dtype: DType,
    comptime b_dtype: DType,
    a: *const tensor.TensorOf(a_dtype),
    b: *const tensor.TensorOf(b_dtype),
) !tensor.TensorOf(.bool) {
    if (!std.mem.eql(usize, a.shape.slice(), b.shape.slice())) return tensor.TensorError.ShapeMismatch;
    var aa = try ctx.prepareContiguous(a_dtype, a);
    defer aa.deinit();
    var bb = try ctx.prepareContiguous(b_dtype, b);
    defer bb.deinit();
    var out = try ctx.empty(.bool, a.shape.slice());
    errdefer out.deinit();
    for (aa.tensor().dataConst(), bb.tensor().dataConst(), out.data()) |av, bv, *dst| {
        const at = dtype_mod.isTruthy(a_dtype, av);
        const bt = dtype_mod.isTruthy(b_dtype, bv);
        dst.* = switch (op) {
            .l_and => at and bt,
            .l_or => at or bt,
            .l_xor => at != bt,
        };
    }
    return out;
}

/// Elementwise logical NOT over truthiness (see `logical`).
pub fn logicalNot(ctx: *ExecContext, comptime dtype: DType, x: *const tensor.TensorOf(dtype)) !tensor.TensorOf(.bool) {
    var xx = try ctx.prepareContiguous(dtype, x);
    defer xx.deinit();
    var out = try ctx.empty(.bool, x.shape.slice());
    errdefer out.deinit();
    for (xx.tensor().dataConst(), out.data()) |xv, *dst| {
        dst.* = !dtype_mod.isTruthy(dtype, xv);
    }
    return out;
}

pub fn addScaledInPlace(ctx: *ExecContext, target: *Tensor, source: *const Tensor, scalar_value: f32) !void {
    try tensor.requireSameShape(target, source);
    if (!target.isContiguous()) return tensor.TensorError.UnsupportedView;

    var ss = try ctx.prepareContiguous(.f32, source);
    defer ss.deinit();
    ctx.enableNativeVectorPoolForWork(target.len(), parallel.vector_elementwise_len_threshold);
    kernels.addScaledSlice(target.data(), ss.tensor().dataConst(), scalar_value);
}

/// `target[..., i] += row_vector[i]` along the last axis `axis`, then `op`
/// when given (a fused bias + activation), in `target`'s storage.
pub fn addAxisVectorInPlace(ctx: *ExecContext, comptime rank: usize, comptime op: ?UnaryOp, target: *Tensor, row_vector: []const f32, comptime axis: usize) !void {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");

    const view = try target.rankView(rank);
    if (!target.isContiguous()) return tensor.TensorError.UnsupportedView;
    const axis_dim = view.shape[axis];
    if (row_vector.len != axis_dim) return tensor.TensorError.ShapeMismatch;
    if (productAfterAxis(rank, view.shape, axis) != 1) return tensor.TensorError.UnsupportedView;

    const rows = productBeforeAxis(rank, view.shape, axis);
    ctx.enableNativeVectorPoolForWork(target.len(), parallel.vector_elementwise_len_threshold);
    kernels.addRowVectorSlice(op, target.data(), row_vector, rows, axis_dim);
}

/// Shared validation for the channel-last per-channel row ops (PReLU,
/// channel affine): `x` contiguous with the channel axis innermost, params
/// rank-1 of that length. Returns `.{ rows, cols }`.
fn channelRowsCols(x: *const Tensor, param_len: usize) !struct { rows: usize, cols: usize } {
    const sh = x.shape.slice();
    if (sh.len == 0) return tensor.TensorError.InvalidShape;
    const cols = sh[sh.len - 1];
    if (cols == 0 or param_len != cols) return tensor.TensorError.ShapeMismatch;
    return .{ .rows = x.len() / cols, .cols = cols };
}

/// Per-channel PReLU over a channel-last tensor (any rank ≥ 1, channel axis
/// innermost): `y = x > 0 ? x : α[c]·x`, `α` rank-1 `[C]`. One fused pass
/// (the composed equivalent is relu + sub + mul + add — 4 passes and 3
/// transients); arithmetic is value-identical to that composition.
pub fn preluChannels(ctx: *ExecContext, x: *const Tensor, alpha: *const Tensor) !Tensor {
    const alpha_view = try alpha.rankView(1);
    var xx = try ctx.prepareContiguous(.f32, x);
    defer xx.deinit();
    var aa = try ctx.prepareContiguous(.f32, alpha);
    defer aa.deinit();
    const rc = try channelRowsCols(xx.tensor(), alpha_view.shape[0]);

    var out = try ctx.empty(.f32, xx.tensor().shape.slice());
    errdefer out.deinit();
    ctx.enableNativeVectorPoolForWork(x.len(), parallel.vector_elementwise_len_threshold);
    kernels.preluChannelsInto(ctx.pc(), out.data(), xx.tensor().dataConst(), aa.tensor().dataConst(), rc.rows, rc.cols);
    return out;
}

/// PReLU input-VJP: `gx = x > 0 ? gy : α[c]·gy`.
pub fn preluChannelsBackwardInput(ctx: *ExecContext, gy: *const Tensor, x: *const Tensor, alpha: *const Tensor) !Tensor {
    try tensor.requireSameShape(gy, x);
    const alpha_view = try alpha.rankView(1);
    var gg = try ctx.prepareContiguous(.f32, gy);
    defer gg.deinit();
    var xx = try ctx.prepareContiguous(.f32, x);
    defer xx.deinit();
    var aa = try ctx.prepareContiguous(.f32, alpha);
    defer aa.deinit();
    const rc = try channelRowsCols(xx.tensor(), alpha_view.shape[0]);

    var out = try ctx.empty(.f32, xx.tensor().shape.slice());
    errdefer out.deinit();
    ctx.enableNativeVectorPoolForWork(x.len(), parallel.vector_elementwise_len_threshold);
    kernels.preluChannelsBackwardInputInto(ctx.pc(), out.data(), gg.tensor().dataConst(), xx.tensor().dataConst(), aa.tensor().dataConst(), rc.rows, rc.cols);
    return out;
}

/// PReLU slope-VJP: `gα[c] = Σ_rows gy·min(x, 0)` → rank-1 `[C]`.
pub fn preluChannelsBackwardAlpha(ctx: *ExecContext, gy: *const Tensor, x: *const Tensor, channels: usize) !Tensor {
    try tensor.requireSameShape(gy, x);
    var gg = try ctx.prepareContiguous(.f32, gy);
    defer gg.deinit();
    var xx = try ctx.prepareContiguous(.f32, x);
    defer xx.deinit();
    const rc = try channelRowsCols(xx.tensor(), channels);

    var out = try ctx.empty(.f32, .{rc.cols});
    errdefer out.deinit();
    kernels.preluChannelsBackwardAlphaInto(ctx.pc(), out.data(), gg.tensor().dataConst(), xx.tensor().dataConst(), rc.rows, rc.cols);
    return out;
}

/// Per-channel affine over a channel-last tensor (any rank ≥ 1, channel axis
/// innermost): `y = x·scale[c] + shift[c]` — the frozen-stats inference
/// BatchNorm as ONE fused pass. Mul-then-add (never contracted to fma), so
/// the values equal the two-pass broadcast mul + add composition bitwise.
/// A null `shift` degrades to the per-channel scale `y = x·scale[c]`.
pub fn channelAffine(ctx: *ExecContext, x: *const Tensor, scale_vec: *const Tensor, shift_vec: ?*const Tensor) !Tensor {
    const scale_view = try scale_vec.rankView(1);
    if (shift_vec) |t| {
        const shift_view = try t.rankView(1);
        if (scale_view.shape[0] != shift_view.shape[0]) return tensor.TensorError.ShapeMismatch;
    }
    var xx = try ctx.prepareContiguous(.f32, x);
    defer xx.deinit();
    var ss = try ctx.prepareContiguous(.f32, scale_vec);
    defer ss.deinit();
    var tt: ?ExecContext.PreparedTensor = if (shift_vec) |t| try ctx.prepareContiguous(.f32, t) else null;
    defer if (tt) |*p| p.deinit();
    const rc = try channelRowsCols(xx.tensor(), scale_view.shape[0]);

    var out = try ctx.empty(.f32, xx.tensor().shape.slice());
    errdefer out.deinit();
    ctx.enableNativeVectorPoolForWork(x.len(), parallel.vector_elementwise_len_threshold);
    kernels.channelAffineInto(ctx.pc(), out.data(), xx.tensor().dataConst(), ss.tensor().dataConst(), if (tt) |*p| p.tensor().dataConst() else null, rc.rows, rc.cols);
    return out;
}

/// Inverted dropout forward: element i keeps `x[i] / (1-p)` iff the
/// 53-bit uniform of `rng.at(seed, i)` is < 1-p, else 0. The mask is never
/// materialized — it is a counter-based function of (seed, element index),
/// so the backward kernel (and a checkpoint recompute) regenerates it
/// exactly. Requires `0 <= p < 1`.
pub fn dropoutForward(ctx: *ExecContext, x: *const Tensor, p: f32, seed: u64) !Tensor {
    return dropoutApply(ctx, x, p, seed);
}

/// Inverted dropout VJP: the gradient passes through kept elements scaled
/// by 1/(1-p) and is 0 at dropped ones — the identical (seed, i) mask and
/// arithmetic as `dropoutForward`, applied to `gy`.
pub fn dropoutBackward(ctx: *ExecContext, gy: *const Tensor, p: f32, seed: u64) !Tensor {
    return dropoutApply(ctx, gy, p, seed);
}

fn dropoutApply(ctx: *ExecContext, x: *const Tensor, p: f32, seed: u64) !Tensor {
    if (!(p >= 0 and p < 1)) return tensor.TensorError.InvalidShape;

    var xx = try ctx.prepareContiguous(.f32, x);
    defer xx.deinit();
    const xp = xx.tensor();
    const input = xp.dataConst();

    var out = try ctx.empty(.f32, xp.shape.slice());
    errdefer out.deinit();

    const base_task: DropoutRangeTask = .{
        .input = input,
        .output = out.data(),
        .keep_cutoff = dropoutKeepCutoff(p),
        .scale = 1.0 / (1.0 - p),
        .seed = seed,
        .start = 0,
        .end = input.len,
    };
    // Counter-based RNG: any flat element range computes independently, so
    // the split is bitwise neutral (same per-element mask and arithmetic
    // for any thread count).
    if (input.len >= parallel.vector_elementwise_len_threshold) {
        if (ctx.dispatchRange(DropoutRangeTask, "start", "end", base_task, input.len, runDropoutRangeTask)) {
            return out;
        }
    }

    dropoutRange(base_task);
    return out;
}

/// Elementwise `op(x)`. One f32 kernel; 16-bit inputs follow the
/// `.widened` policy.
pub fn unary(ctx: *ExecContext, comptime dtype: DType, comptime op: UnaryOp, x: *const tensor.TensorOf(dtype)) !tensor.TensorOf(dtype) {
    const compute = comptime ExecContext.widenedCompute(dtype, "unary");
    var xx = try ctx.prepareAs(dtype, compute, x);
    defer xx.deinit();

    const xp = xx.tensor();
    var out = try ctx.empty(compute, xp.shape.slice());
    errdefer out.deinit();
    ctx.enableNativeVectorPoolForWork(xp.len(), parallel.vector_elementwise_len_threshold);
    kernels.unaryContiguousIntoUnchecked(ctx.pc(), op, &out, xp, xp.len());
    return ctx.storeAs(compute, dtype, out);
}

pub fn leakyRelu(ctx: *ExecContext, comptime dtype: DType, x: *const tensor.TensorOf(dtype), negative_slope: f32) !tensor.TensorOf(dtype) {
    const compute = comptime ExecContext.widenedCompute(dtype, "leakyRelu");
    var xx = try ctx.prepareAs(dtype, compute, x);
    defer xx.deinit();

    const xp = xx.tensor();
    var out = try ctx.empty(compute, xp.shape.slice());
    errdefer out.deinit();
    ctx.enableNativeVectorPoolForWork(xp.len(), parallel.vector_elementwise_len_threshold);
    kernels.leakyReluContiguousIntoUnchecked(ctx.pc(), &out, xp, xp.len(), negative_slope);
    return ctx.storeAs(compute, dtype, out);
}

/// Per-channel Snake activation over `[rows, cols]` rows (the DAC codec op):
/// `y[t,c] = x[t,c] + inv_b[c] * sin(alpha[c] * x[t,c])^2`. The caller
/// precomputes `inv_b = 1/(alpha + 1e-9)` at weight-load time (the reference's
/// convention) — the epsilon is deliberately NOT folded into the kernel.
pub fn snakeRows(ctx: *ExecContext, x: *const Tensor, alpha: *const Tensor, inv_b: *const Tensor) !Tensor {
    const source = try x.rankView(2);
    const rows = source.shape[0];
    const cols = source.shape[1];
    const alpha_view = try alpha.rankView(1);
    if (alpha_view.shape[0] != cols) return tensor.TensorError.ShapeMismatch;
    const inv_view = try inv_b.rankView(1);
    if (inv_view.shape[0] != cols) return tensor.TensorError.ShapeMismatch;

    var xx = try ctx.prepareContiguous(.f32, x);
    defer xx.deinit();
    var aa = try ctx.prepareContiguous(.f32, alpha);
    defer aa.deinit();
    var bb = try ctx.prepareContiguous(.f32, inv_b);
    defer bb.deinit();

    var out = try ctx.empty(.f32, .{ rows, cols });
    errdefer out.deinit();
    ctx.enableNativeVectorPoolForWork(xx.tensor().len(), parallel.vector_elementwise_len_threshold);
    kernels.snakeInto(ctx.pc(), &out, xx.tensor(), aa.tensor().dataConst(), bb.tensor().dataConst(), rows, cols);
    return out;
}

/// VJP of snakeRows wrt the input:
/// `gx[t,c] = gy[t,c] * (1 + inv_b[c]*alpha[c]*sin(2*alpha[c]*x[t,c]))`.
pub fn snakeRowsBackwardInput(ctx: *ExecContext, x: *const Tensor, gy: *const Tensor, alpha: *const Tensor, inv_b: *const Tensor) !Tensor {
    const source = try x.rankView(2);
    const rows = source.shape[0];
    const cols = source.shape[1];
    try tensor.requireSameShape(x, gy);
    const alpha_view = try alpha.rankView(1);
    if (alpha_view.shape[0] != cols) return tensor.TensorError.ShapeMismatch;
    const inv_view = try inv_b.rankView(1);
    if (inv_view.shape[0] != cols) return tensor.TensorError.ShapeMismatch;

    var xx = try ctx.prepareContiguous(.f32, x);
    defer xx.deinit();
    var gg = try ctx.prepareContiguous(.f32, gy);
    defer gg.deinit();
    var aa = try ctx.prepareContiguous(.f32, alpha);
    defer aa.deinit();
    var bb = try ctx.prepareContiguous(.f32, inv_b);
    defer bb.deinit();

    var out = try ctx.empty(.f32, .{ rows, cols });
    errdefer out.deinit();
    ctx.enableNativeVectorPoolForWork(xx.tensor().len(), parallel.vector_elementwise_len_threshold);
    kernels.snakeBackwardInputInto(ctx.pc(), &out, xx.tensor(), gg.tensor(), aa.tensor().dataConst(), bb.tensor().dataConst(), rows, cols);
    return out;
}

/// The two per-channel snake parameter gradients, always filled together
/// (one traversal computes both).
pub const SnakeBackwardParamsResult = struct {
    alpha: Tensor,
    inv_b: Tensor,

    pub fn deinit(self: *SnakeBackwardParamsResult) void {
        self.alpha.deinit();
        self.inv_b.deinit();
        self.* = undefined;
    }
};

/// VJPs of snakeRows wrt the per-channel parameters:
/// `galpha[c] = Σ_t gy[t,c]*inv_b[c]*x[t,c]*sin(2*alpha[c]*x[t,c])`,
/// `ginv_b[c] = Σ_t gy[t,c]*sin(alpha[c]*x[t,c])^2`. `alpha` and `inv_b` are
/// independent inputs at this level (the caller ties `inv_b = 1/(alpha+1e-9)`
/// at load time); both gradients are computed in a single kernel pass.
pub fn snakeRowsBackwardParams(ctx: *ExecContext, x: *const Tensor, gy: *const Tensor, alpha: *const Tensor, inv_b: *const Tensor) !SnakeBackwardParamsResult {
    const source = try x.rankView(2);
    const rows = source.shape[0];
    const cols = source.shape[1];
    try tensor.requireSameShape(x, gy);
    const alpha_view = try alpha.rankView(1);
    if (alpha_view.shape[0] != cols) return tensor.TensorError.ShapeMismatch;
    const inv_view = try inv_b.rankView(1);
    if (inv_view.shape[0] != cols) return tensor.TensorError.ShapeMismatch;

    var xx = try ctx.prepareContiguous(.f32, x);
    defer xx.deinit();
    var gg = try ctx.prepareContiguous(.f32, gy);
    defer gg.deinit();
    var aa = try ctx.prepareContiguous(.f32, alpha);
    defer aa.deinit();
    var bb = try ctx.prepareContiguous(.f32, inv_b);
    defer bb.deinit();

    var galpha = try ctx.empty(.f32, .{cols});
    errdefer galpha.deinit();
    var ginv_b = try ctx.empty(.f32, .{cols});
    errdefer ginv_b.deinit();
    ctx.enableNativeVectorPoolForWork(xx.tensor().len(), parallel.vector_elementwise_len_threshold);
    kernels.snakeBackwardParamsInto(ctx.pc(), &galpha, &ginv_b, xx.tensor(), gg.tensor(), aa.tensor().dataConst(), bb.tensor().dataConst(), rows, cols);
    return .{ .alpha = galpha, .inv_b = ginv_b };
}

/// `cap * tanh(x / cap)`, the logit softcap (Gemma's final-logit cap, the
/// nanochat GPT's). One f32 kernel; 16-bit inputs follow the `.widened`
/// policy.
pub fn softcap(ctx: *ExecContext, comptime dtype: DType, x: *const tensor.TensorOf(dtype), cap: f32) !tensor.TensorOf(dtype) {
    if (!(cap > 0)) return tensor.TensorError.InvalidShape;
    const compute = comptime ExecContext.widenedCompute(dtype, "softcap");
    var xx = try ctx.prepareAs(dtype, compute, x);
    defer xx.deinit();

    const xp = xx.tensor();
    var out = try ctx.empty(compute, xp.shape.slice());
    errdefer out.deinit();
    ctx.enableNativeVectorPoolForWork(xp.len(), parallel.vector_elementwise_len_threshold);
    kernels.softcapContiguousIntoUnchecked(ctx.pc(), &out, xp, xp.len(), cap);
    return ctx.storeAs(compute, dtype, out);
}

pub fn clamp(ctx: *ExecContext, comptime dtype: DType, x: *const tensor.TensorOf(dtype), min_value: f32, max_value: f32) !tensor.TensorOf(dtype) {
    if (min_value > max_value) return tensor.TensorError.InvalidShape;
    const compute = comptime ExecContext.widenedCompute(dtype, "clamp");

    var xx = try ctx.prepareAs(dtype, compute, x);
    defer xx.deinit();

    const xp = xx.tensor();
    var out = try ctx.empty(compute, xp.shape.slice());
    errdefer out.deinit();
    ctx.enableNativeVectorPoolForWork(xp.len(), parallel.vector_elementwise_len_threshold);
    kernels.clampContiguousIntoUnchecked(ctx.pc(), &out, xp, xp.len(), min_value, max_value);
    return ctx.storeAs(compute, dtype, out);
}

/// Sum-reduce `x` down to `target_shape` (array/tuple or slice), the
/// broadcast VJP. An array or tuple shape carries its rank at comptime and
/// skips the target-rank dispatch.
pub fn reduceBroadcast(ctx: *ExecContext, x: *const Tensor, target_shape: anytype) !Tensor {
    const maybe_rank = comptime exec_runtime.shapeComptimeRank(@TypeOf(target_shape));
    if (comptime maybe_rank != null) {
        const target_rank = comptime maybe_rank.?;
        const shape_array = exec_runtime.shapeArray(target_rank, target_shape);
        return dispatchRank(reduceBroadcastSourceDispatched, x.shape.len, .{ ctx, target_rank, x, shape_array });
    }
    return dispatchRank(reduceBroadcastTargetDispatched, target_shape.len, .{ ctx, x, target_shape });
}

fn reduceBroadcastTargetDispatched(
    comptime target_rank: usize,
    ctx: *ExecContext,
    x: *const Tensor,
    target_shape: []const usize,
) !Tensor {
    const shape_array = try shapeArrayFromSlice(target_rank, target_shape);
    return dispatchRank(reduceBroadcastSourceDispatched, x.shape.len, .{ ctx, target_rank, x, shape_array });
}

fn reduceBroadcastSourceDispatched(
    comptime source_rank: usize,
    ctx: *ExecContext,
    comptime target_rank: usize,
    x: *const Tensor,
    target_shape: [target_rank]usize,
) !Tensor {
    return reduceBroadcastFromRankToRank(ctx, source_rank, target_rank, x, target_shape);
}

fn reduceBroadcastFromRankToRank(
    ctx: *ExecContext,
    comptime source_rank: usize,
    comptime target_rank: usize,
    x: *const Tensor,
    target_shape: [target_rank]usize,
) !Tensor {
    if (target_rank > source_rank) return tensor.TensorError.ShapeMismatch;
    const source = try x.rankView(source_rank);
    try validateBroadcastRank(target_rank, source_rank, target_shape, source.shape);
    if (source_rank == target_rank and std.mem.eql(usize, target_shape[0..], source.shape[0..])) {
        return x.cloneView();
    }

    var xx = try ctx.prepareContiguous(.f32, x);
    defer xx.deinit();
    const xp = xx.tensor();

    var out = try ctx.zeros(.f32, target_shape);
    errdefer out.deinit();

    const xd = xp.dataConst();
    const od = out.data();

    if (target_rank == 1 and target_shape[0] == 1) {
        var total: f32 = 0;
        for (xd) |value| total += value;
        od[0] = total;
        return out;
    }

    if (isExactSuffixRank(target_rank, source_rank, target_shape, source.shape)) {
        const inner = tensor.elementCountArrayAssumeValid(target_rank, target_shape);
        var base: usize = 0;
        while (base < xd.len) : (base += inner) {
            for (0..inner) |j| {
                od[j] += xd[base + j];
            }
        }
        return out;
    }

    const out_strides = contiguousStridesArray(target_rank, target_shape);
    const rank_diff = source_rank - target_rank;
    for (xd, 0..) |value, linear| {
        var remainder = linear;
        var out_linear: usize = 0;
        comptime var dim = source_rank;
        inline while (dim > 0) {
            dim -= 1;
            const coord = remainder % source.shape[dim];
            remainder /= source.shape[dim];

            if (dim >= rank_diff) {
                const target_dim = dim - rank_diff;
                if (target_shape[target_dim] == source.shape[dim]) {
                    out_linear += coord * out_strides[target_dim];
                }
            }
        }
        od[out_linear] += value;
    }

    return out;
}

/// Consume `target` and return `op(target, other)` (same shape); ownership
/// as `takeScale`.
pub fn takeElementwise(
    ctx: *ExecContext,
    comptime op: ElementwiseOp,
    target: *Tensor,
    other: *const Tensor,
) !Tensor {
    try tensor.requireSameShape(target, other);

    if (target.canTakeInPlace()) {
        try elementwiseInPlace(ctx, op, target, other);
        return takeTensor(target);
    }

    var result = try elementwise(ctx, .f32, op, target, other);
    errdefer result.deinit();
    return discardTakenInput(target, result);
}

fn elementwiseRank(
    ctx: *ExecContext,
    comptime rank: usize,
    comptime op: ElementwiseOp,
    a: *const Tensor,
    b: *const Tensor,
) !Tensor {
    const shape = try requireSameRankShape(rank, a, b);
    var out = try ctx.empty(.f32, shape);
    errdefer out.deinit();
    try elementwiseRankInto(ctx, rank, op, &out, a, b, shape);
    return out;
}

fn elementwiseRankTyped(
    ctx: *ExecContext,
    comptime dtype: DType,
    comptime rank: usize,
    comptime op: ElementwiseOp,
    a: *const tensor.TensorOf(dtype),
    b: *const tensor.TensorOf(dtype),
) !tensor.TensorOf(dtype_mod.outputDType(.pointwise, dtype)) {
    if (comptime dtype == .f32) return elementwiseRank(ctx, rank, op, a, b);
    if (comptime dtype_mod.supportsIntMath(dtype)) {
        // Integer pointwise: a plain exec loop (the stats.zig precedent —
        // no backend kernel; integers are never the hot path). Wrapping
        // two's-complement arithmetic; `div` is a compile error (integer
        // division is explicit: intDiv).
        const shape = try requireSameRankShapeOf(dtype, rank, a, b);
        var aa = try ctx.prepareContiguous(dtype, a);
        defer aa.deinit();
        var bb = try ctx.prepareContiguous(dtype, b);
        defer bb.deinit();
        var out = try ctx.empty(dtype, shape);
        errdefer out.deinit();
        for (out.data(), aa.tensor().dataConst(), bb.tensor().dataConst()) |*o, x, y| {
            o.* = switch (op) {
                .add => x +% y,
                .sub => x -% y,
                .mul => x *% y,
                .max => @max(x, y),
                .min => @min(x, y),
                .div => @compileError("integer `div` is explicit: use divTrunc/divFloor"),
            };
        }
        return out;
    }
    comptime ensureForwardFloatMath(dtype);
    if (comptime (op == .max or op == .min)) {
        // No typed max/min kernel: the `.widened` policy.
        const compute = comptime ExecContext.widenedCompute(dtype, "max/min");
        var aa = try ctx.prepareAs(dtype, compute, a);
        defer aa.deinit();
        var bb = try ctx.prepareAs(dtype, compute, b);
        defer bb.deinit();
        var out = try elementwiseRank(ctx, rank, op, aa.tensor(), bb.tensor());
        errdefer out.deinit();
        return ctx.storeAs(compute, dtype, out);
    }
    const output_dtype = comptime dtype_mod.outputDType(.pointwise, dtype);

    const shape = try requireSameRankShapeOf(dtype, rank, a, b);
    var aa = try ctx.prepareContiguous(dtype, a);
    defer aa.deinit();
    var bb = try ctx.prepareContiguous(dtype, b);
    defer bb.deinit();

    var out = try ctx.empty(output_dtype, shape);
    errdefer out.deinit();
    ctx.enableNativeVectorPoolForWork(out.len(), parallel.vector_elementwise_len_threshold);
    kernels.elementwiseContiguousIntoTyped(ctx.pc(), dtype, op, &out, aa.tensor(), bb.tensor(), out.len());
    return out;
}

pub const IntDivMode = enum { trunc, floor };

/// Integer division as an explicit op (torch's `/` silently promotes
/// integers to float — a documented divergence: Fucina keeps promotion
/// explicit). `.trunc` rounds toward zero (C semantics), `.floor` toward
/// negative infinity (Python's //). A zero divisor is
/// `error.DivisionByZero`.
pub fn intDiv(
    ctx: *ExecContext,
    comptime dtype: DType,
    comptime rank: usize,
    comptime mode: IntDivMode,
    a: *const tensor.TensorOf(dtype),
    b: *const tensor.TensorOf(dtype),
) !tensor.TensorOf(dtype) {
    comptime {
        if (!dtype_mod.supportsIntMath(dtype)) @compileError("intDiv requires an integer dtype");
    }
    const shape = try requireSameRankShapeOf(dtype, rank, a, b);
    var aa = try ctx.prepareContiguous(dtype, a);
    defer aa.deinit();
    var bb = try ctx.prepareContiguous(dtype, b);
    defer bb.deinit();
    var out = try ctx.empty(dtype, shape);
    errdefer out.deinit();
    const signed = comptime @typeInfo(dtype_mod.Scalar(dtype)).int.signedness == .signed;
    for (out.data(), aa.tensor().dataConst(), bb.tensor().dataConst()) |*o, x, y| {
        if (y == 0) return tensor.TensorError.DivisionByZero;
        if (comptime signed) {
            // minInt / -1 overflows the two's-complement range: wrap to
            // minInt (consistent with the wrapping +%/-%/*% contract).
            if (x == std.math.minInt(dtype_mod.Scalar(dtype)) and y == -1) {
                o.* = x;
                continue;
            }
        }
        o.* = switch (mode) {
            .trunc => @divTrunc(x, y),
            .floor => @divFloor(x, y),
        };
    }
    return out;
}

/// `intDiv` with `.trunc` (C semantics).
pub fn divTrunc(ctx: *ExecContext, comptime dtype: DType, comptime rank: usize, a: *const tensor.TensorOf(dtype), b: *const tensor.TensorOf(dtype)) !tensor.TensorOf(dtype) {
    return intDiv(ctx, dtype, rank, .trunc, a, b);
}

/// `intDiv` with `.floor` (Python's `//`).
pub fn divFloor(ctx: *ExecContext, comptime dtype: DType, comptime rank: usize, a: *const tensor.TensorOf(dtype), b: *const tensor.TensorOf(dtype)) !tensor.TensorOf(dtype) {
    return intDiv(ctx, dtype, rank, .floor, a, b);
}

pub const IntModMode = enum { rem, mod };

/// Integer remainder as an explicit op, the modulus counterpart of
/// `intDiv`: `.rem` pairs `divTrunc` (result takes the sign of the
/// dividend, C's `%`), `.mod` pairs `divFloor` (result takes the sign of
/// the divisor, Python's `%` / numpy). A zero divisor is
/// `error.DivisionByZero`; minInt % -1 is 0 (the wrapping-div contract).
pub fn intMod(
    ctx: *ExecContext,
    comptime dtype: DType,
    comptime rank: usize,
    comptime mode: IntModMode,
    a: *const tensor.TensorOf(dtype),
    b: *const tensor.TensorOf(dtype),
) !tensor.TensorOf(dtype) {
    comptime {
        if (!dtype_mod.supportsIntMath(dtype)) @compileError("intMod requires an integer dtype");
    }
    const shape = try requireSameRankShapeOf(dtype, rank, a, b);
    var aa = try ctx.prepareContiguous(dtype, a);
    defer aa.deinit();
    var bb = try ctx.prepareContiguous(dtype, b);
    defer bb.deinit();
    var out = try ctx.empty(dtype, shape);
    errdefer out.deinit();
    const signed = comptime @typeInfo(dtype_mod.Scalar(dtype)).int.signedness == .signed;
    for (out.data(), aa.tensor().dataConst(), bb.tensor().dataConst()) |*o, x, y| {
        if (y == 0) return tensor.TensorError.DivisionByZero;
        if (comptime signed) {
            // The quotient minInt / -1 overflows, but its remainder is
            // exactly 0; short-circuit so the builtins never see the
            // overflowing division.
            if (x == std.math.minInt(dtype_mod.Scalar(dtype)) and y == -1) {
                o.* = 0;
                continue;
            }
        }
        o.* = switch (mode) {
            .rem => @rem(x, y),
            .mod => @mod(x, y),
        };
    }
    return out;
}

/// `intMod` with `.rem` (C's `%`).
pub fn rem(ctx: *ExecContext, comptime dtype: DType, comptime rank: usize, a: *const tensor.TensorOf(dtype), b: *const tensor.TensorOf(dtype)) !tensor.TensorOf(dtype) {
    return intMod(ctx, dtype, rank, .rem, a, b);
}

/// `intMod` with `.mod` (Python's `%`).
pub fn mod(ctx: *ExecContext, comptime dtype: DType, comptime rank: usize, a: *const tensor.TensorOf(dtype), b: *const tensor.TensorOf(dtype)) !tensor.TensorOf(dtype) {
    return intMod(ctx, dtype, rank, .mod, a, b);
}

pub const IntBitwiseOp = enum { b_and, b_or, b_xor };

/// Bitwise combinators on the integer dtypes (two's-complement bit
/// patterns; distinct from the `.bool` truthiness `logicalAnd/Or/Xor`).
pub fn bitwise(
    ctx: *ExecContext,
    comptime dtype: DType,
    comptime rank: usize,
    comptime op: IntBitwiseOp,
    a: *const tensor.TensorOf(dtype),
    b: *const tensor.TensorOf(dtype),
) !tensor.TensorOf(dtype) {
    comptime {
        if (!dtype_mod.supportsIntMath(dtype)) @compileError("bitwise requires an integer dtype");
    }
    const shape = try requireSameRankShapeOf(dtype, rank, a, b);
    var aa = try ctx.prepareContiguous(dtype, a);
    defer aa.deinit();
    var bb = try ctx.prepareContiguous(dtype, b);
    defer bb.deinit();
    var out = try ctx.empty(dtype, shape);
    errdefer out.deinit();
    for (out.data(), aa.tensor().dataConst(), bb.tensor().dataConst()) |*o, x, y| {
        o.* = switch (op) {
            .b_and => x & y,
            .b_or => x | y,
            .b_xor => x ^ y,
        };
    }
    return out;
}

fn elementwiseRankInto(
    ctx: *ExecContext,
    comptime rank: usize,
    comptime op: ElementwiseOp,
    out: *Tensor,
    a: *const Tensor,
    b: *const Tensor,
    shape: [rank]usize,
) !void {
    const av = try a.rankView(rank);
    const bv = try b.rankView(rank);
    const ov = try out.rankView(rank);
    if (!std.mem.eql(usize, ov.shape[0..], shape[0..])) return tensor.TensorError.ShapeMismatch;

    const len = tensor.elementCountArrayAssumeValid(rank, shape);
    if (ov.isContiguous() and av.isContiguous() and bv.isContiguous()) {
        return backendElementwiseContiguousUnchecked(ctx, op, out, a, b, len);
    }

    if (try tryTailBroadcastElementwise(op, out, a, b)) {
        return;
    }

    var aa = try ctx.prepareContiguous(.f32, a);
    defer aa.deinit();
    var bb = try ctx.prepareContiguous(.f32, b);
    defer bb.deinit();
    return backendElementwiseContiguousUnchecked(ctx, op, out, aa.tensor(), bb.tensor(), len);
}

fn elementwiseInto(
    ctx: *ExecContext,
    comptime op: ElementwiseOp,
    out: *Tensor,
    a: *const Tensor,
    b: *const Tensor,
) !void {
    try tensor.requireSameShape(out, a);

    if (a.isContiguous() and b.isContiguous()) {
        return backendElementwiseContiguousUnchecked(ctx, op, out, a, b, out.len());
    }

    if (try tryTailBroadcastElementwise(op, out, a, b)) {
        return;
    }

    var aa = try ctx.prepareContiguous(.f32, a);
    defer aa.deinit();
    var bb = try ctx.prepareContiguous(.f32, b);
    defer bb.deinit();
    return backendElementwiseContiguousUnchecked(ctx, op, out, aa.tensor(), bb.tensor(), out.len());
}

/// `target = op(target, other)` in `target`'s storage. `target` must be
/// contiguous; `other` is the same shape, or a tail-broadcast row of it.
pub fn elementwiseInPlace(
    ctx: *ExecContext,
    comptime op: ElementwiseOp,
    target: *Tensor,
    other: *const Tensor,
) !void {
    try tensor.requireSameShape(target, other);
    if (!target.isContiguous()) return tensor.TensorError.UnsupportedView;

    if (other.isContiguous()) {
        if (target.len() <= small_in_place_elementwise_len) {
            return elementwiseContiguousInPlace(op, target.data(), other.dataConst());
        }
        return backendElementwiseContiguousUnchecked(ctx, op, target, target, other, target.len());
    }

    if (tryTailBroadcastElementwiseInPlace(op, target, other)) {
        return;
    }

    var materialized = try ctx.materialize(.f32, other);
    defer materialized.deinit();
    return backendElementwiseContiguousUnchecked(ctx, op, target, target, &materialized, target.len());
}

fn backendElementwiseContiguousUnchecked(
    ctx: *ExecContext,
    comptime op: ElementwiseOp,
    out: *Tensor,
    a: *const Tensor,
    b: *const Tensor,
    len: usize,
) void {
    ctx.enableNativeVectorPoolForWork(len, parallel.vector_elementwise_len_threshold);
    return switch (op) {
        .add => kernels.addContiguousIntoUnchecked(ctx.pc(), out, a, b, len),
        .sub => kernels.subContiguousIntoUnchecked(ctx.pc(), out, a, b, len),
        .mul => kernels.mulContiguousIntoUnchecked(ctx.pc(), out, a, b, len),
        .div => kernels.divContiguousIntoUnchecked(ctx.pc(), out, a, b, len),
        .max => kernels.maximumContiguousIntoUnchecked(ctx.pc(), out, a, b, len),
        .min => kernels.minimumContiguousIntoUnchecked(ctx.pc(), out, a, b, len),
    };
}

const small_in_place_elementwise_len = 2048;

fn takeTensor(target: *Tensor) Tensor {
    const out = target.*;
    target.* = undefined;
    return out;
}

fn discardTakenInput(target: *Tensor, result: Tensor) Tensor {
    target.deinit();
    target.* = undefined;
    return result;
}
