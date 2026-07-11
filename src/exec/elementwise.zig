const std = @import("std");
const backend_mod = @import("../backend.zig");
const tensor = @import("../tensor.zig");

const parallel = @import("../parallel.zig");
const dtype_mod = @import("../dtype.zig");
const exec_row_ops = @import("row_ops.zig");
const exec_shape = @import("shape.zig");
const Runtime = @import("runtime.zig").Runtime;
const backend_ops = backend_mod.ops;
const DType = tensor.DType;
const GatedOp = backend_mod.ops.GatedOp;
const UnaryOp = backend_mod.ops.UnaryOp;

const dispatchRank = exec_shape.dispatchRank;
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

pub fn add(rt: *Runtime, a: *const Tensor, b: *const Tensor) !Tensor {
    return elementwiseRuntimeRank(rt, .add, a, b);
}

pub fn sub(rt: *Runtime, a: *const Tensor, b: *const Tensor) !Tensor {
    return elementwiseRuntimeRank(rt, .sub, a, b);
}

pub fn mul(rt: *Runtime, a: *const Tensor, b: *const Tensor) !Tensor {
    return elementwiseRuntimeRank(rt, .mul, a, b);
}

pub fn div(rt: *Runtime, a: *const Tensor, b: *const Tensor) !Tensor {
    return elementwiseRuntimeRank(rt, .div, a, b);
}

pub fn addRank(rt: *Runtime, comptime rank: usize, a: *const Tensor, b: *const Tensor) !Tensor {
    return elementwiseRank(rt, rank, .add, a, b);
}

pub fn addRankTyped(
    rt: *Runtime,
    comptime dtype: DType,
    comptime rank: usize,
    a: *const tensor.TensorOf(dtype),
    b: *const tensor.TensorOf(dtype),
) !tensor.TensorOf(dtype_mod.outputDType(.pointwise, dtype)) {
    return elementwiseRankTyped(rt, dtype, rank, .add, a, b);
}

pub fn subRank(rt: *Runtime, comptime rank: usize, a: *const Tensor, b: *const Tensor) !Tensor {
    return elementwiseRank(rt, rank, .sub, a, b);
}

pub fn subRankTyped(
    rt: *Runtime,
    comptime dtype: DType,
    comptime rank: usize,
    a: *const tensor.TensorOf(dtype),
    b: *const tensor.TensorOf(dtype),
) !tensor.TensorOf(dtype_mod.outputDType(.pointwise, dtype)) {
    return elementwiseRankTyped(rt, dtype, rank, .sub, a, b);
}

pub fn mulRank(rt: *Runtime, comptime rank: usize, a: *const Tensor, b: *const Tensor) !Tensor {
    return elementwiseRank(rt, rank, .mul, a, b);
}

pub fn mulRankTyped(
    rt: *Runtime,
    comptime dtype: DType,
    comptime rank: usize,
    a: *const tensor.TensorOf(dtype),
    b: *const tensor.TensorOf(dtype),
) !tensor.TensorOf(dtype_mod.outputDType(.pointwise, dtype)) {
    return elementwiseRankTyped(rt, dtype, rank, .mul, a, b);
}

pub fn divRank(rt: *Runtime, comptime rank: usize, a: *const Tensor, b: *const Tensor) !Tensor {
    return elementwiseRank(rt, rank, .div, a, b);
}

pub fn divRankTyped(
    rt: *Runtime,
    comptime dtype: DType,
    comptime rank: usize,
    a: *const tensor.TensorOf(dtype),
    b: *const tensor.TensorOf(dtype),
) !tensor.TensorOf(dtype_mod.outputDType(.pointwise, dtype)) {
    return elementwiseRankTyped(rt, dtype, rank, .div, a, b);
}

pub fn maxRankTyped(
    rt: *Runtime,
    comptime dtype: DType,
    comptime rank: usize,
    a: *const tensor.TensorOf(dtype),
    b: *const tensor.TensorOf(dtype),
) !tensor.TensorOf(dtype_mod.outputDType(.pointwise, dtype)) {
    comptime {
        if (!dtype_mod.supportsIntMath(dtype)) @compileError("typed maximum/minimum kernels are integer-only (the float facade widens through f32)");
    }
    return elementwiseRankTyped(rt, dtype, rank, .max, a, b);
}

pub fn minRankTyped(
    rt: *Runtime,
    comptime dtype: DType,
    comptime rank: usize,
    a: *const tensor.TensorOf(dtype),
    b: *const tensor.TensorOf(dtype),
) !tensor.TensorOf(dtype_mod.outputDType(.pointwise, dtype)) {
    comptime {
        if (!dtype_mod.supportsIntMath(dtype)) @compileError("typed maximum/minimum kernels are integer-only (the float facade widens through f32)");
    }
    return elementwiseRankTyped(rt, dtype, rank, .min, a, b);
}

pub fn maxRank(rt: *Runtime, comptime rank: usize, a: *const Tensor, b: *const Tensor) !Tensor {
    return elementwiseRank(rt, rank, .max, a, b);
}

pub fn minRank(rt: *Runtime, comptime rank: usize, a: *const Tensor, b: *const Tensor) !Tensor {
    return elementwiseRank(rt, rank, .min, a, b);
}

pub fn gatedRank(rt: *Runtime, comptime rank: usize, comptime op: GatedOp, a: *const Tensor, b: *const Tensor) !Tensor {
    const shape = try requireSameRankShape(rank, a, b);
    var aa = try rt.prepareContiguous(a);
    defer aa.deinit();
    var bb = try rt.prepareContiguous(b);
    defer bb.deinit();

    const ap = aa.tensor();
    const bp = bb.tensor();
    var out = try rt.emptyRank(rank, shape);
    errdefer out.deinit();
    rt.enableNativeVectorPoolForWork(ap.len(), parallel.vector_elementwise_len_threshold);
    rt.backend.gatedContiguousIntoUnchecked(op, &out, ap, bp, ap.len());
    return out;
}

pub fn gluRank(rt: *Runtime, comptime rank: usize, a: *const Tensor, b: *const Tensor) !Tensor {
    return gatedRank(rt, rank, .glu, a, b);
}

pub fn swigluRank(rt: *Runtime, comptime rank: usize, a: *const Tensor, b: *const Tensor) !Tensor {
    return gatedRank(rt, rank, .swiglu, a, b);
}

pub fn gegluRank(rt: *Runtime, comptime rank: usize, a: *const Tensor, b: *const Tensor) !Tensor {
    return gatedRank(rt, rank, .geglu, a, b);
}

pub fn splitSwiGluAxisRank(rt: *Runtime, comptime rank: usize, x: *const Tensor, comptime axis: usize) !Tensor {
    return splitGatedAxisRankImpl(rt, .swiglu, rank, x, axis);
}

pub fn splitGluAxisRank(rt: *Runtime, comptime rank: usize, x: *const Tensor, comptime axis: usize) !Tensor {
    return splitGatedAxisRankImpl(rt, .glu, rank, x, axis);
}

/// One split-gated forward for both conventions. The gate-half conventions
/// are OPPOSITE (ggml parity): swiglu gates with the FIRST half
/// (`silu(first) * second`), glu with the SECOND (`first * sigmoid(second)`).
fn splitGatedAxisRankImpl(rt: *Runtime, comptime op: GatedOp, comptime rank: usize, x: *const Tensor, comptime axis: usize) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");
    const Task = switch (op) {
        .swiglu => SplitSwiGluTask,
        .glu => SplitGluTask,
        .geglu => @compileError("no split-geglu row kernel or gate-half convention exists"),
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

    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    const input = xx.tensor().dataConst();

    var out = try rt.emptyRank(rank, out_shape);
    errdefer out.deinit();
    const output = out.data();

    const half = axis_dim / 2;
    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);
    if (inner == 1) {
        if (out.len() >= parallel.vector_elementwise_len_threshold / 8) {
            if (rt.workPool()) |pool| {
                const task_count = @min(parallel.cpuThreadCount(parallel.vector_max_threads), outer);
                var tasks: [parallel.vector_max_threads]Task = undefined;
                const base: Task = .{
                    .input = input,
                    .output = output,
                    .axis_dim = axis_dim,
                    .half = half,
                    .outer_start = 0,
                    .outer_end = outer,
                };
                for (0..task_count) |task_i| {
                    tasks[task_i] = base;
                    tasks[task_i].outer_start = task_i * outer / task_count;
                    tasks[task_i].outer_end = (task_i + 1) * outer / task_count;
                }
                pool.parallelChunks(Task, tasks[0..task_count], runTask);
                return out;
            }
        }

        rowsKernel(.{
            .input = input,
            .output = output,
            .axis_dim = axis_dim,
            .half = half,
            .outer_start = 0,
            .outer_end = outer,
        });
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

pub fn splitSwiGluBackwardAxisRank(rt: *Runtime, comptime rank: usize, x: *const Tensor, gy: *const Tensor, comptime axis: usize) !Tensor {
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

    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    var ggy = try rt.prepareContiguous(gy);
    defer ggy.deinit();
    const input = xx.tensor().dataConst();
    const grad = ggy.tensor().dataConst();

    var out = try rt.emptyRank(rank, source.shape);
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
            if (rt.workPool()) |pool| {
                const task_count = @min(parallel.cpuThreadCount(parallel.vector_max_threads), outer);
                var tasks: [parallel.vector_max_threads]SplitSwiGluBackwardTask = undefined;
                for (0..task_count) |task_i| {
                    tasks[task_i] = base_task;
                    tasks[task_i].outer_start = task_i * outer / task_count;
                    tasks[task_i].outer_end = (task_i + 1) * outer / task_count;
                }
                pool.parallelChunks(SplitSwiGluBackwardTask, tasks[0..task_count], runSplitSwiGluBackwardTask);
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

pub fn splitGluBackwardAxisRank(rt: *Runtime, comptime rank: usize, x: *const Tensor, gy: *const Tensor, comptime axis: usize) !Tensor {
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

    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    var ggy = try rt.prepareContiguous(gy);
    defer ggy.deinit();
    const input = xx.tensor().dataConst();
    const grad = ggy.tensor().dataConst();

    var out = try rt.emptyRank(rank, source.shape);
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
            if (rt.workPool()) |pool| {
                const task_count = @min(parallel.cpuThreadCount(parallel.vector_max_threads), outer);
                var tasks: [parallel.vector_max_threads]SplitGluBackwardTask = undefined;
                for (0..task_count) |task_i| {
                    tasks[task_i] = base_task;
                    tasks[task_i].outer_start = task_i * outer / task_count;
                    tasks[task_i].outer_end = (task_i + 1) * outer / task_count;
                }
                pool.parallelChunks(SplitGluBackwardTask, tasks[0..task_count], runSplitGluBackwardTask);
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

pub fn addInPlace(rt: *Runtime, target: *Tensor, other: *const Tensor) !void {
    return elementwiseInPlace(rt, .add, target, other);
}

pub fn subInPlace(rt: *Runtime, target: *Tensor, other: *const Tensor) !void {
    return elementwiseInPlace(rt, .sub, target, other);
}

pub fn mulInPlace(rt: *Runtime, target: *Tensor, other: *const Tensor) !void {
    return elementwiseInPlace(rt, .mul, target, other);
}

pub fn divInPlace(rt: *Runtime, target: *Tensor, other: *const Tensor) !void {
    return elementwiseInPlace(rt, .div, target, other);
}

pub fn takeAdd(rt: *Runtime, target: *Tensor, other: *const Tensor) !Tensor {
    return takeElementwise(rt, .add, target, other);
}

pub fn takeSub(rt: *Runtime, target: *Tensor, other: *const Tensor) !Tensor {
    return takeElementwise(rt, .sub, target, other);
}

pub fn takeMul(rt: *Runtime, target: *Tensor, other: *const Tensor) !Tensor {
    return takeElementwise(rt, .mul, target, other);
}

pub fn takeDiv(rt: *Runtime, target: *Tensor, other: *const Tensor) !Tensor {
    return takeElementwise(rt, .div, target, other);
}

pub fn takeScale(rt: *Runtime, target: *Tensor, scalar_value: f32) !Tensor {
    if (target.canTakeInPlace()) {
        rt.enableNativeVectorPoolForWork(target.len(), parallel.vector_elementwise_len_threshold);
        try rt.backend.scaleInto(target, target, scalar_value);
        return takeTensor(target);
    }

    var result = try scale(rt, target, scalar_value);
    errdefer result.deinit();
    return discardTakenInput(target, result);
}

fn takeUnary(rt: *Runtime, comptime op: UnaryOp, target: *Tensor) !Tensor {
    if (target.canTakeInPlace()) {
        rt.enableNativeVectorPoolForWork(target.len(), parallel.vector_elementwise_len_threshold);
        rt.backend.unaryContiguousIntoUnchecked(op, target, target, target.len());
        return takeTensor(target);
    }

    var result = try unary(rt, op, target);
    errdefer result.deinit();
    return discardTakenInput(target, result);
}

pub fn takeRelu(rt: *Runtime, target: *Tensor) !Tensor {
    return takeUnary(rt, .relu, target);
}

pub fn takeSilu(rt: *Runtime, target: *Tensor) !Tensor {
    return takeUnary(rt, .silu, target);
}

pub fn scale(rt: *Runtime, x: *const Tensor, scalar_value: f32) !Tensor {
    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();

    const xp = xx.tensor();
    var out = try rt.empty(xp.shape.slice());
    errdefer out.deinit();
    rt.enableNativeVectorPoolForWork(xp.len(), parallel.vector_elementwise_len_threshold);
    try rt.backend.scaleInto(&out, xp, scalar_value);
    return out;
}

pub fn addScalar(rt: *Runtime, x: *const Tensor, scalar_value: f32) !Tensor {
    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    const xp = xx.tensor();
    var out = try rt.empty(xp.shape.slice());
    errdefer out.deinit();
    for (xp.dataConst(), out.data()) |value, *dst| dst.* = value + scalar_value;
    return out;
}

pub fn powScalar(rt: *Runtime, x: *const Tensor, exponent: f32) !Tensor {
    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    const xp = xx.tensor();
    var out = try rt.empty(xp.shape.slice());
    errdefer out.deinit();
    for (xp.dataConst(), out.data()) |value, *dst| dst.* = std.math.pow(f32, value, exponent);
    return out;
}

/// Elementwise select: `out[i] = cond[i] != 0 ? x[i] : y[i]` (all same shape).
pub fn where(rt: *Runtime, x: *const Tensor, cond: *const Tensor, y: *const Tensor) !Tensor {
    try tensor.requireSameShape(x, cond);
    try tensor.requireSameShape(x, y);
    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    var cc = try rt.prepareContiguous(cond);
    defer cc.deinit();
    var yy = try rt.prepareContiguous(y);
    defer yy.deinit();
    const xp = xx.tensor();
    var out = try rt.empty(xp.shape.slice());
    errdefer out.deinit();
    for (xp.dataConst(), cc.tensor().dataConst(), yy.tensor().dataConst(), out.data()) |xv, cv, yv, *dst| {
        dst.* = if (cv != 0) xv else yv;
    }
    return out;
}

/// Elementwise masked fill: `out[i] = mask[i] != 0 ? value : x[i]`.
pub fn maskedFill(rt: *Runtime, x: *const Tensor, mask: *const Tensor, value: f32) !Tensor {
    try tensor.requireSameShape(x, mask);
    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    var mm = try rt.prepareContiguous(mask);
    defer mm.deinit();
    const xp = xx.tensor();
    var out = try rt.empty(xp.shape.slice());
    errdefer out.deinit();
    for (xp.dataConst(), mm.tensor().dataConst(), out.data()) |xv, mv, *dst| {
        dst.* = if (mv != 0) value else xv;
    }
    return out;
}

/// Elementwise comparison mask: `out[i] = a[i] <op> b[i] ? 1.0 : 0.0`
/// (same shape only, like `where`; no broadcasting). NaN semantics are IEEE:
/// any comparison involving NaN is false — except `ne`, which is true — so
/// eq(NaN, NaN) = 0 and ne(NaN, x) = 1.
pub fn compare(rt: *Runtime, comptime op: CompareOp, a: *const Tensor, b: *const Tensor) !Tensor {
    try tensor.requireSameShape(a, b);
    var aa = try rt.prepareContiguous(a);
    defer aa.deinit();
    var bb = try rt.prepareContiguous(b);
    defer bb.deinit();
    const ap = aa.tensor();
    var out = try rt.empty(ap.shape.slice());
    errdefer out.deinit();
    for (ap.dataConst(), bb.tensor().dataConst(), out.data()) |av, bv, *dst| {
        dst.* = if (backend_ops.compareScalar(op, av, bv)) 1 else 0;
    }
    return out;
}

/// Elementwise comparison mask vs a scalar RHS:
/// `out[i] = x[i] <op> scalar_value ? 1.0 : 0.0`. Same IEEE NaN contract as
/// `compare` (any comparison involving NaN is false except `ne`).
pub fn compareScalar(rt: *Runtime, comptime op: CompareOp, x: *const Tensor, scalar_value: f32) !Tensor {
    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    const xp = xx.tensor();
    var out = try rt.empty(xp.shape.slice());
    errdefer out.deinit();
    for (xp.dataConst(), out.data()) |xv, *dst| {
        dst.* = if (backend_ops.compareScalar(op, xv, scalar_value)) 1 else 0;
    }
    return out;
}

/// Elementwise logical AND over `!= 0` truthiness (the repo-wide mask
/// convention shared with `where`/`maskedFill`; NaN is `!= 0`, hence truthy):
/// `out[i] = (a[i] != 0 and b[i] != 0) ? 1.0 : 0.0`. Same shape only.
pub fn logicalAnd(rt: *Runtime, a: *const Tensor, b: *const Tensor) !Tensor {
    try tensor.requireSameShape(a, b);
    var aa = try rt.prepareContiguous(a);
    defer aa.deinit();
    var bb = try rt.prepareContiguous(b);
    defer bb.deinit();
    const ap = aa.tensor();
    var out = try rt.empty(ap.shape.slice());
    errdefer out.deinit();
    for (ap.dataConst(), bb.tensor().dataConst(), out.data()) |av, bv, *dst| {
        dst.* = if (av != 0 and bv != 0) 1 else 0;
    }
    return out;
}

/// Elementwise logical OR over `!= 0` truthiness (see `logicalAnd`):
/// `out[i] = (a[i] != 0 or b[i] != 0) ? 1.0 : 0.0`. Same shape only.
pub fn logicalOr(rt: *Runtime, a: *const Tensor, b: *const Tensor) !Tensor {
    try tensor.requireSameShape(a, b);
    var aa = try rt.prepareContiguous(a);
    defer aa.deinit();
    var bb = try rt.prepareContiguous(b);
    defer bb.deinit();
    const ap = aa.tensor();
    var out = try rt.empty(ap.shape.slice());
    errdefer out.deinit();
    for (ap.dataConst(), bb.tensor().dataConst(), out.data()) |av, bv, *dst| {
        dst.* = if (av != 0 or bv != 0) 1 else 0;
    }
    return out;
}

/// Elementwise logical XOR over `!= 0` truthiness (see `logicalAnd`):
/// `out[i] = ((a[i] != 0) != (b[i] != 0)) ? 1.0 : 0.0`. Same shape only.
pub fn logicalXor(rt: *Runtime, a: *const Tensor, b: *const Tensor) !Tensor {
    try tensor.requireSameShape(a, b);
    var aa = try rt.prepareContiguous(a);
    defer aa.deinit();
    var bb = try rt.prepareContiguous(b);
    defer bb.deinit();
    const ap = aa.tensor();
    var out = try rt.empty(ap.shape.slice());
    errdefer out.deinit();
    for (ap.dataConst(), bb.tensor().dataConst(), out.data()) |av, bv, *dst| {
        dst.* = if ((av != 0) != (bv != 0)) 1 else 0;
    }
    return out;
}

/// Elementwise logical NOT over `!= 0` truthiness (see `logicalAnd`):
/// `out[i] = x[i] != 0 ? 0.0 : 1.0`.
pub fn logicalNot(rt: *Runtime, x: *const Tensor) !Tensor {
    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    const xp = xx.tensor();
    var out = try rt.empty(xp.shape.slice());
    errdefer out.deinit();
    for (xp.dataConst(), out.data()) |xv, *dst| {
        dst.* = if (xv != 0) 0 else 1;
    }
    return out;
}

pub fn addScaledInPlace(rt: *Runtime, target: *Tensor, source: *const Tensor, scalar_value: f32) !void {
    try tensor.requireSameShape(target, source);
    if (!target.isContiguous()) return tensor.TensorError.UnsupportedView;

    var ss = try rt.prepareContiguous(source);
    defer ss.deinit();
    rt.enableNativeVectorPoolForWork(target.len(), parallel.vector_elementwise_len_threshold);
    rt.backend.addScaledSliceUnchecked(target.data(), ss.tensor().dataConst(), scalar_value);
}

pub fn addAxisVectorInPlaceRank(rt: *Runtime, comptime rank: usize, target: *Tensor, row_vector: []const f32, comptime axis: usize) !void {
    try addAxisVectorUnaryInPlaceRank(rt, rank, null, target, row_vector, axis);
}

pub fn addAxisVectorUnaryInPlaceRank(rt: *Runtime, comptime rank: usize, comptime op: ?UnaryOp, target: *Tensor, row_vector: []const f32, comptime axis: usize) !void {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");

    const view = try target.rankView(rank);
    if (!target.isContiguous()) return tensor.TensorError.UnsupportedView;
    const axis_dim = view.shape[axis];
    if (row_vector.len != axis_dim) return tensor.TensorError.ShapeMismatch;
    if (productAfterAxis(rank, view.shape, axis) != 1) return tensor.TensorError.UnsupportedView;

    const rows = productBeforeAxis(rank, view.shape, axis);
    rt.enableNativeVectorPoolForWork(target.len(), parallel.vector_elementwise_len_threshold);
    if (comptime op) |actual_op| {
        rt.backend.addRowVectorUnarySliceUnchecked(actual_op, target.data(), row_vector, rows, axis_dim);
    } else {
        rt.backend.addRowVectorSliceUnchecked(target.data(), row_vector, rows, axis_dim);
    }
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
pub fn preluChannels(rt: *Runtime, x: *const Tensor, alpha: *const Tensor) !Tensor {
    const alpha_view = try alpha.rankView(1);
    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    var aa = try rt.prepareContiguous(alpha);
    defer aa.deinit();
    const rc = try channelRowsCols(xx.tensor(), alpha_view.shape[0]);

    var out = try rt.empty(xx.tensor().shape.slice());
    errdefer out.deinit();
    rt.enableNativeVectorPoolForWork(x.len(), parallel.vector_elementwise_len_threshold);
    rt.backend.preluChannelsIntoUnchecked(out.data(), xx.tensor().dataConst(), aa.tensor().dataConst(), rc.rows, rc.cols);
    return out;
}

/// PReLU input-VJP: `gx = x > 0 ? gy : α[c]·gy`.
pub fn preluChannelsBackwardInput(rt: *Runtime, gy: *const Tensor, x: *const Tensor, alpha: *const Tensor) !Tensor {
    try tensor.requireSameShape(gy, x);
    const alpha_view = try alpha.rankView(1);
    var gg = try rt.prepareContiguous(gy);
    defer gg.deinit();
    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    var aa = try rt.prepareContiguous(alpha);
    defer aa.deinit();
    const rc = try channelRowsCols(xx.tensor(), alpha_view.shape[0]);

    var out = try rt.empty(xx.tensor().shape.slice());
    errdefer out.deinit();
    rt.enableNativeVectorPoolForWork(x.len(), parallel.vector_elementwise_len_threshold);
    rt.backend.preluChannelsBackwardInputIntoUnchecked(out.data(), gg.tensor().dataConst(), xx.tensor().dataConst(), aa.tensor().dataConst(), rc.rows, rc.cols);
    return out;
}

/// PReLU slope-VJP: `gα[c] = Σ_rows gy·min(x, 0)` → rank-1 `[C]`.
pub fn preluChannelsBackwardAlpha(rt: *Runtime, gy: *const Tensor, x: *const Tensor, channels: usize) !Tensor {
    try tensor.requireSameShape(gy, x);
    var gg = try rt.prepareContiguous(gy);
    defer gg.deinit();
    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    const rc = try channelRowsCols(xx.tensor(), channels);

    var out = try rt.emptyRank(1, .{rc.cols});
    errdefer out.deinit();
    rt.backend.preluChannelsBackwardAlphaIntoUnchecked(out.data(), gg.tensor().dataConst(), xx.tensor().dataConst(), rc.rows, rc.cols);
    return out;
}

/// Per-channel affine over a channel-last tensor (any rank ≥ 1, channel axis
/// innermost): `y = x·scale[c] + shift[c]` — the frozen-stats inference
/// BatchNorm as ONE fused pass. Mul-then-add (never contracted to fma), so
/// the values equal the two-pass broadcast mul + add composition bitwise.
/// A null `shift` degrades to the per-channel scale `y = x·scale[c]`.
pub fn channelAffine(rt: *Runtime, x: *const Tensor, scale_vec: *const Tensor, shift_vec: ?*const Tensor) !Tensor {
    const scale_view = try scale_vec.rankView(1);
    if (shift_vec) |t| {
        const shift_view = try t.rankView(1);
        if (scale_view.shape[0] != shift_view.shape[0]) return tensor.TensorError.ShapeMismatch;
    }
    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    var ss = try rt.prepareContiguous(scale_vec);
    defer ss.deinit();
    var tt: ?Runtime.PreparedTensor = if (shift_vec) |t| try rt.prepareContiguous(t) else null;
    defer if (tt) |*p| p.deinit();
    const rc = try channelRowsCols(xx.tensor(), scale_view.shape[0]);

    var out = try rt.empty(xx.tensor().shape.slice());
    errdefer out.deinit();
    rt.enableNativeVectorPoolForWork(x.len(), parallel.vector_elementwise_len_threshold);
    rt.backend.channelAffineIntoUnchecked(out.data(), xx.tensor().dataConst(), ss.tensor().dataConst(), if (tt) |*p| p.tensor().dataConst() else null, rc.rows, rc.cols);
    return out;
}

/// Inverted dropout forward: element i keeps `x[i] / (1-p)` iff the
/// 53-bit uniform of `rng.at(seed, i)` is < 1-p, else 0. The mask is never
/// materialized — it is a counter-based function of (seed, element index),
/// so the backward kernel (and a checkpoint recompute) regenerates it
/// exactly. Requires `0 <= p < 1`.
pub fn dropoutForward(rt: *Runtime, x: *const Tensor, p: f32, seed: u64) !Tensor {
    return dropoutApply(rt, x, p, seed);
}

/// Inverted dropout VJP: the gradient passes through kept elements scaled
/// by 1/(1-p) and is 0 at dropped ones — the identical (seed, i) mask and
/// arithmetic as `dropoutForward`, applied to `gy`.
pub fn dropoutBackward(rt: *Runtime, gy: *const Tensor, p: f32, seed: u64) !Tensor {
    return dropoutApply(rt, gy, p, seed);
}

fn dropoutApply(rt: *Runtime, x: *const Tensor, p: f32, seed: u64) !Tensor {
    if (!(p >= 0 and p < 1)) return tensor.TensorError.InvalidShape;

    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    const xp = xx.tensor();
    const input = xp.dataConst();

    var out = try rt.empty(xp.shape.slice());
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
        if (rt.workPool()) |pool| {
            const task_count = @min(parallel.cpuThreadCount(parallel.vector_max_threads), input.len);
            var tasks: [parallel.vector_max_threads]DropoutRangeTask = undefined;
            for (0..task_count) |task_i| {
                tasks[task_i] = base_task;
                tasks[task_i].start = task_i * input.len / task_count;
                tasks[task_i].end = (task_i + 1) * input.len / task_count;
            }
            pool.parallelChunks(DropoutRangeTask, tasks[0..task_count], runDropoutRangeTask);
            return out;
        }
    }

    dropoutRange(base_task);
    return out;
}

pub fn unary(rt: *Runtime, comptime op: UnaryOp, x: *const Tensor) !Tensor {
    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();

    const xp = xx.tensor();
    var out = try rt.empty(xp.shape.slice());
    errdefer out.deinit();
    rt.enableNativeVectorPoolForWork(xp.len(), parallel.vector_elementwise_len_threshold);
    rt.backend.unaryContiguousIntoUnchecked(op, &out, xp, xp.len());
    return out;
}

pub fn relu(rt: *Runtime, x: *const Tensor) !Tensor {
    return unary(rt, .relu, x);
}

pub fn leakyRelu(rt: *Runtime, x: *const Tensor, negative_slope: f32) !Tensor {
    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();

    const xp = xx.tensor();
    var out = try rt.empty(xp.shape.slice());
    errdefer out.deinit();
    rt.enableNativeVectorPoolForWork(xp.len(), parallel.vector_elementwise_len_threshold);
    rt.backend.leakyReluContiguousIntoUnchecked(&out, xp, xp.len(), negative_slope);
    return out;
}

pub fn exp(rt: *Runtime, x: *const Tensor) !Tensor {
    return unary(rt, .exp, x);
}

pub fn sqrt(rt: *Runtime, x: *const Tensor) !Tensor {
    return unary(rt, .sqrt, x);
}

pub fn rsqrt(rt: *Runtime, x: *const Tensor) !Tensor {
    return unary(rt, .rsqrt, x);
}

pub fn sigmoid(rt: *Runtime, x: *const Tensor) !Tensor {
    return unary(rt, .sigmoid, x);
}

pub fn silu(rt: *Runtime, x: *const Tensor) !Tensor {
    return unary(rt, .silu, x);
}

pub fn log(rt: *Runtime, x: *const Tensor) !Tensor {
    return unary(rt, .log, x);
}

pub fn neg(rt: *Runtime, x: *const Tensor) !Tensor {
    return unary(rt, .neg, x);
}

pub fn abs(rt: *Runtime, x: *const Tensor) !Tensor {
    return unary(rt, .abs, x);
}

pub fn sin(rt: *Runtime, x: *const Tensor) !Tensor {
    return unary(rt, .sin, x);
}

pub fn cos(rt: *Runtime, x: *const Tensor) !Tensor {
    return unary(rt, .cos, x);
}

pub fn tanh(rt: *Runtime, x: *const Tensor) !Tensor {
    return unary(rt, .tanh, x);
}

pub fn gelu(rt: *Runtime, x: *const Tensor) !Tensor {
    return unary(rt, .gelu, x);
}

pub fn quickGelu(rt: *Runtime, x: *const Tensor) !Tensor {
    return unary(rt, .quick_gelu, x);
}

/// Per-channel Snake activation over `[rows, cols]` rows (the DAC codec op):
/// `y[t,c] = x[t,c] + inv_b[c] * sin(alpha[c] * x[t,c])^2`. The caller
/// precomputes `inv_b = 1/(alpha + 1e-9)` at weight-load time (the reference's
/// convention) — the epsilon is deliberately NOT folded into the kernel.
pub fn snakeRows(rt: *Runtime, x: *const Tensor, alpha: *const Tensor, inv_b: *const Tensor) !Tensor {
    const source = try x.rankView(2);
    const rows = source.shape[0];
    const cols = source.shape[1];
    const alpha_view = try alpha.rankView(1);
    if (alpha_view.shape[0] != cols) return tensor.TensorError.ShapeMismatch;
    const inv_view = try inv_b.rankView(1);
    if (inv_view.shape[0] != cols) return tensor.TensorError.ShapeMismatch;

    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    var aa = try rt.prepareContiguous(alpha);
    defer aa.deinit();
    var bb = try rt.prepareContiguous(inv_b);
    defer bb.deinit();

    var out = try rt.emptyRank(2, .{ rows, cols });
    errdefer out.deinit();
    rt.enableNativeVectorPoolForWork(xx.tensor().len(), parallel.vector_elementwise_len_threshold);
    rt.backend.snakeInto(&out, xx.tensor(), aa.tensor().dataConst(), bb.tensor().dataConst(), rows, cols);
    return out;
}

/// VJP of snakeRows wrt the input:
/// `gx[t,c] = gy[t,c] * (1 + inv_b[c]*alpha[c]*sin(2*alpha[c]*x[t,c]))`.
pub fn snakeRowsBackwardInput(rt: *Runtime, x: *const Tensor, gy: *const Tensor, alpha: *const Tensor, inv_b: *const Tensor) !Tensor {
    const source = try x.rankView(2);
    const rows = source.shape[0];
    const cols = source.shape[1];
    try tensor.requireSameShape(x, gy);
    const alpha_view = try alpha.rankView(1);
    if (alpha_view.shape[0] != cols) return tensor.TensorError.ShapeMismatch;
    const inv_view = try inv_b.rankView(1);
    if (inv_view.shape[0] != cols) return tensor.TensorError.ShapeMismatch;

    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    var gg = try rt.prepareContiguous(gy);
    defer gg.deinit();
    var aa = try rt.prepareContiguous(alpha);
    defer aa.deinit();
    var bb = try rt.prepareContiguous(inv_b);
    defer bb.deinit();

    var out = try rt.emptyRank(2, .{ rows, cols });
    errdefer out.deinit();
    rt.enableNativeVectorPoolForWork(xx.tensor().len(), parallel.vector_elementwise_len_threshold);
    rt.backend.snakeBackwardInputInto(&out, xx.tensor(), gg.tensor(), aa.tensor().dataConst(), bb.tensor().dataConst(), rows, cols);
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
pub fn snakeRowsBackwardParams(rt: *Runtime, x: *const Tensor, gy: *const Tensor, alpha: *const Tensor, inv_b: *const Tensor) !SnakeBackwardParamsResult {
    const source = try x.rankView(2);
    const rows = source.shape[0];
    const cols = source.shape[1];
    try tensor.requireSameShape(x, gy);
    const alpha_view = try alpha.rankView(1);
    if (alpha_view.shape[0] != cols) return tensor.TensorError.ShapeMismatch;
    const inv_view = try inv_b.rankView(1);
    if (inv_view.shape[0] != cols) return tensor.TensorError.ShapeMismatch;

    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    var gg = try rt.prepareContiguous(gy);
    defer gg.deinit();
    var aa = try rt.prepareContiguous(alpha);
    defer aa.deinit();
    var bb = try rt.prepareContiguous(inv_b);
    defer bb.deinit();

    var galpha = try rt.emptyRank(1, .{cols});
    errdefer galpha.deinit();
    var ginv_b = try rt.emptyRank(1, .{cols});
    errdefer ginv_b.deinit();
    rt.enableNativeVectorPoolForWork(xx.tensor().len(), parallel.vector_elementwise_len_threshold);
    rt.backend.snakeBackwardParamsInto(&galpha, &ginv_b, xx.tensor(), gg.tensor(), aa.tensor().dataConst(), bb.tensor().dataConst(), rows, cols);
    return .{ .alpha = galpha, .inv_b = ginv_b };
}

pub fn clamp(rt: *Runtime, x: *const Tensor, min_value: f32, max_value: f32) !Tensor {
    if (min_value > max_value) return tensor.TensorError.InvalidShape;

    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();

    const xp = xx.tensor();
    var out = try rt.empty(xp.shape.slice());
    errdefer out.deinit();
    rt.enableNativeVectorPoolForWork(xp.len(), parallel.vector_elementwise_len_threshold);
    rt.backend.clampContiguousIntoUnchecked(&out, xp, xp.len(), min_value, max_value);
    return out;
}

pub fn reduceBroadcast(rt: *Runtime, x: *const Tensor, target_shape: []const usize) !Tensor {
    return dispatchRank(reduceBroadcastTargetDispatched, target_shape.len, .{ rt, x, target_shape });
}

/// `pub` so the `exec.ExecContext` non-pub forwarder used by the inline
/// Group-A tests (which drive `reduceBroadcastRank` deliberately) can reach it.
pub fn reduceBroadcastRank(
    rt: *Runtime,
    comptime target_rank: usize,
    x: *const Tensor,
    target_shape: [target_rank]usize,
) !Tensor {
    return dispatchRank(reduceBroadcastSourceDispatched, x.shape.len, .{ rt, target_rank, x, target_shape });
}

fn reduceBroadcastTargetDispatched(
    comptime target_rank: usize,
    rt: *Runtime,
    x: *const Tensor,
    target_shape: []const usize,
) !Tensor {
    return reduceBroadcastRank(rt, target_rank, x, try shapeArrayFromSlice(target_rank, target_shape));
}

fn reduceBroadcastSourceDispatched(
    comptime source_rank: usize,
    rt: *Runtime,
    comptime target_rank: usize,
    x: *const Tensor,
    target_shape: [target_rank]usize,
) !Tensor {
    return reduceBroadcastFromRankToRank(rt, source_rank, target_rank, x, target_shape);
}

fn reduceBroadcastFromRankToRank(
    rt: *Runtime,
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

    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    const xp = xx.tensor();

    var out = try rt.zerosRank(target_rank, target_shape);
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

fn takeElementwise(
    rt: *Runtime,
    comptime op: ElementwiseOp,
    target: *Tensor,
    other: *const Tensor,
) !Tensor {
    try tensor.requireSameShape(target, other);

    if (target.canTakeInPlace()) {
        try elementwiseInPlace(rt, op, target, other);
        return takeTensor(target);
    }

    var result = try elementwiseRuntimeRank(rt, op, target, other);
    errdefer result.deinit();
    return discardTakenInput(target, result);
}

fn elementwiseRuntimeRank(
    rt: *Runtime,
    comptime op: ElementwiseOp,
    a: *const Tensor,
    b: *const Tensor,
) !Tensor {
    return dispatchRank(elementwiseRankDispatched, a.shape.len, .{ rt, op, a, b });
}

fn elementwiseRankDispatched(
    comptime rank: usize,
    rt: *Runtime,
    comptime op: ElementwiseOp,
    a: *const Tensor,
    b: *const Tensor,
) !Tensor {
    return elementwiseRank(rt, rank, op, a, b);
}

fn elementwiseRank(
    rt: *Runtime,
    comptime rank: usize,
    comptime op: ElementwiseOp,
    a: *const Tensor,
    b: *const Tensor,
) !Tensor {
    const shape = try requireSameRankShape(rank, a, b);
    var out = try rt.emptyRank(rank, shape);
    errdefer out.deinit();
    try elementwiseRankInto(rt, rank, op, &out, a, b, shape);
    return out;
}

fn elementwiseRankTyped(
    rt: *Runtime,
    comptime dtype: DType,
    comptime rank: usize,
    comptime op: ElementwiseOp,
    a: *const tensor.TensorOf(dtype),
    b: *const tensor.TensorOf(dtype),
) !tensor.TensorOf(dtype_mod.outputDType(.pointwise, dtype)) {
    if (comptime dtype == .f32) return elementwiseRank(rt, rank, op, a, b);
    if (comptime dtype_mod.supportsIntMath(dtype)) {
        // Integer pointwise: a plain exec loop (the stats.zig precedent —
        // no backend kernel; integers are never the hot path). Wrapping
        // two's-complement arithmetic; `div` is a compile error (integer
        // division is explicit: intDivRankTyped).
        const shape = try requireSameRankShapeOf(dtype, rank, a, b);
        var aa = try rt.prepareContiguousTyped(dtype, a);
        defer aa.deinit();
        var bb = try rt.prepareContiguousTyped(dtype, b);
        defer bb.deinit();
        var out = try rt.emptyRankTyped(dtype, rank, shape);
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
    const output_dtype = comptime dtype_mod.outputDType(.pointwise, dtype);

    const shape = try requireSameRankShapeOf(dtype, rank, a, b);
    var aa = try rt.prepareContiguousTyped(dtype, a);
    defer aa.deinit();
    var bb = try rt.prepareContiguousTyped(dtype, b);
    defer bb.deinit();

    var out = try rt.emptyRankTyped(output_dtype, rank, shape);
    errdefer out.deinit();
    rt.enableNativeVectorPoolForWork(out.len(), parallel.vector_elementwise_len_threshold);
    rt.backend.elementwiseContiguousIntoTyped(dtype, op, &out, aa.tensor(), bb.tensor(), out.len());
    return out;
}

pub const IntDivMode = enum { trunc, floor };

/// Integer division as an explicit op (torch's `/` silently promotes
/// integers to float — a documented divergence: Fucina keeps promotion
/// explicit). `.trunc` rounds toward zero (C semantics), `.floor` toward
/// negative infinity (Python's //). A zero divisor is
/// `error.DivisionByZero`.
pub fn intDivRankTyped(
    rt: *Runtime,
    comptime dtype: DType,
    comptime rank: usize,
    comptime mode: IntDivMode,
    a: *const tensor.TensorOf(dtype),
    b: *const tensor.TensorOf(dtype),
) !tensor.TensorOf(dtype) {
    comptime {
        if (!dtype_mod.supportsIntMath(dtype)) @compileError("intDivRankTyped requires an integer dtype");
    }
    const shape = try requireSameRankShapeOf(dtype, rank, a, b);
    var aa = try rt.prepareContiguousTyped(dtype, a);
    defer aa.deinit();
    var bb = try rt.prepareContiguousTyped(dtype, b);
    defer bb.deinit();
    var out = try rt.emptyRankTyped(dtype, rank, shape);
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

fn elementwiseRankInto(
    rt: *Runtime,
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
        return backendElementwiseContiguousUnchecked(rt, op, out, a, b, len);
    }

    if (try tryTailBroadcastElementwise(op, out, a, b)) {
        return;
    }

    var aa = try rt.prepareContiguous(a);
    defer aa.deinit();
    var bb = try rt.prepareContiguous(b);
    defer bb.deinit();
    return backendElementwiseContiguousUnchecked(rt, op, out, aa.tensor(), bb.tensor(), len);
}

fn elementwiseInto(
    rt: *Runtime,
    comptime op: ElementwiseOp,
    out: *Tensor,
    a: *const Tensor,
    b: *const Tensor,
) !void {
    try tensor.requireSameShape(out, a);

    if (a.isContiguous() and b.isContiguous()) {
        return backendElementwiseContiguousUnchecked(rt, op, out, a, b, out.len());
    }

    if (try tryTailBroadcastElementwise(op, out, a, b)) {
        return;
    }

    var aa = try rt.prepareContiguous(a);
    defer aa.deinit();
    var bb = try rt.prepareContiguous(b);
    defer bb.deinit();
    return backendElementwiseContiguousUnchecked(rt, op, out, aa.tensor(), bb.tensor(), out.len());
}

fn elementwiseInPlace(
    rt: *Runtime,
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
        return backendElementwiseContiguousUnchecked(rt, op, target, target, other, target.len());
    }

    if (tryTailBroadcastElementwiseInPlace(op, target, other)) {
        return;
    }

    var materialized = try rt.materialize(other);
    defer materialized.deinit();
    return backendElementwiseContiguousUnchecked(rt, op, target, target, &materialized, target.len());
}

fn backendElementwiseContiguousUnchecked(
    rt: *Runtime,
    comptime op: ElementwiseOp,
    out: *Tensor,
    a: *const Tensor,
    b: *const Tensor,
    len: usize,
) void {
    rt.enableNativeVectorPoolForWork(len, parallel.vector_elementwise_len_threshold);
    return switch (op) {
        .add => rt.backend.addContiguousIntoUnchecked(out, a, b, len),
        .sub => rt.backend.subContiguousIntoUnchecked(out, a, b, len),
        .mul => rt.backend.mulContiguousIntoUnchecked(out, a, b, len),
        .div => elementwiseContiguousInto(.div, out.data()[0..len], a.dataConst()[0..len], b.dataConst()[0..len]),
        .max => rt.backend.maximumContiguousIntoUnchecked(out, a, b, len),
        .min => rt.backend.minimumContiguousIntoUnchecked(out, a, b, len),
    };
}

const small_in_place_elementwise_len = 2048;

fn ensureForwardFloatMath(comptime dtype: DType) void {
    if (!dtype_mod.supportsForwardFloatMath(dtype)) {
        @compileError("forward math is currently supported only for floating dtypes");
    }
}

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
