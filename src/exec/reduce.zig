//! Sum / mean reductions (whole-tensor + along-axis, typed) + the cumulative
//! (prefix/suffix) sums.
//!
//! Domain module: every op receives an explicit `*ExecContext`. Reduction math
//! dispatches to the backend's `sumInto`/`sumSlice(Typed)` kernels.

const std = @import("std");
const build_options = @import("build_options");
const backend_mod = @import("../backend.zig");
const kernels = backend_mod.kernels;
const parallel = @import("../parallel.zig");
const dtype_mod = @import("../dtype.zig");
const tensor = @import("../tensor.zig");

const shape_mod = @import("../shape.zig");
const ExecContext = @import("../exec.zig").ExecContext;

const DType = tensor.DType;
const Tensor = tensor.Tensor;

const shapeWithoutAxis = shape_mod.withoutAxis;
const contiguousStridesArray = shape_mod.contiguousStrides;
const productAfterAxis = shape_mod.productAfterAxis;
const productBeforeAxis = shape_mod.productBeforeAxis;
const ensureForwardFloatMath = dtype_mod.requireForwardFloatMath;

fn isIntSum(comptime dtype: DType) bool {
    return dtype == .bool or dtype_mod.supportsIntMath(dtype);
}

/// Integer/bool sum: i64 accumulation (wrapping, the +% contract), bool
/// counts `true`s — torch's integer-sum semantics. Plain exec loop.
fn intSumSlice(comptime dtype: DType, values: []const dtype_mod.Scalar(dtype)) i64 {
    var acc: i64 = 0;
    if (comptime dtype == .bool) {
        for (values) |v| acc +%= @intFromBool(v);
    } else {
        for (values) |v| acc +%= v;
    }
    return acc;
}

fn intSumContribution(comptime dtype: DType, value: dtype_mod.Scalar(dtype)) i64 {
    return if (comptime dtype == .bool) @intFromBool(value) else @as(i64, value);
}

// ---------------------------------------------------------------------------
// Non-last-axis reductions: the streaming (outer, axis, inner) walk.
// ---------------------------------------------------------------------------

/// How one axis value folds into its output lane.
const AxisFold = enum { sum, prod, int_sum };

/// Minimum inner lanes per task when a single outer block splits across
/// the pool by lane range: below this the strided row reads are too short
/// for the dispatch to pay.
const axis_fold_min_inner_lanes = 256;

/// A non-last-axis reduction over a contiguous `[outer, axis_dim, inner]`
/// input into `[outer, inner]`: for each outer block the `axis_dim` input
/// rows fold into the output row lane by lane. Every output element meets
/// its axis values in index order, the order the element-linear walk
/// visits them, so the result is bitwise that walk's; the arrays are read
/// contiguously with no per-element delinearization; and outer blocks (or,
/// for one block, inner-lane ranges) split across the pool with disjoint
/// outputs, so any task count gives the same bits.
fn AxisFoldTask(comptime dtype: DType, comptime fold: AxisFold) type {
    return struct {
        input: []const In,
        output: []Out,
        axis_dim: usize,
        inner: usize,
        outer: usize,

        const In = dtype_mod.Scalar(dtype);
        pub const output_dtype = if (fold == .int_sum) .i64 else dtype_mod.outputDType(.reduction, dtype);
        pub const Out = dtype_mod.Scalar(output_dtype);
        const compute_dtype = if (fold == .int_sum) .i64 else dtype_mod.computeDType(.reduction, dtype);
        /// What the streaming arm accumulates into.
        pub const identity: Out = if (fold == .prod) 1 else 0;

        /// The outer-block split: every lane of the blocks `[outer_start, outer_end)`.
        fn runOuter(task: @This(), outer_start: usize, outer_end: usize) void {
            task.foldRange(outer_start, outer_end, 0, task.inner);
        }

        /// The inner-lane split: the lanes `[inner_start, inner_end)` of every block.
        fn runInner(task: @This(), inner_start: usize, inner_end: usize) void {
            task.foldRange(0, task.outer, inner_start, inner_end);
        }

        fn foldRange(task: @This(), outer_start: usize, outer_end: usize, inner_start: usize, inner_end: usize) void {
            const lanes = inner_end - inner_start;
            for (outer_start..outer_end) |outer_i| {
                const out_row = task.output[outer_i * task.inner + inner_start ..][0..lanes];
                const block = outer_i * task.axis_dim;
                for (0..task.axis_dim) |a| {
                    const in_row = task.input[(block + a) * task.inner + inner_start ..][0..lanes];
                    for (out_row, in_row) |*acc, value| {
                        switch (comptime fold) {
                            .sum => if (comptime dtype == .f32) {
                                acc.* += value;
                            } else {
                                const next = dtype_mod.castFloat(output_dtype, compute_dtype, acc.*) + dtype_mod.castFloat(dtype, compute_dtype, value);
                                acc.* = dtype_mod.castFloat(compute_dtype, output_dtype, next);
                            },
                            .prod => acc.* *= value,
                            .int_sum => acc.* +%= intSumContribution(dtype, value),
                        }
                    }
                }
            }
        }
    };
}

/// Fold axis `axis_dim` of the contiguous `[outer, axis_dim, inner]` input
/// into `output` (`[outer, inner]`, pre-filled with the fold's identity).
/// Above the row-kernel work threshold the outer blocks split across the
/// pool, or the inner lanes when there are fewer blocks than workers and
/// the lanes are wide enough; otherwise one serial task.
fn foldAxisStreaming(
    ctx: *ExecContext,
    comptime dtype: DType,
    comptime fold: AxisFold,
    input: []const dtype_mod.Scalar(dtype),
    output: []AxisFoldTask(dtype, fold).Out,
    outer: usize,
    axis_dim: usize,
    inner: usize,
) void {
    const Task = AxisFoldTask(dtype, fold);
    const base_task: Task = .{
        .input = input,
        .output = output,
        .axis_dim = axis_dim,
        .inner = inner,
        .outer = outer,
    };
    if (input.len >= parallel.row_kernel_len_threshold) {
        const workers = parallel.cpuThreadCount(parallel.vector_max_threads);
        if (outer >= workers or inner < 2 * axis_fold_min_inner_lanes) {
            return ctx.forRange(outer, if (outer > 1) outer else 1, base_task, Task.runOuter);
        }
        return ctx.forRange(inner, inner / axis_fold_min_inner_lanes, base_task, Task.runInner);
    }
    Task.runOuter(base_task, 0, outer);
}

fn sumF32(ctx: *ExecContext, x: *const Tensor) !Tensor {
    var xx = try ctx.prepareContiguous(.f32, x);
    defer xx.deinit();

    var out = try ctx.scalar(.f32, 0);
    errdefer out.deinit();
    ctx.enableNativeVectorPoolForWork(xx.tensor().len(), parallel.vector_elementwise_len_threshold);
    try kernels.sumInto(ctx.pc(), &out, xx.tensor());
    return out;
}

/// Sum of all elements to a scalar tensor.
pub fn sum(ctx: *ExecContext, comptime dtype: DType, x: *const tensor.TensorOf(dtype)) !tensor.TensorOf(dtype_mod.outputDType(.reduction, dtype)) {
    if (comptime dtype == .f32) return sumF32(ctx, x);
    if (comptime isIntSum(dtype)) {
        var xx = try ctx.prepareContiguous(dtype, x);
        defer xx.deinit();
        return ctx.scalar(.i64, intSumSlice(dtype, xx.tensor().dataConst()));
    }
    comptime ensureForwardFloatMath(dtype);
    const output_dtype = comptime dtype_mod.outputDType(.reduction, dtype);

    var xx = try ctx.prepareContiguous(dtype, x);
    defer xx.deinit();

    ctx.enableNativeVectorPoolForWork(xx.tensor().len(), parallel.vector_elementwise_len_threshold);
    var out = try ctx.scalar(output_dtype, kernels.sumSliceTyped(dtype, xx.tensor().dataConst()));
    errdefer out.deinit();
    return out;
}

/// Sum along `axis`, the axis removed (`sum` is the all-elements form).
pub fn sumAxis(
    ctx: *ExecContext,
    comptime dtype: DType,
    comptime rank: usize,
    x: *const tensor.TensorOf(dtype),
    comptime axis: usize,
) !tensor.TensorOf(dtype_mod.outputDType(.reduction, dtype)) {
    if (comptime isIntSum(dtype)) return reduceAxis(ctx, dtype, .int_sum, rank, x, axis);
    comptime ensureForwardFloatMath(dtype);
    return reduceAxis(ctx, dtype, .sum, rank, x, axis);
}

/// The three layouts of an axis reduction, one body per fold: rank 1 is
/// the whole-tensor kernel, the last axis one row kernel per output, any
/// other axis the streaming `(outer, axis, inner)` fold. The output is
/// pre-filled with the fold's identity, which the streaming arm
/// accumulates into and the other arms overwrite.
fn reduceAxis(
    ctx: *ExecContext,
    comptime dtype: DType,
    comptime fold: AxisFold,
    comptime rank: usize,
    x: *const tensor.TensorOf(dtype),
    comptime axis: usize,
) !tensor.TensorOf(AxisFoldTask(dtype, fold).output_dtype) {
    const Task = AxisFoldTask(dtype, fold);
    const source = try x.rankView(rank);
    const out_rank = if (rank == 1) 1 else rank - 1;
    const out_shape = shapeWithoutAxis(rank, out_rank, source.shape, axis);

    var xx = try ctx.prepareContiguous(dtype, x);
    defer xx.deinit();
    const xp = xx.tensor();
    const input = xp.dataConst();

    var out = try ctx.full(Task.output_dtype, out_shape, Task.identity);
    errdefer out.deinit();
    const output = out.data();

    if (rank == 1) {
        try reduceWhole(ctx, dtype, fold, &out, xp);
        return out;
    }
    if (comptime axis == rank - 1) {
        const axis_dim = source.shape[axis];
        for (0..output.len) |row| output[row] = reduceRow(dtype, fold, input[row * axis_dim ..][0..axis_dim]);
        return out;
    }
    const g = shape_mod.AxisGeometry.of(rank, source.shape, axis);
    foldAxisStreaming(ctx, dtype, fold, input, output, g.outer, g.axis_dim, g.inner);
    return out;
}

/// The rank-1 arm of `reduceAxis`: one whole-tensor kernel into the scalar
/// output (the pooled SIMD kernels for the float folds).
fn reduceWhole(
    ctx: *ExecContext,
    comptime dtype: DType,
    comptime fold: AxisFold,
    out: *tensor.TensorOf(AxisFoldTask(dtype, fold).output_dtype),
    xp: *const tensor.TensorOf(dtype),
) !void {
    switch (comptime fold) {
        .int_sum => out.data()[0] = intSumSlice(dtype, xp.dataConst()),
        .sum => {
            ctx.enableNativeVectorPoolForWork(xp.len(), parallel.vector_elementwise_len_threshold);
            if (comptime dtype == .f32) {
                try kernels.sumInto(ctx.pc(), out, xp);
            } else {
                out.data()[0] = kernels.sumSliceTyped(dtype, xp.dataConst());
            }
        },
        .prod => {
            ctx.enableNativeVectorPoolForWork(xp.len(), parallel.vector_elementwise_len_threshold);
            try kernels.prodInto(ctx.pc(), out, xp);
        },
    }
}

/// The last-axis arm of `reduceAxis`: one row kernel per output element.
fn reduceRow(
    comptime dtype: DType,
    comptime fold: AxisFold,
    values: []const dtype_mod.Scalar(dtype),
) AxisFoldTask(dtype, fold).Out {
    return switch (comptime fold) {
        .int_sum => intSumSlice(dtype, values),
        .sum => if (comptime dtype == .f32) kernels.sumSlice(values) else kernels.sumSliceTyped(dtype, values),
        .prod => kernels.prodSlice(values),
    };
}

/// Cumulative sum along `axis` (torch.cumsum), preserving the input shape:
/// `out[..., i, ...] = Σ_{j <= i} x[..., j, ...]`. Default build: each row
/// is one serial prefix sum in axis order — bitwise deterministic for any
/// thread count (cold op; no parallel dispatch). With `-Dvector-scan` the
/// scan kernels vectorize (see `scanAxisRankDirected`).
pub fn cumsum(ctx: *ExecContext, comptime dtype: DType, comptime rank: usize, x: *const tensor.TensorOf(dtype), comptime axis: usize) !tensor.TensorOf(dtype) {
    return scanWidened(ctx, dtype, rank, x, axis, .sum, false, "cumsum");
}

/// Reversed cumulative (suffix) sum along `axis`:
/// `out[..., i, ...] = Σ_{j >= i} x[..., j, ...]` — the `cumsum` VJP
/// (a dedicated reverse pass, same determinism contract and the same
/// `-Dvector-scan` gating).
pub fn cumsumReverse(ctx: *ExecContext, comptime dtype: DType, comptime rank: usize, x: *const tensor.TensorOf(dtype), comptime axis: usize) !tensor.TensorOf(dtype) {
    return scanWidened(ctx, dtype, rank, x, axis, .sum, true, "cumsumReverse");
}

/// Cumulative product along `axis` (torch.cumprod), preserving the input
/// shape: `out[..., i, ...] = Π_{j <= i} x[..., j, ...]`. Same contract and
/// `-Dvector-scan` gating as `cumsum`.
pub fn cumprod(ctx: *ExecContext, comptime dtype: DType, comptime rank: usize, x: *const tensor.TensorOf(dtype), comptime axis: usize) !tensor.TensorOf(dtype) {
    return scanWidened(ctx, dtype, rank, x, axis, .prod, false, "cumprod");
}

const ScanOp = backend_mod.ops.ScanOp;

/// The scans share one f32 kernel; 16-bit inputs follow the `.widened`
/// policy.
fn scanWidened(ctx: *ExecContext, comptime dtype: DType, comptime rank: usize, x: *const tensor.TensorOf(dtype), comptime axis: usize, comptime op: ScanOp, comptime reverse: bool, comptime what: []const u8) !tensor.TensorOf(dtype) {
    const compute = comptime ExecContext.widenedCompute(dtype, what);
    var xx = try ctx.prepareAs(dtype, compute, x);
    defer xx.deinit();
    var out = try scanAxisRankDirected(ctx, rank, xx.tensor(), axis, op, reverse);
    errdefer out.deinit();
    return ctx.storeAs(compute, dtype, out);
}

// `scan_vector_width`/`ScanVec` for the linearRecurrence lane strips; the
// cumsum/cumprod scan kernels live in the backend (`kernels.scanRows`/
// `scanColumns`).
const scan_vector_width = 8;
const ScanVec = @Vector(scan_vector_width, f32);

/// Directed inclusive scan (sum or prod, forward or reverse) along `axis`.
///
/// Default build (`-Dvector-scan=false`): the documented serial-per-row
/// scan, bitwise deterministic and sequence-exact.
///
/// `-Dvector-scan=true`:
///   - non-last axes vectorize across `scan_vector_width` independent
///     columns per strip — each lane is one column's serial scan, so the
///     result is BITWISE IDENTICAL to the serial default;
///   - the last axis runs an in-register Hillis–Steele prefix scan per
///     row (log2(W) shifted combines + a running carry) — still bitwise
///     deterministic for any thread count, but the accumulation order
///     differs from the serial default (the sum-SIMD-lanes rounding
///     class; exact for integer-valued data).
fn scanAxisRankDirected(ctx: *ExecContext, comptime rank: usize, x: *const Tensor, comptime axis: usize, comptime op: ScanOp, comptime reverse: bool) !Tensor {
    if (axis >= rank) @compileError("axis out of bounds");

    const source = try x.rankView(rank);
    var xx = try ctx.prepareContiguous(.f32, x);
    defer xx.deinit();
    const input = xx.tensor().dataConst();

    var out = try ctx.empty(.f32, source.shape);
    errdefer out.deinit();
    const output = out.data();

    const axis_dim = source.shape[axis];
    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);

    // The `-Dvector-scan` gate lives inside the kernels: default builds run
    // their serial sequence-exact arms.
    if (inner == 1) {
        kernels.scanRows(op, reverse, input, output, outer, axis_dim);
    } else {
        for (0..outer) |outer_i| {
            const base = outer_i * axis_dim * inner;
            kernels.scanColumns(op, reverse, input[base..][0 .. axis_dim * inner], output[base..][0 .. axis_dim * inner], axis_dim, inner);
        }
    }
    return out;
}

/// First-order linear recurrence along `axis` (the associative-scan
/// primitive of the SSM / linear-attention family):
/// `h_t = a_t · h_{t-1} + b_t` per independent lane (each fixed choice of
/// the non-axis indices), with `h_{-1}` read from `initial` (or 0 when
/// absent). `b` supplies the output shape; `a` is a same-logical-shape
/// tensor — typically the facade's zero-stride broadcast alignment of a
/// lower-rank decay — read through its strides, never materialized.
/// `initial` holds one element per lane, row-major with the axis removed
/// (the shape-minus-axis layout).
///
/// Determinism: one serial pass per lane in axis order, each step evaluated
/// as multiply-then-add — bitwise deterministic for any thread count (cold
/// op; no parallel dispatch, the `cumsum` contract). With `-Dvector-scan`,
/// non-last axes vectorize across `scan_vector_width` independent lanes
/// when the decay's flattened lane strides are contiguous or fully
/// broadcast — every lane runs the identical elementwise op sequence, so
/// the result stays BITWISE IDENTICAL to the serial default. The last axis
/// stays serial per lane even under `-Dvector-scan`: an in-register form
/// would reassociate `a·h + b` and change rounding, so no gated variant
/// exists (unlike cumsum's last-axis prefix scan).
pub fn linearRecurrence(ctx: *ExecContext, comptime rank: usize, b: *const Tensor, a: *const Tensor, comptime axis: usize, initial: ?*const Tensor) !Tensor {
    if (axis >= rank) @compileError("axis out of bounds");

    const source = try b.rankView(rank);
    const a_view = try a.rankView(rank);
    inline for (0..rank) |d| {
        if (a_view.shape[d] != source.shape[d]) return tensor.TensorError.ShapeMismatch;
    }

    const axis_dim = source.shape[axis];
    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);

    var bb = try ctx.prepareContiguous(.f32, b);
    defer bb.deinit();
    const b_data = bb.tensor().dataConst();

    var init_prep: ?ExecContext.PreparedTensor = null;
    defer if (init_prep) |*p| p.deinit();
    var init_data: ?[]const f32 = null;
    if (initial) |ini| {
        if (ini.len() != outer * inner) return tensor.TensorError.ShapeMismatch;
        init_prep = try ctx.prepareContiguous(.f32, ini);
        init_data = init_prep.?.tensor().dataConst();
    }

    var out = try ctx.empty(.f32, source.shape);
    errdefer out.deinit();
    const output = out.data();

    const a_data = a.buffer.data;
    const a_axis_stride = a_view.strides[axis];
    const lane_kind = laneStrideKind(rank, axis, source.shape, a_view.strides);

    for (0..outer) |outer_i| {
        const a_outer_off = a.offset + strideOffset(rank, 0, axis, source.shape, a_view.strides, outer_i);
        var inner_i: usize = 0;

        if (comptime build_options.vector_scan) {
            if (comptime axis != rank - 1) {
                if (lane_kind != .general) {
                    while (inner_i + scan_vector_width <= inner) : (inner_i += scan_vector_width) {
                        var hv: ScanVec = if (init_data) |ini| ini[outer_i * inner + inner_i ..][0..scan_vector_width].* else @splat(0);
                        for (0..axis_dim) |t| {
                            const off = (outer_i * axis_dim + t) * inner + inner_i;
                            const av: ScanVec = switch (lane_kind) {
                                .broadcast => @splat(a_data[a_outer_off + t * a_axis_stride]),
                                .contiguous => a_data[a_outer_off + t * a_axis_stride + inner_i ..][0..scan_vector_width].*,
                                .general => unreachable,
                            };
                            hv = av * hv + @as(ScanVec, b_data[off..][0..scan_vector_width].*);
                            output[off..][0..scan_vector_width].* = hv;
                        }
                    }
                }
            }
        }

        while (inner_i < inner) : (inner_i += 1) {
            const a_off = a_outer_off + strideOffset(rank, axis + 1, rank, source.shape, a_view.strides, inner_i);
            var h: f32 = if (init_data) |ini| ini[outer_i * inner + inner_i] else 0;
            for (0..axis_dim) |t| {
                const off = (outer_i * axis_dim + t) * inner + inner_i;
                h = a_data[a_off + t * a_axis_stride] * h + b_data[off];
                output[off] = h;
            }
        }
    }
    return out;
}

pub const LinearRecurrenceGrads = struct {
    gb: Tensor,
    da: ?Tensor,
    dinitial: ?Tensor,
};

/// VJP of `linearRecurrence`. One reverse serial pass per lane:
/// `gh_t = a_{t+1} · gh_{t+1} + gy_t` (with `gh_T = 0`) — `gb` IS `gh`;
/// `da_t = gh_t · h_{t-1}` with `h_{-1}` = the lane's initial element (or
/// 0); `dinitial = a_0 · gh_0`. `a` is the forward's aligned decay view
/// (read through its strides), `h` the forward OUTPUT. `da` comes back at
/// the full `b` shape — the caller reduces it onto the decay's own shape
/// (the pointwise broadcast-backward rule). Same determinism contract and
/// `-Dvector-scan` lane vectorization as the forward (bitwise identical to
/// the serial pass either way).
pub fn linearRecurrenceBackward(
    ctx: *ExecContext,
    comptime rank: usize,
    gy: *const Tensor,
    a: *const Tensor,
    h: *const Tensor,
    initial: ?*const Tensor,
    comptime axis: usize,
    want_da: bool,
    want_dinitial: bool,
) !LinearRecurrenceGrads {
    if (axis >= rank) @compileError("axis out of bounds");

    const source = try gy.rankView(rank);
    const a_view = try a.rankView(rank);
    inline for (0..rank) |d| {
        if (a_view.shape[d] != source.shape[d]) return tensor.TensorError.ShapeMismatch;
    }

    const axis_dim = source.shape[axis];
    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);

    var gg = try ctx.prepareContiguous(.f32, gy);
    defer gg.deinit();
    const gy_data = gg.tensor().dataConst();
    var hh = try ctx.prepareContiguous(.f32, h);
    defer hh.deinit();
    const h_data = hh.tensor().dataConst();

    var init_prep: ?ExecContext.PreparedTensor = null;
    defer if (init_prep) |*p| p.deinit();
    var init_data: ?[]const f32 = null;
    if (initial) |ini| {
        if (ini.len() != outer * inner) return tensor.TensorError.ShapeMismatch;
        init_prep = try ctx.prepareContiguous(.f32, ini);
        init_data = init_prep.?.tensor().dataConst();
    }

    var gb = try ctx.empty(.f32, source.shape);
    errdefer gb.deinit();
    const gb_data = gb.data();

    var da: ?Tensor = if (want_da) try ctx.empty(.f32, source.shape) else null;
    errdefer if (da) |*t| t.deinit();
    const da_data: ?[]f32 = if (da) |*t| t.data() else null;

    const out_rank = if (rank == 1) 1 else rank - 1;
    var dinitial: ?Tensor = if (want_dinitial) try ctx.empty(.f32, shapeWithoutAxis(rank, out_rank, source.shape, axis)) else null;
    errdefer if (dinitial) |*t| t.deinit();
    const dinit_data: ?[]f32 = if (dinitial) |*t| t.data() else null;

    const a_data = a.buffer.data;
    const a_axis_stride = a_view.strides[axis];
    const lane_kind = laneStrideKind(rank, axis, source.shape, a_view.strides);

    for (0..outer) |outer_i| {
        const a_outer_off = a.offset + strideOffset(rank, 0, axis, source.shape, a_view.strides, outer_i);
        var inner_i: usize = 0;

        if (comptime build_options.vector_scan) {
            if (comptime axis != rank - 1) {
                if (lane_kind != .general) {
                    while (inner_i + scan_vector_width <= inner) : (inner_i += scan_vector_width) {
                        const loadA = struct {
                            inline fn at(kind: LaneStrideKind, data: []const f32, base: usize, stride: usize, t: usize, lane: usize) ScanVec {
                                return switch (kind) {
                                    .broadcast => @splat(data[base + t * stride]),
                                    .contiguous => data[base + t * stride + lane ..][0..scan_vector_width].*,
                                    .general => unreachable,
                                };
                            }
                        }.at;
                        var ghv: ScanVec = @splat(0);
                        var t = axis_dim;
                        while (t > 0) {
                            t -= 1;
                            const off = (outer_i * axis_dim + t) * inner + inner_i;
                            const gyv: ScanVec = gy_data[off..][0..scan_vector_width].*;
                            ghv = if (t + 1 == axis_dim) gyv else loadA(lane_kind, a_data, a_outer_off, a_axis_stride, t + 1, inner_i) * ghv + gyv;
                            gb_data[off..][0..scan_vector_width].* = ghv;
                            if (da_data) |dd| {
                                const hp: ScanVec = if (t > 0)
                                    h_data[off - inner ..][0..scan_vector_width].*
                                else if (init_data) |ini|
                                    ini[outer_i * inner + inner_i ..][0..scan_vector_width].*
                                else
                                    @splat(0);
                                dd[off..][0..scan_vector_width].* = ghv * hp;
                            }
                        }
                        if (dinit_data) |di| {
                            const dv = loadA(lane_kind, a_data, a_outer_off, a_axis_stride, 0, inner_i) * ghv;
                            di[outer_i * inner + inner_i ..][0..scan_vector_width].* = dv;
                        }
                    }
                }
            }
        }

        while (inner_i < inner) : (inner_i += 1) {
            const a_off = a_outer_off + strideOffset(rank, axis + 1, rank, source.shape, a_view.strides, inner_i);
            var gh: f32 = 0;
            var t = axis_dim;
            while (t > 0) {
                t -= 1;
                const off = (outer_i * axis_dim + t) * inner + inner_i;
                gh = if (t + 1 == axis_dim) gy_data[off] else a_data[a_off + (t + 1) * a_axis_stride] * gh + gy_data[off];
                gb_data[off] = gh;
                if (da_data) |dd| {
                    const h_prev: f32 = if (t > 0)
                        h_data[off - inner]
                    else if (init_data) |ini|
                        ini[outer_i * inner + inner_i]
                    else
                        0;
                    dd[off] = gh * h_prev;
                }
            }
            if (dinit_data) |di| {
                di[outer_i * inner + inner_i] = a_data[a_off] * gh;
            }
        }
    }
    return .{ .gb = gb, .da = da, .dinitial = dinitial };
}

const LaneStrideKind = enum { broadcast, contiguous, general };

/// Classify the decay view's flattened lane (inner-axes) access for the
/// `-Dvector-scan` arm: all-zero strides (one decay value per (outer, t)),
/// contiguous strides matching the row-major inner layout (lane offset ==
/// lane index), or anything else (scalar fallback).
fn laneStrideKind(comptime rank: usize, comptime axis: usize, shape: [rank]usize, strides: [rank]usize) LaneStrideKind {
    var expected: usize = 1;
    var contiguous = true;
    var broadcast = true;
    comptime var di: usize = 0;
    inline while (di < rank - axis - 1) : (di += 1) {
        const d = rank - 1 - di;
        if (strides[d] != 0) broadcast = false;
        if (strides[d] != expected) contiguous = false;
        expected *= shape[d];
    }
    if (broadcast) return .broadcast;
    if (contiguous) return .contiguous;
    return .general;
}

/// Strided offset of the `index`-th row-major position over axes
/// `[from, to)`: decomposes `index` (last axis fastest) and dots the
/// per-axis indices with `strides`.
fn strideOffset(comptime rank: usize, comptime from: usize, comptime to: usize, shape: [rank]usize, strides: [rank]usize, index: usize) usize {
    var offset: usize = 0;
    var rem = index;
    comptime var di: usize = 0;
    inline while (di < to - from) : (di += 1) {
        const d = to - 1 - di;
        offset += (rem % shape[d]) * strides[d];
        rem /= shape[d];
    }
    return offset;
}

/// Product along `axis` (torch.prod over a dim), the axis removed —
/// `sumAxis`'s structure at full parity: rank-1 reduces through the
/// pooled SIMD `prodInto`, a last-axis reduction runs one vectorized
/// `prodSlice` per row, and the general axis falls back to the same
/// delinearized scalar accumulation `sum` uses. Like `sum`, the SIMD
/// lane order fixes the float multiplication order per backend.
/// Product over `axis`. One f32 kernel; 16-bit inputs follow the
/// `.widened` policy and return f32 (the reduction output dtype).
pub fn prod(ctx: *ExecContext, comptime dtype: DType, comptime rank: usize, x: *const tensor.TensorOf(dtype), comptime axis: usize) !tensor.TensorOf(dtype_mod.outputDType(.reduction, dtype)) {
    const compute = comptime ExecContext.widenedCompute(dtype, "prod");
    var xx = try ctx.prepareAs(dtype, compute, x);
    defer xx.deinit();
    var out = try reduceAxis(ctx, .f32, .prod, rank, xx.tensor(), axis);
    errdefer out.deinit();
    return ctx.storeAs(compute, comptime dtype_mod.outputDType(.reduction, dtype), out);
}

fn meanAxisF32(ctx: *ExecContext, comptime rank: usize, x: *const Tensor, comptime axis: usize) !Tensor {
    var out = try sumAxis(ctx, .f32, rank, x, axis);
    errdefer out.deinit();
    out.scaleInPlace(1 / @as(f32, @floatFromInt(x.shape.at(axis))));
    return out;
}

// ---------------------------------------------------------------------------
// Masked reductions
//
// Fortran's array intrinsics all carry an optional `mask=`; these are its
// sum/mean twins. The point is fusion: the composed spelling
// (`where(mask, x, 0)` then `sum`) materializes a whole f32 copy of the input
// and walks it a second time, while these accumulate straight out of the
// source.
//
// Accumulation order is the unmasked path's, exactly: a last-axis reduction
// gathers the selected elements of one row into pooled row scratch (a single
// cache-resident row, not a copy of the tensor) and hands it to the same
// `sumSlice`/`sumInto` SIMD kernel, and the non-last-axis arm accumulates in
// linear order like its unmasked twin. An all-true mask therefore reproduces
// the unmasked reduction BITWISE, which is what the tests pin.
//
// Empty lanes are the other half of the Fortran contract. `sum(a, mask=…)`
// over a lane that selects nothing is the operation's identity (0), not an
// error and not a NaN — so a masked reduction has a defined answer where
// `maskedSelect` has to raise `EmptySelection`. `empty` overrides the identity
// where a caller wants a different sentinel; `mean` has no identity, so its
// default empty lane is NaN (0/0) unless one is supplied.
// ---------------------------------------------------------------------------

/// Per-lane mean plus the per-lane count of selected elements. The count is
/// the mean's divisor, and the VJP needs it, so the forward hands it back
/// rather than making the backward recount. The caller owns both.
pub const MaskedMeanResult = struct {
    values: Tensor,
    counts: Tensor,

    pub fn deinit(self: *MaskedMeanResult) void {
        self.values.deinit();
        self.counts.deinit();
        self.* = undefined;
    }
};

/// Sum over `axis` of the elements `mask` selects. `mask` must have `x`'s
/// shape (broadcasting happens above, at the tag layer). Lanes selecting
/// nothing yield `empty orelse 0`.
pub fn sumMasked(
    ctx: *ExecContext,
    comptime mask_dtype: DType,
    comptime rank: usize,
    x: *const Tensor,
    mask: *const tensor.TensorOf(mask_dtype),
    comptime axis: usize,
    empty_value: ?f32,
) !Tensor {
    var counts: ?Tensor = null;
    defer if (counts) |*c| c.deinit();
    return maskedReduce(ctx, mask_dtype, rank, x, mask, axis, empty_value, &counts, false);
}

/// Mean over `axis` of the elements `mask` selects, plus the per-lane
/// selected counts. Lanes selecting nothing yield `empty orelse NaN` (0/0 has
/// no identity to fall back on) and a count of zero.
pub fn meanMasked(
    ctx: *ExecContext,
    comptime mask_dtype: DType,
    comptime rank: usize,
    x: *const Tensor,
    mask: *const tensor.TensorOf(mask_dtype),
    comptime axis: usize,
    empty_value: ?f32,
) !MaskedMeanResult {
    var counts: ?Tensor = null;
    errdefer if (counts) |*c| c.deinit();
    var values = try maskedReduce(ctx, mask_dtype, rank, x, mask, axis, empty_value, &counts, true);
    errdefer values.deinit();
    const result: MaskedMeanResult = .{ .values = values, .counts = counts.? };
    counts = null;
    return result;
}

/// Number of selected elements in a flag row.
fn countRow(comptime mask_dtype: DType, flags: []const dtype_mod.Scalar(mask_dtype)) usize {
    var selected: usize = 0;
    for (flags) |keep| selected += @intFromBool(dtype_mod.isTruthy(mask_dtype, keep));
    return selected;
}

/// Shared masked sum/mean core. `counts_out` receives the per-lane selected
/// counts when `want_mean` is set (the mean divides by them) or when the
/// caller passed a non-null slot; the sum arm only needs them to detect empty
/// lanes, and drops them.
fn maskedReduce(
    ctx: *ExecContext,
    comptime mask_dtype: DType,
    comptime rank: usize,
    x: *const Tensor,
    mask: *const tensor.TensorOf(mask_dtype),
    comptime axis: usize,
    empty_value: ?f32,
    counts_out: *?Tensor,
    comptime want_mean: bool,
) !Tensor {
    if (axis >= rank) @compileError("axis out of bounds");

    const source = try x.rankView(rank);
    const mask_view = try mask.rankView(rank);
    for (0..rank) |dim| {
        if (source.shape[dim] != mask_view.shape[dim]) return tensor.TensorError.ShapeMismatch;
    }

    const out_rank = if (rank == 1) 1 else rank - 1;
    const out_shape = shapeWithoutAxis(rank, out_rank, source.shape, axis);

    var xx = try ctx.prepareContiguous(.f32, x);
    defer xx.deinit();
    const xp = xx.tensor();
    const input = xp.dataConst();

    var mm = try ctx.prepareContiguous(mask_dtype, mask);
    defer mm.deinit();
    const flags = mm.tensor().dataConst();

    var out = try ctx.zeros(.f32, out_shape);
    errdefer out.deinit();
    const output = out.data();

    var counts = try ctx.zeros(.f32, out_shape);
    errdefer counts.deinit();
    const count_data = counts.data();

    const axis_dim = source.shape[axis];
    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);

    // Both arms are branchless selects rather than skips. A data-dependent
    // branch (or a data-dependent store index) per element measured 2.8x
    // SLOWER than the two-pass composition it replaces on the non-last-axis
    // shape; a select vectorizes and keeps every access contiguous.
    //
    // Substituting the identity for an excluded element rather than skipping
    // it also makes the arithmetic EXACTLY the composed spelling's
    // (`maskedFill(mask, 0)` then reduce): same values, same kernel, same
    // order, so the fused entry is bitwise equal to the composition it
    // replaces, and bitwise equal to the unmasked reduction when the mask is
    // all-true. A select is NaN-correct where a multiply by 0/1 would not be:
    // an excluded NaN becomes the identity, it does not poison the lane.
    if (inner == 1) {
        // Last-axis (and rank-1) reduction: write one row through the mask
        // into pooled scratch (a single cache-resident row, not a copy of the
        // tensor) and hand it to the same SIMD kernel the unmasked path uses.
        var scratch = try ctx.empty(.f32, .{axis_dim});
        defer scratch.deinit();
        const row_scratch = scratch.data();
        ctx.enableNativeVectorPoolForWork(axis_dim, parallel.vector_elementwise_len_threshold);

        // Counting is a separate pass over the (now L1-resident) flag row, and
        // only when someone reads it: fusing a scalar `selected` accumulator
        // into the select loop serializes it and blocks vectorization.
        const need_counts = want_mean or empty_value != null;
        for (0..outer) |outer_i| {
            const row = input[outer_i * axis_dim ..][0..axis_dim];
            const row_flags = flags[outer_i * axis_dim ..][0..axis_dim];
            kernels.selectRow(mask_dtype, row, row_flags, row_scratch);
            output[outer_i] = kernels.sumSlice(row_scratch);
            if (need_counts) count_data[outer_i] = @floatFromInt(countRow(mask_dtype, row_flags));
        }
    } else {
        // Non-last axis: accumulate in linear order, exactly what the unmasked
        // non-last-axis arm does, with all four arrays walked contiguously.
        for (0..outer) |outer_i| {
            const base = outer_i * axis_dim * inner;
            const out_base = outer_i * inner;
            const out_row = output[out_base..][0..inner];
            const count_row = count_data[out_base..][0..inner];
            for (0..axis_dim) |a| {
                const in_row = input[base + a * inner ..][0..inner];
                const flag_row = flags[base + a * inner ..][0..inner];
                for (in_row, flag_row, out_row, count_row) |value, keep, *acc, *count| {
                    const on = dtype_mod.isTruthy(mask_dtype, keep);
                    acc.* += if (on) value else 0;
                    count.* += if (on) 1 else 0;
                }
            }
        }
    }

    // Empty lanes: the identity for a sum, NaN for a mean, or the caller's
    // sentinel for either.
    const resolved_empty = empty_value orelse if (want_mean) std.math.nan(f32) else @as(f32, 0);
    if (comptime want_mean) {
        for (output, count_data) |*value, count| {
            value.* = if (count == 0) resolved_empty else value.* / count;
        }
    } else if (empty_value != null) {
        for (output, count_data) |*value, count| {
            if (count == 0) value.* = resolved_empty;
        }
    }

    counts_out.* = counts;
    return out;
}

/// Mean along `axis`, the axis removed (`sumAxis` over the axis extent).
pub fn meanAxis(
    ctx: *ExecContext,
    comptime dtype: DType,
    comptime rank: usize,
    x: *const tensor.TensorOf(dtype),
    comptime axis: usize,
) !tensor.TensorOf(dtype_mod.outputDType(.reduction, dtype)) {
    if (comptime dtype == .f32) return meanAxisF32(ctx, rank, x, axis);
    comptime ensureForwardFloatMath(dtype);
    const compute_dtype = comptime dtype_mod.computeDType(.reduction, dtype);
    const output_dtype = comptime dtype_mod.outputDType(.reduction, dtype);
    var out = try sumAxis(ctx, dtype, rank, x, axis);
    errdefer out.deinit();
    const scale_value: dtype_mod.Scalar(compute_dtype) = 1 / @as(dtype_mod.Scalar(compute_dtype), @floatFromInt(x.shape.at(axis)));
    for (out.data()) |*value| {
        const scaled = dtype_mod.castFloat(output_dtype, compute_dtype, value.*) * scale_value;
        value.* = dtype_mod.castFloat(compute_dtype, output_dtype, scaled);
    }
    return out;
}

/// Segmented sum along `axis` over contiguous index ranges: `offsets` has
/// length `s + 1`, is nondecreasing, and spans the axis exactly
/// (`offsets[0] == 0`, `offsets[s] == shape[axis]`);
/// `out[..., i, ...] = Σ_{j in [offsets[i], offsets[i+1])} x[..., j, ...]`
/// and the axis size becomes `s` (empty segments produce zero rows). Each
/// segment accumulates serially in axis order — bitwise deterministic for
/// any thread count (cold op, no parallel dispatch). The sorted-contiguous
/// restriction of torch.segment_reduce / JAX segment_sum.
pub fn segmentSum(
    ctx: *ExecContext,
    comptime rank: usize,
    x: *const Tensor,
    comptime axis: usize,
    offsets: []const usize,
) !Tensor {
    if (axis >= rank) @compileError("axis out of bounds");
    const source = try x.rankView(rank);
    if (offsets.len < 2) return tensor.TensorError.InvalidShape;
    const segments = offsets.len - 1;
    if (offsets[0] != 0 or offsets[segments] != source.shape[axis]) return tensor.TensorError.InvalidShape;
    for (offsets[0..segments], offsets[1..]) |lo, hi| {
        if (lo > hi) return tensor.TensorError.InvalidShape;
    }

    var out_shape: [rank]usize = source.shape;
    out_shape[axis] = segments;

    var xx = try ctx.prepareContiguous(.f32, x);
    defer xx.deinit();
    const input = xx.tensor().dataConst();

    var out = try ctx.zeros(.f32, out_shape);
    errdefer out.deinit();
    const output = out.data();

    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);
    const n = source.shape[axis];
    for (0..outer) |o| {
        for (0..segments) |s_i| {
            const out_base = (o * segments + s_i) * inner;
            for (offsets[s_i]..offsets[s_i + 1]) |j| {
                const in_base = (o * n + j) * inner;
                for (0..inner) |c| output[out_base + c] += input[in_base + c];
            }
        }
    }
    return out;
}

/// `segmentSum`'s VJP companion: expand the segment axis back to
/// `n` rows, every input row receiving its segment's row:
/// `out[..., j, ...] = gy[..., seg(j), ...]`.
pub fn segmentBroadcast(
    ctx: *ExecContext,
    comptime rank: usize,
    gy: *const Tensor,
    comptime axis: usize,
    offsets: []const usize,
    n: usize,
) !Tensor {
    if (axis >= rank) @compileError("axis out of bounds");
    const source = try gy.rankView(rank);
    if (offsets.len < 2) return tensor.TensorError.InvalidShape;
    const segments = offsets.len - 1;
    if (source.shape[axis] != segments) return tensor.TensorError.ShapeMismatch;
    if (offsets[0] != 0 or offsets[segments] != n) return tensor.TensorError.InvalidShape;

    var out_shape: [rank]usize = source.shape;
    out_shape[axis] = n;

    var gg = try ctx.prepareContiguous(.f32, gy);
    defer gg.deinit();
    const input = gg.tensor().dataConst();

    var out = try ctx.zeros(.f32, out_shape);
    errdefer out.deinit();
    const output = out.data();

    const inner = productAfterAxis(rank, out_shape, axis);
    const outer = productBeforeAxis(rank, out_shape, axis);
    for (0..outer) |o| {
        for (0..segments) |s_i| {
            const in_base = (o * segments + s_i) * inner;
            for (offsets[s_i]..offsets[s_i + 1]) |j| {
                const out_base = (o * n + j) * inner;
                @memcpy(output[out_base..][0..inner], input[in_base..][0..inner]);
            }
        }
    }
    return out;
}
