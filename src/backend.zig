const std = @import("std");
const build_options = @import("build_options");
pub const ops = @import("backend/ops.zig");
pub const packed_matmul = @import("backend/packed.zig");
pub const quantized_matmul = @import("backend/quant.zig");
const dtype_mod = @import("dtype.zig");
const tensor = @import("tensor.zig");
const thread = @import("thread.zig");

pub const dtype_info = dtype_mod;
pub const DType = dtype_mod.DType;
pub const PackedMatmulFormat = packed_matmul.PackedMatmulFormat;
pub const PackedMatmulRhsFor = packed_matmul.PackedMatmulRhsFor;
pub const BlockQ1_0 = quantized_matmul.BlockQ1_0;
pub const BlockQ4_0 = quantized_matmul.BlockQ4_0;
pub const BlockQ4_1 = quantized_matmul.BlockQ4_1;
pub const BlockQ5_0 = quantized_matmul.BlockQ5_0;
pub const BlockQ5_1 = quantized_matmul.BlockQ5_1;
pub const BlockQ2_K = quantized_matmul.BlockQ2_K;
pub const BlockQ3_K = quantized_matmul.BlockQ3_K;
pub const BlockQ4_K = quantized_matmul.BlockQ4_K;
pub const BlockQ5_K = quantized_matmul.BlockQ5_K;
pub const BlockQ6_K = quantized_matmul.BlockQ6_K;
pub const BlockQ8_0 = quantized_matmul.BlockQ8_0;
pub const BlockQ8_1 = quantized_matmul.BlockQ8_1;
pub const BlockQ8_K = quantized_matmul.BlockQ8_K;
pub const BlockIQ1_S = quantized_matmul.BlockIQ1_S;
pub const BlockIQ1_M = quantized_matmul.BlockIQ1_M;
pub const BlockIQ2_XXS = quantized_matmul.BlockIQ2_XXS;
pub const BlockIQ2_XS = quantized_matmul.BlockIQ2_XS;
pub const BlockIQ2_S = quantized_matmul.BlockIQ2_S;
pub const BlockIQ3_XXS = quantized_matmul.BlockIQ3_XXS;
pub const BlockIQ3_S = quantized_matmul.BlockIQ3_S;
pub const BlockIQ4_NL = quantized_matmul.BlockIQ4_NL;
pub const BlockIQ4_XS = quantized_matmul.BlockIQ4_XS;
pub const BlockTQ1_0 = quantized_matmul.BlockTQ1_0;
pub const BlockTQ2_0 = quantized_matmul.BlockTQ2_0;
pub const BlockMXFP4 = quantized_matmul.BlockMXFP4;
pub const BlockNVFP4 = quantized_matmul.BlockNVFP4;
pub const QuantizedMatmulFormat = quantized_matmul.QuantizedMatmulFormat;
pub const supports_q4_k_mmla = quantized_matmul.supports_q4_k_mmla;
pub const PackedRhsLayout = quantized_matmul.PackedRhsLayout;
pub const PackedRhsFor = quantized_matmul.PackedRhsFor;
pub const QuantizedMatmulRhs = quantized_matmul.QuantizedMatmulRhs;
pub const QuantizedMatmulRhsI8 = quantized_matmul.QuantizedMatmulRhsI8;
pub const QuantizedMatmulRhsQ1_0 = quantized_matmul.QuantizedMatmulRhsQ1_0;
pub const QuantizedMatmulRhsQ4_0 = quantized_matmul.QuantizedMatmulRhsQ4_0;
pub const QuantizedMatmulRhsQ4_1 = quantized_matmul.QuantizedMatmulRhsQ4_1;
pub const QuantizedMatmulRhsQ5_0 = quantized_matmul.QuantizedMatmulRhsQ5_0;
pub const QuantizedMatmulRhsQ5_1 = quantized_matmul.QuantizedMatmulRhsQ5_1;
pub const QuantizedMatmulRhsQ2_K = quantized_matmul.QuantizedMatmulRhsQ2_K;
pub const QuantizedMatmulRhsQ3_K = quantized_matmul.QuantizedMatmulRhsQ3_K;
pub const QuantizedMatmulRhsQ4_K = quantized_matmul.QuantizedMatmulRhsQ4_K;
pub const QuantizedMatmulRhsQ4_Kx4 = quantized_matmul.QuantizedMatmulRhsQ4_Kx4;
pub const QuantizedMatmulRhsQ4_Kx8 = quantized_matmul.QuantizedMatmulRhsQ4_Kx8;
pub const QuantizedMatmulRhsQ4_Kx2Mmla = quantized_matmul.QuantizedMatmulRhsQ4_Kx2Mmla;
pub const QuantizedMatmulRhsQ5_K = quantized_matmul.QuantizedMatmulRhsQ5_K;
pub const QuantizedMatmulRhsQ5_Kx8 = quantized_matmul.QuantizedMatmulRhsQ5_Kx8;
pub const QuantizedMatmulRhsQ6_K = quantized_matmul.QuantizedMatmulRhsQ6_K;
pub const QuantizedMatmulRhsQ6_Kx4 = quantized_matmul.QuantizedMatmulRhsQ6_Kx4;
pub const QuantizedMatmulRhsQ8_0 = quantized_matmul.QuantizedMatmulRhsQ8_0;
pub const QuantizedMatmulRhsQ8_0x4 = quantized_matmul.QuantizedMatmulRhsQ8_0x4;
pub const QuantizedMatmulRhsIQ1_S = quantized_matmul.QuantizedMatmulRhsIQ1_S;
pub const QuantizedMatmulRhsIQ1_M = quantized_matmul.QuantizedMatmulRhsIQ1_M;
pub const QuantizedMatmulRhsIQ2_XXS = quantized_matmul.QuantizedMatmulRhsIQ2_XXS;
pub const QuantizedMatmulRhsIQ2_XS = quantized_matmul.QuantizedMatmulRhsIQ2_XS;
pub const QuantizedMatmulRhsIQ2_S = quantized_matmul.QuantizedMatmulRhsIQ2_S;
pub const QuantizedMatmulRhsIQ3_XXS = quantized_matmul.QuantizedMatmulRhsIQ3_XXS;
pub const QuantizedMatmulRhsIQ3_S = quantized_matmul.QuantizedMatmulRhsIQ3_S;
pub const QuantizedMatmulRhsIQ4_NL = quantized_matmul.QuantizedMatmulRhsIQ4_NL;
pub const QuantizedMatmulRhsIQ4_XS = quantized_matmul.QuantizedMatmulRhsIQ4_XS;
pub const QuantizedMatmulRhsTQ1_0 = quantized_matmul.QuantizedMatmulRhsTQ1_0;
pub const QuantizedMatmulRhsTQ2_0 = quantized_matmul.QuantizedMatmulRhsTQ2_0;
pub const QuantizedMatmulRhsMXFP4 = quantized_matmul.QuantizedMatmulRhsMXFP4;
pub const QuantizedMatmulRhsNVFP4 = quantized_matmul.QuantizedMatmulRhsNVFP4;
pub const AnyQuantizedMatmulRhs = quantized_matmul.AnyQuantizedMatmulRhs;
pub const QuantizedRowsQ4_0 = quantized_matmul.QuantizedRowsQ4_0;
pub const QuantizedRowsQ8_0 = quantized_matmul.QuantizedRowsQ8_0;
pub const PackedMatmulRhsI8 = QuantizedMatmulRhsI8;
pub const Tensor = tensor.Tensor;
pub const TensorOf = tensor.TensorOf;
pub const ThreadPool = thread.Pool;
pub const scalar_impl = @import("backend/cpu.zig");
pub const cpu_impl = scalar_impl;
pub const native_impl = @import("backend/native.zig");
// GPU GEMM provider selected by -Dgpu (metal.zig or cuda.zig, via the
// backend/gpu.zig leaf); inert (never analyzed past the `enabled` flag) on
// -Dgpu=none builds.
pub const gpu_impl = @import("backend/gpu.zig").impl;
// Pure-Zig vector kernels backing the native backend, exported so the GEMM
// bench can compare the row-kernel and blocked paths directly.
pub const vector_impl = @import("backend/vector.zig");

/// conv2d geometry (channel-last [H,W,Cin] -> [OH,OW,Cout]); see vector/conv.zig.
pub const Conv2dDims = vector_impl.Conv2dDims;
/// conv1d geometry (general non-causal [T,Cin] -> [T_out,Cout]); see vector/conv.zig.
pub const Conv1dDims = vector_impl.Conv1dDims;
/// pool2d geometry + kind (channel-last [H,W,C] -> [OH,OW,C]); see vector/pool.zig.
pub const PoolKind = vector_impl.PoolKind;
pub const Pool2dDims = vector_impl.Pool2dDims;
/// Winograd F(2×2,3×3) transform geometry; see vector/winograd.zig.
pub const WinogradF2Dims = vector_impl.WinogradF2Dims;

pub const Kind = enum {
    scalar,
    native,
};

pub const active_kind: Kind = switch (build_options.backend_kind) {
    .scalar, .cpu => .scalar,
    .native => .native,
};

pub const native_blas_kind = build_options.blas_kind;
pub const native_uses_blas = build_options.use_blas;
pub const native_uses_accelerate = build_options.blas_kind == .accelerate;
pub const native_blas_threads = build_options.blas_threads;

const active = switch (build_options.backend_kind) {
    .scalar, .cpu => scalar_impl,
    .native => native_impl,
};

const ParallelConfig = active.ParallelConfig;

pub const Backend = struct {
    pub const kind = active_kind;
    // Atomic: kernels may dispatch on other threads (e.g. dot-backward's
    // OneShotWorker) while a lazy tryWorkPool retry publishes the pool.
    // release/acquire so a racing first observer also sees Pool.init's writes.
    parallel_pool: std.atomic.Value(?*thread.Pool) = .init(null),

    pub fn init() Backend {
        return .{};
    }

    pub fn setWorkPool(self: *Backend, pool: ?*thread.Pool) void {
        self.parallel_pool.store(pool, .release);
    }

    fn parallelConfig(self: *const Backend) ParallelConfig {
        return .{ .pool = self.parallel_pool.load(.acquire) };
    }

    pub fn addInto(self: *const Backend, out: *Tensor, a: *const Tensor, b: *const Tensor) !void {
        _ = self;
        return active.addInto(out, a, b);
    }

    pub fn addContiguousIntoUnchecked(
        self: *const Backend,
        out: *Tensor,
        a: *const Tensor,
        b: *const Tensor,
        len: usize,
    ) void {
        return active.addContiguousIntoUncheckedWithConfig(out, a, b, len, self.parallelConfig());
    }

    pub fn maximumContiguousIntoUnchecked(self: *const Backend, out: *Tensor, a: *const Tensor, b: *const Tensor, len: usize) void {
        return active.maximumContiguousIntoUncheckedWithConfig(out, a, b, len, self.parallelConfig());
    }

    pub fn minimumContiguousIntoUnchecked(self: *const Backend, out: *Tensor, a: *const Tensor, b: *const Tensor, len: usize) void {
        return active.minimumContiguousIntoUncheckedWithConfig(out, a, b, len, self.parallelConfig());
    }

    pub fn subInto(self: *const Backend, out: *Tensor, a: *const Tensor, b: *const Tensor) !void {
        _ = self;
        return active.subInto(out, a, b);
    }

    pub fn subContiguousIntoUnchecked(
        self: *const Backend,
        out: *Tensor,
        a: *const Tensor,
        b: *const Tensor,
        len: usize,
    ) void {
        return active.subContiguousIntoUncheckedWithConfig(out, a, b, len, self.parallelConfig());
    }

    pub fn mulInto(self: *const Backend, out: *Tensor, a: *const Tensor, b: *const Tensor) !void {
        _ = self;
        return active.mulInto(out, a, b);
    }

    pub fn mulContiguousIntoUnchecked(
        self: *const Backend,
        out: *Tensor,
        a: *const Tensor,
        b: *const Tensor,
        len: usize,
    ) void {
        return active.mulContiguousIntoUncheckedWithConfig(out, a, b, len, self.parallelConfig());
    }

    pub fn elementwiseContiguousIntoTyped(
        self: *const Backend,
        comptime dtype: DType,
        comptime op: ops.ElementwiseOp,
        out: *tensor.TensorOf(dtype_mod.outputDType(.pointwise, dtype)),
        a: *const tensor.TensorOf(dtype),
        b: *const tensor.TensorOf(dtype),
        len: usize,
    ) void {
        return active.elementwiseContiguousIntoTypedWithConfig(dtype, op, out, a, b, len, self.parallelConfig());
    }

    pub fn scaleInto(self: *const Backend, out: *Tensor, a: *const Tensor, scalar_value: f32) !void {
        return active.scaleIntoWithConfig(out, a, scalar_value, self.parallelConfig());
    }

    pub fn addScaledSliceUnchecked(self: *const Backend, z: []f32, x: []const f32, scalar_value: f32) void {
        _ = self;
        return active.addScaledSlice(z, x, scalar_value);
    }

    pub fn addRowVectorSliceUnchecked(self: *const Backend, z: []f32, row_vector: []const f32, rows: usize, cols: usize) void {
        _ = self;
        return active.addRowVectorSlice(z, row_vector, rows, cols);
    }

    pub fn addRowVectorUnarySliceUnchecked(self: *const Backend, comptime op: ops.UnaryOp, z: []f32, row_vector: []const f32, rows: usize, cols: usize) void {
        _ = self;
        return active.addRowVectorUnarySlice(op, z, row_vector, rows, cols);
    }

    pub fn causalDepthwiseConv1dInto(
        self: *const Backend,
        out: *Tensor,
        input: *const Tensor,
        kernel: *const Tensor,
        state: ?[]const f32,
        seq: usize,
        channels: usize,
        taps: usize,
    ) void {
        return active.causalDepthwiseConv1dIntoWithConfig(out, input, kernel, state, seq, channels, taps, self.parallelConfig());
    }

    pub fn causalDepthwiseConv1dBackwardInputInto(
        self: *const Backend,
        out: *Tensor,
        gy: *const Tensor,
        kernel: *const Tensor,
        seq: usize,
        channels: usize,
        taps: usize,
    ) void {
        return active.causalDepthwiseConv1dBackwardInputIntoWithConfig(out, gy, kernel, seq, channels, taps, self.parallelConfig());
    }

    pub fn causalDepthwiseConv1dBackwardKernelInto(
        self: *const Backend,
        out: *Tensor,
        input: *const Tensor,
        gy: *const Tensor,
        state: ?[]const f32,
        seq: usize,
        channels: usize,
        taps: usize,
    ) void {
        return active.causalDepthwiseConv1dBackwardKernelIntoWithConfig(out, input, gy, state, seq, channels, taps, self.parallelConfig());
    }

    pub fn causalConv1dInto(
        self: *const Backend,
        out: *Tensor,
        input: *const Tensor,
        weight: *const Tensor,
        state: ?[]const f32,
        seq: usize,
        in_channels: usize,
        out_channels: usize,
        taps: usize,
        dilation: usize,
    ) void {
        return active.causalConv1dIntoWithConfig(out, input, weight, state, seq, in_channels, out_channels, taps, dilation, self.parallelConfig());
    }

    pub fn conv2dInto(
        self: *const Backend,
        out: *Tensor,
        input: *const Tensor,
        weight: *const Tensor,
        bias: ?[]const f32,
        dims: Conv2dDims,
    ) void {
        return active.conv2dIntoWithConfig(out, input, weight, bias, dims, self.parallelConfig());
    }

    pub fn conv2dBackwardInputInto(
        self: *const Backend,
        out: *Tensor,
        gy: *const Tensor,
        weight: *const Tensor,
        dims: Conv2dDims,
    ) void {
        return active.conv2dBackwardInputIntoWithConfig(out, gy, weight, dims, self.parallelConfig());
    }

    pub fn conv2dBackwardWeightInto(
        self: *const Backend,
        out: *Tensor,
        input: *const Tensor,
        gy: *const Tensor,
        dims: Conv2dDims,
    ) void {
        return active.conv2dBackwardWeightIntoWithConfig(out, input, gy, dims, self.parallelConfig());
    }

    pub fn im2colInto(
        self: *const Backend,
        col: *Tensor,
        input: *const Tensor,
        dims: Conv2dDims,
    ) void {
        return active.im2colIntoWithConfig(col, input, dims, self.parallelConfig());
    }

    pub fn col2imInto(
        self: *const Backend,
        out: *Tensor,
        col: *const Tensor,
        dims: Conv2dDims,
    ) void {
        return active.col2imIntoWithConfig(out, col, dims, self.parallelConfig());
    }

    pub fn winogradF2WeightTransformInto(self: *const Backend, u: *const [16][]f32, w: []const f32, cout: usize, cin: usize) void {
        return active.winogradF2WeightTransformIntoWithConfig(u, w, cout, cin, self.parallelConfig());
    }

    pub fn winogradF2InputTransformInto(self: *const Backend, v: *const [16][]f32, x: []const f32, dims: WinogradF2Dims) void {
        return active.winogradF2InputTransformIntoWithConfig(v, x, dims, self.parallelConfig());
    }

    pub fn winogradF2OutputTransformInto(self: *const Backend, y: []f32, m: *const [16][]const f32, bias: ?[]const f32, fuse_relu: bool, dims: WinogradF2Dims) void {
        return active.winogradF2OutputTransformIntoWithConfig(y, m, bias, fuse_relu, dims, self.parallelConfig());
    }

    pub fn winogradF4WeightTransformInto(self: *const Backend, u: *const [36][]f32, w: []const f32, cout: usize, cin: usize) void {
        return active.winogradF4WeightTransformIntoWithConfig(u, w, cout, cin, self.parallelConfig());
    }

    pub fn winogradF4InputTransformInto(self: *const Backend, v: *const [36][]f32, x: []const f32, dims: WinogradF2Dims) void {
        return active.winogradF4InputTransformIntoWithConfig(v, x, dims, self.parallelConfig());
    }

    pub fn winogradF4OutputTransformInto(self: *const Backend, y: []f32, m: *const [36][]const f32, bias: ?[]const f32, fuse_relu: bool, dims: WinogradF2Dims) void {
        return active.winogradF4OutputTransformIntoWithConfig(y, m, bias, fuse_relu, dims, self.parallelConfig());
    }

    pub fn pool2dInto(
        self: *const Backend,
        comptime pool_kind: PoolKind,
        out: *Tensor,
        input: *const Tensor,
        dims: Pool2dDims,
    ) void {
        return active.pool2dIntoWithConfig(pool_kind, out, input, dims, self.parallelConfig());
    }

    pub fn avgPool2dBackwardInto(
        self: *const Backend,
        out: *Tensor,
        gy: *const Tensor,
        dims: Pool2dDims,
    ) void {
        return active.avgPool2dBackwardIntoWithConfig(out, gy, dims, self.parallelConfig());
    }

    pub fn maxPool2dBackwardInto(
        self: *const Backend,
        out: *Tensor,
        input: *const Tensor,
        gy: *const Tensor,
        dims: Pool2dDims,
    ) void {
        return active.maxPool2dBackwardIntoWithConfig(out, input, gy, dims, self.parallelConfig());
    }

    pub fn upsample2xNearestInto(
        self: *const Backend,
        out: *Tensor,
        input: *const Tensor,
        h: usize,
        w: usize,
        c: usize,
    ) void {
        return active.upsample2xNearestIntoWithConfig(out, input, h, w, c, self.parallelConfig());
    }

    pub fn preluChannelsIntoUnchecked(self: *const Backend, z: []f32, x: []const f32, alpha: []const f32, rows: usize, cols: usize) void {
        return active.preluChannelsIntoWithConfig(z, x, alpha, rows, cols, self.parallelConfig());
    }

    pub fn preluChannelsBackwardInputIntoUnchecked(self: *const Backend, gx: []f32, gy: []const f32, x: []const f32, alpha: []const f32, rows: usize, cols: usize) void {
        return active.preluChannelsBackwardInputIntoWithConfig(gx, gy, x, alpha, rows, cols, self.parallelConfig());
    }

    pub fn preluChannelsBackwardAlphaIntoUnchecked(self: *const Backend, galpha: []f32, gy: []const f32, x: []const f32, rows: usize, cols: usize) void {
        return active.preluChannelsBackwardAlphaIntoWithConfig(galpha, gy, x, rows, cols, self.parallelConfig());
    }

    pub fn channelAffineIntoUnchecked(self: *const Backend, z: []f32, x: []const f32, scale: []const f32, shift: ?[]const f32, rows: usize, cols: usize) void {
        return active.channelAffineIntoWithConfig(z, x, scale, shift, rows, cols, self.parallelConfig());
    }

    pub fn conv1dInto(
        self: *const Backend,
        out: *Tensor,
        input: *const Tensor,
        weight: *const Tensor,
        dims: Conv1dDims,
    ) void {
        return active.conv1dIntoWithConfig(out, input, weight, dims, self.parallelConfig());
    }

    pub fn conv1dBackwardInputInto(
        self: *const Backend,
        out: *Tensor,
        gy: *const Tensor,
        weight: *const Tensor,
        dims: Conv1dDims,
    ) void {
        return active.conv1dBackwardInputIntoWithConfig(out, gy, weight, dims, self.parallelConfig());
    }

    pub fn conv1dBackwardWeightInto(
        self: *const Backend,
        out: *Tensor,
        input: *const Tensor,
        gy: *const Tensor,
        dims: Conv1dDims,
    ) void {
        return active.conv1dBackwardWeightIntoWithConfig(out, input, gy, dims, self.parallelConfig());
    }

    pub fn col2im1dInto(
        self: *const Backend,
        out: *Tensor,
        col: *const Tensor,
        t_in: usize,
        out_len: usize,
        out_channels: usize,
        taps: usize,
        stride: usize,
        pad: usize,
    ) void {
        return active.col2im1dIntoWithConfig(out, col, t_in, out_len, out_channels, taps, stride, pad, self.parallelConfig());
    }

    pub fn col2im1dBackwardInto(
        self: *const Backend,
        out: *Tensor,
        gy: *const Tensor,
        t_in: usize,
        gy_len: usize,
        out_channels: usize,
        taps: usize,
        stride: usize,
        pad: usize,
    ) void {
        return active.col2im1dBackwardIntoWithConfig(out, gy, t_in, gy_len, out_channels, taps, stride, pad, self.parallelConfig());
    }

    pub fn snakeInto(
        self: *const Backend,
        out: *Tensor,
        x: *const Tensor,
        alpha: []const f32,
        inv_b: []const f32,
        rows: usize,
        cols: usize,
    ) void {
        return active.snakeIntoWithConfig(out, x, alpha, inv_b, rows, cols, self.parallelConfig());
    }

    pub fn snakeBackwardInputInto(
        self: *const Backend,
        out: *Tensor,
        x: *const Tensor,
        gy: *const Tensor,
        alpha: []const f32,
        inv_b: []const f32,
        rows: usize,
        cols: usize,
    ) void {
        return active.snakeBackwardInputIntoWithConfig(out, x, gy, alpha, inv_b, rows, cols, self.parallelConfig());
    }

    pub fn snakeBackwardParamsInto(
        self: *const Backend,
        galpha: *Tensor,
        ginv_b: *Tensor,
        x: *const Tensor,
        gy: *const Tensor,
        alpha: []const f32,
        inv_b: []const f32,
        rows: usize,
        cols: usize,
    ) void {
        return active.snakeBackwardParamsIntoWithConfig(galpha, ginv_b, x, gy, alpha, inv_b, rows, cols, self.parallelConfig());
    }

    pub fn groupNormInto(
        self: *const Backend,
        out: *Tensor,
        x: *const Tensor,
        weight: ?[]const f32,
        bias: ?[]const f32,
        rows: usize,
        cols: usize,
        groups: usize,
        eps: f32,
    ) void {
        return active.groupNormIntoWithConfig(out, x, weight, bias, rows, cols, groups, eps, self.parallelConfig());
    }

    pub fn groupNormBackwardInto(
        self: *const Backend,
        gx: ?*Tensor,
        gw: ?*Tensor,
        gb: ?*Tensor,
        x: *const Tensor,
        gy: *const Tensor,
        weight: ?[]const f32,
        rows: usize,
        cols: usize,
        groups: usize,
        eps: f32,
    ) void {
        return active.groupNormBackwardIntoWithConfig(gx, gw, gb, x, gy, weight, rows, cols, groups, eps, self.parallelConfig());
    }

    pub fn causalConv1dBackwardInputInto(
        self: *const Backend,
        out: *Tensor,
        gy: *const Tensor,
        weight: *const Tensor,
        seq: usize,
        in_channels: usize,
        out_channels: usize,
        taps: usize,
        dilation: usize,
    ) void {
        return active.causalConv1dBackwardInputIntoWithConfig(out, gy, weight, seq, in_channels, out_channels, taps, dilation, self.parallelConfig());
    }

    pub fn causalConv1dBackwardWeightInto(
        self: *const Backend,
        out: *Tensor,
        input: *const Tensor,
        gy: *const Tensor,
        state: ?[]const f32,
        seq: usize,
        in_channels: usize,
        out_channels: usize,
        taps: usize,
        dilation: usize,
    ) void {
        return active.causalConv1dBackwardWeightIntoWithConfig(out, input, gy, state, seq, in_channels, out_channels, taps, dilation, self.parallelConfig());
    }

    pub fn groupedCausalConv1dInto(
        self: *const Backend,
        out: *Tensor,
        input: *const Tensor,
        weight: *const Tensor,
        state: ?[]const f32,
        seq: usize,
        in_channels: usize,
        out_channels: usize,
        taps: usize,
        dilation: usize,
        groups: usize,
    ) void {
        return active.groupedCausalConv1dIntoWithConfig(out, input, weight, state, seq, in_channels, out_channels, taps, dilation, groups, self.parallelConfig());
    }

    pub fn groupedCausalConv1dBackwardInputInto(
        self: *const Backend,
        out: *Tensor,
        gy: *const Tensor,
        weight: *const Tensor,
        seq: usize,
        in_channels: usize,
        out_channels: usize,
        taps: usize,
        dilation: usize,
        groups: usize,
    ) void {
        return active.groupedCausalConv1dBackwardInputIntoWithConfig(out, gy, weight, seq, in_channels, out_channels, taps, dilation, groups, self.parallelConfig());
    }

    pub fn groupedCausalConv1dBackwardWeightInto(
        self: *const Backend,
        out: *Tensor,
        input: *const Tensor,
        gy: *const Tensor,
        state: ?[]const f32,
        seq: usize,
        in_channels: usize,
        out_channels: usize,
        taps: usize,
        dilation: usize,
        groups: usize,
    ) void {
        return active.groupedCausalConv1dBackwardWeightIntoWithConfig(out, input, gy, state, seq, in_channels, out_channels, taps, dilation, groups, self.parallelConfig());
    }

    pub fn unaryContiguousIntoUnchecked(
        self: *const Backend,
        comptime op: ops.UnaryOp,
        out: *Tensor,
        a: *const Tensor,
        len: usize,
    ) void {
        return active.unaryContiguousIntoUncheckedWithConfig(op, out, a, len, self.parallelConfig());
    }

    pub fn leakyReluContiguousIntoUnchecked(
        self: *const Backend,
        out: *Tensor,
        a: *const Tensor,
        len: usize,
        negative_slope: f32,
    ) void {
        return active.leakyReluContiguousIntoUncheckedWithConfig(out, a, len, negative_slope, self.parallelConfig());
    }

    pub fn clampContiguousIntoUnchecked(
        self: *const Backend,
        out: *Tensor,
        a: *const Tensor,
        len: usize,
        min_value: f32,
        max_value: f32,
    ) void {
        return active.clampContiguousIntoUncheckedWithConfig(out, a, len, min_value, max_value, self.parallelConfig());
    }

    pub fn gatedContiguousIntoUnchecked(
        self: *const Backend,
        comptime op: ops.GatedOp,
        out: *Tensor,
        a: *const Tensor,
        b: *const Tensor,
        len: usize,
    ) void {
        return active.gatedContiguousIntoUncheckedWithConfig(op, out, a, b, len, self.parallelConfig());
    }

    pub fn sumInto(self: *const Backend, out: *Tensor, a: *const Tensor) !void {
        return active.sumIntoWithConfig(out, a, self.parallelConfig());
    }

    pub fn sumSlice(self: *const Backend, values: []const f32) f32 {
        _ = self;
        return active.sumSlice(values);
    }

    pub fn prodInto(self: *const Backend, out: *Tensor, a: *const Tensor) !void {
        return active.prodIntoWithConfig(out, a, self.parallelConfig());
    }

    pub fn prodSlice(self: *const Backend, values: []const f32) f32 {
        _ = self;
        return active.prodSlice(values);
    }

    pub fn sumSliceTyped(
        self: *const Backend,
        comptime dtype: DType,
        values: []const dtype_mod.Scalar(dtype),
    ) dtype_mod.Scalar(dtype_mod.outputDType(.reduction, dtype)) {
        return active.sumSliceTypedWithConfig(dtype, values, self.parallelConfig());
    }

    pub fn dotInto(self: *const Backend, out: *Tensor, a: *const Tensor, b: *const Tensor) !void {
        return active.dotIntoWithConfig(out, a, b, self.parallelConfig());
    }

    pub fn dotIntoTyped(
        self: *const Backend,
        comptime dtype: DType,
        out: *tensor.TensorOf(dtype_mod.outputDType(.matmul, dtype)),
        a: *const tensor.TensorOf(dtype),
        b: *const tensor.TensorOf(dtype),
    ) !void {
        return active.dotIntoTypedWithConfig(dtype, out, a, b, self.parallelConfig());
    }

    pub fn matmulInto(self: *const Backend, out: *Tensor, a: *const Tensor, b: *const Tensor) !void {
        _ = self;
        return active.matmulInto(out, a, b);
    }

    pub fn matmul2DIntoUnchecked(
        self: *const Backend,
        out: *Tensor,
        a: *const Tensor,
        b: *const Tensor,
        m: usize,
        n: usize,
        k: usize,
    ) void {
        return active.matmul2DIntoUncheckedWithConfig(out, a, b, m, n, k, self.parallelConfig());
    }

    pub fn matmul2DIntoUncheckedTyped(
        self: *const Backend,
        comptime dtype: DType,
        out: *tensor.TensorOf(dtype_mod.outputDType(.matmul, dtype)),
        a: *const tensor.TensorOf(dtype),
        b: *const tensor.TensorOf(dtype),
        m: usize,
        n: usize,
        k: usize,
    ) void {
        return active.matmul2DIntoUncheckedTypedWithConfig(dtype, out, a, b, m, n, k, self.parallelConfig());
    }

    pub fn packMatmulRhsTyped(
        self: *const Backend,
        comptime dtype: DType,
        allocator: std.mem.Allocator,
        rhs: *const tensor.TensorOf(dtype),
    ) !packed_matmul.PackedMatmulRhsFor(dtype) {
        _ = self;
        return active.packMatmulRhsTyped(dtype, allocator, rhs);
    }

    pub fn matmul2DIntoUncheckedPackedRhsTyped(
        self: *const Backend,
        comptime dtype: DType,
        allocator: std.mem.Allocator,
        out: *tensor.TensorOf(dtype_mod.outputDType(.matmul, dtype)),
        a: *const tensor.TensorOf(dtype),
        rhs: *const packed_matmul.PackedMatmulRhsFor(dtype),
        m: usize,
        n: usize,
        k: usize,
    ) !void {
        return active.matmul2DIntoUncheckedPackedRhsTypedWithConfig(dtype, allocator, out, a, rhs, m, n, k, self.parallelConfig());
    }

    pub fn quantizeMatmulRhsBlockwiseI8(
        self: *const Backend,
        allocator: std.mem.Allocator,
        rhs: *const Tensor,
        group_size: usize,
    ) !quantized_matmul.QuantizedMatmulRhsI8 {
        _ = self;
        return active.quantizeMatmulRhsBlockwiseI8(allocator, rhs, group_size);
    }

    pub fn quantizeMatmulRhsQ4_0(
        self: *const Backend,
        allocator: std.mem.Allocator,
        rhs: *const Tensor,
    ) !quantized_matmul.QuantizedMatmulRhsQ4_0 {
        _ = self;
        return active.quantizeMatmulRhsQ4_0(allocator, rhs);
    }

    pub fn quantizeMatmulRhsQ8_0(
        self: *const Backend,
        allocator: std.mem.Allocator,
        rhs: *const Tensor,
    ) !quantized_matmul.QuantizedMatmulRhsQ8_0 {
        _ = self;
        return active.quantizeMatmulRhsQ8_0(allocator, rhs);
    }

    pub fn supportsQuantizedMatmulRhs(self: *const Backend, format: quantized_matmul.QuantizedMatmulFormat) bool {
        _ = self;
        return quantized_matmul.supportsMatmul(format);
    }

    pub fn matmul2DQuantizedRhs(
        self: *const Backend,
        allocator: std.mem.Allocator,
        out: *Tensor,
        a: *const Tensor,
        rhs: quantized_matmul.AnyQuantizedMatmulRhs,
        m: usize,
        n: usize,
        k: usize,
    ) !void {
        return active.matmul2DQuantizedRhsWithConfig(allocator, out, a, rhs, m, n, k, self.parallelConfig());
    }

    pub fn matmul2DQuantizedRhsQ8_0x4(
        self: *const Backend,
        allocator: std.mem.Allocator,
        out: *Tensor,
        a: *const Tensor,
        rhs: *const quantized_matmul.QuantizedMatmulRhsQ8_0x4,
        m: usize,
        n: usize,
        k: usize,
    ) !void {
        return active.matmul2DQuantizedRhsQ8_0x4WithConfig(allocator, out, a, rhs, m, n, k, self.parallelConfig());
    }

    pub fn matmul2DPackedQ8_0x4LhsRhs(
        self: *const Backend,
        out: *Tensor,
        lhs_blocks: []const quantized_matmul.BlockQ8_0x4,
        rhs: *const quantized_matmul.QuantizedMatmulRhsQ8_0x4,
        m: usize,
        n: usize,
        k: usize,
    ) !void {
        return active.matmul2DPackedQ8_0x4LhsRhsWithConfig(out, lhs_blocks, rhs, m, n, k, self.parallelConfig());
    }

    pub fn matmulPackedQ4_Kx8Q8_Kx4Slice(self: *const Backend, out: []f32, lhs_blocks: []const quantized_matmul.BlockQ8_Kx4, rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_Kx8, m: usize, n: usize, k: usize) void {
        return active.matmulPackedQ4_Kx8Q8_Kx4SliceWithConfig(out, lhs_blocks, rhs, m, n, k, self.parallelConfig());
    }

    pub fn matmulPackedQ4_Kx8RowsSlice(self: *const Backend, out: []f32, lhs_blocks: []const quantized_matmul.BlockQ8_K, rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_Kx8, m: usize, n: usize, k: usize) void {
        return active.matmulPackedQ4_Kx8RowsSliceWithConfig(out, lhs_blocks, rhs, m, n, k, self.parallelConfig());
    }

    pub fn matmulPackedQ5_Kx8Q8_Kx4Slice(self: *const Backend, out: []f32, lhs_blocks: []const quantized_matmul.BlockQ8_Kx4, rhs: *const quantized_matmul.QuantizedMatmulRhsQ5_Kx8, m: usize, n: usize, k: usize) void {
        return active.matmulPackedQ5_Kx8Q8_Kx4SliceWithConfig(out, lhs_blocks, rhs, m, n, k, self.parallelConfig());
    }

    pub fn matmulPackedQ5_Kx8RowsSlice(self: *const Backend, out: []f32, lhs_blocks: []const quantized_matmul.BlockQ8_K, rhs: *const quantized_matmul.QuantizedMatmulRhsQ5_Kx8, m: usize, n: usize, k: usize) void {
        return active.matmulPackedQ5_Kx8RowsSliceWithConfig(out, lhs_blocks, rhs, m, n, k, self.parallelConfig());
    }

    pub fn matmulPackedQ6_Kx4RowsSlice(self: *const Backend, out: []f32, lhs_blocks: []const quantized_matmul.BlockQ8_K, rhs: *const quantized_matmul.QuantizedMatmulRhsQ6_Kx4, m: usize, n: usize, k: usize) void {
        return active.matmulPackedQ6_Kx4RowsSliceWithConfig(out, lhs_blocks, rhs, m, n, k, self.parallelConfig());
    }

    pub fn unaryRowSliceUnchecked(self: *const Backend, comptime op: ops.UnaryOp, z: []f32, x: []const f32) void {
        _ = self;
        return active.unaryRowSlice(op, z, x);
    }

    pub fn mulRowSliceUnchecked(self: *const Backend, z: []f32, x: []const f32, y: []const f32) void {
        _ = self;
        return active.mulRowSlice(z, x, y);
    }

    pub fn matmul2DPackedPaddedQ8_0x4LhsRhs(
        self: *const Backend,
        out: *Tensor,
        lhs_blocks: []const quantized_matmul.BlockQ8_0x4,
        rhs: *const quantized_matmul.QuantizedMatmulRhsQ8_0x4,
        m: usize,
        n: usize,
        k: usize,
    ) !void {
        return active.matmul2DPackedPaddedQ8_0x4LhsRhsWithConfig(out, lhs_blocks, rhs, m, n, k, self.parallelConfig());
    }

    pub fn matmul2DQuantizedRhsQ6_Kx4(
        self: *const Backend,
        allocator: std.mem.Allocator,
        out: *Tensor,
        a: *const Tensor,
        rhs: *const quantized_matmul.QuantizedMatmulRhsQ6_Kx4,
        m: usize,
        n: usize,
        k: usize,
    ) !void {
        return active.matmul2DQuantizedRhsQ6_Kx4WithConfig(allocator, out, a, rhs, m, n, k, self.parallelConfig());
    }

    pub fn matmul2DQuantizedRhsQ4_Kx4(
        self: *const Backend,
        allocator: std.mem.Allocator,
        out: *Tensor,
        a: *const Tensor,
        rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_Kx4,
        m: usize,
        n: usize,
        k: usize,
    ) !void {
        return active.matmul2DQuantizedRhsQ4_Kx4WithConfig(allocator, out, a, rhs, m, n, k, self.parallelConfig());
    }

    pub fn matmul2DQuantizedRhsQ4_Kx8(
        self: *const Backend,
        allocator: std.mem.Allocator,
        out: *Tensor,
        a: *const Tensor,
        rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_Kx8,
        m: usize,
        n: usize,
        k: usize,
    ) !void {
        return active.matmul2DQuantizedRhsQ4_Kx8WithConfig(allocator, out, a, rhs, m, n, k, self.parallelConfig());
    }

    pub fn matmul2DQuantizedRhsQ4_Kx2Mmla(
        self: *const Backend,
        allocator: std.mem.Allocator,
        out: *Tensor,
        a: *const Tensor,
        rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_Kx2Mmla,
        m: usize,
        n: usize,
        k: usize,
    ) !void {
        return active.matmul2DQuantizedRhsQ4_Kx2MmlaWithConfig(allocator, out, a, rhs, m, n, k, self.parallelConfig());
    }

    pub fn matmul2DQuantizedRhsQ5_Kx8(
        self: *const Backend,
        allocator: std.mem.Allocator,
        out: *Tensor,
        a: *const Tensor,
        rhs: *const quantized_matmul.QuantizedMatmulRhsQ5_Kx8,
        m: usize,
        n: usize,
        k: usize,
    ) !void {
        return active.matmul2DQuantizedRhsQ5_Kx8WithConfig(allocator, out, a, rhs, m, n, k, self.parallelConfig());
    }

    pub fn matmulTransAInto(self: *const Backend, out: *Tensor, a: *const Tensor, b: *const Tensor) !void {
        _ = self;
        return active.matmulTransAInto(out, a, b);
    }

    pub fn matmulTransA2DIntoUnchecked(
        self: *const Backend,
        out: *Tensor,
        a: *const Tensor,
        b: *const Tensor,
        m: usize,
        n: usize,
        k: usize,
    ) void {
        return active.matmulTransA2DIntoUncheckedWithConfig(out, a, b, m, n, k, self.parallelConfig());
    }

    pub fn matmulTransBInto(self: *const Backend, out: *Tensor, a: *const Tensor, b: *const Tensor) !void {
        _ = self;
        return active.matmulTransBInto(out, a, b);
    }

    pub fn matmulTransB2DIntoUnchecked(
        self: *const Backend,
        out: *Tensor,
        a: *const Tensor,
        b: *const Tensor,
        m: usize,
        n: usize,
        k: usize,
    ) void {
        return active.matmulTransB2DIntoUncheckedWithConfig(out, a, b, m, n, k, self.parallelConfig());
    }

    pub fn matmulTransB2DIntoUncheckedF16Operands(
        self: *const Backend,
        out: *Tensor,
        a: *const tensor.TensorOf(.f16),
        b: *const tensor.TensorOf(.f16),
        m: usize,
        n: usize,
        k: usize,
    ) void {
        return active.matmulTransB2DIntoUncheckedF16OperandsWithConfig(out, a, b, m, n, k, self.parallelConfig());
    }

    pub fn matmulTransB2DIntoUncheckedBf16Rhs(
        self: *const Backend,
        out: *Tensor,
        a: *const Tensor,
        b: *const tensor.TensorOf(.bf16),
        m: usize,
        n: usize,
        k: usize,
    ) void {
        return active.matmulTransB2DIntoUncheckedBf16RhsWithConfig(out, a, b, m, n, k, self.parallelConfig());
    }

    pub fn matmulBatched2DIntoUnchecked(
        self: *const Backend,
        out: *Tensor,
        a: *const Tensor,
        b: *const Tensor,
        m: usize,
        n: usize,
        k: usize,
        batch_count: usize,
        stride_a: usize,
        stride_b: usize,
        stride_c: usize,
    ) void {
        return active.matmulBatched2DIntoUncheckedWithConfig(
            out,
            a,
            b,
            m,
            n,
            k,
            batch_count,
            stride_a,
            stride_b,
            stride_c,
            self.parallelConfig(),
        );
    }

    pub fn matmulBatchedTransA2DIntoUnchecked(
        self: *const Backend,
        out: *Tensor,
        a: *const Tensor,
        b: *const Tensor,
        m: usize,
        n: usize,
        k: usize,
        batch_count: usize,
        stride_a: usize,
        stride_b: usize,
        stride_c: usize,
    ) void {
        return active.matmulBatchedTransA2DIntoUncheckedWithConfig(
            out,
            a,
            b,
            m,
            n,
            k,
            batch_count,
            stride_a,
            stride_b,
            stride_c,
            self.parallelConfig(),
        );
    }

    pub fn matmulBatchedTransB2DIntoUnchecked(
        self: *const Backend,
        out: *Tensor,
        a: *const Tensor,
        b: *const Tensor,
        m: usize,
        n: usize,
        k: usize,
        batch_count: usize,
        stride_a: usize,
        stride_b: usize,
        stride_c: usize,
    ) void {
        return active.matmulBatchedTransB2DIntoUncheckedWithConfig(
            out,
            a,
            b,
            m,
            n,
            k,
            batch_count,
            stride_a,
            stride_b,
            stride_c,
            self.parallelConfig(),
        );
    }
};

test {
    _ = @import("backend_tests.zig");
    _ = @import("backend/parity_test.zig");
    if (comptime build_options.use_gpu) {
        _ = @import("backend/gpu.zig"); // forwards to the active provider's tests
    }
}
