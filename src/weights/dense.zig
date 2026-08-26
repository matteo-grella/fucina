//! Dense and packed-quantized weight containers: the f32/f16/bf16 weight
//! tensors, the four packed formats (`WeightQ4_K`/`WeightQ5_K`/
//! `WeightQ6_K`/`WeightQ8_0` — one comptime body, four instantiations),
//! their GGUF loaders, and the packed linear forward (`linearSeq`) with
//! its GPU offload try and decode-compact route.

const std = @import("std");

const dtype_mod = @import("../dtype.zig");
const backend_mod = @import("../backend.zig");
const exec_mod = @import("../exec.zig");
const ag_mod = @import("../ag.zig");
const gguf = @import("../gguf.zig");
const tuning = @import("../tuning.zig");

const common = @import("common.zig");
const gpu = @import("gpu.zig");
const host = @import("host.zig");

const offload = backend_mod.offload;
const Tensor = ag_mod.Tensor;
const PackedRhs = ag_mod.PackedRhs;
const DType = dtype_mod.DType;
const ExecContext = exec_mod.ExecContext;
const RhsLifetime = exec_mod.RhsLifetime;
const Error = common.Error;
const QuantWeight = common.QuantWeight;
const LoadOptions = common.LoadOptions;
const blockSlice = common.blockSlice;
const BlockStorage = common.BlockStorage;
const Tag = @TypeOf(.tag);

pub const WeightF32 = Tensor(.{ .out, .in });
pub const WeightF16 = Tensor(.{ .dtype = .f16, .tags = .{ .out, .in } });
pub const WeightBf16 = Tensor(.{ .dtype = .bf16, .tags = .{ .out, .in } });
/// Packed quantized linear weight: the raw block tensor plus its
/// column-interleaved packed RHS, built once at init. One comptime body
/// serves every packed format — the four public names below are distinct
/// instantiations, so `LinearWeight`'s union arms stay nominal and every
/// change to the ownership/residency contract lands in all four at once.
fn PackedQuantWeight(comptime dt: DType) type {
    return struct {
        value: QuantWeight(dt),
        packed_rhs: PackedRhs(dt),
        rhs_lifetime: RhsLifetime = .transient,

        const Self = @This();

        /// The container's block format; `weights.linearSeq` infers its
        /// route from this.
        pub const dtype: DType = dt;

        pub fn init(ctx: *ExecContext, value: QuantWeight(dt)) !Self {
            return initWithRhsLifetime(ctx, value, .transient);
        }

        pub fn initWithRhsLifetime(ctx: *ExecContext, value: QuantWeight(dt), rhs_lifetime: RhsLifetime) !Self {
            var owned = value;
            errdefer owned.deinit();
            var packed_rhs = try owned.packRhs(ctx);
            errdefer packed_rhs.deinit();
            return .{ .value = owned, .packed_rhs = packed_rhs, .rhs_lifetime = rhs_lifetime };
        }

        pub fn deinit(self: *Self) void {
            self.packed_rhs.deinit();
            self.value.deinit();
            self.* = undefined;
        }

        pub fn cloneView(self: *const Self, ctx: *ExecContext) !Self {
            const value = try self.value.withTags(ctx, .{ .out, .in });
            return initWithRhsLifetime(ctx, value, self.rhs_lifetime);
        }

        pub fn concat(self: *const Self, ctx: *ExecContext, comptime tag: Tag, others: []const *const Self) !Self {
            var raw_others = try ctx.allocator.alloc(*const QuantWeight(dt), others.len);
            defer ctx.allocator.free(raw_others);
            for (others, 0..) |other, i| raw_others[i] = &other.value;

            var value = try self.value.concat(ctx, tag, raw_others);
            var owns_value = true;
            errdefer if (owns_value) value.deinit();
            const rhs_lifetime: RhsLifetime = if (try gpu.makeGpuResidentQuantWeight(dt, ctx, &value)) .stable_process else .transient;
            return initWithRhsLifetime(ctx, value, rhs_lifetime) catch |err| {
                owns_value = false;
                return err;
            };
        }
    };
}

pub const WeightQ4_K = PackedQuantWeight(.q4_k);
pub const WeightQ5_K = PackedQuantWeight(.q5_k);
pub const WeightQ6_K = PackedQuantWeight(.q6_k);
pub const WeightQ8_0 = PackedQuantWeight(.q8_0);

/// Try the dense quantized GPU matmul: `out = in · dequant(W)ᵀ` over the raw
/// GGUF blocks (`weight.value`), via the provider's dequant-in-kernel
/// GEMM. Returns null — caller falls back to the CPU packed path — when the GPU
/// is off, the input needs gradients (training), or the exec gate declines
/// (shape/work threshold). Comptime-elided on non-gpu builds.
fn denseQuantGpuTry(
    comptime dtype: DType,
    weight: anytype,
    ctx: *ExecContext,
    input: anytype,
    comptime in_tag: Tag,
    comptime out_tag: Tag,
) !?Tensor(.{ .seq, out_tag }) {
    if (comptime !offload.supportsQuantDType(dtype)) return null;
    if (input.requiresGrad()) return null;
    const m = input.dim(.seq);
    const k = input.dim(in_tag);
    const n = weight.value.dim(.out);
    if (weight.value.dim(.in) != k) return null;
    const wraw = weight.value.asRawTensor();
    if (!wraw.isContiguous()) return null;
    const wbytes = std.mem.sliceAsBytes(wraw.dataConst());
    const nb01 = std.math.divExact(usize, wbytes.len, n) catch return null;
    var out = (try ctx.tryMatmulQuantRhs(dtype, wbytes, weight.rhs_lifetime, nb01, input.asRawTensor(), m, n, k)) orelse return null;
    errdefer out.deinit();
    return try Tensor(.{ .seq, out_tag }).fromTensor(ctx, out);
}

/// Programmatic override for the decode-route gate
/// (`tuning.Table.decode_compact`, where the bandwidth argument lives):
/// `true`/`false` force the compact/packed route, `null` restores the
/// env/default value. The tests' A/B hook; also usable from a CLI flag.
pub fn setDecodeCompact(on: ?bool) void {
    tuning.setField("decode_compact", on);
}

/// Formats whose decode-shape GEMV wins by reading the compact GGUF-native
/// blocks instead of the byte-expanded packed layout (the
/// `tuning.Table.decode_compact` byte ratios). Q8_0's x4 pack carries the
/// same bytes as its compact blocks, so it goes straight to the packed dot.
fn hasCompactDecodeRoute(comptime dt: DType) bool {
    return switch (dt) {
        .q4_k, .q5_k, .q6_k => true,
        .q8_0 => false,
        else => false,
    };
}

/// Packed quantized linear forward, one body for every
/// `PackedQuantWeight` format: the GPU offload try first, then (where the
/// format has one) the decode-shape compact route, then the packed CPU
/// dot. Grad inputs keep the packed path's explicit
/// GradientQuantizedMatmulUnsupported error.
pub fn linearSeq(
    weight: anytype,
    ctx: *ExecContext,
    input: anytype,
    comptime in_tag: Tag,
    comptime out_tag: Tag,
) !Tensor(.{ .seq, out_tag }) {
    const dt = @TypeOf(weight.*).dtype;
    if (try denseQuantGpuTry(dt, weight, ctx, input, in_tag, out_tag)) |r| return r;
    // Decode shapes: contract against the resident GGUF-native compact
    // blocks (`weight.value`) through the public quantized-RHS `dot` —
    // bitwise-equal outputs, fewer weight bytes streamed (see the
    // `tuning.Table.decode_compact` doc). The GPU try stays first so `-Dgpu`
    // builds keep their offload policy ahead of the CPU route choice.
    if (comptime hasCompactDecodeRoute(dt)) {
        if (input.dim(.seq) < 4 and !input.requiresGrad() and tuning.get().decode_compact) {
            var tagged = try weight.value.withTags(ctx, .{ out_tag, in_tag });
            defer tagged.deinit();
            return input.dot(ctx, &tagged, in_tag);
        }
    }
    return input.dotPacked(ctx, &weight.packed_rhs, in_tag, out_tag);
}

pub fn loadDenseF32Weight(ctx: *ExecContext, info: *const gguf.TensorInfo, shape: [2]usize) !WeightF32 {
    var w = try WeightF32.empty(ctx, shape);
    errdefer w.deinit();
    try host.fillF32(try w.data(), info); // validates info.data.len against the shape
    return w;
}

pub fn loadDenseF16Weight(ctx: *ExecContext, info: *const gguf.TensorInfo, shape: [2]usize, options: LoadOptions) !WeightF16 {
    if (info.ggml_type != .f16) return Error.UnsupportedWeightType;
    const len = try std.math.mul(usize, shape[0], shape[1]);
    if (info.data.len != len * @sizeOf(u16)) return Error.InvalidWeightShape;

    // GPU builds: the weight lives in managed/shared resident bytes, so the
    // f16 GEMM offload uses it with zero per-call transfer (registry hit;
    // this path never adopt-copies) while the bytes stay CPU-readable and
    // in-place-trainable. Fallback: plain heap storage.
    if (comptime offload.enabled) {
        if (options.gpu_resident) {
            if (offload.allocResidentBytes(info.data.len)) |dev| {
                @memcpy(dev, info.data);
                return gpu.gpuResidentDenseTensor(.f16, WeightF16, ctx, shape, dev);
            }
        }
    }

    var w = try WeightF16.empty(ctx, shape);
    errdefer w.deinit();
    for (try w.data(), 0..) |*dst, i| {
        const bits = std.mem.readInt(u16, info.data[i * 2 ..][0..2], .little);
        dst.* = @bitCast(bits);
    }
    return w;
}

/// bf16 stays RESIDENT (2 B/weight, like llama.cpp): the linearSeq fast path
/// streams the raw bits through the mixed f32 x bf16 TransB kernel
/// (`matmulHalfRhs(.bf16)`), which widens in-register (u16 << 16, exact)
/// and accumulates in f32. `Scalar(.bf16)` is the raw `u16` bit pattern;
/// the facade elements are `dtype.Bf16` values over the same bits.
pub fn loadDenseBf16Weight(ctx: *ExecContext, info: *const gguf.TensorInfo, shape: [2]usize) !WeightBf16 {
    if (info.ggml_type != .bf16) return Error.UnsupportedWeightType;
    const len = try std.math.mul(usize, shape[0], shape[1]);
    if (info.data.len != len * @sizeOf(u16)) return Error.InvalidWeightShape;

    var w = try WeightBf16.empty(ctx, shape);
    errdefer w.deinit();
    for (try w.data(), 0..) |*dst, i| {
        dst.* = .{ .bits = std.mem.readInt(u16, info.data[i * 2 ..][0..2], .little) };
    }
    return w;
}

pub fn loadQuantizedWeight(comptime dtype: DType, ctx: *ExecContext, info: *const gguf.TensorInfo, shape: [2]usize) !QuantWeight(dtype) {
    const Elem = BlockStorage(dtype);
    const blocks = try blockSlice(Elem, info.data);
    return QuantWeight(dtype).fromBlocks(ctx, shape, blocks);
}

fn LoadedQuantWeight(comptime dtype: DType) type {
    return struct {
        value: QuantWeight(dtype),
        rhs_lifetime: RhsLifetime,
    };
}

fn loadGpuResidentQuantizedWeight(comptime dtype: DType, ctx: *ExecContext, info: *const gguf.TensorInfo, shape: [2]usize, options: LoadOptions) !LoadedQuantWeight(dtype) {
    const Elem = BlockStorage(dtype);
    const blocks = try blockSlice(Elem, info.data);
    if (comptime offload.supportsQuantDType(dtype)) {
        if (options.gpu_resident) {
            if (offload.allocResidentBytes(info.data.len)) |dev| {
                @memcpy(dev, info.data);
                return .{ .value = try gpu.gpuResidentQuantTensor(dtype, ctx, shape, dev), .rhs_lifetime = .stable_process };
            }
        }
    }
    return .{ .value = try QuantWeight(dtype).fromBlocks(ctx, shape, blocks), .rhs_lifetime = .transient };
}

pub fn loadQ6_KWeight(ctx: *ExecContext, info: *const gguf.TensorInfo, shape: [2]usize, options: LoadOptions) !WeightQ6_K {
    const loaded = try loadGpuResidentQuantizedWeight(.q6_k, ctx, info, shape, options);
    return WeightQ6_K.initWithRhsLifetime(ctx, loaded.value, loaded.rhs_lifetime);
}

pub fn loadQ4_KWeight(ctx: *ExecContext, info: *const gguf.TensorInfo, shape: [2]usize, options: LoadOptions) !WeightQ4_K {
    const loaded = try loadGpuResidentQuantizedWeight(.q4_k, ctx, info, shape, options);
    return WeightQ4_K.initWithRhsLifetime(ctx, loaded.value, loaded.rhs_lifetime);
}

pub fn loadQ5_KWeight(ctx: *ExecContext, info: *const gguf.TensorInfo, shape: [2]usize, options: LoadOptions) !WeightQ5_K {
    const loaded = try loadGpuResidentQuantizedWeight(.q5_k, ctx, info, shape, options);
    return WeightQ5_K.initWithRhsLifetime(ctx, loaded.value, loaded.rhs_lifetime);
}

pub fn loadQ8_0Weight(ctx: *ExecContext, info: *const gguf.TensorInfo, shape: [2]usize, options: LoadOptions) !WeightQ8_0 {
    const loaded = try loadGpuResidentQuantizedWeight(.q8_0, ctx, info, shape, options);
    return WeightQ8_0.initWithRhsLifetime(ctx, loaded.value, loaded.rhs_lifetime);
}
