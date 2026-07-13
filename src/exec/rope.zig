//! Rotary position embedding (RoPE) for the eager runtime.
//!
//! Home of `RopeTable`/`RopeMode` (re-exported as `exec.RopeTable`/`exec.RopeMode`
//! for the autograd VJP params) plus the forward rope ops. `sinValues`/`cosValues`
//! are `pub` so `norm.zig`'s fused rms-norm+rope kernel can read them cross-module.
//!
//! Domain module: every op receives an explicit `*Runtime`; imports the runtime
//! + shape leaves; never imports `exec.zig`.

const std = @import("std");
const tensor = @import("../tensor.zig");

const exec_shape = @import("shape.zig");
const Runtime = @import("runtime.zig").Runtime;

const Allocator = std.mem.Allocator;
const Tensor = tensor.Tensor;

const contiguousStridesArray = exec_shape.contiguousStridesArray;

pub const RopeMode = enum {
    interleaved,
    half,
    /// Adjacent (interleaved) pairing over the TRAILING `table.feature_dim`
    /// features: a partial rotary aligned to the end of the feature axis,
    /// with the leading features passed through unchanged (DeepSeek V4's
    /// tail-64 rotary). Identical to `.interleaved` when the table spans the
    /// whole feature axis.
    interleaved_tail,
};

/// On-the-fly RoPE factor source for the unified facade `rope`: positions +
/// theta base, full rotation only. Production paths prepare a `RopeTable`
/// instead (freq_factors/NTK scaling live there).
pub const RopeTheta = struct {
    positions: []const i32,
    theta_base: f32,
};

pub const RopeTable = struct {
    allocator: Allocator,
    positions: []i32,
    theta_base: f32,
    feature_dim: usize,
    pair_count: usize,
    values: []f32,

    pub fn deinit(self: *RopeTable) void {
        self.allocator.free(self.values);
        self.allocator.free(self.positions);
        self.* = undefined;
    }

    pub fn sinValues(self: *const RopeTable) []const f32 {
        const angle_count = self.positions.len * self.pair_count;
        return self.values[0..angle_count];
    }

    pub fn cosValues(self: *const RopeTable) []const f32 {
        const angle_count = self.positions.len * self.pair_count;
        return self.values[angle_count..][0..angle_count];
    }
};

pub fn ropeAxisRank(
    rt: *Runtime,
    comptime rank: usize,
    x: *const Tensor,
    comptime position_axis: usize,
    comptime feature_axis: usize,
    positions: []const i32,
    theta_base: f32,
    comptime mode: RopeMode,
    comptime inverse: bool,
) !Tensor {
    const source = try x.rankView(rank);
    const feature_dim = source.shape[feature_axis];
    var table = try prepareRopeTable(rt, positions, feature_dim, theta_base, inverse);
    defer table.deinit();
    return ropeAxisRankWithTable(rt, rank, x, position_axis, feature_axis, &table, mode);
}

pub fn prepareRopeTable(rt: *Runtime, positions: []const i32, feature_dim: usize, theta_base: f32, inverse: bool) !RopeTable {
    return prepareRopeTableFactors(rt, positions, feature_dim, theta_base, inverse, null);
}

/// As `prepareRopeTable`, but with optional per-pair `freq_factors` (length
/// `feature_dim/2`) that scale each rotary frequency: the angle for pair `i`
/// becomes `(pos / theta_base^(2i/d)) / freq_factors[i]`. This is ggml's
/// `rope_ext` `freq_factors` (a.k.a. proportional / NTK-by-part RoPE): Llama-3
/// long-context scaling and Gemma's global ("full attention") layers both
/// supply it. `freq_factors == null` reproduces plain RoPE exactly.
pub fn prepareRopeTableFactors(
    rt: *Runtime,
    positions: []const i32,
    feature_dim: usize,
    theta_base: f32,
    inverse: bool,
    freq_factors: ?[]const f32,
) !RopeTable {
    if (feature_dim % 2 != 0) return tensor.TensorError.InvalidShape;
    const pair_count = feature_dim / 2;
    if (freq_factors) |ff| {
        if (ff.len != pair_count) return tensor.TensorError.ShapeMismatch;
    }
    const angle_count = try std.math.mul(usize, positions.len, pair_count);
    const values = try rt.allocator.alloc(f32, try std.math.mul(usize, angle_count, 2));
    errdefer rt.allocator.free(values);
    const positions_copy = try rt.allocator.dupe(i32, positions);
    errdefer rt.allocator.free(positions_copy);

    const sin_values = values[0..angle_count];
    const cos_values = values[angle_count..][0..angle_count];
    const sign: f32 = if (inverse) -1 else 1;
    // theta_base^(2i/d) is position-invariant, so hoist the pow; the
    // freq_factors divide must stay per-element — folding it into the cache
    // changes f32 rounding ((pos/a)/b != pos/(a*b)).
    const pow_cache = try rt.allocator.alloc(f32, pair_count);
    defer rt.allocator.free(pow_cache);
    for (pow_cache, 0..) |*p, pair_i| {
        const exponent = @as(f32, @floatFromInt(2 * pair_i)) / @as(f32, @floatFromInt(feature_dim));
        p.* = std.math.pow(f32, theta_base, exponent);
    }
    for (positions, 0..) |position, position_i| {
        const pos = @as(f32, @floatFromInt(position));
        for (0..pair_count) |pair_i| {
            const inv_freq = pos / pow_cache[pair_i];
            const theta = if (freq_factors) |ff| inv_freq / ff[pair_i] else inv_freq;
            const angle_i = position_i * pair_count + pair_i;
            sin_values[angle_i] = sign * @sin(theta);
            cos_values[angle_i] = @cos(theta);
        }
    }

    return .{
        .allocator = rt.allocator,
        .positions = positions_copy,
        .theta_base = theta_base,
        .feature_dim = feature_dim,
        .pair_count = pair_count,
        .values = values,
    };
}

/// DeepSeek-family YaRN inverse-frequency blend, in f64 (the HF
/// DeepseekV2Yarn reference): the plain pow schedule
/// `base^(-2i/dim)`, linearly blended toward `freq/factor` across the
/// correction ramp [beta_fast = 32, beta_slow = 1 rotations]. The cos/sin
/// magnitude correction (mscale) is the caller's business — the DeepSeek
/// ports fold it into the attention scale. `factor <= 1` or
/// `orig_ctx == 0` returns the unblended schedule, so one call covers both
/// rope families. Caller frees the returned slice.
pub fn yarnBlendInvFreqsF64(allocator: Allocator, dim: usize, base: f64, factor: f64, orig_ctx: usize) ![]f64 {
    const pairs = dim / 2;
    const inv_freq = try allocator.alloc(f64, pairs);
    errdefer allocator.free(inv_freq);
    for (inv_freq, 0..) |*f, i| {
        f.* = std.math.pow(f64, base, -(@as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(dim))));
    }
    if (factor > 1.0 and orig_ctx > 0) {
        const orig: f64 = @floatFromInt(orig_ctx);
        const dimFor = struct {
            fn go(rotations: f64, d: f64, b: f64, o: f64) f64 {
                return d * @log(o / (rotations * 2.0 * std.math.pi)) / (2.0 * @log(b));
            }
        }.go;
        const d_f: f64 = @floatFromInt(dim);
        var low = @floor(dimFor(32.0, d_f, base, orig));
        var high = @ceil(dimFor(1.0, d_f, base, orig));
        low = @max(low, 0);
        high = @min(high, d_f - 1);
        for (inv_freq, 0..) |*f, i| {
            const extra = f.*;
            const inter = extra / factor;
            // ramp 0 -> 1 across [low, high]; mask = 1 - ramp keeps the
            // fast-rotating dims extrapolated (original freq).
            var ramp = (@as(f64, @floatFromInt(i)) - low) / @max(high - low, 0.001);
            ramp = @min(@max(ramp, 0.0), 1.0);
            const mask = 1.0 - ramp;
            f.* = inter * (1.0 - mask) + extra * mask;
        }
    }
    return inv_freq;
}

/// Hand-fill a rope table for `count` consecutive positions starting at
/// `pos0` from caller-supplied per-pair inverse frequencies, accumulating
/// each angle in f64 before the f32 cast — for models whose frequency
/// schedule the core cannot rebuild (YaRN blends, per-family bases) and
/// whose reference computes angles in double precision
/// (`prepareRopeTable*` compute f32 angles). `inverse` negates sin (the
/// un-rotation table). `table.feature_dim` spans `2 * inv_freq.len`
/// features, so partial application follows the usual table contract.
pub fn prepareRopeTableInvFreqsF64(rt: *Runtime, pos0: usize, count: usize, inv_freq: []const f64, inverse: bool) !RopeTable {
    const pairs = inv_freq.len;
    const angle_count = try std.math.mul(usize, count, pairs);
    const values = try rt.allocator.alloc(f32, try std.math.mul(usize, angle_count, 2));
    errdefer rt.allocator.free(values);
    const positions = try rt.allocator.alloc(i32, count);
    errdefer rt.allocator.free(positions);
    const sin_values = values[0..angle_count];
    const cos_values = values[angle_count..];
    for (0..count) |i| {
        positions[i] = @intCast(pos0 + i);
        for (0..pairs) |p| {
            const angle = @as(f64, @floatFromInt(pos0 + i)) * inv_freq[p];
            const s: f32 = @floatCast(@sin(angle));
            sin_values[i * pairs + p] = if (inverse) -s else s;
            cos_values[i * pairs + p] = @floatCast(@cos(angle));
        }
    }
    return .{
        .allocator = rt.allocator,
        .positions = positions,
        .theta_base = 0, // hand-filled: never rebuilt from a base
        .feature_dim = 2 * pairs,
        .pair_count = pairs,
        .values = values,
    };
}

pub fn ropeAxisRankWithTable(
    rt: *Runtime,
    comptime rank: usize,
    x: *const Tensor,
    comptime position_axis: usize,
    comptime feature_axis: usize,
    table: *const RopeTable,
    comptime mode: RopeMode,
) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (position_axis >= rank or feature_axis >= rank) @compileError("axis out of bounds");
    if (position_axis == feature_axis) @compileError("position and feature axes must differ");

    const source = try x.rankView(rank);
    const feature_dim = source.shape[feature_axis];
    if (feature_dim % 2 != 0) return tensor.TensorError.InvalidShape;
    if (table.positions.len != source.shape[position_axis]) return tensor.TensorError.InvalidDataLength;
    if (table.feature_dim != feature_dim) return tensor.TensorError.InvalidShape;

    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    const input = xx.tensor().dataConst();

    var out = try rt.emptyRank(rank, source.shape);
    errdefer out.deinit();
    const output = out.data();

    const strides = contiguousStridesArray(rank, source.shape);
    const feature_stride = strides[feature_axis];
    const pair_count = feature_dim / 2;
    const total_vectors = input.len / feature_dim;
    const sin_values = table.sinValues();
    const cos_values = table.cosValues();

    for (0..total_vectors) |vector_i| {
        var remainder = vector_i;
        var base_offset: usize = 0;
        var position_coord: usize = 0;
        comptime var dim = rank;
        inline while (dim > 0) {
            dim -= 1;
            if (dim != feature_axis) {
                const coord = remainder % source.shape[dim];
                remainder /= source.shape[dim];
                base_offset += coord * strides[dim];
                if (dim == position_axis) position_coord = coord;
            }
        }

        for (0..pair_count) |pair_i| {
            const angle_i = position_coord * pair_count + pair_i;
            const sin_value = sin_values[angle_i];
            const cos_value = cos_values[angle_i];

            const first_feature = switch (mode) {
                .interleaved, .interleaved_tail => 2 * pair_i,
                .half => pair_i,
            };
            const second_feature = switch (mode) {
                .interleaved, .interleaved_tail => 2 * pair_i + 1,
                .half => pair_i + pair_count,
            };
            const first_offset = base_offset + first_feature * feature_stride;
            const second_offset = base_offset + second_feature * feature_stride;
            const first = input[first_offset];
            const second = input[second_offset];
            output[first_offset] = first * cos_value - second * sin_value;
            output[second_offset] = first * sin_value + second * cos_value;
        }
    }

    return out;
}

pub fn ropePartialAxisRank(
    rt: *Runtime,
    comptime rank: usize,
    x: *const Tensor,
    comptime position_axis: usize,
    comptime feature_axis: usize,
    rotary_dim: usize,
    positions: []const i32,
    theta_base: f32,
    comptime mode: RopeMode,
    comptime inverse: bool,
) !Tensor {
    var table = try prepareRopeTable(rt, positions, rotary_dim, theta_base, inverse);
    defer table.deinit();
    return ropePartialAxisRankWithTable(rt, rank, x, position_axis, feature_axis, &table, mode);
}

pub fn ropePartialAxisRankWithTable(
    rt: *Runtime,
    comptime rank: usize,
    x: *const Tensor,
    comptime position_axis: usize,
    comptime feature_axis: usize,
    table: *const RopeTable,
    comptime mode: RopeMode,
) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (position_axis >= rank or feature_axis >= rank) @compileError("axis out of bounds");
    if (position_axis == feature_axis) @compileError("position and feature axes must differ");

    const source = try x.rankView(rank);
    const feature_dim = source.shape[feature_axis];
    const rotary_dim = table.feature_dim;
    if (rotary_dim == 0 or rotary_dim > feature_dim or rotary_dim % 2 != 0) return tensor.TensorError.InvalidShape;
    if (table.positions.len != source.shape[position_axis]) return tensor.TensorError.InvalidDataLength;
    if (rotary_dim == feature_dim) return ropeAxisRankWithTable(rt, rank, x, position_axis, feature_axis, table, mode);

    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    const input = xx.tensor().dataConst();

    var out = try rt.emptyRank(rank, source.shape);
    errdefer out.deinit();
    const output = out.data();
    @memcpy(output, input);

    const strides = contiguousStridesArray(rank, source.shape);
    const feature_stride = strides[feature_axis];
    const pair_count = rotary_dim / 2;
    const total_vectors = input.len / feature_dim;
    const sin_values = table.sinValues();
    const cos_values = table.cosValues();
    // Tail alignment: the rotary span sits at the END of the feature axis
    // (the leading `feature_dim - rotary_dim` features pass through).
    const rotary_offset: usize = switch (mode) {
        .interleaved_tail => feature_dim - rotary_dim,
        .interleaved, .half => 0,
    };

    for (0..total_vectors) |vector_i| {
        var remainder = vector_i;
        var base_offset: usize = 0;
        var position_coord: usize = 0;
        comptime var dim = rank;
        inline while (dim > 0) {
            dim -= 1;
            if (dim != feature_axis) {
                const coord = remainder % source.shape[dim];
                remainder /= source.shape[dim];
                base_offset += coord * strides[dim];
                if (dim == position_axis) position_coord = coord;
            }
        }

        for (0..pair_count) |pair_i| {
            const angle_i = position_coord * pair_count + pair_i;
            const sin_value = sin_values[angle_i];
            const cos_value = cos_values[angle_i];

            const first_feature = rotary_offset + switch (mode) {
                .interleaved, .interleaved_tail => 2 * pair_i,
                .half => pair_i,
            };
            const second_feature = rotary_offset + switch (mode) {
                .interleaved, .interleaved_tail => 2 * pair_i + 1,
                .half => pair_i + pair_count,
            };
            const first_offset = base_offset + first_feature * feature_stride;
            const second_offset = base_offset + second_feature * feature_stride;
            const first = input[first_offset];
            const second = input[second_offset];
            output[first_offset] = first * cos_value - second * sin_value;
            output[second_offset] = first * sin_value + second * cos_value;
        }
    }

    return out;
}
