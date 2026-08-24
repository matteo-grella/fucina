//! Rotary position embedding (RoPE) for the eager runtime.
//!
//! Home of `RopeTable`/`RopeMode`/`RopeTableSpec` (re-exported through
//! `exec.zig` for the autograd VJP params and the facade) plus the two
//! forward entries: `prepareRopeTable(spec)` builds a table over any
//! positions source and angle schedule, `ropeWithTable` applies it over a
//! full or partial rotary span. `sinValues`/`cosValues` are `pub` so
//! `norm.zig`'s fused rms-norm+rope kernel can read them cross-module.
//!
//! Domain module: every op receives an explicit `*ExecContext`; imports the
//! shape leaf.

const std = @import("std");
const tensor = @import("../tensor.zig");

const exec_shape = @import("shape.zig");
const ExecContext = @import("../exec.zig").ExecContext;

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

/// On-the-fly factor source for the facade `rope`: positions + theta base,
/// full rotation only. Production paths prepare a `RopeTable` instead
/// (factors / NTK scaling / f64 schedules live in `RopeTableSpec`).
pub const RopeTheta = struct {
    positions: []const i32,
    theta_base: f32,
};

pub const RopeTable = struct {
    allocator: Allocator,
    positions: []i32,
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

/// Where a table's positions come from. Both arms feed one arithmetic body,
/// so a run expressed as a range and the same run expressed as an array
/// produce bitwise identical tables.
pub const RopePositions = union(enum) {
    /// One position per rotated row, any values: ragged batches (several
    /// runs), context shifts (negative deltas).
    explicit: []const i32,
    /// One contiguous run `[origin, origin + len)`: a prefill covers `0..n`,
    /// a decode step `pos0..pos0+n`. Names the run by its origin instead of
    /// materializing it.
    range: tensor.AxisRange,

    fn len(self: RopePositions) usize {
        return switch (self) {
            .explicit => |p| p.len,
            .range => |r| r.len,
        };
    }

    fn at(self: RopePositions, i: usize) i32 {
        return switch (self) {
            .explicit => |p| p[i],
            .range => |r| @intCast(r.at(i)),
        };
    }
};

/// The per-pair angle schedule of a table.
pub const RopeFreqs = union(enum) {
    /// Pair `i` at position `p` rotates by `p / base^(2i/d)`, computed in
    /// f32. `factors` (length `feature_dim/2`) divides each pair's
    /// frequency: ggml's `rope_ext` `freq_factors` (proportional / NTK-by-part
    /// RoPE; Llama-3 long-context scaling, Gemma's global layers). `null`
    /// reproduces plain RoPE exactly.
    theta: struct { base: f32, factors: ?[]const f32 = null },
    /// Caller-supplied per-pair inverse frequencies (length `feature_dim/2`),
    /// each angle accumulated in f64 before the f32 cast: for schedules the
    /// core cannot rebuild (YaRN blends, per-family bases) whose reference
    /// computes angles in double precision. The cos/sin magnitude
    /// correction (mscale) is the caller's business.
    inv_freq_f64: []const f64,
};

/// What `prepareRopeTable` builds: the positions, the rotary span, the angle
/// schedule and the sign. `feature_dim` is the table's authoritative rotary
/// span; a table narrower than the tensor's feature axis is a partial rotary.
pub const RopeTableSpec = struct {
    positions: RopePositions,
    feature_dim: usize,
    freqs: RopeFreqs,
    /// Negate every sin: the un-rotation table.
    inverse: bool = false,
};

/// Full-axis rotation with on-the-fly `.theta` factors over explicit
/// positions. Production paths prepare a `RopeTable` once and apply it per
/// layer with `ropeWithTable`.
pub fn rope(
    ctx: *ExecContext,
    comptime rank: usize,
    x: *const Tensor,
    comptime position_axis: usize,
    comptime feature_axis: usize,
    theta: RopeTheta,
    comptime mode: RopeMode,
    inverse: bool,
) !Tensor {
    const source = try x.rankView(rank);
    var table = try prepareRopeTable(ctx, .{
        .positions = .{ .explicit = theta.positions },
        .feature_dim = source.shape[feature_axis],
        .freqs = .{ .theta = .{ .base = theta.theta_base } },
        .inverse = inverse,
    });
    defer table.deinit();
    return ropeWithTable(ctx, rank, x, position_axis, feature_axis, &table, mode);
}

/// Build the sin/cos table `spec` describes. Both position sources run the
/// exact same arithmetic on the exact same i32 values, so the tables are
/// bitwise identical; the `.theta` schedule computes f32 angles, the
/// `.inv_freq_f64` schedule f64 angles.
pub fn prepareRopeTable(ctx: *ExecContext, spec: RopeTableSpec) !RopeTable {
    if (spec.feature_dim % 2 != 0) return tensor.TensorError.InvalidShape;
    const pair_count = spec.feature_dim / 2;
    switch (spec.freqs) {
        .theta => |t| if (t.factors) |ff| {
            if (ff.len != pair_count) return tensor.TensorError.ShapeMismatch;
        },
        .inv_freq_f64 => |f| if (f.len != pair_count) return tensor.TensorError.ShapeMismatch,
    }
    const position_count = spec.positions.len();
    const angle_count = try std.math.mul(usize, position_count, pair_count);
    const values = try ctx.allocator.alloc(f32, try std.math.mul(usize, angle_count, 2));
    errdefer ctx.allocator.free(values);
    const positions_copy = try ctx.allocator.alloc(i32, position_count);
    errdefer ctx.allocator.free(positions_copy);
    for (positions_copy, 0..) |*slot, i| slot.* = spec.positions.at(i);

    const sin_values = values[0..angle_count];
    const cos_values = values[angle_count..][0..angle_count];
    switch (spec.freqs) {
        .theta => |t| {
            const sign: f32 = if (spec.inverse) -1 else 1;
            // theta_base^(2i/d) is position-invariant, so hoist the pow; the
            // factors divide must stay per-element — folding it into the
            // cache changes f32 rounding ((pos/a)/b != pos/(a*b)).
            const pow_cache = try ctx.allocator.alloc(f32, pair_count);
            defer ctx.allocator.free(pow_cache);
            for (pow_cache, 0..) |*p, pair_i| {
                const exponent = @as(f32, @floatFromInt(2 * pair_i)) / @as(f32, @floatFromInt(spec.feature_dim));
                p.* = std.math.pow(f32, t.base, exponent);
            }
            for (positions_copy, 0..) |position, position_i| {
                const pos = @as(f32, @floatFromInt(position));
                for (0..pair_count) |pair_i| {
                    const inv_freq = pos / pow_cache[pair_i];
                    const angle = if (t.factors) |ff| inv_freq / ff[pair_i] else inv_freq;
                    const angle_i = position_i * pair_count + pair_i;
                    sin_values[angle_i] = sign * @sin(angle);
                    cos_values[angle_i] = @cos(angle);
                }
            }
        },
        .inv_freq_f64 => |inv_freq| {
            for (positions_copy, 0..) |position, position_i| {
                const pos = @as(f64, @floatFromInt(position));
                for (0..pair_count) |pair_i| {
                    const angle = pos * inv_freq[pair_i];
                    const s: f32 = @floatCast(@sin(angle));
                    const angle_i = position_i * pair_count + pair_i;
                    sin_values[angle_i] = if (spec.inverse) -s else s;
                    cos_values[angle_i] = @floatCast(@cos(angle));
                }
            }
        },
    }

    return .{
        .allocator = ctx.allocator,
        .positions = positions_copy,
        .feature_dim = spec.feature_dim,
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
pub fn yarnBlendInvFreqsF64(ctx: *ExecContext, dim: usize, base: f64, factor: f64, orig_ctx: usize) ![]f64 {
    const pairs = dim / 2;
    const inv_freq = try ctx.allocator.alloc(f64, pairs);
    errdefer ctx.allocator.free(inv_freq);
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

// ---------------- Vector pair rotation (feature_stride == 1) ----------------

const rope_vec_width = 8;
const RopeVec = @Vector(rope_vec_width, f32);

/// Rotate one position's pairs when the two features of each pair sit in
/// separate contiguous halves (`.half` pairing): straight vector loads on
/// both halves and on the sin/cos table rows. Same per-element expressions
/// as the scalar loop, so the output is bitwise identical.
fn rotatePairsHalf(output: []f32, input: []const f32, first_base: usize, second_base: usize, sin_row: []const f32, cos_row: []const f32) void {
    const pair_count = sin_row.len;
    var pair_i: usize = 0;
    while (pair_i + rope_vec_width <= pair_count) : (pair_i += rope_vec_width) {
        const first: RopeVec = input[first_base + pair_i ..][0..rope_vec_width].*;
        const second: RopeVec = input[second_base + pair_i ..][0..rope_vec_width].*;
        const sin_vec: RopeVec = sin_row[pair_i..][0..rope_vec_width].*;
        const cos_vec: RopeVec = cos_row[pair_i..][0..rope_vec_width].*;
        output[first_base + pair_i ..][0..rope_vec_width].* = first * cos_vec - second * sin_vec;
        output[second_base + pair_i ..][0..rope_vec_width].* = first * sin_vec + second * cos_vec;
    }
    while (pair_i < pair_count) : (pair_i += 1) {
        const first = input[first_base + pair_i];
        const second = input[second_base + pair_i];
        output[first_base + pair_i] = first * cos_row[pair_i] - second * sin_row[pair_i];
        output[second_base + pair_i] = first * sin_row[pair_i] + second * cos_row[pair_i];
    }
}

/// Rotate one position's pairs when the two features of each pair are
/// adjacent (`.interleaved` pairing): two contiguous vector loads per step,
/// deinterleave/reinterleave with `@shuffle` (the ld2/st2 shape on NEON).
/// Bitwise identical to the scalar loop. Negative shuffle indices select
/// from the second operand (`~i` picks lane `i`).
fn rotatePairsInterleaved(output: []f32, input: []const f32, base: usize, sin_row: []const f32, cos_row: []const f32) void {
    const pair_count = sin_row.len;
    var pair_i: usize = 0;
    while (pair_i + rope_vec_width <= pair_count) : (pair_i += rope_vec_width) {
        const lo: RopeVec = input[base + 2 * pair_i ..][0..rope_vec_width].*;
        const hi: RopeVec = input[base + 2 * pair_i + rope_vec_width ..][0..rope_vec_width].*;
        const first = @shuffle(f32, lo, hi, [rope_vec_width]i32{ 0, 2, 4, 6, -1, -3, -5, -7 });
        const second = @shuffle(f32, lo, hi, [rope_vec_width]i32{ 1, 3, 5, 7, -2, -4, -6, -8 });
        const sin_vec: RopeVec = sin_row[pair_i..][0..rope_vec_width].*;
        const cos_vec: RopeVec = cos_row[pair_i..][0..rope_vec_width].*;
        const rotated_first = first * cos_vec - second * sin_vec;
        const rotated_second = first * sin_vec + second * cos_vec;
        output[base + 2 * pair_i ..][0..rope_vec_width].* = @shuffle(f32, rotated_first, rotated_second, [rope_vec_width]i32{ 0, -1, 1, -2, 2, -3, 3, -4 });
        output[base + 2 * pair_i + rope_vec_width ..][0..rope_vec_width].* = @shuffle(f32, rotated_first, rotated_second, [rope_vec_width]i32{ 4, -5, 5, -6, 6, -7, 7, -8 });
    }
    while (pair_i < pair_count) : (pair_i += 1) {
        const first_offset = base + 2 * pair_i;
        const first = input[first_offset];
        const second = input[first_offset + 1];
        output[first_offset] = first * cos_row[pair_i] - second * sin_row[pair_i];
        output[first_offset + 1] = first * sin_row[pair_i] + second * cos_row[pair_i];
    }
}

/// Apply `table` over (`position_axis`, `feature_axis`). The table's
/// `feature_dim` is the rotary span: equal to the feature axis rotates every
/// pair; smaller rotates the leading `feature_dim` features (`.half`,
/// `.interleaved`) or the trailing ones (`.interleaved_tail`) and passes the
/// rest through unchanged.
pub fn ropeWithTable(
    ctx: *ExecContext,
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

    var xx = try ctx.prepareContiguous(.f32, x);
    defer xx.deinit();
    const input = xx.tensor().dataConst();

    var out = try ctx.empty(.f32, source.shape);
    errdefer out.deinit();
    const output = out.data();
    // Partial span: the pass-through features are copied, the rotated ones
    // overwritten below.
    if (rotary_dim != feature_dim) @memcpy(output, input);

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

        if (feature_stride == 1) {
            const sin_row = sin_values[position_coord * pair_count ..][0..pair_count];
            const cos_row = cos_values[position_coord * pair_count ..][0..pair_count];
            switch (mode) {
                .interleaved, .interleaved_tail => rotatePairsInterleaved(output, input, base_offset + rotary_offset, sin_row, cos_row),
                .half => rotatePairsHalf(output, input, base_offset + rotary_offset, base_offset + rotary_offset + pair_count, sin_row, cos_row),
            }
            continue;
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
