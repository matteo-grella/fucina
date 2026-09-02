//! Rotary position embedding (RoPE) for the eager runtime.
//!
//! Home of `RopeTable`/`RopeMode`/`RopeTableSpec` (re-exported through
//! `exec.zig` for the autograd VJP params and the facade) plus the
//! application entries: `prepareRopeTable(spec)` builds a table over any
//! positions source and angle schedule, `ropeWithTable` applies it over a
//! full or partial rotary span, and `ropeWithTableInverse` applies the
//! exact inverse rotation (sin negated at apply time; the VJP route —
//! records `retain` the forward table instead of cloning it).
//! `sinValues`/`cosValues` are `pub` so `norm.zig`'s fused rms-norm+rope
//! kernel can read them cross-module.
//!
//! `ropeWithTable` splits its vectors across the work pool above the
//! fused-norm sibling's length gate; every vector owns disjoint output
//! features, so the pooled result is bitwise identical to the serial walk
//! (`rope_tests.zig` pins it).
//!
//! Domain module: every op receives an explicit `*ExecContext`; imports the
//! shape leaf and the parallel policy constants.

const std = @import("std");
const parallel = @import("../parallel.zig");
const tensor = @import("../tensor.zig");

const shape_mod = @import("../shape.zig");
const ExecContext = @import("../exec.zig").ExecContext;

const Allocator = std.mem.Allocator;
const Tensor = tensor.Tensor;

const contiguousStridesArray = shape_mod.contiguousStrides;

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
    /// Shared owner count of `positions`/`values`: the prepared original
    /// and every `retain`ed handle each call `deinit` once; the buffers are
    /// freed by the last. Atomic so a graph handed to another thread stays
    /// safe (the tensor-storage refcount convention).
    refs: *std.atomic.Value(usize),

    pub fn deinit(self: *RopeTable) void {
        if (self.refs.fetchSub(1, .acq_rel) == 1) {
            self.allocator.free(self.values);
            self.allocator.free(self.positions);
            self.allocator.destroy(self.refs);
        }
        self.* = undefined;
    }

    /// A new owning handle of the SAME buffers (owner count +1). The RoPE
    /// VJP records retain the forward table this way instead of duplicating
    /// positions and values per record (~1 MB per layer at 2k context) and
    /// apply the inversion at rotation time (`ropeWithTableInverse`).
    pub fn retain(self: *const RopeTable) RopeTable {
        _ = self.refs.fetchAdd(1, .monotonic);
        return self.*;
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
/// A contiguous run of ABSOLUTE indices along one axis: `[origin, origin + len)`.
///
/// This is Fortran's array lower bound as a value. A tensor axis here is
/// 0-origin like NumPy's: `Shape` records how long an axis is, never where it
/// starts, so a view that narrows a positional axis forgets its absolute
/// position and every consumer that needs it takes it through a side channel.
/// The visible cost is the arithmetic-sequence array: a caller that wants
/// "the `n` positions starting at `p0`" allocates `n` integers, fills them
/// with `p0 + i`, passes them down, and frees them, purely to say `p0`.
///
/// `AxisRange` says `p0` instead. It carries no data and owns nothing, so it
/// costs two words wherever an origin needs to travel with a length.
///
/// The default `origin = 0` reproduces the ordinary 0-origin axis, so
/// `.{ .len = n }` is the plain case.
pub const AxisRange = struct {
    /// Absolute index of the axis's first element.
    origin: i64 = 0,
    /// Number of elements the axis spans.
    len: usize,

    /// Absolute index of element `i`. Unchecked: `i < len` is the caller's.
    pub fn at(self: AxisRange, i: usize) i64 {
        return self.origin + @as(i64, @intCast(i));
    }

    /// One past the last absolute index.
    pub fn end(self: AxisRange) i64 {
        return self.origin + @as(i64, @intCast(self.len));
    }

    /// The same run of elements relabelled to start at `new_origin` — what a
    /// zero-copy narrow of a positional axis should do to its origin.
    pub fn rebased(self: AxisRange, new_origin: i64) AxisRange {
        return .{ .origin = new_origin, .len = self.len };
    }

    /// Slide the whole run by `delta` absolute positions.
    pub fn shifted(self: AxisRange, delta: i64) AxisRange {
        return .{ .origin = self.origin + delta, .len = self.len };
    }

    /// The sub-run `[start, start + count)` in LOCAL indices, carrying the
    /// absolute origin forward — the operation a narrow performs and the one
    /// a plain 0-origin axis cannot express.
    pub fn narrowed(self: AxisRange, start: usize, count: usize) AxisRange {
        return .{ .origin = self.at(start), .len = count };
    }

    pub fn contains(self: AxisRange, absolute: i64) bool {
        return absolute >= self.origin and absolute < self.end();
    }

    /// Fill `out` with the absolute indices. The materialization this type
    /// exists to avoid, kept for the interop boundaries that genuinely need a
    /// per-element array (a ragged multi-stream batch, where the positions are
    /// not one run).
    pub fn writeInto(self: AxisRange, out: []i32) !void {
        if (out.len != self.len) return tensor.TensorError.InvalidDataLength;
        for (out, 0..) |*slot, i| slot.* = @intCast(self.at(i));
    }
};

pub const RopePositions = union(enum) {
    /// One position per rotated row, any values: ragged batches (several
    /// runs), context shifts (negative deltas).
    explicit: []const i32,
    /// One contiguous run `[origin, origin + len)`: a prefill covers `0..n`,
    /// a decode step `pos0..pos0+n`. Names the run by its origin instead of
    /// materializing it.
    range: AxisRange,

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
    const values = try ctx.allocator().alloc(f32, try std.math.mul(usize, angle_count, 2));
    errdefer ctx.allocator().free(values);
    const positions_copy = try ctx.allocator().alloc(i32, position_count);
    errdefer ctx.allocator().free(positions_copy);
    const refs = try ctx.allocator().create(std.atomic.Value(usize));
    errdefer ctx.allocator().destroy(refs);
    refs.* = .init(1);
    for (positions_copy, 0..) |*slot, i| slot.* = spec.positions.at(i);

    const sin_values = values[0..angle_count];
    const cos_values = values[angle_count..][0..angle_count];
    switch (spec.freqs) {
        .theta => |t| {
            const sign: f32 = if (spec.inverse) -1 else 1;
            // theta_base^(2i/d) is position-invariant, so hoist the pow; the
            // factors divide must stay per-element — folding it into the
            // cache changes f32 rounding ((pos/a)/b != pos/(a*b)).
            const pow_cache = try ctx.allocator().alloc(f32, pair_count);
            defer ctx.allocator().free(pow_cache);
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
        .allocator = ctx.allocator(),
        .positions = positions_copy,
        .feature_dim = spec.feature_dim,
        .pair_count = pair_count,
        .values = values,
        .refs = refs,
    };
}

// ---------------- Vector pair rotation (feature_stride == 1) ----------------

const rope_vec_width = 8;
const RopeVec = @Vector(rope_vec_width, f32);

/// Rotate one position's pairs when the two features of each pair sit in
/// separate contiguous halves (`.half` pairing): straight vector loads on
/// both halves and on the sin/cos table rows. Same per-element expressions
/// as the scalar loop, so the output is bitwise identical. `inverse`
/// negates each sin lane at load: `-s` is the exact f32 a sign-flipped
/// table would hold, so the un-rotation is bitwise identical to applying a
/// negated table.
fn rotatePairsHalf(comptime inverse: bool, output: []f32, input: []const f32, first_base: usize, second_base: usize, sin_row: []const f32, cos_row: []const f32) void {
    const pair_count = sin_row.len;
    var pair_i: usize = 0;
    while (pair_i + rope_vec_width <= pair_count) : (pair_i += rope_vec_width) {
        const first: RopeVec = input[first_base + pair_i ..][0..rope_vec_width].*;
        const second: RopeVec = input[second_base + pair_i ..][0..rope_vec_width].*;
        const sin_loaded: RopeVec = sin_row[pair_i..][0..rope_vec_width].*;
        const sin_vec = if (inverse) -sin_loaded else sin_loaded;
        const cos_vec: RopeVec = cos_row[pair_i..][0..rope_vec_width].*;
        output[first_base + pair_i ..][0..rope_vec_width].* = first * cos_vec - second * sin_vec;
        output[second_base + pair_i ..][0..rope_vec_width].* = first * sin_vec + second * cos_vec;
    }
    while (pair_i < pair_count) : (pair_i += 1) {
        const first = input[first_base + pair_i];
        const second = input[second_base + pair_i];
        const sin_value = if (inverse) -sin_row[pair_i] else sin_row[pair_i];
        output[first_base + pair_i] = first * cos_row[pair_i] - second * sin_value;
        output[second_base + pair_i] = first * sin_value + second * cos_row[pair_i];
    }
}

/// Rotate one position's pairs when the two features of each pair are
/// adjacent (`.interleaved` pairing): two contiguous vector loads per step,
/// deinterleave/reinterleave with `@shuffle` (the ld2/st2 shape on NEON).
/// Bitwise identical to the scalar loop. Negative shuffle indices select
/// from the second operand (`~i` picks lane `i`).
fn rotatePairsInterleaved(comptime inverse: bool, output: []f32, input: []const f32, base: usize, sin_row: []const f32, cos_row: []const f32) void {
    const pair_count = sin_row.len;
    var pair_i: usize = 0;
    while (pair_i + rope_vec_width <= pair_count) : (pair_i += rope_vec_width) {
        const lo: RopeVec = input[base + 2 * pair_i ..][0..rope_vec_width].*;
        const hi: RopeVec = input[base + 2 * pair_i + rope_vec_width ..][0..rope_vec_width].*;
        const first = @shuffle(f32, lo, hi, [rope_vec_width]i32{ 0, 2, 4, 6, -1, -3, -5, -7 });
        const second = @shuffle(f32, lo, hi, [rope_vec_width]i32{ 1, 3, 5, 7, -2, -4, -6, -8 });
        const sin_loaded: RopeVec = sin_row[pair_i..][0..rope_vec_width].*;
        const sin_vec = if (inverse) -sin_loaded else sin_loaded;
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
        const sin_value = if (inverse) -sin_row[pair_i] else sin_row[pair_i];
        output[first_offset] = first * cos_row[pair_i] - second * sin_value;
        output[first_offset + 1] = first * sin_value + second * cos_row[pair_i];
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
    return ropeWithTableDirection(ctx, rank, x, position_axis, feature_axis, table, mode, false);
}

/// Apply `table` as the exact inverse (transpose) rotation: every sin is
/// negated AT APPLY TIME instead of materializing a sign-flipped table copy
/// (`-s` is the exact f32 such a table would hold, so the output is bitwise
/// identical to `ropeWithTable` over a negated table — `rope_tests.zig`
/// pins it). The RoPE VJPs retain the forward table and route here, which
/// preserves `freq_factors`/NTK scaling baked into the angles.
pub fn ropeWithTableInverse(
    ctx: *ExecContext,
    comptime rank: usize,
    x: *const Tensor,
    comptime position_axis: usize,
    comptime feature_axis: usize,
    table: *const RopeTable,
    comptime mode: RopeMode,
) !Tensor {
    return ropeWithTableDirection(ctx, rank, x, position_axis, feature_axis, table, mode, true);
}

fn ropeWithTableDirection(
    ctx: *ExecContext,
    comptime rank: usize,
    x: *const Tensor,
    comptime position_axis: usize,
    comptime feature_axis: usize,
    table: *const RopeTable,
    comptime mode: RopeMode,
    comptime inverse: bool,
) !Tensor {
    if (position_axis >= rank or feature_axis >= rank) @compileError("axis out of bounds");
    if (position_axis == feature_axis) @compileError("position and feature axes must differ");

    const source = try x.rankView(rank);
    const feature_dim = source.shape[feature_axis];
    const rotary_dim = table.feature_dim;
    if (rotary_dim == 0 or rotary_dim > feature_dim or rotary_dim % 2 != 0) return tensor.TensorError.InvalidShape;
    if (table.positions.len != source.shape[position_axis]) return tensor.TensorError.InvalidDataLength;
    // One body, instantiated per span: the full-span loop carries no
    // pass-through copy and no offset arithmetic.
    if (rotary_dim == feature_dim) return applyRope(ctx, rank, x, position_axis, feature_axis, table, mode, .full, inverse);
    return applyRope(ctx, rank, x, position_axis, feature_axis, table, mode, .partial, inverse);
}

const RotarySpan = enum { full, partial };

fn applyRope(
    ctx: *ExecContext,
    comptime rank: usize,
    x: *const Tensor,
    comptime position_axis: usize,
    comptime feature_axis: usize,
    table: *const RopeTable,
    comptime mode: RopeMode,
    comptime span: RotarySpan,
    comptime inverse: bool,
) !Tensor {
    const source = try x.rankView(rank);
    const feature_dim = source.shape[feature_axis];
    const rotary_dim = switch (span) {
        .full => feature_dim,
        .partial => table.feature_dim,
    };

    var xx = try ctx.prepareContiguous(.f32, x);
    defer xx.deinit();
    const input = xx.tensor().dataConst();

    var out = try ctx.empty(.f32, source.shape);
    errdefer out.deinit();
    const output = out.data();
    // Partial span: the pass-through features are copied, the rotated ones
    // overwritten below.
    if (span == .partial) @memcpy(output, input);

    const strides = contiguousStridesArray(rank, source.shape);
    const feature_stride = strides[feature_axis];
    const pair_count = rotary_dim / 2;
    const total_vectors = input.len / feature_dim;
    const sin_values = table.sinValues();
    const cos_values = table.cosValues();
    // Tail alignment: the rotary span sits at the END of the feature axis
    // (the leading `feature_dim - rotary_dim` features pass through).
    const rotary_offset: usize = switch (span) {
        .full => 0,
        .partial => switch (mode) {
            .interleaved_tail => feature_dim - rotary_dim,
            .interleaved, .half => 0,
        },
    };

    const Task = RopeVectorsTask(rank, position_axis, feature_axis, mode, inverse);
    const task: Task = .{
        .input = input,
        .output = output,
        .sin_values = sin_values,
        .cos_values = cos_values,
        .shape = source.shape,
        .strides = strides,
        .feature_stride = feature_stride,
        .pair_count = pair_count,
        .rotary_offset = rotary_offset,
        .vector_start = 0,
        .vector_end = total_vectors,
    };

    // Same length gate as the fused rms-norm+rope sibling (norm.zig): the
    // pool is engaged for prefill-sized inputs only; a decode step's
    // handful of vectors stays on the calling thread. The gate decides
    // POOLING only, never per-vector math.
    if (total_vectors > 1 and input.len >= parallel.vector_elementwise_len_threshold / 8) {
        if (ctx.dispatchRange(Task, "vector_start", "vector_end", task, total_vectors, Task.run)) return out;
    }
    Task.run(&task);
    return out;
}

/// The per-vector rotation walk of `applyRope`, shaped as a range task so
/// `dispatchRange` can split `[vector_start, vector_end)` across the pool.
/// Each vector reads its own input features and writes its own output
/// features, so any split produces the serial walk's bytes.
fn RopeVectorsTask(
    comptime rank: usize,
    comptime position_axis: usize,
    comptime feature_axis: usize,
    comptime mode: RopeMode,
    comptime inverse: bool,
) type {
    return struct {
        input: []const f32,
        output: []f32,
        sin_values: []const f32,
        cos_values: []const f32,
        shape: [rank]usize,
        strides: [rank]usize,
        feature_stride: usize,
        pair_count: usize,
        rotary_offset: usize,
        vector_start: usize,
        vector_end: usize,

        fn run(task: *const @This()) void {
            const input = task.input;
            const output = task.output;
            const sin_values = task.sin_values;
            const cos_values = task.cos_values;
            const feature_stride = task.feature_stride;
            const pair_count = task.pair_count;
            const rotary_offset = task.rotary_offset;

            for (task.vector_start..task.vector_end) |vector_i| {
                var remainder = vector_i;
                var base_offset: usize = 0;
                var position_coord: usize = 0;
                comptime var dim = rank;
                inline while (dim > 0) {
                    dim -= 1;
                    if (dim != feature_axis) {
                        const coord = remainder % task.shape[dim];
                        remainder /= task.shape[dim];
                        base_offset += coord * task.strides[dim];
                        if (dim == position_axis) position_coord = coord;
                    }
                }

                if (feature_stride == 1) {
                    const sin_row = sin_values[position_coord * pair_count ..][0..pair_count];
                    const cos_row = cos_values[position_coord * pair_count ..][0..pair_count];
                    switch (mode) {
                        .interleaved, .interleaved_tail => rotatePairsInterleaved(inverse, output, input, base_offset + rotary_offset, sin_row, cos_row),
                        .half => rotatePairsHalf(inverse, output, input, base_offset + rotary_offset, base_offset + rotary_offset + pair_count, sin_row, cos_row),
                    }
                    continue;
                }

                for (0..pair_count) |pair_i| {
                    const angle_i = position_coord * pair_count + pair_i;
                    const sin_value = if (inverse) -sin_values[angle_i] else sin_values[angle_i];
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
        }
    };
}
