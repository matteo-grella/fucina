//! f32 tensor methods: variance/standardize, extrema, argmax, multinomial. A mixin over the ag
//! FloatTensor struct; aliased back onto it in ../../tensor.zig.

const std = @import("std");
const tensor_mod = @import("../../../tensor.zig");
const exec_mod = @import("../../../exec.zig");
const dtype_mod = @import("../../../dtype.zig");
const tags_mod = @import("../../../tags.zig");
const backward_stats = @import("../../backward/stats.zig");
const rng = @import("../../../rng.zig");

const RawTensor = tensor_mod.Tensor;
const TensorError = tensor_mod.TensorError;
const ExecContext = exec_mod.ExecContext;
const Tag = tags_mod.Tag;
const removeTag = tags_mod.removeTag;
const replaceTag = tags_mod.replaceTag;
const VarBackward = backward_stats.VarBackward;
const StandardizeBackward = backward_stats.StandardizeBackward;
const MinMaxBackward = backward_stats.MinMaxBackward;

pub fn Ops(comptime Self: type) type {
    return struct {
        const tags = Self.axis_tags;
        const tag_rank = Self.tag_count;
        const tag_count = Self.tag_count;
        const axis = Self.axis;
        const ag_tensor = Self.ag_root;
        const Tensor = ag_tensor.Tensor;
        const plumbing = @import("../plumbing.zig").Mod(ag_tensor);
        const finishOp = plumbing.finishOp;
        const dtype = Self.dtype;
        /// The f32 branch is the differentiable one; every other dtype takes
        /// the constant tail.
        const differentiable = dtype == .f32;
        const finishOrConstant = plumbing.finishOrConstant;
        const reduced_dtype = dtype_mod.outputDType(.reduction, dtype);
        const TensorObject = plumbing.TensorObject;
        const validateMaskedReduceOptions = plumbing.validateMaskedReduceOptions;
        const validateMaskType = plumbing.validateMaskType;
        const maskedReduceEmpty = plumbing.maskedReduceEmpty;
        const MaskedMinMaxBackward = backward_stats.MaskedMinMaxBackward;

        /// Variance over `tag` (the tag is removed like sum/mean): ddof 0 =
        /// biased estimator (the LayerNorm convention), ddof 1 = unbiased
        /// (the torch.var default).
        pub fn variance(self: *const Self, ctx: *ExecContext, comptime tag: Tag, ddof: u1) !Tensor(.{ .dtype = reduced_dtype, .tags = removeTag(tags, tag) }) {
            const result_tags = removeTag(tags, tag);
            const reduce_axis = comptime axis(tag);
            var value = try ctx.varAxis(dtype, tag_rank, self.asRawTensor(), reduce_axis, ddof);
            errdefer value.deinit();
            return finishOrConstant(differentiable, reduced_dtype, result_tags, ctx, value, self.requiresGrad(), VarBackward(tags, reduce_axis), .{ ctx.allocator, self.grad_state, &self.value, ddof });
        }

        /// Standardize over `tag` while preserving shape:
        /// `y = (x - mean(tag)) / denom`. `options` accepts every
        /// `exec.StandardizeOptions` field (ddof/eps/eps_mode/accumulation —
        /// a plain `StandardizeOptions` value still coerces) plus an optional
        /// `.valid_len`: standardize only the first `valid_len` elements of
        /// `tag` — the suffix is masked out, returned as zeros, and receives
        /// zero gradient. Unknown fields are compile errors. Differentiable
        /// in `self`.
        pub fn standardizeAxis(self: *const Self, ctx: *ExecContext, comptime tag: Tag, options: anytype) !Self {
            const Options = @TypeOf(options);
            comptime {
                if (@typeInfo(Options) != .@"struct") @compileError("standardizeAxis: options must be a struct literal, e.g. .{ .ddof = 1 }");
                for (@typeInfo(Options).@"struct".fields) |field| {
                    if (!std.mem.eql(u8, field.name, "valid_len") and !@hasField(exec_mod.StandardizeOptions, field.name))
                        @compileError("standardizeAxis: unknown option ." ++ field.name);
                }
            }
            var exec_options = exec_mod.StandardizeOptions{};
            inline for (@typeInfo(exec_mod.StandardizeOptions).@"struct".fields) |field| {
                if (comptime @hasField(Options, field.name)) @field(exec_options, field.name) = @field(options, field.name);
            }
            const norm_axis = comptime axis(tag);
            if (comptime @hasField(Options, "valid_len")) {
                var value = try ctx.standardizeValidPrefix(tag_rank, self.asRawTensor(), norm_axis, options.valid_len, exec_options);
                errdefer value.deinit();
                return finishOp(tags, ctx, value, self.requiresGrad(), StandardizeBackward(tags, norm_axis), .{ ctx.allocator, self.grad_state, &self.value, @as(?usize, options.valid_len), exec_options });
            }
            var value = try ctx.standardize(tag_rank, self.asRawTensor(), norm_axis, exec_options);
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad(), StandardizeBackward(tags, norm_axis), .{ ctx.allocator, self.grad_state, &self.value, @as(?usize, null), exec_options });
        }

        /// Index of the row maximum along `tag` (torch.argmax over a dim):
        /// a constant i64 tensor, no gradient. Caller-owned even under an
        /// exec scope (the typed-constant ownership rule).
        pub fn argmax(self: *const Self, ctx: *ExecContext, comptime tag: Tag) !Tensor(.{ .dtype = .i64, .tags = removeTag(tags, tag) }) {
            const result_tags = removeTag(tags, tag);
            var value = try ctx.argmax(dtype, tag_rank, self.asRawTensor(), axis(tag));
            errdefer value.deinit();
            return Tensor(.{ .dtype = .i64, .tags = result_tags }).fromTensor(ctx, value);
        }

        /// Categorical sampling from UNNORMALIZED non-negative weight rows
        /// (torch.multinomial): `num_samples` draws per row along
        /// `class_tag` — which must be the last axis, like `routerTopK` —
        /// replaced by `out_tag` in the result. Rank 1 or 2 (the torch
        /// surface). Draw `(row, s)` reads the deterministic counter-based
        /// stream at `(seed, row·num_samples + s)` (see docs/reference/06-the-execution-runtime-execcontext-and-the-memory-model.md), so results are
        /// reproducible and independent of any batching; pass a fresh seed
        /// per call. Rows must hold finite weights `>= 0` with a positive
        /// sum (`InvalidShape` otherwise; NaN/inf rejected). Without
        /// replacement each draw removes the chosen class's mass (torch
        /// semantics); `num_samples` beyond the class count — or beyond the
        /// row's nonzero classes — is `InvalidShape`. The result is a
        /// constant i64 tensor: no gradient, and CALLER-owned even under an
        /// exec scope (the typed-constant ownership rule).
        pub fn multinomial(
            self: *const Self,
            ctx: *ExecContext,
            comptime class_tag: Tag,
            comptime out_tag: Tag,
            num_samples: usize,
            seed: u64,
            replacement: bool,
        ) !Tensor(.{ .dtype = .i64, .tags = replaceTag(tags, class_tag, out_tag) }) {
            comptime {
                if (tag_rank != 1 and tag_rank != 2) @compileError("multinomial takes rank-1 [class] or rank-2 [row, class] weights (the torch surface)");
                if (axis(class_tag) != tag_rank - 1) @compileError("multinomial requires the class tag on the last axis");
            }
            const result_tags = replaceTag(tags, class_tag, out_tag);
            if (num_samples == 0) return TensorError.InvalidShape;

            const raw = self.asRawTensor();
            const classes = raw.shape.at(tag_rank - 1);
            const rows = if (tag_rank == 2) raw.shape.at(0) else 1;
            if (!replacement and num_samples > classes) return TensorError.InvalidShape;

            var prepared: ?RawTensor = null;
            defer if (prepared) |*p| p.deinit();
            const weights_flat = if (raw.isContiguous()) try raw.dataConstChecked() else blk: {
                prepared = try ctx.materialize(.f32, raw);
                break :blk try prepared.?.dataConstChecked();
            };

            const scratch = try ctx.allocator.alloc(f64, classes);
            defer ctx.allocator.free(scratch);

            const out_shape: [tag_rank]usize = if (tag_rank == 2) .{ rows, num_samples } else .{num_samples};
            var out = try Tensor(.{ .dtype = .i64, .tags = result_tags }).empty(ctx, out_shape);
            errdefer out.deinit();
            const out_data = try out.data();

            for (0..rows) |row| {
                const weights = weights_flat[row * classes ..][0..classes];
                var total: f64 = 0;
                for (weights, 0..) |w, c| {
                    if (!(w >= 0) or !std.math.isFinite(w)) return TensorError.InvalidShape;
                    total += w;
                    scratch[c] = if (replacement) total else w;
                }
                if (!(total > 0) or !std.math.isFinite(total)) return TensorError.InvalidShape;

                for (0..num_samples) |s| {
                    // Counter-based [0, 1) draw for (row, s): the uniformFill mapping.
                    const draw = @as(f64, @floatFromInt(rng.at(seed, row * num_samples + s) >> 11)) * 0x1.0p-53;
                    var pick: usize = undefined;
                    if (replacement) {
                        // First index with cumsum > u (zero-weight classes have
                        // zero-width intervals and are never hit).
                        const u = draw * total;
                        var lo: usize = 0;
                        var hi: usize = classes;
                        while (lo < hi) {
                            const mid = lo + (hi - lo) / 2;
                            if (scratch[mid] > u) hi = mid else lo = mid + 1;
                        }
                        pick = @min(lo, classes - 1);
                        // Rounding can land u on the top boundary; step down to mass.
                        while (pick > 0 and weights[pick] == 0) pick -= 1;
                    } else {
                        if (!(total > 0)) return TensorError.InvalidShape; // fewer nonzero classes than draws
                        const u = draw * total;
                        var acc: f64 = 0;
                        var found: ?usize = null;
                        for (scratch[0..classes], 0..) |w, c| {
                            if (w <= 0) continue;
                            acc += w;
                            if (acc > u) {
                                found = c;
                                break;
                            }
                        }
                        if (found == null) {
                            // Rounding hit the top boundary: last remaining class.
                            var c = classes;
                            while (c > 0) {
                                c -= 1;
                                if (scratch[c] > 0) {
                                    found = c;
                                    break;
                                }
                            }
                        }
                        pick = found orelse return TensorError.InvalidShape;
                        total -= scratch[pick];
                        scratch[pick] = 0;
                    }
                    out_data[row * num_samples + s] = @intCast(pick);
                }
            }
            return out;
        }

        /// Max values over `tag` (the tag is removed like sum/mean; argmax
        /// returns the indices). The gradient flows only to the FIRST
        /// occurrence of the extremum along the axis (strict-comparison
        /// tie-break, like PyTorch's torch.max over a dim).
        pub fn max(self: *const Self, ctx: *ExecContext, comptime tag: Tag, opts: anytype) !Tensor(.{ .dtype = reduced_dtype, .tags = removeTag(tags, tag) }) {
            return extremum(self, ctx, tag, .max, opts);
        }

        /// Min values over `tag`; see `max` for gradient/tie-break and mask
        /// semantics (an empty masked lane yields `empty orelse +inf`).
        pub fn min(self: *const Self, ctx: *ExecContext, comptime tag: Tag, opts: anytype) !Tensor(.{ .dtype = reduced_dtype, .tags = removeTag(tags, tag) }) {
            return extremum(self, ctx, tag, .min, opts);
        }

        fn extremum(self: *const Self, ctx: *ExecContext, comptime tag: Tag, comptime op: enum { max, min }, opts: anytype) !Tensor(.{ .dtype = reduced_dtype, .tags = removeTag(tags, tag) }) {
            comptime validateMaskedReduceOptions(@TypeOf(opts));
            const result_tags = removeTag(tags, tag);
            const reduce_axis = comptime axis(tag);
            if (comptime !@hasField(@TypeOf(opts), "mask")) {
                var raw = switch (op) {
                    .max => try ctx.maxAxis(dtype, tag_rank, self.asRawTensor(), reduce_axis),
                    .min => try ctx.minAxis(dtype, tag_rank, self.asRawTensor(), reduce_axis),
                };
                // The first-extremum indices go into the backward node
                // (computed in the forward, not recomputed); the caller only
                // sees values.
                var raw_values: ?RawTensor = raw.values;
                errdefer if (raw_values) |*value| value.deinit();
                defer raw.indices.deinit();
                const out = try finishOrConstant(differentiable, reduced_dtype, result_tags, ctx, raw_values.?, self.requiresGrad(), MinMaxBackward(tags, reduce_axis), .{ ctx.allocator, self.grad_state, &self.value, &raw.indices });
                raw_values = null;
                return out;
            }
            if (comptime !differentiable) @compileError("masked max/min are f32-only; cast to f32 first");

            // Masked arm — `maxval(a, dim, mask)`: `opts` and the mask
            // contract are `sum`'s; tie-break and NaN semantics are the plain
            // arm's, applied to the selected elements only. A lane selecting
            // nothing yields `empty orelse -inf` (`+inf` for min) and
            // receives no gradient (no element participated).
            const Mask = TensorObject(@TypeOf(opts.mask));
            comptime validateMaskType(Mask, "max/min");
            const empty_value = maskedReduceEmpty(opts);
            var raw = switch (op) {
                .max => try ctx.maxMasked(Mask.dtype, tag_rank, self.asRawTensor(), opts.mask.asRawTensor(), reduce_axis, empty_value),
                .min => try ctx.minMasked(Mask.dtype, tag_rank, self.asRawTensor(), opts.mask.asRawTensor(), reduce_axis, empty_value),
            };
            var raw_values: ?RawTensor = raw.values;
            errdefer if (raw_values) |*value| value.deinit();
            defer raw.indices.deinit();
            const out = try finishOp(result_tags, ctx, raw_values.?, self.requiresGrad(), MaskedMinMaxBackward(tags, reduce_axis), .{ ctx.allocator, self.grad_state, &self.value, &raw.indices });
            raw_values = null;
            return out;
        }
    };
}
