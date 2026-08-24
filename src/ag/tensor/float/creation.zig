//! f32 tensor methods: constructors and fills (variable/constant/from*, zeros..eye). A mixin over the ag
//! FloatTensor struct; aliased back onto it in ../../tensor.zig.

const tensor_mod = @import("../../../tensor.zig");
const exec_mod = @import("../../../exec.zig");
const tag_ops = @import("../../../tag_ops.zig");
const core = @import("../../core.zig");
const rng = @import("../../../rng.zig");

const RawTensor = tensor_mod.Tensor;
const TensorError = tensor_mod.TensorError;
const ExecContext = exec_mod.ExecContext;
const GradState = core.GradState;
const validateTensorRank = tag_ops.validateTensorRank;

pub fn Ops(comptime Self: type) type {
    return struct {
        const tags = Self.axis_tags;
        const tensor_rank = Self.tensor_rank;
        const tag_count = Self.tag_count;
        const ag_tensor = Self.ag_root;
        const Tensor = ag_tensor.Tensor;

        /// Consumes `value` on success; on error, ownership stays with the caller.
        pub fn variable(ctx: *ExecContext, value: RawTensor) !Self {
            var v = value;
            try validateTensorRank(.f32, tags, &v);

            const state = try GradState.leaf(ctx.allocator);
            errdefer state.deinit();

            return .{ .value = v, .grad_state = state };
        }

        pub fn variableFromSlice(ctx: *ExecContext, raw_shape: [tensor_rank]usize, values: []const f32) !Self {
            var value = try ctx.fromSlice(.f32, raw_shape, values);
            errdefer value.deinit();
            return Self.variable(ctx, value);
        }

        /// Consumes `value` on success; on error, ownership stays with the caller.
        pub fn constant(ctx: *ExecContext, value: RawTensor) !Self {
            _ = ctx;
            var v = value;
            try validateTensorRank(.f32, tags, &v);
            return .{ .value = v };
        }

        pub fn fromTensor(ctx: *ExecContext, value: RawTensor) !Self {
            return try Self.constant(ctx, value);
        }

        pub fn fromSlice(ctx: *ExecContext, raw_shape: [tensor_rank]usize, values: []const f32) !Self {
            var value = try ctx.fromSlice(.f32, raw_shape, values);
            errdefer value.deinit();
            return try Self.constant(ctx, value);
        }

        /// Wrap caller-owned mutable storage as a no-grad constant tensor.
        /// The returned tensor borrows `values`; callers must keep that slice
        /// alive and unmoved until the tensor is deinitialized.
        pub fn fromBorrowedSlice(ctx: *ExecContext, raw_shape: [tensor_rank]usize, values: []f32) !Self {
            var value = try ctx.fromBorrowedSlice(.f32, raw_shape, values);
            errdefer value.deinit();
            return try Self.constant(ctx, value);
        }

        /// Zero-copy wrap caller-owned READ-ONLY storage (e.g. mmap'd const GGUF
        /// weights) as a no-grad constant tensor, so callers no longer scatter
        /// `@constCast` to turn const file data into a tensor view. The tensor
        /// BORROWS `values`: the slice must outlive the tensor, stay unmoved, and
        /// MUST NOT be mutated through `.data()`. The single internal `@constCast`
        /// is sound only under that read-only contract — use `fromSlice` (which
        /// copies into owned storage) if you need a writable buffer.
        pub fn fromBorrowedConstSlice(ctx: *ExecContext, raw_shape: [tensor_rank]usize, values: []const f32) !Self {
            var value = try ctx.fromBorrowedSlice(.f32, raw_shape, @constCast(values));
            errdefer value.deinit();
            return try Self.constant(ctx, value);
        }

        /// Allocate an uninitialized no-grad tensor of the tag-implied rank.
        pub fn empty(ctx: *ExecContext, raw_shape: [tensor_rank]usize) !Self {
            var value = try ctx.empty(.f32, &raw_shape);
            errdefer value.deinit();
            return try Self.constant(ctx, value);
        }

        /// Allocate a zero-filled no-grad tensor.
        pub fn zeros(ctx: *ExecContext, raw_shape: [tensor_rank]usize) !Self {
            var value = try ctx.zeros(.f32, &raw_shape);
            errdefer value.deinit();
            return try Self.constant(ctx, value);
        }

        /// Allocate a one-filled no-grad tensor.
        pub fn ones(ctx: *ExecContext, raw_shape: [tensor_rank]usize) !Self {
            var value = try ctx.ones(.f32, &raw_shape);
            errdefer value.deinit();
            return try Self.constant(ctx, value);
        }

        /// Allocate a no-grad tensor filled with `fill_value`.
        pub fn full(ctx: *ExecContext, raw_shape: [tensor_rank]usize, fill_value: f32) !Self {
            var value = try ctx.full(.f32, &raw_shape, fill_value);
            errdefer value.deinit();
            return try Self.constant(ctx, value);
        }

        /// Build a single-element no-grad tensor holding `scalar_value`.
        pub fn scalar(ctx: *ExecContext, scalar_value: f32) !Self {
            var value = try ctx.scalar(.f32, scalar_value);
            errdefer value.deinit();
            return try Self.constant(ctx, value);
        }

        /// Rank-1 no-grad tensor holding `start, start+step, …` up to but
        /// excluding `end` (torch.arange with float semantics): element i
        /// is `start + i·step` (not accumulated). `step` must move from
        /// `start` toward `end` — an empty range is `InvalidShape`
        /// (zero-size tensors are not representable), as is `step == 0`.
        pub fn arange(ctx: *ExecContext, start: f32, end: f32, step: f32) !Self {
            comptime if (tag_count != 1) @compileError("arange builds a rank-1 tensor; use a single-tag Tensor type");
            if (step == 0) return TensorError.InvalidShape;
            const span = (end - start) / step;
            if (!(span > 0)) return TensorError.InvalidShape;
            const count: usize = @intFromFloat(@ceil(span));
            var value = try ctx.empty(.f32, &.{count});
            errdefer value.deinit();
            for (value.data(), 0..) |*out, i| out.* = start + @as(f32, @floatFromInt(i)) * step;
            return try Self.constant(ctx, value);
        }

        /// Rank-1 no-grad tensor of `steps` values spaced evenly from
        /// `start` to `end` INCLUSIVE (torch.linspace): element i is
        /// `start + i·(end-start)/(steps-1)` with the final element pinned
        /// to exactly `end`; `steps == 1` yields `{start}`. `steps == 0`
        /// is `InvalidShape` (zero-size tensors are not representable).
        pub fn linspace(ctx: *ExecContext, start: f32, end: f32, steps: usize) !Self {
            comptime if (tag_count != 1) @compileError("linspace builds a rank-1 tensor; use a single-tag Tensor type");
            if (steps == 0) return TensorError.InvalidShape;
            var value = try ctx.empty(.f32, &.{steps});
            errdefer value.deinit();
            const out = value.data();
            if (steps == 1) {
                out[0] = start;
            } else {
                const stride = (end - start) / @as(f32, @floatFromInt(steps - 1));
                for (out, 0..) |*o, i| o.* = start + @as(f32, @floatFromInt(i)) * stride;
                out[steps - 1] = end;
            }
            return try Self.constant(ctx, value);
        }

        /// Rank-2 no-grad one-hot matrix `[indices.len, depth]` (torch
        /// F.one_hot with an explicit class count, as f32): row i holds 1.0
        /// at column `indices[i]`, 0.0 elsewhere. Indices are host-side
        /// like `gather`'s; `indices[i] >= depth` is `IndexOutOfBounds`,
        /// an empty `indices` is `InvalidShape` (zero-size tensors are not
        /// representable). The first tag is the row axis, the second the
        /// class axis.
        pub fn oneHot(ctx: *ExecContext, indices: []const usize, depth: usize) !Self {
            comptime if (tag_count != 2) @compileError("oneHot builds a rank-2 [rows, classes] tensor; use a two-tag Tensor type");
            if (indices.len == 0 or depth == 0) return TensorError.InvalidShape;
            var value = try ctx.zeros(.f32, &.{ indices.len, depth });
            errdefer value.deinit();
            const out = value.data();
            for (indices, 0..) |class_index, row| {
                if (class_index >= depth) return TensorError.IndexOutOfBounds;
                out[row * depth + class_index] = 1;
            }
            return try Self.constant(ctx, value);
        }

        /// No-grad tensor of uniform draws in `[0, 1)` (torch.rand) from
        /// the deterministic counter-based stream at `seed` (see docs/reference/06-the-execution-runtime-execcontext-and-the-memory-model.md;
        /// `fucina.rng`): element i is a pure function of `(seed, i)`, so
        /// the same seed always reproduces the same tensor — the stream IS
        /// the generator abstraction (store the seed, regenerate the
        /// values). Pass a fresh seed per draw (reusing one reuses the
        /// values).
        pub fn rand(ctx: *ExecContext, raw_shape: [tensor_rank]usize, seed: u64) !Self {
            return uniform(ctx, raw_shape, seed, 0, 1);
        }

        /// `rand` over `[lo, hi)` (the `fucina.rng.uniformFill` mapping).
        pub fn uniform(ctx: *ExecContext, raw_shape: [tensor_rank]usize, seed: u64, lo: f32, hi: f32) !Self {
            var value = try ctx.empty(.f32, &raw_shape);
            errdefer value.deinit();
            rng.uniformFill(seed, value.data(), lo, hi);
            return try Self.constant(ctx, value);
        }

        /// No-grad tensor of standard-normal draws (torch.randn) from the
        /// deterministic stream at `seed` (see `rand`); Box-Muller over
        /// the splitmix64 stream (`fucina.rng.gaussianFill`).
        pub fn randn(ctx: *ExecContext, raw_shape: [tensor_rank]usize, seed: u64) !Self {
            return normal(ctx, raw_shape, seed, 0, 1);
        }

        /// `randn` with explicit moments (`fucina.rng.normalFill`).
        pub fn normal(ctx: *ExecContext, raw_shape: [tensor_rank]usize, seed: u64, mean_value: f32, std_dev: f32) !Self {
            var value = try ctx.empty(.f32, &raw_shape);
            errdefer value.deinit();
            rng.normalFill(seed, value.data(), mean_value, std_dev);
            return try Self.constant(ctx, value);
        }

        /// No-grad 0/1 tensor of Bernoulli draws (torch.bernoulli with a
        /// scalar probability): element i is 1.0 iff the `[0, 1)` uniform
        /// stream at `(seed, i)` (see `rand`) draws below `p`. `p` outside
        /// `[0, 1]` is `InvalidShape`.
        pub fn bernoulli(ctx: *ExecContext, raw_shape: [tensor_rank]usize, seed: u64, p: f32) !Self {
            if (!(p >= 0 and p <= 1)) return TensorError.InvalidShape;
            var value = try ctx.empty(.f32, &raw_shape);
            errdefer value.deinit();
            const out = value.data();
            rng.uniformFill(seed, out, 0, 1);
            for (out) |*v| v.* = if (v.* < p) 1 else 0;
            return try Self.constant(ctx, value);
        }

        /// No-grad tensor of standard Gumbel(0, 1) draws — `-ln(-ln(u))`,
        /// the gumbel-max / gumbel-softmax noise — from the deterministic
        /// stream at `seed` (see `rand`; `fucina.rng.gumbelFill` documents
        /// the exact open-interval mapping). Add to logits and take
        /// argmax for a categorical sample, or softmax at a temperature
        /// for its differentiable relaxation.
        pub fn gumbel(ctx: *ExecContext, raw_shape: [tensor_rank]usize, seed: u64) !Self {
            var value = try ctx.empty(.f32, &raw_shape);
            errdefer value.deinit();
            rng.gumbelFill(seed, value.data());
            return try Self.constant(ctx, value);
        }

        /// Identity matrix `[n, n]` (torch.eye) as a no-grad constant: 1.0
        /// on the main diagonal, 0.0 elsewhere. The first tag is the row
        /// axis, the second the column axis. `n == 0` is `InvalidShape`
        /// (zero-size tensors are not representable).
        pub fn eye(ctx: *ExecContext, n: usize) !Self {
            comptime if (tag_count != 2) @compileError("eye builds a rank-2 [n, n] tensor; use a two-tag Tensor type");
            if (n == 0) return TensorError.InvalidShape;
            var value = try ctx.zeros(.f32, &.{ n, n });
            errdefer value.deinit();
            const out = value.data();
            for (0..n) |i| out[i * (n + 1)] = 1;
            return try Self.constant(ctx, value);
        }

        // --- *Like constructors ---------------------------------------------
        // Instance sugar over the static constructors above: same tags and
        // dtype (both are part of `Self`), shape taken from `self`'s logical
        // shape (strided views included). Like every constructor the result
        // is a fresh owned NO-GRAD constant — `self`'s grad state does not
        // carry over — and is never scope-owned.

        /// `empty` with `self`'s shape: uninitialized storage.
        pub fn emptyLike(self: *const Self, ctx: *ExecContext) !Self {
            return Self.empty(ctx, self.shape());
        }

        /// `zeros` with `self`'s shape.
        pub fn zerosLike(self: *const Self, ctx: *ExecContext) !Self {
            return Self.zeros(ctx, self.shape());
        }

        /// `ones` with `self`'s shape.
        pub fn onesLike(self: *const Self, ctx: *ExecContext) !Self {
            return Self.ones(ctx, self.shape());
        }

        /// `full` with `self`'s shape, filled with `fill_value`.
        pub fn fullLike(self: *const Self, ctx: *ExecContext, fill_value: f32) !Self {
            return Self.full(ctx, self.shape(), fill_value);
        }

        // --- Rank-generic matmul --------------------------------------------
        // `out_tags` names the result axes (rank-generic — no fragile
        // tag-composition rule). For tag-semantics contractions, prefer `dot`.
    };
}
