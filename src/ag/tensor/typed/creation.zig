//! Constructors of the typed float and typed scalar branches: the no-grad
//! wrap, slice and borrowed-slice construction, fills, the i64 seed-stream
//! constructors, and the `.bool` band mask. A mixin over the tensor
//! struct; aliased back onto it in ../../tensor.zig.

const tensor_mod = @import("../../../tensor.zig");
const exec_mod = @import("../../../exec.zig");
const tag_ops = @import("../../../tag_ops.zig");
const rng = @import("../../../rng.zig");

const TensorError = tensor_mod.TensorError;
const ExecContext = exec_mod.ExecContext;
const validateTensorRank = tag_ops.validateTensorRank;

pub fn Ops(comptime Self: type) type {
    return struct {
        const dtype = Self.dtype;
        const tags = Self.axis_tags;
        const tensor_rank = Self.tensor_rank;
        const RawT = tensor_mod.TensorOf(dtype);
        const Elem = tensor_mod.Scalar(dtype);

        /// Consumes `value` on success; on error, ownership stays with the caller.
        pub fn constant(ctx: *ExecContext, value: RawT) !Self {
            _ = ctx;
            var v = value;
            try validateTensorRank(dtype, tags, &v);
            return .{ .value = v };
        }

        pub fn fromTensor(ctx: *ExecContext, value: RawT) !Self {
            return try Self.constant(ctx, value);
        }

        pub fn fromSlice(ctx: *ExecContext, raw_shape: [tensor_rank]usize, values: []const Elem) !Self {
            var value = try ctx.fromSlice(dtype, raw_shape, values);
            errdefer value.deinit();
            return try Self.constant(ctx, value);
        }

        /// Zero-copy wrap caller-owned READ-ONLY typed storage as a no-grad
        /// constant tensor without `@constCast` at the call site. Read-only
        /// borrow: `values` must outlive the tensor and must not be mutated
        /// (see the f32 `fromBorrowedConstSlice` contract).
        pub fn fromBorrowedConstSlice(ctx: *ExecContext, raw_shape: [tensor_rank]usize, values: []const Elem) !Self {
            var value = try ctx.fromBorrowedSlice(dtype, raw_shape, @constCast(values));
            errdefer value.deinit();
            return try Self.constant(ctx, value);
        }

        /// Allocate an uninitialized no-grad typed tensor of the tag-implied rank.
        pub fn empty(ctx: *ExecContext, raw_shape: [tensor_rank]usize) !Self {
            var value = try ctx.empty(dtype, raw_shape);
            errdefer value.deinit();
            return try Self.constant(ctx, value);
        }

        /// Allocate a zero-filled no-grad typed tensor.
        pub fn zeros(ctx: *ExecContext, raw_shape: [tensor_rank]usize) !Self {
            var value = try ctx.zeros(dtype, &raw_shape);
            errdefer value.deinit();
            return try Self.constant(ctx, value);
        }

        /// Allocate a one-filled no-grad typed tensor.
        pub fn ones(ctx: *ExecContext, raw_shape: [tensor_rank]usize) !Self {
            var value = try ctx.ones(dtype, &raw_shape);
            errdefer value.deinit();
            return try Self.constant(ctx, value);
        }

        /// No-grad tensor of uniform integer draws in `[low, high)`
        /// (torch.randint) from the deterministic counter-based stream at
        /// `seed` (see docs/reference/06-the-execution-runtime-execcontext-and-the-memory-model.md): element i is a pure function of `(seed, i)` via
        /// the widening multiply-shift map (`fucina.rng.randintFill`).
        /// i64-only (the repo-wide index dtype); cast with `to` for
        /// narrower integers. `low >= high` is `InvalidShape`.
        pub fn randint(ctx: *ExecContext, raw_shape: [tensor_rank]usize, seed: u64, low: i64, high: i64) !Self {
            comptime if (dtype != .i64) @compileError("randint is i64-only (the repo-wide index dtype); cast the result with to()");
            if (low >= high) return TensorError.InvalidShape;
            var value = try ctx.empty(dtype, raw_shape);
            errdefer value.deinit();
            rng.randintFill(seed, value.data(), low, high);
            return try Self.constant(ctx, value);
        }

        /// Rank-1 no-grad random permutation of `{0, ..., n-1}`
        /// (torch.randperm) as i64: Fisher-Yates driven by the
        /// counter-based stream at `seed` (`fucina.rng.randpermFill`;
        /// same seed, same permutation). `n == 0` is `InvalidShape`
        /// (zero-size tensors are not representable).
        pub fn randperm(ctx: *ExecContext, n: usize, seed: u64) !Self {
            comptime {
                if (dtype != .i64) @compileError("randperm is i64-only (the repo-wide index dtype)");
                if (tags.len != 1) @compileError("randperm builds a rank-1 tensor; use a single-tag Tensor type");
            }
            if (n == 0) return TensorError.InvalidShape;
            var value = try ctx.empty(dtype, .{n});
            errdefer value.deinit();
            rng.randpermFill(seed, value.data());
            return try Self.constant(ctx, value);
        }

        /// Rank-2 no-grad band mask (the attention-mask constructor), on
        /// the `.bool` branch only: element `(i, j)` is true iff
        /// `i - j <= lower` and `j - i <= upper`; a null bound is
        /// unbounded on that side. Bounds are signed. Causal keep-set =
        /// `(null, 0)`; sliding window of width W = `(W - 1, 0)`; the
        /// tril(k) keep-set = `(null, k)`; triu(k) = `(-k, null)`. Feed
        /// it to `where`/`maskedFill` (broadcast with `broadcastTo` for
        /// batched scores), or cast with `to(.f32)` for the mask-multiply
        /// idiom. Contradictory bounds yield an all-false mask, not an
        /// error.
        pub fn bandMask(ctx: *ExecContext, raw_shape: [tensor_rank]usize, lower: ?i64, upper: ?i64) !Self {
            comptime {
                if (dtype != .bool) @compileError("bandMask is a .bool mask constructor; use a .bool Tensor type");
                if (tags.len != 2) @compileError("bandMask builds a rank-2 [row, col] mask; use a two-tag Tensor type");
            }
            var value = try ctx.empty(dtype, raw_shape);
            errdefer value.deinit();
            const out = value.data();
            const cols = raw_shape[1];
            for (0..raw_shape[0]) |i| {
                for (0..cols) |j| {
                    const d = @as(i64, @intCast(j)) - @as(i64, @intCast(i)); // j - i
                    const in_lower = if (lower) |l| -d <= l else true;
                    const in_upper = if (upper) |u| d <= u else true;
                    out[i * cols + j] = in_lower and in_upper;
                }
            }
            return try Self.constant(ctx, value);
        }

        /// `empty` with `self`'s shape.
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
    };
}
