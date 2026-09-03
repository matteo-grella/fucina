//! VJPs for contractions: matmul/bmm/dot/einsum and the
//! const-RHS + ternary-STE variants.

const std = @import("std");
const backend_quant = @import("../../backend.zig").quant;
const backend_kernels = @import("../../backend.zig").kernels;
const tensor_mod = @import("../../tensor.zig");
const dtype_mod = @import("../../dtype.zig");
const exec_mod = @import("../../exec.zig");
const parallel = @import("../../parallel.zig");
const tag_ops = @import("../../tag_ops.zig");
const core = @import("../core.zig");
const tags_mod = @import("../../tags.zig");

const RawTensor = tensor_mod.Tensor;
const ExecContext = exec_mod.ExecContext;
const GradState = core.GradState;
const Tag = tags_mod.Tag;
const rawRank = tags_mod.rawRank;
const tagIndex = tags_mod.tagIndex;
const dotResultTags = tags_mod.dotResultTags;
const intersectTags = tags_mod.intersectTags;
const tagsEqual = tags_mod.tagsEqual;

const common = @import("common.zig");
const rawShapeArray = common.rawShapeArray;
const rawShapeArrayOf = common.rawShapeArrayOf;
const tagsDifference = common.tagsDifference;
const contiguousForRead = common.contiguousForRead;
const expandGradientToTags = common.expandGradientToTags;
const runContractionBranches = common.runContractionBranches;

pub fn Matmul2DBackward(comptime trans_b: bool) type {
    return struct {
        parents: [2]?*GradState,
        left: RawTensor,
        right: RawTensor,

        const Self = @This();

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
            if (core.needs(self, 0)) {
                out[0] = if (comptime trans_b)
                    try ctx.matmul(.f32, .plain, gy, &self.right)
                else
                    try ctx.matmul(.f32, .trans_b, gy, &self.right);
            }
            if (core.needs(self, 1)) {
                out[1] = if (comptime trans_b)
                    try ctx.matmul(.f32, .trans_a, gy, &self.left)
                else
                    try ctx.matmul(.f32, .trans_a, &self.left, gy);
            }
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            _ = allocator;
            self.left.deinit();
            self.right.deinit();
        }

        pub const vtable = core.recordVTable(Self);
    };
}

pub fn BmmBackward(comptime kind: exec_mod.BmmKind) type {
    return struct {
        parents: [2]?*GradState,
        left: RawTensor,
        right: RawTensor,

        const Self = @This();

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
            if (core.needs(self, 0)) {
                var full = switch (kind) {
                    .plain => try ctx.bmm(.f32, .trans_b, gy, &self.right),
                    .trans_a => try ctx.bmm(.f32, .trans_b, &self.right, gy),
                    .trans_b => try ctx.bmm(.f32, .plain, gy, &self.right),
                };
                defer full.deinit();
                out[0] = try ctx.reduceBroadcast(&full, self.left.shape.slice());
            }

            if (core.needs(self, 1)) {
                var full = switch (kind) {
                    .plain => try ctx.bmm(.f32, .trans_a, &self.left, gy),
                    .trans_a => try ctx.bmm(.f32, .plain, &self.left, gy),
                    .trans_b => try ctx.bmm(.f32, .trans_a, gy, &self.left),
                };
                defer full.deinit();
                out[1] = try ctx.reduceBroadcast(&full, self.right.shape.slice());
            }
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            _ = allocator;
            self.left.deinit();
            self.right.deinit();
        }

        pub const vtable = core.recordVTable(Self);
    };
}

pub fn DotBackward(comptime left_tags: anytype, comptime right_tags: anytype, comptime contract_tag: Tag) type {
    // `dot` is the single-contract-tag einsum; its VJP record is the einsum
    // one with the canonical dot result order as the equation.
    return EinsumBackward(left_tags, right_tags, dotResultTags(left_tags, right_tags, contract_tag));
}

/// Backward for `einsum` (multi-index tagged contraction). Contractions are
/// closed under differentiation: the gradient w.r.t. one operand is another
/// einsum — the output gradient contracted with the other operand, keeping
/// exactly this operand's recoverable tags — broadcast over any axes the
/// forward summed away. Both operand gradients therefore lower onto GEMM/BMM
/// kernels; there is no pointwise fallback.
pub fn EinsumBackward(comptime left_tags: anytype, comptime right_tags: anytype, comptime out_tags: anytype) type {
    // Unions here are membership sets, not tensor tag tuples: they may
    // legally exceed max_rank (`unionTags`, not `pointwiseResultTags`).
    const left_recover_tags = intersectTags(left_tags, tags_mod.unionTags(out_tags, right_tags));
    const right_recover_tags = intersectTags(right_tags, tags_mod.unionTags(out_tags, left_tags));
    const dropped_tags = tagsDifference(tags_mod.unionTags(left_tags, right_tags), out_tags);

    return struct {
        parents: [2]?*GradState,
        left_shape: [rawRank(left_tags.len)]usize,
        right_shape: [rawRank(right_tags.len)]usize,
        estimated_work: usize,
        left_value: RawTensor,
        right_value: RawTensor,

        const Self = @This();

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
            return runContractionBranches(self, ctx, gy, &out[0], &out[1], core.needs(self, 0), core.needs(self, 1));
        }

        pub fn backwardLeft(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, out: *?RawTensor) !void {
            var grad = try tag_ops.taggedEinsum(.f32, out_tags, gy, ctx, right_tags, &self.right_value, left_recover_tags);
            defer grad.deinit();
            out.* = try expandGradientToTags(left_recover_tags, left_tags, ctx, &grad, self.left_shape);
        }

        pub fn backwardRight(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, out: *?RawTensor) !void {
            var grad = try tag_ops.taggedEinsum(.f32, out_tags, gy, ctx, left_tags, &self.left_value, right_recover_tags);
            defer grad.deinit();
            out.* = try expandGradientToTags(right_recover_tags, right_tags, ctx, &grad, self.right_shape);
        }

        pub fn einsumBackwardWorkEstimate(left_parent: ?*GradState, right_parent: ?*GradState, left: *const RawTensor, right: *const RawTensor) usize {
            var branches: usize = 0;
            if (left_parent != null) branches += 1;
            if (right_parent != null) branches += 1;
            if (branches == 0) return 0;

            var result_elems: usize = 1;
            inline for (out_tags) |tag| {
                result_elems = saturatedMul(result_elems, dimForTag(tag, left, right));
            }
            var dropped_elems: usize = 1;
            inline for (dropped_tags) |tag| {
                dropped_elems = saturatedMul(dropped_elems, dimForTag(tag, left, right));
            }
            return parallel.saturatedMul3(result_elems, dropped_elems, branches);
        }

        fn dimForTag(comptime tag: Tag, left: *const RawTensor, right: *const RawTensor) usize {
            if (comptime tagIndex(left_tags, tag)) |axis| return left.shape.at(axis);
            if (comptime tagIndex(right_tags, tag)) |axis| return right.shape.at(axis);
            unreachable;
        }

        fn saturatedMul(a: usize, b: usize) usize {
            return std.math.mul(usize, a, b) catch std.math.maxInt(usize);
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            _ = allocator;
            self.left_value.deinit();
            self.right_value.deinit();
        }

        pub const vtable = core.recordVTable(Self);
    };
}

/// Backward for `addDot` (`base + left·right`, the addmm form). The base
/// branch is an identity: its contribution is the output gradient itself,
/// shared as a view. The left/right branches are the dot VJP contractions,
/// with the same parallel branch split as `EinsumBackward`.
pub fn AddDotBackward(comptime base_tags: anytype, comptime left_tags: anytype, comptime right_tags: anytype, comptime contract_tag: Tag) type {
    comptime std.debug.assert(tagsEqual(dotResultTags(left_tags, right_tags, contract_tag), base_tags));

    return struct {
        parents: [3]?*GradState,
        estimated_work: usize,
        left_value: RawTensor,
        right_value: RawTensor,

        const Self = @This();

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
            const need_base = core.needs(self, 0);
            const need_left = core.needs(self, 1);
            const need_right = core.needs(self, 2);

            if (need_base) {
                out[0] = try gy.cloneView();
            }

            return runContractionBranches(self, ctx, gy, &out[1], &out[2], need_left, need_right);
        }

        pub fn backwardLeft(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, out: *?RawTensor) !void {
            out.* = try tag_ops.taggedEinsum(.f32, base_tags, gy, ctx, right_tags, &self.right_value, left_tags);
        }

        pub fn backwardRight(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, out: *?RawTensor) !void {
            out.* = try tag_ops.taggedEinsum(.f32, base_tags, gy, ctx, left_tags, &self.left_value, right_tags);
        }

        pub fn workEstimate(left_parent: ?*GradState, right_parent: ?*GradState, left: *const RawTensor, right: *const RawTensor) usize {
            var branches: usize = 0;
            if (left_parent != null) branches += 1;
            if (right_parent != null) branches += 1;
            if (branches == 0) return 0;
            const m = left.shape.at(0);
            const k = left.shape.at(1);
            const n = right.shape.at(1);
            return parallel.saturatedMul3(m * k, n, branches);
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            _ = allocator;
            self.left_value.deinit();
            self.right_value.deinit();
        }

        pub const vtable = core.recordVTable(Self);
    };
}

/// Backward for `dot` against a quantized, f16, or bf16 RHS. The f32 LHS
/// (activation) gradient is computed against the RHS widened to f32 at
/// backward time; the widened copy is transient — weight memory stays
/// quantized between steps — which makes fine-tuning through frozen GGUF
/// weights possible without duplicating them in f32 permanently. A
/// grad-requiring f16/bf16 RHS variable additionally receives its own f32
/// gradient (see `ConstRhsEinsumBackward`).
pub fn ConstRhsDotBackward(
    comptime rhs_dtype: tensor_mod.DType,
    comptime left_tags: anytype,
    comptime right_tags: anytype,
    comptime contract_tag: Tag,
) type {
    // Same delegation as DotBackward: dot is the single-contract-tag einsum
    // (every left tag is recoverable, so no broadcast expansion happens).
    return ConstRhsEinsumBackward(rhs_dtype, left_tags, right_tags, dotResultTags(left_tags, right_tags, contract_tag));
}

/// Backward for `einsum` against a quantized, f16, or bf16 RHS. The f32 LHS
/// gradient is computed against the RHS widened to f32 at backward time (the
/// widened copy is transient -- weight memory stays narrow between steps).
/// When the RHS is a grad-requiring f16/bf16 variable, its gradient also
/// flows -- as f32 (gradients are always f32), the plain einsum of the
/// upstream gradient with the saved f32 LHS. The LHS value is retained only
/// in that case, so frozen-weight fine-tuning keeps its memory profile.
pub fn ConstRhsEinsumBackward(
    comptime rhs_dtype: tensor_mod.DType,
    comptime left_tags: anytype,
    comptime right_tags: anytype,
    comptime out_tags: anytype,
) type {
    const left_recover_tags = intersectTags(left_tags, tags_mod.unionTags(out_tags, right_tags));
    const right_recover_tags = intersectTags(right_tags, tags_mod.unionTags(out_tags, left_tags));

    return struct {
        parents: [2]?*GradState,
        left_shape: [rawRank(left_tags.len)]usize,
        right_shape: [rawRank(right_tags.len)]usize,
        right_value: tensor_mod.TensorOf(rhs_dtype),
        left_value: ?RawTensor,

        const Self = @This();

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
            if (core.needs(self, 0)) {
                var right_f32 = if (comptime dtype_mod.isBlockQuantized(rhs_dtype))
                    try ctx.dequantizeTensor(rhs_dtype, &self.right_value)
                else
                    try ctx.cast(rhs_dtype, .f32, &self.right_value);
                defer right_f32.deinit();

                // dL/dleft is itself a contraction (einsum closure); axes the
                // forward summed away come back as a broadcast.
                var grad = try tag_ops.taggedEinsum(.f32, out_tags, gy, ctx, right_tags, &right_f32, left_recover_tags);
                defer grad.deinit();
                out[0] = try expandGradientToTags(left_recover_tags, left_tags, ctx, &grad, self.left_shape);
            }
            if (core.needs(self, 1)) {
                var grad = try tag_ops.taggedEinsum(.f32, out_tags, gy, ctx, left_tags, &self.left_value.?, right_recover_tags);
                defer grad.deinit();
                out[1] = try expandGradientToTags(right_recover_tags, right_tags, ctx, &grad, self.right_shape);
            }
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            _ = allocator;
            self.right_value.deinit();
            if (self.left_value) |*left| left.deinit();
        }

        pub const vtable = core.recordVTable(Self);
    };
}

/// Backward for `dotTernarySte`: forward ran the flattened activation
/// `x2d [m, k]` against a TQ2_0-encoded snapshot of the latent weight.
/// dx flows through the QUANTIZED weight — the encoded rows are dequantized
/// transiently (scale-correct, mirroring `ConstRhsDotBackward`) and gy is
/// contracted against them. dW is the straight-through estimate: the plain
/// trans_b matmul VJP `gyᵀ @ x` against the latent weight (identity through
/// the quantizer — no clipping or masking).
pub fn TernarySteDotBackward(comptime left_tags: anytype) type {
    return struct {
        parents: [2]?*GradState,
        // Flattened contiguous [m, k] activation view the forward contracted.
        left: RawTensor,
        left_shape: [rawRank(left_tags.len)]usize,
        estimated_work: usize,
        // Encoded weight snapshot; owned (freed by deinit).
        rhs: backend_quant.QuantizedMatmulRhsTQ2_0,

        const Self = @This();

        /// DotBackward's accounting adapted to this op's fixed shapes: each
        /// live branch is one [m, n] x [n, k]-shaped dense contraction, so
        /// work = result_elems (m*n) * contract (k) * branches. The dx
        /// branch additionally dequantizes the [n, k] snapshot — one more
        /// n*k pass, dominated by the contractions above — so the dot-shaped
        /// estimate stays representative.
        pub fn workEstimate(left_parent: ?*GradState, right_parent: ?*GradState, left2d: *const RawTensor, n: usize) usize {
            var branches: usize = 0;
            if (left_parent != null) branches += 1;
            if (right_parent != null) branches += 1;
            if (branches == 0) return 0;

            const m = left2d.shape.at(0);
            const k = left2d.shape.at(1);
            const result_elems = std.math.mul(usize, m, n) catch std.math.maxInt(usize);
            return parallel.saturatedMul3(result_elems, k, branches);
        }

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
            const m = self.left.shape.at(0);
            const k = self.left.shape.at(1);
            const n = self.rhs.n;

            var gy_ready = try contiguousForRead(ctx, gy);
            defer gy_ready.deinit();
            var gy2d = try gy_ready.reshape(&.{ m, n });
            defer gy2d.deinit();

            if (core.needs(self, 0)) {
                var right_f32 = try ctx.empty(.f32, .{ n, k });
                defer right_f32.deinit();
                const rows = right_f32.data();
                for (0..n) |row| {
                    try backend_kernels.dequantizeRowTQ2_0Into(rows[row * k ..][0..k], self.rhs.columnBlocks(row));
                }
                var dx = try ctx.matmul(.f32, .plain, &gy2d, &right_f32);
                errdefer dx.deinit();
                if (!std.mem.eql(usize, dx.shape.slice(), self.left_shape[0..])) {
                    const reshaped = try dx.reshape(self.left_shape[0..]);
                    dx.deinit();
                    dx = reshaped;
                }
                out[0] = dx;
            }

            if (core.needs(self, 1)) {
                out[1] = try ctx.matmul(.f32, .trans_a, &gy2d, &self.left);
            }
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            _ = allocator;
            self.left.deinit();
            self.rhs.deinit();
        }

        pub const vtable = core.recordVTable(Self);
    };
}
