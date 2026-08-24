//! Integer and mask math for the typed scalar branch: explicit division
//! and remainder, the bitwise combinators, and the `.bool` logical
//! combinators. All no-grad constants over the tag-broadcast rule. A mixin
//! over the tensor struct; aliased back onto it in ../../tensor.zig.

const tensor_mod = @import("../../../tensor.zig");
const dtype_mod = @import("../../../dtype.zig");
const exec_mod = @import("../../../exec.zig");
const tag_ops = @import("../../../tag_ops.zig");
const tags_mod = @import("../../../tags.zig");

const ExecContext = exec_mod.ExecContext;
const rawRank = tags_mod.rawRank;
const pointwiseResultTags = tags_mod.pointwiseResultTags;

pub fn Ops(comptime Self: type) type {
    return struct {
        const dtype = Self.dtype;
        const tags = Self.axis_tags;
        const ag_tensor = Self.ag_root;
        const Tensor = ag_tensor.Tensor;
        const plumbing = @import("../plumbing.zig").Mod(ag_tensor);
        const TensorObject = plumbing.TensorObject;
        const tensorObjectPtrFrom = plumbing.tensorObjectPtrFrom;
        const RawT = tensor_mod.TensorOf(dtype);

        fn Out(comptime Other: type) type {
            return Tensor(.{ .dtype = dtype, .tags = pointwiseResultTags(tags, Other.axis_tags) });
        }

        const BoolT = Tensor(.{ .dtype = .bool, .tags = tags });

        /// Explicit integer division with the standard tag-broadcast rule:
        /// `divTrunc` rounds toward zero, `divFloor` toward negative
        /// infinity; a zero divisor is `error.DivisionByZero`; minInt/-1
        /// wraps (the +% contract).
        pub fn divTrunc(self: *const Self, ctx: *ExecContext, other: anytype) !Out(TensorObject(@TypeOf(other))) {
            return binary(.div_trunc, self, ctx, other);
        }

        pub fn divFloor(self: *const Self, ctx: *ExecContext, other: anytype) !Out(TensorObject(@TypeOf(other))) {
            return binary(.div_floor, self, ctx, other);
        }

        /// Explicit integer remainder with the standard tag-broadcast rule:
        /// `rem` pairs `divTrunc` (sign of the dividend, C `%`), `mod` pairs
        /// `divFloor` (sign of the divisor, Python/numpy `%` and Zig's
        /// `@mod`). A zero divisor is `error.DivisionByZero`; minInt % -1
        /// is 0.
        pub fn rem(self: *const Self, ctx: *ExecContext, other: anytype) !Out(TensorObject(@TypeOf(other))) {
            return binary(.rem, self, ctx, other);
        }

        pub fn mod(self: *const Self, ctx: *ExecContext, other: anytype) !Out(TensorObject(@TypeOf(other))) {
            return binary(.mod, self, ctx, other);
        }

        /// Bitwise combinators on two's-complement bit patterns, with the
        /// standard tag-broadcast rule. Integer dtypes only: `.bool` masks
        /// use the truthiness `logicalAnd/Or/Xor`, and floats have no
        /// bit-pattern algebra.
        pub fn bitAnd(self: *const Self, ctx: *ExecContext, other: anytype) !Out(TensorObject(@TypeOf(other))) {
            return binary(.bit_and, self, ctx, other);
        }

        pub fn bitOr(self: *const Self, ctx: *ExecContext, other: anytype) !Out(TensorObject(@TypeOf(other))) {
            return binary(.bit_or, self, ctx, other);
        }

        pub fn bitXor(self: *const Self, ctx: *ExecContext, other: anytype) !Out(TensorObject(@TypeOf(other))) {
            return binary(.bit_xor, self, ctx, other);
        }

        /// Logical ops on the `.bool` branch (the mask combinators): `.bool`
        /// output; `other` may be `.bool` or float (truthiness).
        pub fn logicalAnd(self: *const Self, ctx: *ExecContext, other: anytype) !BoolT {
            return logical(.l_and, self, ctx, other);
        }

        pub fn logicalOr(self: *const Self, ctx: *ExecContext, other: anytype) !BoolT {
            return logical(.l_or, self, ctx, other);
        }

        pub fn logicalXor(self: *const Self, ctx: *ExecContext, other: anytype) !BoolT {
            return logical(.l_xor, self, ctx, other);
        }

        pub fn logicalNot(self: *const Self, ctx: *ExecContext) !BoolT {
            comptime if (dtype != .bool) @compileError("logical ops on the typed branch are .bool-only; cast explicitly");
            var value = try ctx.logicalNot(.bool, self.asRawTensor());
            errdefer value.deinit();
            return BoolT.fromTensor(ctx, value);
        }

        const BinaryOp = enum { div_trunc, div_floor, rem, mod, bit_and, bit_or, bit_xor };

        /// The integer binary family: broadcast both operands to the
        /// pointwise result shape, then run the rank-matched kernel.
        fn binary(comptime op: BinaryOp, self: *const Self, ctx: *ExecContext, other: anytype) !Out(TensorObject(@TypeOf(other))) {
            const Other = TensorObject(@TypeOf(other));
            comptime {
                if (!dtype_mod.supportsIntMath(dtype)) @compileError(switch (op) {
                    .div_trunc, .div_floor => "divTrunc/divFloor are integer ops; floats use div",
                    .rem, .mod => "rem/mod are integer ops; no float counterpart is defined",
                    .bit_and, .bit_or, .bit_xor => "bitAnd/bitOr/bitXor are integer ops; `.bool` masks use logicalAnd/Or/Xor",
                });
                if (Other.dtype != dtype) @compileError("typed pointwise requires matching dtypes; cast explicitly");
            }
            const result_tags = pointwiseResultTags(tags, Other.axis_tags);
            const rank = comptime rawRank(result_tags.len);
            const right = tensorObjectPtrFrom(@TypeOf(other), &other);
            const result_shape = try tag_ops.pointwiseShape(dtype, result_tags, tags, self.asRawTensor(), Other.axis_tags, right.asRawTensor());

            var left_view = try tag_ops.broadcastTensorTo(dtype, tags, self.asRawTensor(), result_tags, result_shape);
            defer left_view.deinit();
            var right_view = try tag_ops.broadcastTensorTo(dtype, Other.axis_tags, right.asRawTensor(), result_tags, result_shape);
            defer right_view.deinit();

            var value: RawT = switch (op) {
                .div_trunc => try ctx.divTrunc(dtype, rank, &left_view, &right_view),
                .div_floor => try ctx.divFloor(dtype, rank, &left_view, &right_view),
                .rem => try ctx.rem(dtype, rank, &left_view, &right_view),
                .mod => try ctx.mod(dtype, rank, &left_view, &right_view),
                .bit_and => try ctx.bitwise(dtype, rank, .b_and, &left_view, &right_view),
                .bit_or => try ctx.bitwise(dtype, rank, .b_or, &left_view, &right_view),
                .bit_xor => try ctx.bitwise(dtype, rank, .b_xor, &left_view, &right_view),
            };
            errdefer value.deinit();
            return Out(Other).fromTensor(ctx, value);
        }

        fn logical(comptime op: exec_mod.LogicalOp, self: *const Self, ctx: *ExecContext, other: anytype) !BoolT {
            const Other = TensorObject(@TypeOf(other));
            comptime {
                if (dtype != .bool) @compileError("logical ops on the typed branch are .bool-only; cast explicitly");
                if (Other.dtype != .bool and !dtype_mod.supportsForwardFloatMath(Other.dtype))
                    @compileError("logical ops take .bool or float operands; cast integer masks explicitly");
            }
            const other_ptr = tensorObjectPtrFrom(@TypeOf(other), &other);
            var value = try ctx.logical(op, .bool, Other.dtype, self.asRawTensor(), other_ptr.asRawTensor());
            errdefer value.deinit();
            return BoolT.fromTensor(ctx, value);
        }
    };
}
