//! Non-f32 float dtypes on the public Tensor: widened forward math against
//! the narrowed f32 reference, f8 round-trips, 16-bit leaves with f32
//! gradients, and master-weight training through the differentiable narrow.

const std = @import("std");
const backend_mod = @import("../../backend.zig");
const dtype_mod = @import("../../dtype.zig");
const exec_mod = @import("../../exec.zig");
const control = @import("../control.zig");
const core = @import("../core.zig");
const ag_tensor = @import("../tensor.zig");
const gradcheck_mod = @import("../gradcheck.zig");

const DType = dtype_mod.DType;
const ExecContext = exec_mod.ExecContext;
const GradState = core.GradState;
const Tensor = ag_tensor.Tensor;
const RawTensor = @import("../../tensor.zig").Tensor;

test "public non-f32 float Tensor exposes forward math at comptime" {
    inline for (.{ DType.bf16, DType.f16, DType.f64 }) |float_dtype| {
        const T = Tensor(.{ .dtype = float_dtype, .tags = .{ .batch, .d } });
        const expected = .{
            "to",         "add",        "sub",      "mul",        "div",        "sum",       "mean",    "sumAll",
            "dot",        "split",      "merge",    "flatten",    "reshape",    "sliceStep", "flip",    "roll",
            "stack",      "repeatAxis", "scale",    "divScalar",  "unary",      "relu",      "exp",     "sqrt",
            "rsqrt",      "sigmoid",    "silu",     "log",        "log1p",      "neg",       "abs",     "sin",
            "cos",        "tanh",       "fastTanh", "softcap",    "gelu",       "quickGelu", "elu",     "geluErf",
            "floor",      "ceil",       "round",    "sign",       "reciprocal", "leakyRelu", "clamp",   "addScalar",
            "subScalar",  "powScalar",  "maximum",  "minimum",    "gated",      "glu",       "swiglu",  "geglu",
            "softmax",    "logSoftmax", "rmsNorm",  "rmsNormMul", "layerNorm",  "cumsum",    "cumprod", "where",
            "maskedFill", "compare",    "pad",      "max",        "min",        "argmax",    "prod",    "variance",
            "logsumexp",  "einsum",
        };
        inline for (expected) |decl_name| {
            if (!@hasDecl(T, decl_name)) @compileError("non-f32 float Tensor missing forward operation: " ++ decl_name);
        }
    }
}

test "public non-f32 Tensor supports tag-only views" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var ids = try Tensor(.{ .dtype = .i64, .rank = 2 }).fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer ids.deinit();

    try std.testing.expect(@TypeOf(ids).axis_tags[0] == ._0);
    try std.testing.expect(@TypeOf(ids).axis_tags[1] == ._1);

    var named = try ids.withTags(&ctx, .{ .batch, .seq });
    defer named.deinit();
    try std.testing.expect(@TypeOf(named).dtype == .i64);
    try std.testing.expect(@TypeOf(named).axis_tags[0] == .batch);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, named.asRawTensor().shape.slice());

    var transposed = try named.transpose(&ctx, .{ .seq, .batch });
    defer transposed.deinit();
    var transposed_data: [6]i64 = undefined;
    try transposed.asRawTensor().copyTo(&transposed_data);
    try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, transposed.asRawTensor().shape.slice());
    try std.testing.expectEqualSlices(i64, &.{ 1, 4, 2, 5, 3, 6 }, &transposed_data);

    var row = try Tensor(.{ .dtype = .i64, .tags = .{.seq} }).fromSlice(&ctx, .{3}, &.{ 7, 8, 9 });
    defer row.deinit();
    var broadcasted = try row.broadcastTo(&ctx, .{ .batch, .seq }, .{ 2, 3 });
    defer broadcasted.deinit();
    var broadcasted_data: [6]i64 = undefined;
    try broadcasted.asRawTensor().copyTo(&broadcasted_data);
    try std.testing.expectEqualSlices(i64, &.{ 7, 8, 9, 7, 8, 9 }, &broadcasted_data);

    var gathered = try named.gather(&ctx, .batch, &.{ 1, 0 }, .token);
    defer gathered.deinit();
    try std.testing.expect(@TypeOf(gathered).axis_tags[0] == .token);
    try std.testing.expectEqualSlices(i64, &.{ 4, 5, 6, 1, 2, 3 }, gathered.asRawTensor().dataConst());

    var narrowed = try named.narrow(&ctx, .seq, 1, 2);
    defer narrowed.deinit();
    var narrowed_data: [4]i64 = undefined;
    try narrowed.asRawTensor().copyTo(&narrowed_data);
    try std.testing.expectEqualSlices(i64, &.{ 2, 3, 5, 6 }, &narrowed_data);

    var extra = try Tensor(.{ .dtype = .i64, .tags = .{ .batch, .seq } }).fromSlice(&ctx, .{ 2, 1 }, &.{ 10, 11 });
    defer extra.deinit();
    var joined = try named.concat(&ctx, .seq, &.{&extra});
    defer joined.deinit();
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 10, 4, 5, 6, 11 }, joined.asRawTensor().dataConst());

    var update = try Tensor(.{ .dtype = .i64, .tags = .{ .batch, .seq } }).fromSlice(&ctx, .{ 2, 2 }, &.{ 20, 21, 22, 23 });
    defer update.deinit();
    var replaced = try joined.setSlice(&ctx, .seq, 1, &update);
    defer replaced.deinit();
    try std.testing.expectEqualSlices(i64, &.{ 1, 20, 21, 10, 4, 22, 23, 11 }, replaced.asRawTensor().dataConst());
}

test "public non-f32 float Tensor supports forward math" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try Tensor(.{ .dtype = .bf16, .tags = .{ .batch, .d } }).fromSlice(&ctx, .{ 2, 2 }, &.{
        dtype_mod.f32ToBf16(1),
        dtype_mod.f32ToBf16(2),
        dtype_mod.f32ToBf16(3),
        dtype_mod.f32ToBf16(4),
    });
    defer a.deinit();
    var b = try Tensor(.{ .dtype = .bf16, .tags = .{ .batch, .d } }).fromSlice(&ctx, .{ 2, 2 }, &.{
        dtype_mod.f32ToBf16(10),
        dtype_mod.f32ToBf16(20),
        dtype_mod.f32ToBf16(30),
        dtype_mod.f32ToBf16(40),
    });
    defer b.deinit();

    var sum = try a.add(&ctx, &b);
    defer sum.deinit();
    try std.testing.expectEqual(@as(f32, 11), dtype_mod.bf16ToF32(sum.asRawTensor().dataConst()[0]));
    try std.testing.expectEqual(@as(f32, 44), dtype_mod.bf16ToF32(sum.asRawTensor().dataConst()[3]));

    var reduced = try sum.sum(&ctx, .d, .{});
    defer reduced.deinit();
    try std.testing.expect(@TypeOf(reduced).dtype == .f32);
    try std.testing.expectEqualSlices(f32, &.{ 33, 77 }, reduced.asRawTensor().dataConst());

    var as_f32 = try reduced.to(&ctx, .f32);
    defer as_f32.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 33, 77 }, as_f32.asRawTensor().dataConst());

    var f32_source = try Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ 1, 2 }, &.{ 1.5, -2.25 });
    defer f32_source.deinit();
    var bf16_cast = try f32_source.to(&ctx, .bf16);
    defer bf16_cast.deinit();
    var f32_roundtrip = try bf16_cast.to(&ctx, .f32);
    defer f32_roundtrip.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1.5, -2.25 }, f32_roundtrip.asRawTensor().dataConst());

    var left = try Tensor(.{ .dtype = .f16, .tags = .{ .m, .k } }).fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer left.deinit();
    var right = try Tensor(.{ .dtype = .f16, .tags = .{ .k, .n } }).fromSlice(&ctx, .{ 3, 2 }, &.{ 7, 8, 9, 10, 11, 12 });
    defer right.deinit();
    var product = try left.dot(&ctx, &right, .k);
    defer product.deinit();
    try std.testing.expectEqualSlices(f16, &.{ 58, 64, 139, 154 }, product.asRawTensor().dataConst());
}

test "public f8 storage tensors round-trip through to casts" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // f32 -> f8_e4m3 -> f32: values on the e4m3 grid come back exactly.
    var src = try Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ 2, 2 }, &.{ 1.0, -2.0, 0.5, 448.0 });
    defer src.deinit();
    var as_e4m3 = try src.to(&ctx, .f8_e4m3);
    defer as_e4m3.deinit();
    try std.testing.expect(@TypeOf(as_e4m3).dtype == .f8_e4m3);
    try std.testing.expectEqualSlices(u8, &.{ 0x38, 0xc0, 0x30, 0x7e }, as_e4m3.asRawTensor().dataConst());
    var back = try as_e4m3.to(&ctx, .f32);
    defer back.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1.0, -2.0, 0.5, 448.0 }, back.asRawTensor().dataConst());

    // Raw-bits construction (the bf16 storage convention) and the e5m2 leg.
    var e5 = try Tensor(.{ .dtype = .f8_e5m2, .tags = .{.d} }).fromSlice(&ctx, .{2}, &.{
        dtype_mod.f32ToF8e5m2(3.0),
        dtype_mod.f32ToF8e5m2(-0.125),
    });
    defer e5.deinit();
    var e5_f32 = try e5.to(&ctx, .f32);
    defer e5_f32.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 3.0, -0.125 }, e5_f32.asRawTensor().dataConst());
}

test "public non-f32 float Tensor dot supports multi-free and batch tags" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var left = try Tensor(.{ .dtype = .f16, .tags = .{ .m, .h, .k } }).fromSlice(&ctx, .{ 2, 2, 2 }, &.{
        1, 2,
        3, 4,
        5, 6,
        7, 8,
    });
    defer left.deinit();
    var right = try Tensor(.{ .dtype = .f16, .tags = .{ .k, .n, .v } }).fromSlice(&ctx, .{ 2, 2, 2 }, &.{
        1, 2,
        3, 4,
        5, 6,
        7, 8,
    });
    defer right.deinit();
    var product = try left.dot(&ctx, &right, .k);
    defer product.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 2, 2 }, product.asRawTensor().shape.slice());
    try std.testing.expectEqualSlices(f16, &.{
        11, 14, 17, 20,
        23, 30, 37, 44,
        35, 46, 57, 68,
        47, 62, 77, 92,
    }, product.asRawTensor().dataConst());

    var batched_left = try Tensor(.{ .dtype = .f16, .tags = .{ .batch, .m, .k } }).fromSlice(&ctx, .{ 2, 2, 2 }, &.{
        1, 2,
        3, 4,
        5, 6,
        7, 8,
    });
    defer batched_left.deinit();
    var batched_right = try Tensor(.{ .dtype = .f16, .tags = .{ .batch, .k, .n } }).fromSlice(&ctx, .{ 2, 2, 2 }, &.{
        1, 2,
        3, 4,
        5, 6,
        7, 8,
    });
    defer batched_right.deinit();
    var batched_product = try batched_left.dot(&ctx, &batched_right, .k);
    defer batched_product.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 2 }, batched_product.asRawTensor().shape.slice());
    try std.testing.expectEqualSlices(f16, &.{
        7,  10,
        15, 22,
        67, 78,
        91, 106,
    }, batched_product.asRawTensor().dataConst());
}

test "typed float widened unary family matches the narrowed f32 reference" {
    @setEvalBranchQuota(1_000_000);
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Positive inputs so log/sqrt/rsqrt stay in-domain for every op.
    var x32 = try Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ 2, 2 }, &.{ 0.5, 1.25, 2.0, 3.5 });
    defer x32.deinit();

    inline for (.{ DType.f16, DType.bf16 }) |float_dtype| {
        const Scalar = dtype_mod.Scalar(float_dtype);
        var x_t = try x32.to(&ctx, float_dtype);
        defer x_t.deinit();

        const unary_names = .{
            "relu", "exp",     "sqrt",  "rsqrt", "sigmoid", "silu",     "log",        "log1p",
            "neg",  "abs",     "sin",   "cos",   "tanh",    "fastTanh", "gelu",       "quickGelu",
            "elu",  "geluErf", "floor", "ceil",  "round",   "sign",     "reciprocal",
        };
        inline for (unary_names) |name| {
            var got = try @field(@TypeOf(x_t), name)(&x_t, &ctx);
            defer got.deinit();
            var ref32 = try @field(@TypeOf(x32), name)(&x32, &ctx);
            defer ref32.deinit();
            var ref = try ref32.to(&ctx, float_dtype);
            defer ref.deinit();
            try std.testing.expectEqualSlices(Scalar, ref.asRawTensor().dataConst(), got.asRawTensor().dataConst());
        }

        // The generic entry, the parameterized pointwise ops, and the
        // scalar variants take the same widen -> f32 -> narrow route.
        var capped_t = try x_t.softcap(&ctx, 30);
        defer capped_t.deinit();
        var capped_ref32 = try x32.softcap(&ctx, 30);
        defer capped_ref32.deinit();
        var capped_ref = try capped_ref32.to(&ctx, float_dtype);
        defer capped_ref.deinit();
        try std.testing.expectEqualSlices(Scalar, capped_ref.asRawTensor().dataConst(), capped_t.asRawTensor().dataConst());

        var via_unary = try x_t.unary(&ctx, .silu);
        defer via_unary.deinit();
        var silu_ref32 = try x32.silu(&ctx);
        defer silu_ref32.deinit();
        var silu_ref = try silu_ref32.to(&ctx, float_dtype);
        defer silu_ref.deinit();
        try std.testing.expectEqualSlices(Scalar, silu_ref.asRawTensor().dataConst(), via_unary.asRawTensor().dataConst());

        var leaky = try x_t.leakyRelu(&ctx, 0.1);
        defer leaky.deinit();
        var leaky_ref32 = try x32.leakyRelu(&ctx, 0.1);
        defer leaky_ref32.deinit();
        var leaky_ref = try leaky_ref32.to(&ctx, float_dtype);
        defer leaky_ref.deinit();
        try std.testing.expectEqualSlices(Scalar, leaky_ref.asRawTensor().dataConst(), leaky.asRawTensor().dataConst());

        var clamped = try x_t.clamp(&ctx, 1.0, 2.0);
        defer clamped.deinit();
        var clamp_ref32 = try x32.clamp(&ctx, 1.0, 2.0);
        defer clamp_ref32.deinit();
        var clamp_ref = try clamp_ref32.to(&ctx, float_dtype);
        defer clamp_ref.deinit();
        try std.testing.expectEqualSlices(Scalar, clamp_ref.asRawTensor().dataConst(), clamped.asRawTensor().dataConst());

        var scaled = try x_t.scale(&ctx, 3.0);
        defer scaled.deinit();
        var shifted = try x_t.addScalar(&ctx, 1.5);
        defer shifted.deinit();
        var shifted_back = try shifted.subScalar(&ctx, 1.5);
        defer shifted_back.deinit();
        var halved = try x_t.divScalar(&ctx, 2.0);
        defer halved.deinit();
        var squared = try x_t.powScalar(&ctx, 2.0);
        defer squared.deinit();
        // 0.5/1.25/2.0/3.5 are exact in f16 AND bf16, so exact-arithmetic
        // scalar results survive the narrow bit-for-bit.
        var squared_ref32 = try x32.powScalar(&ctx, 2.0);
        defer squared_ref32.deinit();
        var squared_ref = try squared_ref32.to(&ctx, float_dtype);
        defer squared_ref.deinit();
        try std.testing.expectEqualSlices(Scalar, squared_ref.asRawTensor().dataConst(), squared.asRawTensor().dataConst());
        try std.testing.expectEqualSlices(Scalar, x_t.asRawTensor().dataConst(), shifted_back.asRawTensor().dataConst());
    }
}

test "typed float widened binary, gated, and mask ops match the narrowed f32 reference" {
    @setEvalBranchQuota(1_000_000);
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a32 = try Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ 2, 2 }, &.{ 1.0, -2.0, 3.0, -4.0 });
    defer a32.deinit();
    var b32 = try Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ 2, 2 }, &.{ -1.5, 2.5, 0.5, 4.0 });
    defer b32.deinit();
    var bias32 = try Tensor(.{.d}).fromSlice(&ctx, .{2}, &.{ 0.5, -0.5 });
    defer bias32.deinit();

    inline for (.{ DType.f16, DType.bf16 }) |float_dtype| {
        const Scalar = dtype_mod.Scalar(float_dtype);
        var a_t = try a32.to(&ctx, float_dtype);
        defer a_t.deinit();
        var b_t = try b32.to(&ctx, float_dtype);
        defer b_t.deinit();
        var bias_t = try bias32.to(&ctx, float_dtype);
        defer bias_t.deinit();

        var hi = try a_t.maximum(&ctx, &b_t);
        defer hi.deinit();
        var hi_ref32 = try a32.maximum(&ctx, &b32);
        defer hi_ref32.deinit();
        var hi_ref = try hi_ref32.to(&ctx, float_dtype);
        defer hi_ref.deinit();
        try std.testing.expectEqualSlices(Scalar, hi_ref.asRawTensor().dataConst(), hi.asRawTensor().dataConst());

        // maximum against a lower-rank operand exercises the tag broadcast.
        var hi_bias = try a_t.minimum(&ctx, &bias_t);
        defer hi_bias.deinit();
        var hi_bias_ref32 = try a32.minimum(&ctx, &bias32);
        defer hi_bias_ref32.deinit();
        var hi_bias_ref = try hi_bias_ref32.to(&ctx, float_dtype);
        defer hi_bias_ref.deinit();
        try std.testing.expectEqualSlices(Scalar, hi_bias_ref.asRawTensor().dataConst(), hi_bias.asRawTensor().dataConst());

        inline for (.{ "glu", "swiglu", "geglu" }) |name| {
            var got = try @field(@TypeOf(a_t), name)(&a_t, &ctx, &b_t);
            defer got.deinit();
            var ref32 = try @field(@TypeOf(a32), name)(&a32, &ctx, &b32);
            defer ref32.deinit();
            var ref = try ref32.to(&ctx, float_dtype);
            defer ref.deinit();
            try std.testing.expectEqualSlices(Scalar, ref.asRawTensor().dataConst(), got.asRawTensor().dataConst());
        }

        // compare is .bool on every branch (torch's comparison dtype).
        var mask = try a_t.compare(&ctx, .gt, 0.0);
        defer mask.deinit();
        comptime std.debug.assert(@TypeOf(mask).dtype == .bool);
        var mask_ref32 = try a32.compare(&ctx, .gt, 0.0);
        defer mask_ref32.deinit();
        try std.testing.expectEqualSlices(bool, try mask_ref32.dataConst(), try mask.dataConst());

        var tensor_mask = try a_t.compare(&ctx, .lt, &b_t);
        defer tensor_mask.deinit();

        var filled = try a_t.maskedFill(&ctx, &tensor_mask, 9.0);
        defer filled.deinit();
        var mask32 = try a32.compare(&ctx, .lt, &b32);
        defer mask32.deinit();
        var filled_ref32 = try a32.maskedFill(&ctx, &mask32, 9.0);
        defer filled_ref32.deinit();
        var filled_ref = try filled_ref32.to(&ctx, float_dtype);
        defer filled_ref.deinit();
        try std.testing.expectEqualSlices(Scalar, filled_ref.asRawTensor().dataConst(), filled.asRawTensor().dataConst());

        var chosen = try a_t.where(&ctx, &tensor_mask, &b_t);
        defer chosen.deinit();
        var chosen_ref32 = try a32.where(&ctx, &mask32, &b32);
        defer chosen_ref32.deinit();
        var chosen_ref = try chosen_ref32.to(&ctx, float_dtype);
        defer chosen_ref.deinit();
        try std.testing.expectEqualSlices(Scalar, chosen_ref.asRawTensor().dataConst(), chosen.asRawTensor().dataConst());
    }
}

test "typed float softmax and norm family matches the narrowed f32 reference" {
    @setEvalBranchQuota(1_000_000);
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x32 = try Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ 2, 3 }, &.{ 0.5, -1.0, 2.0, 1.5, 0.25, -0.75 });
    defer x32.deinit();
    var w32 = try Tensor(.{.d}).fromSlice(&ctx, .{3}, &.{ 1.0, 0.5, 2.0 });
    defer w32.deinit();

    inline for (.{ DType.f16, DType.bf16 }) |float_dtype| {
        const Scalar = dtype_mod.Scalar(float_dtype);
        var x_t = try x32.to(&ctx, float_dtype);
        defer x_t.deinit();
        var w_t = try w32.to(&ctx, float_dtype);
        defer w_t.deinit();

        var soft = try x_t.softmax(&ctx, .d, .{});
        defer soft.deinit();
        var soft_ref32 = try x32.softmax(&ctx, .d, .{});
        defer soft_ref32.deinit();
        var soft_ref = try soft_ref32.to(&ctx, float_dtype);
        defer soft_ref.deinit();
        try std.testing.expectEqualSlices(Scalar, soft_ref.asRawTensor().dataConst(), soft.asRawTensor().dataConst());

        var logsoft = try x_t.logSoftmax(&ctx, .d);
        defer logsoft.deinit();
        var logsoft_ref32 = try x32.logSoftmax(&ctx, .d);
        defer logsoft_ref32.deinit();
        var logsoft_ref = try logsoft_ref32.to(&ctx, float_dtype);
        defer logsoft_ref.deinit();
        try std.testing.expectEqualSlices(Scalar, logsoft_ref.asRawTensor().dataConst(), logsoft.asRawTensor().dataConst());

        var rms = try x_t.rmsNorm(&ctx, .d, 1e-6);
        defer rms.deinit();
        var rms_ref32 = try x32.rmsNorm(&ctx, .d, 1e-6);
        defer rms_ref32.deinit();
        var rms_ref = try rms_ref32.to(&ctx, float_dtype);
        defer rms_ref.deinit();
        try std.testing.expectEqualSlices(Scalar, rms_ref.asRawTensor().dataConst(), rms.asRawTensor().dataConst());

        var rms_mul = try x_t.rmsNormMul(&ctx, .d, &w_t, 1e-6);
        defer rms_mul.deinit();
        var rms_mul_ref32 = try x32.rmsNormMul(&ctx, .d, &w32, 1e-6);
        defer rms_mul_ref32.deinit();
        var rms_mul_ref = try rms_mul_ref32.to(&ctx, float_dtype);
        defer rms_mul_ref.deinit();
        try std.testing.expectEqualSlices(Scalar, rms_mul_ref.asRawTensor().dataConst(), rms_mul.asRawTensor().dataConst());

        var ln = try x_t.layerNorm(&ctx, .d, 1e-5, .{});
        defer ln.deinit();
        var ln_ref32 = try x32.layerNorm(&ctx, .d, 1e-5, .{});
        defer ln_ref32.deinit();
        var ln_ref = try ln_ref32.to(&ctx, float_dtype);
        defer ln_ref.deinit();
        try std.testing.expectEqualSlices(Scalar, ln_ref.asRawTensor().dataConst(), ln.asRawTensor().dataConst());
    }
}

test "typed float widened reductions return f32 and scans keep the dtype" {
    @setEvalBranchQuota(1_000_000);
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Exact in f16 AND bf16: widen(narrow(x)) == x, so the widened
    // reductions must equal the f32 reference bit-for-bit.
    var x32 = try Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ 2, 3 }, &.{ 1.0, -2.5, 3.0, 0.5, 4.0, -1.5 });
    defer x32.deinit();

    inline for (.{ DType.f16, DType.bf16 }) |float_dtype| {
        const Scalar = dtype_mod.Scalar(float_dtype);
        var x_t = try x32.to(&ctx, float_dtype);
        defer x_t.deinit();

        var top = try x_t.max(&ctx, .d, .{});
        defer top.deinit();
        comptime std.debug.assert(@TypeOf(top).dtype == .f32);
        try std.testing.expectEqualSlices(f32, &.{ 3.0, 4.0 }, top.asRawTensor().dataConst());

        var bottom = try x_t.min(&ctx, .d, .{});
        defer bottom.deinit();
        comptime std.debug.assert(@TypeOf(bottom).dtype == .f32);
        try std.testing.expectEqualSlices(f32, &.{ -2.5, -1.5 }, bottom.asRawTensor().dataConst());

        var best = try x_t.argmax(&ctx, .d);
        defer best.deinit();
        comptime std.debug.assert(@TypeOf(best).dtype == .i64);
        try std.testing.expectEqualSlices(i64, &.{ 2, 1 }, best.asRawTensor().dataConst());

        var product = try x_t.prod(&ctx, .d);
        defer product.deinit();
        comptime std.debug.assert(@TypeOf(product).dtype == .f32);
        try std.testing.expectEqualSlices(f32, &.{ -7.5, -3.0 }, product.asRawTensor().dataConst());

        var spread = try x_t.variance(&ctx, .d, 0);
        defer spread.deinit();
        comptime std.debug.assert(@TypeOf(spread).dtype == .f32);
        var spread_ref = try x32.variance(&ctx, .d, 0);
        defer spread_ref.deinit();
        try std.testing.expectEqualSlices(f32, spread_ref.asRawTensor().dataConst(), spread.asRawTensor().dataConst());

        var lse = try x_t.logsumexp(&ctx, .d);
        defer lse.deinit();
        comptime std.debug.assert(@TypeOf(lse).dtype == .f32);
        var lse_ref = try x32.logsumexp(&ctx, .d);
        defer lse_ref.deinit();
        try std.testing.expectEqualSlices(f32, lse_ref.asRawTensor().dataConst(), lse.asRawTensor().dataConst());

        var running = try x_t.cumsum(&ctx, .d);
        defer running.deinit();
        comptime std.debug.assert(@TypeOf(running).dtype == float_dtype);
        var running_ref32 = try x32.cumsum(&ctx, .d);
        defer running_ref32.deinit();
        var running_ref = try running_ref32.to(&ctx, float_dtype);
        defer running_ref.deinit();
        try std.testing.expectEqualSlices(Scalar, running_ref.asRawTensor().dataConst(), running.asRawTensor().dataConst());

        var running_prod = try x_t.cumprod(&ctx, .d);
        defer running_prod.deinit();
        comptime std.debug.assert(@TypeOf(running_prod).dtype == float_dtype);
        var running_prod_ref32 = try x32.cumprod(&ctx, .d);
        defer running_prod_ref32.deinit();
        var running_prod_ref = try running_prod_ref32.to(&ctx, float_dtype);
        defer running_prod_ref.deinit();
        try std.testing.expectEqualSlices(Scalar, running_prod_ref.asRawTensor().dataConst(), running_prod.asRawTensor().dataConst());
    }
}

test "typed float einsum matches the narrowed f32 lowering" {
    @setEvalBranchQuota(1_000_000);
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var left32 = try Tensor(.{ .m, .k }).fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer left32.deinit();
    var right32 = try Tensor(.{ .k, .n }).fromSlice(&ctx, .{ 3, 2 }, &.{ 7, 8, 9, 10, 11, 12 });
    defer right32.deinit();

    inline for (.{ DType.f16, DType.bf16 }) |float_dtype| {
        const Scalar = dtype_mod.Scalar(float_dtype);
        var left_t = try left32.to(&ctx, float_dtype);
        defer left_t.deinit();
        var right_t = try right32.to(&ctx, float_dtype);
        defer right_t.deinit();

        var got = try left_t.einsum(&ctx, &right_t, .{ .m, .n });
        defer got.deinit();
        comptime std.debug.assert(@TypeOf(got).dtype == float_dtype);
        var ref32 = try left32.einsum(&ctx, &right32, .{ .m, .n });
        defer ref32.deinit();
        var ref = try ref32.to(&ctx, float_dtype);
        defer ref.deinit();
        try std.testing.expectEqualSlices(Scalar, ref.asRawTensor().dataConst(), got.asRawTensor().dataConst());

        // einsum agrees with the native typed dot on the same contraction
        // (both accumulate in f32 and narrow once).
        var via_dot = try left_t.dot(&ctx, &right_t, .k);
        defer via_dot.deinit();
        try std.testing.expectEqualSlices(Scalar, via_dot.asRawTensor().dataConst(), got.asRawTensor().dataConst());
    }
}

test "typed float structural ops preserve values across dtypes" {
    @setEvalBranchQuota(1_000_000);
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x32 = try Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ 2, 4 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer x32.deinit();

    inline for (.{ DType.f16, DType.bf16, DType.f64 }) |float_dtype| {
        const Scalar = dtype_mod.Scalar(float_dtype);
        var x_t = try x32.to(&ctx, float_dtype);
        defer x_t.deinit();
        var flat_ref = try x_t.flatten(&ctx, .flat);
        defer flat_ref.deinit();

        var halves = try x_t.split(&ctx, .d, .{ .half, .pair }, .{ 2, 2 });
        defer halves.deinit();
        try std.testing.expectEqualSlices(usize, &.{ 2, 2, 2 }, halves.asRawTensor().shape.slice());
        var remerged = try halves.merge(&ctx, .d, .{ .half, .pair });
        defer remerged.deinit();
        var remerged_mat = try remerged.materialize(&ctx);
        defer remerged_mat.deinit();
        try std.testing.expectEqualSlices(Scalar, x_t.asRawTensor().dataConst(), remerged_mat.asRawTensor().dataConst());

        var reshaped = try x_t.reshape(&ctx, .{ .a, .b }, .{ 4, 2 });
        defer reshaped.deinit();
        try std.testing.expectEqualSlices(usize, &.{ 4, 2 }, reshaped.asRawTensor().shape.slice());
        try std.testing.expectEqualSlices(Scalar, flat_ref.asRawTensor().dataConst(), blk: {
            var reflat = try reshaped.flatten(&ctx, .flat);
            defer reflat.deinit();
            break :blk reflat.asRawTensor().dataConst();
        });

        var stepped = try x_t.sliceStep(&ctx, .d, 0, 2, 2);
        defer stepped.deinit();
        var stepped_mat = try stepped.materialize(&ctx);
        defer stepped_mat.deinit();
        var stepped_ref32 = try x32.sliceStep(&ctx, .d, 0, 2, 2);
        defer stepped_ref32.deinit();
        var stepped_ref = try stepped_ref32.to(&ctx, float_dtype);
        defer stepped_ref.deinit();
        try std.testing.expectEqualSlices(Scalar, stepped_ref.asRawTensor().dataConst(), stepped_mat.asRawTensor().dataConst());

        var flipped = try x_t.flip(&ctx, .d);
        defer flipped.deinit();
        var flipped_ref32 = try x32.flip(&ctx, .d);
        defer flipped_ref32.deinit();
        var flipped_ref = try flipped_ref32.to(&ctx, float_dtype);
        defer flipped_ref.deinit();
        try std.testing.expectEqualSlices(Scalar, flipped_ref.asRawTensor().dataConst(), flipped.asRawTensor().dataConst());

        var rolled = try x_t.roll(&ctx, .d, 1);
        defer rolled.deinit();
        var rolled_ref32 = try x32.roll(&ctx, .d, 1);
        defer rolled_ref32.deinit();
        var rolled_ref = try rolled_ref32.to(&ctx, float_dtype);
        defer rolled_ref.deinit();
        try std.testing.expectEqualSlices(Scalar, rolled_ref.asRawTensor().dataConst(), rolled.asRawTensor().dataConst());

        var stacked = try x_t.stack(&ctx, .copy, 0, &.{&x_t});
        defer stacked.deinit();
        try std.testing.expectEqualSlices(usize, &.{ 2, 2, 4 }, stacked.asRawTensor().shape.slice());

        var repeated = try x_t.repeatAxis(&ctx, .batch, 2);
        defer repeated.deinit();
        try std.testing.expectEqualSlices(usize, &.{ 4, 4 }, repeated.asRawTensor().shape.slice());

        // scale/divScalar run the native typed kernel on every float dtype
        // (f64 included). Integer inputs scaled by 3 are exact everywhere.
        var tripled = try x_t.scale(&ctx, 3.0);
        defer tripled.deinit();
        var thirds = try tripled.divScalar(&ctx, 3.0);
        defer thirds.deinit();
        var tripled_ref32 = try x32.scale(&ctx, 3.0);
        defer tripled_ref32.deinit();
        var tripled_ref = try tripled_ref32.to(&ctx, float_dtype);
        defer tripled_ref.deinit();
        try std.testing.expectEqualSlices(Scalar, tripled_ref.asRawTensor().dataConst(), tripled.asRawTensor().dataConst());
        try std.testing.expectEqualSlices(Scalar, x_t.asRawTensor().dataConst(), thirds.asRawTensor().dataConst());
    }

    // pad computes through f32 and is f16/bf16 only.
    inline for (.{ DType.f16, DType.bf16 }) |float_dtype| {
        const Scalar = dtype_mod.Scalar(float_dtype);
        var x_t = try x32.to(&ctx, float_dtype);
        defer x_t.deinit();
        var padded = try x_t.pad(&ctx, .d, 1, 1, 0.5);
        defer padded.deinit();
        var padded_ref32 = try x32.pad(&ctx, .d, 1, 1, 0.5);
        defer padded_ref32.deinit();
        var padded_ref = try padded_ref32.to(&ctx, float_dtype);
        defer padded_ref.deinit();
        try std.testing.expectEqualSlices(Scalar, padded_ref.asRawTensor().dataConst(), padded.asRawTensor().dataConst());
    }
}

test "16-bit variables receive f32 gradients through dot and einsum" {
    @setEvalBranchQuota(1_000_000);
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .batch, .in }).fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();
    var w32 = try Tensor(.{ .out, .in }).fromSlice(&ctx, .{ 2, 2 }, &.{ 0.5, -1.5, 2, 3 });
    defer w32.deinit();

    inline for (.{ DType.f16, DType.bf16 }) |float_dtype| {
        var w_source = try w32.to(&ctx, float_dtype);
        defer w_source.deinit();
        var w = try @TypeOf(w_source).variable(&ctx, try w_source.asRawTensor().cloneView());
        defer w.deinit();
        try std.testing.expect(w.requiresGrad());

        var y = try x.dot(&ctx, &w, .in);
        defer y.deinit();
        try std.testing.expect(y.requiresGrad());
        var loss = try y.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);

        // gy = 1 everywhere, so dW[o, i] = sum_b x[b, i] regardless of the
        // 16-bit forward rounding — exact in f32.
        var wg = (try w.grad(&ctx)).?;
        defer wg.deinit();
        comptime std.debug.assert(@TypeOf(wg).dtype == .f32);
        try std.testing.expectEqualSlices(f32, &.{ 4, 6, 4, 6 }, try wg.dataConst());

        w.zeroGrad();
        try std.testing.expect((try w.grad(&ctx)) == null);

        var y2 = try x.einsum(&ctx, &w, .{ .batch, .out });
        defer y2.deinit();
        var loss2 = try y2.sumAll(&ctx);
        defer loss2.deinit();
        try loss2.backward(&ctx);
        var wg2 = (try w.grad(&ctx)).?;
        defer wg2.deinit();
        try std.testing.expectEqualSlices(f32, &.{ 4, 6, 4, 6 }, try wg2.dataConst());
    }
}

test "f32 master weights train through the differentiable narrow" {
    @setEvalBranchQuota(1_000_000);
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .batch, .in }).fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();

    inline for (.{ DType.f16, DType.bf16 }) |float_dtype| {
        var w32 = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 2, 2 }, &.{ 0.5, -1.5, 2, 3 });
        defer w32.deinit();

        var h = try w32.to(&ctx, float_dtype);
        defer h.deinit();
        try std.testing.expect(h.requiresGrad());

        var y = try x.dot(&ctx, &h, .in);
        defer y.deinit();
        var loss = try y.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);

        // The cast backward is the identity on the f32 gradient.
        var wg = (try w32.grad(&ctx)).?;
        defer wg.deinit();
        try std.testing.expectEqualSlices(f32, &.{ 4, 6, 4, 6 }, try wg.dataConst());
    }
}

test "grad-carrying narrow is scope-owned inside an exec scope" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .batch, .in }).fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();
    var w32 = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 2, 2 }, &.{ 0.5, -1.5, 2, 3 });
    defer w32.deinit();

    const mark = ctx.openExecScope();
    var h = try w32.to(&ctx, .bf16);
    defer h.deinit(); // borrow: no-op, the scope owns value + node
    try std.testing.expect(h.scope_owned);
    var y = try x.dot(&ctx, &h, .in);
    defer y.deinit();
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    ctx.closeExecScope(mark);

    var wg = (try w32.grad(&ctx)).?;
    defer wg.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 4, 6, 4, 6 }, try wg.dataConst());
}

test "typed forward ops reject grad-requiring operands" {
    @setEvalBranchQuota(1_000_000);
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const W = Tensor(.{ .dtype = .bf16, .tags = .{ .batch, .d } });
    var w = try W.variableFromSlice(&ctx, .{ 2, 2 }, &.{
        dtype_mod.f32ToBf16(1),
        dtype_mod.f32ToBf16(2),
        dtype_mod.f32ToBf16(3),
        dtype_mod.f32ToBf16(4),
    });
    defer w.deinit();

    try std.testing.expectError(error.UnsupportedGradient, w.gelu(&ctx));
    try std.testing.expectError(error.UnsupportedGradient, w.add(&ctx, &w));
    try std.testing.expectError(error.UnsupportedGradient, w.flatten(&ctx, .flat));
    try std.testing.expectError(error.UnsupportedGradient, w.sum(&ctx, .d, .{}));

    // The detached view is a constant again: the whole forward set works.
    var frozen = try w.detach(&ctx);
    defer frozen.deinit();
    try std.testing.expect(!frozen.requiresGrad());
    var activated = try frozen.gelu(&ctx);
    defer activated.deinit();
}
