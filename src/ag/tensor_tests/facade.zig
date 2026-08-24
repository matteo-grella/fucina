//! Comptime surface tests for the public autograd Tensor facade: op-set
//! exposure per dtype spec, rank/axis wrapper parity with the ctx kernels,
//! and rejection rules for no-grad helpers and non-float specs.

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

const util = @import("util.zig");
const expectCloseSlices = util.expectCloseSlices;
const expectedSoftmaxExtWeighted = util.expectedSoftmaxExtWeighted;

test "tagged autograd exposes Tensor facade operations" {
    const T = Tensor(.{ .batch, .d });
    try std.testing.expect(T.dtype == .f32);
    const expected = .{
        "withTags",
        "fromSlice",
        "variableFromSlice",
        "asRawTensor",
        "detach",
        "requiresGrad",
        "backward",
        "backwardWithGrad",
        "grad",
        "shape",
        "to",
        "materialize",
        "alignTo",
        "permuteTo",
        "transpose",
        "insertAxis",
        "squeeze",
        "split",
        "merge",
        "broadcastTo",
        "add",
        "sub",
        "mul",
        "div",
        "causalDepthwiseConv1d",
        "groupedCausalConv1d",
        "gated",
        "glu",
        "swiglu",
        "splitGated",
        "unary",
        "relu",
        "exp",
        "sqrt",
        "rsqrt",
        "sigmoid",
        "silu",
        "log",
        "neg",
        "abs",
        "sin",
        "cos",
        "tanh",
        "fastTanh",
        "gelu",
        "quickGelu",
        "clamp",
        "clampMin",
        "clampMax",
        "sum",
        "mean",
        "variance",
        "standardizeAxis",
        "sumAll",
        "sumMany",
        "flatten",
        "gather",
        "narrow",
        "concat",
        "setSlice",
        "setRows",
        "argmax",
        "max",
        "min",
        "topK",
        "softmax",
        "rmsNorm",
        "layerNorm",
        "crossEntropy",
        "rope",
        "dot",
        "floor",
        "ceil",
        "round",
        "sign",
        "reciprocal",
        "maximum",
        "minimum",
        "pow",
        "isnan",
        "isinf",
        "isfinite",
        "any",
        "all",
        "prod",
        "cumprod",
        "norm",
        "logsumexp",
        "logSoftmax",
        "reshape",
        "sliceStep",
        "select",
        "slice",
        "indexSelect",
        "diagonal",
        "diag",
        "trace",
        "nonzero",
        "indexAdd",
        "takeAlongAxis",
        "scatterAdd",
        "scatter",
        "arange",
        "linspace",
        "oneHot",
        "rand",
        "randn",
        "bernoulli",
    };
    inline for (expected) |name| {
        if (!@hasDecl(T, name)) @compileError("missing tagged autograd op: " ++ name);
    }
}

test "public Tensor accepts non-f32 dtype specs for token tensors" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const TokenIds = Tensor(.{ .dtype = .u16, .tags = .{ .batch, .seq } });
    try std.testing.expect(TokenIds.dtype == .u16);
    // Integer tensors carry the wrapping forward-math set (see docs/reference/04-tensor-operations.md); float
    // `div` stays absent (integer division is explicit divTrunc/divFloor).
    try std.testing.expect(@hasDecl(TokenIds, "add"));
    try std.testing.expect(!@hasDecl(TokenIds, "div"));

    var ids = try TokenIds.fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer ids.deinit();

    try std.testing.expect(!ids.requiresGrad());
    try std.testing.expectEqual(@as(usize, 2), ids.dim(.batch));
    try std.testing.expectEqual(@as(usize, 3), ids.dim(.seq));
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, ids.asRawTensor().shape.slice());
    try std.testing.expectEqualSlices(u16, &.{ 1, 2, 3, 4, 5, 6 }, ids.asRawTensor().dataConst());
}

test "public non-float Tensor excludes autograd at comptime" {
    inline for (.{ DType.bool, DType.u16, DType.i64 }) |non_f32_dtype| {
        const T = Tensor(.{ .dtype = non_f32_dtype, .tags = .{ .batch, .d } });
        if (@hasDecl(T, "variable")) @compileError("non-float Tensor exposes variable");
        if (@hasDecl(T, "variableFromSlice")) @compileError("non-float Tensor exposes variableFromSlice");
        if (@hasDecl(T, "backward")) @compileError("non-float Tensor exposes backward");
        if (@hasDecl(T, "grad")) @compileError("non-float Tensor exposes grad");
    }
}

test "public 16-bit float Tensor exposes leaf autograd but never backward" {
    // f16/bf16 tensors can be trainable LEAVES (f32 gradients); they are
    // never losses, so the backward entry points stay f32-only. The f64
    // decls exist but are comptime errors on instantiation (f64 training
    // is unsupported — gradients are always f32).
    inline for (.{ DType.bf16, DType.f16, DType.f64 }) |float_dtype| {
        const T = Tensor(.{ .dtype = float_dtype, .tags = .{ .batch, .d } });
        inline for (.{ "variable", "variableFromSlice", "grad", "gradView", "zeroGrad", "detach", "requiresGrad" }) |decl_name| {
            if (!@hasDecl(T, decl_name)) @compileError("typed float Tensor missing " ++ decl_name);
        }
        if (@hasDecl(T, "backward")) @compileError("typed float Tensor exposes backward");
        if (@hasDecl(T, "backwardWithGrad")) @compileError("typed float Tensor exposes backwardWithGrad");
    }
}

test "public integer and bool Tensor excludes float math at comptime" {
    inline for (.{ DType.bool, DType.u16, DType.i64 }) |non_float_dtype| {
        comptime expectNoFloatMath(non_float_dtype);
    }
}

fn expectNoFloatMath(comptime non_float_dtype: DType) void {
    const T = Tensor(.{ .dtype = non_float_dtype, .tags = .{ .batch, .d } });
    // `to`, wrapping add/sub/mul, maximum/minimum, divTrunc/divFloor, and
    // the i64-returning sum/sumAll are integer ops now (see docs/reference/04-tensor-operations.md); `div`
    // stays float-only (integer division is explicit).
    const forbidden = .{
        "div",
        "gated",
        "glu",
        "swiglu",
        "unary",
        "relu",
        "exp",
        "sqrt",
        "rsqrt",
        "sigmoid",
        "silu",
        "log",
        "neg",
        "abs",
        "sin",
        "cos",
        "tanh",
        "gelu",
        "quickGelu",
        "clamp",
        "mean",
        "variance",
        "sumMany",
        "argmax",
        "max",
        "min",
        "topK",
        "softmax",
        "rmsNorm",
        "layerNorm",
        "crossEntropy",
        "rope",
        "dot",
    };

    inline for (forbidden) |decl_name| {
        if (@hasDecl(T, decl_name)) @compileError("non-float Tensor exposes float operation: " ++ decl_name);
    }
}

test "tagged autograd numeric rank uses generated axis tags" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(3).variable(
        &ctx,
        try ctx.fromSlice(
            .f32,
            &.{ 2, 3, 2 },
            &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 },
        ),
    );
    defer x.deinit();

    try std.testing.expectEqual(@as(usize, 3), @TypeOf(x).tag_count);
    try std.testing.expect(@TypeOf(x).axis_tags[0] == ._0);
    try std.testing.expect(@TypeOf(x).axis_tags[1] == ._1);
    try std.testing.expect(@TypeOf(x).axis_tags[2] == ._2);
    try std.testing.expectEqual(@as(usize, 2), @TypeOf(x).axis(._2));

    var reduced = try x.sumMany(&ctx, 2);
    defer reduced.deinit();
    try std.testing.expect(@TypeOf(reduced).axis_tags[0] == ._2);
    try std.testing.expectEqualSlices(usize, &.{2}, reduced.asRawTensor().shape.slice());
    try std.testing.expectEqualSlices(f32, &.{ 36, 42 }, reduced.asRawTensor().dataConst());

    var loss = try reduced.sumAll(&ctx);
    defer loss.deinit();

    try loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }, gx.asRawTensor().dataConst());
}

test "public Tensor rank/axis ops match the ctx *AxisRank kernels" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    // softmax-with-scale over the last axis of [H,Q,K]=[2,2,2]
    var s = try Tensor(.{ .h, .q, .k }).fromSlice(&ctx, .{ 2, 2, 2 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer s.deinit();
    var sm = try s.softmax(&ctx, .k, .{ .scale = 0.5 });
    defer sm.deinit();
    var sm_want = try ctx.softmaxExt(3, s.asRawTensor(), 2, .{ .scale = 0.5 });
    defer sm_want.deinit();
    try std.testing.expectEqualSlices(f32, sm_want.dataConst(), sm.asRawTensor().dataConst());

    // affine LayerNorm over .c of [T,C]=[2,3]
    var ln = try Tensor(.{ .t, .c }).fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer ln.deinit();
    var wln = try Tensor(.{.c}).fromSlice(&ctx, .{3}, &.{ 2, 1, 0.5 });
    defer wln.deinit();
    var bln = try Tensor(.{.c}).fromSlice(&ctx, .{3}, &.{ 0.1, -0.1, 0 });
    defer bln.deinit();
    var lno = try ln.layerNorm(&ctx, .c, 1e-5, .{ .weight = wln, .bias = bln });
    defer lno.deinit();
    var lno_want = try ctx.layerNormAffine(2, ln.asRawTensor(), wln.asRawTensor(), bln.asRawTensor(), 1, 1e-5);
    defer lno_want.deinit();
    try std.testing.expectEqualSlices(f32, lno_want.dataConst(), lno.asRawTensor().dataConst());

    // split-glu over .d of [T, 2C]=[2,4]
    var g = try Tensor(.{ .t, .d }).fromSlice(&ctx, .{ 2, 4 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer g.deinit();
    var gg = try g.splitGated(&ctx, .glu, .d, .d);
    defer gg.deinit();
    var gg_want = try ctx.splitGlu(2, g.asRawTensor(), 1);
    defer gg_want.deinit();
    try std.testing.expectEqualSlices(f32, gg_want.dataConst(), gg.asRawTensor().dataConst());

    // causal depthwise conv1d: input [time=4, channel=2], kernel [channel=2, taps=3]
    var cin = try Tensor(.{ .time, .channel }).fromSlice(&ctx, .{ 4, 2 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer cin.deinit();
    var ker = try Tensor(.{ .channel, .taps }).fromSlice(&ctx, .{ 2, 3 }, &.{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 });
    defer ker.deinit();
    var cc = try cin.causalDepthwiseConv1d(&ctx, .time, .channel, .taps, &ker, 1, null);
    defer cc.deinit();
    var cc_want = try ctx.causalDepthwiseConv1d(2, cin.asRawTensor(), ker.asRawTensor(), 0, 1, 1, null);
    defer cc_want.deinit();
    try std.testing.expectEqualSlices(f32, cc_want.dataConst(), cc.asRawTensor().dataConst());

    // silu / relu (existing facade methods) match the ctx kernels (no-grad path)
    var u = try Tensor(.{.d}).fromSlice(&ctx, .{4}, &.{ -2, -1, 1, 2 });
    defer u.deinit();
    var re = try u.relu(&ctx);
    defer re.deinit();
    var re_want = try ctx.relu(u.asRawTensor());
    defer re_want.deinit();
    try std.testing.expectEqualSlices(f32, re_want.dataConst(), re.asRawTensor().dataConst());
    var si = try u.silu(&ctx);
    defer si.deinit();
    var si_want = try ctx.silu(u.asRawTensor());
    defer si_want.deinit();
    for (si_want.dataConst(), si.asRawTensor().dataConst()) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, 1e-6);
    }
}

test "public Tensor rank/axis ops preserve autograd when a VJP exists" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var s = try Tensor(.{ .batch, .d }).variableFromSlice(&ctx, .{ 1, 3 }, &.{ 1, 2, 3 });
    defer s.deinit();
    var weights = try Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ 1, 3 }, &.{ 0.5, -1, 2 });
    defer weights.deinit();
    var sm = try s.softmax(&ctx, .d, .{ .scale = 0.5 });
    defer sm.deinit();
    var weighted = try sm.mul(&ctx, &weights);
    defer weighted.deinit();
    var softmax_loss = try weighted.sumAll(&ctx);
    defer softmax_loss.deinit();
    try softmax_loss.backward(&ctx);

    var expected_probs: [3]f32 = undefined;
    var expected_grad: [3]f32 = undefined;
    expectedSoftmaxExtWeighted(.{ 1, 2, 3 }, .{ 0, 0, 0 }, .{ 0.5, -1, 2 }, 0.5, 1, null, expected_probs[0..], expected_grad[0..]);
    var gs = (try s.grad(&ctx)).?;
    defer gs.deinit();
    try expectCloseSlices(&expected_grad, gs.asRawTensor().dataConst(), 1e-6);

    var ln = try Tensor(.{ .t, .c }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer ln.deinit();
    var wln = try Tensor(.{.c}).variableFromSlice(&ctx, .{3}, &.{ 2, 1, 0.5 });
    defer wln.deinit();
    var bln = try Tensor(.{.c}).variableFromSlice(&ctx, .{3}, &.{ 0.1, -0.1, 0 });
    defer bln.deinit();
    var lno = try ln.layerNorm(&ctx, .c, 1e-5, .{ .weight = &wln, .bias = &bln });
    defer lno.deinit();
    var ln_loss = try lno.sumAll(&ctx);
    defer ln_loss.deinit();
    try ln_loss.backward(&ctx);

    var gln = (try ln.grad(&ctx)).?;
    defer gln.deinit();
    var gwln = (try wln.grad(&ctx)).?;
    defer gwln.deinit();
    var gbln = (try bln.grad(&ctx)).?;
    defer gbln.deinit();
    try std.testing.expectEqual(@as(usize, 6), gln.asRawTensor().dataConst().len);
    try std.testing.expectEqual(@as(usize, 3), gwln.asRawTensor().dataConst().len);
    try std.testing.expectEqual(@as(usize, 3), gbln.asRawTensor().dataConst().len);

    var glu_input = try Tensor(.{ .t, .d }).variableFromSlice(&ctx, .{ 1, 4 }, &.{ 1, 2, 3, 4 });
    defer glu_input.deinit();
    var glu = try glu_input.splitGated(&ctx, .glu, .d, .d);
    defer glu.deinit();
    var glu_loss = try glu.sumAll(&ctx);
    defer glu_loss.deinit();
    try glu_loss.backward(&ctx);

    var gglu = (try glu_input.grad(&ctx)).?;
    defer gglu.deinit();
    try std.testing.expectEqual(@as(usize, 4), gglu.asRawTensor().dataConst().len);

    var conv_input = try Tensor(.{ .time, .channel }).variableFromSlice(&ctx, .{ 4, 2 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer conv_input.deinit();
    var conv_kernel = try Tensor(.{ .channel, .taps }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 });
    defer conv_kernel.deinit();
    var conv = try conv_input.causalDepthwiseConv1d(&ctx, .time, .channel, .taps, &conv_kernel, 1, null);
    defer conv.deinit();
    var conv_loss = try conv.sumAll(&ctx);
    defer conv_loss.deinit();
    try conv_loss.backward(&ctx);

    var gconv_input = (try conv_input.grad(&ctx)).?;
    defer gconv_input.deinit();
    var gconv_kernel = (try conv_kernel.grad(&ctx)).?;
    defer gconv_kernel.deinit();
    try std.testing.expectEqual(@as(usize, 8), gconv_input.asRawTensor().dataConst().len);
    try std.testing.expectEqual(@as(usize, 6), gconv_kernel.asRawTensor().dataConst().len);

    var a = try Tensor(.{ .m, .k }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer a.deinit();
    var b = try Tensor(.{ .k, .n }).variableFromSlice(&ctx, .{ 3, 2 }, &.{ 7, 8, 9, 10, 11, 12 });
    defer b.deinit();
    var mm = try a.matmul(&ctx, &b, .plain, .{ .m, .n });
    defer mm.deinit();
    var mm_loss = try mm.sumAll(&ctx);
    defer mm_loss.deinit();
    try mm_loss.backward(&ctx);
    var ga = (try a.grad(&ctx)).?;
    defer ga.deinit();
    var gb = (try b.grad(&ctx)).?;
    defer gb.deinit();
    try expectCloseSlices(&.{ 15, 19, 23, 15, 19, 23 }, ga.asRawTensor().dataConst(), 1e-6);
    try expectCloseSlices(&.{ 5, 5, 7, 7, 9, 9 }, gb.asRawTensor().dataConst(), 1e-6);

    var at = try Tensor(.{ .m, .k }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer at.deinit();
    var bt = try Tensor(.{ .n, .k }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 7, 9, 11, 8, 10, 12 });
    defer bt.deinit();
    var mmt = try at.matmul(&ctx, &bt, .trans_b, .{ .m, .n });
    defer mmt.deinit();
    var mmt_loss = try mmt.sumAll(&ctx);
    defer mmt_loss.deinit();
    try mmt_loss.backward(&ctx);
    var gat = (try at.grad(&ctx)).?;
    defer gat.deinit();
    var gbt = (try bt.grad(&ctx)).?;
    defer gbt.deinit();
    try expectCloseSlices(&.{ 15, 19, 23, 15, 19, 23 }, gat.asRawTensor().dataConst(), 1e-6);
    try expectCloseSlices(&.{ 5, 7, 9, 5, 7, 9 }, gbt.asRawTensor().dataConst(), 1e-6);

    var bias_x = try Tensor(.{ .batch, .n }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer bias_x.deinit();
    var biased = try bias_x.biasAdd(&ctx, &.{ 10, 20, 30 }, .n);
    defer biased.deinit();
    var bias_loss = try biased.sumAll(&ctx);
    defer bias_loss.deinit();
    try bias_loss.backward(&ctx);
    var gbias_x = (try bias_x.grad(&ctx)).?;
    defer gbias_x.deinit();
    try expectCloseSlices(&.{ 1, 1, 1, 1, 1, 1 }, gbias_x.asRawTensor().dataConst(), 1e-6);

    var zs_x = try Tensor(.{ .batch, .n }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer zs_x.deinit();
    var zs = try zs_x.zeroSlice(&ctx, .n, 1, 1);
    defer zs.deinit();
    var zs_loss = try zs.sumAll(&ctx);
    defer zs_loss.deinit();
    try zs_loss.backward(&ctx);
    var gzs = (try zs_x.grad(&ctx)).?;
    defer gzs.deinit();
    try expectCloseSlices(&.{ 1, 0, 1, 1, 0, 1 }, gzs.asRawTensor().dataConst(), 1e-6);

    var zr_x = try Tensor(.{ .batch, .n }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer zr_x.deinit();
    var zr = try zr_x.zeroRows(&ctx, .batch, &.{0});
    defer zr.deinit();
    var zr_loss = try zr.sumAll(&ctx);
    defer zr_loss.deinit();
    try zr_loss.backward(&ctx);
    var gzr = (try zr_x.grad(&ctx)).?;
    defer gzr.deinit();
    try expectCloseSlices(&.{ 0, 0, 0, 1, 1, 1 }, gzr.asRawTensor().dataConst(), 1e-6);
}

test "public Tensor explicit no-grad helpers reject trainable inputs" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var bias_x = try Tensor(.{ .batch, .n }).variableFromSlice(&ctx, .{ 1, 3 }, &.{ 1, 2, 3 });
    defer bias_x.deinit();
    try std.testing.expectError(error.UnsupportedGradient, bias_x.addAxisVectorInPlace(&ctx, &.{ 1, 2, 3 }, .n));
    try std.testing.expectError(error.UnsupportedGradient, bias_x.addAxisVectorUnaryInPlace(&ctx, .relu, &.{ 1, 2, 3 }, .n));

    var add_dst = try Tensor(.{.d}).fromSlice(&ctx, .{3}, &.{ 1, 2, 3 });
    defer add_dst.deinit();
    var add_src = try Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ 4, 5, 6 });
    defer add_src.deinit();
    try std.testing.expectError(error.UnsupportedGradient, add_dst.addScaledInPlace(&ctx, &add_src, 0.5));

    // conv2d is now differentiable (Conv2dBackward) — its VJP is covered by the
    // gradcheck in ag/gradcheck_tests.zig, so it is no longer a no-grad-only op.
}
