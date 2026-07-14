//! Tests for the `fucina` public facade: the inline-autograd `Tensor` surface
//! (no-grad vs. gradient tensors, mutable-data guards) exposed by `fucina.zig`.

const std = @import("std");

const fucina = @import("fucina.zig");

const ExecContext = fucina.ExecContext;
const Tensor = fucina.Tensor;

test "public facade exports custom VJP" {
    try std.testing.expect(@hasDecl(fucina, "customVjp"));
}

test "public facade exports gradcheck" {
    try std.testing.expect(@hasDecl(fucina, "gradcheck"));
    try std.testing.expect(@hasDecl(fucina, "GradcheckOptions"));
    try std.testing.expect(@hasDecl(fucina, "GradcheckResult"));
}

test "public facade exports grad-control" {
    try std.testing.expect(@hasDecl(fucina, "noGrad"));
    try std.testing.expect(@hasDecl(fucina, "isGradEnabled"));
    try std.testing.expect(@hasDecl(fucina, "NoGradScope"));
}

test "public facade exports ParamRegistry" {
    try std.testing.expect(@hasDecl(fucina, "ParamRegistry"));
    try std.testing.expect(!@hasDecl(fucina, "Module"));
}

test "public facade exports neutral state_dict module" {
    try std.testing.expect(@hasDecl(fucina, "state_dict"));
    try std.testing.expect(@hasDecl(fucina.state_dict, "NamedTensor"));
    try std.testing.expect(@hasDecl(fucina.state_dict, "saveStateDict"));
}

test "public facade exports safetensors codec" {
    try std.testing.expect(@hasDecl(fucina, "safetensors"));
    try std.testing.expect(@hasDecl(fucina.safetensors, "File"));
    try std.testing.expect(@hasDecl(fucina.safetensors, "serialize"));
}

test "public facade exports training checkpoint helper" {
    try std.testing.expect(@hasDecl(fucina, "training_checkpoint"));
    try std.testing.expect(@hasDecl(fucina.training_checkpoint, "TrainerState"));
    try std.testing.expect(@hasDecl(fucina.training_checkpoint, "optimizer_state_file"));
}

test "public Tensor is the inline autograd facade" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(2).fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();
    try std.testing.expect(!x.requiresGrad());
    try std.testing.expect(@TypeOf(x).axis_tags[0] == ._0);
    try std.testing.expect(@TypeOf(x).axis_tags[1] == ._1);

    var y = try x.sum(&ctx, ._1);
    defer y.deinit();
    try std.testing.expect(!y.requiresGrad());
    try std.testing.expectEqualSlices(usize, &.{2}, y.asRawTensor().shape.slice());
    try std.testing.expectEqualSlices(f32, &.{ 6, 15 }, try y.dataConst());
}

test "public Tensor can opt into gradients" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(1).variableFromSlice(&ctx, .{3}, &.{ 1, 2, 3 });
    defer x.deinit();
    try std.testing.expect(x.requiresGrad());

    var y = try x.mul(&ctx, &x);
    defer y.deinit();
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();

    try loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 2, 4, 6 }, try gx.dataConst());

    var gx_view = (try x.gradView(&ctx)).?;
    defer gx_view.deinit();
    var gx_view2 = (try x.gradView(&ctx)).?;
    defer gx_view2.deinit();
    try std.testing.expect(gx_view.asRawTensor().buffer == gx_view2.asRawTensor().buffer);
    try std.testing.expectEqualSlices(f32, &.{ 2, 4, 6 }, try gx_view.dataConst());
}

fn linearCeGcLoss(ctx: *ExecContext, x: *const Tensor(.{ .row, .k }), w: *const Tensor(.{ .class, .k })) !Tensor(.{}) {
    return x.linearCrossEntropyExt(ctx, w, &.{ 2, 9, 0, 5 }, .{ .ignore_index = 9, .label_smoothing = 0.1 });
}

test "public Tensor fused linearCrossEntropyExt matches composed dot + crossEntropyExt" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const rows = 4;
    const in_dim = 6;
    const classes = 9;
    var prng = std.Random.DefaultPrng.init(0x11cef);
    const random = prng.random();
    var x_data: [rows * in_dim]f32 = undefined;
    for (&x_data) |*value| value.* = random.floatNorm(f32);
    var w_data: [classes * in_dim]f32 = undefined;
    for (&w_data) |*value| value.* = random.floatNorm(f32) * 0.5;
    // labels[1] == ignore_index: one masked position.
    const labels = [_]usize{ 2, 9, 0, 5 };
    const options = fucina.CrossEntropyOptions{ .ignore_index = 9, .label_smoothing = 0.1 };

    // Composed reference: dot then crossEntropyExt (two graph nodes).
    var loss_ref: f32 = undefined;
    var gx_ref: [rows * in_dim]f32 = undefined;
    var gw_ref: [classes * in_dim]f32 = undefined;
    {
        var x = try Tensor(.{ .row, .k }).variableFromSlice(&ctx, .{ rows, in_dim }, &x_data);
        defer x.deinit();
        var w = try Tensor(.{ .class, .k }).variableFromSlice(&ctx, .{ classes, in_dim }, &w_data);
        defer w.deinit();
        var logits = try x.dot(&ctx, &w, .k);
        defer logits.deinit();
        var loss = try logits.crossEntropyExt(&ctx, .class, &labels, options);
        defer loss.deinit();
        try loss.backward(&ctx);
        loss_ref = try loss.item();
        var gx = (try x.grad(&ctx)).?;
        defer gx.deinit();
        @memcpy(&gx_ref, try gx.dataConst());
        var gw = (try w.grad(&ctx)).?;
        defer gw.deinit();
        @memcpy(&gw_ref, try gw.dataConst());
    }

    // Fused op: one node, logits internal to the record.
    var x = try Tensor(.{ .row, .k }).variableFromSlice(&ctx, .{ rows, in_dim }, &x_data);
    defer x.deinit();
    var w = try Tensor(.{ .class, .k }).variableFromSlice(&ctx, .{ classes, in_dim }, &w_data);
    defer w.deinit();
    var loss = try x.linearCrossEntropyExt(&ctx, &w, &labels, options);
    defer loss.deinit();
    try loss.backward(&ctx);
    try std.testing.expectApproxEqAbs(loss_ref, try loss.item(), 1e-6);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    for (try gx.dataConst(), gx_ref) |got, want| {
        try std.testing.expect(@abs(got - want) <= 1e-6 + 1e-4 * @abs(want));
    }
    var gw = (try w.grad(&ctx)).?;
    defer gw.deinit();
    for (try gw.dataConst(), gw_ref) |got, want| {
        try std.testing.expect(@abs(got - want) <= 1e-6 + 1e-4 * @abs(want));
    }

    // Finite-difference check of the fused VJP end to end (both operands).
    const result = try fucina.gradcheck(&ctx, linearCeGcLoss, .{ &x, &w }, .{});
    try std.testing.expect(result.checked == rows * in_dim + classes * in_dim);
}

fn linearDistillGcLoss(ctx: *ExecContext, x: *const Tensor(.{ .row, .k }), w: *const Tensor(.{ .class, .k })) !Tensor(.{}) {
    return x.linearDistillExt(ctx, w, &.{ 2, 2, 0, 3 }, &.{ 1, 4, 0, 7 }, &.{ 0.6, 0.3, 0.9, 0.5 }, .{ .loss_scale = 0.7 });
}

test "public Tensor fused linearDistillExt matches the composed sparse-gather route" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const rows = 5;
    const in_dim = 6;
    const classes = 9;
    var prng = std.Random.DefaultPrng.init(0xd157111);
    const random = prng.random();
    var x_data: [rows * in_dim]f32 = undefined;
    for (&x_data) |*value| value.* = random.floatNorm(f32);
    var w_data: [classes * in_dim]f32 = undefined;
    for (&w_data) |*value| value.* = random.floatNorm(f32) * 0.5;
    // Rows 1 and 4 are unsupervised; row 2 carries two entries.
    const t_rows = [_]usize{ 2, 2, 0, 3 };
    const t_classes = [_]usize{ 1, 4, 0, 7 };
    const t_probs = [_]f32{ 0.6, 0.3, 0.9, 0.5 };
    const scale: f32 = 0.7;

    // Composed reference: full logits -> logSoftmax -> flat gather ->
    // prob-weighted mean (the pre-fusion route).
    var loss_ref: f32 = undefined;
    var gx_ref: [rows * in_dim]f32 = undefined;
    var gw_ref: [classes * in_dim]f32 = undefined;
    {
        var x = try Tensor(.{ .row, .k }).variableFromSlice(&ctx, .{ rows, in_dim }, &x_data);
        defer x.deinit();
        var w = try Tensor(.{ .class, .k }).variableFromSlice(&ctx, .{ classes, in_dim }, &w_data);
        defer w.deinit();
        var logits = try x.dot(&ctx, &w, .k);
        defer logits.deinit();
        var logq = try logits.logSoftmax(&ctx, .class);
        defer logq.deinit();
        var flat = try logq.flatten(&ctx, .flat);
        defer flat.deinit();
        var flat_indices: [t_rows.len]usize = undefined;
        var neg_weights: [t_rows.len]f32 = undefined;
        for (&flat_indices, &neg_weights, t_rows, t_classes, t_probs) |*idx, *nw, r, c, p| {
            idx.* = r * classes + c;
            nw.* = -p;
        }
        var picked = try flat.gather(&ctx, .flat, &flat_indices, .entry);
        defer picked.deinit();
        var weights = try Tensor(.{.entry}).fromSlice(&ctx, .{t_rows.len}, &neg_weights);
        defer weights.deinit();
        var weighted = try picked.mul(&ctx, &weights);
        defer weighted.deinit();
        var reduced = try weighted.mean(&ctx, .entry);
        defer reduced.deinit();
        var loss = try reduced.scale(&ctx, scale);
        defer loss.deinit();
        try loss.backward(&ctx);
        loss_ref = try loss.item();
        var gx = (try x.grad(&ctx)).?;
        defer gx.deinit();
        @memcpy(&gx_ref, try gx.dataConst());
        var gw = (try w.grad(&ctx)).?;
        defer gw.deinit();
        @memcpy(&gw_ref, try gw.dataConst());
    }

    // Fused op: one node, selected-row logits internal to the record.
    var x = try Tensor(.{ .row, .k }).variableFromSlice(&ctx, .{ rows, in_dim }, &x_data);
    defer x.deinit();
    var w = try Tensor(.{ .class, .k }).variableFromSlice(&ctx, .{ classes, in_dim }, &w_data);
    defer w.deinit();
    var loss = try x.linearDistillExt(&ctx, &w, &t_rows, &t_classes, &t_probs, .{ .loss_scale = scale });
    defer loss.deinit();
    try loss.backward(&ctx);
    try std.testing.expectApproxEqAbs(loss_ref, try loss.item(), 1e-6);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    for (try gx.dataConst(), gx_ref) |got, want| {
        try std.testing.expect(@abs(got - want) <= 1e-6 + 1e-4 * @abs(want));
    }
    var gw = (try w.grad(&ctx)).?;
    defer gw.deinit();
    for (try gw.dataConst(), gw_ref) |got, want| {
        try std.testing.expect(@abs(got - want) <= 1e-6 + 1e-4 * @abs(want));
    }

    // Finite-difference check of the fused VJP end to end (both operands).
    const result = try fucina.gradcheck(&ctx, linearDistillGcLoss, .{ &x, &w }, .{});
    try std.testing.expect(result.checked == rows * in_dim + classes * in_dim);
}

test "public Tensor mutable data rejects gradient tensors" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var constant = try Tensor(1).fromSlice(&ctx, .{3}, &.{ 1, 2, 3 });
    defer constant.deinit();
    const data = try constant.data();
    data[0] = 4;
    try std.testing.expectEqualSlices(f32, &.{ 4, 2, 3 }, try constant.dataConst());

    var variable = try Tensor(1).variableFromSlice(&ctx, .{3}, &.{ 1, 2, 3 });
    defer variable.deinit();
    try std.testing.expectError(error.MutableDataRequiresNoGrad, variable.data());
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3 }, try variable.dataConst());
}

test "fucina.RawTensor is not re-exported at the public root (Tensor-first guard)" {
    // The raw tensor type is internal/bench-only. Reintroducing `pub const RawTensor`
    // at the module root would also trip the comptime guard in fucina.zig; this test
    // documents the invariant + catches the guard being removed. `internal.RawTensor`
    // and `bench_raw.RawTensor` are the sanctioned raw names and are unaffected.
    try std.testing.expect(!@hasDecl(fucina, "RawTensor"));
    try std.testing.expect(@hasDecl(fucina.internal, "RawTensor")); // the sanctioned in-tree name
}

test "README 'What it looks like' snippet stays compilable and correct" {
    // Pins README.md's tagged-tensor example against API drift: tag-typed
    // model struct, contraction by axis name, the defer-deinit forward that
    // is valid in both modes (eager frees under no-grad, no-op borrow
    // releases under an exec scope), and the exec-scope training step. If
    // this test needs changing, change the README snippet with it.
    const Snippet = struct {
        const Model = struct {
            w1: Tensor(.{ .h1, .in }),
            b1: Tensor(.{.h1}),
            w2: Tensor(.{ .class, .h1 }),
            b2: Tensor(.{.class}),
        };

        fn forward(ctx: *ExecContext, m: *const Model, x: *const Tensor(.{ .batch, .in })) !Tensor(.{ .batch, .class }) {
            var z1 = try x.dot(ctx, &m.w1, .in);
            defer z1.deinit();
            var s1 = try z1.add(ctx, &m.b1);
            defer s1.deinit();
            var a1 = try s1.tanh(ctx);
            defer a1.deinit();
            var z2 = try a1.dot(ctx, &m.w2, .h1);
            defer z2.deinit();
            return try z2.add(ctx, &m.b2);
        }
    };

    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var w1_buf = [_]f32{ 0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8 };
    var b1_buf = [_]f32{ 0.01, 0.02, 0.03, 0.04 };
    var w2_buf = [_]f32{ 0.2, -0.1, 0.05, 0.15, -0.25, 0.35, 0.1, -0.3 };
    var b2_buf = [_]f32{ 0.0, 0.1 };
    var x_buf = [_]f32{ 0.5, -1.0, 1.5, 0.25, -0.75, 2.0 };
    const labels = [_]usize{ 0, 1, 0 };

    var model: Snippet.Model = .{
        .w1 = try Tensor(.{ .h1, .in }).variableFromSlice(&ctx, .{ 4, 2 }, &w1_buf),
        .b1 = try Tensor(.{.h1}).variableFromSlice(&ctx, .{4}, &b1_buf),
        .w2 = try Tensor(.{ .class, .h1 }).variableFromSlice(&ctx, .{ 2, 4 }, &w2_buf),
        .b2 = try Tensor(.{.class}).variableFromSlice(&ctx, .{2}, &b2_buf),
    };
    defer model.w1.deinit();
    defer model.b1.deinit();
    defer model.w2.deinit();
    defer model.b2.deinit();

    var x = try Tensor(.{ .batch, .in }).fromSlice(&ctx, .{ 3, 2 }, &x_buf);
    defer x.deinit();

    // Inference: under no-grad, the forward's defers free each intermediate
    // as soon as it is consumed and the caller owns the result.
    {
        var ng = fucina.noGrad();
        defer ng.close();
        var logits = try Snippet.forward(&ctx, &model, &x);
        defer logits.deinit();
        try std.testing.expect(!logits.requiresGrad());
        for (try logits.dataConst()) |v| try std.testing.expect(std.math.isFinite(v));
    }

    // Training: the SAME forward inside an exec scope; the scope adopts
    // every intermediate and backward() walks the whole step's graph.
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        var logits = try Snippet.forward(&ctx, &model, &x);
        var loss = try logits.crossEntropy(&ctx, .class, &labels);
        try loss.backward(&ctx);
        try std.testing.expect(std.math.isFinite(try loss.item()));
        logits.deinit(); // no-op borrow-release: the scope owns it
        loss.deinit();
    }
}

test "GPU hooks are internal, not public Tensor API" {
    // GPU residency and tracing are implementation details for in-tree loaders
    // and benchmark harnesses. The public root must not grow a device/location
    // surface beside the eager Tensor facade.
    try std.testing.expect(!@hasDecl(fucina, "gpu_enabled"));
    try std.testing.expect(!@hasDecl(fucina, "gpuAllocWeights"));
    try std.testing.expect(!@hasDecl(fucina, "gpuTraceEnabled"));
    try std.testing.expect(!@hasDecl(fucina, "gpuTraceReset"));
    try std.testing.expect(!@hasDecl(fucina, "gpuTraceDump"));
    try std.testing.expect(@hasDecl(fucina.internal, "gpu"));
}
