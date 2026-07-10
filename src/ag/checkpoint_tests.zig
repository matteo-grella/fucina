//! Behavioral tests for activation checkpointing (`ag/checkpoint.zig`).
//! Recompute runs the identical ops on the identical input views, so gradient
//! parity with the plain (non-checkpointed) backward is asserted BITWISE.
const std = @import("std");
const exec_mod = @import("../exec.zig");
const optim = @import("../optim.zig");
const ag_tensor = @import("tensor.zig");
const checkpoint_mod = @import("checkpoint.zig");

const Allocator = std.mem.Allocator;
const ExecContext = exec_mod.ExecContext;
const Tensor = ag_tensor.Tensor;
const checkpoint = checkpoint_mod.checkpoint;
const checkpointWithContext = checkpoint_mod.checkpointWithContext;

const batch = 4;
const n_in = 2;
const n_h1 = 8;
const n_h2 = 8;
const n_class = 2;

fn fillUniform(random: std.Random, buf: []f32, bound: f32) void {
    for (buf) |*value| value.* = (random.float(f32) * 2 - 1) * bound;
}

/// MLP layers written in the defer-deinit forward idiom: checkpoint always
/// runs blocks under an exec scope, so the defers are safe no-ops there, and
/// the same functions double as the plain forward inside scoped tests.
const Blocks = struct {
    fn layer1(ctx: *ExecContext, x: *const Tensor(.{ .batch, .in }), w: *const Tensor(.{ .h1, .in }), b: *const Tensor(.{.h1})) !Tensor(.{ .batch, .h1 }) {
        var z = try x.dot(ctx, w, .in);
        defer z.deinit();
        var s = try z.add(ctx, b);
        defer s.deinit();
        return s.tanh(ctx);
    }

    fn layer2(ctx: *ExecContext, x: *const Tensor(.{ .batch, .h1 }), w: *const Tensor(.{ .h2, .h1 }), b: *const Tensor(.{.h2})) !Tensor(.{ .batch, .h2 }) {
        var z = try x.dot(ctx, w, .h1);
        defer z.deinit();
        var s = try z.add(ctx, b);
        defer s.deinit();
        return s.tanh(ctx);
    }

    fn layer3(ctx: *ExecContext, x: *const Tensor(.{ .batch, .h2 }), w: *const Tensor(.{ .class, .h2 }), b: *const Tensor(.{.class})) !Tensor(.{ .batch, .class }) {
        var z = try x.dot(ctx, w, .h2);
        defer z.deinit();
        return z.add(ctx, b);
    }

    /// Square chain step with three intermediates (dot, tanh, withTags): the
    /// unit of the memory-win measurement.
    fn square(ctx: *ExecContext, x: *const Tensor(.{ .batch, .din }), w: *const Tensor(.{ .dout, .din })) !Tensor(.{ .batch, .din }) {
        var z = try x.dot(ctx, w, .din);
        defer z.deinit();
        var a = try z.tanh(ctx);
        defer a.deinit();
        return a.withTags(ctx, .{ .batch, .din });
    }

    /// Head that reduces to a scalar ({1}-shaped) INSIDE the block: the
    /// checkpointed-loss-tail pattern, where the recompute's incoming
    /// gradient seed is itself a single element.
    fn scalarHead(ctx: *ExecContext, x: *const Tensor(.{ .batch, .in }), w: *const Tensor(.{ .h1, .in }), b: *const Tensor(.{.h1})) !Tensor(.{}) {
        var z = try x.dot(ctx, w, .in);
        defer z.deinit();
        var s = try z.add(ctx, b);
        defer s.deinit();
        var a = try s.tanh(ctx);
        defer a.deinit();
        return a.sumAll(ctx);
    }

    /// Single-element rank-2 output ({1,1} when batch == h1 == 1): still a
    /// one-element tensor to the runtime, unlike the {1}-shaped head above.
    fn onePointLayer(ctx: *ExecContext, x: *const Tensor(.{ .batch, .in }), w: *const Tensor(.{ .h1, .in })) !Tensor(.{ .batch, .h1 }) {
        var z = try x.dot(ctx, w, .in);
        defer z.deinit();
        return z.tanh(ctx);
    }

    const dropout_seed: u64 = 0xd5eed;

    /// Layer-1 variant with dropout inside the block. The mask is regenerated
    /// from the fixed seed (never stored), so the block is a deterministic
    /// pure function of its inputs — exactly the checkpoint recompute
    /// contract — by construction.
    fn dropoutLayer1(ctx: *ExecContext, x: *const Tensor(.{ .batch, .in }), w: *const Tensor(.{ .h1, .in }), b: *const Tensor(.{.h1})) !Tensor(.{ .batch, .h1 }) {
        var z = try x.dot(ctx, w, .in);
        defer z.deinit();
        var s = try z.add(ctx, b);
        defer s.deinit();
        var d = try s.dropout(ctx, 0.5, dropout_seed);
        defer d.deinit();
        return d.tanh(ctx);
    }
};

/// Frozen f16 weight facade: a constant by construction (no grad_state field
/// exists on the type), exactly the kind of tensor that cannot travel through
/// the `inputs` tuple of plain `checkpoint`.
const FrozenW = Tensor(.{ .dtype = .f16, .tags = .{ .h1, .in } });

/// Blocks that take their frozen state through `checkpointWithContext`'s
/// `extra` argument; only the differentiable f32 activations arrive through
/// `inputs`. Same defer-deinit forward idiom as `Blocks`.
const ContextBlocks = struct {
    const FrozenLayer = struct {
        w: *const FrozenW,
        scale: f32,
    };

    /// y = tanh((x · W_frozen) * scale): the f16 weight and the config scalar
    /// both ride in `extra`; only `x` is a differentiable input.
    fn frozenLayer(ctx: *ExecContext, extra: *const FrozenLayer, x: *const Tensor(.{ .batch, .in })) !Tensor(.{ .batch, .h1 }) {
        var z = try x.dot(ctx, extra.w, .in);
        defer z.deinit();
        var s = try z.scale(ctx, extra.scale);
        defer s.deinit();
        return s.tanh(ctx);
    }

    const BiasedLayer = struct {
        w: *const FrozenW,
        bias: []const f32,
        scale: f32,
    };

    /// Like `frozenLayer` plus a bias delivered as a raw `[]const f32` slice
    /// (wrapped into a constant inside the block); `extra` travels by value.
    fn biasedLayer(ctx: *ExecContext, extra: BiasedLayer, x: *const Tensor(.{ .batch, .in })) !Tensor(.{ .batch, .h1 }) {
        var b = try Tensor(.{.h1}).fromSlice(ctx, .{n_h1}, extra.bias);
        defer b.deinit();
        var z = try x.dot(ctx, extra.w, .in);
        defer z.deinit();
        var s = try z.add(ctx, &b);
        defer s.deinit();
        var sc = try s.scale(ctx, extra.scale);
        defer sc.deinit();
        return sc.tanh(ctx);
    }

    const SquareChain = struct {
        w: *const Tensor(.{ .dout, .din }),
    };

    /// `Blocks.square` with the (frozen, constant-f32) weight in `extra`: the
    /// unit of the context-block memory-win measurement.
    fn square(ctx: *ExecContext, extra: *const SquareChain, x: *const Tensor(.{ .batch, .din })) !Tensor(.{ .batch, .din }) {
        var z = try x.dot(ctx, extra.w, .din);
        defer z.deinit();
        var a = try z.tanh(ctx);
        defer a.deinit();
        return a.withTags(ctx, .{ .batch, .din });
    }

    const Dropout = struct {
        p: f32,
        seed: u64,
    };

    /// Dropout whose probability AND seed arrive through `extra`: the
    /// recompute must regenerate the identical mask from the stored value —
    /// the determinism-in-(extra, inputs) contract.
    fn dropoutLayer(ctx: *ExecContext, extra: Dropout, x: *const Tensor(.{ .batch, .in }), w: *const Tensor(.{ .h1, .in }), b: *const Tensor(.{.h1})) !Tensor(.{ .batch, .h1 }) {
        var z = try x.dot(ctx, w, .in);
        defer z.deinit();
        var s = try z.add(ctx, b);
        defer s.deinit();
        var d = try s.dropout(ctx, extra.p, extra.seed);
        defer d.deinit();
        return d.tanh(ctx);
    }
};

const Model = struct {
    w1: Tensor(.{ .h1, .in }),
    b1: Tensor(.{.h1}),
    w2: Tensor(.{ .h2, .h1 }),
    b2: Tensor(.{.h2}),
    w3: Tensor(.{ .class, .h2 }),
    b3: Tensor(.{.class}),

    fn initRandom(ctx: *ExecContext, seed: u64) !Model {
        var prng = std.Random.DefaultPrng.init(seed);
        const random = prng.random();
        var w1_buf: [n_h1 * n_in]f32 = undefined;
        var b1_buf: [n_h1]f32 = undefined;
        var w2_buf: [n_h2 * n_h1]f32 = undefined;
        var b2_buf: [n_h2]f32 = undefined;
        var w3_buf: [n_class * n_h2]f32 = undefined;
        var b3_buf: [n_class]f32 = undefined;
        fillUniform(random, &w1_buf, 1.0 / @sqrt(@as(f32, n_in)));
        fillUniform(random, &b1_buf, 0.1);
        fillUniform(random, &w2_buf, 1.0 / @sqrt(@as(f32, n_h1)));
        fillUniform(random, &b2_buf, 0.1);
        fillUniform(random, &w3_buf, 1.0 / @sqrt(@as(f32, n_h2)));
        fillUniform(random, &b3_buf, 0.1);
        var w1 = try Tensor(.{ .h1, .in }).variableFromSlice(ctx, .{ n_h1, n_in }, &w1_buf);
        errdefer w1.deinit();
        var b1 = try Tensor(.{.h1}).variableFromSlice(ctx, .{n_h1}, &b1_buf);
        errdefer b1.deinit();
        var w2 = try Tensor(.{ .h2, .h1 }).variableFromSlice(ctx, .{ n_h2, n_h1 }, &w2_buf);
        errdefer w2.deinit();
        var b2 = try Tensor(.{.h2}).variableFromSlice(ctx, .{n_h2}, &b2_buf);
        errdefer b2.deinit();
        var w3 = try Tensor(.{ .class, .h2 }).variableFromSlice(ctx, .{ n_class, n_h2 }, &w3_buf);
        errdefer w3.deinit();
        var b3 = try Tensor(.{.class}).variableFromSlice(ctx, .{n_class}, &b3_buf);
        errdefer b3.deinit();
        return .{ .w1 = w1, .b1 = b1, .w2 = w2, .b2 = b2, .w3 = w3, .b3 = b3 };
    }

    fn deinit(self: *Model) void {
        self.w1.deinit();
        self.b1.deinit();
        self.w2.deinit();
        self.b2.deinit();
        self.w3.deinit();
        self.b3.deinit();
    }

    fn zeroGrad(self: *const Model) void {
        self.w1.zeroGrad();
        self.b1.zeroGrad();
        self.w2.zeroGrad();
        self.b2.zeroGrad();
        self.w3.zeroGrad();
        self.b3.zeroGrad();
    }
};

fn makeInput(ctx: *ExecContext, seed: u64) !Tensor(.{ .batch, .in }) {
    var prng = std.Random.DefaultPrng.init(seed);
    var buf: [batch * n_in]f32 = undefined;
    fillUniform(prng.random(), &buf, 1.0);
    return Tensor(.{ .batch, .in }).variableFromSlice(ctx, .{ batch, n_in }, &buf);
}

/// Deep copy of `t`'s accumulated gradient (caller frees).
fn gradData(ctx: *ExecContext, t: anytype) ![]f32 {
    var g = (try t.grad(ctx)).?;
    defer g.deinit();
    return ctx.allocator.dupe(f32, try g.dataConst());
}

fn snapshotGrads(ctx: *ExecContext, model: *const Model, x: anytype) ![7][]f32 {
    var out: [7][]f32 = undefined;
    var filled: usize = 0;
    errdefer for (out[0..filled]) |s| ctx.allocator.free(s);
    out[0] = try gradData(ctx, x);
    filled = 1;
    out[1] = try gradData(ctx, &model.w1);
    filled = 2;
    out[2] = try gradData(ctx, &model.b1);
    filled = 3;
    out[3] = try gradData(ctx, &model.w2);
    filled = 4;
    out[4] = try gradData(ctx, &model.b2);
    filled = 5;
    out[5] = try gradData(ctx, &model.w3);
    filled = 6;
    out[6] = try gradData(ctx, &model.b3);
    filled = 7;
    return out;
}

/// Like `snapshotGrads` but only the layer-1/2 parameters (+ input): for
/// graphs where the third layer is unused and has no gradient.
fn snapshotGradsTwoLayers(ctx: *ExecContext, model: *const Model, x: anytype) ![5][]f32 {
    var out: [5][]f32 = undefined;
    var filled: usize = 0;
    errdefer for (out[0..filled]) |s| ctx.allocator.free(s);
    out[0] = try gradData(ctx, x);
    filled = 1;
    out[1] = try gradData(ctx, &model.w1);
    filled = 2;
    out[2] = try gradData(ctx, &model.b1);
    filled = 3;
    out[3] = try gradData(ctx, &model.w2);
    filled = 4;
    out[4] = try gradData(ctx, &model.b2);
    filled = 5;
    return out;
}

fn freeGrads(allocator: Allocator, grads: []const []f32) void {
    for (grads) |s| allocator.free(s);
}

test "checkpointed middle layer matches plain backward bitwise (unscoped)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try Model.initRandom(&ctx, 42);
    defer model.deinit();
    var x = try makeInput(&ctx, 99);
    defer x.deinit();

    // Plain reference: no scope, so every intermediate stays alive (manual
    // deinits at block exit, after backward) — deinit-ASAP would dangle.
    const plain_snap, const plain_loss = plain: {
        var z1 = try x.dot(&ctx, &model.w1, .in);
        defer z1.deinit();
        var s1 = try z1.add(&ctx, &model.b1);
        defer s1.deinit();
        var a1 = try s1.tanh(&ctx);
        defer a1.deinit();
        var z2 = try a1.dot(&ctx, &model.w2, .h1);
        defer z2.deinit();
        var s2 = try z2.add(&ctx, &model.b2);
        defer s2.deinit();
        var a2 = try s2.tanh(&ctx);
        defer a2.deinit();
        var z3 = try a2.dot(&ctx, &model.w3, .h2);
        defer z3.deinit();
        var logits = try z3.add(&ctx, &model.b3);
        defer logits.deinit();
        var loss = try logits.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);
        const loss_value = try loss.item();
        break :plain .{ try snapshotGrads(&ctx, &model, &x), loss_value };
    };
    defer freeGrads(allocator, &plain_snap);

    model.zeroGrad();
    x.zeroGrad();

    // Same forward with the middle layer checkpointed, still unscoped: the
    // checkpoint result is caller-owned and chains into plain downstream ops.
    const ck_snap, const ck_loss = ck: {
        var z1 = try x.dot(&ctx, &model.w1, .in);
        defer z1.deinit();
        var s1 = try z1.add(&ctx, &model.b1);
        defer s1.deinit();
        var a1 = try s1.tanh(&ctx);
        defer a1.deinit();
        var h2 = try checkpoint(&ctx, Blocks.layer2, .{ &a1, &model.w2, &model.b2 });
        defer h2.deinit();
        var z3 = try h2.dot(&ctx, &model.w3, .h2);
        defer z3.deinit();
        var logits = try z3.add(&ctx, &model.b3);
        defer logits.deinit();
        var loss = try logits.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);
        const loss_value = try loss.item();
        break :ck .{ try snapshotGrads(&ctx, &model, &x), loss_value };
    };
    defer freeGrads(allocator, &ck_snap);

    try std.testing.expectEqual(plain_loss, ck_loss);
    for (plain_snap, ck_snap) |p, c| try std.testing.expectEqualSlices(f32, p, c);
}

test "checkpointed middle layer matches plain backward bitwise (exec scope)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try Model.initRandom(&ctx, 42);
    defer model.deinit();
    var x = try makeInput(&ctx, 99);
    defer x.deinit();

    const plain_snap, const plain_loss = plain: {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const a1 = try Blocks.layer1(&ctx, &x, &model.w1, &model.b1);
        const a2 = try Blocks.layer2(&ctx, &a1, &model.w2, &model.b2);
        const logits = try Blocks.layer3(&ctx, &a2, &model.w3, &model.b3);
        const loss = try logits.sumAll(&ctx);
        try loss.backward(&ctx);
        const loss_value = try loss.item();
        break :plain .{ try snapshotGrads(&ctx, &model, &x), loss_value };
    };
    defer freeGrads(allocator, &plain_snap);

    model.zeroGrad();
    x.zeroGrad();

    const ck_snap, const ck_loss = ck: {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const a1 = try Blocks.layer1(&ctx, &x, &model.w1, &model.b1);
        const h2 = try checkpoint(&ctx, Blocks.layer2, .{ &a1, &model.w2, &model.b2 });
        const logits = try Blocks.layer3(&ctx, &h2, &model.w3, &model.b3);
        const loss = try logits.sumAll(&ctx);
        try loss.backward(&ctx);
        const loss_value = try loss.item();
        break :ck .{ try snapshotGrads(&ctx, &model, &x), loss_value };
    };
    defer freeGrads(allocator, &ck_snap);

    try std.testing.expectEqual(plain_loss, ck_loss);
    for (plain_snap, ck_snap) |p, c| try std.testing.expectEqualSlices(f32, p, c);
}

test "checkpointed scalar-output block matches plain backward bitwise" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(13);
    var w_buf: [n_h1 * n_in]f32 = undefined;
    var b_buf: [n_h1]f32 = undefined;
    fillUniform(prng.random(), &w_buf, 0.7);
    fillUniform(prng.random(), &b_buf, 0.1);

    var x = try makeInput(&ctx, 19);
    defer x.deinit();
    var w = try Tensor(.{ .h1, .in }).variableFromSlice(&ctx, .{ n_h1, n_in }, &w_buf);
    defer w.deinit();
    var b = try Tensor(.{.h1}).variableFromSlice(&ctx, .{n_h1}, &b_buf);
    defer b.deinit();

    // The downstream scale makes the block output's incoming gradient 3, so
    // any deviation from backpropagating it as-is shows up in every grad.
    const plain_snap, const plain_loss = plain: {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const h = try Blocks.scalarHead(&ctx, &x, &w, &b);
        const loss = try h.scale(&ctx, 3);
        try loss.backward(&ctx);
        const loss_value = try loss.item();
        const gx = try gradData(&ctx, &x);
        errdefer allocator.free(gx);
        const gw = try gradData(&ctx, &w);
        errdefer allocator.free(gw);
        const gb = try gradData(&ctx, &b);
        break :plain .{ .{ gx, gw, gb }, loss_value };
    };
    defer inline for (plain_snap) |s| allocator.free(s);

    x.zeroGrad();
    w.zeroGrad();
    b.zeroGrad();

    const ck_snap, const ck_loss = ck: {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const h = try checkpoint(&ctx, Blocks.scalarHead, .{ &x, &w, &b });
        const loss = try h.scale(&ctx, 3);
        try loss.backward(&ctx);
        const loss_value = try loss.item();
        const gx = try gradData(&ctx, &x);
        errdefer allocator.free(gx);
        const gw = try gradData(&ctx, &w);
        errdefer allocator.free(gw);
        const gb = try gradData(&ctx, &b);
        break :ck .{ .{ gx, gw, gb }, loss_value };
    };
    defer inline for (ck_snap) |s| allocator.free(s);

    try std.testing.expectEqual(plain_loss, ck_loss);
    inline for (plain_snap, ck_snap) |p, c| try std.testing.expectEqualSlices(f32, p, c);
}

test "checkpointed single-element rank-2 output matches plain backward bitwise" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(59);
    var w_buf: [n_in]f32 = undefined;
    var x_buf: [n_in]f32 = undefined;
    fillUniform(prng.random(), &w_buf, 0.7);
    fillUniform(prng.random(), &x_buf, 1.0);

    var x = try Tensor(.{ .batch, .in }).variableFromSlice(&ctx, .{ 1, n_in }, &x_buf);
    defer x.deinit();
    var w = try Tensor(.{ .h1, .in }).variableFromSlice(&ctx, .{ 1, n_in }, &w_buf);
    defer w.deinit();

    const plain_snap, const plain_loss = plain: {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const h = try Blocks.onePointLayer(&ctx, &x, &w);
        const loss = try h.sumAll(&ctx);
        try loss.backward(&ctx);
        const loss_value = try loss.item();
        const gx = try gradData(&ctx, &x);
        errdefer allocator.free(gx);
        const gw = try gradData(&ctx, &w);
        break :plain .{ .{ gx, gw }, loss_value };
    };
    defer inline for (plain_snap) |s| allocator.free(s);

    x.zeroGrad();
    w.zeroGrad();

    const ck_snap, const ck_loss = ck: {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const h = try checkpoint(&ctx, Blocks.onePointLayer, .{ &x, &w });
        const loss = try h.sumAll(&ctx);
        try loss.backward(&ctx);
        const loss_value = try loss.item();
        const gx = try gradData(&ctx, &x);
        errdefer allocator.free(gx);
        const gw = try gradData(&ctx, &w);
        break :ck .{ .{ gx, gw }, loss_value };
    };
    defer inline for (ck_snap) |s| allocator.free(s);

    try std.testing.expectEqual(plain_loss, ck_loss);
    inline for (plain_snap, ck_snap) |p, c| try std.testing.expectEqualSlices(f32, p, c);
}

test "checkpoint flows gradients to every input and prunes non-grad inputs" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(7);
    var w_buf: [n_h1 * n_in]f32 = undefined;
    var b_buf: [n_h1]f32 = undefined;
    fillUniform(prng.random(), &w_buf, 0.7);
    fillUniform(prng.random(), &b_buf, 0.1);

    var x = try makeInput(&ctx, 11);
    defer x.deinit();
    var w = try Tensor(.{ .h1, .in }).variableFromSlice(&ctx, .{ n_h1, n_in }, &w_buf);
    defer w.deinit();
    var b = try Tensor(.{.h1}).variableFromSlice(&ctx, .{n_h1}, &b_buf);
    defer b.deinit();

    // All three inputs trainable: grads must match plain bitwise.
    const plain = plain: {
        var z = try x.dot(&ctx, &w, .in);
        defer z.deinit();
        var s = try z.add(&ctx, &b);
        defer s.deinit();
        var a = try s.tanh(&ctx);
        defer a.deinit();
        var loss = try a.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);
        const gx = try gradData(&ctx, &x);
        errdefer allocator.free(gx);
        const gw = try gradData(&ctx, &w);
        errdefer allocator.free(gw);
        const gb = try gradData(&ctx, &b);
        break :plain .{ gx, gw, gb };
    };
    defer inline for (plain) |s| allocator.free(s);

    x.zeroGrad();
    w.zeroGrad();
    b.zeroGrad();

    {
        var h = try checkpoint(&ctx, Blocks.layer1, .{ &x, &w, &b });
        defer h.deinit();
        var loss = try h.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);
    }
    {
        const gx = try gradData(&ctx, &x);
        defer allocator.free(gx);
        const gw = try gradData(&ctx, &w);
        defer allocator.free(gw);
        const gb = try gradData(&ctx, &b);
        defer allocator.free(gb);
        try std.testing.expectEqualSlices(f32, plain[0], gx);
        try std.testing.expectEqualSlices(f32, plain[1], gw);
        try std.testing.expectEqualSlices(f32, plain[2], gb);
    }

    x.zeroGrad();
    w.zeroGrad();
    b.zeroGrad();

    // w as a no-grad constant: the checkpoint node's operand slot is null, the
    // engine passes needs_grad[1] == false, and the recompute rewraps w as a
    // constant — no gradient is produced for it (out[1] stays null).
    var w_const = try Tensor(.{ .h1, .in }).fromSlice(&ctx, .{ n_h1, n_in }, &w_buf);
    defer w_const.deinit();

    const plain_const = plain: {
        var z = try x.dot(&ctx, &w_const, .in);
        defer z.deinit();
        var s = try z.add(&ctx, &b);
        defer s.deinit();
        var a = try s.tanh(&ctx);
        defer a.deinit();
        var loss = try a.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);
        const gx = try gradData(&ctx, &x);
        errdefer allocator.free(gx);
        const gb = try gradData(&ctx, &b);
        break :plain .{ gx, gb };
    };
    defer inline for (plain_const) |s| allocator.free(s);

    x.zeroGrad();
    b.zeroGrad();

    {
        var h = try checkpoint(&ctx, Blocks.layer1, .{ &x, &w_const, &b });
        defer h.deinit();
        var loss = try h.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);
    }
    {
        const gx = try gradData(&ctx, &x);
        defer allocator.free(gx);
        const gb = try gradData(&ctx, &b);
        defer allocator.free(gb);
        try std.testing.expectEqualSlices(f32, plain_const[0], gx);
        try std.testing.expectEqualSlices(f32, plain_const[1], gb);
        try std.testing.expect((try w_const.grad(&ctx)) == null);
    }

    x.zeroGrad();
    b.zeroGrad();

    // All-constant inputs: no node is built, the result is a plain constant —
    // and under a scope it is adopted like any no-grad op result.
    var x_const = try Tensor(.{ .batch, .in }).fromSlice(&ctx, .{ batch, n_in }, &[_]f32{0.5} ** (batch * n_in));
    defer x_const.deinit();
    var b_const = try Tensor(.{.h1}).fromSlice(&ctx, .{n_h1}, &b_buf);
    defer b_const.deinit();
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const base = ctx.rt.scope_entries.items.len;
        const h = try checkpoint(&ctx, Blocks.layer1, .{ &x_const, &w_const, &b_const });
        try std.testing.expect(!h.requiresGrad());
        try std.testing.expect(h.scope_owned);
        try std.testing.expectEqual(base + 1, ctx.rt.scope_entries.items.len);
    }
}

test "chained checkpoints with a multi-consumer output match plain backward" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try Model.initRandom(&ctx, 5);
    defer model.deinit();
    var x = try makeInput(&ctx, 17);
    defer x.deinit();

    // The checkpoint output h2 feeds THREE gradient paths (sumAll + both mul
    // operands): the engine must accumulate them into the checkpoint's
    // GradState before invoking its backward exactly once.
    const plain_snap, const plain_loss = plain: {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const a1 = try Blocks.layer1(&ctx, &x, &model.w1, &model.b1);
        const a2 = try Blocks.layer2(&ctx, &a1, &model.w2, &model.b2);
        const lin = try a2.sumAll(&ctx);
        const sq = try a2.mul(&ctx, &a2);
        const quad = try sq.sumAll(&ctx);
        const loss = try lin.add(&ctx, &quad);
        try loss.backward(&ctx);
        const loss_value = try loss.item();
        break :plain .{ try snapshotGradsTwoLayers(&ctx, &model, &x), loss_value };
    };
    defer freeGrads(allocator, &plain_snap);

    model.zeroGrad();
    x.zeroGrad();

    const ck_snap, const ck_loss = ck: {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const h1 = try checkpoint(&ctx, Blocks.layer1, .{ &x, &model.w1, &model.b1 });
        const h2 = try checkpoint(&ctx, Blocks.layer2, .{ &h1, &model.w2, &model.b2 });
        const lin = try h2.sumAll(&ctx);
        const sq = try h2.mul(&ctx, &h2);
        const quad = try sq.sumAll(&ctx);
        const loss = try lin.add(&ctx, &quad);
        try loss.backward(&ctx);
        const loss_value = try loss.item();
        break :ck .{ try snapshotGradsTwoLayers(&ctx, &model, &x), loss_value };
    };
    defer freeGrads(allocator, &ck_snap);

    try std.testing.expectEqual(plain_loss, ck_loss);
    for (plain_snap, ck_snap) |p, c| try std.testing.expectEqualSlices(f32, p, c);
}

test "checkpointing a deep chain retains materially fewer scope entries" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const d = 8;
    const chain_len = 8;
    var prng = std.Random.DefaultPrng.init(3);
    var w_buf: [d * d]f32 = undefined;
    var x_buf: [batch * d]f32 = undefined;
    fillUniform(prng.random(), &w_buf, 1.0 / @sqrt(@as(f32, d)));
    fillUniform(prng.random(), &x_buf, 1.0);

    var w = try Tensor(.{ .dout, .din }).variableFromSlice(&ctx, .{ d, d }, &w_buf);
    defer w.deinit();
    var x = try Tensor(.{ .batch, .din }).variableFromSlice(&ctx, .{ batch, d }, &x_buf);
    defer x.deinit();

    // Plain: every block intermediate is adopted by the scope and retained
    // until close (3 entries per block: dot, tanh, withTags).
    var plain_entries: usize = 0;
    const plain_gx, const plain_gw, const plain_loss = plain: {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const base = ctx.rt.scope_entries.items.len;
        var h = try Blocks.square(&ctx, &x, &w);
        for (1..chain_len) |_| {
            h = try Blocks.square(&ctx, &h, &w);
        }
        plain_entries = ctx.rt.scope_entries.items.len - base;
        const loss = try h.sumAll(&ctx);
        try loss.backward(&ctx);
        const loss_value = try loss.item();
        const gx = try gradData(&ctx, &x);
        errdefer allocator.free(gx);
        const gw = try gradData(&ctx, &w);
        break :plain .{ gx, gw, loss_value };
    };
    defer allocator.free(plain_gx);
    defer allocator.free(plain_gw);

    x.zeroGrad();
    w.zeroGrad();

    // Checkpointed: only the block OUTPUTS are retained (1 entry per block);
    // the intermediates die inside the checkpoint's inner scope.
    var ck_entries: usize = 0;
    const ck_gx, const ck_gw, const ck_loss = ck: {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const base = ctx.rt.scope_entries.items.len;
        var h = try checkpoint(&ctx, Blocks.square, .{ &x, &w });
        for (1..chain_len) |_| {
            h = try checkpoint(&ctx, Blocks.square, .{ &h, &w });
        }
        ck_entries = ctx.rt.scope_entries.items.len - base;
        const loss = try h.sumAll(&ctx);
        try loss.backward(&ctx);
        const loss_value = try loss.item();
        const gx = try gradData(&ctx, &x);
        errdefer allocator.free(gx);
        const gw = try gradData(&ctx, &w);
        break :ck .{ gx, gw, loss_value };
    };
    defer allocator.free(ck_gx);
    defer allocator.free(ck_gw);

    try std.testing.expectEqual(3 * chain_len, plain_entries);
    try std.testing.expectEqual(chain_len, ck_entries);
    try std.testing.expect(ck_entries * 2 < plain_entries);

    // Same forward, same gradients — bitwise (w accumulates 8 contributions).
    try std.testing.expectEqual(plain_loss, ck_loss);
    try std.testing.expectEqualSlices(f32, plain_gx, ck_gx);
    try std.testing.expectEqualSlices(f32, plain_gw, ck_gw);
}

fn trainStep(
    ctx: *ExecContext,
    model: *const Model,
    x: *const Tensor(.{ .batch, .in }),
    labels: []const usize,
    opt: *optim.SGD,
    comptime use_checkpoint: bool,
) !f32 {
    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    const a1 = if (use_checkpoint)
        try checkpoint(ctx, Blocks.layer1, .{ x, &model.w1, &model.b1 })
    else
        try Blocks.layer1(ctx, x, &model.w1, &model.b1);
    const a2 = if (use_checkpoint)
        try checkpoint(ctx, Blocks.layer2, .{ &a1, &model.w2, &model.b2 })
    else
        try Blocks.layer2(ctx, &a1, &model.w2, &model.b2);
    const logits = try Blocks.layer3(ctx, &a2, &model.w3, &model.b3);
    const loss = try logits.crossEntropy(ctx, .class, labels);
    try loss.backward(ctx);
    try opt.step(ctx);
    opt.zeroGrad();
    return loss.item();
}

test "checkpointed hidden layers train bitwise-identically to plain (SGD)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Tiny separable task: class = sign of the first coordinate.
    const xs = [batch * n_in]f32{ 0.9, 0.2, -0.8, 0.4, 0.6, -0.7, -0.5, -0.3 };
    const labels = [batch]usize{ 1, 0, 1, 0 };
    var x = try Tensor(.{ .batch, .in }).fromSlice(&ctx, .{ batch, n_in }, &xs);
    defer x.deinit();

    const steps = 5;
    var plain_losses: [steps]f32 = undefined;
    var ck_losses: [steps]f32 = undefined;

    var plain_params: ?[][]f32 = null;
    defer if (plain_params) |params| {
        for (params) |s| allocator.free(s);
        allocator.free(params);
    };
    {
        var model = try Model.initRandom(&ctx, 42);
        defer model.deinit();
        var opt = optim.SGD.init(allocator, .{ .lr = 0.2 });
        defer opt.deinit();
        try opt.addParam(&model.w1);
        try opt.addParam(&model.b1);
        try opt.addParam(&model.w2);
        try opt.addParam(&model.b2);
        try opt.addParam(&model.w3);
        try opt.addParam(&model.b3);
        for (&plain_losses) |*loss| {
            loss.* = try trainStep(&ctx, &model, &x, &labels, &opt, false);
        }
        plain_params = try snapshotModelData(allocator, &model);
    }

    var ck_params: ?[][]f32 = null;
    defer if (ck_params) |params| {
        for (params) |s| allocator.free(s);
        allocator.free(params);
    };
    {
        var model = try Model.initRandom(&ctx, 42);
        defer model.deinit();
        var opt = optim.SGD.init(allocator, .{ .lr = 0.2 });
        defer opt.deinit();
        try opt.addParam(&model.w1);
        try opt.addParam(&model.b1);
        try opt.addParam(&model.w2);
        try opt.addParam(&model.b2);
        try opt.addParam(&model.w3);
        try opt.addParam(&model.b3);
        for (&ck_losses) |*loss| {
            loss.* = try trainStep(&ctx, &model, &x, &labels, &opt, true);
        }
        ck_params = try snapshotModelData(allocator, &model);
    }

    for (plain_losses, ck_losses) |p, c| try std.testing.expectEqual(p, c);
    try std.testing.expect(plain_losses[steps - 1] < plain_losses[0]);
    for (plain_params.?, ck_params.?) |p, c| try std.testing.expectEqualSlices(f32, p, c);
}

fn snapshotModelData(allocator: Allocator, model: *const Model) ![][]f32 {
    var out = try allocator.alloc([]f32, 6);
    errdefer allocator.free(out);
    var filled: usize = 0;
    errdefer for (out[0..filled]) |s| allocator.free(s);
    out[0] = try allocator.dupe(f32, try model.w1.dataConst());
    filled = 1;
    out[1] = try allocator.dupe(f32, try model.b1.dataConst());
    filled = 2;
    out[2] = try allocator.dupe(f32, try model.w2.dataConst());
    filled = 3;
    out[3] = try allocator.dupe(f32, try model.b2.dataConst());
    filled = 4;
    out[4] = try allocator.dupe(f32, try model.w3.dataConst());
    filled = 5;
    out[5] = try allocator.dupe(f32, try model.b3.dataConst());
    filled = 6;
    return out;
}

test "recompute error propagates from backward without leaking" {
    const Hostile = struct {
        var fail_recompute: bool = false;

        fn block(ctx: *ExecContext, x: *const Tensor(.{.d})) !Tensor(.{.d}) {
            if (fail_recompute) return error.InducedRecomputeFailure;
            return x.mul(ctx, x);
        }
    };
    Hostile.fail_recompute = false;

    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ 1, 2, 3 });
    defer x.deinit();

    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    const h = try checkpoint(&ctx, Hostile.block, .{&x});
    const loss = try h.sumAll(&ctx);

    Hostile.fail_recompute = true;
    defer Hostile.fail_recompute = false;
    try std.testing.expectError(error.InducedRecomputeFailure, loss.backward(&ctx));
}

test "nested checkpoint recompute is rejected instead of deadlocking" {
    const Nested = struct {
        fn inner(ctx: *ExecContext, x: *const Tensor(.{.d})) !Tensor(.{.d}) {
            return x.mul(ctx, x);
        }

        fn outer(ctx: *ExecContext, x: *const Tensor(.{.d})) !Tensor(.{.d}) {
            return checkpoint(ctx, inner, .{x});
        }
    };

    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ 1, 2, 3 });
    defer x.deinit();

    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    // Forward is fine (no recompute lock is held)...
    const h = try checkpoint(&ctx, Nested.outer, .{&x});
    const loss = try h.sumAll(&ctx);
    // ...but the outer recompute re-creates the inner checkpoint node, whose
    // backward would re-enter the recompute lock on this thread.
    try std.testing.expectError(error.NestedCheckpointRecompute, loss.backward(&ctx));
}

test "checkpointed dropout block matches the un-checkpointed block bitwise" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(23);
    var w_buf: [n_h1 * n_in]f32 = undefined;
    var b_buf: [n_h1]f32 = undefined;
    fillUniform(prng.random(), &w_buf, 0.8);
    fillUniform(prng.random(), &b_buf, 0.1);

    var x = try makeInput(&ctx, 31);
    defer x.deinit();
    var w = try Tensor(.{ .h1, .in }).variableFromSlice(&ctx, .{ n_h1, n_in }, &w_buf);
    defer w.deinit();
    var b = try Tensor(.{.h1}).variableFromSlice(&ctx, .{n_h1}, &b_buf);
    defer b.deinit();

    // Plain reference: dropout in the live graph, mask from the fixed seed.
    var plain_out: []f32 = undefined;
    const plain_snap, const plain_loss = plain: {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const h = try Blocks.dropoutLayer1(&ctx, &x, &w, &b);
        plain_out = try allocator.dupe(f32, try h.dataConst());
        const loss = try h.sumAll(&ctx);
        try loss.backward(&ctx);
        const loss_value = try loss.item();
        const gx = try gradData(&ctx, &x);
        errdefer allocator.free(gx);
        const gw = try gradData(&ctx, &w);
        errdefer allocator.free(gw);
        const gb = try gradData(&ctx, &b);
        break :plain .{ .{ gx, gw, gb }, loss_value };
    };
    defer allocator.free(plain_out);
    defer inline for (plain_snap) |s| allocator.free(s);

    x.zeroGrad();
    w.zeroGrad();
    b.zeroGrad();

    // Checkpointed: the recompute re-runs dropout and must regenerate the
    // exact mask from the stored seed — forward AND gradients are bitwise
    // identical to the plain run.
    var ck_out: []f32 = undefined;
    const ck_snap, const ck_loss = ck: {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const h = try checkpoint(&ctx, Blocks.dropoutLayer1, .{ &x, &w, &b });
        ck_out = try allocator.dupe(f32, try h.dataConst());
        const loss = try h.sumAll(&ctx);
        try loss.backward(&ctx);
        const loss_value = try loss.item();
        const gx = try gradData(&ctx, &x);
        errdefer allocator.free(gx);
        const gw = try gradData(&ctx, &w);
        errdefer allocator.free(gw);
        const gb = try gradData(&ctx, &b);
        break :ck .{ .{ gx, gw, gb }, loss_value };
    };
    defer allocator.free(ck_out);
    defer inline for (ck_snap) |s| allocator.free(s);

    try std.testing.expectEqualSlices(f32, plain_out, ck_out);
    try std.testing.expectEqual(plain_loss, ck_loss);
    inline for (plain_snap, ck_snap) |p, c| try std.testing.expectEqualSlices(f32, p, c);
}

test "checkpointWithContext frozen f16 weight matches plain backward bitwise" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(41);
    var w_buf: [n_h1 * n_in]f32 = undefined;
    fillUniform(prng.random(), &w_buf, 0.8);
    var w16_buf: [n_h1 * n_in]f16 = undefined;
    for (w_buf, &w16_buf) |v, *h| h.* = @floatCast(v);

    var w16 = try FrozenW.fromSlice(&ctx, .{ n_h1, n_in }, &w16_buf);
    defer w16.deinit();
    var x = try makeInput(&ctx, 53);
    defer x.deinit();

    const frozen = ContextBlocks.FrozenLayer{ .w = &w16, .scale = 0.75 };

    // Plain reference: the same block, un-checkpointed, under a scope.
    var plain_out: []f32 = undefined;
    const plain_gx, const plain_loss = plain: {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const h = try ContextBlocks.frozenLayer(&ctx, &frozen, &x);
        plain_out = try allocator.dupe(f32, try h.dataConst());
        const loss = try h.sumAll(&ctx);
        try loss.backward(&ctx);
        const loss_value = try loss.item();
        break :plain .{ try gradData(&ctx, &x), loss_value };
    };
    defer allocator.free(plain_out);
    defer allocator.free(plain_gx);

    x.zeroGrad();

    // Checkpointed: the recompute re-runs the f16 dot + scale with the SAME
    // `extra` value — forward AND gradients are bitwise identical.
    var ck_out: []f32 = undefined;
    const ck_gx, const ck_loss = ck: {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const h = try checkpointWithContext(&ctx, ContextBlocks.frozenLayer, &frozen, .{&x});
        ck_out = try allocator.dupe(f32, try h.dataConst());
        const loss = try h.sumAll(&ctx);
        try loss.backward(&ctx);
        const loss_value = try loss.item();
        break :ck .{ try gradData(&ctx, &x), loss_value };
    };
    defer allocator.free(ck_out);
    defer allocator.free(ck_gx);

    try std.testing.expectEqualSlices(f32, plain_out, ck_out);
    try std.testing.expectEqual(plain_loss, ck_loss);
    try std.testing.expectEqualSlices(f32, plain_gx, ck_gx);
    // The frozen weight is a constant facade: no gradient facility exists.
    try std.testing.expect(!w16.requiresGrad());
}

test "checkpointWithContext extra with slice and config fields matches plain bitwise" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(67);
    var w_buf: [n_h1 * n_in]f32 = undefined;
    var bias_buf: [n_h1]f32 = undefined;
    fillUniform(prng.random(), &w_buf, 0.8);
    fillUniform(prng.random(), &bias_buf, 0.2);
    var w16_buf: [n_h1 * n_in]f16 = undefined;
    for (w_buf, &w16_buf) |v, *h| h.* = @floatCast(v);

    var w16 = try FrozenW.fromSlice(&ctx, .{ n_h1, n_in }, &w16_buf);
    defer w16.deinit();
    var x = try makeInput(&ctx, 71);
    defer x.deinit();

    // `extra` by value: frozen weight pointer + raw bias slice + scalar. The
    // node stores the struct bits; the pointed-at weight and slice stay alive
    // here until backward completes (the `extra` validity contract).
    const biased = ContextBlocks.BiasedLayer{ .w = &w16, .bias = &bias_buf, .scale = 0.5 };

    var plain_out: []f32 = undefined;
    const plain_gx, const plain_loss = plain: {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const h = try ContextBlocks.biasedLayer(&ctx, biased, &x);
        plain_out = try allocator.dupe(f32, try h.dataConst());
        const loss = try h.sumAll(&ctx);
        try loss.backward(&ctx);
        const loss_value = try loss.item();
        break :plain .{ try gradData(&ctx, &x), loss_value };
    };
    defer allocator.free(plain_out);
    defer allocator.free(plain_gx);

    x.zeroGrad();

    var ck_out: []f32 = undefined;
    const ck_gx, const ck_loss = ck: {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const h = try checkpointWithContext(&ctx, ContextBlocks.biasedLayer, biased, .{&x});
        ck_out = try allocator.dupe(f32, try h.dataConst());
        const loss = try h.sumAll(&ctx);
        try loss.backward(&ctx);
        const loss_value = try loss.item();
        break :ck .{ try gradData(&ctx, &x), loss_value };
    };
    defer allocator.free(ck_out);
    defer allocator.free(ck_gx);

    try std.testing.expectEqualSlices(f32, plain_out, ck_out);
    try std.testing.expectEqual(plain_loss, ck_loss);
    try std.testing.expectEqualSlices(f32, plain_gx, ck_gx);
}

test "checkpointing a deep context-block chain retains materially fewer scope entries" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const d = 8;
    const chain_len = 8;
    var prng = std.Random.DefaultPrng.init(3);
    var w_buf: [d * d]f32 = undefined;
    var x_buf: [batch * d]f32 = undefined;
    fillUniform(prng.random(), &w_buf, 1.0 / @sqrt(@as(f32, d)));
    fillUniform(prng.random(), &x_buf, 1.0);

    // The chain weight is frozen (a constant) and rides in `extra`; only the
    // activation chain is differentiable.
    var w = try Tensor(.{ .dout, .din }).fromSlice(&ctx, .{ d, d }, &w_buf);
    defer w.deinit();
    var x = try Tensor(.{ .batch, .din }).variableFromSlice(&ctx, .{ batch, d }, &x_buf);
    defer x.deinit();

    const chain_ctx = ContextBlocks.SquareChain{ .w = &w };

    // Plain: every block intermediate is adopted by the scope and retained
    // until close (3 entries per block: dot, tanh, withTags).
    var plain_entries: usize = 0;
    const plain_gx, const plain_loss = plain: {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const base = ctx.rt.scope_entries.items.len;
        var h = try ContextBlocks.square(&ctx, &chain_ctx, &x);
        for (1..chain_len) |_| {
            h = try ContextBlocks.square(&ctx, &chain_ctx, &h);
        }
        plain_entries = ctx.rt.scope_entries.items.len - base;
        const loss = try h.sumAll(&ctx);
        try loss.backward(&ctx);
        const loss_value = try loss.item();
        break :plain .{ try gradData(&ctx, &x), loss_value };
    };
    defer allocator.free(plain_gx);

    x.zeroGrad();

    // Checkpointed with context: only the block OUTPUTS are retained (1 entry
    // per block) — the memory win is unchanged by the `extra` argument.
    var ck_entries: usize = 0;
    const ck_gx, const ck_loss = ck: {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const base = ctx.rt.scope_entries.items.len;
        var h = try checkpointWithContext(&ctx, ContextBlocks.square, &chain_ctx, .{&x});
        for (1..chain_len) |_| {
            h = try checkpointWithContext(&ctx, ContextBlocks.square, &chain_ctx, .{&h});
        }
        ck_entries = ctx.rt.scope_entries.items.len - base;
        const loss = try h.sumAll(&ctx);
        try loss.backward(&ctx);
        const loss_value = try loss.item();
        break :ck .{ try gradData(&ctx, &x), loss_value };
    };
    defer allocator.free(ck_gx);

    try std.testing.expectEqual(3 * chain_len, plain_entries);
    try std.testing.expectEqual(chain_len, ck_entries);
    try std.testing.expect(ck_entries * 2 < plain_entries);

    try std.testing.expectEqual(plain_loss, ck_loss);
    try std.testing.expectEqualSlices(f32, plain_gx, ck_gx);
}

test "context-block failure propagates from forward and recompute without leaking" {
    const Hostile = struct {
        const Ctx = struct { fail: *const bool };

        fn block(ctx: *ExecContext, extra: Ctx, x: *const Tensor(.{.d})) !Tensor(.{.d}) {
            if (extra.fail.*) return error.InducedContextBlockFailure;
            return x.mul(ctx, x);
        }
    };

    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ 1, 2, 3 });
    defer x.deinit();
    var fail = false;
    const hostile = Hostile.Ctx{ .fail = &fail };

    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);

    // Forward failure: the error surfaces from checkpointWithContext itself
    // (input views + buffers already captured at that point must be freed).
    fail = true;
    try std.testing.expectError(
        error.InducedContextBlockFailure,
        checkpointWithContext(&ctx, Hostile.block, hostile, .{&x}),
    );

    // Recompute failure: forward succeeds, backward re-runs the block with
    // the SAME stored `extra` and hits the (now armed) failure flag.
    fail = false;
    const h = try checkpointWithContext(&ctx, Hostile.block, hostile, .{&x});
    const loss = try h.sumAll(&ctx);
    fail = true;
    try std.testing.expectError(error.InducedContextBlockFailure, loss.backward(&ctx));
}

test "context-block dropout with seed via extra replays bitwise" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(29);
    var w_buf: [n_h1 * n_in]f32 = undefined;
    var b_buf: [n_h1]f32 = undefined;
    fillUniform(prng.random(), &w_buf, 0.8);
    fillUniform(prng.random(), &b_buf, 0.1);

    var x = try makeInput(&ctx, 37);
    defer x.deinit();
    var w = try Tensor(.{ .h1, .in }).variableFromSlice(&ctx, .{ n_h1, n_in }, &w_buf);
    defer w.deinit();
    var b = try Tensor(.{.h1}).variableFromSlice(&ctx, .{n_h1}, &b_buf);
    defer b.deinit();

    const drop = ContextBlocks.Dropout{ .p = 0.5, .seed = 0xfeed5 };

    var plain_out: []f32 = undefined;
    const plain_snap, const plain_loss = plain: {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const h = try ContextBlocks.dropoutLayer(&ctx, drop, &x, &w, &b);
        plain_out = try allocator.dupe(f32, try h.dataConst());
        const loss = try h.sumAll(&ctx);
        try loss.backward(&ctx);
        const loss_value = try loss.item();
        const gx = try gradData(&ctx, &x);
        errdefer allocator.free(gx);
        const gw = try gradData(&ctx, &w);
        errdefer allocator.free(gw);
        const gb = try gradData(&ctx, &b);
        break :plain .{ .{ gx, gw, gb }, loss_value };
    };
    defer allocator.free(plain_out);
    defer inline for (plain_snap) |s| allocator.free(s);

    x.zeroGrad();
    w.zeroGrad();
    b.zeroGrad();

    // The recompute re-runs dropout with the stored `extra` and must
    // regenerate the exact mask from the delivered seed — forward AND
    // gradients are bitwise identical to the plain run.
    var ck_out: []f32 = undefined;
    const ck_snap, const ck_loss = ck: {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const h = try checkpointWithContext(&ctx, ContextBlocks.dropoutLayer, drop, .{ &x, &w, &b });
        ck_out = try allocator.dupe(f32, try h.dataConst());
        const loss = try h.sumAll(&ctx);
        try loss.backward(&ctx);
        const loss_value = try loss.item();
        const gx = try gradData(&ctx, &x);
        errdefer allocator.free(gx);
        const gw = try gradData(&ctx, &w);
        errdefer allocator.free(gw);
        const gb = try gradData(&ctx, &b);
        break :ck .{ .{ gx, gw, gb }, loss_value };
    };
    defer allocator.free(ck_out);
    defer inline for (ck_snap) |s| allocator.free(s);

    try std.testing.expectEqualSlices(f32, plain_out, ck_out);
    try std.testing.expectEqual(plain_loss, ck_loss);
    inline for (plain_snap, ck_snap) |p, c| try std.testing.expectEqualSlices(f32, p, c);
}

const EinsumBlock = struct {
    fn block(ctx: *ExecContext, q: *const Tensor(.{ .b, .g, .i, .d }), k: *const Tensor(.{ .b, .j, .d })) !Tensor(.{ .b, .g, .i, .j }) {
        var scores = try q.einsum(ctx, k, .{ .b, .g, .i, .j });
        defer scores.deinit();
        return scores.tanh(ctx);
    }
};

test "checkpointed einsum block matches plain backward bitwise" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var q = try Tensor(.{ .b, .g, .i, .d }).variableFromSlice(&ctx, .{ 2, 2, 2, 3 }, &.{
        0.5,  -1.0, 0.3,  0.8,  -0.2, 1.1,  0.9, -0.7, 0.4,  -0.5, 1.2,  0.1,
        -0.9, 0.6,  0.2,  -0.3, 0.7,  -1.1, 0.1, 0.4,  -0.6, 0.8,  -0.4, 0.2,
    });
    defer q.deinit();
    var k = try Tensor(.{ .b, .j, .d }).variableFromSlice(&ctx, .{ 2, 2, 3 }, &.{
        0.2, -0.4, 0.1, 0.5, -0.3, 0.6, 0.7, -0.1, 0.3, 0.2, -0.5, 0.4,
    });
    defer k.deinit();

    // Plain reference: inlined ops, every intermediate alive until after
    // backward (calling the block fn directly would deinit the grad-carrying
    // einsum intermediate before backward — the composed-op scope rule).
    const plain_gq, const plain_gk, const plain_loss = plain: {
        var scores = try q.einsum(&ctx, &k, .{ .b, .g, .i, .j });
        defer scores.deinit();
        var a = try scores.tanh(&ctx);
        defer a.deinit();
        var loss = try a.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);
        var gq = (try q.grad(&ctx)).?;
        defer gq.deinit();
        var gk = (try k.grad(&ctx)).?;
        defer gk.deinit();
        break :plain .{
            try allocator.dupe(f32, gq.asRawTensor().dataConst()),
            try allocator.dupe(f32, gk.asRawTensor().dataConst()),
            try loss.item(),
        };
    };
    defer allocator.free(plain_gq);
    defer allocator.free(plain_gk);

    q.zeroGrad();
    k.zeroGrad();

    const ck_loss = ck: {
        var a = try checkpoint(&ctx, EinsumBlock.block, .{ &q, &k });
        defer a.deinit();
        var loss = try a.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);
        break :ck try loss.item();
    };

    try std.testing.expectEqual(plain_loss, ck_loss);
    var gq = (try q.grad(&ctx)).?;
    defer gq.deinit();
    var gk = (try k.grad(&ctx)).?;
    defer gk.deinit();
    try std.testing.expectEqualSlices(f32, plain_gq, gq.asRawTensor().dataConst());
    try std.testing.expectEqualSlices(f32, plain_gk, gk.asRawTensor().dataConst());
}

fn allocationFailureProbe(allocator: std.mem.Allocator) !void {
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .batch, .in }).variableFromSlice(&ctx, .{ 1, 2 }, &.{ 0.5, -0.25 });
    defer x.deinit();
    var w = try Tensor(.{ .h1, .in }).variableFromSlice(&ctx, .{ 2, 2 }, &.{ 0.1, 0.2, -0.3, 0.4 });
    defer w.deinit();
    var b = try Tensor(.{.h1}).variableFromSlice(&ctx, .{2}, &.{ 0.05, -0.15 });
    defer b.deinit();

    var y = try checkpoint(&ctx, Blocks.layer1, .{ &x, &w, &b });
    defer y.deinit();
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    var gw = (try w.grad(&ctx)).?;
    defer gw.deinit();
    var gb = (try b.grad(&ctx)).?;
    defer gb.deinit();
}

test "checkpoint releases exactly once under induced allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationFailureProbe, .{});
}
