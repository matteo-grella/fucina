//! Behavioral tests for the LoRA adapter module (`lora.zig`): zero-delta
//! init, gradient routing around frozen bases, optimizer integration,
//! deterministic init, state-dict round trips, merge parity, dropout
//! determinism, and the scoped train-step loop.
const std = @import("std");
const exec_mod = @import("exec.zig");
const ag = @import("ag.zig");
const dtype_mod = @import("dtype.zig");
const lora = @import("lora.zig");
const optim = @import("optim.zig");
const rng = @import("rng.zig");

const ExecContext = exec_mod.ExecContext;
const Tensor = ag.Tensor;

fn f16Bits(value: f32) u16 {
    const h: f16 = @floatCast(value);
    return @bitCast(h);
}

fn anyNonZero(values: []const f32) bool {
    for (values) |v| if (v != 0) return true;
    return false;
}

fn expectCloseSlices(expected: []const f32, actual: []const f32, tolerance: f32) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |e, a| {
        try std.testing.expectApproxEqAbs(e, a, tolerance);
    }
}

test "lora delta is exactly zero at init and apply returns base bitwise" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var adapter = try lora.Adapter(.in, .out).init(&ctx, 8, 6, .{ .rank = 2, .alpha = 4 }, 42);
    defer adapter.deinit();

    var x_vals: [16]f32 = undefined;
    rng.uniformFill(7, &x_vals, -1, 1);
    var x = try Tensor(.{ .batch, .in }).fromSlice(&ctx, .{ 2, 8 }, &x_vals);
    defer x.deinit();
    var base_vals: [12]f32 = undefined;
    rng.uniformFill(8, &base_vals, -2, 2);
    var base = try Tensor(.{ .batch, .out }).fromSlice(&ctx, .{ 2, 6 }, &base_vals);
    defer base.deinit();

    // B is zero-initialized, so the delta is exactly zero...
    var d = try adapter.delta(&ctx, &x, null);
    defer d.deinit();
    for (try d.dataConst()) |v| try std.testing.expectEqual(@as(f32, 0), v);

    // ...and apply is bitwise the base.
    var y = try adapter.apply(&ctx, &x, &base, null);
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &base_vals, try y.dataConst());
}

test "lora backward reaches A and B; the frozen quantized base path gets no gradient" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const in_dim = dtype_mod.q8_0_block_size; // one Q8_0 block per weight row
    var adapter = try lora.Adapter(.in, .out).init(&ctx, in_dim, 2, .{ .rank = 2, .alpha = 4 }, 0xA);
    defer adapter.deinit();
    // Give B nonzero content the way an optimizer step would (in-place leaf
    // update) so every gradient below is generically nonzero.
    rng.uniformFill(0xB, adapter.b.value.data(), -0.5, 0.5);

    // Frozen Q8_0 weight: no GradState exists for it at all.
    const W = Tensor(.{ .dtype = .q8_0, .tags = .{ .out, .in } });
    var blocks = [_]dtype_mod.BlockQ8_0{
        .{ .d = f16Bits(0.5), .qs = [_]i8{1} ** dtype_mod.q8_0_block_size },
        .{ .d = f16Bits(0.25), .qs = [_]i8{2} ** dtype_mod.q8_0_block_size },
    };
    var wq = try W.fromBlocks(&ctx, .{ 2, in_dim }, &blocks);
    defer wq.deinit();
    try std.testing.expect(!wq.requiresGrad());

    var x_vals: [2 * in_dim]f32 = undefined;
    rng.uniformFill(0xC, &x_vals, -1, 1);
    var x = try Tensor(.{ .batch, .in }).variableFromSlice(&ctx, .{ 2, in_dim }, &x_vals);
    defer x.deinit();

    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const base = try x.dot(&ctx, &wq, .in);
        const y = try adapter.apply(&ctx, &x, &base, null);
        const loss = try y.sumAll(&ctx);
        try loss.backward(&ctx);

        var ga = (try adapter.a.grad(&ctx)).?;
        defer ga.deinit();
        try std.testing.expect(anyNonZero(try ga.dataConst()));
        var gb = (try adapter.b.grad(&ctx)).?;
        defer gb.deinit();
        try std.testing.expect(anyNonZero(try gb.dataConst()));
        // x receives gradient through BOTH the frozen ConstRhsDot base path
        // and the trainable delta path; the weight itself stays grad-free.
        var gx = (try x.grad(&ctx)).?;
        defer gx.deinit();
        try std.testing.expect(anyNonZero(try gx.dataConst()));

        adapter.a.zeroGrad();
        adapter.b.zeroGrad();
        x.zeroGrad();
    }
}

test "lora AdamW step trains A and B end-to-end" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var adapter = try lora.Adapter(.in, .out).init(&ctx, 8, 4, .{ .rank = 2, .alpha = 4 }, 5);
    defer adapter.deinit();

    var x_vals: [24]f32 = undefined;
    rng.uniformFill(6, &x_vals, -1, 1);
    var x = try Tensor(.{ .batch, .in }).fromSlice(&ctx, .{ 3, 8 }, &x_vals);
    defer x.deinit();
    var base_vals: [12]f32 = undefined;
    rng.uniformFill(7, &base_vals, -1, 1);
    var base = try Tensor(.{ .batch, .out }).fromSlice(&ctx, .{ 3, 4 }, &base_vals);
    defer base.deinit();

    var opt = optim.AdamW.init(allocator, .{ .lr = 0.1, .weight_decay = 0 });
    defer opt.deinit();
    try adapter.registerParams(&opt, "layer0");

    var a0: [16]f32 = undefined;
    @memcpy(&a0, try adapter.a.dataConst());
    var b0: [8]f32 = undefined;
    @memcpy(&b0, try adapter.b.dataConst());

    for (0..2) |step_i| {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const y = try adapter.apply(&ctx, &x, &base, null);
        const loss = try y.sumAll(&ctx);
        try loss.backward(&ctx);
        try opt.step(&ctx);
        opt.zeroGrad();

        if (step_i == 0) {
            // After step 1: B moved (its grad is nonzero), A did not — its
            // gradient is exactly zero while B is zero, and weight decay is off.
            try std.testing.expect(!std.mem.eql(f32, &b0, try adapter.b.dataConst()));
            try std.testing.expectEqualSlices(f32, &a0, try adapter.a.dataConst());
        }
    }
    // After step 2, B is nonzero so A received signal and moved too.
    try std.testing.expect(!std.mem.eql(f32, &a0, try adapter.a.dataConst()));

    // The trained adapter changes the forward relative to the frozen base.
    var y_eval = try adapter.apply(&ctx, &x, &base, null);
    defer y_eval.deinit();
    try std.testing.expect(!std.mem.eql(f32, &base_vals, try y_eval.dataConst()));
}

test "lora init: kaiming bound on A, zero B, seed-deterministic, validated config" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const in_dim = 64;
    const bound = @sqrt(1.0 / @as(f32, in_dim));
    var adapter = try lora.Adapter(.in, .out).init(&ctx, in_dim, 16, .{ .rank = 4, .alpha = 8 }, 99);
    defer adapter.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), adapter.scale, 1e-7); // alpha / rank

    const a_data = try adapter.a.dataConst();
    var max_abs: f32 = 0;
    var all_equal = true;
    for (a_data) |v| {
        try std.testing.expect(@abs(v) <= bound + 1e-7);
        max_abs = @max(max_abs, @abs(v));
        if (v != a_data[0]) all_equal = false;
    }
    try std.testing.expect(!all_equal);
    try std.testing.expect(max_abs > 0);
    for (try adapter.b.dataConst()) |v| try std.testing.expectEqual(@as(f32, 0), v);

    // Same seed -> bitwise-identical A.
    var twin = try lora.Adapter(.in, .out).init(&ctx, in_dim, 16, .{ .rank = 4, .alpha = 8 }, 99);
    defer twin.deinit();
    try std.testing.expectEqualSlices(f32, a_data, try twin.a.dataConst());
    var other = try lora.Adapter(.in, .out).init(&ctx, in_dim, 16, .{ .rank = 4, .alpha = 8 }, 100);
    defer other.deinit();
    try std.testing.expect(!std.mem.eql(f32, a_data, try other.a.dataConst()));

    // Config validation.
    const A = lora.Adapter(.in, .out);
    try std.testing.expectError(lora.LoraError.InvalidRank, A.init(&ctx, 8, 4, .{ .rank = 0, .alpha = 1 }, 1));
    try std.testing.expectError(lora.LoraError.InvalidRank, A.init(&ctx, 8, 4, .{ .rank = 5, .alpha = 1 }, 1));
    try std.testing.expectError(lora.LoraError.InvalidDropout, A.init(&ctx, 8, 4, .{ .rank = 2, .alpha = 1, .dropout_p = 1 }, 1));
    try std.testing.expectError(lora.LoraError.InvalidDropout, A.init(&ctx, 8, 4, .{ .rank = 2, .alpha = 1, .dropout_p = -0.1 }, 1));
}

test "lora state-dict round trip restores forwards bitwise under permuted entry order" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var l0 = try lora.Adapter(.in, .out).init(&ctx, 6, 4, .{ .rank = 2, .alpha = 4 }, 1);
    defer l0.deinit();
    var l1 = try lora.Adapter(.in, .out).init(&ctx, 6, 4, .{ .rank = 2, .alpha = 4 }, 2);
    defer l1.deinit();
    rng.uniformFill(101, l0.b.value.data(), -0.5, 0.5);
    rng.uniformFill(102, l1.b.value.data(), -0.5, 0.5);

    var x_vals: [12]f32 = undefined;
    rng.uniformFill(7, &x_vals, -1, 1);
    var x = try Tensor(.{ .batch, .in }).fromSlice(&ctx, .{ 2, 6 }, &x_vals);
    defer x.deinit();
    var base = try Tensor(.{ .batch, .out }).fromSlice(&ctx, .{ 2, 4 }, &([_]f32{0} ** 8));
    defer base.deinit();

    var y0_vals: [8]f32 = undefined;
    var y1_vals: [8]f32 = undefined;
    {
        var y0 = try l0.apply(&ctx, &x, &base, null);
        defer y0.deinit();
        @memcpy(&y0_vals, try y0.dataConst());
        var y1 = try l1.apply(&ctx, &x, &base, null);
        defer y1.deinit();
        @memcpy(&y1_vals, try y1.dataConst());
    }

    var buf: [4096]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    const e0 = try l0.namedTensors("l0");
    const e1 = try l1.namedTensors("l1");
    const save_entries = [_]optim.NamedTensor{ e0[0], e0[1], e1[0], e1[1] };
    try optim.saveStateDict(allocator, &writer, &save_entries);
    const written = writer.buffered();

    // Fresh adapters from different seeds (different A, zero B)...
    var l0b = try lora.Adapter(.in, .out).init(&ctx, 6, 4, .{ .rank = 2, .alpha = 4 }, 8);
    defer l0b.deinit();
    var l1b = try lora.Adapter(.in, .out).init(&ctx, 6, 4, .{ .rank = 2, .alpha = 4 }, 9);
    defer l1b.deinit();
    try std.testing.expect(!std.mem.eql(f32, try l0.a.dataConst(), try l0b.a.dataConst()));

    // ...loaded with the entries permuted relative to stream order: the
    // state dict is name-matched, not positional.
    const m0 = try l0b.namedTensorsMut("l0");
    const m1 = try l1b.namedTensorsMut("l1");
    const load_entries = [_]optim.NamedTensorMut{ m1[1], m0[0], m1[0], m0[1] };
    var reader = std.Io.Reader.fixed(written);
    try optim.loadStateDict(allocator, &reader, &load_entries, .{});

    var y0b = try l0b.apply(&ctx, &x, &base, null);
    defer y0b.deinit();
    try std.testing.expectEqualSlices(f32, &y0_vals, try y0b.dataConst());
    var y1b = try l1b.apply(&ctx, &x, &base, null);
    defer y1b.deinit();
    try std.testing.expectEqualSlices(f32, &y1_vals, try y1b.dataConst());
}

test "lora merge parity: merged weight matches apply; f16 helper round-trips dtype" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var adapter = try lora.Adapter(.in, .out).init(&ctx, 8, 5, .{ .rank = 3, .alpha = 6 }, 3);
    defer adapter.deinit();
    rng.uniformFill(4, adapter.b.value.data(), -0.3, 0.3);

    var x_vals: [16]f32 = undefined;
    rng.uniformFill(5, &x_vals, -1, 1);
    var x = try Tensor(.{ .batch, .in }).fromSlice(&ctx, .{ 2, 8 }, &x_vals);
    defer x.deinit();
    var w_vals: [40]f32 = undefined;
    rng.uniformFill(6, &w_vals, -1, 1);

    // f32: y2 = apply(x, x·W^T) vs y1 = x·(W + scale·B·A)^T. Equal up to a
    // tight tolerance — the accumulation ORDER differs (per-output sum of
    // 8 products + separate rank-3 delta sum vs one merged 8-product sum),
    // so bitwise equality is not expected.
    var w = try Tensor(.{ .out, .in }).fromSlice(&ctx, .{ 5, 8 }, &w_vals);
    defer w.deinit();
    var base = try x.dot(&ctx, &w, .in);
    defer base.deinit();
    var y2 = try adapter.apply(&ctx, &x, &base, null);
    defer y2.deinit();

    var w_merged = try Tensor(.{ .out, .in }).fromSlice(&ctx, .{ 5, 8 }, &w_vals);
    defer w_merged.deinit();
    try adapter.mergeInto(&ctx, &w_merged);
    try std.testing.expect(!std.mem.eql(f32, &w_vals, try w_merged.dataConst()));
    var y1 = try x.dot(&ctx, &w_merged, .in);
    defer y1.deinit();
    try expectCloseSlices(try y2.dataConst(), try y1.dataConst(), 1e-4);

    // The facade mutability gate: merging into a VARIABLE weight is refused.
    var w_var = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 5, 8 }, &w_vals);
    defer w_var.deinit();
    try std.testing.expectError(error.MutableDataRequiresNoGrad, adapter.mergeInto(&ctx, &w_var));

    // Shape mismatch is a runtime error.
    var w_small = try Tensor(.{ .out, .in }).fromSlice(&ctx, .{ 4, 8 }, w_vals[0..32]);
    defer w_small.deinit();
    try std.testing.expectError(error.ShapeMismatch, adapter.mergeInto(&ctx, &w_small));

    // f16 helper: widen -> merge -> cast back, returning a new f16 tensor.
    // Looser tolerance: the merged weights are rounded to f16 once.
    var w16_vals: [40]f16 = undefined;
    for (w_vals, &w16_vals) |v, *h| h.* = @floatCast(v);
    var w16 = try Tensor(.{ .dtype = .f16, .tags = .{ .out, .in } }).fromSlice(&ctx, .{ 5, 8 }, &w16_vals);
    defer w16.deinit();
    var base16 = try x.dot(&ctx, &w16, .in);
    defer base16.deinit();
    var y2h = try adapter.apply(&ctx, &x, &base16, null);
    defer y2h.deinit();

    var merged16 = try adapter.mergeF16(&ctx, &w16);
    defer merged16.deinit();
    comptime std.debug.assert(@TypeOf(merged16).dtype == .f16);
    var y1h = try x.dot(&ctx, &merged16, .in);
    defer y1h.deinit();
    try expectCloseSlices(try y2h.dataConst(), try y1h.dataConst(), 0.05);
}

test "lora dropout: seed-deterministic delta, eval equals the p=0 path, train differs" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Two adapters with identical A (same init seed) and identical raw-written
    // B, differing only in dropout_p.
    var ad05 = try lora.Adapter(.in, .out).init(&ctx, 16, 4, .{ .rank = 2, .alpha = 2, .dropout_p = 0.5 }, 11);
    defer ad05.deinit();
    var ad0 = try lora.Adapter(.in, .out).init(&ctx, 16, 4, .{ .rank = 2, .alpha = 2 }, 11);
    defer ad0.deinit();
    rng.uniformFill(12, ad05.b.value.data(), -0.5, 0.5);
    rng.uniformFill(12, ad0.b.value.data(), -0.5, 0.5);

    var x_vals: [48]f32 = undefined;
    rng.uniformFill(13, &x_vals, 0.25, 1.25); // bounded away from zero
    var x = try Tensor(.{ .batch, .in }).fromSlice(&ctx, .{ 3, 16 }, &x_vals);
    defer x.deinit();

    var d1 = try ad05.delta(&ctx, &x, 7);
    defer d1.deinit();
    var d2 = try ad05.delta(&ctx, &x, 7);
    defer d2.deinit();
    try std.testing.expectEqualSlices(f32, try d1.dataConst(), try d2.dataConst());

    var d3 = try ad05.delta(&ctx, &x, 8);
    defer d3.deinit();
    try std.testing.expect(!std.mem.eql(f32, try d1.dataConst(), try d3.dataConst()));

    // Eval (null seed) skips dropout: bitwise the p=0 adapter's delta.
    var d_eval = try ad05.delta(&ctx, &x, null);
    defer d_eval.deinit();
    var d_p0 = try ad0.delta(&ctx, &x, 99); // seed is ignored at p = 0
    defer d_p0.deinit();
    try std.testing.expectEqualSlices(f32, try d_eval.dataConst(), try d_p0.dataConst());

    // Train mode (masked + 1/(1-p) rescale) differs from eval.
    try std.testing.expect(!std.mem.eql(f32, try d1.dataConst(), try d_eval.dataConst()));
}

test "lora scoped train loop over a frozen f16 base is leak-free and the loss decreases" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var adapter = try lora.Adapter(.in, .out).init(&ctx, 8, 5, .{ .rank = 2, .alpha = 4, .dropout_p = 0.1 }, 21);
    defer adapter.deinit();

    var w16_vals: [40]f16 = undefined;
    var w_seed_vals: [40]f32 = undefined;
    rng.uniformFill(22, &w_seed_vals, -1, 1);
    for (w_seed_vals, &w16_vals) |v, *h| h.* = @floatCast(v);
    var w16 = try Tensor(.{ .dtype = .f16, .tags = .{ .out, .in } }).fromSlice(&ctx, .{ 5, 8 }, &w16_vals);
    defer w16.deinit();

    var x_vals: [32]f32 = undefined;
    rng.uniformFill(23, &x_vals, -1, 1);
    var x = try Tensor(.{ .batch, .in }).fromSlice(&ctx, .{ 4, 8 }, &x_vals);
    defer x.deinit();

    // Overfittable toy target: the frozen base shifted by a constant — the
    // adapter must learn a rank-2 correction toward it.
    var target_vals: [20]f32 = undefined;
    {
        var base0 = try x.dot(&ctx, &w16, .in);
        defer base0.deinit();
        for (try base0.dataConst(), &target_vals) |b, *t| t.* = b + 1.0;
    }
    var target = try Tensor(.{ .batch, .out }).fromSlice(&ctx, .{ 4, 5 }, &target_vals);
    defer target.deinit();

    var opt = optim.AdamW.init(allocator, .{ .lr = 0.05, .weight_decay = 0 });
    defer opt.deinit();
    try adapter.registerParams(&opt, "l0");

    var losses: [3]f32 = undefined;
    for (0..losses.len) |step_i| {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const base = try x.dot(&ctx, &w16, .in);
        const y = try adapter.apply(&ctx, &x, &base, rng.at(0x10ad, step_i));
        const diff = try y.sub(&ctx, &target);
        const sq = try diff.mul(&ctx, &diff);
        const loss = try sq.sumAll(&ctx);
        try loss.backward(&ctx);
        try opt.step(&ctx);
        opt.zeroGrad();
        losses[step_i] = try loss.item();
    }
    try std.testing.expect(losses[2] < losses[0]);
}

test "lora scoped train loop over a frozen bf16 base is leak-free and the loss decreases" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var adapter = try lora.Adapter(.in, .out).init(&ctx, 8, 5, .{ .rank = 2, .alpha = 4, .dropout_p = 0.1 }, 31);
    defer adapter.deinit();
    // Give B nonzero content the way an optimizer step would (as in the q8
    // grad-routing test) so A's gradient below is generically nonzero.
    rng.uniformFill(0xB1, adapter.b.value.data(), -0.5, 0.5);

    // Frozen bf16 base: the dot routes through the mixed f32 x bf16 TransB
    // kernel forward and ConstRhsDotBackward(.bf16) backward — the only
    // differentiable bf16 weight path.
    var wbf_vals: [40]u16 = undefined;
    var w_seed_vals: [40]f32 = undefined;
    rng.uniformFill(32, &w_seed_vals, -1, 1);
    for (w_seed_vals, &wbf_vals) |v, *bits| bits.* = dtype_mod.f32ToBf16(v);
    var wbf = try Tensor(.{ .dtype = .bf16, .tags = .{ .out, .in } }).fromSlice(&ctx, .{ 5, 8 }, &wbf_vals);
    defer wbf.deinit();
    try std.testing.expect(!wbf.requiresGrad());

    var x_vals: [32]f32 = undefined;
    rng.uniformFill(33, &x_vals, -1, 1);
    var x = try Tensor(.{ .batch, .in }).fromSlice(&ctx, .{ 4, 8 }, &x_vals);
    defer x.deinit();

    // Overfittable toy target: the frozen base shifted by a constant — the
    // adapter must learn a rank-2 correction toward it.
    var target_vals: [20]f32 = undefined;
    {
        var base0 = try x.dot(&ctx, &wbf, .in);
        defer base0.deinit();
        for (try base0.dataConst(), &target_vals) |b, *t| t.* = b + 1.0;
    }
    var target = try Tensor(.{ .batch, .out }).fromSlice(&ctx, .{ 4, 5 }, &target_vals);
    defer target.deinit();

    var opt = optim.AdamW.init(allocator, .{ .lr = 0.05, .weight_decay = 0 });
    defer opt.deinit();
    try adapter.registerParams(&opt, "l0");

    var losses: [3]f32 = undefined;
    for (0..losses.len) |step_i| {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const base = try x.dot(&ctx, &wbf, .in);
        const y = try adapter.apply(&ctx, &x, &base, rng.at(0xbf16, step_i));
        const diff = try y.sub(&ctx, &target);
        const sq = try diff.mul(&ctx, &diff);
        const loss = try sq.sumAll(&ctx);
        try loss.backward(&ctx);

        // Mirror the q8 grad-routing test: A and B receive gradient through
        // the frozen bf16 base; the base itself has no GradState at all.
        var ga = (try adapter.a.grad(&ctx)).?;
        defer ga.deinit();
        try std.testing.expect(anyNonZero(try ga.dataConst()));
        var gb = (try adapter.b.grad(&ctx)).?;
        defer gb.deinit();
        try std.testing.expect(anyNonZero(try gb.dataConst()));

        try opt.step(&ctx);
        opt.zeroGrad();
        losses[step_i] = try loss.item();
    }
    try std.testing.expect(losses[2] < losses[0]);
}
