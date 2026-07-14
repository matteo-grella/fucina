//! Mechanism tests for `cartridge.zig`: gradient flow into a trainable KV
//! prefix through concat + grouped attention (the kv_seq > q_seq arm), the
//! prefill-equivalence invariant, sink freezing, and the distillation loss.
//! Torch parity lives in `cartridge_golden_tests.zig`.

const std = @import("std");
const fucina = @import("fucina");
const cartridge = @import("cartridge.zig");

const ExecContext = fucina.ExecContext;
const Tensor = fucina.Tensor;
const Kv = cartridge.Kv;

// Query heads 0,1 read kv head 0; 2,3 read kv head 1 (GQA 4:2).
const gqa_map = [_]usize{ 0, 0, 1, 1 };
const attn_scale: f32 = 0.6;

/// Cartridge attention as one gradcheck-able scalar: sum of
/// attention(q, concat(zk, k_tok), concat(zv, v_tok)) with kv_seq > q_seq.
fn prefixAttentionLoss(
    ctx: *ExecContext,
    q: *const Tensor(.{ .seq, .head, .d }),
    zk: *const Kv,
    zv: *const Kv,
    k_tok: *const Kv,
    v_tok: *const Kv,
) !Tensor(.{}) {
    var k_full = try zk.concat(ctx, .seq, &.{k_tok});
    defer k_full.deinit();
    var v_full = try zv.concat(ctx, .seq, &.{v_tok});
    defer v_full.deinit();
    var y = try q.groupedAttention(ctx, &k_full, &v_full, gqa_map[0..], .attn, attn_scale, .{});
    defer y.deinit();
    return y.sumAll(ctx);
}

test "cartridge prefix attention matches finite differences (kv_seq > q_seq, GQA)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    // T=3 queries over kv_seq = p(2) + 3 = 5: source_offset = 2 in both the
    // forward kernel and the fused backward.
    var q = try Tensor(.{ .seq, .head, .d }).variableFromSlice(&ctx, .{ 3, 4, 2 }, &.{
        0.2,  -0.4, 0.5,  0.1,  -0.3, 0.7,  0.05, -0.6,
        0.4,  0.3,  -0.2, -0.1, 0.6,  -0.5, 0.15, 0.25,
        -0.7, 0.1,  0.35, -0.45, 0.55, 0.65, -0.05, -0.15,
    });
    defer q.deinit();
    var zk = try Kv.variableFromSlice(&ctx, .{ 2, 2, 2 }, &.{
        0.3, -0.2, -0.1, 0.4, 0.7, 0.2, -0.5, 0.1,
    });
    defer zk.deinit();
    var zv = try Kv.variableFromSlice(&ctx, .{ 2, 2, 2 }, &.{
        0.7, -0.6, 0.2, 0.5, -0.3, 0.45, 0.6, -0.25,
    });
    defer zv.deinit();
    var k_tok = try Kv.variableFromSlice(&ctx, .{ 3, 2, 2 }, &.{
        0.1, 0.5, -0.35, 0.2, -0.4, 0.6, 0.3, -0.7, 0.25, -0.15, 0.45, 0.05,
    });
    defer k_tok.deinit();
    var v_tok = try Kv.variableFromSlice(&ctx, .{ 3, 2, 2 }, &.{
        -0.2, 0.4, 0.55, -0.5, 0.15, 0.35, -0.65, 0.1, 0.5, 0.6, -0.1, 0.3,
    });
    defer v_tok.deinit();

    const result = try fucina.gradcheck(&ctx, prefixAttentionLoss, .{ &q, &zk, &zv, &k_tok, &v_tok }, .{});
    try std.testing.expectEqual(@as(usize, 24 + 8 + 8 + 12 + 12), result.checked);
}

/// As `prefixAttentionLoss` with a sliding window of 3 — the Gemma-style
/// local-SWA arm: the window bound cuts THROUGH the prefix, so some
/// queries see only part of it (and the last sees none).
fn windowedPrefixAttentionLoss(
    ctx: *ExecContext,
    q: *const Tensor(.{ .seq, .head, .d }),
    zk: *const Kv,
    zv: *const Kv,
    k_tok: *const Kv,
    v_tok: *const Kv,
) !Tensor(.{}) {
    var k_full = try zk.concat(ctx, .seq, &.{k_tok});
    defer k_full.deinit();
    var v_full = try zv.concat(ctx, .seq, &.{v_tok});
    defer v_full.deinit();
    var y = try q.groupedAttention(ctx, &k_full, &v_full, gqa_map[0..], .attn, attn_scale, .{ .window = 3 });
    defer y.deinit();
    return y.sumAll(ctx);
}

test "windowed prefix attention matches finite differences (SWA cuts the prefix)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    // Same geometry as the full-causal gradcheck (p = 2 prefix rows,
    // 3 token queries, source_offset = 2) with window = 3: query 0 sees
    // the whole prefix, query 1 sees only row 1, query 2 sees none — the
    // exact mask a Gemma local-SWA layer applies over a served cartridge.
    var q = try Tensor(.{ .seq, .head, .d }).variableFromSlice(&ctx, .{ 3, 4, 2 }, &.{
        0.2,  -0.4, 0.5,  0.1,   -0.3, 0.7,  0.05,  -0.6,
        0.4,  0.3,  -0.2, -0.1,  0.6,  -0.5, 0.15,  0.25,
        -0.7, 0.1,  0.35, -0.45, 0.55, 0.65, -0.05, -0.15,
    });
    defer q.deinit();
    var zk = try Kv.variableFromSlice(&ctx, .{ 2, 2, 2 }, &.{
        0.3, -0.2, -0.1, 0.4, 0.7, 0.2, -0.5, 0.1,
    });
    defer zk.deinit();
    var zv = try Kv.variableFromSlice(&ctx, .{ 2, 2, 2 }, &.{
        0.7, -0.6, 0.2, 0.5, -0.3, 0.45, 0.6, -0.25,
    });
    defer zv.deinit();
    var k_tok = try Kv.variableFromSlice(&ctx, .{ 3, 2, 2 }, &.{
        0.1, 0.5, -0.35, 0.2, -0.4, 0.6, 0.3, -0.7, 0.25, -0.15, 0.45, 0.05,
    });
    defer k_tok.deinit();
    var v_tok = try Kv.variableFromSlice(&ctx, .{ 3, 2, 2 }, &.{
        -0.2, 0.4, 0.55, -0.5, 0.15, 0.35, -0.65, 0.1, 0.5, 0.6, -0.1, 0.3,
    });
    defer v_tok.deinit();

    const result = try fucina.gradcheck(&ctx, windowedPrefixAttentionLoss, .{ &q, &zk, &zv, &k_tok, &v_tok }, .{});
    try std.testing.expectEqual(@as(usize, 24 + 8 + 8 + 12 + 12), result.checked);
}

test "cartridge from prefix rows reproduces the full-sequence attention tail" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Full attention over T=6, then a cartridge built from the first p=3
    // K/V rows (1 sink + 2 trainable) serving the last 3 queries: the
    // end-aligned causal kernel must produce the exact tail rows.
    const t_full = 6;
    const p = 3;
    const heads = 4;
    const kv_heads = 2;
    const d = 2;

    var q_data: [t_full * heads * d]f32 = undefined;
    var k_data: [t_full * kv_heads * d]f32 = undefined;
    var v_data: [t_full * kv_heads * d]f32 = undefined;
    fucina.rng.normalFill(1, &q_data, 0, 0.5);
    fucina.rng.normalFill(2, &k_data, 0, 0.5);
    fucina.rng.normalFill(3, &v_data, 0, 0.5);

    var q_full = try Tensor(.{ .seq, .head, .d }).fromSlice(&ctx, .{ t_full, heads, d }, &q_data);
    defer q_full.deinit();
    var k_full = try Kv.fromSlice(&ctx, .{ t_full, kv_heads, d }, &k_data);
    defer k_full.deinit();
    var v_full = try Kv.fromSlice(&ctx, .{ t_full, kv_heads, d }, &v_data);
    defer v_full.deinit();
    var y_full = try q_full.groupedAttention(&ctx, &k_full, &v_full, gqa_map[0..], .attn, attn_scale, .{});
    defer y_full.deinit();

    const row = kv_heads * d;
    var cart = try cartridge.Cartridge.initFromRows(
        &ctx,
        allocator,
        1,
        p,
        kv_heads,
        d,
        &.{k_data[0 .. p * row]},
        &.{v_data[0 .. p * row]},
    );
    defer cart.deinit();

    var q_tail = try Tensor(.{ .seq, .head, .d }).fromSlice(&ctx, .{ t_full - p, heads, d }, q_data[p * heads * d ..]);
    defer q_tail.deinit();
    var k_tok = try Kv.fromSlice(&ctx, .{ t_full - p, kv_heads, d }, k_data[p * row ..]);
    defer k_tok.deinit();
    var v_tok = try Kv.fromSlice(&ctx, .{ t_full - p, kv_heads, d }, v_data[p * row ..]);
    defer v_tok.deinit();

    var k_cat = try cart.layers[0].catK(&ctx, &k_tok);
    defer k_cat.deinit();
    var v_cat = try cart.layers[0].catV(&ctx, &v_tok);
    defer v_cat.deinit();
    var y_tail = try q_tail.groupedAttention(&ctx, &k_cat, &v_cat, gqa_map[0..], .attn, attn_scale, .{});
    defer y_tail.deinit();

    const full = try y_full.dataConst();
    const tail = try y_tail.dataConst();
    try std.testing.expectEqual(@as(usize, (t_full - p) * heads * d), tail.len);
    for (tail, full[p * heads * d ..]) |got, want| {
        try std.testing.expectApproxEqAbs(want, got, 1e-6);
    }
}

test "frozen sink rows stay constant while trainable rows receive gradient" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const p = 3;
    const kv_heads = 2;
    const d = 2;
    const t = 2;
    const row = kv_heads * d;

    var kv_data: [2 * p * row]f32 = undefined;
    fucina.rng.normalFill(7, &kv_data, 0, 0.5);
    var cart = try cartridge.Cartridge.initFromRows(
        &ctx,
        allocator,
        1,
        p,
        kv_heads,
        d,
        &.{kv_data[0 .. p * row]},
        &.{kv_data[p * row ..]},
    );
    defer cart.deinit();

    try std.testing.expect(!cart.layers[0].k_sink.?.requiresGrad());
    try std.testing.expect(!cart.layers[0].v_sink.?.requiresGrad());
    try std.testing.expect(cart.layers[0].k.requiresGrad());
    try std.testing.expect(cart.layers[0].v.requiresGrad());

    var q = try Tensor(.{ .seq, .head, .d }).fromSlice(&ctx, .{ t, 4, d }, &.{
        0.2, -0.4, 0.5, 0.1, -0.3, 0.7, 0.05, -0.6,
        0.4, 0.3, -0.2, -0.1, 0.6, -0.5, 0.15, 0.25,
    });
    defer q.deinit();
    var k_tok = try Kv.fromSlice(&ctx, .{ t, kv_heads, d }, &.{
        0.1, 0.5, -0.35, 0.2, -0.4, 0.6, 0.3, -0.7,
    });
    defer k_tok.deinit();
    var v_tok = try Kv.fromSlice(&ctx, .{ t, kv_heads, d }, &.{
        -0.2, 0.4, 0.55, -0.5, 0.15, 0.35, -0.65, 0.1,
    });
    defer v_tok.deinit();

    var k_cat = try cart.layers[0].catK(&ctx, &k_tok);
    defer k_cat.deinit();
    var v_cat = try cart.layers[0].catV(&ctx, &v_tok);
    defer v_cat.deinit();
    var y = try q.groupedAttention(&ctx, &k_cat, &v_cat, gqa_map[0..], .attn, attn_scale, .{});
    defer y.deinit();
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var gk = (try cart.layers[0].k.grad(&ctx)).?;
    defer gk.deinit();
    var gv = (try cart.layers[0].v.grad(&ctx)).?;
    defer gv.deinit();
    var k_nonzero = false;
    for (gk.asRawTensor().dataConst()) |g| {
        if (g != 0) k_nonzero = true;
    }
    var v_nonzero = false;
    for (gv.asRawTensor().dataConst()) |g| {
        if (g != 0) v_nonzero = true;
    }
    try std.testing.expect(k_nonzero);
    try std.testing.expect(v_nonzero);
}

fn distillMeanLoss(ctx: *ExecContext, logits: *const Tensor(.{ .seq, .vocab })) !Tensor(.{}) {
    return cartridge.distillLoss(ctx, logits, .{
        // Two teacher entries share position 1: the duplicate-index gather
        // gradient accumulation is on the checked path.
        .positions = &.{ 1, 1, 2, 3 },
        .tokens = &.{ 0, 2, 4, 1 },
        .logprobs = &.{ -0.5108256, -1.0498221, -0.1053605, 0.0 },
    }, .{});
}

test "gradcheck validates the distillation loss VJP (duplicate positions)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var logits = try Tensor(.{ .seq, .vocab }).variableFromSlice(&ctx, .{ 3, 5 }, &.{
        0.2,  -0.4, 0.5,  0.1,  -0.3,
        0.7,  0.05, -0.6, 0.4,  0.3,
        -0.2, -0.1, 0.6,  -0.5, 0.15,
    });
    defer logits.deinit();

    const result = try fucina.gradcheck(&ctx, distillMeanLoss, .{&logits}, .{});
    try std.testing.expectEqual(@as(usize, 15), result.checked);
}

test "distillation loss reproduces cross-entropy for a full-mass single entry" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();
    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);

    const values = [_]f32{
        0.2,  -0.4, 0.5,  0.1,
        0.7,  0.05, -0.6, 0.4,
        -0.2, -0.1, 0.6,  -0.5,
    };
    var logits = try Tensor(.{ .seq, .vocab }).fromSlice(&ctx, .{ 3, 4 }, &values);
    defer logits.deinit();

    // Teacher puts all mass on token 2 at position 2 (predicted from row 1):
    // the loss is exactly -log_softmax(row 1)[2].
    var loss = try cartridge.distillLoss(&ctx, &logits, .{
        .positions = &.{2},
        .tokens = &.{2},
        .logprobs = &.{0.0},
    }, .{});
    defer loss.deinit();

    const logit_row = values[4..8];
    var max: f32 = logit_row[0];
    for (logit_row) |x| max = @max(max, x);
    var sum_exp: f32 = 0;
    for (logit_row) |x| sum_exp += @exp(x - max);
    const expected = -(logit_row[2] - max - @log(sum_exp));
    try std.testing.expectApproxEqAbs(expected, (try loss.dataConst())[0], 1e-6);

    // .sum over n entries = n * .mean; loss_scale multiplies through.
    var sum_loss = try cartridge.distillLoss(&ctx, &logits, .{
        .positions = &.{ 2, 2 },
        .tokens = &.{ 2, 2 },
        .logprobs = &.{ 0.0, 0.0 },
    }, .{ .reduction = .sum, .loss_scale = 0.5 });
    defer sum_loss.deinit();
    try std.testing.expectApproxEqAbs(expected, (try sum_loss.dataConst())[0], 1e-6);
}

test "distillation loss rejects malformed targets" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var logits = try Tensor(.{ .seq, .vocab }).fromSlice(&ctx, .{ 2, 3 }, &.{
        0.2, -0.4, 0.5, 0.1, -0.3, 0.7,
    });
    defer logits.deinit();

    // The composite demands an open exec scope before anything else.
    try std.testing.expectError(cartridge.Error.ExecScopeRequired, cartridge.distillLoss(&ctx, &logits, .{
        .positions = &.{1},
        .tokens = &.{1},
        .logprobs = &.{0.0},
    }, .{}));

    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);

    // Position 0 has no preceding logits row.
    try std.testing.expectError(cartridge.Error.InvalidTargets, cartridge.distillLoss(&ctx, &logits, .{
        .positions = &.{0},
        .tokens = &.{1},
        .logprobs = &.{0.0},
    }, .{}));
    // Position past the sequence.
    try std.testing.expectError(cartridge.Error.InvalidTargets, cartridge.distillLoss(&ctx, &logits, .{
        .positions = &.{3},
        .tokens = &.{1},
        .logprobs = &.{0.0},
    }, .{}));
    // Token outside the vocabulary.
    try std.testing.expectError(cartridge.Error.InvalidTargets, cartridge.distillLoss(&ctx, &logits, .{
        .positions = &.{1},
        .tokens = &.{3},
        .logprobs = &.{0.0},
    }, .{}));
    // Length mismatch and empty targets.
    try std.testing.expectError(cartridge.Error.InvalidTargets, cartridge.distillLoss(&ctx, &logits, .{
        .positions = &.{ 1, 2 },
        .tokens = &.{1},
        .logprobs = &.{ 0.0, 0.0 },
    }, .{}));
    try std.testing.expectError(cartridge.Error.InvalidTargets, cartridge.distillLoss(&ctx, &logits, .{
        .positions = &.{},
        .tokens = &.{},
        .logprobs = &.{},
    }, .{}));
}

test "cartridge init validates geometry" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const row = [_]f32{ 1, 2, 3, 4 };
    // frozen_prefix >= p leaves nothing to train.
    try std.testing.expectError(cartridge.Error.InvalidCartridge, cartridge.Cartridge.initFromRows(
        &ctx,
        allocator,
        1,
        1,
        2,
        2,
        &.{&row},
        &.{&row},
    ));
    // Row payload does not match p * kv_heads * head_dim.
    try std.testing.expectError(cartridge.Error.InvalidCartridge, cartridge.Cartridge.initFromRows(
        &ctx,
        allocator,
        0,
        2,
        2,
        2,
        &.{&row},
        &.{&row},
    ));

    // initRandom builds sink + trainable rows for every layer.
    var cart = try cartridge.Cartridge.initRandom(&ctx, allocator, 2, 1, 3, 2, 2, 42, 0.02);
    defer cart.deinit();
    try std.testing.expectEqual(@as(usize, 2), cart.layers.len);
    try std.testing.expectEqual(@as(usize, 2), cart.layers[0].k.dim(.seq));
    try std.testing.expectEqual(@as(usize, 1), cart.layers[0].k_sink.?.dim(.seq));
    try std.testing.expect(cart.layers[1].v.requiresGrad());
}

test "targets from core topK/logsumexp match the reference host scan" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const rows = 5;
    const vocab = 4099;
    const top_k = 20;
    var prng = std.Random.DefaultPrng.init(123);
    const rand = prng.random();
    const data = try allocator.alloc(f32, rows * vocab);
    defer allocator.free(data);
    for (data) |*x| x.* = rand.floatNorm(f32) * 3.0;

    // Reference: the host scan over each full vocab row.
    var ref = cartridge.TargetsBuilder.init(allocator);
    defer ref.deinit();
    for (0..rows) |r| try ref.appendRow(r + 1, data[r * vocab ..][0..vocab], top_k, 0.99);

    // Tensor path: core topK selection + logsumexp partition, host cutoff.
    var logits = try Tensor(.{ .seq, .vocab }).fromSlice(&ctx, .{ rows, vocab }, data);
    defer logits.deinit();
    var log_z = try logits.logsumexp(&ctx, .vocab);
    defer log_z.deinit();
    var top = try logits.topK(&ctx, .vocab, top_k, .top);
    defer top.deinit();
    const top_values = try top.values.dataConst();
    const top_indices = try top.indices.dataConst();
    const log_z_data = try log_z.dataConst();
    var got = cartridge.TargetsBuilder.init(allocator);
    defer got.deinit();
    for (0..rows) |r| {
        try got.appendTopKRow(
            r + 1,
            top_values[r * top_k ..][0..top_k],
            top_indices[r * top_k ..][0..top_k],
            log_z_data[r],
            0.99,
        );
    }

    // Selection must be identical (same descending order, same
    // lowest-index tie-break); logprobs agree up to the core reduction's
    // summation order.
    const want = ref.targets();
    const have = got.targets();
    try std.testing.expectEqualSlices(usize, want.positions, have.positions);
    try std.testing.expectEqualSlices(usize, want.tokens, have.tokens);
    try std.testing.expectEqual(want.logprobs.len, have.logprobs.len);
    for (want.logprobs, have.logprobs) |a, b| {
        try std.testing.expectApproxEqAbs(a, b, 2e-5);
    }
}

test "draft reference roundtrips through the safetensors dict as i64" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var cart = try cartridge.Cartridge.initRandom(&ctx, allocator, 2, 1, 3, 2, 2, 42, 0.02);
    defer cart.deinit();

    // Without a reference nothing is persisted; empty references are invalid.
    try std.testing.expect(cart.draft_reference == null);
    try std.testing.expectError(cartridge.Error.InvalidCartridge, cart.setDraftReference(&ctx, &.{}));

    const corpus = [_]usize{ 7, 151935, 0, 42, 7, 7 };
    try cart.setDraftReference(&ctx, &corpus);
    // The reference is set-once: the artifact carries exactly one corpus.
    try std.testing.expectError(cartridge.Error.InvalidCartridge, cart.setDraftReference(&ctx, &corpus));

    var buf: [64 * 1024]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try cart.saveState(&writer);
    const written = writer.buffered();

    // Geometry-blind reload recovers the rows AND the embedded corpus ids.
    var rebuilt = try cartridge.Cartridge.initFromStateDict(&ctx, allocator, written);
    defer rebuilt.deinit();
    try std.testing.expectEqual(cart.p, rebuilt.p);
    try std.testing.expectEqualSlices(usize, &corpus, rebuilt.draft_reference.?);
    for (cart.layers, rebuilt.layers) |*want, *got| {
        try std.testing.expectEqualSlices(f32, try want.k.dataConst(), try got.k.dataConst());
        try std.testing.expectEqualSlices(f32, try want.v.dataConst(), try got.v.dataConst());
    }
}

test "targets builder keeps descending top-k until the mass threshold" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var builder = cartridge.TargetsBuilder.init(gpa.allocator());
    defer builder.deinit();

    // Softmax of these logits is exactly {0.5, 0.25, 0.15, 0.10} (shifted
    // by a constant to exercise the stable log-softmax path).
    const row = [_]f32{ @log(0.5) + 3, @log(0.25) + 3, @log(0.15) + 3, @log(0.10) + 3 };

    // Mass 0.85: 0.5 + 0.25 = 0.75 < 0.85, the crossing 0.15 entry is kept.
    try builder.appendRow(4, &row, 4, 0.85);
    var t = builder.targets();
    try std.testing.expectEqualSlices(usize, &.{ 4, 4, 4 }, t.positions);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 2 }, t.tokens);
    try std.testing.expectApproxEqAbs(@log(0.5), t.logprobs[0], 1e-6);
    try std.testing.expectApproxEqAbs(@log(0.25), t.logprobs[1], 1e-6);
    try std.testing.expectApproxEqAbs(@log(0.15), t.logprobs[2], 1e-6);

    // top_k caps before the mass threshold is reached.
    try builder.appendRow(5, &row, 2, 0.99);
    t = builder.targets();
    try std.testing.expectEqual(@as(usize, 5), t.positions.len);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1 }, t.tokens[3..]);
}
