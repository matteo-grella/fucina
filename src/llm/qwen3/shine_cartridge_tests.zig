//! SHINE cartridge-readout gates. No PyTorch reference exists for this
//! readout (it is Fucina's extension of the SHINE generator), so the
//! evidence is structural instead of golden: the budget identity, bitwise
//! parity between the inference slicer and the trainer's graph views, a
//! standard-cartridge save/reload round trip, and finite-difference
//! gradient checks through the full lossCartridge chain (generated prefix
//! -> qwen3 cartridge forward -> CE), the `--verify-grads` discipline.
const std = @import("std");
const fucina = @import("fucina");
const qwen3 = @import("model.zig");
const shine = @import("shine.zig");
const shine_train = @import("shine_train.zig");
const qwen3_train = @import("train.zig");
const cartridge = @import("../cartridge.zig");
const weights = @import("fucina").weights;

const ExecContext = fucina.ExecContext;

const base = qwen3.Config{
    .vocab_size = 64,
    .hidden_size = 32,
    .intermediate_size = 48,
    .num_layers = 2,
    .num_attention_heads = 2,
    .num_key_value_heads = 1,
    .head_dim = 16,
    .rms_norm_eps = 1e-6,
    .rope_theta = 10_000,
};
// kv_dim = 16; rows = 2 -> M * H = 2 * 2 * 16 = 64 -> M = 2.
// scale = 1 (not the paper's 0.001): the FD gate needs the generated
// prefix to influence the loss well above f32 resolution.
const sh_config = shine.Config{
    .hidden_size = 32,
    .num_layers = 2,
    .num_mem_token = 2,
    .metalora_r = 2,
    .lora_r = 1,
    .scale = 1.0,
    .m2p_layers = 2,
    .m2p_heads = 2,
    .m2p_ffn = 32,
    .m2p_eps = 1e-5,
    .layer_transformer_first = true,
    .cartridge_rows = 2,
};

fn fillNoise(values: []f32, seed: u32) void {
    var state: u32 = seed | 1;
    for (values) |*v| {
        state = state *% 1664525 +% 1013904223;
        v.* = (@as(f32, @floatFromInt(state >> 9)) / (1 << 23) - 0.5) * 0.2;
    }
}

fn noiseSlice(allocator: std.mem.Allocator, len: usize, seed: u32) ![]f32 {
    const values = try allocator.alloc(f32, len);
    fillNoise(values, seed);
    return values;
}

fn linearNoise(ctx: *ExecContext, allocator: std.mem.Allocator, rows: usize, cols: usize, seed: u32) !weights.LinearWeight {
    const values = try noiseSlice(allocator, rows * cols, seed);
    defer allocator.free(values);
    const w = try weights.WeightF32.fromSlice(ctx, .{ rows, cols }, values);
    return .{ .f32 = w };
}

fn vectorOnes(ctx: *ExecContext, allocator: std.mem.Allocator, comptime tag: @TypeOf(.tag), len: usize) !fucina.Tensor(.{tag}) {
    const values = try allocator.alloc(f32, len);
    defer allocator.free(values);
    @memset(values, 1);
    return fucina.Tensor(.{tag}).fromSlice(ctx, .{len}, values);
}

fn buildModel(ctx: *ExecContext, allocator: std.mem.Allocator) !qwen3.Model {
    var token_embedding = try linearNoise(ctx, allocator, base.vocab_size, base.hidden_size, 11);
    errdefer token_embedding.deinit();
    var output_norm = try vectorOnes(ctx, allocator, .embed, base.hidden_size);
    errdefer output_norm.deinit();
    var output = try linearNoise(ctx, allocator, base.vocab_size, base.hidden_size, 13);
    errdefer output.deinit();

    const kv_head_for_head = try allocator.alloc(usize, base.num_attention_heads);
    errdefer allocator.free(kv_head_for_head);
    const heads_per_kv = base.num_attention_heads / base.num_key_value_heads;
    for (kv_head_for_head, 0..) |*kv_head, head_i| kv_head.* = head_i / heads_per_kv;

    const layers = try allocator.alloc(qwen3.Layer, base.num_layers);
    errdefer allocator.free(layers);
    var built: usize = 0;
    errdefer for (layers[0..built]) |*layer| layer.deinit();
    for (layers, 0..) |*layer, i| {
        const seed: u32 = @intCast(100 + i * 20);
        const q_dim = base.qProjectionDim();
        const kv_dim = base.kvProjectionDim();
        var q_proj = try linearNoise(ctx, allocator, q_dim, base.hidden_size, seed + 1);
        errdefer q_proj.deinit();
        var k_proj = try linearNoise(ctx, allocator, kv_dim, base.hidden_size, seed + 2);
        errdefer k_proj.deinit();
        var v_proj = try linearNoise(ctx, allocator, kv_dim, base.hidden_size, seed + 3);
        errdefer v_proj.deinit();
        var o_proj = try linearNoise(ctx, allocator, base.hidden_size, q_dim, seed + 4);
        errdefer o_proj.deinit();
        var gate_proj = try linearNoise(ctx, allocator, base.intermediate_size, base.hidden_size, seed + 5);
        errdefer gate_proj.deinit();
        var up_proj = try linearNoise(ctx, allocator, base.intermediate_size, base.hidden_size, seed + 6);
        errdefer up_proj.deinit();
        var down_proj = try linearNoise(ctx, allocator, base.hidden_size, base.intermediate_size, seed + 7);
        errdefer down_proj.deinit();
        var attn_norm = try vectorOnes(ctx, allocator, .embed, base.hidden_size);
        errdefer attn_norm.deinit();
        var ffn_norm = try vectorOnes(ctx, allocator, .embed, base.hidden_size);
        errdefer ffn_norm.deinit();
        var q_norm = try vectorOnes(ctx, allocator, .d, base.head_dim);
        errdefer q_norm.deinit();
        var k_norm = try vectorOnes(ctx, allocator, .d, base.head_dim);
        errdefer k_norm.deinit();
        layer.* = .{
            .attn_norm = attn_norm,
            .q_norm = q_norm,
            .k_norm = k_norm,
            .ffn_norm = ffn_norm,
            .attn_proj = .{ .separate = .{ .q_proj = q_proj, .k_proj = k_proj, .v_proj = v_proj } },
            .o_proj = o_proj,
            .ffn = .{ .dense = .{ .input_proj = .{ .separate = .{ .gate_proj = gate_proj, .up_proj = up_proj } }, .down_proj = down_proj } },
        };
        built += 1;
    }

    return .{
        .allocator = allocator,
        .config = base,
        .token_embedding = token_embedding,
        .output_norm = output_norm,
        .output = output,
        .layers = layers,
        .kv_head_for_head = kv_head_for_head,
    };
}

/// Deterministic trainer from noise; `perturb` adds `eps` to ONE strongly
/// coupled parameter (last M2P block's output bias, entry 0 — it feeds the
/// generated rows directly) — the finite-difference seam.
fn buildTrainer(ctx: *ExecContext, allocator: std.mem.Allocator, model: *const qwen3.Model, perturb: f32) !shine_train.ShineTrainer {
    const layers = try allocator.alloc(shine.LayerLora, base.num_layers);
    var built: usize = 0;
    errdefer {
        for (layers[0..built]) |*layer| layer.deinit();
        allocator.free(layers);
    }
    for (layers, 0..) |*layer, i| {
        var pairs: [7]shine.LoraPair = undefined;
        var pair_built: usize = 0;
        errdefer for (pairs[0..pair_built]) |*pair| pair.deinit();
        inline for (shine.modules, 0..) |_, gi| {
            const dims = shine.moduleDims(base, shine.modules[gi]);
            const a_values = try noiseSlice(allocator, dims[0] * sh_config.metalora_r, @intCast(1000 + i * 50 + gi * 2));
            defer allocator.free(a_values);
            const b_values = try noiseSlice(allocator, sh_config.metalora_r * dims[1], @intCast(1001 + i * 50 + gi * 2));
            defer allocator.free(b_values);
            pairs[gi] = try shine_train.pairVariables(ctx, dims[0], sh_config.metalora_r, dims[1], a_values, b_values);
            pair_built = gi + 1;
        }
        layer.* = .{ .q = pairs[0], .k = pairs[1], .v = pairs[2], .o = pairs[3], .gate = pairs[4], .up = pairs[5], .down = pairs[6] };
        built += 1;
    }
    var metalora = shine.LoraSet{ .allocator = allocator, .layers = layers };
    errdefer metalora.deinit();

    const m2p = try allocator.alloc(shine.M2pBlock, sh_config.m2p_layers);
    var m2p_built: usize = 0;
    errdefer {
        for (m2p[0..m2p_built]) |*block| block.deinit(allocator);
        allocator.free(m2p);
    }
    const h = sh_config.hidden_size;
    const ff = sh_config.m2p_ffn;
    for (m2p, 0..) |*block, i| {
        const lens = [_]usize{ 3 * h * h, 3 * h, h * h, h, ff * h, ff, h * ff, h, h, h, h, h };
        var bufs: [12][]f32 = undefined;
        inline for (0..12) |pi| bufs[pi] = try noiseSlice(allocator, lens[pi], @intCast(7000 + i * 20 + pi));
        defer inline for (0..12) |pi| allocator.free(bufs[pi]);
        if (i == sh_config.m2p_layers - 1) bufs[3][0] += perturb; // out_b[0]
        block.* = try shine_train.m2pBlockVariables(ctx, sh_config, .{
            .in_w = bufs[0],
            .in_b = bufs[1],
            .out_w = bufs[2],
            .out_b = bufs[3],
            .ff1_w = bufs[4],
            .ff1_b = bufs[5],
            .ff2_w = bufs[6],
            .ff2_b = bufs[7],
            .norm1_w = bufs[8],
            .norm1_b = bufs[9],
            .norm2_w = bufs[10],
            .norm2_b = bufs[11],
        });
        m2p_built = i + 1;
    }

    const lp = try noiseSlice(allocator, base.num_layers * h, 31337);
    defer allocator.free(lp);
    const tp = try noiseSlice(allocator, sh_config.num_mem_token * h, 31338);
    defer allocator.free(tp);
    var layer_pe = try fucina.Tensor(.{ .layer, .embed }).variableFromSlice(ctx, .{ base.num_layers, h }, lp);
    errdefer layer_pe.deinit();
    const token_pe = try fucina.Tensor(.{ .mem, .embed }).variableFromSlice(ctx, .{ sh_config.num_mem_token, h }, tp);
    return .{
        .allocator = allocator,
        .model = model,
        .config = sh_config,
        .metalora = metalora,
        .m2p = m2p,
        .layer_pe = layer_pe,
        .token_pe = token_pe,
    };
}

test "cartridge readout: budget identity validates and rejects" {
    try sh_config.validate(base);

    var wrong = sh_config;
    wrong.num_mem_token = 3; // 3 * 32 != 2 * 2 * 16
    try std.testing.expectError(shine.Error.InvalidShineConfig, wrong.validate(base));

    var lora_mode = sh_config;
    lora_mode.cartridge_rows = 0; // falls back to the LoRA identity
    try std.testing.expectError(shine.Error.InvalidShineConfig, lora_mode.validate(base));
}

test "cartridge readout: inference slicer and trainer views agree bitwise; round-trip through the standard artifact" {
    const allocator = std.heap.smp_allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // A deterministic generated block stands in for the M2P output.
    const block_len = base.num_layers * sh_config.num_mem_token * sh_config.hidden_size;
    const block_values = try noiseSlice(allocator, block_len, 424242);
    defer allocator.free(block_values);
    var plain = try fucina.Tensor(.{ .layer, .mem, .embed }).fromSlice(&ctx, .{ base.num_layers, sh_config.num_mem_token, sh_config.hidden_size }, block_values);
    defer plain.deinit();

    var cart = try shine.sliceCartridge(&ctx, allocator, base, sh_config, &plain);
    defer cart.deinit();
    try std.testing.expectEqual(sh_config.cartridge_rows, cart.p);
    try std.testing.expectEqual(@as(usize, 0), cart.frozen_prefix);
    try std.testing.expectEqual(base.num_layers, cart.layers.len);
    try std.testing.expectEqual(base.num_key_value_heads, cart.kv_heads);
    try std.testing.expectEqual(base.head_dim, cart.head_dim);

    // Trainer views over the same block: bitwise the same rows.
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        var views = try shine_train.sliceCartridgeViews(&ctx, base, sh_config, &plain);
        defer views.deinit();
        for (cart.layers, views.layers) |*lc, *lv| {
            try std.testing.expectEqualSlices(f32, try lc.k.dataConst(), try lv.k.dataConst());
            try std.testing.expectEqualSlices(f32, try lc.v.dataConst(), try lv.v.dataConst());
        }
    }

    // Standard-cartridge round trip: saveState -> initFromStateDict.
    var buf: [1 << 16]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try cart.saveState(&writer);
    var rebuilt = try cartridge.Cartridge.initFromStateDict(&ctx, allocator, writer.buffered());
    defer rebuilt.deinit();
    try std.testing.expectEqual(cart.p, rebuilt.p);
    try std.testing.expectEqual(cart.layers.len, rebuilt.layers.len);
    for (cart.layers, rebuilt.layers) |*lc, *lr| {
        try std.testing.expectEqualSlices(f32, try lc.k.dataConst(), try lr.k.dataConst());
        try std.testing.expectEqualSlices(f32, try lc.v.dataConst(), try lr.v.dataConst());
    }
}

test "lossCartridge: gradients reach every leaf and match finite differences" {
    const allocator = std.heap.smp_allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try buildModel(&ctx, allocator);
    defer model.deinit();

    const evidence = [_]usize{ 3, 17, 42, 9, 28 };
    const inputs = [_]usize{ 5, 12, 33, 7, 19, 44 };
    const labels = [_]usize{ 12, 33, 7, 19, 44, 2 };

    var trainer = try buildTrainer(&ctx, allocator, &model, 0);
    defer trainer.deinit();

    var loss0: f32 = 0;
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const loss = try trainer.lossCartridge(&ctx, &evidence, &inputs, &labels);
        loss0 = try loss.item();
        try std.testing.expect(std.math.isFinite(loss0));
        try loss.backward(&ctx);
    }

    // Every leaf class received a finite gradient through the chain
    // (generated prefix -> cartridge forward -> CE).
    var grad_a0: f32 = 0;
    {
        var ga = (try trainer.metalora.layers[0].q.a.grad(&ctx)) orelse return error.MissingGrad;
        defer ga.deinit();
        var norm: f64 = 0;
        for (try ga.dataConst()) |g| {
            try std.testing.expect(std.math.isFinite(g));
            norm += @as(f64, g) * g;
        }
        try std.testing.expect(norm > 0);
    }
    {
        var gb = (try trainer.m2p[sh_config.m2p_layers - 1].out_b.grad(&ctx)) orelse return error.MissingGrad;
        defer gb.deinit();
        grad_a0 = (try gb.dataConst())[0];
    }
    inline for (.{ "in_w", "ff1_w" }) |field| {
        var grad = (try @field(trainer.m2p[0], field).grad(&ctx)) orelse return error.MissingGrad;
        defer grad.deinit();
        for (try grad.dataConst()) |g| try std.testing.expect(std.math.isFinite(g));
    }
    {
        var grad = (try trainer.layer_pe.grad(&ctx)) orelse return error.MissingGrad;
        defer grad.deinit();
        for (try grad.dataConst()) |g| try std.testing.expect(std.math.isFinite(g));
    }

    // Central finite difference on the perturbed metalora entry.
    const eps: f32 = 1e-2;
    var loss_plus: f32 = 0;
    var loss_minus: f32 = 0;
    {
        var plus = try buildTrainer(&ctx, allocator, &model, eps);
        defer plus.deinit();
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const loss = try plus.lossCartridge(&ctx, &evidence, &inputs, &labels);
        loss_plus = try loss.item();
    }
    {
        var minus = try buildTrainer(&ctx, allocator, &model, -eps);
        defer minus.deinit();
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const loss = try minus.lossCartridge(&ctx, &evidence, &inputs, &labels);
        loss_minus = try loss.item();
    }
    const fd = (loss_plus - loss_minus) / (2 * eps);
    std.debug.print("shine cartridge FD check: autograd {d} fd {d} (loss {d})\n", .{ grad_a0, fd, loss0 });
    // The gradient must be measurably nonzero AND match the secant.
    try std.testing.expect(@abs(grad_a0) > 1e-6);
    try std.testing.expect(@abs(fd - grad_a0) / @max(@abs(fd), @abs(grad_a0)) < 0.05);
}

test "initRandom + AdamW loop: deterministic init, loss decreases on one triple" {
    const allocator = std.heap.smp_allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try buildModel(&ctx, allocator);
    defer model.deinit();

    var trainer = try shine_train.ShineTrainer.initRandom(&ctx, allocator, &model, sh_config, 7);
    defer trainer.deinit();
    var twin = try shine_train.ShineTrainer.initRandom(&ctx, allocator, &model, sh_config, 7);
    defer twin.deinit();
    try std.testing.expectEqualSlices(
        f32,
        try trainer.metalora.layers[0].q.a.dataConst(),
        try twin.metalora.layers[0].q.a.dataConst(),
    );
    try std.testing.expectEqualSlices(f32, try trainer.m2p[0].in_w.dataConst(), try twin.m2p[0].in_w.dataConst());

    var opt = fucina.optim.AdamW.init(allocator, .{ .lr = 5e-3, .weight_decay = 0 });
    defer opt.deinit();
    try trainer.registerAllParams(&opt);
    var set = fucina.optim.OptimizerSet.init(allocator);
    defer set.deinit();
    try set.add(&opt);

    const evidence = [_]usize{ 3, 17, 42, 9, 28 };
    const inputs = [_]usize{ 5, 12, 33, 7, 19, 44 };
    const labels = [_]usize{ 12, 33, 7, 19, 44, 2 };

    var first: f32 = 0;
    var last: f32 = 0;
    for (0..12) |step_i| {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const loss = try trainer.lossCartridge(&ctx, &evidence, &inputs, &labels);
        try loss.backward(&ctx);
        const value = try loss.item();
        _ = try set.clipGradNorm(&ctx, 1.0);
        try set.step(&ctx);
        set.zeroGrad();
        if (step_i == 0) first = value;
        last = value;
    }
    trainer.freeTransient();
    std.debug.print("shine cartridge AdamW: loss {d} -> {d} over 12 steps\n", .{ first, last });
    try std.testing.expect(last < first);
}
