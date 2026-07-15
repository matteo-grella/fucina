//! Behavioral tests for the Qwen3 LoRA trainer (`qwen3_train.zig`) on a
//! synthetic tiny dense model (no GGUF dependency): inference parity at LoRA
//! init, loss descent, frozen-weight immutability, checkpoint bitwise parity,
//! full-stack finite-difference gradcheck, label masking, adapter
//! persistence, and MoE rejection.
const std = @import("std");
const fucina = @import("fucina");
const qwen3 = @import("model.zig");
const qwen3_train = @import("train.zig");
const weights = @import("../weights.zig");

const ExecContext = fucina.ExecContext;
const optim = fucina.optim;
const rng = fucina.rng;
const Layer = qwen3_train.ModelLayer;

pub const tiny_config = qwen3.Config{
    .vocab_size = 64,
    .hidden_size = 32,
    .intermediate_size = 64,
    .num_layers = 2,
    .num_attention_heads = 2,
    .num_key_value_heads = 1,
    .head_dim = 16,
    .rms_norm_eps = 1e-6,
    .rope_theta = 10_000,
};

const tiny_lora = fucina.lora.Config{ .rank = 4, .alpha = 8 };

/// A fixed batch: inputs predict the next token of `tokens`.
const batch_tokens = [_]usize{ 5, 9, 13, 2, 33, 60, 21, 8, 42, 17, 3, 28 };
const batch_inputs = batch_tokens[0 .. batch_tokens.len - 1];
const batch_labels = batch_tokens[1..];

/// Long fixed batch (seq 64): at q_seq >= 48 (exec's attention_tiled_min_q_seq)
/// with head_dim 16 <= 256 (attention_tile_max_d), training forwards route
/// attention through the query-tiled kernel — the tiled-forward/3-pass-backward
/// mix real training runs. The short batches above stay on the per-query
/// kernels, so without this batch that mix would be untested.
const long_seq = 64;
const long_tokens = blk: {
    var t: [long_seq + 1]usize = undefined;
    for (&t, 0..) |*tok, i| tok.* = (i * 7 + 3) % tiny_config.vocab_size;
    break :blk t;
};
const long_inputs = long_tokens[0..long_seq];
const long_labels = long_tokens[1..];

fn randLinear(ctx: *ExecContext, seed: u64, out_dim: usize, in_dim: usize, bound: f32) !weights.LinearWeight {
    const values = try ctx.allocator.alloc(f32, out_dim * in_dim);
    defer ctx.allocator.free(values);
    rng.uniformFill(seed, values, -bound, bound);
    return .{ .f32 = try weights.WeightF32.fromSlice(ctx, .{ out_dim, in_dim }, values) };
}

/// Replace an f32 LinearWeight with the resident-bf16 arm holding the same
/// values rounded to bf16 (what `LinearWeight.load` produces for bf16 GGUFs).
fn toBf16Weight(ctx: *ExecContext, w: *weights.LinearWeight) !void {
    var converted = switch (w.*) {
        .f32 => |*t| try t.to(ctx, .bf16),
        else => unreachable, // tests only convert the synthetic f32 weights
    };
    errdefer converted.deinit();
    w.deinit();
    w.* = .{ .bf16 = converted };
}

fn randVector(ctx: *ExecContext, seed: u64, comptime tag: @TypeOf(.tag), len: usize) !fucina.Tensor(.{tag}) {
    const values = try ctx.allocator.alloc(f32, len);
    defer ctx.allocator.free(values);
    rng.uniformFill(seed, values, 0.8, 1.2); // norm-weight-shaped: near one
    return fucina.Tensor(.{tag}).fromSlice(ctx, .{len}, values);
}

/// Field-wise teardown for error paths in `buildTinyModel` (Layer's own
/// deinit is private to qwen3.zig; fields are reachable, decls are not).
fn destroyLayer(layer: *Layer) void {
    switch (layer.ffn) {
        .dense => |*dense| {
            dense.down_proj.deinit();
            switch (dense.input_proj) {
                .separate => |*sep| {
                    sep.up_proj.deinit();
                    sep.gate_proj.deinit();
                },
                .fused => |*w| w.deinit(),
            }
        },
        .moe => unreachable, // tests build dense layers only
    }
    layer.o_proj.deinit();
    switch (layer.attn_proj) {
        .separate => |*sep| {
            sep.v_proj.deinit();
            sep.k_proj.deinit();
            sep.q_proj.deinit();
        },
        .fused => |*w| w.deinit(),
    }
    layer.ffn_norm.deinit();
    layer.k_norm.deinit();
    layer.q_norm.deinit();
    layer.attn_norm.deinit();
    layer.* = undefined;
}

fn buildTinyLayer(ctx: *ExecContext, cfg: qwen3.Config, seed: u64) !Layer {
    const q_dim = cfg.num_attention_heads * cfg.head_dim;
    const kv_dim = cfg.num_key_value_heads * cfg.head_dim;

    var attn_norm = try randVector(ctx, rng.at(seed, 0), .embed, cfg.hidden_size);
    errdefer attn_norm.deinit();
    var q_norm = try randVector(ctx, rng.at(seed, 1), .d, cfg.head_dim);
    errdefer q_norm.deinit();
    var k_norm = try randVector(ctx, rng.at(seed, 2), .d, cfg.head_dim);
    errdefer k_norm.deinit();
    var ffn_norm = try randVector(ctx, rng.at(seed, 3), .embed, cfg.hidden_size);
    errdefer ffn_norm.deinit();

    var q_proj = try randLinear(ctx, rng.at(seed, 4), q_dim, cfg.hidden_size, 0.3);
    errdefer q_proj.deinit();
    var k_proj = try randLinear(ctx, rng.at(seed, 5), kv_dim, cfg.hidden_size, 0.3);
    errdefer k_proj.deinit();
    var v_proj = try randLinear(ctx, rng.at(seed, 6), kv_dim, cfg.hidden_size, 0.3);
    errdefer v_proj.deinit();
    var o_proj = try randLinear(ctx, rng.at(seed, 7), cfg.hidden_size, q_dim, 0.3);
    errdefer o_proj.deinit();

    var gate_proj = try randLinear(ctx, rng.at(seed, 8), cfg.intermediate_size, cfg.hidden_size, 0.3);
    errdefer gate_proj.deinit();
    var up_proj = try randLinear(ctx, rng.at(seed, 9), cfg.intermediate_size, cfg.hidden_size, 0.3);
    errdefer up_proj.deinit();
    var down_proj = try randLinear(ctx, rng.at(seed, 10), cfg.hidden_size, cfg.intermediate_size, 0.3);
    errdefer down_proj.deinit();

    return .{
        .attn_norm = attn_norm,
        .q_norm = q_norm,
        .k_norm = k_norm,
        .ffn_norm = ffn_norm,
        .attn_proj = .{ .separate = .{ .q_proj = q_proj, .k_proj = k_proj, .v_proj = v_proj } },
        .o_proj = o_proj,
        .ffn = .{ .dense = .{
            .input_proj = .{ .separate = .{ .gate_proj = gate_proj, .up_proj = up_proj } },
            .down_proj = down_proj,
        } },
    };
}

/// Synthetic dense model with separate projections and seeded f32 weights;
/// tear down with the public `Model.deinit`. Pub: speculative.zig's
/// losslessness tests reuse this scaffolding (imports are memoized per file,
/// so the tests here still register exactly once).
pub fn buildTinyModel(ctx: *ExecContext, seed: u64) !qwen3.Model {
    return buildTinyModelPicked(ctx, tiny_config, seed, &.{ 0, 1 });
}

/// `buildTinyModel` with a config override (same seeded weight recipe):
/// the batch-decode tests need a q8_0-compatible `head_dim % 32 == 0`
/// geometry, and the general (non-paired-GQA) attention path needs
/// `num_attention_heads != 2 * num_key_value_heads`. Layer count comes
/// from `cfg.num_layers`.
pub fn buildTinyModelWithConfig(ctx: *ExecContext, cfg: qwen3.Config, seed: u64) !qwen3.Model {
    const indices = try ctx.allocator.alloc(usize, cfg.num_layers);
    defer ctx.allocator.free(indices);
    for (indices, 0..) |*x, i| x.* = i;
    return buildTinyModelPicked(ctx, cfg, seed, indices);
}

/// `buildTinyModel` whose layer stack is the PICKED layers of the seed-`seed`
/// model (same per-layer weight seeds `rng.at(seed, layer_i)`, same
/// embedding/output seeds) with `config.num_layers` matching — the manual
/// truncation reference for the `forwardHidden` layer-range tests.
fn buildTinyModelPicked(ctx: *ExecContext, cfg_in: qwen3.Config, seed: u64, layer_indices: []const usize) !qwen3.Model {
    var cfg = cfg_in;
    cfg.num_layers = layer_indices.len;
    const allocator = ctx.allocator;

    var token_embedding = try randLinear(ctx, rng.at(seed, 100), cfg.vocab_size, cfg.hidden_size, 0.5);
    errdefer token_embedding.deinit();
    var output_norm = try randVector(ctx, rng.at(seed, 101), .embed, cfg.hidden_size);
    errdefer output_norm.deinit();
    var output = try randLinear(ctx, rng.at(seed, 102), cfg.vocab_size, cfg.hidden_size, 0.5);
    errdefer output.deinit();

    const kv_head_for_head = try allocator.alloc(usize, cfg.num_attention_heads);
    errdefer allocator.free(kv_head_for_head);
    const heads_per_kv = cfg.num_attention_heads / cfg.num_key_value_heads;
    for (kv_head_for_head, 0..) |*kv_head, head_i| kv_head.* = head_i / heads_per_kv;

    const layers = try allocator.alloc(Layer, layer_indices.len);
    errdefer allocator.free(layers);
    var built: usize = 0;
    errdefer for (layers[0..built]) |*layer| destroyLayer(layer);
    for (layers, layer_indices) |*layer, layer_i| {
        layer.* = try buildTinyLayer(ctx, cfg, rng.at(seed, layer_i));
        built += 1;
    }

    return .{
        .allocator = allocator,
        .config = cfg,
        .token_embedding = token_embedding,
        .output_norm = output_norm,
        .output = output,
        .layers = layers,
        .kv_head_for_head = kv_head_for_head,
        .weight_mapping = null,
    };
}

const DefaultTrainer = qwen3_train.Trainer(.{});

fn lossStepOn(ctx: *ExecContext, trainer: anytype, opt: ?*optim.AdamW, inputs: []const usize, labels: []const usize) !f32 {
    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    const loss = try trainer.loss(ctx, inputs, labels);
    try loss.backward(ctx);
    if (opt) |o| {
        try o.step(ctx);
        o.zeroGrad();
    }
    return loss.item();
}

fn lossStep(ctx: *ExecContext, trainer: anytype, opt: ?*optim.AdamW) !f32 {
    return lossStepOn(ctx, trainer, opt, batch_inputs, batch_labels);
}

test "trainable forward matches inference at LoRA init (zero delta)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try buildTinyModel(&ctx, 0xBEEF);
    defer model.deinit();
    var trainer = try DefaultTrainer.init(&ctx, &model, tiny_lora, 1);
    defer trainer.deinit();

    const tokens = [_]usize{ 3, 17, 60, 5, 22, 41, 7, 0 };
    var ref = try model.forwardLastLogits(&ctx, &tokens);
    defer ref.deinit();
    var got = try trainer.evalLastLogits(&ctx, &tokens);
    defer got.deinit();

    const ref_values = try ref.dataConst();
    const got_values = try got.dataConst();
    try std.testing.expectEqual(ref_values.len, got_values.len);
    // Both paths route f32 weights through the same taggedDot kernels; the
    // only divergence is GEMM row-blocking (inference narrows the last layer
    // to the final query, training keeps the full sequence), so the match is
    // tight but not bitwise.
    for (ref_values, got_values) |e, a| try std.testing.expectApproxEqAbs(e, a, 1e-4);
}

test "loss decreases over AdamW steps on a fixed batch" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try buildTinyModel(&ctx, 0xF00D);
    defer model.deinit();
    var trainer = try DefaultTrainer.init(&ctx, &model, tiny_lora, 2);
    defer trainer.deinit();

    var opt = optim.AdamW.init(allocator, .{ .lr = 0.05, .weight_decay = 0 });
    defer opt.deinit();
    try trainer.registerAllParams(&opt);

    var first: f32 = 0;
    var last: f32 = 0;
    for (0..20) |step_i| {
        last = try lossStep(&ctx, &trainer, &opt);
        if (step_i == 0) first = last;
    }
    try std.testing.expect(last < first);
    try std.testing.expect(last < first - 0.1); // by a real margin, not noise
}

test "bf16 frozen base: trainable forward matches inference and loss decreases" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Tiny model with EVERY linear weight on the resident bf16 arm: the
    // embedding table (bf16 getRowsAs), all projections, and the output head
    // (both the inference linearSeq fast path and the trainer's
    // differentiable const-RHS bf16 dot).
    var model = try buildTinyModel(&ctx, 0xBF16);
    defer model.deinit();
    try toBf16Weight(&ctx, &model.token_embedding);
    try toBf16Weight(&ctx, &model.output);
    for (model.layers) |*layer| {
        const attn = &layer.attn_proj.separate;
        try toBf16Weight(&ctx, &attn.q_proj);
        try toBf16Weight(&ctx, &attn.k_proj);
        try toBf16Weight(&ctx, &attn.v_proj);
        try toBf16Weight(&ctx, &layer.o_proj);
        const dense = &layer.ffn.dense;
        const ffn_in = &dense.input_proj.separate;
        try toBf16Weight(&ctx, &ffn_in.gate_proj);
        try toBf16Weight(&ctx, &ffn_in.up_proj);
        try toBf16Weight(&ctx, &dense.down_proj);
    }

    var trainer = try DefaultTrainer.init(&ctx, &model, tiny_lora, 2);
    defer trainer.deinit();

    // At zero-delta init the trainable forward must match inference: both
    // route the bf16 weights through the same mixed f32 x bf16 kernel, the
    // only divergence is GEMM row-blocking (as in the f32 parity test above).
    const tokens = [_]usize{ 3, 17, 60, 5, 22, 41, 7, 0 };
    var ref = try model.forwardLastLogits(&ctx, &tokens);
    defer ref.deinit();
    var got = try trainer.evalLastLogits(&ctx, &tokens);
    defer got.deinit();
    const ref_values = try ref.dataConst();
    const got_values = try got.dataConst();
    try std.testing.expectEqual(ref_values.len, got_values.len);
    for (ref_values, got_values) |e, a| try std.testing.expectApproxEqAbs(e, a, 1e-4);

    // Gradients flow through the frozen bf16 base into the adapters: the
    // loss must actually descend under AdamW.
    var opt = optim.AdamW.init(allocator, .{ .lr = 0.05, .weight_decay = 0 });
    defer opt.deinit();
    try trainer.registerAllParams(&opt);

    var first: f32 = 0;
    var last: f32 = 0;
    for (0..20) |step_i| {
        last = try lossStep(&ctx, &trainer, &opt);
        if (step_i == 0) first = last;
    }
    try std.testing.expect(last < first - 0.1);
}

test "frozen weights stay bitwise unchanged; only adapters carry grads" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try buildTinyModel(&ctx, 0xACE);
    defer model.deinit();
    var trainer = try DefaultTrainer.init(&ctx, &model, tiny_lora, 3);
    defer trainer.deinit();

    // Snapshot representative frozen weights bitwise.
    const embed_data = try model.token_embedding.f32.dataConst();
    const q_proj_data = try model.layers[0].attn_proj.separate.q_proj.f32.dataConst();
    const down_data = try model.layers[1].ffn.dense.down_proj.f32.dataConst();
    const norm_data = try model.output_norm.dataConst();
    const embed_before = try allocator.dupe(f32, embed_data);
    defer allocator.free(embed_before);
    const q_proj_before = try allocator.dupe(f32, q_proj_data);
    defer allocator.free(q_proj_before);
    const down_before = try allocator.dupe(f32, down_data);
    defer allocator.free(down_before);
    const norm_before = try allocator.dupe(f32, norm_data);
    defer allocator.free(norm_before);

    var opt = optim.AdamW.init(allocator, .{ .lr = 0.05, .weight_decay = 0 });
    defer opt.deinit();
    try trainer.registerAllParams(&opt);

    // One backward without a step: only the adapters may carry grads. The
    // frozen weights are constants — no GradState exists for them at all.
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const loss = try trainer.loss(&ctx, batch_inputs, batch_labels);
        try loss.backward(&ctx);
    }
    try std.testing.expect(!model.token_embedding.f32.requiresGrad());
    try std.testing.expect(!model.layers[0].attn_proj.separate.q_proj.f32.requiresGrad());
    try std.testing.expect(!model.output.f32.requiresGrad());
    for (trainer.adapters) |*ads| {
        var ga = (try ads.q.a.grad(&ctx)) orelse return error.MissingGrad;
        defer ga.deinit();
        var gb = (try ads.q.b.grad(&ctx)) orelse return error.MissingGrad;
        defer gb.deinit();
        var gva = (try ads.v.a.grad(&ctx)) orelse return error.MissingGrad;
        defer gva.deinit();
        var gvb = (try ads.v.b.grad(&ctx)) orelse return error.MissingGrad;
        defer gvb.deinit();
    }
    opt.zeroGrad();

    // Train for real, then demand the frozen weights are bitwise untouched.
    for (0..3) |_| _ = try lossStep(&ctx, &trainer, &opt);
    try std.testing.expectEqualSlices(f32, embed_before, try model.token_embedding.f32.dataConst());
    try std.testing.expectEqualSlices(f32, q_proj_before, try model.layers[0].attn_proj.separate.q_proj.f32.dataConst());
    try std.testing.expectEqualSlices(f32, down_before, try model.layers[1].ffn.dense.down_proj.f32.dataConst());
    try std.testing.expectEqualSlices(f32, norm_before, try model.output_norm.dataConst());
}

test "checkpointed layers: loss and adapter grads bitwise equal to plain" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try buildTinyModel(&ctx, 0xCAFE);
    defer model.deinit();

    // Dropout on, so the seed replay in the recompute is actually exercised.
    const dropout_lora = fucina.lora.Config{ .rank = 4, .alpha = 8, .dropout_p = 0.2 };
    var plain = try DefaultTrainer.init(&ctx, &model, dropout_lora, 7);
    defer plain.deinit();
    var ckpt = try DefaultTrainer.init(&ctx, &model, dropout_lora, 7);
    defer ckpt.deinit();
    ckpt.checkpoint_layers = true;

    var plain_loss: f32 = 0;
    var ckpt_loss: f32 = 0;
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const loss = try plain.loss(&ctx, batch_inputs, batch_labels);
        try loss.backward(&ctx);
        plain_loss = try loss.item();
    }
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const loss = try ckpt.loss(&ctx, batch_inputs, batch_labels);
        try loss.backward(&ctx);
        ckpt_loss = try loss.item();
    }
    try std.testing.expectEqual(plain_loss, ckpt_loss);

    for (plain.adapters, ckpt.adapters) |*pads, *cads| {
        inline for (.{ "q", "v" }) |name| {
            var pa = (try @field(pads.*, name).a.grad(&ctx)) orelse return error.MissingGrad;
            defer pa.deinit();
            var ca = (try @field(cads.*, name).a.grad(&ctx)) orelse return error.MissingGrad;
            defer ca.deinit();
            try std.testing.expectEqualSlices(f32, try pa.dataConst(), try ca.dataConst());
            var pb = (try @field(pads.*, name).b.grad(&ctx)) orelse return error.MissingGrad;
            defer pb.deinit();
            var cb = (try @field(cads.*, name).b.grad(&ctx)) orelse return error.MissingGrad;
            defer cb.deinit();
            try std.testing.expectEqualSlices(f32, try pb.dataConst(), try cb.dataConst());
        }
    }
}

test "checkpointed layers at seq 64 (tiled attention): loss and adapter grads bitwise equal to plain" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try buildTinyModel(&ctx, 0x10AD);
    defer model.deinit();

    // Same contract as the seq-11 checkpoint test, but the seq-64 batch
    // routes the forward (and the checkpoint recompute) through the tiled
    // attention kernel while the backward stays on the 3-pass kernels.
    // Dropout on, so the seed replay in the recompute is exercised too.
    const dropout_lora = fucina.lora.Config{ .rank = 4, .alpha = 8, .dropout_p = 0.2 };
    var plain = try DefaultTrainer.init(&ctx, &model, dropout_lora, 23);
    defer plain.deinit();
    var ckpt = try DefaultTrainer.init(&ctx, &model, dropout_lora, 23);
    defer ckpt.deinit();
    ckpt.checkpoint_layers = true;

    var plain_loss: f32 = 0;
    var ckpt_loss: f32 = 0;
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const loss = try plain.loss(&ctx, long_inputs, long_labels);
        try loss.backward(&ctx);
        plain_loss = try loss.item();
    }
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const loss = try ckpt.loss(&ctx, long_inputs, long_labels);
        try loss.backward(&ctx);
        ckpt_loss = try loss.item();
    }
    try std.testing.expectEqual(plain_loss, ckpt_loss);

    for (plain.adapters, ckpt.adapters) |*pads, *cads| {
        inline for (.{ "q", "v" }) |name| {
            inline for (.{ "a", "b" }) |which| {
                var pg = (try @field(@field(pads.*, name), which).grad(&ctx)) orelse return error.MissingGrad;
                defer pg.deinit();
                var cg = (try @field(@field(cads.*, name), which).grad(&ctx)) orelse return error.MissingGrad;
                defer cg.deinit();
                try std.testing.expectEqualSlices(f32, try pg.dataConst(), try cg.dataConst());
            }
        }
    }
}

test "loss decreases over AdamW steps at seq 64 (tiled attention path)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try buildTinyModel(&ctx, 0x64F0);
    defer model.deinit();
    var trainer = try DefaultTrainer.init(&ctx, &model, tiny_lora, 8);
    defer trainer.deinit();

    var opt = optim.AdamW.init(allocator, .{ .lr = 0.05, .weight_decay = 0 });
    defer opt.deinit();
    try trainer.registerAllParams(&opt);

    var first: f32 = 0;
    var last: f32 = 0;
    for (0..12) |step_i| {
        last = try lossStepOn(&ctx, &trainer, &opt, long_inputs, long_labels);
        if (step_i == 0) first = last;
    }
    try std.testing.expect(last < first);
    try std.testing.expect(last < first - 0.1); // by a real margin, not noise
}

// ---------------------------------------------------------------------------
// Full-stack finite-difference gradient check.
//
// Op-level gradchecks cover every individual VJP and the loss-descent tests
// prove the trainer learns, but neither catches WIRING bugs in the composed
// model (wrong residual routing, a contribution dropped through the fused
// q/k-norm+RoPE, a LoRA delta added at the wrong point, CE masking
// interacting with grads). The definitive internal check: central finite
// differences through the ENTIRE `Trainer.loss` forward must match the
// analytical adapter gradients.
// ---------------------------------------------------------------------------

/// FD batch labels: the seq-64 next-token labels with a masked prefix and
/// scattered masked positions, so the CE `ignore_index` masking participates
/// in every loss the check differentiates.
const fd_labels = blk: {
    var labels: [long_seq]usize = long_tokens[1..].*;
    for ([_]usize{ 0, 1, 2, 3, 17, 30, 31, 45, 58 }) |i| labels[i] = qwen3_train.ignore_index;
    break :blk labels;
};

/// One scoped forward of the training loss at the CURRENT adapter values on
/// the FD batch (no backward; closing the scope frees the recorded graph).
fn fdLoss(ctx: *ExecContext, trainer: *DefaultTrainer) !f32 {
    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    const loss = try trainer.loss(ctx, long_inputs, &fd_labels);
    return loss.item();
}

test "full-stack gradcheck: finite differences through Trainer.loss match backward" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // seq 64 >= 48: attention routes through the query-tiled forward kernel
    // (exec.zig's private attention_tiled_min_q_seq = 48, head_dim 16 <= its
    // attention_tile_max_d = 256), so the FD check exercises the
    // tiled-forward/3-pass-backward mix real training runs.
    comptime std.debug.assert(long_seq >= 48);

    var model = try buildTinyModel(&ctx, 0xFD0);
    defer model.deinit();
    // dropout_p == 0 (tiny_lora default) is REQUIRED here: finite differences
    // need a deterministic forward that does not depend on the step counter
    // (`loss` advances the dropout stream per call; with p == 0 dropout is
    // the zero-copy identity on every path, so all FD forwards see the same
    // function of the parameters).
    var trainer = try DefaultTrainer.init(&ctx, &model, tiny_lora, 41);
    defer trainer.deinit();

    var opt = optim.AdamW.init(allocator, .{ .lr = 0.05, .weight_decay = 0 });
    defer opt.deinit();
    try trainer.registerAllParams(&opt);

    // LoRA inits B == 0, which makes every A-gradient exactly zero (dL/dA
    // flows through B) — a gradcheck at init would vacuously pass on half the
    // parameters. Two optimizer steps move B off zero so BOTH A and B carry
    // meaningful gradients at the gradcheck point.
    for (0..2) |_| _ = try lossStepOn(&ctx, &trainer, &opt, long_inputs, &fd_labels);
    {
        var b_norm: f32 = 0;
        for (try trainer.adapters[0].q.b.dataConst()) |v| b_norm += @abs(v);
        try std.testing.expect(b_norm > 0);
    }

    // Analytical gradients: one scoped backward at the gradcheck point.
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const loss = try trainer.loss(&ctx, long_inputs, &fd_labels);
        try loss.backward(&ctx);
    }
    // Frozen base weights are constants: no GradState even after backward.
    try std.testing.expect(!model.token_embedding.f32.requiresGrad());
    try std.testing.expect(!model.layers[0].attn_proj.separate.q_proj.f32.requiresGrad());
    try std.testing.expect(!model.layers[1].ffn.dense.down_proj.f32.requiresGrad());
    try std.testing.expect(!model.output.f32.requiresGrad());

    const Sampled = struct {
        layer: usize,
        proj: []const u8,
        which: []const u8,
        /// Mutable handle into the variable's refcounted storage. The facade's
        /// grad-gated `data()` refuses variables by design; this is the same
        /// raw seam the optimizer writes parameters through (`t.value.data()`,
        /// optim.zig). Safe here because A/B are fresh contiguous buffers the
        /// adapter owns, the forward only READS parameter storage, and every
        /// FD evaluation runs in its own fully-closed exec scope — no
        /// recorded graph outlives a perturbation, exactly like an optimizer
        /// step between training forwards.
        param: []f32,
        /// Cloned analytical gradient (allocator-owned).
        grad: []f32,
    };
    var sampled: [4 * tiny_config.num_layers]Sampled = undefined;
    var n_sampled: usize = 0;
    defer for (sampled[0..n_sampled]) |s| allocator.free(s.grad);
    for (trainer.adapters, 0..) |*ads, layer_i| {
        inline for (.{ "q", "v" }) |proj| {
            inline for (.{ "a", "b" }) |which| {
                const t = &@field(@field(ads.*, proj), which);
                var g = (try t.grad(&ctx)) orelse return error.MissingGrad;
                defer g.deinit();
                sampled[n_sampled] = .{
                    .layer = layer_i,
                    .proj = proj,
                    .which = which,
                    .param = t.value.data(),
                    .grad = try allocator.dupe(f32, try g.dataConst()),
                };
                n_sampled += 1;
            }
        }
    }
    try std.testing.expectEqual(sampled.len, n_sampled);

    // Element selection: EVERY element of layer-0 q's A [4, 32] and B [32, 4]
    // (256 elements — full coverage of one adapter), plus 4 seeded-random
    // elements of each remaining adapter tensor (both layers, q and v, A and
    // B — 24 elements), so every gradient pathway is sampled.
    //
    // eps and tolerance: parameters are O(0.1) and the loss is O(ln 64) ≈ 4 in
    // an f32 forward, so the f32 evaluation error δ on the loss is ~1e-6..1e-5
    // absolute and the FD noise is ~δ/eps; eps = 5e-3 keeps that comfortably
    // under the 1e-3 absolute floor while the O(eps²) truncation term stays
    // negligible. Measured over 3 runs (native + scalar backends): worst
    // |g_num - g_ana| ≈ 2e-4 against tol ≥ 1e-3 — a ≥5x margin, and the
    // forward is deterministic per build, so the check is stable, not flaky.
    const eps: f64 = 5e-3;
    const abs_tol: f64 = 1e-3;
    const rel_tol: f64 = 0.02;

    var prng = std.Random.DefaultPrng.init(0xFD5EED);
    const random = prng.random();

    var dot_sum: f64 = 0;
    var num_sq: f64 = 0;
    var ana_sq: f64 = 0;
    var checked: usize = 0;

    for (sampled[0..n_sampled]) |s| {
        const full = s.layer == 0 and std.mem.eql(u8, s.proj, "q");
        const n = if (full) s.param.len else 4;
        for (0..n) |draw| {
            const i = if (full) draw else random.uintLessThan(usize, s.param.len);
            const g_ana: f64 = s.grad[i];
            const original = s.param[i];
            s.param[i] = original + @as(f32, @floatCast(eps));
            const plus = try fdLoss(&ctx, &trainer);
            s.param[i] = original - @as(f32, @floatCast(eps));
            const minus = try fdLoss(&ctx, &trainer);
            s.param[i] = original; // exact bitwise restore
            const g_num = (@as(f64, plus) - minus) / (2 * eps);

            const dev = @abs(g_num - g_ana);
            const tol = abs_tol + rel_tol * @abs(g_ana);
            if (dev > tol) {
                std.debug.print(
                    "gradcheck mismatch at layers.{d}.{s}.lora_{s}[{d}]: analytical {e} numerical {e} |dev| {e} > tol {e}\n",
                    .{ s.layer, s.proj, s.which, i, g_ana, g_num, dev, tol },
                );
                return error.GradientMismatch;
            }
            dot_sum += g_num * g_ana;
            num_sq += g_num * g_num;
            ana_sq += g_ana * g_ana;
            checked += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 256 + 24), checked);

    // The sampled gradient vectors must also agree in direction — cosine
    // similarity catches a systematic scale/sign error even if every element
    // squeaks under the mixed per-element tolerance.
    try std.testing.expect(ana_sq > 0);
    try std.testing.expect(num_sq > 0);
    const cosine = dot_sum / (@sqrt(num_sq) * @sqrt(ana_sq));
    try std.testing.expect(cosine >= 0.999);

    // Checkpointed run at the SAME parameter point: by the parity contract
    // (proven bitwise at seq 64 by "checkpointed layers at seq 64..."), its
    // grads must equal the plain run's bitwise — which transfers the FD
    // verdict to the checkpointed path without a second FD loop.
    opt.zeroGrad();
    trainer.checkpoint_layers = true;
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const loss = try trainer.loss(&ctx, long_inputs, &fd_labels);
        try loss.backward(&ctx);
    }
    {
        var si: usize = 0;
        for (trainer.adapters) |*ads| {
            inline for (.{ "q", "v" }) |proj| {
                inline for (.{ "a", "b" }) |which| {
                    var g = (try @field(@field(ads.*, proj), which).grad(&ctx)) orelse return error.MissingGrad;
                    defer g.deinit();
                    try std.testing.expectEqualSlices(f32, sampled[si].grad, try g.dataConst());
                    si += 1;
                }
            }
        }
    }

    // ALL-MASKED batch through the full stack: CE masks every position, so
    // the loss is exactly zero and every adapter gradient is exactly zero —
    // ties the masking semantics into the composed backward (meaningful here
    // because B != 0, unlike at init where A-grads vanish regardless).
    opt.zeroGrad();
    trainer.checkpoint_layers = false;
    var all_masked: [long_seq]usize = undefined;
    @memset(&all_masked, qwen3_train.ignore_index);
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const loss = try trainer.loss(&ctx, long_inputs, &all_masked);
        try loss.backward(&ctx);
        try std.testing.expectEqual(@as(f32, 0), try loss.item());
    }
    for (trainer.adapters) |*ads| {
        inline for (.{ "q", "v" }) |proj| {
            inline for (.{ "a", "b" }) |which| {
                var g = (try @field(@field(ads.*, proj), which).grad(&ctx)) orelse return error.MissingGrad;
                defer g.deinit();
                for (try g.dataConst()) |v| try std.testing.expectEqual(@as(f32, 0), v);
            }
        }
    }
}

test "label masking: fully masked loss is zero, partial masking changes it" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try buildTinyModel(&ctx, 0xD1CE);
    defer model.deinit();
    var trainer = try DefaultTrainer.init(&ctx, &model, tiny_lora, 4);
    defer trainer.deinit();

    const sentinel = qwen3_train.ignore_index;
    var all_masked: [batch_inputs.len]usize = undefined;
    @memset(&all_masked, sentinel);
    var half_masked: [batch_inputs.len]usize = undefined;
    @memcpy(&half_masked, batch_labels);
    @memset(half_masked[0 .. batch_inputs.len / 2], sentinel);

    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    const masked = try trainer.loss(&ctx, batch_inputs, &all_masked);
    try std.testing.expectEqual(@as(f32, 0), try masked.item());
    const partial = try trainer.loss(&ctx, batch_inputs, &half_masked);
    const full = try trainer.loss(&ctx, batch_inputs, batch_labels);
    try std.testing.expect(try partial.item() != try full.item());
}

test "adapter persistence: save, load into a fresh trainer, bitwise eval" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try buildTinyModel(&ctx, 0xFADE);
    defer model.deinit();
    var trained = try DefaultTrainer.init(&ctx, &model, tiny_lora, 11);
    defer trained.deinit();

    var opt = optim.AdamW.init(allocator, .{ .lr = 0.05, .weight_decay = 0 });
    defer opt.deinit();
    try trained.registerAllParams(&opt);
    for (0..3) |_| _ = try lossStep(&ctx, &trained, &opt);

    var buf: [64 * 1024]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try trained.saveAdapters(&writer);
    const written = writer.buffered();

    // Different init seed: A differs until the load overwrites it.
    var fresh = try DefaultTrainer.init(&ctx, &model, tiny_lora, 99);
    defer fresh.deinit();
    var reader = std.Io.Reader.fixed(written);
    try fresh.loadAdapters(&reader);

    const tokens = [_]usize{ 12, 7, 51, 30, 9 };
    var expected = try trained.evalLastLogits(&ctx, &tokens);
    defer expected.deinit();
    var actual = try fresh.evalLastLogits(&ctx, &tokens);
    defer actual.deinit();
    try std.testing.expectEqualSlices(f32, try expected.dataConst(), try actual.dataConst());
}

/// The adapter checkpoint's expected entry names for the default (q, v)
/// targets over `tiny_config`'s two layers, in emission order. This is the
/// on-disk schema (docs/TRAINING.md §8): renaming or reordering entries breaks
/// resuming from older checkpoints.
const expected_adapter_names = [_][]const u8{
    "layers.0.q.lora_a", "layers.0.q.lora_b", "layers.0.v.lora_a", "layers.0.v.lora_b",
    "layers.1.q.lora_a", "layers.1.q.lora_b", "layers.1.v.lora_a", "layers.1.v.lora_b",
};

/// The adapter state-dict entries in schema order, named from `names` — the
/// explicit reference writer the golden schema test serializes against.
fn adapterEntries(trainer: *const DefaultTrainer, names: []const []const u8) ![expected_adapter_names.len]optim.NamedTensor {
    var entries: [expected_adapter_names.len]optim.NamedTensor = undefined;
    for (trainer.adapters, 0..) |*ads, layer_i| {
        entries[layer_i * 4 + 0] = try optim.NamedTensor.of(names[layer_i * 4 + 0], &ads.q.a);
        entries[layer_i * 4 + 1] = try optim.NamedTensor.of(names[layer_i * 4 + 1], &ads.q.b);
        entries[layer_i * 4 + 2] = try optim.NamedTensor.of(names[layer_i * 4 + 2], &ads.v.a);
        entries[layer_i * 4 + 3] = try optim.NamedTensor.of(names[layer_i * 4 + 3], &ads.v.b);
    }
    return entries;
}

test "adapter checkpoint schema: byte-identical to the explicit named-tensor writer" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try buildTinyModel(&ctx, 0x60D5);
    defer model.deinit();
    var trainer = try DefaultTrainer.init(&ctx, &model, tiny_lora, 5);
    defer trainer.deinit();

    const entries = try adapterEntries(&trainer, &expected_adapter_names);
    var expected_buf: [64 * 1024]u8 = undefined;
    var expected_writer = std.Io.Writer.fixed(&expected_buf);
    try optim.saveStateDict(allocator, &expected_writer, &entries);

    var actual_buf: [64 * 1024]u8 = undefined;
    var actual_writer = std.Io.Writer.fixed(&actual_buf);
    try trainer.saveAdapters(&actual_writer);
    try std.testing.expectEqualSlices(u8, expected_writer.buffered(), actual_writer.buffered());
}

test "adapter load aliases: a renamed stream entry resolves to the current path" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try buildTinyModel(&ctx, 0xA11A);
    defer model.deinit();
    var source = try DefaultTrainer.init(&ctx, &model, tiny_lora, 7);
    defer source.deinit();

    // Serialize the source adapters with ONE entry under an old path.
    var renamed_names = expected_adapter_names;
    renamed_names[0] = "layers.0.q.lora_a.v0";
    const entries = try adapterEntries(&source, &renamed_names);
    var buf: [64 * 1024]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try optim.saveStateDict(allocator, &writer, &entries);
    const written = writer.buffered();

    // Different init seed: A differs until the load overwrites it. The strict
    // no-alias load must reject the old name (and, transactionally, leave the
    // destinations untouched for the aliased retry below).
    var fresh = try DefaultTrainer.init(&ctx, &model, tiny_lora, 99);
    defer fresh.deinit();
    var strict_reader = std.Io.Reader.fixed(written);
    try std.testing.expectError(error.CheckpointUnknownName, fresh.loadAdapters(&strict_reader));

    var reader = std.Io.Reader.fixed(written);
    try fresh.loadAdaptersWithOptions(&reader, .{
        .aliases = &.{.{ .old = "layers.0.q.lora_a.v0", .new = "layers.0.q.lora_a" }},
    });

    for (source.adapters, fresh.adapters) |*src, *dst| {
        try std.testing.expectEqualSlices(f32, try src.q.a.dataConst(), try dst.q.a.dataConst());
        try std.testing.expectEqualSlices(f32, try src.q.b.dataConst(), try dst.q.b.dataConst());
        try std.testing.expectEqualSlices(f32, try src.v.a.dataConst(), try dst.v.a.dataConst());
        try std.testing.expectEqualSlices(f32, try src.v.b.dataConst(), try dst.v.b.dataConst());
    }
}

/// Batch B for the multi-length tests: a DIFFERENT sequence length than
/// `batch_inputs` (7 vs 11), forcing a second RoPE table in the cache.
const batch_inputs_b = batch_tokens[0..7];
const batch_labels_b = batch_tokens[1..8];

/// Two forwards (different seq lengths), then backward BOTH — gradient
/// accumulation. Under the old one-table-per-forward ownership, loss1's
/// checkpoint nodes would dereference a freed RoPE table during backward.
fn accumulateTwoBatches(ctx: *ExecContext, trainer: *DefaultTrainer, losses: *[2]f32) !void {
    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    const loss1 = try trainer.loss(ctx, batch_inputs, batch_labels);
    const loss2 = try trainer.loss(ctx, batch_inputs_b, batch_labels_b);
    try loss1.backward(ctx);
    try loss2.backward(ctx);
    losses[0] = try loss1.item();
    losses[1] = try loss2.item();
}

test "rope cache: gradient accumulation across seq lengths matches checkpoint-off bitwise" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try buildTinyModel(&ctx, 0xACC0);
    defer model.deinit();

    // Dropout on: the recompute must also replay the per-step seed streams.
    const dropout_lora = fucina.lora.Config{ .rank = 4, .alpha = 8, .dropout_p = 0.2 };
    var plain = try DefaultTrainer.init(&ctx, &model, dropout_lora, 21);
    defer plain.deinit();
    var ckpt = try DefaultTrainer.init(&ctx, &model, dropout_lora, 21);
    defer ckpt.deinit();
    ckpt.checkpoint_layers = true;

    var popt = optim.AdamW.init(allocator, .{ .lr = 0.05, .weight_decay = 0 });
    defer popt.deinit();
    try plain.registerAllParams(&popt);
    var copt = optim.AdamW.init(allocator, .{ .lr = 0.05, .weight_decay = 0 });
    defer copt.deinit();
    try ckpt.registerAllParams(&copt);

    var plain_losses: [2]f32 = undefined;
    var ckpt_losses: [2]f32 = undefined;
    try accumulateTwoBatches(&ctx, &plain, &plain_losses);
    try accumulateTwoBatches(&ctx, &ckpt, &ckpt_losses);
    try std.testing.expectEqualSlices(f32, &plain_losses, &ckpt_losses);

    // Accumulated adapter grads must be bitwise equal...
    for (plain.adapters, ckpt.adapters) |*pads, *cads| {
        inline for (.{ "q", "v" }) |name| {
            inline for (.{ "a", "b" }) |which| {
                var pg = (try @field(@field(pads.*, name), which).grad(&ctx)) orelse return error.MissingGrad;
                defer pg.deinit();
                var cg = (try @field(@field(cads.*, name), which).grad(&ctx)) orelse return error.MissingGrad;
                defer cg.deinit();
                try std.testing.expectEqualSlices(f32, try pg.dataConst(), try cg.dataConst());
            }
        }
    }

    // ...and so must the adapter values after one optimizer step on them.
    try popt.step(&ctx);
    popt.zeroGrad();
    try copt.step(&ctx);
    copt.zeroGrad();
    for (plain.adapters, ckpt.adapters) |*pads, *cads| {
        inline for (.{ "q", "v" }) |name| {
            inline for (.{ "a", "b" }) |which| {
                try std.testing.expectEqualSlices(
                    f32,
                    try @field(@field(pads.*, name), which).dataConst(),
                    try @field(@field(cads.*, name), which).dataConst(),
                );
            }
        }
    }
}

/// loss(); evalLastLogits(different length); loss.backward() — under the old
/// table ownership the eval forward freed the table the pending backward's
/// checkpoint nodes still pointed at.
fn lossEvalBackward(ctx: *ExecContext, trainer: *DefaultTrainer, eval_tokens: []const usize, eval_out: []f32) !f32 {
    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    const loss = try trainer.loss(ctx, batch_inputs, batch_labels);
    var logits = try trainer.evalLastLogits(ctx, eval_tokens);
    defer logits.deinit();
    @memcpy(eval_out, try logits.dataConst());
    try loss.backward(ctx);
    return loss.item();
}

test "rope cache: eval between loss and backward matches checkpoint-off bitwise" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try buildTinyModel(&ctx, 0xEA7);
    defer model.deinit();

    const dropout_lora = fucina.lora.Config{ .rank = 4, .alpha = 8, .dropout_p = 0.2 };
    var plain = try DefaultTrainer.init(&ctx, &model, dropout_lora, 31);
    defer plain.deinit();
    var ckpt = try DefaultTrainer.init(&ctx, &model, dropout_lora, 31);
    defer ckpt.deinit();
    ckpt.checkpoint_layers = true;

    // Different length than the training batch: forces a second RoPE table.
    const eval_tokens = [_]usize{ 4, 9, 1, 33, 7 };
    var plain_eval: [tiny_config.vocab_size]f32 = undefined;
    var ckpt_eval: [tiny_config.vocab_size]f32 = undefined;
    const plain_loss = try lossEvalBackward(&ctx, &plain, &eval_tokens, &plain_eval);
    const ckpt_loss = try lossEvalBackward(&ctx, &ckpt, &eval_tokens, &ckpt_eval);

    try std.testing.expectEqual(plain_loss, ckpt_loss);
    try std.testing.expectEqualSlices(f32, &plain_eval, &ckpt_eval);
    for (plain.adapters, ckpt.adapters) |*pads, *cads| {
        inline for (.{ "q", "v" }) |name| {
            inline for (.{ "a", "b" }) |which| {
                var pg = (try @field(@field(pads.*, name), which).grad(&ctx)) orelse return error.MissingGrad;
                defer pg.deinit();
                var cg = (try @field(@field(cads.*, name), which).grad(&ctx)) orelse return error.MissingGrad;
                defer cg.deinit();
                try std.testing.expectEqualSlices(f32, try pg.dataConst(), try cg.dataConst());
            }
        }
    }
}

test "dropout step counter persists through checkpoint directory state" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try buildTinyModel(&ctx, 0x57E9);
    defer model.deinit();

    const dropout_lora = fucina.lora.Config{ .rank = 4, .alpha = 8, .dropout_p = 0.2 };
    var trained = try DefaultTrainer.init(&ctx, &model, dropout_lora, 13);
    defer trained.deinit();
    var opt = optim.AdamW.init(allocator, .{ .lr = 0.05, .weight_decay = 0 });
    defer opt.deinit();
    try trained.registerAllParams(&opt);
    for (0..2) |_| _ = try lossStep(&ctx, &trained, &opt);

    // Save BEFORE the third loss: loss() advances the step counter. Adapter
    // tensors are a clean safetensors file; the counter lives in JSON state.
    var path_buf: [128]u8 = undefined;
    const dir_path = try std.fmt.bufPrint(&path_buf, "qwen3_train_ckpt_test_{d}", .{std.Io.Clock.real.now(std.testing.io).nanoseconds});
    defer std.Io.Dir.cwd().deleteTree(std.testing.io, dir_path) catch {};
    try fucina.training_checkpoint.beginSave(allocator, std.testing.io, dir_path);
    const adapters_path = try fucina.training_checkpoint.pathJoin(allocator, dir_path, fucina.training_checkpoint.adapters_state_file);
    defer allocator.free(adapters_path);
    const SaveAdapters = struct {
        fn write(t: *const DefaultTrainer, writer: *std.Io.Writer) !void {
            try t.saveAdapters(writer);
        }
    };
    try fucina.training_checkpoint.writeFileAtomic(std.testing.io, adapters_path, &trained, SaveAdapters.write);
    try fucina.training_checkpoint.saveTrainerState(allocator, std.testing.io, dir_path, .{
        .step = trained.step_counter,
        .seed = trained.seed,
        .lora_rank = @intCast(dropout_lora.rank),
        .lora_alpha = @floatCast(dropout_lora.alpha),
        .lora_dropout_p = @floatCast(dropout_lora.dropout_p),
    });

    // The uninterrupted third-step loss (forward only, no optimizer step).
    const expected = blk: {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const loss = try trained.loss(&ctx, batch_inputs, batch_labels);
        break :blk try loss.item();
    };

    // Resume into a fresh trainer. The checkpoint seed/counter drive the
    // dropout stream; adapter tensors stay a standalone safetensors payload.
    var resumed = try DefaultTrainer.init(&ctx, &model, dropout_lora, 13);
    defer resumed.deinit();
    const state = try fucina.training_checkpoint.loadTrainerState(allocator, std.testing.io, dir_path);
    resumed.seed = state.seed;
    resumed.step_counter = state.step;
    {
        var file = try std.Io.Dir.cwd().openFile(std.testing.io, adapters_path, .{});
        defer file.close(std.testing.io);
        var read_buffer: [64 * 1024]u8 = undefined;
        var reader = file.reader(std.testing.io, &read_buffer);
        try resumed.loadAdapters(&reader.interface);
    }
    try std.testing.expectEqual(@as(u64, 2), resumed.step_counter);

    const got = blk: {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const loss = try resumed.loss(&ctx, batch_inputs, batch_labels);
        break :blk try loss.item();
    };
    try std.testing.expectEqual(expected, got);
}

/// Convert every layer's separate q/k/v and gate/up projections into the
/// fused layout the real-GGUF loader produces (`weights.fuseLinear`: rows
/// stacked q,k,v and gate,up along the out axis, parts consumed).
fn fuseModelProjections(ctx: *ExecContext, model: *qwen3.Model) !void {
    for (model.layers) |*layer| {
        switch (layer.attn_proj) {
            .separate => |*sep| {
                var parts = [_]*weights.LinearWeight{ &sep.q_proj, &sep.k_proj, &sep.v_proj };
                const fused = (try weights.fuseLinear(ctx, &parts)) orelse return error.FuseFailed;
                layer.attn_proj = .{ .fused = fused };
            },
            .fused => {},
        }
        switch (layer.ffn) {
            .dense => |*dense| switch (dense.input_proj) {
                .separate => |*sep| {
                    var parts = [_]*weights.LinearWeight{ &sep.gate_proj, &sep.up_proj };
                    const fused = (try weights.fuseLinear(ctx, &parts)) orelse return error.FuseFailed;
                    dense.input_proj = .{ .fused = fused };
                },
                .fused => {},
            },
            .moe => unreachable, // tests build dense layers only
        }
    }
}

test "fused qkv / gate-up arms match the separate arms (eval + adapter grads)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Same seed -> bitwise-identical weights; then fuse the second model.
    var sep_model = try buildTinyModel(&ctx, 0xF05E);
    defer sep_model.deinit();
    var fused_model = try buildTinyModel(&ctx, 0xF05E);
    defer fused_model.deinit();
    try fuseModelProjections(&ctx, &fused_model);

    var sep_tr = try DefaultTrainer.init(&ctx, &sep_model, tiny_lora, 17);
    defer sep_tr.deinit();
    var fused_tr = try DefaultTrainer.init(&ctx, &fused_model, tiny_lora, 17);
    defer fused_tr.deinit();

    const tokens = [_]usize{ 3, 17, 60, 5, 22, 41, 7, 0 };
    var sep_logits = try sep_tr.evalLastLogits(&ctx, &tokens);
    defer sep_logits.deinit();
    var fused_logits = try fused_tr.evalLastLogits(&ctx, &tokens);
    defer fused_logits.deinit();
    const sep_values = try sep_logits.dataConst();
    const fused_values = try fused_logits.dataConst();
    try std.testing.expectEqual(sep_values.len, fused_values.len);
    for (sep_values, fused_values) |e, a| try std.testing.expectApproxEqAbs(e, a, 1e-5);

    // One training step each; the adapter grads must agree tightly. NOT
    // bitwise: the input-gradient VJP of the fused arm is one GEMM whose
    // reduction axis spans the stacked q/k/v (gate/up) rows, while the
    // separate arm accumulates per-projection GEMMs — a different summation
    // order whose low-bit difference propagates to the layers below.
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const loss = try sep_tr.loss(&ctx, batch_inputs, batch_labels);
        try loss.backward(&ctx);
    }
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const loss = try fused_tr.loss(&ctx, batch_inputs, batch_labels);
        try loss.backward(&ctx);
    }
    for (sep_tr.adapters, fused_tr.adapters) |*sads, *fads| {
        inline for (.{ "q", "v" }) |name| {
            inline for (.{ "a", "b" }) |which| {
                var sg = (try @field(@field(sads.*, name), which).grad(&ctx)) orelse return error.MissingGrad;
                defer sg.deinit();
                var fg = (try @field(@field(fads.*, name), which).grad(&ctx)) orelse return error.MissingGrad;
                defer fg.deinit();
                const sgd = try sg.dataConst();
                const fgd = try fg.dataConst();
                try std.testing.expectEqual(sgd.len, fgd.len);
                for (sgd, fgd) |e, a| try std.testing.expectApproxEqAbs(e, a, 1e-5);
            }
        }
    }
}

test "MoE configs are rejected" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try buildTinyModel(&ctx, 0x40E);
    defer model.deinit();
    model.config.num_experts = 4;
    model.config.num_experts_used = 2;
    model.config.moe_intermediate_size = 8;
    try std.testing.expectError(
        qwen3_train.Error.MoeUnsupported,
        DefaultTrainer.init(&ctx, &model, tiny_lora, 5),
    );
}

test "lossExt with default options is bitwise-identical to loss" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try buildTinyModel(&ctx, 0x10E7);
    defer model.deinit();

    // Dropout on: `lossExt` must consume the same per-call seed stream.
    const dropout_lora = fucina.lora.Config{ .rank = 4, .alpha = 8, .dropout_p = 0.2 };
    var base = try DefaultTrainer.init(&ctx, &model, dropout_lora, 11);
    defer base.deinit();
    var ext = try DefaultTrainer.init(&ctx, &model, dropout_lora, 11);
    defer ext.deinit();

    var base_loss: f32 = undefined;
    var ext_loss: f32 = undefined;
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const l = try base.loss(&ctx, batch_inputs, batch_labels);
        try l.backward(&ctx);
        base_loss = try l.item();
    }
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const l = try ext.lossExt(&ctx, batch_inputs, batch_labels, .{});
        try l.backward(&ctx);
        ext_loss = try l.item();
    }
    try std.testing.expectEqual(base_loss, ext_loss);
    try std.testing.expectEqual(base.step_counter, ext.step_counter);

    for (base.adapters, ext.adapters) |*bads, *eads| {
        inline for (.{ "q", "v" }) |name| {
            inline for (.{ "a", "b" }) |which| {
                var bg = (try @field(@field(bads.*, name), which).grad(&ctx)) orelse return error.MissingGrad;
                defer bg.deinit();
                var eg = (try @field(@field(eads.*, name), which).grad(&ctx)) orelse return error.MissingGrad;
                defer eg.deinit();
                try std.testing.expectEqualSlices(f32, try bg.dataConst(), try eg.dataConst());
            }
        }
    }
}

/// Caller-owned f32 copy of a param's accumulated gradient.
fn gradDupe(allocator: std.mem.Allocator, ctx: *ExecContext, t: anytype) ![]f32 {
    var g = (try t.grad(ctx)) orelse return error.MissingGrad;
    defer g.deinit();
    return allocator.dupe(f32, try g.dataConst());
}

test "accumulated scaled grads match manually summed per-batch grads bitwise" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try buildTinyModel(&ctx, 0xACC2);
    defer model.deinit();

    const dropout_lora = fucina.lora.Config{ .rank = 4, .alpha = 8, .dropout_p = 0.2 };
    var acc = try DefaultTrainer.init(&ctx, &model, dropout_lora, 19);
    defer acc.deinit();
    var man = try DefaultTrainer.init(&ctx, &model, dropout_lora, 19);
    defer man.deinit();

    const opts = DefaultTrainer.LossOptions{ .loss_scale = 0.5 };

    // The accumulation recipe: two micro-batches, one exec scope EACH, no
    // zeroGrad between — the second backward ADDS into the persisted grads.
    var acc_losses: [2]f32 = undefined;
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const l = try acc.lossExt(&ctx, batch_inputs, batch_labels, opts);
        try l.backward(&ctx);
        acc_losses[0] = try l.item();
    }
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const l = try acc.lossExt(&ctx, batch_inputs_b, batch_labels_b, opts);
        try l.backward(&ctx);
        acc_losses[1] = try l.item();
    }

    // Manual reference on an identical trainer: run each micro-batch alone
    // (zeroGrad between), capture both grads, and sum g1 + g2 IN THAT ORDER —
    // the same elementwise addition the accumulation's addInPlace performs.
    var man_losses: [2]f32 = undefined;
    const n_grads = tiny_config.num_layers * 4; // layers x {q,v} x {A,B}
    var g1: [n_grads][]f32 = undefined;
    var built_g1: usize = 0;
    defer for (g1[0..built_g1]) |g| allocator.free(g);
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const l = try man.lossExt(&ctx, batch_inputs, batch_labels, opts);
        try l.backward(&ctx);
        man_losses[0] = try l.item();
    }
    for (man.adapters) |*ads| {
        inline for (.{ "q", "v" }) |name| {
            inline for (.{ "a", "b" }) |which| {
                g1[built_g1] = try gradDupe(allocator, &ctx, &@field(@field(ads.*, name), which));
                built_g1 += 1;
                @field(@field(ads.*, name), which).zeroGrad();
            }
        }
    }
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const l = try man.lossExt(&ctx, batch_inputs_b, batch_labels_b, opts);
        try l.backward(&ctx);
        man_losses[1] = try l.item();
    }

    // Same trainer state (seed, step streams) => identical micro losses.
    try std.testing.expectEqualSlices(f32, &acc_losses, &man_losses);

    var idx: usize = 0;
    for (acc.adapters, man.adapters) |*aads, *mads| {
        inline for (.{ "q", "v" }) |name| {
            inline for (.{ "a", "b" }) |which| {
                var ag = (try @field(@field(aads.*, name), which).grad(&ctx)) orelse return error.MissingGrad;
                defer ag.deinit();
                const g2 = try gradDupe(allocator, &ctx, &@field(@field(mads.*, name), which));
                defer allocator.free(g2);
                const expected = try allocator.alloc(f32, g2.len);
                defer allocator.free(expected);
                for (expected, g1[idx], g2) |*e, a, b| e.* = a + b;
                try std.testing.expectEqualSlices(f32, expected, try ag.dataConst());
                idx += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// forwardHidden / lossInjected seam (injection + truncation).
// ---------------------------------------------------------------------------

const Hidden = qwen3_train.Hidden;

test "forwardHidden full depth + output tail reproduces evalLastLogits" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try buildTinyModel(&ctx, 0xF00);
    defer model.deinit();
    var trainer = try DefaultTrainer.init(&ctx, &model, tiny_lora, 3);
    defer trainer.deinit();

    const tokens = [_]usize{ 3, 17, 60, 5, 22, 41, 7, 0 };
    var ref = try trainer.evalLastLogits(&ctx, &tokens);
    defer ref.deinit();

    // Refactor safety: the raw residual + the manual norm/output tail must
    // reproduce the composed logits path bitwise (identical op sequence).
    var got: [tiny_config.vocab_size]f32 = undefined;
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const h = try trainer.forwardHidden(&ctx, &tokens, null, .{});
        const normed = try h.rmsNormMul(&ctx, .embed, &model.output_norm, tiny_config.rms_norm_eps);
        var out_w = try model.output.f32.withTags(&ctx, .{ .vocab, .embed });
        defer out_w.deinit();
        const logits = try normed.dot(&ctx, &out_w, .embed);
        const last = try logits.narrow(&ctx, .seq, tokens.len - 1, 1);
        try last.copyTo(&got);
    }
    try std.testing.expectEqualSlices(f32, try ref.dataConst(), &got);

    // Scope guard: forwardHidden outside an exec scope is an error.
    try std.testing.expectError(
        qwen3_train.Error.ExecScopeRequired,
        trainer.forwardHidden(&ctx, &tokens, null, .{}),
    );
}

test "truncated forwardHidden matches a reference model built from the same layers" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const seed: u64 = 0x7A0C;
    var full = try buildTinyModel(&ctx, seed);
    defer full.deinit();
    // Single-layer reference models sharing the full model's layer weights
    // (identical per-layer seeds): layer 0 checks `layer_count`, layer 1
    // checks `start_layer`. Adapter seeds differ across trainers for the
    // layer-1 case, but at init B == 0 makes every LoRA delta exactly zero,
    // so all forwards below are pure base-model functions.
    var ref0_model = try buildTinyModelPicked(&ctx, tiny_config, seed, &.{0});
    defer ref0_model.deinit();
    var ref1_model = try buildTinyModelPicked(&ctx, tiny_config, seed, &.{1});
    defer ref1_model.deinit();

    var trainer = try DefaultTrainer.init(&ctx, &full, tiny_lora, 5);
    defer trainer.deinit();
    var ref0 = try DefaultTrainer.init(&ctx, &ref0_model, tiny_lora, 5);
    defer ref0.deinit();
    var ref1 = try DefaultTrainer.init(&ctx, &ref1_model, tiny_lora, 5);
    defer ref1.deinit();

    const d = tiny_config.hidden_size;
    var got: [batch_inputs.len * d]f32 = undefined;
    var want: [batch_inputs.len * d]f32 = undefined;
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);

        // layer_count = 1: embedding -> layer 0 only.
        const h_trunc = try trainer.forwardHidden(&ctx, batch_inputs, null, .{ .layer_count = 1 });
        try h_trunc.copyTo(&got);
        const h_ref0 = try ref0.forwardHidden(&ctx, batch_inputs, null, .{});
        try h_ref0.copyTo(&want);
        try std.testing.expectEqualSlices(f32, &want, &got);

        // start_layer = 1: embedding -> layer 1 only.
        const h_start = try trainer.forwardHidden(&ctx, batch_inputs, null, .{ .start_layer = 1, .layer_count = 1 });
        try h_start.copyTo(&got);
        const h_ref1 = try ref1.forwardHidden(&ctx, batch_inputs, null, .{});
        try h_ref1.copyTo(&want);
        try std.testing.expectEqualSlices(f32, &want, &got);

        // Out-of-range layer windows are rejected up front.
        try std.testing.expectError(
            qwen3_train.Error.InvalidLayerRange,
            trainer.forwardHidden(&ctx, batch_inputs, null, .{ .layer_count = 3 }),
        );
        try std.testing.expectError(
            qwen3_train.Error.InvalidLayerRange,
            trainer.forwardHidden(&ctx, batch_inputs, null, .{ .start_layer = 3 }),
        );
        try std.testing.expectError(
            qwen3_train.Error.InvalidLayerRange,
            trainer.forwardHidden(&ctx, batch_inputs, null, .{ .start_layer = 1, .layer_count = 2 }),
        );
    }
}

test "injected forward: causal prefix unchanged, gradient reaches the injected row" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try buildTinyModel(&ctx, 0x1BAD);
    defer model.deinit();
    var trainer = try DefaultTrainer.init(&ctx, &model, tiny_lora, 9);
    defer trainer.deinit();

    const d = tiny_config.hidden_size;
    const pos: usize = 4;
    var row_values: [d]f32 = undefined;
    rng.uniformFill(0xE0B, &row_values, -0.5, 0.5);

    // Causality: positions BEFORE the injection see identical inputs and
    // attend only to identical keys, so their residual rows are bitwise
    // unchanged; the injected position itself must change.
    var plain: [batch_inputs.len * d]f32 = undefined;
    var injected: [batch_inputs.len * d]f32 = undefined;
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const h_plain = try trainer.forwardHidden(&ctx, batch_inputs, null, .{});
        try h_plain.copyTo(&plain);
        // Explicitly-created tensors stay caller-owned even inside a scope.
        var row = try Hidden.fromSlice(&ctx, .{ 1, d }, &row_values);
        defer row.deinit();
        const h_inj = try trainer.forwardHidden(&ctx, batch_inputs, null, .{
            .inject = .{ .pos = pos, .row = &row },
        });
        try h_inj.copyTo(&injected);

        // Out-of-bounds injection positions are rejected up front.
        try std.testing.expectError(
            qwen3_train.Error.InvalidInjection,
            trainer.forwardHidden(&ctx, batch_inputs, null, .{
                .inject = .{ .pos = batch_inputs.len, .row = &row },
            }),
        );
    }
    try std.testing.expectEqualSlices(f32, plain[0 .. pos * d], injected[0 .. pos * d]);
    try std.testing.expect(!std.mem.eql(f32, plain[pos * d .. (pos + 1) * d], injected[pos * d .. (pos + 1) * d]));

    // Injected generation entry: the last-position logits must differ too
    // (the last query attends to the injected key/value).
    var last_plain = try trainer.evalLastLogits(&ctx, batch_inputs);
    defer last_plain.deinit();
    var row_const = try Hidden.fromSlice(&ctx, .{ 1, d }, &row_values);
    defer row_const.deinit();
    var last_inj = try trainer.evalLastLogitsExt(&ctx, batch_inputs, .{
        .inject = .{ .pos = pos, .row = &row_const },
    });
    defer last_inj.deinit();
    try std.testing.expect(!std.mem.eql(f32, try last_plain.dataConst(), try last_inj.dataConst()));

    // A VARIABLE injected row receives a nonzero gradient through the frozen
    // stack (the differentiable setSlice), while adapter grads stay finite
    // (A-grads are exactly zero at init because B == 0; B-grads are live).
    var var_row = try Hidden.variableFromSlice(&ctx, .{ 1, d }, &row_values);
    defer var_row.deinit();
    var labels: [batch_labels.len]usize = batch_labels[0..].*;
    @memset(labels[0 .. pos + 1], qwen3_train.ignore_index); // supervise past the injection
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const loss = try trainer.lossInjected(&ctx, batch_inputs, &labels, .{ .pos = pos, .row = &var_row }, .{});
        try loss.backward(&ctx);
        try std.testing.expect(std.math.isFinite(try loss.item()));
    }
    {
        var g = (try var_row.grad(&ctx)) orelse return error.MissingGrad;
        defer g.deinit();
        var norm: f64 = 0;
        for (try g.dataConst()) |v| {
            try std.testing.expect(std.math.isFinite(v));
            norm += @as(f64, v) * @as(f64, v);
        }
        try std.testing.expect(norm > 0);
    }
    var b_norm: f64 = 0;
    for (trainer.adapters) |*ads| {
        inline for (.{ "q", "v" }) |name| {
            inline for (.{ "a", "b" }) |which| {
                var g = (try @field(@field(ads.*, name), which).grad(&ctx)) orelse return error.MissingGrad;
                defer g.deinit();
                for (try g.dataConst()) |v| {
                    try std.testing.expect(std.math.isFinite(v));
                    if (comptime std.mem.eql(u8, which, "b")) b_norm += @as(f64, v) * @as(f64, v);
                }
            }
        }
    }
    try std.testing.expect(b_norm > 0);
}

test "lossInjected: masking semantics and step-counter advance" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try buildTinyModel(&ctx, 0x1CED);
    defer model.deinit();
    var trainer = try DefaultTrainer.init(&ctx, &model, tiny_lora, 6);
    defer trainer.deinit();

    const d = tiny_config.hidden_size;
    var row_values: [d]f32 = undefined;
    rng.uniformFill(0xCED, &row_values, -0.5, 0.5);
    var row = try Hidden.fromSlice(&ctx, .{ 1, d }, &row_values);
    defer row.deinit();
    const inj = qwen3_train.Injection{ .pos = 2, .row = &row };

    const sentinel = qwen3_train.ignore_index;
    var all_masked: [batch_inputs.len]usize = undefined;
    @memset(&all_masked, sentinel);
    var half_masked: [batch_inputs.len]usize = undefined;
    @memcpy(&half_masked, batch_labels);
    @memset(half_masked[0 .. batch_inputs.len / 2], sentinel);

    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    const step_before = trainer.step_counter;
    const masked = try trainer.lossInjected(&ctx, batch_inputs, &all_masked, inj, .{});
    try std.testing.expectEqual(@as(f32, 0), try masked.item());
    try std.testing.expectEqual(step_before + 1, trainer.step_counter);
    const partial = try trainer.lossInjected(&ctx, batch_inputs, &half_masked, inj, .{});
    const full = try trainer.lossInjected(&ctx, batch_inputs, batch_labels, inj, .{});
    try std.testing.expect(try partial.item() != try full.item());
    // The injection must actually participate: same labels, no injection,
    // via the plain path differs from the injected loss.
    const plain = try trainer.loss(&ctx, batch_inputs, batch_labels);
    try std.testing.expect(try plain.item() != try full.item());
}

test "sum reduction scaled by 1/valid matches mean numerically" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try buildTinyModel(&ctx, 0x5C0F);
    defer model.deinit();
    var trainer = try DefaultTrainer.init(&ctx, &model, tiny_lora, 23);
    defer trainer.deinit();

    // Mask two positions so `.mean`'s denominator is the VALID count, not len.
    var masked: [batch_labels.len]usize = batch_labels[0..].*;
    masked[0] = qwen3_train.ignore_index;
    masked[5] = qwen3_train.ignore_index;
    const valid = masked.len - 2;

    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    // dropout_p == 0, so the two forwards are identical despite the
    // step-counter advance (streams are seeded but never consumed).
    const mean_l = try trainer.lossExt(&ctx, batch_inputs, &masked, .{});
    const sum_l = try trainer.lossExt(&ctx, batch_inputs, &masked, .{
        .reduction = .sum,
        .loss_scale = 1.0 / @as(f32, @floatFromInt(valid)),
    });
    const mean_v = try mean_l.item();
    const sum_v = try sum_l.item();
    try std.testing.expectApproxEqRel(mean_v, sum_v, 1e-6);
}

test "engram graft: zero-init is bitwise identity and trains through the frozen stack" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try buildTinyModel(&ctx, 0xE569);
    defer model.deinit();
    var trainer = try DefaultTrainer.init(&ctx, &model, tiny_lora, 1);
    defer trainer.deinit();

    const engram = @import("../engram.zig");
    const ecfg = engram.Config{
        .hidden_size = tiny_config.hidden_size,
        .hc_mult = 1,
        .n_embed_per_ngram = 8,
        .n_head_per_ngram = 2,
        .engram_vocab_size = &.{ 11, 13 },
        .kernel_size = 2,
        .pad_id = 0,
    };
    var graft = try engram.Engram.init(&ctx, allocator, ecfg, &.{ 0, 1 }, 7, null, .{ .graft_zero_init = true });
    defer graft.deinit();

    // Hash rows for the batch, per plan slot (ids ARE the compressed ids:
    // no lookup).
    var ids: [batch_inputs.len]i64 = undefined;
    for (&ids, batch_inputs) |*dst, tok| dst.* = @intCast(tok);
    const heads = ecfg.headsPerLayer();
    var rows_storage: [2][batch_inputs.len * 4]usize = undefined;
    try std.testing.expectEqual(@as(usize, 4), heads);
    var rows: [2][]const usize = undefined;
    for (0..2) |slot| {
        try graft.plan.hashInto(slot, &ids, &rows_storage[slot]);
        rows[slot] = &rows_storage[slot];
    }
    const opts = qwen3_train.ForwardOptions{ .engram = .{ .model = &graft, .rows = &rows } };

    // Zero-init graft: logits BITWISE identical to the bare trainer forward.
    var bare = try trainer.evalLogits(&ctx, batch_inputs);
    defer bare.deinit();
    var grafted = try trainer.evalLogitsExt(&ctx, batch_inputs, opts);
    defer grafted.deinit();
    try std.testing.expectEqualSlices(f32, try bare.dataConst(), try grafted.dataConst());

    // Training: loss decreases over AdamW steps on the engram params alone
    // (frozen trunk, LoRA untouched), driven through lossForwardExt.
    var opt = optim.AdamW.init(allocator, .{ .lr = 5e-2, .weight_decay = 0 });
    defer opt.deinit();
    try graft.registerParams(&opt);

    var first: f32 = undefined;
    var last: f32 = undefined;
    for (0..8) |step_i| {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        var loss = try trainer.lossForwardExt(&ctx, batch_inputs, batch_labels, opts, .{});
        defer loss.deinit();
        const v = try loss.item();
        if (step_i == 0) first = v;
        last = v;
        try loss.backward(&ctx);
        try opt.step(&ctx);
        opt.zeroGrad();
        graft.zeroGrad();
    }
    try std.testing.expect(last < first);
}
