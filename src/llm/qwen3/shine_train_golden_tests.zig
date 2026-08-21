//! PyTorch golden-parity test for the Zig-native SHINE training step: one
//! full loss + backward on the tiny synthetic model dumped by
//! tools/gen_shine_train_goldens.py (models/shine/train-goldens, skipped
//! when absent). The gate compares the scalar loss and the gradient of
//! every trainable leaf (all Meta-LoRA A/B pairs, every M2P parameter, the
//! positional embeddings) against the reference autograd. Memory tokens
//! are constant zeros on both sides (the reference optimizer excludes
//! them), so their gradient is not part of the gate.
const std = @import("std");
const fucina = @import("fucina");
const qwen3 = @import("model.zig");
const shine = @import("shine.zig");
const shine_train = @import("shine_train.zig");
const qwen3_train = @import("train.zig");
const weights = @import("fucina").weights;

const ExecContext = fucina.ExecContext;

const goldens_dir = "models/shine/train-goldens";

// The generator's fixed tiny geometry.
const base = qwen3.Config{
    .vocab_size = 256,
    .hidden_size = 64,
    .intermediate_size = 96,
    .num_layers = 2,
    .num_attention_heads = 4,
    .num_key_value_heads = 2,
    .head_dim = 16,
    .rms_norm_eps = 1e-6,
    .rope_theta = 10_000,
};
const sh_config = shine.Config{
    .hidden_size = 64,
    .num_layers = 2,
    .num_mem_token = 29,
    .metalora_r = 4,
    .lora_r = 2,
    .scale = 0.001,
    .m2p_layers = 2,
    .m2p_heads = 4,
    .m2p_ffn = 128,
    .m2p_eps = 1e-5,
    .layer_transformer_first = true,
};

fn readF32(allocator: std.mem.Allocator, name: []const u8, expected_len: usize) ![]f32 {
    const path = try std.fmt.allocPrint(allocator, goldens_dir ++ "/{s}", .{name});
    defer allocator.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, allocator, .limited(1 << 26));
    defer allocator.free(bytes);
    if (bytes.len != expected_len * 4) return error.GoldenShapeMismatch;
    const out = try allocator.alloc(f32, expected_len);
    @memcpy(std.mem.sliceAsBytes(out), bytes);
    return out;
}

fn readIdsI32(allocator: std.mem.Allocator, name: []const u8) ![]i32 {
    const path = try std.fmt.allocPrint(allocator, goldens_dir ++ "/{s}", .{name});
    defer allocator.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, allocator, .limited(1 << 20));
    defer allocator.free(bytes);
    const out = try allocator.alloc(i32, bytes.len / 4);
    for (out, 0..) |*v, i| v.* = std.mem.readInt(i32, bytes[i * 4 ..][0..4], .little);
    return out;
}

fn linearFromFile(ctx: *ExecContext, allocator: std.mem.Allocator, name: []const u8, rows: usize, cols: usize) !weights.LinearWeight {
    const values = try readF32(allocator, name, rows * cols);
    defer allocator.free(values);
    const w = try weights.WeightF32.fromSlice(ctx, .{ rows, cols }, values);
    return .{ .f32 = w };
}

fn vectorFromFile(ctx: *ExecContext, allocator: std.mem.Allocator, comptime tag: @TypeOf(.tag), name: []const u8, len: usize) !fucina.Tensor(.{tag}) {
    const values = try readF32(allocator, name, len);
    defer allocator.free(values);
    return fucina.Tensor(.{tag}).fromSlice(ctx, .{len}, values);
}

fn buildModel(ctx: *ExecContext, allocator: std.mem.Allocator) !qwen3.Model {
    var name_buf: [128]u8 = undefined;
    var token_embedding = try linearFromFile(ctx, allocator, "model.model.embed_tokens.weight.f32.bin", base.vocab_size, base.hidden_size);
    errdefer token_embedding.deinit();
    var output_norm = try vectorFromFile(ctx, allocator, .embed, "model.model.norm.weight.f32.bin", base.hidden_size);
    errdefer output_norm.deinit();
    var output = try linearFromFile(ctx, allocator, "model.lm_head.weight.f32.bin", base.vocab_size, base.hidden_size);
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
        const q_dim = base.qProjectionDim();
        const kv_dim = base.kvProjectionDim();
        var q_proj = try linearFromFile(ctx, allocator, try std.fmt.bufPrint(&name_buf, "model.model.layers.{d}.self_attn.q_proj.weight.f32.bin", .{i}), q_dim, base.hidden_size);
        errdefer q_proj.deinit();
        var k_proj = try linearFromFile(ctx, allocator, try std.fmt.bufPrint(&name_buf, "model.model.layers.{d}.self_attn.k_proj.weight.f32.bin", .{i}), kv_dim, base.hidden_size);
        errdefer k_proj.deinit();
        var v_proj = try linearFromFile(ctx, allocator, try std.fmt.bufPrint(&name_buf, "model.model.layers.{d}.self_attn.v_proj.weight.f32.bin", .{i}), kv_dim, base.hidden_size);
        errdefer v_proj.deinit();
        var o_proj = try linearFromFile(ctx, allocator, try std.fmt.bufPrint(&name_buf, "model.model.layers.{d}.self_attn.o_proj.weight.f32.bin", .{i}), base.hidden_size, q_dim);
        errdefer o_proj.deinit();
        var gate_proj = try linearFromFile(ctx, allocator, try std.fmt.bufPrint(&name_buf, "model.model.layers.{d}.mlp.gate_proj.weight.f32.bin", .{i}), base.intermediate_size, base.hidden_size);
        errdefer gate_proj.deinit();
        var up_proj = try linearFromFile(ctx, allocator, try std.fmt.bufPrint(&name_buf, "model.model.layers.{d}.mlp.up_proj.weight.f32.bin", .{i}), base.intermediate_size, base.hidden_size);
        errdefer up_proj.deinit();
        var down_proj = try linearFromFile(ctx, allocator, try std.fmt.bufPrint(&name_buf, "model.model.layers.{d}.mlp.down_proj.weight.f32.bin", .{i}), base.hidden_size, base.intermediate_size);
        errdefer down_proj.deinit();
        var attn_norm = try vectorFromFile(ctx, allocator, .embed, try std.fmt.bufPrint(&name_buf, "model.model.layers.{d}.input_layernorm.weight.f32.bin", .{i}), base.hidden_size);
        errdefer attn_norm.deinit();
        var ffn_norm = try vectorFromFile(ctx, allocator, .embed, try std.fmt.bufPrint(&name_buf, "model.model.layers.{d}.post_attention_layernorm.weight.f32.bin", .{i}), base.hidden_size);
        errdefer ffn_norm.deinit();
        var q_norm = try vectorFromFile(ctx, allocator, .d, try std.fmt.bufPrint(&name_buf, "model.model.layers.{d}.self_attn.q_norm.weight.f32.bin", .{i}), base.head_dim);
        errdefer q_norm.deinit();
        var k_norm = try vectorFromFile(ctx, allocator, .d, try std.fmt.bufPrint(&name_buf, "model.model.layers.{d}.self_attn.k_norm.weight.f32.bin", .{i}), base.head_dim);
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

fn buildTrainer(ctx: *ExecContext, allocator: std.mem.Allocator, model: *const qwen3.Model) !shine_train.ShineTrainer {
    var name_buf: [128]u8 = undefined;
    const layers = try allocator.alloc(shine.LayerLora, base.num_layers);
    var built: usize = 0;
    errdefer {
        for (layers[0..built]) |*layer| layer.deinit();
        allocator.free(layers);
    }
    const groups = [_][]const u8{ "attention.q", "attention.k", "attention.v", "attention.o", "mlp.gate", "mlp.up", "mlp.down" };
    for (layers, 0..) |*layer, i| {
        var pairs: [7]shine.LoraPair = undefined;
        var pair_built: usize = 0;
        errdefer for (pairs[0..pair_built]) |*pair| pair.deinit();
        inline for (shine.modules, groups, 0..) |module, group, gi| {
            _ = module;
            const dims = shine.moduleDims(base, shine.modules[gi]);
            const a_values = try readF32(allocator, try std.fmt.bufPrint(&name_buf, "metalora.{d}.{s}.A.f32.bin", .{ i, group }), dims[0] * sh_config.metalora_r);
            defer allocator.free(a_values);
            const b_values = try readF32(allocator, try std.fmt.bufPrint(&name_buf, "metalora.{d}.{s}.B.f32.bin", .{ i, group }), sh_config.metalora_r * dims[1]);
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
    const hidden = sh_config.hidden_size;
    const ff = sh_config.m2p_ffn;
    for (m2p, 0..) |*block, i| {
        const parts = [_]struct { name: []const u8, len: usize }{
            .{ .name = "self_attn.in_proj_weight", .len = 3 * hidden * hidden },
            .{ .name = "self_attn.in_proj_bias", .len = 3 * hidden },
            .{ .name = "self_attn.out_proj.weight", .len = hidden * hidden },
            .{ .name = "self_attn.out_proj.bias", .len = hidden },
            .{ .name = "linear1.weight", .len = ff * hidden },
            .{ .name = "linear1.bias", .len = ff },
            .{ .name = "linear2.weight", .len = hidden * ff },
            .{ .name = "linear2.bias", .len = hidden },
            .{ .name = "norm1.weight", .len = hidden },
            .{ .name = "norm1.bias", .len = hidden },
            .{ .name = "norm2.weight", .len = hidden },
            .{ .name = "norm2.bias", .len = hidden },
        };
        var loaded: [12][]f32 = undefined;
        var loaded_n: usize = 0;
        defer for (loaded[0..loaded_n]) |values| allocator.free(values);
        for (parts, 0..) |part, pi| {
            loaded[pi] = try readF32(allocator, try std.fmt.bufPrint(&name_buf, "m2p.transformer_layers.{d}.{s}.f32.bin", .{ i, part.name }), part.len);
            loaded_n = pi + 1;
        }
        block.* = try shine_train.m2pBlockVariables(ctx, sh_config, .{
            .in_w = loaded[0],
            .in_b = loaded[1],
            .out_w = loaded[2],
            .out_b = loaded[3],
            .ff1_w = loaded[4],
            .ff1_b = loaded[5],
            .ff2_w = loaded[6],
            .ff2_b = loaded[7],
            .norm1_w = loaded[8],
            .norm1_b = loaded[9],
            .norm2_w = loaded[10],
            .norm2_b = loaded[11],
        });
        m2p_built = i + 1;
    }

    const lp_values = try readF32(allocator, "m2p.layer_pe.f32.bin", base.num_layers * hidden);
    defer allocator.free(lp_values);
    var layer_pe = try fucina.Tensor(.{ .layer, .embed }).variableFromSlice(ctx, .{ base.num_layers, hidden }, lp_values);
    errdefer layer_pe.deinit();
    const tp_values = try readF32(allocator, "m2p.token_pe.f32.bin", sh_config.num_mem_token * hidden);
    defer allocator.free(tp_values);
    var token_pe = try fucina.Tensor(.{ .mem, .embed }).variableFromSlice(ctx, .{ sh_config.num_mem_token, hidden }, tp_values);
    errdefer token_pe.deinit();

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

const GradCheck = struct {
    worst_rel: f32 = 0,
    worst_name: []const u8 = "",

    fn compare(self: *GradCheck, name: []const u8, got: []const f32, want: []const f32) !void {
        try std.testing.expectEqual(want.len, got.len);
        var max_abs: f32 = 0;
        var max_ref: f32 = 0;
        for (got, want) |g, w| {
            max_abs = @max(max_abs, @abs(g - w));
            max_ref = @max(max_ref, @abs(w));
        }
        const rel = if (max_ref == 0) max_abs else max_abs / max_ref;
        if (rel > self.worst_rel) {
            self.worst_rel = rel;
            self.worst_name = name;
        }
        try std.testing.expect(rel < 1e-3);
    }
};

fn goldenGate(checkpoint_layers: bool) !void {
    const allocator = std.heap.smp_allocator;
    // Presence probe before any construction.
    {
        const probe = std.Io.Dir.cwd().readFileAlloc(std.testing.io, goldens_dir ++ "/manifest.json", allocator, .limited(1 << 20)) catch return error.SkipZigTest;
        allocator.free(probe);
    }
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try buildModel(&ctx, allocator);
    defer model.deinit();
    var trainer = try buildTrainer(&ctx, allocator, &model);
    defer trainer.deinit();
    trainer.checkpoint_layers = checkpoint_layers;

    const evidence_i32 = try readIdsI32(allocator, "evidence_ids.i32.bin");
    defer allocator.free(evidence_i32);
    const input_i32 = try readIdsI32(allocator, "input_ids.i32.bin");
    defer allocator.free(input_i32);
    const labels_i32 = try readIdsI32(allocator, "labels.i32.bin");
    defer allocator.free(labels_i32);

    const evidence = try allocator.alloc(usize, evidence_i32.len);
    defer allocator.free(evidence);
    for (evidence, evidence_i32) |*d, s| d.* = @intCast(s);
    const inputs = try allocator.alloc(usize, input_i32.len);
    defer allocator.free(inputs);
    for (inputs, input_i32) |*d, s| d.* = @intCast(s);
    // The reference dumps HF-style labels (aligned with inputs, -100
    // masked) and its loss shifts internally; the Zig trainer takes
    // PRE-SHIFTED next tokens.
    const labels = try allocator.alloc(usize, labels_i32.len);
    defer allocator.free(labels);
    for (labels, 0..) |*d, t| {
        d.* = if (t + 1 < labels_i32.len and labels_i32[t + 1] >= 0)
            @intCast(labels_i32[t + 1])
        else
            qwen3_train.ignore_index;
    }

    const golden_loss = try readF32(allocator, "loss.f32.bin", 1);
    defer allocator.free(golden_loss);

    var loss_value: f32 = 0;
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const loss = try trainer.loss(&ctx, evidence, inputs, labels);
        loss_value = try loss.item();
        try loss.backward(&ctx);
    }
    std.debug.print("shine train loss: got {d} want {d}\n", .{ loss_value, golden_loss[0] });
    try std.testing.expect(@abs(loss_value - golden_loss[0]) < 1e-4 * @max(1.0, @abs(golden_loss[0])));

    var check = GradCheck{};
    var name_buf: [128]u8 = undefined;
    const groups = [_][]const u8{ "attention.q", "attention.k", "attention.v", "attention.o", "mlp.gate", "mlp.up", "mlp.down" };
    for (trainer.metalora.layers, 0..) |*layer, i| {
        inline for (shine.modules, groups, 0..) |module, group, gi| {
            _ = module;
            const dims = shine.moduleDims(base, shine.modules[gi]);
            const pair = layer.pair(shine.modules[gi]);
            var ga = (try pair.a.grad(&ctx)) orelse return error.MissingGrad;
            defer ga.deinit();
            const want_a = try readF32(allocator, try std.fmt.bufPrint(&name_buf, "grad.metalora.{d}.{s}.A.f32.bin", .{ i, group }), dims[0] * sh_config.metalora_r);
            defer allocator.free(want_a);
            try check.compare("metalora.A", try ga.dataConst(), want_a);
            var gb = (try pair.b.grad(&ctx)) orelse return error.MissingGrad;
            defer gb.deinit();
            const want_b = try readF32(allocator, try std.fmt.bufPrint(&name_buf, "grad.metalora.{d}.{s}.B.f32.bin", .{ i, group }), sh_config.metalora_r * dims[1]);
            defer allocator.free(want_b);
            try check.compare("metalora.B", try gb.dataConst(), want_b);
        }
    }

    const m2p_parts = [_]struct { name: []const u8, len: usize }{
        .{ .name = "self_attn.in_proj_weight", .len = 3 * 64 * 64 },
        .{ .name = "self_attn.in_proj_bias", .len = 3 * 64 },
        .{ .name = "self_attn.out_proj.weight", .len = 64 * 64 },
        .{ .name = "self_attn.out_proj.bias", .len = 64 },
        .{ .name = "linear1.weight", .len = 128 * 64 },
        .{ .name = "linear1.bias", .len = 128 },
        .{ .name = "linear2.weight", .len = 64 * 128 },
        .{ .name = "linear2.bias", .len = 64 },
        .{ .name = "norm1.weight", .len = 64 },
        .{ .name = "norm1.bias", .len = 64 },
        .{ .name = "norm2.weight", .len = 64 },
        .{ .name = "norm2.bias", .len = 64 },
    };
    for (trainer.m2p, 0..) |*block, i| {
        const tensors = [_]*const fucina.Tensor(.{ .proj, .embed }){&block.in_w};
        _ = tensors;
        inline for (.{ "in_w", "in_b", "out_w", "out_b", "ff1_w", "ff1_b", "ff2_w", "ff2_b", "norm1_w", "norm1_b", "norm2_w", "norm2_b" }, 0..) |field, pi| {
            var grad = (try @field(block.*, field).grad(&ctx)) orelse return error.MissingGrad;
            defer grad.deinit();
            const want = try readF32(allocator, try std.fmt.bufPrint(&name_buf, "grad.m2p.transformer_layers.{d}.{s}.f32.bin", .{ i, m2p_parts[pi].name }), m2p_parts[pi].len);
            defer allocator.free(want);
            try check.compare(m2p_parts[pi].name, try grad.dataConst(), want);
        }
    }
    {
        var grad = (try trainer.layer_pe.grad(&ctx)) orelse return error.MissingGrad;
        defer grad.deinit();
        const want = try readF32(allocator, "grad.m2p.layer_pe.f32.bin", base.num_layers * 64);
        defer allocator.free(want);
        try check.compare("layer_pe", try grad.dataConst(), want);
    }
    {
        var grad = (try trainer.token_pe.grad(&ctx)) orelse return error.MissingGrad;
        defer grad.deinit();
        const want = try readF32(allocator, "grad.m2p.token_pe.f32.bin", sh_config.num_mem_token * 64);
        defer allocator.free(want);
        try check.compare("token_pe", try grad.dataConst(), want);
    }
    std.debug.print("shine train grads: worst rel {d} ({s})\n", .{ check.worst_rel, check.worst_name });
}

test "SHINE training step: loss and every trainable gradient match the PyTorch reference" {
    try goldenGate(false);
}

test "SHINE training step under per-layer checkpointing: same goldens" {
    try goldenGate(true);
}

test "SHINE packed step: two copies of one example equal the single-example step" {
    const allocator = std.heap.smp_allocator;
    {
        const probe = std.Io.Dir.cwd().readFileAlloc(std.testing.io, goldens_dir ++ "/manifest.json", allocator, .limited(1 << 20)) catch return error.SkipZigTest;
        allocator.free(probe);
    }
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try buildModel(&ctx, allocator);
    defer model.deinit();

    const evidence_i32 = try readIdsI32(allocator, "evidence_ids.i32.bin");
    defer allocator.free(evidence_i32);
    const input_i32 = try readIdsI32(allocator, "input_ids.i32.bin");
    defer allocator.free(input_i32);
    const labels_i32 = try readIdsI32(allocator, "labels.i32.bin");
    defer allocator.free(labels_i32);
    const evidence = try allocator.alloc(usize, evidence_i32.len);
    defer allocator.free(evidence);
    for (evidence, evidence_i32) |*d, v| d.* = @intCast(v);
    const inputs = try allocator.alloc(usize, input_i32.len);
    defer allocator.free(inputs);
    for (inputs, input_i32) |*d, v| d.* = @intCast(v);
    const labels = try allocator.alloc(usize, labels_i32.len);
    defer allocator.free(labels);
    for (labels, 0..) |*d, t| {
        d.* = if (t + 1 < labels_i32.len and labels_i32[t + 1] >= 0)
            @intCast(labels_i32[t + 1])
        else
            qwen3_train.ignore_index;
    }
    const example = shine_train.Example{ .evidence = evidence, .input = inputs, .labels = labels };

    // Three trainers with the same weights: one single step, one packed
    // pair of the SAME example (plain), one packed pair under per-layer
    // checkpointing. Token-mean loss and every gradient must agree with
    // the single step (the pack contributes each example at half weight,
    // twice; recompute-in-backward must not change the math).
    var solo = try buildTrainer(&ctx, allocator, &model);
    defer solo.deinit();
    var packed_trainer = try buildTrainer(&ctx, allocator, &model);
    defer packed_trainer.deinit();
    var ckpt_trainer = try buildTrainer(&ctx, allocator, &model);
    defer ckpt_trainer.deinit();
    ckpt_trainer.checkpoint_layers = true;

    var solo_loss: f32 = 0;
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const l = try solo.loss(&ctx, evidence, inputs, labels);
        solo_loss = try l.item();
        try l.backward(&ctx);
    }
    inline for (.{ &packed_trainer, &ckpt_trainer }, .{ "packed", "packed+ckpt" }) |variant, label| {
        var variant_loss: f32 = 0;
        {
            const scope = ctx.openExecScope();
            defer ctx.closeExecScope(scope);
            const l = try variant.lossPacked(&ctx, &.{ example, example });
            variant_loss = try l.item();
            try l.backward(&ctx);
        }
        std.debug.print("shine packed-pair loss ({s}): solo {d} variant {d}\n", .{ label, solo_loss, variant_loss });
        try std.testing.expect(@abs(solo_loss - variant_loss) < 1e-5 * @max(1.0, @abs(solo_loss)));

        var worst: f32 = 0;
        for (solo.metalora.layers, variant.metalora.layers) |*sl, *pl| {
            inline for (shine.modules) |module| {
                inline for (.{ "a", "b" }) |half| {
                    var sg = (try @field(@field(sl.*, @tagName(module)), half).grad(&ctx)) orelse return error.MissingGrad;
                    defer sg.deinit();
                    var pg = (try @field(@field(pl.*, @tagName(module)), half).grad(&ctx)) orelse return error.MissingGrad;
                    defer pg.deinit();
                    var max_abs: f32 = 0;
                    var max_ref: f32 = 0;
                    for (try sg.dataConst(), try pg.dataConst()) |sv, pv| {
                        max_abs = @max(max_abs, @abs(sv - pv));
                        max_ref = @max(max_ref, @abs(sv));
                    }
                    const rel = if (max_ref == 0) max_abs else max_abs / max_ref;
                    worst = @max(worst, rel);
                    try std.testing.expect(rel < 1e-4);
                }
            }
        }
        std.debug.print("shine packed-pair grads ({s}): worst rel {d}\n", .{ label, worst });
    }
}

/// Deterministic filler for the timing trainer's variables (values are
/// irrelevant to the measurement; the graph shape is what costs).
fn fillNoise(values: []f32, seed: u32) void {
    var state: u32 = seed | 1;
    for (values) |*v| {
        state = state *% 1664525 +% 1013904223;
        v.* = (@as(f32, @floatFromInt(state >> 9)) / (1 << 23) - 0.5) * 0.02;
    }
}

fn noiseSlice(allocator: std.mem.Allocator, len: usize, seed: u32) ![]f32 {
    const values = try allocator.alloc(f32, len);
    fillNoise(values, seed);
    return values;
}

test "SHINE training step timing at 0.6B (informational)" {
    // Meaningful only optimized; the 0.6B graph is ~25x slower in Debug.
    if (@import("builtin").mode == .Debug) return error.SkipZigTest;
    const allocator = std.heap.smp_allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var file = fucina.gguf.File.loadMmap(allocator, std.testing.io, "models/Qwen3-0.6B-BF16.gguf") catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer file.deinit();
    const cfg = try qwen3.Config.fromGguf(&file);
    var model = try qwen3.Model.loadGgufFromFile(&ctx, &file, cfg);
    defer model.deinit();

    const sh = shine.Config{
        .hidden_size = cfg.hidden_size,
        .num_layers = cfg.num_layers,
        .num_mem_token = shine.loraParamsPerLayer(cfg, 8) / cfg.hidden_size,
        .metalora_r = 128,
        .lora_r = 8,
        .scale = 0.001,
        .m2p_layers = 4,
        .m2p_heads = 8,
        .m2p_ffn = 2048,
        .m2p_eps = 1e-5,
        .layer_transformer_first = true,
    };

    // Random-valued trainable leaves at the smoke-run geometry.
    const layers = try allocator.alloc(shine.LayerLora, cfg.num_layers);
    for (layers, 0..) |*layer, i| {
        var pairs: [7]shine.LoraPair = undefined;
        inline for (shine.modules, 0..) |_, gi| {
            const dims = shine.moduleDims(cfg, shine.modules[gi]);
            const a_values = try noiseSlice(allocator, dims[0] * sh.metalora_r, @intCast(i * 100 + gi * 2));
            defer allocator.free(a_values);
            const b_values = try allocator.alloc(f32, sh.metalora_r * dims[1]);
            defer allocator.free(b_values);
            @memset(b_values, 0);
            pairs[gi] = try shine_train.pairVariables(&ctx, dims[0], sh.metalora_r, dims[1], a_values, b_values);
        }
        layer.* = .{ .q = pairs[0], .k = pairs[1], .v = pairs[2], .o = pairs[3], .gate = pairs[4], .up = pairs[5], .down = pairs[6] };
    }
    const metalora = shine.LoraSet{ .allocator = allocator, .layers = layers };

    const m2p = try allocator.alloc(shine.M2pBlock, sh.m2p_layers);
    for (m2p, 0..) |*block, i| {
        const h = sh.hidden_size;
        const ff = sh.m2p_ffn;
        const lens = [_]usize{ 3 * h * h, 3 * h, h * h, h, ff * h, ff, h * ff, h, h, h, h, h };
        var bufs: [12][]f32 = undefined;
        inline for (0..12) |pi| bufs[pi] = try noiseSlice(allocator, lens[pi], @intCast(7000 + i * 20 + pi));
        defer inline for (0..12) |pi| allocator.free(bufs[pi]);
        block.* = try shine_train.m2pBlockVariables(&ctx, sh, .{
            .in_w = bufs[0], .in_b = bufs[1], .out_w = bufs[2], .out_b = bufs[3],
            .ff1_w = bufs[4], .ff1_b = bufs[5], .ff2_w = bufs[6], .ff2_b = bufs[7],
            .norm1_w = bufs[8], .norm1_b = bufs[9], .norm2_w = bufs[10], .norm2_b = bufs[11],
        });
    }
    const lp = try noiseSlice(allocator, cfg.num_layers * sh.hidden_size, 31337);
    defer allocator.free(lp);
    const tp = try noiseSlice(allocator, sh.num_mem_token * sh.hidden_size, 31338);
    defer allocator.free(tp);
    var trainer = shine_train.ShineTrainer{
        .allocator = allocator,
        .model = &model,
        .config = sh,
        .metalora = metalora,
        .m2p = m2p,
        .layer_pe = try fucina.Tensor(.{ .layer, .embed }).variableFromSlice(&ctx, .{ cfg.num_layers, sh.hidden_size }, lp),
        .token_pe = try fucina.Tensor(.{ .mem, .embed }).variableFromSlice(&ctx, .{ sh.num_mem_token, sh.hidden_size }, tp),
    };
    defer trainer.deinit();

    // The smoke recipe's shape: ~512-token context, ~1024-token conversation.
    const evidence = try allocator.alloc(usize, 512);
    defer allocator.free(evidence);
    for (evidence, 0..) |*id, i| id.* = 1000 + (i * 37) % 100_000;
    const inputs = try allocator.alloc(usize, 1024);
    defer allocator.free(inputs);
    for (inputs, 0..) |*id, i| id.* = 2000 + (i * 53) % 100_000;
    const labels = try allocator.alloc(usize, 1024);
    defer allocator.free(labels);
    for (labels, inputs) |*l, id| l.* = id;

    var elapsed_warm: i96 = 0;
    const steps = 3;
    for (0..steps + 1) |step| {
        const t0 = std.Io.Clock.awake.now(std.testing.io).nanoseconds;
        {
            const scope = ctx.openExecScope();
            defer ctx.closeExecScope(scope);
            const loss = try trainer.loss(&ctx, evidence, inputs, labels);
            try loss.backward(&ctx);
        }
        const dt = std.Io.Clock.awake.now(std.testing.io).nanoseconds - t0;
        if (step > 0) elapsed_warm += dt; // step 0 warms the FrozenCache
    }
    const ms_per_step = @as(f64, @floatFromInt(elapsed_warm)) / steps / 1e6;
    const tokens_per_example: f64 = 512 + 1024;
    std.debug.print("shine train 0.6B step: {d:.0} ms, {d:.1} tok/s (evidence 512 + conversation 1024)\n", .{ ms_per_step, tokens_per_example / (ms_per_step / 1000.0) });

    // Packed steps run CHECKPOINTED (recompute-in-backward per layer):
    // only the layer boundaries stay retained, so multi-example packs fit
    // in RAM (the plain packed graph exceeds 64 GB at width 2), and the
    // grad-free block forward makes the packed GEMMs GPU-eligible. Width 4
    // clears the GPU dispatch crossover (4096*3072*1024 > 2^32). Opt-in
    // via FUCINA_SHINE_PACKED_BENCH=1 — a long 0.6B battery.
    if (!fucina.parallel.envFlag("FUCINA_SHINE_PACKED_BENCH")) return;
    trainer.checkpoint_layers = true;
    const example = shine_train.Example{ .evidence = evidence, .input = inputs, .labels = labels };
    const widths = [_]usize{ 1, 2, 4 };
    for (widths) |width| {
        var pack: [4]shine_train.Example = undefined;
        for (pack[0..width]) |*e| e.* = example;
        var packed_ns: i96 = 0;
        for (0..steps + 1) |step| {
            const t0 = std.Io.Clock.awake.now(std.testing.io).nanoseconds;
            {
                const scope = ctx.openExecScope();
                defer ctx.closeExecScope(scope);
                const l = try trainer.lossPacked(&ctx, pack[0..width]);
                try l.backward(&ctx);
            }
            trainer.freeTransient();
            const dt = std.Io.Clock.awake.now(std.testing.io).nanoseconds - t0;
            if (step > 0) packed_ns += dt;
        }
        const packed_ms = @as(f64, @floatFromInt(packed_ns)) / steps / 1e6;
        const w: f64 = @floatFromInt(width);
        std.debug.print("shine train 0.6B ckpt packed-{d} step: {d:.0} ms total, {d:.0} ms/example, {d:.1} tok/s\n", .{ width, packed_ms, packed_ms / w, w * tokens_per_example / (packed_ms / 1000.0) });
    }
}
