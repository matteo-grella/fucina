//! Behavioral tests for the Gemma4 LoRA fine-tuning surface (`gemma4_train.zig`):
//! end-to-end loss + backward gradient-evidence smoke over tiny synthetic
//! dense and MoE models built from random GGUF-shaped weights.
const std = @import("std");
const gemma4_train = @import("gemma4_train.zig");

const fucina = @import("fucina");
const gemma4 = @import("gemma4.zig");
const cartridge = @import("../cartridge.zig");
const kv_cache_mod = @import("../kv_cache.zig");
const gemma_moe = @import("moe.zig");
const weights = @import("../weights.zig");

const Allocator = std.mem.Allocator;
const backend_mod = fucina.internal.backend_mod;
const ExecContext = fucina.ExecContext;
const LinearWeight = weights.LinearWeight;
const Tag = @TypeOf(.tag);
const optim = fucina.optim;
const rng = fucina.rng;

const Trainer = gemma4_train.Trainer;

const tiny_config = gemma4.Config{
    .vocab_size = 32,
    .hidden_size = 16,
    .num_layers = 1,
    .num_attention_heads = 2,
    .head_dim_global = 8,
    .head_dim_swa = 8,
    .sliding_window = 0,
    .shared_kv_layers = 0,
    .rms_norm_eps = 1e-6,
    .rope_theta = 10_000,
    .rope_theta_swa = 10_000,
    .num_experts = 0,
    .num_experts_used = 0,
    .moe_intermediate_size = 0,
    .intermediate_size = 32,
    .per_layer_input_size = 0,
    .final_logit_softcapping = 0,
};

const tiny_moe_config = gemma4.Config{
    .vocab_size = 48,
    .hidden_size = 256,
    .num_layers = 1,
    .num_attention_heads = 4,
    .head_dim_global = 64,
    .head_dim_swa = 64,
    .sliding_window = 0,
    .shared_kv_layers = 0,
    .rms_norm_eps = 1e-6,
    .rope_theta = 10_000,
    .rope_theta_swa = 10_000,
    .num_experts = 4,
    .num_experts_used = 2,
    .moe_intermediate_size = 32,
    .intermediate_size = 64,
    .per_layer_input_size = 0,
    .final_logit_softcapping = 0,
};

fn randLinear(ctx: *ExecContext, seed: u64, out_dim: usize, in_dim: usize, bound: f32) !LinearWeight {
    const values = try ctx.allocator.alloc(f32, out_dim * in_dim);
    defer ctx.allocator.free(values);
    rng.uniformFill(seed, values, -bound, bound);
    return .{ .f32 = try weights.WeightF32.fromSlice(ctx, .{ out_dim, in_dim }, values) };
}

fn randVector(ctx: *ExecContext, seed: u64, comptime tag: Tag, len: usize) !fucina.Tensor(.{tag}) {
    const values = try ctx.allocator.alloc(f32, len);
    defer ctx.allocator.free(values);
    rng.uniformFill(seed, values, 0.8, 1.2);
    return fucina.Tensor(.{tag}).fromSlice(ctx, .{len}, values);
}

fn buildTinyDenseLayer(ctx: *ExecContext, cfg: gemma4.Config, seed: u64) !gemma4.Layer {
    const q_dim = cfg.num_attention_heads * cfg.head_dim_global;
    const kv_dim = cfg.head_dim_global;

    var attn_norm = try randVector(ctx, rng.at(seed, 0), .embed, cfg.hidden_size);
    errdefer attn_norm.deinit();
    var attn_post_norm = try randVector(ctx, rng.at(seed, 1), .embed, cfg.hidden_size);
    errdefer attn_post_norm.deinit();
    var q_norm = try randVector(ctx, rng.at(seed, 2), .d, cfg.head_dim_global);
    errdefer q_norm.deinit();
    var k_norm = try randVector(ctx, rng.at(seed, 3), .d, cfg.head_dim_global);
    errdefer k_norm.deinit();
    var ffn_norm = try randVector(ctx, rng.at(seed, 4), .embed, cfg.hidden_size);
    errdefer ffn_norm.deinit();
    var ffn_post_norm = try randVector(ctx, rng.at(seed, 5), .embed, cfg.hidden_size);
    errdefer ffn_post_norm.deinit();

    var q_proj = try randLinear(ctx, rng.at(seed, 6), q_dim, cfg.hidden_size, 0.25);
    errdefer q_proj.deinit();
    var k_proj = try randLinear(ctx, rng.at(seed, 7), kv_dim, cfg.hidden_size, 0.25);
    errdefer k_proj.deinit();
    var v_proj = try randLinear(ctx, rng.at(seed, 8), kv_dim, cfg.hidden_size, 0.25);
    errdefer v_proj.deinit();
    var o_proj = try randLinear(ctx, rng.at(seed, 9), cfg.hidden_size, q_dim, 0.25);
    errdefer o_proj.deinit();
    var ffn_gate = try randLinear(ctx, rng.at(seed, 10), cfg.intermediate_size, cfg.hidden_size, 0.25);
    errdefer ffn_gate.deinit();
    var ffn_up = try randLinear(ctx, rng.at(seed, 11), cfg.intermediate_size, cfg.hidden_size, 0.25);
    errdefer ffn_up.deinit();
    var ffn_down = try randLinear(ctx, rng.at(seed, 12), cfg.hidden_size, cfg.intermediate_size, 0.25);
    errdefer ffn_down.deinit();

    return .{
        .attn_norm = attn_norm,
        .attn_post_norm = attn_post_norm,
        .attn_proj = .{ .separate = .{ .q_proj = q_proj, .k_proj = k_proj, .v_proj = v_proj } },
        .q_norm = q_norm,
        .k_norm = k_norm,
        .o_proj = o_proj,
        .ffn_norm = ffn_norm,
        .ffn_gate = ffn_gate,
        .ffn_up = ffn_up,
        .ffn_down = ffn_down,
        .ffn_post_norm = ffn_post_norm,
        .moe = null,
        .ple = null,
        .out_scale = null,
    };
}

fn buildTinyMoe(ctx: *ExecContext, cfg: gemma4.Config, seed: u64) !gemma4.MoeFfn {
    const allocator = ctx.allocator;
    const qm = backend_mod.quantized_matmul;
    const hidden = cfg.hidden_size;
    const n_expert = cfg.num_experts;
    const n_ff = cfg.moe_intermediate_size;
    const bpr_gu = hidden / 256;
    const bpr_dn = n_ff / 32;

    var router = try randLinear(ctx, rng.at(seed, 0), n_expert, hidden, 0.1);
    errdefer router.deinit();
    var router_weight = try randVector(ctx, rng.at(seed, 1), .embed, hidden);
    errdefer router_weight.deinit();
    var pre_norm_2 = try randVector(ctx, rng.at(seed, 2), .embed, hidden);
    errdefer pre_norm_2.deinit();
    var post_norm_1 = try randVector(ctx, rng.at(seed, 3), .embed, hidden);
    errdefer post_norm_1.deinit();
    var post_norm_2 = try randVector(ctx, rng.at(seed, 4), .embed, hidden);
    errdefer post_norm_2.deinit();

    const gate = try allocator.alloc(fucina.QuantizedMatmulRhsQ6_Kx4, 0);
    errdefer allocator.free(gate);
    const up = try allocator.alloc(fucina.QuantizedMatmulRhsQ6_Kx4, 0);
    errdefer allocator.free(up);
    const down = try allocator.alloc(fucina.QuantizedMatmulRhsQ8_0x4, 0);
    errdefer allocator.free(down);

    const down_scale = try allocator.alloc(f32, n_expert);
    errdefer allocator.free(down_scale);
    rng.uniformFill(rng.at(seed, 5), down_scale, 0.8, 1.2);

    const gu_blocks = try allocator.alloc(fucina.BlockQ6_K, n_expert * 2 * n_ff * bpr_gu);
    errdefer allocator.free(gu_blocks);
    const dn_blocks = try allocator.alloc(fucina.BlockQ8_0, n_expert * hidden * bpr_dn);
    errdefer allocator.free(dn_blocks);
    {
        const row_gu = try allocator.alloc(f32, hidden);
        defer allocator.free(row_gu);
        for (0..n_expert * 2 * n_ff) |r| {
            rng.uniformFill(rng.at(seed, 100 + r), row_gu, -0.08, 0.08);
            try qm.quantizeRowQ6_KInto(gu_blocks[r * bpr_gu ..][0..bpr_gu], row_gu);
        }

        const row_dn = try allocator.alloc(f32, n_ff);
        defer allocator.free(row_dn);
        for (0..n_expert * hidden) |r| {
            rng.uniformFill(rng.at(seed, 1000 + r), row_dn, -0.08, 0.08);
            try qm.quantizeRowQ8_0Into(dn_blocks[r * bpr_dn ..][0..bpr_dn], row_dn);
        }
    }

    return .{
        .router = router,
        .router_weight = router_weight,
        .pre_norm_2 = pre_norm_2,
        .post_norm_1 = post_norm_1,
        .post_norm_2 = post_norm_2,
        .gate = gate,
        .up = up,
        .down = down,
        .down_scale = down_scale,
        .gpu_weights = .{ .gu = .{ .q6_k = gu_blocks }, .dn_blocks = dn_blocks, .device_owned = false },
    };
}

fn buildTinyModel(ctx: *ExecContext, cfg: gemma4.Config, seed: u64, with_moe: bool) !gemma4.Model {
    const allocator = ctx.allocator;

    var token_embedding = try randLinear(ctx, rng.at(seed, 100), cfg.vocab_size, cfg.hidden_size, 0.35);
    errdefer token_embedding.deinit();
    var output_norm = try randVector(ctx, rng.at(seed, 101), .embed, cfg.hidden_size);
    errdefer output_norm.deinit();
    var output = try randLinear(ctx, rng.at(seed, 102), cfg.vocab_size, cfg.hidden_size, 0.35);
    errdefer output.deinit();

    const swa = try allocator.alloc(bool, cfg.num_layers);
    errdefer allocator.free(swa);
    const head_dim = try allocator.alloc(usize, cfg.num_layers);
    errdefer allocator.free(head_dim);
    const kv_heads = try allocator.alloc(usize, cfg.num_layers);
    errdefer allocator.free(kv_heads);
    const has_kv = try allocator.alloc(bool, cfg.num_layers);
    errdefer allocator.free(has_kv);
    const kv_ref = try allocator.alloc(usize, cfg.num_layers);
    errdefer allocator.free(kv_ref);
    for (0..cfg.num_layers) |i| {
        swa[i] = false;
        head_dim[i] = cfg.head_dim_global;
        kv_heads[i] = 1;
        has_kv[i] = true;
        kv_ref[i] = i;
    }
    var geom = gemma4.LayerGeometry{ .is_swa = swa, .head_dim = head_dim, .kv_heads = kv_heads, .has_kv = has_kv, .kv_ref = kv_ref };
    errdefer geom.deinit(allocator);

    const layers = try allocator.alloc(gemma4.Layer, cfg.num_layers);
    errdefer allocator.free(layers);
    var built: usize = 0;
    errdefer for (layers[0..built]) |*layer| layer.deinit(allocator);
    for (layers, 0..) |*layer, layer_i| {
        layer.* = try buildTinyDenseLayer(ctx, cfg, rng.at(seed, layer_i));
        built += 1;
        if (with_moe) layer.moe = try buildTinyMoe(ctx, cfg, rng.at(seed, 10_000 + layer_i));
    }

    return .{
        .allocator = allocator,
        .config = cfg,
        .geom = geom,
        .token_embedding = token_embedding,
        .output_norm = output_norm,
        .output = output,
        .rope_freqs = null,
        .layers = layers,
        .ple = null,
        .weight_mapping = null,
    };
}

fn buildTinyDenseModel(ctx: *ExecContext, seed: u64) !gemma4.Model {
    return buildTinyModel(ctx, tiny_config, seed, false);
}

fn buildTinyMoeModel(ctx: *ExecContext, seed: u64) !gemma4.Model {
    return buildTinyModel(ctx, tiny_moe_config, seed, true);
}

test "gemma4_train dense trainer loss backward smoke" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try buildTinyDenseModel(&ctx, 0x47454d4d4134);
    defer model.deinit();
    var trainer = try Trainer(.{ .q = true, .v = true }).init(&ctx, &model, .{ .rank = 2, .alpha = 4 }, 17);
    defer trainer.deinit();

    var opt = optim.AdamW.init(allocator, .{ .lr = 0.02, .weight_decay = 0 });
    defer opt.deinit();
    try trainer.registerAllParams(&opt);

    const inputs = [_]usize{ 1, 5, 9, 13, 2, 7 };
    const labels = [_]usize{ 5, 9, 13, 2, 7, 3 };
    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    const loss = try trainer.loss(&ctx, &inputs, &labels);
    try loss.backward(&ctx);

    var saw_grad = false;
    for (trainer.adapters) |*ads| {
        if (try ads.q.b.grad(&ctx)) |g_value| {
            var g = g_value;
            defer g.deinit();
            const values = try g.dataConst();
            for (values) |v| {
                if (v != 0) {
                    saw_grad = true;
                    break;
                }
            }
        }
    }
    try std.testing.expect(saw_grad);
}

/// The adapter checkpoint's expected entry names for (q, v) targets over
/// `tiny_config`'s single layer, in emission order. This is the on-disk
/// schema (docs/TRAINING.md §8): renaming or reordering entries breaks resuming
/// from older checkpoints.
const expected_adapter_names = [_][]const u8{
    "layers.0.q.lora_a", "layers.0.q.lora_b", "layers.0.v.lora_a", "layers.0.v.lora_b",
};

const QvTrainer = Trainer(.{ .q = true, .v = true });

/// The adapter state-dict entries in schema order, named from `names` — the
/// explicit reference writer the golden schema test serializes against.
fn adapterEntries(trainer: *const QvTrainer, names: []const []const u8) ![expected_adapter_names.len]optim.NamedTensor {
    var entries: [expected_adapter_names.len]optim.NamedTensor = undefined;
    for (trainer.adapters, 0..) |*ads, layer_i| {
        entries[layer_i * 4 + 0] = try optim.NamedTensor.of(names[layer_i * 4 + 0], &ads.q.a);
        entries[layer_i * 4 + 1] = try optim.NamedTensor.of(names[layer_i * 4 + 1], &ads.q.b);
        entries[layer_i * 4 + 2] = try optim.NamedTensor.of(names[layer_i * 4 + 2], &ads.v.a);
        entries[layer_i * 4 + 3] = try optim.NamedTensor.of(names[layer_i * 4 + 3], &ads.v.b);
    }
    return entries;
}

test "gemma4_train adapter checkpoint schema: byte-identical to the explicit named-tensor writer" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try buildTinyDenseModel(&ctx, 0x60D5);
    defer model.deinit();
    var trainer = try QvTrainer.init(&ctx, &model, .{ .rank = 2, .alpha = 4 }, 5);
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

test "gemma4_train adapter load aliases: a renamed stream entry resolves to the current path" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try buildTinyDenseModel(&ctx, 0xA11A);
    defer model.deinit();
    var source = try QvTrainer.init(&ctx, &model, .{ .rank = 2, .alpha = 4 }, 7);
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
    var fresh = try QvTrainer.init(&ctx, &model, .{ .rank = 2, .alpha = 4 }, 99);
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

test "gemma4_train moe trainer loss backward smoke" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try buildTinyMoeModel(&ctx, 0x47454d4d414d4f45);
    defer model.deinit();
    var trainer = try Trainer(.{ .q = true, .v = true }).init(&ctx, &model, .{ .rank = 2, .alpha = 4 }, 23);
    defer trainer.deinit();

    const inputs = [_]usize{ 1, 5, 9, 13 };
    const labels = [_]usize{ 5, 9, 13, 2 };
    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    const loss = try trainer.loss(&ctx, &inputs, &labels);
    try loss.backward(&ctx);

    var saw_grad = false;
    for (trainer.adapters) |*ads| {
        if (try ads.q.b.grad(&ctx)) |g_value| {
            var g = g_value;
            defer g.deinit();
            const values = try g.dataConst();
            for (values) |v| {
                if (v != 0) {
                    saw_grad = true;
                    break;
                }
            }
        }
    }
    try std.testing.expect(saw_grad);
}

test "gemma4_train dense trainer lossExt smoke: defaults match loss, sum/scale backward flows" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try buildTinyDenseModel(&ctx, 0x47454d4d4135);
    defer model.deinit();
    var base = try Trainer(.{ .q = true, .v = true }).init(&ctx, &model, .{ .rank = 2, .alpha = 4 }, 17);
    defer base.deinit();
    var ext = try Trainer(.{ .q = true, .v = true }).init(&ctx, &model, .{ .rank = 2, .alpha = 4 }, 17);
    defer ext.deinit();

    const inputs = [_]usize{ 1, 5, 9, 13, 2, 7 };
    const labels = [_]usize{ 5, 9, 13, 2, 7, 3 };
    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);

    // Defaults reproduce `loss` bitwise (same trainer state and seed).
    const base_l = try base.loss(&ctx, &inputs, &labels);
    const ext_l = try ext.lossExt(&ctx, &inputs, &labels, .{});
    try std.testing.expectEqual(try base_l.item(), try ext_l.item());

    // Accumulation arm: `.sum` + 1/valid scaling backpropagates into the
    // adapters (mirrors the plain loss-backward smoke above).
    const scaled = try ext.lossExt(&ctx, &inputs, &labels, .{
        .reduction = .sum,
        .loss_scale = 1.0 / @as(f32, @floatFromInt(labels.len)),
    });
    try std.testing.expectApproxEqRel(try ext_l.item(), try scaled.item(), 1e-6);
    try scaled.backward(&ctx);

    var saw_grad = false;
    for (ext.adapters) |*ads| {
        if (try ads.q.b.grad(&ctx)) |g_value| {
            var g = g_value;
            defer g.deinit();
            const values = try g.dataConst();
            for (values) |v| {
                if (v != 0) {
                    saw_grad = true;
                    break;
                }
            }
        }
    }
    try std.testing.expect(saw_grad);
}

const CartridgeTrainer = Trainer(.{ .q = false, .v = false });

test "gemma4 corpus-init cartridge is logit-equivalent to a real prefill (SWA + global)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var cfg = tiny_config;
    cfg.num_layers = 2;
    cfg.sliding_window = 4;
    var model = try buildTinyModel(&ctx, cfg, 0xca27a1d6e, false);
    defer model.deinit();
    // Layer 1 becomes a local SWA layer whose 4-token window CUTS through
    // the prefix (a query at position 5 sees prefix rows 2..4 only) — the
    // gemma-specific arm of the equivalence.
    model.geom.is_swa[1] = true;

    var trainer = try CartridgeTrainer.init(&ctx, &model, .{ .rank = 1, .alpha = 1 }, 7);
    defer trainer.deinit();

    const full = [_]usize{ 14, 6, 25, 0, 13, 1, 26, 22, 5, 29, 31, 2 };
    const p = 5;
    const suffix = full[p..];

    var teacher = try trainer.evalLogitsExt(&ctx, &full, .{});
    defer teacher.deinit();

    var cart = try trainer.initCartridge(&ctx, full[0..p], 1);
    defer cart.deinit();
    try std.testing.expectEqual(@as(usize, p), cart.p);

    var student = try trainer.evalLogitsExt(&ctx, suffix, .{ .cartridge = &cart });
    defer student.deinit();

    const vocab = model.config.vocab_size;
    const teacher_data = try teacher.dataConst();
    const student_data = try student.dataConst();
    try std.testing.expectEqual(@as(usize, suffix.len * vocab), student_data.len);
    for (student_data, teacher_data[p * vocab ..], 0..) |got, want, i| {
        const tol = 1e-5 + 1e-4 * @abs(want);
        if (@abs(got - want) > tol) {
            std.debug.print("gemma4 prefill-equivalence mismatch at {d}: want {d} got {d}\n", .{ i, want, got });
            return error.PrefillEquivalenceMismatch;
        }
    }
}

test "gemma4 corpus-init cartridge equivalence with rope frequency factors" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var cfg = tiny_config;
    cfg.num_layers = 2;
    cfg.sliding_window = 4;
    var model = try buildTinyModel(&ctx, cfg, 0xca27f4c705, false);
    defer model.deinit();
    model.geom.is_swa[0] = true;
    // Non-trivial per-frequency factors on the global-layer table (the
    // gemma-4 longrope-style rope_freqs arm), combined with the student's
    // offset positions.
    model.rope_freqs = try fucina.Tensor(.{.rope}).fromSlice(&ctx, .{cfg.head_dim_global / 2}, &.{ 1.0, 2.0, 0.5, 1.5 });

    var trainer = try CartridgeTrainer.init(&ctx, &model, .{ .rank = 1, .alpha = 1 }, 7);
    defer trainer.deinit();

    const full = [_]usize{ 14, 6, 25, 0, 13, 1, 26, 22, 5, 29, 31, 2 };
    const p = 5;
    const suffix = full[p..];

    var teacher = try trainer.evalLogitsExt(&ctx, &full, .{});
    defer teacher.deinit();
    var cart = try trainer.initCartridge(&ctx, full[0..p], 1);
    defer cart.deinit();
    var student = try trainer.evalLogitsExt(&ctx, suffix, .{ .cartridge = &cart });
    defer student.deinit();

    const vocab = model.config.vocab_size;
    const teacher_data = try teacher.dataConst();
    const student_data = try student.dataConst();
    for (student_data, teacher_data[p * vocab ..]) |got, want| {
        try std.testing.expect(@abs(got - want) <= 1e-5 + 1e-4 * @abs(want));
    }
}

test "gemma4 distillation pulls a random cartridge toward the teacher (MoE, sink frozen)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try buildTinyMoeModel(&ctx, 0xca27d157111);
    defer model.deinit();
    var trainer = try CartridgeTrainer.init(&ctx, &model, .{ .rank = 1, .alpha = 1 }, 9);
    defer trainer.deinit();

    const full = [_]usize{ 40, 6, 25, 0, 13, 1, 26, 22, 5, 29, 44, 2 };
    const p = 5;
    const suffix = full[p..];

    var teacher = try trainer.evalLogitsExt(&ctx, &full, .{});
    defer teacher.deinit();
    const vocab = model.config.vocab_size;
    const teacher_data = try teacher.dataConst();
    var builder = cartridge.TargetsBuilder.init(allocator);
    defer builder.deinit();
    for (1..suffix.len) |j| {
        const row = teacher_data[(p + j - 1) * vocab ..][0..vocab];
        try builder.appendRow(j, row, 5, 0.99);
    }

    var cart = try cartridge.Cartridge.initRandom(
        &ctx,
        allocator,
        model.config.num_layers,
        1,
        p,
        model.geom.kv_heads[0],
        model.geom.head_dim[0],
        123,
        0.05,
    );
    defer cart.deinit();

    var opt = optim.AdamW.init(allocator, .{ .lr = 2e-2, .weight_decay = 0 });
    defer opt.deinit();
    try cart.registerParams(&opt);

    const sink_before = try allocator.dupe(f32, try cart.layers[0].k_sink.?.dataConst());
    defer allocator.free(sink_before);
    const k_before = try allocator.dupe(f32, try cart.layers[0].k.dataConst());
    defer allocator.free(k_before);

    var first: f32 = 0;
    var last: f32 = 0;
    for (0..8) |step_i| {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        var loss = try trainer.distillLoss(&ctx, suffix, &cart, builder.targets(), null, .{});
        defer loss.deinit();
        try loss.backward(&ctx);
        const value = try loss.item();
        if (step_i == 0) first = value;
        last = value;
        try opt.step(&ctx);
        opt.zeroGrad();
    }
    try std.testing.expect(last < first);
    try std.testing.expectEqualSlices(f32, sink_before, try cart.layers[0].k_sink.?.dataConst());
    var changed = false;
    for (k_before, try cart.layers[0].k.dataConst()) |before, after| {
        if (before != after) {
            changed = true;
            break;
        }
    }
    try std.testing.expect(changed);
}

test "gemma4 forwardStepBatchSpans matches per-stream forwardStepAllLogits" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var cfg = tiny_config;
    cfg.num_layers = 2;
    cfg.sliding_window = 4;
    var model = try buildTinyModel(&ctx, cfg, 0xba7c4592a45, false);
    defer model.deinit();
    model.geom.is_swa[1] = true;

    const prompts = [2][]const usize{ &.{ 3, 9, 27, 5 }, &.{ 11, 2, 30, 8, 17, 1 } };
    const spans_tokens = [2][]const usize{ &.{ 7, 21 }, &.{ 4, 15, 28 } };
    const span_lens = [_]usize{ 2, 3 };
    const vocab = cfg.vocab_size;

    // Reference: each stream alone (prefill, then all-logits step).
    var want: [2][]f32 = undefined;
    defer for (&want) |rows| allocator.free(rows);
    inline for (0..2) |si| {
        var kv = try kv_cache_mod.KvCache.initPerLayer(&ctx, model.geom.kv_heads, model.geom.head_dim, 32);
        defer kv.deinit();
        var prefill = try model.forwardStep(&ctx, &kv, prompts[si], 0);
        prefill.deinit();
        var logits = try model.forwardStepAllLogits(&ctx, &kv, spans_tokens[si], kv.len);
        defer logits.deinit();
        want[si] = try allocator.dupe(f32, try logits.dataConst());
        try std.testing.expectEqual(@as(usize, spans_tokens[si].len * vocab), want[si].len);
    }

    // Ragged batch: same prefixes, both spans in ONE packed pass.
    var kv0 = try kv_cache_mod.KvCache.initPerLayer(&ctx, model.geom.kv_heads, model.geom.head_dim, 32);
    defer kv0.deinit();
    var kv1 = try kv_cache_mod.KvCache.initPerLayer(&ctx, model.geom.kv_heads, model.geom.head_dim, 32);
    defer kv1.deinit();
    var p0 = try model.forwardStep(&ctx, &kv0, prompts[0], 0);
    p0.deinit();
    var p1 = try model.forwardStep(&ctx, &kv1, prompts[1], 0);
    p1.deinit();
    const packed_tokens = [_]usize{ 7, 21, 4, 15, 28 };
    var got = try model.forwardStepBatchSpans(&ctx, &.{ &kv0, &kv1 }, &packed_tokens, &span_lens);
    defer got.deinit();
    const got_rows = try got.dataConst();
    try std.testing.expectEqual(@as(usize, 5 * vocab), got_rows.len);
    try std.testing.expectEqual(kv0.len, prompts[0].len + 2);
    try std.testing.expectEqual(kv1.len, prompts[1].len + 3);

    var at: usize = 0;
    inline for (0..2) |si| {
        for (want[si], got_rows[at * vocab ..][0..want[si].len]) |w, g| {
            try std.testing.expect(@abs(g - w) <= 1e-5 + 1e-4 * @abs(w));
        }
        at += span_lens[si];
    }
}
