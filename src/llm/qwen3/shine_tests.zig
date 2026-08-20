//! Unit tests for the SHINE hypernetwork glue — no model files needed.
//! Golden parity against the PyTorch reference (real Qwen3-8B + released
//! checkpoint) lives in shine_golden_tests.zig, model-gated.
const std = @import("std");
const fucina = @import("fucina");
const qwen3 = @import("model.zig");
const shine = @import("shine.zig");
const kv_cache = @import("../kv_cache.zig");
const scaffolding = @import("train_tests.zig");

const ExecContext = fucina.ExecContext;

test "memory-token budget identity matches the released Qwen3-8B run" {
    // Qwen3-8B: hidden 4096, 36 layers, 32/8 heads at head_dim 128,
    // intermediate 12288. With r=8 over q,k,v,o,gate,up,down the per-layer
    // generated-parameter count must be exactly 148 * 4096 (M = ceil(rD/H)
    // with no remainder — the reference asserts divisibility).
    const base = qwen3.Config{
        .vocab_size = 151_936,
        .hidden_size = 4096,
        .intermediate_size = 12_288,
        .num_layers = 36,
        .num_attention_heads = 32,
        .num_key_value_heads = 8,
        .head_dim = 128,
        .rms_norm_eps = 1e-6,
        .rope_theta = 1_000_000,
    };
    try std.testing.expectEqual(@as(usize, 148 * 4096), shine.loraParamsPerLayer(base, 8));
}

/// Tiny geometry whose per-layer LoRA budget divides the hidden size:
/// r=2 over dims q(8,8) k(8,4) v(8,4) o(8,8) gate(8,12) up(8,12) down(12,8)
/// = 232 = 29 * 8 -> mem = 29 rows of hidden 8.
const tiny = qwen3.Config{
    .vocab_size = 32,
    .hidden_size = 8,
    .intermediate_size = 12,
    .num_layers = 2,
    .num_attention_heads = 2,
    .num_key_value_heads = 1,
    .head_dim = 4,
    .rms_norm_eps = 1e-6,
    .rope_theta = 10_000,
};

test "sliceLora cuts the rl layout in module order with sqrt(scale)" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const r = 2;
    const per_layer = shine.loraParamsPerLayer(tiny, r);
    try std.testing.expectEqual(@as(usize, 232), per_layer);
    const mem = per_layer / tiny.hidden_size;

    const values = try allocator.alloc(f32, tiny.num_layers * per_layer);
    defer allocator.free(values);
    for (values, 0..) |*v, i| v.* = @floatFromInt(i);
    var plain = try fucina.Tensor(.{ .layer, .mem, .embed }).fromSlice(&ctx, .{ tiny.num_layers, mem, tiny.hidden_size }, values);
    defer plain.deinit();

    // scale 4 -> sqrt(scale) 2: every stored value is exactly 2 * index.
    var set = try shine.sliceLora(&ctx, tiny, r, 4.0, &plain);
    defer set.deinit();

    try std.testing.expectEqual(@as(usize, 2), set.layers.len);
    // Layer 0, module q: A is the first in*r = 16 values, [8, 2].
    {
        const a = try set.layers[0].q.a.dataConst();
        try std.testing.expectEqual(@as(usize, 16), a.len);
        try std.testing.expectEqual(@as(f32, 0), a[0]);
        try std.testing.expectEqual(@as(f32, 2 * 15), a[15]);
        const b = try set.layers[0].q.b.dataConst();
        try std.testing.expectEqual(@as(usize, 16), b.len);
        try std.testing.expectEqual(@as(f32, 2 * 16), b[0]);
    }
    // Layer 0, module k starts right after q's A+B (16 + 16 = 32).
    {
        const a = try set.layers[0].k.a.dataConst();
        try std.testing.expectEqual(@as(usize, 16), a.len); // in 8 * r 2
        try std.testing.expectEqual(@as(f32, 2 * 32), a[0]);
        const b = try set.layers[0].k.b.dataConst();
        try std.testing.expectEqual(@as(usize, 8), b.len); // r 2 * out 4
    }
    // Layer 1 restarts at the second flattened layer row.
    {
        const a = try set.layers[1].q.a.dataConst();
        try std.testing.expectEqual(@as(f32, 2 * @as(f32, @floatFromInt(per_layer))), a[0]);
    }
    // The last module (down) ends exactly at the row boundary.
    {
        const b = try set.layers[1].down.b.dataConst();
        try std.testing.expectEqual(@as(f32, 2 * @as(f32, @floatFromInt(2 * per_layer - 1))), b[b.len - 1]);
    }
}

test "adapter artifact round-trips bitwise through the shine-adapter GGUF" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const r = 2;
    const per_layer = shine.loraParamsPerLayer(tiny, r);
    const mem = per_layer / tiny.hidden_size;
    const values = try allocator.alloc(f32, tiny.num_layers * per_layer);
    defer allocator.free(values);
    var state: u32 = 0x9E3779B9;
    for (values) |*v| {
        state = state *% 1664525 +% 1013904223;
        v.* = @as(f32, @floatFromInt(state >> 8)) / (1 << 24);
    }
    var plain = try fucina.Tensor(.{ .layer, .mem, .embed }).fromSlice(&ctx, .{ tiny.num_layers, mem, tiny.hidden_size }, values);
    defer plain.deinit();
    var set = try shine.sliceLora(&ctx, tiny, r, 0.001, &plain);
    defer set.deinit();

    var buf: [64 * 1024]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try shine.saveLora(&set, tiny, allocator, &writer);
    const bytes = try allocator.dupe(u8, writer.buffered());
    // parseOwned takes ownership of `bytes` (freed by file.deinit).
    var file = try fucina.gguf.File.parseOwned(allocator, bytes);
    defer file.deinit();

    var reloaded = try shine.loadLoraFromFile(&ctx, &file, tiny);
    defer reloaded.deinit();
    try std.testing.expectEqual(set.layers.len, reloaded.layers.len);
    for (set.layers, reloaded.layers) |*want, *got| {
        inline for (shine.modules) |module| {
            try std.testing.expectEqualSlices(f32, try want.pair(module).a.dataConst(), try got.pair(module).a.dataConst());
            try std.testing.expectEqualSlices(f32, try want.pair(module).b.dataConst(), try got.pair(module).b.dataConst());
        }
    }

    // A geometry mismatch is refused, never silently mis-sliced.
    var wrong = tiny;
    wrong.hidden_size = 16;
    try std.testing.expectError(shine.Error.InvalidShineConfig, shine.loadLoraFromFile(&ctx, &file, wrong));
}

fn zeroPair(ctx: *ExecContext, in_dim: usize, r: usize, out_dim: usize) !shine.LoraPair {
    var a = try fucina.Tensor(.{ .lin, .lora_r }).zeros(ctx, .{ in_dim, r });
    errdefer a.deinit();
    var b = try fucina.Tensor(.{ .lora_r, .lout }).zeros(ctx, .{ r, out_dim });
    errdefer b.deinit();
    return .{ .a = a, .b = b };
}

fn zeroLoraSet(ctx: *ExecContext, base: qwen3.Config, r: usize) !shine.LoraSet {
    const layers = try ctx.allocator.alloc(shine.LayerLora, base.num_layers);
    var built: usize = 0;
    errdefer {
        for (layers[0..built]) |*layer| layer.deinit();
        ctx.allocator.free(layers);
    }
    for (layers) |*layer| {
        inline for (shine.modules) |module| {
            const dims = shine.moduleDims(base, module);
            @field(layer, @tagName(module)) = try zeroPair(ctx, dims[0], r, dims[1]);
        }
        built += 1;
    }
    return .{ .allocator = ctx.allocator, .layers = layers };
}

test "zero adapter forwardStep matches the base model bitwise" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try scaffolding.buildTinyModelWithConfig(&ctx, tiny, 0x5417E);
    defer model.deinit();
    var zero = try zeroLoraSet(&ctx, tiny, 2);
    defer zero.deinit();

    const prompt = [_]usize{ 3, 11, 7, 29, 2 };

    var kv_base = try model.initKvCache(&ctx, 16);
    defer kv_base.deinit();
    var base_logits = try model.forwardStep(&ctx, &kv_base, &prompt, 0);
    defer base_logits.deinit();

    var kv_lora = try model.initKvCache(&ctx, 16);
    defer kv_lora.deinit();
    var lora_logits = try shine.forwardStep(&model, &zero, &ctx, &kv_lora, &prompt, 0);
    defer lora_logits.deinit();

    // A zero adapter adds exact-zero deltas, so the two paths compute the
    // same values op for op (both below every m-dependent kernel threshold).
    try std.testing.expectEqualSlices(f32, try base_logits.dataConst(), try lora_logits.dataConst());

    // And one decode step on top of the shared cache state agrees too.
    var base_step = try model.forwardStep(&ctx, &kv_base, &.{4}, kv_base.len);
    defer base_step.deinit();
    var lora_step = try shine.forwardStep(&model, &zero, &ctx, &kv_lora, &.{4}, kv_lora.len);
    defer lora_step.deinit();
    try std.testing.expectEqualSlices(f32, try base_step.dataConst(), try lora_step.dataConst());
}
