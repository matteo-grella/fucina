//! Behavioral tests for the Gemma 4 inference module (`gemma4.zig`):
//! per-layer SWA/KV geometry derivation, shared-KV reuse mapping, and the
//! f16-only KV-cache forward-seam guard.
const std = @import("std");
const fucina = @import("fucina");
const gguf = fucina.gguf;
const kv_cache = @import("../kv_cache.zig");
const gemma4 = @import("model.zig");

const ExecContext = fucina.ExecContext;
const KvCache = kv_cache.KvCache;
const Error = gemma4.Error;
const deriveGeometry = gemma4.deriveGeometry;

test "gemma4 per-layer geometry maps an explicit SWA + KV pattern" {
    const allocator = std.testing.allocator;
    const n_layer = 30;
    const global_layers = [_]usize{ 5, 11, 17, 23, 29 };
    var pattern = [_]bool{true} ** n_layer;
    var kv = [_]usize{8} ** n_layer;
    for (global_layers) |g| {
        pattern[g] = false; // global
        kv[g] = 2;
    }

    var geom = try deriveGeometry(allocator, n_layer, &pattern, &kv, 0, 512, 256);
    defer geom.deinit(allocator);

    var globals: usize = 0;
    for (0..n_layer) |il| {
        var is_global = false;
        for (global_layers) |g| {
            if (g == il) is_global = true;
        }
        try std.testing.expectEqual(!is_global, geom.is_swa[il]);
        try std.testing.expectEqual(@as(usize, if (is_global) 512 else 256), geom.head_dim[il]);
        try std.testing.expectEqual(@as(usize, if (is_global) 2 else 8), geom.kv_heads[il]);
        try std.testing.expect(geom.has_kv[il]); // shared_kv_layers = 0
        try std.testing.expectEqual(il, geom.kv_ref[il]);
        if (is_global) globals += 1;
    }
    try std.testing.expectEqual(@as(usize, 5), globals);
}

test "gemma4 rejects a q8_0 KV cache at the forward seam" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var q8 = try KvCache.initWithDtype(&ctx, 1, 2, 64, 4, .q8_0);
    defer q8.deinit();
    try std.testing.expectError(Error.UnsupportedKvCacheDtype, q8.requireF16());

    var f16_cache = try KvCache.initWithDtype(&ctx, 1, 2, 64, 4, .f16);
    defer f16_cache.deinit();
    try f16_cache.requireF16();
}

test "gemma4 shared-KV reuse map is in range" {
    const allocator = std.testing.allocator;
    var pattern = [_]bool{ true, true, false, true, true, true };
    var kv = [_]usize{ 8, 8, 2, 8, 8, 8 };
    var geom = try deriveGeometry(allocator, 6, &pattern, &kv, 2, 256, 128);
    defer geom.deinit(allocator);
    for (0..4) |il| try std.testing.expect(geom.has_kv[il] and geom.kv_ref[il] == il);
    for (4..6) |il| {
        try std.testing.expect(!geom.has_kv[il]);
        try std.testing.expect(geom.kv_ref[il] < 4);
    }
}

// ---------------------------------------------------------------------------
// PLE table: mmap-borrowed vs resident, end to end
// ---------------------------------------------------------------------------

// `enc_arena` must outlive the Writer's finish: addTensor borrows the
// encoded bytes until then.
fn addRandTensor(w: *gguf.Writer, enc_arena: std.mem.Allocator, seed: u64, name: []const u8, dt: gguf.GgmlType, rows: usize, cols: usize, bound: f32) !void {
    const allocator = std.testing.allocator;
    const vals = try allocator.alloc(f32, rows * cols);
    defer allocator.free(vals);
    fucina.rng.uniformFill(seed, vals, -bound, bound);
    const enc = try enc_arena.alloc(u8, try gguf.tensorByteLen(dt, &.{vals.len}));
    try gguf.encodeF32(dt, vals, enc);
    // GGUF dims order: innermost (cols) first; 1-D tensors pass rows=1.
    if (rows == 1) try w.addTensor(name, dt, &.{cols}, enc) else try w.addTensor(name, dt, &.{ cols, rows }, enc);
}

// A complete 1-layer dense Gemma 4 with PLE, written to a real GGUF: the
// mmap load must borrow the PLE table (weight_mapping owned by the model),
// the heap load must not, and both must produce bitwise-identical logits.
test "gemma4 PLE table borrows the mmap and matches the resident path bitwise" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const cfg = gemma4.Config{
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
        .per_layer_input_size = 32, // one q8_0 block per table row
        .final_logit_softcapping = 0,
    };
    const hidden = cfg.hidden_size;
    const q_dim = cfg.num_attention_heads * cfg.head_dim_global;
    const kv_dim = cfg.head_dim_global; // one KV head
    const ple_all = cfg.per_layer_input_size * cfg.num_layers;

    var enc_arena = std.heap.ArenaAllocator.init(allocator);
    defer enc_arena.deinit();
    const enc = enc_arena.allocator();
    var w = gguf.Writer.init(allocator);
    defer w.deinit();
    try w.addMetaInt("gemma4.attention.sliding_window_pattern", u32, 0);
    try w.addMetaInt("gemma4.attention.head_count_kv", u32, 1);
    try addRandTensor(&w, enc, 1, "token_embd.weight", .f32, cfg.vocab_size, hidden, 0.5);
    try addRandTensor(&w, enc, 2, "output_norm.weight", .f32, 1, hidden, 1.0);
    // The lookup table under test: quantized, so the mapped arm's per-row
    // decode (not just the f32 zero-copy fast path) is what the comparison
    // exercises.
    try addRandTensor(&w, enc, 3, "per_layer_token_embd.weight", .q8_0, cfg.vocab_size, ple_all, 0.5);
    try addRandTensor(&w, enc, 4, "per_layer_model_proj.weight", .f32, ple_all, hidden, 0.25);
    try addRandTensor(&w, enc, 5, "per_layer_proj_norm.weight", .f32, 1, cfg.per_layer_input_size, 1.0);
    try addRandTensor(&w, enc, 6, "blk.0.attn_norm.weight", .f32, 1, hidden, 1.0);
    try addRandTensor(&w, enc, 7, "blk.0.post_attention_norm.weight", .f32, 1, hidden, 1.0);
    try addRandTensor(&w, enc, 8, "blk.0.attn_q.weight", .f32, q_dim, hidden, 0.25);
    try addRandTensor(&w, enc, 9, "blk.0.attn_q_norm.weight", .f32, 1, cfg.head_dim_global, 1.0);
    try addRandTensor(&w, enc, 10, "blk.0.attn_k.weight", .f32, kv_dim, hidden, 0.25);
    try addRandTensor(&w, enc, 11, "blk.0.attn_k_norm.weight", .f32, 1, cfg.head_dim_global, 1.0);
    try addRandTensor(&w, enc, 12, "blk.0.attn_v.weight", .f32, kv_dim, hidden, 0.25);
    try addRandTensor(&w, enc, 13, "blk.0.attn_output.weight", .f32, hidden, q_dim, 0.25);
    try addRandTensor(&w, enc, 14, "blk.0.ffn_norm.weight", .f32, 1, hidden, 1.0);
    try addRandTensor(&w, enc, 15, "blk.0.ffn_gate.weight", .f32, cfg.intermediate_size, hidden, 0.25);
    try addRandTensor(&w, enc, 16, "blk.0.ffn_up.weight", .f32, cfg.intermediate_size, hidden, 0.25);
    try addRandTensor(&w, enc, 17, "blk.0.ffn_down.weight", .f32, hidden, cfg.intermediate_size, 0.25);
    try addRandTensor(&w, enc, 18, "blk.0.post_ffw_norm.weight", .f32, 1, hidden, 1.0);
    try addRandTensor(&w, enc, 19, "blk.0.inp_gate.weight", .f32, cfg.per_layer_input_size, hidden, 0.25);
    try addRandTensor(&w, enc, 20, "blk.0.proj.weight", .f32, hidden, cfg.per_layer_input_size, 0.25);
    try addRandTensor(&w, enc, 21, "blk.0.post_norm.weight", .f32, 1, hidden, 1.0);

    const buf = try allocator.alloc(u8, 1 << 20);
    defer allocator.free(buf);
    var sink = std.Io.Writer.fixed(buf);
    try w.finish(&sink);

    var path_buf: [64]u8 = undefined;
    const stamp = std.Io.Clock.real.now(io).nanoseconds;
    const path = try std.fmt.bufPrint(&path_buf, "gemma4_ple_{d}.gguf", .{stamp});
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};
    {
        var f = try std.Io.Dir.cwd().createFile(io, path, .{});
        defer f.close(io);
        try f.writePositionalAll(io, sink.buffered(), 0);
    }

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Mmap load: the PLE table borrows and the model owns the mapping —
    // the file is deinited BEFORE the forward pass, as in Model.loadGguf,
    // so a dangling borrow would surface here, not in production.
    var mapped_model = blk: {
        var file = try gguf.File.loadMmap(allocator, io, path);
        defer file.deinit();
        break :blk try gemma4.Model.loadGgufFromFile(&ctx, &file, cfg);
    };
    defer mapped_model.deinit();
    if (comptime !fucina.internal.gpu.enabled) {
        try std.testing.expect(mapped_model.weight_mapping != null);
        try std.testing.expect(mapped_model.ple.?.tok_embd.borrowsMapping());
    }

    var heap_model = blk: {
        var file = try gguf.File.load(allocator, io, path);
        defer file.deinit();
        break :blk try gemma4.Model.loadGgufFromFile(&ctx, &file, cfg);
    };
    defer heap_model.deinit();
    try std.testing.expect(heap_model.weight_mapping == null);
    try std.testing.expect(!heap_model.ple.?.tok_embd.borrowsMapping());

    const prompt = [_]usize{ 1, 5, 31, 0, 5 };
    var mapped_logits = try mapped_model.forwardLastLogits(&ctx, &prompt);
    defer mapped_logits.deinit();
    var heap_logits = try heap_model.forwardLastLogits(&ctx, &prompt);
    defer heap_logits.deinit();
    try std.testing.expectEqualSlices(f32, try heap_logits.dataConst(), try mapped_logits.dataConst());
}
