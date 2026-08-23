//! Recorded-logits gates for the descriptor runner. The qwen3 family runs
//! ON the runner (`qwen3/model.zig` aliases it), so its correctness is
//! pinned against RECORDED values, not against a hand port: greedy argmax
//! chains (ISA-portable) always, and exact FNV-1a logit-bit hashes on the
//! machine class the goldens were recorded on (aarch64 native, Accelerate
//! BLAS) — a refactor that changes a single output bit fails there. The
//! glm4moe host_reference gate additionally compares the runner against
//! the hand glm4moe port bitwise. Real-model gates skip without models/;
//! all gates are native-only (the scalar leg verifies kernels, not model
//! wiring). Force-imported by `llm.zig`'s test block.

const std = @import("std");
const test_support = @import("test_support.zig");
const fucina = @import("fucina");
const runner = @import("runner.zig");

const gguf = fucina.gguf;
const ExecContext = fucina.ExecContext;

const strict_bits = test_support.strict_bits;
const argmaxRow = test_support.argmaxRow;
const fnvHash = test_support.fnvHash;

/// Recorded forward trace: `chain[0]` is the greedy argmax after prefill,
/// `chain[1..]` the argmaxes of the following decode steps; `prefill_hash`
/// fingerprints the prefill step's full logits row (cached path).
const Golden = struct {
    prefill_hash: u64,
    chain: [5]usize,
};

fn goldenForward(ctx: *ExecContext, model: *runner.Model, prompt: []const usize, golden: Golden) !void {
    // Cross-entry consistency: the no-cache last-logits entry must agree
    // with the cached prefill on the next token. (Not bitwise: the f16 KV
    // round-trip is part of the cached path.)
    {
        var last = try model.forwardLastLogits(ctx, prompt);
        defer last.deinit();
        try std.testing.expectEqual(golden.chain[0], argmaxRow(try last.dataConst()));
    }

    var kv = try model.initKvCache(ctx, 32);
    defer kv.deinit();
    var logits = try model.forwardStep(ctx, &kv, prompt, 0);
    if (strict_bits) try std.testing.expectEqual(golden.prefill_hash, fnvHash(try logits.dataConst()));
    var pos: usize = prompt.len;
    for (golden.chain[0 .. golden.chain.len - 1]) |want| {
        const next = argmaxRow(try logits.dataConst());
        try std.testing.expectEqual(want, next);
        logits.deinit();
        logits = try model.forwardStep(ctx, &kv, &.{next}, pos);
        pos += 1;
    }
    try std.testing.expectEqual(golden.chain[golden.chain.len - 1], argmaxRow(try logits.dataConst()));
    logits.deinit();
}

fn goldenOnGguf(path: []const u8, golden: Golden) !void {
    // Native builds only: real-model goldens verify model wiring, which is
    // backend-independent; the scalar leg would re-run 0.6B forwards on the
    // slow kernels for no added coverage.
    try test_support.requireNative();
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var file = try test_support.openGgufOrSkip(allocator, std.testing.io, path);
    defer file.deinit();

    const desc = try runner.Descriptor.fromGguf(&file);
    var model = try runner.Model.loadGgufFromFile(&ctx, &file, desc);
    defer model.deinit();

    const prompt = [_]usize{ 151644, 872, 198, 9707, 11, 1879, 0, 151645 };
    try goldenForward(&ctx, &model, &prompt, golden);
}

test "runner matches the recorded Qwen3-0.6B forward (Q8_0; skips without models/)" {
    try goldenOnGguf("models/Qwen3-0.6B-Q8_0.gguf", .{
        .prefill_hash = 0x63a59d2f7713b0b8,
        .chain = .{ 198, 151644, 198, 151644, 198 },
    });
}

test "runner matches the recorded Qwen3-0.6B forward (Q4_K_M; skips without models/)" {
    try goldenOnGguf("models/Qwen3-0.6B-Q4_K_M.gguf", .{
        .prefill_hash = 0x7726195312390e83,
        .chain = .{ 198, 151644, 198, 151667, 198 },
    });
}

// ---- synthetic-GGUF fixtures ---------------------------------------------
// Self-contained (no models/ needed): write a tiny model as a real GGUF
// through `gguf.Writer`, parse it back, load it through the runner, and
// require the recorded greedy chain (exact logit bits on the recording
// machine class). Weights are deterministic PRNG draws; the GGUF round-trip
// also pins `fromGguf` and the loaders. The glm fixture additionally runs
// the glm4moe family module for bitwise cross-path parity.

const glm4moe = @import("glm4moe/model.zig");

const FixtureWriter = struct {
    w: gguf.Writer,
    prng: std.Random.DefaultPrng,
    allocator: std.mem.Allocator,
    /// Tensor payloads are BORROWED by the gguf writer until `finish`, so
    /// they live in this arena for the fixture's whole build.
    arena: std.heap.ArenaAllocator,

    fn init(allocator: std.mem.Allocator, seed: u64) FixtureWriter {
        return .{
            .w = gguf.Writer.init(allocator),
            .prng = std.Random.DefaultPrng.init(seed),
            .allocator = allocator,
            .arena = std.heap.ArenaAllocator.init(allocator),
        };
    }

    fn deinit(self: *FixtureWriter) void {
        self.arena.deinit();
        self.w.deinit();
    }

    /// Add an f32 tensor of `dims` (GGUF ne order: innermost first) filled
    /// with small deterministic normal draws.
    fn tensor(self: *FixtureWriter, name: []const u8, dims: []const usize) !void {
        var n: usize = 1;
        for (dims) |d| n *= d;
        const values = try self.arena.allocator().alloc(f32, n);
        const rand = self.prng.random();
        for (values) |*v| v.* = rand.floatNorm(f32) * 0.08;
        try self.w.addTensor(name, .f32, dims, std.mem.sliceAsBytes(values));
    }

    /// As `tensor`, but stored q8_0 (expert stacks: `loadMoeRhs` takes
    /// quantized formats only; dims[0] must be a 32-multiple).
    fn tensorQ8(self: *FixtureWriter, name: []const u8, dims: []const usize) !void {
        var n: usize = 1;
        for (dims) |d| n *= d;
        const values = try self.arena.allocator().alloc(f32, n);
        const rand = self.prng.random();
        for (values) |*v| v.* = rand.floatNorm(f32) * 0.08;
        const bytes = try self.arena.allocator().alignedAlloc(u8, .of(fucina.quant.BlockQ8_0), n / 32 * 34);
        try gguf.encodeF32(.q8_0, values, bytes);
        try self.w.addTensor(name, .q8_0, dims, bytes);
    }

    fn finish(self: *FixtureWriter) ![]u8 {
        var out: std.Io.Writer.Allocating = .init(self.allocator);
        defer out.deinit();
        try self.w.finish(&out.writer);
        return self.allocator.dupe(u8, out.written());
    }
};

const tiny = struct {
    const vocab = 64;
    const hidden = 32;
    const layers = 2;
    const heads = 4;
    const kv_heads = 2;
    const head_dim = 32;
    const ffn = 64;
    const experts = 4;
    const experts_used = 2;
    const expert_ffn = 32;
    const q_dim = heads * head_dim;
    const kv_dim = kv_heads * head_dim;
};

fn writeTinyQwen(allocator: std.mem.Allocator, moe: bool) ![]u8 {
    var f = FixtureWriter.init(allocator, if (moe) 0xA11CE else 0xB0B);
    defer f.deinit();
    const arch: []const u8 = if (moe) "qwen3moe" else "qwen3";
    try f.w.addMetaString("general.architecture", arch);
    var key_buf: [96]u8 = undefined;
    const K = struct {
        fn k(buf: []u8, a: []const u8, suffix: []const u8) ![]const u8 {
            return std.fmt.bufPrint(buf, "{s}.{s}", .{ a, suffix });
        }
    };
    try f.w.addMetaInt(try K.k(&key_buf, arch, "embedding_length"), u32, tiny.hidden);
    try f.w.addMetaInt(try K.k(&key_buf, arch, "feed_forward_length"), u32, tiny.ffn);
    try f.w.addMetaInt(try K.k(&key_buf, arch, "block_count"), u32, tiny.layers);
    try f.w.addMetaInt(try K.k(&key_buf, arch, "attention.head_count"), u32, tiny.heads);
    try f.w.addMetaInt(try K.k(&key_buf, arch, "attention.head_count_kv"), u32, tiny.kv_heads);
    try f.w.addMetaInt(try K.k(&key_buf, arch, "attention.key_length"), u32, tiny.head_dim);
    try f.w.addMetaFloat(try K.k(&key_buf, arch, "attention.layer_norm_rms_epsilon"), f32, 1e-6);
    try f.w.addMetaFloat(try K.k(&key_buf, arch, "rope.freq_base"), f32, 10_000);
    if (moe) {
        try f.w.addMetaInt(try K.k(&key_buf, arch, "expert_count"), u32, tiny.experts);
        try f.w.addMetaInt(try K.k(&key_buf, arch, "expert_used_count"), u32, tiny.experts_used);
        try f.w.addMetaInt(try K.k(&key_buf, arch, "expert_feed_forward_length"), u32, tiny.expert_ffn);
    }

    try f.tensor("token_embd.weight", &.{ tiny.hidden, tiny.vocab });
    try f.tensor("output_norm.weight", &.{tiny.hidden});
    try f.tensor("output.weight", &.{ tiny.hidden, tiny.vocab });
    var name_buf: [96]u8 = undefined;
    for (0..tiny.layers) |i| {
        try f.tensor(try fucina.weights.layerName(&name_buf, i, "attn_norm.weight"), &.{tiny.hidden});
        try f.tensor(try fucina.weights.layerName(&name_buf, i, "attn_q_norm.weight"), &.{tiny.head_dim});
        try f.tensor(try fucina.weights.layerName(&name_buf, i, "attn_k_norm.weight"), &.{tiny.head_dim});
        try f.tensor(try fucina.weights.layerName(&name_buf, i, "ffn_norm.weight"), &.{tiny.hidden});
        try f.tensor(try fucina.weights.layerName(&name_buf, i, "attn_q.weight"), &.{ tiny.hidden, tiny.q_dim });
        try f.tensor(try fucina.weights.layerName(&name_buf, i, "attn_k.weight"), &.{ tiny.hidden, tiny.kv_dim });
        try f.tensor(try fucina.weights.layerName(&name_buf, i, "attn_v.weight"), &.{ tiny.hidden, tiny.kv_dim });
        try f.tensor(try fucina.weights.layerName(&name_buf, i, "attn_output.weight"), &.{ tiny.q_dim, tiny.hidden });
        if (moe) {
            try f.tensor(try fucina.weights.layerName(&name_buf, i, "ffn_gate_inp.weight"), &.{ tiny.hidden, tiny.experts });
            try f.tensorQ8(try fucina.weights.layerName(&name_buf, i, "ffn_gate_exps.weight"), &.{ tiny.hidden, tiny.expert_ffn, tiny.experts });
            try f.tensorQ8(try fucina.weights.layerName(&name_buf, i, "ffn_up_exps.weight"), &.{ tiny.hidden, tiny.expert_ffn, tiny.experts });
            try f.tensorQ8(try fucina.weights.layerName(&name_buf, i, "ffn_down_exps.weight"), &.{ tiny.expert_ffn, tiny.hidden, tiny.experts });
        } else {
            try f.tensor(try fucina.weights.layerName(&name_buf, i, "ffn_gate.weight"), &.{ tiny.hidden, tiny.ffn });
            try f.tensor(try fucina.weights.layerName(&name_buf, i, "ffn_up.weight"), &.{ tiny.hidden, tiny.ffn });
            try f.tensor(try fucina.weights.layerName(&name_buf, i, "ffn_down.weight"), &.{ tiny.ffn, tiny.hidden });
        }
    }
    return f.finish();
}

fn syntheticQwenGolden(moe: bool, golden: Golden) !void {
    try test_support.requireNative();
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const bytes = try writeTinyQwen(allocator, moe);
    var file = try gguf.File.parseOwned(allocator, bytes);
    defer file.deinit();

    const desc = try runner.Descriptor.fromGguf(&file);
    var model = try runner.Model.loadGgufFromFile(&ctx, &file, desc);
    defer model.deinit();

    const prompt = [_]usize{ 5, 61, 2, 33, 17 };
    try goldenForward(&ctx, &model, &prompt, golden);
}

test "runner matches the recorded forward on a synthetic dense GGUF" {
    try syntheticQwenGolden(false, .{
        .prefill_hash = 0x84b988c871d84241,
        .chain = .{ 23, 13, 52, 37, 44 },
    });
}

test "runner matches the recorded forward on a synthetic MoE GGUF (the small-MoE fixture)" {
    try syntheticQwenGolden(true, .{
        .prefill_hash = 0x29724bb80fda2f71,
        .chain = .{ 34, 51, 61, 17, 34 },
    });
}

fn writeTinyGlm(allocator: std.mem.Allocator) ![]u8 {
    var f = FixtureWriter.init(allocator, 0x91A4);
    defer f.deinit();
    try f.w.addMetaString("general.architecture", "glm4moe");
    try f.w.addMetaInt("glm4moe.embedding_length", u32, tiny.hidden);
    try f.w.addMetaInt("glm4moe.feed_forward_length", u32, tiny.ffn);
    try f.w.addMetaInt("glm4moe.block_count", u32, tiny.layers);
    try f.w.addMetaInt("glm4moe.attention.head_count", u32, tiny.heads);
    try f.w.addMetaInt("glm4moe.attention.head_count_kv", u32, tiny.kv_heads);
    try f.w.addMetaInt("glm4moe.attention.key_length", u32, tiny.head_dim);
    try f.w.addMetaFloat("glm4moe.attention.layer_norm_rms_epsilon", f32, 1e-6);
    try f.w.addMetaFloat("glm4moe.rope.freq_base", f32, 10_000);
    try f.w.addMetaInt("glm4moe.rope.dimension_count", u32, tiny.head_dim / 2); // partial rotary
    try f.w.addMetaInt("glm4moe.expert_count", u32, tiny.experts);
    try f.w.addMetaInt("glm4moe.expert_used_count", u32, tiny.experts_used);
    try f.w.addMetaInt("glm4moe.expert_feed_forward_length", u32, tiny.expert_ffn);
    try f.w.addMetaInt("glm4moe.leading_dense_block_count", u32, 1); // layer 0 dense, layer 1 MoE
    try f.w.addMetaInt("glm4moe.expert_shared_count", u32, 1);
    try f.w.addMetaFloat("glm4moe.expert_weights_scale", f32, 1.5);
    try f.w.addMetaInt("glm4moe.expert_gating_func", u32, 2); // sigmoid noaux
    try f.w.addMetaBool("glm4moe.expert_weights_norm", true);

    try f.tensor("token_embd.weight", &.{ tiny.hidden, tiny.vocab });
    try f.tensor("output_norm.weight", &.{tiny.hidden});
    try f.tensor("output.weight", &.{ tiny.hidden, tiny.vocab });
    var name_buf: [96]u8 = undefined;
    for (0..tiny.layers) |i| {
        try f.tensor(try fucina.weights.layerName(&name_buf, i, "attn_norm.weight"), &.{tiny.hidden});
        try f.tensor(try fucina.weights.layerName(&name_buf, i, "post_attention_norm.weight"), &.{tiny.hidden});
        try f.tensor(try fucina.weights.layerName(&name_buf, i, "attn_q.weight"), &.{ tiny.hidden, tiny.q_dim });
        try f.tensor(try fucina.weights.layerName(&name_buf, i, "attn_q.bias"), &.{tiny.q_dim});
        try f.tensor(try fucina.weights.layerName(&name_buf, i, "attn_k.weight"), &.{ tiny.hidden, tiny.kv_dim });
        try f.tensor(try fucina.weights.layerName(&name_buf, i, "attn_k.bias"), &.{tiny.kv_dim});
        try f.tensor(try fucina.weights.layerName(&name_buf, i, "attn_v.weight"), &.{ tiny.hidden, tiny.kv_dim });
        try f.tensor(try fucina.weights.layerName(&name_buf, i, "attn_v.bias"), &.{tiny.kv_dim});
        try f.tensor(try fucina.weights.layerName(&name_buf, i, "attn_output.weight"), &.{ tiny.q_dim, tiny.hidden });
        if (i == 0) {
            try f.tensor(try fucina.weights.layerName(&name_buf, i, "ffn_gate.weight"), &.{ tiny.hidden, tiny.ffn });
            try f.tensor(try fucina.weights.layerName(&name_buf, i, "ffn_up.weight"), &.{ tiny.hidden, tiny.ffn });
            try f.tensor(try fucina.weights.layerName(&name_buf, i, "ffn_down.weight"), &.{ tiny.ffn, tiny.hidden });
        } else {
            try f.tensor(try fucina.weights.layerName(&name_buf, i, "ffn_gate_inp.weight"), &.{ tiny.hidden, tiny.experts });
            try f.tensor(try fucina.weights.layerName(&name_buf, i, "exp_probs_b.bias"), &.{tiny.experts});
            try f.tensorQ8(try fucina.weights.layerName(&name_buf, i, "ffn_gate_exps.weight"), &.{ tiny.hidden, tiny.expert_ffn, tiny.experts });
            try f.tensorQ8(try fucina.weights.layerName(&name_buf, i, "ffn_up_exps.weight"), &.{ tiny.hidden, tiny.expert_ffn, tiny.experts });
            try f.tensorQ8(try fucina.weights.layerName(&name_buf, i, "ffn_down_exps.weight"), &.{ tiny.expert_ffn, tiny.hidden, tiny.experts });
            try f.tensor(try fucina.weights.layerName(&name_buf, i, "ffn_gate_shexp.weight"), &.{ tiny.hidden, tiny.expert_ffn });
            try f.tensor(try fucina.weights.layerName(&name_buf, i, "ffn_up_shexp.weight"), &.{ tiny.hidden, tiny.expert_ffn });
            try f.tensor(try fucina.weights.layerName(&name_buf, i, "ffn_down_shexp.weight"), &.{ tiny.expert_ffn, tiny.hidden });
        }
    }
    return f.finish();
}

test "runner and the glm4moe family module agree bitwise on a synthetic GGUF (host_reference vocabulary)" {
    try test_support.requireNative();
    // Recorded trace for the same fixture: chain[0] = prefill argmax.
    const golden = Golden{
        .prefill_hash = 0xb4ec8486578aa204,
        .chain = .{ 57, 28, 57, 28, 59 },
    };
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Through a real file + mmap: the glm-shaped loaders borrow resident
    // expert stacks from the mapping (both refuse a mapless MoE load).
    const bytes = try writeTinyGlm(allocator);
    defer allocator.free(bytes);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "glm-fixture.gguf", .data = bytes });
    const path = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/glm-fixture.gguf", .{tmp.sub_path});
    defer allocator.free(path);
    // One mapping per model: both loaders take ownership of their file's
    // mmap for the borrowed expert stacks (`takeMapping` is one-shot).
    var file = try gguf.File.loadMmap(allocator, std.testing.io, path);
    defer file.deinit();
    var file_hand = try gguf.File.loadMmap(allocator, std.testing.io, path);
    defer file_hand.deinit();

    const desc = try runner.Descriptor.fromGguf(&file);
    try std.testing.expectEqual(runner.BlockStyle.host_reference, desc.block_style);
    var generic = try runner.Model.loadGgufFromFile(&ctx, &file, desc);
    defer generic.deinit();
    var hand = try glm4moe.Model.loadGgufFromFileOptions(&ctx, &file_hand, 16, .{});
    defer hand.deinit();

    var cache_generic = try generic.initHostCache(16);
    defer cache_generic.deinit();
    var cache_hand = try hand.initCache(16);
    defer cache_hand.deinit();

    // Prefill + greedy chain: every per-position logits row bitwise.
    var prompt = [_]usize{ 3, 47, 20, 11 };
    var next: usize = undefined;
    {
        var got = try generic.hostStep(&ctx, &cache_generic, &prompt);
        defer got.deinit();
        var want = try hand.step(&ctx, &cache_hand, &prompt);
        defer want.deinit();
        const flat = try got.dataConst();
        try std.testing.expectEqualSlices(f32, try want.dataConst(), flat);
        if (strict_bits) try std.testing.expectEqual(golden.prefill_hash, fnvHash(flat));
        next = argmaxRow(flat[(prompt.len - 1) * tiny.vocab ..][0..tiny.vocab]);
        try std.testing.expectEqual(golden.chain[0], next);
    }
    for (golden.chain[1..]) |want_tok| {
        var step_tokens = [_]usize{next};
        var got = try generic.hostStep(&ctx, &cache_generic, &step_tokens);
        defer got.deinit();
        var want = try hand.step(&ctx, &cache_hand, &step_tokens);
        defer want.deinit();
        try std.testing.expectEqualSlices(f32, try want.dataConst(), try got.dataConst());
        next = argmaxRow(try want.dataConst());
        try std.testing.expectEqual(want_tok, next);
    }
}
