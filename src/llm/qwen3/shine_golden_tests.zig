//! PyTorch golden-parity tests for the SHINE port, against the released
//! Yewei-Liu/SHINE-ift_mqa_1qa checkpoint on Qwen3-8B.
//!
//! Model-gated: they need the local artifacts produced by
//!   tools/convert_shine.py  -> models/shine/shine-ift-mqa-1qa.gguf
//!   tools/gen_shine_goldens.py -> models/shine/goldens/*
//!   (+ models/Qwen3-8B-F16.gguf / -BF16.gguf for the encoder/e2e test)
//! and skip silently when any is missing.
//!
//! The reference runs fp32 weights (upcast from the bf16 release); the port
//! runs 16-bit GGUF weights. bf16 -> f16 is exact for every weight in range,
//! so the compared drift is kernel-path reassociation/rounding only.
//!
//! Native backend only: these are real-model golden forwards — they pin
//! MODEL WIRING, not kernel math, so the scalar reference leg skips them
//! (its kernel coverage lives in the exec/backend suites).
const std = @import("std");
const fucina = @import("fucina");
const qwen3 = @import("model.zig");
const shine = @import("shine.zig");

const ExecContext = fucina.ExecContext;
const gguf = fucina.gguf;

const goldens_dir = "models/shine/goldens";
const shine_gguf_path = "models/shine/shine-ift-mqa-1qa.gguf";
// Both 16-bit serving formats when present: f16 GEMMs round the f32
// activations to f16, bf16 keeps the f32 LHS (mixed f32 x bf16 kernel) —
// the greedy gate must hold on each.
const base_gguf_paths = [_][]const u8{ "models/Qwen3-8B-F16.gguf", "models/Qwen3-8B-BF16.gguf" };

// From the released run (manifest.json): the reference tokenizer length
// after the 3 added task tokens, and Qwen3's <|im_end|> id.
const reference_vocab = 151_672;
const im_end = 151_645;

fn readBytes(allocator: std.mem.Allocator, comptime name: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(std.testing.io, goldens_dir ++ "/" ++ name, allocator, .limited(1 << 30));
}

fn readF32(allocator: std.mem.Allocator, comptime name: []const u8, expected_len: usize) ![]f32 {
    const bytes = try readBytes(allocator, name);
    defer allocator.free(bytes);
    if (bytes.len != expected_len * 4) return error.GoldenShapeMismatch;
    const out = try allocator.alloc(f32, expected_len);
    @memcpy(std.mem.sliceAsBytes(out), bytes);
    return out;
}

fn readIds(allocator: std.mem.Allocator, comptime name: []const u8) ![]usize {
    const bytes = try readBytes(allocator, name);
    defer allocator.free(bytes);
    if (bytes.len % 4 != 0) return error.GoldenShapeMismatch;
    const out = try allocator.alloc(usize, bytes.len / 4);
    errdefer allocator.free(out);
    for (out, 0..) |*id, i| {
        const raw = std.mem.readInt(i32, bytes[i * 4 ..][0..4], .little);
        if (raw < 0) return error.GoldenShapeMismatch;
        id.* = @intCast(raw);
    }
    return out;
}

const Drift = struct {
    max_abs: f32 = 0,
    max_ref: f32 = 0,

    fn measure(got: []const f32, ref: []const f32) Drift {
        std.debug.assert(got.len == ref.len);
        var drift = Drift{};
        for (got, ref) |g, r| {
            drift.max_abs = @max(drift.max_abs, @abs(g - r));
            drift.max_ref = @max(drift.max_ref, @abs(r));
        }
        return drift;
    }

    fn rel(self: Drift) f32 {
        return if (self.max_ref == 0) self.max_abs else self.max_abs / self.max_ref;
    }
};

/// Cheap presence probe so a goldens-less checkout skips before any
/// multi-GB GGUF load is attempted.
fn requireGoldens(allocator: std.mem.Allocator) !void {
    const probe = readBytes(allocator, "manifest.json") catch return error.SkipZigTest;
    allocator.free(probe);
}

test "SHINE m2p + sliceLora parity vs the PyTorch reference" {
    if (comptime @import("fucina").internal.backend_mod.active_kind != .native) return error.SkipZigTest; // real-model goldens: native only
    // Loads the 3.3GB f32 SHINE GGUF and runs the full M2P stack — the
    // tracking allocator's overhead is irrelevant at this test's runtime.
    const allocator = std.heap.smp_allocator;
    try requireGoldens(allocator);
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var file = gguf.File.loadMmap(allocator, std.testing.io, shine_gguf_path) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer file.deinit();
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
    var sh = try shine.Shine.loadGgufFromFile(&ctx, &file, base);
    defer sh.deinit();

    const state_len = sh.config.num_layers * sh.config.num_mem_token * sh.config.hidden_size;
    const golden_memory = readF32(allocator, "memory_states.f32.bin", state_len) catch return error.SkipZigTest;
    defer allocator.free(golden_memory);
    const golden_plain = try readF32(allocator, "plain_output.f32.bin", state_len);
    defer allocator.free(golden_plain);

    // M2P from the GOLDEN memory states isolates this stage from encoder
    // drift: pure f32 math on both sides. Walked stage by stage against the
    // reference's layer-0 probes so a mismatch localizes immediately.
    var memory_states = try fucina.Tensor(.{ .layer, .mem, .embed }).fromSlice(&ctx, .{ sh.config.num_layers, sh.config.num_mem_token, sh.config.hidden_size }, golden_memory);
    defer memory_states.deinit();
    const probe_len = sh.config.num_mem_token * sh.config.hidden_size;
    var x = try shine.m2pInput(&sh, &ctx, &memory_states);
    defer x.deinit();
    inline for ([_][]const u8{ "m2p_stage0_probe.f32.bin", "m2p_stage1_probe.f32.bin", "m2p_stage2_probe.f32.bin", "m2p_stage3_probe.f32.bin", "m2p_stage4_probe.f32.bin" }, 0..) |probe_name, stage| {
        const golden_probe = try readF32(allocator, probe_name, probe_len);
        defer allocator.free(golden_probe);
        var layer0 = try x.narrow(&ctx, .layer, 0, 1);
        defer layer0.deinit();
        const probe_drift = Drift.measure(try layer0.dataConst(), golden_probe);
        std.debug.print("shine m2p stage {d}: max|diff| {d} (max|ref| {d}, rel {d})\n", .{ stage, probe_drift.max_abs, probe_drift.max_ref, probe_drift.rel() });
        try std.testing.expect(probe_drift.rel() < 1e-4);
        if (stage < sh.m2p.len) x = try ctx.replace(x, shine.m2pStage(sh.config, sh.m2p, &ctx, stage, &x));
    }
    const plain_drift = Drift.measure(try x.dataConst(), golden_plain);
    std.debug.print("shine m2p plain: max|diff| {d} (max|ref| {d}, rel {d})\n", .{ plain_drift.max_abs, plain_drift.max_ref, plain_drift.rel() });
    try std.testing.expect(plain_drift.rel() < 1e-4);

    // Slicing parity from the golden plain tensor: exact layout, scale-only
    // rounding.
    var golden_plain_tensor = try fucina.Tensor(.{ .layer, .mem, .embed }).fromSlice(&ctx, .{ sh.config.num_layers, sh.config.num_mem_token, sh.config.hidden_size }, golden_plain);
    defer golden_plain_tensor.deinit();
    var set = try shine.sliceLora(&ctx, base, sh.config.lora_r, sh.config.scale, &golden_plain_tensor);
    defer set.deinit();

    const golden_a0 = try readF32(allocator, "lora_l0_q_a.f32.bin", base.hidden_size * sh.config.lora_r);
    defer allocator.free(golden_a0);
    const golden_b0 = try readF32(allocator, "lora_l0_q_b.f32.bin", sh.config.lora_r * base.qProjectionDim());
    defer allocator.free(golden_b0);
    const golden_a35 = try readF32(allocator, "lora_l35_down_a.f32.bin", base.intermediate_size * sh.config.lora_r);
    defer allocator.free(golden_a35);
    const golden_b35 = try readF32(allocator, "lora_l35_down_b.f32.bin", sh.config.lora_r * base.hidden_size);
    defer allocator.free(golden_b35);

    const a0_drift = Drift.measure(try set.layers[0].q.a.dataConst(), golden_a0);
    const b0_drift = Drift.measure(try set.layers[0].q.b.dataConst(), golden_b0);
    const a35_drift = Drift.measure(try set.layers[35].down.a.dataConst(), golden_a35);
    const b35_drift = Drift.measure(try set.layers[35].down.b.dataConst(), golden_b35);
    std.debug.print("shine slice: l0.q a/b max|diff| {d}/{d}, l35.down a/b {d}/{d}\n", .{ a0_drift.max_abs, b0_drift.max_abs, a35_drift.max_abs, b35_drift.max_abs });
    try std.testing.expect(a0_drift.rel() < 1e-6);
    try std.testing.expect(b0_drift.rel() < 1e-6);
    try std.testing.expect(a35_drift.rel() < 1e-6);
    try std.testing.expect(b35_drift.rel() < 1e-6);
}

test "SHINE encoder + end-to-end greedy parity vs the PyTorch reference" {
    if (comptime @import("fucina").internal.backend_mod.active_kind != .native) return error.SkipZigTest; // real-model goldens: native only
    // A Debug 8B forward is a ~25x slowdown; this leg runs only in the
    // optimized gate (`zig build test-llm -Doptimize=ReleaseFast`) so the
    // routine Debug `zig build test` loop stays usable with the model
    // artifacts present.
    if (@import("builtin").mode == .Debug) return error.SkipZigTest;
    const allocator = std.heap.smp_allocator;
    try requireGoldens(allocator);

    var ran_any = false;
    for (base_gguf_paths) |base_path| {
        runEncoderE2e(allocator, base_path) catch |err| switch (err) {
            error.SkipZigTest => continue,
            else => return err,
        };
        ran_any = true;
    }
    if (!ran_any) return error.SkipZigTest;
}

fn runEncoderE2e(allocator: std.mem.Allocator, base_path: []const u8) !void {
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var base_file = gguf.File.loadMmap(allocator, std.testing.io, base_path) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer base_file.deinit();
    const base = try qwen3.Config.fromGguf(&base_file);
    var model = try qwen3.Model.loadGgufFromFile(&ctx, &base_file, base);
    defer model.deinit();

    var sh = blk: {
        var file = gguf.File.loadMmap(allocator, std.testing.io, shine_gguf_path) catch |err| switch (err) {
            error.FileNotFound => return error.SkipZigTest,
            else => return err,
        };
        defer file.deinit();
        break :blk try shine.Shine.loadGgufFromFile(&ctx, &file, base);
    };
    defer sh.deinit();

    const evidence_ids = readIds(allocator, "evidence_ids.i32.bin") catch return error.SkipZigTest;
    defer allocator.free(evidence_ids);

    // Stage 1: encoder parity. Reference: fp32 weights; port: f16 (exact
    // values) — residual-stream drift is kernel-order only, measured
    // against the layer-max magnitude.
    var memory_states = try shine.encodeMemoryStates(&model, &sh, &ctx, evidence_ids);
    defer memory_states.deinit();
    const state_len = sh.config.num_layers * sh.config.num_mem_token * sh.config.hidden_size;
    const golden_memory = try readF32(allocator, "memory_states.f32.bin", state_len);
    defer allocator.free(golden_memory);
    const memory_drift = Drift.measure(try memory_states.dataConst(), golden_memory);
    std.debug.print("shine encoder [{s}] memory states: max|diff| {d} (max|ref| {d}, rel {d})\n", .{ base_path, memory_drift.max_abs, memory_drift.max_ref, memory_drift.rel() });
    // Measured 5.1e-3 rel at the residual-stream outlier (|ref| ~1.7e3):
    // the fp32 reference vs this port's f16 weight/GEMM path, accumulated
    // over 36 layers. The functional gate is the greedy token equality
    // below; this bound catches structural breakage, not kernel drift.
    try std.testing.expect(memory_drift.rel() < 2e-2);

    // Stages 2-3 on OUR encoder output (full-chain adapter).
    var plain = try shine.m2pForward(&sh, &ctx, &memory_states);
    defer plain.deinit();
    var adapter = try shine.sliceLora(&ctx, base, sh.config.lora_r, sh.config.scale, &plain);
    defer adapter.deinit();

    // Stage 4: adapted conversation, both turns.
    inline for ([_][]const u8{ "turn1", "turn2" }) |turn| {
        const prompt = try readIds(allocator, turn ++ "_ids.i32.bin");
        defer allocator.free(prompt);
        const golden_logits = try readF32(allocator, turn ++ "_logits.f32.bin", reference_vocab);
        defer allocator.free(golden_logits);
        const golden_greedy = try readIds(allocator, turn ++ "_greedy_ids.i32.bin");
        defer allocator.free(golden_greedy);

        var kv = try model.initKvCache(&ctx, 256);
        defer kv.deinit();
        var logits = try shine.forwardStep(&model, &adapter, &ctx, &kv, prompt, 0);
        defer logits.deinit();
        const logits_values = (try logits.dataConst())[0..reference_vocab];
        const logit_drift = Drift.measure(logits_values, golden_logits);
        std.debug.print("shine [{s}] {s} logits: max|diff| {d} (max|ref| {d})\n", .{ base_path, turn, logit_drift.max_abs, logit_drift.max_ref });
        // Measured 0.15 on |ref| ~16.6 (0.9%): the f16 activation path
        // again. The greedy token equality below is the functional gate.
        try std.testing.expect(logit_drift.max_abs < 0.5);

        var out_tokens: [16]usize = undefined;
        var greedy_kv = try model.initKvCache(&ctx, 256);
        defer greedy_kv.deinit();
        const produced = try shine.greedy(&model, &adapter, &ctx, &greedy_kv, prompt, &out_tokens, .{
            .max_new_tokens = golden_greedy.len,
            .stop_token = im_end,
            .vocab_limit = reference_vocab,
        });
        std.debug.print("shine [{s}] {s} greedy: got {any}, want {any}\n", .{ base_path, turn, out_tokens[0..produced], golden_greedy });
        try std.testing.expectEqualSlices(usize, golden_greedy, out_tokens[0..produced]);
    }
}
