//! Parity gate for the descriptor runner (gate 1 of the universal-runner
//! design note): on a real Qwen3-0.6B GGUF, the runner driven by the
//! descriptor `fromGguf` derives must reproduce the hand qwen3 port
//! token-for-token — prefill logits bitwise, then every decode step's
//! logits bitwise along a greedy chain, through both models' own KV
//! caches. Skips without models/. Force-imported by `llm.zig`'s test block.

const std = @import("std");
const fucina = @import("fucina");
const runner = @import("runner.zig");
const qwen3 = @import("qwen3/model.zig");

const gguf = fucina.gguf;
const ExecContext = fucina.ExecContext;

fn argmaxRow(row: []const f32) usize {
    var best: usize = 0;
    for (row, 0..) |x, i| {
        if (x > row[best]) best = i;
    }
    return best;
}

fn parityOnGguf(path: []const u8) !void {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var file = gguf.File.loadMmap(allocator, std.testing.io, path) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer file.deinit();

    const desc = try runner.Descriptor.fromGguf(&file);
    var generic = try runner.Model.loadGgufFromFile(&ctx, &file, desc);
    defer generic.deinit();

    var hand = try qwen3.Model.loadGgufFromFile(&ctx, &file, try qwen3.Config.fromGguf(&file));
    defer hand.deinit();

    const prompt = [_]usize{ 151644, 872, 198, 9707, 11, 1879, 0, 151645 };

    // Prefill parity: the no-cache last-logits entry, bitwise.
    {
        var got = try generic.forwardLastLogits(&ctx, &prompt);
        defer got.deinit();
        var want = try hand.forwardLastLogits(&ctx, &prompt);
        defer want.deinit();
        try std.testing.expectEqualSlices(f32, try want.dataConst(), try got.dataConst());
    }

    // Decode parity: each model prefills into ITS OWN cache, then a greedy
    // chain — every step's full logits row must match bitwise (which pins
    // the argmax chain too).
    var kv_generic = try generic.initKvCache(&ctx, 32);
    defer kv_generic.deinit();
    var kv_hand = try hand.initKvCache(&ctx, 32);
    defer kv_hand.deinit();

    var got = try generic.forwardStep(&ctx, &kv_generic, &prompt, 0);
    var want = try hand.forwardStep(&ctx, &kv_hand, &prompt, 0);
    var pos: usize = prompt.len;
    for (0..4) |_| {
        try std.testing.expectEqualSlices(f32, try want.dataConst(), try got.dataConst());
        const next = argmaxRow(try want.dataConst());
        want.deinit();
        got.deinit();
        got = try generic.forwardStep(&ctx, &kv_generic, &.{next}, pos);
        want = try hand.forwardStep(&ctx, &kv_hand, &.{next}, pos);
        pos += 1;
    }
    try std.testing.expectEqualSlices(f32, try want.dataConst(), try got.dataConst());
    want.deinit();
    got.deinit();
}

test "descriptor runner reproduces the hand qwen3 port bitwise (Q8_0; skips without models/)" {
    try parityOnGguf("models/Qwen3-0.6B-Q8_0.gguf");
}

test "descriptor runner reproduces the hand qwen3 port bitwise (Q4_K_M; skips without models/)" {
    try parityOnGguf("models/Qwen3-0.6B-Q4_K_M.gguf");
}
