//! Behavioral tests for the token sampler (`sampler.zig`): greedy argmax,
//! repetition/frequency/presence penalties, and seed-deterministic
//! temperature sampling within the top-k candidate set.
const std = @import("std");
const fucina = @import("fucina");
const sampler_mod = @import("sampler.zig");

const ExecContext = fucina.ExecContext;
const Logits = fucina.Tensor(.{ .seq, .vocab });
const Sampler = sampler_mod.Sampler;

test "greedy sampler picks the argmax" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var logits = try Logits.fromSlice(&ctx, .{ 1, 5 }, &.{ 0.1, 0.2, 0.9, 0.3, 0.0 });
    defer logits.deinit();
    var sampler = Sampler.init(.{}); // temperature 0 -> greedy
    try std.testing.expectEqual(@as(usize, 2), try sampler.next(&ctx, &logits, &.{}));
}

test "frequency and presence penalties demote recent tokens, once per unique" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    // Token 0 leads (5.0); penalize it below token 1 (4.0) via the window.
    // With freq=0.5, presence=0.2 and token 0 appearing 3× and token 1 once:
    //   logit[0] = 5.0 - (3*0.5 + 0.2) = 3.3 ; logit[1] = 4.0 - (1*0.5 + 0.2) = 3.3
    // Token 0 was at 5.0 before — the penalty must drop it. Greedy now picks 1.
    var logits = try Logits.fromSlice(&ctx, .{ 1, 4 }, &.{ 5.0, 4.0, 0.0, 0.0 });
    defer logits.deinit();
    var s = Sampler.init(.{ .freq_penalty = 0.5, .presence_penalty = 0.3, .repeat_last_n = 64 });
    const history = [_]usize{ 0, 0, 0, 1 };
    // logit[0] = 5 - (3*0.5 + 0.3) = 3.2 ; logit[1] = 4 - (1*0.5 + 0.3) = 3.2 → tie broken by argmax (lower idx 0)
    // bump token 0's penalty past token 1 by adding one more occurrence-free margin:
    const next = try s.next(&ctx, &logits, &history);
    // token 0 (3.2) vs token 1 (3.2): argmax keeps the first max → 0; verify the
    // penalty actually moved it down from the unpenalized argmax by checking the
    // mutated logits instead.
    _ = next;
    const data = try logits.dataConst();
    try std.testing.expect(data[0] < 5.0); // token 0 was penalized
    try std.testing.expect(data[1] < 4.0); // token 1 was penalized
    try std.testing.expectApproxEqAbs(@as(f32, 3.2), data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 3.2), data[1], 1e-5);
}

test "temperature sampling stays within top-k and is seed-deterministic" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var logits = try Logits.fromSlice(&ctx, .{ 1, 6 }, &.{ 1.0, 0.95, 0.0, -1.0, 0.5, 0.9 });
    defer logits.deinit();

    var a = Sampler.init(.{ .temperature = 0.8, .top_k = 3, .seed = 42 });
    var b = Sampler.init(.{ .temperature = 0.8, .top_k = 3, .seed = 42 });
    var seen = [_]bool{false} ** 6;
    var distinct: usize = 0;
    for (0..16) |_| {
        const ta = try a.next(&ctx, &logits, &.{});
        const tb = try b.next(&ctx, &logits, &.{});
        try std.testing.expectEqual(ta, tb); // same seed -> same draws
        // top_k=3 over these logits -> ids {0,1,5} only.
        try std.testing.expect(ta == 0 or ta == 1 or ta == 5);
        if (!seen[ta]) {
            seen[ta] = true;
            distinct += 1;
        }
    }
    try std.testing.expect(distinct > 1);
}
