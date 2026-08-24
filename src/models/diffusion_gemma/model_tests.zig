//! Behavioral tests for the DiffusionGemma sampler (`diffusion_gemma.zig`):
//! the entropy-bound sampler pass — per-position argmax, softmax entropy, the
//! multinomial draw, and the sparse self-conditioning candidate collection.
const std = @import("std");
const fucina = @import("fucina");
const diffusion_gemma = @import("model.zig");

const ExecContext = fucina.ExecContext;
const samplerPass = diffusion_gemma.samplerPass;

test "sampler pass: argmax, entropy, multinomial, and SC candidates" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // 2 positions, 4-token vocab. Row 0 is peaked on id 2; row 1 is uniform.
    const v = 4;
    const data = [_]f32{
        0, 0, 8, 0,
        1, 1, 1, 1,
    };
    var logits = try fucina.Tensor(.{ .seq, .vocab }).fromSlice(&ctx, .{ 2, v }, &data);
    defer logits.deinit();

    const u = [_]f32{ 0.5, 0.6 };
    var pass = try samplerPass(&ctx, &logits, 1.0, &u, .{ .sc_p_min = 1e-3, .sc_max_per_row = 8 });
    defer pass.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 2), pass.results[0].argmax);
    // Row 0 entropy ~ 0 (peaked); row 1 = ln(4).
    try std.testing.expect(pass.results[0].entropy < 0.01);
    try std.testing.expectApproxEqAbs(@log(@as(f32, 4)), pass.results[1].entropy, 1e-4);
    // Row 0 multinomial at u=0.5 must hit the dominant token.
    try std.testing.expectEqual(@as(usize, 2), pass.results[0].sampled);
    // Row 1 at u=0.6 hits the third quartile -> id 2.
    try std.testing.expectEqual(@as(usize, 2), pass.results[1].sampled);

    const sc = pass.sc.?;
    // Row 0 keeps only the dominant id (others are ~3e-4 < 1e-3 of mass);
    // row 1 keeps all four at 0.25.
    try std.testing.expectEqual(@as(usize, 1), sc.row_offsets[1] - sc.row_offsets[0]);
    try std.testing.expectEqual(@as(usize, 2), sc.ids[0]);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sc.probs[0], 1e-5);
    try std.testing.expectEqual(@as(usize, 4), sc.row_offsets[2] - sc.row_offsets[1]);
    for (sc.probs[sc.row_offsets[1]..sc.row_offsets[2]]) |p| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.25), p, 1e-5);
    }
}
