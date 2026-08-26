//! Golden parity tests for the delta-rule linear-attention kernels
//! (`delta_attention.zig`) against the Kimi reference implementation.
//! The goldens are produced by `tools/kimi3_goldens.py --kda-op` (the
//! reference `fused_recurrent_kda` with the exact in-kernel flags the Kimi
//! models use) into `models/kimi-k3-0.40b/goldens/`; the test skips when
//! they are absent (models/ is a local, untracked directory).

const std = @import("std");

const fucina = @import("fucina");
const delta_attention = @import("delta_attention.zig");

const ExecContext = fucina.ExecContext;

const goldens_dir = "models/kimi-k3-0.40b/goldens";

fn readGolden(allocator: std.mem.Allocator, comptime name: []const u8, expected_len: usize) !?[]f32 {
    const bytes = std.Io.Dir.cwd().readFileAlloc(
        std.testing.io,
        goldens_dir ++ "/" ++ name ++ ".bin",
        allocator,
        .limited(1 << 24),
    ) catch return null;
    defer allocator.free(bytes);
    if (bytes.len != expected_len * @sizeOf(f32)) return error.GoldenShapeMismatch;
    const out = try allocator.alloc(f32, expected_len);
    @memcpy(std.mem.sliceAsBytes(out), bytes);
    return out;
}

test "kdaRecurrent matches the reference fused_recurrent_kda goldens" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    // Generator shapes (tools/kimi3_goldens.py --kda-op).
    const T = 17;
    const H = 4;
    const K = 32;
    const V = 32;

    const q = (try readGolden(allocator, "kda_q", T * H * K)) orelse return; // goldens absent: skip
    defer allocator.free(q);
    const k = (try readGolden(allocator, "kda_k", T * H * K)) orelse return;
    defer allocator.free(k);
    const v = (try readGolden(allocator, "kda_v", T * H * V)) orelse return;
    defer allocator.free(v);
    const g = (try readGolden(allocator, "kda_g", T * H * K)) orelse return;
    defer allocator.free(g);
    const beta = (try readGolden(allocator, "kda_beta", T * H)) orelse return;
    defer allocator.free(beta);
    const a_log = (try readGolden(allocator, "kda_a_log", H)) orelse return;
    defer allocator.free(a_log);
    const dt_bias = (try readGolden(allocator, "kda_dt_bias", H * K)) orelse return;
    defer allocator.free(dt_bias);
    const o_ref = (try readGolden(allocator, "kda_o", T * H * V)) orelse return;
    defer allocator.free(o_ref);
    const state_ref = (try readGolden(allocator, "kda_state", H * K * V)) orelse return;
    defer allocator.free(state_ref);

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var q_t = try ctx.fromSlice(.f32, &.{ T, H, K }, q);
    defer q_t.deinit();
    var k_t = try ctx.fromSlice(.f32, &.{ T, H, K }, k);
    defer k_t.deinit();
    var v_t = try ctx.fromSlice(.f32, &.{ T, H, V }, v);
    defer v_t.deinit();
    var g_t = try ctx.fromSlice(.f32, &.{ T, H, K }, g);
    defer g_t.deinit();
    var beta_t = try ctx.fromSlice(.f32, &.{ T, H }, beta);
    defer beta_t.deinit();

    var result = try delta_attention.kdaRecurrent(&ctx, &q_t, &k_t, &v_t, &g_t, &beta_t, a_log, dt_bias, null, 0);
    defer result.deinit();

    for (o_ref, result.o.dataConst()) |expected, actual| {
        try std.testing.expectApproxEqAbs(expected, actual, 2e-5);
    }
    for (state_ref, result.state.dataConst()) |expected, actual| {
        try std.testing.expectApproxEqAbs(expected, actual, 2e-4);
    }
}
