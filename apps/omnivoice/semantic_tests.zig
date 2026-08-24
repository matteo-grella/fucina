//! Tests for semantic.zig: wiring order of the res units (pre-activation
//! ELU, skip around the whole ELU-conv-ELU-conv chain) and the block-level
//! bias adds, on tiny synthetic weights. Real-weight parity is covered by
//! the env-gated encode test in rvq_tests.zig.

const std = @import("std");
const fucina = @import("fucina");

const codec = @import("codec.zig");
const semantic = @import("semantic.zig");

const c = 3; // tiny channel count (forward reads dims from the weights)

/// Center-tap identity conv weight in the flat ggml rows layout
/// `w[oc][ic][k]` (0/1 values are exact in f16).
fn identityConv(allocator: std.mem.Allocator, k: usize) !codec.GgmlConvWeight {
    const data = try allocator.alloc(f16, k * c * c);
    @memset(data, 0.0);
    const center = k / 2;
    for (0..c) |ch| data[(ch * c + ch) * k + center] = 1.0;
    return .{ .data = data, .taps = k, .in_per_group = c, .out_ch = c, .groups = 1 };
}

fn zeroConv(allocator: std.mem.Allocator, k: usize) !codec.GgmlConvWeight {
    const data = try allocator.alloc(f16, k * c * c);
    @memset(data, 0.0);
    return .{ .data = data, .taps = k, .in_per_group = c, .out_ch = c, .groups = 1 };
}

fn elu(x: f32) f32 {
    return if (x > 0) x else std.math.expm1(x);
}

const Synthetic = struct {
    sem: codec.SemanticEncoder,

    fn deinit(self: *Synthetic, allocator: std.mem.Allocator) void {
        self.sem.deinit(allocator);
    }
};

/// Builds a SemanticEncoder where every conv is either identity (center
/// tap) or zero: initial conv identity; block 0 res unit 0 identity chain,
/// all other res units zero; block convs identity with bias `bias0/bias1`.
fn buildSynthetic(allocator: std.mem.Allocator, identity_res: bool, bias0: f32, bias1: f32) !Synthetic {
    var sem: codec.SemanticEncoder = undefined;
    sem.conv_w = try identityConv(allocator, 3);
    for (0..2) |bi| {
        for (0..2) |ri| {
            const use_identity = identity_res and bi == 0 and ri == 0;
            sem.blocks[bi].res[ri] = .{
                .conv1_w = if (use_identity) try identityConv(allocator, 3) else try zeroConv(allocator, 3),
                .conv2_w = if (use_identity) try identityConv(allocator, 1) else try zeroConv(allocator, 1),
                .dilation = 1,
            };
        }
        sem.blocks[bi].conv_w = try identityConv(allocator, 3);
        const b = try allocator.alloc(f32, c);
        @memset(b, if (bi == 0) bias0 else bias1);
        sem.blocks[bi].conv_b = b;
    }
    return .{ .sem = sem };
}

test "zero res units pass through; block biases accumulate" {
    const allocator = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var syn = try buildSynthetic(allocator, false, 0.25, -0.5);
    defer syn.deinit(allocator);

    const t = 4;
    var x_data: [t * c]f32 = undefined;
    for (&x_data, 0..) |*dst, i| dst.* = (@as(f32, @floatFromInt(i)) - 5.0) / 3.0;
    var x = try semantic.Act.fromSlice(&ctx, .{ t, c }, &x_data);
    defer x.deinit();

    var out = try semantic.forward(&ctx, &syn.sem, &x);
    defer out.deinit();
    try std.testing.expectEqual(@as(usize, t), out.dim(.seq));
    try std.testing.expectEqual(@as(usize, c), out.dim(.in));

    // Zero res-unit convs make each res unit the identity (skip + 0); the
    // two identity block convs then add their biases: y = x + 0.25 − 0.5.
    // Tolerance covers the two f16 im2col round-trips of the input values.
    const got = try out.dataConst();
    for (got, x_data) |g, xv| {
        try std.testing.expectApproxEqAbs(xv + 0.25 - 0.5, g, 2e-3);
    }
}

test "identity res unit computes skip + elu(elu(x)) (pre-activation order)" {
    const allocator = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var syn = try buildSynthetic(allocator, true, 0.0, 0.0);
    defer syn.deinit(allocator);

    // Single frame with f16-exact values: the center-tap identity convs see
    // only the frame itself, so the wiring order is checked exactly (up to
    // the f16 rounding of the elu outputs inside the conv).
    var x_data = [c]f32{ -1.0, 0.5, -0.25 };
    var x = try semantic.Act.fromSlice(&ctx, .{ 1, c }, &x_data);
    defer x.deinit();

    var out = try semantic.forward(&ctx, &syn.sem, &x);
    defer out.deinit();

    const got = try out.dataConst();
    for (got, x_data) |g, xv| {
        try std.testing.expectApproxEqAbs(xv + elu(elu(xv)), g, 2e-3);
    }
}
