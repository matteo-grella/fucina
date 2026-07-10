const std = @import("std");
const preproc = @import("preproc.zig");

const testing = std.testing;

const limits448 = preproc.Limits{ .patch = 14, .merge_h = 2, .merge_w = 2, .in_token_limit = 25600 };

test "preprocTarget rounds up to merge*patch multiples" {
    var tw: usize = 0;
    var th: usize = 0;
    preproc.preprocTarget(448, 448, limits448, &tw, &th);
    try testing.expectEqual(@as(usize, 448), tw);
    try testing.expectEqual(@as(usize, 448), th);

    preproc.preprocTarget(640, 480, limits448, &tw, &th);
    try testing.expectEqual(@as(usize, 644), tw); // ceil(640/28)*28
    try testing.expectEqual(@as(usize, 504), th); // ceil(480/28)*28
}

test "preprocTarget applies the token-limit downscale" {
    var tw: usize = 0;
    var th: usize = 0;
    // 8000x8000: grid 571*571 = 326041 > 25600 -> scale = sqrt(25600/326041).
    preproc.preprocTarget(8000, 8000, limits448, &tw, &th);
    // scale ≈ 0.280194; 8000*scale ≈ 2241.55 -> int() = 2241 -> ceil/28*28 = 2268.
    try testing.expectEqual(@as(usize, 2268), tw);
    try testing.expectEqual(@as(usize, 2268), th);
}

test "identity-size resize is skipped and patchify normalizes" {
    const allocator = testing.allocator;
    // 28x28 solid gray-ish gradient image -> 2x2 patch grid, no resize needed.
    const w = 28;
    const h = 28;
    var rgb: [w * h * 3]u8 = undefined;
    for (0..h) |y| for (0..w) |x| {
        rgb[(y * w + x) * 3 + 0] = @intCast((x * 255) / (w - 1));
        rgb[(y * w + x) * 3 + 1] = 128;
        rgb[(y * w + x) * 3 + 2] = @intCast((y * 255) / (h - 1));
    };
    var pre = try preproc.preprocess(allocator, &rgb, w, h, limits448);
    defer pre.deinit();
    try testing.expectEqual(@as(usize, 2), pre.gh);
    try testing.expectEqual(@as(usize, 2), pre.gw);
    // Patch t=3 (row 1, col 1), c=1 plane must be 128/127.5-1 everywhere.
    const patch_dim = 14 * 14 * 3;
    const g = pre.pixel_values[3 * patch_dim + 196 ..][0..196];
    for (g) |v| try testing.expectApproxEqAbs(@as(f32, 128.0 / 127.5 - 1.0), v, 1e-7);
}

test "pil bicubic matches Pillow on a hand-checked downscale" {
    // 4x1 -> 2x1 with a=-0.5 antialias: coefficients computed analytically.
    // filterscale=2, support=4, ksize=9; centers 1.0 and 3.0.
    // For out x=0: xmin=0, taps at (i+0.5-1.0)*0.5 for i=0..3 (xmax=4? bounds:
    // xmin=int(1-4+0.5)=-2 -> 0, xmax=int(1+4+0.5)=5 -> clamp 4).
    const allocator = testing.allocator;
    const src = [_]u8{ 0, 0, 0, 100, 100, 100, 200, 200, 200, 255, 255, 255 };
    const out = try preproc.pilBicubicResize(allocator, &src, 4, 1, 2, 1);
    defer allocator.free(out);
    // Weights w_i = cubic(-0.25), cubic(0.25), cubic(0.75), cubic(1.25) then
    // normalized; value = round(sum w_i * v_i).
    var w: [4]f64 = undefined;
    const xs = [_]f64{ -0.25, 0.25, 0.75, 1.25 };
    var ww: f64 = 0;
    for (xs, 0..) |x, i| {
        const a = -0.5;
        const ax = @abs(x);
        w[i] = if (ax < 1) ((a + 2) * ax - (a + 3)) * ax * ax + 1 else (((ax - 5) * ax + 8) * ax - 4) * a;
        ww += w[i];
    }
    var acc: f64 = 0;
    const vals = [_]f64{ 0, 100, 200, 255 };
    for (vals, 0..) |v, i| acc += v * (w[i] / ww);
    const expected: u8 = @intCast(@as(i64, @intFromFloat(@round(acc))));
    try testing.expectEqual(expected, out[0]);
}
