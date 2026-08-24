//! Tests for hubert.zig: feature-extractor length algebra and the
//! ggml-parity grouped conv group mapping the pos_conv path relies on
//! (group g of output channels reads input channels [g·icpg, (g+1)·icpg) —
//! the ggml/PyTorch grouping convention). The full HuBERT forward is
//! covered by the env-gated encode parity test in rvq_tests.zig.

const std = @import("std");
const fucina = @import("fucina");

const codec = @import("codec.zig");
const hubert = @import("hubert.zig");

test "featOutputLength matches the reference per-layer chain" {
    // Bookkeeping example from the reference map (n_24k = 960 ⇒ n_16k = 640
    // ⇒ padded 960): 191 → 95 → 47 → 23 → 11 → 5 → 2.
    try std.testing.expectEqual(@as(?usize, 2), hubert.featOutputLength(960));
    // en_4.wav: 70400 samples @ 16 kHz + 320 pad ⇒ 220 frames (stride 320).
    try std.testing.expectEqual(@as(?usize, 220), hubert.featOutputLength(70720));
    // Too short for the k=10 first conv.
    try std.testing.expectEqual(@as(?usize, null), hubert.featOutputLength(9));

    // Chain each layer manually and compare for a few sizes.
    for ([_]usize{ 960, 4321, 70720 }) |n| {
        var t = n;
        for (codec.hubert_feat_kernels, codec.hubert_feat_strides) |k, s| {
            t = (t - k) / s + 1;
        }
        try std.testing.expectEqual(@as(?usize, t), hubert.featOutputLength(n));
    }
}

test "ggml grouped conv maps group g inputs to group g outputs (pos_conv layout)" {
    const allocator = std.testing.allocator;

    // T=4, C=4, groups=2 (icpg = ocpg = 2), k=2, pad=1 — the even kernel
    // yields T+1 output frames exactly like the k=128 pad=64 pos conv.
    const t_in = 4;
    const c = 4;
    const groups = 2;
    const icpg = c / groups;
    const k = 2;
    const pad = 1;

    // Flat ggml rows w[oc][ic][kk], f16-exact small integers.
    var w_data: [c * icpg * k]f16 = undefined;
    for (0..c) |oc| {
        for (0..icpg) |ic| {
            for (0..k) |kk| {
                w_data[(oc * icpg + ic) * k + kk] = @floatFromInt(1 + oc * 8 + ic * 4 + kk);
            }
        }
    }
    var w = codec.GgmlConvWeight{
        .data = try allocator.dupe(f16, &w_data),
        .taps = k,
        .in_per_group = icpg,
        .out_ch = c,
        .groups = groups,
    };
    defer w.deinit(allocator);

    var x_data: [t_in * c]f32 = undefined;
    for (&x_data, 0..) |*dst, i| dst.* = @floatFromInt(i + 1);

    const out = try codec.ggmlConv1d(allocator, &x_data, t_in, c, &w, null, 1, pad, 1);
    defer allocator.free(out.data);
    const t_out = t_in + 2 * pad - (k - 1) - 1 + 1; // = T + 1 (even kernel)
    try std.testing.expectEqual(@as(usize, t_out), out.t);
    try std.testing.expectEqual(@as(usize, c), out.c);

    // Naive grouped cross-correlation on the zero-padded input. Inputs are
    // small integers, exact in f16, so the result is exact.
    for (0..t_out) |ti| {
        for (0..c) |oc| {
            const g = oc / (c / groups);
            var want: f32 = 0.0;
            for (0..icpg) |ic| {
                for (0..k) |kk| {
                    const pos = ti + kk;
                    if (pos < pad or pos - pad >= t_in) continue;
                    want += @as(f32, @floatCast(w_data[(oc * icpg + ic) * k + kk])) * x_data[(pos - pad) * c + (g * icpg + ic)];
                }
            }
            try std.testing.expectApproxEqAbs(want, out.data[ti * c + oc], 1e-4);
        }
    }
}
