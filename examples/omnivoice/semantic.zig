//! SemanticEncoder forward (semantic-enc.h): refines the mean+decimated
//! HuBERT features `[T_s, 768]` into `e_semantic [T_s, 768]`:
//!
//!   x = conv(k=3, p=1, NO bias)
//!   2 × block: { 2 × res unit [skip = x; x = ELU(x); x = conv1(k=3, p=1,
//!   d=1, NO bias); x = ELU(x); x = conv2(k=1, NO bias); x = skip + x];
//!   then conv(k=3, p=1, WITH bias) }
//!
//! ELU comes BEFORE each conv (pre-activation) and the skip wraps the whole
//! ELU-conv-ELU-conv chain. All channel counts 768, stride 1, T unchanged.
//! Internal layout is Fucina's `[T, C]` rows.

const std = @import("std");
const fucina = @import("fucina");

const codec = @import("codec.zig");

const ExecContext = fucina.ExecContext;

/// Activation rows `[T, 768]` (channel axis tagged `.in`).
pub const Act = fucina.Tensor(.{ .seq, .in });

/// Runs the SemanticEncoder on the HuBERT features (`[T_s, 768]` rows).
/// The whole stack runs host-side with the reference's exact arithmetic
/// (ggml-parity f16 convs + libm-parity ELU).
pub fn forward(ctx: *ExecContext, sem: *const codec.SemanticEncoder, features: *const Act) !Act {
    const allocator = ctx.allocator;

    var x = try codec.ggmlConv1d(allocator, try features.dataConst(), features.dim(.seq), features.dim(.in), &sem.conv_w, null, 1, 1, 1);
    errdefer allocator.free(x.data);

    for (&sem.blocks) |*blk| {
        for (&blk.res) |*ru| {
            // skip = x; ELU → conv1(k=3, p=d, dil=d) → ELU → conv2(k=1);
            // x = skip + x. `x` stays untouched as the skip.
            const e1 = try allocator.dupe(f32, x.data);
            defer allocator.free(e1);
            codec.eluGgml(e1);
            const c1 = try codec.ggmlConv1d(allocator, e1, x.t, x.c, &ru.conv1_w, null, 1, ru.dilation, ru.dilation);
            defer allocator.free(c1.data);
            codec.eluGgml(c1.data);
            const c2 = try codec.ggmlConv1d(allocator, c1.data, c1.t, c1.c, &ru.conv2_w, null, 1, 0, 1);
            std.debug.assert(c2.t == x.t and c2.c == x.c);
            for (c2.data, x.data) |*v, s| v.* = s + v.*; // ggml_add(skip, x)
            allocator.free(x.data);
            x = c2;
        }
        // Post-block conv: k=3, p=1, WITH bias.
        const conved = try codec.ggmlConv1d(allocator, x.data, x.t, x.c, &blk.conv_w, blk.conv_b, 1, 1, 1);
        allocator.free(x.data);
        x = conved;
    }

    const result = try Act.fromSlice(ctx, .{ x.t, x.c }, x.data);
    allocator.free(x.data);
    return result;
}

test {
    _ = @import("semantic_tests.zig");
}
