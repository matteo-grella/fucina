var x = try Tensor(.{.d})
    .fromSlice(&ctx, .{2}, &.{ 1, 2 });
defer x.deinit();
for (0..3) |_| {
    // frees the old x and rebinds in one move —
    // on error the old x stays valid
    x = try ctx.replace(x, x.scale(&ctx, 2.0));
}
