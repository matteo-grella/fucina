const std = @import("std");
const fucina = @import("fucina");

pub fn main(init: std.process.Init) !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .batch, .in }).variable(
        &ctx,
        try ctx.fromSlice(&.{ 1, 2 }, &.{ 2, 3 }),
    );
    defer x.deinit();

    var w = try fucina.Tensor(.{ .in, .out }).variable(
        &ctx,
        try ctx.fromSlice(&.{ 2, 1 }, &.{ 4, 5 }),
    );
    defer w.deinit();

    var y = try x.dot(&ctx, &w, .in);
    defer y.deinit();

    var loss = try y.sumAll(&ctx);
    defer loss.deinit();

    try loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    var gw = (try w.grad(&ctx)).?;
    defer gw.deinit();

    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};
    const gx_data = try gx.dataConst();
    const gw_data = try gw.dataConst();
    try stdout.print("loss={d}\n", .{try loss.item()});
    try stdout.print("grad_x=[{d}, {d}]\n", .{ gx_data[0], gx_data[1] });
    try stdout.print("grad_w=[{d}, {d}]\n", .{ gw_data[0], gw_data[1] });
}
