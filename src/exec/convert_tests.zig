//! Typed cast kernels: bf16 vector lanes against the scalar converters.
//! Force-imported by `exec.zig`.

const std = @import("std");
const backend_mod = @import("../backend.zig");
const exec = @import("../exec.zig");
const exec_elementwise = @import("elementwise.zig");
const exec_row_ops = @import("row_ops.zig");
const exec_moe_chain = @import("moe_chain.zig");
const dtype_mod = @import("../dtype.zig");
const fpenv = @import("../fpenv.zig");
const parallel = @import("../parallel.zig");
const rng = @import("../rng.zig");
const tensor = @import("../tensor.zig");

const Allocator = std.mem.Allocator;
const Tensor = tensor.Tensor;
const ExecContext = exec.ExecContext;
const LayoutClass = exec.LayoutClass;
const CrossEntropyOptions = exec.CrossEntropyOptions;
const Reduction = exec.Reduction;

test "castTyped bf16 vector lanes match the scalar converters bit-for-bit" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Edge patterns: signed zeros, exact/tie/rounding mantissas, ±inf,
    // quiet + signaling NaN payloads, subnormals, max finite. 67 elements
    // force both the vector body and the scalar tail.
    const patterns = [_]u32{
        0x00000000, 0x80000000, 0x3f800000, 0x3f808000, 0x3f818000, 0x3f7fffff,
        0x7f800000, 0xff800000, 0x7fc00001, 0x7f800001, 0xffc12345, 0x80000001,
        0x00000001, 0x807fffff, 0x40490fdb, 0x7f7fffff, 0xff7fffff, 0x33800000,
        0xc2f6e979, 0x3d800800,
    };
    var values: [67]f32 = undefined;
    for (&values, 0..) |*v, i| v.* = @bitCast(patterns[i % patterns.len]);

    var x = try ctx.fromSlice(&.{values.len}, &values);
    defer x.deinit();
    var narrowed = try ctx.castTyped(.f32, .bf16, &x);
    defer narrowed.deinit();
    for (narrowed.dataConst(), values) |got, value| {
        try std.testing.expectEqual(dtype_mod.f32ToBf16(value), got);
    }

    var widened = try ctx.castTyped(.bf16, .f32, &narrowed);
    defer widened.deinit();
    for (widened.dataConst(), narrowed.dataConst()) |got, bits| {
        // Bit compare: NaN payloads must survive, and nan != nan by value.
        try std.testing.expectEqual(@as(u32, @bitCast(dtype_mod.bf16ToF32(bits))), @as(u32, @bitCast(got)));
    }
}
