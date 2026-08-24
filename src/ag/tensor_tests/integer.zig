//! Integer and bool tensors: wrapping and explicit integer math, bitwise
//! combinators, comparison and logical masks, saturating casts, and
//! mask-driven where/maskedFill with gradients.

const std = @import("std");
const backend_mod = @import("../../backend.zig");
const dtype_mod = @import("../../dtype.zig");
const exec_mod = @import("../../exec.zig");
const control = @import("../control.zig");
const core = @import("../core.zig");
const ag_tensor = @import("../tensor.zig");
const gradcheck_mod = @import("../gradcheck.zig");

const DType = dtype_mod.DType;
const ExecContext = exec_mod.ExecContext;
const GradState = core.GradState;
const Tensor = ag_tensor.Tensor;
const RawTensor = @import("../../tensor.zig").Tensor;

const util = @import("util.zig");
const expectCloseSlices = util.expectCloseSlices;

test "public Tensor where / maskedFill / zeroSlice / zeroRows values" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const T = Tensor(.{.d});
    var x = try T.fromSlice(&ctx, .{4}, &.{ 1, 2, 3, 4 });
    defer x.deinit();
    var y = try T.fromSlice(&ctx, .{4}, &.{ 10, 20, 30, 40 });
    defer y.deinit();

    // where: cond ? x : y
    var cond = try T.fromSlice(&ctx, .{4}, &.{ 1, 0, 1, 0 });
    defer cond.deinit();
    var w = try x.where(&ctx, cond, y);
    defer w.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 20, 3, 40 }, w.asRawTensor().dataConst());

    // maskedFill: mask ? value : x
    var mask = try T.fromSlice(&ctx, .{4}, &.{ 0, 1, 0, 1 });
    defer mask.deinit();
    var mf = try x.maskedFill(&ctx, mask, 99);
    defer mf.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 99, 3, 99 }, mf.asRawTensor().dataConst());

    // zeroSlice / zeroRows vs the ctx kernels
    const M = Tensor(.{ .batch, .n });
    var m = try M.fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer m.deinit();
    var zs = try m.zeroSlice(&ctx, .n, 1, 1);
    defer zs.deinit();
    var zs_want = try ctx.zeroSlice(2, m.asRawTensor(), 1, 1, 1);
    defer zs_want.deinit();
    try std.testing.expectEqualSlices(f32, zs_want.dataConst(), zs.asRawTensor().dataConst());

    var zr = try m.zeroRows(&ctx, .batch, &.{1});
    defer zr.deinit();
    var zr_want = try ctx.zeroRows(2, m.asRawTensor(), 0, &.{1});
    defer zr_want.deinit();
    try std.testing.expectEqualSlices(f32, zr_want.dataConst(), zr.asRawTensor().dataConst());
}

test "public Tensor comparison and logical ops produce constant 0/1 masks" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const V = Tensor(.{.d});
    // Variables in, constant masks out: comparisons are non-differentiable
    // (the argmax precedent), so downstream where/maskedFill treat them as
    // plain masks.
    var a = try V.variableFromSlice(&ctx, .{4}, &.{ 1, 2, 3, 4 });
    defer a.deinit();
    var b = try V.variableFromSlice(&ctx, .{4}, &.{ 1, 5, 2, 4 });
    defer b.deinit();

    // torch.lt(a, b).float() = {0, 1, 0, 0}; torch.ge = {1, 0, 1, 1}.
    var lt = try a.compare(&ctx, .lt, &b);
    defer lt.deinit();
    comptime std.debug.assert(@TypeOf(lt).dtype == .bool);
    try std.testing.expect(!lt.requiresGrad());
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false }, try lt.dataConst());
    var ge = try a.compare(&ctx, .ge, &b);
    defer ge.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, true }, try ge.dataConst());

    // torch.eq(a, 4) and torch.le(a, 2): .bool masks.
    var eq4 = try a.compare(&ctx, .eq, 4);
    defer eq4.deinit();
    try std.testing.expect(!eq4.requiresGrad());
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, true }, try eq4.dataConst());
    var le2 = try a.compare(&ctx, .le, 2);
    defer le2.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, false }, try le2.dataConst());

    // Logical combinators over truthiness: .bool in, .bool out.
    var both = try le2.logicalAnd(&ctx, &eq4);
    defer both.deinit();
    try std.testing.expect(!both.requiresGrad());
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false }, try both.dataConst());
    var either = try le2.logicalOr(&ctx, &eq4);
    defer either.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, try either.dataConst());
    var one_of = try le2.logicalXor(&ctx, &eq4);
    defer one_of.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, try one_of.dataConst());
    var neither = try either.logicalNot(&ctx);
    defer neither.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false }, try neither.dataConst());
    // Count the mask (i64) and cast for the mask-multiply idiom.
    var n_true = try either.sumAll(&ctx);
    defer n_true.deinit();
    try std.testing.expectEqual(@as(i64, 3), try n_true.item());
    var as_f32 = try either.to(&ctx, .f32);
    defer as_f32.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 0, 1 }, try as_f32.dataConst());

    // The mask feeds where/maskedFill directly: where(le2, a, b).
    var picked = try a.where(&ctx, le2, &b);
    defer picked.deinit();
    try expectCloseSlices(&.{ 1, 2, 2, 4 }, try picked.dataConst(), 0);
}

test "public Tensor isnan isinf isfinite produce constant masks" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const inf = std.math.inf(f32);
    const V = Tensor(.{.d});
    var x = try V.fromSlice(&ctx, .{5}, &.{ 1, inf, -inf, std.math.nan(f32), 0 });
    defer x.deinit();

    var nan_mask = try x.isnan(&ctx);
    defer nan_mask.deinit();
    comptime std.debug.assert(@TypeOf(nan_mask).dtype == .bool);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, true, false }, try nan_mask.dataConst());
    var inf_mask = try x.isinf(&ctx);
    defer inf_mask.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, false, false }, try inf_mask.dataConst());
    var finite_mask = try x.isfinite(&ctx);
    defer finite_mask.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false, true }, try finite_mask.dataConst());

    // Masks are constants even off a grad-tracked source, and need no scope.
    var xv = try V.variableFromSlice(&ctx, .{5}, &.{ 1, inf, -inf, std.math.nan(f32), 0 });
    defer xv.deinit();
    var m = try xv.isfinite(&ctx);
    defer m.deinit();
    try std.testing.expect(!m.requiresGrad());
}

test "public Tensor any all reductions follow torch truthiness" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const M = Tensor(.{ .row, .col });
    // Rows: {0, 0, 0} → any 0, all 0; {1, 0, 2} → any 1, all 0;
    // {3, NaN, -1} → NaN is truthy (torch.any/all): any 1, all 1.
    var x = try M.fromSlice(&ctx, .{ 3, 3 }, &.{ 0, 0, 0, 1, 0, 2, 3, std.math.nan(f32), -1 });
    defer x.deinit();

    var any_row = try x.any(&ctx, .col);
    defer any_row.deinit();
    comptime std.debug.assert(@TypeOf(any_row).dtype == .bool);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true }, try any_row.dataConst());
    var all_row = try x.all(&ctx, .col);
    defer all_row.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, false, true }, try all_row.dataConst());

    var any_scalar = try x.anyAll(&ctx);
    defer any_scalar.deinit();
    try std.testing.expectEqual(true, try any_scalar.item());
    var all_scalar = try x.allAll(&ctx);
    defer all_scalar.deinit();
    try std.testing.expectEqual(false, try all_scalar.item());
}

test "integer tensors: rem/mod remainders and bitwise combinators" {
    @setEvalBranchQuota(1_000_000);
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // rem pairs divTrunc (sign of the dividend), mod pairs divFloor
    // (sign of the divisor) — the mixed-sign quartet separates them.
    const I8 = Tensor(.{ .dtype = .i8, .tags = .{.d} });
    var a = try I8.fromSlice(&ctx, .{4}, &.{ -7, -7, 7, 7 });
    defer a.deinit();
    var b = try I8.fromSlice(&ctx, .{4}, &.{ 2, -2, 2, -2 });
    defer b.deinit();
    var r = try a.rem(&ctx, &b);
    defer r.deinit();
    try std.testing.expectEqualSlices(i8, &.{ -1, -1, 1, 1 }, try r.dataConst());
    var m = try a.mod(&ctx, &b);
    defer m.deinit();
    try std.testing.expectEqualSlices(i8, &.{ 1, -1, 1, -1 }, try m.dataConst());

    // minInt % -1 is 0 (the wrapping-div contract), zero divisor errors.
    var min_int = try I8.fromSlice(&ctx, .{4}, &.{ -128, -128, 5, 5 });
    defer min_int.deinit();
    var neg_one = try I8.fromSlice(&ctx, .{4}, &.{ -1, -1, -1, -1 });
    defer neg_one.deinit();
    var wrapped = try min_int.mod(&ctx, &neg_one);
    defer wrapped.deinit();
    try std.testing.expectEqualSlices(i8, &.{ 0, 0, 0, 0 }, try wrapped.dataConst());
    var zero = try I8.fromSlice(&ctx, .{4}, &.{ 1, 0, 1, 1 });
    defer zero.deinit();
    try std.testing.expectError(error.DivisionByZero, a.rem(&ctx, &zero));
    try std.testing.expectError(error.DivisionByZero, a.mod(&ctx, &zero));

    // Bitwise combinators work on the two's-complement bit patterns.
    var x = try I8.fromSlice(&ctx, .{4}, &.{ -5, 12, 0, -1 });
    defer x.deinit();
    var y = try I8.fromSlice(&ctx, .{4}, &.{ 3, 10, -1, -1 });
    defer y.deinit();
    var xor = try x.bitXor(&ctx, &y);
    defer xor.deinit();
    try std.testing.expectEqualSlices(i8, &.{ -8, 6, -1, 0 }, try xor.dataConst());
    var band = try x.bitAnd(&ctx, &y);
    defer band.deinit();
    try std.testing.expectEqualSlices(i8, &.{ 3, 8, 0, -1 }, try band.dataConst());
    var bor = try x.bitOr(&ctx, &y);
    defer bor.deinit();
    try std.testing.expectEqualSlices(i8, &.{ -5, 14, -1, -1 }, try bor.dataConst());

    // The n-gram-hash composition at i64 width — wrapping mul, xor, and
    // floored mod against a broadcast [head] modulus vector — matches
    // numpy exactly (values precomputed with numpy 2.4 int64 semantics).
    const Row = Tensor(.{ .dtype = .i64, .tags = .{.seq} });
    const Mods = Tensor(.{ .dtype = .i64, .tags = .{.head} });
    var t0 = try Row.fromSlice(&ctx, .{3}, &.{ 5, 1, 3 });
    defer t0.deinit();
    var t1 = try Row.fromSlice(&ctx, .{3}, &.{ 2, 5, 1 });
    defer t1.deinit();
    var m0 = try Row.fromSlice(&ctx, .{3}, &.{ 6148914691236517205, 6148914691236517205, 6148914691236517205 });
    defer m0.deinit();
    var m1 = try Row.fromSlice(&ctx, .{3}, &.{ -7905747460161236407, -7905747460161236407, -7905747460161236407 });
    defer m1.deinit();
    var p0 = try t0.mul(&ctx, &m0);
    defer p0.deinit();
    var p1 = try t1.mul(&ctx, &m1);
    defer p1.deinit();
    try std.testing.expectEqualSlices(i64, &.{ -6148914691236517207, 6148914691236517205, -1 }, try p0.dataConst());
    var mix = try p0.bitXor(&ctx, &p1);
    defer mix.deinit();
    try std.testing.expectEqualSlices(i64, &.{ -8198552921648689605, -8198552921648689608, 7905747460161236406 }, try mix.dataConst());
    var mods = try Mods.fromSlice(&ctx, .{2}, &.{ 97, 89 });
    defer mods.deinit();
    var hashes = try mix.mod(&ctx, &mods);
    defer hashes.deinit();
    try std.testing.expectEqual(@as(usize, 3), hashes.dim(.seq));
    try std.testing.expectEqual(@as(usize, 2), hashes.dim(.head));
    try std.testing.expectEqualSlices(i64, &.{ 72, 2, 69, 88, 53, 66 }, try hashes.dataConst());
}

test "integer tensors: wrapping pointwise, explicit division, i64 reductions" {
    @setEvalBranchQuota(1_000_000);
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const I8 = Tensor(.{ .dtype = .i8, .tags = .{.d} });
    var a = try I8.fromSlice(&ctx, .{4}, &.{ 127, -128, 10, -7 });
    defer a.deinit();
    var b = try I8.fromSlice(&ctx, .{4}, &.{ 1, -1, 3, 2 });
    defer b.deinit();

    // Wrapping two's-complement arithmetic.
    var summed = try a.add(&ctx, &b);
    defer summed.deinit();
    try std.testing.expectEqualSlices(i8, &.{ -128, 127, 13, -5 }, try summed.dataConst());
    var product = try a.mul(&ctx, &b);
    defer product.deinit();
    try std.testing.expectEqualSlices(i8, &.{ 127, -128, 30, -14 }, try product.dataConst());

    // Explicit division: trunc toward zero vs floor toward -inf.
    var q_trunc = try a.divTrunc(&ctx, &b);
    defer q_trunc.deinit();
    try std.testing.expectEqualSlices(i8, &.{ 127, -128, 3, -3 }, try q_trunc.dataConst());
    var q_floor = try a.divFloor(&ctx, &b);
    defer q_floor.deinit();
    try std.testing.expectEqualSlices(i8, &.{ 127, -128, 3, -4 }, try q_floor.dataConst());
    var neg = try I8.fromSlice(&ctx, .{4}, &.{ -7, -7, 7, 7 });
    defer neg.deinit();
    var two = try I8.fromSlice(&ctx, .{4}, &.{ 2, -2, 2, -2 });
    defer two.deinit();
    var nt = try neg.divTrunc(&ctx, &two);
    defer nt.deinit();
    try std.testing.expectEqualSlices(i8, &.{ -3, 3, 3, -3 }, try nt.dataConst());
    var nf = try neg.divFloor(&ctx, &two);
    defer nf.deinit();
    try std.testing.expectEqualSlices(i8, &.{ -4, 3, 3, -4 }, try nf.dataConst());

    var zero = try I8.fromSlice(&ctx, .{4}, &.{ 1, 0, 1, 1 });
    defer zero.deinit();
    try std.testing.expectError(error.DivisionByZero, a.divTrunc(&ctx, &zero));

    // maximum/minimum run natively (no NaN business on ints).
    var hi = try a.maximum(&ctx, &b);
    defer hi.deinit();
    try std.testing.expectEqualSlices(i8, &.{ 127, -1, 10, 2 }, try hi.dataConst());
    var lo = try a.minimum(&ctx, &b);
    defer lo.deinit();
    try std.testing.expectEqualSlices(i8, &.{ 1, -128, 3, -7 }, try lo.dataConst());

    // Reductions accumulate in i64 and RETURN i64 (torch's integer-sum
    // dtype) — a u16 row sum cannot overflow silently.
    const U16 = Tensor(.{ .dtype = .u16, .tags = .{ .row, .col } });
    var wide = try U16.fromSlice(&ctx, .{ 2, 3 }, &.{ 65535, 65535, 65535, 1, 2, 3 });
    defer wide.deinit();
    var row_sum = try wide.sum(&ctx, .col, .{});
    defer row_sum.deinit();
    comptime std.debug.assert(@TypeOf(row_sum).dtype == .i64);
    try std.testing.expectEqualSlices(i64, &.{ 196605, 6 }, try row_sum.dataConst());
    var total = try wide.sumAll(&ctx);
    defer total.deinit();
    try std.testing.expectEqual(@as(i64, 196611), try total.item());
}

test "integer and bool casts: wrap, saturate, count" {
    @setEvalBranchQuota(1_000_000);
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // int -> int wraps (two's complement), int -> float is exact here.
    const I32 = Tensor(.{ .dtype = .i32, .tags = .{.d} });
    var x = try I32.fromSlice(&ctx, .{4}, &.{ 300, -1, 65536, -129 });
    defer x.deinit();
    var narrow8 = try x.to(&ctx, .i8);
    defer narrow8.deinit();
    try std.testing.expectEqualSlices(i8, &.{ 44, -1, 0, 127 }, try narrow8.dataConst());
    var as_f32 = try x.to(&ctx, .f32);
    defer as_f32.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 300, -1, 65536, -129 }, try as_f32.dataConst());

    // float -> int truncates toward zero and SATURATES; NaN -> 0.
    var f = try Tensor(.{.d}).fromSlice(&ctx, .{5}, &.{ 300.9, -300.9, 1.9, -1.9, std.math.nan(f32) });
    defer f.deinit();
    var sat = try f.to(&ctx, .i8);
    defer sat.deinit();
    try std.testing.expectEqualSlices(i8, &.{ 127, -128, 1, -1, 0 }, try sat.dataConst());

    // anything -> bool is != 0 (NaN -> true); bool -> number is 0/1;
    // bool sum counts.
    var mask = try f.to(&ctx, .bool);
    defer mask.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, true }, try mask.dataConst());
    var zeros_too = try Tensor(.{.d}).fromSlice(&ctx, .{4}, &.{ 0, 2, 0, -3 });
    defer zeros_too.deinit();
    var mask2 = try zeros_too.to(&ctx, .bool);
    defer mask2.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, try mask2.dataConst());
    var count = try mask2.sumAll(&ctx);
    defer count.deinit();
    comptime std.debug.assert(@TypeOf(count).dtype == .i64);
    try std.testing.expectEqual(@as(i64, 2), try count.item());
    var back = try mask2.to(&ctx, .f32);
    defer back.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0, 1, 0, 1 }, try back.dataConst());
}

test "bool masks route where and maskedFill gradients" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const V = Tensor(.{.d});
    var a = try V.variableFromSlice(&ctx, .{4}, &.{ 1, 2, 3, 4 });
    defer a.deinit();
    var b = try V.variableFromSlice(&ctx, .{4}, &.{ 10, 20, 30, 40 });
    defer b.deinit();
    // cond = {false, true, false, true}: gradient splits by the bool mask.
    var cond2 = try V.fromSlice(&ctx, .{4}, &.{ 0, 1, 0, 1 });
    defer cond2.deinit();
    var mask = try cond2.compare(&ctx, .ge, 1);
    defer mask.deinit();

    var picked = try a.where(&ctx, &mask, &b);
    defer picked.deinit();
    try expectCloseSlices(&.{ 10, 2, 30, 4 }, try picked.dataConst(), 0);
    var loss = try picked.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    var ga = (try a.grad(&ctx)).?;
    defer ga.deinit();
    try expectCloseSlices(&.{ 0, 1, 0, 1 }, try ga.dataConst(), 0);
    var gb = (try b.grad(&ctx)).?;
    defer gb.deinit();
    try expectCloseSlices(&.{ 1, 0, 1, 0 }, try gb.dataConst(), 0);

    // maskedFill with the bool mask: grad zeroed exactly where filled.
    a.zeroGrad();
    var filled = try a.maskedFill(&ctx, &mask, 9.0);
    defer filled.deinit();
    try expectCloseSlices(&.{ 1, 9, 3, 9 }, try filled.dataConst(), 0);
    var loss2 = try filled.sumAll(&ctx);
    defer loss2.deinit();
    try loss2.backward(&ctx);
    var ga2 = (try a.grad(&ctx)).?;
    defer ga2.deinit();
    try expectCloseSlices(&.{ 1, 0, 1, 0 }, try ga2.dataConst(), 0);
}
