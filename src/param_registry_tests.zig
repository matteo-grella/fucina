//! Behavioral tests for the parameter registry (`param_registry.zig`): named
//! registration + duplicate rejection, save/load round-trips independent of
//! registration order, reflective `collect` walks (nested structs, arrays,
//! pointers/slices/optionals, prefixes, frozen constants, f16/bf16 dtypes),
//! and optimizer composition through the public autograd facade.
const std = @import("std");
const exec_mod = @import("exec.zig");
const tensor_mod = @import("tensor.zig");
const ag = @import("ag.zig");
const optim = @import("optim.zig");
const state_dict = @import("state_dict.zig");
const param_registry = @import("param_registry.zig");

const ExecContext = exec_mod.ExecContext;
const DType = tensor_mod.DType;
const Tensor = ag.Tensor;
const ParamRegistry = param_registry.ParamRegistry;

test "ParamRegistry saves and loads named parameters independent of registration order" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var w = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer w.deinit();
    var b = try Tensor(.{.out}).variableFromSlice(&ctx, .{2}, &.{ -1, 0.5 });
    defer b.deinit();

    var registry = ParamRegistry.init(allocator);
    defer registry.deinit();
    try registry.addParam("linear.weight", &w);
    try registry.addParam("linear.bias", &b);
    try std.testing.expectEqual(@as(usize, 2), registry.parameterCount());

    var buf: [4096]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try registry.saveStateDict(&writer);
    const written = writer.buffered();

    var w2 = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 2, 2 }, &.{ 0, 0, 0, 0 });
    defer w2.deinit();
    var b2 = try Tensor(.{.out}).variableFromSlice(&ctx, .{2}, &.{ 0, 0 });
    defer b2.deinit();

    var registry2 = ParamRegistry.init(allocator);
    defer registry2.deinit();
    try registry2.addParam("linear.bias", &b2);
    try registry2.addParam("linear.weight", &w2);

    var reader = std.Io.Reader.fixed(written);
    try registry2.loadStateDict(&reader, .{});
    try std.testing.expectEqualSlices(f32, try w.dataConst(), try w2.dataConst());
    try std.testing.expectEqualSlices(f32, try b.dataConst(), try b2.dataConst());
}

test "ParamRegistry zeroGrad and addParamsTo compose with optimizers" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var w = try Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ 1, -2, 3 });
    defer w.deinit();

    var registry = ParamRegistry.init(allocator);
    defer registry.deinit();
    try registry.addParam("w", &w);

    var opt = optim.SGD.init(allocator, .{ .lr = 0.1 });
    defer opt.deinit();
    try registry.addParamsTo(&opt);

    {
        var y = try w.mul(&ctx, &w);
        defer y.deinit();
        var loss = try y.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);
    }
    var grad = (try w.grad(&ctx)).?;
    defer grad.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 2, -4, 6 }, try grad.dataConst());

    registry.zeroGrad();
    try std.testing.expect((try w.grad(&ctx)) == null);

    {
        var y = try w.mul(&ctx, &w);
        defer y.deinit();
        var loss = try y.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);
    }
    try opt.step(&ctx);
    registry.zeroGrad();
    try std.testing.expectEqualSlices(f32, &.{ 0.8, -1.6, 2.4 }, try w.dataConst());
}

test "ParamRegistry rejects duplicate parameter names" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try Tensor(.{.d}).variableFromSlice(&ctx, .{1}, &.{1});
    defer a.deinit();
    var b = try Tensor(.{.d}).variableFromSlice(&ctx, .{1}, &.{2});
    defer b.deinit();

    var registry = ParamRegistry.init(allocator);
    defer registry.deinit();
    try registry.addParam("x", &a);
    try std.testing.expectError(state_dict.Error.CheckpointDuplicateName, registry.addParam("x", &b));
}

test "collect registers flat struct fields by field name and round-trips" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const Model = struct {
        w1: Tensor(.{ .o, .i }),
        b1: Tensor(.{.o}),
    };
    var model = Model{
        .w1 = try Tensor(.{ .o, .i }).variableFromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 }),
        .b1 = try Tensor(.{.o}).variableFromSlice(&ctx, .{2}, &.{ 5, 6 }),
    };
    defer {
        model.w1.deinit();
        model.b1.deinit();
    }

    var registry = ParamRegistry.init(allocator);
    defer registry.deinit();
    try registry.collect(&model);
    try std.testing.expectEqual(@as(usize, 2), registry.parameterCount());
    try std.testing.expectEqualStrings("w1", registry.params.items[0].name);
    try std.testing.expectEqualStrings("b1", registry.params.items[1].name);

    var buf: [4096]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try registry.saveStateDict(&writer);

    var dst = Model{
        .w1 = try Tensor(.{ .o, .i }).variableFromSlice(&ctx, .{ 2, 2 }, &.{ 0, 0, 0, 0 }),
        .b1 = try Tensor(.{.o}).variableFromSlice(&ctx, .{2}, &.{ 0, 0 }),
    };
    defer {
        dst.w1.deinit();
        dst.b1.deinit();
    }
    var dst_registry = ParamRegistry.init(allocator);
    defer dst_registry.deinit();
    try dst_registry.collect(&dst);
    var reader = std.Io.Reader.fixed(writer.buffered());
    try dst_registry.loadStateDict(&reader, .{});
    try std.testing.expectEqualSlices(f32, try model.w1.dataConst(), try dst.w1.dataConst());
    try std.testing.expectEqualSlices(f32, try model.b1.dataConst(), try dst.b1.dataConst());
}

test "collect nests structs with dotted names and round-trips by name" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const Lin = struct { w: Tensor(.{ .o, .i }), b: Tensor(.{.o}) };
    const Net = struct { enc: Lin, head: Lin };
    const make = struct {
        fn lin(c: *ExecContext, wv: f32, bv: f32) !Lin {
            return .{
                .w = try Tensor(.{ .o, .i }).variableFromSlice(c, .{ 1, 1 }, &.{wv}),
                .b = try Tensor(.{.o}).variableFromSlice(c, .{1}, &.{bv}),
            };
        }
    };
    var net = Net{ .enc = try make.lin(&ctx, 1, 2), .head = try make.lin(&ctx, 3, 4) };
    defer {
        net.enc.w.deinit();
        net.enc.b.deinit();
        net.head.w.deinit();
        net.head.b.deinit();
    }

    var registry = ParamRegistry.init(allocator);
    defer registry.deinit();
    try registry.collect(&net);
    try std.testing.expectEqual(@as(usize, 4), registry.parameterCount());
    try std.testing.expectEqualStrings("enc.w", registry.params.items[0].name);
    try std.testing.expectEqualStrings("enc.b", registry.params.items[1].name);
    try std.testing.expectEqualStrings("head.w", registry.params.items[2].name);
    try std.testing.expectEqualStrings("head.b", registry.params.items[3].name);

    var buf: [4096]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try registry.saveStateDict(&writer);

    var dst = Net{ .enc = try make.lin(&ctx, 0, 0), .head = try make.lin(&ctx, 0, 0) };
    defer {
        dst.enc.w.deinit();
        dst.enc.b.deinit();
        dst.head.w.deinit();
        dst.head.b.deinit();
    }
    var dst_registry = ParamRegistry.init(allocator);
    defer dst_registry.deinit();
    try dst_registry.collect(&dst);
    var reader = std.Io.Reader.fixed(writer.buffered());
    try dst_registry.loadStateDict(&reader, .{});
    try std.testing.expectEqualSlices(f32, try net.head.w.dataConst(), try dst.head.w.dataConst());
    try std.testing.expectEqualSlices(f32, try net.enc.b.dataConst(), try dst.enc.b.dataConst());
}

test "collect indexes arrays of structs with dotted indices" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const Block = struct { w: Tensor(.{.d}) };
    const Net = struct { layers: [2]Block };
    var net = Net{ .layers = .{
        .{ .w = try Tensor(.{.d}).variableFromSlice(&ctx, .{2}, &.{ 1, 2 }) },
        .{ .w = try Tensor(.{.d}).variableFromSlice(&ctx, .{2}, &.{ 3, 4 }) },
    } };
    defer for (&net.layers) |*b| b.w.deinit();

    var registry = ParamRegistry.init(allocator);
    defer registry.deinit();
    try registry.collect(&net);
    try std.testing.expectEqual(@as(usize, 2), registry.parameterCount());
    try std.testing.expectEqualStrings("layers.0.w", registry.params.items[0].name);
    try std.testing.expectEqualStrings("layers.1.w", registry.params.items[1].name);
}

test "collect follows mutable pointer, slice, and optional fields" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const Block = struct { w: Tensor(.{.d}) };
    const Net = struct {
        direct: *Tensor(.{.d}),
        layers: []Block,
        maybe: ?Tensor(.{.d}),
    };

    var direct = try Tensor(.{.d}).variableFromSlice(&ctx, .{2}, &.{ 1, 2 });
    defer direct.deinit();
    var blocks = [_]Block{
        .{ .w = try Tensor(.{.d}).variableFromSlice(&ctx, .{1}, &.{3}) },
        .{ .w = try Tensor(.{.d}).variableFromSlice(&ctx, .{1}, &.{4}) },
    };
    defer for (&blocks) |*block| block.w.deinit();
    var net = Net{
        .direct = &direct,
        .layers = blocks[0..],
        .maybe = try Tensor(.{.d}).variableFromSlice(&ctx, .{1}, &.{5}),
    };
    defer if (net.maybe) |*maybe| maybe.deinit();

    var registry = ParamRegistry.init(allocator);
    defer registry.deinit();
    try registry.collect(&net);
    try std.testing.expectEqual(@as(usize, 4), registry.parameterCount());
    try std.testing.expectEqualStrings("direct", registry.params.items[0].name);
    try std.testing.expectEqualStrings("layers.0.w", registry.params.items[1].name);
    try std.testing.expectEqualStrings("layers.1.w", registry.params.items[2].name);
    try std.testing.expectEqualStrings("maybe", registry.params.items[3].name);
}

test "collectPrefixed applies a leading prefix" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const Model = struct { weight: Tensor(.{.d}) };
    var model = Model{ .weight = try Tensor(.{.d}).variableFromSlice(&ctx, .{1}, &.{1}) };
    defer model.weight.deinit();

    var registry = ParamRegistry.init(allocator);
    defer registry.deinit();
    try registry.collectPrefixed("model", &model);
    try std.testing.expectEqual(@as(usize, 1), registry.parameterCount());
    try std.testing.expectEqualStrings("model.weight", registry.params.items[0].name);
}

test "collect registers constants frozen, skips non-tensor fields, trains only variables" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const Model = struct {
        w: Tensor(.{.d}), // variable (trainable)
        steps: u32, // non-tensor: ignored
        frozen: Tensor(.{.d}), // constant: checkpoint-only
    };
    var model = Model{
        .w = try Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ 1, -2, 3 }),
        .steps = 7,
        .frozen = try Tensor(.{.d}).fromSlice(&ctx, .{3}, &.{ 9, 9, 9 }),
    };
    defer {
        model.w.deinit();
        model.frozen.deinit();
    }

    var registry = ParamRegistry.init(allocator);
    defer registry.deinit();
    try registry.collect(&model);
    try std.testing.expectEqual(@as(usize, 2), registry.parameterCount()); // w + frozen, not steps
    try std.testing.expectEqualStrings("w", registry.params.items[0].name);
    try std.testing.expectEqualStrings("frozen", registry.params.items[1].name);

    var opt = optim.SGD.init(allocator, .{ .lr = 0.1 });
    defer opt.deinit();
    try registry.addParamsTo(&opt); // only the variable is registered

    {
        var y = try model.w.mul(&ctx, &model.w);
        defer y.deinit();
        var loss = try y.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);
    }
    try opt.step(&ctx);
    registry.zeroGrad();

    try std.testing.expectEqualSlices(f32, &.{ 0.8, -1.6, 2.4 }, try model.w.dataConst());
    try std.testing.expectEqualSlices(f32, &.{ 9, 9, 9 }, try model.frozen.dataConst()); // untouched
}

test "collect registers f16/bf16 fields at native dtype and round-trips" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // f32 source cast to f16/bf16 constants (autograd is f32-only, so reduced
    // precision tensors are frozen — the resident-low-precision-weights case).
    var a = try Tensor(.{.d}).fromSlice(&ctx, .{4}, &.{ 1, 2, 3, 4 });
    defer a.deinit();
    var b = try a.to(&ctx, .f16);
    defer b.deinit();
    var c = try a.to(&ctx, .bf16);
    defer c.deinit();

    const Model = struct { a: @TypeOf(a), b: @TypeOf(b), c: @TypeOf(c) };
    var model = Model{ .a = a, .b = b, .c = c }; // shallow views; a/b/c own the storage

    var registry = ParamRegistry.init(allocator);
    defer registry.deinit();
    try registry.collect(&model);
    try std.testing.expectEqual(@as(usize, 3), registry.parameterCount());
    try std.testing.expectEqual(DType.f32, registry.params.items[0].dtype);
    try std.testing.expectEqual(DType.f16, registry.params.items[1].dtype);
    try std.testing.expectEqual(DType.bf16, registry.params.items[2].dtype);

    var buf: [4096]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try registry.saveStateDict(&writer);

    // Fresh zeroed model of the same dtypes; a successful load proves the wire
    // stored native dtypes (a dtype mismatch would be rejected on load).
    var za = try Tensor(.{.d}).fromSlice(&ctx, .{4}, &.{ 0, 0, 0, 0 });
    defer za.deinit();
    var zb = try za.to(&ctx, .f16);
    defer zb.deinit();
    var zc = try za.to(&ctx, .bf16);
    defer zc.deinit();
    var dst = Model{ .a = za, .b = zb, .c = zc };

    var dst_registry = ParamRegistry.init(allocator);
    defer dst_registry.deinit();
    try dst_registry.collect(&dst);
    var reader = std.Io.Reader.fixed(writer.buffered());
    try dst_registry.loadStateDict(&reader, .{});

    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(a.value.dataConst()), std.mem.sliceAsBytes(za.value.dataConst()));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(b.value.dataConst()), std.mem.sliceAsBytes(zb.value.dataConst()));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(c.value.dataConst()), std.mem.sliceAsBytes(zc.value.dataConst()));
}

test "collect descends tagged unions into the active arm under the same prefix" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var w32 = try Tensor(.{ .out, .in }).fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer w32.deinit();
    var w16 = try w32.to(&ctx, .f16);
    defer w16.deinit();

    // Mirrors llm.weights.LinearWeight: one dtype arm live per weight, plus a
    // non-tensor arm that must be skipped when active.
    const Linear = union(enum) {
        f32: @TypeOf(w32),
        f16: @TypeOf(w16),
        packed_only: u32,
    };
    const Model = struct {
        proj: [3]Linear,
    };
    var model = Model{ .proj = .{
        .{ .f32 = w32 },
        .{ .f16 = w16 },
        .{ .packed_only = 7 },
    } };

    var registry = ParamRegistry.init(allocator);
    defer registry.deinit();
    try registry.collect(&model);

    // The union arm adds no path segment: "proj.<i>" regardless of dtype arm;
    // the non-tensor arm contributes nothing.
    try std.testing.expectEqual(@as(usize, 2), registry.parameterCount());
    try std.testing.expectEqualStrings("proj.0", registry.view(0).name);
    try std.testing.expectEqual(DType.f32, registry.view(0).dtype);
    try std.testing.expectEqualStrings("proj.1", registry.view(1).name);
    try std.testing.expectEqual(DType.f16, registry.view(1).dtype);
    try std.testing.expect(!registry.view(0).trainable); // constants stay frozen
}

test "view exposes name, dtype, mutable bytes, and trainability" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var w = try Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ 1, 2, 3 });
    defer w.deinit();
    var frozen = try Tensor(.{.d}).fromSlice(&ctx, .{2}, &.{ 5, 6 });
    defer frozen.deinit();

    var registry = ParamRegistry.init(allocator);
    defer registry.deinit();
    try registry.addParam("w", &w);
    try registry.addParam("frozen", &frozen);

    const wv = registry.view(0);
    try std.testing.expectEqualStrings("w", wv.name);
    try std.testing.expectEqual(DType.f32, wv.dtype);
    try std.testing.expect(wv.trainable);
    const fv = registry.view(1);
    try std.testing.expect(!fv.trainable);

    // The bytes view aliases the live tensor storage (mutations flow through).
    const wf: []f32 = @alignCast(std.mem.bytesAsSlice(f32, wv.bytes));
    wf[1] = 42;
    try std.testing.expectEqual(@as(f32, 42), (try w.dataConst())[1]);
}
