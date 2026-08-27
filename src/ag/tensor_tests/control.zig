//! Autograd graph and scope mechanics: backward consumption, detach, noGrad
//! scopes, exec-scope buffer ownership, and exactly-once release under
//! induced allocation failure.

const std = @import("std");
const backend_mod = @import("../../backend.zig");
const dtype_mod = @import("../../dtype.zig");
const exec_mod = @import("../../exec.zig");
const control = @import("../control.zig");
const core = @import("../core.zig");
const ag_tensor = @import("../tensor.zig");

const DType = dtype_mod.DType;
const ExecContext = exec_mod.ExecContext;
const GradState = core.GradState;
const Tensor = ag_tensor.Tensor;
const RawTensor = @import("../../tensor.zig").Tensor;

test "tagged public grad state does not duplicate the raw value" {
    if (@hasField(GradState, "value")) @compileError("GradState must not own Tensor.value");
}

test "tagged public tensor detach severs the gradient graph" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ 1, 2, 3 });
    defer x.deinit();

    var y = try x.mul(&ctx, &x);
    defer y.deinit();
    try std.testing.expect(y.requiresGrad());

    var detached = try y.detach(&ctx);
    defer detached.deinit();
    try std.testing.expect(!detached.requiresGrad());
    try std.testing.expectEqualSlices(f32, &.{ 1, 4, 9 }, try detached.dataConst());

    var loss = try detached.sumAll(&ctx);
    defer loss.deinit();
    try std.testing.expect(!loss.requiresGrad());
    try std.testing.expectError(error.NoGradientGraph, loss.backward(&ctx));
    try std.testing.expect((try x.grad(&ctx)) == null);
}

test "tagged public backwardWithGrad seeds non-scalar outputs" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{.d}).variableFromSlice(&ctx, .{2}, &.{ 3, 4 });
    defer x.deinit();
    var c = try Tensor(.{.d}).fromSlice(&ctx, .{2}, &.{ 10, 20 });
    defer c.deinit();
    var y = try x.mul(&ctx, &c);
    defer y.deinit();

    // A non-scalar output has no implicit seed; a mis-shaped output
    // gradient is rejected before any state changes.
    try std.testing.expectError(error.MissingOutputGradient, y.backward(&ctx));
    var bad_grad = try Tensor(.{.d}).fromSlice(&ctx, .{3}, &.{ 1, 1, 1 });
    defer bad_grad.deinit();
    try std.testing.expectError(error.ShapeMismatch, y.backwardWithGrad(&ctx, &bad_grad));

    // The output gradient is read as a value: dloss/dx = grad_output * c.
    var grad_output = try Tensor(.{.d}).fromSlice(&ctx, .{2}, &.{ 1, 10 });
    defer grad_output.deinit();
    try y.backwardWithGrad(&ctx, &grad_output);
    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 10, 200 }, try gx.dataConst());
    try std.testing.expectEqualSlices(f32, &.{ 1, 10 }, try grad_output.dataConst());

    // The completed pass consumed the graph; constants have no graph at all.
    try std.testing.expectError(error.BackwardAlreadyRun, y.backwardWithGrad(&ctx, &grad_output));
    try std.testing.expectError(error.NoGradientGraph, c.backwardWithGrad(&ctx, &grad_output));
}

test "tagged public tensor rejects a second backward over a consumed graph" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{.d}).variableFromSlice(&ctx, .{2}, &.{ 3, 5 });
    defer x.deinit();

    var sq = try x.mul(&ctx, &x);
    defer sq.deinit();
    var loss = try sq.sumAll(&ctx);
    defer loss.deinit();

    // Scalar outputs may take an explicit output gradient too:
    // dloss/dx = 2 * 2x.
    var grad_output = try Tensor(.{}).scalar(&ctx, 2);
    defer grad_output.deinit();
    try loss.backwardWithGrad(&ctx, &grad_output);
    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 12, 20 }, try gx.dataConst());

    // Interior states keep their gradients, so a second pass over the SAME
    // graph would compound them; it fails loudly instead — zeroGrad resets
    // gradients, not the consumed graph.
    try std.testing.expectError(error.BackwardAlreadyRun, loss.backward(&ctx));
    x.zeroGrad();
    loss.zeroGrad();
    try std.testing.expectError(error.BackwardAlreadyRun, loss.backward(&ctx));

    // The micro-batch idiom is untouched: a FRESH graph over the same leaf
    // runs and accumulates into the leaf as before.
    var sq2 = try x.mul(&ctx, &x);
    defer sq2.deinit();
    var loss2 = try sq2.sumAll(&ctx);
    defer loss2.deinit();
    try loss2.backward(&ctx);
    var gx2 = (try x.grad(&ctx)).?;
    defer gx2.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 6, 10 }, try gx2.dataConst());
}

test "tagged public noGrad scope suppresses graph recording" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ 1, 2, 3 });
    defer x.deinit();

    {
        var guard = control.noGrad();
        defer guard.close();
        var y = try x.mul(&ctx, &x);
        defer y.deinit();
        try std.testing.expect(!y.requiresGrad());
        try std.testing.expectEqualSlices(f32, &.{ 1, 4, 9 }, try y.dataConst());
    }

    var y = try x.mul(&ctx, &x);
    defer y.deinit();
    try std.testing.expect(y.requiresGrad());
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 2, 4, 6 }, try gx.dataConst());
}

test "tagged autograd no-grad expressions do not retain graph intermediates" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();
    var bias = try Tensor(.{.d}).fromSlice(&ctx, .{3}, &.{ 10, 20, 30 });
    defer bias.deinit();

    var y = try x.add(&ctx, &bias);
    defer y.deinit();

    try std.testing.expect(!y.requiresGrad());
    try std.testing.expect(y.grad_state == null);
    try std.testing.expectEqualSlices(f32, &.{ 11, 22, 33, 14, 25, 36 }, y.asRawTensor().dataConst());
}

test "tagged public tensor stores no-grad values inline" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const raw = try ctx.fromSlice(.f32, .{3}, &.{ 1, 2, 3 });
    const buffer = raw.buffer;
    var x = try Tensor(1).fromTensor(&ctx, raw);
    defer x.deinit();

    try std.testing.expect(x.grad_state == null);
    try std.testing.expect(x.asRawTensor().buffer == buffer);

    var y = try x.add(&ctx, x);
    defer y.deinit();
    try std.testing.expect(y.grad_state == null);
    try std.testing.expectEqualSlices(f32, &.{ 2, 4, 6 }, y.asRawTensor().dataConst());
}

test "tagged public tensor applies no-grad views without graph state" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();

    var p = try x.permuteTo(&ctx, .{ .d, .batch });
    defer p.deinit();
    var p_data = [_]f32{0} ** 6;
    try p.asRawTensor().copyTo(&p_data);
    try std.testing.expect(p.grad_state == null);
    try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, p.asRawTensor().shape.slice());
    try std.testing.expectEqualSlices(f32, &.{ 1, 4, 2, 5, 3, 6 }, &p_data);

    var expanded = try x.insertAxis(&ctx, .head, 1);
    defer expanded.deinit();
    try std.testing.expect(expanded.grad_state == null);
    try std.testing.expectEqualSlices(usize, &.{ 2, 1, 3 }, expanded.asRawTensor().shape.slice());

    var squeezed = try expanded.squeeze(&ctx, .head);
    defer squeezed.deinit();
    var squeezed_data = [_]f32{0} ** 6;
    try squeezed.asRawTensor().copyTo(&squeezed_data);
    try std.testing.expect(squeezed.grad_state == null);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6 }, &squeezed_data);

    var split = try x.split(&ctx, .d, .{ .row, .col }, .{ 1, 3 });
    defer split.deinit();
    try std.testing.expect(split.grad_state == null);
    try std.testing.expectEqualSlices(usize, &.{ 2, 1, 3 }, split.asRawTensor().shape.slice());

    var merged = try split.merge(&ctx, .d2, .{ .row, .col });
    defer merged.deinit();
    var merged_data = [_]f32{0} ** 6;
    try merged.asRawTensor().copyTo(&merged_data);
    try std.testing.expect(merged.grad_state == null);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6 }, &merged_data);
}

test "tagged public tensor keeps constant operands alive for gradients" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(1).variableFromSlice(&ctx, .{3}, &.{ 1, 2, 3 });
    defer x.deinit();
    var scale = try Tensor(1).fromSlice(&ctx, .{3}, &.{ 10, 20, 30 });
    defer scale.deinit();

    var y = try x.mul(&ctx, &scale);
    defer y.deinit();
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();

    try loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 10, 20, 30 }, gx.asRawTensor().dataConst());
    try std.testing.expect((try scale.grad(&ctx)) == null);
}

fn allocationFailureProbe(allocator: std.mem.Allocator) !void {
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ 1, 2, 3 });
    defer x.deinit();
    var w = try Tensor(.{ .out, .d }).fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer w.deinit();

    var y = try w.dot(&ctx, &x, .d);
    defer y.deinit();
    var z = try y.mul(&ctx, &y);
    defer z.deinit();
    var loss = try z.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
}

test "tagged autograd ops release exactly once under induced allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationFailureProbe, .{});
}

fn allocationFailureProbeMulUnscoped(allocator: std.mem.Allocator) !void {
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ 1, 2, 3 });
    defer x.deinit();

    var y = try x.mul(&ctx, &x);
    defer y.deinit();
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
}

test "tagged tracked mul (unscoped) releases exactly once under induced allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationFailureProbeMulUnscoped, .{});
}

fn allocationFailureProbeMulScoped(allocator: std.mem.Allocator) !void {
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ 1, 2, 3 });
    defer x.deinit();

    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const y = try x.mul(&ctx, &x);
        var loss = try y.sumAll(&ctx);
        try loss.backward(&ctx);
    }

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
}

test "tagged tracked mul (exec scope) releases exactly once under induced allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationFailureProbeMulScoped, .{});
}

/// Minimal counting wrapper (mirrors bench/alloc.zig CountingAllocator, which
/// tests cannot import across the bench/src module boundary).
const AllocCounter = struct {
    child: std.mem.Allocator,
    alloc_count: usize = 0,

    fn allocator(self: *AllocCounter) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .remap = remap,
                .free = free,
            },
        };
    }

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *AllocCounter = @ptrCast(@alignCast(ctx));
        const ptr = self.child.rawAlloc(len, alignment, ret_addr) orelse return null;
        self.alloc_count += 1;
        return ptr;
    }

    fn resize(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *AllocCounter = @ptrCast(@alignCast(ctx));
        return self.child.rawResize(buf, alignment, new_len, ret_addr);
    }

    fn remap(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        const self: *AllocCounter = @ptrCast(@alignCast(ctx));
        return self.child.rawRemap(buf, alignment, new_len, ret_addr);
    }

    fn free(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        const self: *AllocCounter = @ptrCast(@alignCast(ctx));
        self.child.rawFree(buf, alignment, ret_addr);
    }
};

test "tagged tracked pointwise op on a warmed context performs exactly one allocation" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var counter = AllocCounter{ .child = gpa.allocator() };

    var ctx: ExecContext = undefined;
    ctx.init(counter.allocator());
    defer ctx.deinit();

    var x = try Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ 1, 2, 3 });
    defer x.deinit();

    // Warm the runtime: the first op's value buffer (and any lazy runtime
    // state) is allocated here and returns to the buffer pool on deinit.
    var warm = try x.mul(&ctx, &x);
    warm.deinit();

    // The counted op reuses the pooled value buffer, so the only heap
    // allocation left is the single co-allocated GradState+record node.
    counter.alloc_count = 0;
    var y = try x.mul(&ctx, &x);
    defer y.deinit();
    try std.testing.expectEqual(@as(usize, 1), counter.alloc_count);
}

test "exec scope owns differentiable intermediates through backward" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ 1, 2, 3 });
    defer x.deinit();

    // Two steps with per-iteration scopes: no keep, no defer, no leaks.
    for (0..2) |_| {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const y = try x.mul(&ctx, &x);
        const z = try y.mul(&ctx, &x);
        var loss = try z.sumAll(&ctx);
        try loss.backward(&ctx);

        var gx = (try x.grad(&ctx)).?; // gradients stay caller-owned
        defer gx.deinit();
        try std.testing.expectEqualSlices(f32, &.{ 3, 12, 27 }, try gx.dataConst());
        x.zeroGrad();
    }
}

test "exec scope adopts no-grad op results (constants, argmax, topK)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var c = try Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ 2, 4 }, &.{ 1, 5, 2, 3, -1, 2, 4, 0 });
    defer c.deinit();

    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    const doubled = try c.add(&ctx, &c);
    // Index outputs are i64 typed constants and scope borrows like every
    // other op result: their deinit is a no-op, the scope releases at close.
    var arg = try doubled.argmax(&ctx, .d);
    defer arg.deinit();
    try std.testing.expect(arg.scope_owned);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2 }, try arg.dataConst());
    var top = try doubled.topK(&ctx, .d, 2, .k);
    defer top.indices.deinit();
    try std.testing.expect(top.values.scope_owned and top.indices.scope_owned);
    try std.testing.expectEqualSlices(f32, &.{ 10, 6, 8, 4 }, try top.values.dataConst());
    try std.testing.expectEqualSlices(i64, &.{ 1, 3, 2, 1 }, try top.indices.dataConst());
}

test "typed results are scope borrows: deinit is a no-op and the scope releases at close" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .batch, .d }).variableFromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();

    const scope = ctx.openExecScope();
    const base = ctx.rt.scopes.entries.items.len;
    var idx = try x.argmax(&ctx, .d); // i64: one entry, the buffer
    try std.testing.expect(idx.scope_owned);
    try std.testing.expectEqual(base + 1, ctx.rt.scopes.entries.items.len);
    var mask = try x.compare(&ctx, .gt, 2); // bool: one entry
    try std.testing.expect(mask.scope_owned);
    try std.testing.expectEqual(base + 2, ctx.rt.scopes.entries.items.len);
    var narrow = try x.to(&ctx, .bf16); // grad-carrying 16-bit: buffer + node
    try std.testing.expect(narrow.scope_owned);
    try std.testing.expectEqual(base + 4, ctx.rt.scopes.entries.items.len);
    try std.testing.expectEqualSlices(i64, &.{ 1, 1 }, try idx.dataConst());

    // The caller's deinit is a no-op on every borrow, whatever the dtype;
    // the scope's close is the single release (a double release would
    // trip the buffer pool and the DebugAllocator).
    idx.deinit();
    mask.deinit();
    narrow.deinit();
    try std.testing.expectEqual(base + 4, ctx.rt.scopes.entries.items.len);
    ctx.closeExecScope(scope);
    try std.testing.expectEqual(base, ctx.rt.scopes.entries.items.len);
}

test "a borrow's deinit after its scope closed is still a no-op" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{.d}).fromSlice(&ctx, .{2}, &.{ 1, 2 });
    defer x.deinit();

    var y: Tensor(.{.d}) = undefined;
    var idx: Tensor(.{ .dtype = .i64, .tags = .{} }) = undefined;
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        y = try x.add(&ctx, &x);
        idx = try y.argmax(&ctx, .d);
        try std.testing.expectEqual(@as(i64, 1), try idx.item());
    }
    // The handles are invalid borrows now (their storage went at close);
    // releasing them is still a no-op, so a defer placed outside the scope
    // cannot double free. Reading them would be use-after-free.
    y.deinit();
    idx.deinit();
}

test "nested exec scopes release only their suffix" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{.d}).variableFromSlice(&ctx, .{2}, &.{ 2, 3 });
    defer x.deinit();

    const outer = ctx.openExecScope();
    defer ctx.closeExecScope(outer);
    const a = try x.mul(&ctx, &x); // owned by the outer scope

    {
        const inner = ctx.openExecScope();
        defer ctx.closeExecScope(inner);
        const b = try a.mul(&ctx, &x); // owned by the inner scope
        try std.testing.expectEqualSlices(f32, &.{ 8, 27 }, try b.dataConst());
    }

    // `a` survives the inner close and is still usable.
    const c2 = try a.add(&ctx, &a);
    try std.testing.expectEqualSlices(f32, &.{ 8, 18 }, try c2.dataConst());
}

// This test documents WHY exec scopes are a training tool and not a
// replacement for deinit-ASAP in inference code: with deinit-ASAP a chain of
// same-shaped ops recycles ~2 pooled buffers (O(1) working set, warm
// addresses), while a held scope keeps every intermediate live until close
// (O(N) working set, cold addresses). See the note at the end of
// docs/MEMORY-MODEL.md §5.
test "exec scope holds buffers until close; deinit-ASAP recycles through the pool" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    const side: usize = 64;
    const chain_len: usize = 16;
    const data = try allocator.alloc(f32, side * side);
    defer allocator.free(data);
    @memset(data, 0.5);

    // Variant A: deinit-ASAP (the inference idiom) — bounded working set.
    var peak_asap: usize = 0;
    {
        var ctx: ExecContext = undefined;
        ctx.init(allocator);
        defer ctx.deinit();
        var c = try Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ side, side }, data);
        defer c.deinit();
        const base = ctx.rt.buffers.outstandingBuffers();
        var cur = try c.add(&ctx, &c);
        for (1..chain_len) |_| {
            const next = try cur.add(&ctx, &c);
            peak_asap = @max(peak_asap, ctx.rt.buffers.outstandingBuffers() - base);
            cur.deinit(); // previous intermediate returns to the pool immediately
            cur = next;
        }
        cur.deinit();
    }
    try std.testing.expect(peak_asap <= 2);

    // Variant B: one exec scope held to the end — every intermediate lives.
    var peak_scope: usize = 0;
    {
        var ctx: ExecContext = undefined;
        ctx.init(allocator);
        defer ctx.deinit();
        var c = try Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ side, side }, data);
        defer c.deinit();
        const base = ctx.rt.buffers.outstandingBuffers();
        const scope = ctx.openExecScope();
        var cur = try c.add(&ctx, &c);
        for (1..chain_len) |_| {
            cur = try cur.add(&ctx, &c);
            peak_scope = @max(peak_scope, ctx.rt.buffers.outstandingBuffers() - base);
        }
        ctx.closeExecScope(scope);
        try std.testing.expectEqual(base, ctx.rt.buffers.outstandingBuffers());
    }
    try std.testing.expectEqual(chain_len, peak_scope);
}

// The write-once property: engine-style forward code (defer-deinit everywhere,
// ctx.replace for the residual stream) is inference code when no scope is
// open, and training code when one is — deinit on scope-owned results is a
// safe no-op (arena semantics), so neither double-frees nor leaks.
fn engineStyleForward(ctx: *ExecContext, w: *const Tensor(.{ .out, .in }), x0: *const Tensor(.{ .batch, .in })) !Tensor(.{}) {
    var h = try x0.dot(ctx, w, .in);
    defer h.deinit();
    var x = try h.withTags(ctx, .{ .batch, .in });
    defer x.deinit();
    for (0..3) |_| {
        // residual-style carry: old released by replace (no-op when scope-owned)
        x = try ctx.replace(x, blockStep(ctx, w, &x));
    }
    var sq = try x.mul(ctx, &x);
    defer sq.deinit();
    return sq.sumAll(ctx);
}

fn blockStep(ctx: *ExecContext, w: *const Tensor(.{ .out, .in }), x: *const Tensor(.{ .batch, .in })) !Tensor(.{ .batch, .in }) {
    var z = try x.dot(ctx, w, .in);
    defer z.deinit();
    var a = try z.tanh(ctx);
    defer a.deinit();
    var renamed = try a.withTags(ctx, .{ .batch, .in });
    defer renamed.deinit();
    return renamed.add(ctx, x);
}

test "scope-owned deinit is a no-op: the same engine-style forward runs unscoped and scoped" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x0 = try Tensor(.{ .batch, .in }).fromSlice(&ctx, .{ 2, 2 }, &.{ 0.5, -0.25, 0.125, 0.75 });
    defer x0.deinit();

    // Inference mode: no scope, every defer is a real release (deinit-ASAP).
    var inference_loss: f32 = 0;
    {
        var w = try Tensor(.{ .out, .in }).fromSlice(&ctx, .{ 2, 2 }, &.{ 0.4, -0.3, 0.2, 0.6 });
        defer w.deinit();
        var loss = try engineStyleForward(&ctx, &w, &x0);
        defer loss.deinit();
        inference_loss = try loss.item();
    }

    // Training mode: SAME code under a scope — defers no-op on scope-owned
    // results, the graph survives to backward, grads flow.
    {
        var w = try Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 2, 2 }, &.{ 0.4, -0.3, 0.2, 0.6 });
        defer w.deinit();
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        var loss = try engineStyleForward(&ctx, &w, &x0);
        defer loss.deinit(); // no-op: scope-owned
        try std.testing.expectApproxEqAbs(inference_loss, try loss.item(), 1e-6);
        try loss.backward(&ctx);
        var gw = (try w.grad(&ctx)).?;
        defer gw.deinit();
        for (try gw.dataConst()) |g| try std.testing.expect(g != 0);
        w.zeroGrad();
    }
}
