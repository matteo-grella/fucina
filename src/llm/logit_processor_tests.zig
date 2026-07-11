//! Behavioral tests for the logit-processor seam (`logit_processor.zig` +
//! `sampler.Sampler.processor`): mask application on the greedy and sampled
//! paths, exactly-one-commit-per-selection accounting, the all-masked loud
//! failure, and history visibility.

const std = @import("std");
const fucina = @import("fucina");
const sampler_mod = @import("sampler.zig");
const logit_processor = @import("logit_processor.zig");

const ExecContext = fucina.ExecContext;
const Logits = fucina.Tensor(.{ .seq, .vocab });
const Sampler = sampler_mod.Sampler;
const LogitProcessor = logit_processor.LogitProcessor;

/// Test double: masks every token id NOT in `allowed` to -inf, records every
/// commit and reset, and snapshots the history length it saw last.
const MaskProcessor = struct {
    allowed: []const usize,
    committed: std.ArrayList(usize) = .empty,
    allocator: std.mem.Allocator,
    resets: usize = 0,
    last_history_len: ?usize = null,

    fn deinit(self: *MaskProcessor) void {
        self.committed.deinit(self.allocator);
    }

    fn processor(self: *MaskProcessor) LogitProcessor {
        return .{ .ptr = self, .vtable = &vtable };
    }

    const vtable = LogitProcessor.VTable{
        .process = process,
        .commit = commit,
        .reset = reset,
    };

    fn process(ptr: *anyopaque, logits: []f32, history: []const usize) anyerror!void {
        const self: *MaskProcessor = @ptrCast(@alignCast(ptr));
        self.last_history_len = history.len;
        outer: for (logits, 0..) |*l, tok| {
            for (self.allowed) |a| if (a == tok) continue :outer;
            l.* = -std.math.inf(f32);
        }
    }

    fn commit(ptr: *anyopaque, token: usize) anyerror!void {
        const self: *MaskProcessor = @ptrCast(@alignCast(ptr));
        try self.committed.append(self.allocator, token);
    }

    fn reset(ptr: *anyopaque) anyerror!void {
        const self: *MaskProcessor = @ptrCast(@alignCast(ptr));
        self.resets += 1;
    }
};

test "greedy path: mask overrides the argmax, selection is committed" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var proc = MaskProcessor{ .allowed = &.{ 1, 4 }, .allocator = allocator };
    defer proc.deinit();

    // Unmasked argmax would be token 2; the mask forces the best allowed one.
    var logits = try Logits.fromSlice(&ctx, .{ 1, 5 }, &.{ 0.1, 0.2, 0.9, 0.3, 0.15 });
    defer logits.deinit();

    var s = Sampler.init(.{}); // greedy
    s.processor = proc.processor();
    try std.testing.expectEqual(@as(usize, 1), try s.next(&ctx, &logits, &.{ 7, 8 }));
    try std.testing.expectEqualSlices(usize, &.{1}, proc.committed.items);
    try std.testing.expectEqual(@as(?usize, 2), proc.last_history_len);
}

test "sampled path: only unmasked tokens are ever drawn, one commit per draw" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var proc = MaskProcessor{ .allowed = &.{ 0, 3 }, .allocator = allocator };
    defer proc.deinit();

    var s = Sampler.init(.{ .temperature = 1.5, .top_k = 5, .seed = 123 });
    s.processor = proc.processor();

    for (0..32) |_| {
        // Fresh row per draw: process() mutates the logits in place.
        var logits = try Logits.fromSlice(&ctx, .{ 1, 5 }, &.{ 0.4, 2.0, 1.5, 0.6, 1.0 });
        defer logits.deinit();
        const tok = try s.next(&ctx, &logits, &.{});
        try std.testing.expect(tok == 0 or tok == 3);
    }
    try std.testing.expectEqual(@as(usize, 32), proc.committed.items.len);
}

test "an all-masking processor fails loudly on both paths" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var proc = MaskProcessor{ .allowed = &.{}, .allocator = allocator };
    defer proc.deinit();

    var greedy = Sampler.init(.{});
    greedy.processor = proc.processor();
    var l1 = try Logits.fromSlice(&ctx, .{ 1, 4 }, &.{ 1, 2, 3, 4 });
    defer l1.deinit();
    try std.testing.expectError(error.AllTokensMasked, greedy.next(&ctx, &l1, &.{}));

    var sampled = Sampler.init(.{ .temperature = 0.8, .seed = 9 });
    sampled.processor = proc.processor();
    var l2 = try Logits.fromSlice(&ctx, .{ 1, 4 }, &.{ 1, 2, 3, 4 });
    defer l2.deinit();
    try std.testing.expectError(error.AllTokensMasked, sampled.next(&ctx, &l2, &.{}));
    try std.testing.expectEqual(@as(usize, 0), proc.committed.items.len);
}

test "without a processor the sampler pipeline is unchanged (no commit hook)" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var logits = try Logits.fromSlice(&ctx, .{ 1, 5 }, &.{ 0.1, 0.2, 0.9, 0.3, 0.0 });
    defer logits.deinit();
    var s = Sampler.init(.{});
    try std.testing.expectEqual(@as(usize, 2), try s.next(&ctx, &logits, &.{}));
}

test "penalties compose with the mask: -inf stays -inf through the penalty pass" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var proc = MaskProcessor{ .allowed = &.{ 0, 2 }, .allocator = allocator };
    defer proc.deinit();

    // Token 2 (the allowed argmax) is in the penalty window; the repeat
    // penalty demotes it below token 0 but must not resurrect masked ids.
    var logits = try Logits.fromSlice(&ctx, .{ 1, 4 }, &.{ 1.0, 5.0, 1.2, 5.0 });
    defer logits.deinit();
    var s = Sampler.init(.{ .repeat_penalty = 2.0, .repeat_last_n = 8 });
    s.processor = proc.processor();
    const tok = try s.next(&ctx, &logits, &.{2});
    try std.testing.expectEqual(@as(usize, 0), tok);
}
