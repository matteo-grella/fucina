//! End-to-end golden parity for the Kimi-K3 architecture model against the
//! tiny f32 reference checkpoint (`models/kimi-k3-0.40b`, untracked; the
//! goldens come from `tools/kimi3_goldens.py`). Skips when the local model
//! or goldens are absent. Layer-level attention outputs localize a failure
//! to a component before the full-logits check runs.
//!
//! Native backend only: these are real-model golden forwards — they pin
//! MODEL WIRING, not kernel math, so the scalar reference leg skips them
//! (its kernel coverage lives in the exec/backend suites).

const std = @import("std");
const test_support = @import("../../test_support.zig");

const model_mod = @import("model.zig");

const model_dir = "models/kimi-k3-0.40b";
const goldens_dir = model_dir ++ "/goldens";

// tools/kimi3_goldens.py TOKEN_IDS.
const token_ids = [_]u32{ 1000, 2534, 77, 4096, 163, 999, 5000, 42, 31337, 7, 250, 88 };

fn readGolden(allocator: std.mem.Allocator, name: []const u8) !?[]f32 {
    var path_buf: [256]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "{s}/{s}.bin", .{ goldens_dir, name });
    const bytes = std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, allocator, .limited(1 << 27)) catch return null;
    defer allocator.free(bytes);
    if (bytes.len % @sizeOf(f32) != 0) return error.GoldenShapeMismatch;
    const out = try allocator.alloc(f32, bytes.len / @sizeOf(f32));
    @memcpy(std.mem.sliceAsBytes(out), bytes);
    return out;
}

test "kimi3 model matches the reference checkpoint logits" {
    try test_support.requireNative();
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    const io = std.testing.io;

    const logits_ref = (try readGolden(allocator, "logits")) orelse return; // no local goldens: skip
    defer allocator.free(logits_ref);

    var ctx: model_mod.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = model_mod.Model.load(allocator, io, &ctx, model_dir) catch |err| switch (err) {
        error.FileNotFound => return,
        else => {
            std.debug.print("kimi3 load failed: {t}\n", .{err});
            return err;
        },
    };
    defer model.deinit();

    const ProbeState = struct {
        allocator: std.mem.Allocator,

        fn onStage(context: *anyopaque, name: []const u8, layer_idx: usize, values: []const f32) void {
            const state: *@This() = @ptrCast(@alignCast(context));
            var name_buf: [128]u8 = undefined;
            const golden_name = std.fmt.bufPrint(&name_buf, "language_model_model_layers_{d}_{s}", .{ layer_idx, name }) catch return;
            const expected = (readGolden(state.allocator, golden_name) catch return) orelse return;
            defer state.allocator.free(expected);
            if (expected.len != values.len) {
                std.debug.print("kimi3 probe L{d} {s}: LEN {d} vs {d}\n", .{ layer_idx, name, expected.len, values.len });
                return;
            }
            var max_diff: f32 = 0;
            for (expected, values) |e, v| max_diff = @max(max_diff, @abs(e - v));
            std.debug.print("kimi3 probe L{d} {s}: max|diff| {d}\n", .{ layer_idx, name, max_diff });
        }
    };
    var probe_state = ProbeState{ .allocator = allocator };
    const probe = model_mod.Model.Probe{ .context = @ptrCast(&probe_state), .callback = ProbeState.onStage };

    var logits = model.forwardProbed(&ctx, &token_ids, &probe) catch |err| {
        std.debug.print("kimi3 forward failed: {t}\n", .{err});
        return err;
    };
    defer logits.deinit();

    const actual = logits.dataConst();
    try std.testing.expectEqual(logits_ref.len, actual.len);
    var max_diff: f32 = 0;
    var max_at: usize = 0;
    for (logits_ref, actual, 0..) |expected, value, i| {
        const diff = @abs(expected - value);
        if (diff > max_diff) {
            max_diff = diff;
            max_at = i;
        }
    }
    std.debug.print("kimi3 logits max |diff| = {d} at {d} (ref {d}, got {d})\n", .{ max_diff, max_at, logits_ref[max_at], actual[max_at] });
    try std.testing.expect(max_diff < 5e-3);
}
