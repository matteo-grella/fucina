//! Behavioral tests for the shared GGUF loader glue (`gguf_meta.zig`): the
//! per-family zero-policy split of the metadata readers, and the parallel
//! layer loader's partial-failure semantics (deinit only the layers that
//! loaded, return the first error in layer order).
const std = @import("std");
const fucina = @import("fucina");
const gguf_meta = @import("gguf_meta.zig");

const ExecContext = fucina.ExecContext;
const gguf = fucina.gguf;

/// Minimal metadata-only GGUF: one positive int, one zero int, one negative
/// int, and one float under a test arch prefix.
fn buildMetaFile(allocator: std.mem.Allocator) !gguf.File {
    var w = gguf.Writer.init(allocator);
    defer w.deinit();
    try w.addMetaString("general.architecture", "testarch");
    try w.addMetaInt("testarch.block_count", u32, 24);
    try w.addMetaInt("testarch.shared_kv_layers", u32, 0);
    try w.addMetaInt("testarch.negative", i32, -3);
    try w.addMetaFloat("testarch.eps", f32, 1e-6);

    const buf = try allocator.alloc(u8, 4096);
    defer allocator.free(buf);
    var sink = std.Io.Writer.fixed(buf);
    try w.finish(&sink);
    return gguf.File.parseOwned(allocator, try allocator.dupe(u8, sink.buffered()));
}

test "meta readers: positive/missing keys behave the same under both zero policies" {
    const allocator = std.testing.allocator;
    var file = try buildMetaFile(allocator);
    defer file.deinit();

    try std.testing.expectEqual(@as(usize, 24), try gguf_meta.metaInt(&file, "testarch", "block_count", .reject_zero));
    try std.testing.expectEqual(@as(usize, 24), try gguf_meta.metaInt(&file, "testarch", "block_count", .accept_zero));
    try std.testing.expectEqual(@as(?usize, 24), gguf_meta.metaIntOpt(&file, "testarch", "block_count", .reject_zero));
    try std.testing.expectEqual(@as(?usize, 24), gguf_meta.metaIntOpt(&file, "testarch", "block_count", .accept_zero));

    try std.testing.expectError(error.InvalidConfig, gguf_meta.metaInt(&file, "testarch", "missing", .reject_zero));
    try std.testing.expectError(error.InvalidConfig, gguf_meta.metaInt(&file, "testarch", "missing", .accept_zero));
    try std.testing.expectEqual(@as(?usize, null), gguf_meta.metaIntOpt(&file, "testarch", "missing", .reject_zero));
    try std.testing.expectEqual(@as(?usize, null), gguf_meta.metaIntOpt(&file, "testarch", "missing", .accept_zero));
}

test "meta readers: zero splits by policy, negative is invalid under both" {
    const allocator = std.testing.allocator;
    var file = try buildMetaFile(allocator);
    defer file.deinit();

    // Present-but-zero: rejected like missing (qwen3) vs read as 0 (gemma).
    try std.testing.expectError(error.InvalidConfig, gguf_meta.metaInt(&file, "testarch", "shared_kv_layers", .reject_zero));
    try std.testing.expectEqual(@as(usize, 0), try gguf_meta.metaInt(&file, "testarch", "shared_kv_layers", .accept_zero));
    try std.testing.expectEqual(@as(?usize, null), gguf_meta.metaIntOpt(&file, "testarch", "shared_kv_layers", .reject_zero));
    try std.testing.expectEqual(@as(?usize, 0), gguf_meta.metaIntOpt(&file, "testarch", "shared_kv_layers", .accept_zero));

    // Negative values never map to a usize.
    try std.testing.expectError(error.InvalidConfig, gguf_meta.metaInt(&file, "testarch", "negative", .reject_zero));
    try std.testing.expectError(error.InvalidConfig, gguf_meta.metaInt(&file, "testarch", "negative", .accept_zero));
    try std.testing.expectEqual(@as(?usize, null), gguf_meta.metaIntOpt(&file, "testarch", "negative", .accept_zero));
}

test "meta readers: float read + oversized key formats as absent" {
    const allocator = std.testing.allocator;
    var file = try buildMetaFile(allocator);
    defer file.deinit();

    try std.testing.expectEqual(@as(f32, 1e-6), try gguf_meta.metaFloat(&file, "testarch", "eps"));
    try std.testing.expectEqual(@as(?f32, 1e-6), gguf_meta.metaFloatOpt(&file, "testarch", "eps"));
    try std.testing.expectError(error.InvalidConfig, gguf_meta.metaFloat(&file, "testarch", "missing"));
    try std.testing.expectEqual(@as(?f32, null), gguf_meta.metaFloatOpt(&file, "testarch", "missing"));

    // Keys that overflow the format buffer read as absent, not out-of-bounds.
    const long_arch = "a" ** 200;
    try std.testing.expectEqual(@as(?usize, null), gguf_meta.metaIntOpt(&file, long_arch, "block_count", .accept_zero));
    try std.testing.expectError(error.InvalidConfig, gguf_meta.metaFloat(&file, long_arch, "eps"));
}

// ---------------------------------------------------------------------------
// parallelLoadLayers — fake Layer/Loader pair. `live` counts successful loads
// minus deinits (atomically: loads run on the work pool), so it must return
// to 0 after a partial failure's cleanup.
// ---------------------------------------------------------------------------

const TestLayer = struct {
    value: usize,
};

const TestLoader = struct {
    live: *std.atomic.Value(isize),
    fail_at: ?usize = null,
    second_fail_at: ?usize = null,

    pub fn load(self: TestLoader, layer_i: usize) !TestLayer {
        if (self.fail_at == layer_i) return error.SyntheticLoadFailure;
        if (self.second_fail_at == layer_i) return error.SyntheticSecondFailure;
        _ = self.live.fetchAdd(1, .monotonic);
        return .{ .value = layer_i };
    }

    pub fn deinitLayer(self: TestLoader, layer: *TestLayer) void {
        _ = self.live.fetchSub(1, .monotonic);
        layer.* = undefined;
    }
};

test "parallelLoadLayers: success fills every slot in layer order" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var live = std.atomic.Value(isize).init(0);
    const loader = TestLoader{ .live = &live };

    var layers: [16]TestLayer = undefined;
    try gguf_meta.parallelLoadLayers(TestLayer, TestLoader, &ctx, loader, &layers);
    try std.testing.expectEqual(@as(isize, layers.len), live.load(.monotonic));
    for (&layers, 0..) |*layer, i| {
        try std.testing.expectEqual(i, layer.value);
        loader.deinitLayer(layer);
    }
    try std.testing.expectEqual(@as(isize, 0), live.load(.monotonic));
}

test "parallelLoadLayers: partial failure deinits only the loaded layers" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var live = std.atomic.Value(isize).init(0);
    const loader = TestLoader{ .live = &live, .fail_at = 5 };

    var layers: [16]TestLayer = undefined;
    try std.testing.expectError(
        error.SyntheticLoadFailure,
        gguf_meta.parallelLoadLayers(TestLayer, TestLoader, &ctx, loader, &layers),
    );
    // Every layer that loaded (all but the failed slot) was deinitialized.
    try std.testing.expectEqual(@as(isize, 0), live.load(.monotonic));
}

test "parallelLoadLayers: multiple failures return the first error in layer order" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var live = std.atomic.Value(isize).init(0);
    const loader = TestLoader{ .live = &live, .fail_at = 9, .second_fail_at = 3 };

    var layers: [16]TestLayer = undefined;
    try std.testing.expectError(
        error.SyntheticSecondFailure, // slot 3 fails before slot 9 in layer order
        gguf_meta.parallelLoadLayers(TestLayer, TestLoader, &ctx, loader, &layers),
    );
    try std.testing.expectEqual(@as(isize, 0), live.load(.monotonic));
}

test "parallelLoadLayers: empty layer slice is a no-op" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var live = std.atomic.Value(isize).init(0);
    const loader = TestLoader{ .live = &live };

    var layers: [0]TestLayer = undefined;
    try gguf_meta.parallelLoadLayers(TestLayer, TestLoader, &ctx, loader, &layers);
    try std.testing.expectEqual(@as(isize, 0), live.load(.monotonic));
}
