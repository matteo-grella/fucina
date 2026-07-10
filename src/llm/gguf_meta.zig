//! Flat GGUF loader glue shared by the model families (`llm/<family>/`):
//! `<arch>.<suffix>` metadata readers and the comptime-generic parallel layer
//! loader. Family model files keep thin wrappers that pin their per-family
//! policy (zero handling, layer load/deinit shape).

const std = @import("std");
const fucina = @import("fucina");

const ExecContext = fucina.ExecContext;
const gguf = fucina.gguf;

pub const Error = error{InvalidConfig};

/// Whether a present-but-zero integer key counts as valid. Families disagree
/// on purpose: qwen3 treats 0 like a missing key everywhere, while gemma reads
/// legitimately-zero keys (e.g. `attention.shared_kv_layers`).
pub const ZeroPolicy = enum { reject_zero, accept_zero };

/// Read `<arch>.<suffix>` as a usize; missing/negative (and, under
/// `.reject_zero`, zero) values are `Error.InvalidConfig`.
pub fn metaInt(file: *const gguf.File, arch: []const u8, suffix: []const u8, zero: ZeroPolicy) Error!usize {
    return metaIntOpt(file, arch, suffix, zero) orelse Error.InvalidConfig;
}

/// As `metaInt`, but invalid values read as absent (`null`).
pub fn metaIntOpt(file: *const gguf.File, arch: []const u8, suffix: []const u8, zero: ZeroPolicy) ?usize {
    var buf: [128]u8 = undefined;
    const key = std.fmt.bufPrint(&buf, "{s}.{s}", .{ arch, suffix }) catch return null;
    const v = file.getInt(key) orelse return null;
    if (v < 0) return null;
    if (zero == .reject_zero and v == 0) return null;
    return @intCast(v);
}

/// Read `<arch>.<suffix>` as an f32; missing is `Error.InvalidConfig`.
pub fn metaFloat(file: *const gguf.File, arch: []const u8, suffix: []const u8) Error!f32 {
    return metaFloatOpt(file, arch, suffix) orelse Error.InvalidConfig;
}

/// As `metaFloat`, but a missing key reads as absent (`null`).
pub fn metaFloatOpt(file: *const gguf.File, arch: []const u8, suffix: []const u8) ?f32 {
    var buf: [128]u8 = undefined;
    const key = std.fmt.bufPrint(&buf, "{s}.{s}", .{ arch, suffix }) catch return null;
    const v = file.getFloat(key) orelse return null;
    return @floatCast(v);
}

/// Load all model layers, in parallel across the exec work pool when
/// available. Layer loads are independent and the ExecContext allocator +
/// buffer pool are thread-safe, so the multi-GB weight copy+pack becomes an
/// N-core job (the dominant chunk of model load time). On any failure only
/// the layers that DID load are deinitialized before the first error (in
/// layer order) is returned.
///
/// `Loader` is a small per-family adapter value providing:
///   fn load(self: Loader, layer_i: usize) !Layer
///   fn deinitLayer(self: Loader, layer: *Layer) void
pub fn parallelLoadLayers(
    comptime Layer: type,
    comptime Loader: type,
    ctx: *ExecContext,
    loader: Loader,
    layers: []Layer,
) !void {
    const pool = ctx.workPool() orelse {
        var loaded: usize = 0;
        errdefer for (layers[0..loaded]) |*layer| loader.deinitLayer(layer);
        for (layers, 0..) |*layer, layer_i| {
            layer.* = try loader.load(layer_i);
            loaded += 1;
        }
        return;
    };

    const allocator = ctx.allocator;
    const slots = try allocator.alloc(?anyerror, layers.len);
    defer allocator.free(slots);
    @memset(slots, null);

    const Task = struct {
        loader: Loader,
        layer: *Layer,
        layer_i: usize,
        slot: *?anyerror,

        fn run(task: *const @This()) void {
            if (task.loader.load(task.layer_i)) |loaded| {
                task.layer.* = loaded;
            } else |err| {
                task.slot.* = err;
            }
        }
    };

    const tasks = try allocator.alloc(Task, layers.len);
    defer allocator.free(tasks);
    for (tasks, 0..) |*task, i| {
        task.* = .{ .loader = loader, .layer = &layers[i], .layer_i = i, .slot = &slots[i] };
    }

    pool.parallelChunks(Task, tasks, Task.run);

    for (slots) |slot| {
        if (slot) |err| {
            // Deinit only the layers that loaded (slot == null means success).
            for (slots, 0..) |s, i| {
                if (s == null) loader.deinitLayer(&layers[i]);
            }
            return err;
        }
    }
}

test {
    _ = @import("gguf_meta_tests.zig");
}
